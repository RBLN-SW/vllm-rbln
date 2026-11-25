# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import TYPE_CHECKING, NamedTuple, Union

import torch
from vllm.forward_context import get_forward_context

if TYPE_CHECKING:
    from torch_rbln.memory import HostMemoryRegistration


def is_prefill_step() -> bool:
    """The step's prefill/decode classification, read off the attention metadata
    the runner put in the forward context (one per layer; every layer agrees)."""
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()))
    return attn_metadata.is_prefill


def run_mm_encoder(
    encoder: Callable[..., torch.Tensor],
    inputs: list[torch.Tensor],
    num_embeddings: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    embeds = encoder(*inputs)[:num_embeddings]
    if dtype is None:
        return embeds.cpu()
    return embeds.to(device="cpu", dtype=dtype)


class EncoderInput(NamedTuple):
    """Shape and staging policy for one vision encoder input."""

    name: str
    shape: tuple[int, ...]
    axis: int
    pad: float
    dtype: torch.dtype | None = None
    allow_prefix_copy: bool = True
    use_registered_host_buffer: bool = False

    def nbytes(self, default_dtype: torch.dtype) -> int:
        return prod(self.shape) * (self.dtype or default_dtype).itemsize


def encoder_input_sizes(
    buckets: Sequence[int],
    specs_for_bucket: Callable[[int], Sequence[EncoderInput]],
    default_dtype: torch.dtype,
) -> list[int]:
    """Return the byte size of each device tensor allocated across the buckets."""
    return [
        spec.nbytes(default_dtype)
        for bucket in dict.fromkeys(buckets)
        for spec in specs_for_bucket(bucket)
    ]


class EncoderInputBuffers:
    """Reuse device input buffers by bucket.

    stage() and warmup_inputs() update tensors returned by earlier calls for
    the same bucket in place. Prefix copies can leave old padding values, so
    callers must refresh the attention mask for each input.
    """

    def __init__(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
        *,
        specs_by_bucket: Mapping[int, Sequence[EncoderInput]],
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._specs_by_bucket = dict(specs_by_bucket)
        self._buffers_by_bucket: dict[int, list[torch.Tensor]] = {}
        self._registered_host_buffers: dict[
            tuple[int, str], tuple[torch.Tensor, HostMemoryRegistration] | None
        ] = {}

    def _get_or_create_registered_host_buffer(
        self, bucket_size: int, spec: EncoderInput
    ) -> torch.Tensor | None:
        if self.device.type != "rbln":
            return None
        key = (bucket_size, spec.name)
        if key in self._registered_host_buffers:
            cached = self._registered_host_buffers[key]
            return cached[0] if cached is not None else None

        dtype = spec.dtype or self.dtype
        nbytes = spec.nbytes(self.dtype)
        with torch.inference_mode(False):
            host = torch.rbln.huge_host_empty(nbytes).view(dtype).view(spec.shape)
        registration = torch.rbln.host_register_memory(host, device=self.device)
        if registration.covers(host.data_ptr(), nbytes):
            self._registered_host_buffers[key] = (host, registration)
            return host

        registration.close()
        self._registered_host_buffers[key] = None
        return None

    def _get_or_create_bucket(
        self,
        bucket_size: int,
    ) -> tuple[Sequence[EncoderInput], list[torch.Tensor]]:
        specs = self._specs_by_bucket[bucket_size]
        buffers = self._buffers_by_bucket.get(bucket_size)
        if buffers is None:
            # Plain tensors, not inference tensors: staging writes them in place
            # inside and outside inference mode alike.
            with torch.inference_mode(False):
                buffers = [self._allocate_input(spec) for spec in specs]
            self._buffers_by_bucket[bucket_size] = buffers
        return specs, buffers

    def _allocate_input(self, spec: EncoderInput) -> torch.Tensor:
        dtype = spec.dtype or self.dtype
        buffer = torch.full(spec.shape, spec.pad, dtype=dtype, device=self.device)
        if self.device.type == "rbln" and spec.use_registered_host_buffer:
            # Streamed patches start in flat device memory on node 0, chiplet 0.
            torch.rbln.bind_device_memory(buffer)
        return buffer

    def warmup_inputs(
        self,
        bucket_size: int,
        *,
        fill_overrides: Mapping[str, float],
    ) -> list[torch.Tensor]:
        """Fill the bucket for warmup and prepare its host staging buffers."""
        specs, buffers = self._get_or_create_bucket(bucket_size)
        for spec, buffer in zip(specs, buffers, strict=True):
            if spec.use_registered_host_buffer:
                self._get_or_create_registered_host_buffer(bucket_size, spec)
            buffer.fill_(fill_overrides.get(spec.name, spec.pad))
        return buffers

    def stage(
        self,
        bucket_size: int,
        input_tensors: Sequence[torch.Tensor],
        num_valid: int,
    ) -> list[torch.Tensor]:
        """Copy each tensor over the whole buffer or its valid prefix.

        The shape must match the spec outside the bucket axis. If
        allow_prefix_copy is False, the bucket axis must also match in full.
        """
        specs, buffers = self._get_or_create_bucket(bucket_size)
        # Validate every input before writing any reused buffers.
        for spec, tensor in zip(specs, input_tensors, strict=True):
            axis = spec.axis if spec.axis >= 0 else len(spec.shape) + spec.axis
            if (
                tensor.ndim != len(spec.shape)
                or tensor.shape[:axis] != spec.shape[:axis]
                or tensor.shape[axis + 1 :] != spec.shape[axis + 1 :]
            ):
                raise ValueError(
                    f"Encoder input {spec.name!r} has shape {tuple(tensor.shape)}, "
                    f"expected {spec.shape} with only bucket axis {spec.axis} "
                    "allowed to vary"
                )
            length = tensor.shape[spec.axis]
            if length == bucket_size:
                continue
            if not spec.allow_prefix_copy:
                raise ValueError(
                    f"Encoder input {spec.name!r} requires a full "
                    f"{bucket_size}-entry tensor "
                    f"along axis {spec.axis}, got {length} entries"
                )
            assert length == num_valid, (
                f"{spec.name!r} received {length} entries for "
                f"{num_valid} valid entries in a {bucket_size}-entry bucket"
            )
        for buffer, spec, tensor in zip(buffers, specs, input_tensors, strict=True):
            length = tensor.shape[spec.axis]
            full = length == bucket_size
            dst = buffer if full else buffer.narrow(spec.axis, 0, length)
            if (
                spec.use_registered_host_buffer
                and length
                and tensor.device.type == "cpu"
            ):
                host = self._get_or_create_registered_host_buffer(bucket_size, spec)
                if host is not None:
                    host = host if full else host.narrow(spec.axis, 0, length)
                    host.copy_(tensor)
                    tensor = host
            dst.copy_(tensor)
        return buffers


class BufferedVisionEncoder:
    """Keep a compiled encoder callable with its input staging buffers."""

    def __init__(
        self,
        encoder: Callable[..., torch.Tensor],
        *,
        specs_by_bucket: Mapping[int, Sequence[EncoderInput]],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.encoder = encoder
        self.input_buffers = EncoderInputBuffers(
            device, dtype, specs_by_bucket=specs_by_bucket
        )


def pad(
    x: torch.Tensor, dim: int, target_len: int, pad_value: Union[int, float] = 0
) -> torch.Tensor:
    """Pad along the given dimension to target_len using pad_value."""
    current = x.size(dim)
    if current >= target_len:
        # NOTE: dynamo distinguishes views and non-views for inputs,
        # so ensure that the output is always a non-view.
        return x if x._base is None else x.clone()

    pad_shape = list(x.shape)
    pad_shape[dim] = target_len - current
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def pad_speculative_draft_tokens(
    input_ids: torch.Tensor,
    num_scheduled_tokens: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    """
    Pad per-request draft tokens to a uniform length (max across requests)
    by inserting zeros.

    Assumes `input_ids` is a 1D concatenation of per-request draft tokens
    in request order.
    Example 1:
      input_ids = [3925, 3823, 1694, 477] or [13, 18, 19, 20]
      num_scheduled_tokens = [1, 3]
    returns:
      [3925, 0, 0, 3823, 1694, 477] or [13, 0, 0, 18, 19, 20]

    Example 2:
      input_ids = [3363, 315, 11]
      num_scheduled_tokens = [2, 1]
      max_len = 3
    returns:
      [3363, 315, 0, 11, 0, 0]
    """
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}")

    if num_scheduled_tokens.ndim != 1:
        raise ValueError(
            f"num_scheduled_tokens must be 1D, got shape={num_scheduled_tokens.shape}"
        )

    num_reqs = num_scheduled_tokens.numel()
    max_sched = num_scheduled_tokens.max().item()

    if max_len is not None:
        if max_len < max_sched:
            raise ValueError(
                f"max_len({max_len}) must be >= max(num_scheduled_tokens)({max_sched})"
            )
        max_sched = max_len

    # Create flattened destination indices
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, device=num_scheduled_tokens.device), num_scheduled_tokens
    )
    token_offsets = (
        torch.arange(input_ids.numel(), device=num_scheduled_tokens.device)
        - num_scheduled_tokens.cumsum(0)[req_indices]
        + num_scheduled_tokens[req_indices]
    )
    dest_indices = req_indices * max_sched + token_offsets

    # Scatter input tokens into padded output
    out = torch.zeros(
        num_reqs * max_sched, device=input_ids.device, dtype=input_ids.dtype
    )
    out.index_copy_(0, dest_indices, input_ids)

    return out
