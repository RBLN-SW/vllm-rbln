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

from collections.abc import Callable, Sequence
from typing import NamedTuple, Union

import torch
from vllm.forward_context import get_forward_context


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
    """One input of a compiled vision encoder at one bucket: its shape, the axis
    the bucket runs along, and the values its valid and padded entries take when
    nothing is copied in."""

    shape: tuple[int, ...]
    axis: int
    valid: float
    pad: float


class EncoderInputBuffers:
    """Device buffers a vision encoder's inputs are staged into, one set a bucket.

    Uploading into a freshly allocated device tensor runs several times slower
    than into one written before, so each bucket keeps its input tensors across
    calls. Staging copies the valid prefix of each input from the host and
    refills the padding on the device, so only the image's own bytes cross the
    link and the host builds no padded copies.
    """

    def __init__(self, device: torch.device | str, dtype: torch.dtype) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self._buffers: dict[int, list[torch.Tensor]] = {}

    def get(self, bucket: int, specs: Sequence[EncoderInput]) -> list[torch.Tensor]:
        """The bucket's buffers, allocated on first use."""
        buffers = self._buffers.get(bucket)
        if buffers is None:
            # Plain tensors, not inference tensors: staging writes them in place
            # inside and outside inference mode alike.
            with torch.inference_mode(False):
                buffers = [
                    torch.empty(spec.shape, dtype=self.dtype, device=self.device)
                    for spec in specs
                ]
            self._buffers[bucket] = buffers
        return buffers

    def stage(
        self,
        bucket: int,
        specs: Sequence[EncoderInput],
        prefixes: Sequence[torch.Tensor | None],
        num_valid: int,
    ) -> list[torch.Tensor]:
        """Fill the bucket's buffers: the first `num_valid` entries along each
        input's axis from its host prefix (or its valid value when the prefix is
        None), the rest with its pad value."""
        buffers = self.get(bucket, specs)
        for buffer, spec, prefix in zip(buffers, specs, prefixes, strict=True):
            head = buffer.narrow(spec.axis, 0, num_valid)
            if prefix is None:
                head.fill_(spec.valid)
            else:
                head.copy_(prefix)
            if num_valid < bucket:
                buffer.narrow(spec.axis, num_valid, bucket - num_valid).fill_(spec.pad)
        return buffers


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
