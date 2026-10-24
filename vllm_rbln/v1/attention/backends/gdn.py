# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""GatedDeltaNet (Qwen3-Next, Qwen3.5 `linear_attention`) state on RBLN.

The core kernels `rbln_custom_ops::gdn_prefill` / `gdn_decode` update the state
rows [slot, slot + batch), so a request's state has to sit on its row of the
step's batch. `RBLNGDNStateRows` keeps that invariant with device row copies
before each step, and parks the states of live requests that are not in the
step above the padded batch. The states never enter a KV cache group: the
layer's `get_kv_cache_spec` is patched to None (patches/qwen3_5.py), and the
runner builds this group next to the KV cache groups.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# The core kernels take 128-wide heads and run prefill in 128-token chunks; the
# recurrent state of one head is kept blocked as [Dv / 64, Dk, 64].
GDN_HEAD_DIM = 128
GDN_PREFILL_CHUNK = 128
_STATE_BLOCKS = (GDN_HEAD_DIM // 64, GDN_HEAD_DIM, 64)
_CONV_LANES = 64


def stateless_gdn_layers(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> list[str]:
    """The Mamba-family layers that own no KV cache group, in model order."""
    grouped = {
        name for group in kv_cache_config.kv_cache_groups for name in group.layer_names
    }
    layers = get_layers_from_vllm_config(vllm_config, MambaBase)
    return [name for name in layers if name not in grouped]


@dataclass
class GDNLayerState:
    # [rows, width, channels / 64, 64]: the conv ring, row p % width holds position p
    conv_state: torch.Tensor
    # [rows, heads_k, ratio, 2, 128, 64]: the recurrent state in the kernel's layout
    ssm_state: torch.Tensor
    conv_width: int


@dataclass
class RBLNGDNAttentionMetadata:
    is_prefill: bool
    batch_size: int
    seq_len: int
    # int16 scalar, the first state row the core kernels update (always 0: the
    # rows are permuted to the batch before the step)
    slot: torch.Tensor
    # [batch] in the model dtype for the core kernel, int32 for the conv kernel
    has_initial_state: torch.Tensor
    has_initial_state_int: torch.Tensor
    # [batch] int32
    state_index: torch.Tensor
    seq_idx: torch.Tensor
    query_len: torch.Tensor
    # [1, 1, 1, seq_len] float32 on prefill: 1 for valid tokens, 0 for padding
    valid_mask: torch.Tensor | None
    states: dict[str, GDNLayerState]


class RBLNGDNStateRows:
    """Which request's state sits on which row, and the row copies that keep a
    request on its batch row.

    Rows [0, batch) hold the step's requests in batch order; rows at or above the
    padded batch park the live requests the step leaves out, so a padded row never
    overwrites a live state. The last row is scratch for the swaps.
    """

    def __init__(self, num_rows: int, tensors: Sequence[torch.Tensor]):
        for tensor in tensors:
            assert tensor.shape[0] == num_rows + 1, tensor.shape
        self.num_rows = num_rows
        self.tensors = list(tensors)
        self.owner: list[str | None] = [None] * num_rows
        self.row_of: dict[str, int] = {}

    def release(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            row = self.row_of.pop(req_id, None)
            if row is not None:
                self.owner[row] = None

    def _swap(self, a: int, b: int) -> None:
        if a == b:
            return
        scratch = self.num_rows
        for tensor in self.tensors:
            tensor[scratch].copy_(tensor[a])
            tensor[a].copy_(tensor[b])
            tensor[b].copy_(tensor[scratch])
        self.owner[a], self.owner[b] = self.owner[b], self.owner[a]
        for row in (a, b):
            if (owner := self.owner[row]) is not None:
                self.row_of[owner] = row

    def _park(self, row: int, batch: int) -> None:
        for free in range(batch, self.num_rows):
            if self.owner[free] is None:
                self._swap(row, free)
                return
        raise RuntimeError(
            f"no free GatedDeltaNet state row above the batch of {batch}: "
            f"{len(self.row_of)} requests hold rows, {self.num_rows} rows exist"
        )

    def place(self, req_ids: Sequence[str] | None, batch: int) -> None:
        """Put `req_ids[b]`'s state on row b and clear rows [len(req_ids), batch)
        of live states. A request seen for the first time takes its row as is;
        its first prefill starts from zero through `has_initial_state`."""
        ids = list(req_ids or ())
        assert len(ids) <= batch <= self.num_rows, (len(ids), batch, self.num_rows)
        for row, req_id in enumerate(ids):
            current = self.row_of.get(req_id)
            if current is None:
                if self.owner[row] is not None:
                    self._park(row, batch)
                self.owner[row] = req_id
                self.row_of[req_id] = row
            elif current != row:
                self._swap(current, row)
        for row in range(len(ids), batch):
            if self.owner[row] is not None:
                self._park(row, batch)


@register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)
class RBLNGDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RBLN_GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RBLNGDNAttentionMetadataBuilder"]:
        return RBLNGDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


class RBLNGDNAttentionMetadataBuilder(
    AttentionMetadataBuilder[RBLNGDNAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        max_batch_pad: int,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.prefill_chunk_size = vllm_config.scheduler_config.max_num_batched_tokens
        self.dtype = vllm_config.model_config.dtype
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        # Live requests fit the parking region even when every one of them sits
        # out a step; the extra max_num_seqs rows absorb requests the engine
        # preempted and has not finished.
        num_rows = max_batch_pad + 2 * max_num_seqs
        layers = vllm_config.compilation_config.static_forward_context
        self.states: dict[str, GDNLayerState] = {}
        tensors: list[torch.Tensor] = []
        for name in layer_names:
            state = self._allocate(layers[name], num_rows + 1)
            self.states[name] = state
            tensors += [state.conv_state, state.ssm_state]
        self.rows = RBLNGDNStateRows(num_rows, tensors)
        self._staged: dict[tuple, torch.Tensor] = {}
        logger.info(
            "GatedDeltaNet state: %d layers x %d rows (batch up to %d, %d parked)",
            len(layer_names),
            num_rows,
            max_batch_pad,
            2 * max_num_seqs,
        )

    def _allocate(self, layer, rows: int) -> GDNLayerState:
        tp_size = layer.tp_size
        heads_k = layer.num_k_heads // tp_size
        ratio = layer.num_v_heads // layer.num_k_heads
        channels = layer.conv_dim // tp_size
        if layer.head_k_dim != GDN_HEAD_DIM or layer.head_v_dim != GDN_HEAD_DIM:
            raise NotImplementedError(
                "The RBLN GatedDeltaNet kernels take 128-wide heads, got "
                f"{layer.head_k_dim} / {layer.head_v_dim}"
            )
        if channels % _CONV_LANES:
            raise NotImplementedError(
                f"The RBLN conv kernel takes channels in blocks of {_CONV_LANES}, got "
                f"{channels} per rank"
            )
        width = layer.conv_kernel_size - 1
        return GDNLayerState(
            conv_state=torch.zeros(
                rows,
                width,
                channels // _CONV_LANES,
                _CONV_LANES,
                dtype=self.dtype,
                device=self.device,
            ),
            ssm_state=torch.zeros(
                rows,
                heads_k,
                ratio,
                *_STATE_BLOCKS,
                dtype=self.dtype,
                device=self.device,
            ),
            conv_width=width,
        )

    def _stage(self, t: torch.Tensor, slot: str) -> torch.Tensor:
        """Copy a host tensor into this builder's persistent device buffer, keyed
        by name so same-shaped tensors never share one (see the flash builder)."""
        key = (slot, t.shape, t.dtype)
        if (buf := self._staged.get(key)) is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            self._staged[key] = buf
        buf.copy_(t)
        return buf

    def release(self, req_ids: Sequence[str]) -> None:
        self.rows.release(req_ids)

    def build(  # type: ignore[override]
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        req_ids: Sequence[str] | None = None,
    ) -> RBLNGDNAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_computed = common_attn_metadata.seq_lens[:num_reqs] - query_lens
        first_positions = positions[query_start_loc[:num_reqs]]

        if is_prefill:
            assert num_reqs == 1, "RBLN prefills one request per step"
            batch, seq_len = 1, self.prefill_chunk_size
        else:
            batch, seq_len = batch_pad, 1
            assert int(query_lens.max()) == 1, (
                "a decode step takes one token per request"
            )
        self.rows.place(req_ids, batch)

        def padded(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(batch, dtype=torch.int32)
            out[:num_reqs] = values.to(torch.int32)
            return out

        has_initial_state = padded(num_computed > 0)
        valid_mask = None
        if is_prefill:
            valid_mask = (torch.arange(seq_len) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, 1, seq_len)
        return RBLNGDNAttentionMetadata(
            is_prefill=is_prefill,
            batch_size=batch,
            seq_len=seq_len,
            slot=self._stage(torch.zeros((), dtype=torch.int16), "slot"),
            has_initial_state=self._stage(
                has_initial_state.to(self.dtype), "has_initial_state"
            ),
            has_initial_state_int=self._stage(
                has_initial_state, "has_initial_state_int"
            ),
            state_index=self._stage(
                torch.arange(batch, dtype=torch.int32), "state_index"
            ),
            seq_idx=self._stage(padded(first_positions), "seq_idx"),
            query_len=self._stage(padded(query_lens), "query_len"),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
            states=self.states,
        )
