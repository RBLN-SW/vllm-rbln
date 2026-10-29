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

A layer's state is a `MambaSpec` KV cache group of one block a request
(block_size = max_model_len): the scheduler holds a block for every live
request and the group's block table names it. The state itself lives on this
backend's rows, not in the KV cache pool: the KDA core kernels
(`rbln_custom_ops::kda_prefill` / `kda_decode`) take a whole state tensor and
the compiler lowers a dtype view to a plain view, so a state can neither be
carved out of a page nor share an attention layer's page. `RBLNGDNStateRows`
maps each live block to a row; a block seen for the first time starts from
zero through `has_initial_state`.

KDA is GatedDeltaNet with a per-channel decay: the layer broadcasts its
per-head decay over the key channels and repeats q / k over each key head's
value heads. The short conv (`rbln_custom_ops::kda_conv`) keeps one ring each
for q, k and v on the same rows.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# The core kernels take 128-wide heads; prefill runs in 128-token chunks (the
# KDA kernel's own chunk is 64). A head's state is kept as R1's bytes: 64 rows
# of 512 bytes a [128, 128] bf16 state.
GDN_HEAD_DIM = 128
GDN_PREFILL_CHUNK = 128
R1_ROW_BYTES = 512
_CONV_LANES = 64


def gdn_state_rows(num_v_heads: int, dtype: torch.dtype) -> int:
    """The R1 rows a block's `num_v_heads` [128, 128] states take."""
    return num_v_heads * GDN_HEAD_DIM * GDN_HEAD_DIM * dtype.itemsize // R1_ROW_BYTES


@dataclass
class GDNLayerState:
    # [rows, width, channels / 64, 64] each: the conv rings of q, k and v, row
    # p % width holding position p
    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor
    # [rows, 64 * heads_v, 512] uint8: R1's bytes of the [heads_v, 128, 128] states
    ssm_state: torch.Tensor
    conv_width: int


@dataclass
class RBLNGDNAttentionMetadata:
    is_prefill: bool
    batch_size: int
    seq_len: int
    # [batch] each row's state row (see RBLNGDNStateRows): int32 on the device
    # for the conv kernel; int16 on the host for the core kernels, which read
    # their indices through the const buffer (a device copy crashes the runtime)
    state_index: torch.Tensor
    core_state_index: torch.Tensor
    # [batch] 1 where the row carries a cached state, likewise twice
    has_initial_state: torch.Tensor
    core_has_initial_state: torch.Tensor
    # [batch] int32: the position of the step's first token, and its valid tokens
    seq_idx: torch.Tensor
    query_len: torch.Tensor
    # [1, 1, seq_len] float32 on prefill: 1 for valid tokens, 0 for padding
    valid_mask: torch.Tensor | None
    states: dict[str, GDNLayerState]


class RBLNGDNStateRows:
    """Which state block sits on which row.

    Rows [0, num_live) hold the blocks of running requests; a block is given a
    row when it is first scheduled and gives it back when its request finishes
    or is preempted (`release`). Rows [num_live, 2 * num_live) are scratch, one
    per batch position: a padded row and a dummy run (block 0, vLLM's null
    block) write there, each position its own row since the kernels take
    distinct rows.
    """

    def __init__(self, num_live: int):
        self.num_live = num_live
        self.row_of: dict[int, int] = {}
        self._free = list(range(num_live - 1, -1, -1))

    def release(self, blocks: Iterable[int]) -> None:
        for block in blocks:
            if (row := self.row_of.pop(block, None)) is not None:
                self._free.append(row)

    def rows(self, blocks: Sequence[int], batch: int) -> list[int]:
        """The row of each scheduled block (`blocks[b]` for batch row b)."""
        rows = []
        for position in range(batch):
            block = blocks[position] if position < len(blocks) else 0
            if block == 0:
                rows.append(self.num_live + position)
                continue
            if (row := self.row_of.get(block)) is None:
                if not self._free:
                    raise RuntimeError(
                        f"no free GatedDeltaNet state row for block {block}: "
                        f"{len(self.row_of)} live blocks hold the {self.num_live} rows"
                    )
                row = self.row_of[block] = self._free.pop()
            rows.append(row)
        return rows


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
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.prefill_chunk_size = vllm_config.scheduler_config.max_num_batched_tokens
        self.dtype = vllm_config.model_config.dtype
        # At most max_num_seqs requests run, and the decode batch is padded to
        # at most max_num_seqs rows.
        num_live = vllm_config.scheduler_config.max_num_seqs
        self.rows = RBLNGDNStateRows(num_live)
        layers = vllm_config.compilation_config.static_forward_context
        self.states = {
            name: self._allocate(layers[name], 2 * num_live) for name in layer_names
        }
        self._staged: dict[tuple, torch.Tensor] = {}
        logger.info(
            "GatedDeltaNet state: %d layers x %d rows (%d live blocks, %d scratch)",
            len(layer_names),
            2 * num_live,
            num_live,
            num_live,
        )

    def _allocate(self, layer, rows: int) -> GDNLayerState:
        tp_size = layer.tp_size
        heads_k = layer.num_k_heads // tp_size
        heads_v = layer.num_v_heads // tp_size
        if layer.head_k_dim != GDN_HEAD_DIM or layer.head_v_dim != GDN_HEAD_DIM:
            raise NotImplementedError(
                "The RBLN GatedDeltaNet kernels take 128-wide heads, got "
                f"{layer.head_k_dim} / {layer.head_v_dim}"
            )
        width = layer.conv_kernel_size - 1

        def ring(heads: int) -> torch.Tensor:
            blocks = heads * GDN_HEAD_DIM // _CONV_LANES
            return torch.zeros(
                rows, width, blocks, _CONV_LANES, dtype=self.dtype, device=self.device
            )

        return GDNLayerState(
            conv_q=ring(heads_k),
            conv_k=ring(heads_k),
            conv_v=ring(heads_v),
            ssm_state=torch.zeros(
                rows,
                gdn_state_rows(heads_v, self.dtype),
                R1_ROW_BYTES,
                dtype=torch.uint8,
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

    def release(self, blocks: Iterable[int]) -> None:
        self.rows.release(blocks)

    def build(  # type: ignore[override]
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
    ) -> RBLNGDNAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_computed = common_attn_metadata.seq_lens[:num_reqs] - query_lens
        first_positions = positions[query_start_loc[:num_reqs]]
        # One block a request: the table is one column wide.
        blocks = common_attn_metadata.block_table_tensor[:num_reqs, 0].tolist()

        if is_prefill:
            assert num_reqs == 1, "RBLN prefills one request per step"
            batch, seq_len = 1, self.prefill_chunk_size
        else:
            batch, seq_len = batch_pad, 1
            assert int(query_lens.max()) == 1, (
                "a decode step takes one token per request"
            )
        rows = self.rows.rows(blocks, batch)

        def padded(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(batch, dtype=torch.int32)
            out[:num_reqs] = values.to(torch.int32)
            return out

        valid_mask = None
        if is_prefill:
            valid_mask = (torch.arange(seq_len) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, seq_len)
        return RBLNGDNAttentionMetadata(
            is_prefill=is_prefill,
            batch_size=batch,
            seq_len=seq_len,
            state_index=self._stage(torch.tensor(rows, dtype=torch.int32), "rows"),
            core_state_index=torch.tensor(rows, dtype=torch.int16),
            has_initial_state=self._stage(
                padded(num_computed > 0), "has_initial_state"
            ),
            core_has_initial_state=padded(num_computed > 0).to(torch.int16),
            seq_idx=self._stage(padded(first_positions), "seq_idx"),
            query_len=self._stage(padded(query_lens), "query_len"),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
            states=self.states,
        )
