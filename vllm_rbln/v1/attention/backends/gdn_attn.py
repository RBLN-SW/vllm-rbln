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

"""GatedDeltaNet (Qwen3.5 `linear_attention`) state on RBLN.

A layer's `MambaSpec` describes a full-attention KV page, including unused
capacity. The runner binds that shape so sharing layers use one graph input
and one chiplet cut. The kernels size a contiguous state prefix per local
page from Q/K/V and preserve trailing padding. State uses the activation
dtype (dlfloat16 inside the kernels).

The builder addresses a request's one state block by its group's block table;
`mamba_block_size` is the model length. A padding batch row uses null block 0.
The short-conv ring is a block-aux cache on those same block ids. Its channels
and the delta-rule heads share the key-head order from `grouped_qkv_rows`.
"""

from dataclasses import dataclass

import torch
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

import vllm_rbln.utils as rbln_utils
from vllm_rbln.config import RBLNConfig
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo


def grouped_qkv_rows(heads_k: int, ratio: int, head_dim: int) -> torch.Tensor:
    """The rows of [q heads | k heads | v heads] regrouped by key head: q_h, k_h,
    then key head h's `ratio` value heads, for each h. The short conv runs one
    call over this order, so RSD's cut of its channels is the delta-rule
    kernels' cut of their heads and no chiplet reads another's output."""
    q = torch.arange(heads_k).view(heads_k, 1)
    k = q + heads_k
    v = (torch.arange(heads_k * ratio) + 2 * heads_k).view(heads_k, ratio)
    heads = torch.cat([q, k, v], dim=1).reshape(-1, 1)
    return (heads * head_dim + torch.arange(head_dim)).reshape(-1)


@dataclass(kw_only=True)
class RBLNGDNAttentionMetadata(GDNAttentionMetadata):
    """Upstream's GDN metadata as one RBLN step fills it, and the operands the
    RBLN kernels take beside it.

    A step is one prefill request (`num_prefills` 1) or a decode batch
    (`num_decodes` requests), never both and never speculative, so the
    `spec_*` fields stay unset, as do the ones upstream fills for its own
    kernels (chunk indices, query start locs, the causal_conv1d pointers).
    `non_spec_state_indices_tensor` (the state block each row loads) and
    `has_initial_state` (1 where that block carries a state, prefill only) are
    padded to the step's batch and kept [batch] int16 on the host: the
    delta-rule kernels read them through the const buffer (a device copy
    crashes the runtime). As upstream, a prefill step names them again as the
    chunk kernel's `prefill_state_indices` and `prefill_has_initial_state`.
    """

    # The compiled step: [1, prefill chunk] or [padded batch, 1].
    batch_size: int
    seq_len: int
    # [batch] int16 on the host: the state block each row stores.
    store_state_indices: torch.Tensor
    # [batch] int32 on the device: what kda_conv reads a row's ring by (its
    # block, the step's first position, its valid tokens, 1 where the ring
    # carries a state) and the block it stores the ring to.
    conv_state_index: torch.Tensor
    conv_seq_idx: torch.Tensor
    conv_query_len: torch.Tensor
    conv_has_initial_state: torch.Tensor
    conv_store_index: torch.Tensor
    # [1, 1, seq_len] float32 on prefill: 1 for valid tokens, 0 for padding
    valid_mask: torch.Tensor | None
    # Attached by the runner (attach_kv_cache_bindings), as on the flash metadata.
    kv_caches: list[torch.Tensor] | None = None
    kv_cache_view_infos: list[KVCacheViewInfo] | None = None

    @property
    def is_prefill(self) -> bool:
        """The step kind the dispatcher keys its graph on, as the flash metadata's."""
        return self.num_prefills > 0


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


class RBLNGDNAttentionMetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    """The flash builder for its step shape and staged device buffers; the
    metadata it builds is the GDN kernels'."""

    def _padded(self, values: torch.Tensor, batch: int) -> torch.Tensor:
        """`values` as int32, padded with 0 to the step's `batch` rows."""
        return rbln_utils.pad(values.to(torch.int32), 0, batch)

    def build(  # type: ignore[override]
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        skip_attn_masks: bool = False,
        back_pad: torch.Tensor | None = None,
        is_prompt_sorted: bool = False,
        num_accepted: torch.Tensor | None = None,
        state_load_pos: torch.Tensor | None = None,
    ) -> RBLNGDNAttentionMetadata:
        """The draft rows are refused with speculative decoding."""
        assert num_accepted is None and state_load_pos is None, (
            "speculative decoding is refused for GatedDeltaNet models"
        )
        m = common_attn_metadata
        num_reqs = m.num_reqs
        query_start_loc = m.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        context_lens_tensor = m.seq_lens[:num_reqs] - query_lens
        first_positions = positions[query_start_loc[:num_reqs]]

        if is_prefill:
            assert num_reqs == 1, "RBLN prefills one request per step"
            num_prefills, num_prefill_tokens, num_decodes = 1, int(query_lens[0]), 0
            # The runner pads a prefill step to the shortest chunk that holds it.
            rbln_config: RBLNConfig = self.vllm_config.additional_config
            assert rbln_config.prefill_chunk_size is not None  # resolved at start-up
            batch = 1
            seq_len = min(
                c for c in rbln_config.prefill_chunk_size if c >= num_prefill_tokens
            )
        else:
            batch, seq_len = batch_pad, 1
            assert int(query_lens.max()) == 1, (
                "a decode step takes one token per request"
            )
            num_prefills, num_prefill_tokens, num_decodes = 0, 0, num_reqs

        # TODO(prefix caching): one column while mamba_block_size is the model
        # length, so a row stores to the block it loads; a step crossing into a
        # new block stores to the block of its last scheduled token instead.
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        blocks = self._padded(block_table_tensor[:num_reqs, 0], batch)
        non_spec_state_indices_tensor = blocks.to(torch.int16)

        valid_mask = None
        if is_prefill:
            valid_mask = (torch.arange(seq_len) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, seq_len)
        has_initial_state = self._padded(context_lens_tensor > 0, batch)
        prefill_has_initial_state = (
            has_initial_state.to(torch.int16) if is_prefill else None
        )
        return RBLNGDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decodes,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=prefill_has_initial_state,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            prefill_state_indices=non_spec_state_indices_tensor if is_prefill else None,
            prefill_has_initial_state=prefill_has_initial_state,
            batch_size=batch,
            seq_len=seq_len,
            store_state_indices=blocks.to(torch.int16),
            conv_state_index=self._stage(blocks, "conv_state_index"),
            conv_seq_idx=self._stage(
                self._padded(first_positions, batch), "conv_seq_idx"
            ),
            conv_query_len=self._stage(
                self._padded(query_lens, batch), "conv_query_len"
            ),
            conv_has_initial_state=self._stage(
                has_initial_state, "conv_has_initial_state"
            ),
            conv_store_index=self._stage(blocks, "conv_store_index"),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
        )
