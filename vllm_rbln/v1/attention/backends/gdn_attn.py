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

A layer's state is a `MambaSpec` block declared as a full-attention KV page, so
the layers sharing a buffer are one graph input with one chiplet cut. The
kernels read a state prefix of each local page and preserve the trailing
padding. A request holds one state block (`mamba_block_size` is the model
length); a padding row uses null block 0.

The short-conv ring is a block-aux cache on the same block ids, its channels
in the delta-rule kernels' key-head order (`grouped_qkv_rows`).
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
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo


def grouped_qkv_rows(num_k_heads: int, ratio: int, head_dim: int) -> torch.Tensor:
    """The rows of [q heads | k heads | v heads] regrouped by key head: q_h, k_h,
    then key head h's `ratio` value heads. The short conv runs in this order so
    that RSD cuts its channels where it cuts the delta-rule kernels' heads."""
    q = torch.arange(num_k_heads).view(num_k_heads, 1)
    k = q + num_k_heads
    v = (torch.arange(num_k_heads * ratio) + 2 * num_k_heads).view(num_k_heads, ratio)
    heads = torch.cat([q, k, v], dim=1).reshape(-1, 1)
    return (heads * head_dim + torch.arange(head_dim)).reshape(-1)


@dataclass(kw_only=True)
class RBLNGDNAttentionMetadata(GDNAttentionMetadata):
    """Upstream's GDN metadata for one RBLN step, plus the operands of the RBLN
    kernels.

    A step is one prefill request or a decode batch, never both and never
    speculative, so the `spec_*` fields and the ones for upstream's own kernels
    stay unset. The state indices are padded to the step's batch and stay int16
    on the host: the kernels read them through the const buffer, and a device
    copy crashes the runtime.
    """

    # The compiled step's batch: 1 on prefill, the padded batch on decode.
    batch_size: int
    # [batch] int16 on the host: the state block each row stores.
    store_state_indices: torch.Tensor
    # [batch] int32 on the device: kda_conv's per-row operands.
    conv_load_blocks: torch.Tensor
    conv_start_positions: torch.Tensor
    conv_query_lens: torch.Tensor
    conv_has_initial_state: torch.Tensor
    conv_store_blocks: torch.Tensor
    # [1, 1, largest prefill chunk] float32 on prefill: 1 for valid tokens, 0
    # for padding. The layer takes the step's chunk of it.
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
        mm_prefix_row_ends: torch.Tensor | None = None,
    ) -> RBLNGDNAttentionMetadata:
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

            batch_size = 1
            num_prefills = 1
            num_prefill_tokens = int(query_lens[0])
            num_decodes = 0
        else:
            assert int(query_lens.max()) == 1, (
                "a decode step takes one token per request"
            )

            batch_size = batch_pad
            num_prefills = 0
            num_prefill_tokens = 0
            num_decodes = num_reqs

        # TODO(prefix caching): a row stores to the block it loads, which holds
        # while mamba_block_size is the model length (one block-table column).
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        blocks = self._padded(block_table_tensor[:num_reqs, 0], batch_size)
        non_spec_state_indices_tensor = blocks.to(torch.int16)

        valid_mask = None
        if is_prefill:
            max_chunk = self.chunked_prefill_size
            valid_mask = (torch.arange(max_chunk) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, max_chunk)
        has_initial_state = self._padded(context_lens_tensor > 0, batch_size)
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
            batch_size=batch_size,
            store_state_indices=blocks.to(torch.int16),
            conv_load_blocks=self._stage(blocks, "conv_load_blocks"),
            conv_start_positions=self._stage(
                self._padded(first_positions, batch_size), "conv_start_positions"
            ),
            conv_query_lens=self._stage(
                self._padded(query_lens, batch_size), "conv_query_lens"
            ),
            conv_has_initial_state=self._stage(
                has_initial_state, "conv_has_initial_state"
            ),
            conv_store_blocks=self._stage(blocks, "conv_store_blocks"),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
        )
