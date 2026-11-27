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

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

import vllm_rbln.utils as rbln_utils
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo


@dataclass(kw_only=True)
class RBLNBaseMambaAttentionMetadata(BaseMambaAttentionMetadata):
    """Upstream's Mamba metadata for one RBLN step, plus the operands of the RBLN
    kernels, which read none of upstream's tensors (those stay None)."""

    batch_size: int
    has_initial_state: torch.Tensor
    load_blocks: torch.Tensor
    store_blocks: torch.Tensor
    conv_load_blocks: torch.Tensor
    conv_start_positions: torch.Tensor
    conv_query_lens: torch.Tensor
    conv_store_blocks: torch.Tensor
    num_accepted: torch.Tensor | None = None
    # Attached by the runner, as on the flash metadata.
    kv_caches: list[torch.Tensor] | None = None
    kv_cache_view_infos: list[KVCacheViewInfo] | None = None
    kv_cache_positions: dict[str, int] = field(default_factory=dict)

    @property
    def is_prefill(self) -> bool:
        """The step kind the dispatcher keys its graph on, as the flash metadata's."""
        return self.num_prefills > 0


class RBLNBaseMambaAttentionMetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    metadata_cls: ClassVar[type[RBLNBaseMambaAttentionMetadata]]

    def _padded(self, values: torch.Tensor, batch: int) -> torch.Tensor:
        """`values` as int32, padded with 0 to the step's `batch` rows."""
        return rbln_utils.pad(values.to(torch.int32), 0, batch)

    def _get_state_blocks(
        self, block_table: torch.Tensor, num_tokens: torch.Tensor, batch: int
    ) -> torch.Tensor:
        """Each request's block holding its state after `num_tokens` tokens."""
        blocks = mamba_get_block_table_tensor(
            block_table,
            num_tokens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        return self._padded(blocks[:, 0], batch)

    def stage(  # type: ignore[override]
        self,
        metadata: Any,
        group_shared: dict[tuple, torch.Tensor] | None = None,
    ) -> None:
        metadata.has_initial_state = self._stage(
            metadata.has_initial_state, "has_initial_state", group_shared
        )
        metadata.conv_load_blocks = self._stage(
            metadata.conv_load_blocks, "conv_load_blocks"
        )
        metadata.conv_start_positions = self._stage(
            metadata.conv_start_positions, "conv_start_positions", group_shared
        )
        metadata.conv_query_lens = self._stage(
            metadata.conv_query_lens, "conv_query_lens", group_shared
        )
        metadata.conv_store_blocks = self._stage(
            metadata.conv_store_blocks, "conv_store_blocks"
        )
        metadata.num_accepted = self._stage(metadata.num_accepted, "num_accepted")

    def _compute_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        back_pad: torch.Tensor | None = None,
        num_accepted: torch.Tensor | None = None,
        state_load_pos: torch.Tensor | None = None,
    ) -> Any:
        m = common_attn_metadata
        num_reqs = m.num_reqs
        query_start_loc = m.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        # An MTP decode window is padded at the back to 1 + drafts tokens.
        valid_lens = query_lens
        if back_pad is not None:
            valid_lens = query_lens - back_pad[:num_reqs]
        context_lens_tensor = m.seq_lens[:num_reqs] - valid_lens
        first_positions = positions[query_start_loc[:num_reqs]]

        if is_prefill:
            assert num_reqs == 1, "RBLN prefills one request per step"
            batch_size = 1
            num_prefills = 1
            num_prefill_tokens = int(query_lens[0])
            num_decodes = 0
        else:
            batch_size = batch_pad
            spec = self.vllm_config.speculative_config
            query_len = 1 + (spec.num_speculative_tokens if spec is not None else 0)
            assert bool((query_lens == query_len).all()), (
                f"a decode step takes {query_len} tokens a request, got {query_lens}"
            )
            num_prefills = 0
            num_prefill_tokens = 0
            num_decodes = num_reqs

        block_table = m.block_table_tensor[:num_reqs]
        load_tokens = context_lens_tensor
        if state_load_pos is not None:
            load_tokens = state_load_pos[:num_reqs] + 1
        load_blocks = self._get_state_blocks(block_table, load_tokens, batch_size)
        store_blocks = self._get_state_blocks(
            block_table, m.seq_lens[:num_reqs], batch_size
        )
        return self.metadata_cls(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decodes,
            num_reqs=num_reqs,
            has_initial_states_p=None,
            query_start_loc_p=None,
            num_computed_tokens_p=None,
            state_indices_tensor_p=None,
            state_indices_tensor_d=None,
            query_start_loc_d=None,
            num_accepted_tokens=None,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=m.seq_lens[:num_reqs],
            batch_size=batch_size,
            has_initial_state=self._padded(context_lens_tensor > 0, batch_size),
            load_blocks=load_blocks.to(torch.int16),
            store_blocks=store_blocks.to(torch.int16),
            conv_load_blocks=load_blocks,
            conv_start_positions=self._padded(first_positions, batch_size),
            conv_query_lens=self._padded(valid_lens, batch_size),
            conv_store_blocks=store_blocks,
            num_accepted=None
            if num_accepted is None
            else self._padded(num_accepted, batch_size),
        )
