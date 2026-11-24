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

from dataclasses import dataclass, fields

import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

import vllm_rbln.utils as rbln_utils

from .flash_attention import (
    RBLNFlashAttentionBackend,
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)


@dataclass
class RBLNInklingAttentionMetadata(RBLNFlashAttentionMetadata):
    conv_load_blocks: torch.Tensor | None = None
    conv_store_blocks: torch.Tensor | None = None
    conv_start_positions: torch.Tensor | None = None
    conv_query_lens: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None


class RBLNInklingAttentionBackend(RBLNFlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RBLN_INKLING_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RBLNInklingAttentionMetadataBuilder"]:
        return RBLNInklingAttentionMetadataBuilder


class RBLNInklingAttentionMetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    def stage(  # type: ignore[override]
        self,
        metadata: RBLNInklingAttentionMetadata,
        group_shared: dict[tuple, torch.Tensor] | None = None,
    ) -> None:
        super().stage(metadata, group_shared)
        metadata.conv_load_blocks = self._stage(
            metadata.conv_load_blocks, "conv_load_blocks"
        )
        metadata.conv_store_blocks = self._stage(
            metadata.conv_store_blocks, "conv_store_blocks"
        )
        metadata.conv_start_positions = self._stage(
            metadata.conv_start_positions, "conv_start_positions", group_shared
        )
        metadata.conv_query_lens = self._stage(
            metadata.conv_query_lens, "conv_query_lens", group_shared
        )
        metadata.has_initial_state = self._stage(
            metadata.has_initial_state, "has_initial_state", group_shared
        )

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
        group_shared: dict[tuple, torch.Tensor] | None = None,
        defer_staging: bool = False,
    ) -> RBLNInklingAttentionMetadata:
        flash = super().build(
            common_attn_metadata,
            positions,
            batch_pad,
            is_prefill,
            skip_attn_masks=skip_attn_masks,
            back_pad=back_pad,
            is_prompt_sorted=is_prompt_sorted,
            mm_prefix_row_ends=mm_prefix_row_ends,
            group_shared=group_shared,
            defer_staging=True,
        )
        m = common_attn_metadata
        num_reqs = m.num_reqs
        query_start_loc = m.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        valid_lens = (
            query_lens if back_pad is None else query_lens - back_pad[:num_reqs]
        )
        seq_lens = m.seq_lens[:num_reqs]
        context_lens = seq_lens - valid_lens
        block_table = m.block_table_tensor[:num_reqs].long()
        block_size = self.kv_cache_spec.block_size

        def block_of(num_tokens: torch.Tensor) -> torch.Tensor:
            column = ((num_tokens - 1).clamp(min=0) // block_size).long()
            return block_table.gather(1, column[:, None])[:, 0]

        batch = 1 if is_prefill else batch_pad

        def padded(values: torch.Tensor) -> torch.Tensor:
            return rbln_utils.pad(values.to(torch.int32), 0, batch)

        metadata = RBLNInklingAttentionMetadata(
            **{f.name: getattr(flash, f.name) for f in fields(flash)},
            conv_load_blocks=padded(
                torch.where(context_lens > 0, block_of(context_lens), 0)
            ),
            conv_store_blocks=padded(block_of(seq_lens)),
            conv_start_positions=padded(positions[query_start_loc[:num_reqs]]),
            conv_query_lens=padded(valid_lens),
            has_initial_state=padded(context_lens > 0),
        )
        if not defer_staging:
            self.stage(metadata, group_shared)
        return metadata
