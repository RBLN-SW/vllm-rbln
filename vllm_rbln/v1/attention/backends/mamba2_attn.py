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

from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)

from vllm_rbln.v1.attention.backends.mamba_attn import (
    RBLNBaseMambaAttentionMetadata,
    RBLNBaseMambaAttentionMetadataBuilder,
)


@register_backend(MambaAttentionBackendEnum.MAMBA2, is_mamba=True)
class RBLNMamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RBLN_MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RBLNMamba2AttentionMetadataBuilder"]:
        return RBLNMamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass(kw_only=True)
class RBLNMamba2AttentionMetadata(RBLNBaseMambaAttentionMetadata):
    valid_mask: torch.Tensor | None = None


class RBLNMamba2AttentionMetadataBuilder(RBLNBaseMambaAttentionMetadataBuilder):
    metadata_cls = RBLNMamba2AttentionMetadata

    def stage(  # type: ignore[override]
        self,
        metadata: Any,
        group_shared: dict[tuple, torch.Tensor] | None = None,
    ) -> None:
        super().stage(metadata, group_shared)
        metadata.valid_mask = self._stage(metadata.valid_mask, "valid_mask")

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
    ) -> RBLNMamba2AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            positions,
            batch_pad,
            is_prefill,
            back_pad=back_pad,
            num_accepted=num_accepted,
            state_load_pos=state_load_pos,
        )
        valid_mask = None
        if common.num_prefills > 0:
            max_chunk = self.chunked_prefill_size
            valid_mask = torch.arange(max_chunk) < common.num_prefill_tokens
            valid_mask = valid_mask.to(torch.float32).view(1, 1, max_chunk)
        metadata = replace(common, valid_mask=valid_mask)
        if not defer_staging:
            self.stage(metadata, group_shared)
        return metadata
