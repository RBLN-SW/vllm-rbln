# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Generic, TypeVar

import torch
from vllm.logger import init_logger

from .optimum_attention_strategy import (
    AttentionStrategy,
    EntryT,
    HybridAttentionImageEntry,
    HybridAttentionImageStrategy,
    HybridR1,
    HybridR2,
    Result1T,
    Result2T,
)

logger = init_logger(__name__)
StrategyT = TypeVar("StrategyT", bound=AttentionStrategy[Any, Any, Any])


class AttentionManager(Generic[StrategyT, EntryT, Result1T, Result2T]):
    def __init__(self, strategy: StrategyT):
        self._s: StrategyT = strategy

    def add(self, running_requests_id: str, local_table_id: int, **kwargs) -> None:
        self._s.add(running_requests_id, local_table_id, **kwargs)

    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> Any:
        return self._s.get(
            is_prompt,
            decoder_batch_size,
            running_requests_ids,
            finished_requests_ids,
            **kwargs,
        )

    def preprocess(
        self,
        local_block_table_ids: list[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> Any:
        return self._s.preprocess(
            local_block_table_ids,
            cache_positions,
            request_nums,
            decoder_batch_size,
            **kwargs,
        )

    def clear(self):
        self._s.clear()

    def pop(self, request_id: str):
        self._s.pop(request_id)


class HybridAttentionImageManager(
    AttentionManager[
        HybridAttentionImageStrategy, HybridAttentionImageEntry, HybridR1, HybridR2
    ]
):
    def update(
        self,
        running_requests_ids: list[str],
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        updated_attention_mask = self._s.update_attention_mask(
            attention_mask,
            cache_position,
        )
        self._s.update_hybrid_attention_table(
            running_requests_ids,
            updated_attention_mask,
        )
        return updated_attention_mask
