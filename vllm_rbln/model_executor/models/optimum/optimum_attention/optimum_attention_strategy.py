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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Union

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class HybridAttentionImageEntry:
    pad_len: int
    attention_mask: torch.Tensor


EntryT = TypeVar("EntryT", bound=HybridAttentionImageEntry)
Result1T = TypeVar("Result1T")
Result2T = TypeVar("Result2T")


class AttentionStrategy(ABC, Generic[EntryT, Result1T, Result2T]):
    def __init__(self):
        self.table: dict[str, EntryT] = {}

    @abstractmethod
    def add(
        self,
        running_requests_id: str,
        pad_len: int,
        attention_mask: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def get(self, running_requests_ids: list[str]) -> Result1T: ...

    @abstractmethod
    def preprocess(
        self,
        cache_positions: torch.Tensor,
        decoder_batch_size: int,
        pad_lens: list[int],
        attention_masks: list[torch.Tensor],
    ) -> Result2T: ...

    def pop(self, request_id: str) -> None:
        self.table.pop(request_id, None)

    def clear(self):
        self.table.clear()

    def pad_to_2d(
        self,
        original_values: Union[list[int], list[torch.Tensor], torch.Tensor],
        rows: int,
        cols: int,
        pad_value: int = 0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if isinstance(original_values, list) and original_values:
            original_value = original_values[0]
            if isinstance(original_value, int):
                dtype = torch.int16 if dtype is None else dtype
                valid_nums = len(original_values)
                padded = torch.full((rows, cols), pad_value, dtype=dtype)
                original_tensor = torch.tensor(original_values, dtype=dtype).unsqueeze(
                    1
                )
            elif isinstance(original_value, torch.Tensor):
                dtype = original_value.dtype if dtype is None else dtype
                valid_nums = len(original_values)
                padded = torch.full((rows, cols), pad_value, dtype=dtype)
                original_tensor = torch.cat(original_values)
            else:
                raise RuntimeError("Invalid type of input.")

        elif isinstance(original_values, torch.Tensor):
            original_tensor = original_values
            dtype = original_tensor.dtype
            valid_nums = original_tensor.shape[0]
            padded = torch.full((rows, cols), pad_value, dtype=dtype)
        else:
            raise RuntimeError("Invalid type of input.")

        padded[:valid_nums] = original_tensor
        return padded


HybridR1 = tuple[list[int], list[torch.Tensor]]
HybridR2 = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HybridAttentionImageStrategy(
    AttentionStrategy[HybridAttentionImageEntry, HybridR1, HybridR2]
):
    def add(
        self,
        running_requests_id: str,
        pad_len: int,
        attention_mask: torch.Tensor,
    ) -> None:
        self.table[running_requests_id] = HybridAttentionImageEntry(
            pad_len=pad_len,
            attention_mask=attention_mask,
        )

    def get(self, running_requests_ids: list[str]) -> HybridR1:
        pad_lens: list[int] = []
        attention_masks: list[torch.Tensor] = []
        for request_id in running_requests_ids:
            entry = self.table[request_id]
            pad_lens.append(entry.pad_len)
            attention_masks.append(entry.attention_mask)
        return pad_lens, attention_masks

    def preprocess(
        self,
        cache_positions: torch.Tensor,
        decoder_batch_size: int,
        pad_lens: list[int],
        attention_masks: list[torch.Tensor],
    ) -> HybridR2:
        padded_pad_len = self.pad_to_2d(pad_lens, decoder_batch_size, 1, 0)
        padded_cache_positions = self.pad_to_2d(
            cache_positions, decoder_batch_size, 1, 0
        )
        padded_attention_mask = self.pad_to_2d(
            attention_masks, decoder_batch_size, attention_masks[0].shape[1], 0
        )
        # FIXME remove the logic
        position_ids = padded_cache_positions
        padded_cache_positions = position_ids + padded_pad_len

        return (
            padded_cache_positions,
            position_ids,
            padded_attention_mask,
        )

    def update_hybrid_attention_table(
        self, running_requests_ids: list[str], attention_mask: torch.Tensor
    ) -> None:
        """
        Update the sliding window table with a new attention mask.
        """
        for idx, request_id in enumerate(running_requests_ids):
            self.table[request_id].attention_mask = attention_mask[idx : idx + 1]

    def update_attention_mask(
        self, attention_mask: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        """
        To enable attention for the newly generated tokens,
        set their corresponding `cache_position` values
        in the `attention_mask` to 1.
        """

        rows = torch.arange(attention_mask.shape[0])
        cols = cache_position.squeeze(1)

        attention_mask[rows, cols] = 1
        return attention_mask
