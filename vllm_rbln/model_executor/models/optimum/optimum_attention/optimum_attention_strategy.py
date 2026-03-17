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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Union, cast

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class InnerAttentionEntry:
    local_table_id: int


@dataclass
class HybridAttentionImageEntry(InnerAttentionEntry):
    pad_len: int
    attention_mask: torch.Tensor


EntryT = TypeVar("EntryT", bound=InnerAttentionEntry)
Result1T = TypeVar("Result1T")
Result2T = TypeVar("Result2T")


class AttentionStrategy(ABC, Generic[EntryT, Result1T, Result2T]):
    def __init__(self):
        self.table: dict[str, EntryT] = {}

    @abstractmethod
    def add(self, running_requests_id: str, local_table_id: int, **kwargs) -> None: ...

    @abstractmethod
    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> Result1T: ...

    @abstractmethod
    def preprocess(
        self,
        local_block_table_ids: list[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> Result2T: ...

    def pop(self, request_id: str) -> None:
        self.table.pop(request_id, None)

    def clear(self):
        self.table.clear()

    def get_table_mapping_values(
        self,
        decoder_batch_size: int,
        is_prompt: bool,
        finished_requests_ids: list[str],  # FIXME to be removed
        running_requests_ids: list[str],
        get_entry_fn: Callable[[Any], Any] | None = None,
        get_extra_values_fn: Callable[[Any], Union[Any, tuple[Any, ...]]] | None = None,
    ) -> Union[list[int], tuple[list[int], ...]]:
        if is_prompt:
            used_ids = {
                get_entry_fn(v) if get_entry_fn else v for v in self.table.values()
            }
            available_ids = set(range(decoder_batch_size)) - used_ids
            assert available_ids, "No available table IDs"
            table_id = min(available_ids)
            return [table_id]

        table_ids = []
        extra_values = []

        for request_id in running_requests_ids:
            entry = self.table[request_id]
            table_id = get_entry_fn(entry) if get_entry_fn else entry
            table_ids.append(table_id)

            if get_extra_values_fn:
                result = get_extra_values_fn(entry)
                if not isinstance(result, tuple):
                    result = (result,)
                extra_values.append(result)

        if get_extra_values_fn:
            extra_values_lists: list[list[Any]] = [
                list(col) for col in zip(*extra_values)
            ]
            return (table_ids, *extra_values_lists)
        return table_ids

    def pad_list22dtensor(
        self,
        original_list: list[int],
        rows: int,
        cols: int,
        pad_value: int = 0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.int16
        valid_nums = len(original_list)
        padded = torch.full((rows, cols), pad_value, dtype=dtype)
        original_tensor = torch.tensor(original_list, dtype=dtype).unsqueeze(1)
        padded[:valid_nums] = original_tensor
        return padded

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


InnerR1 = list[int]
InnerR2 = tuple[torch.Tensor, torch.Tensor]


class InnerAttentionStrategy(AttentionStrategy[InnerAttentionEntry, InnerR1, InnerR2]):
    def add(
        self,
        running_requests_id: str,
        local_table_id: int,
        **kwargs,
    ) -> None:
        self.table[running_requests_id] = InnerAttentionEntry(
            local_table_id=local_table_id,
        )

    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> list[int]:
        result = self.get_table_mapping_values(
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
        )

        table_ids = cast(list[int], result)
        return table_ids

    def preprocess(
        self,
        local_block_table_ids: list[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Determine padding value for local_block_table_id
        used_ids = set(local_block_table_ids)
        pad_value = next((i for i in range(decoder_batch_size) if i not in used_ids), 0)

        padded_local_block_table_ids = self.pad_to_2d(
            local_block_table_ids, decoder_batch_size, 1, pad_value, torch.int16
        )
        padded_cache_positions = self.pad_to_2d(
            cache_positions, decoder_batch_size, 1, 0
        )

        return (
            padded_local_block_table_ids,
            padded_cache_positions,
        )


HybridR1 = tuple[list[int], list[int], list[torch.Tensor]]
HybridR2 = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class HybridAttentionImageStrategy(
    AttentionStrategy[HybridAttentionImageEntry, HybridR1, HybridR2]
):
    def __init__(self, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id

    def add(self, running_requests_id: str, local_table_id: int, **kwargs) -> None:
        pad_len: int | None = kwargs.get("pad_len")
        attention_mask: torch.Tensor | None = kwargs.get("attention_mask")
        assert pad_len is not None
        assert attention_mask is not None

        self.table[running_requests_id] = HybridAttentionImageEntry(
            local_table_id=local_table_id,
            pad_len=pad_len,
            attention_mask=attention_mask,
        )

    def get(
        self,
        is_prompt: bool,
        decoder_batch_size: int,
        running_requests_ids: list[str],
        finished_requests_ids: list[str],
        **kwargs,
    ) -> tuple[list[int], list[int], list[torch.Tensor]]:
        get_extra_values_fn = None
        input_ids: torch.Tensor | None = kwargs.get("input_ids")
        assert input_ids is not None

        if is_prompt:
            attention_mask = (input_ids != self.pad_token_id).to(torch.int64).squeeze(0)
        else:
            get_extra_values_fn = lambda entry: (
                entry.pad_len,
                entry.attention_mask,
            )

        result = self.get_table_mapping_values(
            decoder_batch_size,
            is_prompt,
            finished_requests_ids,
            running_requests_ids,
            get_entry_fn=lambda entry: entry.local_table_id,
            get_extra_values_fn=get_extra_values_fn,
        )

        if is_prompt:
            table_ids = cast(list[int], result)
            return table_ids, [], [attention_mask]
        else:
            result = cast(tuple[list[int], list[int], list[torch.Tensor]], result)
            table_ids, pad_lens, attention_masks = result
            return table_ids, pad_lens, attention_masks

    def preprocess(
        self,
        local_block_table_ids: list[int],
        cache_positions: torch.Tensor,
        request_nums: int,
        decoder_batch_size: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_lens: list[int] | None = kwargs.get("pad_lens")
        attention_masks: list[torch.Tensor] | None = kwargs.get("attention_masks")

        assert pad_lens is not None
        assert attention_masks is not None

        used_ids = set(local_block_table_ids)
        pad_value = next((i for i in range(decoder_batch_size) if i not in used_ids), 0)

        padded_local_block_table_ids = self.pad_to_2d(
            local_block_table_ids, decoder_batch_size, 1, pad_value, torch.int16
        )
        padded_pad_len = self.pad_to_2d(pad_lens, decoder_batch_size, 1, 0)
        padded_cache_positions = self.pad_to_2d(
            cache_positions, decoder_batch_size, 1, 0
        )
        padded_attention_mask = self.pad_to_2d(
            attention_masks, decoder_batch_size, attention_masks[0].shape[1], 0
        )

        # cache_positions:
        #  the index including padding between text and image
        # pad_lens:
        #   the size of padding
        # position_ids:
        #   the index of the token to be decoded in the sequence.
        position_ids = padded_cache_positions - padded_pad_len

        return (
            padded_local_block_table_ids,
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
