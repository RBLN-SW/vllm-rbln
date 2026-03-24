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

import pytest
import torch

from vllm_rbln.lora.mask import LoRAMask
from vllm_rbln.v1.worker.rbln_model_runner import create_lora_mask


def get_random_id_to_index(
    num_loras: int, num_slots: int, log: bool = True
) -> list[int | None]:
    """Creates a random lora_id_to_index mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping.
                   Must be larger than num_loras.
        log: Whether to log the output.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots."
        )

    slots: list[int | None] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def test_lora_mask():
    LoRAMask.set_lora_mask(torch.randn(10, 10))
    mask = LoRAMask.get_lora_mask()

    assert mask.shape[0] == 10
    assert mask.shape[1] == 10


@pytest.mark.parametrize("num_seqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 256])
@pytest.mark.parametrize("num_loras", [1, 2, 4])
def test_create_lora_mask(num_seqs, seq_len, num_loras):
    max_loras = 8
    max_lora_rank = 8
    lora_dtype = torch.bfloat16

    input_ids = torch.randint(0, 6000, (num_seqs, seq_len), dtype=torch.int64)
    id_to_index = get_random_id_to_index(num_loras, max_loras)
    lora_ids = torch.randint(0, num_loras + 1, (num_seqs,)).tolist()

    lora_mask = create_lora_mask(
        input_ids,
        lora_ids,
        id_to_index,
        max_loras,
        max_lora_rank,
        lora_dtype=lora_dtype,
        device=input_ids.device,
    )

    expected_shape = (num_seqs * seq_len, max_loras * max_lora_rank)
    assert lora_mask.shape == expected_shape
    assert lora_mask.dtype == lora_dtype

    for i, lora_id in enumerate(lora_ids):
        if lora_id == 0:
            continue

        lora_index = id_to_index.index(lora_id)
        start_row = i * seq_len
        start_col = lora_index * max_lora_rank

        mask_section = lora_mask[
            start_row : start_row + seq_len, start_col : start_col + max_lora_rank
        ]
        assert torch.all(mask_section == 1.0)
