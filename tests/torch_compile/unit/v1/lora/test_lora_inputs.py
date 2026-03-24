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

from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.v1.worker.rbln_model_runner import create_sampler_indices_padded

STAGES = [True, False]  # prefill(True) stage and decode(False) stage


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


def test_lora_inputs():
    LoRAInputs.set_sampler_indices_padded(torch.randn(10, 10))
    sampler_indices_padded = LoRAInputs.get_sampler_indices_padded()

    assert sampler_indices_padded.shape[0] == 10
    assert sampler_indices_padded.shape[1] == 10


@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("stage", STAGES)
def test_create_sampler_indices_padded(num_loras, stage):
    max_num_seqs = 8
    max_loras = 8

    id_to_index = get_random_id_to_index(num_loras, max_loras)
    lora_ids = torch.randint(0, num_loras + 1, (1 if stage else max_num_seqs,)).tolist()

    if stage and len(lora_ids) > 1:
        # the case that the number of sequences is greater than 1
        # and current stage is prefill.
        with pytest.raises(AssertionError):
            sampler_indices_padded = create_sampler_indices_padded(
                lora_ids, id_to_index, max_num_seqs, stage, max_loras, "cpu"
            )
    else:
        sampler_indices_padded = create_sampler_indices_padded(
            lora_ids, id_to_index, max_num_seqs, stage, max_loras, "cpu"
        )

        assert sampler_indices_padded.dtype == torch.long
        assert len(sampler_indices_padded) == (1 if stage else max_num_seqs)

        for i in range(len(lora_ids)):
            if lora_ids[i] > 0:
                index = id_to_index.index(lora_ids[i])
            else:
                index = max_loras if stage else max_num_seqs

            expected_value = i + (index * len(lora_ids))
            assert sampler_indices_padded[i] == expected_value
