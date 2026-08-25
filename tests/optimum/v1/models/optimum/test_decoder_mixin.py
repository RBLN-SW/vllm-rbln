# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumDecoderMixin


def test_padding_row_does_not_alias_a_real_slot():
    padded = RBLNOptimumDecoderMixin.pad_cache_slot_ids(
        torch.tensor([0], dtype=torch.int16), 4
    )

    assert padded[0, 0] == 0  # the scheduled request keeps its slot
    assert 0 not in {int(padded[row, 0]) for row in range(1, 4)}
