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
"""Unit tests for the shared optimum decoder mixin.

``pad_cache_slot_ids`` belongs to no single model: the sliding-window,
EXAONE-4.5 and Gemma3 wrappers all pad their decode cache slot ids through it.
"""

import torch

from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumDecoderMixin


def test_padding_row_does_not_alias_a_real_slot():
    """A padding row must not point at a scheduled request's cache slot.

    One request is pinned to slot 0 and the decode batch is padded to four
    rows. The compiled graph computes every row, so a padding row that reused
    slot 0 would write into that request's per-sequence cache -- the naive
    "pad with 0" corrupts a live request.

    The function only sees the ids scheduled this step, so a slot held by a
    request that is running but unscheduled is outside what this can check.
    """
    padded = RBLNOptimumDecoderMixin.pad_cache_slot_ids(
        torch.tensor([0], dtype=torch.int16), 4
    )

    assert padded[0, 0] == 0  # the scheduled request keeps its slot
    assert 0 not in {int(padded[row, 0]) for row in range(1, 4)}
