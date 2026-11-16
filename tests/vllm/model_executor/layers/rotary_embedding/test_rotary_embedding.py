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

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.patches.rotary_embedding import patched_rope_forward_oot


@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("is_neox_style", [False, True])
def test_a_query_alone_rotates_as_it_does_beside_a_key(rotary_dim, is_neox_style):
    table = torch.linspace(-1.0, 1.0, 16 * rotary_dim).reshape(16, rotary_dim)
    rope = SimpleNamespace(
        head_size=64,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
        cos_cache=table.cos(),
        sin_cache=table.sin(),
    )
    positions = torch.arange(4).view(1, 4)
    query = torch.linspace(-2.0, 2.0, 4 * 2 * 64).view(1, 4, 2 * 64)
    key = torch.linspace(2.0, -2.0, 4 * 64).view(1, 4, 64)
    alone, no_key = patched_rope_forward_oot(rope, positions, query.clone(), None)
    paired, _ = patched_rope_forward_oot(rope, positions, query.clone(), key)
    assert no_key is None
    torch.testing.assert_close(alone, paired)
