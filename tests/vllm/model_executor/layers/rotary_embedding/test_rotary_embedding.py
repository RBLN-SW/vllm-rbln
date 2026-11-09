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

import torch

from vllm_rbln.patches.rotary_embedding import patched_rope_forward_oot

_MAX_POS, _HEAD = 16, 64


def _rope():
    table = torch.linspace(-1.0, 1.0, _MAX_POS * _HEAD).reshape(_MAX_POS, _HEAD)
    return SimpleNamespace(
        head_size=_HEAD,
        rotary_dim=_HEAD,
        is_neox_style=True,
        cos_cache=table.cos(),
        sin_cache=table.sin(),
    )


def test_a_query_alone_rotates_as_it_does_beside_a_key():
    # A KV-shared layer (Gemma4 MTP) has no key to rotate.
    rope = _rope()
    positions = torch.arange(4).view(1, 4)
    query = torch.linspace(-2.0, 2.0, 4 * 2 * _HEAD).view(1, 4, 2 * _HEAD)
    key = torch.linspace(2.0, -2.0, 4 * _HEAD).view(1, 4, _HEAD)

    alone, no_key = patched_rope_forward_oot(rope, positions, query.clone(), None)
    paired, _ = patched_rope_forward_oot(rope, positions, query.clone(), key)

    assert no_key is None
    torch.testing.assert_close(alone, paired)
