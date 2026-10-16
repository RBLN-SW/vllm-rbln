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

"""A.X-K3 config: the DSA index-share rule and expert-group routing."""

from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from vllm_rbln.patches.axk3.config import AXK3Config


def _config(**kwargs) -> AXK3Config:
    return AXK3Config(num_hidden_layers=12, index_topk=2048, **kwargs)


@pytest.mark.parametrize(
    ("freq", "offset", "computing"),
    [
        # The A.X-K3 checkpoint: layers 1, 5, 9 (1-indexed) own an indexer.
        (4, 1, [0, 4, 8]),
        # vLLM's DeepSeek-V3.2 default offset: the first two layers compute.
        (4, 2, [0, 1, 5, 9]),
        # The converter writes max(offset, 1), so 0 behaves like 1.
        (4, 0, [0, 4, 8]),
    ],
)
def test_index_share_layers(freq, offset, computing):
    config = _config(index_topk_freq=freq, index_skip_topk_offset=offset)

    owners = [i for i in range(12) if not config.is_index_skip_layer(i)]
    assert owners == computing
    for layer in range(12):
        source = config.index_source_layer(layer)
        assert source in computing and source <= layer
        assert source == max(c for c in computing if c <= layer)


def test_every_layer_owns_an_indexer_without_index_share():
    config = _config()

    assert not any(config.is_index_skip_layer(i) for i in range(12))


def test_a_dense_config_has_no_skip_layer():
    config = AXK3Config(num_hidden_layers=12, index_topk_freq=4)

    assert not any(config.is_index_skip_layer(i) for i in range(12))


def test_a_single_expert_group_routes_as_plain_topk():
    # The grouped path's single-group mask scatter does not lower on RBLN.
    config = _config(n_group=1, topk_group=1)

    assert config.n_group is None and config.topk_group is None


def test_several_expert_groups_keep_grouped_routing():
    config = _config(n_group=8, topk_group=4)

    assert (config.n_group, config.topk_group) == (8, 4)


def test_config_unpickles_where_vllm_rbln_has_not_been_imported(tmp_path):
    # Same constraint as AXK2Config: a spawned engine core unpickles the config
    # before the plugin is loaded there.
    blob = tmp_path / "axk3_config.pkl"
    blob.write_bytes(pickle.dumps(_config(index_topk_freq=4)))

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import pickle; pickle.loads(open({str(blob)!r}, 'rb').read())",
        ],
        capture_output=True,
        text=True,
    )

    assert child.returncode == 0, child.stderr
