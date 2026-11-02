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
from types import SimpleNamespace

import pytest
import torch

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


def test_axk3_registers_its_mla_convertor(monkeypatch):
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
    )

    from vllm_rbln.patches.axk3 import register_axk3
    from vllm_rbln.patches.model_arch_config import KvLoraRankMLAConvertor

    # The registration may already have run in this process, and re-running it
    # has to stay harmless; only the registry writes are undone per test.
    monkeypatch.setattr(ModelRegistry, "register_model", lambda *a, **k: None)
    monkeypatch.delitem(_CONFIG_REGISTRY, "axk3", raising=False)
    monkeypatch.delitem(MODEL_ARCH_CONFIG_CONVERTORS, "axk3", raising=False)

    register_axk3()

    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS["axk3"]
    assert convertor_cls is KvLoraRankMLAConvertor
    config = SimpleNamespace(model_type="axk3", kv_lora_rank=512)
    assert convertor_cls(config, config).is_deepseek_mla()


def test_forward_runs_only_the_layers_this_rank_built(monkeypatch):
    """`VLLM_RBLN_NUM_HIDDEN_LAYERS` builds the first N layers by shrinking the
    `end_layer` that `make_layers` sees; the list it returns still has one entry
    per layer, the rest `PPMissingLayer`. That placeholder returns its first
    argument, so a forward that walks the whole list hands `positions` on as the
    hidden state from layer N."""
    import vllm.distributed.parallel_state as parallel_state
    import vllm.distributed.utils as dist_utils
    from vllm.model_executor.models.utils import PPMissingLayer, make_layers

    import vllm_rbln.patches.axk3.model as axk3_model
    from vllm_rbln import envs
    from vllm_rbln.patches.distributed_utils import patched_get_pp_indices

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden_states, topk_by_layer):
            return hidden_states + 1, None

    monkeypatch.setattr(envs, "VLLM_RBLN_NUM_HIDDEN_LAYERS", 2)
    monkeypatch.setattr(dist_utils, "get_pp_indices", patched_get_pp_indices)
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)

    model = axk3_model.AXK3Model.__new__(axk3_model.AXK3Model)
    torch.nn.Module.__init__(model)
    model.num_streams = 1
    model.start_layer, model.end_layer, model.layers = make_layers(
        4, lambda prefix: Layer(), prefix="model.layers"
    )
    assert [type(layer) for layer in model.layers] == [
        Layer,
        Layer,
        PPMissingLayer,
        PPMissingLayer,
    ]
    model.embed_input_ids = lambda input_ids: torch.zeros(len(input_ids), 8)
    model.final_gated_residual = SimpleNamespace(read=lambda hidden: (hidden, None))

    out = model.forward(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        intermediate_tensors=None,
    )

    # Two built layers, each adding one; nothing from the unbuilt tail.
    torch.testing.assert_close(out, torch.full((3, 8), 2.0))


@pytest.mark.parametrize("skip_topk", [False, True])
def test_decoder_layer_returns_topk_without_writing_the_shared_dict(
    monkeypatch, skip_topk
):
    """The decoder layer may run as a nested compile region, which must not
    write the outer `topk_by_layer`. It returns the top-k it computed, or None
    for a skip layer that reuses its source layer's top-k."""
    import vllm_rbln.patches.axk3.model as axk3_model

    computed = torch.tensor([7])
    received = []

    def gated_mla_forward(mla, positions, hidden_states, topk_indices):
        received.append(topk_indices)
        return hidden_states, computed

    monkeypatch.setattr(axk3_model, "_gated_mla_forward", gated_mla_forward)
    layer = axk3_model.AXK3DecoderLayer.__new__(axk3_model.AXK3DecoderLayer)
    torch.nn.Module.__init__(layer)
    residual = SimpleNamespace(
        read=lambda hidden: (hidden, None), write=lambda res, x, gate: res + x
    )
    layer.attn_gated_residual = layer.mlp_gated_residual = residual
    layer.self_attn = SimpleNamespace(mla_attn=None)
    layer.mlp = lambda x: x
    layer.layer_idx, layer.index_source_layer, layer.skip_topk = 1, 0, skip_topk
    source = torch.tensor([3])
    topk_by_layer = {0: source}

    hidden, topk = layer(torch.arange(2), torch.ones(2, 4), topk_by_layer)

    assert list(topk_by_layer) == [0] and topk_by_layer[0] is source
    assert received[0] is (source if skip_topk else None)
    assert topk is (None if skip_topk else computed)
    torch.testing.assert_close(hidden, torch.full((2, 4), 4.0))


def test_eagle3_is_refused_on_both_aux_layer_setup_paths():
    """The runner resolves the aux layers from the speculative config, or from
    the model's default, then calls the setter; neither path may silently
    accept a target whose forward returns no aux hidden states."""
    from vllm.model_executor.models.interfaces import supports_eagle3

    import vllm_rbln.patches.axk3.model as axk3_model

    model = axk3_model.AXK3ForCausalLM.__new__(axk3_model.AXK3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(layers=[None] * 12, aux_hidden_state_layers=())

    # Inherited from the K2 base, which is why the runner reaches the setup at all.
    assert supports_eagle3(model)
    with pytest.raises(NotImplementedError, match="EAGLE3"):
        model.get_eagle3_default_aux_hidden_state_layers()
    with pytest.raises(NotImplementedError, match="EAGLE3"):
        model.set_aux_hidden_state_layers((2, 6, 9))
