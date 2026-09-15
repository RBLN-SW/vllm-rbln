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

"""VLLM_RBLN_NUM_HIDDEN_LAYERS truncation, driven through the real
``make_layers`` with a stub layer type -- no checkpoint, no device."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import DeepseekV3Config
from vllm.distributed import parallel_state
from vllm.distributed import utils as distributed_utils
from vllm.model_executor.models import utils as models_utils
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
)

from vllm_rbln import patches
from vllm_rbln.patches.distributed_utils import patched_get_pp_indices
from vllm_rbln.patches.registry import get_registered_patch_descriptors

TOTAL_LAYERS = 32
LIMIT = 3


class StubLayer(nn.Module):
    """Stands in for a decoder layer: one named parameter, nothing else."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(torch.zeros(1))


class StubModel(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    """Install the replacement directly, so these pass however the session was
    started (--num-hidden-layers 0 never applies it). That the registry wires it
    up for real is TestPatchWiring's job."""
    monkeypatch.setattr(distributed_utils, "get_pp_indices", patched_get_pp_indices)
    # get_pp_missing_layer_names caches on id(model); a freed model from an
    # earlier test can hand its id -- and its entry -- to ours.
    monkeypatch.setattr(models_utils, "_model_to_pp_missing_layer_names", {})


def build(
    monkeypatch,
    limit: int,
    *,
    total: int = TOTAL_LAYERS,
    pp_rank: int = 0,
    pp_size: int = 1,
) -> tuple[int, int, nn.ModuleList]:
    monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", str(limit))
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: type("PP", (), {"rank_in_group": pp_rank, "world_size": pp_size})(),
    )
    return make_layers(total, StubLayer, prefix="layers")


class TestPatchWiring:
    """Without this, a typo in the target or a module missing from
    patches/__init__ would leave every other test here passing."""

    def test_listed_in_the_patch_package(self):
        assert hasattr(patches, "distributed_utils")

    def test_registered_against_get_pp_indices(self):
        [descriptor] = [
            d
            for d in get_registered_patch_descriptors()
            if d.replacement is patched_get_pp_indices
        ]
        assert descriptor.target == "vllm.distributed.utils.get_pp_indices"

    def test_gated_on_the_variable(self, monkeypatch):
        [descriptor] = [
            d
            for d in get_registered_patch_descriptors()
            if d.replacement is patched_get_pp_indices
        ]
        monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", "0")
        assert not descriptor.condition()
        monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", str(LIMIT))
        assert descriptor.condition()


def test_builds_only_the_first_n_layers(monkeypatch):
    start, end, layers = build(monkeypatch, LIMIT)

    assert (start, end) == (0, LIMIT)
    # The list keeps its full length: indices still address the checkpoint's.
    assert len(layers) == TOTAL_LAYERS
    assert [type(layer) for layer in layers[:LIMIT]] == [StubLayer] * LIMIT
    assert {type(layer) for layer in layers[LIMIT:]} == {PPMissingLayer}


def test_dropped_layers_read_as_missing_to_the_weight_loaders(monkeypatch):
    """The half that replaces the loader override: every upstream loader asks
    is_pp_missing_parameter before touching a weight."""
    model = StubModel(build(monkeypatch, LIMIT)[2])

    assert not is_pp_missing_parameter(f"layers.{LIMIT - 1}.weight", model)
    assert is_pp_missing_parameter(f"layers.{LIMIT}.weight", model)
    assert is_pp_missing_parameter(f"layers.{TOTAL_LAYERS - 1}.weight", model)


def test_a_model_shorter_than_the_limit_is_left_alone(monkeypatch):
    start, end, layers = build(monkeypatch, TOTAL_LAYERS * 2)

    assert (start, end) == (0, TOTAL_LAYERS)
    assert PPMissingLayer not in {type(layer) for layer in layers}


def test_zero_disables_the_truncation(monkeypatch):
    start, end, layers = build(monkeypatch, 0)

    assert (start, end) == (0, TOTAL_LAYERS)
    assert PPMissingLayer not in {type(layer) for layer in layers}


def test_pipeline_stages_split_the_kept_layers(monkeypatch):
    """Spread rather than piled on rank 0: no stage may end up empty, or its
    attention-free spec caps every other stage at one KV block."""
    stages = [
        build(monkeypatch, LIMIT, pp_rank=rank, pp_size=2)[:2] for rank in range(2)
    ]

    assert all(end > start for start, end in stages)
    assert stages == [(0, 2), (2, LIMIT)]


def test_fewer_layers_than_stages_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="pipeline_parallel_size=4"):
        build(monkeypatch, LIMIT, pp_size=4)


# DeepSeek-V3's published config; DeepseekV3Config itself carries no
# num_nextn_predict_layers, since HF's own implementation has no MTP.
DEEPSEEK_V3 = dict(
    num_hidden_layers=61, num_nextn_predict_layers=1, first_k_dense_replace=3
)
MTP_WEIGHT = "model.layers.61.mtp_block.self_attn.o_proj.weight"


def deepseek_v3_config(**overrides) -> DeepseekV3Config:
    return DeepseekV3Config(**{**DEEPSEEK_V3, **overrides})


class TestDeepSeekV3MTP:
    """The MTP block is built outside make_layers, at an index taken from
    config.num_hidden_layers -- which this truncation deliberately never sets."""

    def test_mtp_sits_past_the_kept_layers_and_the_stack(self, monkeypatch):
        config = deepseek_v3_config()
        _, end, layers = build(monkeypatch, LIMIT, total=config.num_hidden_layers)
        mtp_index = get_spec_layer_idx_from_weight_name(config, MTP_WEIGHT)

        # Where DeepSeekMultiTokenPredictor keys its block and where the loader
        # looks for its weights -- one index, past the whole decoder stack.
        assert mtp_index == config.num_hidden_layers == len(layers) == 61
        assert mtp_index >= end

    def test_a_kept_layer_is_not_claimed_by_the_drafter(self, monkeypatch):
        config = deepseek_v3_config()
        build(monkeypatch, LIMIT, total=config.num_hidden_layers)

        for index in range(LIMIT):
            name = f"model.layers.{index}.mlp.gate_proj.weight"
            assert get_spec_layer_idx_from_weight_name(config, name) is None

    def test_overriding_the_config_would_aim_the_mtp_at_a_real_layer(self):
        """Why the truncation moved off hf_overrides: shrinking the config makes
        the drafter claim layer 3 -- a real decoder layer -- and drops the
        checkpoint's own MTP weights."""
        overridden = deepseek_v3_config(num_hidden_layers=LIMIT)

        assert (
            get_spec_layer_idx_from_weight_name(
                overridden, f"model.layers.{LIMIT}.mlp.gate_proj.weight"
            )
            == LIMIT
        )
        assert get_spec_layer_idx_from_weight_name(overridden, MTP_WEIGHT) is None
