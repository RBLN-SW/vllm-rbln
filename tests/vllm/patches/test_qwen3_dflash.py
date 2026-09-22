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

"""The DFlash head is named past the target's full depth, on every rank.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

import vllm_rbln.patches.qwen3_dflash as patch_module

# The suite truncates models via VLLM_RBLN_NUM_HIDDEN_LAYERS and the patched
# get_pp_indices honors it; the split this reasons about is the real 62-layer one.
from vllm_rbln.patches.distributed_utils import (
    original_get_pp_indices as get_pp_indices,
)
from vllm_rbln.v1.worker.utils import pipeline_adjusted_layer_index

# A depth and split where the per-rank count differs from the total, so the two
# naming rules disagree. The head's own depth is its config's
# `num_hidden_layers`; 5 stands in for a head deeper than EAGLE3's one layer.
NUM_LAYERS = 62
PP_SIZE = 4
DRAFT_LAYERS = 5


def _vllm_config(per_rank_layers: int):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: NUM_LAYERS,
            get_num_layers=lambda parallel_config: per_rank_layers,
        ),
    )


@pytest.fixture
def captured(monkeypatch):
    """Record what the patch hands upstream instead of running upstream."""
    seen = {}

    def stub_init(model, *, vllm_config, start_layer_id=0, prefix=""):
        seen["start_layer_id"] = start_layer_id
        seen["prefix"] = prefix

    monkeypatch.setattr(patch_module, "_orig_model_init", stub_init)
    return seen


# The per-rank counts this split gives, which is what upstream passes, and the
# full depth. Only PP=1 makes the two agree.
@pytest.mark.parametrize("per_rank_layers", [15, 16, NUM_LAYERS])
def test_the_head_is_named_past_the_full_depth(per_rank_layers, captured):
    patch_module.patched_dflash_qwen3_model_init(
        object(), vllm_config=_vllm_config(per_rank_layers), prefix=""
    )

    assert captured["start_layer_id"] == NUM_LAYERS


def test_the_prefix_reaches_upstream(captured):
    # Upstream builds every weight and KV name under it, including the head's
    # own layers -- the names this patch is here to place. A value the stub
    # would also produce by defaulting cannot tell forwarding from silence.
    patch_module.patched_dflash_qwen3_model_init(
        object(), vllm_config=_vllm_config(15), prefix="model.draft"
    )

    assert captured["prefix"] == "model.draft"


def test_the_patch_is_the_one_installed():
    # Every other case here calls the replacement directly, so none of them
    # would notice the registration failing to take.
    assert DFlashQwen3Model.__init__ is patch_module.patched_dflash_qwen3_model_init


@pytest.mark.parametrize("rank", range(PP_SIZE))
def test_the_named_band_lands_after_every_target_layer(rank):
    # The point of the rename: RBLN's index rule routes a name at or past the
    # full depth to the end of the rank's compacted KV list. This head is a band
    # of names, so the band has to land there contiguously and in order.
    start, end = get_pp_indices(NUM_LAYERS, rank, PP_SIZE)
    model_config = SimpleNamespace(
        get_layers_start_end_indices=lambda parallel_config: (start, end),
        get_total_num_hidden_layers=lambda: NUM_LAYERS,
    )

    band = [
        pipeline_adjusted_layer_index(
            f"model.layers.{NUM_LAYERS + i}.self_attn.attn", model_config, None, 1
        )
        for i in range(DRAFT_LAYERS)
    ]
    targets = [
        pipeline_adjusted_layer_index(
            f"model.layers.{i}.self_attn.attn", model_config, None, 1
        )
        for i in range(start, end)
    ]

    assert targets == list(range(end - start))
    assert band == list(range(end - start, end - start + DRAFT_LAYERS))
