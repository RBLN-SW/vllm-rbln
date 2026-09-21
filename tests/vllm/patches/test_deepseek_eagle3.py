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

"""The DeepSeek EAGLE3 head is named past the target's full depth, on every rank.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.model_executor.models.deepseek_eagle3 import DeepseekV2Eagle3Model

import vllm_rbln.patches.deepseek_eagle3 as patch_module

# The suite truncates models via VLLM_RBLN_NUM_HIDDEN_LAYERS and the patched
# get_pp_indices honors it; the split this reasons about is the real 61-layer one.
from vllm_rbln.patches.distributed_utils import (
    original_get_pp_indices as get_pp_indices,
)
from vllm_rbln.v1.worker.utils import pipeline_adjusted_layer_index

NUM_LAYERS = 61
PP_SIZE = 8
# An A.X-K2 target is a DSA model, so every target layer takes two KV slots (MLA
# plus the lightning indexer's key cache) and the head has to clear both.
NUM_ATTN_MODULE = 2


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

    monkeypatch.setattr(patch_module, "_orig_model_init", stub_init)
    return seen


# 7 and 8 are the per-rank counts on the 61-layer pp8 split, which is what upstream
# passes; 61 is the full depth. Only PP=1 makes the two agree.
@pytest.mark.parametrize("per_rank_layers", [7, 8, NUM_LAYERS])
def test_the_head_is_named_past_the_full_depth(per_rank_layers, captured):
    patch_module.patched_eagle3_deepseek_model_init(
        object(), vllm_config=_vllm_config(per_rank_layers), prefix=""
    )

    assert captured["start_layer_id"] == NUM_LAYERS


@pytest.mark.parametrize("rank", range(PP_SIZE))
def test_the_named_head_lands_after_every_target_layer(rank):
    # The point of the rename: RBLN's index rule routes a name at or past the full
    # depth to the end of the rank's compacted KV list. Upstream's per-rank name
    # would sort in among the target layers instead.
    start, end = get_pp_indices(NUM_LAYERS, rank, PP_SIZE)
    model_config = SimpleNamespace(
        get_layers_start_end_indices=lambda parallel_config: (start, end),
        get_total_num_hidden_layers=lambda: NUM_LAYERS,
    )

    head = pipeline_adjusted_layer_index(
        f"model.layers.{NUM_LAYERS}.self_attn.attn", model_config, None, NUM_ATTN_MODULE
    )
    targets = [
        pipeline_adjusted_layer_index(
            f"model.layers.{i}.self_attn.{module}",
            model_config,
            None,
            NUM_ATTN_MODULE,
        )
        for i in range(start, end)
        for module in ("attn", "indexer.k_cache")
    ]

    assert targets == list(range((end - start) * NUM_ATTN_MODULE))
    assert head == (end - start) * NUM_ATTN_MODULE


def test_the_patch_is_the_one_installed():
    assert (
        DeepseekV2Eagle3Model.__init__
        is patch_module.patched_eagle3_deepseek_model_init
    )
