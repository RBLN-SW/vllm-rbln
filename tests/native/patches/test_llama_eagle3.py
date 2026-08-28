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

"""The EAGLE3 head is named past the target's full depth, on every rank.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

import pytest
from vllm.model_executor.models.llama_eagle3 import LlamaModel as Eagle3LlamaModel

import vllm_rbln.patches.llama_eagle3 as patch_module

# The suite truncates models via VLLM_RBLN_NUM_HIDDEN_LAYERS and the patched
# get_pp_indices honors it; the split this reasons about is the real 62-layer one.
from vllm_rbln.patches.distributed_utils import (
    original_get_pp_indices as get_pp_indices,
)
from vllm_rbln.v1.worker.utils import pipeline_adjusted_layer_index

NUM_LAYERS = 62
PP_SIZE = 4


def _vllm_config(per_rank_layers: int):
    """A config shaped like the one the head's __init__ actually reads.

    `model_config` is the target's and `speculative_config.draft_model_config` the
    draft's -- the two the upstream __init__ mixes, and the reason a per-rank count
    can pass for the full depth.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: NUM_LAYERS,
            get_num_layers=lambda parallel_config: per_rank_layers,
            hf_text_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS),
        ),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=1),
            ),
        ),
    )


@pytest.fixture
def captured(monkeypatch):
    """Record what the patch hands upstream instead of running upstream."""
    seen = {}

    def stub_init(model, *, vllm_config, start_layer_id=0, prefix=""):
        # Upstream builds the layers here, so anything the layers read has to be
        # set on the config before this point.
        seen["start_layer_id"] = start_layer_id
        seen["target_layer_count"] = getattr(
            vllm_config.speculative_config.draft_model_config.hf_config,
            "target_layer_count",
            None,
        )

    monkeypatch.setattr(patch_module, "_orig_model_init", stub_init)
    return seen


# 15 is stage 3's own layer count on the 62-layer PP4 split, which is what upstream
# passes; 62 is the full depth. Only the first rank makes the two agree.
@pytest.mark.parametrize("per_rank_layers", [15, 16, NUM_LAYERS])
def test_the_head_is_named_past_the_full_depth(per_rank_layers, captured):
    patch_module.patched_eagle3_llama_model_init(
        object(), vllm_config=_vllm_config(per_rank_layers), prefix=""
    )

    assert captured["start_layer_id"] == NUM_LAYERS


@pytest.mark.parametrize("per_rank_layers", [15, 16, NUM_LAYERS])
def test_target_layer_count_stays_paired_with_the_name(per_rank_layers, captured):
    # `llama.py` recovers a draft-relative index as `extract_layer_index(prefix) -
    # target_layer_count`, i.e. off the name, not off the loop counter. Raising the
    # name alone would give stage 3 `62 - 15 = 47` and trip the layer_types assert.
    patch_module.patched_eagle3_llama_model_init(
        object(), vllm_config=_vllm_config(per_rank_layers), prefix=""
    )

    assert captured["target_layer_count"] == captured["start_layer_id"]


@pytest.mark.parametrize("rank", range(PP_SIZE))
def test_the_named_head_lands_after_every_target_layer(rank):
    # The point of the rename: RBLN's index rule routes a name at or past the full
    # depth to the end of the rank's compacted KV list. Upstream's per-rank name
    # would sort in among the target layers instead.
    from types import SimpleNamespace

    start, end = get_pp_indices(NUM_LAYERS, rank, PP_SIZE)
    model_config = SimpleNamespace(
        get_layers_start_end_indices=lambda parallel_config: (start, end),
        get_total_num_hidden_layers=lambda: NUM_LAYERS,
    )

    head = pipeline_adjusted_layer_index(
        f"model.layers.{NUM_LAYERS}.self_attn.attn", model_config, None, 1
    )
    targets = [
        pipeline_adjusted_layer_index(
            f"model.layers.{i}.self_attn.attn", model_config, None, 1
        )
        for i in range(start, end)
    ]

    assert targets == list(range(end - start))
    assert head == end - start


def test_the_patch_is_the_one_installed():
    assert Eagle3LlamaModel.__init__ is patch_module.patched_eagle3_llama_model_init
