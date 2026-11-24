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
from unittest.mock import patch

import pytest

from vllm_rbln.model_executor.layers.quantization.nvfp4 import (
    RBLNModelOptNvFp4Config,
)
from vllm_rbln.model_executor.models import inkling
from vllm_rbln.model_executor.models.inkling import RBLNInklingForCausalLM

MAPPER = RBLNInklingForCausalLM.hf_to_vllm_mapper
EXPERTS = "model.llm.layers.3.mlp.experts."
ROUTED = "model.layers.3.mlp.experts.routed_experts."


@pytest.mark.parametrize(
    ("checkpoint", "param"),
    [
        (EXPERTS + "w13_weight", ROUTED + "w13_weight"),
        (EXPERTS + "w13_weight.scale", ROUTED + "w13_weight_scale"),
        (EXPERTS + "w2_weight.scale", ROUTED + "w2_weight_scale"),
        (EXPERTS + "w13_weight.scale2", ROUTED + "w13_weight_scale_2"),
        (EXPERTS + "w2_weight.scale2", ROUTED + "w2_weight_scale_2"),
        (EXPERTS + "w13_weight.input_amax", ROUTED + "w13_input_scale"),
        (EXPERTS + "w2_weight.input_amax", ROUTED + "w2_input_scale"),
        (EXPERTS + "w13_weight.original_shape", None),
    ],
)
def test_nvfp4_expert_tensors_map_onto_routed_expert_params(checkpoint, param):
    assert MAPPER._map_name(checkpoint) == param


def test_modelopt_exclusions_reach_the_rbln_modules():
    # Entries as Inkling-Small-NVFP4's hf_quant_config.json writes them.
    config = RBLNModelOptNvFp4Config(
        "NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=[
            "model.llm.unembed",
            "model.llm.layers.0.attn",
            "model.llm.layers.0.mlp.w13_dn",
            "model.llm.layers.0.mlp.w2_md",
            "model.llm.layers.2.mlp.experts",
            "model.llm.layers.3.attn",
            "model.llm.layers.3.mlp.gate",
            "model.llm.layers.3.mlp.shared_experts",
        ],
    )
    config.apply_vllm_mapper(MAPPER)

    for prefix in (
        "lm_head",
        "model.layers.0.attn.q_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.2.mlp.experts",
        "model.layers.3.attn.r_proj",
        "model.layers.3.mlp.gate",
        "model.layers.3.mlp.shared_experts.1.gate_up_proj",
    ):
        assert config.is_layer_excluded(prefix), prefix
    assert not config.is_layer_excluded("model.layers.3.mlp.experts")


def test_speculative_decoding_is_refused():
    # The runner replays accepted drafts only for MambaSpec groups, which
    # Inkling's rings are not, so nothing downstream would catch it.
    config = SimpleNamespace(
        use_sconv=True,
        use_embed_norm=True,
        q_bias=False,
        o_bias=False,
        final_logit_softcapping=None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        speculative_config=SimpleNamespace(method="ngram"),
    )
    with (
        patch.object(
            inkling, "get_pp_group", return_value=SimpleNamespace(world_size=1)
        ),
        pytest.raises(NotImplementedError, match="speculative decoding"),
    ):
        RBLNInklingForCausalLM(vllm_config=vllm_config)
