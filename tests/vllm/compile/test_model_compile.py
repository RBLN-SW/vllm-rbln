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

# Whole-model compile-and-run smoke on a real NPU: each model compiles and
# generates tokens.

import os

import pytest

from tests.vllm.compile.models import MODELS
from tests.vllm.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.vllm.runners import DPRequest
from tests.vllm.utils import devices_needed, rbln_device_count

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_TOKENS = 8

# Compiling a whole model outlasts any default timeout; thread method because
# SIGALRM cannot fire while the compiler is down in C.
_TIMEOUT_S = int(os.environ.get("VLLM_RBLN_TEST_COMPILE_TIMEOUT_S", 3600))


def _generate(spec: CompileModelSpec, vllm_runner, async_vllm_runner) -> list:
    """One output per DP rank, or one output for a non-DP spec. DP has to go
    through AsyncLLM: the sync LLM rejects data_parallel_size > 1."""
    dp = spec.engine_kwargs.get("data_parallel_size", 1)
    if dp == 1:
        with vllm_runner(spec.model, **spec.engine_kwargs) as model:
            return model.generate_greedy([PROMPT], MAX_TOKENS)
    with async_vllm_runner(spec.model, **spec.engine_kwargs) as model:
        return model.generate_greedy(
            [DPRequest(PROMPT, MAX_TOKENS, dp_rank=rank) for rank in range(dp)]
        )


@pytest.mark.model_compile
@pytest.mark.timeout(_TIMEOUT_S, method="thread")
@pytest.mark.parametrize("spec", spec_params(MODELS))
def test_compile_and_generate(
    vllm_runner, async_vllm_runner, spec: CompileModelSpec, monkeypatch
) -> None:
    needed = devices_needed(spec.engine_kwargs, spec.rsd)
    available = rbln_device_count()
    if available < needed:
        pytest.skip(f"{spec.test_id} needs {needed} NPUs, host has {available}")

    apply_spec_envs(spec, monkeypatch)
    outputs = _generate(spec, vllm_runner, async_vllm_runner)

    assert len(outputs) == spec.engine_kwargs.get("data_parallel_size", 1)
    for rank, (token_ids, _text) in enumerate(outputs):
        assert token_ids, f"{spec.test_id} rank{rank} compiled but generated no tokens"


@pytest.mark.model_compile
@pytest.mark.timeout(_TIMEOUT_S, method="thread")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_minimax_m3_small_config_compiles_and_generates(
    vllm_runner, tmp_path, kv_cache_dtype
):
    from vllm import SamplingParams
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    config = MiniMaxM3TextConfig(
        architectures=["MiniMaxM3SparseForCausalLM"],
        vocab_size=256,
        hidden_size=512,
        intermediate_size=256,
        dense_intermediate_size=512,
        shared_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        num_mtp_modules=0,
        moe_layer_freq=[0, 1],
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.1.block_sparse_moe.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            },
        },
    )
    config.sparse_attention_config.update(
        sparse_num_index_heads=1,
        sparse_attention_freq=[0, 1],
        sparse_disable_index_value=[0, 1],
    )
    config.save_pretrained(tmp_path)

    # SwiGLU-OAI is supported by the compiler's packed NVFP4 MoE kernel,
    # not its unquantized MoE kernel. Dense/attention weights stay in BF16.
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        block_size=1024,
        num_gpu_blocks_override=5,
        kv_cache_dtype=kv_cache_dtype,
    ) as model:
        outputs = model.llm.generate(
            [{"prompt_token_ids": [1] * 140}, {"prompt_token_ids": [2] * 270}],
            SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True),
        )

    assert len(outputs) == 2
    assert all(len(output.outputs[0].token_ids) == MAX_TOKENS for output in outputs)
