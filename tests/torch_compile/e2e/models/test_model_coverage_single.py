# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from .utils import patch_and_run


@pytest.fixture
def common_env():
    return {
        "RBLN_KERNEL_MODE": "triton",
        "VLLM_RBLN_USE_VLLM_MODEL": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
    }


@pytest.fixture
def prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


# for the sake of test environment and prevent downloading models twice,
# always use instruct/chat model if possible

# TODO: uncomment following models and adjust their arguments properly
# after SWA is implemented
# * mistralai/Mistral-7B-Instruct-v0.1
# * google/gemma-3-270m-it
# * google/gemma-3-1b-it
# * google/gemma-2-2b-it
targets = [
    # LlamaForCausalLM
    {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # ExaoneForCausalLM
    {
        "model": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
        "trust_remote_code": True,
    },
    # Qwen2ForCausalLM
    {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # MistralForCausalLM
    # {
    #     "model": "mistralai/Mistral-7B-Instruct-v0.1",
    #     "max_model_len": 2*4096,
    #     "block_size": 4096,
    #     "enable_chunked_prefill": True,
    #     "max_num_batched_tokens": 128,
    #     "max_num_seqs": 1,
    # },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
    },
    # GPT2LMHeadModel
    {
        "model": "openai-community/gpt2",
        "max_model_len": 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "openai-community/gpt2-medium",
        "max_model_len": 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "openai-community/gpt2-large",
        "max_model_len": 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "openai-community/gpt2-xl",
        "max_model_len": 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # Qwen3ForCausalLM
    {
        "model": "Qwen/Qwen3-0.6B",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "Qwen/Qwen3-1.7B",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "Qwen/Qwen3-4B",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # Gemma3ForCausalLM
    # {
    #     "model": "google/gemma-3-270m-it",
    #     "max_model_len": 4*1024,
    #     "block_size": 1024,
    #     "enable_chunked_prefill": True,
    #     "max_num_batched_tokens": 128,
    #     "max_num_seqs": 8,
    # },
    # {
    #     "model": "google/gemma-3-1b-it",
    #     "max_model_len": 4*1024,
    #     "block_size": 1024,
    #     "enable_chunked_prefill": True,
    #     "max_num_batched_tokens": 128,
    #     "max_num_seqs": 8,
    # },
    # Gemma2ForCausalLM
    # {
    #     "model": "google/gemma-2-2b-it",
    #     "max_model_len": 4*1024,
    #     "block_size": 1024,
    #     "enable_chunked_prefill": True,
    #     "max_num_batched_tokens": 128,
    #     "max_num_seqs": 8,
    # },
    # GemmaForCausalLM
    {
        "model": "google/gemma-1.1-2b-it",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "google/gemma-2b-it",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # PhiForCausalLM
    {
        "model": "microsoft/phi-1",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "microsoft/phi-1_5",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "microsoft/phi-2",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # OPTForCausalLM
    {
        "model": "facebook/opt-2.7b",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    {
        "model": "facebook/opt-6.7b",
        "max_model_len": 2 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
    # GraniteForCausalLM
    {
        "model": "ibm-granite/granite-3.3-2b-instruct",
        "max_model_len": 4 * 1024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 8,
    },
]


def run_vllm(llm_kwargs, prompts):
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(**llm_kwargs)
        for output in llm.generate(prompts, SamplingParams(temperature=0.0)):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    except Exception as e:
        raise e


@pytest.mark.parametrize("target", targets)
def test_model_coverage(
    monkeypatch: pytest.MonkeyPatch,
    target,
    common_env,
    prompts,
):
    test_env = target.pop("extra_env", {})
    test_env.update(common_env)
    patch_and_run(
        monkeypatch,
        test_env,
        run_vllm,
        llm_kwargs=target,
        prompts=prompts,
    )
