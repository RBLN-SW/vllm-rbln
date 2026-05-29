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

import os

os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "0"
os.environ["VLLM_RBLN_SAMPLER"] = "0"
os.environ["VLLM_RBLN_ENFORCE_MODEL_FP32"] = "1"

from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LORA_MODEL_ID = "jeeejeee/llama32-3b-text2sql-spider"


def create_test_prompts(
    lora_path: str | None, lora_id: int = -1
) -> list[tuple[str, SamplingParams, LoRARequest | None]]:
    if lora_path is None:
        return [
            (
                "A robot may not injure a human being",
                SamplingParams(temperature=0.0, max_tokens=128),
                None,
            ),
        ]

    if "sql" in lora_path.lower():
        prompt = (
            "[user] Write a SQL query to answer the question based on the "
            "table schema.\n\n context: CREATE TABLE table_name_74 "
            "(icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO "
            "for lilongwe international airport [/user] [assistant]"
        )
        return [
            (
                prompt,  # noqa
                SamplingParams(
                    max_tokens=256,
                ),
                LoRARequest("sql-lora", lora_id, lora_path),
            ),
        ]

    raise NotImplementedError


def process_requests(
    engine: LLMEngine,
    prompts: list[tuple[str, SamplingParams, LoRARequest | None]],
) -> list[RequestOutput]:
    request_id = 0

    print("-" * 50)
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, lora_request = prompts.pop(0)
            engine.add_request(
                str(request_id), prompt, sampling_params, lora_request=lora_request
            )
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(
                    f"Prompt: {request_output.prompt!r}, Generated text: {request_output.outputs[0].text!r}"  # noqa
                )
                print("-" * 50)


def initialize_engine() -> LLMEngine:
    engine_args = EngineArgs(
        model=MODEL_ID,
        max_model_len=8192,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=1,
        # cannot set float32 due to vLLM's pydantic restriction
        # lora_dtype="float32",
    )
    return LLMEngine.from_engine_args(engine_args)


if __name__ == "__main__":
    engine = initialize_engine()
    lora_path = snapshot_download(repo_id=LORA_MODEL_ID)

    # Note: The order of prompts matters
    prompts = create_test_prompts(lora_path, 1) + create_test_prompts(None)

    process_requests(engine, prompts)
