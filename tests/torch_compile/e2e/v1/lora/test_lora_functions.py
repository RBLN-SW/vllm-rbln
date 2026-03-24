# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""
Script to test add_lora, remove_lora, pin_lora, list_loras functions.
"""

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "charent/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync(monkeypatch):
    monkeypatch.setenv("RBLN_PROFILER", "0")
    monkeypatch.setenv("RBLN_KERNEL_MODEL", "triton")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    max_loras = 4
    engine_args = EngineArgs(
        model=MODEL_PATH,
        max_model_len=4096,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_cpu_loras=max_loras,
    )

    llm = LLMEngine.from_engine_args(engine_args)

    def run_check(fn, args, expected: list):
        fn(args)
        assert set(llm.list_loras()) == set(expected)

    run_check(llm.add_lora, make_lora_request(1), [1])
    run_check(llm.add_lora, make_lora_request(2), [1, 2])

    # Pin LoRA 1 and test that it is never removed on subsequent adds.
    run_check(llm.pin_lora, 1, [1, 2])
    run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
    run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
    run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
    run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
    run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
    run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
    run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
    run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

    # Remove LoRA 1 and continue adding.
    run_check(llm.remove_lora, 1, [8, 9, 10])
    run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

    # Remove all LoRAs.
    run_check(llm.remove_lora, 13, [12, 10, 11])
    run_check(llm.remove_lora, 12, [10, 11])
    run_check(llm.remove_lora, 11, [10])
    run_check(llm.remove_lora, 10, [])
