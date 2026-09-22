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

"""Models the compile-and-run smoke exercises (spec type in model_specs)."""

from __future__ import annotations

from tests.vllm.model_specs import REBEL, CompileModelSpec

OPT_ENVS = {
    "VLLM_RBLN_BATCH_ATTN_OPT": "1",
}

_MINIMAX_BASE = CompileModelSpec(
    "MiniMaxAI/MiniMax-M2.7",
    {
        "max_num_seqs": 1,
        "max_model_len": 51200,
        "block_size": 1024,
        "max_num_batched_tokens": 512,
        "enable_expert_parallel": True,
    },
    OPT_ENVS,
    chips=REBEL,
)

MODELS: list[CompileModelSpec] = [
    CompileModelSpec(
        "Qwen/Qwen3-30B-A3B",
        {
            "max_num_seqs": 1,
            "max_model_len": 40960,
            "block_size": 8192,
            "tensor_parallel_size": 4,
            "enable_expert_parallel": False,
        },
        OPT_ENVS,
        chips=REBEL,
    ),
    CompileModelSpec(
        "openai/gpt-oss-120b",
        {
            "max_num_seqs": 1,
            "max_model_len": 131072,
            "block_size": 1024,
            "max_num_batched_tokens": 512,
            "data_parallel_size": 4,
            "enable_expert_parallel": True,
        },
        {
            "VLLM_RBLN_SUB_BLOCK_CACHE": "0",
            **OPT_ENVS,
        },
        chips=REBEL,
        num_hidden_layers=4,
    ),
    _MINIMAX_BASE.variant(data_parallel_size=4),
    _MINIMAX_BASE.variant(tensor_parallel_size=4),
    _MINIMAX_BASE.variant(max_num_seqs=4, pipeline_parallel_size=4),
]
