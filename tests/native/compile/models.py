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

from tests.native.model_specs import ATOM, REBEL, CompileModelSpec

OPT_ENVS = {
    "VLLM_RBLN_BATCH_ATTN_OPT": "1",
    "VLLM_RBLN_SORT_BATCH": "1",
}

_QWEN3_30B_A3B_BASE = CompileModelSpec(
    "Qwen/Qwen3-30B-A3B",
    {
        "max_num_seqs": 1,
        "max_model_len": 40960,
        "block_size": 8192,
        "tensor_parallel_size": 8,
        "enable_expert_parallel": True,
    },
    OPT_ENVS,
)
_MINIMAX_BASE = CompileModelSpec(
    "MiniMaxAI/MiniMax-M2.7",
    {
        "max_num_seqs": 1,
        "max_model_len": 204800,
        "block_size": 1024,
        "max_num_batched_tokens": 512,
        "enable_expert_parallel": True,
    },
    OPT_ENVS,
    chips=REBEL,
)

MODELS: list[CompileModelSpec] = [
    _QWEN3_30B_A3B_BASE.variant(rsd=4, tensor_parallel_size=8, chips=ATOM),
    _QWEN3_30B_A3B_BASE.variant(
        tensor_parallel_size=4, enable_expert_parallel=False, chips=REBEL
    ),
    CompileModelSpec(
        "Qwen/Qwen1.5-MoE-A2.7B",
        {
            "max_num_seqs": 1,
            "max_model_len": 8192,
            "block_size": 4096,
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
        },
        OPT_ENVS,
        rsd=4,
        chips=ATOM,
    ),
    CompileModelSpec(
        "openai/gpt-oss-20b",
        {
            "max_num_seqs": 1,
            "max_model_len": 131072,
            "block_size": 8192,
            "tensor_parallel_size": 8,
            "enable_expert_parallel": True,
        },
        {
            "VLLM_RBLN_SUB_BLOCK_CACHE": "0",
            **OPT_ENVS,
        },
        rsd=4,
        chips=ATOM,
        num_hidden_layers=4,
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
    _MINIMAX_BASE.variant(pipeline_parallel_size=4),
]
