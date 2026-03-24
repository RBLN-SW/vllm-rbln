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
import torch
from vllm import LLM

from .utils import patch_and_run

LLM_PARAMS = [
    {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "max_model_len": 2024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 4,
    },
]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def run_vllm_score(llm_kwargs: dict) -> None:
    task = "Given a query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the "
        "movement of planets around the sun.",
    ]
    input_prompts = queries + documents

    try:
        llm = LLM(task="embed", **llm_kwargs)

        outputs = llm.embed(input_prompts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        scores = embeddings[:2] @ embeddings[2:].T

        assert scores[0][0] > scores[0][1], (
            f'The score between the "{queries[0]}" and the "{documents[0]}" '
            f'should be larger than the score between the "{queries[0]}" '
            f'and the "{documents[1]}".'
        )
        assert scores[1][0] < scores[1][1], (
            f'The score between the "{queries[1]}" and the "{documents[0]}" '
            f'should be smaller than the score between the "{queries[1]}" '
            f'and the "{documents[1]}".'
        )

    except Exception as e:
        raise e


@pytest.mark.parametrize("llm_params", LLM_PARAMS)
def test_pooling_model_embed(
    monkeypatch: pytest.MonkeyPatch,
    llm_params: dict,
) -> None:
    env = {
        "VLLM_RBLN_USE_VLLM_MODEL": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
    }

    patch_and_run(monkeypatch, env, run_vllm_score, llm_kwargs=llm_params)
