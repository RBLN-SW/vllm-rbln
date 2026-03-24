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
from vllm import LLM

from .utils import patch_and_run

LLM_PARAMS = [
    {
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "max_model_len": 2024,
        "block_size": 1024,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 4,
        "hf_overrides": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    },
]


def run_vllm_score(llm_kwargs: dict) -> None:
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the "
        + "requirements based on the Query and the Instruct provided. "
        + 'Note that the answer can only be "yes" or "no".<|im_end|>\n'
        + "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    document_template = "<Document>: {doc}{suffix}"

    instruction = "Given a query, retrieve relevant passages that answer the query"

    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]

    templated_queries = [
        query_template.format(prefix=prefix, instruction=instruction, query=query)
        for query in queries
    ]
    templated_documents = [
        document_template.format(doc=doc, suffix=suffix) for doc in documents
    ]

    llm = LLM(runner="pooling", **llm_kwargs)

    outputs = llm.score(templated_queries, templated_documents)

    assert outputs[0].outputs.score > 0.8, (
        f"Score ({outputs[0].outputs.score}) should be large."
        f" <Query>: {queries[0]} <Document>: {documents[0]}."
    )
    assert outputs[1].outputs.score < 0.2, (
        f"Score ({outputs[1].outputs.score}) should be small."
        f" <Query>: {queries[1]} <Document>: {documents[1]}."
    )
    assert outputs[2].outputs.score > 0.8, (
        f"Score ({outputs[2].outputs.score}) should be large."
        f" <Query>: {queries[2]} <Document>: {documents[2]}."
    )


@pytest.mark.parametrize("llm_params", LLM_PARAMS)
def test_pooling_model_score(
    monkeypatch: pytest.MonkeyPatch,
    llm_params: dict,
) -> None:
    env = {
        "VLLM_RBLN_USE_VLLM_MODEL": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
    }

    patch_and_run(monkeypatch, env, run_vllm_score, llm_kwargs=llm_params)
