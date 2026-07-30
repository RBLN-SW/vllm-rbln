# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
from transformers import AutoTokenizer
from vllm import LLM

QUERIES = [
    "What is the capital of China?",
    "Explain gravity",
    "What is the capital of China?",
]

DOCUMENTS = [
    "The capital of China is Beijing.",
    (
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and "
        "is responsible for the movement of planets around the sun."
    ),
    # Deliberately irrelevant, to show the scores separate.
    "Gravity gives weight to physical objects.",
]


def main(
    max_seq_len: int = 32768,
    model: str = "Qwen/Qwen3-Reranker-0.6B",
):
    llm = LLM(
        model=model,
        runner="pooling",
        block_size=4096,
        max_model_len=max_seq_len,
        # This is how upstream vLLM loads the original reranker. We keep the
        # same three keys, so the same code runs on either backend:
        # https://github.com/vllm-project/vllm/blob/v0.22.0/examples/pooling/score/qwen3_reranker_offline.py
        # - the architecture override is how vLLM picks convert=classify,
        # - the two label tokens name the logits the score is read from,
        #   false first,
        # - and the flag marks this as the original, unconverted checkpoint.
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    # `score()` pairs each query with each document as `query`/`document` roles,
    # which is exactly what the reranker's own chat template selects on, so the
    # shipped template renders the full judge prompt for us.
    chat_template = AutoTokenizer.from_pretrained(model).chat_template

    outputs = llm.score(QUERIES, DOCUMENTS, chat_template=chat_template)
    print(f"scores: {[output.outputs.score for output in outputs]}")


if __name__ == "__main__":
    fire.Fire(main)
