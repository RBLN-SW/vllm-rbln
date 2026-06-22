# Copyright 2025 Rebellions Inc. All rights reserved.
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

import torch
from vllm import LLM


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def main():
    # Each query comes with a one-sentence instruction that describes the task
    task = "Given a query, retrieve relevant passages that answer the query"
    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
        get_detailed_instruct(task, "Who is the president of the United States?"),
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        + "It gives weight to physical objects and is responsible for the "
        + "movement of planets around the sun.",
        "The president of the United States is Donald Trump.",
    ]
    input_texts = queries + documents
    model = LLM(
        model="Qwen/Qwen3-Embedding-4B",
        max_model_len=2024,
        block_size=512,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        task="embed",
    )

    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = embeddings[:3] @ embeddings[3:].T
    for score in scores:
        print(score.tolist())
    # [0.7661870121955872, 0.11317858844995499, 0.2081689089536667]
    # [0.030531929805874825, 0.6107323169708252, 0.07793198525905609]
    # [0.21299228072166443, 0.12848089635372162, 0.7656189799308777


if __name__ == "__main__":
    main()
