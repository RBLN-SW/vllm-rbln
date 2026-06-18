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

import json

import fire
import torch
from vllm import LLM, PoolingParams

THRESHOLD = 0.2


def get_input_prompts(prompt_json: str) -> list[str]:
    with open(prompt_json) as file:
        prompt = file.readlines()

    return prompt


def compare_copy_prompt_task_result(scores: list[float], golden_json: str):
    with open(golden_json) as f:
        golden = json.load(f)

    for i, similarity in enumerate(scores):
        golden_similarity = golden[i]["golden_similarity"]
        diff = abs(similarity - golden_similarity)
        print(
            "Difference: {:.3f} Similarity : {:.3f}, Golden Similarity: {:.3f}".format(
                diff, similarity, golden_similarity
            )
        )
        if abs(similarity - golden_similarity) > THRESHOLD:
            print(f"The Error is higher than the threshold ({THRESHOLD})")
            exit(1)


def main(
    model: str = "/bge-m3-1k-batch4",
    num_input_prompt: int = 3,
    q_prompt_txt: str = "/prompts/q_prompts.txt",
    p_prompt_txt: str = "/prompts/p_prompts.txt",
    golden_json: str = "/golden/golden_bge_m3_result_qp_prompts.json",
):
    llm = LLM(model=model)
    pooling_params = PoolingParams(task="embed")

    q_prompt = get_input_prompts(q_prompt_txt)[:num_input_prompt]
    p_prompt = get_input_prompts(p_prompt_txt)[:num_input_prompt]

    assert len(q_prompt) == len(p_prompt)
    q_result = llm.encode(q_prompt, pooling_params)
    p_result = llm.encode(p_prompt, pooling_params)

    scores = []

    for idx, (q, p) in enumerate(zip(q_result, p_result)):
        q_embedding = q.outputs.data
        p_embedding = p.outputs.data

        q_embedding = torch.nn.functional.normalize(q_embedding, p=2, dim=0)
        p_embedding = torch.nn.functional.normalize(p_embedding, p=2, dim=0)

        score = q_embedding @ p_embedding.T

        scores.append(float(score))

    # compare
    compare_copy_prompt_task_result(scores, golden_json)


if __name__ == "__main__":
    fire.Fire(main)
