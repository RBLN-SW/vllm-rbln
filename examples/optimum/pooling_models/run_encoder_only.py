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

import asyncio
import json

import fire
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams

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


async def encode(engine, prompt, request_id):
    pooling_params = PoolingParams(task="embed")
    results_generator = engine.encode(
        prompt=prompt, pooling_params=pooling_params, request_id=str(request_id)
    )
    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def get_result(engine, model_id, prompt, num_input_prompt):
    futures = []

    for i, p in enumerate(prompt):
        if i == num_input_prompt:
            break

        futures.append(asyncio.create_task(encode(engine, p, i)))
    results = await asyncio.gather(*futures)

    return results


async def main(
    model_id: str,
    num_input_prompt: int,
    q_prompt_txt: str,
    p_prompt_txt: str,
    golden_json: str,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    q_prompt = get_input_prompts(q_prompt_txt)
    p_prompt = get_input_prompts(p_prompt_txt)

    assert len(q_prompt) == len(p_prompt)
    q_result = await get_result(engine, model_id, q_prompt, num_input_prompt)
    p_result = await get_result(engine, model_id, p_prompt, num_input_prompt)

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


def entry_point(
    num_input_prompt: int = 3,
    model_id: str = "/bge-m3-1k-batch4",
    q_prompt_txt: str = "/prompts/q_prompts.txt",
    p_prompt_txt: str = "/prompts/p_prompts.txt",
    golden_json: str = "/golden/golden_bge_m3_result_qp_prompts.json",
):
    asyncio.run(
        main(
            num_input_prompt=num_input_prompt,
            model_id=model_id,
            q_prompt_txt=q_prompt_txt,
            p_prompt_txt=p_prompt_txt,
            golden_json=golden_json,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
