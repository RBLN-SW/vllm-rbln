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

import fire
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine, PoolingParams


def get_input_prompts() -> list[str]:
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    # Each query must come with a one-sentence instruction
    # that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        (
            "Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and "
            "is responsible for the movement of planets around the sun."
        ),
    ]

    inputs_texts = queries + documents
    return inputs_texts


async def embed(engine: AsyncLLMEngine, prompt: str, model: str, request_id: int):
    print(f"embed request_id={request_id}, prompt={prompt}")
    pooling_params = PoolingParams(task="embed")
    results_generator = engine.encode(
        prompt,
        pooling_params,
        str(request_id),
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id, runner="pooling")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt_list = get_input_prompts()
    if len(prompt_list) > 2 * num_input_prompt:
        raise RuntimeError(
            "The len(QUERIES) and len(DOCUMENTS) ",
            "should be equal with 2 * `num_input_prompt`.",
        )
    futures = []
    for i, p in enumerate(prompt_list):
        if i == num_input_prompt * 2:
            break
        futures.append(
            asyncio.create_task(
                embed(
                    engine,
                    prompt=p,
                    model=model_id,
                    request_id=i,
                )
            )
        )

    outputs = await asyncio.gather(*futures)

    embeddings = torch.stack([o.outputs.data for o in outputs])
    scores = embeddings[:num_input_prompt] @ embeddings[num_input_prompt:].T

    print(f"scores: {scores.tolist()}")


def entry_point(
    num_input_prompt: int = 2,
    model_id: str = "/qwen3-0.6b-b1-embedding",
):
    asyncio.run(
        main(
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
