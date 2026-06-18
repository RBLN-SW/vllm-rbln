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
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompts(batch_size: int):
    from datasets import load_dataset

    images = []
    dataset = (
        load_dataset("takara-ai/image_captions", split="train", streaming=True)
        .take(batch_size)
        .shuffle(seed=42)
    )

    for example in dataset:
        images.append(example["image"])
    # NOTE
    # "caption en" means "generate caption in English"
    # "caption es" means "generate caption in Spanish"
    texts = ["caption en"] * batch_size

    return [
        {"prompt": text, "multi_modal_data": {"image": image}}
        for text, image in zip(texts, images)
    ]


async def generate(engine: AsyncLLMEngine, eos_token_id, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(
            temperature=0,
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[eos_token_id],
            max_tokens=200,
        ),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    inputs: list,
    model_id: str,
    eos_token_id: int,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    futures = []
    for request_id, request in enumerate(inputs):
        futures.append(
            asyncio.create_task(generate(engine, eos_token_id, request_id, request))
        )

    results = await asyncio.gather(*futures)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


def entry_point(
    num_input_prompt: int = 10,
    model_id: str = "./paligemma2_b2",
):
    inputs = generate_prompts(num_input_prompt)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    asyncio.run(
        main(
            inputs=inputs,
            model_id=model_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
