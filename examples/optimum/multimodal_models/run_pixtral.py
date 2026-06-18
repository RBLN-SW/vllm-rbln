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
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": dataset[i]["query"],
                    },
                ],
            },
        ]
        for i in range(batch_size)
    ]
    images = [[dataset[i]["image"]] for i in range(batch_size)]
    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = [
        {"prompt": text, "multi_modal_data": {"image": image}}
        for text, image in zip(texts, images)
    ]
    labels = [dataset[i]["label"] for i in range(batch_size)]
    return inputs, labels


async def generate(engine: AsyncLLMEngine, tokenizer, request_id, request):
    results_generator = engine.generate(
        request,
        SamplingParams(
            temperature=0,
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=500,
        ),
        str(request_id),
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def main(
    num_input_prompt: int,
    model_id: str,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs, labels = generate_prompts(num_input_prompt, model_id)

    futures = []
    for request_id, request in enumerate(inputs):
        futures.append(
            asyncio.create_task(generate(engine, tokenizer, request_id, request))
        )

    results = await asyncio.gather(*futures)
    for i, (result, label) in enumerate(zip(results, labels)):
        label_str = str(label)
        output = result.outputs[0].text

        print("=" * 80)
        print(f"[{i}] Label:")
        print(f"{label_str}\n")
        print(f"[{i}] Model Output:")
        print(output)
        print("=" * 80 + "\n")


def entry_point(
    num_input_prompt: int = 4,
    model_id: str = "/pixtral-12b-b4",
):
    asyncio.run(
        main(
            num_input_prompt=num_input_prompt,
            model_id=model_id,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
