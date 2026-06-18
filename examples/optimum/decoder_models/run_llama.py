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
from simphile import jaccard_similarity
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


async def generate(
    engine: AsyncLLMEngine, prompt: str, model: str, request_id: int, max_tokens: int
):
    print(f"generate request_id={request_id}, prompt={prompt}")
    example_input = {
        "stream": True,
        "temperature": 0.0,
        "request_id": str(request_id),
    }
    # start the generation
    conversation = [{"role": "user", "content": prompt}]
    tokenizer = AutoTokenizer.from_pretrained(model)
    chat = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    results_generator = engine.generate(
        chat,
        SamplingParams(
            temperature=example_input["temperature"],
            ignore_eos=False,
            skip_special_tokens=True,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=max_tokens,
        ),
        example_input["request_id"],
    )

    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


def get_input_prompts(prompt_txt: str) -> list[str]:
    with open(prompt_txt) as file:
        prompt = file.readlines()

    return prompt


def compare_copy_prompt_task_result(
    result,
    golden_json,
    write_txt="compare_summary.txt",
):
    with open(golden_json) as f:
        golden = json.load(f)

    total_score = 0
    for i, r in enumerate(result):
        inference_output_text = r.outputs[0].text
        print(inference_output_text)

        golden_prompt = golden[i]["output_prompt"][0]
        similarity = jaccard_similarity(golden_prompt, inference_output_text)
        print(f"Similarity score : {similarity}")
        total_score += similarity

    total_avg = total_score / len(result)
    return total_avg


async def main(
    max_seq_len: int,
    num_input_prompt: int,
    model_id: str,
    prompt_txt: str,
    golden_json: str,
):
    engine_args = AsyncEngineArgs(model=model_id)

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    prompt = get_input_prompts(prompt_txt)
    futures = []
    for i, p in enumerate(prompt):
        if i == num_input_prompt:
            break

        futures.append(
            asyncio.create_task(
                generate(
                    engine,
                    prompt=p,
                    model=model_id,
                    request_id=i,
                    max_tokens=max_seq_len,
                )
            )
        )

    result = await asyncio.gather(*futures)

    score = compare_copy_prompt_task_result(result, golden_json)
    if score < 0.97:
        print(f"score is lower than threshold({score})")
        exit(1)


def entry_point(
    max_seq_len: int = 4096,
    num_input_prompt: int = 1,
    model_id: str = "/llama2-7b_batch2",
    prompt_txt: str = "/prompts/copy_prompts.txt",
    golden_json: str = "/golden/golden_llama7b_result_copy_prompts.json",
):
    asyncio.run(
        main(
            max_seq_len=max_seq_len,
            num_input_prompt=num_input_prompt,
            model_id=model_id,
            prompt_txt=prompt_txt,
            golden_json=golden_json,
        )
    )


if __name__ == "__main__":
    fire.Fire(entry_point)
