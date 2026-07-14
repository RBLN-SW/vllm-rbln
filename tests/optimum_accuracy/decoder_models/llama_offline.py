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
from simphile import jaccard_similarity
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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


def main(
    max_seq_len: int = 4096,
    num_input_prompt: int = 1,
    model: str = "/llama2-7b_batch2",
    prompt_txt: str = "/prompts/copy_prompts.txt",
    golden_json: str = "/golden/golden_llama7b_result_copy_prompts.json",
):
    llm = LLM(model=model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts = get_input_prompts(prompt_txt)[:num_input_prompt]
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=max_seq_len,
    )

    result = llm.generate(chats, sampling_params)

    score = compare_copy_prompt_task_result(result, golden_json)
    if score < 0.97:
        print(f"score is lower than threshold({score})")
        exit(1)


if __name__ == "__main__":
    fire.Fire(main)
