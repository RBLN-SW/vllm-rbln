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

import fire
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_prompts(batch_size: int, model: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(
        seed=42
    )

    prompts = []
    for i in range(batch_size):
        image = dataset[i]["image"]
        question = dataset[i]["question"]

        # Use simple QA template because BLIP2 don't have default chat template.
        text_prompt = f"Question: {question}\nAnswer:"

        prompts.append({"prompt": text_prompt, "multi_modal_data": {"image": [image]}})

    return prompts


def main(
    num_input_prompt: int = 10,
    model: str = "Salesforce/blip2-opt-2.7b",
):
    # `max_model_len` of BLIP2 model is 2048
    # and `block_size` cannot exceeds `max_model_len`.
    llm = LLM(model=model, block_size=2048)
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = generate_prompts(num_input_prompt, model)

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=200,
    )

    results = llm.generate(inputs, sampling_params)

    for i, result in enumerate(results):
        output = result.outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


if __name__ == "__main__":
    fire.Fire(main)
