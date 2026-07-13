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
import os

import fire
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams


def generate_prompts(batch_size: int, model: str):
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="train").shuffle(seed=42)
    processor = AutoProcessor.from_pretrained(model, padding_side="left")
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


def main(
    num_input_prompt: int = 4,
    model: str = "mistral-community/pixtral-12b",
):
    os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = "4"
    llm = LLM(model=model, block_size=4096)
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs, labels = generate_prompts(num_input_prompt, model)

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=500,
    )

    results = llm.generate(inputs, sampling_params)

    for i, (result, label) in enumerate(zip(results, labels)):
        label_str = str(label)
        output = result.outputs[0].text

        print("=" * 80)
        print(f"[{i}] Label:")
        print(f"{label_str}\n")
        print(f"[{i}] Model Output:")
        print(output)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    fire.Fire(main)
