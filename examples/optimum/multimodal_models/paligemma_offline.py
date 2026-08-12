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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_prompts(batch_size: int):
    images = []
    texts = []
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


def main(
    num_input_prompt: int = 10,
    model: str = "google/paligemma2-3b-pt-224",
    max_num_seqs: int = 1,
    max_model_len: int = 8192,
    block_size: int = None,  # if None, will be set to max_model_len
    num_devices: int = 4,
):
    # Compile shape follows the rebel_compiler multi-modal compile.py pattern:
    # num_devices → env var, batch_size / max_seq_len → additional_config
    # ["rbln_config"]; block_size stays a top-level vLLM kwarg.
    os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = str(num_devices)
    if block_size is None:
        block_size = max_model_len
    llm = LLM(
        model=model,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = generate_prompts(num_input_prompt)

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
