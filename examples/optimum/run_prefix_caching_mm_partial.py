# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prefix caching with a *partial* multimodal hit.

Each prompt is built as:

    [shared_image] [shared prefix text] [unique_image] [unique question]

The leading ``[shared_image] + prefix`` is identical across every prompt, so
after the first request its KV cache is populated. On later requests that region
is a prefix-cache hit, and the shared image's vision-encoder run is skipped. The
trailing ``[unique_image]`` differs per prompt, so it is never cached and MUST
still be encoded on every request even though the earlier part of the prompt hit
the cache.

This is the case the model runner must get right: ``total_cached_length > 0`` and
yet an uncached multimodal item remains after the cache boundary. ``_prepare_prefill``
encodes only the multimodal items whose placeholder tokens fall after the cached
prefix, so the shared image is dropped while the unique image is kept.

Correctness check: outputs with prefix caching must match the no-cache baseline.
If the uncached image were wrongly dropped (or the cached one wrongly kept, which
would misalign the placeholder scatter), the cached-path outputs would diverge.
"""

import os
import time

from datasets import load_dataset
from transformers import AutoProcessor

os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = "4"
from vllm import LLM, SamplingParams

# NOTE: The model must support at least 2 images per prompt. llava-v1.6 and
# Qwen2.5-VL both do. Swap in a locally compiled checkpoint as needed.
MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

prefix = (
    "You are given two images. The first image is a fixed reference scene that "
    "is the same for every question. Study it carefully as background context. "
    "First, describe the reference image in one short sentence. Then look at the "
    "second image and answer the question about it.\n"
    "Format your response exactly as:\n"
    "Reference: <one sentence about the first image>\n"
    "Answer: <your answer about the second image>\n\n"
    "Question: "
)

prompts = [
    "What is the largest mammal in the world?",
    "Who developed the theory of relativity?",
    "Where is the Great Wall of China located?",
    "Where does the process of photosynthesis occur?",
    "What does the Pythagorean theorem state?",
    "What is the chemical symbol for gold?",
]


def build_prompts():
    """Build prompts that share the first image + prefix but not the second.

    Returns a list of vLLM inputs, each carrying two images: a shared reference
    image (identical across the batch) and a per-prompt unique image.
    """
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")
    shared_image = dataset[0]["image"]
    # One distinct image per prompt, taken from the rest of the dataset.
    unique_images = [dataset[i + 1]["image"] for i in range(len(prompts))]

    processor = AutoProcessor.from_pretrained(MODEL)
    generating_prompts = []
    for prompt, unique_image in zip(prompts, unique_images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # shared reference image (cached prefix)
                    {"type": "text", "text": prefix + prompt},
                    {"type": "image"},  # unique image (never cached)
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        generating_prompts.append(
            {
                "prompt": text,
                "multi_modal_data": {"image": [shared_image, unique_image]},
            }
        )
    return generating_prompts


sampling_params = SamplingParams(temperature=0.0, max_tokens=200)


def run(llm, generating_prompts, label):
    print(f"Results with {label}")
    start_time = time.time()
    outputs = llm.generate(generating_prompts, sampling_params)
    elapsed = time.time() - start_time

    texts = []
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        texts.append(generated_text)
        print(f"Prompt: {output.prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)
    return texts, elapsed


def main():
    generating_prompts = build_prompts()

    # Baseline: no prefix caching.
    regular_llm = LLM(
        model=MODEL,
        block_size=16384,
        max_num_seqs=1,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 2},
    )
    regular_texts, wo_prefix_time = run(
        regular_llm, generating_prompts, "`enable_prefix_caching=False`"
    )
    del regular_llm

    # Prefix caching enabled.
    prefix_cached_llm = LLM(
        model=MODEL,
        block_size=16384,
        max_num_seqs=1,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 2},
    )
    # Warmup with the first prompt so the shared image + prefix KV is cached
    # before the measured run, guaranteeing the later prompts are partial hits.
    prefix_cached_llm.generate(generating_prompts[0], sampling_params)
    cached_texts, w_prefix_time = run(
        prefix_cached_llm, generating_prompts, "`enable_prefix_caching=True`"
    )

    generated_same = all(
        regular_texts[i] == cached_texts[i] for i in range(len(prompts))
    )
    print(f"Generated answers are the same: {generated_same}")
    print(f"Time without prefix caching: {wo_prefix_time} sec")
    print(f"Time with prefix caching: {w_prefix_time} sec")
    assert generated_same, (
        "Prefix-caching outputs diverged from the baseline: the uncached image "
        "after the cache boundary was likely not encoded correctly."
    )


if __name__ == "__main__":
    main()
