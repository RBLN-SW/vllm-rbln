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

    [shared_image] [shared instruction] [unique_image] [shared question]

The leading ``[shared_image] + instruction`` is identical across every prompt, so
after the first request its KV cache is populated. On later requests that region
is a prefix-cache hit, and the shared image's vision-encoder run is skipped. The
``[unique_image]`` differs per prompt, so it is never cached and MUST still be
encoded on every request even though the earlier part of the prompt hit the cache.

This is the case the model runner must get right: ``total_cached_length > 0`` and
yet an uncached multimodal item remains after the cache boundary. ``_prepare_prefill``
encodes only the multimodal items whose placeholder tokens fall after the cached
prefix, so the shared image is dropped while the unique image is kept.

The question deliberately asks *only about the second (unique) image* and demands
a short answer. Two properties matter:

* Image-dependence: the answer depends on the uncached image's content, so if that
  image were wrongly dropped (or the cached one wrongly kept, misaligning the
  placeholder scatter), the answer would change and the assertion would fail. A
  general-knowledge question would pass even with the image dropped, testing
  nothing.
* Short + constrained output: reusing cached KV is not guaranteed to be
  bit-identical to recomputing it, so over a long free-form generation greedy
  decoding can diverge from the baseline even when the cache is correct. A short,
  low-entropy answer keeps the exact-match check robust to that numeric noise.
"""

import os
import time
from itertools import islice

from datasets import load_dataset
from transformers import AutoProcessor

os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = "4"
from vllm import LLM, SamplingParams

# NOTE: The model must support at least 2 images per prompt. llava-v1.6 and
# Qwen2.5-VL both do. Swap in a locally compiled checkpoint as needed.
MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

# Number of prompts to run. Each uses the same shared image + text but a
# different unique second image.
NUM_PROMPTS = 6

# Shared question placed after the images. It asks for a brief description of each
# image, so the answer covers the uncached second image too; if that image were
# dropped on the partial hit, its part of the answer would change and the assertion
# would fail.
QUESTION = (
    "There are two images above. Describe each image in one sentence: "
    "one sentence for the first image and one sentence for the second image."
)


def build_prompts():
    """Build prompts that share the first image + text but not the second image.

    Returns a list of vLLM inputs, each carrying two images: a shared reference
    image (identical across the batch) and a per-prompt unique image. Only the
    unique image varies, so the cacheable prefix ends right before it.
    """
    # Stream so only the first NUM_PROMPTS + 1 rows are fetched instead of the
    # whole (multi-million-image) dataset.
    dataset = load_dataset(
        "lmms-lab/LLaVA-ReCap-CC3M", split="train", streaming=True
    )
    images = [row["image"] for row in islice(dataset, NUM_PROMPTS + 1)]
    shared_image = images[0]
    # One distinct image per prompt.
    unique_images = images[1:]

    processor = AutoProcessor.from_pretrained(MODEL)
    generating_prompts = []
    for unique_image in unique_images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # shared reference image (cached prefix)
                    {"type": "image"},  # unique image (never cached)
                    {"type": "text", "text": QUESTION},
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
        print("@@@ shared_image:", shared_image)
        print("@@@ unique_image:", unique_image)
    return generating_prompts


# Greedy + short output: keeps the baseline-vs-cached comparison deterministic and
# robust to small cached-vs-recomputed KV numeric differences.
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
        regular_texts[i] == cached_texts[i] for i in range(len(generating_prompts))
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
