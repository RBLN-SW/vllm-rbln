# SPDX-License-Identifier: Apache-2.0
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
import time

from transformers import AutoProcessor

os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = "4"
from vllm import LLM, SamplingParams


def load_shared_image():
    """Load a single image shared by every prompt.

    Sharing one image across the batch lets the image + text-prefix KV cache
    be computed once and reused, which is what prefix caching exercises here.
    """
    from datasets import load_dataset

    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train")
    return dataset[0]["image"]


# NOTE: This is just a running example. For benchmarking purpose,
# please see benchmarks/benchmark_prefix_caching.py

# Common prefix.
# prefix = (
#     "You are an experienced and insightful school principal, highly skilled in "
#     "strategically managing and guiding a diverse team of faculty, instructional "
#     "specialists, and support staff across grade levels. Draft 10–15 thoughtful, "
#     "open-ended questions for a potential first grade Head Teacher candidate at my "
#     "independent K–12, all-girls’ school. Our institution strongly emphasizes "
#     "collaboration, a nurturing sense of community, joyful discovery throughout "
#     "academic and co-curricular life, and the cultivation of life-long curiosity, "
#     "resilience, and learning habits. The candidate is interviewing for a first-round "
#     "panel conversation related to an 8th grade Mathematics teaching role. They bring "
#     "over 5 years of professional experience, having served as an assistant teacher "
#     "in a large, co-educational public school, with substantial background in "
#     "curriculum design, classroom leadership, and instructional strategies for "
#     "middle school mathematics students."
# )
# Shared instruction + output template. Kept identical across prompts so the
# image + this prefix can be reused by prefix caching; only the trailing
# question varies. The explicit two-step task and fixed format make the model
# both describe the image and answer, instead of just completing the sentence.
prefix = (
    "First, describe the image in one short sentence. "
    "Then, on a new line, answer the question.\n"
    "Format your response exactly as:\n"
    "Description: <one sentence>\n"
    "Answer: <your answer>\n\n"
    "Question: "
)

# Sample prompts. Phrased as explicit questions so the model does not treat
# them as a fill-in-the-blank continuation and skip the description.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    "What is the largest mammal in the world?",
    "Who developed the theory of relativity?",
    "Where is the Great Wall of China located?",
    "Where does the process of photosynthesis occur?",
    "What does the Pythagorean theorem state?",
    "What is the chemical symbol for gold?",
]

MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

# Every prompt shares the same image and the same prefix text, so the
# image + prefix KV cache can be reused across the batch. Run each prompt
# through the chat template so the `<image>` placeholder token is inserted
# for the multimodal processor.
shared_image = load_shared_image()
processor = AutoProcessor.from_pretrained(MODEL)
generating_prompts = []
for prompt in prompts:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prefix + prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    generating_prompts.append(
        {"prompt": text, "multi_modal_data": {"image": shared_image}}
    )

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)


def main():
    # Create an LLM without prefix caching as a baseline.
    regular_llm = LLM(
        model=MODEL,
        block_size=16384,
        # max_model_len=32768,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )

    print("Results without `enable_prefix_caching`")

    # ruff: noqa: E501
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start_time = time.time()
    outputs = regular_llm.generate(generating_prompts, sampling_params)
    end_time = time.time()
    wo_prefix_time = end_time - start_time

    regular_generated_texts = []
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        regular_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Destroy the LLM object and free up the GPU memory.
    del regular_llm
    # cleanup_dist_env_and_memory()

    # Create an LLM with prefix caching enabled.
    prefix_cached_llm = LLM(
        model=MODEL,
        block_size=16384,
        # max_model_len=32768,
        max_num_seqs=1,
        enable_prefix_caching=True,
    )

    # Warmup so that the shared prompt's KV cache is computed.
    # prefix_cached_llm.generate(generating_prompts[0], sampling_params)

    # Generate with prefix caching.
    start_time = time.time()
    outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
    end_time = time.time()
    w_prefix_time = end_time - start_time
    print("Results with `enable_prefix_caching`")

    cached_generated_texts = []
    # Print the outputs. You should see the same outputs as before.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Compare the results and display the speedup
    generated_same = all(
        [
            regular_generated_texts[i] == cached_generated_texts[i]
            for i in range(len(prompts))
        ]
    )
    print(f"Generated answers are the same: {generated_same}")
    print(f"Time without prefix caching: {wo_prefix_time} sec")
    print(f"Time with prefix caching: {w_prefix_time} sec")


if __name__ == "__main__":
    main()
