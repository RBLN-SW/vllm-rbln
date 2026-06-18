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

import random
import time

import wikipedia
from vllm import LLM, SamplingParams

# NOTE: This is just a running example. For benchmarking purpose,
# please see benchmarks/benchmark_prefix_caching.py


# Common prefix.
def get_system_prompted_questions():
    prefix = (
        "You are an experienced and insightful school principal, highly skilled in "
        "strategically managing and guiding a diverse team of faculty, instructional "
        "specialists, and support staff across grade levels. Draft 10–15 thoughtful, "
        "open-ended questions for a potential first grade Head Teacher candidate at my "
        "independent K–12, all-girls’ school. Our institution strongly emphasizes "
        "collaboration, a nurturing sense of community, joyful discovery throughout "
        "academic and co-curricular life, and the cultivation of life-long curiosity, "
        "resilience, and learning habits. The candidate is interviewing for a first-round "
        "panel conversation related to an 8th grade Mathematics teaching role. They bring "
        "over 5 years of professional experience, having served as an assistant teacher "
        "in a large, co-educational public school, with substantial background in "
        "curriculum design, classroom leadership, and instructional strategies for "
        "middle school mathematics students."
    )
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "The largest mammal in the world is",
        "The theory of relativity was developed by",
        "The Great Wall of China is located in",
        "The process of photosynthesis occurs in",
        "The Pythagorean theorem states that",
        "The chemical symbol for gold is",
    ]
    return [prefix + prompt for prompt in prompts]


def get_wiki_based_questions():
    wikipedia.set_lang("en")
    template = """
    DOCUMENT:
    {document}

    QUESTION:
    {question}

    INSTRUCTIONS:
    Answer the users QUESTION using the DOCUMENT text above.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENT doesn’t contain the facts to answer the QUESTION return NONE.

    ANSWER:
    """
    doc = wikipedia.page("Artificial intelligence").content[:3000]
    questions = [
        "When is the AI winter?",
        "Who is the father of AI?",
        "What is the Turing Test?",
    ]
    return [template.format(document=doc, question=question) for question in questions]


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)
MODEL = "./llama3.2-3b-rbln-b3"


def main():
    # Create an LLM without prefix caching as a baseline.
    prompts = get_system_prompted_questions() + get_wiki_based_questions()
    random.seed(42)
    random.shuffle(prompts)

    regular_llm = LLM(
        model=MODEL,
        enable_prefix_caching=False,
    )

    print("Results without `enable_prefix_caching`")

    # ruff: noqa: E501
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start_time = time.time()
    outputs = regular_llm.generate(prompts, sampling_params)
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
        enable_prefix_caching=True,
    )

    # Warmup so that the shared prompt's KV cache is computed.
    # prefix_cached_llm.generate(prompts[0], sampling_params)

    # Generate with prefix caching.
    start_time = time.time()
    outputs = prefix_cached_llm.generate(prompts, sampling_params)
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
