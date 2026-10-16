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

"""Extend full-block prefix cache hits by reusing sub-blocks on RBLN.

See docs/sub_block_prefix_caching.md for configuration and device requirements.
"""

import argparse

from automatic_prefix_caching_offline import make_prompt
from vllm import LLM, SamplingParams

from vllm_rbln.config import RBLNConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    block_size = 1024
    llm = LLM(
        model=args.model,
        model_impl="vllm",
        block_size=block_size,
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        additional_config=RBLNConfig(
            enable_sub_block_cache=True,
            sub_block_size=128,
        ),
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    tokenizer = llm.get_tokenizer()
    questions = [
        "What is the age of John Doe?",
        "What is the age of Zack Blue?",
        "What is the occupation of Jane Smith?",
        "Which country does Alice Johnson live in?",
    ]

    # Complete each request before submitting the next so its KV is cached.
    cached_tokens = []
    for question in questions:
        prompt = make_prompt(tokenizer, question)
        output = llm.generate([prompt], sampling_params)[0]
        cached_tokens.append(output.num_cached_tokens)
        sub_block_tokens = output.num_cached_tokens % block_size
        print("-" * 60)
        print(f"Question: {question}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Cached prompt tokens: {output.num_cached_tokens}")
        print(f"Cached tokens beyond full blocks: {sub_block_tokens}")

    assert cached_tokens[0] == 0, "The first request should have a cold cache."
    assert all(
        count >= block_size and count % block_size > 0 for count in cached_tokens[1:]
    ), "Later requests should reuse complete KV blocks and additional sub-blocks."


if __name__ == "__main__":
    main()
