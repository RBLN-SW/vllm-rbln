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

os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"
os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_ENABLE_WARM_UP"] = "1"
# vLLM(v0.10.2) bug: speculative decoding works only in multi-processing.
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

MODEL_ID = "meta-llama/Llama-3.2-1B"
NUM_SPECULATIVE_TOKENS = 4
NUM_PROMPT_LOOKUP_MAX = 5
NUM_PROMPT_LOOKUP_MIN = 2


def main():
    # Create an LLM.
    llm = LLM(
        model=MODEL_ID,
        max_model_len=2048,
        block_size=2048,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "prompt_lookup_max": NUM_PROMPT_LOOKUP_MAX,
            "prompt_lookup_min": NUM_PROMPT_LOOKUP_MIN,
        },
        disable_log_stats=False,
    )

    prompts = [
        "A robot may not injure a human being",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=128)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    for output in outputs:
        print("-" * 50)
        print(f"prompt: {output.prompt}")
        print(f"generated text: {output.outputs[0].text}")
        print("-" * 50)

    try:
        metrics = llm.get_metrics()
    except AssertionError:
        print("Failed to load metrics.")
        return

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * NUM_SPECULATIVE_TOKENS
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
