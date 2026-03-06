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
import argparse
import os

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")
os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
os.environ.setdefault("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_RBLN_ENABLE_WARM_UP", "0")
# vLLM(v0.10.2) bug: speculative decoding works only in multi-processing.
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NUM_SPECULATIVE_TOKENS = 4
MAX_MODEL_LEN = 2048
MAX_NUM_BATCHED_TOKENS = 256
MAX_NUM_SEQS = 4
DEFAULT_MAX_TOKENS = 128
DEFAULT_PROMPTS = [
    "A robot may not injure a human being",
    "The capital of France is",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run suffix-decoding speculative decoding on vLLM-RBLN."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=NUM_SPECULATIVE_TOKENS,
        help=(
            "Maximum speculative token budget used by suffix decoding. "
            "Actual speculative length per step is dynamic."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--block-size", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--suffix-decoding-max-tree-depth",
        type=int,
        default=NUM_SPECULATIVE_TOKENS,
    )
    parser.add_argument(
        "--suffix-decoding-max-cached-requests",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--suffix-decoding-max-spec-factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--suffix-decoding-min-token-prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PROMPTS,
        help="One or more prompts to generate from.",
    )
    return parser.parse_args()


def _check_suffix_dependency() -> None:
    try:
        import arctic_inference  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Suffix decoding requires arctic-inference. "
            "Install it with `pip install arctic-inference==0.1.1`."
        ) from exc


def main() -> None:
    args = parse_args()
    _check_suffix_dependency()

    llm = LLM(
        model=args.model_id,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": args.num_speculative_tokens,
            "suffix_decoding_max_tree_depth": args.suffix_decoding_max_tree_depth,
            "suffix_decoding_max_cached_requests": (
                args.suffix_decoding_max_cached_requests
            ),
            "suffix_decoding_max_spec_factor": args.suffix_decoding_max_spec_factor,
            "suffix_decoding_min_token_prob": args.suffix_decoding_min_token_prob,
        },
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(args.prompts, sampling_params=sampling_params)

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
    acceptance_counts = [0] * args.num_speculative_tokens
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
            num_positions = min(len(metric.values), len(acceptance_counts))
            for pos in range(num_positions):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    for pos, accepted_at_pos in enumerate(acceptance_counts):
        acceptance_rate = accepted_at_pos / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {pos}: {acceptance_rate:.2f}")


if __name__ == "__main__":
    main()
