# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark RBLN speculative decoding through the public vLLM engine.

Select a card with RBLN_VISIBLE_DEVICES and run from the repository root:
    uv run --no-sync python -m benchmarks.benchmark_spec_decode --model MODEL

Use the same model, arguments and card for each revision. For paired results,
run with --repeats 1 and rotate the checkout order across rounds.
"""

import argparse
from statistics import median
from time import perf_counter

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter

DRAFT_TOKENS_METRIC = "vllm:spec_decode_num_draft_tokens"


def draft_tokens(llm: LLM) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and metric.name == DRAFT_TOKENS_METRIC
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.spec_tokens,
            args.output_len,
            args.warmup,
            args.repeats,
        )
        < 1
    ):
        parser.error("batch size, token counts, warmup, and repeats must be positive")

    llm = LLM(
        model=args.model,
        model_impl="vllm",
        max_num_seqs=args.batch_size,
        max_model_len=max(512, args.output_len + 128),
        block_size=128,
        max_num_batched_tokens=128,
        enable_prefix_caching=False,
        disable_log_stats=False,
        additional_config={"use_custom_sampler": True},
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": args.spec_tokens,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 3,
        },
    )

    # The repeated text gives the built-in n-gram proposer matching prefixes.
    prompts = [" red green blue" * 16 + " red"] * args.batch_size
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=40,
        top_p=0.9,
        max_tokens=args.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)

    before = draft_tokens(llm)
    samples = []
    for _ in range(args.repeats):
        start = perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = perf_counter() - start
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        samples.append(tokens / elapsed)

    drafted = draft_tokens(llm) - before
    if not drafted:
        raise RuntimeError(
            "The measured requests produced no speculative draft tokens."
        )
    print(f"Draft tokens during measurement: {drafted}")
    print(f"Median generation throughput: {median(samples):.2f} output tokens/s")
    print("Throughput per repeat: " + ", ".join(f"{sample:.2f}" for sample in samples))


if __name__ == "__main__":
    main()
