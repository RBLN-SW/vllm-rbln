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

The workload uses fixed prose prompts and per-request sampling presets. Use the
same model, arguments and card for each revision. For paired results, run with
--repeats 1 and rotate the checkout order across rounds.
"""

import argparse
from statistics import mean, median
from time import perf_counter

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter

DRAFT_TOKENS_METRIC = "vllm:spec_decode_num_draft_tokens"

PROMPTS = (
    "Explain why a glass of ice water collects droplets on the outside. "
    "Use an example a ten-year-old would understand.",
    "Write a short email asking a colleague to move tomorrow's meeting "
    "to Thursday afternoon. Keep the tone friendly and professional.",
    "Continue this field journal in the same style. On Monday we counted "
    "the gulls by the harbor. On Tuesday we counted the ducks by the bridge.",
    "Summarize this note in one sentence. The museum will open two hours late "
    "on Saturday because staff are installing a new exhibit.",
    "Suggest a simple vegetarian dinner for four using chickpeas, spinach, "
    "and a can of tomatoes. Include a quick side dish.",
    "Translate this announcement into Spanish. The last train leaves at "
    "eleven tonight, and tickets are available at the station.",
    "Write a Python function that groups filenames by extension. Explain "
    "what the function should do with filenames that have no extension.",
    "Continue this checklist with two similar reminders. Before you leave "
    "the lab, close the windows. Before you leave the lab, turn off the lights.",
    "Compare taking the bus with riding a bicycle for a five-kilometer "
    "commute on a rainy day. Give one advantage of each.",
    "Retell this scene from the dog's point of view. When the doorbell rang, "
    "Mina put down her book and walked over to the front door.",
    "Rewrite this sentence in plain English. Visitors must present a valid "
    "ticket before entering the exhibition hall.",
    "Give three questions a manager could ask during a first meeting with "
    "a new team member. Explain the purpose of each question.",
    "Write the next two lines in this weather log using the same format. "
    "At six in the morning it was cool and clear. At noon it was warm and clear.",
    "Explain how to estimate the cost of painting a room. State what "
    "measurements are needed and what assumptions affect the estimate.",
    "Recommend a quiet afternoon itinerary for someone visiting a small "
    "coastal town without a car. Allow time for lunch and a short walk.",
    "Copy this sentence and then explain it in your own words. Please leave "
    "the spare key with the concierge before you check out.",
)

# Temperature, top-k, and top-p presets assigned in request order.
SAMPLING_PROFILES = (
    (0.2, 20, 0.85),
    (0.4, 32, 0.90),
    (0.6, 40, 0.92),
    (0.7, 50, 0.95),
    (0.7, 0, 0.92),
    (0.85, 40, 1.0),
    (0.85, 64, 0.95),
    (1.0, 50, 0.98),
)


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

    prompts = []
    sampling = []
    for index in range(args.batch_size):
        prompts.append(PROMPTS[index % len(PROMPTS)])
        temperature, top_k, top_p = SAMPLING_PROFILES[index % len(SAMPLING_PROFILES)]
        sampling.append(
            SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=args.output_len,
                ignore_eos=True,
            )
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
    print(f"Average generation throughput: {mean(samples):.2f} output tokens/s")
    print(f"Minimum generation throughput: {min(samples):.2f} output tokens/s")
    print(f"Maximum generation throughput: {max(samples):.2f} output tokens/s")
    print("Throughput per repeat: " + ", ".join(f"{sample:.2f}" for sample in samples))


if __name__ == "__main__":
    main()
