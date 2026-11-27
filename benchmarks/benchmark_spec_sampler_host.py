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

"""Measure the RBLN rejection sampler with its compiled NPU graph.

Run from the repository root with
`uv run --no-sync python -m benchmarks.benchmark_spec_sampler_host`.
For a revision comparison, run this same script in each checkout.
"""

import argparse
from statistics import median
from time import perf_counter_ns

import torch

from benchmarks.benchmark_sampler import _create_sampling_metadata_from_config
from vllm_rbln.v1.sample.rbln_rejection_sampler import RBLNRejectionSamplerImpl


@torch.inference_mode()
def measure(
    impl: RBLNRejectionSamplerImpl,
    args: argparse.Namespace,
    change_every: int,
    partial_every: int,
    inference_metadata: bool,
) -> float:
    batch = args.batch_size
    width = args.spec_tokens
    vocab = args.vocab_size

    with torch.inference_mode(inference_metadata):
        metadata = _create_sampling_metadata_from_config(
            {
                "temperature": 0.8,
                "all_greedy": args.greedy,
                "all_random": not (args.greedy or args.mixed),
                "top_k": None if args.greedy else min(40, vocab),
                "top_p": None if args.greedy else 0.9,
                "no_penalties": True,
            },
            batch,
            vocab,
            torch.device("cpu"),
        )
        temperature = metadata.temperature
        assert temperature is not None
        if args.greedy:
            metadata.temperature = None
        elif args.mixed:
            temperature[1::2] = 0
            assert metadata.top_p is not None
            metadata.top_p[1::2] = 1

    full_counts = [width] * batch
    partial_counts = [*full_counts[:-1], width - 1]
    draft = torch.zeros(batch * width, dtype=torch.int32)
    logits = torch.zeros(batch * width, vocab)
    bonus_logits = torch.zeros(batch, vocab)
    full_cumulative = torch.tensor(full_counts, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    partial_cumulative = full_cumulative.clone()
    partial_cumulative[-1] -= 1
    full = (draft, full_counts, width, full_cumulative, logits)
    partial = (
        draft[:-1],
        partial_counts,
        max(partial_counts),
        partial_cumulative,
        logits[:-1],
    )
    impl.invalidate_sampling_param_cache()

    def run(steps: int) -> None:
        for step in range(steps):
            if change_every and (step + 1) % change_every == 0:
                temperature[0] = 0.7 if (step // change_every) % 2 == 0 else 0.8
                impl.invalidate_sampling_param_cache()
            ids, counts, max_drafts, cumulative, target = (
                partial if partial_every and (step + 1) % partial_every == 0 else full
            )
            output = impl.rejection_sample(
                ids,
                counts,
                max_drafts,
                cumulative,
                None,
                target,
                None,
                metadata,
                bonus_logits=bonus_logits,
                reuse_sampling_param_tensors=True,
            )
            # Reading the output waits for the runtime to finish sampling.
            output.tolist()

    run(args.warmup)
    timings = []
    for _ in range(args.repeats):
        start = perf_counter_ns()
        run(args.steps)
        timings.append((perf_counter_ns() - start) / args.steps / 1_000)
    return median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--greedy", action="store_true", help="Use only greedy rows.")
    sampling.add_argument(
        "--mixed", action="store_true", help="Use greedy and random rows."
    )
    args = parser.parse_args()
    if args.batch_size < 1 or (args.mixed and args.batch_size < 2):
        parser.error("--batch-size must be positive, and at least 2 with --mixed")
    if not 2 <= args.spec_tokens <= 32:
        parser.error("--spec-tokens must be between 2 and 32")
    if args.vocab_size < 2 or args.steps < 32 or args.repeats < 1 or args.warmup < 16:
        parser.error(
            "need vocab size >= 2, steps >= 32, repeats >= 1, and warmup >= 16"
        )

    torch.set_num_threads(1)
    impl = RBLNRejectionSamplerImpl(num_spec_tokens=args.spec_tokens)
    with torch.inference_mode():
        impl.initialize_sampling_inputs(
            args.batch_size,
            args.vocab_size,
            torch.get_default_dtype(),
            torch.get_default_device(),
        )
    cases = [
        ("stable full batch", 0, 0, False),
        ("params change every 16 steps", 16, 0, False),
        ("params change every step", 1, 0, False),
        ("partial drafts every 8 steps", 0, 8, False),
        ("stable inference tensors", 0, 0, True),
    ]
    mode = "greedy" if args.greedy else "mixed" if args.mixed else "random"
    print(
        f"RBLN rejection sampler | B={args.batch_size}, K={args.spec_tokens}, "
        f"V={args.vocab_size} | {mode}"
    )
    print(f"Median of {args.repeats} runs with {args.steps} steps each")
    print(f"{'Workload':32} {'us/step':>12}")
    for label, change_every, partial_every, inference_metadata in cases:
        if args.greedy and (change_every or inference_metadata):
            continue
        elapsed = measure(impl, args, change_every, partial_every, inference_metadata)
        print(f"{label:32} {elapsed:12.2f}")


if __name__ == "__main__":
    main()
