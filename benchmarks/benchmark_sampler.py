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
import contextlib
import time

import numpy as np
import rebel
import torch
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample import WARM_UP_CONFIGS, RBLNSampler
from vllm_rbln.v1.worker.metrics import PerformanceTracker, collect_metrics

MAX_NUM_PROMPT_TOKENS = 64


def _create_penalty_tensor(
    batch_size: int, penalty_value: float, device: torch.device
) -> torch.Tensor:
    return torch.full(
        (batch_size,), fill_value=penalty_value, dtype=torch.float, device=device
    )


def _create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


def _create_sampling_metadata_from_config(
    config: dict,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    num_output_tokens: int = 1,
) -> SamplingMetadata:
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist()
        )
        prompt_token_ids.append(
            np.random.randint(
                0, vocab_size, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
            ).tolist()
        )

    temperature = torch.full((batch_size,), config["temperature"], device=device)

    top_p = None
    if config.get("top_p") is not None:
        top_p = torch.full(
            (batch_size,), config["top_p"], dtype=torch.float32, device=device
        )

    top_k = None
    if config.get("top_k") is not None:
        top_k = torch.full(
            (batch_size,), config["top_k"], dtype=torch.int32, device=device
        )

    no_penalties = config["no_penalties"]
    freq_val = config.get("frequency_penalties", 0.0)
    pres_val = config.get("presence_penalties", 0.0)
    rep_val = config.get("repetition_penalties", 1.0)

    prompt_tokens_tensor = None
    if not no_penalties:
        prompt_tokens_tensor = _create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        )

    fake_sampling_metadata = SamplingMetadata(
        temperature=temperature,
        all_greedy=config["all_greedy"],
        all_random=config["all_random"],
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=prompt_tokens_tensor,
        output_token_ids=output_token_ids,
        spec_token_ids=[[] for _ in range(batch_size)],
        frequency_penalties=_create_penalty_tensor(batch_size, freq_val, device),
        presence_penalties=_create_penalty_tensor(batch_size, pres_val, device),
        repetition_penalties=_create_penalty_tensor(batch_size, rep_val, device),
        no_penalties=no_penalties,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    return fake_sampling_metadata


def _create_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    return torch.randn(batch_size, vocab_size, device="cpu")


def run_benchmark(
    benchmark_config: dict,
    batch_size: int,
    vocab_size: int,
    warmup_iters: int,
    benchmark_iters: int,
):
    torch._dynamo.config.recompile_limit = len(WARM_UP_CONFIGS)
    sampler = RBLNSampler(seed=42)
    sampler = torch.compile(sampler, dynamic=False, fullgraph=False)
    sampler_performance_tracker = PerformanceTracker("SAMPLER")

    logits = _create_logits(batch_size, vocab_size)

    # warmup: iterate over all WARM_UP_CONFIGS, matching actual vllm-rbln behavior
    for _ in range(warmup_iters):
        for config in WARM_UP_CONFIGS:
            sampling_metadata = _create_sampling_metadata_from_config(
                config=config,
                batch_size=batch_size,
                vocab_size=vocab_size,
                device=logits.device,
            )
            sampler(logits, sampling_metadata)

    print(f"Running benchmark: {benchmark_config['name']}")
    print(f"Batch size: {batch_size}, Vocab size: {vocab_size}")
    print()

    # benchmark run with the user-specified config
    sampling_metadata = _create_sampling_metadata_from_config(
        config=benchmark_config,
        batch_size=batch_size,
        vocab_size=vocab_size,
        device=logits.device,
    )

    for _ in range(benchmark_iters):
        if hasattr(rebel, "capture_reports"):
            capture_ctx = rebel.capture_reports()
        else:
            capture_ctx = contextlib.nullcontext()
        start_time = time.perf_counter()
        with capture_ctx as model_reports:
            sampler(logits, sampling_metadata)
        collect_metrics(
            sampler_performance_tracker,
            is_prefill=False,
            start_time=start_time,
            end_time=time.perf_counter(),
            reports=model_reports,
            token_count=0,
        )
    sampler_performance_tracker.print_final_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton vs PyTorch sort-based top-k/top-p implementations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to test (default: 1)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=151936,
        help="Vocabulary size to test (default: 151936)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=8,
        help="Number of warmup iterations (default: 8)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=8,
        help="Number of benchmark iterations (default: 8)",
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        choices=[c["name"] for c in WARM_UP_CONFIGS],
        default="no_penalty_greedy",
        help=f"Benchmark config name (default: no_penalty_greedy). "
        f"Choices: {[c['name'] for c in WARM_UP_CONFIGS]}",
    )

    args = parser.parse_args()

    # Find the benchmark config by name
    benchmark_config = None
    for config in WARM_UP_CONFIGS:
        if config["name"] == args.benchmark_config:
            benchmark_config = config
            break
    if benchmark_config is None:
        raise ValueError(f"Unknown benchmark config: {args.benchmark_config}")

    # Print configuration
    print(f"Batch size: {args.batch_size}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Benchmark config: {args.benchmark_config}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.benchmark_iters}")
    print()

    run_benchmark(
        benchmark_config=benchmark_config,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )


if __name__ == "__main__":
    main()
