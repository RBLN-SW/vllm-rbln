import argparse
import contextlib
import time
from dataclasses import dataclass

import numpy as np
import rebel
import torch
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample import WARM_UP_CONFIGS, RBLNSampler
from vllm_rbln.v1.worker.metrics import PerformanceTracker

MAX_NUM_PROMPT_TOKENS = 64


def _collect_metrics(
    performance_tracker: PerformanceTracker,
    is_prefill: bool,
    start_time: float,
    end_time: float,
    reports: list[dict],
    token_count: int,
) -> None:
    execution_time = end_time - start_time
    host_time = None
    device_time = None
    ccl_time = None
    if reports is not None and len(reports) > 0:
        host_time = reports[0].get("total_host", None)
        device_time = reports[0].get("total_device", None)
        ccl_time = reports[0].get("total_ccl", None)
    if is_prefill:
        performance_tracker.record_prefill(
            execution_time,
            token_count,
            host_time=host_time,
            device_time=device_time,
            ccl_time=ccl_time,
        )
    else:
        performance_tracker.record_decode(
            execution_time,
            token_count,
            host_time=host_time,
            device_time=device_time,
            ccl_time=ccl_time,
        )


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


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    temperature: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
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
    if top_p is None and top_k is None:
        is_greedy = True
    else:
        is_greedy = False
    fake_sampling_metadata = SamplingMetadata(
        temperature=temperature,
        all_greedy=is_greedy,
        all_random=not is_greedy,
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=_create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        spec_token_ids=[[] for _ in range(batch_size)],
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    return fake_sampling_metadata


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    batch_size: int
    vocab_size: int
    # k and p can be tensors or None
    temperature: torch.Tensor | None  # [batch_size] or None
    k_values: torch.Tensor | None  # [batch_size] or None
    p_values: torch.Tensor | None  # [batch_size] or None
    description: str = ""


def create_benchmark_configs(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    device: str = "cpu",
    sampling_type: str = "greedy",
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            # Greedy
            if sampling_type == "greedy":
                temperature = torch.full((batch_size,), 0.0, device=device)
                config = BenchmarkConfig(
                    name=f"bs{batch_size}_vocab{vocab_size}",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=None,
                    temperature=temperature,
                    description="Greedy",
                )
                configs.append(config)
            elif sampling_type == "topkp":
                # Top-k / Top-p
                k_values = torch.full(
                    (batch_size,), 30, dtype=torch.int32, device=device
                )
                p_values = torch.full(
                    (batch_size,), 0.99, dtype=torch.float32, device=device
                )
                temperature = torch.full((batch_size,), 0.8, device=device)
                config = BenchmarkConfig(
                    name=f"bs{batch_size}_vocab{vocab_size}_topk30toppt99",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_values,
                    p_values=p_values,
                    temperature=temperature,
                    description="Top-k with k=30 and Top-p with p=0.99",
                )
                configs.append(config)
    return configs


def create_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    # Create random logits for testing
    return torch.randn(batch_size, vocab_size, device="cpu")


def run_benchmark(
    user_config: BenchmarkConfig,
    warmup_configs: list[BenchmarkConfig],
    warmup_iters: int,
    benchmark_iters: int,
):
    torch._dynamo.config.recompile_limit = len(warmup_configs) * len(WARM_UP_CONFIGS)
    sampler = RBLNSampler(seed=42)
    sampler = torch.compile(sampler, dynamic=False, fullgraph=False)
    sampler_performance_tracker = PerformanceTracker("SAMPLER")

    logits = create_logits(user_config.batch_size, user_config.vocab_size)

    # warmup iterations
    for _ in range(warmup_iters):
        for config in warmup_configs:
            sampling_metadata = _create_default_sampling_metadata(
                num_output_tokens=1,
                batch_size=config.batch_size,
                vocab_size=config.vocab_size,
                device=logits.device,
                temperature=config.temperature,
                top_p=config.p_values,
                top_k=config.k_values,
            )
            sampler(logits, sampling_metadata)

    print(f"Running benchmark: {user_config.name}")
    print(f"Description: {user_config.description}")
    print(f"Batch size: {user_config.batch_size}, Vocab size: {user_config.vocab_size}")
    print()

    sampling_metadata = _create_default_sampling_metadata(
        num_output_tokens=1,
        batch_size=user_config.batch_size,
        vocab_size=user_config.vocab_size,
        device=logits.device,
        temperature=user_config.temperature,
        top_p=user_config.p_values,
        top_k=user_config.k_values,
    )

    for _ in range(benchmark_iters):
        # Benchmark iterations
        if hasattr(rebel, "capture_reports"):
            capture_ctx = rebel.capture_reports()
        else:
            # use a dummy context manager that does nothing
            capture_ctx = contextlib.nullcontext()
        start_time = time.perf_counter()
        with capture_ctx as model_reports:
            sampler(logits, sampling_metadata)
        # Collect metrics from the sampler
        _collect_metrics(
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
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to test (default: 1)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[151936],
        help="Vocabulary sizes to test (default: 151936)",
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
        "--sampling-type", type=str, choices=["greedy", "topkp"], default="greedy"
    )

    args = parser.parse_args()

    # Print configuration
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.benchmark_iters}")
    print()

    # Create configs
    warmup_configs = create_benchmark_configs(
        args.batch_sizes,
        args.vocab_sizes,
        device="cpu",
        sampling_type=args.sampling_type,
    )
    user_config = None
    # extract the user specified config
    for config in warmup_configs:
        if (
            args.sampling_type == "greedy"
            and config.description == "Greedy"
            or args.sampling_type == "topkp"
            and config.description.startswith("Top")
        ):
            user_config = config
        else:
            raise ValueError(f"Unexpected sampling type: {args.sampling_type}")

    result = run_benchmark(
        user_config,
        warmup_configs=warmup_configs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )


if __name__ == "__main__":
    main()
