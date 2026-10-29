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

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import torch
from vllm.sampling_params import _SAMPLING_EPS, SamplingParams
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_rbln.v1.sample.rbln_logits_processor import build_rbln_logitsprocs

# A.X-K2's vocabulary, a production size.
VOCAB_SIZE = 163840


def create_allowed_token_ids(
    batch_size: int,
    vocab_size: int,
    num_allowed_token_ids: int,
    device: torch.device,
) -> torch.Tensor | None:
    mask: torch.Tensor | None = None
    for i in range(batch_size):
        if i % 2 == 1:
            continue
        if mask is None:
            mask = torch.zeros(
                (batch_size, vocab_size), dtype=torch.bool, device=device
            )
        start = min(i, vocab_size - 1)
        end = min(i + num_allowed_token_ids, vocab_size - 1)
        mask[i, start:end] = True
    return mask


def make_input_batch(
    max_num_reqs: int,
    vocab_size: int = VOCAB_SIZE,
    device: torch.device | str = "cpu",
    *,
    cls: type[InputBatch] = InputBatch,
    build: Callable[..., LogitsProcessors] = build_rbln_logitsprocs,
    **cls_kwargs: Any,
) -> InputBatch:
    """An empty `cls` with the logits processors `build` makes."""
    device = torch.device(device)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
        speculative_config=None,
    )
    return cls(
        max_num_reqs=max_num_reqs,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=device,
        vocab_size=vocab_size,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
        logitsprocs=build(vllm_config, device, False, False),
        **cls_kwargs,
    )


def add_requests(
    input_batch: InputBatch,
    requests: Sequence[tuple[str, SamplingParams]],
    *,
    prompt_token_ids: Sequence[int] = (1, 2, 3),
    output_token_ids: Sequence[int] = (),
) -> None:
    """Every request shares one prompt and one output so far."""
    for req_id, params in requests:
        input_batch.add_request(
            CachedRequestState(
                req_id=req_id,
                prompt_token_ids=list(prompt_token_ids),
                mm_features=None,
                sampling_params=params,
                pooling_params=None,
                generator=None,
                block_ids=([0, 1],),
                num_computed_tokens=0,
                output_token_ids=list(output_token_ids),
            )
        )


def build_input_batch(
    requests: Sequence[tuple[str, SamplingParams]],
    vocab_size: int = VOCAB_SIZE,
    device: torch.device | str = "cpu",
    *,
    build: Callable[..., LogitsProcessors] = build_rbln_logitsprocs,
    **request_kwargs: Any,
) -> InputBatch:
    """Feed real `SamplingParams` through an `InputBatch`, as the runner does."""
    input_batch = make_input_batch(len(requests), vocab_size, device, build=build)
    add_requests(input_batch, requests, **request_kwargs)
    input_batch.refresh_metadata()
    return input_batch


def weighted_token_ids(num_weighted: int, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Ids spread over the whole vocab, first and last included."""
    return torch.linspace(0, vocab_size - 1, num_weighted).round().long()


def make_logits(
    weights: Sequence[int],
    vocab_size: int = VOCAB_SIZE,
    tail_mass: float = 0.01,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Logits whose softmax weighs `weighted_token_ids()[i]` by `weights[i]` and
    spreads `tail_mass` over the rest. Rounded to `dtype`, so build the expected
    distribution from the row, not from `weights`."""
    logits = torch.empty(vocab_size, dtype=torch.float64)
    total_weight = sum(weights)
    num_tail = vocab_size - len(weights)
    logits.fill_(math.log(tail_mass / (1.0 - tail_mass) * total_weight / num_tail))
    logits[weighted_token_ids(len(weights), vocab_size)] = torch.tensor(
        weights, dtype=torch.float64
    ).log()
    return logits.to(dtype)


def min_margin_logits(
    argmax: int, second: int, vocab_size: int = VOCAB_SIZE
) -> torch.Tensor:
    """A bf16 row where `argmax` beats `second` by one ulp, the smallest margin."""
    logits = torch.full((vocab_size,), -100.0, dtype=torch.bfloat16)
    logits[second] = 10.0
    logits[argmax] = 10.0625
    return logits


def expected_probs(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    *,
    top_k_top_p: Callable[
        [torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor
    ] = apply_top_k_top_p_pytorch,
    **request_kwargs,
) -> torch.Tensor:
    """The per-row distribution upstream's Sampler draws from, built from its own
    steps in float32. A greedy row is one-hot. `top_k_top_p` replaces upstream's
    filter for a kernel that cuts differently."""
    logits = logits.to("cpu", torch.float32, copy=True)
    metadata = build_input_batch(
        [(str(i), p) for i, p in enumerate(params)],
        logits.shape[-1],
        build=build_logitsprocs,
        **request_kwargs,
    ).sampling_metadata
    logits = Sampler().apply_logits_processors(
        logits, metadata, predict_bonus_token=False
    )
    greedy = torch.nn.functional.one_hot(
        logits.argmax(dim=-1), logits.shape[-1]
    ).float()
    if metadata.all_greedy:
        return greedy
    logits = Sampler.apply_temperature(
        logits, metadata.temperature, metadata.all_random
    )
    for processor in metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    probs = top_k_top_p(logits, metadata.top_k, metadata.top_p).softmax(dim=-1)
    is_greedy = (metadata.temperature < _SAMPLING_EPS).unsqueeze(dim=1)
    return torch.where(is_greedy, greedy, probs)


def draw(
    sampler: Callable[[torch.Tensor, SamplingMetadata], SamplerOutput],
    logits: torch.Tensor,
    metadata: SamplingMetadata,
    num_launches: int,
) -> torch.Tensor:
    """Token ids from `num_launches` runs, shaped (num_launches, rows). Each run
    gets a fresh copy, since the sampler scales `logits` in place."""
    return torch.stack(
        [
            sampler(logits.clone(), metadata).sampled_token_ids.view(-1).cpu()
            for _ in range(num_launches)
        ]
    )


def assert_samples_match(
    samples: torch.Tensor, probs: torch.Tensor, min_expected: float = 5.0
) -> None:
    """Chi-squared fit of `samples` against the 1-D `probs`. A rare-token pool
    too small to count folds into the smallest bin instead of being dropped.
    The ~10 sigma bound keeps unseeded runs from flaking."""
    samples = samples.flatten().cpu().long()
    probs = probs.cpu().double()
    counts = torch.bincount(samples, minlength=probs.numel()).double()

    impossible = (probs == 0) & (counts > 0)
    assert not impossible.any(), (
        f"sampled zero-probability tokens {impossible.nonzero().flatten().tolist()}"
    )

    expected = probs * samples.numel()
    common = expected >= min_expected
    rare_observed, rare_expected = counts[~common].sum(), expected[~common].sum()
    observed, expected = counts[common], expected[common]
    if rare_expected >= min_expected:
        observed = torch.cat([observed, rare_observed.view(1)])
        expected = torch.cat([expected, rare_expected.view(1)])
    elif common.any():
        smallest = expected.argmin()
        observed[smallest] += rare_observed
        expected[smallest] += rare_expected

    df = observed.numel() - 1
    if df < 1:
        # One bin left, so only the zero-probability check applies.
        return
    chi2 = ((observed - expected) ** 2 / expected).sum().item()
    bound = df + 10 * math.sqrt(2 * df)
    assert chi2 < bound, f"chi2={chi2:.1f} >= bound={bound:.1f} (df={df})"
