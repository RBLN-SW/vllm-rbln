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

import torch
from vllm.sampling_params import _SAMPLING_EPS, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
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


def build_input_batch(
    requests: Sequence[tuple[str, SamplingParams]],
    vocab_size: int = VOCAB_SIZE,
    device: torch.device | str = "cpu",
    *,
    build: Callable[..., LogitsProcessors] = build_rbln_logitsprocs,
    prompt_token_ids: Sequence[int] = (1, 2, 3),
    output_token_ids: Sequence[int] = (),
) -> InputBatch:
    """Feed real `SamplingParams` through an `InputBatch`, as the runner does.
    Every request shares one prompt and one output so far."""
    device = torch.device(device)
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=len(requests)),
        speculative_config=None,
    )
    input_batch = InputBatch(
        max_num_reqs=len(requests),
        max_model_len=64,
        max_num_batched_tokens=64,
        device=device,
        vocab_size=vocab_size,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
        logitsprocs=build(vllm_config, device, False, False),
    )
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
    input_batch.refresh_metadata()
    return input_batch


def weighted_token_ids(num_weighted: int, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Ids spread over the whole vocab, first and last included."""
    return torch.linspace(0, vocab_size - 1, num_weighted).round().to(torch.int64)


def logits_from_weights(
    weights: Sequence[int],
    vocab_size: int = VOCAB_SIZE,
    tail_mass: float = 0.01,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Weights `weighted_token_ids()` by `weights` and spreads `tail_mass` over
    the rest, rounded to `dtype`. Expected probs come from the row, not `weights`.

    For example, `logits_from_weights(weights=[1, 3], vocab_size=5, tail_mass=0.3)`
    splits 0.7 between tokens 0 and 4 at 1:3. Its softmax is [0.175, 0.1, 0.1,
    0.1, 0.525] before the bf16 rounding and [0.1747, ..., 0.5256] after it.
    """
    logits = torch.empty(vocab_size, dtype=torch.float32)
    total_weight = sum(weights)
    num_tail = vocab_size - len(weights)
    logits.fill_(math.log(tail_mass / (1.0 - tail_mass) * total_weight / num_tail))
    logits[weighted_token_ids(num_weighted=len(weights), vocab_size=vocab_size)] = (
        torch.tensor(weights, dtype=torch.float32).log()
    )
    return logits.to(dtype)


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
        requests=[(str(i), p) for i, p in enumerate(params)],
        vocab_size=logits.shape[-1],
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


def assert_samples_match(
    samples: torch.Tensor, probs: torch.Tensor, min_expected: float = 5.0
) -> None:
    """Chi-squared fit of `samples` against the 1-D `probs`. A rare-token pool
    too small to count folds into the smallest bin instead of being dropped.
    The ~10 sigma bound keeps unseeded runs from flaking."""
    samples = samples.flatten().cpu().to(torch.int64)
    probs = probs.cpu().float()
    counts = torch.bincount(samples, minlength=probs.numel()).float()

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
