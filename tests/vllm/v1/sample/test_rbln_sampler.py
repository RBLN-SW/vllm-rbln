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
from collections.abc import Callable
from typing import Any

import pytest
import torch
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler
from vllm_rbln.v1.worker.rbln_model_runner import (
    _depad_sampler_output,
    _pad_sampling_metadata,
)

from . import utils
from .utils import (
    assert_samples_match,
    build_input_batch,
    expected_probs,
    logits_from_weights,
    weighted_token_ids,
)

# Logits arrive in the model's dtype, and RBLN serves all three.
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture
def sampler() -> RBLNSampler:
    return RBLNSampler()


@pytest.fixture
def op_args(sampler) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Record the (top_k, top_p) tensors the sampling op receives."""
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = sampler.topk_topp_sampler._compiled_rbln_topk_topp_sampler

    def spy(logits, temperature, k, p):
        recorded.append((k, p))
        return original(logits, temperature, k, p)

    sampler.topk_topp_sampler._compiled_rbln_topk_topp_sampler = spy
    return recorded


def near_tied_logits(
    top_token: int,
    second_token: int,
    vocab_size: int = utils.VOCAB_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """A row where `top_token` beats `second_token` by the smallest margin the
    NPU resolves in `dtype`, which a greedy row must still pick."""
    logits = torch.full((vocab_size,), -100.0, dtype=dtype)
    logits[second_token] = 10.0
    # One ulp is 8 * eps in [8, 16). The NPU reads one float16 ulp, or a float32
    # gap below 0.0078, as a tie and takes the lower id, so neither margin goes
    # below two float16 ulps.
    float16_ulp = 8 * torch.finfo(torch.float16).eps
    logits[top_token] = 10.0 + max(8 * torch.finfo(dtype).eps, 2 * float16_ulp)
    return logits


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.use_device
def test_all_greedy_batch_takes_the_argmax_op(sampler, op_args, dtype):
    """`rbln::argmax` must resolve the near tie."""
    row = near_tied_logits(
        top_token=WEIGHTED[20], second_token=WEIGHTED[5], dtype=dtype
    )

    samples = sample(
        logits=row.expand(ROWS, -1),
        params=[SamplingParams(temperature=0.0)] * ROWS,
        launches=2,
        sampler=sampler,
    )

    # An all-greedy batch skips the top-k/top-p op entirely.
    assert op_args == []
    assert (samples == WEIGHTED[20]).all(), samples.unique().tolist()


########################### sampled distributions ###########################
# Every row shares one logits row: WEIGHTED[i] weighs i + 1, the tail holds 1%.
ROWS = 256
WEIGHTED = weighted_token_ids(num_weighted=32).tolist()
TAIL_TOKEN = 1
LAUNCHES = 40


def sample(
    logits: torch.Tensor,
    params: list[SamplingParams],
    launches: int = LAUNCHES,
    sampler: Callable[[torch.Tensor, SamplingMetadata], SamplerOutput] | None = None,
    **request_kwargs: Any,
) -> torch.Tensor:
    """Token ids shaped (launches, rows), drawn from `sampler`,
    which wraps one as the runner does."""
    device = current_platform.device_type
    metadata = build_input_batch(
        requests=[(str(i), p) for i, p in enumerate(params)],
        device=device,
        **request_kwargs,
    ).sampling_metadata
    logits = logits.contiguous().to(device)
    sampler = sampler or RBLNSampler()
    # A fresh copy per launch, since the sampler scales `logits` in place.
    return torch.stack(
        [
            sampler(logits.clone(), metadata).sampled_token_ids.view(-1).cpu()
            for _ in range(launches)
        ]
    )


def npu_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """How `rbln::top_k_top_p` cuts on the NPU: the intersection of top-k and
    top-p, without upstream's renormalization over the top-k."""
    probs, order = logits.softmax(dim=-1).sort(dim=-1, descending=True)
    keep = torch.ones_like(probs, dtype=torch.bool)
    if k is not None:
        keep &= torch.arange(probs.shape[-1]) < k.unsqueeze(dim=1)
    if p is not None:
        above = probs.cumsum(dim=-1) - probs
        keep &= above < p.unsqueeze(dim=1)
        keep[:, 0] = True
    mask = torch.empty_like(keep).scatter_(-1, order, keep)
    return logits.masked_fill(~mask, float("-inf"))


def case(params: SamplingParams, id: str, **request_kwargs: Any):
    return pytest.param(params, request_kwargs, id=id)


DISTRIBUTION_CASES = [
    # Temperature alone.
    case(SamplingParams(temperature=0.5), "temperature_0.5"),
    case(SamplingParams(temperature=1.0), "temperature_1.0"),
    case(SamplingParams(temperature=1.5), "temperature_1.5"),
    # Upstream's TestFlashInferDistributionMatch cases, 0.98 for 0.99: a 0.99
    # nucleus ends at the 1% tail, where bf16 rounding decides the cut.
    case(SamplingParams(top_k=8), "top_k_8"),
    case(SamplingParams(top_k=16), "top_k_16"),
    case(SamplingParams(top_p=0.5), "top_p_0.5"),
    case(SamplingParams(top_p=0.7), "top_p_0.7"),
    case(SamplingParams(top_p=0.98), "top_p_0.98"),
    # Both set: the NPU and upstream keep all of the top 8.
    case(SamplingParams(top_k=8, top_p=0.9), "top_k_8_top_p_0.9"),
    # Upstream keeps 2 and the NPU all 4.
    case(SamplingParams(top_k=4, top_p=0.5), "top_k_4_top_p_0.5"),
    # Temperature scales the logits before top-p cuts the nucleus.
    case(SamplingParams(temperature=0.7, top_p=0.8), "temperature_top_p"),
    # min_p compares against the temperature-scaled distribution.
    case(SamplingParams(temperature=1.5, min_p=0.3), "min_p"),
    case(
        SamplingParams(temperature=0.7, min_p=0.05, top_k=16, top_p=0.9),
        "min_p_top_k_top_p",
    ),
    # Penalties reshape the logits before top-p sees them.
    case(
        SamplingParams(
            presence_penalty=0.5,
            frequency_penalty=0.3,
            repetition_penalty=1.2,
            top_p=0.9,
        ),
        "penalties_top_p",
        prompt_token_ids=[WEIGHTED[29]],
        output_token_ids=[WEIGHTED[31], WEIGHTED[30], WEIGHTED[30]],
    ),
    # The bias lifts the least likely weighted token into the top-k.
    case(SamplingParams(logit_bias={WEIGHTED[0]: 4.0}, top_k=8), "logit_bias_top_k"),
    case(
        SamplingParams(
            allowed_token_ids=[*WEIGHTED[16:], TAIL_TOKEN], top_p=0.9, min_p=0.05
        ),
        "allowed_token_ids_top_p_min_p",
    ),
    # The tokenizer normally fills `_bad_words_token_ids` in from `bad_words`.
    case(
        SamplingParams(
            _bad_words_token_ids=[[WEIGHTED[31]], [WEIGHTED[28]]],
            top_p=0.9,
            min_p=0.05,
        ),
        "bad_words_top_p_min_p",
    ),
    # Stop tokens stay masked until min_tokens are out.
    case(
        SamplingParams(
            temperature=0.8, min_tokens=4, stop_token_ids=[WEIGHTED[31], WEIGHTED[29]]
        ),
        "min_tokens",
    ),
]


@pytest.mark.parametrize(("params", "request_kwargs"), DISTRIBUTION_CASES)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.use_device
def test_sampled_distribution(params, request_kwargs, dtype):
    row = logits_from_weights(weights=range(1, 33), dtype=dtype)
    expected = expected_probs(
        logits=row.unsqueeze(0),
        params=[params],
        top_k_top_p=npu_top_k_top_p,
        **request_kwargs,
    )[0]

    samples = sample(
        logits=row.expand(ROWS, -1), params=[params] * ROWS, **request_kwargs
    )

    assert_samples_match(samples=samples, probs=expected)


@pytest.mark.use_device
def test_structured_output_mask_with_top_p_min_p():
    """A token masked to -inf must never come back, whatever the filters keep."""
    row = logits_from_weights(weights=range(1, 33))
    row[[WEIGHTED[31], WEIGHTED[30], *range(WEIGHTED[0] + 1, WEIGHTED[1])]] = float(
        "-inf"
    )
    params = SamplingParams(top_p=0.9, min_p=0.05)
    expected = expected_probs(logits=row.unsqueeze(0), params=[params])[0]

    samples = sample(logits=row.expand(ROWS, -1), params=[params] * ROWS)

    assert_samples_match(samples=samples, probs=expected)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        # torch.float16,
        torch.float32,
    ],
    ids=str,
)
@pytest.mark.use_device
def test_mixed_batch_keeps_each_row_to_its_own_params(dtype):
    """One op call serves the mixed batch, greedy rows as top_k=1. Each row must
    keep its own distribution."""
    weighted_row = logits_from_weights(weights=range(1, 33), dtype=dtype)
    groups = [
        (
            near_tied_logits(
                top_token=WEIGHTED[20], second_token=WEIGHTED[5], dtype=dtype
            ),
            SamplingParams(temperature=0.0),
        ),
        (weighted_row, SamplingParams(temperature=1.0)),
        (weighted_row, SamplingParams(temperature=0.7, top_k=8)),
        (weighted_row, SamplingParams(temperature=1.2, top_p=0.7)),
        (
            weighted_row,
            SamplingParams(
                presence_penalty=0.5,
                frequency_penalty=0.3,
                repetition_penalty=1.2,
                top_p=0.9,
            ),
        ),
    ]
    # Only the penalized group reads the history.
    history: dict[str, Any] = {
        "prompt_token_ids": [WEIGHTED[29]],
        "output_token_ids": [WEIGHTED[31], WEIGHTED[30], WEIGHTED[30]],
    }
    per_group = ROWS // len(groups)
    logits = torch.stack([row for row, _ in groups for _ in range(per_group)])
    params = [p for _, p in groups for _ in range(per_group)]
    expected = expected_probs(
        logits=torch.stack([row for row, _ in groups]),
        params=[p for _, p in groups],
        **history,
    )

    samples = sample(logits=logits, params=params, **history)

    for g, probs in enumerate(expected):
        assert_samples_match(
            samples=samples[:, g * per_group : (g + 1) * per_group], probs=probs
        )


@pytest.mark.use_device
def test_padded_bucket_keeps_live_rows_to_their_params():
    """Padding the metadata with copies of the last live row must not change
    any live row's distribution."""
    weighted_row = logits_from_weights(weights=range(1, 33))
    groups = [
        (
            near_tied_logits(top_token=WEIGHTED[20], second_token=WEIGHTED[5]),
            SamplingParams(temperature=0.0),
        ),
        (weighted_row, SamplingParams(temperature=1.5, min_p=0.3)),
        (weighted_row, SamplingParams(logit_bias={WEIGHTED[0]: 4.0}, top_k=8)),
    ]
    per_group = 64
    num_reqs = per_group * len(groups)
    assert num_reqs < ROWS
    params = [p for _, p in groups for _ in range(per_group)]
    logits = torch.stack(
        [row for row, _ in groups for _ in range(per_group)]
        + [weighted_row] * (ROWS - num_reqs)
    )
    expected = expected_probs(
        logits=torch.stack([row for row, _ in groups]),
        params=[p for _, p in groups],
    )

    sampler = RBLNSampler()

    def padded(logits, metadata):
        out = sampler(logits, _pad_sampling_metadata(metadata, ROWS))
        return _depad_sampler_output(out, num_reqs)

    samples = sample(logits=logits, params=params, sampler=padded)

    assert samples.shape[1] == num_reqs
    for g, probs in enumerate(expected):
        assert_samples_match(
            samples=samples[:, g * per_group : (g + 1) * per_group], probs=probs
        )


########################### pairwise exact sampling ###########################
# Four candidates, the rest far below. The parameters pick the winner. Under
# any penalties, A stays at least 1 away from B and C.
A = WEIGHTED[31]  # 9.0, in the history for the penalties; the allow-list drops it
B = WEIGHTED[0]  # 8.0, min_tokens' stop token; the grammar drops it
C = WEIGHTED[16]  # 1.0, the bias lifts it to the top; bad words drop it
D = WEIGHTED[8]  # -4.0, nothing drops it
# History for the penalties, two outputs short of min_tokens=4.
EXACT_HISTORY: dict[str, Any] = {"prompt_token_ids": [A], "output_token_ids": [A, A]}


def exact_logits(structured: bool, dtype: torch.dtype) -> torch.Tensor:
    row = torch.full((utils.VOCAB_SIZE,), -30.0, dtype=dtype)
    row[[A, B, C, D]] = torch.tensor([9.0, 8.0, 1.0, -4.0], dtype=dtype)
    if structured:
        row[B] = float("-inf")
    return row


def exact_case(
    id: str,
    *,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    presence: bool = False,
    frequency: bool = False,
    repetition: bool = False,
    logit_bias: bool = False,
    allowed: bool = False,
    bad_words: bool = False,
    min_tokens: bool = False,
    structured: bool = False,
    mixed: bool = False,
):
    params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        presence_penalty=2.0 if presence else 0.0,
        frequency_penalty=2.0 if frequency else 0.0,
        repetition_penalty=1.5 if repetition else 1.0,
        logit_bias={C: 12.0} if logit_bias else None,
        allowed_token_ids=[B, C, D] if allowed else None,
        min_tokens=4 if min_tokens else 0,
        stop_token_ids=[B] if min_tokens else None,
        _bad_words_token_ids=[[C]] if bad_words else None,
    )
    return pytest.param(params, structured, mixed, id=id)


# Every pair of parameters meets in some row, on and off. A random row leaves
# only the winner by top_k=1, top_p=0.5 or min_p=0.5.
EXACT_CASES = [
    exact_case(
        "random_A_drops_B_C",
        temperature=1.3,
        top_k=1,
        presence=True,
        frequency=True,
        repetition=True,
        bad_words=True,
        min_tokens=True,
        structured=True,
        mixed=True,
    ),
    exact_case(
        "greedy_B_drops_A_C", min_p=0.5, logit_bias=True, allowed=True, bad_words=True
    ),
    exact_case(
        "greedy_C_drops_A_B",
        top_k=1,
        top_p=0.5,
        frequency=True,
        repetition=True,
        allowed=True,
        min_tokens=True,
    ),
    exact_case(
        "random_A_drops_B",
        temperature=0.7,
        top_k=1,
        top_p=0.5,
        min_p=0.5,
        presence=True,
        structured=True,
        mixed=True,
    ),
    exact_case(
        "random_C_lifted_drops_A_B",
        temperature=1.3,
        top_p=0.5,
        presence=True,
        repetition=True,
        logit_bias=True,
        allowed=True,
        min_tokens=True,
        mixed=True,
    ),
    exact_case(
        "greedy_A_drops_B_C",
        top_p=0.5,
        frequency=True,
        logit_bias=True,
        bad_words=True,
        structured=True,
        mixed=True,
    ),
    exact_case(
        "random_C_drops_A_B",
        temperature=0.7,
        min_p=0.5,
        frequency=True,
        repetition=True,
        allowed=True,
        structured=True,
    ),
    exact_case(
        "greedy_C_lifted_drops_B",
        top_k=1,
        min_p=0.5,
        presence=True,
        logit_bias=True,
        min_tokens=True,
    ),
]


@pytest.mark.parametrize(("params", "structured", "mixed"), EXACT_CASES)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.use_device
def test_sample_only_the_winner(params, structured, mixed, dtype):
    """Where the parameters leave one candidate, every draw must be it. A mixed
    row shares the batch with a row of the opposite kind."""
    row = exact_logits(structured=structured, dtype=dtype)
    if params.temperature == 0.0:
        sibling = SamplingParams(temperature=1.0, top_k=1)
    else:
        sibling = SamplingParams(temperature=0.0)
    groups = [params, sibling] if mixed else [params]
    per_group = ROWS // len(groups)
    expected = expected_probs(
        logits=row.expand(len(groups), -1), params=groups, **EXACT_HISTORY
    )

    samples = sample(
        logits=row.expand(ROWS, -1),
        params=[p for p in groups for _ in range(per_group)],
        launches=2,
        **EXACT_HISTORY,
    )

    for g, probs in enumerate(expected):
        winners = probs.nonzero().flatten().tolist()
        assert len(winners) == 1, f"group {g} keeps {winners}, not one token"
        group = samples[:, g * per_group : (g + 1) * per_group]
        assert (group == winners[0]).all(), group.unique().tolist()
