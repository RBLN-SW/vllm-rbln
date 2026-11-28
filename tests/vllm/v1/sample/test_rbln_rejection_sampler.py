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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_rbln.v1.sample import rbln_rejection_sampler as module
from vllm_rbln.v1.sample.ops.top_k_top_p import (
    GREEDY_TOP_K,
    GREEDY_TOP_P,
    build_op_top_k_top_p,
)
from vllm_rbln.v1.sample.rbln_rejection_sampler import (
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
    RBLNRejectionSampler,
    RBLNRejectionSamplerImpl,
)
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

from .utils import VOCAB_SIZE as LARGE_VOCAB_SIZE
from .utils import build_input_batch

DEVICE = torch.device("cpu")
VOCAB_SIZE = 8
NUM_SPEC_TOKENS = 2


def make_sampling_metadata(
    temperature: torch.Tensor | None,
    all_greedy: bool,
    all_random: bool,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> SamplingMetadata:
    """Build a SamplingMetadata carrying only the fields this impl reads.

    Unlike the helper in test_torch_rejection_sampler, this one allows a mixed
    batch (`all_greedy` and `all_random` both False).
    """
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.fixture
def impl(monkeypatch, request) -> RBLNRejectionSamplerImpl:
    monkeypatch.setattr(module, "compile_sampler", lambda target, _: target)
    sampler = RBLNRejectionSamplerImpl(
        compile_context=Mock(),
        num_spec_tokens=NUM_SPEC_TOKENS,
    )
    batch_size = getattr(request, "param", 2)
    if batch_size is not None:
        sampler.initialize_sampling_inputs(
            batch_size, VOCAB_SIZE, torch.float32, DEVICE
        )
    return sampler


########################### build_op_top_k_top_p ###########################
def test_all_greedy_is_rejected():
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    with pytest.raises(AssertionError):
        build_op_top_k_top_p(metadata, 3, VOCAB_SIZE, DEVICE)


def test_all_random_pure_multinomial():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    # Pure multinomial: no filter, so the compiler runs the multinomial kernel.
    assert top_k is None
    assert top_p is None


def test_all_random_top_k_only():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([3, VOCAB_SIZE], dtype=torch.int32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert top_k is metadata.top_k
    assert top_p is None


def test_all_random_top_p_only():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([0.9, 1.0], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert top_k is None
    assert top_p is metadata.top_p


def test_all_random_top_k_and_top_p():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([3, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([0.9, 1.0], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert top_k is metadata.top_k
    assert top_p is metadata.top_p


def test_mixed_pure_multinomial():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([2.0, 0.0]),
        all_greedy=False,
        all_random=False,
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert torch.equal(
        top_k, torch.tensor([VOCAB_SIZE, GREEDY_TOP_K], dtype=torch.int32)
    )
    assert top_p is None


def test_mixed_top_k_only():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
        top_k=torch.tensor([VOCAB_SIZE, 3], dtype=torch.int32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert torch.equal(top_k, torch.tensor([GREEDY_TOP_K, 3], dtype=torch.int32))
    assert top_p is None


def test_mixed_top_p_only():
    """A greedy row is encoded as `top_k == 1`, so `top_k` cannot be elided
    from a mixed batch even when no request uses top-k."""
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert torch.equal(
        top_k, torch.tensor([GREEDY_TOP_K, VOCAB_SIZE], dtype=torch.int32)
    )
    assert torch.equal(top_p, torch.tensor([GREEDY_TOP_P, 0.9], dtype=torch.float32))


def test_mixed_top_k_and_top_p():
    # Row 0 is greedy, and vLLM rewrites its params to the same values a random
    # request without top-k/top-p carries -- which is why the op cannot tell the
    # two apart from `sampling_metadata` alone.
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0, 2.0]),
        all_greedy=False,
        all_random=False,
        top_k=torch.tensor([VOCAB_SIZE, 4, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([1.0, 1.0, 0.9], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 3, VOCAB_SIZE, DEVICE)

    assert torch.equal(
        top_k, torch.tensor([GREEDY_TOP_K, 4, VOCAB_SIZE], dtype=torch.int32)
    )
    assert torch.equal(
        top_p, torch.tensor([GREEDY_TOP_P, 1.0, 0.9], dtype=torch.float32)
    )


def test_greedy_request_of_a_mixed_batch_carries_neutral_top_k_top_p():
    """The upstream contract the greedy detection rests on.

    vLLM hands a greedy request the values that disable both filters --
    `top_p=1.0` and `top_k=vocab_size` -- whatever the caller asked for, which is
    exactly what a random request without filters carries. Temperature is the
    only thing left that marks the row.
    """
    # The greedy request asks for both filters; the random ones are what make
    # vLLM materialize the per-request tensors for the whole batch.
    input_batch = build_input_batch(
        requests=[
            ("greedy", SamplingParams(temperature=0.0, top_k=4, top_p=0.5)),
            ("random_top_k", SamplingParams(temperature=1.0, top_k=4)),
            ("random_top_p", SamplingParams(temperature=1.0, top_p=0.9)),
        ],
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )

    metadata = input_batch.sampling_metadata
    greedy = input_batch.req_id_to_index["greedy"]

    assert not metadata.all_greedy
    assert not metadata.all_random
    assert metadata.top_k is not None
    assert metadata.top_p is not None
    assert metadata.top_k[greedy].item() == VOCAB_SIZE
    assert metadata.top_p[greedy].item() == 1.0
    assert metadata.temperature[greedy].item() == 0.0

    # Which is why the row has to be re-encoded before it reaches the op.
    top_k, top_p = build_op_top_k_top_p(
        metadata, input_batch.num_reqs, VOCAB_SIZE, DEVICE
    )
    assert top_k[greedy].item() == GREEDY_TOP_K
    assert top_p[greedy].item() == GREEDY_TOP_P


# Every kind of row the op can receive, and the (k, p) pair it must arrive with.
# `V` disables top-k and `1.0` disables top-p, so only the greedy row is rewritten.
ROW_KINDS = [
    ("greedy", SamplingParams(temperature=0.0), GREEDY_TOP_K, GREEDY_TOP_P),
    ("pure_random", SamplingParams(temperature=1.0), VOCAB_SIZE, 1.0),
    ("top_k_only", SamplingParams(temperature=1.0, top_k=3), 3, 1.0),
    ("top_p_only", SamplingParams(temperature=1.0, top_p=0.9), VOCAB_SIZE, 0.9),
    ("top_k_top_p", SamplingParams(temperature=1.0, top_k=2, top_p=0.8), 2, 0.8),
]


def test_every_row_kind_gets_its_own_top_k_top_p():
    """All five kinds in one batch, since the op reads k/p per row."""
    input_batch = build_input_batch(
        requests=[(name, params) for name, params, _, _ in ROW_KINDS],
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )

    top_k, top_p = build_op_top_k_top_p(
        input_batch.sampling_metadata, input_batch.num_reqs, VOCAB_SIZE, DEVICE
    )

    for name, _, expected_k, expected_p in ROW_KINDS:
        row = input_batch.req_id_to_index[name]
        assert top_k[row].item() == expected_k, name
        assert top_p[row].item() == pytest.approx(expected_p), name


########################### apply_sampling_constraints ###########################
def test_apply_sampling_constraints_leaves_all_greedy_logits_untouched(impl):
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    out = impl.apply_sampling_constraints(
        logits.clone(),
        cu_num_draft_tokens=torch.tensor([1, 2]),
        sampling_metadata=metadata,
    )

    # The op resolves these rows to their argmax, so no host-side rewrite is
    # needed -- not even temperature scaling.
    assert torch.equal(out, logits)


def test_apply_sampling_constraints_leaves_mixed_batch_logits_untouched(impl):
    """Mixed batches are not scaled here either -- the NPU impl is a no-op."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 2.0]),
        all_greedy=False,
        all_random=False,
    )

    out = impl.apply_sampling_constraints(
        logits.clone(),
        cu_num_draft_tokens=torch.tensor([1, 2]),
        sampling_metadata=metadata,
    )

    assert torch.equal(out, logits)


########################### rejection_sample ###########################
def make_target_probs(argmax_token_ids: list[int]) -> torch.Tensor:
    """One row per draft position, peaked at the given token but far from
    one-hot: the argmax holds 0.4 of the mass and the rest is spread evenly."""
    num_tokens = len(argmax_token_ids)
    probs = torch.full(
        (num_tokens, VOCAB_SIZE),
        0.6 / (VOCAB_SIZE - 1),
        dtype=torch.float32,
    )
    for row, token_id in enumerate(argmax_token_ids):
        probs[row, token_id] = 0.4
    return probs


def run_rejection_sample(
    impl: RBLNRejectionSamplerImpl,
    draft_token_ids: list[int],
    target_argmax_token_ids: list[int],
    bonus_token_ids: list[int],
    metadata: SamplingMetadata,
    num_draft_tokens: list[int] | None = None,
    bonus_from_logits: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    reuse_sampling_param_tensors: bool = False,
) -> torch.Tensor:
    """Run the impl on a batch of drafts packed in `draft_token_ids` order.

    `num_draft_tokens` defaults to every request holding NUM_SPEC_TOKENS drafts.
    `bonus_from_logits` hands the graph bonus logits peaked at `bonus_token_ids`
    instead of the ids, the way a greedy step does.
    """
    if num_draft_tokens is None:
        num_draft_tokens = [NUM_SPEC_TOKENS] * (len(draft_token_ids) // NUM_SPEC_TOKENS)
    return impl.rejection_sample(
        draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int32),
        num_draft_tokens=num_draft_tokens,
        max_spec_len=NUM_SPEC_TOKENS,
        cu_num_draft_tokens=torch.cumsum(
            torch.tensor(num_draft_tokens, dtype=torch.int32), dim=0
        ),
        draft_probs=None,
        target_logits=make_target_probs(target_argmax_token_ids),
        bonus_token_ids=None
        if bonus_from_logits
        else torch.tensor(bonus_token_ids, dtype=torch.int64).unsqueeze(-1),
        sampling_metadata=metadata,
        synthetic_mode=synthetic_conditional_rates is not None,
        synthetic_conditional_rates=synthetic_conditional_rates,
        bonus_logits=make_target_probs(bonus_token_ids) if bonus_from_logits else None,
        reuse_sampling_param_tensors=reuse_sampling_param_tensors,
    )


@pytest.mark.parametrize("impl", [3], indirect=True)
@pytest.mark.parametrize("trial", range(10))
def test_greedy_rows_accept_exactly_the_target_argmax(impl, trial):
    """A greedy row accepts iff its draft is the target argmax, and recovers the
    argmax at the first mismatch.

    `target_probs` puts only 0.4 on the argmax, so a row treated as random would
    diverge from this expectation within a few trials.
    """
    metadata = make_sampling_metadata(
        # Row 0 greedy, row 1 greedy, row 2 random.
        temperature=torch.tensor([0.0, 0.0, 1.0]),
        all_greedy=False,
        all_random=False,
        # The greedy rows carry the params vLLM assigns them, so only the
        # temperature marks them as greedy.
        top_k=torch.tensor([VOCAB_SIZE, VOCAB_SIZE, 1], dtype=torch.int32),
        top_p=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    )

    output = run_rejection_sample(
        impl,
        # Row 0 matches both argmaxes; row 1 mismatches at position 1; row 2
        # (random, top_k=1) matches both argmaxes.
        draft_token_ids=[3, 5, 2, 4, 6, 1],
        target_argmax_token_ids=[3, 5, 2, 7, 6, 1],
        bonus_token_ids=[10, 11, 12],
        metadata=metadata,
    )

    expected = torch.tensor(
        [
            [3, 5, 10],  # all accepted -> bonus token
            [2, 7, PLACEHOLDER_TOKEN_ID],  # accept one, then recover the argmax
            [6, 1, 12],  # top_k=1 row behaves like the greedy rows
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)


def test_greedy_bonus_token_is_the_argmax_of_its_logits(impl):
    """A greedy step hands the graph the bonus rows' logits; their argmax lands
    where the bonus token id would."""
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    output = run_rejection_sample(
        impl,
        # Row 0 matches both argmaxes; row 1 mismatches at position 1.
        draft_token_ids=[3, 5, 2, 4],
        target_argmax_token_ids=[3, 5, 2, 7],
        bonus_token_ids=[6, 1],
        metadata=metadata,
        bonus_from_logits=True,
    )

    expected = torch.tensor(
        [
            [3, 5, 6],  # all accepted -> the bonus row's argmax
            [2, 7, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)


def test_requests_with_fewer_drafts_than_the_padded_length(impl):
    """A request may bring fewer drafts than `num_spec_tokens`, which the impl
    pads to. Its bonus token then lands right after its own last draft."""
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    output = run_rejection_sample(
        impl,
        draft_token_ids=[3, 2, 4],
        target_argmax_token_ids=[3, 2, 7],
        bonus_token_ids=[10, 11],
        metadata=metadata,
        num_draft_tokens=[1, 2],
    )

    expected = torch.tensor(
        [
            # One draft, accepted -> bonus token at column 1, not at column 2.
            [3, 10, PLACEHOLDER_TOKEN_ID],
            [2, 7, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)


def test_random_step_draws_the_bonus_token_in_the_graph(impl):
    """A random step hands the graph the bonus rows' logits too, and the graph
    draws the bonus token under the row's own temperature, top_k and top_p.

    Row 0 is a greedy request of the mixed batch (top_k == 1), so it must land
    on its argmax every time. Row 1 spreads its mass over two tokens and must
    reach both across the draws: an argmax would only ever pick the first.
    """
    metadata = make_sampling_metadata(
        temperature=torch.tensor([GREEDY_TEMPERATURE, 1.0]),
        all_greedy=False,
        all_random=False,
        top_k=torch.tensor([GREEDY_TOP_K, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([GREEDY_TOP_P, 1.0]),
    )
    bonus_logits = torch.full((2, VOCAB_SIZE), -100.0)
    bonus_logits[0, 6] = 0.0
    bonus_logits[0, 1] = -1.0
    bonus_logits[1, 2] = 0.0
    bonus_logits[1, 5] = 0.0
    # Peaked hard enough that every draft is accepted, so the bonus slot is
    # always filled.
    target_logits = make_target_probs([3, 5, 2, 4]) * 100

    torch.manual_seed(0)
    drawn = set()
    for _ in range(32):
        output = impl.rejection_sample(
            draft_token_ids=torch.tensor([3, 5, 2, 4], dtype=torch.int32),
            num_draft_tokens=[2, 2],
            max_spec_len=NUM_SPEC_TOKENS,
            cu_num_draft_tokens=torch.tensor([2, 4], dtype=torch.int32),
            draft_probs=None,
            target_logits=target_logits,
            bonus_token_ids=None,
            sampling_metadata=metadata,
            bonus_logits=bonus_logits,
        )
        assert torch.equal(
            output[:, :2], torch.tensor([[3, 5], [2, 4]], dtype=torch.int32)
        )
        assert int(output[0, 2]) == 6
        drawn.add(int(output[1, 2]))
    assert drawn == {2, 5}


@pytest.mark.use_device
@pytest.mark.parametrize(
    "method",
    [
        {"rejection_sample_method": "standard"},
        {
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": [1.0] * NUM_SPEC_TOKENS,
        },
    ],
    ids=["standard", "synthetic"],
)
# 0.01 is sharp enough that a random draw always lands on the argmax.
@pytest.mark.parametrize("temperature", [0.0, 0.01], ids=["greedy", "random"])
def test_fused_penalties_follow_the_drafts(method, temperature):
    """The graph penalizes each row for the drafts it follows."""
    device = current_platform.device_type
    params = [
        SamplingParams(temperature=temperature, repetition_penalty=1.3),
        SamplingParams(temperature=temperature, frequency_penalty=2.0),
    ]
    batch = build_input_batch(
        [(str(i), p) for i, p in enumerate(params)],
        device=device,
        prompt_token_ids=[11, 4000, 9000],
        output_token_ids=[4000, 77],
    )
    drafts = [[555, 600], [800]]
    for spec_token_ids, ids in zip(batch.spec_token_ids, drafts):
        spec_token_ids.extend(ids)
    # Per row: the argmax, then the token left once it is penalized.
    rows = [(4000, 555), (555, 600), (4000, 800), (600, 700), (800, 900)]
    logits = torch.full((len(rows), LARGE_VOCAB_SIZE), -2.0)
    for row, (first, second) in enumerate(rows):
        logits[row, first] = 5.0
        logits[row, second] = 4.5
    metadata = SpecDecodeMetadata.make_dummy(drafts, torch.device(device))
    metadata.target_logits_indices = torch.arange(3, device=device)
    metadata.bonus_logits_indices = torch.arange(3, 5, device=device)

    sampler = RBLNRejectionSampler(
        RBLNSampler(fused_sampling=True),
        spec_config=SimpleNamespace(num_speculative_tokens=NUM_SPEC_TOKENS, **method),
        use_rbln_sampler=True,
    )
    logits = logits.to(device, torch.bfloat16)
    sampler.initialize_sampling_inputs(
        len(drafts), logits.shape[-1], logits.dtype, logits.device
    )
    graph = Mock(wraps=sampler.impl._compiled_rejection_sample)
    sampler.impl._compiled_rejection_sample = graph
    output = sampler(
        metadata,
        None,
        logits,
        batch.sampling_metadata,
        input_batch=batch,
    )

    # Twelve op inputs, then the plan's cidx and lanes.
    assert len(graph.call_args.args) == 14
    assert output.sampled_token_ids.cpu().tolist() == [
        [555, 600, 700],
        [800, 900, PLACEHOLDER_TOKEN_ID],
    ]


# ---------------------------------------------------------------------------
# Synthetic acceptance (rejection_sample_method="synthetic")
# ---------------------------------------------------------------------------

# The drafts miss the target argmax at both positions, so standard rejection
# would stop at position 0. Only a synthetic rate can accept them.
SYNTHETIC_DRAFTS = [3, 5]
SYNTHETIC_ARGMAX = [6, 7]


def run_synthetic(impl, rates: list[float]) -> torch.Tensor:
    """One greedy request drafting `SYNTHETIC_DRAFTS`, with 1 as its bonus token."""
    return run_rejection_sample(
        impl,
        draft_token_ids=SYNTHETIC_DRAFTS,
        target_argmax_token_ids=SYNTHETIC_ARGMAX,
        bonus_token_ids=[1],
        metadata=make_sampling_metadata(
            temperature=None, all_greedy=True, all_random=False
        ),
        synthetic_conditional_rates=torch.tensor(rates),
    )


# The synthetic count decides how far the row runs; `recovered_token_ids` is
# filled only where the op itself stopped, so the boundary token is always that
# one and never a zeroed slot.
@pytest.mark.parametrize(
    "rates,expected,rule",
    [
        (
            [1.0, 1.0],
            [[3, 5, 1]],
            "every position accepts, so the row runs on to its bonus token",
        ),
        (
            [1.0, 0.0],
            [[3, 6, PLACEHOLDER_TOKEN_ID]],
            "stopping past the op's own stop still carries the op's token, 6, "
            "rather than the zero sitting in position 1's slot",
        ),
        (
            [0.0, 1.0],
            [[6, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
            "a rejection ends the run, so position 1 cannot accept on its own",
        ),
    ],
    ids=["accepts_every_position", "stops_after_the_op", "rejection_ends_the_run"],
)
@pytest.mark.parametrize("impl", [1], indirect=True)
def test_synthetic_rates_decide_the_row(impl, rates, expected, rule):
    assert run_synthetic(impl, rates).tolist() == expected, rule


@pytest.mark.parametrize("impl", [1], indirect=True)
def test_synthetic_rejection_reuses_the_bonus_when_the_op_accepted_everything(impl):
    """An op that accepted every draft recovered nothing, so the only token it
    drew is its bonus; a synthetic rejection has to carry that one."""
    output = run_rejection_sample(
        impl,
        draft_token_ids=SYNTHETIC_ARGMAX,
        target_argmax_token_ids=SYNTHETIC_ARGMAX,
        bonus_token_ids=[1],
        metadata=make_sampling_metadata(
            temperature=None, all_greedy=True, all_random=False
        ),
        synthetic_conditional_rates=torch.tensor([1.0, 0.0]),
    )
    assert output.tolist() == [[6, 1, PLACEHOLDER_TOKEN_ID]]


def test_synthetic_acceptance_is_capped_by_the_drafted_count(impl):
    """Every position accepts, so a request that drafted fewer tokens than the
    padded width must still stop at its own last draft and put its bonus there.
    The second row drafted the full width and is the control."""
    output = run_rejection_sample(
        impl,
        draft_token_ids=[3, 2, 4],
        target_argmax_token_ids=[6, 7, 7],
        bonus_token_ids=[10, 11],
        metadata=make_sampling_metadata(
            temperature=None, all_greedy=True, all_random=False
        ),
        num_draft_tokens=[1, 2],
        synthetic_conditional_rates=torch.ones(NUM_SPEC_TOKENS),
    )
    assert output.tolist() == [
        [3, 10, PLACEHOLDER_TOKEN_ID],
        [2, 4, 11],
    ]


@pytest.mark.parametrize("impl", [1], indirect=True)
def test_synthetic_acceptance_stops_at_an_out_of_range_draft_id(impl):
    """An id at or past the vocabulary counts as a rejection, as in the Torch
    kernels: the row stops there and carries the op's token."""
    output = run_rejection_sample(
        impl,
        draft_token_ids=[3, VOCAB_SIZE],
        target_argmax_token_ids=SYNTHETIC_ARGMAX,
        bonus_token_ids=[1],
        metadata=make_sampling_metadata(
            temperature=None, all_greedy=True, all_random=False
        ),
        synthetic_conditional_rates=torch.ones(NUM_SPEC_TOKENS),
    )
    assert output.tolist() == [[3, 6, PLACEHOLDER_TOKEN_ID]]


# ---------------------------------------------------------------------------
# Staged sampling inputs
# ---------------------------------------------------------------------------


def _recording(impl, monkeypatch):
    """Count top_k/top_p builds and record the temperature the graph reads."""
    builds: list[int] = []
    seen: list[list[float]] = []
    real_build = module.build_op_top_k_top_p

    def build(*args, **kwargs):
        builds.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(module, "build_op_top_k_top_p", build)
    graph = impl._compiled_rejection_sample

    def record(*args):
        seen.append(args[5].tolist())
        return graph(*args)

    impl._compiled_rejection_sample = record
    return builds, seen


def _random_metadata():
    return make_sampling_metadata(
        temperature=torch.tensor([0.5, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([4, 4], dtype=torch.int32),
    )


def test_sampling_param_tensors_follow_fused_penalty_mode(impl):
    inputs = impl._rejection_sampling_inputs
    assert inputs is not None
    metadata = make_sampling_metadata(torch.ones(2), all_greedy=False, all_random=True)
    cumulative = torch.tensor([2, 4], dtype=torch.int32)

    for has_penalty in (False, True, True, False):
        tensors = inputs.prepare_graph_sampling_params(
            metadata,
            cumulative,
            4,
            full_drafts=True,
            bonus_in_graph=True,
            has_penalty=has_penalty,
            reuse_sampling_param_tensors=True,
        )
        if has_penalty:
            assert tensors.top_k_per_request is not None
            assert tensors.top_k_per_request.tolist() == [VOCAB_SIZE] * 2
            assert tensors.top_p_per_request is not None
            assert tensors.top_p_per_request.tolist() == [1.0] * 2
        else:
            assert tensors.top_k_per_request is None
            assert tensors.top_p_per_request is None


def test_sampling_param_tensors_follow_partial_draft_counts(impl, monkeypatch):
    builds, seen = _recording(impl, monkeypatch)
    metadata = _random_metadata()
    for step, (counts, ids) in enumerate(
        [
            ([2, 2], [3, 5, 2, 4]),
            ([2, 1], [3, 5, 2]),
            ([2, 1], [5, 2, 3]),
            ([2, 1], [3, 5, 2]),
            ([1, 2], [3, 5, 2]),
            ([2, 2], [3, 5, 2, 4]),
        ]
    ):
        if step == 3:
            metadata.temperature[1] = 3.0
            impl.invalidate_sampling_param_cache()
        run_rejection_sample(
            impl,
            ids,
            ids,
            [1, 1],
            metadata,
            num_draft_tokens=counts,
            bonus_from_logits=True,
            reuse_sampling_param_tensors=True,
        )

    assert len(builds) == 2
    assert seen == [
        [0.5, 0.5, 2.0, 2.0],
        [0.5, 0.5, 2.0, 1.0],
        [0.5, 0.5, 2.0, 1.0],
        [0.5, 0.5, 3.0, 1.0],
        [0.5, 3.0, 3.0, 1.0],
        [0.5, 0.5, 3.0, 3.0],
    ]


def test_direct_calls_prepare_sampling_param_tensors_every_time(impl, monkeypatch):
    builds, seen = _recording(impl, monkeypatch)
    metadata = _random_metadata()
    run_rejection_sample(impl, [3, 5, 2, 4], [3, 5, 2, 4], [1, 1], metadata)
    metadata.temperature[1] = 3.0
    run_rejection_sample(impl, [3, 5, 2, 4], [3, 5, 2, 4], [1, 1], metadata)
    assert len(builds) == 2
    assert seen[-1] == [0.5, 0.5, 3.0, 3.0]


@pytest.mark.parametrize("impl", [None], indirect=True)
def test_sampling_validates_the_allocated_device(impl):
    requested_device = torch.device("cpu", 0)
    impl.initialize_sampling_inputs(2, VOCAB_SIZE, torch.float32, requested_device)
    run_rejection_sample(impl, [3, 5, 2, 4], [3, 5, 2, 4], [1, 1], _random_metadata())


def test_sampling_rejects_mismatched_batch_size(impl):
    small_batch = make_sampling_metadata(
        temperature=torch.tensor([3.0]), all_greedy=False, all_random=True
    )
    with pytest.raises(AssertionError, match="Rejection sampler input signature"):
        run_rejection_sample(impl, [3, 5], [3, 5], [1], small_batch)


def test_draft_count_buffers_are_reused_across_full_and_partial_steps(
    impl, monkeypatch
):
    graph = impl._compiled_rejection_sample
    addresses = []
    versions = []
    values = []

    def record(*args):
        counts = (args[8], args[2])
        addresses.append(tuple(t.data_ptr() for t in counts))
        versions.append(tuple(t._version for t in counts))
        values.append([t.tolist() for t in counts])
        return graph(*args)

    monkeypatch.setattr(impl, "_compiled_rejection_sample", record)
    metadata = make_sampling_metadata(
        temperature=None, all_greedy=True, all_random=False
    )
    for _ in range(2):
        run_rejection_sample(impl, [3, 5, 2, 4], [3, 5, 2, 4], [1, 1], metadata)
    run_rejection_sample(
        impl, [3, 5, 2], [3, 5, 2], [1, 1], metadata, num_draft_tokens=[2, 1]
    )
    impl.invalidate_sampling_param_cache()
    run_rejection_sample(
        impl, [5, 2, 3], [5, 2, 3], [1, 1], metadata, num_draft_tokens=[2, 1]
    )
    run_rejection_sample(
        impl, [2, 4], [2, 4], [1, 1], metadata, num_draft_tokens=[0, 2]
    )
    for _ in range(2):
        run_rejection_sample(impl, [3, 5, 2, 4], [3, 5, 2, 4], [1, 1], metadata)

    full = [[2, 2], [2, 4]]
    partial = [[2, 1], [2, 3]]
    assert values == [full, full, partial, partial, [[0, 2], [0, 2]], full, full]
    assert len(set(addresses)) == 1
    assert versions[0] == versions[1]
    assert versions[2] == versions[3]
    assert versions[5] == versions[6]
