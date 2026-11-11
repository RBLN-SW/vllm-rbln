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
import random

import numpy as np
import pytest
import torch
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

from vllm_rbln.v1.sample.ops.fused_penalty import (
    K_BUCKETS,
    FusedPenaltyState,
    apply_fused_penalty,
    k_buckets,
    row_layout,
)
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

from .test_rbln_sampler import npu_top_k_top_p
from .utils import (
    VOCAB_SIZE,
    assert_samples_match,
    build_input_batch,
    evenly_spaced_token_ids,
    expected_probs,
    logits_from_weights,
)


def dense_reference(logits, prompts, outputs, rep, freq, pres, biases):
    """Upstream order: logit bias, then repetition, frequency, presence."""
    out = logits.clone()
    for i in range(len(prompts)):
        for t, v in biases.get(i, {}).items():
            out[i, t] += v
        for t in set(prompts[i]) | set(outputs[i]):
            out[i, t] = out[i, t] / rep[i] if out[i, t] > 0 else out[i, t] * rep[i]
        for t in set(outputs[i]):
            out[i, t] -= freq[i] * outputs[i].count(t) + pres[i]
    return out


@pytest.mark.parametrize(
    ("num_rows", "vocab", "base"),
    [
        (1, 16384, K_BUCKETS),
        (4, 16384, K_BUCKETS),
        (16, 16384, K_BUCKETS),
        # ~150 chunks per request outgrow 64 slots: k moves up a bucket.
        (4, 16384, (64, 128)),
    ],
)
def test_state_matches_dense_across_steps(num_rows, vocab, base):
    """Requests join, leave, reorder and grow one token per step."""
    rng = random.Random(0)
    torch.manual_seed(0)
    state = FusedPenaltyState(base)
    buckets = k_buckets(row_layout(num_rows, vocab)[1], base)
    ks = []
    reqs: dict[str, dict] = {}
    for step in range(12):
        if step % 4 == 0:
            while len(reqs) < num_rows:
                reqs[f"r{rng.randrange(10**9)}"] = dict(
                    prompt=[rng.randrange(vocab) for _ in range(170)],
                    outputs=[],
                    rep=rng.choice([1.0, 1.05, 1.2]),
                    freq=rng.choice([0.0, 0.5]),
                    pres=rng.choice([0.0, 0.3]),
                    bias={rng.randrange(vocab): rng.choice([-100.0, 100.0, 2.5])},
                )
        if step % 5 == 3 and len(reqs) > 1:
            reqs.pop(next(iter(reqs)))
        ids = list(reqs)
        rng.shuffle(ids)
        for rid in ids:
            r = reqs[rid]
            r["outputs"].append(rng.choice(r["prompt"] + [rng.randrange(vocab)]))
        live = [reqs[r] for r in ids]

        plan = state.update(
            ids,
            lambda i, live=live: np.asarray(live[i]["prompt"]),
            [r["outputs"] for r in live],
            np.array([r["rep"] for r in live]),
            np.array([r["freq"] for r in live]),
            np.array([r["pres"] for r in live]),
            {i: r["bias"] for i, r in enumerate(live)},
            vocab,
            num_rows,
        )
        assert plan is not None
        rows, row_vocab, k, cidx, lanes = plan
        assert k in buckets
        ks.append(k)
        logits = torch.randn(num_rows, vocab) * 3
        got = apply_fused_penalty(
            logits.clone(), cidx, lanes.float(), rows, row_vocab, k
        )
        want = dense_reference(
            logits[: len(ids)],
            [r["prompt"] for r in live],
            [r["outputs"] for r in live],
            [float(np.float16(r["rep"])) for r in live],
            [r["freq"] for r in live],
            [r["pres"] for r in live],
            {i: r["bias"] for i, r in enumerate(live)},
        )
        torch.testing.assert_close(got[: len(ids)], want, atol=1e-3, rtol=1e-3)
        assert torch.equal(got[len(ids) :], logits[len(ids) :])
    assert max(ks) > buckets[0] or len(buckets) == 1


def _update(state, prompts, vocab, num_rows):
    """One step over `prompts`, a {request id: prompt} dict."""
    return state.update(
        list(prompts),
        lambda i: np.asarray(list(prompts.values())[i]),
        [[] for _ in prompts],
        np.full(len(prompts), 1.05),
        np.zeros(len(prompts)),
        np.zeros(len(prompts)),
        {},
        vocab,
        num_rows,
    )


def test_k_buckets_stay_within_the_row():
    # 1187 and 2374 chunks: K past the row's chunk count fails to compile.
    assert k_buckets(151936 // 2) == (256, 512, 1024, 1152)
    assert k_buckets(151936) == (256, 512, 1024, 2368)


def test_k_drops_back_after_a_long_request_leaves():
    state = FusedPenaltyState((64, 128))
    vocab = 16384  # 256 chunks per row
    long_prompt = [c * 64 for c in range(200)]
    assert _update(state, {"long": long_prompt, "a": [1]}, vocab, 2)[2] == 256
    assert _update(state, {"a": [1], "b": [3]}, vocab, 2)[2] == 64


def test_a_prefill_step_keeps_the_decoding_requests():
    # Prefill and decode never share a step, so a prefill step lists only the
    # new request. The decoding ones must not be rebuilt after it.
    state = FusedPenaltyState()
    vocab = 16384
    decode = {"a": [1, 2, 3], "b": [4, 5, 6]}
    _update(state, decode, vocab, 2)
    kept = {rid: state._reqs[rid] for rid in decode}
    _update(state, {"c": [7, 8]}, vocab, 1)
    _update(state, decode, vocab, 2)
    assert all(state._reqs[rid] is req for rid, req in kept.items())
    state.forget(["a", "c"])
    assert set(state._reqs) == {"b"}


def test_frequency_only_request_takes_no_prompt_slots():
    state = FusedPenaltyState()
    vocab = 16384
    prompt = [c * 64 for c in range(200)]
    step = state.update(
        ["r"],
        lambda i: np.asarray(prompt),
        [[3, 3, 5000]],
        np.ones(1),
        np.full(1, 0.5),
        np.zeros(1),
        {},
        vocab,
        2,
    )
    assert step is not None
    assert state._reqs["r"].used == 2  # chunks of 3 and 5000 only


def test_request_past_the_cap_keeps_the_upstream_path():
    vocab = 151936  # 2374 chunks, cap 2368
    fits = {"r": [c * 64 for c in range(2368)]}
    over = {"r": [c * 64 for c in range(2370)]}
    assert _update(FusedPenaltyState(), fits, vocab, 2)[2] == 2368
    assert _update(FusedPenaltyState(), over, vocab, 2) is None


def _run_sampler(params, logits, fused: bool, expect_kinds: set[str]):
    device = current_platform.device_type
    batch = build_input_batch(
        [(str(i), p) for i, p in enumerate(params)],
        device=device,
        prompt_token_ids=[11, 4000, 9000],
        output_token_ids=[4000, 77],
    )
    sampler = RBLNSampler(fused_sampling=fused)
    out = sampler(logits.clone().to(device), batch.sampling_metadata, input_batch=batch)
    # The fused run must take the fused graphs, not fall back to upstream.
    assert {key[0] for key in sampler._fused_graphs} == expect_kinds
    return out.sampled_token_ids.view(-1).cpu()


def _logits(rows: int) -> torch.Tensor:
    logits = torch.full((rows, VOCAB_SIZE), -2.0)
    logits[:, 4000] = 5.0  # seen: penalties push it below 4.5
    logits[:, 555] = 4.5
    return logits.to(torch.bfloat16)


@pytest.mark.use_device
def test_sampler_matches_dense_path():
    """Greedy picks under penalties and logit bias match the upstream path."""
    params = [
        SamplingParams(temperature=0.0, repetition_penalty=1.3),
        SamplingParams(temperature=0.0, frequency_penalty=2.0, presence_penalty=1.0),
        SamplingParams(temperature=0.0, logit_bias={123: 100.0}),
        SamplingParams(
            temperature=0.0, logit_bias={4000: -100.0}, repetition_penalty=1.1
        ),
    ]
    logits = _logits(len(params))
    dense = _run_sampler(params, logits, fused=False, expect_kinds=set())
    fused = _run_sampler(params, logits, fused=True, expect_kinds={"greedy"})
    assert fused.tolist() == dense.tolist() == [555, 555, 123, 555]


@pytest.mark.use_device
def test_min_p_reads_penalized_logits():
    """With min_p on, penalties apply before it, through the penalize graph."""
    params = [
        SamplingParams(temperature=1.0, min_p=0.5, repetition_penalty=1.3),
        SamplingParams(temperature=1.0, min_p=0.5, logit_bias={123: 100.0}),
    ]
    logits = _logits(len(params))
    logits[:, 555] = 10.0  # min_p keeps only the clear winner
    logits[1, 123] = -100.0  # bias lifts it to 0, still below 555
    dense = _run_sampler(params, logits, fused=False, expect_kinds=set())
    fused = _run_sampler(params, logits, fused=True, expect_kinds={"penalize"})
    assert fused.tolist() == dense.tolist() == [555, 555]


# The weighted row: WEIGHTED[i] weighs i + 1. The heaviest tokens are the ones the
# prompt and output repeat, so penalties and bias reshape the distribution.
WEIGHTED = evenly_spaced_token_ids(num_ids=32).tolist()
PROMPT = WEIGHTED[26:32]
OUTPUT = [WEIGHTED[31], WEIGHTED[30], WEIGHTED[31]]
BIAS = {WEIGHTED[20]: 1.5, WEIGHTED[29]: -100.0}
ROWS, LAUNCHES = 256, 40


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(
            temperature=0.8,
            top_p=0.99,
            top_k=30,
            repetition_penalty=1.05,
            logit_bias=BIAS,
        ),
        SamplingParams(
            temperature=1.0,
            repetition_penalty=1.3,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        ),
        SamplingParams(temperature=1.0, top_k=8, logit_bias=BIAS),
        SamplingParams(temperature=1.0, min_p=0.05, repetition_penalty=1.3),
    ],
    ids=["temp_topp_topk_rep_bias", "rep_freq_pres", "topk_bias", "min_p_rep"],
)
@pytest.mark.use_device
def test_sampled_distribution_matches_upstream(params):
    """Samples follow upstream's distribution under penalties and logit bias."""
    device = current_platform.device_type
    row = logits_from_weights(weights=range(1, 33))
    expected = expected_probs(
        logits=row.unsqueeze(0),
        params=[params],
        top_k_top_p=npu_top_k_top_p,
        prompt_token_ids=PROMPT,
        output_token_ids=OUTPUT,
    )[0]
    batch = build_input_batch(
        [(str(i), params) for i in range(ROWS)],
        device=device,
        prompt_token_ids=PROMPT,
        output_token_ids=OUTPUT,
    )
    sampler = RBLNSampler(fused_sampling=True)
    logits = row.expand(ROWS, -1).contiguous().to(device)
    samples = torch.stack(
        [
            sampler(logits.clone(), batch.sampling_metadata, input_batch=batch)
            .sampled_token_ids.view(-1)
            .cpu()
            for _ in range(LAUNCHES)
        ]
    )
    assert sampler._fused_graphs, "the fused graphs were not taken"
    assert_samples_match(samples=samples, probs=expected)
