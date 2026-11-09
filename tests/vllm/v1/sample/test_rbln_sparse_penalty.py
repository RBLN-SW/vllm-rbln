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

from vllm_rbln.v1.sample.ops.sparse_penalty import (
    K_MIN,
    SparsePenaltyState,
    apply_sparse_penalty,
)
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

from .utils import VOCAB_SIZE, build_input_batch


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
    ("num_rows", "vocab", "k_min"),
    [
        (1, 4096, K_MIN),
        (4, 4096, K_MIN),
        (16, 4096, K_MIN),
        # ~150 chunks per request outgrow 64 slots: k moves up and rebuilds.
        (4, 16384, 64),
    ],
)
def test_state_matches_dense_across_steps(num_rows, vocab, k_min):
    """Requests join, leave, reorder and grow one token per step."""
    rng = random.Random(0)
    torch.manual_seed(0)
    state = SparsePenaltyState(k_min=k_min)
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

        rows, row_vocab, k, cidx, lanes = state.update(
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
        logits = torch.randn(num_rows, vocab) * 3
        got = apply_sparse_penalty(
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
    assert state.k > k_min or vocab // 64 <= k_min


def _run_sampler(params, logits, sparse: bool, expect_kinds: set[str]):
    device = current_platform.device_type
    batch = build_input_batch(
        [(str(i), p) for i, p in enumerate(params)],
        device=device,
        prompt_token_ids=[11, 4000, 9000],
        output_token_ids=[4000, 77],
    )
    sampler = RBLNSampler(sparse_penalty=sparse)
    out = sampler(logits.clone().to(device), batch.sampling_metadata, input_batch=batch)
    # The sparse run must take the fused graphs, not fall back to upstream.
    assert {key[0] for key in sampler._sparse_graphs} == expect_kinds
    return out.sampled_token_ids.view(-1).cpu()


def _logits(rows: int) -> torch.Tensor:
    logits = torch.full((rows, VOCAB_SIZE), -2.0)
    logits[:, 4000] = 5.0  # seen: penalties push it below 4.5
    logits[:, 555] = 4.5
    return logits.to(torch.bfloat16)


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
    dense = _run_sampler(params, logits, sparse=False, expect_kinds=set())
    sparse = _run_sampler(params, logits, sparse=True, expect_kinds={"greedy"})
    assert sparse.tolist() == dense.tolist() == [555, 555, 123, 555]


def test_min_p_reads_penalized_logits():
    """With min_p on, penalties apply before it, through the penalize graph."""
    params = [
        SamplingParams(temperature=1.0, min_p=0.5, repetition_penalty=1.3),
        SamplingParams(temperature=1.0, min_p=0.5, logit_bias={123: 100.0}),
    ]
    logits = _logits(len(params))
    logits[:, 555] = 10.0  # min_p keeps only the clear winner
    logits[1, 123] = -100.0  # bias lifts it to 0, still below 555
    dense = _run_sampler(params, logits, sparse=False, expect_kinds=set())
    sparse = _run_sampler(params, logits, sparse=True, expect_kinds={"penalize"})
    assert sparse.tolist() == dense.tolist() == [555, 555]
