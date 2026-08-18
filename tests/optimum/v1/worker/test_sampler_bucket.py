# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

from vllm_rbln.utils.optimum.bucket import select_bucket_size

from .utils import (
    _schedule_cached_reqs,
    _schedule_new_request_from_request,
    create_model_runner,
    fake_load_model,
    make_request,
)

TOP_TOKEN_ID = 9
RUNNER_UP_TOKEN_ID = 5


@pytest.fixture(autouse=True)
def _rbln_sampler_env(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")


@pytest.fixture(params=["bucket_ladder", "multiple_decoder"])
def make_runner(request):
    """Build a runner with each bucket source prepare_rbln_sampler
    supports: the synthetic get_bucket_sizes ladder (single decoder) and
    compiled decoder_batch_sizes (use_multiple_decoder=True).
    """

    def _make(max_num_seqs):
        if request.param == "multiple_decoder":
            decoder_batch_sizes = tuple(b for b in (1, 4, 8) if b <= max_num_seqs)
            return create_model_runner(
                max_num_seqs=max_num_seqs,
                decoder_batch_sizes=decoder_batch_sizes,
            )
        return create_model_runner(max_num_seqs=max_num_seqs)

    return _make


def _assert_padded_bucket(runner, num_live: int):
    """The last step ran on a padded bucket: the bucket must come from
    the runner's own bucket table, and must exceed the live request
    count so padding rows actually exist.
    """
    assert runner.bucket_size == select_bucket_size(num_live, runner.bucket_sizes)
    assert runner.bucket_size > num_live


def _rig_forward(runner, token_logits: dict[int, float], base: float = 0.0):
    """Make the model emit fixed logits so sampling is predictable.

    Mirrors RBLNOptimumForCausalLM.forward's output contract: prefill
    returns one row for the scheduled request, decode returns one row
    per live request (the model-side bucket padding already sliced off).
    """

    def rigged_forward(model_input, **kwargs):
        num_rows = 1 if model_input.is_prompt else runner.input_batch.num_reqs
        vocab_size = runner.model_config.get_vocab_size()
        logits = torch.full((num_rows, 1, vocab_size), base, dtype=torch.float32)
        for token_id, value in token_logits.items():
            logits[..., token_id] = value
        return logits

    runner.model.forward = rigged_forward


def _prefill(runner, req, index):
    scheduler_output = _schedule_new_request_from_request(
        req, block_ids=([index],), outer_block_ids=[index]
    )
    runner.execute_model(scheduler_output)
    output = runner.sample_tokens(grammar_output=None)
    req.num_computed_tokens = len(req.prompt_token_ids)
    return output


def _decode(runner, reqs, finished_req_ids=None):
    scheduler_output = _schedule_cached_reqs(
        reqs,
        new_block_ids=[None] * len(reqs),
        finished_req_ids=finished_req_ids,
    )
    runner.execute_model(scheduler_output)
    output = runner.sample_tokens(grammar_output=None)
    for req in reqs:
        req.num_computed_tokens += 1
    return output


def _sampled_by_req(output) -> dict[str, int]:
    return {
        req_id: token_ids[0]
        for req_id, token_ids in zip(
            output.req_ids, output.sampled_token_ids, strict=True
        )
    }


def _get_min_p_proc(runner) -> MinPLogitsProcessor:
    return next(
        p
        for p in runner.input_batch.logitsprocs.all
        if isinstance(p, MinPLogitsProcessor)
    )


@pytest.mark.parametrize(
    "num_seqs, expected_bucket_sizes",
    [
        pytest.param(1, [1], id="1_seq"),
        pytest.param(2, [1, 2], id="2_seq"),
        pytest.param(16, [1, 2, 4, 8, 16], id="16_seq"),
        pytest.param(17, [1, 2, 4, 8, 16, 17], id="17_seq"),
        pytest.param(61, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 61], id="61_seq"),
        # Powers of two to 16, then step 8 to 256, then step 16 to 512, and
        # a non-bucket max_num_seqs is appended as its own final bucket.
        pytest.param(
            512,
            [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16)],
            id="512_seq",
        ),
        pytest.param(
            515,
            [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16), 515],
            id="515_seq",
        ),
    ],
)
def test_get_bucket_sizes(monkeypatch, num_seqs: int, expected_bucket_sizes: list[int]):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    runner = create_model_runner(max_num_seqs=num_seqs)
    fake_load_model(runner)
    bucket_sizes = runner.get_bucket_sizes(num_seqs)
    assert bucket_sizes == expected_bucket_sizes
    assert len(runner.pooled_tensors) == len(expected_bucket_sizes)


def test_min_p_with_padded_bucket(make_runner):
    """min_p's dense tensor must match the padded (bucket_size) logits,
    not num_reqs. With max_num_seqs=4 and 3 live requests the pooled
    logits have 4 rows; a num_reqs-sized min_p ([3, 1]) fails to
    broadcast in max_probabilities.mul_(self.min_p). Regression test for
    refresh_metadata_rbln passing num_reqs to get_and_reset.
    """
    runner = make_runner(4)

    # The top token holds only ~5% probability; min_p=0.5 masks all other
    # tokens, so random sampling becomes deterministic only when min_p is
    # actually applied.
    _rig_forward(runner, {TOP_TOKEN_ID: 8.0})

    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=1.0,
            min_p=0.5,
        )
        for i in range(3)
    ]

    for i, req in enumerate(reqs):
        _prefill(runner, req, i)

    # Decode all together: num_reqs=3 < bucket_size, so padding rows exist.
    output = _decode(runner, reqs)

    _assert_padded_bucket(runner, num_live=3)
    min_p_proc = _get_min_p_proc(runner)
    assert min_p_proc.min_p.shape[0] == runner.bucket_size
    for token_id in _sampled_by_req(output).values():
        assert token_id == TOP_TOKEN_ID


def test_min_tokens_with_padded_bucket(make_runner):
    """min_tokens must keep masking each live row's stop token in the
    padded bucket, then release every row after min_tokens tokens.
    """
    runner = make_runner(4)

    min_tokens = 3
    # Greedy picks the stop token unless min_tokens masks it.
    _rig_forward(
        runner, {TOP_TOKEN_ID: 10.0, RUNNER_UP_TOKEN_ID: 5.0}, base=-10.0
    )

    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=0.0,
            min_tokens=min_tokens,
            stop_token_ids=[TOP_TOKEN_ID],
        )
        for i in range(3)
    ]

    sampled: dict[str, list[int]] = {req.request_id: [] for req in reqs}
    for i, req in enumerate(reqs):
        output = _prefill(runner, req, i)
        sampled[req.request_id].append(output.sampled_token_ids[0][0])

    for _ in range(min_tokens):
        output = _decode(runner, reqs)
        for req_id, token_id in _sampled_by_req(output).items():
            sampled[req_id].append(token_id)

    _assert_padded_bucket(runner, num_live=3)
    for tokens in sampled.values():
        assert tokens[:min_tokens] == [RUNNER_UP_TOKEN_ID] * min_tokens
        assert tokens[min_tokens] == TOP_TOKEN_ID


def test_logit_bias_with_padded_bucket(make_runner):
    """Each row's logit_bias must lift only that row's token in the
    padded bucket - no cross-row leakage, no effect from the padding row.
    """
    runner = make_runner(4)

    biased_token_ids = [20, 21, 22]
    # Every biased token starts below the top token (5 + 20 > 10 only
    # with the bias applied to its own row).
    _rig_forward(
        runner,
        {TOP_TOKEN_ID: 10.0} | {t: 5.0 for t in biased_token_ids},
        base=-10.0,
    )

    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=0.0,
            logit_bias={biased_token_ids[i]: 20.0},
        )
        for i in range(3)
    ]

    for i, req in enumerate(reqs):
        output = _prefill(runner, req, i)
        assert output.sampled_token_ids[0][0] == biased_token_ids[i]

    for _ in range(2):
        output = _decode(runner, reqs)
        _assert_padded_bucket(runner, num_live=3)
        sampled = _sampled_by_req(output)
        for i, req in enumerate(reqs):
            assert sampled[req.request_id] == biased_token_ids[i]


def test_removed_request_leaves_clean_padding_row(make_runner):
    """When a request finishes and the batch shrinks below the bucket,
    the vacated row becomes padding. Its leftover min_p and temperature
    must be reset to no-ops so the padding row cannot corrupt sampling.
    """
    runner = make_runner(4)

    _rig_forward(runner, {TOP_TOKEN_ID: 8.0})

    # temperature=0.7 (not the 1.0 default) so a stale slot is detectable.
    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=0.7,
            min_p=0.5,
        )
        for i in range(4)
    ]

    for i, req in enumerate(reqs):
        _prefill(runner, req, i)

    # Full bucket first: num_reqs=4, no padding row.
    output = _decode(runner, reqs)
    for token_id in _sampled_by_req(output).values():
        assert token_id == TOP_TOKEN_ID

    # Finish the middle request: condense moves req_3 into its slot, the
    # last row becomes padding, and the bucket stays above num_reqs.
    live = [reqs[0], reqs[2], reqs[3]]
    output = _decode(runner, live, finished_req_ids=["req_1"])

    _assert_padded_bucket(runner, num_live=3)
    min_p_proc = _get_min_p_proc(runner)
    assert tuple(min_p_proc.min_p.shape) == (runner.bucket_size, 1)
    assert all(float(min_p_proc.min_p_cpu[i]) == 0.5 for i in range(3))
    for row in range(3, runner.bucket_size):
        assert float(min_p_proc.min_p_cpu[row]) == 0.0
        assert float(runner.input_batch.temperature_cpu_tensor[row]) == 1.0
    sampled = _sampled_by_req(output)
    assert set(sampled) == {"req_0", "req_2", "req_3"}
    for token_id in sampled.values():
        assert token_id == TOP_TOKEN_ID


def test_min_p_tracks_bucket_transitions(make_runner):
    """Growing or shrinking the batch across a bucket boundary must
    resize min_p to the new bucket on the same step - a stale shape
    would fail to broadcast against the pooled logits.
    """
    runner = make_runner(8)

    _rig_forward(runner, {TOP_TOKEN_ID: 8.0})

    def new_req(i):
        return make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=1.0,
            min_p=0.5,
        )

    def assert_decode(reqs, finished_req_ids=None):
        output = _decode(runner, reqs, finished_req_ids=finished_req_ids)
        _assert_padded_bucket(runner, num_live=len(reqs))
        min_p_proc = _get_min_p_proc(runner)
        assert min_p_proc.min_p.shape[0] == runner.bucket_size
        for token_id in _sampled_by_req(output).values():
            assert token_id == TOP_TOKEN_ID
        return runner.bucket_size

    reqs = [new_req(i) for i in range(3)]
    for i, req in enumerate(reqs):
        _prefill(runner, req, i)
    small_bucket = assert_decode(reqs)

    # Grow the batch to 5 requests.
    for i in range(3, 5):
        req = new_req(i)
        _prefill(runner, req, i)
        reqs.append(req)
    large_bucket = assert_decode(reqs)
    # The growth must actually cross a bucket boundary, or this test
    # would not exercise a transition.
    assert large_bucket > small_bucket

    # Shrink back below the boundary.
    reqs = reqs[:3]
    assert assert_decode(reqs, finished_req_ids=["req_3", "req_4"]) == small_bucket


def test_mixed_sampling_params_with_padded_bucket(make_runner):
    """Rows with different sampling params (greedy, min_p, logit_bias)
    must each get only their own params in one padded batch.
    """
    runner = make_runner(4)

    # top ~5% under temperature=1.0 so row 1 is deterministic only via
    # min_p; runner-up + 20 beats top only on the biased row.
    _rig_forward(runner, {TOP_TOKEN_ID: 8.0, RUNNER_UP_TOKEN_ID: 4.0})

    reqs = [
        make_request(
            request_id="req_0",
            prompt_token_ids=[1, 2, 3],
            temperature=0.0,
        ),
        make_request(
            request_id="req_1",
            prompt_token_ids=[1, 2, 3],
            temperature=1.0,
            min_p=0.5,
        ),
        make_request(
            request_id="req_2",
            prompt_token_ids=[1, 2, 3],
            temperature=0.0,
            logit_bias={RUNNER_UP_TOKEN_ID: 20.0},
        ),
    ]
    expected = {
        "req_0": TOP_TOKEN_ID,
        "req_1": TOP_TOKEN_ID,
        "req_2": RUNNER_UP_TOKEN_ID,
    }

    for i, req in enumerate(reqs):
        output = _prefill(runner, req, i)
        assert output.sampled_token_ids[0][0] == expected[req.request_id]

    for _ in range(3):
        output = _decode(runner, reqs)
        _assert_padded_bucket(runner, num_live=3)
        assert _sampled_by_req(output) == expected


def test_logprobs_with_padded_bucket(make_runner):
    """Logprobs must be gathered from the live rows of the padded
    logits: one entry per live request, none for the padding row.
    """
    runner = make_runner(4)

    _rig_forward(
        runner, {TOP_TOKEN_ID: 10.0, RUNNER_UP_TOKEN_ID: 5.0}, base=-10.0
    )

    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            temperature=0.0,
            logprobs=2,
        )
        for i in range(3)
    ]

    for i, req in enumerate(reqs):
        _prefill(runner, req, i)

    output = _decode(runner, reqs)

    _assert_padded_bucket(runner, num_live=3)
    for token_id in _sampled_by_req(output).values():
        assert token_id == TOP_TOKEN_ID
    assert output.logprobs is not None
    assert len(output.logprobs.logprob_token_ids) == 3
    for row in output.logprobs.logprob_token_ids:
        # The sampled (greedy top) token leads each row's gathered ids.
        assert row[0] == TOP_TOKEN_ID
