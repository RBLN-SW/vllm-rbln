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

"""The three output comparisons, in increasing tolerance: equal, almost equal,
logprobs close. Fixtures are the real numbers measured off the eagle e2e
divergence, so a threshold change shows up as a verdict change here."""

from __future__ import annotations

import math

import pytest

from tests.native.utils import (
    NearTieWarning,
    check_logprobs_close,
    check_outputs_almost_equal,
    check_outputs_equal,
)

# Llama-3.1-8B truncated to 3 layers, generated index 3: the position where the
# eagle e2e run's reference and spec paths split. Measured in fp32 on CPU.
LEADING = 113526  # the spec run's pick
RUNNER_UP = 26494  # the reference run's pick
LEADING_LOGPROB = -4.3295
RUNNER_UP_LOGPROB = -4.3538
FILLER = {124512: -4.8294, 26966: -5.0267, 6339: -5.0666}


def step(pick: int, other: int, other_logprob: float) -> dict[int, float]:
    """One step's top-k as the runner reports it: {token id: logprob}."""
    own_logprob = LEADING_LOGPROB if pick == LEADING else RUNNER_UP_LOGPROB
    return {pick: own_logprob, other: other_logprob, **FILLER}


def outputs(pick: int, other: int, other_logprob: float):
    return [([pick], "", [step(pick, other, other_logprob)])]


@pytest.fixture
def eagle_divergence():
    """Both runs, each rating the other's pick as measured."""
    return (
        outputs(LEADING, RUNNER_UP, RUNNER_UP_LOGPROB),
        outputs(RUNNER_UP, LEADING, LEADING_LOGPROB),
    )


def test_the_measured_gap_is_a_few_percent_of_probability():
    """0.0243 nats is the whole question: it is what the two comparisons are
    thresholding, expressed as a probability ratio."""
    gap = LEADING_LOGPROB - RUNNER_UP_LOGPROB
    assert gap == pytest.approx(0.0243, abs=1e-4)
    assert math.exp(gap) - 1 == pytest.approx(0.0246, abs=1e-3)


def test_almost_equal_accepts_what_the_npu_runs_actually_reported():
    """The e2e numbers, not the fp32 ones: at that position the reference rated
    the pair a literal tie and the spec run rated it 0.0625 -- two bf16 steps
    apart, around a true gap of 0.0243. The threshold has to clear that."""
    tie = -4.3295
    lhs = [([RUNNER_UP], "", [{RUNNER_UP: tie, LEADING: tie}])]
    rhs = [([LEADING], "", [{LEADING: tie, RUNNER_UP: tie - 0.0625}])]

    with pytest.warns(NearTieWarning):
        check_outputs_almost_equal(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )


def test_equal_rejects_it(eagle_divergence):
    lhs, rhs = eagle_divergence
    with pytest.raises(AssertionError):
        check_outputs_equal(
            outputs_0_lst=[(ids, text) for ids, text, _ in lhs],
            outputs_1_lst=[(ids, text) for ids, text, _ in rhs],
            name_0="reference",
            name_1="spec",
        )


def test_almost_equal_accepts_it(eagle_divergence):
    lhs, rhs = eagle_divergence
    with pytest.warns(NearTieWarning):
        check_outputs_almost_equal(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )


def test_almost_equal_rejects_a_pick_outside_the_leading_pair():
    """Three candidates inside the probability tolerance, so only the ranking
    can reject: each run's pick puts the other's third. This is the criterion
    check_logprobs_close does not have."""
    middle = 999_999
    lhs = [([LEADING], "", [{LEADING: -4.3295, middle: -4.3300, RUNNER_UP: -4.3305}])]
    rhs = [([RUNNER_UP], "", [{RUNNER_UP: -4.3295, middle: -4.3300, LEADING: -4.3305}])]

    with pytest.warns(NearTieWarning):
        check_logprobs_close(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )
    with pytest.raises(AssertionError, match="outside the leading 2"):
        check_outputs_almost_equal(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )


@pytest.mark.parametrize("gap", [0.15, 0.3, 0.49])
def test_almost_equal_rejects_gaps_check_logprobs_close_allows(gap):
    """The band between the two thresholds: tolerated as a near-tie, rejected as
    almost equal."""
    lhs = outputs(LEADING, RUNNER_UP, LEADING_LOGPROB - gap)
    rhs = [
        (
            [RUNNER_UP],
            "",
            [{RUNNER_UP: LEADING_LOGPROB - gap, LEADING: LEADING_LOGPROB, **FILLER}],
        )
    ]

    with pytest.warns(NearTieWarning):
        check_logprobs_close(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )
    with pytest.raises(AssertionError, match="not a near-tie"):
        check_outputs_almost_equal(
            outputs_0_lst=lhs, outputs_1_lst=rhs, name_0="reference", name_1="spec"
        )
