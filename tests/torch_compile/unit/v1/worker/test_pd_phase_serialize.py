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

"""Tests for VLLM_RBLN_MOE_PD_MIX=0: cross-DP mixed prefill/decode steps are
serialized into two phase-pure forwards via the _pd_phase_vote pre-vote."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_rbln.forward_context import RBLNDPMetadata
from vllm_rbln.v1.worker.rbln_model_runner import (
    PD_PHASE_DECODE,
    PD_PHASE_IDLE,
    PD_PHASE_PREFILL,
    RBLNModelRunner,
)


def _stub(dp_size=2, serialize=True):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=dp_size,
                data_parallel_rank=0,
            )
        ),
        serialize_pd_phases=serialize,
        dummy_run=MagicMock(),
    )


def _patch_votes(per_rank_votes):
    return patch.object(
        RBLNDPMetadata,
        "num_tokens_across_dp",
        return_value=torch.tensor(per_rank_votes, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "votes,expected_mixed",
    [
        ([PD_PHASE_PREFILL, PD_PHASE_DECODE], True),
        ([PD_PHASE_PREFILL, PD_PHASE_DECODE, PD_PHASE_IDLE], True),
        ([PD_PHASE_DECODE, PD_PHASE_DECODE], False),
        ([PD_PHASE_PREFILL, PD_PHASE_PREFILL], False),
        ([PD_PHASE_PREFILL, PD_PHASE_IDLE], False),
        ([PD_PHASE_IDLE, PD_PHASE_IDLE], False),
    ],
)
def test_pd_phase_vote_mixed_detection(votes, expected_mixed):
    stub = _stub(dp_size=len(votes))
    with _patch_votes(votes):
        assert RBLNModelRunner._pd_phase_vote(stub, PD_PHASE_IDLE) is expected_mixed


def test_idle_rank_runs_two_dummies_on_mixed_step():
    stub = _stub()
    stub._pd_phase_vote = MagicMock(return_value=True)
    RBLNModelRunner.execute_dummy_batch(stub)
    assert stub.dummy_run.call_count == 2
    stub._pd_phase_vote.assert_called_once_with(PD_PHASE_IDLE)


def test_idle_rank_runs_one_dummy_on_uniform_step():
    stub = _stub()
    stub._pd_phase_vote = MagicMock(return_value=False)
    RBLNModelRunner.execute_dummy_batch(stub)
    assert stub.dummy_run.call_count == 1


def test_idle_rank_skips_vote_when_mix_allowed():
    stub = _stub(serialize=False)
    stub._pd_phase_vote = MagicMock()
    RBLNModelRunner.execute_dummy_batch(stub)
    stub._pd_phase_vote.assert_not_called()
    assert stub.dummy_run.call_count == 1


def test_prefill_rank_fires_phase2_dummy_after_sampling():
    stub = _stub()
    stub._pd_phase2_pending = True
    stub._sample_tokens_impl = MagicMock(return_value="output")
    result = RBLNModelRunner.sample_tokens(stub, None)
    assert result == "output"
    assert stub._pd_phase2_pending is False
    # The dummy run (phase 2) fires after the real sampling/propose.
    stub.dummy_run.assert_called_once()


def test_no_phase2_dummy_when_not_pending():
    stub = _stub()
    stub._pd_phase2_pending = False
    stub._sample_tokens_impl = MagicMock(return_value="output")
    RBLNModelRunner.sample_tokens(stub, None)
    stub.dummy_run.assert_not_called()
