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

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest

from tests.native.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.native.runners import DPRequest
from tests.native.utils import (
    check_outputs_equal,
    devices_needed,
    rbln_device_count,
)

DP_MODELS: list[CompileModelSpec] = [
    CompileModelSpec(
        "openai/gpt-oss-20b",
        {"data_parallel_size": 4},
        {"VLLM_RBLN_SUB_BLOCK_CACHE": "0"},
    ),
]

PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_TOKENS = 32
# Finishes mid-decode of a MAX_TOKENS request, so its rank goes idle early.
SHORT_MAX_TOKENS = 4

# Hang detector, applied per test; the first test of a spec carries that spec's
# engine build, which compiles once per DP rank.
_TIMEOUT_S = int(os.environ.get("VLLM_RBLN_TEST_DP_TIMEOUT_S", 3600))

pytestmark = [
    pytest.mark.model_compile,
    # thread, not the default signal: a rank blocked in a collective is below
    # Python and would not handle SIGALRM until it returns.
    pytest.mark.timeout(_TIMEOUT_S, method="thread"),
]


@dataclass(frozen=True)
class DPLane:
    """One engine and the spec it was built from."""

    spec: CompileModelSpec
    runner: Any

    @property
    def dp_size(self) -> int:
        return self.spec.engine_kwargs["data_parallel_size"]

    def generate_greedy(self, requests: list[DPRequest]) -> list[tuple[list[int], str]]:
        return self.runner.generate_greedy(requests)


@pytest.fixture(scope="module", params=spec_params(DP_MODELS))
def dp_lane(request, async_vllm_runner):
    spec: CompileModelSpec = request.param
    needed = devices_needed(spec.engine_kwargs, spec.rsd)
    available = rbln_device_count()
    if available < needed:
        pytest.skip(f"{spec.test_id} needs {needed} NPUs, host has {available}")

    # Module-scoped and parametrized: pytest groups a spec's tests together, so
    # only one engine holds the devices at a time.
    #
    # RBLN_DEVICES is left as the host set it: both branches of
    # _init_device_env give each DP rank its own slice, and CI runs with it set.
    with pytest.MonkeyPatch.context() as mp:
        apply_spec_envs(spec, mp)
        with async_vllm_runner(spec.model, **spec.engine_kwargs) as runner:
            assert runner.engine.vllm_config.model_config.is_moe, (
                f"{spec.model} is not MoE, so this lane runs independent "
                f"replicas rather than data parallel"
            )
            yield DPLane(spec, runner)


@pytest.fixture(scope="module")
def symmetric_outputs(dp_lane):
    """Every rank busy with the same prompt -- the reference for the asymmetric
    runs, and the only run here with no idle rank."""
    return dp_lane.generate_greedy(
        [DPRequest(PROMPT, MAX_TOKENS, dp_rank=rank) for rank in range(dp_lane.dp_size)]
    )


def test_every_rank_produces_the_same_output(symmetric_outputs) -> None:
    # Ranks differ only in which NPU slice they hold, so a divergence here is a
    # per-rank fault: mis-assigned devices, or DP padding leaking into logits.
    for rank in range(1, len(symmetric_outputs)):
        check_outputs_equal(
            outputs_0_lst=[symmetric_outputs[0]],
            outputs_1_lst=[symmetric_outputs[rank]],
            name_0="rank0",
            name_1=f"rank{rank}",
        )


def test_output_survives_idle_peers(dp_lane, symmetric_outputs) -> None:
    # Every other rank dummy-runs (RBLNModelRunner._dummy_run) for the whole of
    # rank 0's decode -- the shape #894 got wrong.
    outputs = dp_lane.generate_greedy([DPRequest(PROMPT, MAX_TOKENS, dp_rank=0)])

    check_outputs_equal(
        outputs_0_lst=[symmetric_outputs[0]],
        outputs_1_lst=outputs,
        name_0="rank0 with every rank busy",
        name_1=f"rank0 with {dp_lane.dp_size - 1} idle peers",
    )


def test_output_survives_a_peer_finishing_early(dp_lane, symmetric_outputs) -> None:
    # rank0 drops out mid-flight, so the group's shape changes under the survivor.
    last = dp_lane.dp_size - 1
    outputs = dp_lane.generate_greedy(
        [
            DPRequest(PROMPT, SHORT_MAX_TOKENS, dp_rank=0),
            DPRequest(PROMPT, MAX_TOKENS, dp_rank=last),
        ]
    )
    short, long = outputs

    check_outputs_equal(
        outputs_0_lst=[symmetric_outputs[last]],
        outputs_1_lst=[long],
        name_0=f"rank{last} with every rank busy",
        name_1=f"rank{last} outliving rank0",
    )
    # Bounded, not fixed: an earlier EOS still leaves the rank idle early.
    short_ids = short[0]
    assert 0 < len(short_ids) <= SHORT_MAX_TOKENS, (
        f"rank0 asked for at most {SHORT_MAX_TOKENS} tokens, got {len(short_ids)}"
    )
    assert short_ids == symmetric_outputs[0][0][: len(short_ids)], (
        "rank0's tokens changed when it shared the group with a longer request"
    )
