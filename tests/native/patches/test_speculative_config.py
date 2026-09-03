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

"""The two `SpeculativeConfig` patches: the draft's parallel config, and the
drafting reservation DFlash does not need.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_rbln.patches.speculative_config import (
    create_draft_parallel_config,
    max_num_new_slots_for_drafting,
    upstream_max_num_new_slots_for_drafting,
)


@pytest.mark.parametrize("target_pp", [1, 2, 4])
def test_draft_never_inherits_target_pipeline_size(target_pp):
    target = ParallelConfig(pipeline_parallel_size=target_pp, tensor_parallel_size=1)

    draft = create_draft_parallel_config(target, 1)

    assert draft.pipeline_parallel_size == 1
    # world_size is derived in the validator, so it has to come out right at
    # construction; there is no field to correct afterwards.
    assert draft.world_size == 1


def test_the_patch_is_the_one_installed():
    assert (
        SpeculativeConfig.create_draft_parallel_config is create_draft_parallel_config
    )


def test_tensor_parallel_size_still_comes_from_the_argument():
    target = ParallelConfig(pipeline_parallel_size=4, tensor_parallel_size=1)

    draft = create_draft_parallel_config(target, 2)

    assert draft.tensor_parallel_size == 2
    assert draft.world_size == 2


def test_the_draft_head_is_what_fails_supports_pp():
    # Why the inherited pipeline size is fatal rather than merely wasteful: the
    # draft head reaches verify_with_parallel_config and cannot satisfy it. If
    # upstream ever gives the head an intermediate_tensors forward, this flips and
    # the patch has lost its reason.
    assert supports_pp(LlamaForCausalLM)
    assert not supports_pp(Eagle3LlamaForCausalLM)


def _spec_config(method: str, *, num_speculative_tokens: int = 3) -> SpeculativeConfig:
    """A `SpeculativeConfig` carrying only what the reservation reads.

    `__post_init__` resolves a draft checkpoint, which these tests have no use
    for; `parallel_drafting` is what it would set for this method.
    """
    config = SpeculativeConfig.__new__(SpeculativeConfig)
    config.method = method
    config.parallel_drafting = method == "dflash"
    config.num_speculative_tokens = num_speculative_tokens
    return config


def _reserve(method: str, **kwargs) -> int:
    return SpeculativeConfig.max_num_new_slots_for_drafting.fget(
        _spec_config(method, **kwargs)
    )


def test_dflash_reserves_no_slots():
    assert _reserve("dflash") == 0


def test_upstream_still_reserves_for_dflash():
    # Why the patch exists rather than what it does: upstream counts dflash's
    # mask tokens as slots appended to the target's batch. If it ever stops,
    # the patch has lost its reason.
    assert upstream_max_num_new_slots_for_drafting.fget(_spec_config("dflash")) > 0


@pytest.mark.parametrize(
    ("method", "expected"),
    [("draft_model", 1), ("eagle3", 0), ("ngram", 0), ("medusa", 0)],
)
def test_every_other_method_keeps_upstream_reservation(method, expected):
    assert _reserve(method) == expected


def test_the_reservation_patch_is_the_one_installed():
    assert (
        SpeculativeConfig.max_num_new_slots_for_drafting
        is max_num_new_slots_for_drafting
    )


@pytest.mark.parametrize(
    ("method", "reserved"), [("dflash", 0), ("draft_model", 1), ("eagle3", 0)]
)
def test_the_budget_upstream_computes_from_the_reservation(method, reserved):
    # The payoff: with nothing reserved, `max_num_scheduled_tokens` lands on the
    # full budget, so the prefill chunk keeps the KV block boundary.
    budget, seqs = 512, 4
    config = SimpleNamespace(
        speculative_config=_spec_config(method),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=budget,
            max_num_seqs=seqs,
            max_num_scheduled_tokens=None,
        ),
    )

    VllmConfig._set_max_num_scheduled_tokens(config)

    scheduler_config = config.scheduler_config
    assert scheduler_config.max_num_scheduled_tokens == budget - reserved * seqs
