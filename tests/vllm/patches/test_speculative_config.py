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

"""The two `SpeculativeConfig` patches: the draft's expert parallelism, and the
drafting reservation DFlash does not need.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig

from vllm_rbln.patches.speculative_config import (
    _orig_create_draft_parallel_config,
    create_draft_parallel_config,
    max_num_new_slots_for_drafting,
    upstream_max_num_new_slots_for_drafting,
)


def test_the_draft_never_inherits_expert_parallelism():
    # A MoE target with an EAGLE-family head: upstream forwards the flag and
    # `verify_with_parallel_config` then refuses the head for having no experts.
    target = ParallelConfig(tensor_parallel_size=1, enable_expert_parallel=True)

    draft = create_draft_parallel_config(target, 1)

    assert not draft.enable_expert_parallel


def test_everything_else_still_comes_from_upstream():
    # Why this delegates instead of building a ParallelConfig of its own: a
    # hand-rolled copy silently drops whatever field upstream adds next.
    target = ParallelConfig(
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_parallel_loading_workers=3,
    )

    ours = create_draft_parallel_config(target, 2)
    theirs = _orig_create_draft_parallel_config(target, 2)
    theirs.enable_expert_parallel = False

    assert ours == theirs


def test_the_patch_is_the_one_installed():
    assert (
        SpeculativeConfig.create_draft_parallel_config is create_draft_parallel_config
    )


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


def _budget_check(method: str, budget: int) -> None:
    """Run the reservation through upstream's config-time budget validation."""
    VllmConfig._set_max_num_scheduled_tokens(
        SimpleNamespace(
            speculative_config=_spec_config(method),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=budget,
                max_num_seqs=4,
                max_num_scheduled_tokens=None,
            ),
        )
    )


def test_a_zeroed_reservation_keeps_a_tight_budget_legal():
    # The payoff: upstream refuses a budget it cannot fit the drafting slots
    # into, and dflash reserves `num_speculative_tokens` of them. Zeroing the
    # reservation is what keeps a budget that tight usable on RBLN.
    tight = _spec_config("dflash").num_speculative_tokens - 1

    _budget_check("dflash", tight)

    with pytest.raises(ValueError, match="enough slots"):
        VllmConfig._set_max_num_scheduled_tokens(
            SimpleNamespace(
                speculative_config=SimpleNamespace(
                    max_num_new_slots_for_drafting=tight
                ),
                scheduler_config=SimpleNamespace(
                    max_num_batched_tokens=tight,
                    max_num_seqs=4,
                    max_num_scheduled_tokens=None,
                ),
            )
        )
