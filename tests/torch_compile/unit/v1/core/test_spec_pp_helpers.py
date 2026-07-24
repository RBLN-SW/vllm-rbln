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

"""Unit tests for the speculative-decode + pipeline-parallelism scheduler/runner
helpers in ``vllm_rbln.v1.core.utils``:

  * ``should_defer_spec_step`` -- the running-loop anchor-reconciled deferral
    predicate (a verify is scheduled only once its base/anchor token is
    reconciled under sync-scheduling PP).
  * ``num_base_tokens`` / ``resolve_propagated_token_write`` -- the non-last-PP-
    rank multi-accept token propagation bookkeeping (bring the recorded-token
    cursor up to committed_tip by writing the scheduler-propagated payload at
    absolute position).

These are pure functions, so they are exercised directly (no scheduler or
runner instance).
"""

from vllm_rbln.v1.core.utils import (
    num_base_tokens,
    resolve_propagated_token_write,
    should_defer_spec_step,
)


class TestSpecDecodePropagationHelpers:
    """Pure helpers extracted from the scheduler + runner non-last-rank token
    propagation: num_base_tokens and resolve_propagated_token_write."""

    def test_num_base_tokens(self):
        num_sched = {"A": 4, "B": 1}
        drafts = {"A": [1, 2, 3]}  # B carries no drafts this step
        assert num_base_tokens(num_sched, drafts, "A") == 1  # 4 - 3 drafts
        assert num_base_tokens(num_sched, drafts, "B") == 1  # 1 - 0
        assert num_base_tokens(num_sched, drafts, "missing") == 0

    def test_write_normal_decode_writes_newest_token(self):
        # base=1, cursor caught up (== num_computed). Payload is extended
        # backward by num_spec (positions 97..100); write only the newest.
        payload = [10, 11, 12, 13]
        assert resolve_propagated_token_write(
            cursor=100, num_computed_tokens=100, base=1, new_token_ids=payload
        ) == (101, [13])

    def test_write_multi_accept_lag_fills_gap(self):
        # After a verify accepted 3 drafts the cursor lags num_computed by 3;
        # the extended payload lets PP0 fill positions 97..100 by absolute pos.
        payload = [10, 11, 12, 13]
        assert resolve_propagated_token_write(
            cursor=97, num_computed_tokens=100, base=1, new_token_ids=payload
        ) == (101, [10, 11, 12, 13])

    def test_write_nothing_when_cursor_at_tip(self):
        assert (
            resolve_propagated_token_write(
                cursor=101, num_computed_tokens=100, base=1, new_token_ids=[10, 11]
            )
            is None
        )

    def test_write_empty_payload_advances_cursor_only(self):
        # Async GPU-broadcast path: no payload, cursor still advances.
        assert resolve_propagated_token_write(
            cursor=100, num_computed_tokens=100, base=1, new_token_ids=[]
        ) == (101, [])

    def test_write_out_of_window_falls_back_to_tail(self):
        # Defensive: a payload too short to cover [cursor, committed_tip) falls
        # back to its tail so the cursor still advances in-bounds.
        assert resolve_propagated_token_write(
            cursor=97, num_computed_tokens=100, base=1, new_token_ids=[13]
        ) == (101, [13])


class TestShouldDeferSpecStep:
    """Pure predicate for the spec+PP running-loop deferral."""

    def test_disabled_when_spec_off(self):
        # num_spec_tokens == 0: never defers, even for a negative num_new.
        assert should_defer_spec_step(0, [], -3) is False
        assert should_defer_spec_step(0, [], 0) is False

    def test_drafts_held_defers_on_base_le_zero(self):
        # base = num_new - len(drafts). drafts=[1,2,3].
        assert should_defer_spec_step(3, [1, 2, 3], 3) is True  # base 0
        assert should_defer_spec_step(3, [1, 2, 3], 0) is True  # base -3
        assert should_defer_spec_step(3, [1, 2, 3], 4) is False  # base 1

    def test_no_drafts_defers_only_on_negative(self):
        # base == num_new. Only the post-verify overshoot (negative) defers;
        # the mundane == 0 and any positive are left to the caller.
        assert should_defer_spec_step(3, [], -3) is True
        assert should_defer_spec_step(3, [], 0) is False
        assert should_defer_spec_step(3, [], 1) is False
