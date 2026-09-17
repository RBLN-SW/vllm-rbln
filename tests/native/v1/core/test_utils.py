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

# Pure-function helpers in v1/core/utils shared by the RBLN scheduler and runner:
# decode_batch_size, the step phase, and the spec-decode + PP token bookkeeping.
# The decode-batch admission budget classes are exercised in
# test_rbln_scheduler.py, as is is_prefill against real requests.

from types import SimpleNamespace

import pytest

from vllm_rbln.v1.core.utils import (
    decode_batch_size,
    num_base_tokens,
    resolve_propagated_token_write,
    should_defer_spec_step,
    step_is_prefill,
    sub_block_size_in_use,
)


def _sched_out(num_scheduled_tokens, spec=None):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=spec or {},
    )


class TestStepIsPrefill:
    # The phase the runner selects its compiled graph with: more than one
    # non-draft token for some request.

    def test_decode_step(self):
        assert step_is_prefill(_sched_out({"a": 1, "b": 1})) is False

    def test_prefill_chunk(self):
        assert step_is_prefill(_sched_out({"a": 512})) is True

    def test_single_token_step_is_decode(self):
        # Where a 1-token prompt, a chunked prefill's last token and a full
        # remote-KV or prefix-cache match all land -- is_prefill() reports decode
        # for each, so this has to as well.
        assert step_is_prefill(_sched_out({"a": 1})) is False

    def test_drafts_do_not_inflate_the_phase(self):
        # 1 base + 4 drafts is a decode step; only the base counts.
        assert step_is_prefill(_sched_out({"a": 5}, spec={"a": [1, 2, 3, 4]})) is False

    def test_empty_step(self):
        assert step_is_prefill(_sched_out({})) is False

    def test_a_mixed_step_reads_prefill(self):
        # Unreachable (the scheduler never mixes) and asserted against there.
        # Pinned because any() picks the safe side: a prefill graph can still
        # take a single-token query, a decode graph cannot take a chunk.
        assert step_is_prefill(_sched_out({"a": 1, "b": 512})) is True


@pytest.mark.parametrize(
    ("max_num_seqs", "pp_size", "expected"),
    [
        (16, 1, 16),  # non-PP: unchanged
        (16, 2, 8),
        (16, 4, 4),
        (2, 2, 1),  # small batch splits down to 1 per stage
        (4, 2, 2),
    ],
)
def test_decode_batch_size_divides_by_pp(max_num_seqs, pp_size, expected):
    assert decode_batch_size(max_num_seqs, pp_size) == expected


def test_decode_batch_size_pp1_is_identity():
    # pp_size == 1 (non-PP) returns max_num_seqs unchanged.
    for n in (1, 2, 8, 37, 256):
        assert decode_batch_size(n, 1) == n


class TestSpecDecodePropagationHelpers:
    # Pure helpers extracted from the scheduler + runner non-last-rank token
    # propagation: num_base_tokens and resolve_propagated_token_write.

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

    def test_write_out_of_window_asserts_broken_invariant(self):
        # A payload too short to cover [cursor, committed_tip) means the
        # token-propagation invariant is broken; assert instead of returning a
        # short slice. committed_tip=101, span=4, payload len=1 -> lo=-3.
        with pytest.raises(AssertionError, match="invariant is broken"):
            resolve_propagated_token_write(
                cursor=97, num_computed_tokens=100, base=1, new_token_ids=[13]
            )


class TestShouldDeferSpecStep:
    # Pure predicate for the spec+PP running-loop deferral.

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


class TestSubBlockSizeInUse:
    """The one place the sub-block configuration is decided: the scheduler's
    manager choice, the worker's copy-stream reserve, and every rule."""

    @pytest.fixture
    def eligible(self, monkeypatch):
        from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager

        def _set(value):
            monkeypatch.setattr(
                RBLNKVCacheManager,
                "can_use_sub_block_caching",
                staticmethod(lambda cfg, size: value),
            )

        return _set

    @pytest.fixture(autouse=True)
    def _not_cr13(self, monkeypatch):
        # The runner's own card would otherwise decide the decoupled cases.
        from vllm_rbln import platform

        monkeypatch.setattr(
            platform.rebel, "get_npu_name", lambda *a, **kw: "RBLN-CA25"
        )

    def _call(self, **kw):
        args = dict(
            enable_prefix_caching=True,
            sub_block_cache=True,
            block_size=1024,
            max_num_batched_tokens=512,
            kv_cache_config=object(),
        )
        args.update(kw)
        return sub_block_size_in_use(**args)

    def test_defaults_to_the_prefill_chunk(self, eligible):
        eligible(True)
        assert self._call() == 512

    def test_none_without_prefix_caching(self, eligible):
        eligible(True)
        assert self._call(enable_prefix_caching=False) is None

    def test_none_when_the_flag_is_off(self, eligible):
        eligible(True)
        assert self._call(sub_block_cache=False) is None

    def test_none_when_the_config_is_ineligible(self, eligible):
        eligible(False)
        assert self._call() is None

    @pytest.mark.parametrize(
        ("off", "named"),
        [
            ({"sub_block_cache": False}, "enable_sub_block_cache"),
            ({"enable_prefix_caching": False}, "enable_prefix_caching"),
        ],
    )
    def test_a_size_that_something_turns_off_is_rejected(self, eligible, off, named):
        # Without this the size is dropped and the run quietly has no sub-block
        # caching at all.
        eligible(True)
        with pytest.raises(ValueError, match=named):
            self._call(sub_block_size=128, **off)

    def test_a_chunk_outside_the_block_bounds_is_rejected(self, eligible):
        eligible(True)
        with pytest.raises(ValueError, match="block_size >="):
            self._call(block_size=256, max_num_batched_tokens=512, sub_block_size=256)

    def test_a_size_the_kv_cache_cannot_hold_is_rejected(self, eligible):
        # A size that is not a divisor of block_size, or a spec with no token
        # axis to slice: without this the size is dropped and the run quietly
        # has no sub-block caching at all.
        eligible(False)
        with pytest.raises(ValueError, match="not one this KV cache can hold"):
            self._call(sub_block_size=128)

    def test_the_derived_default_bows_out_where_it_does_not_fit(self, eligible):
        # Nobody asked for sub-blocks here, so a chunk wider than the block
        # steps aside rather than refusing to start.
        eligible(True)
        assert self._call(block_size=256, max_num_batched_tokens=512) is None

    def test_a_size_below_the_chunk_needs_cr13(self, eligible, monkeypatch):
        # The multi-block store the decoupled size needs is CR13-only.
        from vllm_rbln import platform

        eligible(True)
        with pytest.raises(ValueError, match="REBEL CR13"):
            self._call(sub_block_size=128)
        monkeypatch.setattr(
            platform.rebel, "get_npu_name", lambda *a, **kw: "RBLN-CR13"
        )
        assert self._call(sub_block_size=128) == 128
