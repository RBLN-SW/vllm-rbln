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

"""Unit tests for the query backfill past-token approach.

Two test groups:
  1) Scheduler-side: a boundary-affected req gets `slide_distance` recorded
     in RBLNSchedulerOutput.spec_decode_slide_distance, and
     num_scheduled_tokens / scheduled_spec_decode_tokens are trimmed to the
     effective_remaining advance.
  2) Runner-side: with `slide_distance` and the corresponding
     token_ids_cpu / num_computed_tokens_cpu state, the input building
     math (mirroring _prepare_inputs) produces positions that start at
     T - slide and an input_ids vector whose first `slide` entries are
     the already-decoded past tokens.

The runner-side test reimplements the sliding math directly (rather than
spinning up a full RBLNModelRunner instance) so the test is fast and
isolated. The math mirrors the per-req block at the top of
RBLNModelRunner._prepare_inputs.
"""

import pytest

from tests.torch_compile.unit.v1.core.utils import (
    advance_to_decode,
    create_requests,
    create_scheduler,
)
from vllm_rbln.v1.core.utils import (
    num_base_tokens,
    resolve_propagated_token_write,
    should_defer_spec_step,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 1024
_NUM_SPEC_TOKENS = 3
_MAX_SPEC_DECODE_LEN = _NUM_SPEC_TOKENS + 1  # 4


def _scheduler():
    """A scheduler configured for fixed-length spec decode (num_spec=3)."""
    return create_scheduler(
        block_size=_BLOCK_SIZE,
        num_blocks=100,
        max_num_seqs=10,
        num_speculative_tokens=_NUM_SPEC_TOKENS,
    )


def _request(num_tokens, req_id):
    return create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        block_size=_BLOCK_SIZE,
        max_tokens=2048,
        req_ids=[req_id],
    )[0]


# ---------------------------------------------------------------------------
# Scheduler-side tests: slide_distance, num_scheduled, drafts trimming
# ---------------------------------------------------------------------------


class TestSchedulerBackfill:
    def test_no_boundary_full_spec_no_slide(self):
        """prompt=1024 → remaining_in_block=1024 ≫ max_spec=4 → no slide,
        full spec runs."""
        scheduler = _scheduler()
        req = _request(1024, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [1] * _NUM_SPEC_TOKENS

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert rid not in sched_out.spec_decode_slide_distance
        # 1 base + num_spec_tokens drafts = 4
        assert sched_out.num_scheduled_tokens[rid] == _MAX_SPEC_DECODE_LEN
        assert len(sched_out.scheduled_spec_decode_tokens[rid]) == _NUM_SPEC_TOKENS

    def test_boundary_records_slide_and_trims_advance(self):
        """prompt=1020 → remaining_in_block=4 (full); after one accepted
        step remaining drops below max_spec. Force the boundary by using
        a longer prompt so remaining_in_block is short.

        Use prompt=1022 → remaining_in_block=2 → effective_remaining=2 →
        slide=2, num_scheduled trimmed to 2 (1 base + 1 draft kept).
        """
        scheduler = _scheduler()
        req = _request(1022, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [11, 22, 33]  # 3 drafts proposed

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.spec_decode_slide_distance[rid] == 2
        # effective_remaining = block_size - 1022 % 1024 = 2
        assert sched_out.num_scheduled_tokens[rid] == 2
        # drafts trimmed to (effective_remaining - 1) = 1
        assert sched_out.scheduled_spec_decode_tokens[rid] == [11]

    def test_boundary_remaining_one_drops_all_drafts(self):
        """remaining_in_block=1 → effective_remaining=1 → slide=3,
        num_scheduled=1, no drafts kept."""
        scheduler = _scheduler()
        req = _request(1023, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.spec_decode_slide_distance[rid] == 3
        assert sched_out.num_scheduled_tokens[rid] == 1
        # No drafts survive when only 1 token of advance fits.
        assert rid not in sched_out.scheduled_spec_decode_tokens

    def test_step_no_spec_required_stays_false_under_backfill(self):
        """The legacy collective flag must remain False — backfill handles
        every boundary case per-req."""
        scheduler = _scheduler()
        req = _request(1022, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        assert sched_out.step_no_spec_required is False

    def test_two_reqs_only_boundary_one_slides(self):
        """Two reqs: A has plenty of room (no slide), B is at boundary
        (slides). Verifies per-req independence of the decision."""
        scheduler = _scheduler()
        req_a = _request(512, "A")
        req_b = _request(1022, "B")
        advance_to_decode(scheduler, req_a)
        advance_to_decode(scheduler, req_b)
        req_a.spec_token_ids = [1] * _NUM_SPEC_TOKENS
        req_b.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        assert "A" not in sched_out.spec_decode_slide_distance
        assert sched_out.num_scheduled_tokens["A"] == _MAX_SPEC_DECODE_LEN
        assert len(sched_out.scheduled_spec_decode_tokens["A"]) == _NUM_SPEC_TOKENS

        assert sched_out.spec_decode_slide_distance["B"] == 2
        assert sched_out.num_scheduled_tokens["B"] == 2
        assert sched_out.scheduled_spec_decode_tokens["B"] == [11]


class TestSchedulerBackfillUnderPP:
    """Query backfill (spec decode) composed with pipeline parallelism.

    Backfill shapes the decode QUERY axis (each decode req's window is padded
    to num_spec_tokens + 1); the PP per-step decode cap
    (max_num_seqs // pipeline_parallel_size) bounds the BATCH axis. The two are
    orthogonal, so PP must not change the per-req backfill decision, and the
    cap must not shrink an admitted req's spec query window.
    """

    @pytest.fixture(autouse=True)
    def _static_decode_cap(self, monkeypatch):
        # These tests assert against the STATIC per-step decode cap
        # (max_num_seqs // pp). Disable the dynamic PP-balancing spread (on by
        # default), which would otherwise size the cap to ceil(demand / pp) and
        # change which reqs are admitted -- orthogonal to the backfill decision
        # under test here.
        monkeypatch.setenv("VLLM_RBLN_PP_BALANCE_DECODE_BATCH", "0")

    @pytest.mark.parametrize("pp_size", [1, 2])
    def test_backfill_decision_invariant_under_pp(self, pp_size):
        """The per-req backfill decision (slide / trim / drafts) is identical
        under pp_size=1 and pp_size=2: A (mid-block) stays full-spec with no
        slide; B (at the block boundary) slides 2 and trims to one draft. The
        batch cap (max_num_seqs // pp) is not reached here (2 reqs, cap 5)."""
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,  # pp=2 -> per-step cap 5, well above 2 reqs
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=pp_size,
        )
        req_a = _request(512, "A")  # mid-block: full spec, no slide
        req_b = _request(1022, "B")  # near boundary: slide 2, one draft kept
        advance_to_decode(scheduler, req_a)
        advance_to_decode(scheduler, req_b)
        req_a.spec_token_ids = [1] * _NUM_SPEC_TOKENS
        req_b.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        assert "A" not in sched_out.spec_decode_slide_distance
        assert sched_out.num_scheduled_tokens["A"] == _MAX_SPEC_DECODE_LEN
        assert len(sched_out.scheduled_spec_decode_tokens["A"]) == _NUM_SPEC_TOKENS
        assert sched_out.spec_decode_slide_distance["B"] == 2
        assert sched_out.num_scheduled_tokens["B"] == 2
        assert sched_out.scheduled_spec_decode_tokens["B"] == [11]
        assert sched_out.step_no_spec_required is False

    def test_decode_cap_bounds_batch_but_preserves_spec_window(self):
        """The PP decode cap limits how many spec-decodes join one step, but
        never shrinks an admitted req's spec query window.

        pp=2, max_num_seqs=4 -> per-step cap 2. Three mid-block spec-decodes
        are ready; exactly two are admitted and each keeps its full window
        (1 base + num_spec drafts, no slide), the third defers to a later step.
        """
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=4,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=2,  # per-step decode cap = 4 // 2 = 2
        )
        reqs = [_request(512, f"d{i}") for i in range(3)]  # all mid-block
        for r in reqs:
            advance_to_decode(scheduler, r)
        for r in reqs:
            r.spec_token_ids = [1] * _NUM_SPEC_TOKENS

        sched_out = scheduler.schedule()

        scheduled = [
            r.request_id for r in reqs if r.request_id in sched_out.num_scheduled_tokens
        ]
        assert len(scheduled) == 2, (
            f"per-step decode cap is 2, got {len(scheduled)} scheduled"
        )
        # each admitted spec-decode keeps its full query window, uncapped.
        for rid in scheduled:
            assert sched_out.num_scheduled_tokens[rid] == _MAX_SPEC_DECODE_LEN
            assert len(sched_out.scheduled_spec_decode_tokens[rid]) == _NUM_SPEC_TOKENS
            assert rid not in sched_out.spec_decode_slide_distance
        assert sched_out.step_no_spec_required is False

    @pytest.mark.parametrize("pp_size", [1, 2])
    def test_variable_length_no_spec_fallback_invariant_under_pp(self, pp_size):
        """The variable-length (ngram) cross-block no-spec fallback decision is
        identical under pp_size=1 and pp_size=2: entering a fresh block with a
        draft shortfall crosses a block boundary and elects no-spec (no slide
        recorded); the same shortfall mid-block backfills in-block (full spec,
        slide == num_spec). PP shapes the batch axis, not this per-req/per-step
        query-window decision."""
        # Cross-block: fresh block entry (num_computed == block_size), 0 drafts
        # -> desired slide num_spec > used_in_block 0 -> no-spec.
        sched_cross = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=pp_size,
        )
        assert sched_cross.vllm_config.speculative_config.method == "ngram"
        req_cross = _request(_BLOCK_SIZE, "X")
        advance_to_decode(sched_cross, req_cross)
        req_cross.spec_token_ids = []  # variable-length proposer found no match
        out_cross = sched_cross.schedule()
        assert out_cross.step_no_spec_required is True
        assert "X" not in out_cross.spec_decode_slide_distance

        # In-block: mid-block (used_in_block large), same 0-draft shortfall
        # backfills in-block -> full spec, slide == num_spec, no no-spec.
        sched_in = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=pp_size,
        )
        req_in = _request(1500, "Y")
        advance_to_decode(sched_in, req_in)
        req_in.spec_token_ids = []
        out_in = sched_in.schedule()
        assert out_in.step_no_spec_required is False
        assert out_in.spec_decode_slide_distance["Y"] == _NUM_SPEC_TOKENS

    def test_spec_optimistic_overshoot_defers_not_negative(self):
        """A spec request whose num_computed_tokens has been optimistically
        advanced PAST num_tokens_with_spec must be deferred, never scheduled
        with a negative num_new_tokens.

        Spec decode advances num_computed_tokens at schedule time (by
        1 + num_drafts, assuming acceptance) and only reconciles it in
        update_from_output. Under PP the engine keeps pipeline_parallel_size
        batches in flight, so a running request can be re-scheduled before its
        prior step reconciles -- with few requests filling the pipeline its
        num_computed_tokens overshoots num_tokens_with_spec (simulated here by
        advancing it directly). The base-count deferral gate catches this: the
        request has spec_token_ids and its base (num_tokens - num_computed) is
        <= 0 (anchor in flight), so it is deferred before num_new_tokens is even
        computed -- no negative num_scheduled_tokens / bogus spec slide leaks
        (which would trip `total_num_scheduled_tokens > 0` in the runner).
        """
        scheduler = _scheduler()
        healthy = _request(512, "healthy")  # mid-block: full spec, num_new = 4
        overshot = _request(512, "overshot")
        advance_to_decode(scheduler, healthy)
        advance_to_decode(scheduler, overshot)
        healthy.spec_token_ids = [1, 2, 3]
        overshot.spec_token_ids = [1, 2, 3]
        # Simulate the PP optimistic overshoot: push num_computed_tokens past
        # num_tokens_with_spec so num_new_tokens = num_tokens_with_spec - it < 0.
        overshot.num_computed_tokens = overshot.num_tokens_with_spec + 2

        sched_out = scheduler.schedule()

        # The overshot request is deferred (not scheduled this step).
        assert "overshot" not in sched_out.num_scheduled_tokens
        assert "overshot" not in sched_out.spec_decode_slide_distance
        # The healthy request is unaffected -- the skip is per-request.
        assert sched_out.num_scheduled_tokens["healthy"] == _MAX_SPEC_DECODE_LEN
        # No negative/zero counts leak out; the runner's `> 0` assert holds.
        assert sched_out.total_num_scheduled_tokens > 0
        assert all(v >= 1 for v in sched_out.num_scheduled_tokens.values())

    def test_spec_overshoot_empty_drafts_defers_not_negative(self):
        """Post-verify overshoot with the drafts already CLEARED: num_computed
        is optimistically advanced past num_tokens while spec_token_ids is
        empty, so num_new_tokens goes negative. The base-count deferral gate is
        guarded by `request.spec_token_ids`, so it does NOT fire here; the
        `num_new_tokens <= 0` guard must catch it. Otherwise a negative
        num_scheduled_tokens (plus a bogus spec slide) leaks and trips the
        runner's `total_num_scheduled_tokens > 0` assert. Discriminates the
        `<= 0` guard from a plain `== 0` (regression guard)."""
        scheduler = _scheduler()
        req = _request(512, "X")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = []  # drafts consumed by the prior verify step
        # Optimistic overshoot: num_computed runs past num_tokens -> num_new < 0.
        req.num_computed_tokens = req.num_tokens + _NUM_SPEC_TOKENS

        sched_out = scheduler.schedule()

        assert "X" not in sched_out.num_scheduled_tokens  # deferred
        assert "X" not in sched_out.spec_decode_slide_distance
        assert sched_out.total_num_scheduled_tokens >= 0  # no negative leak

    def test_spec_pp_defers_verify_when_anchor_in_flight(self):
        """A verify whose anchor (base) token is still in flight -- num_computed
        == num_tokens, so base == 0 -- must be deferred, even though the drafts
        keep num_new_tokens > 0. This is the exact empty-new_token_ids trigger
        on the non-last PP rank (the IndexError). The `== 0` guard alone does
        NOT catch it (num_new == num_spec here); the base-count deferral gate
        does. Discriminates the gate from the plain num_new guard."""
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=2,
        )
        req = _request(512, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [1, 2, 3]
        # Anchor in flight: base == 0. With drafts, num_new would be num_spec.
        req.num_computed_tokens = req.num_tokens
        assert req.num_tokens_with_spec - req.num_computed_tokens == _NUM_SPEC_TOKENS

        sched_out = scheduler.schedule()

        # Deferred by the base-count gate (a plain `== 0` guard would not fire).
        assert "A" not in sched_out.num_scheduled_tokens

    def test_spec_pp_extends_new_token_ids_for_multi_accept(self):
        """Under spec + PP the scheduler extends the non-last-rank new_token_ids
        payload backward by num_spec so PP0 can catch up its recorded-token
        cursor after a prior verify's multi-token accept (bug#2). The payload
        spans all_token_ids[num_computed - num_spec : num_computed + base]
        instead of the base-only [num_computed : num_computed + base]."""
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            pipeline_parallel_size=2,
        )
        req = _request(512, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [1, 2, 3]
        nc_pre = req.num_computed_tokens  # payload is sliced pre-advance

        sched_out = scheduler.schedule()

        base = sched_out.num_scheduled_tokens["A"] - len(
            sched_out.scheduled_spec_decode_tokens.get("A", [])
        )
        cached = sched_out.scheduled_cached_reqs
        idx = list(cached.req_ids).index("A")
        new_token_ids = list(cached.new_token_ids[idx])
        # Extended backward by num_spec (not the base-only length).
        assert len(new_token_ids) == base + _NUM_SPEC_TOKENS
        assert new_token_ids == list(
            req.all_token_ids[nc_pre - _NUM_SPEC_TOKENS : nc_pre + base]
        )


# ---------------------------------------------------------------------------
# Cross-block no-spec fallback: variable-length proposers can hit a step whose
# backfill window would cross a KV block boundary and must drop to no-spec;
# fixed-length proposers never reach that branch (guarded by an assert).
# ---------------------------------------------------------------------------


def _set_method(scheduler, method):
    """Override the proposer method on the scheduler's SpeculativeConfig.

    Only the cross-block guard reads ``method``; the scheduling math itself is
    method-agnostic. Setting it lets us exercise fixed-length behavior without
    standing up a real draft model. ``object.__setattr__`` bypasses pydantic
    frozen-field protection.
    """
    object.__setattr__(scheduler.vllm_config.speculative_config, "method", method)


class TestCrossBlockNoSpecFallback:
    # ---- variable-length: the problem situation occurs AND no-spec fixes it --

    def test_variable_length_cross_block_elects_no_spec(self):
        """ngram (variable-length), just entered a new block
        (num_computed=1024 -> used_in_block=0), proposer returned 0 drafts ->
        desired_slide = 4 - 1 = 3 > 0 -> backfill would cross into the previous
        block. The scheduler must elect no-spec and NOT record a (cross-block)
        slide for that req."""
        scheduler = _scheduler()  # default method == "ngram" (variable-length)
        assert scheduler.vllm_config.speculative_config.method == "ngram"
        req = _request(1024, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = []  # variable-length proposer found no match

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is True
        # cross-block must NOT record an in-block slide; it drops to no-spec
        assert rid not in sched_out.spec_decode_slide_distance

    def test_variable_length_in_block_shortfall_keeps_full_spec(self):
        """Same 0-draft shortfall but mid-block (num_computed=1500 ->
        used_in_block=476): desired_slide=3 <= 476 stays in-block -> slide is
        recorded and no-spec is NOT elected. Confirms the fallback only fires
        on a genuine block crossing, not on every draft shortfall."""
        scheduler = _scheduler()
        req = _request(1500, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = []

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is False
        assert sched_out.spec_decode_slide_distance[rid] == _NUM_SPEC_TOKENS

    @pytest.mark.parametrize(
        "used_in_block,num_draft,expect_no_spec",
        [
            # At a fresh block entry (used_in_block=0) any shortfall crosses:
            (0, 0, True),  # slide 3 > 0
            (0, 1, True),  # slide 2 > 0
            (0, 2, True),  # slide 1 > 0
            (0, 3, False),  # slide 0 -> full spec, no slide at all
            # 2 tokens into the block, only the largest shortfall crosses:
            (2, 0, True),  # slide 3 > 2
            (2, 1, False),  # slide 2 <= 2 -> in-block backfill
            # 3 tokens in: even a zero-draft shortfall fits in-block:
            (3, 0, False),  # slide 3 <= 3 -> in-block backfill
        ],
    )
    def test_variable_length_cross_block_threshold(
        self, used_in_block, num_draft, expect_no_spec
    ):
        """num_draft (= len(spec_token_ids)) and the block-entry offset jointly
        decide the fallback: cross-block (no-spec) iff
        (num_spec - num_draft) > used_in_block. Sweeps the exact threshold."""
        scheduler = _scheduler()  # ngram (variable-length)
        req = _request(_BLOCK_SIZE + used_in_block, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [0] * num_draft

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is expect_no_spec
        if expect_no_spec:
            # cross-block elects no-spec and records no slide
            assert rid not in sched_out.spec_decode_slide_distance
        else:
            # in-block: slide recorded iff there was a shortfall to pad
            desired_slide = _NUM_SPEC_TOKENS - num_draft
            if desired_slide > 0:
                assert sched_out.spec_decode_slide_distance[rid] == desired_slide

    # ---- fixed-length: no-spec never arises ---------------------------------

    def test_fixed_length_block_entry_no_no_spec(self):
        """Fixed-length (eagle) always supplies num_spec drafts. At a fresh
        block entry desired_slide = 4 - 4 = 0 -> no slide, no no-spec."""
        scheduler = _scheduler()
        _set_method(scheduler, "eagle")
        req = _request(1024, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [1, 2, 3]  # fixed-length always supplies num_spec

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is False
        assert rid not in sched_out.spec_decode_slide_distance

    def test_fixed_length_boundary_squeeze_stays_in_block(self):
        """Fixed-length at the block-END squeeze (num_computed=1022 ->
        remaining=2) backfills in-block (slide=2 <= used_in_block=1022) and
        never elects no-spec."""
        scheduler = _scheduler()
        _set_method(scheduler, "medusa")
        req = _request(1022, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is False
        assert sched_out.spec_decode_slide_distance[rid] == 2

    # ---- invariant guard: fixed-length must never reach cross-block ---------

    @pytest.mark.parametrize(
        "m,expect_assert",
        [
            # m = max_model_len % block_size = the final partial block's size.
            # The scheduler caps each decode step to
            #   num_new_tokens <= max_model_len - 1 - num_computed   (rbln_scheduler
            # reserves the last position; "input position must not exceed
            # max_model_len" under spec decode). So at the final-block entry
            # (used_in_block=0, remaining_in_maxlen=m) even with FULL drafts:
            #   old_n  = min(num_spec+1, m-1)
            #   new_n  = min(old_n, m) = old_n
            #   desired_slide = num_spec+1 - old_n  > 0  iff  m-1 < num_spec+1
            # => cross-block iff m <= num_spec+1, safe iff m >= num_spec+2.
            # The shortfall is from the maxlen reservation, NOT a draft shortage.
            (2, True),  # rem_max=2 -> old_n=1 -> slide 3 -> cross-block
            (3, True),  # rem_max=3 -> old_n=2 -> slide 2 -> cross-block
            (4, True),  # rem_max=4 -> old_n=3 -> slide 1 (m == num_spec+1)
            (5, False),  # rem_max=5 -> old_n=4 -> slide 0 (m == num_spec+2, safe)
            (6, False),  # safe
        ],
    )
    def test_fixed_length_maxlen_edge_threshold(self, m, expect_assert):
        """The ONLY real way a fixed-length proposer reaches cross-block: a
        misaligned max_model_len whose final block is too short to hold a full
        num_spec+1 window. The shortfall comes from the scheduler's maxlen
        reservation (max_model_len-1), NOT a draft shortage (eagle always
        supplies num_spec). Sweeps m: m <= num_spec+1 must fail loudly (assert),
        m >= num_spec+2 stays full-spec. (m==0 aligned and m==1 degenerate are
        safe and covered by the other fixed-length tests.)

        self.max_model_len is read from model_config, not the create_scheduler
        arg (which only sets scheduler_config), so it is patched directly.
        """
        scheduler = _scheduler()
        scheduler.max_model_len = _BLOCK_SIZE + m  # final block holds m tokens
        _set_method(scheduler, "eagle")
        req = _request(_BLOCK_SIZE, "A")  # decode starts at the block boundary
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [1, 2, 3]  # full drafts (realistic fixed-length)

        if expect_assert:
            with pytest.raises(AssertionError, match="fixed-length"):
                scheduler.schedule()
        else:
            sched_out = scheduler.schedule()
            assert sched_out.step_no_spec_required is False


# ---------------------------------------------------------------------------
# Runner-side test: query window math + input_ids inclusion of past tokens
# ---------------------------------------------------------------------------


class TestRunnerBackfillMath:
    """Mirror the per-req block at the top of
    RBLNModelRunner._prepare_inputs to verify that past tokens land
    at the start of input_ids and positions shift backward by slide."""

    def _run_sliding_math(
        self,
        *,
        num_reqs,
        num_scheduled_per_req,
        slide_per_req,
        num_computed_per_req,
        token_ids_cpu,
        max_model_len,
    ):
        """Reimplement the backfill chunk of _prepare_inputs as pure numpy
        so we can assert on positions / input_ids without spinning up the
        full runner."""
        import numpy as np

        num_scheduled = np.array(num_scheduled_per_req, dtype=np.int32)
        slide_arr = np.array(slide_per_req, dtype=np.int32)
        num_computed_cpu = np.array(num_computed_per_req, dtype=np.int32)

        query_lengths = num_scheduled + slide_arr
        total_query_tokens = int(query_lengths.sum())

        req_indices = np.repeat(np.arange(num_reqs), query_lengths)
        # Per-req local arange [0..query_lengths[i]-1] concatenated.
        arange = np.concatenate([np.arange(ql, dtype=np.int32) for ql in query_lengths])

        positions = num_computed_cpu[req_indices] - slide_arr[req_indices] + arange
        token_indices = positions + req_indices * max_model_len
        input_ids = token_ids_cpu.flatten()[token_indices]
        return positions, input_ids, total_query_tokens

    def test_single_req_no_slide_identity(self):
        """Without slide the math reproduces the standard flow exactly."""
        import numpy as np

        max_model_len = 2048
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        token_ids[0, 100] = 100
        token_ids[0, 101] = 101
        token_ids[0, 102] = 102
        token_ids[0, 103] = 103

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=1,
            num_scheduled_per_req=[4],  # full window, no boundary
            slide_per_req=[0],
            num_computed_per_req=[100],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        assert positions.tolist() == [100, 101, 102, 103]
        assert input_ids.tolist() == [100, 101, 102, 103]
        assert total == 4

    def test_single_req_boundary_slide_2_prepends_past(self):
        """req at T=1022, remaining_in_block=2, num_spec_tokens=3:
        scheduler sets num_scheduled=2 (1 base + 1 draft) and slide=2.
        Runner builds a 4-position window [1020..1023]; the first two
        slots carry past tokens, the last two carry base+draft."""
        import numpy as np

        max_model_len = 2048
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        token_ids[0, 1020] = 700  # past
        token_ids[0, 1021] = 701  # past
        token_ids[0, 1022] = 702  # base (last sampled prev step)
        token_ids[0, 1023] = 703  # draft

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=1,
            num_scheduled_per_req=[2],
            slide_per_req=[2],
            num_computed_per_req=[1022],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        # Window slid back by 2 so all 4 positions are within current block.
        assert positions.tolist() == [1020, 1021, 1022, 1023]
        # First two are past tokens; last two are base + draft.
        assert input_ids.tolist() == [700, 701, 702, 703]
        assert total == _MAX_SPEC_DECODE_LEN  # 4

    def test_single_req_boundary_slide_3_only_base_in_window(self):
        """remaining_in_block=1 case: slide=3, num_scheduled=1.
        Window = [T-3..T] = 3 past + 1 base (no drafts)."""
        import numpy as np

        max_model_len = 2048
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        token_ids[0, 1020] = 800
        token_ids[0, 1021] = 801
        token_ids[0, 1022] = 802
        token_ids[0, 1023] = 803  # base

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=1,
            num_scheduled_per_req=[1],
            slide_per_req=[3],
            num_computed_per_req=[1023],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        assert positions.tolist() == [1020, 1021, 1022, 1023]
        assert input_ids.tolist() == [800, 801, 802, 803]
        assert total == _MAX_SPEC_DECODE_LEN

    def test_zero_drafts_non_boundary_slide_full_locally(self):
        """Local rank with zero drafts off the boundary: previously this
        rank would have voted query_len=1 (no-spec) and relied on cross-DP
        MAX to lift to num_spec_tokens+1. Now the scheduler fills the
        deficit locally via slide=num_spec_tokens, so the runner builds
        a full num_spec_tokens+1 window without any cross-DP help."""
        import numpy as np

        max_model_len = 2048
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        # Past tokens at positions [97..99] then base at 100.
        token_ids[0, 97] = 970
        token_ids[0, 98] = 980
        token_ids[0, 99] = 990
        token_ids[0, 100] = 1000  # base

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=1,
            num_scheduled_per_req=[1],
            slide_per_req=[_NUM_SPEC_TOKENS],
            num_computed_per_req=[100],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        # Full num_spec_tokens+1 window built purely locally.
        assert positions.tolist() == [97, 98, 99, 100]
        assert input_ids.tolist() == [970, 980, 990, 1000]
        assert total == _MAX_SPEC_DECODE_LEN

    def test_partial_drafts_non_boundary_slide_pads_remainder(self):
        """Local rank with 1 draft (out of num_spec_tokens=3) off the
        boundary: scheduler slides by num_spec - kept = 2 so the window
        becomes 2 past + 1 base + 1 draft = num_spec_tokens+1."""
        import numpy as np

        max_model_len = 2048
        token_ids = np.zeros((1, max_model_len), dtype=np.int32)
        # Past at [98..99], base at 100, draft at 101.
        token_ids[0, 98] = 980
        token_ids[0, 99] = 990
        token_ids[0, 100] = 1000  # base
        token_ids[0, 101] = 1010  # draft

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=1,
            num_scheduled_per_req=[2],  # 1 base + 1 draft
            slide_per_req=[2],
            num_computed_per_req=[100],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        assert positions.tolist() == [98, 99, 100, 101]
        assert input_ids.tolist() == [980, 990, 1000, 1010]
        assert total == _MAX_SPEC_DECODE_LEN

    def test_mixed_batch_per_req_independence(self):
        """Req A: no slide (full window starts at T). Req B: slide=2.
        Both contribute their own 4-position window into the flat layout
        without interfering — input_ids concatenates per-req."""
        import numpy as np

        max_model_len = 2048
        # Req A starts at position 100, no boundary issue.
        # Req B starts at position 1022, boundary case (slide=2).
        token_ids = np.zeros((2, max_model_len), dtype=np.int32)
        token_ids[0, 100] = 1000
        token_ids[0, 101] = 1001
        token_ids[0, 102] = 1002
        token_ids[0, 103] = 1003
        token_ids[1, 1020] = 2020
        token_ids[1, 1021] = 2021
        token_ids[1, 1022] = 2022
        token_ids[1, 1023] = 2023

        positions, input_ids, total = self._run_sliding_math(
            num_reqs=2,
            # A: full 4 positions scheduled, no slide.
            # B: only 2 logical positions advance, 2 past pulled in.
            num_scheduled_per_req=[4, 2],
            slide_per_req=[0, 2],
            num_computed_per_req=[100, 1022],
            token_ids_cpu=token_ids,
            max_model_len=max_model_len,
        )

        # 4 (A) + 4 (B window incl. past) = 8 total query tokens.
        assert total == 8
        assert positions.tolist() == [
            100,
            101,
            102,
            103,  # A window
            1020,
            1021,
            1022,
            1023,  # B window, slid back by 2
        ]
        assert input_ids.tolist() == [
            1000,
            1001,
            1002,
            1003,
            2020,
            2021,
            2022,
            2023,
        ]


# ---------------------------------------------------------------------------
# Sampler logits indices: past positions excluded automatically
# ---------------------------------------------------------------------------


class TestBackfillLogitsIndices:
    """Verify that the existing _calc_spec_decode_metadata math, when fed
    query-aware cu_num_tokens (= cumsum of query_lengths, backfill-aware),
    yields logits_indices that point only at the NEW positions of each
    req's window — past positions are excluded automatically. No code
    change was required for this; the test pins down the invariant so
    future refactors don't accidentally break it.

    The formula being exercised (mirroring _calc_spec_decode_metadata):
        cu_prev_end      = cu_num_scheduled_tokens - num_sampled_tokens
        logits_indices   = repeat(cu_prev_end, num_sampled_tokens) + arange
    where
        num_sampled_tokens      = num_draft_tokens + 1 = effective_remaining
        cu_num_scheduled_tokens = cumsum of query_lengths (incl. slide)
    """

    def _calc_logits_indices(self, query_lengths, num_draft_tokens):
        """Reimplement the per-step block of _calc_spec_decode_metadata as
        pure numpy. Returns (logits_indices, target_logits_indices,
        bonus_logits_indices)."""
        import numpy as np

        query_lengths_np = np.asarray(query_lengths, dtype=np.int32)
        num_draft = np.asarray(num_draft_tokens, dtype=np.int32)
        cu_num_scheduled_tokens = np.cumsum(query_lengths_np)
        num_sampled_tokens = num_draft + 1

        cu_prev_end = cu_num_scheduled_tokens - num_sampled_tokens
        arange = np.concatenate(
            [np.arange(s, dtype=np.int32) for s in num_sampled_tokens]
        )
        logits_indices = np.repeat(cu_prev_end, num_sampled_tokens) + arange

        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)
        if int(num_draft.sum()) > 0:
            target_arange = np.concatenate(
                [np.arange(d, dtype=np.int32) for d in num_draft]
            )
            target_logits_indices = (
                np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft)
                + target_arange
            )
        else:
            target_logits_indices = np.zeros(0, dtype=np.int32)
        bonus_logits_indices = cu_num_sampled_tokens - 1

        return logits_indices, target_logits_indices, bonus_logits_indices

    def test_no_slide_full_spec_indices(self):
        """Baseline: no boundary, drafts=3. logits_indices covers all 4
        sampled positions (base + 3 drafts)."""
        logits_indices, target, bonus = self._calc_logits_indices(
            query_lengths=[4], num_draft_tokens=[3]
        )
        assert logits_indices.tolist() == [0, 1, 2, 3]
        assert target.tolist() == [0, 1, 2]
        assert bonus.tolist() == [3]

    def test_slide_2_drafts_1_skips_past_positions(self):
        """Boundary slide=2, drafts=1. Flat layout per req =
        [past, past, base, draft]. logits_indices must skip pasts
        (flat 0, 1) and only point at base (2) and draft (3)."""
        logits_indices, target, bonus = self._calc_logits_indices(
            query_lengths=[4], num_draft_tokens=[1]
        )
        assert logits_indices.tolist() == [2, 3]
        assert target.tolist() == [0]
        assert bonus.tolist() == [1]

    def test_slide_3_no_drafts_only_base_logit(self):
        """Extreme boundary (remaining=1): slide=3, drafts=0.
        Flat layout = [past, past, past, base]. Only the base logit is
        sampled; no drafts to validate."""
        logits_indices, target, bonus = self._calc_logits_indices(
            query_lengths=[4], num_draft_tokens=[0]
        )
        assert logits_indices.tolist() == [3]
        assert target.tolist() == []
        assert bonus.tolist() == [0]

    def test_mixed_batch_logits_indices_skip_per_req(self):
        """Two reqs:
          A: query_length=4, drafts=3 (no slide) -> all 4 positions sampled.
          B: query_length=4, drafts=1 (slide=2)  -> only base+draft sampled.
        Flat layout: [A0, A1, A2, A3,  B_past, B_past, B_base, B_draft]
        Expected logits_indices = [0, 1, 2, 3,  6, 7] (B's pasts 4, 5
        excluded)."""
        logits_indices, target, bonus = self._calc_logits_indices(
            query_lengths=[4, 4], num_draft_tokens=[3, 1]
        )
        assert logits_indices.tolist() == [0, 1, 2, 3, 6, 7]
        assert target.tolist() == [0, 1, 2, 4]
        assert bonus.tolist() == [3, 5]


# ---------------------------------------------------------------------------
# Rejection sampler input integrity: draft_token_ids extraction
# ---------------------------------------------------------------------------


class TestBackfillDraftTokenExtraction:
    """Verify that the draft token ids the rejection sampler validates
    against are the post-trim drafts (not the pre-slide originals).

    In _calc_spec_decode_metadata the draft token tensor is extracted as:
        draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]
    Under backfill, input_ids has past tokens prepended at flat positions
    [0..slide-1], the base at [slide], and the (effective_remaining-1)
    surviving drafts at [slide+1..query_length-1]. The extraction must
    pull exactly those surviving drafts — no past, no original-but-dropped
    drafts.
    """

    def _extract_draft_tokens(self, input_ids_flat, query_lengths, num_draft_tokens):
        """Mirror _calc_spec_decode_metadata's draft_token_ids extraction."""
        import numpy as np

        query_lengths_np = np.asarray(query_lengths, dtype=np.int32)
        num_draft = np.asarray(num_draft_tokens, dtype=np.int32)
        cu_num_scheduled_tokens = np.cumsum(query_lengths_np)
        num_sampled_tokens = num_draft + 1
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)

        cu_prev_end = cu_num_scheduled_tokens - num_sampled_tokens
        sampled_arange = np.concatenate(
            [np.arange(s, dtype=np.int32) for s in num_sampled_tokens]
        )
        logits_indices = np.repeat(cu_prev_end, num_sampled_tokens) + sampled_arange

        if int(num_draft.sum()) > 0:
            draft_arange = np.concatenate(
                [np.arange(d, dtype=np.int32) for d in num_draft]
            )
            target_logits_indices = (
                np.repeat(cu_num_sampled_tokens - num_sampled_tokens, num_draft)
                + draft_arange
            )
        else:
            target_logits_indices = np.zeros(0, dtype=np.int32)

        # Equivalent of: input_ids[logits_indices][target_logits_indices + 1]
        sampled_input_ids = input_ids_flat[logits_indices]
        if target_logits_indices.size > 0:
            draft_token_ids = sampled_input_ids[target_logits_indices + 1]
        else:
            draft_token_ids = sampled_input_ids[np.zeros(0, dtype=np.int32)]
        return draft_token_ids

    def test_no_slide_extracts_all_drafts(self):
        """Baseline: full spec, no slide. Drafts extracted are exactly the
        proposed drafts D1, D2, D3."""
        import numpy as np

        # Flat input_ids for 1 req with no slide: [base, D1, D2, D3]
        input_ids = np.array([500, 11, 22, 33], dtype=np.int32)
        draft_tokens = self._extract_draft_tokens(
            input_ids, query_lengths=[4], num_draft_tokens=[3]
        )
        assert draft_tokens.tolist() == [11, 22, 33]

    def test_slide_2_extracts_only_surviving_draft(self):
        """Backfill scenario: slide=2 prepends 2 past tokens; only 1 draft
        survives the scheduler's trim. The extraction must skip the past
        slots AND skip the original-but-dropped drafts (D2, D3) — only
        the kept draft D1 should be returned."""
        import numpy as np

        # Flat input_ids for boundary req: [past0, past1, base, D1]
        # Note D2, D3 do NOT appear in input_ids at all — scheduler
        # already trimmed scheduled_spec_decode_tokens to [D1] before
        # runner builds input_ids.
        input_ids = np.array([700, 701, 500, 11], dtype=np.int32)
        draft_tokens = self._extract_draft_tokens(
            input_ids, query_lengths=[4], num_draft_tokens=[1]
        )
        assert draft_tokens.tolist() == [11]

    def test_slide_3_no_drafts_empty_extraction(self):
        """Extreme boundary: slide=3, drafts=0. No drafts to extract."""
        import numpy as np

        input_ids = np.array([700, 701, 702, 500], dtype=np.int32)
        draft_tokens = self._extract_draft_tokens(
            input_ids, query_lengths=[4], num_draft_tokens=[0]
        )
        assert draft_tokens.tolist() == []

    def test_mixed_batch_extracts_per_req_drafts(self):
        """Mixed batch: A has 3 drafts (no slide), B has 1 draft
        (slide=2). Extraction returns A's 3 drafts followed by B's 1
        surviving draft — nothing from B's past or B's dropped drafts."""
        import numpy as np

        # A: [a_base, A_D1, A_D2, A_D3], B: [b_past0, b_past1, b_base, B_D1]
        input_ids = np.array(
            [
                100,
                11,
                22,
                33,  # req A
                700,
                701,
                200,
                99,  # req B
            ],
            dtype=np.int32,
        )
        draft_tokens = self._extract_draft_tokens(
            input_ids, query_lengths=[4, 4], num_draft_tokens=[3, 1]
        )
        # A's 3 drafts (D1, D2, D3) + B's 1 surviving draft (D1).
        assert draft_tokens.tolist() == [11, 22, 33, 99]


# ---------------------------------------------------------------------------
# Edge cases: spec disabled, prefill-only, boundary not triggered
# ---------------------------------------------------------------------------


class TestBackfillEdgeCases:
    """Verify the query backfill logic stays a no-op when spec decode is
    disabled (num_spec_tokens=0), when the running req is in prefill
    phase, or when the boundary simply isn't reached. These are the
    guards the per-req backfill block depends on; a regression that
    removes one of them would silently change behavior for non-spec or
    prefill workloads.
    """

    def test_no_spec_configured_no_slide_entry(self):
        """num_spec_tokens=0 disables spec entirely; scheduler must not
        record any slide_distance for any req."""
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=None,  # spec decode OFF
        )
        req = _request(1022, "A")  # would be a boundary case if spec were on
        advance_to_decode(scheduler, req)

        sched_out = scheduler.schedule()

        # No slide map entries when spec decode is disabled.
        assert sched_out.spec_decode_slide_distance == {}
        # Standard single-token decode advance.
        assert sched_out.num_scheduled_tokens[req.request_id] == 1
        assert req.request_id not in sched_out.scheduled_spec_decode_tokens

    def test_prefill_req_no_slide_entry(self):
        """The backfill decision is gated on `not is_prefill(request)`,
        so a req still in prefill must never appear in
        spec_decode_slide_distance even if num_computed % block_size is
        near the boundary."""
        scheduler = _scheduler()
        # Long prompt so we'd hit boundary IF this were a decode req.
        req = _request(1022, "A")
        # Do NOT advance_to_decode — leave the req in prefill phase.
        scheduler.add_request(req)

        sched_out = scheduler.schedule()

        # Prefill reqs are excluded from the backfill block (is_prefill
        # guard), so the dict stays empty regardless of position-in-block.
        assert sched_out.spec_decode_slide_distance == {}

    def test_no_boundary_no_slide_entry_far_from_block_end(self):
        """A decode req whose full num_spec_tokens+1 window fits inside
        the current block AND whose proposer returned the full set of
        drafts must not get a slide entry (sanity for the
        condition `desired_slide > 0`)."""
        scheduler = _scheduler()
        # remaining_in_block from this position = block_size - 100 = 924,
        # comfortably larger than max_spec_decode_len (4).
        req = _request(100, "A")
        advance_to_decode(scheduler, req)
        req.spec_token_ids = [11, 22, 33]

        sched_out = scheduler.schedule()

        assert req.request_id not in sched_out.spec_decode_slide_distance
        assert sched_out.num_scheduled_tokens[req.request_id] == _MAX_SPEC_DECODE_LEN
        assert (
            len(sched_out.scheduled_spec_decode_tokens[req.request_id])
            == _NUM_SPEC_TOKENS
        )


# ---------------------------------------------------------------------------
# Variable-length proposer support: backfill pads query window when the
# proposer (ngram, suffix decoding, etc.) returns fewer than
# num_spec_tokens drafts, even off the block boundary. This is the
# unified always-full-spec design — every decode step's query length is
# num_spec_tokens + 1 at runtime.
# ---------------------------------------------------------------------------


class TestBackfillVariableLengthPadding:
    """Verify backfill fires off the boundary when the proposer returns
    fewer than num_spec_tokens drafts. The shortage is padded with past
    positions so the runtime query window stays at num_spec_tokens + 1.
    """

    def test_zero_drafts_far_from_boundary_slides_full(self):
        """ngram miss (0 drafts proposed) far from any boundary should
        still record slide_distance = num_spec_tokens so the runtime
        query window is padded to num_spec_tokens + 1."""
        scheduler = _scheduler()
        # num_computed=100, remaining_in_block = 924 (no boundary), but
        # the proposer returns 0 drafts → padding needed.
        req = _request(100, "A")
        advance_to_decode(scheduler, req)
        # No spec_token_ids set → proposer miss equivalent.

        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.spec_decode_slide_distance[rid] == _NUM_SPEC_TOKENS
        # Actual advance is still 1 (only the base, no drafts kept).
        assert sched_out.num_scheduled_tokens[rid] == 1
        assert rid not in sched_out.scheduled_spec_decode_tokens

    def test_partial_drafts_far_from_boundary_slides_to_full(self):
        """ngram partial hit (k < num_spec drafts) far from boundary
        should record slide_distance = num_spec_tokens - k so the
        runtime window is padded to num_spec_tokens + 1."""
        scheduler = _scheduler()
        req = _request(100, "A")
        advance_to_decode(scheduler, req)
        # Proposer returns 1 draft (< num_spec_tokens=3).
        req.spec_token_ids = [42]

        sched_out = scheduler.schedule()

        rid = req.request_id
        # slide = num_spec_tokens (3) - kept_drafts (1) = 2.
        assert sched_out.spec_decode_slide_distance[rid] == 2
        # Actual advance: 1 base + 1 draft = 2.
        assert sched_out.num_scheduled_tokens[rid] == 2
        # The single proposed draft is retained (boundary didn't squeeze).
        assert sched_out.scheduled_spec_decode_tokens[rid] == [42]

    def test_partial_drafts_at_boundary_combines_pad_and_trim(self):
        """When BOTH variable-length padding AND boundary squeeze apply,
        slide_distance covers the combined deficit and drafts get
        trimmed to fit `effective_remaining - 1`."""
        scheduler = _scheduler()
        # num_computed=1022 → remaining_in_block=2 → effective_remaining=2.
        req = _request(1022, "A")
        advance_to_decode(scheduler, req)
        # Proposer returns 2 drafts.
        req.spec_token_ids = [11, 22]

        sched_out = scheduler.schedule()

        rid = req.request_id
        # new_n = min(old_n=3, effective_remaining=2) = 2
        # slide = max_spec_decode_len(4) - new_n(2) = 2
        assert sched_out.spec_decode_slide_distance[rid] == 2
        assert sched_out.num_scheduled_tokens[rid] == 2
        # Drafts trimmed to (new_n - 1) = 1 kept.
        assert sched_out.scheduled_spec_decode_tokens[rid] == [11]


# ---------------------------------------------------------------------------
# Issue 1: a prompt shorter than num_spec_tokens has too few committed
# positions to backfill a full speculative window (desired_slide >
# available_past). That is a SUBSET of the can't-backfill condition, so the
# scheduler must elect no-spec instead of crashing on the old hard assert.
# ---------------------------------------------------------------------------


class TestShortPromptNoSpecFallback:
    def test_short_prompt_elects_no_spec_not_assert(self):
        """prompt (2) < num_spec_tokens (3). After one decode num_computed=2,
        but a full window needs to slide back desired_slide=3 positions
        (available_past=2 < 3). The old code asserted ("prompt shorter than
        num_spec_tokens") and crashed the engine. Since available_past >=
        tokens_used_in_block always holds, this case is just a cross-block
        backfill that can't reach full spec -> it must elect no-spec.

        BEFORE FIX (bug reproduced): with the hard
        ``assert desired_slide <= available_past`` still present in
        rbln_scheduler.py (Issue 1), ``scheduler.schedule()`` below raises
        ``AssertionError: ... prompt is shorter than num_spec_tokens`` and the
        engine crashes -> this test FAILS.
        AFTER FIX: that assert is removed; control falls through to the
        cross-block branch which elects no-spec -> this test PASSES.
        """
        scheduler = _scheduler()  # ngram (variable-length)
        assert scheduler.vllm_config.speculative_config.method == "ngram"
        req = _request(2, "A")  # prompt < num_spec_tokens
        advance_to_decode(scheduler, req)
        assert req.num_computed_tokens == 2
        req.spec_token_ids = []  # no drafts -> desired_slide = num_spec = 3

        # Must NOT raise AssertionError ("prompt is shorter than ...").
        sched_out = scheduler.schedule()

        rid = req.request_id
        assert sched_out.step_no_spec_required is True
        # cross-block / short-prompt must NOT record an in-block slide
        assert rid not in sched_out.spec_decode_slide_distance

    # NOTE: a fixed-length proposer cannot reach the short-prompt
    # can't-backfill condition: it always supplies num_spec drafts, so
    # desired_slide = (num_spec+1) - (1 + num_spec) = 0 and the sliding block is
    # skipped entirely. The only way fixed-length reaches the cross-block guard
    # (and must fail loudly) is the misaligned max_model_len edge, covered by
    # TestCrossBlockNoSpecFallback.test_fixed_length_maxlen_edge_threshold.


# ---------------------------------------------------------------------------
# Issue 2: on a cross-block no-spec step the scheduler optimistically advances
# num_computed_tokens by (1 bonus + num_drafts) in _update_after_schedule. The
# runner's no-spec scrub must NOT clear scheduled_spec_decode_tokens, because
# the engine's update_from_output reads it to roll the draft advance back
# (num_rejected = num_draft_tokens - num_accepted). Clearing it strands the
# advance and over-counts num_computed_tokens by the dropped draft count on
# EVERY cross-block no-spec step.
# ---------------------------------------------------------------------------

from tests.torch_compile.unit.v1.core.utils import create_runner_output  # noqa: E402
from vllm_rbln.v1.worker.rbln_model_runner import (  # noqa: E402
    scrub_scheduler_output_for_no_spec,
)


class TestCrossBlockNoSpecRollback:
    def test_cross_block_no_spec_rolls_back_num_computed(self):
        """Reproduces the reported Issue 2 (observed, not hypothetical).

        BEFORE FIX (bug reproduced): with
        ``scheduler_output.scheduled_spec_decode_tokens.clear()`` still present
        in ``scrub_scheduler_output_for_no_spec``, the cleared dict makes the
        engine's ``update_from_output`` skip the draft rollback, so the final
        assert sees ``num_computed_tokens == 1026`` (1024 + 1 bonus + 1
        never-verified draft) and this test FAILS (``1026 == 1025``).
        AFTER FIX: the clear is removed (the dict survives for the rollback) and
        ``_prepare_inputs`` is told to take the no-spec path via
        ``spec_decode_max_query_len==1`` instead, so the draft is rolled back to
        ``num_computed_tokens == 1025`` and this test PASSES.
        """
        scheduler = _scheduler()  # ngram (variable-length)
        req = _request(_BLOCK_SIZE, "A")  # decode starts at a fresh block entry
        advance_to_decode(scheduler, req)
        assert req.num_computed_tokens == _BLOCK_SIZE  # 1024

        # 1 draft at a fresh block entry: old_n = 1 bonus + 1 draft = 2,
        # desired_slide = (num_spec+1) - 2 = 2 > used_in_block(0) -> cross-block
        # -> no-spec, but a draft WAS scheduled (so there is something to roll
        # back).
        req.spec_token_ids = [777]

        sched_out = scheduler.schedule()
        rid = req.request_id
        assert sched_out.step_no_spec_required is True
        # _update_after_schedule advanced num_computed optimistically by old_n=2
        # (1 bonus + 1 draft); not yet rolled back.
        assert req.num_computed_tokens == _BLOCK_SIZE + 2  # 1026
        # the scheduler must keep the draft recorded for the rollback below
        assert sched_out.scheduled_spec_decode_tokens[rid] == [777]

        # The runner forces query_len=1 for the collective no-spec step.
        scrub_scheduler_output_for_no_spec(sched_out)
        assert sched_out.num_scheduled_tokens[rid] == 1

        # The model emits a single (no-spec) token; the engine must roll the
        # never-verified draft back out of num_computed_tokens.
        scheduler.update_from_output(sched_out, create_runner_output(sched_out, 1))

        # Correct: 1024 prompt + 1 real decoded token = 1025. The dropped draft
        # is rolled back. BUG (scheduled_spec_decode_tokens cleared in the
        # scrub): rollback is skipped and it stays at 1026.
        assert req.num_computed_tokens == _BLOCK_SIZE + 1  # 1025


# ---------------------------------------------------------------------------
# Issue 3: no-spec is a DECODE concern (cross-block backfill on a decoding
# request). In DP/EP each rank runs prefill-only OR decode-only. When a decode
# rank elects no-spec, the cross-DP OR-reduce forces EVERY rank to the no-spec
# path -- including a PREFILL rank. The scrub then wrongly clamps that rank's
# prefill query_len (its chunk) to 1, so the prefill's sampled token is later
# discarded (seq_lens < num_tokens) and the request is lost. no-spec must apply
# to decode requests only; a prefill's query_len must be left untouched.
# ---------------------------------------------------------------------------


class TestNoSpecScrubPrefill:
    def test_prefill_query_len_intact_without_scrub(self):
        """Baseline: a prefill req is scheduled with its full chunk (>1)."""
        scheduler = _scheduler()
        req = _request(512, "P")
        scheduler.add_request(req)  # prefill phase (NOT advanced to decode)
        sched_out = scheduler.schedule()
        rid = req.request_id
        assert sched_out.num_scheduled_tokens[rid] > 1

    def test_no_spec_scrub_must_not_clamp_prefill(self):
        """BUG repro: the no-spec scrub (fired by a peer decode rank's
        cross-block fallback via the cross-DP OR-reduce) must NOT clamp a
        PREFILL rank's query_len to 1.

        BEFORE FIX: scrub sets num_scheduled_tokens[prefill]=1 -> this FAILS.
        AFTER FIX: prefill query_len preserved -> PASSES.
        """
        scheduler = _scheduler()
        req = _request(512, "P")
        scheduler.add_request(req)  # prefill-only rank
        sched_out = scheduler.schedule()
        rid = req.request_id
        chunk = sched_out.num_scheduled_tokens[rid]
        assert chunk > 1, "precondition: prefill scheduled with a real chunk"

        # A peer decode rank elected no-spec -> this prefill rank is scrubbed.
        # scrub excludes prefill reqs (via is_prefill) from the clamp.
        scrub_scheduler_output_for_no_spec(sched_out)

        assert sched_out.num_scheduled_tokens[rid] == chunk, (
            f"prefill query_len wrongly clamped to "
            f"{sched_out.num_scheduled_tokens[rid]} (was {chunk}) by no-spec scrub"
        )

    def test_no_spec_scrub_must_not_clamp_intermediate_chunked_prefill(self):
        """Intermediate chunked prefill: a CACHED req still processing its
        prompt (num_output == 0, num_computed < num_prompt) must ALSO be
        excluded from the no-spec scrub -- not just the first (new-req) chunk.
        """
        scheduler = create_scheduler(
            block_size=_BLOCK_SIZE,
            num_blocks=100,
            max_num_seqs=10,
            num_speculative_tokens=_NUM_SPEC_TOKENS,
            long_prefill_token_threshold=256,  # force multi-chunk prefill
        )
        req = _request(1024, "C")  # 1024-token prompt -> 256-token chunks
        scheduler.add_request(req)
        rid = req.request_id

        # chunk 1: NEW req prefill (no output committed -> still prefilling).
        sched1 = scheduler.schedule()
        assert sched1.num_scheduled_tokens[rid] == 256
        scheduler.update_from_output(sched1, create_runner_output(sched1, None))

        # chunk 2: now a CACHED req, still prefilling (num_output == 0).
        sched2 = scheduler.schedule()
        assert req.num_computed_tokens < req.num_prompt_tokens  # still prefilling
        assert not sched2.scheduled_new_reqs  # exercises the cached-req path
        chunk = sched2.num_scheduled_tokens[rid]
        assert chunk > 1

        # A peer decode rank elected no-spec -> this chunked-prefill rank is
        # scrubbed. The intermediate chunk must NOT be clamped to 1.
        scrub_scheduler_output_for_no_spec(sched2)
        assert sched2.num_scheduled_tokens[rid] == chunk, (
            f"intermediate chunked-prefill query_len wrongly clamped to "
            f"{sched2.num_scheduled_tokens[rid]} (was {chunk}) by no-spec scrub"
        )

    def test_no_spec_scrub_still_clamps_decode(self):
        """Guard the fix's scope: a DECODE req (num_output > 0) must still be
        clamped to query_len=1 by the no-spec scrub."""
        scheduler = _scheduler()
        req = _request(_BLOCK_SIZE, "D")
        advance_to_decode(scheduler, req)  # now decode
        req.spec_token_ids = []  # cross-block no-spec election
        sched_out = scheduler.schedule()
        rid = req.request_id
        assert sched_out.step_no_spec_required is True
        # decode req (num_output > 0) is NOT a prefill -> gets clamped.
        scrub_scheduler_output_for_no_spec(sched_out)
        assert sched_out.num_scheduled_tokens[rid] == 1


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
