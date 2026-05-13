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

"""Unit tests for the sliding-window past-token approach.

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

from tests.torch_compile.unit.v1.core.utils import (
    advance_to_decode,
    create_requests,
    create_scheduler,
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


class TestSchedulerSliding:
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

    def test_step_no_spec_required_stays_false_under_sliding(self):
        """The legacy collective flag must remain False — sliding handles
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


# ---------------------------------------------------------------------------
# Runner-side test: query window math + input_ids inclusion of past tokens
# ---------------------------------------------------------------------------


class TestRunnerSlidingMath:
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
        """Reimplement the sliding chunk of _prepare_inputs as pure numpy
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


class TestSlidingLogitsIndices:
    """Verify that the existing _calc_spec_decode_metadata math, when fed
    query-aware cu_num_tokens (= cumsum of query_lengths, sliding-aware),
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


class TestSlidingDraftTokenExtraction:
    """Verify that the draft token ids the rejection sampler validates
    against are the post-trim drafts (not the pre-slide originals).

    In _calc_spec_decode_metadata the draft token tensor is extracted as:
        draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]
    Under sliding, input_ids has past tokens prepended at flat positions
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
        """Sliding scenario: slide=2 prepends 2 past tokens; only 1 draft
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


class TestSlidingEdgeCases:
    """Verify the sliding-window logic stays a no-op when spec decode is
    disabled (num_spec_tokens=0), when the running req is in prefill
    phase, or when the boundary simply isn't reached. These are the
    guards the per-req sliding block depends on; a regression that
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
        """The sliding decision is gated on `not is_prefill(request)`,
        so a req still in prefill must never appear in
        spec_decode_slide_distance even if num_computed % block_size is
        near the boundary."""
        scheduler = _scheduler()
        # Long prompt so we'd hit boundary IF this were a decode req.
        req = _request(1022, "A")
        # Do NOT advance_to_decode — leave the req in prefill phase.
        scheduler.add_request(req)

        sched_out = scheduler.schedule()

        # Prefill reqs are excluded from the sliding block (is_prefill
        # guard), so the dict stays empty regardless of position-in-block.
        assert sched_out.spec_decode_slide_distance == {}

    def test_no_boundary_no_slide_entry_far_from_block_end(self):
        """A decode req whose full num_spec_tokens+1 window fits inside
        the current block must not get a slide entry (sanity for the
        condition `effective_remaining < max_spec_decode_len`)."""
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
