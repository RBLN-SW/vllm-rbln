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

"""Utility helpers for the RBLN scheduler (``rbln_scheduler.py``).

Small, independently-testable pieces factored out of ``RBLNScheduler`` so the
large upstream-copied ``schedule()`` stays readable; some are shared with
``RBLNModelRunner`` (``decode_batch_size``, the spec-decode ``num_base_tokens`` /
``resolve_propagated_token_write`` token bookkeeping). The per-step decode-batch
admission budget lives here too (``DecodeBatchBudget``): RBLN compiles a fixed
decode-batch shape, so each PP microbatch must stay <=
``max_num_seqs // pipeline_parallel_size``.
"""


def decode_batch_size(max_num_seqs: int, pipeline_parallel_size: int) -> int:
    """Per-PP-stage (compiled) decode batch size — the single source of truth.

    ``max_num_seqs`` is the running-queue cap (as upstream). Under PP the engine
    keeps ``pipeline_parallel_size`` microbatches in flight, so to keep total
    in-flight sequences <= ``max_num_seqs`` the batch RBLN *compiles* per stage
    is ``max_num_seqs // pipeline_parallel_size``. Unlike GPU (dynamic shapes),
    RBLN compiles a fixed shape, so this derived value -- not ``max_num_seqs`` --
    is what the runner buckets to and the scheduler caps at. ``pp_size == 1``
    returns ``max_num_seqs``; callers ensure ``max_num_seqs >= pp`` (validated in
    ``check_and_update_config``).
    """
    return max_num_seqs // pipeline_parallel_size


def should_defer_spec_step(
    num_spec_tokens: int,
    spec_token_ids: list[int],
    num_new_tokens: int,
) -> bool:
    """Whether ``schedule()``'s running loop must defer a spec-decode step under
    sync-scheduling PP because it has no reconciled BASE (anchor) token yet.

    Called with the RAW ``num_new_tokens`` (pre-cap); base is
    ``num_new_tokens - len(spec_token_ids)``. Branch on whether the proposer
    holds drafts:

    - drafts held: ``num_new_tokens`` is draft-inflated, so gate on base itself
      (``base <= 0`` means no fresh anchor to verify against yet).
    - no drafts: ``base == num_new_tokens``, negative only in post-verify
      overshoot; the plain ``== 0`` / no-match cases fall to the caller.

    Returns ``False`` when spec is off (``num_spec_tokens <= 0``); inert for pp=1.
    """
    if num_spec_tokens <= 0:
        return False
    if spec_token_ids:
        return num_new_tokens - len(spec_token_ids) <= 0
    return num_new_tokens < 0


def num_base_tokens(
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list[int]],
    req_id: str,
) -> int:
    """Number of non-draft (anchor / decode) tokens scheduled for ``req_id`` in
    a step: the total scheduled tokens minus the speculative draft tokens.

    This is the count that advances the request's committed sequence (the base
    decode token plus any catch-up), as opposed to the unverified drafts. It is
    the shared notion of "base" used by the scheduler's non-last-rank token
    propagation and by the runner's token-write / output bookkeeping.
    """
    return num_scheduled_tokens.get(req_id, 0) - len(
        scheduled_spec_decode_tokens.get(req_id, ())
    )


def resolve_propagated_token_write(
    cursor: int,
    num_computed_tokens: int,
    base: int,
    new_token_ids: list[int],
) -> tuple[int, list[int]] | None:
    """Plan the non-last PP rank's write of scheduler-propagated tokens.

    Under sync-scheduling PP the scheduler ships ``new_token_ids`` covering the
    committed-token tail (extended backward by ``num_spec_tokens`` to span a
    prior verify's multi-accept lag). This rank brings its cursor
    (``num_tokens_no_spec``) up to ``committed_tip = num_computed_tokens + base``
    by writing the slice landing at ``[cursor, committed_tip)`` by ABSOLUTE
    position, idempotently overwriting mis-speculated slots. Returns
    ``(committed_tip, tokens)`` to write at ``token_ids_cpu[cursor:committed_tip]``,
    or ``None`` if the cursor already reached it.

    Sync-scheduling only (RBLN force-disables async); async would need the
    prev_sampled_token_ids broadcast path, not surface here.
    """
    committed_tip = num_computed_tokens + base
    if committed_tip <= cursor:
        return None
    span = committed_tip - cursor
    # `cursor`'s token sits at offset `lo` in new_token_ids, which covers
    # all_token_ids[committed_tip - len : committed_tip].
    lo = cursor - (committed_tip - len(new_token_ids))
    # Invariant (sync-scheduling PP): the scheduler extends new_token_ids backward
    # by num_spec_tokens, so the payload always covers [cursor, committed_tip) --
    # a non-last rank never lags num_computed by more than one verify's worth. The
    # assert below fails fast if that breaks, instead of an opaque broadcast error
    # deep in _update_states. (lo + span == len(new_token_ids), so no separate
    # upper bound to check.)
    assert lo >= 0, (
        f"propagated payload ({len(new_token_ids)}) shorter than the write span "
        f"({span}): a non-last PP rank's cursor lags num_computed by more than "
        "num_spec_tokens -- the token-propagation invariant is broken."
    )
    return committed_tip, new_token_ids[lo : lo + span]


class DecodeBatchBudget:
    """Per-step decode-admission budget, shared across ``schedule()``'s running
    and waiting loops. One instance per ``schedule()`` call.

    Two caps:
    - hard: the compiled decode-bucket ceiling (``max_num_seqs // pp``); the
      batch may never exceed it or the runner has no bucket -> crash.
    - soft: ``ceil(demand / pp)``, the PP-balanced spreading target that keeps
      each microbatch small enough to fill the pipeline. It equals ``demand``
      (a no-op) when ``pp == 1``, so non-PP decode is unaffected.
    """

    def __init__(self, hard_cap: int, soft_cap: int) -> None:
        assert hard_cap >= 1 and soft_cap >= 1
        self._hard_cap = hard_cap
        self._soft_cap = soft_cap
        self._count = 0

    @classmethod
    def for_step(
        cls, max_num_seqs: int, pipeline_parallel_size: int, demand: int
    ) -> "DecodeBatchBudget":
        """Budget for one step. ``demand`` is running decodes + ready remote-KV
        (invariant under admission -- promoting a ready remote-KV keeps the
        total -- so a step-start snapshot is exact)."""
        hard = decode_batch_size(max_num_seqs, pipeline_parallel_size)
        soft = max(1, -(-demand // pipeline_parallel_size))  # ceil(demand / pp)
        return cls(hard_cap=hard, soft_cap=soft)

    def can_admit(self, *, apply_soft_cap: bool = True) -> bool:
        """True iff one more decode may be admitted this step. The hard cap
        always applies; the soft cap is skipped (``apply_soft_cap=False``) for
        joins outside the demand snapshot (full local prefix match /
        resumed-after-eviction), which then face only the hard cap.
        """
        if self._count >= self._hard_cap:
            return False
        return not (apply_soft_cap and self._count >= self._soft_cap)

    def admit(self, n: int = 1) -> None:
        self._count += n

    def discard(self, n: int = 1) -> None:
        """Un-admit ``n`` decode requests dropped from this step's batch.

        Unlike ``reset()`` (whole batch dropped), removes only requests that left
        while others stay admitted -- the PRIORITY-policy preemption in the
        running loop -- so the next ``can_admit()`` isn't stopped early on a
        stale over-count.
        """
        # A discard without a matching admit would drive the count negative,
        # silently disabling both caps (can_admit only tests >=) -> overschedule.
        assert self._count >= n, (
            f"discard({n}) with count={self._count}: discard without a matching admit"
        )
        self._count -= n

    def reset(self) -> None:
        """Clear the admitted count (decode batch dropped, e.g. by eviction)."""
        self._count = 0

    @property
    def count(self) -> int:
        return self._count
