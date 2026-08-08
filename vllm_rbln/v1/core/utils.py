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
admission budget lives here too (``DecodeBatchBudget`` + ``DecodeCapPolicy`` +
``DecodeAdmissionController``): RBLN compiles a fixed decode-batch shape, so each
PP microbatch must stay <= ``max_num_seqs // pipeline_parallel_size``.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class DecodeCapPolicy(Protocol):
    """Supplies the per-step decode-batch cap (max decode requests admitted
    to one PP microbatch).

    Injected into ``DecodeBatchBudget`` so the admission logic stays fixed
    while the *sizing* rule can evolve — e.g. a future dynamic policy that
    splits available decodes across PP stages to avoid microbatch collapse."""

    def cap(self) -> int: ...


class StaticDecodeCapPolicy:
    """Fixed cap = ``max_num_seqs // pipeline_parallel_size``.

    Must equal ``RBLNModelRunner.bucketing_manager.max_batch_size`` so the
    scheduled decode batch always maps onto a compiled decode-batch bucket.
    ``pp_size == 1``
    degenerates to ``max_num_seqs`` (the cap is then a no-op, since the
    running queue is already bounded by ``max_num_seqs``).
    """

    def __init__(self, cap: int) -> None:
        assert cap >= 1, f"decode-batch cap must be >= 1, got {cap}"
        self._cap = cap

    def cap(self) -> int:
        return self._cap


class DynamicDecodeCapPolicy:
    """Cap that spreads decode demand across PP microbatches to avoid PP-depth
    collapse: after a drain the static ``max_num_seqs // pp_size`` cap lets all
    decodes pack into one microbatch, idling the other stages. Sizing each step
    to ``ceil(demand / pp_size)`` keeps the pipeline full.

    ``num_demand_decodes`` must include remote-KV requests ready to join (not
    just running decodes), or the ramp stalls with no headroom. Demand is
    invariant under admission (promoting a ready remote-KV keeps the total), so a
    step-start snapshot is exact even as the running/ready split shifts.

    Clamped to ``static_max_cap`` (the compiled ceiling) and floored at 1;
    ``pp_size == 1`` is a no-op.
    """

    def __init__(
        self,
        static_max_cap: int,
        pipeline_parallel_size: int,
        num_demand_decodes: int,
    ) -> None:
        assert static_max_cap >= 1, f"static_max_cap must be >= 1, got {static_max_cap}"
        # ceil(num_demand_decodes / pipeline_parallel_size)
        spread = -(-num_demand_decodes // pipeline_parallel_size)
        self._cap = max(1, min(static_max_cap, spread))

    def cap(self) -> int:
        return self._cap


class DecodeBatchBudget:
    """Tracks how many decode requests have been admitted to the current
    step's batch, shared across ``schedule()``'s running and waiting loops.

    Lifecycle: one instance per ``schedule()`` call. ``admit()`` on every
    decode added to the step; ``can_admit()`` gates further admissions in
    both loops; ``reset()`` clears the count when a prefill evicts the
    decode batch (the "disable mixed batching" path).
    """

    def __init__(self, cap_policy: DecodeCapPolicy, hard_cap: int) -> None:
        self._cap_policy = cap_policy
        # Compiled decode-bucket ceiling (max_num_seqs // pp). The batch may
        # never exceed it or the runner has no bucket -> crash. The policy's
        # cap() is the (<=) soft/spreading cap (== hard_cap in static mode).
        self._hard_cap = hard_cap
        self._count = 0

    def can_admit(self, *, apply_soft_cap: bool = True) -> bool:
        """True iff one more decode may be admitted this step.

        * hard cap (always): the compiled bucket ceiling; exceeding it crashes
          the runner.
        * soft cap (``apply_soft_cap``): the ``ceil(demand/pp)`` spreading
          target for the budgeted demand. Skip it for joins not in the demand
          snapshot (full local prefix match, resumed-after-eviction). In static
          mode the two caps are equal.
        """
        if self._count >= self._hard_cap:
            return False
        # Soft (spreading) cap for the budgeted demand; skipped for
        # demand-unbudgeted joins, which then face only the hard cap above.
        return not (apply_soft_cap and self._count >= self._cap_policy.cap())

    def admit(self, n: int = 1) -> None:
        """Record ``n`` decode requests as admitted to this step's batch."""
        self._count += n

    def discard(self, n: int = 1) -> None:
        """Un-admit ``n`` decode requests dropped from this step's batch.

        Unlike ``reset()`` (whole batch dropped), removes only requests that left
        while others stay admitted -- the PRIORITY-policy preemption in the
        running loop -- so the next ``can_admit()`` isn't stopped early on a
        stale over-count.
        """
        self._count -= n

    def reset(self) -> None:
        """Clear the admitted count (decode batch dropped, e.g. by eviction)."""
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def cap(self) -> int:
        return self._cap_policy.cap()


class DecodeAdmissionController:
    """Per-scheduler factory for per-step decode-batch admission budgets.

    Built once from config; produces a fresh ``DecodeBatchBudget`` per
    ``schedule()`` call. Owns the PP decode-cap machinery (compiled per-stage
    size, static policy, and the static-vs-PP-balanced-dynamic choice). Dynamic
    capping applies only when ``pp_balance_decode`` and
    ``pipeline_parallel_size > 1``.
    """

    def __init__(
        self,
        max_num_seqs: int,
        pipeline_parallel_size: int,
        pp_balance_decode: bool,
    ) -> None:
        self._pp_size = pipeline_parallel_size
        self._max_decode_batch_size = decode_batch_size(
            max_num_seqs, pipeline_parallel_size
        )
        self._static_policy = StaticDecodeCapPolicy(self._max_decode_batch_size)
        # Demand-spread capping is only meaningful under PP.
        self._balance = pp_balance_decode and pipeline_parallel_size > 1

    def make_budget(self, demand_fn: Callable[[], int]) -> DecodeBatchBudget:
        """Return a fresh budget for this step.

        ``demand_fn`` is a zero-arg callable returning the total decode demand
        (running decodes + ready remote-KV). It is invoked **only** when PP
        balancing is active, so the static path pays nothing to compute it.
        """
        if self._balance:
            policy: DecodeCapPolicy = DynamicDecodeCapPolicy(
                self._max_decode_batch_size, self._pp_size, demand_fn()
            )
        else:
            policy = self._static_policy
        return DecodeBatchBudget(policy, hard_cap=self._max_decode_batch_size)
