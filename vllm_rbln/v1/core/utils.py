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

A home for small, self-contained, independently-testable pieces of RBLN
scheduling/engine logic factored out of ``RBLNScheduler`` so that
``schedule()`` (a large copy of upstream's method with RBLN-specific tweaks)
stays readable. Unlike ``vllm_rbln.utils`` (general torch-tensor primitives),
these encode engine/scheduler semantics; some are shared with the runner
(``RBLNModelRunner``) -- e.g. ``decode_batch_size`` and the spec-decode
``num_base_tokens`` / ``resolve_propagated_token_write`` token bookkeeping.
Add further scheduler utilities here.

The speculative-decode + PP helpers -- ``should_defer_spec_step`` (running-loop
anchor-reconciled deferral), ``num_base_tokens`` and
``resolve_propagated_token_write`` (non-last-rank token propagation) -- live
next to their docstrings above/below.

The per-step decode-batch admission budget (``DecodeBatchBudget`` +
``DecodeCapPolicy``):

Under pipeline parallelism the engine keeps ``pipeline_parallel_size``
microbatches in flight (one per PP stage). To bound the total in-flight
decode requests to ``max_num_seqs`` and to never exceed the runner's
compiled decode-batch bucket, each single step's decode batch must be
capped at ``max_num_seqs // pipeline_parallel_size`` (==
``RBLNModelRunner.max_batch_size``). ``schedule()`` admits decode requests
from two places — the running loop and the waiting-loop remote-KV promotion
(P/D-disaggregated decode) — so the cap is centralized into one budget both
loops share, rather than enforced per-loop.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable


def decode_batch_size(max_num_seqs: int, pipeline_parallel_size: int) -> int:
    """Per-PP-stage (compiled) decode batch size — the single source of truth.

    ``max_num_seqs`` is the max number of concurrently running sequences (the
    scheduler's running-queue cap), consistent with upstream vLLM and with the
    non-PP case; the KV cache is provisioned to hold that many concurrent
    sequences. Under pipeline parallelism the engine keeps
    ``pipeline_parallel_size`` microbatches in flight, so to keep total
    in-flight sequences <= ``max_num_seqs`` the batch RBLN must *compile* for
    one PP stage is that cap split across stages:

        decode_batch_size = max_num_seqs // pipeline_parallel_size

    Unlike GPU (dynamic batch shapes, no division), RBLN compiles a fixed
    decode-batch shape, so this derived value — not ``max_num_seqs`` itself —
    is what the runner buckets to and the scheduler caps at. ``pp_size == 1``
    returns ``max_num_seqs`` unchanged (non-PP). Callers must ensure
    ``max_num_seqs >= pipeline_parallel_size`` (validated at config time in
    ``RBLNPlatform.check_and_update_config``); otherwise this floors to 0.
    """
    return max_num_seqs // pipeline_parallel_size


def should_defer_spec_step(
    num_spec_tokens: int,
    spec_token_ids: list[int],
    num_new_tokens: int,
) -> bool:
    """Whether ``schedule()``'s running loop must defer a speculative-decode
    step under sync-scheduling PP because it has no reconciled BASE (anchor)
    token to schedule yet.

    Called with the RAW ``num_new_tokens`` (before the scheduler's caps). The
    base count is ``num_new_tokens - len(spec_token_ids)``. The branch is on
    whether the proposer currently holds drafts (``spec_token_ids``), which only
    selects how the base is tested:

    - Drafts held: ``num_new_tokens`` is draft-inflated, so gate on the base
      itself. ``base <= 0`` means there is no fresh anchor token for the drafts
      to verify against yet (the token-producing step has not reconciled);
      scheduling anyway hands the non-last PP rank an empty token payload.
    - No drafts held (just consumed, or no proposer match): ``base ==
      num_new_tokens``, which goes negative only in the post-verify overshoot
      (num_computed_tokens still optimistically advanced). The mundane ``== 0``
      (no work this step) and the no-match case are left to the caller's plain
      upstream defer.

    Returns ``False`` when spec is disabled (``num_spec_tokens <= 0``), so
    non-spec decode is untouched; also inert for pp=1, where synchronous
    scheduling keeps the base reconciled.
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

    Under sync-scheduling pipeline parallelism the scheduler ships
    ``new_token_ids`` covering the request's committed-token tail (extended
    backward by ``num_spec_tokens`` so it always spans a prior verify's
    multi-accept lag). This rank brings its recorded-token cursor
    (``num_tokens_no_spec``) up to
    ``committed_tip = num_computed_tokens + base`` by writing the payload slice
    that lands at ``[cursor, committed_tip)`` by ABSOLUTE position, idempotently
    overwriting any mis-speculated slots.

    Returns ``(committed_tip, tokens)`` where ``tokens`` is the slice to write
    at ``token_ids_cpu[cursor:committed_tip]`` -- empty when the payload is
    empty (e.g. the async-scheduling GPU-broadcast path), in which case only the
    cursor advances -- or ``None`` if the cursor has already reached
    ``committed_tip`` (nothing to write).
    """
    committed_tip = num_computed_tokens + base
    if committed_tip <= cursor:
        return None
    span = committed_tip - cursor
    if not new_token_ids:
        return committed_tip, []
    # The payload covers all_token_ids[committed_tip - len : committed_tip], so
    # the token for absolute position `cursor` sits at offset `lo`.
    lo = cursor - (committed_tip - len(new_token_ids))
    if lo >= 0 and lo + span <= len(new_token_ids):
        return committed_tip, new_token_ids[lo : lo + span]
    # Defensive: the payload should always cover [cursor, committed_tip); if it
    # somehow does not, fall back to its tail so the cursor still advances
    # without an out-of-bounds slice.
    return committed_tip, new_token_ids[-span:]


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

    Must equal ``RBLNModelRunner.max_batch_size`` so the scheduled decode
    batch always maps onto a compiled decode-batch bucket. ``pp_size == 1``
    degenerates to ``max_num_seqs`` (the cap is then a no-op, since the
    running queue is already bounded by ``max_num_seqs``).
    """

    def __init__(self, cap: int) -> None:
        assert cap >= 1, f"decode-batch cap must be >= 1, got {cap}"
        self._cap = cap

    def cap(self) -> int:
        return self._cap


class DynamicDecodeCapPolicy:
    """Cap that spreads the decode *demand* across the PP pipeline.

    Avoids PP-depth collapse: after a pipeline drain (e.g. a
    prefill stall) every decode becomes reschedulable at once, and the static
    ``max_num_seqs // pp_size`` cap lets them pack into a single microbatch ->
    the other ``pp_size - 1`` stages idle (depth-1, ~2x decode latency). Sizing
    each step's batch to ``ceil(demand / pp_size)`` instead spreads the decode
    demand across ``pp_size`` microbatches so the pipeline stays full.

    ``num_demand_decodes`` is the total decode demand, NOT just the currently
    running decodes: it must also include remote-KV requests that are ready to
    join the decode batch (P/D-disaggregated decode). Using running-only would
    leave no headroom to admit those, stalling the ramp (the cap is fully
    consumed by the requests already cycling). Crucially, demand is *invariant*
    under admission -- promoting a ready remote-KV moves it from "ready" to
    "running" without changing the total -- so a single step-start snapshot is
    exact even though the running/ready split shifts during the waiting loop.

    The cap is **clamped to** ``static_max_cap`` (== the runner's compiled
    decode-batch ceiling) so it can only go *lower*, never overflow the bucket;
    and floored at 1. ``pp_size == 1`` yields ``min(static_max_cap, demand)``,
    a no-op (only ``demand`` decodes are admittable anyway).

    Constructed per ``schedule()`` call with that step's decode demand.
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
        """True iff one more decode request may be admitted this step.

        Two limits:
          * hard cap (always): the compiled bucket ceiling; exceeding it
            crashes the runner. Applies to every candidate decode join.
          * soft cap (``apply_soft_cap``): the balance/spreading target
            ``ceil(demand/pp)`` for the *budgeted* decode demand (running
            decodes + ready remote-KV). Skip it (``apply_soft_cap=False``) for
            joins not in the demand snapshot (full local prefix-cache match,
            resumed-after-eviction) — they only face the hard limit. In static
            mode the two caps are equal, so the flag has no effect.
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

        Unlike ``reset()`` (whole batch dropped by a prefill eviction), this
        removes only the requests that left the batch while others remain
        admitted -- currently the PRIORITY-policy preemption in the running
        loop, which evicts an already-scheduled decode to free KV blocks.
        Keeping the count in step with the batch keeps the subsequent
        ``can_admit()`` gate from stopping early on a stale (over)count.
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

    Built once from scheduler config; produces a fresh ``DecodeBatchBudget``
    for each ``schedule()`` call. It owns the PP decode-cap machinery — the
    compiled per-stage batch size, the static cap policy, and the choice
    between the static cap and the PP-balanced dynamic cap — so ``schedule()``
    only has to supply the demand and use the returned budget.

    Dynamic (demand-spread) capping applies only when ``pp_balance_decode`` is
    enabled *and* ``pipeline_parallel_size > 1``; otherwise the fixed
    ``max_num_seqs // pp_size`` cap is used.
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
