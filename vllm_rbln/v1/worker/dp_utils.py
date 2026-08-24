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

"""Which padded batch a step runs, agreed across the data-parallel ranks.

RBLN compiles fixed decode shapes, so every step must land on a compiled
``(num_reqs_padded, query_len)`` pair -- and under data parallelism every rank
must land on the *same* one, or the ranks enter different graphs and the
collective hangs. The layers, mirroring upstream's ``vllm/v1/worker/dp_utils.py``:

- ``_run_ar``: one int per rank, sum-reduced into a tensor.
- ``synchronize_dp_ranks``: bit-packs this rank's counts, all-reduces them and
  unpacks a ``DPStatus`` -- transport only, it decides nothing.
- ``determine_batch_execution_and_padding``: the rule, a pure function of this
  rank's counts plus the status (None on a single rank), so every route is
  testable without a runner.
- ``coordinate_batch_across_dp``: the two together, for callers under DP.
- ``determine_draft_batch_execution_and_padding``: the draft's rule. Its shapes
  follow different bounds, but it runs inside a step that already agreed, so it
  reads the published status instead of reducing again.

``(num_reqs, query_len)`` are the free variables; tokens are their product. The
result therefore always states ``query_len`` outright: callers stage from it and
never divide ``num_tokens_padded`` to recover it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_dp_group


class BatchRoute(Enum):
    """Which rule decided this step's shape.

    Not derivable from the numbers: two rules can land on the same
    ``(num_reqs_padded, query_len, num_tokens_padded)`` -- an unspecialized DP
    decode and a decode alongside a prefilling peer both give ``(8, 1,
    max_num_tokens)`` -- so a log line or a test that only checked the values
    could not tell a right-numbers-wrong-rule step from a correct one. Nothing
    branches on it: callers run on the numbers and log the route, and the one
    route that has no numbers says so by coming back without a descriptor.
    """

    PINNED = "PINNED"
    """Warm-up dictated the token dimension: it compiles the shapes only DP
    asymmetry produces, which a symmetric warm-up group cannot reach on its own."""
    LOCAL = "LOCAL"
    """Single-DP: nothing to agree with, so the local bucket wins."""
    UNSPECIALIZED = "UNSPECIALIZED"
    """DP with specialize-moe-decode off: local bucket, tokens padded to the target."""
    ANY_PREFILL = "ANY_PREFILL"
    """Some rank is prefilling, so decoding ranks run the padded-decode graph."""
    ALL_IDLE = "ALL_IDLE"
    """Every rank drained: there is no shape to run, so the descriptor is None."""
    QLEN_ASYM = "QLEN_ASYM"
    """Busy ranks disagree on query_len; only the top bucket's padding is warmed."""
    AGREED = "AGREED"
    """The normal path: the busy ranks share a query_len and a bucket."""


@dataclass(frozen=True)
class DPStatus:
    """Per-rank counts and flags from one all-reduce, index-aligned by DP rank.

    Everything is data: this layer reports what the ranks said and leaves the
    policy to the rules. The step publishes it so that anything running inside
    the same step -- the draft -- decides from it rather than reducing again.
    """

    num_tokens: tuple[int, ...]
    num_reqs: tuple[int, ...]
    is_prefill: tuple[bool, ...]
    is_idle: tuple[bool, ...]
    """A DP-idle dummy step, which must not drive the shape."""
    num_tokens_across_dp: torch.Tensor
    """``num_tokens`` as the tensor ``set_forward_context`` consumes."""


@dataclass(frozen=True)
class ShapeConfig:
    """The compiled-shape snapshot of this worker."""

    decode_batch_buckets: Sequence[int]
    """Ascending compiled decode batch sizes."""
    find_bucket: Callable[[int], int]
    """``BucketingManager.find_decode_batch_bucket``, injected so the
    smallest-bucket-that-fits rule has one implementation."""
    max_num_tokens: int
    """Prefill-sized token dimension, used as the padding target."""
    specialized_moe_decode: bool


@dataclass(frozen=True)
class BatchDescriptor:
    """Describes the padded batch this rank runs, i.e. which compiled graph the
    step dispatches to. Same role as upstream's ``BatchDescriptor``, with the
    dimensions RBLN compiles for."""

    num_reqs_padded: int
    """Compiled batch dimension. On a prefill step this is ``num_reqs``: RBLN
    prefills one request at a time and the decode padding does not apply."""
    query_len: int
    """Tokens per request to stage. Equals this rank's own ``num_tokens //
    num_reqs`` except on ``AGREED``, where an idle rank adopts the busy ranks'
    length so it runs their graph."""
    num_tokens_padded: int | None
    """Token dimension of the graph the step runs, or None when the caller does
    not pad tokens (single-DP). Always at least ``num_reqs * query_len`` -- what
    this rank stages fits, and pads up to the dimension when it is smaller. The
    exact value follows from which warmed graph the route picked."""


def determine_batch_execution_and_padding(
    *,
    cfg: ShapeConfig,
    num_reqs: int,
    num_tokens: int,
    is_prefill: bool,
    status: DPStatus | None,
    pinned_num_tokens_padded: int | None = None,
) -> tuple[BatchDescriptor | None, BatchRoute]:
    """Decide this step's padded batch, or that there is none to run.

    ``status`` is None for single-DP. ``num_tokens`` must be ``num_reqs *
    query_len``: RBLN runs one query length per step, so a non-uniform batch is a
    caller bug rather than a shape to be chosen.

    A drained group comes back as ``(None, ALL_IDLE)``: every rank reads the same
    status, so they stop together, and there is no shape to state for a step none
    of them runs.

    ``pinned_num_tokens_padded`` is warm-up asking for a specific token dimension
    instead of a decided one; the returned descriptor is still the one value the
    callers read.
    """
    assert num_reqs >= 1, f"num_reqs must be >= 1, got {num_reqs}"
    assert num_tokens % num_reqs == 0, (
        f"num_tokens={num_tokens} is not a multiple of num_reqs={num_reqs}: "
        "a step's query length must be uniform"
    )
    query_len = num_tokens // num_reqs

    # A prefill step is not padded to a decode bucket; keep num_reqs so no caller
    # has to override the value afterwards.
    local_num_reqs_padded = num_reqs if is_prefill else cfg.find_bucket(num_reqs)

    if pinned_num_tokens_padded is not None:
        return (
            BatchDescriptor(
                num_reqs_padded=local_num_reqs_padded,
                query_len=query_len,
                num_tokens_padded=pinned_num_tokens_padded,
            ),
            BatchRoute.PINNED,
        )

    if status is None:
        return (
            BatchDescriptor(
                num_reqs_padded=local_num_reqs_padded,
                query_len=query_len,
                num_tokens_padded=None,
            ),
            BatchRoute.LOCAL,
        )

    busy = [i for i, idle in enumerate(status.is_idle) if not idle]
    if not busy:
        return None, BatchRoute.ALL_IDLE

    if not cfg.specialized_moe_decode:
        # Answered before the phase routes below: every bucket is compiled with
        # this token dimension here, so a prefilling peer leaves this rank's own
        # bucket usable and there is nothing to agree on.
        return (
            BatchDescriptor(
                num_reqs_padded=local_num_reqs_padded,
                query_len=query_len,
                num_tokens_padded=cfg.max_num_tokens,
            ),
            BatchRoute.UNSPECIALIZED,
        )

    if any(status.is_prefill):
        # A rank that prefills forces the token dimension to the prefill target,
        # which the small decode buckets cannot satisfy -- the decoding ranks run
        # the top bucket's padded-decode graph instead.
        return (
            BatchDescriptor(
                num_reqs_padded=(
                    num_reqs if is_prefill else cfg.decode_batch_buckets[-1]
                ),
                query_len=query_len,
                num_tokens_padded=cfg.max_num_tokens,
            ),
            BatchRoute.ANY_PREFILL,
        )

    assert all(status.num_tokens[i] % status.num_reqs[i] == 0 for i in busy), (
        "a busy rank reported a non-uniform query length: "
        f"num_tokens={status.num_tokens} num_reqs={status.num_reqs}"
    )
    query_lens = [status.num_tokens[i] // status.num_reqs[i] for i in busy]
    max_query_len = max(query_lens)

    if min(query_lens) != max_query_len:
        # Spec-decode asymmetry (one rank drafted, another did not). Only the top
        # bucket is warmed for it, and each rank pads its own tokens up to the
        # graph's dimension, so query_len stays this rank's own.
        top = cfg.decode_batch_buckets[-1]
        return (
            BatchDescriptor(
                num_reqs_padded=top,
                query_len=query_len,
                num_tokens_padded=top * max_query_len,
            ),
            BatchRoute.QLEN_ASYM,
        )

    # The busy ranks agree, so decide from them and let an idle rank adopt the
    # result -- its own (1 request, 1 token) must not shrink the graph.
    num_reqs_padded = cfg.find_bucket(max(status.num_reqs[i] for i in busy))
    return (
        BatchDescriptor(
            num_reqs_padded=num_reqs_padded,
            query_len=max_query_len,
            num_tokens_padded=num_reqs_padded * max_query_len,
        ),
        BatchRoute.AGREED,
    )


def determine_draft_batch_execution_and_padding(
    *,
    cfg: ShapeConfig,
    status: DPStatus | None,
    dp_rank: int,
    num_reqs: int,
    num_tokens: int,
    is_prefill: bool,
    draft_has_moe: bool,
    first_pass: bool = True,
    pinned_num_tokens_padded: int | None = None,
) -> tuple[BatchDescriptor, torch.Tensor | None]:
    """Decide the draft's padded batch from the status the step published.

    The draft runs inside a step whose ranks have already reduced, so it reads
    what they reported instead of reducing again. Its bounds are its own: the
    token dimension follows the busiest rank rather than a length the group
    agreed on, so ``query_len`` stays what this rank stages.

    Past the first pass every rank runs the drafting loop one token per request
    deep, whatever it staged before, so the counts are read at that length: the
    request counts are the token counts, and no rank is prefilling any more.

    ``draft_has_moe`` says whether the draft's forward reads the token dimension
    at all. Without fused MoE nothing does, so the batch stays this rank's own
    bucket rather than the one a peer's phase would ask for.

    Returns the descriptor and the per-rank token counts the draft's forward
    context consumes.
    """
    assert num_reqs >= 1, f"num_reqs must be >= 1, got {num_reqs}"
    assert num_tokens % num_reqs == 0, (
        f"num_tokens={num_tokens} is not a multiple of num_reqs={num_reqs}: "
        "a draft pass runs one query length"
    )
    query_len = num_tokens // num_reqs
    num_reqs_padded = num_reqs if is_prefill else cfg.find_bucket(num_reqs)

    if status is None:
        return (
            BatchDescriptor(
                num_reqs_padded=num_reqs_padded,
                query_len=query_len,
                num_tokens_padded=pinned_num_tokens_padded,
            ),
            None,
        )

    assert status.num_reqs[dp_rank] == num_reqs, (
        "the draft is not on the step this status came from: "
        f"reported {status.num_reqs[dp_rank]} requests, running {num_reqs}"
    )

    if first_pass:
        tokens_per_rank = status.num_tokens
        any_prefill = any(status.is_prefill)
        tokens_across_dp = status.num_tokens_across_dp
        # An idle rank reported the minimal entry so that it would not drive the
        # decision, then staged the length that decision handed back, so its
        # token count is the only one that does not describe what its draft runs.
        assert status.is_idle[dp_rank] or status.num_tokens[dp_rank] == num_tokens, (
            "the draft is not on the step this status came from: "
            f"reported {status.num_tokens[dp_rank]} tokens, running {num_tokens}"
        )
    else:
        assert query_len == 1, f"the drafting loop runs one token, got {query_len}"
        tokens_per_rank = status.num_reqs
        any_prefill = False
        tokens_across_dp = torch.tensor(status.num_reqs, dtype=torch.int32)

    if not draft_has_moe:
        # Nothing in this draft reads the token dimension, so it states what this
        # pass stages rather than a dimension the ranks have to share.
        num_tokens_padded = num_reqs_padded * query_len
    elif cfg.specialized_moe_decode and not is_prefill:
        if any_prefill:
            # The prefilling rank dictates the dimension, and only the top bucket
            # is compiled with it.
            num_reqs_padded = cfg.decode_batch_buckets[-1]
            num_tokens_padded = cfg.max_num_tokens
        else:
            num_reqs_padded = cfg.find_bucket(max(status.num_reqs))
            num_tokens_padded = num_reqs_padded * max(
                tokens // reqs for tokens, reqs in zip(tokens_per_rank, status.num_reqs)
            )
    else:
        # The dispatch dimension has to be the same on every rank, and this is the
        # one value they all reach without agreeing on a batch.
        num_tokens_padded = cfg.max_num_tokens
    if pinned_num_tokens_padded is not None:
        num_tokens_padded = pinned_num_tokens_padded

    # The fused-MoE dispatch pads what a pass stages up to this dimension and only
    # up to it: staging past it hands the all-gather a tensor the peers do not
    # expect. Every caller reaches it through a length the ranks reported or the
    # one they settled on, so the bound holds -- this states it rather than
    # leaving a pass that got the length wrong to run truncated.
    assert num_reqs_padded * query_len <= num_tokens_padded, (
        f"the draft stages {num_reqs_padded} x {query_len} tokens, more than the "
        f"{num_tokens_padded} the group settled on"
    )

    return (
        BatchDescriptor(
            num_reqs_padded=num_reqs_padded,
            query_len=query_len,
            num_tokens_padded=num_tokens_padded,
        ),
        tokens_across_dp,
    )


def _run_ar(value: int, dp_size: int, dp_rank: int) -> torch.Tensor:
    """Sum-reduce one int per rank into a CPU tensor of size dp_size.

    Every rank contributes into its own slot and zero elsewhere, so the sum is a
    gather. Runs on the CPU group: the values are tiny and the caller is on the
    host path.
    """
    slots = [0] * dp_size
    slots[dp_rank] = value
    tensor = torch.tensor(slots, device="cpu", dtype=torch.int32)
    dist.all_reduce(tensor, group=get_dp_group().cpu_group)
    return tensor


def _synchronize_dp_ranks(
    num_tokens: int,
    num_reqs: int,
    dp_size: int,
    dp_rank: int,
    is_prefill: bool,
    is_idle: bool = False,
) -> DPStatus:
    """All-reduce per-rank (num_tokens, num_reqs, is_prefill, is_idle)
    across DP via a single bit-packed int32 and split the result back out.

    Transport only: it reports what every rank said and leaves the shape
    decision to ``determine_batch_execution_and_padding``.

    Bit layout (int32, low to high; bit 31 is left unused to keep the packed
    value non-negative for the sum-based all-gather):
        bits  0..15  num_tokens  (max 65535)
        bits 16..28  num_reqs    (max 8191; >> any real max_num_seqs)
        bit  29      is_prefill flag
        bit  30      is_idle flag (DP-idle dummy step)
    """
    token_bits = 16
    req_bits = 13
    token_mask = (1 << token_bits) - 1
    req_mask_raw = (1 << req_bits) - 1
    req_mask_shifted = req_mask_raw << token_bits
    prefill_flag = 1 << (token_bits + req_bits)
    idle_flag = 1 << (token_bits + req_bits + 1)

    assert num_tokens <= token_mask, (
        f"num_tokens={num_tokens} exceeds bit-packed limit {token_mask}"
    )
    assert num_reqs <= req_mask_raw, (
        f"num_reqs={num_reqs} exceeds bit-packed limit {req_mask_raw}"
    )

    encoded = num_tokens | (num_reqs << token_bits)
    if is_prefill:
        encoded |= prefill_flag
    if is_idle:
        encoded |= idle_flag

    encoded_across_dp = _run_ar(encoded, dp_size, dp_rank)

    token_mask_t = torch.tensor([token_mask] * dp_size, device="cpu", dtype=torch.int32)
    num_tokens_across_dp_cpu = encoded_across_dp & token_mask_t

    # Unpack once into ints: the decision reads these several times, and one
    # all-reduce carries only dp_size numbers.
    packed_per_rank = encoded_across_dp.tolist()
    return DPStatus(
        num_tokens=tuple(value & token_mask for value in packed_per_rank),
        num_reqs=tuple(
            (value & req_mask_shifted) >> token_bits for value in packed_per_rank
        ),
        is_prefill=tuple(bool(value & prefill_flag) for value in packed_per_rank),
        is_idle=tuple(bool(value & idle_flag) for value in packed_per_rank),
        num_tokens_across_dp=num_tokens_across_dp_cpu,
    )


def coordinate_batch_across_dp(
    *,
    cfg: ShapeConfig,
    dp_size: int,
    dp_rank: int,
    num_reqs: int,
    num_tokens: int,
    is_prefill: bool,
    is_idle: bool = False,
    pinned_num_tokens_padded: int | None = None,
) -> tuple[BatchDescriptor | None, BatchRoute, DPStatus]:
    """Agree on this step's padded batch with the other DP ranks.

    Only for ``dp_size > 1``: it joins the collective, so every rank of the group
    must call it on the same step or they deadlock. A single-DP caller decides
    directly with the rule. The decision comes back with the status the ranks
    reported -- token counts, request counts, whether any rank is
    prefilling -- so anything that would otherwise repeat the collective reads
    those numbers instead.
    """
    status = _synchronize_dp_ranks(
        num_tokens, num_reqs, dp_size, dp_rank, is_prefill, is_idle
    )
    batch_desc, route = determine_batch_execution_and_padding(
        cfg=cfg,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        is_prefill=is_prefill,
        status=status,
        pinned_num_tokens_padded=pinned_num_tokens_padded,
    )
    return batch_desc, route, status
