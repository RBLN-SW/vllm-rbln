# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Size the KV cache from the compiled programs' device placement.

Two questions, two sources. *How many bytes does one more block cost on each
chiplet* is a compile-time contract: every dynamic-shape input of a program
carries a `PhysicalPlacement` whose shard extents are expressions over the
dynamic dim. *How many bytes are already spoken for on each chiplet* is runtime
state, read from a per-chiplet memory snapshot after warm-up.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# `(node_id, chiplet_id)` -- the granularity device memory pools are carved at.
Unit = tuple[int, int]
# The runtime's caching allocator (rebel caching_allocator.cc): requests are
# rounded to 4 KiB and served best-fit from the pool's free blocks; a miss maps
# a new segment of 2 MiB (request <= 1 MiB), 20 MiB (<= 10 MiB) or the request
# rounded up to 2 MiB. The block is split when the remainder is >= 512 B in the
# small pool or > 1 MiB in the large pool, so several requests share a segment.
_ALLOC_MIN_BLOCK = 4096
_ALLOC_SMALL_MAX = 1 << 20
_ALLOC_SMALL_BLOCK = 2 << 20
_ALLOC_MEDIUM_MAX = 10 << 20
_ALLOC_MEDIUM_BLOCK = 20 << 20
_ALLOC_LARGE_ROUND = 2 << 20


def allocation_size(nbytes: int) -> int:
    """Segment the caching allocator maps for a request of `nbytes` that no free
    block can serve."""
    if nbytes <= 0:
        return 0
    if nbytes <= _ALLOC_SMALL_MAX:
        return _ALLOC_SMALL_BLOCK
    if nbytes <= _ALLOC_MEDIUM_MAX:
        return _ALLOC_MEDIUM_BLOCK
    return -(-nbytes // _ALLOC_LARGE_ROUND) * _ALLOC_LARGE_ROUND


def allocator_reserved(requests: Iterable[int]) -> int:
    """Device bytes the caching allocator holds after serving `requests` in
    order with nothing freed in between: the segments it maps, with the
    remainder of a split block serving later requests of the same pool."""
    free: dict[bool, list[int]] = {True: [], False: []}
    reserved = 0
    for nbytes in requests:
        if nbytes <= 0:
            continue
        size = -(-nbytes // _ALLOC_MIN_BLOCK) * _ALLOC_MIN_BLOCK
        small = size <= _ALLOC_SMALL_MAX
        pool = free[small]
        at = bisect.bisect_left(pool, size)
        if at < len(pool):
            block = pool.pop(at)
        else:
            block = allocation_size(size)
            reserved += block
        remainder = block - size
        splits = (
            remainder >= _ALLOC_MIN_BLOCK if small else remainder > _ALLOC_SMALL_MAX
        )
        if splits:
            bisect.insort(pool, remainder)
    return reserved


_KEY_RE = re.compile(r"^npu\.(\d+)\.chiplet\.(\d+)\.(.+)$")
_TRAILING_BITS_RE = re.compile(r"(\d+)$")


def eval_placement_dim(dim: Any, num_blocks: int) -> int:
    """Evaluate one `PlacementDim` with every dynamic-shape variable bound to
    `num_blocks`.

    Only `num_blocks` can be marked dynamic, so every symbol is that count.
    """
    if isinstance(dim, bool):
        raise ValueError(f"placement dim cannot be a bool: {dim!r}")
    if isinstance(dim, int):
        return dim
    if isinstance(dim, tuple) and dim:
        kind, *operands = dim
        if kind == "symbol":
            return num_blocks
        if kind == "prod":
            return math.prod(eval_placement_dim(x, num_blocks) for x in operands)
        if kind == "sum":
            return sum(eval_placement_dim(x, num_blocks) for x in operands)
    raise ValueError(f"unrecognised placement dim expression: {dim!r}")


def placement_itemsize(dtype: str) -> int:
    """Bytes per element of a physical placement dtype ("dlfloat16", "float16",
    "bfloat16", "e4m3_float8", "int8", ...)."""
    name = dtype.strip().lower()
    if name == "bool":
        return 1
    match = _TRAILING_BITS_RE.search(name)
    if match is None:
        raise ValueError(f"cannot derive an element size from dtype {dtype!r}")
    bits = int(match.group(1))
    if bits <= 0:
        raise ValueError(f"cannot derive an element size from dtype {dtype!r}")
    return -(-bits // 8)


def _tensor_fingerprint(spec: Any) -> tuple:
    """What a KV input looks like regardless of how it is sharded."""
    placement = spec.physical_placement
    return (tuple(spec.shape), tuple(placement.shape), str(placement.dtype))


def _input_key(spec: Any) -> tuple:
    """Order-independent identity of a KV input, for comparing programs."""
    placement = spec.physical_placement
    return (
        *_tensor_fingerprint(spec),
        tuple(
            sorted(
                (int(s.node_id), int(s.chiplet_id), tuple(s.slice_shape))
                for s in placement.shards
            )
        ),
    )


def select_kv_input_groups(programs: Sequence[Any]) -> list[tuple[list[Any], Any]]:
    """The distinct sets of KV inputs the programs bind, one `(specs, program)`
    per set, in first-seen order.

    Only the KV caches are marked dynamic, so a program's dynamic-shape inputs
    are exactly its KV inputs; a static program (compute_logits) has none. The
    target's prefill and decode programs bind the same tensors and so agree on
    the placement; a speculative drafter's programs bind their own tensors and
    form a second set, which the sizing adds on top. Two programs whose inputs
    match in shape and dtype but not in shards are the same tensors placed two
    ways -- the runtime would re-place the cache on every switch -- and refuse.
    """
    groups: dict[tuple, tuple[list[Any], Any]] = {}
    fingerprints: dict[tuple, tuple] = {}
    for program in programs:
        specs = [
            spec for spec in program.input_specs if spec.physical_placement is not None
        ]
        if not specs:
            continue
        key = tuple(sorted(_input_key(spec) for spec in specs))
        if key in groups:
            continue
        fingerprint = tuple(sorted(_tensor_fingerprint(spec) for spec in specs))
        if fingerprint in fingerprints:
            other = groups[fingerprints[fingerprint]][1]
            raise RuntimeError(
                "compiled programs disagree on the KV cache placement: "
                f"{_program_name(other)} and {_program_name(program)} bind the "
                f"same {len(specs)} KV input(s) with different shard layouts. A KV "
                "tensor can only hold one placement at a time."
            )
        groups[key] = (specs, program)
        fingerprints[fingerprint] = key
    if not groups:
        raise RuntimeError(
            f"none of the {len(programs)} compiled program(s) carries a dynamic-shape "
            "KV input; was VLLM_CACHE_ROOT replaying a static build?"
        )
    return list(groups.values())


def _program_name(program: Any) -> str:
    name = getattr(program, "name", "")
    return f"program {name!r}" if name else "an unnamed program"


def dynamic_extent(spec: Any) -> int:
    """The compiled extent of `spec`'s dynamic dim: the kernel block count the
    program was traced with."""
    for index, dim in enumerate(spec.physical_placement.shape):
        if not isinstance(dim, int):
            return int(spec.shape[index])
    raise ValueError(f"input {spec.name!r} has a placement but no dynamic dim")


def kv_requests_per_unit(
    specs: Iterable[Any], num_blocks: int, hint_blocks: int
) -> dict[Unit, list[int]]:
    """Bytes of each KV shard on each (node, chiplet) at `num_blocks`, in the
    order the shards are allocated.

    An input's dynamic dim counts *kernel* blocks, `dynamic_extent / hint_blocks`
    of them per manager block (a sliding-window layer splits each block into
    `block_size / sliding_window`), so its symbol is bound to that multiple.
    """
    if hint_blocks <= 0:
        raise ValueError(f"hint_blocks must be positive, got {hint_blocks}")
    requests: dict[Unit, list[int]] = {}
    for spec in specs:
        placement = spec.physical_placement
        extent = dynamic_extent(spec)
        if extent % hint_blocks:
            raise RuntimeError(
                f"input {spec.name!r} was compiled with a dynamic extent of {extent}, "
                f"not a multiple of the {hint_blocks}-block compile hint."
            )
        symbol = (extent // hint_blocks) * num_blocks
        itemsize = placement_itemsize(placement.dtype)
        for shard in placement.shards:
            elems = math.prod(
                eval_placement_dim(dim, symbol) for dim in shard.slice_shape
            )
            unit = (int(shard.node_id), int(shard.chiplet_id))
            requests.setdefault(unit, []).append(elems * itemsize)
    return requests


def kv_bytes_per_unit(
    specs: Iterable[Any], num_blocks: int, hint_blocks: int
) -> dict[Unit, int]:
    """Bytes the KV inputs occupy on each (node, chiplet) at `num_blocks`."""
    return {
        unit: sum(shards)
        for unit, shards in kv_requests_per_unit(specs, num_blocks, hint_blocks).items()
    }


@dataclass(frozen=True)
class KvGrowth:
    """Per-unit cost of the KV cache as a function of `num_blocks`."""

    per_block: dict[Unit, int]
    hint_blocks: int
    num_inputs: int
    specs: tuple[Any, ...]

    def bytes_at(self, num_blocks: int) -> dict[Unit, int]:
        return {unit: num_blocks * cost for unit, cost in self.per_block.items()}

    def allocated_at(self, num_blocks: int) -> dict[Unit, int]:
        """What the caching allocator reserves on each unit for the shards
        `bytes_at` counts."""
        return {
            unit: allocator_reserved(shards)
            for unit, shards in kv_requests_per_unit(
                self.specs, num_blocks, self.hint_blocks
            ).items()
        }


def kv_growth(specs: Sequence[Any], hint_blocks: int) -> KvGrowth:
    """Per-block growth per unit, checked to be linear through the origin.

    The compiler refuses to pad, transform or shard a dynamic dim, which is what
    makes shard bytes exactly `num_blocks * per_block`. The check turns a future
    relaxation of that rule into a start-up error instead of a wrong size.
    """
    at_zero = kv_bytes_per_unit(specs, 0, hint_blocks)
    at_one = kv_bytes_per_unit(specs, 1, hint_blocks)
    per_block = {
        unit: slope
        for unit, total in at_one.items()
        if (slope := total - at_zero.get(unit, 0)) > 0
    }
    if not per_block:
        raise RuntimeError(
            "the KV placements have no per-block growth on any chiplet, i.e. the "
            "artifacts were not compiled with a dynamic KV dim."
        )
    offsets = {unit: b for unit, b in at_zero.items() if b}
    at_hint = {
        unit: b
        for unit, b in kv_bytes_per_unit(specs, hint_blocks, hint_blocks).items()
        if b
    }
    expected = {unit: hint_blocks * slope for unit, slope in per_block.items()}
    if offsets or at_hint != expected:
        raise RuntimeError(
            "KV placement bytes are not linear through the origin in num_blocks "
            f"(at 0: {offsets}, at {hint_blocks}: {at_hint}, expected "
            f"{expected}); the compiler's dynamic-input contract changed."
        )
    return KvGrowth(
        per_block=per_block,
        hint_blocks=hint_blocks,
        num_inputs=len(specs),
        specs=tuple(specs),
    )


@dataclass(frozen=True)
class ChipletMemory:
    """Device DRAM of one (node, chiplet) as the budget sees it."""

    total: int
    used: int


def _per_unit(stats: Mapping[str, int], field: str) -> dict[Unit, int]:
    out: dict[Unit, int] = {}
    for key, value in stats.items():
        match = _KEY_RE.match(key)
        if match is None or match.group(3) != field:
            continue
        out[(int(match.group(1)), int(match.group(2)))] = int(value)
    return out


def snapshot_from_driver(stats: Mapping[str, int]) -> dict[Unit, ChipletMemory]:
    """From `torch.rbln.mem_get_info_per_chiplet()`: the driver's view, every
    process included."""
    total = _per_unit(stats, "total")
    used = _per_unit(stats, "used")
    if not total or set(total) != set(used):
        raise RuntimeError(
            "mem_get_info_per_chiplet() reply has no per-chiplet total/used pairs "
            f"(total units={sorted(total)}, used units={sorted(used)})."
        )
    return {u: ChipletMemory(total=total[u], used=used[u]) for u in sorted(total)}


def snapshot_from_allocator(
    stats: Mapping[str, int],
    *,
    memory_per_chiplet: int,
    foreign_card_used_bytes: int,
    reserve_bytes: int,
) -> dict[Unit, ChipletMemory]:
    """From `torch.rbln.memory_stats_per_chiplet()`: this process's caching
    allocator only.

    Two things the allocator cannot see are added back: other tenants' usage,
    sampled card-wide before this worker allocated and spread evenly, and a
    fixed reserve for the runtime's own direct allocations (command streams).
    """
    reserved = _per_unit(stats, "reserved.current")
    if not reserved:
        raise RuntimeError(
            "memory_stats_per_chiplet() reported no `reserved.current` per chiplet; "
            "the allocator has no device context yet."
        )
    if memory_per_chiplet <= 0:
        raise ValueError(
            f"memory_per_chiplet must be positive, got {memory_per_chiplet}"
        )
    foreign_per_unit = max(0, foreign_card_used_bytes) // len(reserved)
    return {
        u: ChipletMemory(
            total=memory_per_chiplet,
            used=reserved[u] + foreign_per_unit + reserve_bytes,
        )
        for u in sorted(reserved)
    }


@dataclass(frozen=True)
class UnitFit:
    """How one unit's budget was spent; for the log line and the refusal."""

    total: int
    budget: int
    used: int
    kv_resident: int
    reserve: int
    base: int
    per_block: int
    num_blocks: int


def max_num_blocks(
    snapshot: Mapping[Unit, ChipletMemory],
    growth: KvGrowth,
    *,
    gpu_memory_utilization: float,
    kv_resident: Mapping[Unit, int],
    reserve_bytes: int = 0,
) -> tuple[int, dict[Unit, UnitFit]]:
    """Largest `num_blocks` whose KV cache fits every chiplet's budget.

    `used` was sampled with the KV cache resident at `kv_resident` bytes, so the
    non-KV base is `used - kv_resident + reserve_bytes`, the reserve standing
    for device memory the runtime allocates only once requests flow; the answer
    satisfies `base + allocated(n) <= total * gpu_memory_utilization` on every
    unit the KV cache grows on, `allocated` counting each shard at the size the
    caching allocator reserves. Units it does not touch are not sized here.
    """
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )
    missing = sorted(set(growth.per_block) - set(snapshot))
    if missing:
        raise RuntimeError(
            f"the memory snapshot has no entry for (node, chiplet) {missing} that "
            f"the KV cache grows on (snapshot covers {sorted(snapshot)})."
        )
    room: dict[Unit, int] = {}
    linear: dict[Unit, int] = {}
    for unit, per_block in growth.per_block.items():
        mem = snapshot[unit]
        budget = int(mem.total * gpu_memory_utilization)
        base = mem.used - int(kv_resident.get(unit, 0)) + reserve_bytes
        room[unit] = budget - base
        linear[unit] = max(0, room[unit] // per_block)

    def fits_all(n: int) -> bool:
        allocated = growth.allocated_at(n)
        return all(allocated[unit] <= room[unit] for unit in room)

    # fits_all is monotone in n: the allocator's rounding costs a few blocks
    # below the linear bound, so bisect for the largest n that fits.
    low, high = 0, min(linear.values())
    while low < high:
        mid = (low + high + 1) // 2
        if fits_all(mid):
            low = mid
        else:
            high = mid - 1
    num_blocks = low

    fits: dict[Unit, UnitFit] = {}
    for unit, per_block in sorted(growth.per_block.items()):
        mem = snapshot[unit]
        fits[unit] = UnitFit(
            total=mem.total,
            budget=int(mem.total * gpu_memory_utilization),
            used=mem.used,
            kv_resident=int(kv_resident.get(unit, 0)),
            reserve=reserve_bytes,
            base=mem.used - int(kv_resident.get(unit, 0)) + reserve_bytes,
            per_block=per_block,
            num_blocks=linear[unit],
        )
    return num_blocks, fits


def format_fits(fits: Mapping[Unit, UnitFit]) -> str:
    return " ".join(
        f"{n}:{c}(total={f.total} budget={f.budget} used={f.used} "
        f"kv_resident={f.kv_resident} reserve={f.reserve} base={f.base} "
        f"per_block={f.per_block} "
        f"blocks={f.num_blocks})"
        for (n, c), f in sorted(fits.items())
    )


def format_placements(specs: Iterable[Any]) -> str:
    """One line per KV input, field by field so the record survives compiler
    builds; the only record of the per-shard extents."""
    lines = []
    for spec in specs:
        placement = spec.physical_placement
        shards = ", ".join(
            f"({int(s.node_id)},{int(s.chiplet_id)})={tuple(s.slice_shape)}"
            for s in placement.shards
        )
        lines.append(
            f"{spec.name or '?'} compiled={tuple(spec.shape)} "
            f"shape={tuple(placement.shape)} dtype={placement.dtype} shards=[{shards}]"
        )
    return "; ".join(lines)
