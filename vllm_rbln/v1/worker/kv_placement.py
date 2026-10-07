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
"""Size the KV cache from the compiled programs' device placement: the per-block
cost per chiplet comes from the shards of each dynamic input's `rebel.v2.Arg`, the
bytes already spoken for from a per-chiplet memory snapshot."""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# `(node, chiplet)`: the granularity of the device memory pools.
Unit = tuple[int, int]
# torch-rbln RBLNCachingAllocator.h constants; `allocator_reserved` replays its
# best-fit + split behaviour, under which small requests share a segment.
_ALLOC_SMALL_SIZE = 1 << 20
_ALLOC_SMALL_ROUND = 512
_ALLOC_SMALL_SEGMENT = 2 << 20
_ALLOC_LARGE_ROUND = 2 << 20


def _rounded(nbytes: int) -> int:
    unit = _ALLOC_SMALL_ROUND if nbytes <= _ALLOC_SMALL_SIZE else _ALLOC_LARGE_ROUND
    return -(-nbytes // unit) * unit


def allocator_reserved(requests: Iterable[int]) -> int:
    """Device bytes the caching allocator holds after serving `requests` in
    order: mapped segments, with split remainders serving later requests."""
    free: dict[bool, list[int]] = {True: [], False: []}
    reserved = 0
    for nbytes in requests:
        if nbytes <= 0:
            continue
        size = _rounded(nbytes)
        small = size <= _ALLOC_SMALL_SIZE
        pool = free[small]
        at = bisect.bisect_left(pool, size)
        if at < len(pool):
            block = pool.pop(at)
        else:
            block = _ALLOC_SMALL_SEGMENT if small else size
            reserved += block
        remainder = block - size
        if remainder >= (_ALLOC_SMALL_ROUND if small else _ALLOC_LARGE_ROUND):
            bisect.insort(pool, remainder)
    return reserved


_KEY_RE = re.compile(r"^npu\.(\d+)\.chiplet\.(\d+)\.(.+)$")


def is_kv_input(spec: Any) -> bool:
    """Whether `spec` is an input the program takes of any extent: a KV cache,
    the only input ever marked dynamic."""
    return spec.arg is not None and bool(spec.arg.logical.dynamic_axes)


def _dynamic_axis(spec: Any) -> Any:
    (axis,) = spec.arg.logical.dynamic_axes
    return axis


def _tensor_fingerprint(spec: Any) -> tuple:
    """What a KV input looks like regardless of how the device holds it."""
    return (tuple(spec.shape), str(spec.dtype))


def _input_key(spec: Any) -> tuple:
    """Order-independent identity of a KV input, for comparing programs."""
    return (
        *_tensor_fingerprint(spec),
        spec.arg.type_id,
        tuple(
            sorted(
                (s["node"], s["chiplet"], s["min_nbytes"], s["step_nbytes"])
                for s in spec.arg.shards
            )
        ),
    )


def select_kv_input_groups(programs: Sequence[Any]) -> list[tuple[list[Any], Any]]:
    """The distinct sets of KV inputs the programs bind (the target's, a
    drafter's), one `(specs, program)` each, in first-seen order. Inputs matching
    in shape and dtype but not in how the device holds them are the same tensors
    placed two ways, and refuse."""
    groups: dict[tuple, tuple[list[Any], Any]] = {}
    fingerprints: dict[tuple, tuple] = {}
    for program in programs:
        specs = [spec for spec in program.input_specs if is_kv_input(spec)]
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
                f"same {len(specs)} KV input(s) laid out differently. A KV "
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
    """The traced extent of `spec`'s dynamic axis: the kernel block count the
    program was traced with."""
    return int(spec.shape[_dynamic_axis(spec).axis])


def kv_requests_per_unit(
    specs: Iterable[Any], num_blocks: int, hint_blocks: int
) -> dict[Unit, list[int]]:
    """Bytes of each KV shard on each (node, chiplet) at `num_blocks`, in
    allocation order. A dynamic axis counts kernel blocks, so it is set to
    `num_blocks * dynamic_extent / hint_blocks`."""
    if hint_blocks <= 0:
        raise ValueError(f"hint_blocks must be positive, got {hint_blocks}")
    requests: dict[Unit, list[int]] = {}
    for spec in specs:
        extent = dynamic_extent(spec)
        if extent % hint_blocks:
            raise RuntimeError(
                f"input {spec.name!r} was traced with a dynamic extent of {extent}, "
                f"not a multiple of the {hint_blocks}-block compile hint."
            )
        steps = (extent // hint_blocks) * num_blocks - _dynamic_axis(spec).min
        for shard in spec.arg.shards:
            nbytes = shard["min_nbytes"] + shard["step_nbytes"] * steps
            unit = (int(shard["node"]), int(shard["chiplet"]))
            requests.setdefault(unit, []).append(nbytes)
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
    """Per-block growth per unit, checked to be linear through the origin (the
    compiler does not pad or transform a dynamic axis)."""
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
            "artifacts were not compiled with a dynamic KV axis."
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
    """From `torch.rbln.memory_stats_per_chiplet()`: this process's allocator,
    plus the other tenants' usage spread evenly and the runtime reserve."""
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
    """Largest `num_blocks` with `base + allocated(n) <= total * gmu` on every
    unit the KV cache grows on, `base = used - kv_resident + reserve_bytes`."""
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

    # The allocator's rounding makes the fit a step function of n; scan down
    # from the linear bound, which is an upper bound on the answer.
    num_blocks = min(linear.values())
    while num_blocks > 0 and not fits_all(num_blocks):
        num_blocks -= 1

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
    """One line per KV input; the only record of the per-shard extents."""
    lines = []
    for spec in specs:
        shards = ", ".join(
            f"({s['node']},{s['chiplet']})={s['min_nbytes']}+{s['step_nbytes']}/step"
            for s in spec.arg.shards
        )
        lines.append(
            f"{spec.name or '?'} traced={tuple(spec.shape)} "
            f"physical={spec.arg.physical} shards=[{shards}]"
        )
    return "; ".join(lines)
