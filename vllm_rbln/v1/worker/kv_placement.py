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
cost per chiplet comes from each dynamic input's `PhysicalPlacement`, the bytes
already spoken for from a per-chiplet memory snapshot."""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# `(node_id, chiplet_id)`: the granularity of the device memory pools.
Unit = tuple[int, int]
# rebel runtime caching_allocator.h constants; `allocator_reserved` replays its
# best-fit + split behaviour, under which several requests share a segment.
_ALLOC_MIN_BLOCK = 4096
_ALLOC_SMALL_MAX = 1 << 20
_ALLOC_SMALL_BLOCK = 2 << 20
_ALLOC_MEDIUM_MAX = 10 << 20
_ALLOC_MEDIUM_BLOCK = 20 << 20
_ALLOC_LARGE_ROUND = 2 << 20


def allocation_size(nbytes: int) -> int:
    """Segment the caching allocator maps for a request no free block serves."""
    if nbytes <= 0:
        return 0
    if nbytes <= _ALLOC_SMALL_MAX:
        return _ALLOC_SMALL_BLOCK
    if nbytes <= _ALLOC_MEDIUM_MAX:
        return _ALLOC_MEDIUM_BLOCK
    return -(-nbytes // _ALLOC_LARGE_ROUND) * _ALLOC_LARGE_ROUND


def allocator_reserved(requests: Iterable[int]) -> int:
    """Device bytes the caching allocator holds after serving `requests` in
    order: mapped segments, with split remainders serving later requests."""
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
    """Evaluate one `PlacementDim` with every symbol bound to `num_blocks`, the
    only dim ever marked dynamic."""
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
    """The distinct sets of KV inputs the programs bind (the target's, a
    drafter's), one `(specs, program)` each, in first-seen order. Inputs matching
    in shape and dtype but not in shards are the same tensors placed two ways,
    and refuse."""
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


def dynamic_dim(spec: Any) -> int:
    """The index of `spec`'s dynamic dim."""
    for index, dim in enumerate(spec.physical_placement.shape):
        if not isinstance(dim, int):
            return index
    raise ValueError(f"input {spec.name!r} has a placement but no dynamic dim")


def dynamic_extent(spec: Any) -> int:
    """The compiled extent of `spec`'s dynamic dim: the kernel block count the
    program was traced with."""
    return int(spec.shape[dynamic_dim(spec)])


def rebound_extent(spec: Any, num_blocks: int, hint_blocks: int) -> int:
    """`spec`'s dynamic dim at `num_blocks`. It counts kernel blocks, so it binds
    to `num_blocks * dynamic_extent / hint_blocks`."""
    extent = dynamic_extent(spec)
    if extent % hint_blocks:
        raise RuntimeError(
            f"input {spec.name!r} was compiled with a dynamic extent of {extent}, "
            f"not a multiple of the {hint_blocks}-block compile hint."
        )
    return (extent // hint_blocks) * num_blocks


def relatched_shapes(
    program: Any, num_blocks: int, hint_blocks: int
) -> list[list[int]]:
    """`program`'s input shapes at `num_blocks`: the compiled shapes with every
    dynamic dim rebound."""
    if hint_blocks <= 0:
        raise ValueError(f"hint_blocks must be positive, got {hint_blocks}")
    shapes = []
    for spec in program.input_specs:
        shape = [int(dim) for dim in spec.shape]
        if spec.physical_placement is not None:
            shape[dynamic_dim(spec)] = rebound_extent(spec, num_blocks, hint_blocks)
        shapes.append(shape)
    return shapes


def kv_requests_per_unit(
    specs: Iterable[Any], num_blocks: int, hint_blocks: int
) -> dict[Unit, list[int]]:
    """Bytes of each KV shard on each (node, chiplet) at `num_blocks`, in
    allocation order. A dynamic dim counts kernel blocks, so its symbol is bound
    to `num_blocks * dynamic_extent / hint_blocks`."""
    if hint_blocks <= 0:
        raise ValueError(f"hint_blocks must be positive, got {hint_blocks}")
    requests: dict[Unit, list[int]] = {}
    for spec in specs:
        placement = spec.physical_placement
        symbol = rebound_extent(spec, num_blocks, hint_blocks)
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
    """Per-block growth per unit, checked to be linear through the origin (the
    compiler does not pad or transform a dynamic dim)."""
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

    # Not monotone in n: a shard crossing 10 MiB drops from a 20 MiB segment to
    # 2 MiB rounding, so a count can fit while smaller ones do not. Scan down
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
