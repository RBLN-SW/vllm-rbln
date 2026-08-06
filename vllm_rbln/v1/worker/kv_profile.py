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
"""Merge `rebel` KV-cache memory profiles into a single joint profile."""

# NOTE(RBLN): the reduction is load-bearing and nothing here can test it -- `min`
# is optimistic, summing the growth double-counts, and every merge recorded on
# hardware had N=2, where all candidate rules are algebraically identical.
# See docs/dynamic_kv_cache.md, "How the per-artifact profiles are merged".

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# NOTE(RBLN): log analysis recovers base_bytes / bytes_per_block by scanning for
# these tokens, so the merged key must not contain the source one.
SOURCE_PROFILE_LOG_KEY = "device_regions"
MERGED_PROFILE_LOG_KEY = "merged_regions"


@dataclass(frozen=True)
class MergedMemoryRegion:
    """One region of the merged profile; fields mirror `PyRblnMemoryRegion`."""

    node_id: int
    chiplet_id: int
    base_bytes: int
    bytes_per_block: int
    alignment: int


@dataclass
class MergedKvCacheMemoryProfile:
    """Duck-typed stand-in for `rebel._C.PyRblnKvCacheMemoryProfile`."""

    device_regions: list[MergedMemoryRegion]
    host_base_bytes: int = 0
    host_bytes_per_block: int = 0
    # Bookkeeping for logs / the prediction-vs-measurement gate. Not read by
    # `max_num_blocks`.
    num_source_profiles: int = 0
    num_shared_base_regions: int = 0
    num_private_base_regions: int = 0
    num_growth_regions: int = 0


def _region_key(region: Any) -> tuple[int, int, int, int]:
    """Dedup key; `alignment` is excluded, it is a literal constant."""
    return (
        int(region.node_id),
        int(region.chiplet_id),
        int(region.base_bytes),
        int(region.bytes_per_block),
    )


def _to_merged(region: Any) -> MergedMemoryRegion:
    return MergedMemoryRegion(
        node_id=int(region.node_id),
        chiplet_id=int(region.chiplet_id),
        base_bytes=int(region.base_bytes),
        bytes_per_block=int(region.bytes_per_block),
        alignment=int(region.alignment),
    )


@dataclass
class _Split:
    base: list[Any] = field(default_factory=list)
    growth: list[Any] = field(default_factory=list)


def _split_regions(profile: Any) -> _Split:
    out = _Split()
    for region in profile.device_regions:
        if int(region.bytes_per_block) > 0:
            out.growth.append(region)
        else:
            out.base.append(region)
    return out


def _merge_base_by_multiset(
    splits: list[_Split],
) -> tuple[list[MergedMemoryRegion], int, int]:
    """Shared == multiset intersection; the remainder is private and summed."""
    counters = [Counter(_region_key(r) for r in s.base) for s in splits]
    shared = counters[0].copy()
    for counter in counters[1:]:
        shared &= counter

    prototypes: dict[tuple[int, int, int, int], Any] = {}
    for split in splits:
        for region in split.base:
            prototypes.setdefault(_region_key(region), region)

    merged: list[MergedMemoryRegion] = []
    for key, count in shared.items():
        merged.extend([_to_merged(prototypes[key])] * count)
    num_shared = len(merged)

    for counter in counters:
        for key, count in (counter - shared).items():
            merged.extend([_to_merged(prototypes[key])] * count)
    num_private = len(merged) - num_shared

    logger.info(
        "[Dynamic KV] base regions deduped: shared=%d private=%d",
        num_shared,
        num_private,
    )
    return merged, num_shared, num_private


def _merge_growth(splits: list[_Split]) -> list[MergedMemoryRegion]:
    """Count per-block growth once per (node, chiplet)."""
    # NOTE(RBLN): kept as individual regions, not collapsed per chiplet --
    # `max_num_blocks` aligns every region on its own.
    totals: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for index, split in enumerate(splits):
        acc: Counter = Counter()
        for region in split.growth:
            acc[(int(region.node_id), int(region.chiplet_id))] += int(
                region.bytes_per_block
            )
        for unit, total in acc.items():
            totals[unit][index] = total

    merged: list[MergedMemoryRegion] = []
    for unit in sorted(totals):
        per_profile = totals[unit]
        if len(set(per_profile.values())) > 1:
            logger.warning(
                "[Dynamic KV] per-block growth disagrees at (node=%d, chiplet=%d): "
                "%s. Taking the maximum; the KV layout may not be shared.",
                unit[0],
                unit[1],
                {i: v for i, v in sorted(per_profile.items())},
            )
        # Largest per-block growth wins; ties go to the earliest profile.
        winner = min((-total, index) for index, total in per_profile.items())[1]
        for region in splits[winner].growth:
            if (int(region.node_id), int(region.chiplet_id)) == unit:
                merged.append(_to_merged(region))
    return merged


def merge_kv_cache_memory_profiles(
    profiles: list[Any],
) -> MergedKvCacheMemoryProfile:
    """Merge per-artifact KV-cache memory profiles into one joint profile."""
    if not profiles:
        raise ValueError("merge_kv_cache_memory_profiles() needs >= 1 profile")

    splits = [_split_regions(p) for p in profiles]

    base, num_shared, num_private = _merge_base_by_multiset(splits)
    growth = _merge_growth(splits)

    host_base_bytes = max((int(p.host_base_bytes) for p in profiles), default=0)
    host_bytes_per_block = max(
        (int(p.host_bytes_per_block) for p in profiles), default=0
    )

    merged = MergedKvCacheMemoryProfile(
        device_regions=base + growth,
        host_base_bytes=host_base_bytes,
        host_bytes_per_block=host_bytes_per_block,
        num_source_profiles=len(profiles),
        num_shared_base_regions=num_shared,
        num_private_base_regions=num_private,
        num_growth_regions=len(growth),
    )
    logger.info(
        "[Dynamic KV] merged %d compiled profile(s): regions=%d "
        "(base=%d growth=%d) host_base_bytes=%d host_bytes_per_block=%d",
        len(profiles),
        len(merged.device_regions),
        len(base),
        len(growth),
        host_base_bytes,
        host_bytes_per_block,
    )
    return merged


def build_per_chiplet_budget(
    num_nodes: int,
    num_chiplets: int,
    budget_bytes_per_chiplet: int,
) -> dict[int, dict[int, int]]:
    """Return the `{node_id: {chiplet_id: bytes}}` map `max_num_blocks` wants."""
    # NOTE(RBLN): every combination is filled in -- a unit the profile uses but
    # the map omits is silently budgeted 0 bytes, i.e. an answer of 0 blocks.
    if num_nodes <= 0 or num_chiplets <= 0:
        raise ValueError(
            f"invalid device topology: num_nodes={num_nodes} "
            f"num_chiplets={num_chiplets}"
        )
    return {
        node_id: {
            chiplet_id: budget_bytes_per_chiplet for chiplet_id in range(num_chiplets)
        }
        for node_id in range(num_nodes)
    }


def assert_budget_covers_profile(
    profile: Any,
    budget: dict[int, dict[int, int]],
) -> None:
    """Fail loudly when the budget map misses a (node, chiplet) the profile uses."""
    used = {(int(r.node_id), int(r.chiplet_id)) for r in profile.device_regions}
    covered = {
        (node_id, chiplet_id)
        for node_id, chiplets in budget.items()
        for chiplet_id in chiplets
    }
    missing = sorted(used - covered)
    if missing:
        raise RuntimeError(
            f"device budget map misses (node, chiplet) the profile uses: {missing} "
            f"(covered={sorted(covered)}); max_num_blocks would budget them 0 bytes."
        )


def per_chiplet_usage(profile: Any, num_blocks: int) -> dict[tuple[int, int], int]:
    """Predicted aligned bytes per (node, chiplet), mirroring `max_num_blocks`."""
    usage: Counter = Counter()
    for region in profile.device_regions:
        alignment = int(region.alignment)
        size = int(region.base_bytes) + num_blocks * int(region.bytes_per_block)
        if alignment > 1:
            # Integer align_up; float division would lose precision on the byte
            # counts involved here.
            size = -(-size // alignment) * alignment
        usage[(int(region.node_id), int(region.chiplet_id))] += size
    return dict(usage)


def format_profile_for_log(profile: Any, key: str = SOURCE_PROFILE_LOG_KEY) -> str:
    """One-line, machine-readable record of a KV-cache memory profile.

    Written out field by field rather than via ``repr()`` so the record stays
    identical across compiler builds.
    """
    regions = ", ".join(
        "PyRblnMemoryRegion("
        f"node_id={int(r.node_id)}, chiplet_id={int(r.chiplet_id)}, "
        f"base_bytes={int(r.base_bytes)}, "
        f"bytes_per_block={int(r.bytes_per_block)}, "
        f"alignment={int(r.alignment)})"
        for r in profile.device_regions
    )
    return (
        f"{key}=[{regions}] "
        f"host_base_bytes={int(profile.host_base_bytes)} "
        f"host_bytes_per_block={int(profile.host_bytes_per_block)}"
    )
