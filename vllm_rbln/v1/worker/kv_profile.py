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
"""Merging of `rebel` KV-cache memory profiles into a single joint profile.

`executor.kv_cache_memory_profile()` describes the memory one compiled artifact
allocates as, per device region, an affine function of the KV `num_blocks`
(`base_bytes + bytes_per_block * n`) plus a per-region alignment. A vLLM rank
holds several artifacts (prefill, one per decode batch bucket, a specialised MoE
decode graph, a spec-decode drafter, ...) and they *share* the KV tensors and the
weights while each keeps a small private footprint (command streams, IO
buffers). Solving each profile on its own and taking the ``min`` is therefore
optimistic: no single profile sees the other artifacts' private regions.

This module builds one duck-typed *joint* profile that
`rebel.kv_cache.max_num_blocks` can consume directly:

* **base regions** (``bytes_per_block == 0``) are summed across profiles, with
  shared regions counted once. Identity comes from ``region_id`` when the
  compiler exposes it; otherwise a ``(node_id, chiplet_id, base_bytes,
  bytes_per_block)`` multiset intersection is used as a fallback so the code
  also works against compiler builds that predate the identity fields.
* **growth regions** (``bytes_per_block > 0``) are counted **once per
  (node, chiplet)**. They are `RblnSlot` device allocations pointing at the very
  KV tensors vLLM handed the runtime, so summing them across profiles would
  double-count the same bytes. The profile with the largest per-(node, chiplet)
  growth total wins and contributes its regions verbatim, which keeps
  `max_num_blocks`' per-region ``align_up`` accounting intact (collapsing 36
  regions of 0.5 MiB/block into one of 18 MiB/block changes the answer).

Nothing here re-implements the sort/bisect in `rebel/kv_cache.py`; the merged
object is only a container.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "MergedMemoryRegion",
    "MergedKvCacheMemoryProfile",
    "merge_kv_cache_memory_profiles",
    "build_per_chiplet_budget",
    "assert_budget_covers_profile",
    "per_chiplet_usage",
]


@dataclass(frozen=True)
class MergedMemoryRegion:
    """One device allocation region of the merged profile.

    Field names mirror `rebel._C.PyRblnMemoryRegion`; `max_num_blocks` reads
    them by duck typing only.
    """

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
    dedup_strategy: str = "none"
    num_shared_base_regions: int = 0
    num_private_base_regions: int = 0
    num_growth_regions: int = 0


def _region_id(region: Any) -> str | None:
    """Return the compiler-provided region identity, or None when absent.

    ``region_id`` is added by the local compiler change that exposes region
    identity (`BuildMemoryProfile`). Released builds do not have it, so every
    caller must tolerate ``None``.
    """
    value = getattr(region, "region_id", None)
    if value is None:
        return None
    value = str(value)
    return value or None


def _is_shared_reference(region: Any) -> bool:
    return bool(getattr(region, "is_shared_reference", False))


def _fallback_key(region: Any) -> tuple[int, int, int, int]:
    """Dedup key. `alignment` is excluded: it is filled from a literal constant
    and carries no discriminating information.
    """
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


def _merge_base_by_identity(
    splits: list[_Split],
) -> tuple[list[MergedMemoryRegion], int, int]:
    """Dedup base regions by ``(node_id, chiplet_id, region_id)``.

    A region is counted once no matter how many artifacts report it. That also
    covers shared references whose owning artifact was never queried: the bytes
    are allocated on the device either way, so they must stay in the accounting.
    """
    merged: list[MergedMemoryRegion] = []
    seen: set[tuple[int, int, str]] = set()
    num_shared_refs = 0
    num_duplicates = 0
    for split in splits:
        for region in split.base:
            rid = _region_id(region)
            assert rid is not None
            key = (int(region.node_id), int(region.chiplet_id), rid)
            if _is_shared_reference(region):
                num_shared_refs += 1
            if key in seen:
                num_duplicates += 1
                continue
            seen.add(key)
            merged.append(_to_merged(region))
    logger.info(
        "[Dynamic KV] base regions deduped by region_id: kept=%d dropped=%d "
        "shared_reference_records=%d",
        len(merged),
        num_duplicates,
        num_shared_refs,
    )
    return merged, num_duplicates, num_shared_refs


def _merge_base_by_multiset(
    splits: list[_Split],
) -> tuple[list[MergedMemoryRegion], int, int]:
    """Shared == multiset intersection across every profile.

    Shared regions are counted once; what remains in each profile after removing
    the shared multiset is its private footprint and is summed.
    """
    counters = [Counter(_fallback_key(r) for r in s.base) for s in splits]
    shared = counters[0].copy()
    for counter in counters[1:]:
        shared &= counter

    prototypes: dict[tuple[int, int, int, int], Any] = {}
    for split in splits:
        for region in split.base:
            prototypes.setdefault(_fallback_key(region), region)

    merged: list[MergedMemoryRegion] = []
    for key, count in shared.items():
        region = _to_merged(prototypes[key])
        for _ in range(count):
            merged.append(region)
    num_shared = len(merged)

    for counter in counters:
        for key, count in (counter - shared).items():
            region = _to_merged(prototypes[key])
            for _ in range(count):
                merged.append(region)
    num_private = len(merged) - num_shared

    logger.info(
        "[Dynamic KV] base regions deduped by (node, chiplet, base_bytes, "
        "bytes_per_block) multiset intersection: shared=%d private=%d",
        num_shared,
        num_private,
    )
    return merged, num_shared, num_private


def _merge_growth(splits: list[_Split]) -> list[MergedMemoryRegion]:
    """Count per-block growth once per (node, chiplet).

    Regions are kept individually rather than collapsed per chiplet because
    `max_num_blocks` aligns every region on its own.
    """
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
                "[Dynamic KV] per-block growth disagrees across compiled "
                "artifacts at (node=%d, chiplet=%d): %s. Taking the maximum; "
                "the artifacts may not share one KV layout.",
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
    """Merge per-artifact KV-cache memory profiles into one joint profile.

    Args:
        profiles: `PyRblnKvCacheMemoryProfile`-shaped objects, one per compiled
            artifact that consumes the KV cache.

    Raises:
        ValueError: when `profiles` is empty.
    """
    if not profiles:
        raise ValueError("merge_kv_cache_memory_profiles() needs >= 1 profile")

    splits = [_split_regions(p) for p in profiles]

    have_identity = all(
        _region_id(region) is not None
        for split in splits
        for region in split.base + split.growth
    )
    if have_identity:
        base, num_duplicates, _num_shared_refs = _merge_base_by_identity(splits)
        strategy = "region_id"
        # Regions reported by more than one artifact are the shared ones; what is
        # left is private to a single artifact.
        num_shared = num_duplicates
        num_private = len(base) - num_duplicates
    else:
        base, num_shared, num_private = _merge_base_by_multiset(splits)
        strategy = "multiset"

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
        dedup_strategy=strategy,
        num_shared_base_regions=num_shared,
        num_private_base_regions=num_private,
        num_growth_regions=len(growth),
    )
    logger.info(
        "[Dynamic KV] merged %d compiled profile(s) via %s: regions=%d "
        "(base=%d growth=%d) host_base_bytes=%d host_bytes_per_block=%d",
        len(profiles),
        strategy,
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
    """Return the `{node_id: {chiplet_id: bytes}}` map `max_num_blocks` wants.

    Every (node, chiplet) combination is filled in: a unit the profile uses but
    the mapping omits is silently budgeted 0 bytes, which turns a plumbing
    mistake into a plausible-looking answer of 0 blocks.
    """
    if num_nodes <= 0 or num_chiplets <= 0:
        raise ValueError(
            f"invalid device topology: num_nodes={num_nodes} "
            f"num_chiplets={num_chiplets}"
        )
    return {
        node_id: {
            chiplet_id: budget_bytes_per_chiplet
            for chiplet_id in range(num_chiplets)
        }
        for node_id in range(num_nodes)
    }


def assert_budget_covers_profile(
    profile: Any,
    budget: dict[int, dict[int, int]],
) -> None:
    """Fail loudly when the budget map misses a (node, chiplet) the profile uses.

    Without this `max_num_blocks` charges the missing unit against a 0-byte
    budget and returns 0 blocks with no error.
    """
    used = {(int(r.node_id), int(r.chiplet_id)) for r in profile.device_regions}
    covered = {
        (node_id, chiplet_id)
        for node_id, chiplets in budget.items()
        for chiplet_id in chiplets
    }
    missing = sorted(used - covered)
    if missing:
        raise RuntimeError(
            "device budget map does not cover every (node_id, chiplet_id) the "
            f"compiled profile allocates on: missing={missing} "
            f"covered={sorted(covered)}. rebel.kv_cache.max_num_blocks would "
            "silently budget 0 bytes for those units."
        )


def per_chiplet_usage(profile: Any, num_blocks: int) -> dict[tuple[int, int], int]:
    """Predicted aligned bytes per (node, chiplet) at `num_blocks`.

    Mirrors the accounting inside `rebel.kv_cache.max_num_blocks` so the worker
    can compare its prediction against the runtime's own allocation report.
    """
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
