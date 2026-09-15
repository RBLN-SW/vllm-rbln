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
"""Placement-based sizing of the KV cache (`VLLM_RBLN_USE_DYNAMIC_KV_CACHE`):
the state machine; `kv_placement` holds the arithmetic."""

import copy
import gc
import os
import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig

import vllm_rbln.envs as envs
from vllm_rbln.compilation.backends import set_compile_stage
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.utils import sub_block_size_in_use
from vllm_rbln.v1.worker.kv_placement import (
    ChipletMemory,
    KvGrowth,
    Unit,
    UnitFit,
    format_fits,
    format_placements,
    kv_growth,
    max_num_blocks,
    select_kv_input_groups,
    snapshot_from_allocator,
    snapshot_from_driver,
)
from vllm_rbln.v1.worker.utils import (
    compile_and_warmup_skip_reason,
    estimate_available_memory,
    minimum_kv_blocks,
    rescale_kv_cache_config,
)

try:
    import torch.rbln  # noqa: F401

    has_torch_rbln = True
except ImportError:
    has_torch_rbln = False

logger = init_logger(__name__)


class DynamicKvMode(Enum):
    DISABLED = "disabled"
    INERT = "inert"
    PINNED = "pinned"
    DRY_RUN = "dry_run"
    ACTIVE = "active"


def resolve_mode(
    *,
    use_dynamic_kv: bool,
    dry_run: bool,
    num_gpu_blocks_override: int | None,
    compile_skip_reason: str | None,
) -> tuple[DynamicKvMode, str | None]:
    """The mode and its reason. Every input is static config or env, so the
    decision is final at init time."""
    if not use_dynamic_kv:
        return DynamicKvMode.DISABLED, None
    if compile_skip_reason is not None:
        return DynamicKvMode.INERT, compile_skip_reason
    if dry_run:
        return DynamicKvMode.DRY_RUN, None
    if num_gpu_blocks_override is not None:
        return (
            DynamicKvMode.PINNED,
            f"--num-gpu-blocks-override={num_gpu_blocks_override}",
        )
    return DynamicKvMode.ACTIVE, None


def refuse(mode: DynamicKvMode, message: str) -> None:
    """Raise, or report and continue in a dry run: the dry run observes, it does
    not decide whether a configuration boots."""
    if mode is DynamicKvMode.DRY_RUN:
        logger.warning("[Dynamic KV] dry run: %s", message)
        return
    raise RuntimeError(message)


@dataclass(frozen=True)
class KvSizing:
    """One rank's count and the fit behind it; `if_resident` only when the
    compile-time cache could not be released first."""

    num_blocks: int
    fits: dict[Unit, UnitFit]
    hint_blocks: int
    growth: KvGrowth
    if_resident: int | None = None


# Trace hint for the mark_dynamic'd KV dim, not a capacity; dynamo specializes
# a smaller dim away.
COMPILE_KV_CACHE_NUM_BLOCKS = 4
# Allocator-snapshot fallback only: runtime allocations the allocator never sees.
DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES = 48 * 1024 * 1024
# Per chiplet, for the copy command streams sub-block prefix caching uploads.
DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES = 64 * 1024 * 1024


def kv_cache_config_at(cfg: KVCacheConfig, num_blocks: int) -> KVCacheConfig:
    """A copy of `cfg` retargeted at `num_blocks` (tensor sizes scale with it)."""
    scaled = copy.copy(cfg)
    scaled.kv_cache_tensors = copy.deepcopy(cfg.kv_cache_tensors)
    rescale_kv_cache_config(scaled, num_blocks)
    return scaled


def empty_rbln_device_caches() -> bool:
    """Return every *free* block the rbln caching allocator holds to the driver."""
    # The allocator otherwise releases cached blocks only after a failed
    # allocation, so freed bytes keep counting in `dram_used`. Never raises.
    if not has_torch_rbln:
        return False
    try:
        # NOTE(RBLN): is_available() raises on a malformed RBLN_* config.
        if not torch.rbln.is_available():
            return False
        device_count = torch.rbln.device_count()
    except Exception as exc:
        logger.warning(
            "could not query the rbln devices to empty their allocator caches: "
            "%s. Freed KV bytes stay reserved and keep counting per chiplet.",
            exc,
        )
        return False

    for index in range(device_count):
        try:
            torch.rbln.empty_cache(index)
        except Exception as exc:
            logger.warning(
                "torch.rbln.empty_cache(%d) failed: %s. Freed KV blocks stay "
                "reserved and keep counting per chiplet.",
                index,
                exc,
            )
    return device_count > 0


class DynamicKvSizer:
    """The worker's dynamic-KV state machine: shrink -> capture -> snapshot ->
    size -> release -> reallocate -> materialize -> check."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model_runner: Any,
        foreign_dram_used_bytes: int,
    ) -> None:
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model_runner = model_runner
        self.device: torch.device = vllm_config.device_config.device
        self.rank: int = vllm_config.parallel_config.rank
        self.mode, self.mode_reason = resolve_mode(
            use_dynamic_kv=envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE,
            dry_run=envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN,
            num_gpu_blocks_override=self.cache_config.num_gpu_blocks_override,
            compile_skip_reason=compile_and_warmup_skip_reason(vllm_config),
        )
        # The count vLLM sized, held while the cache is shrunk for the compile.
        self.kv_blocks_before_shrink: int | None = None
        self.programs: list[Any] = []
        self.expected_used: dict[Unit, int] = {}
        # Other tenants' card DRAM at init; the allocator-snapshot fallback only.
        self.foreign_dram_used_bytes = foreign_dram_used_bytes
        device_env = current_platform.device_control_env_var
        logger.debug(
            "foreign device DRAM at worker init: %d bytes (%s=%s)",
            foreign_dram_used_bytes,
            device_env,
            os.environ.get(device_env, ""),
        )

    @property
    def compiled_with_shrunk_cache(self) -> bool:
        return self.kv_blocks_before_shrink is not None

    def record_programs(self, programs: list[Any]) -> None:
        self.programs.extend(programs)
        logger.info(
            "[Dynamic KV] captured %d compiled program(s) during warm-up.",
            len(programs),
        )

    def pre_compile_estimate(self, estimate_kwargs: dict[str, Any]) -> int:
        """The bytes vllm sizes the compile-time cache from: the estimate fed
        the per-chiplet snapshot on a real device, floored at one request when
        the shrink makes that estimate a placeholder."""
        if self.mode is DynamicKvMode.DISABLED:
            return estimate_available_memory(**estimate_kwargs)
        # The exact DRAM figure and the driver's capacity are the resize's, not
        # this estimate's: every other mode serves it as vllm sized it.
        estimate_kwargs = {
            **estimate_kwargs,
            "exact_dram": self.mode is DynamicKvMode.ACTIVE,
        }
        if not torch.rbln.is_dummy_device():
            snapshot, source = self.memory_snapshot(self.device)
            if self.mode is DynamicKvMode.DRY_RUN:
                measured = estimate_available_memory(
                    **{**estimate_kwargs, "exact_dram": True},
                    chiplet_memory=snapshot,
                )
                logger.info(
                    "[Dynamic KV] dry run: the %s memory snapshot of %s would put the "
                    "pre-compile estimate at %.2f GiB; keeping the whole-card formula.",
                    source,
                    self.device,
                    measured / 1024**3,
                )
            else:
                estimate_kwargs = {**estimate_kwargs, "chiplet_memory": snapshot}
                logger.info(
                    "[Dynamic KV] pre-compile estimate from the %s memory snapshot of "
                    "%s.",
                    source,
                    self.device,
                )

        estimate = estimate_available_memory(**estimate_kwargs)
        one_request = sum(
            spec.max_memory_usage_bytes(self.vllm_config)
            for spec in self.model_runner.get_kv_cache_spec().values()
        )
        if self.mode is DynamicKvMode.ACTIVE and estimate < one_request:
            # vllm refuses a pool below one request against this estimate; the
            # real count is sized from the device after warm-up. Only under the
            # shrink: every other mode serves this estimate, so raising it here
            # would change the pool instead of reporting on it.
            logger.warning(
                "[Dynamic KV] the pre-compile estimate (%.2f GiB) is short of one "
                "max-length request (%.2f GiB); raising it to that so the compile "
                "proceeds. The pool is sized from the device after warm-up and "
                "refused there if one request does not fit.",
                estimate / 1024**3,
                one_request / 1024**3,
            )
            estimate = one_request
        return estimate

    def shrink_for_compile(self, kv_cache_config: KVCacheConfig) -> KVCacheConfig:
        """A small-KV-cache copy of the config, or it unchanged; its
        `num_blocks` is the hint warm-up traces the dynamic dim with."""
        if self.mode is DynamicKvMode.DISABLED:
            return kv_cache_config
        if self.mode is DynamicKvMode.INERT:
            # Nothing compiles, so a shrink would set the latch with nothing to resize.
            logger.warning(
                "[Dynamic KV] compile/warm-up is skipped (%s), so the cache stays "
                "at the estimated %d blocks and this feature does nothing for "
                "this run.",
                self.mode_reason,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        if self.mode is DynamicKvMode.DRY_RUN:
            logger.warning(
                "[Dynamic KV] dry run: compiling at the %d blocks vllm sized; the "
                "count this feature would pick is only logged after warm-up.",
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        if self.mode is DynamicKvMode.PINNED:
            logger.warning(
                "[Dynamic KV] %s pins the count; no shrink and no resize. "
                "Compiling at %d blocks.",
                self.mode_reason,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        compile_num_blocks = COMPILE_KV_CACHE_NUM_BLOCKS
        if compile_num_blocks >= kv_cache_config.num_blocks:
            # No shrink means no resize: the run would silently serve the
            # pre-compile estimate. Refuse instead.
            raise RuntimeError(
                f"the {compile_num_blocks}-block compile hint is not below the "
                f"{kv_cache_config.num_blocks} blocks vllm estimated, so there is "
                "nothing to shrink and no resize would run. See "
                "docs/dynamic_kv_cache.md."
            )

        shrunk = kv_cache_config_at(kv_cache_config, compile_num_blocks)
        self.kv_blocks_before_shrink = kv_cache_config.num_blocks
        logger.info(
            "[Dynamic KV] compiling with %d KV blocks instead of %d; resized after "
            "warm-up from the compiled profile.",
            compile_num_blocks,
            kv_cache_config.num_blocks,
        )
        return shrunk

    def assert_attention_layout(self) -> None:
        """Every attention layer must dispatch to a paged causal or sliding-window
        naive kernel (`is_causal`, not `is_normal`). Runs here, not in platform
        validation: the layers exist only after the model build."""
        if self.mode is DynamicKvMode.DISABLED:
            return
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        offenders: list[str] = []
        for layer_name, layer in attn_layers.items():
            impl = layer.impl
            is_causal = getattr(impl, "is_causal", None)
            is_normal = getattr(impl, "is_normal", None)
            if is_causal is not True or is_normal is not False:
                offenders.append(
                    f"{layer_name}(is_causal={is_causal}, is_normal={is_normal})"
                )
        if offenders:
            refuse(
                self.mode,
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires every layer to dispatch "
                "to a paged causal or sliding-window naive kernel. Offending: "
                + ", ".join(offenders[:8])
                + (f" (+{len(offenders) - 8} more)" if len(offenders) > 8 else ""),
            )

    def assert_cache_layout(self) -> None:
        """The KV bindings must satisfy the compiler's dynamic-input rules; reads
        state `initialize_kv_cache` fills."""
        if self.mode is DynamicKvMode.DISABLED:
            return
        mr = self.model_runner

        # The compiler admits a dynamic input through view ops into several
        # attention calls, but not the same view into two calls.
        if mr.shared_kv_cache_layers:
            refuse(
                self.mode,
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support cross-layer KV "
                f"sharing, but {len(mr.shared_kv_cache_layers)} layer(s) reuse "
                "another layer's KV cache.",
            )

    def capture_programs(self):
        """Scope that records the programs warm-up builds, when the flag is on."""
        if self.mode is DynamicKvMode.DISABLED:
            return nullcontext(None)
        if not has_torch_rbln:
            message = (
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE needs torch_rbln's "
                "capture_programs(); torch.rbln is not importable."
            )
            if self.mode is DynamicKvMode.DRY_RUN:
                # Nothing to capture means nothing to report, which the sizing
                # step says; it must not stop a run the dry run cannot change.
                logger.warning("[Dynamic KV] dry run: %s", message)
                return nullcontext(None)
            raise RuntimeError(message)
        return torch.rbln.capture_programs()

    def collect_runtimes(self) -> list[Any]:
        """Every rbln runtime warm-up built, deduplicated across programs."""
        runtimes: list[Any] = []
        seen: set[int] = set()
        for program in self.programs:
            runtime = program.runtime
            if id(runtime) in seen:
                continue
            seen.add(id(runtime))
            runtimes.append(runtime)
        return runtimes

    def memory_snapshot(
        self, device: torch.device
    ) -> tuple[dict[Unit, ChipletMemory], str]:
        """Per-(node, chiplet) `(total, used)` and its source: the driver, or
        this process's allocator plus the reserve and the foreign usage."""
        query = getattr(torch.rbln, "mem_get_info_per_chiplet", None)
        if query is not None:
            try:
                return snapshot_from_driver(query(device)), "driver"
            except RuntimeError as exc:
                logger.warning(
                    "[Dynamic KV] mem_get_info_per_chiplet(%s) is unavailable (%s); "
                    "sizing from this process's allocator instead.",
                    device,
                    exc,
                )
        else:
            logger.warning(
                "[Dynamic KV] this torch_rbln has no mem_get_info_per_chiplet(); "
                "sizing from this process's allocator instead."
            )
        # Cached-but-free blocks would otherwise count as used.
        torch.rbln.empty_cache(device)
        memory_per_chiplet = int(
            torch.rbln.get_device_properties(device).memory_per_chiplet
        )
        snapshot = snapshot_from_allocator(
            torch.rbln.memory_stats_per_chiplet(device),
            memory_per_chiplet=memory_per_chiplet,
            foreign_card_used_bytes=self.foreign_dram_used_bytes,
            reserve_bytes=DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES,
        )
        logger.info(
            "[Dynamic KV] allocator snapshot: memory_per_chiplet=%d foreign_used=%d "
            "reserve=%d units=%d",
            memory_per_chiplet,
            self.foreign_dram_used_bytes,
            DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES,
            len(snapshot),
        )
        return snapshot, "allocator"

    def copy_stream_reserve_bytes(self) -> int:
        """Per-chiplet bytes to keep out of the KV budget when the scheduler
        will run sub-block prefix caching."""
        in_use = sub_block_size_in_use(
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
            sub_block_cache=self.vllm_config.additional_config.enable_sub_block_cache,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            kv_cache_config=self.model_runner.kv_cache_config,
        )
        return DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES if in_use is not None else 0

    def compute_num_blocks(self) -> int | None:
        """How many KV blocks fit this device, from the placement and a memory
        snapshot. Reallocates nothing; None means the path is not in play."""
        if self.mode is DynamicKvMode.PINNED:
            logger.info(
                "[Dynamic KV] %s is set; leaving the KV cache alone.",
                self.mode_reason,
            )
            return None
        dry_run = self.mode is DynamicKvMode.DRY_RUN
        if not dry_run and self.kv_blocks_before_shrink is None:
            # The branch that cancelled the shrink already logged why.
            logger.warning(
                "[Dynamic KV] the cache was not shrunk, so no placement is queried "
                "and the count stays at the %d blocks vllm estimated.",
                self.model_runner.kv_cache_config.num_blocks,
            )
            return None
        if dry_run and not self.programs:
            logger.warning(
                "[Dynamic KV] dry run: no compiled program was captured, so there is "
                "nothing to report."
            )
            return None

        if torch.rbln.is_dummy_device():
            # Compile-only: no device memory to size against.
            logger.warning(
                "[Dynamic KV] RBLN_DUMMY_DEVICE is set, so there is no device to "
                "measure; keeping the %d blocks vllm estimated for this compile-only "
                "run.",
                self.kv_blocks_before_shrink
                or self.model_runner.kv_cache_config.num_blocks,
            )
            return None

        if dry_run:
            try:
                sizing = self._propose_kv_size()
            except RuntimeError as exc:
                logger.warning(
                    "[Dynamic KV] dry run: the count could not be computed (%s); "
                    "nothing is resized.",
                    exc,
                )
                return None
            self.log_dry_run(sizing)
            return None
        return self._size_kv_and_release().num_blocks

    def _kv_growth_from_programs(self) -> tuple[KvGrowth, int, torch.device]:
        """The KV growth the captured programs' placements imply, the hint they
        were traced with, and the device they run on."""
        programs = list(self.programs)
        groups = select_kv_input_groups(programs)
        hint_blocks = self.model_runner.kv_cache_config.num_blocks
        for group_specs, group_program in groups:
            logger.info(
                "[Dynamic KV] KV placement from %s (%d program(s) captured, %d KV "
                "input set(s), %d dynamic input(s) in this set, hint=%d blocks): %s",
                group_program.name or "an unnamed program",
                len(programs),
                len(groups),
                len(group_specs),
                hint_blocks,
                format_placements(group_specs),
            )
        specs = [spec for group_specs, _ in groups for spec in group_specs]
        # Groups are keyed by the whole input set, so two programs binding
        # overlapping but unequal sets form two groups and the slope charges the
        # shared tensors twice. InputSpec carries no tensor identity to dedupe
        # by, so log what the slope is summed over next to what vllm allocated.
        logger.info(
            "[Dynamic KV] the slope is summed over %d KV input(s) in %d set(s); "
            "vllm allocated %d KV cache tensor(s) for %d layer(s).",
            len(specs),
            len(groups),
            len(self.model_runner.kv_cache_config.kv_cache_tensors),
            sum(
                len(t.shared_by)
                for t in self.model_runner.kv_cache_config.kv_cache_tensors
            ),
        )
        growth = kv_growth(specs, hint_blocks)
        program = groups[0][1]
        device = program.device if program.device is not None else self.device
        return growth, hint_blocks, device

    def _size_kv_from_snapshot(
        self,
        growth: KvGrowth,
        device: torch.device,
        *,
        kv_resident: Mapping[Unit, int],
    ) -> tuple[int, dict[Unit, UnitFit], dict[Unit, ChipletMemory], int]:
        """Snapshot the device and size against it; `kv_resident` is what the
        snapshot still holds of the KV cache."""
        snapshot, source = self.memory_snapshot(device)
        logger.info(
            "[Dynamic KV] %s memory snapshot of %s: %s",
            source,
            device,
            {f"{n}:{c}": (m.total, m.used) for (n, c), m in sorted(snapshot.items())},
        )

        reserve_bytes = self.copy_stream_reserve_bytes()
        if reserve_bytes:
            logger.info(
                "[Dynamic KV] sub-block prefix caching is on; reserving %d bytes per "
                "chiplet for its copy command streams.",
                reserve_bytes,
            )

        gmu = self.cache_config.gpu_memory_utilization
        num_blocks, fits = max_num_blocks(
            snapshot,
            growth,
            gpu_memory_utilization=gmu,
            kv_resident=kv_resident,
            reserve_bytes=reserve_bytes,
        )
        logger.info(
            "[Dynamic KV] rank %d: computed_num_blocks=%d gpu_memory_utilization=%.3f "
            "per unit: %s",
            self.rank,
            num_blocks,
            gmu,
            format_fits(fits),
        )
        if num_blocks <= 0:
            raise RuntimeError(
                "[Dynamic KV] no KV block fits: on some chiplet the non-KV base "
                "already exceeds the budget. Per unit: "
                f"{format_fits(fits)}. Raise --gpu-memory-utilization or give the "
                "model more devices."
            )
        predicted = growth.allocated_at(num_blocks)
        logger.info(
            "[Dynamic KV] predicted KV bytes per (node, chiplet) at %d blocks: %s "
            "total=%d",
            num_blocks,
            {f"{n}:{c}": b for (n, c), b in sorted(predicted.items())},
            sum(predicted.values()),
        )
        self.expected_used = {
            unit: fits[unit].base + predicted[unit] for unit in predicted
        }
        return num_blocks, fits, snapshot, reserve_bytes

    def _size_kv_and_release(self) -> KvSizing:
        """Release the compile-time cache, then size from what the runtime
        actually handed back; no KV cache is bound until `apply_num_blocks`."""
        growth, hint_blocks, device = self._kv_growth_from_programs()
        self.release_kv_cache_tensors(self.model_runner.kv_cache_config)
        num_blocks, fits, _, _ = self._size_kv_from_snapshot(
            growth, device, kv_resident={}
        )
        return KvSizing(num_blocks, fits, hint_blocks, growth)

    def _propose_kv_size(self) -> KvSizing:
        """Size without touching the cache, reporting the count under both
        readings of the resident compile-time cache."""
        growth, hint_blocks, device = self._kv_growth_from_programs()
        num_blocks, fits, snapshot, reserve_bytes = self._size_kv_from_snapshot(
            growth, device, kv_resident=growth.allocated_at(hint_blocks)
        )
        if_resident, _ = max_num_blocks(
            snapshot,
            growth,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            kv_resident={},
            reserve_bytes=reserve_bytes,
        )
        return KvSizing(num_blocks, fits, hint_blocks, growth, if_resident)

    def log_fit_check(self, num_blocks: int) -> None:
        """Measured `used` against the prediction, once the resized cache is
        physically allocated."""
        snapshot, source = self.memory_snapshot(self.device)
        parts = []
        for (node, chiplet), expected in sorted(self.expected_used.items()):
            memory = snapshot.get((node, chiplet))
            if memory is None:
                parts.append(f"{node}:{chiplet}(expected={expected} measured=?)")
                continue
            budget = int(memory.total * self.cache_config.gpu_memory_utilization)
            parts.append(
                f"{node}:{chiplet}(expected={expected} measured={memory.used} "
                f"diff={memory.used - expected:+d} "
                f"budget_left={budget - memory.used:+d})"
            )
        logger.info(
            "[Dynamic KV] fit check after reallocating to %d blocks, %s snapshot: %s",
            num_blocks,
            source,
            " ".join(parts),
        )

    def log_dry_run(self, sizing: KvSizing) -> None:
        """The dry-run report: today's count against each chiplet's budget and
        the count this feature would pick."""
        num_blocks, fits, current, growth, if_resident = (
            sizing.num_blocks,
            sizing.fits,
            sizing.hint_blocks,
            sizing.growth,
            sizing.if_resident,
        )
        now = growth.allocated_at(current)
        proposed = growth.allocated_at(num_blocks)

        def pct(used: int, of: int) -> float:
            # A dry run never fails the run, not even on a malformed snapshot.
            return 100.0 * used / of if of else float("nan")

        per_unit = []
        for (node, chiplet), fit in sorted(fits.items()):
            unit = (node, chiplet)
            headroom = fit.budget - fit.base - now[unit]
            at_proposed = fit.base + proposed[unit]
            per_unit.append(
                f"{node}:{chiplet}(kv_now={now[unit]} base={fit.base} "
                f"budget={fit.budget} headroom={headroom} = "
                f"{headroom // fit.per_block:+d} blocks, now "
                f"{pct(fit.base + now[unit], fit.budget):.1f}% of budget; "
                f"at {num_blocks} blocks: used={at_proposed} "
                f"budget_left={fit.budget - at_proposed} "
                f"total_left={fit.total - at_proposed} = "
                f"{pct(at_proposed, fit.budget):.1f}% of budget, "
                f"{pct(at_proposed, fit.total):.1f}% of DRAM)"
            )
        minimum = minimum_kv_blocks(self.vllm_config, self.model_runner.kv_cache_config)
        logger.warning(
            "[Dynamic KV] dry run: vllm sized %d blocks, this feature would set %d "
            "(%+d) if the runtime hands the current cache back, %s if it stays "
            "resident; the pool needs %d (one request %d, decode batch %d, +1 null "
            "block), so the count %s. Per (node, chiplet): %s. Nothing is resized.",
            current,
            num_blocks,
            num_blocks - current,
            if_resident,
            minimum.needed,
            minimum.one_request,
            minimum.decode_batch,
            "would be accepted" if num_blocks >= minimum.needed else "would be REFUSED",
            " ".join(per_unit),
        )

    def apply_num_blocks(self, n: int | None) -> int | None:
        """Resize the KV cache to the count the engine settled on; None puts the
        pre-shrink count back."""
        before_shrink = self.kv_blocks_before_shrink
        target = before_shrink if n is None else n
        if target is None:
            return None
        self.kv_blocks_before_shrink = None

        current = self.model_runner.kv_cache_config.num_blocks
        if target == current and self.model_runner.kv_caches:
            logger.info(
                "[Dynamic KV] KV cache already holds %d blocks; nothing to reallocate.",
                target,
            )
            return target

        if n is None:
            if torch.rbln.is_dummy_device():
                # The dummy UMD still enforces its limit; nothing runs after warm-up.
                logger.info(
                    "[Dynamic KV] compile-only run: keeping the %d-block compile "
                    "cache instead of the %d blocks vllm estimated.",
                    current,
                    target,
                )
                return current
            logger.warning(
                "[Dynamic KV] restoring KV cache to the %d blocks vllm sized it "
                "with (compiled with %d).",
                target,
                current,
            )
        self.reallocate(target)
        self.materialize()
        if n is not None and self.expected_used:
            self.log_fit_check(target)
        return target

    def materialize(self) -> None:
        """One decode step so the pool's physical allocation lands at boot, not
        on the first request."""
        num_reqs = min(self.model_runner.bucketing_manager.decode_batch_buckets)
        with set_compile_stage("warmup"), self.model_runner.offload_context():
            self.model_runner._dummy_run(num_reqs, 1, False)

    def release_kv_cache_tensors(self, old_cfg: KVCacheConfig) -> None:
        """Drop every reference to the outgoing KV cache and free its device DRAM
        before the replacement is allocated, or the peak is base + old + new."""
        mr = self.model_runner

        kv_device_types = {kv_cache.device.type for kv_cache in mr.kv_caches}
        was_device_resident = bool(kv_device_types - {"meta", "cpu"})

        # The rebind reassigns all three from one ordered name list.
        mr.kv_caches = []
        mr.kv_cache_bases = []
        mr.kv_cache_names = []

        # Each layer's view is parked on its Attention module; the next bind
        # overwrites it only after the new tensors exist.
        forward_context = mr.compilation_config.static_forward_context
        unbound = 0
        for layer_name in dict.fromkeys(
            name for t in old_cfg.kv_cache_tensors for name in t.shared_by
        ):
            layer = forward_context.get(layer_name)
            if layer is None:
                logger.warning(
                    "[Dynamic KV] layer %s has a KV cache tensor but no entry in "
                    "the static forward context; its binding cannot be dropped "
                    "before the reallocation.",
                    layer_name,
                )
                continue
            layer.kv_cache = None
            unbound += 1

        # A reference cycle would defer the free past the new allocation.
        gc.collect()

        released = empty_rbln_device_caches()
        logical_bytes = sum(t.size for t in old_cfg.kv_cache_tensors)
        logger.info(
            "[Dynamic KV] released the outgoing %d-block KV cache: "
            "outgoing_kv_logical_bytes=%d unbound_layers=%d kv_device_types=%s "
            "allocator_cache_emptied=%s device_resident=%s allocator_after=%s",
            old_cfg.num_blocks,
            logical_bytes,
            unbound,
            sorted(kv_device_types),
            released,
            was_device_resident,
            self.allocator_state_per_chiplet(),
        )

    def allocator_state_per_chiplet(self) -> str:
        """`allocated/reserved` per chiplet from this process's allocator, or
        why it could not be read."""
        stats_fn = getattr(torch.rbln, "memory_stats_per_chiplet", None)
        if stats_fn is None or torch.rbln.is_dummy_device():
            return "unavailable"
        try:
            stats = stats_fn(self.device)
        except RuntimeError as exc:
            return f"unavailable ({exc})"
        per_unit: dict[str, list[str]] = {}
        for key, value in sorted(stats.items()):
            match = re.match(
                r"^npu\.(\d+)\.chiplet\.(\d+)\.(allocated|reserved)\.current$", key
            )
            if match:
                per_unit.setdefault(f"{match.group(1)}:{match.group(2)}", []).append(
                    f"{match.group(3)}={value}"
                )
        return " ".join(f"{u}({' '.join(v)})" for u, v in per_unit.items()) or repr(
            stats
        )

    def reallocate(self, new_num_blocks: int) -> None:
        """Rebuild only the KV cache tensors at `new_num_blocks`; the dim is
        `mark_dynamic`'d, so nothing recompiles."""
        # Not `initialize_kv_cache()`: `initialize_attn_backend` asserts the
        # attn groups are empty, and nothing there depends on num_blocks.
        mr = self.model_runner
        old_cfg = mr.kv_cache_config
        old_num_blocks = old_cfg.num_blocks

        new_cfg = kv_cache_config_at(old_cfg, new_num_blocks)
        self.cache_config.num_gpu_blocks = new_num_blocks
        self.cache_config.num_cpu_blocks = new_num_blocks

        logger.info(
            "[Dynamic KV] reallocating KV cache: %d -> %d blocks",
            old_num_blocks,
            new_num_blocks,
        )
        mr.kv_cache_config = new_cfg
        # Release before allocating (see `release_kv_cache_tensors`); the
        # sizing has usually done it already.
        if mr.kv_caches:
            self.release_kv_cache_tensors(old_cfg)
        mr.initialize_kv_cache_tensors(new_cfg, mr._kernel_block_sizes)

        # Warm-up latched the adaptive buffer sizes at the old num_blocks;
        # without this the next forward raises "variable dim changed".
        runtimes = self.collect_runtimes()
        for runtime in runtimes:
            runtime.reset_adaptive_buffers()
        logger.info(
            "[Dynamic KV] reset_adaptive_buffers() on %d runtime(s).",
            len(runtimes),
        )
