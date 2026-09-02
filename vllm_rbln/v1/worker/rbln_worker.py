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
"""A RBLN worker class."""

import copy
import gc
import os
import time
from types import NoneType
from typing import TYPE_CHECKING, Any

import numba
import torch

try:
    import torch.rbln

    has_torch_rbln = True
except ImportError:
    has_torch_rbln = False

import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.exc import BackendCompilerFailed
from vllm.config import (
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_rbln.envs as envs
from vllm_rbln.compilation.backends import set_compile_stage
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    finalize_kv_cache_registrations,
)
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.kv_profile import (
    MERGED_PROFILE_LOG_KEY,
    assert_budget_covers_profile,
    build_per_chiplet_budget,
    format_profile_for_log,
    merge_kv_cache_memory_profiles,
    per_chiplet_usage,
)
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.utils import (
    estimate_available_memory,
    estimate_model_kernel_size,
    get_rbln_planned_affinity_cpu_count,
    read_rbln_card_dram_total_bytes,
    read_rbln_card_dram_used_bytes,
    rescale_kv_cache_config,
    set_cpu_affinity,
    set_omp_num_threads,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

# Trace hint for the mark_dynamic'd KV dim, not a capacity: dynamo specializes
# a smaller dim away, and below this artifacts abort on device at larger
# max_num_batched_tokens.
COMPILE_KV_CACHE_NUM_BLOCKS = 4
# Held back from every chiplet's budget for device memory no profile region
# describes.
DYNAMIC_KV_UNPROFILED_RESERVE_BYTES = 48 * 1024 * 1024


def _kv_cache_config_at(cfg: KVCacheConfig, num_blocks: int) -> KVCacheConfig:
    """A copy of `cfg` retargeted at `num_blocks` (tensor sizes scale with it)."""
    scaled = copy.copy(cfg)
    scaled.kv_cache_tensors = copy.deepcopy(cfg.kv_cache_tensors)
    rescale_kv_cache_config(scaled, num_blocks)
    return scaled


def empty_rbln_device_caches() -> bool:
    """Return every *free* block the rbln caching allocator holds to the driver."""
    # NOTE(RBLN): the allocator otherwise releases cached blocks only as a retry
    # after an allocation fails, so freed bytes keep counting in sysfs
    # `dram_used`. Never raises: this runs during start-up.
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


class RBLNWorker(WorkerBase):
    """A worker class that executes the model on RBLN NPUs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self._init_device_env()

        self._rbln_host_threads_before_compile_ready = False
        self._rbln_cpu_affinity_applied = False

        # num_blocks vLLM sized the cache with, stashed while it is shrunk for
        # the compile. None when no shrink is pending.
        self._kv_blocks_before_shrink: int | None = None
        # Other tenants' device DRAM, sampled before this worker allocates
        # anything. Card-scope; only `_dynamic_kv_chiplet_budget` reads it.
        self._foreign_dram_used_bytes = 0
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            self._foreign_dram_used_bytes = read_rbln_card_dram_used_bytes()
            logger.debug(
                "foreign device DRAM at worker init: %d bytes (RBLN_DEVICES=%s)",
                self._foreign_dram_used_bytes,
                os.environ.get("RBLN_DEVICES", ""),
            )

        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config

        if self.profiler_config.profiler not in ("torch", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.parallel_config.disable_custom_all_reduce = True

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is not supported on RBLN, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Sleep mode is not supported on RBLN, ignore it.")
        pass

    def _init_device_env(self) -> None:
        world_size = self.parallel_config.world_size // envs.VLLM_RBLN_NUM_RAY_NODES
        env_var = current_platform.device_control_env_var

        num_devices = envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK
        total_device_count = world_size * num_devices

        if env_var not in os.environ:
            dev_begin = total_device_count * self.parallel_config.data_parallel_rank
            dev_end = dev_begin + total_device_count
            device_ids = [str(i) for i in range(dev_begin, dev_end)]
            start_idx = self.local_rank * num_devices
            end_idx = start_idx + num_devices
            selected_devices = ",".join(device_ids[start_idx:end_idx])
        else:
            # vLLM 0.24 stopped narrowing the device-control env var per DP rank
            # and puts the mapping on the config instead, so under DP the env var
            # now holds the whole deployment's list (vllm/v1/engine/utils.py,
            # set_assigned_physical_gpu_ids_for_dp_rank). getattr: older vLLM has
            # no such field.
            assigned = getattr(self.parallel_config, "assigned_physical_gpu_ids", None)
            if assigned:
                device_ids = [str(i) for i in assigned]
            else:
                device_ids = os.environ[env_var].split(",")
            assert len(device_ids) == world_size, (
                f"device_ids: {device_ids} should have device count: {world_size}"
            )
            try:
                device_id = int(device_ids[self.local_rank])
                start_idx = device_id * num_devices
                end_idx = start_idx + num_devices
                device_ids = [str(i) for i in range(start_idx, end_idx)]
                selected_devices = ",".join(device_ids)
            except ValueError as e:
                raise ValueError(
                    f"device_ids: {device_ids} should be a list of integers"
                ) from e

        os.environ[env_var] = selected_devices
        logger.info(
            "Local rank: %d, Selected devices: %s",
            self.local_rank,
            selected_devices,
        )

        if has_torch_rbln and num_devices > 1:
            os.environ["RBLN_NPUS_PER_DEVICE"] = str(num_devices)

    @instrument(span_name="Init device")
    def init_device(self) -> None:
        self.device = self.device_config.device

        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: RBLNModelRunner = RBLNModelRunner(
            self.vllm_config, self.device
        )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    def load_model(self):
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Estimate KV-cache DRAM, discounting the fixed command-stream buffers
        that warm-up's compiled decode runtimes reserve.

        One runtime per (decode bucket, query length): non-spec has a single
        query length (1); spec adds a second (num_spec + 1) per bucket;
        specialized-MoE decode repeats those plus one DP-asymmetric spec dummy.
        A draft model, when present, adds its own -- one per bucket, plus the
        specialized-MoE fallback. Counting all of them keeps the KV-block estimate
        from over-reserving and OOMing at runtime.
        """
        params_dict = dict(self.model_runner.model.named_parameters())
        device_name = current_platform.get_device_name().lower()
        assert "rbln" in device_name

        has_specialized_moe_decode = self.model_runner.specialized_moe_decode
        decode_batch_buckets_count = (
            self.model_runner.bucketing_manager.decode_batch_buckets_count
        )

        spec_enabled = self.speculative_config is not None
        num_decode_query_lens = 2 if spec_enabled else 1
        num_runtimes = 1 + decode_batch_buckets_count * num_decode_query_lens
        if has_specialized_moe_decode:
            num_runtimes += num_decode_query_lens
            if spec_enabled:
                num_runtimes += 1

        ratio: float = 1.0
        if self.model_config.quantization is not None:
            logger.info(
                "model quantization scheme = %s", self.model_config.quantization
            )
            # FIXME(RBLN) - for now, mxfp4/fp8 quantization is only supported
            quantization = self.model_config.quantization
            assert quantization in (
                "mxfp4",
                "gpt_oss_mxfp4",
                "fp8",
                "compressed-tensors",
            )

            if quantization == "compressed-tensors":
                qcfg = (
                    getattr(self.model_config.hf_config, "quantization_config", {})
                    or {}
                )
                groups = qcfg.get("config_groups", {})
                num_bits_set: set[int] = set()
                for group_cfg in groups.values():
                    nb = group_cfg.get("weights", {}).get("num_bits")
                    if nb is not None:
                        num_bits_set.add(nb)
                if not num_bits_set:
                    logger.warning(
                        "compressed-tensors quantization_config has no num_bits; "
                        "assuming 8-bit (fp8)."
                    )
                    num_bits = 8
                elif len(num_bits_set) == 1:
                    (num_bits,) = num_bits_set
                else:
                    raise RuntimeError(
                        f"compressed-tensors config has mixed bit-widths "
                        f"{num_bits_set}; not supported."
                    )

                if num_bits == 8:
                    quantization = "fp8"
                elif num_bits == 4:
                    quantization = "int4"
                else:
                    raise RuntimeError(
                        f"compressed-tensors {num_bits=} is not supported; "
                        f"only 4-bit (int4) or 8-bit (fp8)."
                    )

            if quantization == "fp8":
                nbits_per_param = 8
                packed_num_elems = 1
            elif quantization == "int4":
                nbits_per_param = 4
                packed_num_elems = 1
            elif quantization in ("mxfp4", "gpt_oss_mxfp4"):
                if "ca" in device_name:
                    # ATOM DOES NOT support mxfp4 quantization, handled by bf16
                    nbits_per_param = 16
                    # mlp weight scale is merged into params
                    # FIXME(RBLN) - expert scale merged into expert weight param
                    # ratio scale vs weight = 1 : 16
                    ratio = 16 / 17
                elif "cr" in device_name:
                    # REBEL can support mxfp4 quantization
                    nbits_per_param = 4
                else:
                    raise ValueError(
                        "invalid RBLN architecture, candidates = [ATOM(ca), REBEL(cr)]"
                    )
                # pack 2 mxfp4 elems into single uint8 elem
                packed_num_elems = 8 // 4
            else:
                raise ValueError(
                    "invalid quantization scheme, candidates = [fp8, int4, mxfp4]"
                )

        else:
            nbits_per_param = 16
            packed_num_elems = 1

        n_model_bytes = 0
        for value in params_dict.values():
            if value.is_floating_point():
                n_model_bytes += value.numel() * value.element_size()
            else:
                n_model_bytes += int(
                    value.numel() * packed_num_elems * ratio * nbits_per_param // 8
                )

        logger.info("n_model_bytes = %.2f GB", n_model_bytes / 1024**3)

        estimate_kwargs = dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            num_runtimes=num_runtimes,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
        )

        speculative_config = self.speculative_config
        drafter = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(drafter, "model", None)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        draft_parallel_config = getattr(
            speculative_config,
            "draft_parallel_config",
            None,
        )

        if draft_model is not None and draft_model_config is not None:
            if draft_parallel_config is None:
                draft_parallel_config = self.parallel_config

            draft_quantization = getattr(draft_model_config, "quantization", None)
            if (
                draft_quantization is not None
                and (method := getattr(speculative_config, "method", None)) != "mtp"
            ):
                # MTP draft shares the target checkpoint and inherits its
                # quantization (e.g. fp8 for DeepSeek-V3),
                # Eagle/Medusa draft are separately-trained models and
                # quantized variants are not validated on RBLN yet.
                raise ValueError(
                    f"draft model quantization is not supported for "
                    f"{method=}: {draft_quantization}"
                )

            model_kernel_size = estimate_model_kernel_size(
                model_config=self.model_config,
                parallel_config=self.parallel_config,
                n_model_bytes=n_model_bytes,
            )

            # Draft runtimes: one per bucket, plus the specialized-MoE fallback.
            # TODO(RBLN): an undercount since the draft started compiling both decode
            # query lengths. Reserving for what it actually compiles needs the count
            # split by speculative method, which the medusa path would want too.
            num_draft_runtimes = 1 + decode_batch_buckets_count
            if has_specialized_moe_decode:
                num_draft_runtimes += 1
            draft_n_model_bytes = 0

            for value in draft_model.parameters():
                draft_n_model_bytes += value.numel() * value.element_size()

            draft_kernel_size = estimate_model_kernel_size(
                model_config=draft_model_config,
                parallel_config=draft_parallel_config,
                n_model_bytes=draft_n_model_bytes,
            )
            estimate_kwargs["num_runtimes"] = num_runtimes + num_draft_runtimes
            estimate_kwargs["kernel_size"] = model_kernel_size + draft_kernel_size
            logger.info("draft_n_model_bytes = %.2f GB", draft_n_model_bytes / 1024**3)
            logger.info(
                "draft_model_kernel_size = %.2f GB",
                draft_kernel_size / 1024**3,
            )
        else:
            estimate_kwargs["n_model_bytes"] = n_model_bytes

        available_memory_estimate = estimate_available_memory(**estimate_kwargs)

        logger.info(
            "available_memory_estimate = %.2f GiB", available_memory_estimate / 1024**3
        )

        return available_memory_estimate

    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:
        """Get KV connector metadata from this worker if available.

        Returned dict is keyed by ``(pp_rank, tp_rank)``.
        """

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate RBLN KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.cache_config.num_cpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        dynamic_kv = envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE
        if dynamic_kv:
            self._assert_dynamic_kv_attention_layout()

        self.model_runner.initialize_kv_cache(
            self._maybe_shrink_kv_cache_for_compile(kv_cache_config)
        )

        if dynamic_kv:
            self._assert_dynamic_kv_cache_layout()

    def _compile_and_warmup_skip_reason(self) -> str | None:
        """Why the compile and warm-up will be skipped, or None if they will run."""
        if self.model_config.enforce_eager:
            return "enforce_eager is set"
        if not envs.VLLM_RBLN_COMPILE_MODEL:
            return "VLLM_RBLN_COMPILE_MODEL is off"
        if not envs.VLLM_RBLN_ENABLE_WARM_UP:
            return "VLLM_RBLN_ENABLE_WARM_UP is off"
        return None

    def _maybe_shrink_kv_cache_for_compile(
        self, kv_cache_config: KVCacheConfig
    ) -> KVCacheConfig:
        """Return a small-KV-cache copy of the config, or it unchanged.

        The cache allocated here is what `warmup_model()` traces, so its
        `num_blocks` becomes the hint of the `mark_dynamic`'d dim.
        """
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            return kv_cache_config
        skip_reason = self._compile_and_warmup_skip_reason()
        if skip_reason is not None:
            # Nothing compiles, so no artifact carries a profile: shrinking
            # anyway would set the latch and then find no runtimes to resize.
            logger.warning(
                "[Dynamic KV] compile/warm-up is skipped (%s), so the cache stays "
                "at the estimated %d blocks and this feature does nothing for "
                "this run.",
                skip_reason,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        override = self.cache_config.num_gpu_blocks_override
        if override is not None:
            logger.warning(
                "[Dynamic KV] --num-gpu-blocks-override=%d pins the count; no "
                "shrink and no resize. Compiling at %d blocks.",
                override,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        compile_num_blocks = COMPILE_KV_CACHE_NUM_BLOCKS
        if compile_num_blocks >= kv_cache_config.num_blocks:
            # Cancelling the shrink cancels the resize too, so serving on would
            # leave the run on the pre-compile estimate -- #42 reproducing
            # silently, with nobody having asked for it. Refuse instead.
            raise RuntimeError(
                f"the {compile_num_blocks}-block compile hint is not below the "
                f"{kv_cache_config.num_blocks} blocks vllm estimated, so there is "
                "nothing to shrink and no resize would run. See "
                "docs/dynamic_kv_cache.md."
            )

        shrunk = _kv_cache_config_at(kv_cache_config, compile_num_blocks)
        self._kv_blocks_before_shrink = kv_cache_config.num_blocks
        logger.info(
            "[Dynamic KV] compiling with %d KV blocks instead of %d; resized after "
            "warm-up from the compiled profile.",
            compile_num_blocks,
            kv_cache_config.num_blocks,
        )
        return shrunk

    def _assert_dynamic_kv_attention_layout(self) -> None:
        """Guard: every attention layer must dispatch to the paged-causal kernel.

        i.e. `sliding_window is None`, `is_causal` True and `is_normal` False;
        `is_normal` becomes True when `block_size == max_model_len`. Lives in
        the worker, not platform config validation: `get_layers_from_vllm_config`
        reads `static_forward_context`, which only the model build fills.
        """
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        offenders: list[str] = []
        for layer_name, layer in attn_layers.items():
            impl = layer.impl
            sliding_window = getattr(impl, "sliding_window", None)
            is_causal = getattr(impl, "is_causal", None)
            is_normal = getattr(impl, "is_normal", None)
            if (
                sliding_window is not None
                or is_causal is not True
                or is_normal is not False
            ):
                offenders.append(
                    f"{layer_name}(sliding_window={sliding_window}, "
                    f"is_causal={is_causal}, is_normal={is_normal})"
                )
        if offenders:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires every layer to dispatch "
                "to paged_flash_causal_attention_naive_*. Offending: "
                + ", ".join(offenders[:8])
                + (f" (+{len(offenders) - 8} more)" if len(offenders) > 8 else "")
            )

    def _assert_dynamic_kv_cache_layout(self) -> None:
        """Guard: the KV bindings must satisfy the compiler's dynamic-input rules."""
        # NOTE(RBLN): the compiler requires each mark_dynamic'd KV input to have
        # exactly one use, and that use to be paged_flash_causal_attention_naive_*.
        # Both checks read state `initialize_kv_cache` fills, so moving them
        # earlier makes them pass vacuously.
        mr = self.model_runner

        # NOTE(RBLN): a deduped base makes the per-layer kv_cache a view built
        # inside the graph, so mark_dynamic is a no-op, and marking the base is
        # rejected by the compiler ('kind is not call_arg').
        if mr.kv_cache_bases:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires KV base deduplication "
                f"to be inactive, but {len(mr.kv_cache_bases)} deduped bases "
                "were built."
            )

        # NOTE(RBLN): one tensor reaching two attention calls gives the marked
        # input two uses, which the compiler's validator rejects.
        if mr.shared_kv_cache_layers:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support cross-layer KV "
                f"sharing, but {len(mr.shared_kv_cache_layers)} layer(s) reuse "
                "another layer's KV cache."
            )

    def _collect_dynamic_kv_runtimes(self) -> list[Any]:
        """Every rbln runtime that holds a slice of the KV cache.

        The runner's holder is the whole set: spec decode, the only other
        producer of KV-holding runtimes, is refused under this flag.
        """
        runtimes: list[Any] = []
        seen: set[int] = set()
        for runtime in self.model_runner.runtime_holder:
            if id(runtime) in seen:
                continue
            seen.add(id(runtime))
            runtimes.append(runtime)
        return runtimes

    def _dynamic_kv_device_topology(self, runtimes: list[Any]) -> tuple[int, int]:
        """`(num_nodes, num_chiplets)` from a runtime's allocation report.

        Not from the memory profile: that would make the coverage check circular.
        """
        for runtime in runtimes:
            alloc = runtime.get_alloc_per_chiplet(1)
            if not alloc:
                continue
            num_nodes = len(alloc)
            chiplet_counts = {len(per_node) for per_node in alloc}
            if len(chiplet_counts) != 1:
                raise RuntimeError(
                    "get_alloc_per_chiplet() reported a ragged topology "
                    f"({[len(x) for x in alloc]}); cannot derive a uniform "
                    "per-chiplet budget."
                )
            num_chiplets = next(iter(chiplet_counts))
            if num_chiplets <= 0:
                continue
            return num_nodes, num_chiplets
        raise RuntimeError("no rbln runtime reported a per-chiplet allocation shape.")

    def _charge_retained_compile_kv_cache(
        self,
        budget: dict[int, dict[int, int]],
        merged: Any,
    ) -> dict[int, dict[int, int]]:
        """Charge the compile-time KV cache that TP>=2 does not give back.

        The compile cache is per-block growth, so neither the capacity-derived
        budget nor `max_num_blocks`' own `base_bytes` subtraction accounts for
        it; on TP>=2 it is not returned and the process cannot see that it was
        not. Charging TP=1 would cost blocks it does get back.

        DP+EP keeps the cache resident at tp_size=1 and is therefore not
        charged here; that gap is in docs/dynamic_kv_cache.md.
        """
        resident_blocks = self.model_runner.kv_cache_config.num_blocks
        tp_size = self.parallel_config.tensor_parallel_size
        if tp_size <= 1:
            return budget

        # Difference of two aligned usages, not num_blocks * bytes_per_block:
        # alignment is per region, so this matches what `max_num_blocks` does.
        at_compile = per_chiplet_usage(merged, resident_blocks)
        at_zero = per_chiplet_usage(merged, 0)

        charged = dict(budget)
        retained_by_unit: dict[tuple[int, int], int] = {}
        for unit, used in at_compile.items():
            retained = used - at_zero.get(unit, 0)
            if retained <= 0:
                continue
            node_id, chiplet_id = unit
            charged[node_id] = dict(charged[node_id])
            charged[node_id][chiplet_id] -= retained
            retained_by_unit[unit] = retained

        # INFO: the charge working as designed. The hint is a constant, so the
        # lower block count is the price of the parallelism, not a knob.
        logger.info(
            "[Dynamic KV] tp=%d keeps the %d-block compile cache; charged per "
            "chiplet: %s. Expect fewer blocks than TP=1.",
            tp_size,
            resident_blocks,
            {f"{n}:{c}": v for (n, c), v in sorted(retained_by_unit.items())},
        )

        exhausted = {
            f"{node_id}:{chiplet_id}": remaining
            for node_id, chiplets in charged.items()
            for chiplet_id, remaining in chiplets.items()
            if remaining <= 0
        }
        if exhausted:
            # The hint is a constant, so the budget is the only side to move.
            raise RuntimeError(
                "the retained compile-time KV cache leaves no budget on "
                f"{exhausted}; raise --gpu-memory-utilization or lower --block-size."
            )
        return charged

    def _dynamic_kv_chiplet_budget(self, num_chiplets: int) -> int:
        """Per-chiplet byte budget for `max_num_blocks`.

        `floor(dram_total / num_chiplets * gmu)` minus the unprofiled reserve
        minus other tenants' usage.
        """
        # Capacity-derived on purpose: `max_num_blocks` subtracts the profile's
        # own `base_bytes`, so a free-memory figure would charge it twice.
        dram_total = read_rbln_card_dram_total_bytes()
        if dram_total is None:
            raise RuntimeError(
                "cannot read /sys/class/rebellions/rbln*/dram_total, which "
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE needs to budget per chiplet."
            )
        capacity_per_chiplet = dram_total // num_chiplets
        gmu = self.cache_config.gpu_memory_utilization
        gmu_budget = int(capacity_per_chiplet * gmu)
        raw_budget = gmu_budget - DYNAMIC_KV_UNPROFILED_RESERVE_BYTES
        # Card-scope figure, so scale it: charging the whole card to every chiplet
        # drove the budget negative. An even spread is the same assumption
        # `capacity_per_chiplet` already makes.
        foreign_per_chiplet = self._foreign_dram_used_bytes // num_chiplets
        budget = raw_budget - foreign_per_chiplet
        logger.info(
            "[Dynamic KV] per-chiplet budget: dram_total=%d chiplets=%d "
            "capacity=%d gpu_memory_utilization=%.3f gmu_budget=%d "
            "unprofiled_reserve=%d raw=%d foreign_used=%d "
            "foreign_per_chiplet=%d adjusted=%d",
            dram_total,
            num_chiplets,
            capacity_per_chiplet,
            gmu,
            gmu_budget,
            DYNAMIC_KV_UNPROFILED_RESERVE_BYTES,
            raw_budget,
            self._foreign_dram_used_bytes,
            foreign_per_chiplet,
            budget,
        )
        if budget <= 0:
            raise RuntimeError(
                f"per-chiplet KV budget is non-positive ({budget} bytes) after the "
                f"reserve and {foreign_per_chiplet} bytes of foreign usage."
            )
        return budget

    def compute_dynamic_kv_num_blocks(self) -> int | None:
        """Ask the compiled artifacts how many KV blocks fit this device.

        Runs after warm-up and reallocates nothing; the engine takes the minimum
        across ranks and hands it back through `apply_dynamic_kv_num_blocks`.
        None means the path is not in play.
        """
        if self.cache_config.num_gpu_blocks_override is not None:
            logger.info(
                "[Dynamic KV] --num-gpu-blocks-override=%d is set; leaving the "
                "KV cache alone.",
                self.cache_config.num_gpu_blocks_override,
            )
            return None
        if self._kv_blocks_before_shrink is None:
            # The branch that cancelled the shrink already logged why.
            logger.warning(
                "[Dynamic KV] the cache was not shrunk, so no profile is queried "
                "and the count stays at the %d blocks vllm estimated.",
                self.model_runner.kv_cache_config.num_blocks,
            )
            return None

        runtimes = self._collect_dynamic_kv_runtimes()
        profiles = []
        for runtime in runtimes:
            executor = getattr(runtime, "_executor", None)
            if executor is None:
                logger.warning(
                    "[Dynamic KV] runtime %s exposes no _executor; skipping it.",
                    type(runtime).__name__,
                )
                continue
            try:
                # No public wrapper exists for this query yet.
                profile = executor.kv_cache_memory_profile()
            except RuntimeError as exc:
                message = str(exc)
                if "no dynamic-shape variable" in message:
                    reason = "static artifact (no dynamic-shape variable)"
                elif "Unsupported operation" in message:
                    reason = "artifact type does not support the query"
                else:
                    reason = "unexpected RuntimeError"
                logger.warning(
                    "[Dynamic KV] kv_cache_memory_profile() unavailable for one "
                    "runtime (%s): %s. Continuing with the rest.",
                    reason,
                    message,
                )
                continue
            profiles.append(profile)
            # The only record of base_bytes / bytes_per_block / alignment; see
            # kv_profile.SOURCE_PROFILE_LOG_KEY.
            logger.info(
                "[Dynamic KV] compiled profile: %s rank=%d profile_index=%d "
                "runtime=%s#%d num_regions=%d",
                format_profile_for_log(profile),
                self.rank,
                len(profiles) - 1,
                type(runtime).__name__,
                len(profiles) - 1,
                len(profile.device_regions),
            )

        if not profiles:
            raise RuntimeError(
                f"[Dynamic KV] not one of the {len(runtimes)} rbln runtimes "
                "returned a KV memory profile; was VLLM_CACHE_ROOT replaying a "
                "static build?"
            )

        merged = merge_kv_cache_memory_profiles(profiles)
        logger.info(
            "[Dynamic KV] merged profile: %s rank=%d num_source_profiles=%d "
            "num_regions=%d shared_base=%d private_base=%d growth=%d",
            format_profile_for_log(merged, MERGED_PROFILE_LOG_KEY),
            self.rank,
            merged.num_source_profiles,
            len(merged.device_regions),
            merged.num_shared_base_regions,
            merged.num_private_base_regions,
            merged.num_growth_regions,
        )
        num_nodes, num_chiplets = self._dynamic_kv_device_topology(runtimes)
        budget_per_chiplet = self._dynamic_kv_chiplet_budget(num_chiplets)
        budget = build_per_chiplet_budget(num_nodes, num_chiplets, budget_per_chiplet)
        assert_budget_covers_profile(merged, budget)
        budget = self._charge_retained_compile_kv_cache(budget, merged)

        # NOTE(RBLN): `rebel/__init__.py` does not import `rebel.kv_cache`, so
        # this exact import form is the only reliable one.
        from rebel.kv_cache import max_num_blocks

        try:
            num_blocks = max_num_blocks(merged, budget, per_node_budget=False)
        except ValueError as exc:
            raise RuntimeError(
                "[Dynamic KV] the merged profile has no per-block growth, i.e. the "
                "artifacts were not compiled with a dynamic KV dim."
            ) from exc

        base_usage = per_chiplet_usage(merged, 0)
        logger.info(
            "[Dynamic KV] merged profile predicts base bytes per (node, chiplet): %s",
            {k: v for k, v in sorted(base_usage.items())},
        )
        if num_blocks <= 0:
            raise RuntimeError(
                "[Dynamic KV] max_num_blocks() returned 0: the per-chiplet budget "
                f"of {budget_per_chiplet} bytes does not fit the base usage "
                f"{dict(sorted(base_usage.items()))}."
            )

        fit_usage = per_chiplet_usage(merged, num_blocks)
        logger.info(
            "[Dynamic KV] rank %d: computed_num_blocks=%d "
            "chiplet_budget_bytes=%d predicted usage per (node, chiplet)=%s "
            "total=%d",
            self.rank,
            num_blocks,
            budget_per_chiplet,
            {k: v for k, v in sorted(fit_usage.items())},
            sum(fit_usage.values()),
        )
        for runtime in runtimes:
            logger.info(
                "[Dynamic KV] runtime %s get_alloc_per_chiplet(1)=%s",
                type(runtime).__name__,
                runtime.get_alloc_per_chiplet(1),
            )

        return num_blocks

    def apply_dynamic_kv_num_blocks(self, n: int | None) -> int | None:
        """Resize the KV cache to the block count the engine settled on.

        `n` is None when no usable count was computed; the pre-shrink number is
        put back then, or the server would serve from the tiny compile cache.
        """
        before_shrink = self._kv_blocks_before_shrink
        target = before_shrink if n is None else n
        if target is None:
            return None
        self._kv_blocks_before_shrink = None

        current = self.model_runner.kv_cache_config.num_blocks
        if target == current:
            # The latch already describes reality, so no reset is needed either.
            logger.info(
                "[Dynamic KV] KV cache already holds %d blocks; nothing to reallocate.",
                target,
            )
            return target

        if n is None:
            logger.warning(
                "[Dynamic KV] restoring KV cache to the %d blocks vllm sized it "
                "with (compiled with %d).",
                target,
                current,
            )
        self._reallocate_kv_cache(target)
        self._materialize_kv_cache()
        return target

    def _materialize_kv_cache(self) -> None:
        """One decode step so the resized pool is paid for at boot, not by a user.

        The reallocation leaves physical allocation to the next forward, which
        would otherwise land on the first request.
        """
        # The smallest decode bucket warmup already compiled: no new graph.
        num_reqs = min(self.model_runner.bucketing_manager.decode_batch_buckets)
        with set_compile_stage("warmup"), self.model_runner.offload_context():
            self.model_runner._dummy_run(num_reqs, 1, False)

    def _release_kv_cache_tensors(self, old_cfg: KVCacheConfig) -> None:
        """Drop every reference to the outgoing KV cache and free its device DRAM.

        Called *before* the replacement is allocated. Otherwise the old tensors
        outlive the free, the allocator keeps their blocks reserved, and the
        peak is base + old + new rather than base + max(old, new).
        """
        mr = self.model_runner

        # Read residency from the tensors, not from the env, and before they go.
        kv_device_types = {kv_cache.device.type for kv_cache in mr.kv_caches}
        was_device_resident = bool(kv_device_types - {"meta", "cpu"})

        # NOTE(RBLN): the rebind (initialize_kv_cache_tensors) reassigns
        # kv_caches and kv_cache_names from one ordered name list and rebuilds
        # kv_cache_bases, so drop all three stale bindings together before the
        # reallocation.
        mr.kv_caches = []
        mr.kv_cache_bases = []
        mr.kv_cache_names = []

        # NOTE(RBLN): the rebind also parks each layer's view on the Attention
        # module; the next bind overwrites it only *after* the new tensors
        # exist, which is the window this closes.
        forward_context = mr.compilation_config.static_forward_context
        unbound = 0
        # `KVCacheTensor.shared_by` is the same list `_allocate_kv_cache_tensors`
        # keys its output on, so this is exactly the set of bound layers.
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

        # NOTE(RBLN): every per-layer tensor is a view and keeps its base alive,
        # so a reference cycle would defer the free past the new allocation.
        gc.collect()

        released = empty_rbln_device_caches()
        logical_bytes = sum(t.size for t in old_cfg.kv_cache_tensors)
        # Not observable in-process; confirm from sysfs `dram_used` across the
        # resize instead.
        logger.info(
            "[Dynamic KV] released the outgoing %d-block KV cache: "
            "outgoing_kv_logical_bytes=%d unbound_layers=%d kv_device_types=%s "
            "allocator_cache_emptied=%s device_resident=%s",
            old_cfg.num_blocks,
            logical_bytes,
            unbound,
            sorted(kv_device_types),
            released,
            was_device_resident,
        )

    def _reallocate_kv_cache(self, new_num_blocks: int) -> None:
        """Rebuild only the KV cache *tensors* at `new_num_blocks`.

        No recompilation happens because the affected dim is `mark_dynamic`'d;
        the physical allocation happens on the next forward.
        """
        # NOTE(RBLN): `initialize_kv_cache()` must not be re-run --
        # `initialize_attn_backend` asserts `len(self.attn_groups) == 0`, and the
        # backends and input batch depend on block_size, not num_blocks.
        mr = self.model_runner
        old_cfg = mr.kv_cache_config
        old_num_blocks = old_cfg.num_blocks

        new_cfg = _kv_cache_config_at(old_cfg, new_num_blocks)
        self.cache_config.num_gpu_blocks = new_num_blocks
        self.cache_config.num_cpu_blocks = new_num_blocks

        logger.info(
            "[Dynamic KV] reallocating KV cache: %d -> %d blocks",
            old_num_blocks,
            new_num_blocks,
        )
        mr.kv_cache_config = new_cfg
        # Order is load-bearing: see `_release_kv_cache_tensors`. It also does
        # the `mr.kv_caches = []` that the rebind reassigns.
        self._release_kv_cache_tensors(old_cfg)
        # Re-applies mark_dynamic and rebinds the KV caches itself.
        mr.initialize_kv_cache_tensors(new_cfg, mr._kernel_block_sizes)

        if mr.kv_cache_bases:
            raise RuntimeError(
                "KV base deduplication became active after the dynamic-KV "
                f"reallocation ({len(mr.kv_cache_bases)} bases); the "
                "mark_dynamic'd layer tensors are no longer graph inputs."
            )

        # NOTE(RBLN): warm-up latched the adaptive buffer sizes at the old
        # num_blocks; without this clear, the next forward raises 'variable dim
        # changed after adaptive buffers were fixed'. No getattr: a missing
        # symbol must fail here, not on the first request.
        runtimes = self._collect_dynamic_kv_runtimes()
        for runtime in runtimes:
            runtime.reset_adaptive_buffers()
        logger.info(
            "[Dynamic KV] reset_adaptive_buffers() on %d runtime(s).",
            len(runtimes),
        )

    @instrument(span_name="Warmup (NPU)")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # NOTE(RBLN): Manual timing since RBLN does not support @support_torch_compile.
        st = time.perf_counter()

        # NOTE(RBLN): Thread policy + RBLN_NUM_THREADS must be set
        # before compile/warm-up. CPU affinity is applied afterward.
        self._ensure_rbln_host_threads_before_compile()

        try:
            if (skip := self._compile_and_warmup_skip_reason()) is not None:
                logger.info("Skipping compile_or_warm_up_model (%s).", skip)
            else:
                self.model_runner.warmup_model()

                # Connectors that defer KV-cache registration (RBLN NIXL D2D
                # and LMCache) finalize it here: the KV cache physical views
                # only exist once warm-up has run the compiled model. Walk the
                # connector tree (incl. MultiConnector children) so the hook
                # still runs when combined with other connectors. Only on a
                # successful warm-up — not on the skipped or failed path.
                if has_kv_transfer_group():
                    finalize_kv_cache_registrations(get_kv_transfer_group())

                # NOTE(RBLN): the sampler warm-up and the deferred KV-cache
                # registration above are per-rank, so ranks reach this point
                # hundreds of ms apart. Nothing left before the first request is
                # collective, so that skew would otherwise land in the first
                # forward's DP all-reduce and be billed to the prefill it runs.
                if self.parallel_config.data_parallel_size > 1:
                    logger.info("Warm-up done; waiting for the other DP ranks.")
                    dist.barrier(group=get_dp_group().cpu_group)
                    logger.info("All DP ranks left warm-up.")

        except BackendCompilerFailed as e:

            def is_rbln_oom_error(exc: BaseException | None) -> bool:
                if not isinstance(exc, RuntimeError):
                    return False

                return any(
                    isinstance(arg, str)
                    and (
                        "SYS_ENOMEM: Out of memory" in arg
                        or "SYS_EBUSY: Lack of device memory" in arg
                    )
                    for arg in exc.args
                )

            if is_rbln_oom_error(e.inner_exception):
                blocks = self.model_runner.kv_cache_config.num_blocks
                if self._kv_blocks_before_shrink is not None:
                    # The KV cache is not what exhausted the device at this size,
                    # so --num-gpu-blocks-override is the wrong advice here.
                    raise RuntimeError(
                        f"Not enough memory to compile against the {blocks}-block "
                        "compile-time KV cache. Reduce --max-num-batched-tokens, "
                        "--max-model-len or --max-num-seqs, or raise "
                        "--tensor-parallel-size."
                    ) from e
                raise RuntimeError(
                    f"Not enough memory for {blocks} blocks of KV cache. "
                    "Try reducing the number of blocks by setting "
                    "--num-gpu-blocks-override."
                ) from e
            raise
        finally:
            # NOTE(RBLN): Apply CPU affinity only after compile/warm-up.
            self._ensure_rbln_cpu_affinity_after_warmup()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return CompilationTimes(language_model=time.perf_counter() - st, encoder=0.0)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def _send_handoff(self, tensors: dict) -> None:
        """Hand this stage's output on; a seam the metrics patch wraps."""
        # NOTE(RBLN): DO NOT all_gather_group for RBLN pp
        get_pp_group().send_tensor_dict(tensors)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0

        if forward_pass and not get_pp_group().is_first_rank:
            # NOTE(RBLN): DO NOT all_gather_group for RBLN pp
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict()
            )

        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
        if isinstance(output, ModelRunnerOutput | NoneType):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != ("external_launcher")
            and not get_pp_group().is_last_rank
        )

        self._send_handoff(output.tensors)

        # Non-last PP rank: the model runner already surfaces this rank's
        # KV-connector output through the two-phase sample_tokens() path
        # (mirroring the upstream model runner). The engine consumes this
        # execute_model result only for error propagation, so return None
        # rather than emitting the same finished send/recv notifications here.
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from vllm.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # Build the full trace name
            trace_name = (
                f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
            )

            # Create the profiler wrapper only on the first start call
            if self.profiler is None:
                from vllm.profiler.wrapper import TorchProfilerActivityMap

                activities = ["CPU"]
                if "RBLN" in TorchProfilerActivityMap:
                    activities.append("RBLN")

                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=activities,
                    )
                    logger.debug(
                        "Starting torch profiler with tarce name: %s", trace_name
                    )
                else:
                    raise ValueError(
                        f"Invalid proifler value of {self.profiler_config.profiler}."
                    )

            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        # Serving-time DP-idle step: this rank has no real work. Run a non-warmup
        # dummy (warmup=False) so it contributes a minimal (num_reqs=1, qlen=1)
        # entry to the cross-DP collective, is EXCLUDED from the shape decision,
        # then adopts the busy-decided shape and runs the same compiled decode
        # graph the busy ranks run -- so an idle rank never drags the collective
        # into a fall-back route nor lands on an uncompiled shape.
        self.model_runner._dummy_run(1, 1, is_prefill=False, warmup=False)

    # def add_lora(self, lora_request: LoRARequest) -> bool:
    #     return self.model_runner.add_lora(lora_request)

    # def remove_lora(self, lora_id: int) -> bool:
    #     return self.model_runner.remove_lora(lora_id)

    # def list_loras(self) -> set[int]:
    #     return self.model_runner.list_loras()

    # def pin_lora(self, lora_id: int) -> bool:
    #     return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def shutdown(self) -> None:
        self._release_offload_temp_storage()

        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

    def reset_encoder_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_encoder_cache", None)
        if callable(reset_fn):
            reset_fn()

    def _release_offload_temp_storage(self) -> None:
        # The runtime drops the offload dir on teardown, but that runs last and vLLM
        # SIGKILLs a worker seconds after asking it to stop, so reclaim up front.
        if not has_torch_rbln:
            return
        try:
            num_removed = torch.rbln.release_offload_temp_storage()
        except Exception:
            logger.exception("Failed to release RBLN offload temp storage")
            return
        if num_removed:
            logger.info("Released %d RBLN offload temp file(s)", num_removed)

    def _ensure_rbln_host_threads_before_compile(self) -> None:
        """Set OpenMP / torch / numba threads before ``warm_up_model()`` without
        CPU affinity.

        Affinity is applied later (after warm-up) so ``torch.compile`` / dummy
        compile sees an unpinned CPU mask while thread counts and
        ``RBLN_NUM_THREADS`` match Dynamo. Default thread count uses the same
        logical CPU count ``set_cpu_affinity`` will pin to (NUMA / DP split),
        not the pre-split ``sched_getaffinity`` mask.
        """
        if self._rbln_host_threads_before_compile_ready:
            return

        allocated_cpus = get_rbln_planned_affinity_cpu_count(
            self.rank,
            self.local_rank,
            self.parallel_config,
        )
        num_threads = max(2, allocated_cpus // 2)
        set_omp_num_threads(
            self.rank,
            self.local_rank,
            num_threads,
        )

        # NOTE(RBLN): numba is used throughout vllm code base (especially in spec-dec)
        # however accessing numba thread settings somewhat affects torch
        # thread settings and cause global state change leading to recompilation.
        # Thus the only solution for now is to set both thread settings to identical
        # value in correct order like below

        # Code below sets numba num thread to torch num thread and
        # potentially change torch num thread to other value
        numba.set_num_threads(torch.get_num_threads())

        # Code below restores torch num thread to its original value
        # before numba.set_num_threads
        torch.set_num_threads(numba.get_num_threads())

        self._rbln_host_threads_before_compile_ready = True

    def _ensure_rbln_cpu_affinity_after_warmup(self) -> None:
        """Pin CPU affinity after ``warm_up_model()``; does not change torch
        thread counts."""
        if self._rbln_cpu_affinity_applied:
            return

        set_cpu_affinity(
            self.rank,
            self.local_rank,
            self.parallel_config,
        )
        self._rbln_cpu_affinity_applied = True


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "gloo",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    world_size = parallel_config.world_size

    # Set envs for RCCL
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    if parallel_config.data_parallel_size > 1:
        world_size_across_dp = parallel_config.world_size_across_dp
        dp_rank = parallel_config.data_parallel_rank
        rank_across_dp = dp_rank * world_size
        rank_across_dp += rank
        logger.info(
            "world_size_across_dp = %s, rank_across_dp = %s",
            world_size_across_dp,
            rank_across_dp,
        )
        # consider across_dp
        os.environ["LOCAL_RANK"] = str(rank_across_dp)
        os.environ["WORLD_SIZE"] = str(world_size_across_dp)

    new_backend = backend
    if envs.VLLM_RBLN_AUTO_PORT:
        if has_torch_rbln:
            new_backend = "rbln-ccl"
            os.environ["RCCL_PORT_GEN"] = "1"
        else:
            logger.warning(
                "Cannot use auto port because torch-rbln is not installed. "
                "You may need to install torch-rbln to use auto port feature."
            )

    init_distributed_environment(
        world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend=new_backend,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )
