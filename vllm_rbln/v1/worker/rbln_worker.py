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
from collections.abc import Mapping
from contextlib import nullcontext
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
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
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
from vllm_rbln.config import build_rbln_config, set_rbln_config
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.utils import (
    finalize_kv_cache_registrations,
)
from vllm_rbln.logger import init_logger
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
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.utils import (
    estimate_available_memory,
    estimate_model_kernel_size,
    get_rbln_planned_affinity_cpu_count,
    read_rbln_card_dram_used_bytes,
    rescale_kv_cache_config,
    set_cpu_affinity,
    set_omp_num_threads,
    worker_fail_fast,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

# Trace hint for the mark_dynamic'd KV dim, not a capacity: dynamo specializes
# a smaller dim away, and below this artifacts abort on device at larger
# max_num_batched_tokens.
COMPILE_KV_CACHE_NUM_BLOCKS = 4
# Allocator-snapshot fallback only: device memory the caching allocator never
# sees (the runtime's direct allocations, e.g. command streams).
DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES = 48 * 1024 * 1024
# Sub-block prefix caching copies partial blocks with command streams the
# runtime uploads per request; keep this much per chiplet out of the KV budget.
DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES = 64 * 1024 * 1024


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
        self.fail_fast = issubclass(Executor.get_class(vllm_config), MultiprocExecutor)

        # Before _init_device_env(), which reads device-count options.
        set_rbln_config(build_rbln_config(vllm_config.additional_config))

        self._init_device_env()

        self._rbln_host_threads_before_compile_ready = False
        self._rbln_cpu_affinity_applied = False

        # num_blocks vLLM sized the cache with, stashed while it is shrunk for
        # the compile. None when no shrink is pending.
        self._kv_blocks_before_shrink: int | None = None
        # Programs warm-up built, captured for the dynamic-KV sizing.
        self._dynamic_kv_programs: list[Any] = []
        # Per-unit `used` the sizing predicts once the cache is reallocated.
        self._dynamic_kv_expected_used: dict[Unit, int] = {}
        # Other tenants' device DRAM, sampled before this worker allocates
        # anything. Card-scope; only the allocator-snapshot fallback reads it.
        self._foreign_dram_used_bytes = 0
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            self._foreign_dram_used_bytes = read_rbln_card_dram_used_bytes()
            logger.debug(
                "foreign device DRAM at worker init: %d bytes "
                "(RBLN_VISIBLE_DEVICES=%s)",
                self._foreign_dram_used_bytes,
                os.environ.get("RBLN_VISIBLE_DEVICES", ""),
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
        env_var = current_platform.device_control_env_var
        num_devices = envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK

        dp_rank = self.parallel_config.data_parallel_rank_local or 0
        slot = dp_rank * self.parallel_config.world_size + self.local_rank
        first = slot * num_devices
        # The visible variant, because every worker process is handed a data
        # parallel mapping of one device per rank that the logical variant
        # prefers (MultiprocExecutor.worker_main), and a rank owning
        # num_devices NPUs cannot be described by one entry.
        try:
            selected = [
                current_platform.visible_device_id_to_physical_device_id(first + offset)
                for offset in range(num_devices)
            ]
        except IndexError as e:
            raise ValueError(
                f"rank slot {slot} needs {num_devices} NPU(s) from index "
                f"{first} of {env_var}={os.environ.get(env_var, '')!r}. "
                "One entry per NPU is expected."
            ) from e

        selected_devices = ",".join(str(device) for device in selected)
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
                "modelopt_mixed",
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
            elif quantization == "modelopt_mixed":
                # The fp8 weights and both NVFP4 scales are float dtypes and are
                # counted by element_size() below
                nbits_per_param = 4
                packed_num_elems = 8 // 4
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

        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE and not torch.rbln.is_dummy_device():
            snapshot, source = self._dynamic_kv_memory_snapshot(self.device)
            if envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN:
                measured = estimate_available_memory(
                    **estimate_kwargs, chiplet_memory=snapshot
                )
                logger.info(
                    "[Dynamic KV] dry run: the %s memory snapshot of %s would put the "
                    "pre-compile estimate at %.2f GiB; keeping the whole-card formula.",
                    source,
                    self.device,
                    measured / 1024**3,
                )
            else:
                estimate_kwargs["chiplet_memory"] = snapshot
                logger.info(
                    "[Dynamic KV] pre-compile estimate from the %s memory snapshot of "
                    "%s.",
                    source,
                    self.device,
                )

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
        if envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN:
            logger.warning(
                "[Dynamic KV] dry run: compiling at the %d blocks vllm sized; the "
                "count this feature would pick is only logged after warm-up.",
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
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
        """Guard: every attention layer must dispatch to a paged naive kernel the
        compiler admits a dynamic KV input for (flash causal or sliding window).

        i.e. `is_causal` True and `is_normal` False; `is_normal` becomes True
        when `block_size == max_model_len`. Lives in the worker, not platform
        config validation: `get_layers_from_vllm_config` reads
        `static_forward_context`, which only the model build fills.
        """
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
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires every layer to dispatch "
                "to a paged causal or sliding-window naive kernel. Offending: "
                + ", ".join(offenders[:8])
                + (f" (+{len(offenders) - 8} more)" if len(offenders) > 8 else "")
            )

    def _assert_dynamic_kv_cache_layout(self) -> None:
        """Guard: the KV bindings must satisfy the compiler's dynamic-input rules."""
        # NOTE(RBLN): reads state `initialize_kv_cache` fills, so moving it
        # earlier makes it pass vacuously.
        mr = self.model_runner

        # NOTE(RBLN): the compiler admits a dynamic input through view ops into
        # several paged attention calls, but not the same view into two calls.
        if mr.shared_kv_cache_layers:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support cross-layer KV "
                f"sharing, but {len(mr.shared_kv_cache_layers)} layer(s) reuse "
                "another layer's KV cache."
            )

    def _capture_dynamic_kv_programs(self):
        """Scope that records the programs warm-up builds, when the flag is on."""
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            return nullcontext(None)
        if not has_torch_rbln:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE needs torch_rbln's "
                "capture_programs(); torch.rbln is not importable."
            )
        return torch.rbln.capture_programs()

    def _collect_dynamic_kv_runtimes(self) -> list[Any]:
        """Every rbln runtime warm-up built; each KV-holding one binds a slice of
        the KV cache.

        Spec decode, the only other producer of KV-holding runtimes, is refused
        under this flag.
        """
        runtimes: list[Any] = []
        seen: set[int] = set()
        for program in self._dynamic_kv_programs:
            runtime = program.runtime
            if id(runtime) in seen:
                continue
            seen.add(id(runtime))
            runtimes.append(runtime)
        return runtimes

    def _dynamic_kv_memory_snapshot(
        self, device: torch.device
    ) -> tuple[dict[Unit, ChipletMemory], str]:
        """Per-(node, chiplet) `(total, used)` of the device, and where it came from.

        The driver's figures (`mem_get_info_per_chiplet`) see every process and
        every allocation. Without them -- an older UMD/KMD or torch_rbln -- this
        process's caching allocator stands in, topped up with the reserve and the
        other tenants' usage sampled at init.
        """
        rbln = torch.rbln
        query = getattr(rbln, "mem_get_info_per_chiplet", None)
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
        # The runtime hands cached-but-free blocks back only on a failed
        # allocation, so they would otherwise count as used here.
        rbln.empty_cache(device)
        memory_per_chiplet = int(rbln.get_device_properties(device).memory_per_chiplet)
        snapshot = snapshot_from_allocator(
            rbln.memory_stats_per_chiplet(device),
            memory_per_chiplet=memory_per_chiplet,
            foreign_card_used_bytes=self._foreign_dram_used_bytes,
            reserve_bytes=DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES,
        )
        logger.info(
            "[Dynamic KV] allocator snapshot: memory_per_chiplet=%d foreign_used=%d "
            "reserve=%d units=%d",
            memory_per_chiplet,
            self._foreign_dram_used_bytes,
            DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES,
            len(snapshot),
        )
        return snapshot, "allocator"

    def _kv_copy_stream_reserve_bytes(self) -> int:
        """Per-chiplet bytes to keep out of the KV budget when the scheduler
        will run sub-block prefix caching (the same predicate `RBLNScheduler`
        uses)."""
        if not self.cache_config.enable_prefix_caching:
            return 0
        if not build_rbln_config(self.vllm_config.additional_config).sub_block_cache:
            return 0
        # Imported here: the manager pulls in vllm.distributed.kv_events (numba).
        from vllm_rbln.v1.core.rbln_kv_cache_manager import RBLNKVCacheManager

        sub_block_size = self.scheduler_config.max_num_batched_tokens
        if not RBLNKVCacheManager.can_use_sub_block_caching(
            self.model_runner.kv_cache_config, sub_block_size
        ):
            return 0
        return DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES

    def compute_dynamic_kv_num_blocks(self) -> int | None:
        """How many KV blocks fit this device, from the compiled placement and a
        memory snapshot taken with the compile-time cache resident.

        Runs after warm-up and reallocates nothing; the engine takes the minimum
        across ranks and hands it back through `apply_dynamic_kv_num_blocks`.
        None means the path is not in play.
        """
        dry_run = envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN
        if not dry_run and self.cache_config.num_gpu_blocks_override is not None:
            logger.info(
                "[Dynamic KV] --num-gpu-blocks-override=%d is set; leaving the "
                "KV cache alone.",
                self.cache_config.num_gpu_blocks_override,
            )
            return None
        if not dry_run and self._kv_blocks_before_shrink is None:
            # The branch that cancelled the shrink already logged why.
            logger.warning(
                "[Dynamic KV] the cache was not shrunk, so no placement is queried "
                "and the count stays at the %d blocks vllm estimated.",
                self.model_runner.kv_cache_config.num_blocks,
            )
            return None
        if dry_run and not self._dynamic_kv_programs:
            logger.warning(
                "[Dynamic KV] dry run: no compiled program was captured, so there is "
                "nothing to report."
            )
            return None

        if torch.rbln.is_dummy_device():
            # A compile-only run has no device memory to size against; the
            # count stays at the estimate and the device run sizes for real.
            logger.warning(
                "[Dynamic KV] RBLN_DUMMY_DEVICE is set, so there is no device to "
                "measure; keeping the %d blocks vllm estimated for this compile-only "
                "run.",
                self._kv_blocks_before_shrink
                or self.model_runner.kv_cache_config.num_blocks,
            )
            return None

        if dry_run:
            try:
                num_blocks, fits, hint_blocks, growth = (
                    self._dynamic_kv_num_blocks_from_placement()
                )
            except RuntimeError as exc:
                logger.warning(
                    "[Dynamic KV] dry run: the count could not be computed (%s); "
                    "nothing is resized.",
                    exc,
                )
                return None
            self._log_dynamic_kv_dry_run(num_blocks, fits, hint_blocks, growth)
            return None
        num_blocks, _, _, _ = self._dynamic_kv_num_blocks_from_placement()
        return num_blocks

    def _dynamic_kv_num_blocks_from_placement(
        self,
    ) -> tuple[int, dict[Unit, UnitFit], int, KvGrowth]:
        """The count that fits, the per-unit fit behind it, the hint the
        programs were traced with, and the growth they imply."""
        programs = list(self._dynamic_kv_programs)
        groups = select_kv_input_groups(programs)
        hint_blocks = self.model_runner.kv_cache_config.num_blocks
        # The only record of the per-shard extents.
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
        growth = kv_growth(specs, hint_blocks)

        program = groups[0][1]
        device = program.device if program.device is not None else self.device
        snapshot, source = self._dynamic_kv_memory_snapshot(device)
        logger.info(
            "[Dynamic KV] %s memory snapshot of %s: %s",
            source,
            device,
            {f"{n}:{c}": (m.total, m.used) for (n, c), m in sorted(snapshot.items())},
        )

        # `used` includes the compile-time KV cache. TP=1 gives it back at the
        # reallocation, so it is not base; TP>=2 does not, and the process cannot
        # observe that it did not, so it stays charged. DP+EP keeps it resident
        # at tp_size=1 and slips through; see docs/dynamic_kv_cache.md.
        tp_size = self.parallel_config.tensor_parallel_size
        kv_resident = growth.allocated_at(hint_blocks) if tp_size <= 1 else {}
        if tp_size > 1:
            logger.info(
                "[Dynamic KV] tp=%d keeps the %d-block compile cache resident; "
                "charged as base. Expect fewer blocks than TP=1.",
                tp_size,
                hint_blocks,
            )

        reserve_bytes = self._kv_copy_stream_reserve_bytes()
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
        self._dynamic_kv_expected_used = {
            unit: fits[unit].base + predicted[unit] for unit in predicted
        }
        return num_blocks, fits, hint_blocks, growth

    def _log_dynamic_kv_fit_check(self, num_blocks: int) -> None:
        """Measured `used` against what the sizing predicted, once the resized
        cache is physically allocated; the allocator's behaviour shows up here."""
        snapshot, source = self._dynamic_kv_memory_snapshot(self.device)
        parts = []
        for (node, chiplet), expected in sorted(self._dynamic_kv_expected_used.items()):
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

    def _log_dynamic_kv_dry_run(
        self,
        num_blocks: int,
        fits: Mapping[Unit, UnitFit],
        current: int,
        growth: KvGrowth,
    ) -> None:
        """How the `current` blocks vllm sized sit in each chiplet's budget next
        to the `num_blocks` this feature would pick. Resizes nothing."""
        allocated = growth.allocated_at(current)
        per_unit = []
        for (node, chiplet), fit in sorted(fits.items()):
            kv_now = allocated[(node, chiplet)]
            headroom = fit.budget - fit.base - kv_now
            per_unit.append(
                f"{node}:{chiplet}(kv_now={kv_now} base={fit.base} budget={fit.budget} "
                f"headroom={headroom} = {headroom // fit.per_block:+d} blocks)"
            )
        needed = -(-self.model_config.max_model_len // self.cache_config.block_size)
        logger.warning(
            "[Dynamic KV] dry run: vllm sized %d blocks, this feature would set %d "
            "(%+d); one request of max_model_len=%d needs %d blocks, so the count "
            "%s. Per (node, chiplet): %s. Nothing is resized.",
            current,
            num_blocks,
            num_blocks - current,
            self.model_config.max_model_len,
            needed,
            "would be accepted" if num_blocks >= needed else "would be REFUSED",
            " ".join(per_unit),
        )

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
            if torch.rbln.is_dummy_device():
                # The dummy UMD still enforces its memory limit; nothing runs
                # after warm-up here, so the compile cache is enough.
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
        self._reallocate_kv_cache(target)
        self._materialize_kv_cache()
        if n is not None and self._dynamic_kv_expected_used:
            self._log_dynamic_kv_fit_check(target)
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
                with self._capture_dynamic_kv_programs() as programs:
                    self.model_runner.warmup_model()
                if programs is not None:
                    self._dynamic_kv_programs.extend(programs)
                    logger.info(
                        "[Dynamic KV] captured %d compiled program(s) during warm-up.",
                        len(programs),
                    )

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
    @worker_fail_fast
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def _send_handoff(self, tensors: dict) -> None:
        """Hand this stage's output on; a seam the metrics patch wraps."""
        # NOTE(RBLN): DO NOT all_gather_group for RBLN pp
        get_pp_group().send_tensor_dict(tensors)

    @torch.inference_mode()
    @worker_fail_fast
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        intermediate_tensors = None

        if (
            scheduler_output.total_num_scheduled_tokens > 0
            and not get_pp_group().is_first_rank
        ):
            intermediate_tensors = self.model_runner.recv_intermediate_tensors()

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

    @worker_fail_fast
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
