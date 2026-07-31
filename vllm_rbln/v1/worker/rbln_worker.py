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
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_rbln.envs as envs
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
    set_cpu_affinity,
    set_omp_num_threads,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput


def empty_rbln_device_caches() -> bool:
    """Return every *free* block the rbln caching allocator holds to the driver.

    Without this a freed tensor's block stays reserved -- the allocator only
    releases cached blocks as a *retry* after an allocation fails -- so the bytes
    keep counting against the chiplet in sysfs `dram_used`. Only blocks that were
    never split are freed, and it is a no-op for a device this process never
    allocated on. Never raises: this runs during start-up.
    """
    if not has_torch_rbln:
        return False
    try:
        # is_available() raises on a malformed RBLN_* config, hence the guard.
        if not torch.rbln.is_available():
            return False
        device_count = torch.rbln.device_count()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "could not query the rbln devices to empty their allocator caches: "
            "%s. Memory freed by the dynamic-KV resize stays reserved by the "
            "caching allocator and keeps counting against the chiplet budget.",
            exc,
        )
        return False

    for index in range(device_count):
        try:
            torch.rbln.empty_cache(index)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "torch.rbln.empty_cache(%d) failed: %s. Freed KV blocks stay "
                "reserved by the caching allocator and keep counting against "
                "the chiplet's memory budget.",
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

        # --- dynamic KV cache (VLLM_RBLN_USE_DYNAMIC_KV_CACHE) ---
        # num_blocks vLLM sized the cache with, stashed while it is shrunk for
        # the compile. None when no shrink is pending.
        self._kv_blocks_before_shrink: int | None = None
        # Runtimes whose profile fed the block-count answer. Each must have its
        # adaptive-size latch cleared after the reallocation or its next forward
        # raises inside the rbln runtime.
        self._dynamic_kv_profiled_runtime_ids: set[int] = set()
        # Other tenants' device DRAM, sampled before this worker allocates
        # anything. Card-scope -- sysfs has no per-chiplet breakdown.
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
        params_dict = dict(self.model_runner.model.named_parameters())
        device_name = current_platform.get_device_name().lower()
        assert "rbln" in device_name

        specialized_moe_decode = int(self.model_runner.specialized_moe_decode)
        decode_batch_buckets_count = (
            self.model_runner.bucketing_manager.decode_batch_buckets_count
        )

        num_runtimes = 1 + decode_batch_buckets_count + specialized_moe_decode

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

        speculative_config = getattr(self, "speculative_config", None)
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

            num_draft_runtimes = 1 + decode_batch_buckets_count
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

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

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
            self._assert_dynamic_kv_scheduler_handoff_installed()
            self._assert_dynamic_kv_transfer_absent()

        self.model_runner.initialize_kv_cache(
            self._maybe_shrink_kv_cache_for_compile(kv_cache_config)
        )

        if dynamic_kv:
            self._assert_dynamic_kv_cache_layout()

    # ------------------------------------------------------------------
    # Dynamic KV cache (VLLM_RBLN_USE_DYNAMIC_KV_CACHE)
    # ------------------------------------------------------------------

    def _maybe_shrink_kv_cache_for_compile(
        self, kv_cache_config: KVCacheConfig
    ) -> KVCacheConfig:
        """Return a `VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS`-sized copy of the
        config, or `kv_cache_config` unchanged when the shrink is not enabled.

        The cache allocated here is the one `warmup_model()` traces, so its
        `num_blocks` becomes the trace-time hint of the `mark_dynamic`'d dim.
        Building the artifact's buffers at the full size is pointless when the
        runtime resizes them anyway, and warm-up is safe small: every dummy
        request points at block id 0.

        The shrink is ANDed with `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` and with the
        *absence* of `--num-gpu-blocks-override`, because in both cases nothing
        would put the full cache back and the scheduler would hand out block ids
        the shrunk cache does not have.
        """
        compile_num_blocks = envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS
        if compile_num_blocks <= 0:
            return kv_cache_config
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            logger.warning(
                "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=%d is ignored because "
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE is off; without the dynamic KV "
                "path nothing would restore the full cache after the compile.",
                compile_num_blocks,
            )
            return kv_cache_config
        override = self.cache_config.num_gpu_blocks_override
        if override is not None:
            logger.warning(
                "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=%d is ignored because "
                "--num-gpu-blocks-override=%d pins the block count: the dynamic "
                "KV path then computes no new size, so nothing would restore "
                "the full cache after the compile and the server would serve "
                "from the %d-block compile cache while the scheduler hands out "
                "block ids for %d. Compiling at the pinned size instead.",
                compile_num_blocks,
                override,
                compile_num_blocks,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config
        if compile_num_blocks >= kv_cache_config.num_blocks:
            logger.warning(
                "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=%d is not below the %d "
                "blocks vllm sized the KV cache with; compiling as-is.",
                compile_num_blocks,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config

        shrunk = copy.copy(kv_cache_config)
        shrunk.kv_cache_tensors = copy.deepcopy(kv_cache_config.kv_cache_tensors)
        shrunk.num_blocks = compile_num_blocks
        # Each kv_cache_tensor's size is num_blocks * group.page_size_bytes;
        # scale proportionally so the tensor allocation stays consistent with
        # num_blocks (mirrors `_reallocate_kv_cache`).
        for kv_tensor in shrunk.kv_cache_tensors:
            kv_tensor.size = (
                kv_tensor.size * compile_num_blocks
            ) // kv_cache_config.num_blocks

        self._kv_blocks_before_shrink = kv_cache_config.num_blocks
        logger.info(
            "[Dynamic KV] compiling with a %d-block KV cache instead of %d "
            "(VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS); the cache is resized "
            "after warm-up.",
            compile_num_blocks,
            kv_cache_config.num_blocks,
        )
        return shrunk

    def _assert_dynamic_kv_scheduler_handoff_installed(self) -> None:
        """Guard: nothing may shrink the KV cache unless something will resize it.

        Both rpcs are only ever driven by the `EngineCore._initialize_kv_caches`
        patch. Without it the cache would stay at the compile-time size while the
        scheduler hands out block ids for the pre-compile estimate -- the exact
        failure this feature exists to remove.
        """
        if not envs.VLLM_RBLN_USE_VLLM_MODEL:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires "
                "VLLM_RBLN_USE_VLLM_MODEL=1: the optimum path uses runtimes "
                "that ignore adaptive buffer sizes, and the RBLN patch registry "
                "(which carries the scheduler hand-off) is not applied there."
            )

        from vllm.v1.engine.core import EngineCore

        from vllm_rbln.patches.dynamic_kv import patched_initialize_kv_caches

        if EngineCore._initialize_kv_caches is not patched_initialize_kv_caches:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE is set but the "
                "EngineCore._initialize_kv_caches patch is not installed, so "
                "nothing would query the compiled profile or tell the scheduler "
                "the new block count."
            )

        self._assert_dynamic_kv_compiler_support()

    def _assert_dynamic_kv_compiler_support(self) -> None:
        """Guard: the installed rebel-compiler must carry the #10678 API.

        `rebel.kv_cache.max_num_blocks` and `DynamoRuntime.reset_adaptive_buffers`
        both arrived in the same commit. Both are reached only *after* the model
        has compiled and warmed up, so on an older compiler the run pays the full
        compile and then dies on a bare ImportError / AttributeError. Probe them
        up front instead, before anything is compiled.
        """
        try:
            from rebel.kv_cache import max_num_blocks  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires a rebel-compiler that "
                "provides rebel.kv_cache.max_num_blocks (rebel_compiler #10678). "
                f"The installed one does not: {exc}"
            ) from exc

        try:
            from rebel.sync_runtime import DynamoRuntime
        except ImportError:
            # The class moved or is not importable here; the per-runtime type
            # check in _assert_dynamo_runtimes still covers the real objects.
            return
        # Defined on BaseRuntime, which DynamoRuntime inherits from.
        if not hasattr(DynamoRuntime, "reset_adaptive_buffers"):
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires a rebel-compiler whose "
                "DynamoRuntime provides reset_adaptive_buffers() "
                "(rebel_compiler #10678); without it the KV cache cannot be "
                "resized after warm-up."
            )

    def _assert_dynamic_kv_transfer_absent(self) -> None:
        """Guard: a KV connector and dynamic KV cannot be combined.

        `compile_or_warm_up_model` calls `finalize_kv_cache_registrations`, which
        hands the connector the KV cache's physical views. Reallocating the KV
        tensors right afterwards would leave those registrations pointing at
        freed memory.
        """
        if has_kv_transfer_group():
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE cannot be combined with a KV "
                "transfer connector: the connector registers the KV cache's "
                "physical views during warm-up and the dynamic-KV resize "
                "invalidates them."
            )

    def _assert_dynamic_kv_cache_layout(self) -> None:
        """Guard: the KV layout must satisfy the compiler's dynamic-input rules.

        The compiler requires every `mark_dynamic`'d KV input to have exactly one
        use, and that use to be a
        `paged_flash_causal_attention_naive_{prefill,decode}` call. Each item
        below breaks half of that invariant and each fails *silently* -- a no-op
        `mark_dynamic`, or a confusing 'has N uses' rejection -- unless checked.
        """
        mr = self.model_runner

        if mr.kv_cache_bases:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires KV base deduplication "
                f"to be inactive, but {len(mr.kv_cache_bases)} deduped bases "
                "were built. The per-layer kv_cache would then be a view "
                "created inside the traced graph, which makes mark_dynamic a "
                "no-op, and marking the base itself is rejected by the "
                "compiler ('kind is not call_arg')."
            )

        if mr.shared_kv_cache_layers:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support cross-layer KV "
                f"sharing, but {len(mr.shared_kv_cache_layers)} layer(s) reuse "
                "another layer's KV cache. The same tensor object reaching two "
                "attention calls gives the marked input two uses, which the "
                "compiler's dynamic-input validator rejects."
            )

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
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires every attention layer "
                "to dispatch to paged_flash_causal_attention_naive_{prefill,"
                "decode}, i.e. sliding_window is None, is_causal is True and "
                "is_normal is False (is_normal becomes True when block_size == "
                "max_model_len). Offending layers: "
                + ", ".join(offenders[:8])
                + (f" (+{len(offenders) - 8} more)" if len(offenders) > 8 else "")
            )

    def _collect_dynamic_kv_runtimes(self) -> list[Any]:
        """Every rbln runtime that holds a slice of the KV cache.

        `runtime_holder` is shared with the spec-decode drafter, and any extra
        holder on the drafter is folded in as well: a runtime missed here never
        gets its adaptive-size latch cleared, and its first forward after the
        resize dies inside the rbln runtime.
        """
        runtimes: list[Any] = []
        seen: set[int] = set()
        holders = [getattr(self.model_runner, "runtime_holder", None)]
        drafter = getattr(self.model_runner, "drafter", None)
        if drafter is not None:
            holders.append(getattr(drafter, "runtime_holder", None))
        for holder in holders:
            for runtime in holder or []:
                if id(runtime) in seen:
                    continue
                seen.add(id(runtime))
                runtimes.append(runtime)
        return runtimes

    def _assert_dynamo_runtimes(self, runtimes: list[Any]) -> None:
        """Guard: adaptive buffer sizes are a `DynamoRuntime`-only feature.

        The other runtime classes ignore the resize without a word, so the server
        would run on the compile-time cache while the scheduler hands out the new
        block count.
        """
        offenders = [
            type(runtime).__name__
            for runtime in runtimes
            if type(runtime).__name__ != "DynamoRuntime"
        ]
        if offenders:
            raise RuntimeError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires every KV-holding "
                "runtime to be a rebel DynamoRuntime (the torch.compile path, "
                "VLLM_RBLN_USE_VLLM_MODEL=1). Found: "
                f"{sorted(set(offenders))}. Other runtime classes silently "
                "ignore adaptive buffer sizes."
            )

    def _dynamic_kv_device_topology(self, runtimes: list[Any]) -> tuple[int, int]:
        """`(num_nodes, num_chiplets)` from a runtime's allocation report.

        Taken from `get_alloc_per_chiplet(1)`'s shape rather than the memory
        profile: deriving the budget keys from the profile would make the
        coverage assert circular.
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
        raise RuntimeError(
            "no rbln runtime reported a per-chiplet allocation shape; cannot "
            "determine the (node, chiplet) grid the KV budget must cover."
        )

    def _dynamic_kv_chiplet_budget(self, num_chiplets: int) -> int:
        """Per-chiplet byte budget for `max_num_blocks`.

        `floor(dram_total / num_chiplets * gmu)` minus the unprofiled reserve
        minus other tenants' usage. Deliberately *capacity*-derived rather than a
        measured free figure: `max_num_blocks` subtracts the profile's
        `base_bytes` itself, and those are already resident by now, so a measured
        free value would charge the base twice.
        """
        dram_total = read_rbln_card_dram_total_bytes()
        if dram_total is None:
            raise RuntimeError(
                "cannot read per-card DRAM capacity from sysfs "
                "(/sys/class/rebellions/rbln*/dram_total); "
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE needs it to budget per chiplet."
            )
        capacity_per_chiplet = dram_total // num_chiplets
        gmu = self.cache_config.gpu_memory_utilization
        reserve = envs.VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES
        if reserve < 0:
            raise RuntimeError(
                "VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES must not be "
                f"negative (got {reserve}); a negative reserve would hand the KV "
                "cache more memory than gpu_memory_utilization allows."
            )
        gmu_budget = int(capacity_per_chiplet * gmu)
        raw_budget = gmu_budget - reserve
        # Card-scope figure against a per-chiplet budget, so it must be scaled.
        # Charging the whole card total to every chiplet drove the budget negative
        # and aborted start-up on a configuration that serves fine once scaled.
        # An even spread is an assumption, but the same one `capacity_per_chiplet`
        # above already makes.
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
            reserve,
            raw_budget,
            self._foreign_dram_used_bytes,
            foreign_per_chiplet,
            budget,
        )
        if budget <= 0:
            raise RuntimeError(
                f"per-chiplet KV budget is non-positive ({budget} bytes) after "
                f"subtracting {reserve} bytes of unprofiled reserve and "
                f"{foreign_per_chiplet} bytes of foreign device usage "
                f"({self._foreign_dram_used_bytes} bytes across "
                f"{num_chiplets} chiplets)."
            )
        return budget

    def compute_dynamic_kv_num_blocks(self) -> int | None:
        """Ask the compiled artifacts how many KV blocks fit this device.

        Runs after warm-up and reallocates nothing: the engine collects one
        answer per rank, takes the minimum, and hands it back through
        `apply_dynamic_kv_num_blocks`.

        Returns:
            The block count that fits, or None when the path is not in play or
            the answer cannot be trusted.
        """
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            return None
        if self.cache_config.num_gpu_blocks_override is not None:
            logger.info(
                "[Dynamic KV] --num-gpu-blocks-override=%d is set; leaving the "
                "KV cache alone.",
                self.cache_config.num_gpu_blocks_override,
            )
            return None
        if self._kv_blocks_before_shrink is None:
            logger.warning(
                "[Dynamic KV] the KV cache was not shrunk for the compile "
                "(VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=%d); skipping the "
                "profile query.",
                envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS,
            )
            return None

        runtimes = self._collect_dynamic_kv_runtimes()
        if not runtimes:
            logger.error(
                "[Dynamic KV] no rbln runtime was registered during warm-up; "
                "cannot query the compiled memory profile."
            )
            return None
        self._assert_dynamo_runtimes(runtimes)

        profiles = []
        profiled_ids: set[int] = set()
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
                    "runtime (%s): %s. Continuing with the remaining runtimes; "
                    "the holder legitimately contains graphs without KV inputs.",
                    reason,
                    message,
                )
                continue
            profiles.append(profile)
            profiled_ids.add(id(runtime))
            # The full profile, verbatim, one line per artifact: the only record
            # of base_bytes / bytes_per_block / alignment, since neither sysfs nor
            # the runtime's allocation log exposes the per-block cost. Without it
            # a finished run cannot be re-analysed offline. The identity fields go
            # after the region list so a reader that splits at `device_regions=[`
            # can still say which artifact the regions came from.
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
            logger.error(
                "[Dynamic KV] not one of the %d rbln runtimes returned a KV "
                "memory profile; the model was probably not compiled with a "
                "mark_dynamic'd KV cache.",
                len(runtimes),
            )
            return None

        merged = merge_kv_cache_memory_profiles(profiles)
        # `merged_regions=`, not `device_regions=`: a reader scanning for the
        # latter would otherwise pick the merge up as one more source profile.
        logger.info(
            "[Dynamic KV] merged profile: %s rank=%d num_source_profiles=%d "
            "dedup=%s num_regions=%d shared_base=%d private_base=%d growth=%d",
            format_profile_for_log(merged, MERGED_PROFILE_LOG_KEY),
            self.rank,
            merged.num_source_profiles,
            merged.dedup_strategy,
            len(merged.device_regions),
            merged.num_shared_base_regions,
            merged.num_private_base_regions,
            merged.num_growth_regions,
        )
        num_nodes, num_chiplets = self._dynamic_kv_device_topology(runtimes)
        budget_per_chiplet = self._dynamic_kv_chiplet_budget(num_chiplets)
        budget = build_per_chiplet_budget(num_nodes, num_chiplets, budget_per_chiplet)
        assert_budget_covers_profile(merged, budget)

        # `rebel/__init__.py` does not import `rebel.kv_cache`, so this exact
        # import form is the only reliable one. Lazy: only needed once a compiled
        # artifact exists.
        from rebel.kv_cache import max_num_blocks

        try:
            num_blocks = max_num_blocks(merged, budget, per_node_budget=False)
        except ValueError:
            logger.error(
                "[Dynamic KV] the merged profile has no per-block memory "
                "growth, i.e. the artifacts were NOT compiled with a dynamic KV "
                "dim. Check that mark_dynamic ran (VLLM_RBLN_USE_DYNAMIC_KV_"
                "CACHE must be set before the compile) and that the compile "
                "cache was not reused from a static build."
            )
            return None

        base_usage = per_chiplet_usage(merged, 0)
        logger.info(
            "[Dynamic KV] merged profile predicts base bytes per (node, chiplet): %s",
            {k: v for k, v in sorted(base_usage.items())},
        )
        if num_blocks <= 0:
            logger.error(
                "[Dynamic KV] max_num_blocks() returned 0: not even num_blocks=0 "
                "fits the per-chiplet budget of %d bytes. Predicted base usage "
                "%s.",
                budget_per_chiplet,
                {k: v for k, v in sorted(base_usage.items())},
            )
            return None

        fit_usage = per_chiplet_usage(merged, num_blocks)
        # Field names, not prose: a log reader keys on `computed_num_blocks` /
        # `chiplet_budget_bytes` to recover the answer and its budget.
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

        self._dynamic_kv_profiled_runtime_ids = profiled_ids
        return num_blocks

    def apply_dynamic_kv_num_blocks(self, n: int | None) -> int | None:
        """Resize the KV cache to the block count the engine settled on.

        Args:
            n: The cross-rank block count, or None to put back what vLLM sized
                the cache with before the shrink -- otherwise the server runs on
                the tiny compile cache while the scheduler believes the full
                number.

        Returns:
            The block count now in effect, or None when nothing was done.
        """
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            return None

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
            self._dynamic_kv_profiled_runtime_ids = set()
            return target

        if n is None:
            logger.warning(
                "[Dynamic KV] restoring KV cache to the %d blocks vllm sized it "
                "with (compiled with %d).",
                target,
                current,
            )
        self._reallocate_kv_cache(target)
        return target

    def _kv_cache_layer_names(self, kv_cache_config: KVCacheConfig) -> list[str]:
        """Every layer name `bind_kv_cache` would have bound for this config.

        `KVCacheTensor.shared_by` is the same list `_allocate_kv_cache_tensors`
        keys its output on, so this covers exactly the bound layers.
        """
        names: list[str] = []
        seen: set[str] = set()
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for layer_name in kv_cache_tensor.shared_by:
                if layer_name in seen:
                    continue
                seen.add(layer_name)
                names.append(layer_name)
        return names

    def _release_kv_cache_tensors(self, old_cfg: KVCacheConfig) -> None:
        """Drop every reference to the outgoing KV cache and free its device DRAM.

        Called *before* the replacement is allocated. `initialize_kv_cache_tensors`
        rebinds the runner list and the forward context, so otherwise the old
        tensors outlive the free and the allocator keeps their blocks reserved --
        measured at 4.5 GiB per card, the dominant term in that run's
        `gpu_memory_utilization` overshoot. Releasing first also caps the peak at
        `base + max(old, new)` rather than `base + old + new`.
        """
        mr = self.model_runner

        # Whether the outgoing cache was device-resident decides whether the
        # accounting below means anything: with VLLM_RBLN_USE_DEVICE_TENSOR off
        # the KV tensors live on `meta` and the device buffers belong to the rbln
        # runtime. Read from the tensors, not from the env, and before they go.
        kv_device_types = sorted(
            {
                getattr(getattr(kv_cache, "device", None), "type", None) or "unknown"
                for kv_cache in mr.kv_caches
            }
        )
        was_device_resident = bool(set(kv_device_types) - {"unknown", "meta", "cpu"})

        # (1) The runner's own bindings. Clearing kv_caches is required either
        # way -- upstream `bind_kv_cache` asserts the list starts empty -- and
        # kv_cache_names must stay in step with it because
        # `initialize_kv_cache_tensors` asserts equal length.
        mr.kv_caches = []
        mr.kv_cache_bases = []
        mr.kv_cache_names = []

        # (2) `bind_kv_cache` also parks each layer's view on the Attention
        # module in the static forward context. The next bind overwrites those
        # attributes, but only *after* the new tensors exist -- the window this
        # method closes. Nothing on RBLN reads the attribute, but the reference
        # is real. Read the context off the model runner, i.e. the same
        # expression that wrote it, so a rename breaks both at once.
        forward_context = mr.compilation_config.static_forward_context
        unbound = 0
        for layer_name in self._kv_cache_layer_names(old_cfg):
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

        # (3) Every per-layer tensor is a view and a view keeps its base alive,
        # so the storage only dies with the last one. A reference cycle would
        # defer the free past the replacement's allocation, losing the point.
        gc.collect()

        released = empty_rbln_device_caches()
        logical_bytes = sum(t.size for t in old_cfg.kv_cache_tensors)
        # Whether those bytes came back is not observable in-process -- the
        # allocator's reserved-bytes counter does not track them. Confirm from
        # the card's sysfs `dram_used` across the resize instead.
        logger.info(
            "[Dynamic KV] released the outgoing %d-block KV cache: "
            "outgoing_kv_logical_bytes=%d unbound_layers=%d kv_device_types=%s "
            "allocator_cache_emptied=%s device_resident=%s",
            old_cfg.num_blocks,
            logical_bytes,
            unbound,
            kv_device_types,
            released,
            was_device_resident,
        )

    def _reallocate_kv_cache(self, new_num_blocks: int) -> None:
        """Rebuild only the KV cache *tensors* at `new_num_blocks`.

        `initialize_kv_cache()` is deliberately not re-run: the attention
        backends, metadata builders and input batch depend on block_size /
        max_model_len rather than num_blocks, are already initialized, and
        `initialize_attn_backend` asserts `len(self.attn_groups) == 0`.

        No recompilation happens because the affected dim is `mark_dynamic`'d;
        the physical allocation happens on the next forward. The outgoing cache
        is released first -- nothing else in the process ever frees it.
        """
        mr = self.model_runner
        old_cfg = mr.kv_cache_config
        old_num_blocks = old_cfg.num_blocks
        assert old_num_blocks > 0, "cannot rescale a KV cache of 0 blocks"

        new_cfg = copy.copy(old_cfg)
        new_cfg.kv_cache_tensors = copy.deepcopy(old_cfg.kv_cache_tensors)
        new_cfg.num_blocks = new_num_blocks
        # Each kv_cache_tensor's size is num_blocks * group.page_size_bytes.
        for kv_tensor in new_cfg.kv_cache_tensors:
            kv_tensor.size = (kv_tensor.size * new_num_blocks) // old_num_blocks

        self.cache_config.num_gpu_blocks = new_num_blocks
        self.cache_config.num_cpu_blocks = new_num_blocks

        logger.warning(
            "[Dynamic KV] reallocating KV cache: %d -> %d blocks",
            old_num_blocks,
            new_num_blocks,
        )
        mr.kv_cache_config = new_cfg
        # Order is load-bearing: see `_release_kv_cache_tensors`. It also does
        # the `mr.kv_caches = []` that upstream bind_kv_cache() asserts on.
        self._release_kv_cache_tensors(old_cfg)
        # Re-applies mark_dynamic and calls bind_kv_cache itself.
        mr.initialize_kv_cache_tensors(new_cfg, mr._kernel_block_sizes)

        if mr.kv_cache_bases:
            raise RuntimeError(
                "KV base deduplication became active after the dynamic-KV "
                f"reallocation ({len(mr.kv_cache_bases)} bases); the "
                "mark_dynamic'd layer tensors are no longer graph inputs."
            )

        # Warm-up latched every runtime's adaptive buffer sizes at the old
        # num_blocks. Clear it so the next forward re-applies them at the new
        # size. No getattr: a missing symbol must fail here, not on first request.
        runtimes = self._collect_dynamic_kv_runtimes()
        reset_ids: set[int] = set()
        for runtime in runtimes:
            runtime.reset_adaptive_buffers()
            reset_ids.add(id(runtime))
        missing = self._dynamic_kv_profiled_runtime_ids - reset_ids
        assert not missing, (
            f"{len(missing)} runtime(s) contributed a KV memory profile but "
            "were not reset after the reallocation; their first forward would "
            "raise 'variable dim changed after adaptive buffers were fixed'."
        )
        assert reset_ids, (
            "the KV cache was reallocated but no rbln runtime had its adaptive "
            "buffer latch cleared."
        )
        logger.info(
            "[Dynamic KV] reset_adaptive_buffers() on %d runtime(s) "
            "(%d contributed a profile).",
            len(reset_ids),
            len(self._dynamic_kv_profiled_runtime_ids),
        )
        self._dynamic_kv_profiled_runtime_ids = set()

    @instrument(span_name="Warmup (NPU)")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # NOTE(RBLN): Manual timing since RBLN does not support @support_torch_compile.
        st = time.perf_counter()

        # NOTE(RBLN): Thread policy + RBLN_NUM_THREADS must be set
        # before compile/warm-up. CPU affinity is applied afterward.
        self._ensure_rbln_host_threads_before_compile()

        try:
            if (
                self.model_config.enforce_eager
                or not envs.VLLM_RBLN_COMPILE_MODEL
                or not envs.VLLM_RBLN_ENABLE_WARM_UP
            ):
                logger.info("Skipping compile_or_warm_up_model.")
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
                raise RuntimeError(
                    "Not enough memory for "
                    f"{self.model_runner.kv_cache_config.num_blocks} "
                    "blocks of KV cache. Try reducing the number of blocks "
                    "by setting --num-gpu-blocks-override."
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

        # NOTE(RBLN): DO NOT all_gather_group for RBLN pp
        get_pp_group().send_tensor_dict(output.tensors)

        # For PP with a KV connector, surface the connector output the
        # model runner attached to the intermediate tensors so finished
        # send/recv notifications still propagate from non-last ranks.
        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None
        if (
            not kv_connector_output.finished_sending
            and not kv_connector_output.finished_recving
        ):
            return EMPTY_MODEL_RUNNER_OUTPUT

        empty_output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        empty_output.kv_connector_output = kv_connector_output
        return empty_output

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
        bucket_size = self.model_runner.bucketing_manager.find_decode_batch_bucket(1)
        query_len = 1 + self.model_runner.num_spec_tokens
        self.model_runner._dummy_run(bucket_size, query_len, is_prefill=False)

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
        self.model_runner.performance_ctx.print_stats()

        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

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
