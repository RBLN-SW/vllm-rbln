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
import os
import time
from types import NoneType
from typing import TYPE_CHECKING

import numba
import torch

try:
    import torch.rbln

    has_torch_rbln = True
except ImportError:
    has_torch_rbln = False

import torch.nn as nn
from torch._dynamo.exc import BackendCompilerFailed
from vllm.config import VllmConfig, set_current_vllm_config
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
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
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

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner
from vllm_rbln.v1.worker.utils import (
    device_dram_bytes,
    estimate_available_memory,
    estimate_model_kernel_size,
    get_rbln_planned_affinity_cpu_count,
    set_cpu_affinity,
    set_omp_num_threads,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput


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
        self.device = self.device_config.device

        self.local_world_size = (
            self.parallel_config.world_size // envs.VLLM_RBLN_NUM_RAY_NODES
        )

        self._init_device_env()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self._rbln_host_threads_before_compile_ready = False
        self._rbln_cpu_affinity_applied = False
        # num_blocks vllm sized the KV cache with, stashed while the cache is
        # temporarily shrunk for compilation. None when no shrink is pending.
        self._kv_blocks_after_compile: int | None = None

        profiler_config = vllm_config.profiler_config
        # Set up profiler if profiling is enabled
        if profiler_config.torch_profiler_dir:
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                profiler_config.torch_profiler_dir,
            )
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s,use_gzip=%s",
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
                profiler_config.torch_profiler_use_gzip,
            )
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=f"{vllm_config.instance_id}-rank-{self.rank}",
                local_rank=self.local_rank,
                activities=["CPU"],
            )
        else:
            self.profiler = None

        self.parallel_config.disable_custom_all_reduce = True

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on RBLN, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on RBLN, ignore it.")
        pass

    def _init_device_env(self) -> None:
        world_size = self.local_world_size
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

    def init_device(self) -> None:
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
            self.vllm_config,
            self.device,
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

        num_runtimes = 1 + (1 + specialized_moe_decode) * decode_batch_buckets_count

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
                    raise ValueError(
                        f"compressed-tensors config has mixed bit-widths "
                        f"{num_bits_set}; not supported."
                    )

                if num_bits == 8:
                    # fp8 weights stay 1 byte/elem (float8) on device; counted in
                    # the floating-point branch below. Reuse the "fp8" path.
                    quantization = "fp8"
                elif num_bits == 4:
                    # w4a16 (compressed-tensors int4) via
                    # RBLNInt8UnpackedLinearKernel. Weights are unpacked to int8
                    # host params but stored 4-bit on device (RBLN_QUANT_BITS=4).
                    quantization = "int4"
                else:
                    raise ValueError(
                        f"compressed-tensors num_bits={num_bits} not supported; "
                        f"only 4-bit (int4) or 8-bit (fp8)."
                    )

            if quantization == "fp8":
                nbits_per_param = 8
                packed_num_elems = 1
            elif quantization == "int4":
                # Unpacked int8 weight param (1 logical elem per element) is
                # packed to 4-bit on device.
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
            spec_method = getattr(speculative_config, "method", None)
            if draft_quantization is not None and spec_method != "mtp":
                # MTP draft shares the target checkpoint and inherits its
                # quantization (e.g. fp8 for DeepSeek-V3),
                # Eagle/Medusa draft are separately-trained models and
                # quantized variants are not validated on RBLN yet.
                raise ValueError(
                    f"draft model quantization is not supported for "
                    f"method={spec_method}: {draft_quantization}"
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

        self.model_runner.initialize_kv_cache(
            self._maybe_shrink_kv_cache_for_compile(kv_cache_config)
        )

    def _maybe_shrink_kv_cache_for_compile(
        self, kv_cache_config: KVCacheConfig
    ) -> KVCacheConfig:
        """Return a `VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS`-sized copy of the
        config, or `kv_cache_config` unchanged when the env var is unset.

        The KV cache allocated here is the one `warm_up_model()` traces, so its
        num_blocks becomes the trace-time hint of the `mark_dynamic`'d dim and
        sizes the compiled artifact's device buffers. Allocating the real cache
        (thousands of blocks, from `determine_available_memory()`) makes those
        buffers huge for no reason: the dim is dynamic, so the runtime resizes
        them anyway. Compile against a small cache instead and put the real one
        back afterwards -- `_maybe_recompute_kv_blocks_from_compiled_profile()`
        grows it to the device maximum, or
        `_maybe_restore_kv_blocks_after_compile()` restores vllm's own number.

        Warm-up is safe at this size: every dummy request points at block id 0
        (`_add_dummy_requests`), so nothing indexes past the shrunk cache.
        """
        compile_num_blocks = envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS
        if compile_num_blocks <= 0:
            return kv_cache_config
        if compile_num_blocks >= kv_cache_config.num_blocks:
            logger.warning(
                "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=%d is not below the %d "
                "blocks vllm sized the KV cache with; compiling as-is.",
                compile_num_blocks,
                kv_cache_config.num_blocks,
            )
            return kv_cache_config

        shrunk = copy.deepcopy(kv_cache_config)
        shrunk.num_blocks = compile_num_blocks
        # Each kv_cache_tensor's size is num_blocks * group.page_size_bytes;
        # scale proportionally, mirroring `_reallocate_kv_cache`.
        for kv_tensor in shrunk.kv_cache_tensors:
            kv_tensor.size = (
                kv_tensor.size * compile_num_blocks
            ) // kv_cache_config.num_blocks

        self._kv_blocks_after_compile = kv_cache_config.num_blocks
        logger.info(
            "Compiling with a %d-block KV cache instead of %d "
            "(VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS); the cache is resized "
            "after warm-up.",
            compile_num_blocks,
            kv_cache_config.num_blocks,
        )
        return shrunk

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

    def compile_or_warm_up_model(self) -> CompilationTimes:
        st = time.perf_counter()
        if self.parallel_config.data_parallel_size > 1:
            if envs.VLLM_RBLN_DP_IMPL == "padded_decode":
                max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
                max_num_seqs = self.scheduler_config.max_num_seqs
                # TODO: consider relaxing this constraint
                assert max_num_batched_tokens % max_num_seqs == 0, (
                    "max_num_batched_tokens must be divisible by max_num_seqs"
                )
            elif envs.VLLM_RBLN_DP_IMPL == "dummy_prefill":
                raise ValueError(
                    "dummy_prefill is not supported in v1 worker"
                    "and will be deprecated in the future"
                )
            self.model_runner.prepare_dummy_run()

        # Thread policy + RBLN_NUM_THREADS before compile/warm-up; affinity after.
        self._ensure_rbln_host_threads_before_compile()

        if (
            self.model_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
            or not envs.VLLM_RBLN_ENABLE_WARM_UP
        ):
            logger.warning("skipping compile_or_warm_up_model")

            self._ensure_rbln_cpu_affinity_after_warmup()
            return CompilationTimes(
                language_model=time.perf_counter() - st, encoder=0.0
            )
        else:
            try:
                self.model_runner.warm_up_model()

            except BackendCompilerFailed as e:

                def is_oom(exc):
                    if isinstance(exc, RuntimeError):
                        for arg in exc.args:
                            if isinstance(arg, str) and (
                                "SYS_ENOMEM: Out of memory" in arg
                                or "SYS_EBUSY: Lack of device memory" in arg
                            ):
                                return True
                    return False

                if is_oom(e.inner_exception):
                    raise RuntimeError(
                        "Not enough memory for "
                        f"{self.model_runner.kv_cache_config.num_blocks} "
                        "blocks of KV cache. Try reducing the number of blocks "
                        "by setting --num-gpu-blocks-override."
                    ) from e

                raise

        # After compile/warm-up: if dynamic KV is enabled, query the rbln
        # executor's actual per-block bytes and reallocate KV cache with
        # the maximized num_blocks (replaces the manual estimate that was
        # used at initialize_from_config time). No-op when the env var is
        # off or the model was not compiled with a mark_dynamic'd KV.
        self._maybe_recompute_kv_blocks_from_compiled_profile()

        # If the cache was shrunk just for the compile and nothing above
        # resized it, put back the num_blocks vllm sized it with.
        self._maybe_restore_kv_blocks_after_compile()

        # After warm-up: apply CPU affinity only (threads already set pre-compile).
        self._ensure_rbln_cpu_affinity_after_warmup()
        self.model_runner._enable_performance_tracker()

        # TODO(RBLN): support encoder's compilation time
        return CompilationTimes(language_model=time.perf_counter() - st, encoder=0.0)

    def _maybe_restore_kv_blocks_after_compile(self) -> None:
        """Undo `_maybe_shrink_kv_cache_for_compile()`.

        No-op unless the KV cache was shrunk for the compile and still sits at
        that size -- `_reallocate_kv_cache()` clears the stashed target, so a
        dynamic-KV resize that already ran is never overwritten here.
        """
        target_num_blocks = self._kv_blocks_after_compile
        if target_num_blocks is None:
            return
        self._kv_blocks_after_compile = None

        logger.info(
            "Restoring the KV cache to the %d blocks vllm sized it with "
            "(compiled with %d).",
            target_num_blocks,
            self.model_runner.kv_cache_config.num_blocks,
        )
        self._reallocate_kv_cache(target_num_blocks)

    def _maybe_recompute_kv_blocks_from_compiled_profile(self) -> None:
        """If VLLM_RBLN_USE_DYNAMIC_KV_CACHE is enabled, query each
        compiled artifact's `kv_cache_memory_profile()` for the exact
        per-block bytes (incl. real chiplet sharding / 2 MiB alignment)
        and reallocate the KV cache with a maximized num_blocks.

        Must be called AFTER `warm_up_model()` so the rbln executor
        exists in `model_runner.runtime_holder`.
        """
        if not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            return

        runtime_holder = getattr(self.model_runner, "runtime_holder", None)
        if not runtime_holder:
            logger.warning(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE is set but "
                "runtime_holder is empty — skipping KV reallocation."
            )
            return

        # rebel.kv_cache.max_num_blocks models each compiled KV allocation
        # region as an affine function of num_blocks
        # (base_bytes + bytes_per_block * n) plus a per-region device
        # alignment, and returns the largest n whose aligned footprint fits the
        # supplied per-(node, chiplet) budget. Imported lazily: this path only
        # runs once the model is compiled, so rebel is guaranteed available.
        from rebel.kv_cache import max_num_blocks

        kv_cache_config = self.model_runner.kv_cache_config
        old_num_blocks = kv_cache_config.num_blocks
        if old_num_blocks <= 0:
            logger.warning("Existing num_blocks <= 0; skipping reallocation.")
            return

        # The budget must be the *whole* device DRAM, not vllm's KV-only estimate.
        # The compiled profile covers everything the artifact allocates -- weights,
        # constants, command streams -- and max_num_blocks subtracts those
        # base_bytes itself. Feeding it vllm's estimate (which already had the
        # weights and per-runtime buffers taken out) charged the same static
        # footprint twice and shrank num_blocks well below what the device holds.
        total_device_bytes = device_dram_bytes(
            self.cache_config.gpu_memory_utilization
        )
        if total_device_bytes <= 0:
            logger.warning("Device DRAM budget <= 0; skipping reallocation.")
            return

        # All compiled artifacts share the same dynamic-KV layout (same
        # mark_dynamic hint / same dynamic_shape_info JSON). Ask each artifact's
        # profile for the max num_blocks that fits the device and take the most
        # constraining (min) across artifacts. min() is right for what the
        # artifacts share (the KV tensors and the weights they all point at), but
        # it does not add up the private regions each artifact holds on its own
        # (command streams, IO buffers -- ~0.1 GiB apiece here). The profile does
        # not expose region identity, so they cannot be de-duplicated; the
        # gpu_memory_utilization headroom absorbs them for now.
        # TODO(joonsoo): Verify whether this code works on multi-chiplet environment.
        # TODO(joonsoo): Account for per-artifact private regions across runtimes.
        new_num_blocks: int | None = None
        for runtime in runtime_holder:
            executor = getattr(runtime, "_executor", None)
            if executor is None:
                continue
            try:
                profile = executor.kv_cache_memory_profile()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "kv_cache_memory_profile() failed (%s); skipping "
                    "dynamic KV reallocation.",
                    e,
                )
                return

            # `total_device_bytes` covers every device this rank owns. A device's
            # chiplets share one DRAM pool, so budget per node (chiplet-agnostic):
            # max_num_blocks(per_node_budget=True) pools every chiplet's regions
            # on a node against a single per-node budget. Split the DRAM budget
            # evenly across the nodes the profile uses (one node in the common
            # single-device case, so each node gets the full budget).
            node_ids = {r.node_id for r in profile.device_regions}
            per_node_bytes = total_device_bytes // max(1, len(node_ids))
            available_device_bytes: dict[int, int] = {
                node_id: per_node_bytes for node_id in node_ids
            }

            try:
                fit = max_num_blocks(
                    profile, available_device_bytes, per_node_budget=True
                )
            except ValueError:
                # No per-block growth → this artifact was not compiled with a
                # mark_dynamic'd KV dim, so num_blocks is not memory-bounded.
                logger.warning(
                    "[Dynamic KV] compiled profile has no per-block memory "
                    "growth; skipping reallocation."
                )
                return

            logger.warning(
                "[Dynamic KV] compiled profile: device_regions=%s "
                "host_base_bytes=%s host_bytes_per_block=%s -> max_num_blocks=%d",
                profile.device_regions,
                profile.host_base_bytes,
                profile.host_bytes_per_block,
                fit,
            )
            new_num_blocks = fit if new_num_blocks is None else min(new_num_blocks, fit)

        if not new_num_blocks or new_num_blocks <= 0:
            logger.warning(
                "[Dynamic KV] no fittable num_blocks from compiled artifacts; "
                "skipping reallocation."
            )
            return

        logger.warning(
            "[Dynamic KV] device_budget_bytes=%d, old_num_blocks=%d, "
            "computed_num_blocks=%d",
            total_device_bytes,
            old_num_blocks,
            new_num_blocks,
        )

        # The scheduler's block pool was built from what
        # `determine_available_memory()` reported, before the cache was shrunk
        # for the compile; it never learns about this reallocation. Extra blocks
        # are simply left unused, but a smaller cache means the scheduler can
        # hand out block ids that no longer exist.
        scheduler_num_blocks = self._kv_blocks_after_compile or old_num_blocks
        if new_num_blocks < scheduler_num_blocks:
            logger.warning(
                "[Dynamic KV] computed_num_blocks=%d is BELOW the %d blocks the "
                "scheduler was sized with; it may reference blocks the cache "
                "does not have.",
                new_num_blocks,
                scheduler_num_blocks,
            )

        if new_num_blocks == old_num_blocks:
            logger.warning(
                "[Dynamic KV] computed_num_blocks=%d == old_num_blocks; "
                "no reallocation needed.",
                new_num_blocks,
            )
            self._kv_blocks_after_compile = None
            return

        self._reallocate_kv_cache(new_num_blocks)

    def _reallocate_kv_cache(self, new_num_blocks: int) -> None:
        """Re-allocate the model_runner's KV cache tensors with `new_num_blocks`.

        Only the KV-cache *tensors* are rebuilt: num_blocks changes their byte
        size, and `mark_dynamic` is re-applied by
        `RBLNModelRunner.initialize_kv_cache_tensors`. No torch.compile recompile
        occurs because the affected dim is mark_dynamic'd, and the rbln runtime's
        `apply_adaptive_size_buffers` picks up the new shape on the next forward.

        We deliberately do NOT call the full `initialize_kv_cache()` again: the
        attention backends, metadata builders and input batch depend on
        block_size / max_model_len, not on the total num_blocks, and are already
        initialized. Re-running them trips
        `initialize_attn_backend`'s `assert len(self.attn_groups) == 0`.
        """
        mr = self.model_runner
        old_cfg = mr.kv_cache_config
        old_num_blocks = old_cfg.num_blocks
        # This resize supersedes any pending compile-shrink restore.
        self._kv_blocks_after_compile = None

        new_cfg = copy.deepcopy(old_cfg)
        new_cfg.num_blocks = new_num_blocks
        # Each kv_cache_tensor's size is num_blocks * group.page_size_bytes.
        # Scale proportionally so all derived fields stay consistent.
        for kv_tensor in new_cfg.kv_cache_tensors:
            kv_tensor.size = (kv_tensor.size * new_num_blocks) // old_num_blocks

        self.cache_config.num_gpu_blocks = new_num_blocks
        self.cache_config.num_cpu_blocks = new_num_blocks

        logger.warning(
            "[Dynamic KV] reallocating KV cache: %d → %d blocks",
            old_num_blocks,
            new_num_blocks,
        )
        mr.kv_cache_config = new_cfg
        # bind_kv_cache() asserts the runner's kv_caches list starts empty, so
        # clear the bindings from the initial allocation before re-binding.
        mr.kv_caches = []
        mr.initialize_kv_cache_tensors(new_cfg, mr.kernel_block_sizes)

        # The warm-up run already latched each rbln runtime's adaptive buffer
        # sizes at the pre-reallocation num_blocks. Clear that latch so the next
        # forward re-applies apply_adaptive_size_buffers at the new num_blocks
        # (which also rewrites + refreshes the per-input device_config).
        for runtime in getattr(mr, "runtime_holder", None) or []:
            reset = getattr(runtime, "reset_adaptive_buffers", None)
            if reset is not None:
                reset()

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
            # NOTE - DO NOT all_gather_group for RBLN pp
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

        # NOTE - DO NOT all_gather_group for RBLN pp
        get_pp_group().send_tensor_dict(output.tensors)
        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if (
            not kv_connector_output.finished_sending
            and not kv_connector_output.finished_recving
        ):
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            # only print profiler results on rank 0
            if self.local_rank == 0:
                print(
                    self.profiler.profiler.key_averages().table(
                        sort_by="self_cpu_time_total"
                    )
                )

    def execute_dummy_batch(self) -> None:
        self._ensure_rbln_host_threads_before_compile()
        self.model_runner.dummy_run()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def shutdown(self) -> None:
        logger.info("v1 rbln_worker shutdown called")
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        if envs.VLLM_RBLN_METRICS:
            if self.model_runner.performance_tracker:
                self.model_runner.performance_tracker.print_final_stats()
            if self.model_runner.sampler_performance_tracker:
                self.model_runner.sampler_performance_tracker.print_final_stats()
            if self.model_runner.e2e_performance_tracker:
                self.model_runner.e2e_performance_tracker.print_final_stats()


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
