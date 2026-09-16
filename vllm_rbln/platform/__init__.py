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

import os
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

import rebel
from torch._dynamo import register_backend
from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

import vllm_rbln.logger  # noqa: F401
from vllm_rbln import envs

logger = init_logger(__name__)
# Earliest point at which `vllm.envs` is guaranteed to exist, and still before
# any engine code reads a variable.
envs.publish_to_vllm_envs()

try:
    import torch.rbln  # noqa: F401

    HAS_TORCH_RBLN: bool = True
except ImportError:
    HAS_TORCH_RBLN = False

USE_DEVICE_TENSOR: bool = (
    envs.VLLM_RBLN_USE_VLLM_MODEL and envs.VLLM_RBLN_USE_DEVICE_TENSOR
)
# RBLN default for an unset max_num_seqs (upstream vLLM defaults to 256).
RBLN_DEFAULT_MAX_NUM_SEQS = 1
# Superseded by RblnPlatform.device_control_env_var.
DEPRECATED_DEVICE_CONTROL_ENV_VAR = "RBLN_DEVICES"


def bypass_backend(graph_module: torch.fx.GraphModule, example_inputs):
    return graph_module.forward


register_backend(name="bypass", compiler_fn=bypass_backend)


def _impl():
    """The module owning the selected model path.

    Imported lazily so the path modules can read this module's constants at
    their own import time, and one branch at a time so that neither path
    imports the other's module.
    """
    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        from vllm_rbln.platform import vllm_impl

        return vllm_impl

    from vllm_rbln.platform import optimum_impl

    return optimum_impl


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT

    # Compute device_name/device_type/dist_backend once at class definition
    # from env vars so that subprocesses spawned under
    # VLLM_WORKER_MULTIPROC_METHOD=spawn (which re-import this module fresh)
    # observe identical values to the parent without any extra plumbing.
    plugin_name: str = "rbln"
    device_name: str = "rbln" if USE_DEVICE_TENSOR else "cpu"
    device_type: str = "rbln" if USE_DEVICE_TENSOR else "cpu"
    dist_backend: str = "rbln-ccl" if USE_DEVICE_TENSOR else ""
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    device_control_env_var: str = "RBLN_VISIBLE_DEVICES"
    simple_compile_backend = "bypass"

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend is None:
            selected_backend = (
                AttentionBackendEnum.FLASH_ATTN_MLA
                if attn_selector_config.use_mla
                else AttentionBackendEnum.FLASH_ATTN
            )
        if selected_backend and selected_backend not in (
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASH_ATTN_MLA,
        ):
            raise ValueError(f"Cannot use {selected_backend} backend on RBLN.")

        if attn_selector_config.use_sparse and not attn_selector_config.use_mla:
            raise NotImplementedError("Sparse Attention is not supported on RBLN.")

        logger.info("Using %s Backend", selected_backend)

        return selected_backend.get_path()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # No NPU mounted (e.g., CPU-only compile worker): fall back to the env var
        # the compiler CI sets - RBLN_FORCE_NPU_NAME (RBLN_TARGET_SOC = legacy).
        device_name = (
            rebel.get_npu_name(device_id)
            or os.environ.get("RBLN_FORCE_NPU_NAME")
            or os.environ.get("RBLN_TARGET_SOC")
        )
        if not device_name:
            raise RuntimeError(
                "Could not determine the RBLN NPU name "
                f"(rebel.get_npu_name({device_id}) returned None). On a host "
                "without an NPU mounted (e.g., a CPU-only compile worker running "
                "with VLLM_RBLN_COMPILE_ONLY=1), set RBLN_FORCE_NPU_NAME to the "
                "target NPU (e.g., RBLN-CA25) so compilation can target it."
            )
        return device_name

    @classmethod
    def is_cr13(cls) -> bool:
        return cls.get_device_name().strip().upper() == "RBLN-CR13"

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        rebel.manual_seed(seed)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        logger.warning("set_device is not supported on RBLN.")
        pass

    @classmethod
    def _override_default_max_num_seqs(cls) -> None:
        """Default an unset max_num_seqs to RBLN_DEFAULT_MAX_NUM_SEQS.

        Wraps EngineArgs.get_batch_defaults() so RBLN's default applies to both
        `vllm serve` and `LLM(...)`. Explicit values are not None and untouched.
        """
        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_rbln_max_num_seqs_patched", False):
            return

        orig_get_batch_defaults = EngineArgs.get_batch_defaults.__func__

        def get_batch_defaults(cls_, world_size):
            from vllm.usage.usage_lib import UsageContext

            default_batched_tokens, _ = orig_get_batch_defaults(cls_, world_size)
            # Cover every usage context plus None (create_engine_config's
            # usage_context is UsageContext | None);
            # otherwise .get(ctx, DEFAULT_MAX_NUM_SEQS) falls through to 128.
            default_max_num_seqs = {
                ctx: RBLN_DEFAULT_MAX_NUM_SEQS for ctx in UsageContext
            }
            default_max_num_seqs[None] = RBLN_DEFAULT_MAX_NUM_SEQS
            return default_batched_tokens, default_max_num_seqs

        EngineArgs.get_batch_defaults = classmethod(get_batch_defaults)
        EngineArgs._rbln_max_num_seqs_patched = True

    @classmethod
    def _adopt_deprecated_device_control_env_var(cls) -> None:
        """Fold ``RBLN_DEVICES`` into ``device_control_env_var`` and unset it.

        The runtime takes both names but prefers ``RBLN_DEVICES``, so one left
        in the environment would override the pool a worker narrows itself to
        and put every rank on the same NPUs.
        """
        legacy = os.environ.pop(DEPRECATED_DEVICE_CONTROL_ENV_VAR, None)
        if legacy is None:
            return
        in_effect = os.environ.setdefault(cls.device_control_env_var, legacy)
        logger.warning_once(
            "%s is deprecated and will be removed in a future release. Please "
            "use %s instead; this run uses %s=%s.",
            DEPRECATED_DEVICE_CONTROL_ENV_VAR,
            cls.device_control_env_var,
            cls.device_control_env_var,
            in_effect,
        )

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        # Early enough that vLLM has read neither the device-control env var nor
        # max_num_seqs, which is still None here.
        cls._adopt_deprecated_device_control_env_var()
        cls._override_default_max_num_seqs()
        _impl().patch_upstream()

        if parser is None:
            return

        for action in parser._actions:
            if action.dest == "device":
                action.choices.append("rbln")

        for action in parser._actions:
            if action.dest == "block_size":
                action.choices = None  # Override choices

        # Not dispatched: the flags cover both paths, because which one runs is
        # not known until the arguments are parsed.
        from vllm_rbln.config import add_rbln_cli_args

        add_rbln_cli_args(parser)

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError(
                "VLLM_USE_V2_MODEL_RUNNER is not supported for RBLN backend."
            )

        # NOTE(RBLN): checked here, not inside the selected path module -- the
        # optimum path is exactly where an unsupported flag would go unnoticed.
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            cls._validate_dynamic_kv_config(vllm_config)

        _impl().check_and_update(vllm_config)

        parallel_config = vllm_config.parallel_config
        if parallel_config.distributed_executor_backend not in (None, "mp", "uni"):
            logger.warning(
                "%s is not supported on RBLN. Keeping the selected distributed "
                "executor backend; use 'mp' for supported multi-worker execution.",
                parallel_config.distributed_executor_backend,
            )

    @staticmethod
    def _validate_dynamic_kv_config(vllm_config: VllmConfig) -> None:
        """Reject configurations the dynamic-KV path cannot size.

        A dry run reports them instead: it changes nothing, so refusing would
        stop a run that the flag off would have served. Reasons per shape:
        docs/dynamic_kv_cache.md, "Unsupported Configurations".
        """
        dry_run = envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN

        def reject(message: str) -> None:
            if dry_run:
                logger.warning("dynamic KV cache dry run: %s", message)
                return
            raise ValueError(message)

        if not envs.VLLM_RBLN_USE_VLLM_MODEL:
            reject(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1 requires "
                "VLLM_RBLN_USE_VLLM_MODEL=1; see docs/dynamic_kv_cache.md."
            )

        if not USE_DEVICE_TENSOR:
            reject(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires "
                "VLLM_RBLN_USE_DEVICE_TENSOR=1; without it the artifact carries "
                "no dynamic KV dimension."
            )

        if vllm_config.kv_transfer_config is not None:
            reject(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE cannot be combined with a KV "
                "transfer connector; the resize invalidates its registrations."
            )

    @classmethod
    def register_custom_kv_cache_specs(cls, vllm_config: "VllmConfig") -> None:
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        from vllm_rbln.v1.kv_cache import (
            RBLNSlidingWindowManager,
            RBLNSlidingWindowSpec,
        )

        KVCacheSpecRegistry.register(
            RBLNSlidingWindowSpec,
            RBLNSlidingWindowManager,
            uniform_type_base_spec=RBLNSlidingWindowSpec,
        )

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.distributed.rbln_communicator.RblnCommunicator"  # noqa

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_rbln.lora.punica_wrapper.punica_rbln.PunicaWrapperRBLN"

    @classmethod
    def can_update_inplace(cls) -> bool:
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        # kv_buffer_device "cpu" is the host-bounce path; "rbln" is the D2D
        # path (upstream NixlConnectorWorker.__init__ rejects kv_buffer_device
        # values not listed here). Listed under both device_types because
        # device_type is "rbln" only when VLLM_RBLN_USE_DEVICE_TENSOR and
        # VLLM_RBLN_USE_VLLM_MODEL are both set.
        return {
            "cpu": ("cpu", "rbln"),
            "rbln": ("rbln", "cpu"),
        }

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        return "DRAM"

    @classmethod
    def discover_numa_topology(cls) -> list[list[int]]:
        """
        Discover NUMA topology and keep the last physical core of each numa
        into one core group list for nixl start_kv_load()
        """
        return []

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Set some additional forward context for the current platform if needs.
        """
        additional_kwargs: dict[str, Any] = {}
        if "kv_cache_bases" in kwargs:
            additional_kwargs["kv_cache_bases"] = kwargs["kv_cache_bases"]

        return additional_kwargs
