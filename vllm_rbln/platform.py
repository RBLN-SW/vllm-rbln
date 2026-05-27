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

import contextlib
import os
from typing import TYPE_CHECKING

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
from vllm.platforms import Platform, PlatformEnum
from vllm.utils.torch_utils import _StreamPlaceholder

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.converter import sync_vllm_and_optimum
from vllm_rbln.utils.optimum.predicates import is_qwen3_pooling
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_multi_modal,
    is_pooling_arch,
)

logger = init_logger(__name__)

# RBLN default for an unset max_num_seqs (upstream vLLM defaults to 256).
RBLN_DEFAULT_MAX_NUM_SEQS = 1


def bypass_backend(graph_module: torch.fx.GraphModule, example_inputs):
    return graph_module.forward


register_backend(name="bypass", compiler_fn=bypass_backend)


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT

    # Compute device_name/device_type/dist_backend once at class definition
    # from env vars so that subprocesses spawned under
    # VLLM_WORKER_MULTIPROC_METHOD=spawn (which re-import this module fresh)
    # observe identical values to the parent without any extra plumbing.
    _USE_DEVICE_TENSOR: bool = (
        envs.VLLM_RBLN_USE_VLLM_MODEL and envs.VLLM_RBLN_USE_DEVICE_TENSOR
    )
    plugin_name: str = "rbln"
    device_name: str = "rbln" if _USE_DEVICE_TENSOR else "cpu"
    device_type: str = "rbln" if _USE_DEVICE_TENSOR else "cpu"
    dist_backend: str = "rbln-ccl" if _USE_DEVICE_TENSOR else ""
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    simple_compile_backend = "bypass"
    device_control_env_var: str = "RBLN_DEVICES"
    current_stream = _StreamPlaceholder

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # rebel.get_npu_name() returns None on a host without an NPU mounted
        # (e.g. a CPU-only compile worker) and otherwise falls back to the
        # RBLN_TARGET_SOC env var. When it is None we cannot determine a target
        # SOC, so surface an actionable error instead of a bare AssertionError.
        device_name = rebel.get_npu_name(device_id)
        if not device_name:
            raise RuntimeError(
                "Could not determine the RBLN NPU name "
                f"(rebel.get_npu_name({device_id}) returned None). On a host "
                "without an NPU mounted (e.g. a CPU-only compile worker running "
                "with VLLM_RBLN_COMPILE_ONLY=1), set RBLN_TARGET_SOC to the "
                "target SOC (e.g. RBLN-CA25) so compilation can target it."
            )
        return device_name

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        logger.warning("set_device is not supported on RBLN.")
        pass

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.distributed.rbln_communicator.RblnCommunicator"  # noqa

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
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: RBLN_DEFAULT_MAX_NUM_SEQS,
                UsageContext.OPENAI_API_SERVER: RBLN_DEFAULT_MAX_NUM_SEQS,
            }
            return default_batched_tokens, default_max_num_seqs

        EngineArgs.get_batch_defaults = classmethod(get_batch_defaults)
        EngineArgs._rbln_max_num_seqs_patched = True

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        # Runs before max_num_seqs is resolved from None to its default.
        cls._override_default_max_num_seqs()

        if parser is None:
            return

        for action in parser._actions:
            if action.dest == "device":
                action.choices.append("rbln")

        for action in parser._actions:
            if action.dest == "block_size":
                action.choices = None  # Override choices

    @classmethod
    def validate_and_setup_prerequisite(cls, vllm_config: VllmConfig) -> None:
        scheduler_config = vllm_config.scheduler_config
        if not scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "RBLN does not officially support disabling chunked prefill. "
                "Please don't disable chunked prefill by yourself."
            )

        if envs.VLLM_RBLN_COMPILE_ONLY:
            # Compile-only injects the compile_only torch.compile option. The
            # optimum-rbln path is not torch.compile-based, so the flag has no
            # meaning there and conflicts with that path; it only applies to the
            # vLLM-native (torch.compile) path, which VLLM_RBLN_USE_VLLM_MODEL
            # selects.
            if not envs.VLLM_RBLN_USE_VLLM_MODEL:
                raise ValueError(
                    "VLLM_RBLN_COMPILE_ONLY=1 is a torch.compile option and only "
                    "applies to the vLLM-native model path; set "
                    "VLLM_RBLN_USE_VLLM_MODEL=1 to use it. The optimum-rbln path "
                    "is not torch.compile-based, so compile-only conflicts with "
                    "it."
                )
            if envs.VLLM_DISABLE_COMPILE_CACHE:
                # Compile-only compiles each graph and writes the .rbln artifact
                # to the compile cache (the runtime is built on a dummy device
                # so no NPU is needed). With the cache disabled there is nowhere
                # to write the artifact, so the two options are mutually
                # exclusive.
                raise ValueError(
                    "VLLM_RBLN_COMPILE_ONLY=1 needs the compile cache enabled "
                    "to write compiled artifacts to disk; do not set "
                    "VLLM_DISABLE_COMPILE_CACHE=1 together with it."
                )

        parallel_config = vllm_config.parallel_config
        use_model_parallel = (
            parallel_config.tensor_parallel_size > 1
            or parallel_config.pipeline_parallel_size > 1
            or parallel_config.data_parallel_size > 1
            or parallel_config.enable_expert_parallel
        )
        if use_model_parallel:
            if envs.VLLM_RBLN_PROFILER:
                raise RuntimeError(
                    "RBLN_PROFILER is not supported when using vLLM model parallel "
                    "(TP, DP, EP, or PP)."
                )
            os.environ["RBLN_CTX_STANDALONE"] = "1"
            if os.environ.get("RBLN_RUNTIME_FORCE_SYNC") == "1":
                logger.warning(
                    "RBLN_RUNTIME_FORCE_SYNC=1 forces the synchronous runtime, "
                    "which may cause performance degradation "
                    "when using vLLM model parallel (TP, DP, EP, or PP)."
                )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.async_scheduling:
            scheduler_config.async_scheduling = False
            logger.warning("Async scheduler not supported on RBLN.")

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            cls.validate_and_setup_prerequisite(vllm_config)

            if envs.VLLM_RBLN_ENFORCE_MODEL_FP32:
                logger.info("original model_config.dtype = %s", model_config.dtype)
                if model_config.dtype == torch.bfloat16:
                    logger.warning("bfloat16 is not supported on RBLN.")

                # FIXME - force model dtype into fp32 for graph compilation
                model_config.dtype = torch.float
                assert model_config.dtype == torch.float
                logger.info("RBLN enforce model_config.dtype as torch.float")

                if (lora_config := vllm_config.lora_config) is not None:
                    lora_config.lora_dtype = torch.float
                    logger.info("RBLN enforce lora_config.lora_dtype as torch.float")

                if (speculative_config := vllm_config.speculative_config) is not None:
                    speculative_config.draft_model_config.dtype = torch.float
                    logger.info("RBLN enforce draft_model_config.dtype as torch.float")
            else:
                dtype = model_config.dtype
                if (
                    dtype != torch.bfloat16
                    and dtype != torch.float16
                    and dtype != torch.float
                ):
                    logger.warning(
                        "%s not supported on RBLN, only fp32,fp16,bf16 supported", dtype
                    )
                    model_config.dtype = torch.float
                logger.info("RBLN use model_config.dtype = %s", model_config.dtype)

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = (
                    "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"
                )
            scheduler_config.scheduler_cls = (
                "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"
            )

            # FIXME(jiwoo.park) This is a temporary workaround.
            if model_config.enforce_eager:
                if not envs.VLLM_RBLN_USE_DEVICE_TENSOR:
                    raise ValueError(
                        "enforce_eager=True requires VLLM_RBLN_USE_DEVICE_TENSOR=1. "
                        "Eager mode bypasses torch.compile, so ops must dispatch "
                        "to a real device='rbln' rather than the compile-backend "
                        "fake-CPU tensors used by the default vLLM model path."
                    )
                hf_config = vllm_config.model_config.hf_config
                assert not hasattr(hf_config, "sliding_window") or not getattr(
                    hf_config, "use_sliding_window", True
                )

                # NOTE - force dtype into fp16 for eager mode
                model_config.dtype = torch.float16

                if (lora_config := vllm_config.lora_config) is not None:
                    lora_config.lora_dtype = torch.float16

        else:
            # NOTE(eunji.lee):
            # It is for multimodal models
            # to generate inputs as fp32, not bfloat16
            # even though the model is compiled with bfloat16
            model_config.dtype = torch.float
            assert model_config.dtype == torch.float

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = (
                    "vllm_rbln.v1.worker.optimum_worker.RBLNOptimumWorker"
                )
            scheduler_config.scheduler_cls = (
                "vllm_rbln.v1.core.optimum_scheduler.RBLNOptimumScheduler"
            )

            assert vllm_config.parallel_config.tensor_parallel_size == 1, (
                "Cannot set tensor_parallel_size for pre-compiled optimum-rbln models. "
                "If you want to compile with tensor parallelism in vllm-rbln, "
                "please use the `VLLM_RBLN_TP_SIZE` environment variable instead."
            )
            assert vllm_config.parallel_config.pipeline_parallel_size == 1, (
                "Pipeline parallelism is not supported in optimum-rbln."
            )
            assert vllm_config.speculative_config is None, (
                "Speculative decoding is not supported in optimum-rbln."
            )
            # T5EncoderModel is encoder-only but inherits T5Config which has
            # is_encoder_decoder=True. This causes vllm to route inputs
            # through the enc-dec path, prepending decoder_start_token_id and
            # breaking CLS pooling. Set it to False for pooling models.
            # ModelConfig.is_encoder_decoder is a @cached_property that's
            # already evaluated by this point, so invalidate the cache too.
            hf_config = model_config.hf_config
            if is_pooling_arch(hf_config) and getattr(
                hf_config, "is_encoder_decoder", False
            ):
                hf_config.is_encoder_decoder = False
                with contextlib.suppress(KeyError):
                    del model_config.__dict__["is_encoder_decoder"]

            cls.disable_unsupported_prefix_caching(vllm_config)
            sync_vllm_and_optimum(vllm_config)

        if (
            parallel_config.distributed_executor_backend is not None
            and parallel_config.distributed_executor_backend != "mp"
        ):
            logger.warning(
                (
                    "%s is not supported on RBLN, fallback to mp "
                    "distributed executor backend."
                ),
                parallel_config.distributed_executor_backend,
            )

        assert not envs.VLLM_USE_V2_MODEL_RUNNER, (
            "v2 model runner is not supported for RBLN backend."
        )

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            from vllm.config import CompilationMode

            if vllm_config.compilation_config.mode != CompilationMode.NONE:
                logger.info("RBLN doesn't @support_torch_compile decorator")
                vllm_config.compilation_config.mode = CompilationMode.NONE
                if (
                    len(vllm_config.compilation_config.custom_ops) == 1
                    and vllm_config.compilation_config.custom_ops[0] == "none"
                ):
                    vllm_config.compilation_config.custom_ops = []

            if not model_config.disable_cascade_attn:
                logger.info(
                    "The cascade attention is disabled because RBLN does not support it"
                )
                model_config.disable_cascade_attn = True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.FLASH_ATTN:
            logger.info("Cannot use %s backend on RBLN.", selected_backend)
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on RBLN.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on RBLN.")

        attn_backend_cls = (
            "vllm_rbln.v1.attention.backends.flash_attention.RBLNAttentionBackend"
        )
        logger.info("Using RBLN Attention Backend: %s", attn_backend_cls)

        return attn_backend_cls

    @classmethod
    def _disable_prefix_caching(cls, vllm_config: VllmConfig, reason: str) -> None:
        """Disable prefix caching with warning message."""
        logger.warning(
            "Prefix caching is not available for %s. "
            "It has been automatically disabled.",
            reason,
        )
        vllm_config.cache_config.enable_prefix_caching = False

    @classmethod
    def disable_unsupported_prefix_caching(cls, vllm_config: VllmConfig) -> None:
        if not vllm_config.cache_config.enable_prefix_caching:
            return

        hf_config = vllm_config.model_config.hf_config

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if getattr(hf_config, "sliding_window", None) is not None and getattr(
                hf_config, "use_sliding_window", True
            ):
                cls._disable_prefix_caching(vllm_config, "sliding window models")

        else:
            # Prefix caching is supported only for decoder-only models for now.
            if is_qwen3_pooling(vllm_config.model_config):
                # Qwen3 pooling model does not support prefix caching for now.
                cls._disable_prefix_caching(vllm_config, "Qwen3 pooling models")
            elif is_enc_dec_arch(hf_config):
                cls._disable_prefix_caching(vllm_config, "encoder-decoder models")
            elif is_multi_modal(hf_config):
                cls._disable_prefix_caching(vllm_config, "multimodal models")
            elif is_pooling_arch(hf_config):
                cls._disable_prefix_caching(vllm_config, "pooling models")
            elif getattr(hf_config, "sliding_window", None) is not None and getattr(
                hf_config, "use_sliding_window", True
            ):
                cls._disable_prefix_caching(vllm_config, "sliding window models")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_rbln.lora.punica_wrapper.punica_rbln.PunicaWrapperRBLN"

    @classmethod
    def can_update_inplace(cls) -> bool:
        return False

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        # RBLN reports device_type="cpu" (see device_name). NIXL looks up
        # _NIXL_SUPPORTED_DEVICE[device_type], so to allow the device-to-device
        # path (kv_buffer_device="rbln", RblnNixlDirectConnector) we must
        # register "rbln" as an allowed kv_buffer under the "cpu" key.
        # "cpu" kv_buffer stays allowed for the host-bounce RblnNixlConnector.
        # Which connector may use which kv_buffer is enforced by an assert in
        # each connector (host-bounce rejects "rbln"; direct requires "rbln").
        return {
            "cpu": ("cpu", "rbln"),
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
