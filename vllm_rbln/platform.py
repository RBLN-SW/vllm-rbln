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
from vllm.platforms import Platform, PlatformEnum

from vllm_rbln import envs
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
    # TODO(RBLN): GroupCoordinator uses the device_name
    # when torch.device(device_name) is called.
    # But we don't support the 'rbln' device yet.
    # To support this, we must use PyTorch-RBLN
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    device_control_env_var: str = "RBLN_DEVICES"
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
        if selected_backend and selected_backend != AttentionBackendEnum.CUSTOM:
            logger.info("Cannot use %s backend on RBLN.", selected_backend)
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on RBLN.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on RBLN.")

        logger.info("Using %s Backend", selected_backend)

        return selected_backend.get_path()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        assert (device_name := rebel.get_npu_name(device_id))
        return device_name

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        rebel.manual_seed(seed)

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
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError(
                "VLLM_USE_V2_MODEL_RUNNER is not supported for RBLN backend."
            )

        attention_config = vllm_config.attention_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if attention_config.backend is None:
            attention_config.backend = AttentionBackendEnum.CUSTOM

        if scheduler_config.async_scheduling:
            logger.warning(
                "Asynchronous scheduling is not supported on RBLN. "
                "Overriding scheduler_config.async_scheduling to False."
            )
            scheduler_config.async_scheduling = False

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if vllm_config.lora_config is not None:
                raise ValueError("LoRA is not supported on RBLN.")

            cls._validate_and_setup_prerequisite(vllm_config)

            if envs.VLLM_RBLN_ENFORCE_MODEL_FP32:
                if model_config.dtype != torch.float32:
                    # FIXME(RBLN): force model dtype into fp32 for graph compilation
                    original_dtype = model_config.dtype
                    model_config.dtype = torch.float32
                    logger.info(
                        "Overriding model_config.dtype from %s to %s.",
                        original_dtype,
                        model_config.dtype,
                    )
            else:
                if model_config.dtype not in (
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ):
                    logger.warning(
                        "Unsupported dtype for RBLN: %s. Falling back to %s. "
                        "Supported dtypes are torch.float32, torch.float16, "
                        "and torch.bfloat16.",
                        model_config.dtype,
                        torch.float32,
                    )
                    model_config.dtype = torch.float32

            logger.info("Using model_config.dtype for RBLN: %s", model_config.dtype)

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = (
                    "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"
                )
            scheduler_config.scheduler_cls = (
                "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"
            )

            # FIXME(jiwoo.park) This is a temporary workaround.
            if model_config.enforce_eager:
                hf_config = vllm_config.model_config.hf_config
                assert not hasattr(hf_config, "sliding_window") or not getattr(
                    hf_config, "use_sliding_window", True
                )

                RblnPlatform.device_type = "rbln"
                vllm_config.device_config.device_type = RblnPlatform.device_type
                vllm_config.device_config.device = torch.device(
                    RblnPlatform.device_type
                )
                # RBLN(NOTE): force dtype into fp16 for eager mode
                model_config.dtype = torch.float16

            from vllm.config import CompilationMode

            if vllm_config.compilation_config.mode != CompilationMode.NONE:
                logger.info(
                    "vLLM compilation mode is not used on RBLN because "
                    "@support_torch_compile is not supported. "
                    "Overriding compilation_config.mode from %s to %s.",
                    vllm_config.compilation_config.mode,
                    CompilationMode.NONE,
                )
                vllm_config.compilation_config.mode = CompilationMode.NONE
                if (
                    len(vllm_config.compilation_config.custom_ops) == 1
                    and vllm_config.compilation_config.custom_ops[0] == "none"
                ):
                    logger.debug(
                        "Clearing compilation_config.custom_ops because "
                        "vLLM compilation mode is disabled on RBLN."
                    )
                    vllm_config.compilation_config.custom_ops = []

            if not model_config.disable_cascade_attn:
                logger.warning(
                    "Cascade attention is not supported on RBLN. "
                    "Overriding model_config.disable_cascade_attn to True."
                )
                model_config.disable_cascade_attn = True

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

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.distributed.rbln_communicator.RblnCommunicator"  # noqa

    @classmethod
    def _validate_and_setup_prerequisite(cls, vllm_config: VllmConfig) -> None:
        scheduler_config = vllm_config.scheduler_config
        if not scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "Disabling chunked prefill is not supported on RBLN. "
                "Please enable chunked prefill by yourself."
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
                raise ValueError(
                    "RBLN_PROFILER is not supported when using vLLM model parallel "
                    "(TP, DP, EP, or PP)."
                )

            if (
                parallel_config.data_parallel_size > 1
                and scheduler_config.max_num_batched_tokens
                % scheduler_config.max_num_seqs
                != 0
            ):
                raise ValueError(
                    "max_num_batched_tokens must be divisible by max_num_seqs "
                    "when DP enabled."
                )

            if (
                parallel_config.data_parallel_size > 1
                or parallel_config.enable_expert_parallel
            ) and not envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
                raise ValueError(
                    "VLLM_RBLN_USE_MOE_TOKENS_MASK is required when DP or EP enabled: "
                    "the mask marks padded tokens introduced by DP multicast. "
                    "Set VLLM_RBLN_USE_MOE_TOKENS_MASK=1 (default)."
                )

            os.environ["RBLN_CTX_STANDALONE"] = "1"
            ccl_async_mode = os.environ.get("RBLN_FORCE_CCL_ASYNC")
            # NOTE If users don't set RBLN_FORCE_CCL_ASYNC, we will set it to 1
            # to enable async mode by default for better performance.
            # However, if users explicitly set RBLN_FORCE_CCL_ASYNC to 0,
            # we will respect their choice but print a warning message.
            if ccl_async_mode is None:
                os.environ["RBLN_FORCE_CCL_ASYNC"] = "1"
            elif ccl_async_mode == "0":
                logger.warning(
                    "RBLN_FORCE_CCL_ASYNC is set to 0, "
                    "which may cause performance degradation "
                    "when using vLLM model parallel (TP, DP, EP, or PP)."
                )

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
        return {
            "rbln": ("cpu",),
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
