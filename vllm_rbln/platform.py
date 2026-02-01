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
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    VllmConfig = None

import rebel
from torch._dynamo import register_backend
from vllm.platforms import Platform, PlatformEnum
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import _StreamPlaceholder

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.configuration import (is_qwen3_pooling,
                                                   sync_with_rbln_config)
from vllm_rbln.utils.optimum.registry import (is_enc_dec_arch, is_multi_modal,
                                              is_pooling_arch)

logger = init_logger(__name__)


def bypass_backend(graph_module: torch.fx.GraphModule, example_inputs):
    return graph_module.forward


register_backend(name="bypass", compiler_fn=bypass_backend)


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT

    # TODO(jiwoo.park): GroupCoordinator uses the device_name
    # when torch.device(device_name) is called.
    # But we don't support the 'rbln'' device yet.
    # To support this, we must use PyTorch-RBLN
    plugin_name: str = "rbln"
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    simple_compile_backend = "bypass"
    device_control_env_var: str = "RBLN_DEVICES"
    current_stream = _StreamPlaceholder

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return rebel.get_npu_name(device_id)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.distributed.rbln_communicator.RblnCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
                                ) -> None:
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
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if envs.VLLM_RBLN_ENFORCE_MODEL_FP32:
                logger.info("original model_config.dtype = %s",
                            model_config.dtype)
                if model_config.dtype == torch.bfloat16:
                    logger.warning("bfloat16 is not supported on RBLN.")

                # FIXME - force model dtype into fp32 for graph compilation
                model_config.dtype = torch.float
                assert model_config.dtype == torch.float
                logger.info("RBLN enforce model_config.dtype as torch.float")

                if (lora_config := vllm_config.lora_config) is not None:
                    lora_config.lora_dtype = torch.float
                    logger.info(
                        "RBLN enforce lora_config.lora_dtype as torch.float")
            else:
                dtype = model_config.dtype
                logger.info("original model_config.dtype = %s", dtype)
                if dtype != torch.bfloat16 and dtype != torch.float16 \
                            and dtype != torch.float:
                    logger.warning(
                        "%s not supported on RBLN, "
                        "only fp32,fp16,bf16 supported", dtype)
                    model_config.dtype = torch.float
                logger.info("RBLN use model_config.dtype = %s",
                            model_config.dtype)

            if envs.VLLM_USE_V1:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm_rbln.v1.worker.rbln_worker.RBLNWorker")
                scheduler_config.scheduler_cls = (
                    "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler")
            else:
                if parallel_config.worker_cls == "auto":
                    parallel_config.worker_cls = (
                        "vllm_rbln.worker.worker.RBLNWorker")
                scheduler_config.scheduler_cls = (
                    "vllm_rbln.core.scheduler.RBLNScheduler")

            # FIXME(jiwoo.park) This is a temporary workaround.
            if model_config.enforce_eager:
                hf_config = vllm_config.model_config.hf_config
                assert not hasattr(hf_config, "sliding_window") \
                    or not getattr(hf_config, "use_sliding_window", True)

                RblnPlatform.device_type = "rbln"
                vllm_config.device_config.device_type = RblnPlatform.device_type
                vllm_config.device_config.device = (torch.device(
                    RblnPlatform.device_type))
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
                parallel_config.worker_cls = \
                    "vllm_rbln.v1.worker.optimum_worker.RBLNOptimumWorker"
            scheduler_config.scheduler_cls = \
                    "vllm_rbln.v1.core.optimum_scheduler.RBLNOptimumScheduler"

            assert vllm_config.parallel_config.tensor_parallel_size == 1, (
                "Tensor parallelism is set when compiled in optimum-rbln.")
            assert vllm_config.parallel_config.pipeline_parallel_size == 1, (
                "Pipeline parallelism is not supported in optimum-rbln.")
            assert vllm_config.speculative_config is None, (
                "Speculative decoding is not supported in vLLM RBLN.")
            cls.disable_unsupported_prefix_caching(vllm_config)
            sync_with_rbln_config(vllm_config)

        if (parallel_config.distributed_executor_backend is not None
                and parallel_config.distributed_executor_backend != "mp"):
            logger.warning(
                ("%s is not supported on RBLN, fallback to mp "
                 "distributed executor backend."),
                parallel_config.distributed_executor_backend,
            )

        assert (not vllm_config.speculative_config
                ), "Speculative decoding not yet supported for RBLN backend."
        assert not envs.VLLM_USE_V2_MODEL_RUNNER, \
            "v2 model runner is not supported for RBLN backend."

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            from vllm.config import CompilationLevel

            if (vllm_config.compilation_config.level
                    != CompilationLevel.NO_COMPILATION):
                logger.info("RBLN doesn't @support_torch_compile decorator")
                vllm_config.compilation_config.level = (
                    CompilationLevel.NO_COMPILATION)
                if (len(vllm_config.compilation_config.custom_ops) == 1
                        and vllm_config.compilation_config.custom_ops[0]
                        == "none"):
                    vllm_config.compilation_config.custom_ops = []

            if not model_config.disable_cascade_attn:
                logger.info("The cascade attention is disabled"
                            " because RBLN does not support it")
                model_config.disable_cascade_attn = True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink: bool,
    ) -> str:
        if envs.VLLM_USE_V1:
            attn_backend_cls = ("vllm_rbln.v1.attention.backends."
                                "flash_attention.RBLNAttentionBackend")
        else:
            attn_backend_cls = ("vllm_rbln.attention.backends."
                                "flash_attention.RBLNAttentionBackend")
        logger.info("Using RBLN Attention Backend: %s", attn_backend_cls)

        return attn_backend_cls

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    def _disable_prefix_caching(cls, vllm_config: VllmConfig,
                                reason: str) -> None:
        """Disable prefix caching with warning message."""
        logger.warning(
            "Prefix caching is not available for %s. "
            "It has been automatically disabled.", reason)
        vllm_config.cache_config.enable_prefix_caching = None

    @classmethod
    def disable_unsupported_prefix_caching(cls,
                                           vllm_config: VllmConfig) -> None:
        if not vllm_config.cache_config.enable_prefix_caching:
            return

        hf_config = vllm_config.model_config.hf_config

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if getattr(hf_config, "sliding_window", None) is not None \
                   and getattr(hf_config, "use_sliding_window", True):
                cls._disable_prefix_caching(vllm_config,
                                            "sliding window models")

        else:
            # Prefix caching is supported only for decoder-only models for now.
            if is_qwen3_pooling(vllm_config):
                # Qwen3 pooling model does not support prefix caching for now.
                cls._disable_prefix_caching(vllm_config,
                                            "Qwen3 pooling models")
            elif is_enc_dec_arch(hf_config):
                cls._disable_prefix_caching(vllm_config,
                                            "encoder-decoder models")
            elif is_multi_modal(hf_config):
                cls._disable_prefix_caching(vllm_config, "multimodal models")
            elif is_pooling_arch(hf_config):
                cls._disable_prefix_caching(vllm_config, "pooling models")
            elif getattr(hf_config, "sliding_window", None) is not None \
                    and getattr(hf_config, "use_sliding_window", True):
                cls._disable_prefix_caching(vllm_config,
                                            "sliding window models")

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
    def device_id_to_physical_device_id(cls, device_id: int):
        # overrides for RSD (rbln scalable devices)
        # dp device ids for RBLN SHOULD consider rsd size
        rsd_size = envs.VLLM_RBLN_TP_SIZE
        assert rsd_size >= 1
        if cls.device_control_env_var in os.environ and os.environ[
                cls.device_control_env_var] != "":
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_ids = ""
            start_device_id = device_id * rsd_size
            for rsd_id in range(rsd_size - 1):
                physical_device_ids += str(device_ids[start_device_id +
                                                      rsd_id])
                physical_device_ids += ","
            physical_device_ids += str(device_ids[start_device_id + rsd_size -
                                                  1])
            logger.info("RBLN DP physical_device_ids = %s",
                        physical_device_ids)
            return physical_device_ids
        else:
            return device_id
