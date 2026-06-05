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

import vllm_rbln.rbln_envs as envs


def _try_import_optional(module_name: str):
    import importlib

    from vllm_rbln.logger import init_logger

    logger = init_logger(__name__)
    try:
        importlib.import_module(module_name)
    except (ImportError, AttributeError) as e:
        logger.warning("[RBLN] Skipping optional patch %s: %s", module_name, e)


def _patch_cleanup_dist_env_and_memory():
    from vllm.v1.engine import core as engine_core
    from vllm_rbln.logger import init_logger

    if getattr(engine_core, "_rbln_cleanup_patched", False):
        return

    logger = init_logger(__name__)
    orig_cleanup = engine_core.cleanup_dist_env_and_memory

    def cleanup_dist_env_and_memory(*args, **kwargs):
        try:
            return orig_cleanup(*args, **kwargs)
        except RuntimeError as e:
            if "Cannot access accelerator device when none is available" not in str(e):
                raise
            logger.debug(
                "Skipping accelerator cache cleanup because no accelerator is available"
            )
            return None

    engine_core.cleanup_dist_env_and_memory = cleanup_dist_env_and_memory
    engine_core._rbln_cleanup_patched = True


def register():
    """Register the RBLN platform."""
    return "vllm_rbln.platform.RblnPlatform"


def register_model():
    if not envs.VLLM_RBLN_USE_VLLM_MODEL:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "T5WithLMHeadModel",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "T5ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "T5EncoderModel",
            "vllm_rbln.model_executor.models.optimum.encoder:RBLNOptimumForEncoderModel",
        )
        ModelRegistry.register_model(
            "Gemma3ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.gemma3:RBLNOptimumGemma3ForConditionalGeneration",
        )


def register_ops():
    # torch 2.10 added a strict raise in CompileEventLogger.increment_toplevel
    # / add_to_set_toplevel when no outermost chromium event is active. The
    # RBLN custom torch.compile backend ends up calling those without a
    # propagated event (wrap-based approaches at warm_up_model / execute_model
    # / dummy_run did not work for this code path), so we silence the raise
    # here. See ``_torch_dynamo_compat.py`` for details.
    import vllm_rbln._torch_dynamo_compat  # noqa
    import vllm_rbln.distributed.ec_transfer.ec_connector.factory  # noqa

    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        _patch_cleanup_dist_env_and_memory()
        import vllm_rbln.model_executor.layers.attention.attention  # noqa
        import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa
        import vllm_rbln.forward_context  # noqa
        import vllm_rbln.lora.layer  # noqa
        import vllm_rbln.model_executor.layers.fused_moe.layer  # noqa
        import vllm_rbln.model_executor.layers.logits_processor  # noqa
        _try_import_optional(
            "vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision"
        )
        _try_import_optional("vllm_rbln.model_executor.layers.quantization.mxfp4")
        _try_import_optional("vllm_rbln.model_executor.layers.quantization.fp8")
        import vllm_rbln.model_executor.layers.rotary_embedding.base  # noqa
        import vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope  # noqa
        import vllm_rbln.model_executor.layers.vocab_parallel_embedding  # noqa
        import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
        import vllm_rbln.models.deepseek_v2  # noqa
        import vllm_rbln.models.gpt_oss  # noqa
        import vllm_rbln.models.qwen2  # noqa
        import vllm_rbln.models.qwen2_moe  # noqa
        import vllm_rbln.models.qwen3  # noqa
        import vllm_rbln.models.qwen3_moe  # noqa
        import vllm_rbln.models.minimax_m2  # noqa
        import vllm_rbln.models.utils  # noqa
        from vllm_rbln.triton_kernels import attention  # noqa
        from vllm_rbln.triton_kernels import causal_attention  # noqa
        from vllm_rbln.triton_kernels import flash_attention  # noqa
        from vllm_rbln.triton_kernels import flash_causal_attention  # noqa
        from vllm_rbln.triton_kernels import sliding_window_attention  # noqa
