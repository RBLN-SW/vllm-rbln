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
    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        import vllm_rbln.attention.layer  # noqa
        import vllm_rbln.forward_context  # noqa
        import vllm_rbln.lora.layer  # noqa
        import vllm_rbln.model_executor.layers.fused_moe.layer  # noqa
        import vllm_rbln.model_executor.layers.fused_moe.shared_fused_moe  # noqa
        import vllm_rbln.model_executor.layers.logits_processor  # noqa
        import vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision  # noqa
        import vllm_rbln.model_executor.layers.quantization.mxfp4  # noqa
        import vllm_rbln.model_executor.layers.quantization.fp8  # noqa
        import vllm_rbln.model_executor.layers.rotary_embedding.base  # noqa
        import vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope  # noqa
        import vllm_rbln.model_executor.layers.vocab_parallel_embedding  # noqa
        import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
        import vllm_rbln.models.deepseek_v2  # noqa
        import vllm_rbln.models.gpt_oss  # noqa
        import vllm_rbln.models.qwen2_moe  # noqa
        import vllm_rbln.models.qwen3  # noqa
        import vllm_rbln.models.qwen3_moe  # noqa
        import vllm_rbln.models.minimax_m2  # noqa
        import vllm_rbln.models.utils  # noqa
        import vllm_rbln.v1.spec_decoding.eagle  # noqa
        from vllm_rbln.triton_kernels import attention  # noqa
        from vllm_rbln.triton_kernels import causal_attention  # noqa
        from vllm_rbln.triton_kernels import flash_attention  # noqa
        from vllm_rbln.triton_kernels import flash_causal_attention  # noqa
        from vllm_rbln.triton_kernels import sliding_window_attention  # noqa
