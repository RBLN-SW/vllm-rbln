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

from vllm_rbln import envs


def register():
    """Register the RBLN platform."""
    return "vllm_rbln.platform.RblnPlatform"


def register_model():
    if not envs.VLLM_RBLN_USE_VLLM_MODEL:
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "Gemma3ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.gemma3:RBLNOptimumGemma3ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "Gemma4ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.gemma4:RBLNOptimumGemma4ForConditionalGeneration",
        )


def register_ops():
    import vllm_rbln.distributed.ec_transfer.ec_connector.factory  # noqa

    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        from vllm_rbln.patches import apply_registered_patches, apply_registrations

        apply_registrations()
        apply_registered_patches()

        # import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa
        # import vllm_rbln.lora.layer  # noqa
        # import vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision  # noqa
        # import vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope  # noqa
        # import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
        # import vllm_rbln.models.deepseek_v2  # noqa
        # import vllm_rbln.models.minimax_m2  # noqa
