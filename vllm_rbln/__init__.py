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

# NOTE(eunji.lee): Keep this module free of top-level imports.
# vLLM resolves platform plugins lazily on the first
# `vllm.platforms.current_platform` access, which almost any
# vLLM import triggers. If importing `vllm_rbln` pulls in vLLM before `register`
# is bound, the plugin loader re-enters this partially initialized module,
# `getattr(module, "register")` raises, and vLLM falls back to CpuPlatform.


def register():
    """Register the RBLN platform."""
    return "vllm_rbln.platform.RblnPlatform"


def register_model():
    # FIXME(kblee): A.X K2 (axk2) is not in upstream vLLM yet.
    from vllm_rbln import envs

    if not envs.VLLM_RBLN_USE_VLLM_MODEL:
        return

    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from vllm_rbln.patches import axk2
    from vllm_rbln.patches.axk2.config import AXK2Config

    _CONFIG_REGISTRY[axk2.MODEL_TYPE] = AXK2Config
    ModelRegistry.register_model(axk2.ARCH, axk2.MODEL_CLASS_PATH)


def register_ops():
    import vllm_rbln.distributed.ec_transfer.ec_connector.factory  # noqa
    from vllm_rbln import envs

    if envs.VLLM_RBLN_USE_VLLM_MODEL:
        from vllm_rbln.patches import apply_registered_patches, apply_registrations

        apply_registrations()
        apply_registered_patches()

        # TODO(RBLN): remove the following imports after we have a better way
        import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa
        import vllm_rbln.model_executor.model_loader.weight_loader  # noqa
