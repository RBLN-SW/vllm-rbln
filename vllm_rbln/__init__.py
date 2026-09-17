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
    """Nothing to do, and kept so an installed copy still resolves this name.

    The A.X K2 registration this used to hold runs from the patch registry now.
    It has to: this entry point is called before the arguments are parsed, too
    early to know the model path, while the architecture is resolved in the
    process that parses them.
    """
    # TODO(vllm-rbln>=0.12.0): delete, with the entry point in pyproject.toml.


def register_ops():
    import os

    import vllm_rbln.distributed.ec_transfer.ec_connector.factory  # noqa
    from vllm_rbln import envs
    from vllm_rbln.platform import RblnPlatform

    # Both `vllm serve` and `LLM(...)` load the plugins before they build a
    # config, and only the first of them parses arguments, so this is the one
    # point early enough to see every `create_engine_config` call.
    RblnPlatform._capture_model_impl()

    # Only the path a parent already resolved can be acted on here: this runs
    # before the arguments are parsed, so in the process that parses them the
    # variable is unset and the platform hook applies the same set afterwards.
    # Not `model_impl_from_env()`, which answers there too, from the deprecated
    # variable: a patch applied on that guess outlives a flag that disagrees.
    if os.environ.get(envs.RESOLVED_MODEL_IMPL_ENV) == "vllm":
        from vllm_rbln.patches import apply_registered_patches, apply_registrations

        apply_registrations()
        apply_registered_patches()

        # TODO(RBLN): remove the following imports after we have a better way
        import vllm_rbln.distributed.kv_transfer.kv_connector.factory  # noqa
