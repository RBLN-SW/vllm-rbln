# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RBLN registration + speculative-config wiring for MiniMax-M2 MTP.

Upstream vLLM ships the MiniMax-M2 base model and its
``get_spec_layer_idx_from_weight_name`` helper, but has **no** MTP draft
architecture, no ``minimax_m2_mtp`` MTP model type, and no ``hf_config_override``
branch that maps MiniMax-M2 onto an MTP draft. This module adds all three so that
``--speculative-config '{"method": "mtp", ...}'`` on a MiniMax-M2 checkpoint that
ships trained MTP heads resolves to :class:`vllm_rbln.models.minimax_m2_mtp.MiniMaxM2MTP`.
"""

from typing import Literal, get_args

from transformers import PretrainedConfig

from vllm_rbln.logger import init_logger
from vllm_rbln.patches.registry import add_registration, register_patch

logger = init_logger(__name__)

_MINIMAX_M2_MTP_ARCH = "MiniMaxM2MTP"
_MINIMAX_M2_MTP_MODEL_TYPE = "minimax_m2_mtp"


@add_registration(
    reason=(
        "Register the MiniMax-M2 MTP draft architecture and declare "
        "'minimax_m2_mtp' as a recognized MTP model type so that "
        "SpeculativeConfig auto-detects method='mtp' for it."
    ),
)
def register_minimax_m2_mtp() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        _MINIMAX_M2_MTP_ARCH,
        "vllm_rbln.models.minimax_m2_mtp:MiniMaxM2MTP",
    )

    # Extend the MTPModelTypes Literal so SpeculativeConfig's method detection
    # (`model_type in get_args(MTPModelTypes)`) recognizes minimax_m2_mtp.
    import vllm.config.speculative as speculative

    existing = get_args(speculative.MTPModelTypes)
    if _MINIMAX_M2_MTP_MODEL_TYPE not in existing:
        speculative.MTPModelTypes = Literal[
            existing + (_MINIMAX_M2_MTP_MODEL_TYPE,)
        ]
        logger.debug(
            "Added '%s' to SpeculativeConfig MTPModelTypes.",
            _MINIMAX_M2_MTP_MODEL_TYPE,
        )


# Upstream ``SpeculativeConfig.hf_config_override``, captured before the patch
# registry swaps it out. Kept at module scope (not closed over) so the
# replacement below stays a module-level function: ``ModelConfig`` stores this
# function in its ``hf_overrides`` field (see upstream speculative.py:
# ``hf_overrides=SpeculativeConfig.hf_config_override``), and that config is
# pickled when data-parallel / multiprocessing spawns engine processes. A
# nested ``<locals>`` closure is not picklable by reference and breaks spawn.
_original_hf_config_override = None


def _patched_hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _original_hf_config_override(hf_config)

    # Upstream maps DeepSeek / GLM / MiMo / ... onto their MTP drafts, but
    # nothing maps MiniMax-M2. Mirror that pattern: key off num_mtp_modules
    # (M2's spec-layer count field) rather than num_nextn_predict_layers.
    if (
        hf_config.architectures
        and hf_config.architectures[0] == "MiniMaxM2ForCausalLM"
        and getattr(hf_config, "num_mtp_modules", 0)
    ):
        n_predict = getattr(hf_config, "num_mtp_modules", None)
        hf_config.model_type = _MINIMAX_M2_MTP_MODEL_TYPE
        hf_config.update(
            {
                "n_predict": n_predict,
                "architectures": [_MINIMAX_M2_MTP_ARCH],
            }
        )

    return hf_config


# Capture the upstream static method (unwrapped to a plain function via
# class attribute access) before the patch registry swaps it out.
def _register_hf_config_override_patch() -> None:
    from vllm.config.speculative import SpeculativeConfig

    global _original_hf_config_override
    _original_hf_config_override = SpeculativeConfig.hf_config_override

    register_patch(
        target="vllm.config.speculative.SpeculativeConfig.hf_config_override",
        reason=(
            "Map MiniMax-M2 (architectures=['MiniMaxM2ForCausalLM'], "
            "num_mtp_modules > 0) onto the MiniMaxM2MTP draft architecture with "
            "n_predict = num_mtp_modules. Upstream hf_config_override has no "
            "MiniMax-M2 branch."
        ),
        key="vllm_rbln.patches.minimax_m2_mtp.hf_config_override",
    )(_patched_hf_config_override)


_register_hf_config_override_patch()
