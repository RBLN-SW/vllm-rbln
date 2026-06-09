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
"""Encoder-cache (EC) disaggregation: producer-side model.

The EC producer runs only the vision encoder (no LLM prefill/decode compiled
models). This module owns that special construction so RBLNOptimumModelBase
stays free of any producer/EC awareness — the model runner calls
``build_ec_producer_model`` when it is running in the producer-only role.
"""

from typing import Any

import torch.nn as nn
from vllm.config import VllmConfig

# Architectures that can run as a disaggregated-encoder (EC) producer.
EC_PRODUCER_SUPPORTED_ARCHITECTURES = frozenset(
    {"RBLNQwen3VLForConditionalGeneration"}
)


class _ProducerOptimumModelProxy:
    """Lightweight stand-in for the full optimum model on EC producers.

    Only the visual encoder submodule is loaded; LLM compiled models
    (.rbln files for prefill/decode) are never touched.
    """

    def __init__(self, visual: Any, rbln_config: Any) -> None:
        self.visual = visual
        self.rbln_config = rbln_config

    def get_kvcache_num_blocks(self) -> int:
        return getattr(self.rbln_config, "kvcache_num_blocks", 1)

    def get_attn_impl(self) -> None:
        return None


def load_ec_producer_model(
    model_cls: Any, model_cls_name: str, valid_path: str
) -> _ProducerOptimumModelProxy:
    """Load only the vision encoder and wrap it in a proxy.

    Raises ValueError if the architecture does not support EC disaggregation.
    """
    if model_cls_name not in EC_PRODUCER_SUPPORTED_ARCHITECTURES:
        raise ValueError("Disaggregation is not supported for this model.")
    visual = model_cls.load_visual_encoder(valid_path)
    return _ProducerOptimumModelProxy(visual, visual.rbln_config)


def build_ec_producer_model(vllm_config: VllmConfig) -> nn.Module:
    """Build the producer-only model for the EC producer role.

    Reuses the host VLM wrapper's multimodal encode surface (embed_multimodal,
    encode, _parse_and_validate_*) by subclassing it, but replaces the heavy
    __init__ with a light one that loads only the vision encoder — never the
    LLM compiled models — and leaves KV cache disabled.
    """
    # Imported lazily to avoid an import cycle (model_base / qwen_vl import
    # chain pulls this module in for the runner).
    from .model_base import resolve_optimum_model

    valid_path, model_cls, model_cls_name = resolve_optimum_model(vllm_config)
    if valid_path is None:
        raise ValueError(
            "Disaggregated Encoder is only supported for a pre-compiled model."
        )
    if model_cls_name not in EC_PRODUCER_SUPPORTED_ARCHITECTURES:
        raise ValueError("Disaggregation is not supported for this model.")

    from .qwen_vl import RBLNOptimumQwen3VLForConditionalGeneration

    class _Qwen3VLECProducer(RBLNOptimumQwen3VLForConditionalGeneration):
        """Vision-encoder-only Qwen3-VL for the EC producer role."""

        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.scheduler_config = vllm_config.scheduler_config
            self.cache_config = vllm_config.cache_config
            self.batch_size = self.scheduler_config.max_num_seqs
            # No LLM, so no KV cache; the worker's determine_available_memory()
            # short-circuits when the adapter is None.
            self.kv_block_adapter = None
            self.rope_deltas = {}
            self.model = load_ec_producer_model(
                model_cls, model_cls_name, valid_path
            )
            self.rbln_model_config = self.model.rbln_config
            self.attn_impl = self.model.get_attn_impl()
            self.supports_transcription_only = False

    return _Qwen3VLECProducer()
