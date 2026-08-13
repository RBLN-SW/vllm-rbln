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
"""Dtype-aware variants of vLLM's builtin logits processors.

The RBLN sampler keeps logits in the model dtype instead of upcasting
them to float32, while the builtin processors create their constant
tensors as float32. Mixed-dtype in-place ops (``index_put_``, ``mul_``,
``+=``) raise a RuntimeError, so these subclasses keep the constants in
the model dtype.
"""
import itertools
from collections.abc import Sequence

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (
    STR_POOLING_REJECTS_LOGITSPROCS,
    STR_SPEC_DEC_REJECTS_LOGITSPROCS,
    LogitsProcessors,
    _load_custom_logitsprocs,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

logger = init_logger(__name__)


class RBLNMinTokensLogitsProcessor(MinTokensLogitsProcessor):
    # index_put_ requires the value dtype to exactly match the logits
    # dtype, and two dtypes reach one instance within a single spec-decode
    # step: apply() sees model-dtype logits from the RBLN sampler, while
    # apply_with_spec_decode() sees the float32-upcast target logits from
    # the rejection sampler. The -inf constant is therefore synced to the
    # incoming logits dtype per call, with one cached tensor per dtype.
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self._neg_inf_tensors = {self.neg_inf_tensor.dtype: self.neg_inf_tensor}

    def _sync_neg_inf_dtype(self, dtype: torch.dtype):
        tensor = self._neg_inf_tensors.get(dtype)
        if tensor is None:
            tensor = self._neg_inf_tensors[dtype] = self.neg_inf_tensor.to(dtype)
        self.neg_inf_tensor = tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            self._sync_neg_inf_dtype(logits.dtype)
        return super().apply(logits)

    def apply_with_spec_decode(
        self, logits: torch.Tensor, num_draft_tokens: list[int]
    ) -> torch.Tensor:
        if self.min_toks:
            self._sync_neg_inf_dtype(logits.dtype)
        return super().apply_with_spec_decode(logits, num_draft_tokens)


class RBLNLogitBiasLogitsProcessor(LogitBiasLogitsProcessor):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self._logits_dtype = vllm_config.model_config.dtype

    def update_state(self, batch_update: BatchUpdate | None):
        # bias_tensor is rebuilt as float32 on every state change, so it
        # must be re-cast here rather than once in __init__.
        super().update_state(batch_update)
        if self.biases and self.bias_tensor.dtype != self._logits_dtype:
            self.bias_tensor = self.bias_tensor.to(self._logits_dtype)


class RBLNMinPLogitsProcessor(MinPLogitsProcessor):
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self._logits_dtype = vllm_config.model_config.dtype

    def update_state(self, batch_update: BatchUpdate | None):
        # min_p is re-sliced from the float32 buffer on state changes;
        # apply() multiplies it in place into model-dtype probabilities.
        super().update_state(batch_update)
        if self.min_p_count and self.min_p.dtype != self._logits_dtype:
            self.min_p = self.min_p.to(self._logits_dtype)


RBLN_BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    RBLNMinTokensLogitsProcessor,
    RBLNLogitBiasLogitsProcessor,
    RBLNMinPLogitsProcessor,
]


def build_rbln_logitsprocs(
    vllm_config: VllmConfig,
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors:
    """Mirror vLLM's build_logitsprocs with dtype-aware builtin processors."""
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        return LogitsProcessors()

    if vllm_config.speculative_config:
        if custom_logitsprocs:
            raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [RBLNMinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )

    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory)
        for ctor in itertools.chain(
            RBLN_BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes
        )
    )
