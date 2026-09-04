# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

logger = init_logger(__name__)


class RBLNMinTokensLogitsProcessor(MinTokensLogitsProcessor):
    # index_put_ requires the value dtype to exactly match the logits
    # dtype, and two dtypes reach one instance within a single spec-decode
    # step: apply() sees model-dtype logits from the RBLN sampler, while
    # apply_with_spec_decode() sees the float32-upcast target logits from
    # the rejection sampler. The -inf constant is therefore synced to the
    # incoming logits dtype per call, with one cached tensor per dtype.
    neg_inf_tensor: torch.Tensor

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
    # bias_tensor is rebuilt as float32 on every state change, so a one-time
    # cast would not stick: it is synced to the incoming logits dtype on each
    # apply() call.
    bias_tensor: torch.Tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases and self.bias_tensor.dtype != logits.dtype:
            self.bias_tensor = self.bias_tensor.to(logits.dtype)
        return super().apply(logits)


class RBLNMinPLogitsProcessor(MinPLogitsProcessor):
    # min_p is re-sliced from the float32 buffer on state changes and
    # multiplied in place into model-dtype probabilities, so it is synced
    # to the incoming logits dtype on each apply() call.
    min_p: torch.Tensor

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits
        if self.min_p.dtype != logits.dtype:
            self.min_p = self.min_p.to(logits.dtype)

        # NOTE(RBLN): update_state sizes min_p to the live request count, but
        # the native runner hands the sampler bucket-padded logits on decode
        # and leading-rows-only logits on prefill -- the same row contract the
        # runner's _pad_rows applies to temperature/top_k/top_p. Upstream's
        # apply multiplies min_p into a logits-rows-sized tensor in place, so a
        # row mismatch raises. Rebuild it around a row-matched view; the body
        # below is otherwise upstream's apply. Zero pad rows are no-ops: their
        # adjusted_min_p is 0 and no probability is below it. On the optimum
        # path RBLNInputBatch already sizes min_p to the bucket, so the sizing
        # branches never fire there.
        min_p = self.min_p
        num_rows = logits.shape[0]
        if min_p.shape[0] > num_rows:
            min_p = min_p[:num_rows]
        elif min_p.shape[0] < num_rows:
            min_p = torch.cat([min_p, min_p.new_zeros(num_rows - min_p.shape[0], 1)])

        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        adjusted_min_p = max_probabilities.mul_(min_p)
        invalid_token_mask = probability_values < adjusted_min_p
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


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
