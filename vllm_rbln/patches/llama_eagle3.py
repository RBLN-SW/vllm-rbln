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
"""EAGLE3 draft->target vocab scatter index: keep it a named buffer.

Without this, EAGLE3 SIGSEGVs during KV warmup, before the server comes up.

Upstream builds the scatter index inside ``compute_logits`` as
``arange(draft_vocab_size) + draft_id_to_target_id``. Both operands are constant
with respect to the forward input, so the subgraph folds into a single
*anonymous* constant. Under weight-free loading, tensors carry no data at trace
time and are filled afterwards by ``data_ptr`` with a name fallback over
``named_parameters() + named_buffers()``; a folded constant matches neither
channel and is skipped **silently**. It reaches execution holding whatever its
placeholder held -- arbitrary indices into a 200064-wide scatter -- and that
out-of-bounds host write is the segfault.

The fix keeps the index a named buffer and its subgraph non-constant:

``__init__``
    register ``target_ids`` as ``arange`` -- identity and in range, so even an
    unfilled buffer cannot write out of bounds.
``load_weights``
    fill it from the loaded mapping (``arange + d2t``). Upstream leaves
    ``draft_id_to_target_id`` zero when the checkpoint has no mapping, which
    keeps the buffer at identity.
``compute_logits``
    add an input-derived zero so the subgraph is not folded. Index values are
    unchanged.

This mirrors the ``mask_hidden`` buffer already in the same class, which
upstream fills the same way for the same reason.

``deepseek_eagle3.Eagle3DeepseekV2ForCausalLM.compute_logits`` has the same
defect. It is deliberately not patched: we have no DeepSeek-family EAGLE3 head
to verify against.
"""

import logging

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

_TARGET = "vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM"

# Captured at import time: the registry replaces targets outright, so wrapping
# upstream behaviour means holding the original here rather than copying its body.
_orig_init = Eagle3LlamaForCausalLM.__init__
_orig_load_weights = Eagle3LlamaForCausalLM.load_weights

_WHY = (
    "the draft->target scatter index is input-independent, so it folds into an "
    "anonymous constant; weight-free apply resolves tensors by "
    "named_parameters/named_buffers and silently skips it, leaving garbage "
    "indices -> out-of-bounds host-op write -> segfault in KV warmup"
)


@register_patch(
    target=f"{_TARGET}.__init__",
    reason=f"Register the scatter index as a named buffer: {_WHY}.",
)
def patched_eagle3_llama_init(
    self, *, vllm_config: VllmConfig, prefix: str = ""
) -> None:
    # Keep the explicit `vllm_config` / `prefix` names: `initialize_model`
    # picks the construction path by inspecting them, so a `*args, **kwargs`
    # replacement makes every EAGLE3 load fail with "missing 1 required
    # keyword-only argument: 'vllm_config'". (The attention/mla `__init__`
    # patches are safe with *args because those classes are constructed
    # directly rather than resolved through `initialize_model`.)
    _orig_init(self, vllm_config=vllm_config, prefix=prefix)
    # arange == identity mapping: in range even if never filled.
    self.register_buffer(
        "target_ids",
        torch.arange(self.config.draft_vocab_size, dtype=torch.long),
        persistent=False,
    )


@register_patch(
    target=f"{_TARGET}.load_weights",
    reason=f"Fill the scatter-index buffer from the loaded d2t mapping: {_WHY}.",
)
def patched_eagle3_llama_load_weights(self, weights):
    loaded = _orig_load_weights(self, weights)
    if self.draft_id_to_target_id is None:
        return loaded
    base = torch.arange(
        self.config.draft_vocab_size,
        dtype=torch.long,
        device=self.target_ids.device,
    )
    # Upstream skips draft_id_to_target_id in the loader when the checkpoint has
    # no mapping, so it stays zero here and the buffer stays identity.
    self.target_ids.copy_(
        base + self.draft_id_to_target_id.to(self.target_ids.device).long()
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "EAGLE3 scatter index filled: targets [%d, %d], target vocab %d",
            int(self.target_ids.min()),
            int(self.target_ids.max()),
            self.config.vocab_size,
        )
    return loaded


@register_patch(
    target=f"{_TARGET}.compute_logits",
    reason=f"Keep the scatter-index subgraph non-constant: {_WHY}.",
)
def patched_eagle3_llama_compute_logits(
    self, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    if self.draft_id_to_target_id is None:
        assert logits.shape[1] == self.config.vocab_size, (
            "Expected logits to have shape "
            f"(*, {self.config.vocab_size}), but got {logits.shape}"
        )
        return logits

    # NOTE(RBLN): upstream computes the index here as
    #   base = torch.arange(self.config.draft_vocab_size, device=logits.device)
    #   targets = base + self.draft_id_to_target_id
    # which is input-independent and therefore foldable. Deriving a zero from
    # `logits` keeps the subgraph live so `target_ids` survives as a named
    # buffer for the weight-free apply pass. Index values are unchanged.
    #
    # The zero has to stay value-derived -- a shape-derived one (`zeros_like`)
    # is foldable, which is the defect being worked around -- so it inherits
    # whatever lm_head emitted: a non-finite element makes `x * 0.0` a NaN and
    # `NaN -> long` an arbitrary value, which would push every entry of
    # `targets` out of range and reintroduce exactly the out-of-bounds scatter
    # this patch exists to prevent. The clamp bounds that: `target_ids` is
    # already in range by construction, so on the normal path (zero == 0) it is
    # a no-op.
    zero = (logits.reshape(-1)[:1].sum() * 0.0).to(torch.long)
    targets = (self.target_ids + zero).clamp(0, self.config.vocab_size - 1)
    logits_new = logits.new_full(
        (logits.shape[0], self.config.vocab_size),
        float("-inf"),
    )
    logits_new[:, targets] = logits
    return logits_new
