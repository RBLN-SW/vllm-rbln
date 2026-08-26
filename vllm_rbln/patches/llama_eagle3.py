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
"""EAGLE3 draft->target vocab scatter index: keep it a NAMED buffer.

The drafter emits ids in its own 32k draft vocabulary and scatters them into the
target's 200k vocabulary. Upstream builds that scatter index inside
``compute_logits``::

    base = torch.arange(self.config.draft_vocab_size, device=logits.device)
    targets = base + self.draft_id_to_target_id

Both operands are constants with respect to the forward input, so dynamo
const-folds the subgraph into a SINGLE anonymous constant. Under RBLN weight-free
loading, the host-op apply pass resolves tensors through
``named_parameters()`` / ``named_buffers()`` -- an anonymous folded constant is in
neither, so it is never filled. It reaches execution holding whatever was in the
FakeTensorMode allocation: garbage indices into a 200k-wide scatter, i.e. an
out-of-bounds host-op write, which segfaults during KV warmup before the server
ever comes up.

The fix keeps the index NAMED and keeps the subgraph NON-constant:

1. ``__init__`` registers ``target_ids`` as a buffer, initialised to ``arange``
   (identity, fully in range) so that even an unfilled buffer is safe.
2. ``load_weights`` fills it from the real loaded mapping (``arange + d2t``).
   When the checkpoint carries no mapping, ``d2t`` is all zeros and the buffer
   stays identity.
3. ``compute_logits`` ties the index to an input-derived zero, so the subgraph
   is not constant and cannot be folded. ``target_ids`` then stays a ``get_attr``
   buffer that the apply pass fills with real, in-range values before execution.

This mirrors the existing ``mask_hidden`` buffer pattern in the same class.

NOTE: ``deepseek_eagle3.Eagle3DeepseekV2ForCausalLM.compute_logits`` builds the
index the same way and has the same defect. It is deliberately NOT patched here
because we have no DeepSeek-family EAGLE3 head to verify against; see the commit
message.
"""

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

_REASON_SUFFIX = (
    "the draft->target scatter index is input-independent, so it const-folds "
    "into an anonymous constant that weight-free apply (which resolves by "
    "named_parameters/named_buffers) never fills -- garbage indices, "
    "out-of-bounds host-op write, segfault in KV warmup."
)


@register_patch(
    target=f"{_TARGET}.__init__",
    reason=f"Register the scatter index as a named buffer: {_REASON_SUFFIX}",
)
def patched_eagle3_llama_init(
    self, *, vllm_config: VllmConfig, prefix: str = ""
) -> None:
    # The signature has to name vllm_config and prefix, not absorb them into
    # **kwargs: `initialize_model` inspects it and takes the new-style path only
    # when both names are present (model_loader/utils.py:56-62). A wrapper that
    # forwards opaquely is read as an old-style model class and called with no
    # arguments at all.
    _orig_init(self, vllm_config=vllm_config, prefix=prefix)
    # Identity default: safe to execute even if never filled.
    self.register_buffer(
        "target_ids",
        torch.arange(self.config.draft_vocab_size, dtype=torch.long),
        persistent=False,
    )


@register_patch(
    target=f"{_TARGET}.load_weights",
    reason=(
        f"Fill the named scatter index from the loaded d2t mapping: {_REASON_SUFFIX}"
    ),
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
    # No mapping in the checkpoint -> d2t is all zeros -> stays identity.
    self.target_ids.copy_(
        base + self.draft_id_to_target_id.to(self.target_ids.device).long()
    )
    logger.debug(
        "EAGLE3 scatter index filled: targets [%d, %d], target vocab %d",
        int(self.target_ids.min()),
        int(self.target_ids.max()),
        self.config.vocab_size,
    )
    return loaded


@register_patch(
    target=f"{_TARGET}.compute_logits",
    reason=f"Keep the scatter-index subgraph non-constant: {_REASON_SUFFIX}",
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
    # which is input-independent and therefore foldable. Adding an
    # input-derived zero keeps the subgraph live so `target_ids` survives as a
    # named buffer for the apply pass. The value is unchanged.
    zero = (logits.reshape(-1)[:1].sum() * 0.0).to(torch.long)
    targets = self.target_ids + zero
    logits_new = logits.new_full(
        (logits.shape[0], self.config.vocab_size),
        float("-inf"),
    )
    logits_new[:, targets] = logits
    return logits_new
