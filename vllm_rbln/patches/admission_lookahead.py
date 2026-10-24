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
"""Count the step after the prefill in the full-sequence admission gate."""

import inspect
from contextvars import ContextVar
from typing import Any

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

from vllm_rbln.patches.registry import register_patch

# TODO(vllm-project/vllm#63288): delete this module once the vllm pin has it.
# Held at import time: the registry replaces the attributes outright.
kv_cache_manager_original_allocate_slots = KVCacheManager.allocate_slots
kv_cache_coordinator_original_get_num_blocks_to_allocate = (
    KVCacheCoordinator.get_num_blocks_to_allocate
)
ALLOCATE_SLOTS_SIGNATURE = inspect.signature(kv_cache_manager_original_allocate_slots)

# (decode step tokens, lookahead, max_model_len) of the gated allocate_slots
# call in flight. Unset elsewhere, so upstream's other admission-cap callers
# count what they did.
admission_margin: ContextVar[tuple[int, int, int] | None] = ContextVar(
    "rbln_admission_margin", default=None
)


@register_patch(
    target="vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots",
    reason=(
        "upstream's full-sequence admission gate counts request.num_tokens "
        "alone, but the last prefill chunk allocates the lookahead (RBLN raises "
        "it to num_spec_tokens even for ngram) and the decode step after it "
        "adds the sampled and draft tokens. With no mixed batching, a prompt "
        "admitted short preempts itself there while the decodes holding the "
        "pool cannot run. Upstream has no num_decode_step_tokens."
    ),
)
def patched_allocate_slots(
    self: KVCacheManager,
    *args: Any,
    num_decode_step_tokens: int = 0,
    **kwargs: Any,
) -> KVCacheBlocks | None:
    # Every caller passes full_sequence_must_fit by keyword.
    if not kwargs.get("full_sequence_must_fit"):
        return kv_cache_manager_original_allocate_slots(self, *args, **kwargs)
    call = ALLOCATE_SLOTS_SIGNATURE.bind(self, *args, **kwargs).arguments
    token = admission_margin.set(
        (
            num_decode_step_tokens,
            call.get("num_lookahead_tokens", 0),
            self.max_model_len,
        )
    )
    try:
        return kv_cache_manager_original_allocate_slots(self, *args, **kwargs)
    finally:
        admission_margin.reset(token)


@register_patch(
    target=(
        "vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator"
        ".get_num_blocks_to_allocate"
    ),
    reason=(
        "num_tokens is meant to include the lookahead and num_tokens_main_model "
        "to exclude it, but the admission gate passes the prompt for both. "
        "Adds what the patched allocate_slots recorded, the decode step to both "
        "and the lookahead to num_tokens, on the gate's call only "
        "(apply_admission_cap=True), so upstream's block arithmetic counts."
    ),
)
def patched_get_num_blocks_to_allocate(
    self: KVCacheCoordinator, *args: Any, **kwargs: Any
) -> int:
    margin = admission_margin.get()
    # The gate passes every argument by keyword; a positional one raises.
    if margin is not None and kwargs.get("apply_admission_cap"):
        num_decode_step_tokens, lookahead, max_model_len = margin
        num_tokens_main_model = min(
            kwargs["num_tokens_main_model"] + num_decode_step_tokens, max_model_len
        )
        kwargs["num_tokens_main_model"] = num_tokens_main_model
        kwargs["num_tokens"] = min(num_tokens_main_model + lookahead, max_model_len)
    return kv_cache_coordinator_original_get_num_blocks_to_allocate(
        self, *args, **kwargs
    )
