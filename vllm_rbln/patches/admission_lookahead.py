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
"""Count the spec decode lookahead in the full-sequence admission gate."""

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

# (lookahead, max_model_len) of the gated allocate_slots call in flight. Unset
# elsewhere, so upstream's other admission-cap callers count what they did.
admission_lookahead: ContextVar[tuple[int, int] | None] = ContextVar(
    "rbln_admission_lookahead", default=None
)


@register_patch(
    target="vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots",
    reason=(
        "upstream's full-sequence admission gate counts request.num_tokens "
        "alone, but the final prefill chunk also allocates the spec decode "
        "lookahead, which RBLN raises to num_spec_tokens even for ngram. With "
        "no mixed batching, a prompt admitted one block short preempts itself "
        "on that chunk while the decodes holding the pool never run. Upstream "
        "has no hook for the token count the gate passes on."
    ),
)
def patched_allocate_slots(
    self: KVCacheManager, *args: Any, **kwargs: Any
) -> KVCacheBlocks | None:
    # Every caller passes full_sequence_must_fit by keyword.
    if not kwargs.get("full_sequence_must_fit"):
        return kv_cache_manager_original_allocate_slots(self, *args, **kwargs)
    call = ALLOCATE_SLOTS_SIGNATURE.bind(self, *args, **kwargs).arguments
    token = admission_lookahead.set(
        (call.get("num_lookahead_tokens", 0), self.max_model_len)
    )
    try:
        return kv_cache_manager_original_allocate_slots(self, *args, **kwargs)
    finally:
        admission_lookahead.reset(token)


@register_patch(
    target=(
        "vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator"
        ".get_num_blocks_to_allocate"
    ),
    reason=(
        "num_tokens is meant to include the lookahead and num_tokens_main_model "
        "to exclude it, but the admission gate passes the prompt for both. "
        "Adds the lookahead the patched allocate_slots recorded, to the gate's "
        "call only (apply_admission_cap=True), so upstream's block arithmetic "
        "does the counting."
    ),
)
def patched_get_num_blocks_to_allocate(
    self: KVCacheCoordinator, *args: Any, **kwargs: Any
) -> int:
    gate = admission_lookahead.get()
    # The gate passes every argument by keyword; a positional num_tokens raises.
    if gate is not None and kwargs.get("apply_admission_cap"):
        lookahead, max_model_len = gate
        kwargs["num_tokens"] = min(kwargs["num_tokens"] + lookahead, max_model_len)
    return kv_cache_coordinator_original_get_num_blocks_to_allocate(
        self, *args, **kwargs
    )
