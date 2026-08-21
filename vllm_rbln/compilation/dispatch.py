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

from types import CodeType
from typing import Any

import torch
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def _value_key(value: Any) -> Any:
    """The part of an argument that decides which graph it belongs to."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, torch.Tensor):
        # torch.Size is a tuple, so this stays hashable and allocation-free.
        return value.shape, value.dtype
    if hasattr(value, "items"):  # IntermediateTensors and friends
        return tuple((k, t.shape, t.dtype) for k, t in value.items())
    return type(value).__name__


def _forward_context_key() -> tuple[Any, ...]:
    """Graph-selecting state that arrives through the forward context.

    The staged tensors do not carry all of it: two calls with identical input
    shapes still need different graphs when the DP padding width differs (the
    graph reads it as `dp_metadata.max_pads_across_dp.shape[0]`), when the
    attention metadata says prefill rather than decode, or when a KV connector
    is attached (which changes the attention wrapper's graph).

    Not every dispatched target runs inside a forward context -- the model
    forward does, the samplers run after it has been exited -- so a missing
    context is a normal state and contributes nothing to the key.
    """
    if not is_forward_context_available():
        return (None, None, has_kv_transfer_group())

    ctx = get_forward_context()

    dp_metadata = getattr(ctx, "dp_metadata", None)
    max_pads = getattr(dp_metadata, "max_pads_across_dp", None)
    max_pad = None if max_pads is None else max_pads.shape[0]

    attn_metadata = ctx.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    elif isinstance(attn_metadata, list):
        first = attn_metadata[0] if attn_metadata else None
        attn_metadata = next(iter(first.values()), None) if first else None

    return (
        getattr(attn_metadata, "is_prefill", None),
        max_pad,
        has_kv_transfer_group(),
    )


class Dispatcher:
    """Owns the graph selection that Dynamo's cache lookup normally does.

    Deciding which compiled graph a call belongs to costs a frame-eval entry
    plus a guard-tree walk every step, and the answer is something the runner
    already knows. So this keeps the mapping itself: the first call for a key
    goes through Dynamo (which compiles it if needed) and the transformed
    bytecode is recorded, and every later call with that key runs the bytecode
    directly -- no frame eval, no guards.

    The table is a cache, not an allow-list: an unseen key is compiled on the
    spot rather than rejected. What is rejected is a key that maps to two
    different graphs -- that means the key is coarser than Dynamo's guards, and
    serving either graph would be wrong.

    Assumes one caller at a time: the swap goes through the function object, so
    two threads calling the same target with different keys would collide. The
    RBLN platform forces async scheduling off, so the step loop is serial.
    """

    def __init__(self, target: Any, compiled: Any) -> None:
        if not hasattr(target, "__code__"):
            raise TypeError(
                "Dispatcher needs a plain function to swap code on, got "
                f"{type(target).__name__}"
            )
        self._target = target
        self._compiled = compiled
        self._original_code: CodeType = target.__code__
        self._codes: dict[tuple, CodeType] = {}
        self._unregistered = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = (
            tuple(_value_key(a) for a in args),
            tuple((name, _value_key(v)) for name, v in sorted(kwargs.items())),
            *_forward_context_key(),
        )
        code = self._codes.get(key)
        if code is None:
            result = self._compiled(*args, **kwargs)
            self._register(key)
            return result

        self._target.__code__ = code
        try:
            return self._target(*args, **kwargs)
        finally:
            self._target.__code__ = self._original_code

    def _register(self, key: tuple) -> None:
        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(
            self._original_code
        )
        if not entries:
            # Dynamo ran the frame without leaving a cache entry (it bails out
            # once accumulated_cache_size_limit is hit, for one). Nothing to
            # dispatch to, so this key keeps taking the Dynamo path.
            self._unregistered += 1
            logger.warning_once(
                "no Dynamo cache entry for %s; %d key(s) will keep paying the "
                "frame-eval cost",
                self._original_code.co_name,
                self._unregistered,
            )
            return

        # Cache entries are MRU-ordered, so the call above put the graph it used
        # at the head of the list.
        code = entries[0].code
        previous = self._codes.get(key)
        if previous is not None and previous is not code:
            raise RuntimeError(
                f"dispatch key {key} selected two different graphs for "
                f"{self._original_code.co_name}; the key is missing a factor "
                "that Dynamo's guards do distinguish"
            )
        self._codes[key] = code
        logger.debug(
            "dispatch: registered %s as entry %d for %s",
            code.co_name,
            len(self._codes),
            self._original_code.co_name,
        )
