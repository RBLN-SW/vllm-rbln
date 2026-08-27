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

import sys
from types import CodeType, FunctionType, MethodType
from typing import Any

import torch
from torch._C._profiler import _RecordFunctionFast
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.sequence import IntermediateTensors

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

# FunctionType's constructor takes five of a function's slots and the rest have
# to be assigned, or the clone is not the target with a different body. Keyword
# defaults are the one that changes results rather than just introspection.
# test_dispatch.py enumerates the slots off the type, so a Python release that
# adds one fails the test instead of silently dropping it here.
_COPIED_SLOTS = (
    "__kwdefaults__",
    "__qualname__",
    "__module__",
    "__doc__",
    "__annotations__",
    # A function carries type params only from 3.12, as does the test's enumeration.
    *(("__type_params__",) if sys.version_info >= (3, 12) else ()),
)


def _value_key(value: Any) -> Any:
    """The part of an argument that decides which graph it belongs to.

    A tensor contributes the members Dynamo's TENSOR_MATCH compares, less
    requires_grad, which the step loop's inference_mode pins.
    """
    if isinstance(value, torch.Tensor):
        return value.shape, value.dtype, value.device, value.stride()
    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.dtype)):
        # Bare, True would fold into 1 and 1 into 1.0: Python hashes them equal
        # where Dynamo specialises them apart.
        return type(value), value
    if isinstance(value, IntermediateTensors):
        return tuple((k, _value_key(t)) for k, t in value.items())
    raise TypeError(
        f"Dispatcher cannot key a {type(value).__name__} argument. Dynamo may "
        "guard on its value, and a key that does not carry it would serve the "
        "graph recorded for a different value."
    )


def _forward_context_key() -> tuple[Any, ...]:
    """Graph-selecting state that arrives through the forward context.

    The staged tensors do not carry all of it: two calls with identical input
    shapes still need different graphs when the DP padding width differs (the
    graph reads it as `dp_metadata.max_pads_across_dp.shape[0]`), when the
    attention metadata says prefill rather than decode, or when a KV connector
    is attached (which changes the attention wrapper's graph).

    Not every dispatched target runs inside a forward context -- the model
    forward does, the medusa drafter runs after execute_model has left it -- so
    a missing context is a normal state and contributes nothing to the key.

    A missing context, missing DP metadata and a missing pad width are states
    that really occur. A missing attribute is not, and must not be defaulted:
    it would drop a key component instead of failing.
    """
    if not is_forward_context_available():
        return (None, None, has_kv_transfer_group())

    ctx = get_forward_context()

    dp_metadata = ctx.dp_metadata
    max_pads = None if dp_metadata is None else dp_metadata.max_pads_across_dp
    max_pad = None if max_pads is None else max_pads.shape[0]

    attn_metadata = ctx.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)

    return (
        None if attn_metadata is None else attn_metadata.is_prefill,
        max_pad,
        has_kv_transfer_group(),
    )


class Dispatcher:
    """Owns the graph selection that Dynamo's cache lookup normally does.

    Deciding which compiled graph a call belongs to costs a frame-eval entry
    plus a guard-tree walk every step, and the answer is something the runner
    already knows. So this keeps the mapping itself: the first call for a key
    goes through Dynamo (which compiles it if needed) and the transformed
    bytecode is recorded as a clone of the target, and every later call with
    that key runs the clone directly -- no frame eval, no guards.

    A clone rather than the target with its `__code__` swapped, because a swap
    is visible to every other holder of the target for as long as it is
    installed. One clone per key keeps the fast path free of shared mutable
    state, and lets a bound method be dispatched without touching the code
    object its whole class shares.

    The table is a cache, not an allow-list: an unseen key is compiled on the
    spot rather than rejected. Key completeness is a precondition this class
    does not check -- once a key is registered its graph is served forever, so
    a key coarser than Dynamo's guards is wrong rather than reported. What
    keeps it honest is `_value_key` refusing an argument it cannot key.
    """

    def __init__(self, target: Any, compiled: Any) -> None:
        # Annotated Any, not FunctionType: a function is a descriptor, so a
        # FunctionType-annotated attribute reads back as MethodType in mypy.
        self._function: Any
        self._instance: Any
        if isinstance(target, MethodType):
            self._function = target.__func__
            self._instance = target.__self__
        elif isinstance(target, FunctionType):
            self._function = target
            self._instance = None
        else:
            raise TypeError(
                "Dispatcher needs a function or a bound method, got "
                f"{type(target).__name__}"
            )
        self._compiled = compiled
        self._original_code: CodeType = self._function.__code__
        self._graphs: dict[tuple, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = (
            tuple(_value_key(a) for a in args),
            tuple((name, _value_key(v)) for name, v in sorted(kwargs.items())),
            *_forward_context_key(),
        )
        graph, scope = self._graphs.get(key, (None, None))
        if graph is None:
            result = self._compiled(*args, **kwargs)
            self._register(key)
            return result
        with scope:
            return graph(*args, **kwargs)

    def _register(self, key: tuple) -> None:
        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(
            self._original_code
        )
        if not entries:
            # Dynamo ran the frame without leaving a cache entry -- the frame was
            # skipped, or an entry it had was invalidated. Nothing to dispatch to,
            # so this key keeps taking the Dynamo path.
            logger.warning_once(
                "no Dynamo cache entry for %s; keys that reach this path keep "
                "paying the frame-eval cost",
                self._original_code.co_name,
            )
            return

        # Cache entries are MRU-ordered, so the call above left the graph it
        # used at the head of the list. That identification holds only while
        # nothing else calls the target in between, which the serial step loop
        # guarantees; a second concurrent caller would need this serialised.
        entry = entries[0]
        # Bypassing the frame eval also bypasses the profiler scope it opens, so
        # the region is invisible unless we open one. Dynamo's compile id keeps a
        # dispatched trace comparable with one taken without the dispatcher, and
        # this scope is gated on the profiler, not on an env var like vLLM's.
        self._graphs[key] = (
            self._clone(entry.code),
            _RecordFunctionFast(f"Dispatched Region: {entry.compile_id}"),
        )
        logger.debug(
            "dispatch: registered %s as entry %d for %s",
            entry.code.co_name,
            len(self._graphs),
            self._original_code.co_name,
        )

    def _clone(self, code: CodeType) -> Any:
        clone = FunctionType(
            code,
            self._function.__globals__,
            self._function.__name__,
            self._function.__defaults__,
            self._function.__closure__,
        )
        for slot in _COPIED_SLOTS:
            setattr(clone, slot, getattr(self._function, slot))
        clone.__dict__.update(self._function.__dict__)
        return clone if self._instance is None else MethodType(clone, self._instance)
