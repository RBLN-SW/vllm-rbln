# Copyright 2025 Rebellions Inc. All rights reserved.
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
"""Logging wrapper for the rbln torch.compile backend.

Dynamo invokes the backend once per unique guarded input shape; each call
is either a fresh compile (cache miss) or a `.rbln` cache load (cache hit).
We wrap rebel.core.torch_compile.rbln_backend so every invocation is
logged. Invocations outside the warm-up window are flagged as warnings,
since they mean an input shape was not pre-compiled and the compile (slow)
or cache load happened on the serving hot path.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

import torch
from rebel.core.torch_compile import rbln_backend as _rbln_backend

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_warmup_active: bool = False

_SELF_FILENAME = os.path.abspath(__file__)

_CALL_CHAIN_DEPTH = 3


def set_warmup_active(active: bool) -> None:
    global _warmup_active
    _warmup_active = active


def _summarize_inputs(inputs: list[Any]) -> str:
    parts = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            parts.append(f"{tuple(t.shape)}:{t.dtype}")
        else:
            parts.append(type(t).__name__)
    return "[" + ", ".join(parts) + "]"


def _find_call_chain(depth: int = _CALL_CHAIN_DEPTH) -> str:
    """Walk the stack and return up to `depth` user-level frames as a chain.

    rbln_backend is invoked from deep inside Dynamo; we skip torch / rebel /
    this file's frames and format the deepest user frame leftmost.
    """
    chain: list[str] = []
    for frame in reversed(traceback.extract_stack()):
        path = frame.filename
        if "/torch/" in path or "/rebel/" in path:
            continue
        if os.path.abspath(path) == _SELF_FILENAME:
            continue
        chain.append(f"{os.path.basename(path)}:{frame.lineno}({frame.name})")
        if len(chain) >= depth:
            break
    return " <- ".join(chain) if chain else "?"


def logged_rbln_backend(graph_module: Any, inputs: list[Any], **kwargs: Any) -> Any:
    in_warmup = _warmup_active
    log = logger.info if in_warmup else logger.warning
    phase = "warm-up" if in_warmup else "HOT PATH"
    log(
        "rbln_backend [%s] %s: inputs=%s",
        phase,
        _find_call_chain(),
        _summarize_inputs(inputs),
    )
    t0 = time.perf_counter()
    result = _rbln_backend(graph_module, inputs, **kwargs)
    log("rbln_backend done: %.2fs", time.perf_counter() - t0)
    return result
