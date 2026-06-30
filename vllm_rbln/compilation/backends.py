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
import itertools
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
from rebel.core.torch_compile import rbln_backend as rbln_torch_compile_backend

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

_compile_counter = itertools.count(1)
_current_stage = ContextVar("rbln_backend_current_stage", default="runtime")
_SELF_FILE = Path(__file__).resolve()

_DTYPE_SHORT = {
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "f8e4m3fn",
    torch.float8_e5m2: "f8e5m2",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.bool: "bool",
}


def fmt_dtype(dtype: torch.dtype) -> str:
    return _DTYPE_SHORT.get(dtype, str(dtype).replace("torch.", ""))


def fmt_shape(x: torch.Tensor) -> str:
    return "[" + ",".join(str(d) for d in x.shape) + "]"


def placeholder_names(gm: fx.GraphModule, max_len: int = 20) -> list[str]:
    def fmt_name(name: str) -> str:
        if len(name) <= max_len:
            return name
        return ".." + name[-max_len:]

    return [fmt_name(node.name) for node in gm.graph.nodes if node.op == "placeholder"]


def fmt_input(name: str, x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return f"{name}:{fmt_dtype(x.dtype)}{fmt_shape(x)}"
    return f"{name}:{type(x).__name__}={x!r}"


@contextmanager
def set_compile_stage(stage: str):
    token = _current_stage.set(stage)
    try:
        yield
    finally:
        _current_stage.reset(token)


def current_stage() -> str:
    return _current_stage.get()


def _format_callsite(frame: traceback.FrameSummary) -> str:
    path = Path(frame.filename).resolve()
    parts = path.parts

    if "vllm_rbln" in parts:
        idx = parts.index("vllm_rbln")
        rel_path = "/".join(parts[idx:])
    else:
        rel_path = path.name

    return f"{rel_path}:{frame.lineno}({frame.name})"


def _find_rbln_callsite() -> str:
    # Walk from the current frame outward and return the nearest vllm_rbln
    # frame that triggered this Dynamo backend invocation.
    for frame in reversed(traceback.extract_stack()):
        if frame.filename.startswith("<"):
            continue

        path = Path(frame.filename).resolve()
        if path == _SELF_FILE:
            continue
        if "vllm_rbln" not in path.parts:
            continue

        return _format_callsite(frame)

    return "unknown"


# TODO(RBLN): Implement RBLN-specific backend like VllmBackend
def _rbln_backend(
    graph: fx.GraphModule, example_inputs: Sequence[Any], **kwargs: Any
) -> Any:
    compile_id = next(_compile_counter)
    names = placeholder_names(graph)
    parts = []
    for i, x in enumerate(example_inputs):
        name = names[i] if i < len(names) else f"arg{i}"
        parts.append(fmt_input(name, x))

    stage = current_stage()
    log_fn = logger.warning if stage == "runtime" else logger.debug
    log_fn(
        "rbln_backend: stage=%s #%d caller=%s graph inputs=%s",
        stage,
        compile_id,
        _find_rbln_callsite(),
        "; ".join(parts),
    )
    return rbln_torch_compile_backend(graph, example_inputs, **kwargs)


rbln_backend = _rbln_backend
