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

import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch

from vllm_rbln import envs
from vllm_rbln.compilation.backends import rbln_backend
from vllm_rbln.compilation.dispatch import Dispatcher

CompiledTarget = TypeVar("CompiledTarget")

_DYNAMO_CONFIGURED = False


def _ensure_torch_dynamo_configured() -> None:
    """Apply dynamo settings for RBLN compilation."""
    global _DYNAMO_CONFIGURED
    if _DYNAMO_CONFIGURED:
        return

    torch._dynamo.config.cache_size_limit = 64

    _DYNAMO_CONFIGURED = True


def compile(
    target: CompiledTarget,
    *,
    backend: str | Callable = rbln_backend,
    dynamic: bool = False,
    fullgraph: bool = False,
    num_devices: int | None = None,
    guard_filter_fn: Callable | None = None,
    use_cache: bool = True,
    cache_dir: str = "",
    use_direct_dispatch: bool = False,
) -> CompiledTarget:
    if use_direct_dispatch and not fullgraph:
        # A dispatched call runs one code object, so whatever Dynamo leaves
        # outside the graph is unreachable: a graph break's resume function only
        # runs compiled under the frame eval, and a recompile-limit bail-out adds
        # no cache entry, so _register binds the previous call's graph to the new
        # key. fullgraph turns both into an exception.
        raise ValueError("use_direct_dispatch requires fullgraph=True")

    _ensure_torch_dynamo_configured()

    options: dict[str, Any] = {}
    if num_devices is not None:
        options["devices"] = num_devices
    if guard_filter_fn is not None:
        options["guard_filter_fn"] = guard_filter_fn
    if envs.VLLM_RBLN_COMPILE_ONLY:
        options["mode"] = "compile_only"
    if use_cache and not envs.VLLM_DISABLE_COMPILE_CACHE:
        options["cache_dir"] = cache_dir or os.path.join(envs.VLLM_CACHE_ROOT, "rbln")

    compiled = torch.compile(
        target,
        backend=backend,
        dynamic=dynamic,
        fullgraph=fullgraph,
        options=options,
    )
    if use_direct_dispatch:
        return cast(CompiledTarget, Dispatcher(target, compiled))
    return cast(CompiledTarget, compiled)
