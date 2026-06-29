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
from rebel import CompileContext
from vllm.distributed import get_dp_group, get_pp_group, get_tp_group

from vllm_rbln import envs
from vllm_rbln.compilation.backends import rbln_backend

CompiledTarget = TypeVar("CompiledTarget")

_DYNAMO_CONFIGURED = False


def _ensure_torch_dynamo_configured() -> None:
    """Apply dynamo settings for RBLN compilation."""
    global _DYNAMO_CONFIGURED
    if _DYNAMO_CONFIGURED:
        return

    # To prevent nn.modules parameters to be modmel input, set false.
    # If this flag is set, nn.modules parameters are treated as model input.
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    torch._dynamo.config.cache_size_limit = 64

    _DYNAMO_CONFIGURED = True


def create_compile_context(
    use_weight_sharing: bool = False, use_global_ctx: bool = False
) -> CompileContext:
    return CompileContext(
        use_weight_sharing=use_weight_sharing, use_global_ctx=use_global_ctx
    )


def build_process_group_dict() -> dict[str, list[int]]:
    """Build process group metadata consumed by the RBLN torch.compile backend."""
    tp = get_tp_group()
    pp = get_pp_group()
    dp = get_dp_group()

    return {
        tp.device_group.group_name: tp.ranks,
        tp.cpu_group.group_name: tp.ranks,
        pp.device_group.group_name: pp.ranks,
        pp.cpu_group.group_name: pp.ranks,
        dp.device_group.group_name: dp.ranks,
        dp.cpu_group.group_name: dp.ranks,
    }


def compile(
    target: CompiledTarget,
    *,
    backend: str | Callable = rbln_backend,
    dynamic: bool = False,
    fullgraph: bool = False,
    compile_context: CompileContext | None = None,
    num_devices: int | None = None,
    model_trace_method: str = "",
    process_group_dict: dict[str, list[int]] | None = None,
    guard_filter_fn: Callable | None = None,
    runtime_holder: list | None = None,
    mode: str = "",
    cache_dir: str = "",
) -> CompiledTarget:
    _ensure_torch_dynamo_configured()

    options = {}

    def set_option(key: str, value: Any) -> None:
        if value is None or value == "":
            return
        options[key] = value

    set_option("compile_context", compile_context)
    set_option("num_devices", num_devices)
    set_option("model_trace_method", model_trace_method)
    set_option("process_group_dict", process_group_dict)
    set_option("guard_filter_fn", guard_filter_fn)
    set_option("_runtime_holder", runtime_holder)
    set_option("mode", mode)
    if not envs.VLLM_DISABLE_COMPILE_CACHE:
        set_option("cache_dir", cache_dir or os.path.join(envs.VLLM_CACHE_ROOT, "rbln"))

    return cast(
        CompiledTarget,
        torch.compile(
            target,
            backend=backend,
            dynamic=dynamic,
            fullgraph=fullgraph,
            options=options,
        ),
    )
