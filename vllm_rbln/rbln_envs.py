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

import os
from typing import TYPE_CHECKING

# Canonical readers live in torch_rbln so the OS-env names
# (`RBLN_PROFILER`, `RBLN_USE_CUSTOM_KERNEL`) have one source of truth.
from torch_rbln import (  # noqa: E402
    is_profiler_enabled as _torch_rbln_is_profiler_enabled,
)
from torch_rbln import (
    use_custom_kernel as _torch_rbln_use_custom_kernel,
)
from vllm.envs import environment_variables as vllm_envs

if TYPE_CHECKING:
    VLLM_RBLN_COMPILE_MODEL: bool = True
    VLLM_RBLN_COMPILE_STRICT_MODE: bool = False
    VLLM_RBLN_TP_SIZE: int = 1
    VLLM_RBLN_ENABLE_WARM_UP: bool = True
    VLLM_RBLN_SPECIALIZE_MOE_DECODE: bool = True
    VLLM_RBLN_FLASH_CAUSAL_ATTN: bool = True
    VLLM_RBLN_DISABLE_MM: bool = False
    VLLM_RBLN_DP_IMPL: str = "padded_decode"
    VLLM_RBLN_USE_MOE_TOKENS_MASK: bool = True
    VLLM_RBLN_ENFORCE_MODEL_FP32: bool = False
    VLLM_RBLN_MOE_CUSTOM_KERNEL: bool = True
    VLLM_RBLN_MOE_USE_OPT_KERNEL: bool = True
    VLLM_RBLN_DP_INPUT_ALL_GATHER: bool = True
    VLLM_RBLN_LOGITS_ALL_GATHER: bool = True
    VLLM_RBLN_NUM_RAY_NODES: int = 1
    VLLM_RBLN_METRICS: bool = False
    VLLM_RBLN_NUMA: bool = True
    VLLM_RBLN_SORT_BATCH: bool = False
    VLLM_RBLN_USE_CUSTOM_KERNEL: bool = False
    VLLM_RBLN_AUTO_PORT: bool = True
    VLLM_RBLN_MOE_REDUCE_SCATTER: bool = False
    VLLM_RBLN_SUB_BLOCK_CACHE: bool = True
    VLLM_RBLN_USE_FLASHINFER: bool = False


def get_dp_impl() -> str:
    dp_impl = os.environ.get("VLLM_RBLN_DP_IMPL")
    if dp_impl is None:
        return "padded_decode"
    # default is padded_decode
    # dummy_prefill will be deprecated in the future
    choices = set(["padded_decode", "dummy_prefill"])
    current_impl = dp_impl.lower()
    if current_impl not in choices:
        raise ValueError(
            f"Invalid VLLM_RBLN_DP_IMPL: {current_impl}, Valid choices: {choices}"
        )
    return current_impl


# extended environments
environment_variables = {
    **vllm_envs,
    # If true, will compile models using torch.compile.
    # Otherwise, run the CPU eager mode, if possible.
    "VLLM_RBLN_COMPILE_MODEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_MODEL", "True").lower() in ("true", "1")
        )
    ),
    # If true, will compile models using strict mode.
    "VLLM_RBLN_COMPILE_STRICT_MODE": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_STRICT_MODE", "False").lower()
            in ("true", "1")
        )
    ),
    # TP Size for RSD.
    "VLLM_RBLN_TP_SIZE": lambda: int(os.environ.get("VLLM_RBLN_TP_SIZE", 1)),
    # Enable warm_up
    "VLLM_RBLN_ENABLE_WARM_UP": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENABLE_WARM_UP", "True").lower() in ("true", "1")
        )
    ),
    # Use flash attention for causal attention
    "VLLM_RBLN_FLASH_CAUSAL_ATTN": (
        lambda: (
            os.environ.get("VLLM_RBLN_FLASH_CAUSAL_ATTN", "True").lower()
            in ("true", "1")
        )
    ),
    # Disable multimodal input
    "VLLM_RBLN_DISABLE_MM": (
        lambda: os.environ.get("VLLM_RBLN_DISABLE_MM", "False").lower() in ("true", "1")
    ),
    # DP implementation, see choices in get_dp_impl
    "VLLM_RBLN_DP_IMPL": get_dp_impl,
    # If true, it uses the tokens mask applied to moe expert kernel
    "VLLM_RBLN_USE_MOE_TOKENS_MASK": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_MOE_TOKENS_MASK", "True").lower()
            in ("true", "1")
        )
    ),
    # If true, it specializes the cases where all instances are at decode stage
    "VLLM_RBLN_SPECIALIZE_MOE_DECODE": (
        lambda: (
            os.environ.get("VLLM_RBLN_SPECIALIZE_MOE_DECODE", "True").lower()
            in ("true", "1")
        )
    ),
    # enforce model data type into fp32 not model_config.dtype
    "VLLM_RBLN_ENFORCE_MODEL_FP32": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENFORCE_MODEL_FP32", "False").lower()
            in ("true", "1")
        )
    ),
    # use moe custom kernel, by default disabled
    "VLLM_RBLN_MOE_CUSTOM_KERNEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_MOE_CUSTOM_KERNEL", "True").lower()
            in ("true", "1")
        )
    ),
    # enable moe optimization if RBLN_MoE_OPT is set to 1
    "VLLM_RBLN_MOE_USE_OPT_KERNEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_MOE_USE_OPT_KERNEL", "True").lower()
            in ("true", "1")
        )
    ),
    # DP_INPUT_ALL_GATHER, use DP input all_gather
    "VLLM_RBLN_DP_INPUT_ALL_GATHER": (
        lambda: (
            os.environ.get("VLLM_RBLN_DP_INPUT_ALL_GATHER", "True").lower()
            in ("true", "1")
        )
    ),
    # LOGITS_ALL_GATHER, include logits all_gather into model compilation
    "VLLM_RBLN_LOGITS_ALL_GATHER": (
        lambda: (
            os.environ.get("VLLM_RBLN_LOGITS_ALL_GATHER", "True").lower()
            in ("true", "1")
        )
    ),
    # Number of Ray nodes
    "VLLM_RBLN_NUM_RAY_NODES": lambda: int(
        os.environ.get("VLLM_RBLN_NUM_RAY_NODES", 1)
    ),
    "VLLM_RBLN_METRICS": (
        lambda: os.environ.get("VLLM_RBLN_METRICS", "False").lower() in ("true", "1")
    ),
    # Enable NUMA-based CPU affinity binding for OpenMP threads
    "VLLM_RBLN_NUMA": (
        lambda: os.environ.get("VLLM_RBLN_NUMA", "True").lower() in ("true", "1")
    ),
    "VLLM_RBLN_SORT_BATCH": (
        lambda: os.environ.get("VLLM_RBLN_SORT_BATCH", "False").lower() in ("true", "1")
    ),
    # Auto port
    "VLLM_RBLN_AUTO_PORT": (
        lambda: os.environ.get("VLLM_RBLN_AUTO_PORT", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_USE_CUSTOM_KERNEL": _torch_rbln_use_custom_kernel,
    # Use reduce_scatter instead of all_reduce in MoE combine phase
    "VLLM_RBLN_MOE_REDUCE_SCATTER": (
        lambda: (
            os.environ.get("VLLM_RBLN_MOE_REDUCE_SCATTER", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_PROFILER": _torch_rbln_is_profiler_enabled,
    # Enable sub-block prefix caching.
    # Sub-block size equals max_num_batched_tokens (prefill chunk size).
    "VLLM_RBLN_SUB_BLOCK_CACHE": lambda: (
        os.environ.get("VLLM_RBLN_SUB_BLOCK_CACHE", "True").lower() in ("true", "1")
    ),
    # Use flashinfer_rbln paged attention kernels instead of the built-in
    # rbln_custom_ops / rbln_triton_ops attention dispatch.
    # Set VLLM_RBLN_USE_FLASHINFER=1 to enable.
    "VLLM_RBLN_USE_FLASHINFER": lambda: (
        os.environ.get("VLLM_RBLN_USE_FLASHINFER", "False").lower() in ("true", "1")
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


vllm_envs.update(environment_variables)
