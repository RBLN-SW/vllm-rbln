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

from vllm.envs import environment_variables as vllm_envs

if TYPE_CHECKING:
    VLLM_RBLN_COMPILE_MODEL: bool = True
    VLLM_RBLN_COMPILE_STRICT_MODE: bool = False
    VLLM_RBLN_TP_SIZE: int = 1
    VLLM_RBLN_SAMPLER: bool = True
    VLLM_RBLN_ENABLE_WARM_UP: bool = True
    VLLM_RBLN_USE_VLLM_MODEL: bool = False
    VLLM_RBLN_SPECIALIZE_MOE_DECODE: bool = True
    VLLM_RBLN_FLASH_CAUSAL_ATTN: bool = True
    VLLM_RBLN_BATCH_ATTN_OPT: bool = False
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
    VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY: str = "exponential"
    VLLM_RBLN_DECODE_BATCH_BUCKET_MIN: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_STEP: int = 2
    VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS: list[int] = []
    VLLM_RBLN_USE_CUSTOM_KERNEL: bool = False
    VLLM_RBLN_AUTO_PORT: bool = True


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


def get_decode_batch_bucket_strategy() -> str:
    decode_batch_bucket_strategy = os.environ.get(
        "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY"
    )
    if decode_batch_bucket_strategy is None:
        return "exponential"
    choices = set(["exponential", "exp", "linear", "manual"])
    current_strategy = decode_batch_bucket_strategy.lower()
    if current_strategy not in choices:
        raise ValueError(
            f"Invalid VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY: {current_strategy}, "
            f"Valid choices: {choices}",
        )
    if current_strategy == "manual":
        buckets = get_decode_batch_bucket_manual_buckets()
        if len(buckets) < 1:
            raise ValueError(
                "There must be at least one decode batch size in the manual buckets"
            )
    elif current_strategy == "exp":
        return "exponential"
    return current_strategy


def get_decode_batch_bucket_manual_buckets() -> list[int]:
    manual_buckets = os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS")
    if manual_buckets is None:
        return []
    try:
        buckets = [int(bucket) for bucket in manual_buckets.split(",")]
        if any(bucket <= 0 for bucket in buckets):
            raise ValueError(
                "All decode batch bucket manual buckets must be greater than 0"
            )
        if len(buckets) < 1:
            raise ValueError(
                "There must be at least one decode batch size in the manual buckets"
            )
        if len(buckets) != len(set(buckets)):
            raise ValueError("All decode batch bucket manual buckets must be unique")
        return buckets
    except ValueError as e:
        raise ValueError(
            f"Invalid VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS: "
            f"{manual_buckets}, {e}"
        ) from e


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
    # Use customized sampler
    "VLLM_RBLN_SAMPLER": (
        lambda: os.environ.get("VLLM_RBLN_SAMPLER", "True").lower() in ("true", "1")
    ),
    # Enable warm_up
    "VLLM_RBLN_ENABLE_WARM_UP": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENABLE_WARM_UP", "True").lower() in ("true", "1")
        )
    ),
    # If true, it uses the natively compiled vLLM model
    # rather than the optimum-rbln compiled model.
    "VLLM_RBLN_USE_VLLM_MODEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_VLLM_MODEL", "False").lower() in ("true", "1")
        )
    ),
    # Use flash attention for causal attention
    "VLLM_RBLN_FLASH_CAUSAL_ATTN": (
        lambda: (
            os.environ.get("VLLM_RBLN_FLASH_CAUSAL_ATTN", "True").lower()
            in ("true", "1")
        )
    ),
    # Use batch attention optimization for paged attention
    "VLLM_RBLN_BATCH_ATTN_OPT": (
        lambda: (
            os.environ.get("VLLM_RBLN_BATCH_ATTN_OPT", "False").lower() in ("true", "1")
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
    # Decode batch bucket strategy [exponential, exp, linear, manual]
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY": get_decode_batch_bucket_strategy,
    # Decode batch bucket min
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MIN": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_MIN", 1)
    ),
    # Decode batch bucket step
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STEP": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_STEP", 2)
    ),
    # Decode batch bucket limit
    "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT", 1)
    ),
    # Auto port
    "VLLM_RBLN_AUTO_PORT": (
        lambda: os.environ.get("VLLM_RBLN_AUTO_PORT", "False").lower() in ("true", "1")
    ),
    # Decode batch bucket manual buckets
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": get_decode_batch_bucket_manual_buckets,  # noqa E501
    "VLLM_RBLN_USE_CUSTOM_KERNEL": (
        lambda: os.environ.get("RBLN_USE_CUSTOM_KERNEL", "False").lower()
        in ("true", "1")
    ),
    "VLLM_RBLN_PROFILER": (
        lambda: os.environ.get("RBLN_PROFILER", "False").lower() in ("true", "1")
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
