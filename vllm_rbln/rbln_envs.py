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

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    # ====================================================================
    # Path selector: the value of VLLM_RBLN_USE_VLLM_MODEL splits the model
    # path in two, which decides which variables below take effect.
    # ====================================================================
    VLLM_RBLN_USE_VLLM_MODEL: bool = False

    # ====================================================================
    # Common: read regardless of the VLLM_RBLN_USE_VLLM_MODEL value
    # ====================================================================
    VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK: int = 1
    VLLM_RBLN_SAMPLER: bool = True
    VLLM_RBLN_ENABLE_WARM_UP: bool = True
    VLLM_RBLN_METRICS: bool = False
    VLLM_RBLN_METRICS_FILE: str = ""
    VLLM_RBLN_NUMA: bool = True

    # ====================================================================
    # Read only when VLLM_RBLN_USE_VLLM_MODEL=False
    # ====================================================================
    # (none currently; only the common variables are read in this case)

    # ====================================================================
    # Read only when VLLM_RBLN_USE_VLLM_MODEL=True
    # ====================================================================
    # --- COMPILE / RUNTIME ---
    VLLM_RBLN_COMPILE_MODEL: bool = True
    VLLM_RBLN_COMPILE_STRICT_MODE: bool = False
    VLLM_RBLN_COMPILE_ONLY: bool = False
    VLLM_RBLN_USE_DEVICE_TENSOR: bool = False
    VLLM_RBLN_DISABLE_OFFLOAD: bool = False
    # Default follows VLLM_RBLN_USE_DEVICE_TENSOR (see use_auto_port), so it is
    # False unless device-tensor mode is enabled.
    VLLM_RBLN_AUTO_PORT: bool = False
    VLLM_RBLN_ENFORCE_MODEL_FP32: bool = False
    VLLM_RBLN_NUM_RAY_NODES: int = 1
    VLLM_RBLN_PROFILER: bool = False
    # --- ATTENTION ---
    VLLM_RBLN_FLASH_CAUSAL_ATTN: bool = True
    VLLM_RBLN_BATCH_ATTN_OPT: bool = False
    VLLM_RBLN_USE_CUSTOM_KERNEL: bool = False
    # --- MODEL INPUT / SCHEDULING ---
    VLLM_RBLN_DISABLE_MM: bool = False
    VLLM_RBLN_SORT_BATCH: bool = False
    VLLM_RBLN_SUB_BLOCK_CACHE: bool = True
    VLLM_RBLN_LOGITS_ALL_GATHER: bool = True
    # --- DATA PARALLEL ---
    VLLM_RBLN_DP_IMPL: str = "padded_decode"
    # --- MOE ---
    VLLM_RBLN_SPECIALIZE_MOE_DECODE: bool = True
    VLLM_RBLN_USE_MOE_TOKENS_MASK: bool = True
    VLLM_RBLN_MOE_REDUCE_SCATTER: bool = False
    VLLM_RBLN_DISPATCH_ALL2ALL: bool = False
    VLLM_RBLN_COMBINE_ALL2ALL: bool = False
    # --- DECODE BATCH BUCKET ---
    VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY: str = "exponential"
    VLLM_RBLN_DECODE_BATCH_BUCKET_MIN: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_STEP: int = 2
    VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS: list[int] = []
    # --- QUANTIZATION ---
    VLLM_RBLN_USE_W8A8_FP8: bool = False
    # --- NIXL ---
    VLLM_RBLN_NIXL_SWA_VIEW_OPT: bool = False


def get_num_devices_per_local_rank() -> int:
    """Number of NPU devices assigned to each local rank.

    Resolves ``VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK``. For backward
    compatibility the deprecated ``VLLM_RBLN_TP_SIZE`` is still honored as a
    fallback when the new variable is unset, and emits a deprecation warning.
    """
    new_value = os.environ.get("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK")
    legacy_value = os.environ.get("VLLM_RBLN_TP_SIZE")

    if legacy_value is not None:
        logger.warning_once(
            "VLLM_RBLN_TP_SIZE is deprecated and will be removed in a future "
            "release. Please use VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK instead."
        )
        if new_value is None:
            return int(legacy_value)

    return int(new_value) if new_value is not None else 1


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


def use_auto_port() -> bool:
    raw = os.environ.get("VLLM_RBLN_AUTO_PORT")
    if raw is not None:
        return raw.lower() in ("true", "1")
    # Default follows device-tensor mode: auto port is on when
    # VLLM_RBLN_USE_DEVICE_TENSOR is enabled.
    return os.environ.get("VLLM_RBLN_USE_DEVICE_TENSOR", "False").lower() in (
        "true",
        "1",
    )


# extended environments
environment_variables = {
    **vllm_envs,
    # ====================================================================
    # Path selector: the value of VLLM_RBLN_USE_VLLM_MODEL splits the model
    # path in two, which decides which variables below take effect.
    # ====================================================================
    # Splits the model path in two; selects which model implementation is used.
    "VLLM_RBLN_USE_VLLM_MODEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_VLLM_MODEL", "False").lower() in ("true", "1")
        )
    ),
    # ====================================================================
    # Common: read regardless of the VLLM_RBLN_USE_VLLM_MODEL value
    # ====================================================================
    # Number of NPU devices per local rank (was VLLM_RBLN_TP_SIZE).
    "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK": get_num_devices_per_local_rank,
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
    "VLLM_RBLN_METRICS": (
        lambda: os.environ.get("VLLM_RBLN_METRICS", "False").lower() in ("true", "1")
    ),
    # Mirror the final performance report to this file (in addition to stdout).
    # The worker pid is appended before the extension to keep TP/DP workers
    # from clobbering each other. Empty disables file output.
    "VLLM_RBLN_METRICS_FILE": lambda: os.environ.get("VLLM_RBLN_METRICS_FILE", ""),
    # Enable NUMA-based CPU affinity binding for OpenMP threads
    "VLLM_RBLN_NUMA": (
        lambda: os.environ.get("VLLM_RBLN_NUMA", "True").lower() in ("true", "1")
    ),
    # ====================================================================
    # Read only when VLLM_RBLN_USE_VLLM_MODEL=False
    # ====================================================================
    # (none currently; only the common variables are read in this case)
    # ====================================================================
    # Read only when VLLM_RBLN_USE_VLLM_MODEL=True
    # ====================================================================
    # --- COMPILE / RUNTIME ---
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
    # Compile-only mode for NPU-less (CPU-only) hosts such as CI build workers.
    # When set, the rbln torch.compile backend compiles + caches each graph and
    # builds its runtime on a dummy device (no NPU required); the populated
    # cache is later reused by a real NPU host via cache-hit. The target SOC is
    # taken from rebel.get_npu_name(), which falls back to RBLN_TARGET_SOC, so
    # set RBLN_TARGET_SOC (e.g. RBLN-CA25) on a host without an NPU mounted.
    "VLLM_RBLN_COMPILE_ONLY": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_ONLY", "False").lower() in ("true", "1")
        )
    ),
    # Use RBLN device tensors end-to-end (platform device_type="rbln",
    # KV cache / inputs on device, CPU-first attention metadata, padded
    # sampling metadata, no CompileContext). Opt-in until stable.
    "VLLM_RBLN_USE_DEVICE_TENSOR": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_DEVICE_TENSOR", "False").lower()
            in ("true", "1")
        )
    ),
    # Disable RBLN file offloading during model load / warm-up even when
    # VLLM_RBLN_USE_DEVICE_TENSOR is set. Kill-switch for the offload path;
    # weight host backings stay resident instead of being paged to disk.
    "VLLM_RBLN_DISABLE_OFFLOAD": (
        lambda: (
            os.environ.get("VLLM_RBLN_DISABLE_OFFLOAD", "False").lower()
            in ("true", "1")
        )
    ),
    # Auto port
    "VLLM_RBLN_AUTO_PORT": use_auto_port,
    # enforce model data type into fp32 not model_config.dtype
    "VLLM_RBLN_ENFORCE_MODEL_FP32": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENFORCE_MODEL_FP32", "False").lower()
            in ("true", "1")
        )
    ),
    # Number of Ray nodes
    "VLLM_RBLN_NUM_RAY_NODES": lambda: int(
        os.environ.get("VLLM_RBLN_NUM_RAY_NODES", 1)
    ),
    "VLLM_RBLN_PROFILER": (
        lambda: os.environ.get("RBLN_PROFILER", "False").lower() in ("true", "1")
    ),
    # --- ATTENTION ---
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
    "VLLM_RBLN_USE_CUSTOM_KERNEL": (
        lambda: (
            os.environ.get("RBLN_USE_CUSTOM_KERNEL", "False").lower() in ("true", "1")
        )
    ),
    # --- MODEL INPUT / SCHEDULING ---
    # Disable multimodal input
    "VLLM_RBLN_DISABLE_MM": (
        lambda: os.environ.get("VLLM_RBLN_DISABLE_MM", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_SORT_BATCH": (
        lambda: os.environ.get("VLLM_RBLN_SORT_BATCH", "False").lower() in ("true", "1")
    ),
    # Enable sub-block prefix caching.
    # Sub-block size equals max_num_batched_tokens (prefill chunk size).
    "VLLM_RBLN_SUB_BLOCK_CACHE": lambda: (
        os.environ.get("VLLM_RBLN_SUB_BLOCK_CACHE", "True").lower() in ("true", "1")
    ),
    # LOGITS_ALL_GATHER, include logits all_gather into model compilation
    "VLLM_RBLN_LOGITS_ALL_GATHER": (
        lambda: (
            os.environ.get("VLLM_RBLN_LOGITS_ALL_GATHER", "True").lower()
            in ("true", "1")
        )
    ),
    # --- DATA PARALLEL ---
    # DP implementation, see choices in get_dp_impl
    "VLLM_RBLN_DP_IMPL": get_dp_impl,
    # --- MOE ---
    # If true, it specializes the cases where all instances are at decode stage
    "VLLM_RBLN_SPECIALIZE_MOE_DECODE": (
        lambda: (
            os.environ.get("VLLM_RBLN_SPECIALIZE_MOE_DECODE", "True").lower()
            in ("true", "1")
        )
    ),
    # If true, it uses the tokens mask applied to moe expert kernel
    "VLLM_RBLN_USE_MOE_TOKENS_MASK": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_MOE_TOKENS_MASK", "True").lower()
            in ("true", "1")
        )
    ),
    # Use reduce_scatter instead of all_reduce in MoE combine phase
    "VLLM_RBLN_MOE_REDUCE_SCATTER": (
        lambda: (
            os.environ.get("VLLM_RBLN_MOE_REDUCE_SCATTER", "False").lower()
            in ("true", "1")
        )
    ),
    # Use all2all dispatch instead of all-gather for MoE DP dispatch
    "VLLM_RBLN_DISPATCH_ALL2ALL": (
        lambda: (
            os.environ.get("VLLM_RBLN_DISPATCH_ALL2ALL", "False").lower()
            in ("true", "1")
        )
    ),
    # Use all2all combine instead of reduce-scatter for MoE DP combine
    "VLLM_RBLN_COMBINE_ALL2ALL": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMBINE_ALL2ALL", "False").lower()
            in ("true", "1")
        )
    ),
    # --- DECODE BATCH BUCKET ---
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
    # Use W8A8 block fp8 (quantize activations to fp8) instead of W8A16
    # (weight-only fp8 dequant). evt0 does not support w8a8; opt-in for evt1.
    "VLLM_RBLN_USE_W8A8_FP8": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_W8A8_FP8", "False").lower() in ("true", "1")
        )
    ),
    # Decode batch bucket limit
    "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT", 1)
    ),
    # Decode batch bucket manual buckets
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": get_decode_batch_bucket_manual_buckets,  # noqa E501
    # --- NIXL ---
    # Publish a second SWA-sized descriptor range alongside the Full-sized
    # range at the same NIXL base addresses, so SWA groups transfer only
    # `sliding_window` bytes per block over RDMA. Host-side h2d/d2h still
    # moves the full block — only the remote RDMA payload is trimmed.
    "VLLM_RBLN_NIXL_SWA_VIEW_OPT": lambda: (
        os.environ.get("VLLM_RBLN_NIXL_SWA_VIEW_OPT", "False").lower() in ("true", "1")
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


vllm_envs.update(environment_variables)
