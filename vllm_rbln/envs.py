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
    VLLM_RBLN_METRICS_DIR: str = ""
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
    VLLM_RBLN_USE_DEVICE_TENSOR: bool = True
    VLLM_RBLN_DISABLE_OFFLOAD: bool = False
    # Default follows VLLM_RBLN_USE_DEVICE_TENSOR (see use_auto_port), so it is
    # True unless device-tensor mode is explicitly disabled.
    VLLM_RBLN_AUTO_PORT: bool = True
    VLLM_RBLN_ENFORCE_MODEL_FP32: bool = False
    VLLM_RBLN_NUM_RAY_NODES: int = 1
    # --- DYNAMIC KV CACHE ---
    VLLM_RBLN_USE_DYNAMIC_KV_CACHE: bool = False
    # Debug-only override of the derived compile-time hint (see
    # compile_kv_cache_num_blocks); 0 opts out of the shrink and the resize.
    # Kept a literal because this block is only read by type checkers; the value
    # in force is DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS below.
    VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS: int = 8
    VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES: int = 48 * 1024 * 1024
    # --- ATTENTION ---
    VLLM_RBLN_FLASH_CAUSAL_ATTN: bool = True
    VLLM_RBLN_BATCH_ATTN_OPT: bool = False
    VLLM_RBLN_USE_CUSTOM_KERNEL: bool = False
    # --- MODEL INPUT / SCHEDULING ---
    VLLM_RBLN_SORT_BATCH: bool = False
    VLLM_RBLN_SUB_BLOCK_CACHE: bool = True
    # --- MOE ---
    VLLM_RBLN_SPECIALIZE_MOE_DECODE: bool = True
    VLLM_RBLN_USE_MOE_TOKENS_MASK: bool = True
    VLLM_RBLN_DISPATCH_ALL2ALL: bool = False
    VLLM_RBLN_COMBINE_ALL2ALL: bool = False
    # --- DECODE BATCH BUCKET ---
    VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY: str = "exponential"
    VLLM_RBLN_DECODE_BATCH_BUCKET_MIN: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_STEP: int = 2
    VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS: list[int] = []
    # --- KV CONNECTOR ---
    VLLM_RBLN_NIXL_SWA_VIEW_OPT: bool = False
    # --- QUANTIZATION ---
    VLLM_RBLN_USE_W8A16: bool = False

_W8A8_CAPABLE_NPUS = frozenset({"RBLN-CR13", "RBLN-CR23"})

_USE_W8A16: bool | None = None


def get_use_w8a16() -> bool:
    value = os.environ.get("VLLM_RBLN_USE_W8A16")
    if value is not None:
        return value.lower() in ("true", "1")

    global _USE_W8A16
    if _USE_W8A16 is not None:
        return _USE_W8A16

    from vllm.platforms import current_platform

    try:
        device_name = current_platform.get_device_name() or ""
    except Exception:
        device_name = ""

    _USE_W8A16 = (
        not device_name or device_name.strip().upper() not in _W8A8_CAPABLE_NPUS
    )
    return _USE_W8A16


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


def is_set(name: str) -> bool:
    """Whether `name` is present in the environment, whatever its value.

    Lets a caller tell "the operator chose this" from "this is the default",
    which a getter's return value alone cannot express once the default is
    non-zero.
    """
    return name in os.environ


# The compile-time KV block count used when the operator sets nothing. The
# traced `num_blocks` only supplies the hint of the mark_dynamic'd dim, and
# warm-up points every dummy request at block 0, so a small cache is enough;
# 8 is the value the measured dynamic-KV runs used.
DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS = 8


def compile_kv_cache_num_blocks() -> int:
    """Blocks the KV cache is compiled against, under the dynamic-KV path.

    Unset means `DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS`, so the feature needs one
    public switch (VLLM_RBLN_USE_DYNAMIC_KV_CACHE) rather than two variables that
    must agree. An explicit value is a debug override; an explicit 0 opts out of
    the shrink, and therefore out of the resize.

    Deliberately *not* derived from VLLM_RBLN_USE_DYNAMIC_KV_CACHE.
    `use_auto_port` below shows the cost of that shape: it re-spells the sibling
    variable's default ("True") in a second place, so the two have to be kept in
    step by hand. Here that line would be the one missed the day the dynamic flag
    becomes default-on -- and missing it restores exactly the half-enabled state
    this default exists to remove. The flag is checked where the shrink is
    decided instead.

    Never raises on an unset or blank value: vLLM's `enable_envs_cache` evaluates
    every registered getter once, RBLN's included (see the `vllm_envs.update`
    call at the end of this module), so a getter that raises would break
    deployments that never touch this feature. A non-numeric value still raises,
    as it did before.
    """
    raw = os.environ.get("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS")
    if raw is None or not raw.strip():
        return DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS
    return int(raw)


def use_auto_port() -> bool:
    raw = os.environ.get("VLLM_RBLN_AUTO_PORT")
    if raw is not None:
        return raw.lower() in ("true", "1")
    # Default follows device-tensor mode: auto port is on when
    # VLLM_RBLN_USE_DEVICE_TENSOR is enabled.
    return os.environ.get("VLLM_RBLN_USE_DEVICE_TENSOR", "True").lower() in (
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
    # Directory for per-worker JSON performance reports (empty disables).
    "VLLM_RBLN_METRICS_DIR": lambda: os.environ.get("VLLM_RBLN_METRICS_DIR", ""),
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
    # taken from rebel.get_npu_name(); on a host without an NPU mounted set
    # RBLN_FORCE_NPU_NAME (e.g., RBLN-CA25) so the target can still be resolved.
    "VLLM_RBLN_COMPILE_ONLY": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_ONLY", "False").lower() in ("true", "1")
        )
    ),
    # Use RBLN device tensors end-to-end (platform device_type="rbln",
    # KV cache / inputs on device, CPU-first attention metadata, padded
    # sampling metadata, no CompileContext). Enabled by default; set to False
    # to fall back to the host-tensor path.
    "VLLM_RBLN_USE_DEVICE_TENSOR": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_DEVICE_TENSOR", "True").lower()
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
    # --- DYNAMIC KV CACHE ---
    # Size the KV cache from each compiled artifact's kv_cache_memory_profile()
    # instead of from the pre-compile estimate. Requires the torch.compile path
    # (VLLM_RBLN_USE_VLLM_MODEL=1): only DynamoRuntime applies adaptive buffer
    # sizes, the others ignore them silently.
    "VLLM_RBLN_USE_DYNAMIC_KV_CACHE": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "False").lower()
            in ("true", "1")
        )
    ),
    # Debug-only override of the compile-time block count derived above; unset is
    # DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS. Smaller is cheaper: on TP=1 the cache
    # is returned and costs nothing, on TP>=2 it is not, so the final block count
    # loses exactly this many blocks (measured: MiniMax TP4+EP, 62 MiB per block
    # per chiplet -- 64 blocks there is 4 runs out of 4 dead on the first
    # request). Values at or above what vllm sized the cache with cancel the
    # shrink, which also cancels the resize; 0 opts out of both on purpose.
    # Below 8 is unmeasured, and 1 risks dynamo specializing a size-1 dim into a
    # static artifact.
    "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS": compile_kv_cache_num_blocks,
    # Bytes held back from every chiplet's dynamic-KV budget for device memory
    # `kv_cache_memory_profile()` does not describe: the profile is not a
    # complete inventory, so eager buffers and command-stream pools that match no
    # region would otherwise be handed to KV blocks. Sized from the sum of those
    # items on a measured run, rounded up. 0 disables it.
    "VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES": lambda: int(
        os.environ.get(
            "VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES", 48 * 1024 * 1024
        )
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
    "VLLM_RBLN_SORT_BATCH": (
        lambda: os.environ.get("VLLM_RBLN_SORT_BATCH", "False").lower() in ("true", "1")
    ),
    # Enable sub-block prefix caching.
    # Sub-block size equals max_num_batched_tokens (prefill chunk size).
    "VLLM_RBLN_SUB_BLOCK_CACHE": lambda: (
        os.environ.get("VLLM_RBLN_SUB_BLOCK_CACHE", "True").lower() in ("true", "1")
    ),
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
    # Decode batch bucket limit
    "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT", 1)
    ),
    # Decode batch bucket manual buckets
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": get_decode_batch_bucket_manual_buckets,  # noqa E501
    # --- KV CONNECTOR ---
    # Publish a second SWA-sized descriptor range alongside the Full-sized
    # range at the same NIXL base addresses, so SWA groups transfer only
    # `sliding_window` bytes per block over RDMA. Host-side h2d/d2h still
    # moves the full block — only the remote RDMA payload is trimmed.
    "VLLM_RBLN_NIXL_SWA_VIEW_OPT": (
        lambda: (
            os.environ.get("VLLM_RBLN_NIXL_SWA_VIEW_OPT", "False").lower()
            in ("true", "1")
        )
    ),
    # --- QUANTIZATION ---
    "VLLM_RBLN_USE_W8A16": get_use_w8a16,
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


vllm_envs.update(environment_variables)
