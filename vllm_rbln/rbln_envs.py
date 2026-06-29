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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.envs import environment_variables as vllm_envs

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class EnvMeta:
    """Documentation metadata for a VLLM_RBLN_* environment variable.

    SSOT for the env var docs. The runtime parsing lives in
    ``environment_variables``; this only describes the variable for
    linting and doc generation.
    """

    description: str
    default: object = None
    type: str = ""
    deprecated: str = ""  # non-empty marks the var deprecated (for docs/lint)


# SSOT for VLLM_RBLN_* env var documentation. Every VLLM_RBLN_* key in
# ``environment_variables`` must have an entry here (enforced by
# tools/pre_commit/check_env_metadata.py). Docs are generated from this.
ENV_METADATA: dict[str, EnvMeta] = {
    "VLLM_RBLN_COMPILE_MODEL": EnvMeta(
        "If true, compile models using torch.compile. "
        "Otherwise, run the CPU eager mode if possible.",
        default=True, type="bool"),
    "VLLM_RBLN_COMPILE_STRICT_MODE": EnvMeta(
        "If true, compile models using torch.compile strict mode.",
        default=False, type="bool"),
    "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK": EnvMeta(
        "Number of NPU devices assigned to each local rank. The "
        "deprecated VLLM_RBLN_TP_SIZE is honored as a fallback.",
        default=1, type="int"),
    "VLLM_RBLN_SAMPLER": EnvMeta(
        "Use the customized RBLN sampler.",
        default=True, type="bool"),
    "VLLM_RBLN_ENABLE_WARM_UP": EnvMeta(
        "Enable model warm-up before serving.",
        default=True, type="bool"),
    "VLLM_RBLN_USE_VLLM_MODEL": EnvMeta(
        "If true, use the natively compiled vLLM model instead of the "
        "optimum-rbln compiled model.",
        default=False, type="bool"),
    "VLLM_RBLN_FLASH_CAUSAL_ATTN": EnvMeta(
        "Use flash attention for causal attention.",
        default=True, type="bool"),
    "VLLM_RBLN_BATCH_ATTN_OPT": EnvMeta(
        "Use batch attention optimization for paged attention.",
        default=False, type="bool"),
    "VLLM_RBLN_DISABLE_MM": EnvMeta(
        "Disable multimodal input.",
        default=False, type="bool"),
    "VLLM_RBLN_DP_IMPL": EnvMeta(
        "Data-parallel implementation. Choices: padded_decode, "
        "dummy_prefill (dummy_prefill will be deprecated).",
        default="padded_decode", type="str"),
    "VLLM_RBLN_USE_MOE_TOKENS_MASK": EnvMeta(
        "If true, apply the tokens mask to the MoE expert kernel.",
        default=True, type="bool"),
    "VLLM_RBLN_SPECIALIZE_MOE_DECODE": EnvMeta(
        "If true, specialize the case where all instances are at the "
        "decode stage for MoE models.",
        default=True, type="bool"),
    "VLLM_RBLN_ENFORCE_MODEL_FP32": EnvMeta(
        "Enforce the model data type to fp32 instead of "
        "model_config.dtype.",
        default=False, type="bool"),
    "VLLM_RBLN_DP_INPUT_ALL_GATHER": EnvMeta(
        "Use DP input all_gather.",
        default=True, type="bool"),
    "VLLM_RBLN_LOGITS_ALL_GATHER": EnvMeta(
        "Include the logits all_gather in model compilation.",
        default=True, type="bool"),
    "VLLM_RBLN_NUM_RAY_NODES": EnvMeta(
        "Number of Ray nodes.",
        default=1, type="int"),
    "VLLM_RBLN_METRICS": EnvMeta(
        "Enable performance metrics collection.",
        default=False, type="bool"),
    "VLLM_RBLN_METRICS_FILE": EnvMeta(
        "Mirror the final performance report to this file (in addition "
        "to stdout). The worker pid is appended before the extension to "
        "keep TP/DP workers from clobbering each other. Empty disables "
        "file output.",
        default="", type="str"),
    "VLLM_RBLN_NUMA": EnvMeta(
        "Enable NUMA-based CPU affinity binding for OpenMP threads.",
        default=True, type="bool"),
    "VLLM_RBLN_SORT_BATCH": EnvMeta(
        "Sort the batch before execution.",
        default=False, type="bool"),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY": EnvMeta(
        "Decode batch bucket strategy. Choices: exponential, exp, "
        "linear, manual.",
        default="exponential", type="str"),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MIN": EnvMeta(
        "Minimum decode batch bucket size.",
        default=1, type="int"),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STEP": EnvMeta(
        "Decode batch bucket step size.",
        default=2, type="int"),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT": EnvMeta(
        "Maximum decode batch bucket size.",
        default=1, type="int"),
    "VLLM_RBLN_AUTO_PORT": EnvMeta(
        "Automatically pick a free port. Defaults to on when "
        "VLLM_RBLN_USE_DEVICE_TENSOR is enabled.",
        default=None, type="bool"),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": EnvMeta(
        "Comma-separated decode batch sizes used when the bucket "
        "strategy is 'manual'.",
        default=[], type="list[int]"),
    "VLLM_RBLN_USE_CUSTOM_KERNEL": EnvMeta(
        "Use custom kernels. Controlled by the RBLN_USE_CUSTOM_KERNEL "
        "env var (not VLLM_RBLN_USE_CUSTOM_KERNEL).",
        default=False, type="bool"),
    "VLLM_RBLN_MOE_REDUCE_SCATTER": EnvMeta(
        "Use reduce_scatter instead of all_reduce in the MoE combine "
        "phase.",
        default=False, type="bool"),
    "VLLM_RBLN_PROFILER": EnvMeta(
        "Enable the RBLN profiler. Controlled by the RBLN_PROFILER env "
        "var (not VLLM_RBLN_PROFILER).",
        default=False, type="bool"),
    "VLLM_RBLN_DISPATCH_ALL2ALL": EnvMeta(
        "Use all2all dispatch instead of all-gather for MoE DP dispatch.",
        default=False, type="bool"),
    "VLLM_RBLN_COMBINE_ALL2ALL": EnvMeta(
        "Use all2all combine instead of reduce-scatter for MoE DP "
        "combine.",
        default=False, type="bool"),
    "VLLM_RBLN_SUB_BLOCK_CACHE": EnvMeta(
        "Enable sub-block prefix caching. Sub-block size equals "
        "max_num_batched_tokens (prefill chunk size).",
        default=True, type="bool"),
    "VLLM_RBLN_USE_DEVICE_TENSOR": EnvMeta(
        "Use RBLN device tensors end-to-end (platform device_type "
        "'rbln', KV cache / inputs on device, CPU-first attention "
        "metadata, padded sampling metadata, no CompileContext). "
        "Opt-in until stable.",
        default=False, type="bool"),
    "VLLM_RBLN_DISABLE_OFFLOAD": EnvMeta(
        "Disable RBLN file offloading during model load / warm-up even "
        "when VLLM_RBLN_USE_DEVICE_TENSOR is set. Kill-switch for the "
        "offload path; weight host backings stay resident instead of "
        "being paged to disk.",
        default=False, type="bool"),
    "VLLM_RBLN_COMPILE_ONLY": EnvMeta(
        "Compile-only mode for NPU-less (CPU-only) hosts such as CI "
        "build workers. Compiles + caches each graph on a dummy device; "
        "the cache is later reused by a real NPU host via cache-hit. "
        "The target SOC is taken from rebel.get_npu_name(), which falls "
        "back to RBLN_TARGET_SOC (e.g. RBLN-CA25) on a host without an "
        "NPU.",
        default=False, type="bool"),
}


if TYPE_CHECKING:
    VLLM_RBLN_COMPILE_MODEL: bool = True
    VLLM_RBLN_COMPILE_STRICT_MODE: bool = False
    VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK: int = 1
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
    VLLM_RBLN_DP_INPUT_ALL_GATHER: bool = True
    VLLM_RBLN_LOGITS_ALL_GATHER: bool = True
    VLLM_RBLN_NUM_RAY_NODES: int = 1
    VLLM_RBLN_METRICS: bool = False
    VLLM_RBLN_METRICS_FILE: str = ""
    VLLM_RBLN_NUMA: bool = True
    VLLM_RBLN_SORT_BATCH: bool = False
    VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY: str = "exponential"
    VLLM_RBLN_DECODE_BATCH_BUCKET_MIN: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_STEP: int = 2
    VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT: int = 1
    VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS: list[int] = []
    VLLM_RBLN_USE_CUSTOM_KERNEL: bool = False
    VLLM_RBLN_AUTO_PORT: bool = True
    VLLM_RBLN_DISPATCH_ALL2ALL: bool = False
    VLLM_RBLN_COMBINE_ALL2ALL: bool = False
    VLLM_RBLN_MOE_REDUCE_SCATTER: bool = False
    VLLM_RBLN_SUB_BLOCK_CACHE: bool = True
    VLLM_RBLN_USE_DEVICE_TENSOR: bool = False
    VLLM_RBLN_DISABLE_OFFLOAD: bool = False
    VLLM_RBLN_COMPILE_ONLY: bool = False


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
    "VLLM_RBLN_COMPILE_MODEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_MODEL", "True").lower() in ("true", "1")
        )
    ),
    "VLLM_RBLN_COMPILE_STRICT_MODE": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_STRICT_MODE", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK": get_num_devices_per_local_rank,
    "VLLM_RBLN_SAMPLER": (
        lambda: os.environ.get("VLLM_RBLN_SAMPLER", "True").lower() in ("true", "1")
    ),
    "VLLM_RBLN_ENABLE_WARM_UP": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENABLE_WARM_UP", "True").lower() in ("true", "1")
        )
    ),
    "VLLM_RBLN_USE_VLLM_MODEL": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_VLLM_MODEL", "False").lower() in ("true", "1")
        )
    ),
    "VLLM_RBLN_FLASH_CAUSAL_ATTN": (
        lambda: (
            os.environ.get("VLLM_RBLN_FLASH_CAUSAL_ATTN", "True").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_BATCH_ATTN_OPT": (
        lambda: (
            os.environ.get("VLLM_RBLN_BATCH_ATTN_OPT", "False").lower() in ("true", "1")
        )
    ),
    "VLLM_RBLN_DISABLE_MM": (
        lambda: os.environ.get("VLLM_RBLN_DISABLE_MM", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_DP_IMPL": get_dp_impl,
    "VLLM_RBLN_USE_MOE_TOKENS_MASK": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_MOE_TOKENS_MASK", "True").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_SPECIALIZE_MOE_DECODE": (
        lambda: (
            os.environ.get("VLLM_RBLN_SPECIALIZE_MOE_DECODE", "True").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_ENFORCE_MODEL_FP32": (
        lambda: (
            os.environ.get("VLLM_RBLN_ENFORCE_MODEL_FP32", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_DP_INPUT_ALL_GATHER": (
        lambda: (
            os.environ.get("VLLM_RBLN_DP_INPUT_ALL_GATHER", "True").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_LOGITS_ALL_GATHER": (
        lambda: (
            os.environ.get("VLLM_RBLN_LOGITS_ALL_GATHER", "True").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_NUM_RAY_NODES": lambda: int(
        os.environ.get("VLLM_RBLN_NUM_RAY_NODES", 1)
    ),
    "VLLM_RBLN_METRICS": (
        lambda: os.environ.get("VLLM_RBLN_METRICS", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_METRICS_FILE": lambda: os.environ.get("VLLM_RBLN_METRICS_FILE", ""),
    "VLLM_RBLN_NUMA": (
        lambda: os.environ.get("VLLM_RBLN_NUMA", "True").lower() in ("true", "1")
    ),
    "VLLM_RBLN_SORT_BATCH": (
        lambda: os.environ.get("VLLM_RBLN_SORT_BATCH", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY": get_decode_batch_bucket_strategy,
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MIN": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_MIN", 1)
    ),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STEP": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_STEP", 2)
    ),
    "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT": lambda: int(
        os.environ.get("VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT", 1)
    ),
    "VLLM_RBLN_AUTO_PORT": use_auto_port,
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": get_decode_batch_bucket_manual_buckets,  # noqa E501
    "VLLM_RBLN_USE_CUSTOM_KERNEL": (
        lambda: (
            os.environ.get("RBLN_USE_CUSTOM_KERNEL", "False").lower() in ("true", "1")
        )
    ),
    "VLLM_RBLN_MOE_REDUCE_SCATTER": (
        lambda: (
            os.environ.get("VLLM_RBLN_MOE_REDUCE_SCATTER", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_PROFILER": (
        lambda: os.environ.get("RBLN_PROFILER", "False").lower() in ("true", "1")
    ),
    "VLLM_RBLN_DISPATCH_ALL2ALL": (
        lambda: (
            os.environ.get("VLLM_RBLN_DISPATCH_ALL2ALL", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_COMBINE_ALL2ALL": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMBINE_ALL2ALL", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_SUB_BLOCK_CACHE": lambda: (
        os.environ.get("VLLM_RBLN_SUB_BLOCK_CACHE", "True").lower() in ("true", "1")
    ),
    "VLLM_RBLN_USE_DEVICE_TENSOR": (
        lambda: (
            os.environ.get("VLLM_RBLN_USE_DEVICE_TENSOR", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_DISABLE_OFFLOAD": (
        lambda: (
            os.environ.get("VLLM_RBLN_DISABLE_OFFLOAD", "False").lower()
            in ("true", "1")
        )
    ),
    "VLLM_RBLN_COMPILE_ONLY": (
        lambda: (
            os.environ.get("VLLM_RBLN_COMPILE_ONLY", "False").lower() in ("true", "1")
        )
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


vllm_envs.update(environment_variables)
