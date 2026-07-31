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
"""Hand the dynamic-KV block count to the scheduler.

Under `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` a worker only learns after warm-up how
many KV blocks its device holds. vLLM's admission side never finds out on its
own -- it was sized from the pre-compile estimate -- so the scheduler hands out
block ids the cache may not have.

`EngineCore._initialize_kv_caches` is the one place covering both the
single-process and TP>1 cases: it warms the workers up before returning, and the
caller only builds the `Scheduler` afterwards. Both facts are asserted below so a
vLLM upgrade that reorders them fails loudly instead of mis-sizing the pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vllm
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig

import vllm_rbln.envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.patches.registry import register_patch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Captured at import time: the registry replaces the attribute outright, so the
# original has to be held onto here to be able to call it.
engine_core_original_initialize_kv_caches = EngineCore._initialize_kv_caches

# vLLM release this patch was written against. A mismatch is not fatal -- the
# structural asserts below are the real check -- but it is worth logging.
_VERIFIED_VLLM_VERSION = "0.22.0"


def _dynamic_kv_enabled() -> bool:
    return bool(envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE)


def resolve_rank_num_blocks(num_blocks_per_rank: list[Any]) -> int | None:
    """Reduce the per-rank answers to the single block count to apply.

    All ranks run the same gates, so the result should be uniformly None or
    uniformly an int. A mixed result means a rank could not validate a number
    against its own device, and applying another rank's answer there risks an
    OOM -- so it is refused.

    Returns:
        None when every rank declined, otherwise the minimum across ranks
        (upstream's own reduction is a min; a rank with more blocks than the
        agreed number simply leaves the extras unused).
    """
    if not num_blocks_per_rank:
        return None
    if all(n is None for n in num_blocks_per_rank):
        return None
    if any(n is None for n in num_blocks_per_rank):
        raise RuntimeError(
            "dynamic KV cache: some ranks computed a block count and some did "
            f"not ({num_blocks_per_rank}). Applying another rank's number to a "
            "rank whose own profile query failed would be unvalidated against "
            "that device."
        )
    invalid = [n for n in num_blocks_per_rank if not isinstance(n, int) or n <= 0]
    if invalid:
        raise RuntimeError(
            "dynamic KV cache: ranks returned invalid block counts "
            f"({num_blocks_per_rank})."
        )
    return min(num_blocks_per_rank)


def rescale_kv_cache_config(kv_cache_config: KVCacheConfig, num_blocks: int) -> None:
    """Retarget `kv_cache_config` at `num_blocks`, in place.

    `KVCacheTensor.size` is `num_blocks * page_size_bytes` and consumers read it
    rather than recomputing, so it has to move with `num_blocks`.
    """
    old_num_blocks = kv_cache_config.num_blocks
    assert old_num_blocks > 0, "cannot rescale a KV cache config of 0 blocks"
    kv_cache_config.num_blocks = num_blocks
    for kv_tensor in kv_cache_config.kv_cache_tensors:
        kv_tensor.size = (kv_tensor.size * num_blocks) // old_num_blocks


@register_patch(
    target="vllm.v1.engine.core.EngineCore._initialize_kv_caches",
    reason=(
        "VLLM_RBLN_USE_DYNAMIC_KV_CACHE resizes the KV cache after warm-up, "
        "once the compiled artifact can report how many blocks fit the device. "
        "Re-announce that number to the scheduler's block pool, which was "
        "otherwise sized from the pre-compile estimate."
    ),
    condition=_dynamic_kv_enabled,
)
def patched_initialize_kv_caches(
    self: EngineCore, vllm_config: "VllmConfig"
) -> KVCacheConfig:
    kv_cache_config = engine_core_original_initialize_kv_caches(self, vllm_config)

    # --- version-drift guards ---------------------------------------------
    if vllm.__version__ != _VERIFIED_VLLM_VERSION:
        logger.warning(
            "the RBLN dynamic-KV scheduler hand-off was verified against vLLM "
            "%s but this is vLLM %s; check that "
            "EngineCore._initialize_kv_caches still warms up the workers before "
            "the Scheduler is constructed.",
            _VERIFIED_VLLM_VERSION,
            vllm.__version__,
        )
    if not isinstance(kv_cache_config, KVCacheConfig):
        raise RuntimeError(
            "EngineCore._initialize_kv_caches no longer returns a KVCacheConfig "
            f"(got {type(kv_cache_config).__name__}); the RBLN dynamic-KV "
            "scheduler hand-off must be revisited."
        )
    if getattr(self, "scheduler", None) is not None:
        raise RuntimeError(
            "the scheduler already exists when _initialize_kv_caches returns, "
            "so its block pool was built from the pre-resize block count. The "
            "RBLN dynamic-KV scheduler hand-off must be revisited."
        )

    # An explicit override is the user pinning the block count; the workers make
    # the same check, so nothing was resized either.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        logger.info(
            "dynamic KV cache: --num-gpu-blocks-override=%d is set; leaving the "
            "scheduler block pool at %d blocks.",
            override,
            kv_cache_config.num_blocks,
        )
        return kv_cache_config

    num_blocks_per_rank = self.model_executor.collective_rpc(
        "compute_dynamic_kv_num_blocks"
    )
    num_blocks = resolve_rank_num_blocks(num_blocks_per_rank)
    logger.info(
        "dynamic KV cache: per-rank block counts %s -> %s",
        num_blocks_per_rank,
        num_blocks,
    )

    if num_blocks is None:
        # Nothing usable. Tell the workers so they put back the block count
        # vLLM sized the cache with before the compile-time shrink.
        self.model_executor.collective_rpc(
            "apply_dynamic_kv_num_blocks", args=(None,)
        )
        return kv_cache_config

    self.model_executor.collective_rpc(
        "apply_dynamic_kv_num_blocks", args=(num_blocks,)
    )

    old_num_blocks = kv_cache_config.num_blocks
    rescale_kv_cache_config(kv_cache_config, num_blocks)
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    # The frontend picks this up on its own: EngineCoreReadyResponse is built
    # from cache_config.num_gpu_blocks after __init__ has finished.

    logger.info(
        "dynamic KV cache: scheduler block pool resized %d -> %d blocks",
        old_num_blocks,
        num_blocks,
    )
    _log_gpu_kv_cache_size(vllm_config, kv_cache_config)
    return kv_cache_config


def _log_gpu_kv_cache_size(
    vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig
) -> None:
    """Re-announce the KV cache size after the resize.

    The original line is emitted inside `get_kv_cache_configs`, i.e. before
    warm-up, so it reports the pre-resize number and `logger.info_once` will not
    print it a second time.
    """
    try:
        from vllm.v1.core.kv_cache_utils import (
            get_max_concurrency_for_kv_cache_config,
        )

        max_model_len = vllm_config.model_config.max_model_len
        max_concurrency = get_max_concurrency_for_kv_cache_config(
            vllm_config, kv_cache_config
        )
        logger.info(
            "GPU KV cache size (after dynamic KV resize): %s tokens "
            "(num_blocks=%d, maximum concurrency for %s tokens per request: "
            "%.2fx)",
            f"{int(max_concurrency * max_model_len):,}",
            kv_cache_config.num_blocks,
            f"{max_model_len:,}",
            max_concurrency,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Reporting must never break start-up.
        logger.warning(
            "could not recompute the GPU KV cache size after the dynamic KV "
            "resize (num_blocks=%d): %s",
            kv_cache_config.num_blocks,
            exc,
        )
