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
"""Hand the dynamic-KV block count to the scheduler after warm-up."""

# NOTE(RBLN): `EngineCore._initialize_kv_caches` is the only hook that works for
# both single-process and TP>1 -- it warms the workers up before returning, and
# the caller builds the `Scheduler` only afterwards. The second fact is re-checked
# below, so a vLLM upgrade that reorders them fails loudly instead of mis-sizing.

from typing import TYPE_CHECKING, Any

from vllm.utils.math_utils import cdiv
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig

import vllm_rbln.envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.patches.registry import register_patch
from vllm_rbln.v1.worker.utils import rescale_kv_cache_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Held at import time: the registry replaces the attribute outright.
engine_core_original_initialize_kv_caches = EngineCore._initialize_kv_caches


def resolve_rank_num_blocks(num_blocks_per_rank: list[Any]) -> int | None:
    """Reduce the per-rank answers to the single block count to apply.

    All ranks run the same gates, so a mixed result means one rank's query
    failed. The reduction is a min, as upstream's own is.
    """
    if all(n is None for n in num_blocks_per_rank):
        return None
    if any(n is None for n in num_blocks_per_rank):
        raise RuntimeError(
            "dynamic KV cache: some ranks computed a block count and some did not "
            f"({num_blocks_per_rank}); one rank's profile query failed."
        )
    invalid = [n for n in num_blocks_per_rank if not isinstance(n, int) or n <= 0]
    if invalid:
        raise RuntimeError(
            "dynamic KV cache: ranks returned invalid block counts "
            f"({num_blocks_per_rank})."
        )
    return min(num_blocks_per_rank)


@register_patch(
    target="vllm.v1.engine.core.EngineCore._initialize_kv_caches",
    reason=(
        "VLLM_RBLN_USE_DYNAMIC_KV_CACHE resizes the KV cache after warm-up, "
        "once the compiled artifact can report how many blocks fit the device. "
        "Re-announce that number to the scheduler's block pool, which was "
        "otherwise sized from the pre-compile estimate."
    ),
    condition=lambda: envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE,
)
def patched_initialize_kv_caches(
    self: EngineCore, vllm_config: "VllmConfig"
) -> KVCacheConfig:
    kv_cache_config = engine_core_original_initialize_kv_caches(self, vllm_config)

    if getattr(self, "scheduler", None) is not None:
        raise RuntimeError(
            "the scheduler already exists when _initialize_kv_caches returns, so "
            "its block pool predates the resize; revisit the dynamic-KV patch."
        )

    # The worker gates on the override too, so nothing has been resized here.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        # WARNING: two features that each size the KV cache are on, one ignored.
        logger.warning(
            "dynamic KV cache: --num-gpu-blocks-override=%d wins over "
            "VLLM_RBLN_USE_DYNAMIC_KV_CACHE; the block pool stays at %d.",
            override,
            kv_cache_config.num_blocks,
        )
        return kv_cache_config

    num_blocks_per_rank = self.model_executor.collective_rpc(
        "compute_dynamic_kv_num_blocks"
    )
    num_blocks = resolve_rank_num_blocks(num_blocks_per_rank)
    logger.info(
        "dynamic KV cache: compute_dynamic_kv_num_blocks=%s per rank -> apply(%s)",
        num_blocks_per_rank,
        num_blocks,
    )

    if num_blocks is None:
        # Nothing usable: tell the workers to put the pre-shrink count back.
        self.model_executor.collective_rpc("apply_dynamic_kv_num_blocks", args=(None,))
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
    assert_kv_cache_fits_one_request(vllm_config, kv_cache_config)
    _log_gpu_kv_cache_size(vllm_config, kv_cache_config)
    return kv_cache_config


def assert_kv_cache_fits_one_request(
    vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig
) -> None:
    """Fail loudly when the resized pool cannot hold a single max-length request."""
    # NOTE(RBLN): upstream's `check_enough_kv_cache_memory` runs against the
    # pre-compile estimate and nothing re-checks the number substituted here, so
    # without this the server starts and then rejects every request.
    block_size = vllm_config.cache_config.block_size
    max_model_len = vllm_config.model_config.max_model_len
    if not block_size or not max_model_len:
        return
    needed = cdiv(max_model_len, block_size)
    if kv_cache_config.num_blocks >= needed:
        return
    raise ValueError(
        f"The KV cache sized from the compiled profile holds "
        f"{kv_cache_config.num_blocks} blocks, but a single request of "
        f"max_model_len={max_model_len} needs {needed} at block_size="
        f"{block_size}. Reduce max_model_len, raise gpu_memory_utilization, or "
        f"give the model more devices."
    )


def _log_gpu_kv_cache_size(
    vllm_config: "VllmConfig", kv_cache_config: KVCacheConfig
) -> None:
    """Re-announce the KV cache size after the resize.

    Upstream emits its line before warm-up and `info_once` will not repeat it,
    so these keep upstream's wording for whatever parses the first one.
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
            "GPU KV cache size: %s tokens (num_blocks=%d, after the dynamic KV resize)",
            f"{int(max_concurrency * max_model_len):,}",
            kv_cache_config.num_blocks,
        )
        logger.info(
            "Maximum concurrency for %s tokens per request: %.2fx (after the "
            "dynamic KV resize)",
            f"{max_model_len:,}",
            max_concurrency,
        )
    except Exception as exc:
        # Reporting must never break start-up.
        logger.warning(
            "could not recompute the GPU KV cache size at num_blocks=%d: %s",
            kv_cache_config.num_blocks,
            exc,
        )
