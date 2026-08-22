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
"""LMCache's mp adapters size paged-KV work in *pages* under page layout.

`LMCacheMPConnector` passes `vllm_block_size=vllm_config.cache_config.block_size`
to both adapters. Without page layout that is the same number as the KV tensor's
block dimension, so the two readings agree and nobody notices they answer
different questions:

    cache_config.block_size   the scheduler's block, and the hash block
    <KV tensor>.shape[-2]     the block the paged KV is actually laid out in

Page layout separates them. The runner restates each group's spec to the kernel
block and **deliberately** leaves `cache_config.block_size` at the page -- the
engine core reads cache_config afterwards, where it becomes the scheduler and
hash block, and a kernel-sized value there contradicts the page-sized spec the
scheduler kept (`RBLNModelRunner._maybe_restate_page_layout`, "Restate the spec
only").

So the adapters size paged-KV work in pages while the tensors -- and, once
vllm-rbln converts them, the block ids -- are kernel blocks:

    blocks_in_chunk = lmcache_tokens_per_chunk // vllm_block_size
                    = 4096 // 512 = 8      (should be 4096 // 4096 = 1)

Measured 2026-08-22 (MiniMax-M2.5 / R100 2P2D):

    ValueError: block_ids length (6) must be at least len(chunks) (6)
                * blocks_per_chunk (8)

Six kernel blocks arrived; the adapter wanted 6 x 8 page-sized ones.

Only that one argument is redirected. `cache_config.block_size` has other
readers inside `LMCacheMPConnector` that genuinely want the page -- notably the
`max_num_batched_tokens >= block_size` check, which the kernel block would fail
(512 < 4096) on exactly the configuration page layout is for. LMCache already
prefers `group.kv_cache_spec.block_size` elsewhere in the same file, so this
brings the adapters in line with that.

TODO: drop this once LMCache takes the paged-KV block from the KV cache spec
(or from the tensor) rather than from `cache_config`.
"""

from typing import Any

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

# Captured before the registry swaps them in. lmcache is optional: images built
# for the in-process (non-mp) stack do not ship it.
try:
    from lmcache.integration.vllm import (  # type: ignore[import-not-found]
        vllm_multi_process_adapter as _mp,
    )

    _ORIGINAL_INITS = {
        "LMCacheMPSchedulerAdapter": _mp.LMCacheMPSchedulerAdapter.__init__,
        "LMCacheMPWorkerAdapter": _mp.LMCacheMPWorkerAdapter.__init__,
    }
except Exception:  # noqa: BLE001 - absence is normal, not an error
    _ORIGINAL_INITS = {}


def _lmcache_mp_available() -> bool:
    return bool(_ORIGINAL_INITS)


def _kernel_block_size(vllm_config: Any) -> int | None:
    """The kernel block, when page layout is on and actually applies.

    Mirrors `RBLNModelRunner._maybe_restate_page_layout`'s guard so the adapter
    and the worker cannot disagree about whether the geometry was converted.
    """
    import vllm_rbln.envs as envs

    if not envs.VLLM_RBLN_PAGE_LAYOUT:
        return None
    additional = getattr(vllm_config, "additional_config", None)
    kernel_block_size = additional.get("attn_block_size") if additional else None
    if not kernel_block_size:
        return None
    kernel_block_size = int(kernel_block_size)
    page_size = vllm_config.cache_config.block_size
    if kernel_block_size == page_size or kernel_block_size % page_size != 0:
        return None
    return kernel_block_size


def _redirect(name: str, self: Any, args: tuple, kwargs: dict) -> Any:
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is None:
        for arg in args:
            if hasattr(arg, "cache_config") and hasattr(arg, "model_config"):
                vllm_config = arg
                break
    if vllm_config is not None and "vllm_block_size" in kwargs:
        kernel_block = _kernel_block_size(vllm_config)
        if kernel_block is not None and kwargs["vllm_block_size"] != kernel_block:
            logger.info(
                "Page layout: %s takes the kernel block (%d) as the paged-KV "
                "block, not cache_config's page (%s).",
                name,
                kernel_block,
                kwargs["vllm_block_size"],
            )
            kwargs["vllm_block_size"] = kernel_block
    return _ORIGINAL_INITS[name](self, *args, **kwargs)


@register_patch(
    target=(
        "lmcache.integration.vllm.vllm_multi_process_adapter."
        "LMCacheMPSchedulerAdapter.__init__"
    ),
    reason=(
        "Page layout: the paged-KV block is the kernel block, but LMCache reads "
        "cache_config.block_size (the page). TODO: drop once LMCache takes it "
        "from the KV cache spec."
    ),
    condition=_lmcache_mp_available,
)
def patched_scheduler_adapter_init(self, *args, **kwargs):
    return _redirect("LMCacheMPSchedulerAdapter", self, args, kwargs)


@register_patch(
    target=(
        "lmcache.integration.vllm.vllm_multi_process_adapter."
        "LMCacheMPWorkerAdapter.__init__"
    ),
    reason=(
        "Page layout: the paged-KV block is the kernel block, but LMCache reads "
        "cache_config.block_size (the page). TODO: drop once LMCache takes it "
        "from the KV cache spec."
    ),
    condition=_lmcache_mp_available,
)
def patched_worker_adapter_init(self, *args, **kwargs):
    return _redirect("LMCacheMPWorkerAdapter", self, args, kwargs)
