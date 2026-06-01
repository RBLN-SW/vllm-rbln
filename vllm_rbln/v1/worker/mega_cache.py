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
"""torch.compiler mega-cache bundle helpers for the rbln model runner.

Wraps `torch.compiler.{save,load}_cache_artifacts()` with a per-(model, rank)
file under VLLM_CACHE_ROOT. The rbln dynamo backend pushes `.rbln` blobs into
`CacheArtifactManager` during compile; here we persist/restore the bundle as
an atomic unit only when warm-up has fully succeeded.
"""

from __future__ import annotations

import hashlib
import os
import re

import torch
import vllm.envs as envs
from rebel.core import mega_cache as rbln_mega_cache

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model).strip("_") or "unknown"


def bundle_path(model: str) -> str:
    """Per-(model, local_rank) bundle path under VLLM_CACHE_ROOT.

    local_rank disambiguates processes on the same node (each binds to a
    distinct rbln NPU). Assumes VLLM_CACHE_ROOT is node-local; on a shared
    filesystem, callers should override VLLM_CACHE_ROOT per node.
    """
    raw = model or "unknown"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    local_rank = os.environ.get("LOCAL_RANK", "0")
    return os.path.join(
        envs.VLLM_CACHE_ROOT,
        "rbln",
        f"{_safe_name(raw)}-{suffix}",
        f"rank{local_rank}",
        "mega_cache.bin",
    )


def cache_root() -> str:
    """Directory the rbln backend should use for populate/lookup."""
    return os.path.join(envs.VLLM_CACHE_ROOT, "rbln")


def load(model: str) -> None:
    """Restore artifacts from disk so first-compile cache-hits."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    rbln_mega_cache.set_dir(cache_root())
    path = bundle_path(model)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "rb") as src:
            torch.compiler.load_cache_artifacts(src.read())
        logger.info("Loaded rbln mega-cache bundle from %s", path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to load rbln mega-cache bundle: %s", exc)


def save(model: str) -> None:
    """Persist artifacts atomically. Call only after warm-up succeeds."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    path = bundle_path(model)
    try:
        rbln_mega_cache.flush_to_bundle()
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return
        artifact_bytes, _ = result
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as dst:
            dst.write(artifact_bytes)
        os.replace(tmp_path, path)
        logger.info("Saved rbln mega-cache bundle to %s", path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to save rbln mega-cache bundle: %s", exc)
