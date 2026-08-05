# Copyright 2026 Rebellions Inc. All rights reserved.

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

Wraps `torch.compiler.{save,load}_cache_artifacts()` with a
per-(model, config-signature, rank) file under VLLM_CACHE_ROOT. The rbln dynamo
backend pushes `.rbln` blobs into
`CacheArtifactManager` during compile; here we persist/restore the bundle as
an atomic unit only when warm-up has fully succeeded.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re

import torch
import vllm.envs as envs

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model).strip("_") or "unknown"


def _rebel_major_minor(version: str | None = None) -> str:
    """major.minor of the rebel version; patch bumps stay cache-compatible."""
    if version is None:
        try:
            import rebel

            version = getattr(rebel, "__version__", "") or ""
        except Exception:  # pylint: disable=broad-exception-caught
            version = ""
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    return f"{match.group(1)}.{match.group(2)}" if match else "unknown"


# Per-worker identity that must NOT enter the bundle signature: the rank subdir
# separates workers and DP replicas compile identically, so the DP-degree lives
# in parallel_config.compute_hash, not the per-worker index. vLLM's
# compile_factors already drops LOCAL_RANK / ports but keeps VLLM_DP_RANK*.
_PER_WORKER_ENV_DROP = frozenset({"VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL"})

# rbln vars that never change the compiled graph (runtime, scheduling, KV
# transfer). COMPILE_ONLY must stay here so a CPU-compiled bundle hits on the
# NPU serving host.
_RUNTIME_ENV_DROP = frozenset(
    {
        "VLLM_RBLN_ENABLE_WARM_UP",
        "VLLM_RBLN_METRICS",
        "VLLM_RBLN_METRICS_FILE",
        "VLLM_RBLN_NUMA",
        "VLLM_RBLN_PROFILER",
        "VLLM_RBLN_AUTO_PORT",
        "VLLM_RBLN_SUB_BLOCK_CACHE",
        "VLLM_RBLN_SORT_BATCH",
        "VLLM_RBLN_DISABLE_OFFLOAD",
        "VLLM_RBLN_NIXL_SWA_VIEW_OPT",
        "VLLM_RBLN_COMPILE_ONLY",
        # Sampler graphs are compiled with use_cache=False (#798), so they never
        # enter the bundle and this flag cannot change its contents. Model graphs
        # compile before the sampler in warmup, so their keys are unaffected too.
        "VLLM_RBLN_SAMPLER",
    }
)


def _compile_env_factors() -> str:
    """vLLM's worker-aligned compile env hash (rbln vars are merged into that
    registry by importing vllm_rbln.envs; per-worker and runtime-only vars are
    dropped so all ranks share one signature)."""
    import vllm.envs as vllm_envs
    from vllm.config.utils import hash_factors

    # Import for the side effect: vllm_rbln.envs merges VLLM_RBLN_* into vLLM's
    # environment_variables registry, which is what compile_factors() walks.
    import vllm_rbln.envs  # noqa: F401

    factors = vllm_envs.compile_factors()
    for name in _PER_WORKER_ENV_DROP | _RUNTIME_ENV_DROP:
        factors.pop(name, None)
    return hash_factors(factors)


def _stable_compute_hash(vllm_config) -> str:
    """vllm_config.compute_hash() leaks per-launch auto-queried open ports via
    ParallelConfig port fields that are not in its compute_hash ignored set (e.g.
    _coord_store_port), so the hash changes every launch. Neutralize every
    "port"-named parallel field around the call — ports never affect the compiled
    graph — so the signature is launch-stable."""
    import dataclasses

    pc = getattr(vllm_config, "parallel_config", None)
    saved: dict = {}
    if pc is not None and dataclasses.is_dataclass(pc):
        for field in dataclasses.fields(pc):
            if "port" in field.name.lower():
                try:
                    saved[field.name] = getattr(pc, field.name)
                    object.__setattr__(pc, field.name, 0)
                except Exception:  # pylint: disable=broad-exception-caught
                    saved.pop(field.name, None)
    try:
        return vllm_config.compute_hash()
    finally:
        for name, value in saved.items():
            with contextlib.suppress(Exception):
                object.__setattr__(pc, name, value)


def config_signature(vllm_config) -> str:
    """Hash of everything that changes the compiled graphs, keying the bundle.

    vLLM config hash + vLLM's worker-aligned compile env factors + rebel
    major.minor. Env factors drop per-worker vars (LOCAL_RANK, ports, ...) and
    per-launch open ports are blanked, so all TP/DP ranks across launches share
    one signature and the rank subdir isolates their shards.
    """
    cfg = _stable_compute_hash(vllm_config)
    env = _compile_env_factors()
    rebel_ver = _rebel_major_minor()
    digest = hashlib.sha1("|".join([cfg, env, f"rebel={rebel_ver}"]).encode("utf-8"))
    sig = digest.hexdigest()[:16]
    logger.info(
        "mega-cache config_signature=%s (cfg=%s env=%s rebel=%s)",
        sig,
        cfg[:8],
        hashlib.sha1(env.encode("utf-8")).hexdigest()[:8],
        rebel_ver,
    )
    return sig


def bundle_path(model: str, sig: str) -> str:
    """Per-(model, config-signature, local_rank) bundle path under VLLM_CACHE_ROOT.

    `sig` isolates compile configs; local_rank isolates same-node NPUs. Assumes
    VLLM_CACHE_ROOT is node-local; override per node on a shared filesystem.
    """
    raw = model or "unknown"
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    local_rank = os.environ.get("LOCAL_RANK", "0")
    return os.path.join(
        envs.VLLM_CACHE_ROOT,
        "rbln",
        f"{_safe_name(raw)}-{suffix}",
        sig,
        f"rank{local_rank}",
        "mega_cache.bin",
    )


def cache_root() -> str:
    """Directory the rbln backend should use for populate/lookup."""
    return os.path.join(envs.VLLM_CACHE_ROOT, "rbln")


def load(model: str, sig: str) -> None:
    """Restore artifacts from disk so first-compile cache-hits."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    from rebel.core import mega_cache as rbln_mega_cache

    rbln_mega_cache.set_dir(cache_root())
    path = bundle_path(model, sig)
    if not os.path.isfile(path):
        return
    try:
        with open(path, "rb") as src:
            torch.compiler.load_cache_artifacts(src.read())
        logger.info("Loaded rbln mega-cache bundle from %s", path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to load rbln mega-cache bundle: %s", exc)


def save(model: str, sig: str) -> None:
    """Persist artifacts atomically. Call only after warm-up succeeds."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    from rebel.core import mega_cache as rbln_mega_cache

    rbln_mega_cache.set_dir(cache_root())
    path = bundle_path(model, sig)
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
