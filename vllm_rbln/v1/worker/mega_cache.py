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

Persists/restores `torch.compiler.{save,load}_cache_artifacts()` bundles as a
per-(model, config-signature, rank) file under VLLM_CACHE_ROOT.
"""

import contextlib
import errno
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


# Allowlist, not compile_factors(): that walks ~240 vLLM env vars, so host
# paths and ports alone discarded the bundle.
_RBLN_COMPILE_ENV = frozenset(
    {
        "VLLM_RBLN_USE_VLLM_MODEL",
        "VLLM_RBLN_COMPILE_MODEL",
        "VLLM_RBLN_COMPILE_STRICT_MODE",
        "VLLM_RBLN_USE_DEVICE_TENSOR",
        "VLLM_RBLN_ENFORCE_MODEL_FP32",
        "VLLM_RBLN_USE_W8A16",
        "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",
        "VLLM_RBLN_NUM_RAY_NODES",
        "VLLM_RBLN_FLASH_CAUSAL_ATTN",
        "VLLM_RBLN_BATCH_ATTN_OPT",
        "VLLM_RBLN_USE_CUSTOM_KERNEL",
        "VLLM_RBLN_SPECIALIZE_MOE_DECODE",
        "VLLM_RBLN_USE_MOE_TOKENS_MASK",
        "VLLM_RBLN_DISPATCH_ALL2ALL",
        "VLLM_RBLN_COMBINE_ALL2ALL",
        "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY",
        "VLLM_RBLN_DECODE_BATCH_BUCKET_MIN",
        "VLLM_RBLN_DECODE_BATCH_BUCKET_STEP",
        "VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT",
        "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS",
    }
)


def _compile_env_factors() -> str:
    """Hash of the rbln compile env allowlist, rank- and host-invariant."""
    from vllm.config.utils import hash_factors, normalize_value

    import vllm_rbln.envs as rbln_envs

    factors: dict[str, object] = {
        name: normalize_value(getattr(rbln_envs, name, None))
        for name in _RBLN_COMPILE_ENV
    }
    return hash_factors(factors)


def _stable_compute_hash(vllm_config) -> str:
    """compute_hash() leaks per-launch auto-queried ports via ParallelConfig
    port fields; zero them around the call so the hash is launch-stable."""
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
    """vLLM config hash + rbln compile env + rebel major.minor; launch- and
    host-stable, shared by all TP/DP ranks (the rank subdir isolates shards)."""
    cfg = _stable_compute_hash(vllm_config)
    env = _compile_env_factors()
    rebel_ver = _rebel_major_minor()
    digest = hashlib.sha1("|".join([cfg, env, f"rebel={rebel_ver}"]).encode("utf-8"))
    sig = digest.hexdigest()[:16]
    logger.info(
        "mega-cache config_signature=%s (cfg=%s env=%s rebel=%s)",
        sig,
        cfg[:8],
        env[:8],
        rebel_ver,
    )
    return sig


def bundle_path(model: str, sig: str) -> str:
    """Per-(model, sig, local_rank) bundle path under VLLM_CACHE_ROOT
    (assumed node-local; override per node on a shared filesystem)."""
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
            info = torch.compiler.load_cache_artifacts(src.read())
        if info is None:
            logger.warning(
                "Ignored an unreadable rbln mega-cache bundle at %s; recompiling",
                path,
            )
        else:
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
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        rbln_mega_cache.flush_to_bundle()
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return
        artifact_bytes, _ = result
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "wb") as dst:
            dst.write(artifact_bytes)
            # Delayed allocation defers ENOSPC to flush, so a bundle could be
            # renamed into place truncated without this.
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_path, path)
        logger.info(
            "Saved rbln mega-cache bundle to %s (%.1f MiB)",
            path,
            len(artifact_bytes) / (1 << 20),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        out_of_space = isinstance(exc, OSError) and exc.errno in (
            errno.ENOSPC,
            errno.EDQUOT,
        )
        logger.error(
            "Could not persist the rbln mega-cache bundle to %s: %s%s. The "
            "previous bundle is left untouched, so every restart recompiles "
            "from scratch until this is resolved.",
            path,
            exc,
            " (out of disk space)" if out_of_space else "",
        )
