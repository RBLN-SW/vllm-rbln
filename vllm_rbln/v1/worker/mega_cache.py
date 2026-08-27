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

Persists/restores `torch.compiler.{save,load}_cache_artifacts()` bundles under a
per-(model, config-signature, rank) directory in VLLM_CACHE_ROOT.

A bundle is a base file plus append-only increments: `mega_cache.bin`, then
`mega_cache.inc.<n>.bin`. `save_cache_artifacts()` serializes only the artifacts
*this* session compiled -- a loaded hit is never re-recorded -- so writing its
output over the base would drop every graph that hit. Each save becomes a new
part instead, and load replays them all.
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

_INCREMENT_RE = re.compile(r"^mega_cache\.inc\.(\d+)\.bin$")


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


def _compile_env_factors() -> str:
    """Hash of rbln_envs.RBLN_COMPILE_ENV, rank- and host-invariant.

    Keys on the rbln partition, not compile_factors(): that walks ~240 vLLM
    env vars, so host paths and ports alone discarded the bundle.
    """
    from vllm.config.utils import hash_factors, normalize_value

    import vllm_rbln.envs as rbln_envs

    factors: dict[str, object] = {
        name: normalize_value(getattr(rbln_envs, name, None))
        for name in rbln_envs.RBLN_COMPILE_ENV
    }
    return hash_factors(factors)


def _npu_name() -> str:
    """Target NPU name, from the source the per-graph `meta=npu:...` hash uses."""
    try:
        from vllm_rbln.platform import RblnPlatform

        return RblnPlatform.get_device_name() or "unknown"
    except Exception:  # pylint: disable=broad-exception-caught
        return "unknown"


def _warmup_graph_set_factors(vllm_config) -> str:
    """Hash of the graph-shaping config that compute_hash() leaves out.

    Decode buckets, KV cache input, decode query length, which drafter graphs
    exist. Without them two runs share a bundle that only partly hits.
    """
    from vllm.config.utils import hash_factors, normalize_value

    scheduler = getattr(vllm_config, "scheduler_config", None)
    cache = getattr(vllm_config, "cache_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    # Not draft_parallel_config: it carries the same per-launch ports.
    draft = getattr(spec, "draft_model_config", None)
    factors: dict[str, object] = {
        "max_num_seqs": normalize_value(getattr(scheduler, "max_num_seqs", None)),
        "num_gpu_blocks_override": normalize_value(
            getattr(cache, "num_gpu_blocks_override", None)
        ),
        "gpu_memory_utilization": normalize_value(
            getattr(cache, "gpu_memory_utilization", None)
        ),
        "num_speculative_tokens": normalize_value(
            getattr(spec, "num_speculative_tokens", None)
        ),
        "spec_method": normalize_value(getattr(spec, "method", None)),
        "draft_model": draft.compute_hash() if draft is not None else None,
        "draft_tensor_parallel_size": normalize_value(
            getattr(spec, "draft_tensor_parallel_size", None)
        ),
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
    """vLLM config hash + warm-up graph set + rbln compile env + NPU name +
    rebel major.minor; launch- and host-stable, shared by all TP/DP ranks (the
    rank subdir isolates shards)."""
    cfg = _stable_compute_hash(vllm_config)
    graphs = _warmup_graph_set_factors(vllm_config)
    env = _compile_env_factors()
    npu = _npu_name()
    rebel_ver = _rebel_major_minor()
    digest = hashlib.sha1(
        "|".join([cfg, graphs, env, f"npu={npu}", f"rebel={rebel_ver}"]).encode("utf-8")
    )
    sig = digest.hexdigest()[:16]
    logger.info(
        "mega-cache config_signature=%s (cfg=%s graphs=%s env=%s npu=%s rebel=%s)",
        sig,
        cfg[:8],
        graphs[:8],
        env[:8],
        npu,
        rebel_ver,
    )
    return sig


def bundle_path(model: str, sig: str) -> str:
    """Base part of the per-(model, sig, local_rank) bundle under VLLM_CACHE_ROOT
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


def _increment_path(directory: str, index: int) -> str:
    return os.path.join(directory, f"mega_cache.inc.{index}.bin")


def _increment_indices(directory: str) -> list[int]:
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    matches = (_INCREMENT_RE.match(name) for name in names)
    return sorted(int(m.group(1)) for m in matches if m is not None)


def bundle_parts(path: str) -> list[str]:
    """Every part of the bundle at `path`, base first, then increments in order."""
    directory = os.path.dirname(path)
    parts = [path] if os.path.isfile(path) else []
    parts.extend(_increment_path(directory, i) for i in _increment_indices(directory))
    return parts


def _next_part_path(path: str) -> str:
    """Where this session's artifacts go: the base if it is not there yet, else
    a fresh increment."""
    if not os.path.isfile(path):
        return path
    directory = os.path.dirname(path)
    index = max(_increment_indices(directory), default=0) + 1
    while os.path.exists(_increment_path(directory, index)):
        index += 1
    return _increment_path(directory, index)


def cache_root() -> str:
    """Directory the rbln backend should use for populate/lookup."""
    return os.path.join(envs.VLLM_CACHE_ROOT, "rbln")


def _artifact_count(info) -> int:
    """Artifacts a torch CacheInfo carries; 0 if torch stops exposing them."""
    artifacts = getattr(info, "artifacts", None)
    if not isinstance(artifacts, dict):
        return 0
    return sum(len(keys) for keys in artifacts.values())


def load(model: str, sig: str) -> None:
    """Restore artifacts from every part of the bundle so first-compile
    cache-hits."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    from rebel.core import mega_cache as rbln_mega_cache

    rbln_mega_cache.set_dir(cache_root())
    path = bundle_path(model, sig)
    parts = bundle_parts(path)
    if not parts:
        return
    restored = 0
    for part in parts:
        try:
            with open(part, "rb") as src:
                info = torch.compiler.load_cache_artifacts(src.read())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to load rbln mega-cache bundle: %s", exc)
            continue
        if info is None:
            logger.warning(
                "Ignored an unreadable rbln mega-cache part at %s; recompiling "
                "whatever it held",
                part,
            )
            continue
        restored += _artifact_count(info)
    logger.info(
        "Loaded rbln mega-cache bundle from %s: %d part(s), %d artifact(s)",
        os.path.dirname(path),
        len(parts),
        restored,
    )


def save(model: str, sig: str) -> None:
    """Persist this session's artifacts as a new bundle part, atomically. Call
    only after warm-up succeeds."""
    if envs.VLLM_DISABLE_COMPILE_CACHE:
        return
    from rebel.core import mega_cache as rbln_mega_cache

    rbln_mega_cache.set_dir(cache_root())
    path = bundle_path(model, sig)
    tmp_path = None
    try:
        rbln_mega_cache.flush_to_bundle()
        result = torch.compiler.save_cache_artifacts()
        if result is None:
            return
        artifact_bytes, info = result
        os.makedirs(os.path.dirname(path), exist_ok=True)
        target = _next_part_path(path)
        tmp_path = f"{target}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as dst:
            dst.write(artifact_bytes)
            # Delayed allocation defers ENOSPC to flush, so a bundle could be
            # renamed into place truncated without this.
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_path, target)
        logger.info(
            "Saved rbln mega-cache bundle to %s (%.1f MiB, %d artifact(s))",
            target,
            len(artifact_bytes) / (1 << 20),
            _artifact_count(info),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
        out_of_space = isinstance(exc, OSError) and exc.errno in (
            errno.ENOSPC,
            errno.EDQUOT,
        )
        logger.error(
            "Could not persist the rbln mega-cache bundle under %s: %s%s. The "
            "parts already on disk are left untouched, so the graphs this run "
            "compiled are recompiled on every restart until this is resolved.",
            os.path.dirname(path),
            exc,
            " (out of disk space)" if out_of_space else "",
        )
