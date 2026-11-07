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
"""Locate compiled artifacts and derive their compile-cache paths."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model_config import RBLNCompileSpec

RBLN_CONFIG_FILE = "rbln_config.json"


def is_compiled_dir(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, RBLN_CONFIG_FILE))


def resolve_compiled_model_path(
    model_path: str, cached_model_path: str | None
) -> str | None:
    """Prefer an explicit compiled model directory over the generated cache."""
    for path in (model_path, cached_model_path):
        if is_compiled_dir(path):
            return path
    return None


def load_compiled_rbln_config(model_path: str) -> dict | None:
    """Read rbln_config.json from the selected compiled model directory."""
    rbln_config_path = Path(model_path) / RBLN_CONFIG_FILE
    if not rbln_config_path.is_file():
        return None
    with open(rbln_config_path, encoding="utf-8") as f:
        return json.load(f)


# Keys that affect only runtime loading, not the compiled binary, and
# therefore must be stripped before hashing so the same compiled artifact
# is shared between compile-only and inference invocations.
_RUNTIME_ONLY_KEYS: frozenset[str] = frozenset(
    {"create_runtimes", "device", "kvcache_num_blocks"}
)


def _strip_runtime_only_keys(obj: dict) -> dict:
    """Recursively drop :data:`_RUNTIME_ONLY_KEYS` from nested dict/list."""
    if isinstance(obj, dict):
        return {
            k: _strip_runtime_only_keys(v)
            for k, v in obj.items()
            if k not in _RUNTIME_ONLY_KEYS
        }
    if isinstance(obj, list):
        return [_strip_runtime_only_keys(item) for item in obj]
    return obj


def generate_compile_cache_name(spec: RBLNCompileSpec) -> str:
    """Hash compile-relevant spec fields into a stable artifact directory name."""
    config_dict: dict[str, Any] = {
        field.name: getattr(spec, field.name) for field in fields(spec)
    }
    config_dict["rbln_config"] = _strip_runtime_only_keys(config_dict["rbln_config"])
    model_name = str(config_dict["model"])

    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    sanitized_name = model_name.replace("/", "_").replace(":", "_")
    return f"{sanitized_name}_{config_hash}"


def get_compile_cache_path(spec: RBLNCompileSpec, cache_root: str) -> str:
    """Return the cache location for a spec without reading or writing files."""
    return os.path.join(
        cache_root, "compiled_models", generate_compile_cache_name(spec)
    )
