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
"""Locate compiled model directories and read their rbln_config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .compile_spec import RBLNCompileSpec, compile_cache_name

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


def get_compile_cache_path(spec: RBLNCompileSpec, cache_root: str) -> str:
    """Return the cache location for a spec without reading or writing files."""
    return os.path.join(cache_root, "compiled_models", compile_cache_name(spec))
