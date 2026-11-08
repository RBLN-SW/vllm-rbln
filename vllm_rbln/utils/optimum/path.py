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
"""Locate compiled model directories and construct cache paths."""

from __future__ import annotations

import os

RBLN_CONFIG_FILE = "rbln_config.json"


def is_compiled_dir(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, RBLN_CONFIG_FILE))


def resolve_compiled_model_path(
    model_path: str, artifact_path: str | None
) -> str | None:
    """Prefer an explicit compiled model directory over the generated cache."""
    for path in (model_path, artifact_path):
        if is_compiled_dir(path):
            return path
    return None


def get_compile_cache_path(cache_name: str, cache_root: str) -> str:
    """Return the cache location without reading or writing files."""
    return os.path.join(cache_root, "compiled_models", cache_name)
