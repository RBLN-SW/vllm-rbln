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

from typing import Any

# Keys that affect only runtime loading, not the compiled binary, and
# therefore must be stripped before hashing so the same compiled artifact
# is shared between compile-only and inference invocations.
_RUNTIME_ONLY_KEYS: frozenset[str] = frozenset(
    {"create_runtimes", "devices", "kvcache_num_blocks"}
)


def strip_runtime_only_keys(obj: dict) -> dict:
    """
    Recursively drop :data:`_RUNTIME_ONLY_KEYS` from nested dict/list.
    """
    if isinstance(obj, dict):
        return {
            k: strip_runtime_only_keys(v)
            for k, v in obj.items()
            if k not in _RUNTIME_ONLY_KEYS
        }
    if isinstance(obj, list):
        return [strip_runtime_only_keys(item) for item in obj]
    return obj


def keep_only_device_keys(obj: dict) -> dict:
    """
    Recursively keep only ``devices`` entries from nested dict/list.
    """
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "devices":
            result[k] = v
        elif isinstance(v, dict):
            filtered = keep_only_device_keys(v)
            if filtered:
                result[k] = filtered
    return result
