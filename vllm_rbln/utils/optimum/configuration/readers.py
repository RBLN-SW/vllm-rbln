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

"""Read compiled RBLN metadata and access its configuration fields.

Field readers accept a ``rbln_config.json`` dict or an ``RBLNModelConfig``.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from vllm_rbln.utils.optimum.path import RBLN_CONFIG_FILE

if TYPE_CHECKING:
    from optimum.rbln.configuration_utils import RBLNModelConfig


# Either a plain dict (from rbln_config.json) or an RBLNModelConfig instance.
RblnConfigLike = Union[dict, "RBLNModelConfig"]


def load_compiled_rbln_config(model_path: str) -> dict | None:
    """Read rbln_config.json from the selected compiled model directory."""
    rbln_config_path = Path(model_path) / RBLN_CONFIG_FILE
    if not rbln_config_path.is_file():
        return None
    with open(rbln_config_path, encoding="utf-8") as f:
        return json.load(f)


def get_field(cfg: RblnConfigLike, key: str, default: Any = None) -> Any:
    """Access a config value from either a dict or an RBLNModelConfig instance."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_submodule(cfg: RblnConfigLike, submodule: str) -> RblnConfigLike | None:
    """Get a submodule config; returns ``None`` if the submodule is absent."""
    if isinstance(cfg, dict):
        return cfg.get(submodule)
    return getattr(cfg, submodule, None)


def language_model_configs(cfg: RblnConfigLike) -> list[RblnConfigLike]:
    """Return language_model, text_model and root configs in precedence order."""
    return [
        sub_cfg
        for name in ("language_model", "text_model")
        if (sub_cfg := get_submodule(cfg, name)) is not None
    ] + [cfg]


def _num_devices_of(cfg: RblnConfigLike) -> int | None:
    val = get_field(cfg, "num_devices")
    if val is not None:
        assert isinstance(val, int), (
            f"num_devices must be an int, got {type(val).__name__}"
        )
        assert val > 0, "num_devices must be a positive integer"
        return val
    compile_cfgs = get_field(cfg, "_compile_cfgs")
    if isinstance(compile_cfgs, list):
        for entry in compile_cfgs:
            entry_val = get_field(entry, "num_devices")
            if entry_val is not None:
                assert isinstance(entry_val, int), (
                    f"num_devices must be an int, got {type(entry_val).__name__}"
                )
                assert entry_val > 0, "num_devices must be a positive integer"
                return entry_val
    return None


def read_num_devices(cfg: RblnConfigLike) -> int | None:
    for submodule_name in ("language_model", "text_model"):
        sub_cfg = get_submodule(cfg, submodule_name)
        if sub_cfg is None:
            continue
        sub_val = _num_devices_of(sub_cfg)
        if sub_val is not None:
            return sub_val
    return _num_devices_of(cfg)


def read_image_prefill_chunk_size(cfg: RblnConfigLike) -> list[int] | None:
    """Resolve image-prefill buckets exactly as optimum-rbln persists them.

    optimum-rbln stores ``image_prefill_chunk_size`` under one key but with two
    shapes: gemma3 persists a scalar ``int`` (a single bucket); gemma4 persists
    a descending ``list[int]`` (multiple buckets). Both are normalized to a
    ``list[int]``. Any other type is a config error. Returns ``None`` when the
    key is absent.
    """
    sizes = get_field(cfg, "image_prefill_chunk_size")
    if sizes is None:
        return None
    if isinstance(sizes, list):
        return list(sizes)
    # bool is an int subclass; reject it explicitly.
    if isinstance(sizes, int) and not isinstance(sizes, bool):
        return [sizes]
    raise TypeError(
        "image_prefill_chunk_size must be an int or list[int], got "
        f"{type(sizes).__name__}"
    )


def read_kvcache_block_size(cfg: RblnConfigLike) -> int | None:
    """Resolve ``kvcache_block_size``, reconciling it with ``kvcache_partition_len``.

    Some rbln_config payloads only carry ``kvcache_partition_len``; when both
    are present they must agree.
    """
    kvcache_block_size = get_field(cfg, "kvcache_block_size")
    kvcache_partition_len = get_field(cfg, "kvcache_partition_len")
    if kvcache_partition_len is None:
        return kvcache_block_size
    if kvcache_block_size is None:
        return kvcache_partition_len
    assert kvcache_partition_len == kvcache_block_size, (
        "kvcache_partition_len must equal kvcache_block_size. "
        "Please check the values in rbln_config.json"
    )
    return kvcache_block_size


def read_language_model_fields(cfg: RblnConfigLike, *keys: str) -> dict[str, Any]:
    """Select each field in language_model, text_model, root order, skipping None.

    Visit every source so lower-priority alias conflicts are still rejected.
    """
    values: dict[str, Any] = {}
    for source in language_model_configs(cfg):
        for key in keys:
            value: Any
            if key == "kvcache_block_size":
                value = read_kvcache_block_size(source)
            elif key == "image_prefill_chunk_size":
                value = read_image_prefill_chunk_size(source)
            else:
                value = get_field(source, key)
            if value is not None:
                values.setdefault(key, value)
    return values
