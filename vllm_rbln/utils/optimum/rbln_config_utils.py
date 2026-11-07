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

"""Read optimum-rbln config fields shared by compile specs and vLLM config updates.

Each reader accepts either a ``rbln_config.json`` dict or an ``RBLNModelConfig``.
"""

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from optimum.rbln.configuration_utils import RBLNModelConfig


# Either a plain dict (from rbln_config.json) or an RBLNModelConfig instance.
RblnConfigLike = Union[dict, "RBLNModelConfig"]


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


def resolve_num_devices(cfg: RblnConfigLike, *, default: int = 1) -> int:
    for submodule_name in ("language_model", "text_model"):
        sub_cfg = get_submodule(cfg, submodule_name)
        if sub_cfg is None:
            continue
        sub_val = _num_devices_of(sub_cfg)
        if sub_val is not None:
            return sub_val
    return _num_devices_of(cfg) or default


def resolve_image_prefill_chunk_size(cfg: RblnConfigLike) -> list[int] | None:
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


def resolve_kvcache_block_size(cfg: RblnConfigLike) -> int | None:
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


_PREFILL_CHUNK_SIZE_BY_FAMILY = {
    "ca": 128,  # ATOM
    "cr": 512,  # REBEL
}


def default_prefill_chunk_size() -> int:
    from vllm_rbln.platform import RblnPlatform

    device_name = RblnPlatform.get_device_name().lower()
    for family, chunk_size in _PREFILL_CHUNK_SIZE_BY_FAMILY.items():
        if family in device_name:
            return chunk_size
    raise RuntimeError(
        f"Unknown NPU device {device_name!r}; "
        "expected an ATOM (ca) or REBEL (cr) device."
    )
