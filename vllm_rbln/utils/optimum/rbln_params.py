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

"""Helpers for reading and parsing rbln_config.json parameters."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from optimum.rbln.configuration_utils import RBLNModelConfig

from vllm_rbln.logger import init_logger
from vllm_rbln.utils.optimum.registry import (
    is_enc_dec_arch,
    is_multi_modal,
    is_pooling_arch,
)

logger = init_logger(__name__)

# Either a plain dict (from rbln_config.json) or an RBLNModelConfig instance.
RblnConfigLike = Union[dict, RBLNModelConfig]


def _cfg_get(cfg: RblnConfigLike, key: str, default: Any = None) -> Any:
    """Access a config value from either a dict or an RBLNModelConfig instance."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_get_submodule(cfg: RblnConfigLike, submodule: str) -> RblnConfigLike | None:
    """Get a submodule config; returns ``None`` if the submodule is absent."""
    if isinstance(cfg, dict):
        return cfg.get(submodule)
    return getattr(cfg, submodule, None)


def get_rbln_config(vllm_config: VllmConfig) -> dict | None:
    rbln_config_path = Path(
        os.path.join(vllm_config.model_config.model, "rbln_config.json")
    )
    if not rbln_config_path.exists():  # for pytest
        logger.warning(
            "rbln_config.json not found in model directory: %s. "
            "Using `block_size` from vllm_config.cache_config instead.",
            rbln_config_path,
        )
        return None
    with open(rbln_config_path, encoding="utf-8") as f:
        rbln_config = json.load(f)
    return rbln_config


@dataclass
class CompleteRBLNParams:
    """:class:`RBLNParams` with every field guaranteed present.

    Returned by :meth:`RBLNParams.assert_complete` so callers that rely on
    the values being populated get non-Optional types.
    """

    num_blocks: int
    batch_size: int
    max_seq_len: int
    kvcache_block_size: int
    prefill_chunk_size: int = 128
    tensor_parallel_size: int = 1


@dataclass
class RBLNParams:
    """Parameters derived from an optimum-rbln ``rbln_config.json``.

    Any field may be ``None`` when parsed from a user-supplied partial config
    (e.g. the ``additional_config`` passed before compilation). Call
    :meth:`assert_complete` once a fully materialised rbln_config is expected;
    it returns a :class:`CompleteRBLNParams` whose fields are non-Optional.
    """

    num_blocks: int | None = None
    batch_size: int | None = None
    max_seq_len: int | None = None
    kvcache_block_size: int | None = None
    prefill_chunk_size: int = 128
    tensor_parallel_size: int = 1

    @classmethod
    def from_rbln_config(
        cls, vllm_config: VllmConfig, rbln_config: RblnConfigLike
    ) -> "RBLNParams":
        """Parse rbln_config according to the model architecture."""
        hf_config = vllm_config.model_config.hf_config
        tensor_parallel_size = _cfg_get(rbln_config, "tensor_parallel_size", 1)

        if is_enc_dec_arch(hf_config):
            params = cls._parse_enc_dec(rbln_config)
        elif is_multi_modal(hf_config):
            params = cls._parse_multimodal(rbln_config)
        elif is_pooling_arch(hf_config):
            params = cls._parse_pooling(rbln_config)
        else:
            params = cls._parse_decoder(rbln_config)

        params.tensor_parallel_size = tensor_parallel_size
        return params

    def assert_complete(self) -> CompleteRBLNParams:
        """Validate every required field is set, returning a narrowed view."""
        assert self.num_blocks is not None, (
            "num_blocks must be specified in rbln_config.json"
        )
        assert self.kvcache_block_size is not None, (
            "kvcache_block_size must be specified in rbln_config.json"
        )
        assert self.batch_size is not None, (
            "batch_size must be specified in rbln_config.json"
        )
        assert self.max_seq_len is not None, (
            "max_seq_len must be specified in rbln_config.json"
        )
        return CompleteRBLNParams(
            num_blocks=self.num_blocks,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            kvcache_block_size=self.kvcache_block_size,
            prefill_chunk_size=self.prefill_chunk_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    @classmethod
    def _parse_enc_dec(cls, cfg: RblnConfigLike) -> "RBLNParams":
        max_seq_len = _cfg_get(cfg, "dec_max_seq_len")
        return cls(
            num_blocks=_cfg_get(cfg, "kvcache_num_blocks"),
            batch_size=_cfg_get(cfg, "batch_size"),
            max_seq_len=max_seq_len,
            kvcache_block_size=max_seq_len,
        )

    @classmethod
    def _parse_pooling(cls, cfg: RblnConfigLike) -> "RBLNParams":
        max_seq_len = _cfg_get(cfg, "max_seq_len")
        batch_size = _cfg_get(cfg, "batch_size")
        # For pooling models each sequence occupies exactly one block.
        num_blocks = _cfg_get(cfg, "kvcache_num_blocks")
        if num_blocks is None:
            num_blocks = batch_size
        return cls(
            num_blocks=num_blocks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=max_seq_len,
        )

    @classmethod
    def _parse_decoder(cls, cfg: RblnConfigLike) -> "RBLNParams":
        kvcache_block_size = _resolve_kvcache_block_size(cfg, arch="decoder")
        return cls(
            num_blocks=_cfg_get(cfg, "kvcache_num_blocks"),
            batch_size=_cfg_get(cfg, "batch_size"),
            max_seq_len=_cfg_get(cfg, "max_seq_len"),
            kvcache_block_size=kvcache_block_size,
            prefill_chunk_size=_cfg_get(cfg, "prefill_chunk_size", 128),
        )

    @classmethod
    def _parse_multimodal(cls, cfg: RblnConfigLike) -> "RBLNParams":
        kvcache_block_size = _resolve_kvcache_block_size(cfg, arch="multi-modal")
        batch_size = _cfg_get(cfg, "batch_size")
        max_seq_len = _cfg_get(cfg, "max_seq_len")
        num_blocks = _cfg_get(cfg, "kvcache_num_blocks")
        # Whisper exposes dec_max_seq_len at the top level; FIXME move to enc-dec.
        if max_seq_len is None:
            max_seq_len = _cfg_get(cfg, "dec_max_seq_len")

        # Fall back to a known submodule when the main module does not expose
        # these fields (e.g. language_model / text_model for some VLMs).
        if kvcache_block_size is None:
            for submodule_name in ("language_model", "text_model"):
                sub_cfg = _cfg_get_submodule(cfg, submodule_name)
                if sub_cfg is None:
                    continue
                kvcache_block_size = _cfg_get(sub_cfg, "kvcache_block_size")
                if kvcache_block_size is not None:
                    batch_size = _cfg_get(sub_cfg, "batch_size")
                    max_seq_len = _cfg_get(sub_cfg, "max_seq_len")
                    num_blocks = _cfg_get(sub_cfg, "kvcache_num_blocks")
                    break

        return cls(
            num_blocks=num_blocks,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kvcache_block_size=kvcache_block_size,
        )


def _resolve_kvcache_block_size(cfg: RblnConfigLike, *, arch: str) -> int | None:
    """Resolve ``kvcache_block_size``, reconciling it with ``kvcache_partition_len``.

    Some rbln_config payloads only carry ``kvcache_partition_len``; when both
    are present they must agree.
    """
    kvcache_block_size = _cfg_get(cfg, "kvcache_block_size")
    kvcache_partition_len = _cfg_get(cfg, "kvcache_partition_len")
    if kvcache_partition_len is None:
        return kvcache_block_size
    if kvcache_block_size is None:
        return kvcache_partition_len
    assert kvcache_partition_len == kvcache_block_size, (
        f"kvcache_partition_len must equal kvcache_block_size for {arch} models. "
        "Please check the values in rbln_config.json"
    )
    return kvcache_block_size
