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
from pathlib import Path
from typing import TYPE_CHECKING

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


def _cfg_get(cfg, key: str, default=None):
    """Access a config value from either a dict or an RBLNModelConfig instance."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_get_submodule(cfg, submodule: str):
    """Get a submodule config from either a dict or an RBLNModelConfig instance.
    Returns None if the submodule doesn't exist."""
    if isinstance(cfg, dict):
        return cfg.get(submodule)
    sub = getattr(cfg, submodule, None)
    # RBLNModelConfig stores submodules as attributes;
    # filter out None / non-existent submodules.
    return sub


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


def get_rbln_params(
    vllm_config: VllmConfig,
    rbln_config: dict | RBLNModelConfig,
) -> tuple[int, int, int, int, int]:
    kvcache_block_size = None
    prefill_chunk_size = 128
    batch_size = None
    max_seq_len = None

    if is_enc_dec_arch(vllm_config.model_config.hf_config):
        max_seq_len = _cfg_get(rbln_config, "dec_max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = _cfg_get(rbln_config, "batch_size")
        num_blocks = _cfg_get(rbln_config, "kvcache_num_blocks")
    elif is_multi_modal(vllm_config.model_config.hf_config):
        # Get configurations from main module (e.g. Qwen2.5-VL, Whisper)
        kvcache_block_size = _cfg_get(rbln_config, "kvcache_block_size")
        batch_size = _cfg_get(rbln_config, "batch_size")
        max_seq_len = _cfg_get(rbln_config, "max_seq_len")
        num_blocks = _cfg_get(rbln_config, "kvcache_num_blocks")
        if max_seq_len is None:  # Whisper FIXME to be moved to enc-dec
            max_seq_len = _cfg_get(rbln_config, "dec_max_seq_len")
        # Get configurations from submodule
        if kvcache_block_size is None:
            submodules = ["language_model", "text_model"]
            for submodule in submodules:
                sub_cfg = _cfg_get_submodule(rbln_config, submodule)
                if sub_cfg is not None:
                    kvcache_block_size = _cfg_get(sub_cfg, "kvcache_block_size")
                    batch_size = _cfg_get(sub_cfg, "batch_size")
                    max_seq_len = _cfg_get(sub_cfg, "max_seq_len")
                    num_blocks = _cfg_get(sub_cfg, "kvcache_num_blocks")
                    if kvcache_block_size is not None:
                        break

    elif is_pooling_arch(vllm_config.model_config.hf_config):
        max_seq_len = _cfg_get(rbln_config, "max_seq_len")
        kvcache_block_size = max_seq_len
        batch_size = _cfg_get(rbln_config, "batch_size")
        num_blocks = _cfg_get(rbln_config, "kvcache_num_blocks")
        if num_blocks is None:
            num_blocks = batch_size  # for pooling models, each sequence is one block
    else:
        # decoder
        kvcache_block_size = _cfg_get(rbln_config, "kvcache_block_size")
        prefill_chunk_size = _cfg_get(rbln_config, "prefill_chunk_size", 128)
        batch_size = _cfg_get(rbln_config, "batch_size")
        max_seq_len = _cfg_get(rbln_config, "max_seq_len")
        num_blocks = _cfg_get(rbln_config, "kvcache_num_blocks")

    assert num_blocks is not None, "num_blocks must be specified in rbln_config.json"

    assert kvcache_block_size is not None, (
        "kvcache_block_size must be specified in rbln_config.json"
    )
    assert batch_size is not None, "batch_size must be specified in rbln_config.json"
    assert max_seq_len is not None, "max_seq_len must be specified in rbln_config.json"
    # NOTE:
    # prefill_chunk_size is only used for decoder-only models
    # with prefix caching
    return num_blocks, batch_size, max_seq_len, kvcache_block_size, prefill_chunk_size
