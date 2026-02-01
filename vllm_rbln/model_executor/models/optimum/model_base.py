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
import math
import os
from pathlib import Path
from typing import Any, Optional

import optimum.rbln
import torch
import torch.nn as nn
from optimum.rbln.transformers.models.decoderonly import (
    decoderonly_runtime_utils as runtime_utils)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.utils.optimum.common import select_bucket_size
from vllm_rbln.utils.optimum.registry import get_rbln_model_info

logger = init_logger(__name__)


class KVCacheBlockAdapter:
    """
     KV cache block allocation behavior (v1 vs v0).
    +-------------------+---------------------+------------------+
    | Condition         | v1                  | v0               |
    +-------------------+---------------------+------------------+
    | is_full_block     | n() + 1             | n()              |
    | not is_full_block | n()                 | max(0, n() - 1)  |
    +-------------------+---------------------+------------------+
    n = estimated_num_blocks()
    
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        estimated_kvcache_num_blocks: int,
    ):
        self.vllm_config = vllm_config
        self.estimated_kvcache_num_blocks = estimated_kvcache_num_blocks

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _estimated_num_blocks(self) -> int:
        return self._env_int(
            "VLLM_RBLN_NPU_NUM_BLOCKS",
            int(self.estimated_kvcache_num_blocks),
        )

    def is_full_block_available(self) -> bool:
        """True if we can allocate a full batch worth of blocks."""
        estimated = self._estimated_num_blocks()

        if self.vllm_config.cache_config.enable_prefix_caching:
            block_size = self.vllm_config.additional_config["attn_block_size"]

        else:
            block_size = self.vllm_config.cache_config.block_size

        max_model_len = self.vllm_config.model_config.max_model_len
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs

        blocks_per_seq = math.ceil(max_model_len / block_size)
        ideal_total = max_num_seqs * blocks_per_seq
        return estimated >= ideal_total

    def get_available_num_blocks(self) -> int:
        if self.vllm_config.cache_config.enable_prefix_caching:
            ob_size = self.vllm_config.additional_config["attn_block_size"]
            ib_size = self.vllm_config.cache_config.block_size
            blk_ratio = ob_size // ib_size
        else:
            blk_ratio = 1

        if self.is_full_block_available():
            new_estimated = self._estimated_num_blocks() * blk_ratio
            return new_estimated + 1

        new_estimated = (self._estimated_num_blocks() - 1) * blk_ratio + 1
        return new_estimated


class RBLNOptimumModelBase(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.init_model()
        self.batch_size = self.scheduler_config.max_num_seqs
        self.kv_block_adapter = KVCacheBlockAdapter(
            vllm_config, self._resolve_kvcache_num_blocks())

    def _resolve_kvcache_num_blocks(self) -> int:
        """Prefer model-provided KV-cache block count; 
           else fall back to config."""
        value: Optional[Any] = None

        getter = getattr(self.model, "get_kvcache_num_blocks", None)
        if callable(getter):
            value = getter()
        elif hasattr(self.model.rbln_config, "kvcache_num_blocks"):
            value = self.model.rbln_config.kvcache_num_blocks
        else:
            value = self.scheduler_config.max_num_seqs  # fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(self.scheduler_config.max_num_seqs)

    def init_model(self) -> None:
        config = self.model_config.hf_config
        model_name, model_cls_name = get_rbln_model_info(config)

        if isinstance(self.model_config.model, str | Path) and os.path.exists(
                self.model_config.model):
            model_path = Path(self.model_config.model)
            if model_path.is_dir() and any(
                    model_path.glob('rbln_config.json')):
                compiled_path = self.model_config.model
            else:
                compiled_path = None
        else:
            compiled_path = None

        if compiled_path is None or not os.path.exists(compiled_path):
            raise RuntimeError(
                f"Compiled model path does not exist: {compiled_path}")

        # huggingface model class name
        logger.info("model_name = %s, model_cls_name = %s, model_path = %s",
                    model_name, model_cls_name, compiled_path)

        self.supports_transcription_only = (
            model_cls_name == "RBLNOptimumWhisperForConditionalGeneration")

        # huggingface model class
        model_cls = getattr(optimum.rbln, model_cls_name)
        assert model_cls is not None
        model = model_cls.from_pretrained(compiled_path, export=False)
        self.model = model
        self.rbln_model_config = model.rbln_config
        self.attn_impl = model.get_attn_impl() if hasattr(
            model, "get_attn_impl") else None

    @property
    def dtype(self) -> torch.dtype:
        assert self.model.rbln_config.dtype is not None
        return self.model.rbln_config.dtype


class RBLNOptimumDecoderMixin:

    def setup_decoder_mixin(
        self,
        attn_impl: str,
        vocab_size: int,
        use_multiple_decoder: bool,
        default_batch_size: int,
        decoder_batch_sizes: list[int],
        num_blocks: int,
    ):
        self.attn_impl = attn_impl
        self.use_multiple_decoder = use_multiple_decoder
        # FIXME: self.batch_size != self.decoder_batch_size ?
        self.decoder_batch_size = default_batch_size
        if self.use_multiple_decoder:
            self.decoder_batch_sizes = tuple(reversed(decoder_batch_sizes))

        self.logits_processor = LogitsProcessor(vocab_size,
                                                logits_as_input=True)
        self.available_blocks = torch.arange(
            0,
            num_blocks,
            dtype=torch.int16,
        )

    def pad_decoder_items(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        input_block_ids: Optional[torch.Tensor] = None,
        padded_batch_size: Optional[int] = None,
        dummy_block: Optional[int] = None,
    ):
        assert input_ids.shape[1] == 1
        if input_block_ids is None and padded_batch_size is None:
            raise ValueError(
                "Either input_block_ids or padded_batch_size must be provided."
            )
        elif input_block_ids is not None and padded_batch_size is not None:
            raise ValueError(
                "Cannot provide both input_block_ids and padded_batch_size.")

        if padded_batch_size is None:
            padded_batch_size = self.decoder_batch_size

        original_batch_size = input_ids.shape[0]

        padded_input_ids = torch.zeros(padded_batch_size,
                                       1,
                                       dtype=input_ids.dtype)
        padded_position_ids = torch.zeros(padded_batch_size,
                                          1,
                                          dtype=positions.dtype)
        padded_block_tables = torch.zeros(padded_batch_size,
                                          block_tables.shape[1],
                                          dtype=block_tables.dtype).fill_(-1)

        mask = torch.ones_like(
            padded_block_tables,
            dtype=torch.bool,
            device=block_tables.device,
        )

        if input_block_ids is None:
            padded_input_ids[:original_batch_size] = input_ids
            padded_position_ids[:original_batch_size] = positions
            padded_block_tables[:original_batch_size] = block_tables
            mask[:original_batch_size, :] = False
        else:
            padded_input_ids[input_block_ids] = input_ids
            padded_position_ids[input_block_ids] = positions
            padded_block_tables[input_block_ids] = block_tables
            mask[input_block_ids, :] = False

        if torch.any(mask):
            if dummy_block is not None:
                padding_blocks = torch.tensor([dummy_block],
                                              dtype=block_tables.dtype)
            else:
                padding_blocks = self.available_blocks[
                    ~torch.isin(self.available_blocks, block_tables.flatten())]
            padded_block_tables[mask] = padding_blocks[0]
        return padded_input_ids, padded_position_ids, padded_block_tables

    def preprocess_for_decoder(
        self,
        is_prompt: bool,
        block_tables: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        input_block_ids: Optional[list[int]] = None,
        dummy_block: Optional[int] = None,
    ):
        padded_batch_size = None
        # 1. Set the type
        # TODO: Does it require changing the dtype dynamically?
        input_ids = input_ids.to(
            torch.int64) if input_ids is not None else None
        cache_position = cache_position.to(
            torch.int32) if cache_position is not None else None
        block_tables = block_tables.to(torch.int16)

        # 2. Adjust the shape of tensors by squeezing and padding
        if is_prompt:
            block_tables = block_tables.squeeze(0)
            padded_batch_size = 1
        else:
            if input_block_ids is None:
                padded_batch_size = self.decoder_batch_size
                if input_ids is not None:
                    request_nums = input_ids.shape[0]
                # Select lower-bounded batch size in case of multiple decoders
                if self.use_multiple_decoder:
                    padded_batch_size = select_bucket_size(
                        request_nums, self.decoder_batch_sizes)

            input_ids, cache_position, block_tables = self.pad_decoder_items(
                input_ids,
                cache_position,
                block_tables,
                input_block_ids=input_block_ids,
                padded_batch_size=padded_batch_size,
                dummy_block=dummy_block,
            )
        kwargs = {
            "block_tables": block_tables,
            "padded_batch_size": padded_batch_size,
            "input_ids": input_ids,
            "cache_position": cache_position,
        }
        return kwargs

    def _copy_cached_kv_blocks(self,
                               prefill_decoder: runtime_utils.RBLNRuntimeModel,
                               cached_block_tables: list[int],
                               cached_lengths: list[int],
                               block_tables: torch.Tensor):
        """Copy cached KV blocks from source to destination blocks.
        
        Args:
            cached_block_tables: List of source block IDs to copy from
            cached_lengths: List of cached lengths for each block
            block_tables: Tensor containing destination block IDs
        """
        if not cached_block_tables:
            return

        if len(cached_block_tables) != len(cached_lengths):
            raise ValueError(
                "Mismatch between cached_block_tables length (%s) "
                "and cached_lengths length (%s)", len(cached_block_tables),
                len(cached_lengths))

        # Convert to list once for efficiency
        dst_blocks = block_tables[0].tolist()

        for block_idx, (src_block, dst_block) in enumerate(
                zip(cached_block_tables, dst_blocks)):
            try:
                prefill_decoder.runtime._copy_kv_cache(
                    src_block, dst_block, cached_lengths[block_idx])
                logger.debug(
                    "Successfully copied KV cache from block %d to block %d",
                    src_block, dst_block)
            except Exception as e:
                error_msg = (
                    "Failed to copy KV cache from block %d to block %d "
                    "at index %d: %s", src_block, dst_block, block_idx, e)
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    # It is required for decoder models in openai api server
    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.logits_processor(None, hidden_states, sampling_metadata)
