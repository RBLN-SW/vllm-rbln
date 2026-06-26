# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_rbln.envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.attention.kv_cache_bindings import KVCacheViewInfo

from ..ops.attention_naive import (
    attention_naive_decode,
    attention_naive_prefill,
)
from ..ops.causal_attention_naive import (
    causal_attention_naive_decode,
    causal_attention_naive_prefill,
)
from ..ops.flash_attention_naive import (
    flash_attention_naive_decode,
    flash_attention_naive_prefill,
)
from ..ops.flash_causal_attention_naive import (
    flash_causal_attention_naive_decode,
    flash_causal_attention_naive_prefill,
)
from ..ops.sliding_window_attention_naive import (
    sliding_window_attention_naive_decode,
    sliding_window_attention_naive_prefill,
)

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.CUSTOM)
class RBLNFlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttentionImpl"]:
        return RBLNFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """kv cache shape
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 32/8 = 4
        # D - head_size
        # L - q_len
        list of kv cache = [num_layer][kv=2]
        kv_cache_shape= [B, H, 1, S, D]
        query_shape   = [1, H, G, L, D]
        """
        return (2, num_blocks, num_kv_heads, 1, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]


@dataclass
class RBLNFlashAttentionMetadata:
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    # For RBLN Attention
    is_prefill: bool
    attn_masks: torch.Tensor | None = None
    kv_caches: list[torch.Tensor] | None = None
    kv_cache_view_infos: list[KVCacheViewInfo] | None = None

    # For sliding window attention
    cache_seq_lens: torch.Tensor | None = None
    cache_offsets: torch.Tensor | None = None
    local_block_tables: torch.Tensor | None = None
    swa_attn_masks: torch.Tensor | None = None

    def __post_init__(self):
        # FIXME(RBLN): to_dynamic_index does not accept int64 inputs.Thus in the
        # VLLM_RBLN_USE_CUSTOM_KERNEL=0 path, rebel-compiler automatically converts
        # integer tensor inputs to a supported dtype.
        # However, this preprocessing is somewhat missing in the triton-rbln kernel
        # path(VLLM_RBLN_USE_CUSTOM_KERNEL=1), so we explicitly cast the input to a
        # supported dtype here. This can be removed when the triton-rbln kernel path
        # performs the same dtype conversion.

        if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return

        self.seq_lens = self.seq_lens.to(torch.int32)
        self.cache_seq_lens = (
            self.cache_seq_lens.to(torch.int32) if self.cache_seq_lens else None
        )
        self.cache_offsets = (
            self.cache_offsets.to(torch.int32) if self.cache_offsets else None
        )


class RBLNFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[RBLNFlashAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config

        self.block_size = kv_cache_spec.block_size
        self.chunked_prefill_size = self.scheduler_config.max_num_batched_tokens
        self.enforce_eager = get_current_vllm_config().model_config.enforce_eager
        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
        self.is_batch_attention_opt = envs.VLLM_RBLN_BATCH_ATTN_OPT

    def reorder_batch(
        self, input_batch: "InputBatch", scheduler_output: "SchedulerOutput"
    ) -> bool:
        return False

    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
    ) -> RBLNFlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        block_tables_tensor = common_attn_metadata.block_table_tensor

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        num_computed_tokens = seq_lens - query_seq_lens_cpu

        seq_idx = positions[query_start_loc_cpu[:num_reqs]].view(-1, 1)

        # The length of the partition equals the block size.
        partition_len = self.block_size
        # num_partition is derived from max_model_len (not hardware block count)
        # to ensure seq_idx/seq_lens dimensions stay within block_table bounds.
        max_seq_len = self.model_config.max_model_len
        num_partition = max_seq_len // partition_len

        cs = seq_idx.repeat(1, num_partition)
        pidx = torch.arange(num_partition, dtype=torch.int32)
        seq_lens_tensor = torch.clamp(cs - pidx * partition_len, 0, partition_len)

        attn_masks = None
        if is_prefill:
            # NOTE(RBLN): block_tables_tensor for prefill must be a 1D tensor.
            block_tables_tensor = block_tables_tensor[0]
            if not self.is_causal:
                prefill_chunk_size = self.chunked_prefill_size
                chunked_attention_mask = torch.zeros(
                    1,
                    1,
                    1,
                    prefill_chunk_size,
                    max_seq_len,
                    dtype=torch.float16 if self.enforce_eager else torch.float32,
                )
                causal_mask = 1 - torch.triu(
                    torch.ones(1, 1, prefill_chunk_size, prefill_chunk_size),
                    diagonal=1,
                )
                step = seq_idx[0]
                if step >= prefill_chunk_size:
                    chunked_attention_mask[:, :, :, :, :step] = 1
                chunked_attention_mask[:, :, :, :, step : step + prefill_chunk_size] = (
                    causal_mask
                )
                attn_masks = chunked_attention_mask
                attn_masks = attn_masks.to(self.device)
        else:
            seq_idx = rbln_utils.pad(seq_idx, 0, batch_pad)
            seq_lens_tensor = rbln_utils.pad(seq_lens_tensor, 0, batch_pad)
            block_tables_tensor = rbln_utils.pad(block_tables_tensor, 0, batch_pad)
            if not self.is_causal:
                decode_attention_mask = torch.zeros(
                    batch_pad,
                    1,
                    1,
                    1,
                    max_seq_len,
                    dtype=torch.float16 if self.enforce_eager else torch.float32,
                )
                for batch_index, batch_step in enumerate(seq_lens):
                    decode_attention_mask[batch_index, :, :, :, : batch_step + 1] = 1
                attn_masks = decode_attention_mask
                attn_masks = attn_masks.to(self.device)

        cache_seq_lens = None
        cache_offsets = None
        local_block_tables = None
        swa_attn_masks = None
        if sliding_window := getattr(self.kv_cache_spec, "sliding_window", None):
            num_computed_tokens = num_computed_tokens[:num_reqs].view(-1, 1)
            seq_lens = seq_lens[:num_reqs].view(-1, 1)
            query_lens = seq_lens - num_computed_tokens
            cache_seq_lens = torch.clamp(num_computed_tokens, max=sliding_window)
            cache_offsets = cache_seq_lens + query_lens
            if not is_prefill:
                cache_seq_lens = rbln_utils.pad(cache_seq_lens, 0, batch_pad)
                cache_offsets = rbln_utils.pad(cache_offsets, 0, batch_pad)
                # Generate sliding window attention mask for decode
                # mask[b, s] = 1.0 if s <= cache_seq_lens[b] else 0.0
                positions = torch.arange(sliding_window)[None, :]
                swa_attn_masks = torch.where(positions - cache_seq_lens > 0, 0.0, 1.0)[
                    :, None, None, :
                ]

            local_block_tables = block_tables_tensor[..., :1]

        # * seq_idx(batch attention opt decode) - [B, 1],
        #   for each batch, have sequence offset
        # * seq_lens_tensor(otherwise)      - [B, P],
        #   have dynamic size for each partition
        attn_metadata = RBLNFlashAttentionMetadata(
            seq_lens=seq_lens_tensor.to(self.device)
            if not self.is_batch_attention_opt or is_prefill or batch_pad <= 1
            else seq_idx.to(self.device),
            block_tables=block_tables_tensor.to(self.device),
            is_prefill=is_prefill,
            attn_masks=attn_masks,
            cache_seq_lens=cache_seq_lens.to(self.device)
            if cache_seq_lens is not None
            else None,
            cache_offsets=cache_offsets.to(self.device)
            if cache_offsets is not None
            else None,
            local_block_tables=local_block_tables.to(self.device)
            if local_block_tables is not None
            else None,
            swa_attn_masks=swa_attn_masks.to(self.device)
            if swa_attn_masks is not None
            else None,
        )

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class RBLNFlashAttentionImpl(AttentionImpl[RBLNFlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        vllm_config = get_current_vllm_config()
        self.enforce_eager = vllm_config.model_config.enforce_eager
        self.device = vllm_config.device_config.device
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in RBLN.")
        if logits_soft_cap is not None:
            logger.warning_once(
                "RBLN Attention Backend does not support logits soft cap. "
                "Outputs may be slightly off."
            )
            logits_soft_cap = None

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = torch.tensor(scale, device=self.device)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = RBLNFlashAttentionBackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by RBLNFlashAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )
        if kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "FP8 KV cache is not supported by RBLNFlashAttention."
            )
        self.attn_type = attn_type

        # TODO(RBLN): We need to apply sinks attn kernel.
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )
            if len(self.sinks.size()) == 1:
                self.sinks = self.sinks[:, None]

        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
        self.is_batch_attention_opt = envs.VLLM_RBLN_BATCH_ATTN_OPT
        self.is_normal = (self.block_size == self.max_model_len) and (
            self.sinks is None
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> None:
        """Forward pass with RBLNFlashAttention.

        Args:
            query:  shape = [num_tokens, num_heads, head_size]
            key:    shape = [num_tokens, num_kv_heads, head_size]
            value:  shape = [num_tokens, num_kv_heads, head_size]
            kv_cache shape= [2, num_blocks, num_kv_heads, 1,
                                block_size, head_size]

        Shape that we expect:
            kv_cache  = [2, num_blocks, num_kv_heads, 1, block_size, head_size]
            key       = [1, num_kv_heads, 1, block_size, head_size]
            query     = [1, num_kv_heads, 4, query_len, head_size]
            key_t     = [1, num_kv_heads, 1, head_size, block_size]

        Returns:
            attn_out  = [num_tokens, num_heads, head_size]

            hidden_size = num_heads * head_size
        """
        # B - num_blocks == num_partitions
        # S - block_size == partition_size
        # H - num_kv_heads
        # G - num_heads / num_kv_heads = 4
        # D - head_size
        # L - query length
        # C - max_seq_len
        # NB- num batch
        if output_scale is not None or output_block_scale is not None:
            raise RuntimeError(
                "Fused output quantization is not yet supported for RBLNFlashAttention."
            )

        # 1. query reshape for custom operation
        num_tokens = query.shape[0]
        b_size = attn_metadata.seq_lens.shape[0]
        q_len = num_tokens // b_size
        query = query.view(b_size, q_len, self.num_heads, self.head_size).transpose(
            1, 2
        )
        query = query.view(
            b_size, self.num_kv_heads, self.num_queries_per_kv, q_len, self.head_size
        )
        key = key.view(b_size, q_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        key = key.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)
        value = value.view(b_size, q_len, self.num_kv_heads, self.head_size).transpose(
            1, 2
        )
        value = value.view(b_size, self.num_kv_heads, 1, q_len, self.head_size)

        # NOTE - for cache update,
        # slot mapping will be necessary from sequence index
        # slot_mapping = [block_number, block_offset]

        # flash_attention_naive extended to have cache update
        # cache update is included into flash attention
        # but not within partition loop
        # input = {q, k, v, kv_cache, mask, scalar_scale,
        # seq_lens, block_table, slot_mapping}
        # output = {attn_output}
        # q, k, v = [batch,H,G,L,D]
        # key/value cache = [B,H,1,S,D]
        # mask  = [1,1,1,L,C]
        # o = [batch,H,G,L,D]

        # build attention mask within [0, 1]
        # - attention mask SHOULD be causal mask based on query length
        # - attention mask is used for masked softmax not actual value
        # if there is not positional embedding,
        # it can be merged into attention mask
        # attn_masks = _make_alibi_bias(alibi_slopes, dtype, seq_lens)
        # seq_lens_tensor (1, num_partition = 128k / k = 128)
        # ex) tensor[partition0 = 1024, partition1 = 10,
        # partition2 = 0, partition3 = 0] for len=1034
        # block_tables tensor (1, num_blocks = 256)
        # ex) tensor[block0 : 0, block1 : 100,
        #  block2: 10, block3: 5, ...]
        # attn_output = [batch,H,4,L,D]
        if self.sliding_window is not None:
            assert self.sliding_window == kv_cache.size(-2), (
                "SWA kernel_block_size must match window_size"
            )
            assert attn_metadata.cache_seq_lens is not None
            assert attn_metadata.cache_offsets is not None

            if attn_metadata.is_prefill:
                attn_output = sliding_window_attention_naive_prefill(
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    self.sinks,
                )
            else:
                attn_output = sliding_window_attention_naive_decode(
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    attn_metadata.swa_attn_masks
                    if self.is_batch_attention_opt and b_size > 1
                    else None,
                    self.sinks,
                )

        elif self.is_causal:
            if self.is_normal:
                if attn_metadata.is_prefill:
                    attn_output = causal_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = causal_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
            else:
                # * batched attention - seq_lens[B, 1] == seq_idx,
                #   original sequence index
                # * otherwise         - seq_lens[B, P] == seq_lens_tensor,
                #   dynamic size for each partition
                if attn_metadata.is_prefill:
                    attn_output = flash_causal_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = flash_causal_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
        else:
            if self.is_normal:
                if attn_metadata.is_prefill:
                    attn_output = attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens,
                        self.scale,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
            else:
                if attn_metadata.is_prefill:
                    attn_output = flash_attention_naive_prefill(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )
                else:
                    attn_output = flash_attention_naive_decode(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens,
                        attn_metadata.block_tables,
                        self.sinks,
                    )

        # 2. attention output reshape for attention backend return
        # attn_output = [batch,H*4,L,D] -> [batch,L,H*4,D] -> [batch*L,H*4,D]
        if self.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            attn_output = attn_output.reshape(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.reshape(
                b_size * q_len, self.num_heads, self.head_size
            )
        else:
            attn_output = attn_output.view(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.view(
                b_size * q_len, self.num_heads, self.head_size
            )

        output.copy_(attn_output)
