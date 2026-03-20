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
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_rbln.rbln_envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


# RBLN custom op (flash causal attention naive prefill/decode w/o attn mask)
@torch.library.custom_op(
    "rbln_custom_ops::flash_causal_attention_naive_prefill", mutates_args=["kv_cache"]
)
def flash_causal_attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Expected tensor shapes:
    - q: [batch, n_kv_heads, n_groups, seq_len, head_dim]
      Query states for multiple tokens
    - k: [batch, n_kv_heads, 1, seq_len, head_dim]
      Key states for current input
    - v: [batch, n_kv_heads, 1, seq_len, head_dim]
      Value states for current input
    - kv_cache: [2, num_blocks, n_kv_heads, 1, partition_size, head_dim]
      Key and value cache
    - seq_idx: [batch, num_partitions]
      number of already cached tokens in each partition
    - block_tables: [num_partitions,] for prefill,
                    [batch, num_partitions] for decode
    - sinks: [n_heads, sink_len] (optional)

    Returns:
        Tensor: attn_output: [batch, n_kv_heads, n_groups, seq_len, head_dim]

    batch size is assumed to be 1 for prefill.
    """
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_causal_attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::flash_causal_attention_naive_decode", mutates_args=["kv_cache"]
)
def flash_causal_attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_causal_attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


from .flash_attention import (
    RBLNAttentionBackend,
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
    RBLNFlashAttentionImpl,
)


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class RBLNFlashAttnMLABackend(RBLNAttentionBackend):
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @staticmethod
    def get_name() -> str:
        return "RBLN_FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttnMLAImpl"]:
        return RBLNFlashAttnMLAImpl

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """kv cache shape
        # B - num_blocks
        # S - block_size
        # H - head_size
        """
        return (num_blocks, block_size, head_size)


class RBLNFlashAttnMLAImpl(RBLNFlashAttentionImpl[RBLNFlashAttentionMetadata]):
    def forward(
        self,
        layer: torch.nn.Module,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            q : torch.size([batch, num_tokens, num_heads, qk_head_dim])
            kv_c_normed: torch.size([batch, num_tokens, kv_lora_rank])
            k_pe: torch.size([batch, num_tokens, 1, qk_rope_head_dim])
            kv_cache: torch.size([num_blocks, block_size, num_kv_heads(= kv_lora_rank + qk_rope_head_dim)])
            attn_metadata: RBLNFlashAttentionMetadata
            output: torch.size([batch, num_tokens, num_heads * v_head_dim])
            output_scale: torch.size([batch, num_tokens, num_heads * v_head_dim])
            output_block_scale: torch.size([batch, num_tokens, num_heads * v_head_dim])

        Returns:
            attn_out  = (batch_size, seq_len, num_heads * v_head_dim)
        """
        b_size, q_len, _, _ = q.size()
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
        assert kv_cache is not None

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window attention is not supported for MLA"
            )
        elif not self.is_causal:
            raise NotImplementedError("Non-causal attention is not supported for MLA")
        # actually non-flash paged attention DOES NOT use slot_mapping
        else:
            if self.is_normal:
                assert attn_metadata.seq_lens is not None
                assert attn_metadata.block_tables is not None

                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_decode
                        )
                    else:
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_decode
                        )

                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = causal_attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = causal_attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode
                        )
                    else:
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode
                        )
                else:
                    flash_causal_attention_naive_prefill = (
                        flash_causal_attention_naive_prefill_impl
                    )
                    flash_causal_attention_naive_decode = (
                        flash_causal_attention_naive_decode_impl
                    )

                # * batched attention - seq_lens[B, 1] == seq_idx,
                #   original sequence index
                # * otherwise         - seq_lens[B, P] == dyn_size_for_partitions,
                #   dynamic size for each partition
                if not attn_metadata.is_prefill:
                    decode_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        decode_args.append(self.sinks)
                    attn_output = flash_causal_attention_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
                    ]
                    if not envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        prefill_args.append(self.sinks)
                    attn_output = flash_causal_attention_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )

        # 2. attention output reshape for attention backend return
        # attn_output = [batch,H*4,L,D] -> [batch,L,H*4,D] -> [batch,L,H*4*D]
        if self.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            attn_output = attn_output.reshape(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.reshape(
                b_size, q_len, self.num_heads * self.head_size
            )
        else:
            attn_output = attn_output.view(
                b_size, self.num_heads, q_len, self.head_size
            ).transpose(1, 2)
            attn_output = attn_output.view(
                b_size, q_len, self.num_heads * self.head_size
            )
        # attn_output = [batch,L,H*4*D]
        return attn_output
