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

import math
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


@torch.library.custom_op(
    "rbln_custom_ops::attention_naive_prefill", mutates_args=["kv_cache"]
)
def attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::attention_naive_decode", mutates_args=["kv_cache"]
)
def attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::causal_attention_naive_prefill", mutates_args=["kv_cache"]
)
def causal_attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::causal_attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::causal_attention_naive_decode", mutates_args=["kv_cache"]
)
def causal_attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::causal_attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


# RBLN custom op (flash attention naive prefill/decode)
@torch.library.custom_op(
    "rbln_custom_ops::flash_attention_naive_prefill", mutates_args=["kv_cache"]
)
def flash_attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
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
    - mask: [batch, 1, 1, seq_len, max_seq_len]
    - seq_idx: [batch, num_partitions]
      number of already cached tokens in each partition
    - block_tables: [num_partitions,] for prefill,
                    [batch, num_partitions] for decode
    - sinks: [n_heads, sink_len] (optional)

    Returns:
        Tensor: attn_output: [batch, n_kv_heads, n_groups, seq_len, head_dim]

    batch size is assumed to be 1 for prefill.
    """
    if not envs.VLLM_RBLN_COMPILE_MODEL:
        # attn_weights = MM(q,kt) * scale
        # attn_weights = add(attn_weights + mask)
        # attn_weights = softmax(attn_weights)
        # MM(attn_weights, v)
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        s = seq_idx[0][0]
        e = s + seq_len
        # NOTE: this reference impl works only for single partition
        block = block_tables[0].to(torch.int32)
        k_state = (
            kv_cache[0][block].unsqueeze(0).slice_scatter(k, dim=3, start=s, end=e)
        )
        v_state = (
            kv_cache[1][block].unsqueeze(0).slice_scatter(v, dim=3, start=s, end=e)
        )
        kv_cache[0][block] = k_state.squeeze(0)
        kv_cache[1][block] = v_state.squeeze(0)
        attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
        causal_mask = torch.where(
            mask[:, :, :, :, :partition] > 0, 0.0, -float("inf")
        ).to(attn_weights.dtype)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_state)
        return attn_output
    else:
        return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::flash_attention_naive_decode", mutates_args=["kv_cache"]
)
def flash_attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if not envs.VLLM_RBLN_COMPILE_MODEL:
        # NOTE: this reference impl works only for batch_size=1
        assert q.size(0) == 1
        partition = kv_cache.size(-2)
        seq_len = q.size(-2)
        s = seq_idx[0][0]
        e = s + seq_len
        # NOTE: this reference impl works only for single partition
        block = block_tables[0][0].to(torch.int32)
        k_state = (
            kv_cache[0][block].unsqueeze(0).slice_scatter(k, dim=3, start=s, end=e)
        )
        v_state = (
            kv_cache[1][block].unsqueeze(0).slice_scatter(v, dim=3, start=s, end=e)
        )
        kv_cache[0][block] = k_state.squeeze(0)
        kv_cache[1][block] = v_state.squeeze(0)
        attn_weights = torch.matmul(q, k_state.transpose(3, 4)) * scale
        causal_mask = torch.where(
            mask[:, :, :, :, :partition] > 0, 0.0, -float("inf")
        ).to(attn_weights.dtype)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_state)
        return attn_output
    else:
        return torch.empty_like(q)


@torch.library.register_fake("rbln_custom_ops::flash_attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


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
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(q)

    # This is just reference to test vllm-rbln independently of the actual RBLN
    # custom op implementation, so it implements simple non-flash attention.

    batch_size, n_kv_heads, n_groups, seq_len, head_dim = q.shape
    partition_size = kv_cache.size(-2)
    num_partitions = block_tables.shape[0]

    # Calculate the starting position (number of tokens already in cache)
    # seq_idx contains tokens per partition that are already cached
    cache_start_pos = int(seq_idx[0].sum().item())
    total_seq_len = cache_start_pos + seq_len

    # Step 1: Write KV cache
    # We need to write seq_len new tokens starting at cache_start_pos
    for p in range(num_partitions):
        block_idx = block_tables[p].to(torch.int32)

        # Calculate how many tokens to write to this partition
        partition_start = p * partition_size
        partition_end = (p + 1) * partition_size

        # Tokens we're writing go from cache_start_pos to
        # cache_start_pos + seq_len
        write_start = max(cache_start_pos, partition_start)
        write_end = min(cache_start_pos + seq_len, partition_end)

        if write_start >= write_end:
            continue

        num_tokens_to_write = write_end - write_start
        offset_in_partition = write_start - partition_start
        offset_in_input = write_start - cache_start_pos

        k_slice = k[:, :, :, offset_in_input : offset_in_input + num_tokens_to_write, :]
        v_slice = v[:, :, :, offset_in_input : offset_in_input + num_tokens_to_write, :]

        kv_cache[
            0,
            block_idx,
            :,
            :,
            offset_in_partition : offset_in_partition + num_tokens_to_write,
            :,
        ] = k_slice.squeeze(0)
        kv_cache[
            1,
            block_idx,
            :,
            :,
            offset_in_partition : offset_in_partition + num_tokens_to_write,
            :,
        ] = v_slice.squeeze(0)

    # Step 2: Gather KV cache for the entire sequence
    k_gathered = torch.zeros(
        batch_size,
        n_kv_heads,
        1,
        total_seq_len,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )
    v_gathered = torch.zeros(
        batch_size,
        n_kv_heads,
        1,
        total_seq_len,
        head_dim,
        dtype=v.dtype,
        device=v.device,
    )

    gathered_pos = 0
    for p in range(num_partitions):
        block_idx = block_tables[p].to(torch.int32)

        # Calculate how many tokens are in this partition after writing
        partition_start = p * partition_size
        tokens_in_partition = min(total_seq_len - partition_start, partition_size)

        if tokens_in_partition <= 0:
            break

        k_gathered[:, :, :, gathered_pos : gathered_pos + tokens_in_partition, :] = (
            kv_cache[0, block_idx, :, :, :tokens_in_partition, :]
        )
        v_gathered[:, :, :, gathered_pos : gathered_pos + tokens_in_partition, :] = (
            kv_cache[1, block_idx, :, :, :tokens_in_partition, :]
        )
        gathered_pos += tokens_in_partition

    # Step 3: Compute causal attention (with sinks, if any)
    # attn_weights: [batch, n_kv_heads, n_groups, seq_len, total_seq_len]
    attn_weights = torch.matmul(q, k_gathered.transpose(3, 4)) * scale

    # Create causal mask
    # Query positions are from cache_start_pos to cache_start_pos + seq_len
    query_positions = torch.arange(
        cache_start_pos, cache_start_pos + seq_len, device=q.device
    )
    key_positions = torch.arange(total_seq_len, device=q.device)

    # Causal mask: query can only attend to keys at positions <= query position
    causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

    # Convert to attention mask format
    causal_mask = torch.where(causal_mask, 0.0, float("-inf")).to(attn_weights.dtype)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    attn_weights = attn_weights + causal_mask

    # Apply attention sinks if provided
    # sinks shape: [n_heads, sink_len] ->
    # expand to [batch, n_kv_heads, n_groups, seq_len, sink_len]
    if sinks is not None:
        # sinks: [n_heads, sink_len] where n_heads = n_kv_heads * n_groups
        sink_len = sinks.size(-1)
        # Reshape sinks to match attention weight dimensions
        sinks_expanded = sinks.view(n_kv_heads, n_groups, 1, sink_len)
        sinks_expanded = sinks_expanded.expand(
            batch_size, n_kv_heads, n_groups, seq_len, sink_len
        )
        # Concatenate sink logits to attention weights
        combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)
        # Stabilize softmax by subtracting max
        combined_logits = (
            combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        )
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        # Drop the sink probabilities before matmul with values
        attn_weights = probs[..., :-sink_len]
    else:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, v_gathered)

    return attn_output


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
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(q)

    # This is just reference to test vllm-rbln independently of the actual RBLN
    # custom op implementation, so it implements simple non-flash attention.

    batch_size, n_kv_heads, n_groups, seq_len, head_dim = q.shape
    partition_size = kv_cache.size(-2)
    num_partitions = block_tables.shape[1]

    outputs = []
    for b in range(batch_size):
        # Calculate the starting position (number of tokens already in cache)
        cache_start_pos = int(seq_idx[b].sum().item())
        total_seq_len = cache_start_pos + seq_len  # seq_len is 1 for decode

        if total_seq_len == 0:
            outputs.append(torch.zeros_like(q[b : b + 1]))
            continue

        # Step 1: Write KV cache
        # Find which partition to write to
        for p in range(num_partitions):
            block_idx = block_tables[b, p].to(torch.int32)

            partition_start = p * partition_size
            partition_end = (p + 1) * partition_size

            write_start = max(cache_start_pos, partition_start)
            write_end = min(cache_start_pos + seq_len, partition_end)

            if write_start >= write_end:
                continue

            num_tokens_to_write = write_end - write_start
            offset_in_partition = write_start - partition_start
            offset_in_input = write_start - cache_start_pos

            k_slice = k[
                b : b + 1,
                :,
                :,
                offset_in_input : offset_in_input + num_tokens_to_write,
                :,
            ]
            v_slice = v[
                b : b + 1,
                :,
                :,
                offset_in_input : offset_in_input + num_tokens_to_write,
                :,
            ]

            kv_cache[
                0,
                block_idx,
                :,
                :,
                offset_in_partition : offset_in_partition + num_tokens_to_write,
                :,
            ] = k_slice.squeeze(0)
            kv_cache[
                1,
                block_idx,
                :,
                :,
                offset_in_partition : offset_in_partition + num_tokens_to_write,
                :,
            ] = v_slice.squeeze(0)

        # Step 2: Gather KV cache for the entire sequence
        k_gathered = torch.zeros(
            1, n_kv_heads, 1, total_seq_len, head_dim, dtype=k.dtype, device=k.device
        )
        v_gathered = torch.zeros(
            1, n_kv_heads, 1, total_seq_len, head_dim, dtype=v.dtype, device=v.device
        )

        gathered_pos = 0
        for p in range(num_partitions):
            block_idx = block_tables[b, p].to(torch.int32)

            partition_start = p * partition_size
            tokens_in_partition = min(total_seq_len - partition_start, partition_size)

            if tokens_in_partition <= 0:
                break

            k_gathered[
                :, :, :, gathered_pos : gathered_pos + tokens_in_partition, :
            ] = kv_cache[0, block_idx, :, :, :tokens_in_partition, :]
            v_gathered[
                :, :, :, gathered_pos : gathered_pos + tokens_in_partition, :
            ] = kv_cache[1, block_idx, :, :, :tokens_in_partition, :]
            gathered_pos += tokens_in_partition

        # Step 3: Compute causal attention (with sinks, if any)
        q_b = q[b : b + 1]
        attn_weights = torch.matmul(q_b, k_gathered.transpose(3, 4)) * scale

        # For decode, query is at position total_seq_len - 1
        # It can attend to all previous positions (0 to total_seq_len - 1)
        # So no causal masking needed for decode

        # Apply attention sinks if provided
        # sinks shape: [n_heads, sink_len] ->
        # expand to [1, n_kv_heads, n_groups, seq_len, sink_len]
        if sinks is not None:
            sink_len = sinks.size(-1)
            # Reshape sinks to match attention weight dimensions
            sinks_expanded = sinks.view(n_kv_heads, n_groups, 1, sink_len)
            sinks_expanded = sinks_expanded.expand(
                1, n_kv_heads, n_groups, seq_len, sink_len
            )
            # Concatenate sink logits to attention weights
            combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)
            # Stabilize softmax by subtracting max
            combined_logits = (
                combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            )
            probs = torch.nn.functional.softmax(combined_logits, dim=-1)
            # Drop the sink probabilities before matmul with values
            attn_weights = probs[..., :-sink_len]
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v_gathered)
        outputs.append(attn_output)

    return torch.cat(outputs, dim=0)


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


@torch.library.custom_op(
    "rbln_custom_ops::sliding_window_attention_naive_prefill", mutates_args=["kv_cache"]
)
def sliding_window_attention_naive_prefill_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
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
    - kv_cache: [2, num_blocks, n_kv_heads, 1, window_size, head_dim]
      Key and value cache
    - cache_seq_len: [batch, 1]
      number of tokens already cached
    - cache_offset: [batch, 1]
      ending position after insertion (cache_seq_len + query_len)
    - scale: []. Attention scale factor
    - block_tables: [batch] for prefill, [batch, 1] for decode
    - sinks: [n_heads, sink_len] (optional)

    Returns:
        Tensor: attn_output: [batch, n_kv_heads, n_groups, seq_len, head_dim]

    batch size is assumed to be 1 for prefill.
    """
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(q)

    window_size = kv_cache.size(-2)
    seq_len = q.size(-2)
    cache_start = int(cache_seq_len[0][0].item())
    cache_end = int(cache_offset[0][0].item())
    block = int(block_tables[0].item())

    k_cache = kv_cache[0][block].unsqueeze(0)
    k_cache_curr = torch.cat([k_cache[:, :, :, :cache_start, :], k], dim=3)
    k_cache_curr = rbln_utils.pad(
        k_cache_curr,
        3,
        window_size + seq_len,
    )
    k_cache_slice = k_cache_curr[
        :, :, :, max(0, cache_end - window_size) : cache_end, :
    ]
    k_cache_slice = rbln_utils.pad(
        k_cache_slice,
        3,
        window_size,
    )
    kv_cache[0][block] = k_cache_slice.squeeze(0)

    v_cache = kv_cache[1][block].unsqueeze(0)
    v_cache_curr = torch.cat([v_cache[:, :, :, :cache_start, :], v], dim=3)
    v_cache_curr = rbln_utils.pad(
        v_cache_curr,
        3,
        window_size + seq_len,
    )
    v_cache_slice = v_cache_curr[
        :, :, :, max(0, cache_end - window_size) : cache_end, :
    ]
    v_cache_slice = rbln_utils.pad(
        v_cache_slice,
        3,
        window_size,
    )
    kv_cache[1][block] = v_cache_slice.squeeze(0)

    attn_weights = torch.matmul(q, k_cache_curr.transpose(3, 4)) * scale

    ones = torch.ones(window_size + seq_len, window_size + seq_len)
    mask_full = torch.tril(ones) - torch.tril(ones, diagonal=-window_size)
    mask = mask_full[None, None, None, cache_start : cache_start + seq_len, :]
    mask = torch.where(mask > 0, 0.0, float("-inf")).to(attn_weights.dtype)

    attn_weights = attn_weights + mask

    if sinks is not None:
        sink_len = sinks.size(-1)
        n_kv_heads = q.size(1)
        n_groups = q.size(2)
        sinks_expanded = sinks.view(n_kv_heads, n_groups, 1, sink_len)
        sinks_expanded = sinks_expanded.expand(
            1, n_kv_heads, n_groups, seq_len, sink_len
        )
        combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)
        combined_logits = (
            combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        )
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        attn_weights = probs[..., :-sink_len]
    else:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, v_cache_curr)

    return attn_output


@torch.library.register_fake("rbln_custom_ops::sliding_window_attention_naive_prefill")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "rbln_custom_ops::sliding_window_attention_naive_decode", mutates_args=["kv_cache"]
)
def sliding_window_attention_naive_decode_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.empty_like(q)

    window_size = kv_cache.size(-2)
    batch_size = q.size(0)

    outputs = []
    for r in range(batch_size):
        cache_start = int(cache_seq_len[r][0].item())
        cache_end = int(cache_offset[r][0].item())
        if cache_end - cache_start <= 0:
            outputs.append(torch.zeros_like(q[r : r + 1]))
            continue
        block = int(block_tables[r][0].item())

        q_r = q[r : r + 1]
        k_r = k[r : r + 1]
        v_r = v[r : r + 1]

        k_cache = kv_cache[0][block].unsqueeze(0)
        k_cache_curr = torch.cat([k_cache[:, :, :, :cache_start, :], k_r], dim=3)
        k_cache_curr = rbln_utils.pad(
            k_cache_curr,
            3,
            window_size + 1,
        )
        k_cache_slice = k_cache_curr[
            :, :, :, max(0, cache_end - window_size) : cache_end, :
        ]
        k_cache_slice = rbln_utils.pad(
            k_cache_slice,
            3,
            window_size,
        )
        kv_cache[0][block] = k_cache_slice.squeeze(0)

        v_cache = kv_cache[1][block].unsqueeze(0)
        v_cache_curr = torch.cat([v_cache[:, :, :, :cache_start, :], v_r], dim=3)
        v_cache_curr = rbln_utils.pad(
            v_cache_curr,
            3,
            window_size + 1,
        )
        v_cache_slice = v_cache_curr[
            :, :, :, max(0, cache_end - window_size) : cache_end, :
        ]
        v_cache_slice = rbln_utils.pad(
            v_cache_slice,
            3,
            window_size,
        )
        kv_cache[1][block] = v_cache_slice.squeeze(0)

        attn_weights = torch.matmul(q_r, k_cache_curr.transpose(3, 4)) * scale

        ones = torch.ones(window_size + 1, window_size + 1)
        mask_full = torch.tril(ones) - torch.tril(ones, diagonal=-window_size)
        mask = mask_full[None, None, None, cache_start : cache_start + 1, :]
        mask = torch.where(mask > 0, 0.0, float("-inf")).to(attn_weights.dtype)

        attn_weights = attn_weights + mask

        if sinks is not None:
            sink_len = sinks.size(-1)
            n_kv_heads = q.size(1)
            n_groups = q.size(2)
            sinks_expanded = sinks.view(n_kv_heads, n_groups, 1, sink_len)
            sinks_expanded = sinks_expanded.expand(1, n_kv_heads, n_groups, 1, sink_len)
            combined_logits = torch.cat([attn_weights, sinks_expanded], dim=-1)
            combined_logits = (
                combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            )
            probs = torch.nn.functional.softmax(combined_logits, dim=-1)
            attn_weights = probs[..., :-sink_len]
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v_cache_curr)
        outputs.append(attn_output)

    return torch.cat(outputs, dim=0)


@torch.library.register_fake("rbln_custom_ops::sliding_window_attention_naive_decode")
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    dummy: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


# RBLN custom op (cache update)
# NYI, custom op interface is only registered for test
# inputs = {cache, state, batch, seq}
@torch.library.custom_op("rbln_custom_ops::rbln_cache_update", mutates_args=())
def rbln_cache_update_impl(
    cache: torch.Tensor, state: torch.Tensor, slot_mapping: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(cache)


@torch.library.register_fake("rbln_custom_ops::rbln_cache_update")
def _(
    cache: torch.Tensor, state: torch.Tensor, slot_mapping: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(cache)


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class RBLNAttentionBackend(AttentionBackend):
    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "RBLN_ATTN"

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

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: dict[int, int],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: dict[int, list[int]],
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the RBLN backend.")


@dataclass
class RBLNFlashAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool | None
    common_prefix_len: int | None
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # For RBLN Attention
    attn_masks: torch.Tensor | None = None
    kv_caches: list[torch.Tensor] | None = None
    # for sliding window attention
    cache_seq_lens: torch.Tensor | None = None
    cache_offsets: torch.Tensor | None = None
    local_block_tables: torch.Tensor | None = None


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
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.scheduler_config = vllm_config.scheduler_config

        # self.runner = runner
        # self.input_batch = runner.input_batch
        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.chunked_prefill = (
            self.scheduler_config.enable_chunked_prefill
            or self.cache_config.enable_prefix_caching
        )
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
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        num_tokens=None,
        positions=None,
        batch_pad=None,
    ) -> RBLNFlashAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        query_max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_tables_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        query_seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        num_computed_tokens = seq_lens - query_seq_lens

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        seq_idx = positions[:num_reqs].view(-1, 1)

        # The length of the partition equals the block size.
        partition_len = self.block_size
        # no. of block(HW constraint) determines max sequence length.
        # max_model_len(Model constraint) determines max sequence length.
        # One of them is selected for max_seq_len.
        block_length = self.cache_config.num_gpu_blocks * partition_len
        max_seq_len = min(self.model_config.max_model_len, block_length)

        num_partition = max_seq_len // partition_len
        cs = seq_idx.repeat(1, num_partition)
        pidx = torch.arange(num_partition, dtype=torch.int32)
        # RBLN - seq_lens tensor dtype SHOULD be int16
        dyn_size_for_partitions = torch.clamp(
            cs - pidx * partition_len, 0, partition_len
        ).to(torch.int16)
        seq_lens_tensor = dyn_size_for_partitions

        assert num_tokens is not None, (
            "num_tokens is required for RBLN Attention Backend"
        )
        assert batch_pad is not None, "batch_pad is required for RBLN Attention Backend"
        is_prefills = num_computed_tokens[:num_reqs].numpy() < num_tokens[:num_reqs] - 1
        # The prefill and decode cannot be mixed.
        assert len(is_prefills) > 0 and all(
            is_prefill == is_prefills[0] for is_prefill in is_prefills[:num_reqs]
        )

        attn_masks = None
        if is_prefills[0]:
            # NOTE(jiwoo.park) prefill's block_tables must be a 1D tensor.
            block_tables_tensor = block_tables_tensor[0]
            if not self.is_causal:
                prefill_chunk_size = (
                    self.chunked_prefill_size
                    if self.chunked_prefill
                    else 1 << (math.ceil(math.log2(query_max_seq_len)))
                )
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
            # batch padding
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

        cache_seq_lens, cache_offsets, local_block_tables = None, None, None
        if sliding_window := getattr(self.kv_cache_spec, "sliding_window", None):
            num_computed_tokens = (
                num_computed_tokens[:num_reqs].view(-1, 1).to(torch.int16)
            )
            seq_lens = seq_lens[:num_reqs].view(-1, 1).to(torch.int16)
            query_lens = seq_lens - num_computed_tokens
            cache_seq_lens = torch.clamp(num_computed_tokens, max=sliding_window)
            cache_offsets = cache_seq_lens + query_lens
            if not is_prefills[0]:
                cache_seq_lens = rbln_utils.pad(cache_seq_lens, 0, batch_pad)
                cache_offsets = rbln_utils.pad(cache_offsets, 0, batch_pad)
            local_block_tables = block_tables_tensor[..., :1]

        # * seq_idx(batch attention opt decode) - [B, 1],
        #   for each batch, have sequence offset
        # * seq_lens_tensor(otherwise)      - [B, P],
        #   have dynamic size for each partition
        attn_metadata = RBLNFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=query_max_seq_len,
            seq_lens=seq_lens_tensor.to(self.device)
            if not self.is_batch_attention_opt or is_prefills[0]
            else seq_idx.to(self.device),
            block_tables=block_tables_tensor.to(self.device),
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=None,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
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
        # unused?
        self.need_mask = (
            self.alibi_slopes is not None or self.sliding_window is not None
        )

        supported_head_sizes = RBLNAttentionBackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                "Torch SDPA backend does not support FP8 KV cache. "
                "Please use xFormers backend instead."
            )
        self.attn_type = attn_type

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
        self.is_normal = self.block_size == self.max_model_len

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query:  shape = [num_tokens, num_heads * head_size]
            key:    shape = [num_tokens, num_kv_heads * head_size]
            value:  shape = [num_tokens, num_kv_heads * head_size]
            kv_cache shape= [2, num_blocks,
                                block_size * num_kv_heads * head_size]

        Shape that we expect:
            kv_cache  = [2][num_blocks, num_kv_heads, 1, block_size, head_size]
            key       = [1, num_kv_heads, 1, block_size, head_size]
            query     = [1, num_kv_heads, 4, query_len, head_size]
            key_t     = [1, num_kv_heads, 1, head_size, block_size]
        Returns:
            attn_out  = [num_tokens, num_heads * head_size]

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

        # 1. query reshape for custom operation
        # query = [b_size(batch), q_len(query len), num_heads * head_size]
        b_size, q_len, _ = query.size()
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
            assert self.sliding_window == kv_cache.size(-2), (
                "SWA kernel_block_size must match window_size"
            )
            assert attn_metadata.cache_seq_lens is not None
            assert attn_metadata.cache_offsets is not None
            if envs.VLLM_RBLN_COMPILE_MODEL:
                if envs.VLLM_RBLN_KERNEL_MODE == "torch_triton":
                    sliding_window_attention_naive_prefill = (
                        torch.ops.rbln_triton_ops.sliding_window_attention_naive_prefill
                    )
                    sliding_window_attention_naive_decode = (
                        torch.ops.rbln_triton_ops.sliding_window_attention_naive_decode
                    )
                elif (
                    envs.VLLM_RBLN_KERNEL_MODE == "triton"
                    or envs.VLLM_RBLN_KERNEL_MODE == "str"
                ):
                    sliding_window_attention_naive_prefill = (
                        torch.ops.rbln_custom_ops.sliding_window_attention_naive_prefill
                    )
                    sliding_window_attention_naive_decode = (
                        torch.ops.rbln_custom_ops.sliding_window_attention_naive_decode
                    )
                else:
                    raise ValueError(
                        f"Invalid VLLM_RBLN_KERNEL_MODE: {envs.VLLM_RBLN_KERNEL_MODE}"
                    )
            else:
                sliding_window_attention_naive_prefill = (
                    sliding_window_attention_naive_prefill_impl
                )
                sliding_window_attention_naive_decode = (
                    sliding_window_attention_naive_decode_impl
                )

            if q_len == 1:
                attn_output = sliding_window_attention_naive_decode(  # noqa: E501
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    self.scale,  # dummy
                    self.sinks,
                )
            else:
                attn_output = sliding_window_attention_naive_prefill(  # noqa: E501
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata.cache_seq_lens,
                    attn_metadata.cache_offsets,
                    self.scale,
                    attn_metadata.local_block_tables,
                    self.scale,  # dummy
                    self.sinks,
                )
        # actually non-flash paged attention DOES NOT use slot_mapping
        elif self.is_causal:
            if self.is_normal:
                assert attn_metadata.seq_lens is not None
                assert attn_metadata.block_tables is not None

                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_KERNEL_MODE == "torch_triton":
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_triton_ops.causal_attention_naive_decode
                        )
                    elif (
                        envs.VLLM_RBLN_KERNEL_MODE == "triton"
                        or envs.VLLM_RBLN_KERNEL_MODE == "str"
                    ):
                        causal_attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_prefill
                        )
                        causal_attention_naive_decode = (
                            torch.ops.rbln_custom_ops.causal_attention_naive_decode
                        )
                    else:
                        raise ValueError(
                            f"Invalid VLLM_RBLN_KERNEL_MODE: "
                            f"{envs.VLLM_RBLN_KERNEL_MODE}"
                        )

                if q_len == 1:
                    attn_output = causal_attention_naive_decode(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    )
                else:
                    attn_output = causal_attention_naive_prefill(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    )
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_KERNEL_MODE == "torch_triton":
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode
                        )
                    elif (
                        envs.VLLM_RBLN_KERNEL_MODE == "triton"
                        or envs.VLLM_RBLN_KERNEL_MODE == "str"
                    ):
                        flash_causal_attention_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill
                        )
                        flash_causal_attention_naive_decode = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode
                        )
                    else:
                        raise ValueError(
                            f"Invalid VLLM_RBLN_KERNEL_MODE: "
                            f"{envs.VLLM_RBLN_KERNEL_MODE}"
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
                if q_len == 1:
                    attn_output = flash_causal_attention_naive_decode(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
                    )
                else:
                    attn_output = flash_causal_attention_naive_prefill(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
                    )
        else:
            if self.is_normal:
                assert attn_metadata.attn_masks is not None
                assert attn_metadata.seq_lens is not None
                assert attn_metadata.block_tables is not None

                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_KERNEL_MODE == "torch_triton":
                        attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.attention_naive_prefill
                        )
                        attention_naive_decode = (
                            torch.ops.rbln_triton_ops.attention_naive_decode
                        )
                    elif (
                        envs.VLLM_RBLN_KERNEL_MODE == "triton"
                        or envs.VLLM_RBLN_KERNEL_MODE == "str"
                    ):
                        attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.attention_naive_prefill
                        )
                        attention_naive_decode = (
                            torch.ops.rbln_custom_ops.attention_naive_decode
                        )
                    else:
                        raise ValueError(
                            f"Invalid VLLM_RBLN_KERNEL_MODE: "
                            f"{envs.VLLM_RBLN_KERNEL_MODE}"
                        )

                if q_len == 1:
                    attn_output = attention_naive_decode(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    )
                else:
                    attn_output = attention_naive_prefill(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        attn_metadata.seq_lens.to(torch.int16),
                        self.scale,
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy (required by rbln_triton_ops signature)
                    )
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_KERNEL_MODE == "torch_triton":
                        flash_attention_naive_prefill = (
                            torch.ops.rbln_triton_ops.flash_attention_naive_prefill
                        )
                        flash_attention_naive_decode = (
                            torch.ops.rbln_triton_ops.flash_attention_naive_decode
                        )
                    elif (
                        envs.VLLM_RBLN_KERNEL_MODE == "triton"
                        or envs.VLLM_RBLN_KERNEL_MODE == "str"
                    ):
                        flash_attention_naive_prefill = (
                            torch.ops.rbln_custom_ops.flash_attention_naive_prefill
                        )
                        flash_attention_naive_decode = (
                            torch.ops.rbln_custom_ops.flash_attention_naive_decode
                        )
                    else:
                        raise ValueError(
                            f"Invalid VLLM_RBLN_KERNEL_MODE: "
                            f"{envs.VLLM_RBLN_KERNEL_MODE}"
                        )
                else:
                    flash_attention_naive_prefill = flash_attention_naive_prefill_impl
                    flash_attention_naive_decode = flash_attention_naive_decode_impl

                if q_len == 1:
                    attn_output = flash_attention_naive_decode(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
                    )
                else:
                    attn_output = flash_attention_naive_prefill(  # noqa: E501
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata.attn_masks,
                        self.scale,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,  # dummy
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
