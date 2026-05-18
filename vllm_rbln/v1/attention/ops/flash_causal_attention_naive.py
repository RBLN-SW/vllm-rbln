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

import torch

from vllm_rbln import envs

from . import triton_flash_causal_attention  # noqa: F401


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


def flash_causal_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
            )

    raise NotImplementedError


def flash_causal_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
            )

    raise NotImplementedError
