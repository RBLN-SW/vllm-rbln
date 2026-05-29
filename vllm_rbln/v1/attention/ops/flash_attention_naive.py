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

from . import triton_flash_attention_naive  # noqa: F401


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


def flash_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                mask,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.flash_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                mask,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
                sinks,
            )

    raise NotImplementedError


def flash_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                mask,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.flash_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                mask,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
                sinks,
            )

    raise NotImplementedError
