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
from vllm_rbln import utils as rbln_utils

from ..ops import triton_sliding_window_attention_naive  # noqa: F401


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
    attn_mask: torch.Tensor | None = None,
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
    attn_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


def sliding_window_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.sliding_window_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.sliding_window_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
                sinks,
            )

    raise NotImplementedError


def sliding_window_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.sliding_window_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.sliding_window_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
                attn_mask,
                sinks,
            )

    raise NotImplementedError
