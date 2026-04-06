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
"""Triton kernels for Attention"""

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


@triton.jit
def flash_attention_naive_prefill(
    query,
    key,
    value,
    kv_cache,
    mask,
    output,
    qk_scale,
    seq_idx,
    block_table,
    block_size,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    NB: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    NP: tl.constexpr = C // P
    for batch_id in tl.static_range(0, NB, 1):
        Q_block_ptr = tl.make_block_ptr(
            base=query,
            shape=(NB, H, G, L, D),
            strides=(H * G * L * D, G * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, G, L, D),
            order=(4, 3, 2, 1, 0),
        )
        M_block_ptr = tl.make_block_ptr(
            base=mask,
            shape=(NB, 1, 1, L, C),
            strides=(L * C, L * C, L * C, C, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, 1, 1, L, P),
            order=(4, 3, 2, 1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=output,
            shape=(NB, H, G, L, D),
            strides=(H * G * L * D, G * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, G, L, D),
            order=(4, 3, 2, 1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            base=key,
            shape=(NB, H, 1, L, D),
            strides=(H * 1 * L * D, 1 * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, 1, L, D),
            order=(4, 3, 2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=value,
            shape=(NB, H, 1, L, D),
            strides=(H * 1 * L * D, 1 * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, 1, L, D),
            order=(4, 3, 2, 1, 0),
        )
        q = tl.load(Q_block_ptr)
        k_state = tl.load(k_block_ptr)
        v_state = tl.load(v_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, H, 1, L, D))
        v_state = tl.reshape(v_state, (1, 1, H, 1, L, D))

        attn_out_i = tl.zeros([1, H, G, L, D], dtype=tl.float32)
        row_sum_i = tl.zeros([1, H, G, L, 1], dtype=tl.float32)
        row_max_i = tl.zeros([1, H, G, L, 1], dtype=tl.float32)

        for partition_id in tl.static_range(0, NP, 1):
            BT_block_ptr = tl.make_block_ptr(
                base=block_table,
                shape=(NP,),
                strides=(1,),
                offsets=(partition_id,),
                block_shape=(1,),
                order=(0,),
            )
            tl.static_assert(len(BT_block_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)
            SP_block_ptr = tl.make_block_ptr(
                base=seq_idx,
                shape=(NB, NP),
                strides=(NP, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )
            block_number = rblib.to_dynamic_index(BT_block_ptr)
            block_offset = rblib.to_dynamic_index(SP_block_ptr)
            block_number = block_number.cast(tl.int32)
            block_offset = block_offset.cast(tl.int32)

            if rblib.partition_skip(block_offset) == False:  # noqa: E712
                k_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, B, H, 1, P, D),
                    strides=(B * H * 1 * P * D, H * 1 * P * D, 1 * P * D, P * D, D, 1),
                    offsets=(0, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, H, 1, P, D),
                    order=(5, 4, 3, 2, 1, 0),
                )
                v_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, B, H, 1, P, D),
                    strides=(B * H * 1 * P * D, H * 1 * P * D, 1 * P * D, P * D, D, 1),
                    offsets=(1, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, H, 1, P, D),
                    order=(5, 4, 3, 2, 1, 0),
                )

                k = rblib.dynamic_load(k_cache_ptr, 4, block_offset)
                k_insert = rblib.insert(k, k_state, 4, block_offset)
                rblib.dynamic_store(k_cache_ptr, k_insert, 4, block_offset + L)

                k_insert = tl.reshape(k_insert, (1, H, 1, P, D))
                k = tl.permute(k_insert, (0, 1, 2, 4, 3))
                k = tl.broadcast_to(k, (1, H, G, D, P))

                qk = tl.dot(q, k)
                qk_scaled = qk * qk_scale
                attn_mask = tl.load(M_block_ptr)

                v = rblib.dynamic_load(v_cache_ptr, 4, block_offset)
                v_insert = rblib.insert(v, v_state, 4, block_offset)
                rblib.dynamic_store(v_cache_ptr, v_insert, 4, block_offset + L)

                v_insert = tl.reshape(v_insert, (1, H, 1, P, D))
                v = tl.broadcast_to(v_insert, (1, H, G, P, D))
                if partition_id > 0:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.flash_attn_tile(qk_scaled, attn_mask, row_max_i)
                    )
                else:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.flash_attn_tile(qk_scaled, attn_mask)
                    )

                attn_out_cur = tl.dot(row_exp_normalize, v)
                if partition_id > 0:
                    row_sum_i, attn_out_i = rblib.flash_attn_recompute(
                        row_max_i,
                        row_max_global,
                        row_sum_i,
                        row_sum_cur,
                        attn_out_i,
                        attn_out_cur,
                    )
                else:
                    row_sum_i = row_sum_cur
                    attn_out_i = attn_out_cur
                row_max_i = row_max_global

            M_block_ptr = tl.advance(M_block_ptr, (0, 0, 0, 0, P))

        attn_out = attn_out_i / row_sum_i
        tl.store(O_block_ptr, attn_out)


@triton.jit
def flash_attention_naive_decode(
    query,
    key,
    value,
    kv_cache,
    mask,
    output,
    qk_scale,
    seq_idx,
    block_table,
    block_size,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    NB: tl.constexpr,
    P: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    NP: tl.constexpr = C // P
    for batch_id in tl.static_range(0, NB, 1):
        Q_block_ptr = tl.make_block_ptr(
            base=query,
            shape=(NB, H, G, L, D),
            strides=(H * G * L * D, G * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, G, L, D),
            order=(4, 3, 2, 1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=output,
            shape=(NB, H, G, L, D),
            strides=(H * G * L * D, G * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, G, L, D),
            order=(4, 3, 2, 1, 0),
        )
        M_block_ptr = tl.make_block_ptr(
            base=mask,
            shape=(NB, 1, 1, L, C),
            strides=(L * C, L * C, L * C, C, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, 1, 1, L, P),
            order=(4, 3, 2, 1, 0),
        )

        k_block_ptr = tl.make_block_ptr(
            base=key,
            shape=(NB, H, 1, L, D),
            strides=(H * 1 * L * D, 1 * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, 1, L, D),
            order=(4, 3, 2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=value,
            shape=(NB, H, 1, L, D),
            strides=(H * 1 * L * D, 1 * L * D, L * D, D, 1),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, H, 1, L, D),
            order=(4, 3, 2, 1, 0),
        )
        q = tl.load(Q_block_ptr)
        k_state = tl.load(k_block_ptr)
        v_state = tl.load(v_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, H, 1, L, D))
        v_state = tl.reshape(v_state, (1, 1, H, 1, L, D))

        attn_out_i = tl.zeros([1, H, G, L, D], dtype=tl.float32)
        row_sum_i = tl.zeros([1, H, G, L, 1], dtype=tl.float32)
        row_max_i = tl.zeros([1, H, G, L, 1], dtype=tl.float32)

        for partition_id in tl.static_range(0, NP, 1):
            BT_block_ptr = tl.make_block_ptr(
                base=block_table,
                shape=(NB, NP),
                strides=(NP, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )
            tl.static_assert(len(BT_block_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)
            SP_block_ptr = tl.make_block_ptr(
                base=seq_idx,
                shape=(NB, NP),
                strides=(NP, 1),
                offsets=(batch_id, partition_id),
                block_shape=(1, 1),
                order=(1, 0),
            )
            block_number = rblib.to_dynamic_index(BT_block_ptr)
            block_offset = rblib.to_dynamic_index(SP_block_ptr)
            block_number = block_number.cast(tl.int32)
            block_offset = block_offset.cast(tl.int32)

            if rblib.partition_skip(block_offset) == False:  # noqa: E712
                k_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, B, H, 1, P, D),
                    strides=(B * H * 1 * P * D, H * 1 * P * D, 1 * P * D, P * D, D, 1),
                    offsets=(0, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, H, 1, P, D),
                    order=(5, 4, 3, 2, 1, 0),
                )
                v_cache_ptr = tl.make_block_ptr(
                    base=kv_cache,
                    shape=(2, B, H, 1, P, D),
                    strides=(B * H * 1 * P * D, H * 1 * P * D, 1 * P * D, P * D, D, 1),
                    offsets=(1, block_number, 0, 0, 0, 0),
                    block_shape=(1, 1, H, 1, P, D),
                    order=(5, 4, 3, 2, 1, 0),
                )
                k = rblib.dynamic_load(k_cache_ptr, 4, block_offset)
                k_insert = rblib.insert(k, k_state, 4, block_offset)
                rblib.dynamic_store(k_cache_ptr, k_insert, 4, block_offset + L)
                k_insert = tl.reshape(k_insert, (1, H, 1, P, D))
                k = tl.permute(k_insert, (0, 1, 2, 4, 3))
                k = tl.broadcast_to(k, (1, H, G, D, P))

                qk = tl.dot(q, k)
                qk_scaled = qk * qk_scale
                attn_mask = tl.load(M_block_ptr)

                v = rblib.dynamic_load(v_cache_ptr, 4, block_offset)
                v_insert = rblib.insert(v, v_state, 4, block_offset)
                rblib.dynamic_store(v_cache_ptr, v_insert, 4, block_offset + L)

                v_insert = tl.reshape(v_insert, (1, H, 1, P, D))
                v = tl.broadcast_to(v_insert, (1, H, G, P, D))
                if partition_id > 0:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.flash_attn_tile(qk_scaled, attn_mask, row_max_i)
                    )
                else:
                    row_max_global, row_exp_normalize, row_sum_cur = (
                        rblib.flash_attn_tile(qk_scaled, attn_mask)
                    )

                attn_out_cur = tl.dot(row_exp_normalize, v)
                if partition_id > 0:
                    row_sum_i, attn_out_i = rblib.flash_attn_recompute(
                        row_max_i,
                        row_max_global,
                        row_sum_i,
                        row_sum_cur,
                        attn_out_i,
                        attn_out_cur,
                    )
                else:
                    row_sum_i = row_sum_cur
                    attn_out_i = attn_out_cur
                row_max_i = row_max_global

            M_block_ptr = tl.advance(M_block_ptr, (0, 0, 0, 0, P))

        attn_out = attn_out_i / row_sum_i
        tl.store(O_block_ptr, attn_out)


def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,), host_layout="1:2:3")
    rblib.write_rtosa(kernel, args)

    return kernel


@triton_op("rbln_triton_ops::flash_attention_naive_prefill", mutates_args=())
def flash_attention_naive_prefill_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
    mask = mask.to(torch.float32)
    qk_scale = qk_scale.to(torch.float32)

    output = torch.empty_like(query)

    query = rblib.align_tensor_last_dim_to_64(query)
    key = rblib.align_tensor_last_dim_to_64(key)
    value = rblib.align_tensor_last_dim_to_64(value)

    NUM_HEAD = query.shape[1]
    NUM_GROUP = query.shape[2]
    HEAD_DIM = query.shape[-1]
    QUERY_LEN = query.shape[-2]
    PARTITION_SIZE = kv_cache.shape[-2]
    MAX_SEQ_LEN = mask.shape[-1]
    NUM_BLOCK = kv_cache.shape[1]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        mask,
        output,
        qk_scale,
        seq_idx,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        NUM_BATCH,
        PARTITION_SIZE,
        MAX_SEQ_LEN,
        NUM_BLOCK,
        DIM_BLOCK_TABLE,
    ]
    warmup(flash_attention_naive_prefill, *params)
    return output.to(original_dtype)


@triton_op("rbln_triton_ops::flash_attention_naive_decode", mutates_args=())
def flash_attention_naive_decode_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
    mask = mask.to(torch.float32)
    qk_scale = qk_scale.to(torch.float32)

    output = torch.empty_like(query)

    query = rblib.align_tensor_last_dim_to_64(query)
    key = rblib.align_tensor_last_dim_to_64(key)
    value = rblib.align_tensor_last_dim_to_64(value)

    NUM_HEAD = query.shape[1]
    NUM_GROUP = query.shape[2]
    HEAD_DIM = query.shape[-1]
    QUERY_LEN = query.shape[-2]
    PARTITION_SIZE = kv_cache.shape[-2]
    MAX_SEQ_LEN = mask.shape[-1]
    NUM_BLOCK = kv_cache.shape[1]
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        kv_cache,
        mask,
        output,
        qk_scale,
        seq_idx,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        NUM_BATCH,
        PARTITION_SIZE,
        MAX_SEQ_LEN,
        NUM_BLOCK,
        DIM_BLOCK_TABLE,
    ]

    warmup(flash_attention_naive_decode, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::flash_attention_naive_prefill")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)


@register_fake("rbln_triton_ops::flash_attention_naive_decode")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    qk_scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_table: torch.Tensor,
    dummy0: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)
