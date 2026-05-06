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
def attention_naive_prefill(
    query_base,
    key_base,
    value_base,
    kv_cache_base,
    attn_mask_base,
    output_base,
    seq_idx_base,
    qk_scale,
    block_table_base,
    dummy,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,  # 128(prefill) or 1(decode)
    PARTITION_SIZE: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    NUM_PARTITION: tl.constexpr = 1
    DYNAMIC_AXIS: tl.constexpr = 4

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        # -- get physical block index from block table --
        block_table_ptr = tl.make_block_ptr(
            base=block_table_base,
            shape=(NUM_PARTITION,),
            strides=(1,),
            offsets=(batch_id,),
            block_shape=(1,),
            order=(0,),
        )
        tl.static_assert(len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)
        # seq_idx = {batch, num_partition}
        seq_idx_ptr = tl.make_block_ptr(
            base=seq_idx_base,
            shape=(NUM_BATCH, PARTITION_SIZE // PARTITION_SIZE),
            strides=(PARTITION_SIZE // PARTITION_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        block_number = rblib.to_dynamic_index(block_table_ptr)
        block_offset = rblib.to_dynamic_index(seq_idx_ptr)
        block_number = block_number.cast(tl.int32)
        block_offset = block_offset.cast(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=key_base,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * 1 * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=value_base,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * 1 * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        k_cache_base_ptr = tl.make_block_ptr(
            base=kv_cache_base,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            strides=(
                2 * NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                1 * PARTITION_SIZE * HEAD_DIM,
                PARTITION_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(0, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        v_cache_base_ptr = tl.make_block_ptr(
            base=kv_cache_base,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            strides=(
                2 * NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                1 * PARTITION_SIZE * HEAD_DIM,
                PARTITION_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(1, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        query_ptr = tl.make_block_ptr(
            base=query_base,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * NUM_GROUP * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        output_ptr = tl.make_block_ptr(
            base=output_base,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * NUM_GROUP * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        attn_mask_ptr = tl.make_block_ptr(
            base=attn_mask_base,
            shape=(NUM_BATCH, 1, 1, QUERY_LEN, PARTITION_SIZE),
            strides=(
                1 * 1 * QUERY_LEN * PARTITION_SIZE,
                1 * QUERY_LEN * PARTITION_SIZE,
                QUERY_LEN * PARTITION_SIZE,
                PARTITION_SIZE,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, 1, 1, QUERY_LEN, PARTITION_SIZE),
            order=(4, 3, 2, 1, 0),
        )

        k_state = tl.load(k_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        k_base = rblib.dynamic_load(k_cache_base_ptr, DYNAMIC_AXIS, block_offset)
        k_insert = rblib.insert(k_base, k_state, DYNAMIC_AXIS, block_offset)
        rblib.dynamic_store(
            k_cache_base_ptr, k_insert, DYNAMIC_AXIS, block_offset + QUERY_LEN
        )

        q = tl.load(query_ptr)
        # to fuse transpose, broadcast ops into matmul op, make sure the sequence
        # to be transpose - broadcast - matmul. if the sequence is broadcast -
        # transpose - matmul, it may not be fused (NYI)
        k_insert = tl.reshape(k_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM))
        k = tl.permute(k_insert, (0, 1, 2, 4, 3))
        k = tl.broadcast_to(k, (1, NUM_HEAD, NUM_GROUP, HEAD_DIM, PARTITION_SIZE))
        # 2.1 a = MM(Q, Kt)
        qk = tl.dot(q, k)  # (1,h,g,l,d) x (1,h,g,d,p) = (1,h,g,l,p)
        # 2.2 b = a * qk_scale
        qk_scaled = qk * qk_scale  # (1,h,g,l,p)
        attn_mask = tl.load(attn_mask_ptr)  # (1,1,g,l,p)
        # 2.3 c = b + attention_mask
        # 2.4 d = softmax(c)
        softmax_masked_qk_scaled = rblib.masked_softmax(qk_scaled, attn_mask)

        v_state = tl.load(v_block_ptr)
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v_base = rblib.dynamic_load(v_cache_base_ptr, DYNAMIC_AXIS, block_offset)
        v_insert = rblib.insert(v_base, v_state, DYNAMIC_AXIS, block_offset)
        rblib.dynamic_store(
            v_cache_base_ptr, v_insert, DYNAMIC_AXIS, block_offset + QUERY_LEN
        )

        v = tl.reshape(v_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM))
        v = tl.broadcast_to(v, (1, NUM_HEAD, NUM_GROUP, PARTITION_SIZE, HEAD_DIM))
        # 2.5 O = MM(d, V)
        attn_out = tl.dot(
            softmax_masked_qk_scaled, v
        )  # (1,h,g,l,p) x (1,h,g,p,d) = (1,h,g,l,d)
        tl.store(output_ptr, attn_out)  # (1,h,g,l,d)


@triton.jit
def attention_naive_decode(
    query_base,
    key_base,
    value_base,
    kv_cache_base,
    attn_mask_base,
    output_base,
    seq_idx_base,
    qk_scale,
    block_table_base,
    dummy,
    NUM_HEAD: tl.constexpr,
    NUM_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_BATCH: tl.constexpr,
    DIM_BLOCK_TABLE: tl.constexpr,
):
    NUM_PARTITION: tl.constexpr = 1
    DYNAMIC_AXIS: tl.constexpr = 4

    for batch_id in tl.static_range(0, NUM_BATCH, 1):
        # -- get physical block index from block table --
        block_table_ptr = tl.make_block_ptr(
            base=block_table_base,
            shape=(NUM_BATCH, NUM_PARTITION),
            strides=(NUM_PARTITION, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )
        tl.static_assert(len(block_table_ptr.type.element_ty.shape) == DIM_BLOCK_TABLE)
        # seq_idx = {batch, num_partition}
        seq_idx_ptr = tl.make_block_ptr(
            base=seq_idx_base,
            shape=(NUM_BATCH, PARTITION_SIZE // PARTITION_SIZE),
            strides=(PARTITION_SIZE // PARTITION_SIZE, 1),
            offsets=(batch_id, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        block_number = rblib.to_dynamic_index(block_table_ptr)
        block_offset = rblib.to_dynamic_index(seq_idx_ptr)
        block_number = block_number.cast(tl.int32)
        block_offset = block_offset.cast(tl.int32)

        k_block_ptr = tl.make_block_ptr(
            base=key_base,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * 1 * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=value_base,
            shape=(NUM_BATCH, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * 1 * QUERY_LEN * HEAD_DIM,
                1 * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        k_cache_base_ptr = tl.make_block_ptr(
            base=kv_cache_base,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            strides=(
                2 * NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                1 * PARTITION_SIZE * HEAD_DIM,
                PARTITION_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(0, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        v_cache_base_ptr = tl.make_block_ptr(
            base=kv_cache_base,
            shape=(2, NUM_BATCH, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            strides=(
                2 * NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                NUM_BATCH * NUM_HEAD * 1 * PARTITION_SIZE * HEAD_DIM,
                1 * PARTITION_SIZE * HEAD_DIM,
                PARTITION_SIZE * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(1, block_number, 0, 0, 0, 0),
            block_shape=(1, 1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM),
            order=(5, 4, 3, 2, 1, 0),
        )
        query_ptr = tl.make_block_ptr(
            base=query_base,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * NUM_GROUP * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        output_ptr = tl.make_block_ptr(
            base=output_base,
            shape=(NUM_BATCH, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            strides=(
                NUM_HEAD * NUM_GROUP * QUERY_LEN * HEAD_DIM,
                NUM_GROUP * QUERY_LEN * HEAD_DIM,
                QUERY_LEN * HEAD_DIM,
                HEAD_DIM,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, NUM_HEAD, NUM_GROUP, QUERY_LEN, HEAD_DIM),
            order=(4, 3, 2, 1, 0),
        )
        attn_mask_ptr = tl.make_block_ptr(
            base=attn_mask_base,
            shape=(NUM_BATCH, 1, 1, QUERY_LEN, PARTITION_SIZE),
            strides=(
                1 * 1 * QUERY_LEN * PARTITION_SIZE,
                1 * QUERY_LEN * PARTITION_SIZE,
                QUERY_LEN * PARTITION_SIZE,
                PARTITION_SIZE,
                1,
            ),
            offsets=(batch_id, 0, 0, 0, 0),
            block_shape=(1, 1, 1, QUERY_LEN, PARTITION_SIZE),
            order=(4, 3, 2, 1, 0),
        )

        k_state = tl.load(k_block_ptr)
        k_state = tl.reshape(k_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        k_base = rblib.dynamic_load(k_cache_base_ptr, DYNAMIC_AXIS, block_offset)
        k_insert = rblib.insert(k_base, k_state, DYNAMIC_AXIS, block_offset)
        rblib.dynamic_store(
            k_cache_base_ptr, k_insert, DYNAMIC_AXIS, block_offset + QUERY_LEN
        )

        q = tl.load(query_ptr)
        # to fuse transpose, broadcast ops into matmul op, make sure the sequence
        # to be transpose - broadcast - matmul. if the sequence is broadcast -
        # transpose - matmul, it may not be fused (NYI)
        k_insert = tl.reshape(k_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM))
        k = tl.permute(k_insert, (0, 1, 2, 4, 3))
        k = tl.broadcast_to(k, (1, NUM_HEAD, NUM_GROUP, HEAD_DIM, PARTITION_SIZE))
        # 2.1 a = MM(Q, Kt)
        qk = tl.dot(q, k)  # (1,h,g,l,d) x (1,h,g,d,p) = (1,h,g,l,p)
        # 2.2 b = a * qk_scale
        qk_scaled = qk * qk_scale
        attn_mask = tl.load(attn_mask_ptr)
        # 2.3 c = b + attention_mask
        # 2.4 d = softmax(c)
        softmax_masked_qk_scaled = rblib.masked_softmax(qk_scaled, attn_mask)

        v_state = tl.load(v_block_ptr)
        v_state = tl.reshape(v_state, (1, 1, NUM_HEAD, 1, QUERY_LEN, HEAD_DIM))
        v_base = rblib.dynamic_load(v_cache_base_ptr, DYNAMIC_AXIS, block_offset)
        v_insert = rblib.insert(v_base, v_state, DYNAMIC_AXIS, block_offset)
        rblib.dynamic_store(
            v_cache_base_ptr, v_insert, DYNAMIC_AXIS, block_offset + QUERY_LEN
        )

        v = tl.reshape(v_insert, (1, NUM_HEAD, 1, PARTITION_SIZE, HEAD_DIM))
        v = tl.broadcast_to(v, (1, NUM_HEAD, NUM_GROUP, PARTITION_SIZE, HEAD_DIM))
        # 2.5 O = MM(d, V)
        attn_out = tl.dot(
            softmax_masked_qk_scaled, v
        )  # (1,h,g,l,p) x (1,h,g,p,d) = (1,h,g,l,d)
        tl.store(output_ptr, attn_out)  # (1,h,4,l,d)


def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,), host_layout="1:2:3")
    rblib.write_rtosa(kernel, args)

    return kernel


@triton_op("rbln_triton_ops::attention_naive_prefill", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    seq_idx: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    mask = mask.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
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
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        mask,
        kv_cache,
        output,
        seq_idx,
        qk_scale,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        PARTITION_SIZE,
        NUM_BATCH,
        DIM_BLOCK_TABLE,
    ]
    warmup(attention_naive_prefill, *params)

    return output.to(original_dtype)


@triton_op("rbln_triton_ops::attention_naive_decode", mutates_args=())
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    seq_idx: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    original_dtype = query.dtype

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    mask = mask.to(torch.float32)
    kv_cache = kv_cache.to(torch.float32)
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
    NUM_BATCH = query.shape[0]
    DIM_BLOCK_TABLE = block_table.dim()

    params = [
        query,
        key,
        value,
        mask,
        kv_cache,
        output,
        seq_idx,
        qk_scale,
        block_table,
        qk_scale,
        NUM_HEAD,
        NUM_GROUP,
        HEAD_DIM,
        QUERY_LEN,
        PARTITION_SIZE,
        NUM_BATCH,
        DIM_BLOCK_TABLE,
    ]

    warmup(attention_naive_decode, *params)

    return output.to(original_dtype)


@register_fake("rbln_triton_ops::attention_naive_prefill")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    seq_idx: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)


@register_fake("rbln_triton_ops::attention_naive_decode")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: torch.Tensor,
    seq_idx: torch.Tensor,
    qk_scale: torch.Tensor,
    block_table: torch.Tensor,
    dummy: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(query)
