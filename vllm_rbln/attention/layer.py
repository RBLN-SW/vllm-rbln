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


import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    validate_kv_sharing_target,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

# @FIXME(RBLN): We hope to remove the Custom Attention forward.
# The original vLLM forward function will be used in the future.


def __custom_init__(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int | None = None,
    alibi_slopes: list[float] | None = None,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    logits_soft_cap: float | None = None,
    per_layer_sliding_window: int | None = None,
    prefix: str = "",
    attn_type: str = AttentionType.DECODER,
    kv_sharing_target_layer_name: str | None = None,
    attn_backend: type[AttentionBackend] | None = None,
    **extra_impl_args,
) -> None:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.
    """
    nn.Module.__init__(self)

    if per_layer_sliding_window is not None:
        # per-layer sliding window
        sliding_window = per_layer_sliding_window
    elif cache_config is not None:
        # model-level sliding window
        sliding_window = cache_config.sliding_window
    else:
        sliding_window = None

    vllm_config = get_current_vllm_config()
    if cache_config is not None:
        kv_cache_dtype = cache_config.cache_dtype
        calculate_kv_scales = cache_config.calculate_kv_scales
    else:
        kv_cache_dtype = "auto"
        calculate_kv_scales = False
    self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        kv_cache_dtype, vllm_config.model_config
    )
    self.kv_cache_dtype = kv_cache_dtype
    self.calculate_kv_scales = calculate_kv_scales
    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
    )

    # Initialize KV cache quantization attributes
    _init_kv_cache_quant(self, quant_config, prefix)

    self.quant_config = quant_config

    self.num_heads = num_heads
    self.head_size = head_size
    self.num_kv_heads = num_kv_heads
    self.sliding_window = sliding_window
    self.has_sink = extra_impl_args.get("sinks") is not None

    # During model initialization, the default dtype is set as the model
    # weight and activation dtype.
    dtype = torch.get_default_dtype()
    if attn_backend is None:
        self.attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            use_mla=False,
            has_sink=self.has_sink,
            attn_type=attn_type,
        )
    else:
        self.attn_backend = attn_backend

    impl_cls = self.attn_backend.get_impl_cls()
    self.impl = impl_cls(
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        **extra_impl_args,
    )
    backend_name = self.attn_backend.get_name()
    self.backend = AttentionBackendEnum.__members__.get(backend_name)
    self.dtype = dtype

    # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
    # torch.compile works by registering the attention as one giant
    # opaque custom op. For other platforms, we directly call them
    # and let torch.compile handle them.
    self.use_direct_call = not current_platform.opaque_attention_op()

    self.use_output = self.attn_backend.accept_output_buffer
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = self
    self.layer_name = prefix
    self.attn_type = attn_type

    if kv_sharing_target_layer_name is not None:
        validate_kv_sharing_target(
            prefix,
            kv_sharing_target_layer_name,
            compilation_config.static_forward_context,
        )
    self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    # use a placeholder kv cache tensor during init, which will be replaced
    # by bind_kv_cache
    # this variable will not be accessed if use_direct_call is True
    self.kv_cache = [
        torch.tensor([])
        for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
    ]

    # Initialize q/k/v range constants.
    self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
    self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    # for attn backends supporting query quantization
    self.query_quant = None
    if self.impl.supports_quant_query_input and self.kv_cache_dtype.startswith("fp8"):
        self.query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    # NOTE(jiwoo.park) layer index is required to use external binding KV cache.
    self.layer_index = extract_layer_index(self.layer_name)

    # NOTE - consider PP
    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    start, end = model_config.get_layers_start_end_indices(parallel_config)
    # assert self.layer_index >= start and self.layer_index < end
    self.layer_index -= start


def custom_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    # For some alternate attention backends like MLA the attention output
    # shape does not match the query shape, so we optionally let the model
    # definition specify the output tensor shape.
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.

    Attention metadata (`attn_metadata`) is set using a context manager in
    the model runner's `execute_model` method. It is accessed via forward
    context using
    `vllm.forward_context.get_forward_context().attn_metadata`.
    """
    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
    output_dtype = query.dtype
    if self.query_quant is not None:
        # quantizing with a simple torch operation enables
        # torch.compile to fuse this into previous ops
        # which reduces overheads during decoding.
        # Otherwise queries are quantized using custom ops
        # which causes decoding overheads
        assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

        # check if query quantization is supported
        if self.impl.supports_quant_query_input():
            query, _ = self.query_quant(query, self._q_scale)

    if self.use_output:
        output_shape = output_shape if output_shape is not None else query.shape
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size)
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            """
            NOTE(jiwoo.park) - To represent kv cache as model input,
            modify attention
            instead of attention layer's embedded kv cache(self.kv_cache),
            use attention metadata's kv cache.
            attention metadata's kv cache must equal
            the attention layer's embedded kv cache.
            """
            assert attn_metadata.kv_caches is not None
            assert self.layer_index < len(attn_metadata.kv_caches)
            self_kv_cache = attn_metadata.kv_caches[self.layer_index]
            self.impl.forward(
                self,
                query,
                key,
                value,
                self_kv_cache,
                attn_metadata,
                output=output,
            )
        else:
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, self.layer_name
            )
        return output.view(-1, hidden_size)
    else:
        if self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            """
            NOTE(jiwoo.park) - To represent kv cache as model input,
            modify attention
            instead of attention layer's embedded kv cache(self.kv_cache),
            use attention metadata's kv cache.
            attention metadata's kv cache must equal
            the attention layer's embedded kv cache.
            """
            assert attn_metadata.kv_caches is not None
            assert self.layer_index < len(attn_metadata.kv_caches)
            self_kv_cache = attn_metadata.kv_caches[self.layer_index]
            return self.impl.forward(
                self, query, key, value, self_kv_cache, attn_metadata
            )
        else:
            return torch.ops.vllm.unified_attention(query, key, value, self.layer_name)


def custom_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    # Block size may get updated after model loading, refresh it
    block_size = vllm_config.cache_config.block_size
    # Should not be called for enc-dec or encoder-only attention.
    assert self.attn_type == AttentionType.DECODER
    if self.sliding_window is not None:
        assert not vllm_config.model_config.use_mla, (
            "MLA is not supported for slidingwindow"
        )
        return RBLNSlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=self.sliding_window,
        )
    else:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


Attention.__init__ = __custom_init__
Attention.forward = custom_attention_forward
Attention.get_kv_cache_spec = custom_get_kv_cache_spec
