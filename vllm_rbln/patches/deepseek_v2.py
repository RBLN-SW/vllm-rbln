# Copyright 2026 Rebellions Inc. All rights reserved.
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
from rebel.ops.torch_custom_ops import attn as _rbln_attn_ops  # noqa: F401
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MoE,
    DeepseekV32IndexerCache,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.patches.models_utils import (
    rbln_num_attn_module,
    rbln_pipeline_adjusted_layer_index,
)
from vllm_rbln.v1.attention.backends.mla.indexer import (
    RBLNDeepseekV32IndexerBackend,
    RBLNDeepseekV32IndexerScaleBackend,
)

logger = init_logger(__name__)
_original_indexer_cache_init = DeepseekV32IndexerCache.__init__


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV2MoE.forward",
    reason=(
        "Replace DeepseekV2MoE.forward with an RBLN-friendly form: call the "
        "RBLNMoERunner with a `router` callback and keep 3-D tensors (no "
        "reshape)."
    ),
)
def patched_deepseek_v2_moe_forward(
    self: DeepseekV2MoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    # Fix FP16 overflow
    # See DeepseekV2DecoderLayer for more details.
    if hidden_states.dtype != torch.float16:
        final_hidden_states *= self.routed_scaling_factor

    shared_output = None
    if self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
        if hidden_states.dtype == torch.float16:
            shared_output *= 1.0 / self.routed_scaling_factor
        if not self.is_sequence_parallel:
            final_hidden_states = final_hidden_states + shared_output
            shared_output = None

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output
    # FIXME(RBLN) - DO NOT reshape
    # return final_hidden_states.view(orig_shape)
    return final_hidden_states


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV2Attention.forward",
    reason=(
        "RBLN non-MLA (use_mla=False) fallback path: materialize K/V per head "
        "and pad to qk_head_dim so the regular RBLN attention backend can run "
        "DeepSeek attention. Unused when use_mla=True (DeepseekV2MLAAttention)."
    ),
)
def patched_deepseek_v2_attention_forward(
    self: DeepseekV2Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, _, _ = hidden_states.shape
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    else:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
    kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_a = self.kv_a_layernorm(kv_a.contiguous())
    kv = self.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    if q_nope.dim() != q_pe.dim():
        q_pe = q_pe.squeeze(0)
    if k_nope.dim() != k_pe.dim():
        k_pe = k_pe.squeeze(0)

    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.repeat(1, self.num_local_heads, 1)], dim=-1)
    # padding value to qk_head_dim for alignment
    if self.qk_head_dim != self.v_head_dim:
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
    q = q.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    k = k.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    v = v.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    attn_output = self.attn(q, k, v)
    if self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(batch, -1, self.num_local_heads * self.v_head_dim)

    output, _ = self.o_proj(attn_output)
    return output


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache.__init__",
    reason=(
        "Give the indexer KV cache the RBLN head_dim/dtype "
        "and a layer_index that accounts for the extra indexer module per decoder "
        "layer (num_attn_module=2), so the runner binds a distinct cache for it."
    ),
)
def rbln_indexer_cache_init(self, *args, **kwargs) -> None:
    _original_indexer_cache_init(self, *args, **kwargs)
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    self.head_dim = vllm_config.model_config.hf_text_config.index_head_dim
    cache_dtype = vllm_config.cache_config.cache_dtype
    self.is_fp8_cache = bool(cache_dtype) and cache_dtype.startswith("fp8")
    self.dtype = torch.float8_e4m3fn if self.is_fp8_cache else torch.bfloat16

    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(model_config, cache_dtype)
    self.layer_index = rbln_pipeline_adjusted_layer_index(
        self.prefix, model_config, vllm_config.parallel_config, num_attn_module
    )

    # register fp16 scale cache (2-cache spec).
    self.scale_cache = None
    if self.is_fp8_cache:
        scale = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        _original_indexer_cache_init(
            scale,
            head_dim=1,
            dtype=torch.float16,
            prefix=f"{self.prefix}.scale_cache",
            cache_config=self.cache_config,
        )
        scale.get_attn_backend = lambda: RBLNDeepseekV32IndexerScaleBackend
        scale.layer_index = rbln_pipeline_adjusted_layer_index(
            scale.prefix, model_config, vllm_config.parallel_config, num_attn_module
        )
        self.scale_cache = scale


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache.get_attn_backend",
    reason="Use in the RBLN indexer backend.",
)
def rbln_indexer_cache_get_attn_backend(self):
    return RBLNDeepseekV32IndexerBackend


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.Indexer.forward",
    reason=("Run the lightning indexer through the RBLN. "),
)
def rbln_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape

    # q [B, S, n_head, head_dim]
    q, _ = self.wq_b(qr)
    q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    # k [B, S, head_dim]
    kw, _ = self.wk_weights_proj(hidden_states)
    k = kw[..., : self.head_dim]
    weights = kw[..., self.head_dim :] * (self.n_head**-0.5)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(
        k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    # q,k rope
    k_pe = k_pe.unsqueeze(2)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
    k_pe = k_pe.squeeze(2)

    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe, k_nope], dim=-1)  # [B, S, head_dim]

    q_indexer = q.transpose(1, 2).contiguous()
    k_indexer_cur = k.contiguous()

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[self.k_cache.prefix]
    k_cache = _resolve_kv_cache(attn_metadata, self.k_cache.layer_index)

    scale_cache = None
    if getattr(self.k_cache, "scale_cache", None) is not None:
        scale_cache = _resolve_kv_cache(
            attn_metadata, self.k_cache.scale_cache.layer_index
        ).squeeze(-1)  # [num_block, ps, 1] -> [num_block, ps]

    weights = weights.contiguous()  # [B, T, n_head]
    softmax_scale = torch.tensor(
        self.softmax_scale, dtype=torch.float32, device=q_indexer.device
    )
    topk_index = torch.ops.rbln_custom_ops.sparse_attn_deepseek_indexer(
        q_indexer,
        k_indexer_cur,
        k_cache,
        softmax_scale,
        weights,
        attn_metadata.seq_lens.to(torch.int32),
        attn_metadata.block_tables,
        self.topk_tokens,
        scale_cache,
    )
    return topk_index
