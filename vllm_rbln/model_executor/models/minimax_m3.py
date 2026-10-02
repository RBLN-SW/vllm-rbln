# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
    set_default_torch_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.flash_attention import _fp8_cache_dtype
from vllm_rbln.v1.attention.backends.minimax_m3 import (
    MSA_SPARSE_BLOCK_SIZE,
    RBLNMiniMaxM3IndexerBackend,
    RBLNMiniMaxM3SparseBackend,
)
from vllm_rbln.v1.worker.utils import (
    num_attn_module as rbln_num_attn_module,
)
from vllm_rbln.v1.worker.utils import (
    pipeline_adjusted_layer_index,
)

logger = init_logger(__name__)


def _layer_index_of(prefix: str) -> int:
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(
        model_config, vllm_config.cache_config.cache_dtype
    )
    return pipeline_adjusted_layer_index(
        prefix, model_config, vllm_config.parallel_config, num_attn_module
    )


def _per_head_norm(
    x: torch.Tensor, norm: GemmaRMSNorm, num_heads: int, head_dim: int
) -> torch.Tensor:
    shape = x.shape
    x = x.view(*shape[:-1], num_heads, head_dim)
    return norm(x).view(shape)


class RBLNMiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (the leading dense layers and the shared expert)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        self.swiglu_alpha = float(config.swiglu_alpha)
        self.swiglu_beta = float(getattr(config, "swiglu_beta", 1.0))
        self.swiglu_limit = float(config.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        x = gate * torch.sigmoid(gate * self.swiglu_alpha) * (up + self.swiglu_beta)
        x, _ = self.down_proj(x)
        return x


class RBLNMiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE with a routing-bias correction and a shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                RBLNMiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        # Router weights are fp32 in the checkpoint; the gate is computed in fp32.
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: RBLNMiniMaxM3MLP | None = None
        if self.n_shared_experts:
            self.shared_experts = RBLNMiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=getattr(config, "swiglu_beta", 1.0),
            routed_scaling_factor=self.routed_scaling_factor,
            router_logits_dtype=self.gate.out_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router=lambda x: self.gate(x.to(torch.float32))[0],
        )
        final_hidden_states = final_hidden_states * self.routed_scaling_factor
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                hidden_states
            )
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.to(hidden_states.dtype)


class RBLNMiniMaxM3Attention(nn.Module):
    """Dense attention with per-head Gemma QK norm and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.indexer_cache = RBLNMiniMaxM3IndexerCache(
            head_dim=config.sparse_attention_config["sparse_index_dim"],
            prefix=f"{prefix}.attn.indexer",
            cache_config=cache_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = _per_head_norm(q, self.q_norm, self.num_heads, self.head_dim)
        k = _per_head_norm(k, self.k_norm, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class RBLNMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """Key-only side cache of the lightning indexer"""

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        vllm_config = get_current_vllm_config()
        cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
        self.fp8_dtype = _fp8_cache_dtype(cache_dtype)
        self.dtype = (
            kv_cache_dtype_str_to_dtype(cache_dtype, vllm_config.model_config)
            if self.fp8_dtype is not None
            else torch.bfloat16
        )
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_index = _layer_index_of(prefix)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector per token (not K + V).
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[RBLNMiniMaxM3IndexerBackend]:
        return RBLNMiniMaxM3IndexerBackend


class RBLNMiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer"""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        # index_q has one head per kv head and shards like K/V.
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.topk_blocks = int(sparse_cfg["sparse_topk_blocks"])
        if int(sparse_cfg.get("sparse_block_size", MSA_SPARSE_BLOCK_SIZE)) != (
            MSA_SPARSE_BLOCK_SIZE
        ):
            raise NotImplementedError(
                "the RBLN MSA kernels are fixed to a 128-token sparse block; got "
                f"sparse_block_size={sparse_cfg.get('sparse_block_size')}"
            )
        if int(sparse_cfg.get("sparse_init_block", 0)) != 0:
            raise NotImplementedError("sparse_init_block != 0 is not supported")
        if int(sparse_cfg.get("sparse_local_block", 1)) != 1:
            raise NotImplementedError("the kernels always attend the local block")
        if sparse_cfg.get("sparse_score_type", "max") != "max":
            raise NotImplementedError("only the block-max index score is supported")

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        # index_dim == head_dim for M3, so the index branch shares the RoPE.
        assert self.idx_head_dim == self.head_dim
        self.index_rotary_emb = self.rotary_emb

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_fp8_dtype = _fp8_cache_dtype(self.kv_cache_dtype)
        if (
            self.kv_cache_dtype not in ("auto", "bfloat16")
            and self.kv_cache_fp8_dtype is None
        ):
            raise NotImplementedError(
                "the RBLN MSA attention kernel reads a bf16 or fp8 K/V cache; got "
                f"kv_cache_dtype={self.kv_cache_dtype!r}"
            )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by the runner's bind
        self.layer_index = _layer_index_of(self.layer_name)

        self.indexer_cache = RBLNMiniMaxM3IndexerCache(
            head_dim=self.idx_head_dim,
            prefix=f"{self.layer_name}.indexer",
            cache_config=cache_config,
        )

        self.scale_tensor = torch.tensor(
            self.scaling, dtype=torch.float32, device=vllm_config.device_config.device
        )
        self.k_scale_tensor = torch.tensor(
            1.0, dtype=torch.float32, device=vllm_config.device_config.device
        )
        self.v_scale_tensor = torch.tensor(
            1.0, dtype=torch.float32, device=vllm_config.device_config.device
        )

    def get_attn_backend(self) -> type[RBLNMiniMaxM3SparseBackend]:
        return RBLNMiniMaxM3SparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        num_heads, num_kv, groups, head_dim = (
            self.num_heads,
            self.num_kv_heads,
            self.num_queries_per_kv,
            self.head_dim,
        )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, index_q, index_k = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ],
            dim=-1,
        )
        q = _per_head_norm(q, self.q_norm, num_heads, head_dim)
        k = _per_head_norm(k, self.k_norm, num_kv, head_dim)
        q, k = self.rotary_emb(positions, q, k)
        index_q = _per_head_norm(
            index_q, self.index_q_norm, self.num_idx_heads, self.idx_head_dim
        )
        index_k = self.index_k_norm(index_k)
        index_q, index_k = self.index_rotary_emb(positions, index_q, index_k)

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            main_metadata = attn_metadata[self.layer_name]
            index_metadata = attn_metadata[self.indexer_cache.prefix]
        else:
            main_metadata = index_metadata = attn_metadata
        kv_cache = _resolve_kv_cache(main_metadata, self.layer_index)
        index_cache = _resolve_kv_cache(index_metadata, self.indexer_cache.layer_index)

        q5 = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        q5 = q5.view(batch, num_kv, groups, seq_len, head_dim)
        k5 = k.view(batch, seq_len, num_kv, head_dim).transpose(1, 2)
        k5 = k5.view(batch, num_kv, 1, seq_len, head_dim)
        v5 = v.view(batch, seq_len, num_kv, head_dim).transpose(1, 2)
        v5 = v5.view(batch, num_kv, 1, seq_len, head_dim)
        index_q4 = (
            index_q.view(batch, seq_len, self.num_idx_heads, self.idx_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        index_k3 = index_k.contiguous()

        topk_index = torch.ops.rbln_custom_ops.sparse_attn_minimax_m3_indexer(
            index_q4,
            index_k3,
            index_cache,
            self.scale_tensor,
            index_metadata.seq_lens.to(torch.int32),
            index_metadata.block_tables,
            self.topk_blocks,
            *(
                (self.k_scale_tensor, self.indexer_cache.fp8_dtype)
                if self.indexer_cache.fp8_dtype is not None
                else ()
            ),
        )
        attn_output = torch.ops.rbln_custom_ops.sparse_attn_minimax_m3_msa(
            q5,
            k5,
            v5,
            kv_cache,
            self.scale_tensor,
            main_metadata.seq_lens.to(torch.int32),
            main_metadata.block_tables,
            topk_index,
            *(
                (self.k_scale_tensor, self.v_scale_tensor, self.kv_cache_fp8_dtype)
                if self.kv_cache_fp8_dtype is not None
                else ()
            ),
        )
        # [B, H_kv, G, L, D] -> [B, L, H * D]
        attn_output = attn_output.view(batch, num_heads, seq_len, head_dim).transpose(
            1, 2
        )
        attn_output = attn_output.reshape(batch, seq_len, num_heads * head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class RBLNMiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

        sparse_cfg = getattr(config, "sparse_attention_config", None)
        sparse_freq = sparse_cfg.get("sparse_attention_freq") if sparse_cfg else None
        if sparse_freq and sparse_freq[layer_id] != 0:
            self.self_attn = RBLNMiniMaxM3SparseAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )
        else:
            self.self_attn = RBLNMiniMaxM3Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        self.is_moe_layer = moe_layer_freq is None or moe_layer_freq[layer_id] != 0
        if self.is_moe_layer:
            self.block_sparse_moe = RBLNMiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = RBLNMiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual


class RBLNMiniMaxM3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNMiniMaxM3DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Checkpoint experts use w1=gate, w2=down, w3=up.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # The MTP module is not modeled.
            if "mtp." in name:
                continue
            # The checkpoint stores the MXFP8 block scales as `weight_scale_inv`;
            # the linear methods expose them as `weight_scale`.
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class RBLNMiniMaxM3SparseForCausalLM(nn.Module, SupportsPP):
    """MiniMax M3 (sparse/dense backbone) for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = RBLNMiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def _load_upstream_common(name: str) -> ModuleType:
    """Import ``vllm.models.minimax_m3.common.<name>`` without running the
    package ``__init__``, which imports the NVIDIA model (CUDA custom ops) at
    import time. The ``common`` modules only use absolute imports."""
    full_name = f"vllm.models.minimax_m3.common.{name}"
    if (module := sys.modules.get(full_name)) is not None:
        return module
    import vllm.models

    path = Path(vllm.models.__file__).parent / "minimax_m3" / "common" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(full_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_mm_preprocess = _load_upstream_common("mm_preprocess")
_vision_tower = _load_upstream_common("vision_tower")


# (72 x 72 grid, 1008 x 1008 px = max_image_resolution 1008 / patch_size 14).
MM_ENCODER_PATCH_BUCKET = 5184


class _RBLNVisionEncoder(nn.Module):
    def __init__(self, tower: nn.Module):
        super().__init__()
        vm = tower.vision_model
        attn = vm.encoder.layers[0].self_attn
        self.tower = tower
        self.num_heads = attn.num_heads_per_partition
        self.head_dim = attn.head_dim
        w = vm.embeddings.patch_embedding.weight
        self.patch_weight = w.reshape(w.shape[0], -1)
        half = (vm.t_dim + vm.h_dim + vm.w_dim) // 2
        rot = torch.zeros(self.head_dim, self.head_dim)
        for j in range(half):
            rot[j + half, j] = -1.0
            rot[j, j + half] = 1.0
        self.rot = rot.to(device=w.device, dtype=w.dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_bias: torch.Tensor,
    ) -> torch.Tensor:
        vm = self.tower.vision_model
        n = pixel_values.shape[0]
        hidden = vm.pre_layrnorm(F.linear(pixel_values, self.patch_weight))
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        scale = self.head_dim**-0.5
        for layer in vm.encoder.layers:
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(layer.layer_norm1(hidden))
            qkv = qkv.view(n, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [N, heads, D]
            q = q * cos + torch.matmul(q, self.rot) * sin
            k = k * cos + torch.matmul(k, self.rot) * sin
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            scores = torch.matmul(q, k.transpose(1, 2)) * scale + key_bias[0]
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
            out, _ = attn.out_proj(out.transpose(0, 1).reshape(n, -1))
            hidden = hidden + out
            mlp, _ = layer.fc1(layer.layer_norm2(hidden))
            mlp, _ = layer.fc2(layer.act(mlp))
            hidden = hidden + mlp
        hidden = self.tower.multi_modal_projector(hidden)
        return self.tower.patch_merge_mlp(hidden)


@MULTIMODAL_REGISTRY.register_processor(
    _mm_preprocess.MiniMaxM3VLMultiModalProcessor,
    info=_mm_preprocess.MiniMaxM3VLProcessingInfo,
    dummy_inputs=_mm_preprocess.MiniMaxM3VLDummyInputsBuilder,
)
class RBLNMiniMaxM3SparseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return _mm_preprocess.MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return _mm_preprocess.MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vision_tower = None

        mm_config = vllm_config.model_config.multimodal_config
        if get_pp_group().is_first_rank and any(
            mm_config.get_limit_per_prompt(m) > 0 for m in ("image", "video")
        ):
            with set_default_torch_dtype(torch.float32):
                self.vision_tower = _vision_tower.MiniMaxVLVisionModel(
                    config=PretrainedConfig.from_dict(config.vision_config),
                    text_hidden_size=config.text_config.hidden_size,
                    projector_hidden_size=getattr(
                        config, "projector_hidden_size", None
                    ),
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )

        object.__setattr__(self, "_mm_encoder", None)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["MiniMaxM3SparseForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    @property
    def logits_processor(self) -> nn.Module:
        return self.language_model.logits_processor

    def compile_mm_encoder(self, compile_fn) -> None:
        """Called by the runner at model load with its compile wrapper."""
        assert self.vision_tower is not None
        encoder = _RBLNVisionEncoder(self.vision_tower).eval()
        object.__setattr__(self, "_mm_encoder", compile_fn(encoder))

    def _encoder_inputs(
        self, pixel_values: torch.Tensor | None, grid_thw: list[int]
    ) -> list[torch.Tensor]:
        """Pad one image to the bucket and build its host 3D-RoPE and key mask.
        ``pixel_values`` None gives an all-padding (warm-up) image."""
        tower = self.vision_tower
        assert tower is not None
        vm = tower.vision_model
        bucket = MM_ENCODER_PATCH_BUCKET
        num_patches = math.prod(grid_thw)
        assert num_patches <= bucket, (
            f"an image of {num_patches} patches (grid {grid_thw}) exceeds the "
            f"vision encoder's {bucket}-patch bucket"
        )
        dtype = tower.dtype
        device = next(tower.parameters()).device
        patch_dim = vm.embeddings.patch_embedding.weight[0].numel()
        padded = torch.zeros(bucket, patch_dim, dtype=dtype)
        if pixel_values is not None:
            padded[:num_patches] = pixel_values.to(dtype)

        t, h, w = grid_thw
        merge = tower.spatial_merge_size

        def spatial_ids(ids: torch.Tensor) -> torch.Tensor:
            return (
                ids.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .unsqueeze(0)
                .expand(t, -1, -1, -1, -1)
                .flatten()
            )

        t_ids = torch.arange(t).unsqueeze(1).expand(-1, h * w).flatten()
        h_ids = spatial_ids(torch.arange(h).unsqueeze(1).expand(-1, w))
        w_ids = spatial_ids(torch.arange(w).unsqueeze(0).expand(h, -1))
        freqs = torch.cat(
            [
                torch.outer(t_ids.float(), vm.inv_freq_t.cpu()),
                torch.outer(h_ids.float(), vm.inv_freq_h.cpu()),
                torch.outer(w_ids.float(), vm.inv_freq_w.cpu()),
            ],
            dim=-1,
        )
        half = freqs.shape[-1]
        head_dim = vm.encoder.layers[0].self_attn.head_dim
        cos = torch.ones(bucket, head_dim)
        sin = torch.zeros(bucket, head_dim)
        cos[:num_patches, :half] = cos[:num_patches, half : 2 * half] = freqs.cos()
        sin[:num_patches, :half] = sin[:num_patches, half : 2 * half] = freqs.sin()

        key_bias = torch.zeros(1, 1, 1, bucket)
        key_bias[..., num_patches:] = -30000.0
        return [x.to(device=device, dtype=dtype) for x in (padded, cos, sin, key_bias)]

    def warmup_mm_encoder(self) -> None:
        """Run the compiled encoder once on an all-padding image."""
        side = math.isqrt(MM_ENCODER_PATCH_BUCKET)
        with torch.inference_mode():
            self._mm_encoder(*self._encoder_inputs(None, [1, side, side]))

    def _encode_image(
        self, pixel_values: torch.Tensor, grid_thw: list[int]
    ) -> torch.Tensor:
        assert self.vision_tower is not None and self._mm_encoder is not None
        embeds = self._mm_encoder(*self._encoder_inputs(pixel_values, grid_thw))
        merge = self.vision_tower.spatial_merge_size
        return embeds[: math.prod(grid_thw) // (merge * merge)].cpu()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if "pixel_values_videos" in kwargs:
            raise NotImplementedError(
                "MiniMax-M3 video inputs are not supported on RBLN"
            )
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return ()
        assert self._mm_encoder is not None, "the vision encoder is not compiled"
        image_grid_thw = kwargs["image_grid_thw"]
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_grid_thw, torch.Tensor)
        grid_thw = image_grid_thw.tolist()
        sizes = [math.prod(g) for g in grid_thw]
        with torch.inference_mode():
            return tuple(
                self._encode_image(pv, g)
                for pv, g in zip(pixel_values.split(sizes), grid_thw)
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_embeds: torch.Tensor | None = None,
        mm_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            mm_embeds is not None
            and inputs_embeds is None
            and get_pp_group().is_first_rank
        ):
            assert mm_mask is not None
            text_embeds = self.language_model.embed_input_ids(input_ids)
            inputs_embeds = torch.where(mm_mask > 0.5, mm_embeds, text_embeds)
            input_ids = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def maybe_text_only():
            for name, weight in self.hf_to_vllm_mapper.apply(weights):
                if self.vision_tower is None and name.startswith("vision_tower."):
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(maybe_text_only())
