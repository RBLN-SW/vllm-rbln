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

"""MiniMax-M3 (text backbone) for RBLN.

Vendored from upstream ``vllm.models.minimax_m3`` (vLLM 0.24), which cannot
run here: it binds FlashInfer Gemma norms, the fused CUDA
``fused_minimax_m3_qknorm_rope_kv_insert`` and the SM100/Triton block-sparse
attends at import. This copy keeps the module tree and weight names (so the
NVFP4 checkpoint and the ModelOpt mixed-precision config resolve unchanged)
and replaces the compute with RBLN-friendly code:

* hidden states stay 3-D ``[B, L, H]`` end-to-end (RBLN convention);
* Gemma RMSNorm and the per-head QK norms are plain torch;
* the dense layers use the generic ``Attention`` (RBLN flash backend);
* the sparse layers run the MSA lightning indexer and the block-sparse GQA
  attention through ``rbln_custom_ops.sparse_attn_minimax_indexer`` /
  ``sparse_attn_minimax_attn``, each reading its paged cache from the
  attention metadata (a graph input) like the DSA path does;
* MoE goes through the RBLN ``MoERunner`` (router callback) with the
  routed-scaling factor and the shared expert applied here.

The MTP head and the vision tower are not modeled; their weights are skipped.
"""

from collections.abc import Iterable

import torch
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
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
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
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.patches.attention import _resolve_kv_cache
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


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    """Layer ids whose attention runs the extra sparse "index" branch."""
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _layer_index_of(prefix: str) -> int:
    """Pipeline-adjusted index of this layer's cache in the runner's compacted
    KV cache list (``num_attn_module`` slots per decoder layer)."""
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(
        model_config, vllm_config.cache_config.cache_dtype
    )
    return pipeline_adjusted_layer_index(
        prefix, model_config, vllm_config.parallel_config, num_attn_module
    )


class RBLNGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``, in fp32.

    With ``residual`` the pre-norm residual add is fused and the updated
    ``(x, residual)`` pair is returned, like vLLM's RMSNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.variance_epsilon)
        xf = xf * (1.0 + self.weight.to(torch.float32))
        return xf.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


def _per_head_norm(
    x: torch.Tensor, norm: RBLNGemmaRMSNorm, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Apply a head-dim norm to ``[B, L, num_heads * head_dim]``."""
    shape = x.shape
    x = x.view(*shape[:-1], num_heads, head_dim)
    return norm(x).view(shape)


def _swiglu_oai(x: torch.Tensor, alpha: float, beta: float, limit: float) -> torch.Tensor:
    """``gate.clamp(max=limit) * sigmoid(alpha * gate) * (up.clamp(+-limit) + beta)``
    over a ``[..., 2 * I]`` (gate | up) tensor."""
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + beta)


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
        x = _swiglu_oai(gate_up, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class RBLNMiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE with a routing-bias correction and a shared expert.

    The RBLN ``MoERunner`` takes the router as a callable (routing runs after
    the DP multicast) and does not apply ``routed_scaling_factor`` nor the
    shared expert, so both are applied here, as the DeepSeek-V2 RBLN patch does.
    """

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

        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
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

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 is loaded packed ([all gates; all ups]), so the uninterleaved
            # SwiGLU-OAI variant. The RBLN packed-FP4 MoE kernel reads this
            # name and turns on its alpha / limit clamp (+1 up bias) path.
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

        self.q_norm = RBLNGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RBLNGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        # Dense layers carry no index-key cache, but the runner binds
        # `num_attn_module` caches per decoder layer, so give it one to keep the
        # compacted cache list aligned (one 128-wide vector per token).
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
    """Key-only side cache of the lightning indexer (one index key per token).

    Registers itself in the static forward context so the KV-cache manager
    allocates it, like ``DeepseekV32IndexerCache``.
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = torch.bfloat16
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
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
    """Block-sparse attention layer with the lightning-indexer branch.

    Owns the projections (fused [q | k | v | index_q | index_k]), the per-head
    QK norms and RoPE, the main paged K/V cache (registered here under
    ``{prefix}.attn``) and the indexer's key cache (``{prefix}.attn.indexer``).
    ``index_{v,o}_proj`` never exist for M3 (``sparse_disable_index_value``).
    """

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

        self.q_norm = RBLNGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RBLNGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        self.index_q_norm = RBLNGemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = RBLNGemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        # index_dim == head_dim for M3, so the index branch shares the RoPE.
        assert self.idx_head_dim == self.head_dim
        self.index_rotary_emb = self.rotary_emb

        # Attention-backend wiring (the main paged K/V cache).
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "the RBLN MSA attention kernel reads a bf16 K/V cache; got "
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

        # One fused projection: [q | k | v | index_q | index_k].
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
        index_q = _per_head_norm(index_q, self.index_q_norm, self.num_idx_heads, self.idx_head_dim)
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

        # RBLN GQA layout: query [B, H_kv, G, L, D] with head h = kv * G + g,
        # key / value [B, H_kv, 1, L, D]; index query [B, H_idx, L, 128].
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

        # Token positions of the top-k (+ local) blocks, per (batch, index head,
        # query): ascending, -1 padded; the attend consumes them like DSA's top-k.
        topk_index = torch.ops.rbln_custom_ops.sparse_attn_minimax_indexer(
            index_q4,
            index_k3,
            index_cache,
            self.scale_tensor,
            index_metadata.seq_lens.to(torch.int32),
            index_metadata.block_tables,
            self.topk_blocks,
        )
        attn_output = torch.ops.rbln_custom_ops.sparse_attn_minimax_attn(
            q5,
            k5,
            v5,
            kv_cache,
            self.scale_tensor,
            main_metadata.seq_lens.to(torch.int32),
            main_metadata.block_tables,
            topk_index,
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

        if layer_id in _sparse_attention_layer_ids(config):
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

        # Dense layers store the FFN under `mlp`; MoE layers under
        # `block_sparse_moe` -- the checkpoint's naming.
        self.is_moe_layer = _is_moe_layer(config, layer_id)
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

        self.input_layernorm = RBLNGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RBLNGemmaRMSNorm(
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
            self.norm = RBLNGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        # q/k/v_proj -> fused qkv_proj (plus index_q/index_k_proj on sparse
        # layers); gate_proj/up_proj -> fused gate_up_proj. Leading dots keep
        # `q_proj` from matching `index_q_proj`.
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
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
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


# Vision-side weight prefixes of the VL checkpoint; the tower is not modeled.
_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


class RBLNMiniMaxM3SparseForConditionalGeneration(nn.Module, SupportsPP):
    """Text-only entry point for the MiniMax M3 (VL) checkpoint.

    Builds the text backbone from ``config.text_config`` under the
    ``language_model`` prefix (the checkpoint's naming) and drops the vision
    tower's weights.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def text_only():
            for name, weight in weights:
                if name.startswith(_VISION_PREFIXES):
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(text_only())
