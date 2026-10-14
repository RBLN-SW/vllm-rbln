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

"""GLM-5.3 (glm5_next) text backbone for RBLN.

Upstream's ``vllm.models.glm5next.nvidia`` binds FLA / TileLang / DeepGEMM kernels
and the hybrid mamba state manager. This keeps its module tree and weight names and
runs:

* mHC (4 residual streams) through ``hc_stream_reduce`` / ``hc_residual_add`` /
  ``sinkhorn_normalize``;
* KDA layers through ``kda_conv`` and ``kda_prefill`` / ``kda_decode``, the
  recurrent state and the q / k / v conv rings each one ``RBLNSlidingWindowSpec``
  slot per request;
* DSA layers (NoPE MLA) through ``glm5_kpool_compress``,
  ``sparse_attn_glm5_indexer`` and ``sparse_attn_glm5_mla`` over paged latent /
  indexer key / indexer scale caches.

Every decoder layer owns four caches, named ``{prefix}.attn.{slot}``. Text only: the
vision tower and the MTP head are not modeled and their weights are skipped. TP 1.
"""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from rebel.ops.torch_custom_ops import kda as _rbln_kda_ops  # noqa: F401
from rebel.ops.torch_custom_ops import mhc as _rbln_mhc_ops  # noqa: F401
from rebel.ops.torch_custom_ops import sparse_attn_glm5 as _rbln_glm5_ops
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_rbln.config import get_rbln_config
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)
from vllm_rbln.v1.attention.backends.glm5_next import (
    RBLNGlm5IndexerBackend,
    RBLNGlm5LatentBackend,
    RBLNGlm5RingBackend,
)
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec
from vllm_rbln.v1.worker.utils import (
    num_attn_module as rbln_num_attn_module,
)
from vllm_rbln.v1.worker.utils import (
    pipeline_adjusted_layer_index,
)

# The four caches of a decoder layer, by the slot in its cache names.
SLOT_KDA_STATE, SLOT_KDA_CONV_Q, SLOT_KDA_CONV_K, SLOT_KDA_CONV_V = range(4)
SLOT_DSA_LATENT, SLOT_DSA_INDEX_KEY, SLOT_DSA_INDEX_SCALE, SLOT_DSA_INDEX_TAIL = range(
    4
)

KDA_HEAD_DIM = 128
# The conv ring's width: the 3 taps before the current token (no MTP verify tokens).
KDA_CONV_RING = 4
# kda_prefill runs whole 128-token chunks.
KDA_CHUNK = 128
# The sparse MLA kernel's table must span a multiple of this many tokens.
MLA_TABLE_TOKENS = 16384
INDEX_KPOOL = _rbln_glm5_ops.KPOOL
INDEX_HEAD_DIM = _rbln_glm5_ops.INDEX_HEAD_DIM


def _layer_index_of(prefix: str) -> int:
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(
        model_config, vllm_config.cache_config.cache_dtype
    )
    return pipeline_adjusted_layer_index(
        prefix, model_config, vllm_config.parallel_config, num_attn_module
    )


def _metadata(prefix: str) -> RBLNFlashAttentionMetadata:
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return attn_metadata[prefix]
    return attn_metadata


class _RBLNGlm5Cache(nn.Module, AttentionLayerBase):
    """A cache with no compute of its own, registered so the KV-cache manager allocates
    it and the runner binds it as a graph input."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_index = _layer_index_of(prefix)

    def forward(self) -> None: ...


class RBLNGlm5RingCache(_RBLNGlm5Cache):
    """One ``[rows, width]`` slot per request. ``RBLNSlidingWindowSpec`` budgets
    K + V, so it is declared with half the row width."""

    def __init__(self, prefix: str, rows: int, width: int, dtype: torch.dtype) -> None:
        super().__init__(prefix)
        assert width % 2 == 0
        self.rows = rows
        self.width = width
        self.dtype = dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return RBLNSlidingWindowSpec(
            block_size=self.rows,
            num_kv_heads=1,
            head_size=self.width // 2,
            dtype=self.dtype,
            sliding_window=self.rows,
        )

    def get_attn_backend(self) -> type[RBLNGlm5RingBackend]:
        return RBLNGlm5RingBackend

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, RBLNFlashAttentionMetadata]:
        """(ring [slots, rows, width], slot [B, 1], metadata). Read only the metadata
        fields an op takes: a graph input nothing uses is pruned from the compiled
        module, and the runtime then binds the inputs out of step."""
        metadata = _metadata(self.prefix)
        ring = _resolve_kv_cache(metadata, self.layer_index)
        # The hybrid allocator unifies page sizes by growing a smaller spec's block
        # size, so a page may hold k slots; the request's is the first.
        per_page = ring.numel() // (ring.shape[0] * self.rows * self.width)
        ring = ring.view(ring.shape[0] * per_page, self.rows, self.width)
        slot = metadata.local_block_tables.reshape(-1, 1).to(torch.int32) * per_page
        return ring, slot, metadata


class RBLNGlm5PagedCache(_RBLNGlm5Cache):
    """A paged cache of ``block_size / tokens_per_state`` rows a block."""

    def __init__(
        self,
        prefix: str,
        head_size: int,
        dtype: torch.dtype,
        tokens_per_state: int,
        backend: type,
    ) -> None:
        super().__init__(prefix)
        self.head_size = head_size
        self.dtype = dtype
        self.tokens_per_state = tokens_per_state
        self.backend = backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            tokens_per_state=self.tokens_per_state,
        )

    def get_attn_backend(self) -> type:
        return self.backend

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, RBLNFlashAttentionMetadata]:
        """(cache, block_table, metadata)."""
        metadata = _metadata(self.prefix)
        return (
            _resolve_kv_cache(metadata, self.layer_index),
            metadata.block_tables,
            metadata,
        )


class _ConvWeight(nn.Module):
    """A depthwise conv's taps as ``[channels, taps]`` (checkpoint ``[C, 1, taps]``)."""

    def __init__(self, channels: int, taps: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, taps), requires_grad=False)
        self.weight.weight_loader = self._load

    @staticmethod
    def _load(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight.reshape(param.shape))


class RBLNRMSNorm(nn.Module):
    """RMSNorm in fp32 reading its weight as a parameter: vLLM's passes ``weight.data``,
    which export lifts to a constant the compiler resolves by shape alone."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(x.dtype)


def _swiglu_clamped(gate_up: torch.Tensor, limit: float | None) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class RBLNGlm5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None,
        quant_config: QuantizationConfig | None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(_swiglu_clamped(gate_up, self.swiglu_limit))
        return x


class RBLNGlm5MoE(nn.Module):
    """Sigmoid-routed (noaux_tc) MoE with a shared expert. The RBLN ``MoERunner``
    takes the router as a callable and applies neither ``routed_scaling_factor`` nor
    the shared expert, so both are applied here, as the DeepSeek-V2 RBLN patch does."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert config.hidden_act == "silu"
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        assert config.topk_method == "noaux_tc"
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.shared_experts = RBLNGlm5MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.swiglu_limit,
            quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            # One group is plain top-k; the grouped path's [T, 1, 288] view breaks the
            # 128-byte alignment RSD needs.
            use_grouped_topk=config.n_group > 1,
            num_expert_group=config.n_group if config.n_group > 1 else None,
            topk_group=config.topk_group if config.n_group > 1 else None,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            router_logits_dtype=torch.float32,
            swiglu_limit=config.swiglu_limit,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routed = self.experts(
            hidden_states=hidden_states,
            router=lambda x: self.gate(x.to(torch.float32))[0],
        )
        return routed * self.routed_scaling_factor + self.shared_experts(hidden_states)


class RBLNGlm5KDA(nn.Module):
    """Kimi Delta Attention with GLM-5.3's bounded sigmoid gate."""

    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        assert self.head_dim == KDA_HEAD_DIM
        self.lower_bound = config.linear_lower_bound
        proj = self.num_heads * self.head_dim
        taps = config.linear_conv_kernel_dim

        def linear(name: str, in_size: int, out_size: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                in_size, out_size, bias=False, prefix=f"{prefix}.{name}"
            )

        self.q_proj = linear("q_proj", hidden_size, proj)
        self.k_proj = linear("k_proj", hidden_size, proj)
        self.v_proj = linear("v_proj", hidden_size, proj)
        self.b_proj = linear("b_proj", hidden_size, self.num_heads)
        self.f_a_proj = linear("f_a_proj", hidden_size, self.head_dim)
        self.f_b_proj = linear("f_b_proj", self.head_dim, proj)
        self.g_a_proj = linear("g_a_proj", hidden_size, self.head_dim)
        self.g_b_proj = linear("g_b_proj", self.head_dim, proj)
        self.q_conv1d = _ConvWeight(proj, taps)
        self.k_conv1d = _ConvWeight(proj, taps)
        self.v_conv1d = _ConvWeight(proj, taps)
        self.A_log = nn.Parameter(
            torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
        )
        self.dt_bias = nn.Parameter(
            torch.empty(proj, dtype=torch.float32), requires_grad=False
        )
        # f_b(f_a(x)) and g_b(g_a(x)) as one [proj, hidden] weight each (fuse_low_rank):
        # RSD cuts the rank-128 contraction 2 ways and reduces onto 2 of its 4 nodes,
        # which then strands the KDA state there.
        self.register_buffer(
            "f_weight", torch.empty(proj, hidden_size), persistent=False
        )
        self.register_buffer(
            "g_weight", torch.empty(proj, hidden_size), persistent=False
        )
        self.o_norm = RBLNRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            proj, hidden_size, bias=False, prefix=f"{prefix}.o_proj"
        )

        dtype = get_current_vllm_config().model_config.dtype
        self.state = RBLNGlm5RingCache(
            f"{prefix}.attn.{SLOT_KDA_STATE}", proj, self.head_dim, dtype
        )
        conv_rows = proj * KDA_CONV_RING // 64
        self.conv_q, self.conv_k, self.conv_v = (
            RBLNGlm5RingCache(f"{prefix}.attn.{slot}", conv_rows, 64, dtype)
            for slot in (SLOT_KDA_CONV_Q, SLOT_KDA_CONV_K, SLOT_KDA_CONV_V)
        )

    @torch.no_grad()
    def fuse_low_rank(self) -> None:
        for fused, a, b in (
            (self.f_weight, self.f_a_proj, self.f_b_proj),
            (self.g_weight, self.g_a_proj, self.g_b_proj),
        ):
            fused.copy_(b.weight.float() @ a.weight.float())

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)

        state, slot, metadata = self.state.inputs()
        seq_idx = metadata.seq_lens.to(torch.int32)
        query_len = (metadata.cache_offsets - metadata.cache_seq_lens).to(torch.int32)
        has_initial_state = (seq_idx > 0).to(torch.int32).view(-1)
        slot = slot.view(-1)

        def conv(x: torch.Tensor, cache: RBLNGlm5RingCache, weight: torch.Tensor):
            ring, conv_slot, _ = cache.inputs()
            return torch.ops.rbln_custom_ops.kda_conv(
                x,
                weight,
                ring,
                conv_slot.view(-1),
                seq_idx.view(-1),
                query_len.view(-1),
                has_initial_state,
                KDA_CONV_RING,
            )

        q = conv(self.q_proj(hidden_states)[0], self.conv_q, self.q_conv1d.weight)
        k = conv(self.k_proj(hidden_states)[0], self.conv_k, self.k_conv1d.weight)
        v = conv(self.v_proj(hidden_states)[0], self.conv_v, self.v_conv1d.weight)
        q = self._l2norm(q.view(shape))
        k = self._l2norm(k.view(shape))
        v = v.view(shape)

        f = F.linear(hidden_states, self.f_weight).view(shape)
        decay = self.A_log.exp().view(-1, 1) * (
            f.float() + self.dt_bias.view(shape[2:])
        )
        g = self.lower_bound * torch.sigmoid(decay)  # [B, T, H, D] f32
        beta = torch.sigmoid(self.b_proj(hidden_states)[0].float())  # [B, T, H]

        state = state.view(-1, self.num_heads, self.head_dim, self.head_dim)
        q, k, v, g = (x.transpose(1, 2) for x in (q, k, v, g))
        beta = beta.transpose(1, 2).to(hidden_states.dtype)
        if metadata.is_prefill:
            # Padding tokens carry the state through unchanged: no decay, no update.
            valid = torch.arange(tokens, device=g.device).view(
                1, 1, -1
            ) < query_len.view(-1, 1, 1)
            g = g * valid.unsqueeze(-1)
            beta = beta * valid
            pad = -tokens % KDA_CHUNK
            if pad:
                q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
                beta = F.pad(beta, (0, pad))
            o = torch.ops.rbln_custom_ops.kda_prefill(
                q, k, v, g, beta, state, slot, has_initial_state
            )[:, :, :tokens]
        else:
            o = torch.ops.rbln_custom_ops.kda_decode(q, k, v, g, beta, state, slot)

        o = o.transpose(1, 2)  # [B, T, H, D]
        gate = F.linear(hidden_states, self.g_weight).view(shape)
        o = self.o_norm(o) * torch.sigmoid(gate.float()).to(o.dtype)
        return self.o_proj(o.reshape(batch, tokens, -1))[0]


class RBLNGlm5Indexer(nn.Module):
    """The kpool lightning indexer: 4-token pools, Hadamard-rotated fp8 keys. The
    rotation is folded into ``wq_b`` at load (``fold_hadamard``)."""

    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        assert config.index_kpool == INDEX_KPOOL
        assert config.index_head_dim == INDEX_HEAD_DIM
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk // config.index_kpool
        self.weights_scale = self.head_dim**-0.5 * self.n_head**-0.5
        hidden_size = config.hidden_size

        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_head * self.head_dim,
            bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size, self.head_dim, bias=False, prefix=f"{prefix}.wk"
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size, self.n_head, bias=False, prefix=f"{prefix}.weights_proj"
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size), requires_grad=False
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(INDEX_KPOOL, self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )

        attn_prefix = prefix.rsplit(".indexer", 1)[0]
        self.key_cache = RBLNGlm5PagedCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_KEY}",
            self.head_dim,
            torch.float8_e4m3fn,
            INDEX_KPOOL,
            RBLNGlm5IndexerBackend,
        )
        # f16 rows of the key row's bytes, so both get one page size (a block's E scales
        # use its first E * 2 bytes) without a dtype view in the graph. It is another
        # KV cache group, so it takes its own block table.
        self.scale_cache = RBLNGlm5PagedCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_SCALE}",
            self.head_dim // 2,
            torch.float16,
            INDEX_KPOOL,
            RBLNGlm5IndexerBackend,
        )
        self.tail_cache = RBLNGlm5RingCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_TAIL}",
            INDEX_KPOOL,
            2 * self.head_dim,
            torch.bfloat16,
        )
        # The tail ring is one slot per request: a block holding every position.
        max_model_len = get_current_vllm_config().model_config.max_model_len
        self.tail_block_size = -(-max_model_len // INDEX_KPOOL) * INDEX_KPOOL

    @torch.no_grad()
    def fold_hadamard(self) -> None:
        w = self.wq_b.weight.data.view(self.n_head, self.head_dim, -1)
        rot = _rbln_glm5_ops.hadamard_128().to(w.device)
        w.copy_(torch.einsum("ij,hjk->hik", rot, w.float()).to(w.dtype))

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, is_prompt_sorted: bool
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        q = self.wq_b(qr)[0].view(batch, tokens, self.n_head, self.head_dim)
        key = self.wk(hidden_states)[0]
        key = F.layer_norm(
            key.float(), (self.head_dim,), self.k_norm.weight, self.k_norm.bias, 1e-6
        ).to(hidden_states.dtype)
        gate = F.linear(hidden_states, self.index_kpool_compress_gate)
        weights = (
            F.linear(hidden_states.float(), self.weights_proj.weight.float())
            * self.weights_scale
        )

        tail, tail_slot, metadata = self.tail_cache.inputs()
        seq_idx = metadata.seq_lens.to(torch.int32)
        query_len = (metadata.cache_offsets - metadata.cache_seq_lens).to(torch.int32)
        k_cur = torch.ops.rbln_custom_ops.glm5_kpool_compress(
            key,
            gate,
            self.index_kpool_compress_ape,
            tail,
            seq_idx,
            query_len,
            tail_slot,
            self.tail_block_size,
        )
        key_cache, key_table, _ = self.key_cache.inputs()
        scale_cache, scale_table, _ = self.scale_cache.inputs()
        scale_cache = scale_cache.reshape(scale_cache.shape[0], -1, 1)
        return torch.ops.rbln_custom_ops.sparse_attn_glm5_indexer(
            q.transpose(1, 2),
            k_cur,
            key_cache,
            scale_cache,
            weights,
            seq_idx,
            query_len,
            key_table,
            self.topk,
            scale_table,
        )


class RBLNGlm5DSA(nn.Module):
    """DeepSeek sparse attention without RoPE: absorbed MLA over the indexer's pools."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.qk_rope_head_dim == 0
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        self.q_a_proj = ReplicatedLinear(
            hidden_size,
            config.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RBLNRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RBLNRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.indexer = RBLNGlm5Indexer(config, f"{prefix}.indexer")
        dtype = get_current_vllm_config().model_config.dtype
        self.latent_cache = RBLNGlm5PagedCache(
            f"{prefix}.attn.{SLOT_DSA_LATENT}",
            self.kv_lora_rank,
            dtype,
            1,
            RBLNGlm5LatentBackend,
        )
        self.scale = self.qk_nope_head_dim**-0.5
        # The runner sorts a batch longest first under the same condition
        # (RBLNModelRunner.sort_batch_by_length).
        self.is_prompt_sorted = (
            current_platform.is_cr13() or get_rbln_config().use_batch_attn_opt
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        qr = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(qr)[0].view(
            batch, tokens, self.num_heads, self.qk_nope_head_dim
        )
        kv_c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states)[0])
        topk = self.indexer(hidden_states, qr, self.is_prompt_sorted)

        kv_b = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_uk, w_uv = kv_b.unsqueeze(0).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=2
        )
        # matmul, not einsum: the attention returns f32 (normalized and cast by a
        # later compiler pass), which relay's einsum refuses next to bf16 weights.
        q_mla = torch.matmul(q.transpose(1, 2), w_uk)  # [B, H, T, L]
        latent_cache, block_table, metadata = self.latent_cache.inputs()
        # The kernel walks the table in groups of 16384 tokens: a max_model_len below
        # that gets null-block (0) columns, past every sequence, so never read.
        block_size = latent_cache.shape[1]
        pad = -block_table.shape[-1] * block_size % MLA_TABLE_TOKENS // block_size
        if pad:
            block_table = F.pad(block_table, (0, pad))
        o = torch.ops.rbln_custom_ops.sparse_attn_glm5_mla(
            q_mla,
            kv_c,
            latent_cache,
            self.scale,
            metadata.seq_lens.to(torch.int32),
            block_table,
            topk,
            is_prompt_sorted=self.is_prompt_sorted,
        )
        o = torch.matmul(o, w_uv.transpose(2, 3))  # [B, H, T, V]
        o = o.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(o)[0]


class RBLNGlm5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.n = config.mhc_num_residual_streams
        self.sinkhorn_iters = config.mhc_sinkhorn_iterations
        self.post_mult = config.mhc_post_mult_value

        if config.layer_types[layer_idx] == "linear_attention":
            self.self_attn: nn.Module = RBLNGlm5KDA(config, f"{prefix}.self_attn")
        else:
            assert config.layer_types[layer_idx] == "deepseek_sparse_attention"
            self.self_attn = RBLNGlm5DSA(config, quant_config, f"{prefix}.self_attn")
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp: nn.Module = RBLNGlm5MoE(config, quant_config, f"{prefix}.mlp")
        else:
            self.mlp = RBLNGlm5MLP(
                config.hidden_size,
                config.intermediate_size,
                config.swiglu_limit,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RBLNRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RBLNRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        mix = (2 + self.n) * self.n
        d_model = self.n * config.hidden_size
        for kind in ("attn", "ffn"):
            for name, shape in (
                ("fn", (mix, d_model)),
                ("base", (mix,)),
                ("scale", (3,)),
            ):
                self.register_parameter(
                    f"hc_{kind}_{name}",
                    nn.Parameter(
                        torch.empty(shape, dtype=torch.float32), requires_grad=False
                    ),
                )
            # The 3 scales spread over the mix's lanes (expand_hc_scales): a slice of a
            # small weight runs as a host op whose constant gets no buffer slot.
            self.register_buffer(
                f"hc_{kind}_scale_lanes",
                torch.empty(mix, dtype=torch.float32),
                persistent=False,
            )

    @torch.no_grad()
    def expand_hc_scales(self) -> None:
        n = self.n
        lanes = torch.tensor([n, n, n * n], device=self.hc_attn_scale.device)
        for kind in ("attn", "ffn"):
            scale = getattr(self, f"hc_{kind}_scale")
            getattr(self, f"hc_{kind}_scale_lanes").copy_(
                scale.repeat_interleave(lanes)
            )

    def _hc_mix(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_pre, z_post, comb) of a sublayer, from the RMS-scaled stream mix."""
        batch, n, tokens, hidden = streams.shape
        flat = streams.transpose(1, 2).reshape(batch, tokens, n * hidden).float()
        mixes = F.linear(flat, fn) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.rms_eps
        )
        # The mHC ops take bf16 operands (they compute in fp32).
        z = (mixes * scale + base).to(streams.dtype)
        z_pre, z_post = z[..., :n], z[..., n : 2 * n]
        comb = torch.ops.rbln_custom_ops.sinkhorn_normalize(
            z[..., 2 * n :],
            iters=self.sinkhorn_iters,
            eps=self.hc_eps,
        )
        return z_pre, z_post, comb

    def _sublayer(self, streams, fn, scale, base, norm, sublayer) -> torch.Tensor:
        z_pre, z_post, comb = self._hc_mix(streams, fn, scale, base)
        x = torch.ops.rbln_custom_ops.hc_stream_reduce(streams, z_pre, eps=self.hc_eps)
        x = sublayer(norm(x))
        return torch.ops.rbln_custom_ops.hc_residual_add(
            x, streams, z_post, comb, post_mult=self.post_mult
        )

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """streams [B, n, T, H] -> [B, n, T, H]."""
        streams = self._sublayer(
            streams,
            self.hc_attn_fn,
            self.hc_attn_scale_lanes,
            self.hc_attn_base,
            self.input_layernorm,
            self.self_attn,
        )
        return self._sublayer(
            streams,
            self.hc_ffn_fn,
            self.hc_ffn_scale_lanes,
            self.hc_ffn_base,
            self.post_attention_layernorm,
            self.mlp,
        )


class RBLNGlm5Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        assert config.mhc
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("GLM-5.3 on RBLN runs at tensor parallel size 1")
        quant_config = vllm_config.quant_config
        self.n = config.mhc_num_residual_streams

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNGlm5DecoderLayer(
                config, int(prefix.rsplit(".", 1)[1]), quant_config, prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RBLNRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        batch, tokens, hidden = x.shape
        streams = x.unsqueeze(1).expand(batch, self.n, tokens, hidden).contiguous()
        # islice, not a slice: slicing a ModuleList builds a new one each call, whose
        # modules dynamo then lifts as graph inputs instead of constant weights.
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(streams)
        return self.norm(streams.mean(dim=1))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Layers not built here: the MTP head, or past a
            # VLLM_RBLN_NUM_HIDDEN_LAYERS cut.
            if name.startswith("layers.") and not (
                self.start_layer <= int(name.split(".", 2)[1]) < self.end_layer
            ):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.expand_hc_scales()
            if isinstance(layer.self_attn, RBLNGlm5DSA):
                layer.self_attn.indexer.fold_hadamard()
            else:
                layer.self_attn.fuse_low_rank()
        return loaded_params


class RBLNGlm5NextForCausalLM(nn.Module):
    """The text model of ``Glm5NextForConditionalGeneration`` checkpoints."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "model.visual.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = RBLNGlm5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
