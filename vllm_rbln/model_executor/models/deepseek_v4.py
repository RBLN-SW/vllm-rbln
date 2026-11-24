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

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import (
    RBLNDeepseekV4BlockFp8LinearMethod,
    e8m0_to_bf16,
)
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.deepseek_v4 import (
    DSV4_INDEX_HEAD_DIM,
    DSV4_KV8_ROW_BYTES,
    RBLNDeepseekV4RingBackend,
    RBLNDeepseekV4StateSpec,
    dsv4_paged_backend,
)
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec

# Cache names distinguish the six possible buffers in each decoder layer.
SLOT_SWA, SLOT_CMP, SLOT_IDX_K, SLOT_IDX_SCALE, SLOT_STATE, SLOT_IDX_STATE = range(6)
DSV4_NUM_CACHE_SLOTS = 6


def _compress_ratio(config: PretrainedConfig, layer_id: int) -> int:
    ratios = config.compress_ratios
    if layer_id >= config.num_hidden_layers:
        # Truncating target layers must not move MTP ratios away from the tail.
        layer_id += (
            len(ratios) - config.num_nextn_predict_layers - config.num_hidden_layers
        )
    return int(ratios[layer_id])


class _RBLNDeepseekV4Cache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self) -> None: ...


class RBLNDeepseekV4RingCache(_RBLNDeepseekV4Cache):
    """A per-request ring: ``[slot, window, width]`` (the SWA ring or a compressor
    state)."""

    def __init__(
        self, prefix: str, window: int, width: int, dtype: torch.dtype
    ) -> None:
        super().__init__(prefix)
        self.window = window
        self.width = width
        self.dtype = dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        speculative = vllm_config.speculative_config
        drafts = speculative.num_speculative_tokens if speculative is not None else 0
        capacity = 1 << (self.window + drafts - 1).bit_length()
        prefix_caching = vllm_config.cache_config.enable_prefix_caching
        checkpoints = 2 if prefix_caching else 0
        return RBLNDeepseekV4StateSpec(
            num_state_slots=(1 + checkpoints)
            * vllm_config.scheduler_config.max_num_seqs
            + 1,
            prefix_caching=prefix_caching,
            is_draft=prefix_caching
            and extract_layer_index(self.prefix, num_attn_module=DSV4_NUM_CACHE_SLOTS)
            // DSV4_NUM_CACHE_SLOTS
            >= vllm_config.model_config.hf_text_config.num_hidden_layers,
            block_size=capacity,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
        )

    def get_attn_backend(self) -> type[RBLNDeepseekV4RingBackend]:
        return RBLNDeepseekV4RingBackend

    def inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RBLNFlashAttentionMetadata]:
        metadata = get_forward_context().attn_metadata[self.prefix]
        ring = _resolve_kv_cache(metadata, self.prefix)
        slots = metadata.block_tables.reshape(-1, 1).to(torch.int32)
        query_len = metadata.cache_offsets - metadata.cache_seq_lens
        return ring, slots, query_len.to(torch.int32), metadata


def _paged_block_size(vllm_config: VllmConfig, compress_ratio: int) -> int:
    block_size = vllm_config.cache_config.block_size
    entries = 64 if compress_ratio == 128 else 256
    unit = entries * compress_ratio
    if block_size % unit == 0:
        return block_size
    if unit % block_size == 0:
        return unit
    raise ValueError(
        f"DeepSeek-V4: --block-size {block_size} and the "
        f"ratio-{compress_ratio} cache's "
        f"{unit}-token unit ({entries} entries a block) do not divide"
    )


class RBLNDeepseekV4PagedCache(_RBLNDeepseekV4Cache):
    """Compressed KV8 entries or fp8 indexer keys, grouped by token block size."""

    def __init__(
        self, prefix: str, compress_ratio: int, width: int, dtype: torch.dtype
    ) -> None:
        super().__init__(prefix)
        self.compress_ratio = compress_ratio
        self.width = width
        self.dtype = dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = _paged_block_size(vllm_config, self.compress_ratio)
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
        )

    def get_attn_backend(self):
        return dsv4_paged_backend(self.compress_ratio)

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(cache, block_table)."""
        metadata = get_forward_context().attn_metadata[self.prefix]
        return _resolve_kv_cache(metadata, self.prefix), metadata.block_tables


class RBLNDeepseekV4AuxCache(_RBLNDeepseekV4Cache):
    def __init__(
        self, prefix: str, owner: RBLNDeepseekV4PagedCache, dtype: torch.dtype
    ) -> None:
        super().__init__(prefix)
        self.owner_prefix = owner.prefix
        self.owner_ratio = owner.compress_ratio
        self.dtype = dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None

    def get_aux_cache_spec(self, vllm_config: VllmConfig) -> RBLNBlockAuxSpec:
        rows = _paged_block_size(vllm_config, self.owner_ratio) // self.owner_ratio
        return RBLNBlockAuxSpec(self.owner_prefix, (rows,), self.dtype)

    def get_attn_backend(self):
        return dsv4_paged_backend(self.owner_ratio)

    def inputs(self) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata[self.owner_prefix]
        return _resolve_kv_cache(metadata, self.prefix)


_TID2EID_LANES = 64


class RBLNDeepseekV4Compressor(nn.Module):
    """Gated pooling of ``compress_ratio`` tokens into one entry, then RMSNorm + RoPE
    at the block's first position."""

    def __init__(
        self,
        config: PretrainedConfig,
        compress_ratio: int,
        head_dim: int,
        state_slot: int,
        rope: DeepseekV4ScalingRotaryEmbedding,
        prefix: str,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.coff = 2 if compress_ratio == 4 else 1
        width = self.coff * head_dim
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, width, dtype=torch.float32), requires_grad=False
        )
        # bf16 in the checkpoint (unquantized); kv | gate fused.
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            config.hidden_size,
            [width, width],
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,  # the pooling runs in fp32
            prefix=f"{prefix}.fused_wkv_wgate",
            disable_tp=True,
        )
        self.norm = RMSNorm(head_dim, config.rms_norm_eps)
        self.rope = rope
        self.state_cache = RBLNDeepseekV4RingCache(
            prefix=f"{prefix}.state.{state_slot}",
            window=self.coff * compress_ratio,
            width=2 * width,
            # Preserve projection values across partial compression groups.
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        """x [B, L, H] -> [B, ceil(L / r), head_dim] candidates (normed, RoPE'd)."""
        kv_score, _ = self.fused_wkv_wgate(x.float())
        kv, score = kv_score.chunk(2, dim=-1)
        state, slot, query_len, _ = self.state_cache.inputs()
        pooled = torch.ops.rbln_custom_ops.deepseek_v4_compressor(
            kv.float().contiguous(),
            score.float().contiguous(),
            self.ape,
            state,
            seq_idx,
            query_len,
            slot,
            self.compress_ratio,
        )
        entries = self.norm(pooled.to(x.dtype))
        # Candidate j is entry seq_idx // r + j; it is rotated at its first token.
        r = self.compress_ratio
        j = torch.arange(entries.shape[1], device=x.device, dtype=seq_idx.dtype)
        positions = (seq_idx // r + j[None, :]) * r
        # Padded candidates can extend beyond max_model_len; their rotation is unused.
        positions = positions.minimum((seq_idx + query_len - 1).clamp_min(0))
        return self.rope(positions.long(), entries)[0]


class RBLNDeepseekV4Indexer(nn.Module):
    """Lightning indexer of a CSA layer over its own compressed keys (fp8, V3.2
    format).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        rope: DeepseekV4ScalingRotaryEmbedding,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        assert self.head_dim == DSV4_INDEX_HEAD_DIM
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.rope = rope
        self.compressor = RBLNDeepseekV4Compressor(
            config,
            4,
            self.head_dim,
            SLOT_IDX_STATE,
            rope,
            f"{prefix}.compressor",
        )
        self.k_cache = RBLNDeepseekV4PagedCache(
            f"{prefix}.k_cache.{SLOT_IDX_K}",
            4,
            self.head_dim,
            torch.float8_e4m3fn,
        )
        self.k_scale = RBLNDeepseekV4AuxCache(
            f"{prefix}.k_scale.{SLOT_IDX_SCALE}",
            self.k_cache,
            torch.float16,
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        seq_idx: torch.Tensor,
        query_len: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, _ = self.wq_b(qr)
        q, _ = self.rope(positions, q.view(batch, seq_len, self.n_heads, self.head_dim))
        k_cur = self.compressor(x, seq_idx)
        weights, _ = self.weights_proj(x)
        weights = weights.float() * self.weight_scale
        k_cache, block_table = self.k_cache.inputs()
        k_scale = self.k_scale.inputs()
        return torch.ops.rbln_custom_ops.sparse_attn_glm5_indexer(
            q.transpose(1, 2).contiguous(),
            k_cur.contiguous(),
            k_cache,
            k_scale,
            weights.contiguous(),
            seq_idx,
            query_len,
            block_table,
            self.topk,
            None,
            4,
        )


class RBLNDeepseekV4Attention(nn.Module):
    """MQA over SWA + compressed entries with a sink; low-rank q, grouped low-rank o."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "RBLN DeepSeek-V4 implements the KV8 kernels only; run with "
                f"--kv-cache-dtype fp8 (got {cache_dtype!r})"
            )
        tp_size = get_tensor_model_parallel_world_size()
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        assert self.n_groups % tp_size == 0
        self.n_local_groups = self.n_groups // tp_size
        self.window = config.sliding_window
        self.eps = config.rms_norm_eps
        self.compress_ratio = _compress_ratio(config, layer_id)
        assert self.compress_ratio in (0, 4, 128), self.compress_ratio

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_a",
        )
        self.register_buffer(  # filled by finalize_wo_a() after loading
            "wo_a_t",
            torch.empty(
                1,
                self.n_local_groups,
                self.n_heads * self.head_dim // self.n_groups,
                self.o_lora_rank,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
        )

        self.rope = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_dim,
            max_position_embeddings=vllm_config.model_config.max_model_len,
            compress_ratio=self.compress_ratio,
        )
        self.swa_cache = RBLNDeepseekV4RingCache(
            prefix=f"{prefix}.swa_cache.{SLOT_SWA}",
            window=self.window,
            width=DSV4_KV8_ROW_BYTES,
            dtype=torch.uint8,
        )
        self.compressor = None
        self.cmp_cache = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = RBLNDeepseekV4Compressor(
                config,
                self.compress_ratio,
                self.head_dim,
                SLOT_STATE,
                self.rope,
                f"{prefix}.compressor",
            )
            self.cmp_cache = RBLNDeepseekV4PagedCache(
                f"{prefix}.cmp_cache.{SLOT_CMP}",
                self.compress_ratio,
                DSV4_KV8_ROW_BYTES,
                torch.uint8,
            )
            if self.compress_ratio == 4:
                self.indexer = RBLNDeepseekV4Indexer(
                    config, self.rope, quant_config, f"{prefix}.indexer"
                )
        # a Python float: a compile-time constant every RSD node holds
        self.scale = self.head_dim**-0.5

    def finalize_wo_a(self) -> None:
        weight = self.wo_a.weight.detach().to("cpu")
        method = self.wo_a.quant_method
        if isinstance(method, RBLNDeepseekV4BlockFp8LinearMethod):
            block_n, block_k = (int(v) for v in method.weight_block_size)
            n, k = weight.shape
            inv = self.wo_a.weight_scale_inv.detach().to("cpu")
            if inv.dtype in (torch.float8_e8m0fnu, torch.uint8):
                inv = e8m0_to_bf16(inv)
            scale = inv.to(torch.float32).repeat_interleave(block_n, dim=0)[:n]
            # bf16 product, the same rounding as the in-graph dequantized_weight it
            # replaces
            bf = torch.bfloat16
            weight = (
                weight.view(n, k // block_k, block_k).to(bf) * scale.to(bf)[:, :, None]
            ).view(n, k)
        weight = weight.to(torch.bfloat16).view(
            self.n_local_groups, self.o_lora_rank, -1
        )
        self.wo_a_t = (
            weight.transpose(1, 2).unsqueeze(0).contiguous().to(self.wo_a.weight.device)
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv, _ = self.fused_wqa_wkv(x)
        q_a, kv = qkv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(q_a)
        q, _ = self.wq_b(qr)
        q = q.view(batch, seq_len, self.n_local_heads, self.head_dim)
        qf = q.float()
        q = (qf * torch.rsqrt(qf.square().mean(dim=-1, keepdim=True) + self.eps)).to(
            x.dtype
        )
        q, _ = self.rope(positions, q)
        kv, _ = self.rope(positions, self.kv_norm(kv))  # [B, L, 512]

        swa, swa_slot, query_len, metadata = self.swa_cache.inputs()
        seq_idx = metadata.cache_seq_lens.to(torch.int32)
        common = (
            q.transpose(1, 2).contiguous(),
            kv.contiguous(),
            swa,
            swa_slot,
            self.attn_sink,
            self.scale,
            seq_idx,
            query_len,
        )
        if self.compress_ratio:
            assert self.compressor is not None and self.cmp_cache is not None
            kv_cmp = self.compressor(x, seq_idx).contiguous()
            cmp_cache, cmp_block_table = self.cmp_cache.inputs()
            compressed = (self.compress_ratio, kv_cmp, cmp_cache, cmp_block_table)
            if self.indexer is not None:  # CSA: the indexer's top-k entries
                topk_index = self.indexer(x, qr, positions, seq_idx, query_len)
                o = torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4_csa(
                    *common, *compressed, topk_index
                )
            else:  # HCA: every completed entry
                o = torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4_hca(
                    *common, *compressed
                )
        else:
            o = torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4_swa(*common)
        # o [B, H, L, D]
        if seq_len > 1:
            o, _ = self.rope(positions, o, inverse=True, head_major=True)
            o = o.transpose(1, 2)
        else:
            # Per-batch decode positions need token-major rotary lowering at L=1.
            o, _ = self.rope(positions, o.transpose(1, 2), inverse=True)
        o = o.reshape(batch, seq_len, self.n_local_groups, -1)
        o = torch.matmul(o.transpose(1, 2), self.wo_a_t.to(o.dtype))
        out, _ = self.wo_b(o.transpose(1, 2).flatten(2))
        return out


class RBLNDeepseekV4MLP(nn.Module):
    """Shared expert: SwiGLU with the gate / up clamp."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
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
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        self.limit = float(config.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.float().chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        out, _ = self.down_proj((F.silu(gate) * up).to(x.dtype))
        return out


class RBLNDeepseekV4MoE(nn.Module):
    def __init__(self, vllm_config: VllmConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        self.top_k = config.num_experts_per_tok
        self.is_hash = layer_id < config.num_hash_layers

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        if self.is_hash:
            self.gate.e_score_correction_bias = None
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(config.vocab_size, _TID2EID_LANES, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.gate.tid2eid = None

        self.shared_experts = RBLNDeepseekV4MLP(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config,
            f"{prefix}.shared_experts",
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            swiglu_limit=config.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        expert_ids = None
        if self.is_hash:
            expert_ids = F.embedding(input_ids.reshape(-1), self.gate.tid2eid)
        out = self.experts(
            hidden_states=hidden_states,
            router=lambda x: self.gate(x.float())[0],
            expert_ids=expert_ids,
        )
        out = out * self.routed_scaling_factor + self.shared_experts(hidden_states)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.to(hidden_states.dtype)


class RBLNDeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        layer_id = int(prefix.split(sep=".")[-1])
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_mult = 2.0
        self.attn = RBLNDeepseekV4Attention(vllm_config, layer_id, f"{prefix}.attn")
        self.ffn = RBLNDeepseekV4MoE(vllm_config, layer_id, f"{prefix}.ffn")
        self.attn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        for name, shape in (
            ("hc_attn_fn", (mix_hc, hc_dim)),
            ("hc_ffn_fn", (mix_hc, hc_dim)),
            ("hc_attn_base", (mix_hc,)),
            ("hc_ffn_base", (mix_hc,)),
            ("hc_attn_scale", (3,)),
            ("hc_ffn_scale", (3,)),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False
                ),
            )
        # Per-mix expansions of hc_*_scale, filled by finalize_hc_scales() after
        # loading.
        for name in ("hc_attn_scale_vec", "hc_ffn_scale_vec"):
            self.register_buffer(
                name, torch.empty(mix_hc, dtype=torch.float32), persistent=False
            )

    def finalize_hc_scales(self) -> None:
        n = self.hc_mult
        lanes = torch.tensor([n, n, n * n])
        for kind in ("attn", "ffn"):
            scale = getattr(self, f"hc_{kind}_scale").cpu()
            getattr(self, f"hc_{kind}_scale_vec").copy_(scale.repeat_interleave(lanes))

    def _sublayer(self, streams, fn, scale, base, norm, sublayer) -> torch.Tensor:
        # The compiler cuts the mhc_* kernels over the chiplets, by tokens in
        # prefill and by batch rows in decode, so the streams stay split between
        # sublayers (fp32 holding bf16 values) and only the sublayer input is
        # gathered.
        ops = torch.ops.rbln_custom_ops
        streams_f32 = streams.float()
        logits = ops.mhc_mix(streams_f32, fn, scale, base, eps=self.norm_eps)
        x = ops.mhc_read(streams_f32, logits, eps=self.hc_eps).to(norm.weight.dtype)
        y = sublayer(norm(x))
        return ops.mhc_write(
            y.float(),
            streams_f32,
            logits,
            post_mult=self.hc_post_mult,
            iters=self.hc_iters,
            eps=self.hc_eps,
        ).to(streams.dtype)

    def forward(
        self, positions: torch.Tensor, streams: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """streams [B, n, T, H] -> [B, n, T, H]."""
        streams = self._sublayer(
            streams,
            self.hc_attn_fn,
            self.hc_attn_scale_vec,
            self.hc_attn_base,
            self.attn_norm,
            lambda h: self.attn(positions, h),
        )
        return self._sublayer(
            streams,
            self.hc_ffn_fn,
            self.hc_ffn_scale_vec,
            self.hc_ffn_base,
            self.ffn_norm,
            lambda h: self.ffn(h, input_ids),
        )


def _load_deepseek_v4_weights(
    module: nn.Module,
    config: PretrainedConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
    layers: list[nn.Module],
) -> set[str]:
    """Load target or MTP decoder weights using names relative to ``module``."""
    stacked_params_mapping = [
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
        ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
    ]
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    n_local_heads = config.num_attention_heads // tp_size
    params_dict = dict(module.named_parameters())
    expert_mapping = fused_moe_make_expert_params_mapping(
        module,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=config.n_routed_experts,
    )
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if ".mtp." in name or name.startswith("mtp.") or ".mtp" in name.split(".")[0]:
            continue
        if ".experts." in name and ".shared_experts." not in name:
            # Packed e2m1 (I8) and e8m0 scales are byte containers: keep the bytes.
            if loaded_weight.dtype in (torch.int8, torch.float8_e8m0fnu):
                loaded_weight = loaded_weight.view(torch.uint8)
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, module) or mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(
                    param, loaded_weight, mapped, shard_id=shard_id, expert_id=expert_id
                )
                loaded_params.add(mapped)
                break
            continue
        for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            mapped = name.replace(weight_name, param_name)
            if is_pp_missing_parameter(mapped, module) or mapped not in params_dict:
                break
            param = params_dict[mapped]
            param.weight_loader(param, loaded_weight, stacked_shard_id)
            loaded_params.add(mapped)
            break
        else:
            if is_pp_missing_parameter(name, module) or name not in params_dict:
                continue
            param = params_dict[name]
            if name.endswith("attn_sink"):
                start = tp_rank * n_local_heads
                param.data.copy_(loaded_weight[start : start + n_local_heads])
            elif name.endswith("tid2eid"):
                param.data.zero_()
                param.data[:, : loaded_weight.shape[1]].copy_(
                    loaded_weight.to(torch.bfloat16)
                )
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
    for layer in layers:
        if isinstance(layer, PPMissingLayer):
            continue
        layer.finalize_hc_scales()
        layer.attn.finalize_wo_a()
    return loaded_params


def _hc_head_hidden(
    flat, hc_mult, fn, scale, base, norm_eps, hc_eps, norm
) -> torch.Tensor:
    """Reduce flat pre-hc-head streams [..., n * H] and normalize to [..., H]."""
    lead = flat.shape[:-1]
    hidden = flat.shape[-1] // hc_mult
    rows = flat.reshape(1, -1, hc_mult * hidden)  # [1, N, n * H]
    x = rows.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    z_pre = F.linear(x, fn) * scale + base
    streams = rows.reshape(1, -1, hc_mult, hidden).transpose(1, 2)  # [1, n, N, H]
    out = torch.ops.rbln_custom_ops.hc_stream_reduce(streams.float(), z_pre, eps=hc_eps)
    return norm(out.to(flat.dtype)).reshape(*lead, hidden)


class RBLNDeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        # PP boundaries carry all mHC streams as [B, T, n * H].
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["streams"], config.hc_mult * config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNDeepseekV4DecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, self.norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            h = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            batch, tokens, hidden = h.shape
            streams = (
                h.unsqueeze(1).expand(batch, self.hc_mult, tokens, hidden).contiguous()
            )
        else:
            assert intermediate_tensors is not None
            flat_in = intermediate_tensors["streams"]
            batch, tokens, _ = flat_in.shape
            hidden = flat_in.shape[-1] // self.hc_mult
            streams = (
                flat_in.view(batch, tokens, self.hc_mult, hidden)
                .transpose(1, 2)
                .contiguous()
            )

        input_dtype = streams.dtype
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(positions, streams, input_ids)

        streams = streams.to(input_dtype)
        flat = streams.transpose(1, 2).reshape(batch, tokens, self.hc_mult * hidden)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"streams": flat})
        return flat

    def head_hidden(self, flat_streams: torch.Tensor) -> torch.Tensor:
        """[..., n * H] flat streams -> [..., H]: the hc head and the final norm."""
        return _hc_head_hidden(
            flat_streams,
            self.hc_mult,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm_eps,
            self.hc_eps,
            self.norm,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_deepseek_v4_weights(self, self.config, weights, list(self.layers))


class RBLNDeepseekV4ForCausalLM(nn.Module, SupportsPP):
    """DeepSeek-V4 (Flash) for causal language modeling."""

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        indexer_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
        if indexer_dtype != "fp8":
            raise ValueError(
                "DeepSeek-V4 on RBLN supports indexer_kv_dtype='fp8' only; "
                f"got {indexer_dtype!r}."
            )
        chunks = vllm_config.additional_config.prefill_chunk_size
        assert chunks is not None
        scheduler = vllm_config.scheduler_config
        prefill_tokens = max(chunks)
        for cap in (
            scheduler.max_num_scheduled_tokens,
            scheduler.long_prefill_token_threshold or None,
        ):
            if cap is not None:
                prefill_tokens = min(prefill_tokens, cap)
        graph_tokens = min(chunk for chunk in chunks if chunk >= prefill_tokens)
        # Multi-entry compression currently groups rows from chunk offset zero.
        # Single-entry graphs can resume inside a compression block using the ring.
        if any(
            ratio and graph_tokens > ratio and prefill_tokens % ratio
            for ratio in config.compress_ratios
        ):
            raise ValueError(
                f"DeepSeek-V4 prefill steps of {prefill_tokens} tokens in a "
                f"{graph_tokens}-token graph do not preserve the compression boundary. "
                "Choose aligned prefill_chunk_size, max_num_scheduled_tokens and "
                "long_prefill_token_threshold; unaligned multi-entry compression "
                "is not supported by the RBLN compiler yet."
            )
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = RBLNDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
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
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # hidden_states: the flat pre-hc_head streams of the sampled rows
        return self.logits_processor(
            self.lm_head, self.model.head_hidden(hidden_states)
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (n, w) for n, w in weights if not n.startswith(("model.mtp.", "mtp."))
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class RBLNDeepseekV4MTPLayer(RBLNDeepseekV4DecoderLayer):
    """Apply the checkpoint's SWA MTP layer to projected streams and embeddings."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        hidden = config.hidden_size
        self.enorm = RMSNorm(hidden, self.norm_eps)
        self.hnorm = RMSNorm(hidden, self.norm_eps)
        self.e_proj = ReplicatedLinear(
            hidden,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            hidden,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.h_proj",
        )
        hc_dim = self.hc_mult * hidden
        for name, shape in (
            ("hc_head_fn", (self.hc_mult, hc_dim)),
            ("hc_head_base", (self.hc_mult,)),
            ("hc_head_scale", (1,)),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False
                ),
            )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(hidden, self.norm_eps)

    def forward_mtp(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        previous: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """inputs_embeds [B, T, H], previous [B, T, n * H] (flat pre-hc_head streams)
        -> [B, T, n * H], this layer's flat pre-hc_head streams.
        """
        batch, tokens, hidden = inputs_embeds.shape
        n = self.hc_mult
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        e = self.e_proj(self.enorm(inputs_embeds))[0]  # [B, T, H]
        h = self.h_proj(self.hnorm(previous.reshape(batch, tokens, n, hidden)))[0]
        streams = (h + e.unsqueeze(2)).transpose(1, 2).contiguous()  # [B, n, T, H]
        streams = self.forward(positions, streams, input_ids).to(inputs_embeds.dtype)
        return streams.transpose(1, 2).reshape(batch, tokens, n * hidden)

    def head_hidden(self, flat_streams: torch.Tensor) -> torch.Tensor:
        return _hc_head_hidden(
            flat_streams,
            self.hc_mult,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.norm_eps,
            self.hc_eps,
            self.shared_head.norm,
        )


class RBLNDeepseekV4MTPModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): RBLNDeepseekV4MTPLayer(
                    vllm_config=vllm_config, prefix=f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_deepseek_v4_weights(
            self, self.config, weights, list(self.layers.values())
        )


class RBLNDeepseekV4MTP(nn.Module):
    """MTP draft model sharing the target lm_head and non-PP embeddings."""

    has_own_lm_head = False
    hf_to_vllm_mapper = RBLNDeepseekV4ForCausalLM.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = RBLNDeepseekV4MTPModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self._mtp_layers = list(self.model.layers.values())
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        layer = self._mtp_layers[spec_step_index % len(self._mtp_layers)]
        return layer.forward_mtp(positions, inputs_embeds, hidden_states, input_ids)

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_index: int = 0
    ) -> torch.Tensor | None:
        layer = self._mtp_layers[spec_step_index % len(self._mtp_layers)]
        return self.logits_processor(self.lm_head, layer.head_hidden(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        start = self.config.num_hidden_layers

        own_embed = get_pp_group().world_size > 1

        def mtp_only():
            for name, weight in weights:
                if own_embed and name == "embed.weight":
                    yield name, weight
                    continue
                if not name.startswith("mtp."):
                    continue
                _, idx, rest = name.split(".", 2)
                if rest.startswith("norm."):
                    rest = "shared_head." + rest
                yield f"layers.{start + int(idx)}.{rest}", weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mtp_only(), mapper=self.hf_to_vllm_mapper)
