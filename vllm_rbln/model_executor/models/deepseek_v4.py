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

"""DeepSeek-V4 (Flash) for RBLN. KV8 only.

Vendored from upstream ``vllm.models.deepseek_v4`` (vLLM 0.26), which binds FlashMLA /
FlashInfer / DeepGEMM / tilelang kernels at import. This copy keeps the checkpoint's module tree
and weight names (through upstream's name mapper) and computes the way the reference
``inference/model.py`` does:

* hidden states stay ``[B, L, hc, H]`` (the ``hc_mult`` = 4 hyper-connection copies) between
  layers; mHC pre / post / head mixing (Sinkhorn) is plain torch;
* attention is MQA with one 512-channel entry (448 nope + 64 rope) per token, K == V. Every
  layer keeps the last ``sliding_window`` raw tokens; CSA layers (compress ratio 4) add the
  indexer's top-k compressed entries, HCA layers (128) all compressed entries; one softmax with
  a per-head sink. The three pieces are ``rbln_custom_ops.deepseek_v4_compressor`` /
  ``sparse_attn_deepseek_v4_indexer`` / ``sparse_attn_deepseek_v4``, which read and write their
  caches (graph inputs, resolved from the attention metadata);
* the indexer keeps the DeepSeek-V3.2 fp8 key cache; no Hadamard, no FP4 simulation;
* the output de-rotates its rope channels and goes through the grouped ``wo_a`` (bmm) + ``wo_b``;
* MoE goes through the RBLN ``MoERunner``: sqrt-softplus scores with the routing bias, or the
  per-token expert table on the leading hash-routed layers; routed scaling and the shared
  expert are applied here.

The MTP head is not modeled; its weights are skipped.
"""

import math
from collections.abc import Iterable
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
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
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_rbln.logger import init_logger
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.deepseek_v4 import (
    DSV4_INDEX_HEAD_DIM,
    DSV4_KV8_ROW_BYTES,
    RBLNDeepseekV4RingBackend,
    dsv4_paged_backend,
)
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

logger = init_logger(__name__)

# Cache slots of a decoder layer (the second integer of the cache's layer name, which
# `extract_layer_index` reads as the sub-index); a layer carries only the ones its compress
# ratio needs, and `_cache_ordinal` maps (layer, slot) to the position in the runner's
# compacted cache list.
# A compressed-cache block holds whole 64-entry chunks on each of the 4 RSD shards.
DSV4_MIN_BLOCK_ENTRIES = 256

SLOT_SWA, SLOT_CMP, SLOT_IDX_K, SLOT_IDX_SCALE, SLOT_STATE, SLOT_IDX_STATE = range(6)
DSV4_NUM_CACHE_SLOTS = 6


def _layer_slots(compress_ratio: int) -> tuple[int, ...]:
    if compress_ratio == 4:
        return tuple(range(DSV4_NUM_CACHE_SLOTS))
    if compress_ratio == 128:
        return (SLOT_SWA, SLOT_CMP, SLOT_STATE)
    return (SLOT_SWA,)


def _compress_ratio(config: PretrainedConfig, layer_id: int) -> int:
    ratios = config.compress_ratios
    return int(ratios[layer_id]) if layer_id < len(ratios) else 0


def _cache_ordinal(layer_id: int, slot: int) -> int:
    """Position of (layer, slot) in the runner's cache list: sorted by layer * 6 + slot over
    the caches this pipeline stage's layers own."""
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    config = model_config.hf_text_config
    start, _ = model_config.get_layers_start_end_indices(vllm_config.parallel_config)
    ordinal = sum(len(_layer_slots(_compress_ratio(config, i))) for i in range(start, layer_id))
    return ordinal + _layer_slots(_compress_ratio(config, layer_id)).index(slot)


def _register_layer(module: nn.Module, prefix: str) -> None:
    compilation_config = get_current_vllm_config().compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = module


def _metadata(prefix: str):
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return attn_metadata[prefix]
    return attn_metadata


# --------------------------------------------------------------------------------------------
# caches
# --------------------------------------------------------------------------------------------


class RBLNDeepseekV4RingCache(nn.Module, AttentionLayerBase):
    """A per-request ring: ``[slot, window, width]`` (the SWA ring or a compressor state).

    ``RBLNSlidingWindowSpec`` budgets K + V, so it is declared with half the row width and
    the backend reshapes the page to one ``width``-wide row per position.
    """

    def __init__(
        self, prefix: str, layer_id: int, slot: int, window: int, width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert width % 2 == 0
        self.prefix = prefix
        self.window = window
        self.width = width
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register_layer(self, prefix)
        self.layer_index = _cache_ordinal(layer_id, slot)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return RBLNSlidingWindowSpec(
            block_size=self.window,
            num_kv_heads=1,
            head_size=self.width // 2,
            dtype=self.dtype,
            sliding_window=self.window,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[RBLNDeepseekV4RingBackend]:
        return RBLNDeepseekV4RingBackend


class RBLNDeepseekV4PagedCache(nn.Module, AttentionLayerBase):
    """A paged cache of compressed entries (``block_size / r`` a block): the compressed KV
    (768-byte KV8 rows), the indexer key (fp8) or its scale (f16)."""

    def __init__(
        self, prefix: str, layer_id: int, slot: int, compress_ratio: int, width: int,
        dtype: torch.dtype, blocks_per_key_block: int = 1,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        # > 1 for the indexer scale cache: its block spans that many key blocks' tokens, so its
        # page equals the key cache's and vLLM's page unification keeps the two in step (the
        # indexer converter views it as [NB * n, E] and addresses key block b as b * n).
        self.blocks_per_key_block = blocks_per_key_block
        self.width = width
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register_layer(self, prefix)
        self.layer_index = _cache_ordinal(layer_id, slot)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # The CP kernels cut a block into 4 shards of whole 64-entry chunks: at least 256
        # entries a block (HCA: 32768 tokens).
        block_size = max(vllm_config.cache_config.block_size, DSV4_MIN_BLOCK_ENTRIES * self.compress_ratio)
        return MLAAttentionSpec(
            block_size=block_size * self.blocks_per_key_block,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self):
        return dsv4_paged_backend(self.compress_ratio)


def _ring_inputs(cache: RBLNDeepseekV4RingCache) -> tuple[torch.Tensor, ...]:
    """(ring tensor, slot [B, 1], query_len [B, 1]) of a ring cache."""
    metadata = _metadata(cache.prefix)
    ring = _resolve_kv_cache(metadata, cache.layer_index)
    # The hybrid allocator unifies page sizes by growing a smaller spec's block size, so a
    # ring's page may hold k rings: view them as k slots and take a request's first.
    rings_per_page = ring.numel() // (ring.shape[0] * cache.window * cache.width)
    assert ring.numel() == ring.shape[0] * rings_per_page * cache.window * cache.width, (
        f"{cache.prefix}: page of {ring.numel() // ring.shape[0]} elements is not whole rings "
        f"of {cache.window} x {cache.width}"
    )
    ring = ring.view(ring.shape[0] * rings_per_page, cache.window, cache.width)
    # [B, 1] like cache_offsets: a flat [B] device input only the host reads has no device
    # consumer, and the compiler then leaves it without a per-node address (ISSUES B10).
    slots = metadata.local_block_tables.reshape(-1, 1).to(torch.int32) * rings_per_page
    query_len = metadata.cache_offsets - metadata.cache_seq_lens
    return ring, slots, query_len.to(torch.int32)


def _paged_inputs(cache: RBLNDeepseekV4PagedCache) -> tuple[torch.Tensor, ...]:
    """(cache tensor, block_table [B, P]) of a paged cache. Its seq_lens is not read: the ops
    take the step position from the ring caches, and an input the graph captures but no op
    uses is pruned from the compiled module (the runtime then sees too many inputs)."""
    metadata = _metadata(cache.prefix)
    return _resolve_kv_cache(metadata, cache.layer_index), metadata.block_tables


# --------------------------------------------------------------------------------------------
# small modules
# --------------------------------------------------------------------------------------------


class RBLNRMSNorm(nn.Module):
    """``w * x * rsqrt(mean(x^2) + eps)`` in fp32 (the checkpoint stores w in bf16)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * xf).to(x.dtype)


def _yarn_inv_freq(
    rope_dim: int, base: float, factor: float, original_len: int, beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """The reference ``precompute_freqs_cis`` frequencies (YaRN when original_len > 0)."""
    freqs = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if original_len <= 0:
        return freqs

    def correction_dim(rot: float) -> float:
        return rope_dim * math.log(original_len / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    smooth = 1 - ramp
    return freqs / factor * (1 - smooth) + freqs * smooth


@cache
def _rope_table(
    rope_dim: int, max_pos: int, base: float, factor: float, original_len: int,
    beta_fast: int, beta_slow: int,
) -> torch.Tensor:
    inv_freq = _yarn_inv_freq(rope_dim, base, factor, original_len, beta_fast, beta_slow)
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)  # [max_pos, rope_dim]


class RBLNDeepseekV4Rope(nn.Module):
    """Interleaved (GPT-J pair) RoPE on the last ``rope_dim`` channels.

    Compressed layers use YaRN with ``compress_rope_theta``; pure-SWA layers the base
    ``rope_theta`` without YaRN (the reference's ``original_seq_len = 0``).

    Written as ``x * cos + rotate_gptj(x) * sin`` over pair-duplicated cos / sin (as the RBLN
    RotaryEmbedding patch) so the compiler can lower the rotation to its rotary primitive; the
    nope channels pass around it (slice + concat). The inverse rotation (-theta) is
    ``x * cos - rotate_gptj(x) * sin`` over the same tables: the subtract is the pattern the
    compiler maps to the primitive's inverse rotation.
    """

    def __init__(self, config: PretrainedConfig, compress_ratio: int, max_pos: int) -> None:
        super().__init__()
        self.rope_dim = config.qk_rope_head_dim
        scaling = getattr(config, "rope_scaling", None) or {}
        if compress_ratio:
            base = float(config.compress_rope_theta)
            original_len = int(scaling.get("original_max_position_embeddings", 0))
        else:
            base, original_len = float(config.rope_theta), 0
        table = _rope_table(
            self.rope_dim,
            max_pos,
            base,
            float(scaling.get("factor", 1.0)),
            original_len,
            int(scaling.get("beta_fast", 32)),
            int(scaling.get("beta_slow", 1)),
        )
        half = self.rope_dim // 2
        # [max_pos, rope_dim] each, every frequency repeated for its (even, odd) channel pair
        cos = table[:, :half].repeat_interleave(2, dim=-1)
        sin = table[:, half:].repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """x [B, L, ..., D]; positions [B, L]. Rotates x[..., -rope_dim:]."""
        batch, seq_len = positions.shape[0], positions.shape[1]
        flat = positions.flatten()
        cos = self.cos_cache.index_select(0, flat).view(batch, seq_len, -1).to(x.dtype)
        sin = self.sin_cache.index_select(0, flat).view(batch, seq_len, -1).to(x.dtype)
        for _ in range(x.dim() - 3):
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        nope, rope = x[..., : -self.rope_dim], x[..., -self.rope_dim :]
        if inverse:
            rot = rope * cos - rotate_gptj(rope) * sin
        else:
            rot = rope * cos + rotate_gptj(rope) * sin
        return torch.cat([nope, rot], dim=-1)


_TID2EID_LANES = 64


def _hc_scale_vec(scale: torch.Tensor, hc: int) -> torch.Tensor:
    """[3] (pre, post, comb) scales -> [(2 + hc) * hc], one per mix; built on the host."""
    s = scale.detach().to("cpu", torch.float32)
    vec = torch.cat([s[0].expand(hc), s[1].expand(hc), s[2].expand(hc * hc)])
    return vec.to(scale.device)


def _hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc: int,
    iters: int, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The reference ``hc_split_sinkhorn``: mixes [..., (2 + hc) * hc] -> pre [.., hc],
    post [.., hc], comb [.., hc, hc] (Sinkhorn-normalized). ``hc_scale`` is the per-mix
    expansion of the checkpoint's 3 scales (``_hc_scale_vec``): one multiply, one constant."""
    z = mixes * hc_scale + hc_base
    pre = torch.sigmoid(z[..., :hc]) + eps
    post = 2 * torch.sigmoid(z[..., hc : 2 * hc])
    comb = z[..., 2 * hc :]
    comb = comb.unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# --------------------------------------------------------------------------------------------
# compressor / indexer / attention
# --------------------------------------------------------------------------------------------


class RBLNDeepseekV4Compressor(nn.Module):
    """Gated pooling of ``compress_ratio`` tokens into one entry, then RMSNorm + RoPE at the
    block's first position. The pooling and its per-request ring live in
    ``rbln_custom_ops.deepseek_v4_compressor``."""

    def __init__(
        self, config: PretrainedConfig, layer_id: int, compress_ratio: int, head_dim: int,
        state_slot: int, rope: RBLNDeepseekV4Rope, prefix: str,
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
        self.norm = RBLNRMSNorm(head_dim, config.rms_norm_eps)
        self.rope = rope
        self.state_cache = RBLNDeepseekV4RingCache(
            prefix=f"{prefix}.state.{state_slot}",
            layer_id=layer_id,
            slot=state_slot,
            window=self.coff * compress_ratio,
            width=2 * width,
            # bf16: the device pools in dlf16 either way
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        """x [B, L, H] -> [B, ceil(L / r), head_dim] candidates (normed, RoPE'd)."""
        kv_score, _ = self.fused_wkv_wgate(x.float())
        kv, score = kv_score.chunk(2, dim=-1)
        state, slot, query_len = _ring_inputs(self.state_cache)
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
        return self.rope(entries, positions.long())


class RBLNDeepseekV4Indexer(nn.Module):
    """Lightning indexer of a CSA layer over its own compressed keys (fp8, V3.2 format)."""

    def __init__(
        self, config: PretrainedConfig, layer_id: int, rope: RBLNDeepseekV4Rope,
        quant_config: QuantizationConfig | None, prefix: str,
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
            config.hidden_size, self.n_heads, bias=False, quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.rope = rope
        self.compressor = RBLNDeepseekV4Compressor(
            config, layer_id, 4, self.head_dim, SLOT_IDX_STATE, rope, f"{prefix}.compressor"
        )
        self.k_cache = RBLNDeepseekV4PagedCache(
            f"{prefix}.k_cache.{SLOT_IDX_K}", layer_id, SLOT_IDX_K, 4, self.head_dim,
            torch.float8_e4m3fn,
        )
        self.k_scale = RBLNDeepseekV4PagedCache(
            f"{prefix}.k_scale.{SLOT_IDX_SCALE}", layer_id, SLOT_IDX_SCALE, 4, 1,
            torch.float16,
            # one f16 scale a key row of head_dim fp8 bytes: the key cache's page
            blocks_per_key_block=self.head_dim // 2,
        )

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor,
        seq_idx: torch.Tensor, query_len: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, _ = self.wq_b(qr)
        q = self.rope(q.view(batch, seq_len, self.n_heads, self.head_dim), positions)
        k_cur = self.compressor(x, seq_idx)
        weights, _ = self.weights_proj(x)
        weights = weights.float() * self.weight_scale
        k_cache, block_table = _paged_inputs(self.k_cache)
        k_scale = _resolve_kv_cache(_metadata(self.k_scale.prefix), self.k_scale.layer_index)
        return torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4_indexer(
            q.transpose(1, 2).contiguous(),
            k_cur.contiguous(),
            k_cache,
            k_scale,
            weights.contiguous(),
            seq_idx,
            query_len,
            block_table,
            self.topk,
            4,
        )


class RBLNDeepseekV4Attention(nn.Module):
    """MQA over SWA + compressed entries with a sink; low-rank q, grouped low-rank o."""

    def __init__(
        self, vllm_config: VllmConfig, layer_id: int, prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        cache_dtype = vllm_config.cache_config.cache_dtype
        if not (cache_dtype and cache_dtype.startswith("fp8")):
            raise NotImplementedError(
                "RBLN DeepSeek-V4 implements the KV8 kernels only; run with "
                f"--kv-cache-dtype fp8 (got {cache_dtype!r})"
            )
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
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
        self.q_norm = RBLNRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RBLNRMSNorm(self.head_dim, self.eps)
        # Grouped: o [.., g, H * D / g] x wo_a[g] -> [.., g, o_lora]; the weight is read
        # dequantized and multiplied per group (a bmm).
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
                1, self.n_local_groups, self.n_heads * self.head_dim // self.n_groups,
                self.o_lora_rank, dtype=torch.bfloat16,
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

        max_pos = vllm_config.model_config.max_model_len
        self.rope = RBLNDeepseekV4Rope(config, self.compress_ratio, max_pos)
        self.swa_cache = RBLNDeepseekV4RingCache(
            prefix=f"{prefix}.swa_cache.{SLOT_SWA}",
            layer_id=layer_id,
            slot=SLOT_SWA,
            window=self.window,
            width=DSV4_KV8_ROW_BYTES,
            dtype=torch.int8,
        )
        self.compressor = None
        self.cmp_cache = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = RBLNDeepseekV4Compressor(
                config, layer_id, self.compress_ratio, self.head_dim, SLOT_STATE, self.rope,
                f"{prefix}.compressor",
            )
            self.cmp_cache = RBLNDeepseekV4PagedCache(
                f"{prefix}.cmp_cache.{SLOT_CMP}", layer_id, SLOT_CMP, self.compress_ratio,
                DSV4_KV8_ROW_BYTES, torch.int8,
            )
            if self.compress_ratio == 4:
                self.indexer = RBLNDeepseekV4Indexer(
                    config, layer_id, self.rope, quant_config, f"{prefix}.indexer"
                )
        self.scale_tensor = torch.tensor(self.head_dim**-0.5, dtype=torch.float32)

    def finalize_wo_a(self) -> None:
        """wo_a as the [1, g, d, r] bf16 batched-matmul weight, dequantized once on the host after
        loading (as MLA's W_UV): an einsum lowers to a host op, and an in-graph fp8 dequant +
        transpose in front of the matmul fails the fp8 weight-format annotation."""
        weight = self.wo_a.weight.detach().to("cpu")
        method = self.wo_a.quant_method
        if hasattr(method, "dequantized_weight"):
            block_n, block_k = (int(v) for v in method.weight_block_size)
            n, k = weight.shape
            if hasattr(self.wo_a, "weight_scale"):  # folded already: [N, K / 128]
                scale = self.wo_a.weight_scale.detach().to("cpu", torch.float32)
            else:  # load_weights runs before the fold: the checkpoint's [N / 128, K / 128]
                from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import e8m0_to_bf16

                inv = self.wo_a.weight_scale_inv.detach().to("cpu")
                if inv.dtype in (torch.float8_e8m0fnu, torch.uint8):
                    inv = e8m0_to_bf16(inv)
                scale = inv.to(torch.float32).repeat_interleave(block_n, dim=0)[:n]
            # bf16 product, the same rounding as the in-graph dequantized_weight it replaces
            bf = torch.bfloat16
            weight = (
                weight.view(n, k // block_k, block_k).to(bf) * scale.to(bf)[:, :, None]
            ).view(n, k)
        weight = weight.to(torch.bfloat16).view(self.n_local_groups, self.o_lora_rank, -1)
        self.wo_a_t = weight.transpose(1, 2).unsqueeze(0).contiguous().to(self.wo_a.weight.device)

    def _wo_a(self, o: torch.Tensor) -> torch.Tensor:
        """o [B, L, g, H * D / g] -> [B, L, g * o_lora]: [B, g, L, d] @ [1, g, d, r] -> [B, g, L, r]."""
        out = torch.matmul(o.transpose(1, 2), self.wo_a_t.to(o.dtype))
        return out.transpose(1, 2).flatten(2)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv, _ = self.fused_wqa_wkv(x)
        q_a, kv = qkv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(q_a)
        q, _ = self.wq_b(qr)
        q = q.view(batch, seq_len, self.n_local_heads, self.head_dim)
        qf = q.float()
        q = (qf * torch.rsqrt(qf.square().mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        q = self.rope(q, positions)
        kv = self.rope(self.kv_norm(kv), positions)  # [B, L, 512]

        swa, swa_slot, query_len = _ring_inputs(self.swa_cache)
        seq_idx = _metadata(self.swa_cache.prefix).seq_lens.to(torch.int32)
        common = (
            q.transpose(1, 2).contiguous(),
            kv.contiguous(),
            swa,
            swa_slot,
            self.attn_sink,
            self.scale_tensor.to(x.device),
            seq_idx,
            query_len,
        )
        if self.compress_ratio:
            kv_cmp = self.compressor(x, seq_idx).contiguous()
            cmp_cache, cmp_block_table = _paged_inputs(self.cmp_cache)
            topk_index = None
            if self.indexer is not None:
                topk_index = self.indexer(x, qr, positions, seq_idx, query_len)
            o = torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4(
                *common,
                self.compress_ratio,
                kv_cmp,
                cmp_cache,
                cmp_block_table,
                topk_index,
            )
        else:
            o = torch.ops.rbln_custom_ops.sparse_attn_deepseek_v4_swa(*common)
        # o [B, H, L, D]
        o = self.rope(o.transpose(1, 2), positions, inverse=True)  # [B, L, H, D]
        o = o.reshape(batch, seq_len, self.n_local_groups, -1)
        out, _ = self.wo_b(self._wo_a(o))
        return out


# --------------------------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------------------------


class RBLNDeepseekV4MLP(nn.Module):
    """Shared expert: SwiGLU with the gate / up clamp."""

    def __init__(
        self, config: PretrainedConfig, intermediate_size: int,
        quant_config: QuantizationConfig | None, prefix: str,
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
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
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
            # bf16 rows padded to 64 lanes: looked up as a device embedding (an int32 table
            # gather lowers to one host index per token). Expert ids < 256 are exact in bf16.
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
        self.experts = FusedMoE(
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

    def _router(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        # Hash layers: the expert ids are a bf16 table lookup of the local tokens; under DP the
        # runner carries them through its hidden-state gather (`gather_extra`) and hands the
        # gathered rows back as `extra`.
        eids_local = (
            F.embedding(input_ids.reshape(-1), self.gate.tid2eid) if self.is_hash else None
        )

        def route(x: torch.Tensor, extra: torch.Tensor | None = None):
            logits, _ = self.gate(x.float())
            if not self.is_hash:
                return logits
            eids = eids_local if extra is None else extra
            return logits, eids[:, : self.top_k].to(torch.int32)

        if self.is_hash:
            route.gather_extra = eids_local
        return route

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.experts(
            hidden_states=hidden_states, router=self._router(hidden_states, input_ids)
        )
        out = out * self.routed_scaling_factor + self.shared_experts(hidden_states)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.to(hidden_states.dtype)


# --------------------------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------------------------


class RBLNDeepseekV4DecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        layer_id = int(prefix.split(sep=".")[-1])
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = RBLNDeepseekV4Attention(vllm_config, layer_id, f"{prefix}.attn")
        self.ffn = RBLNDeepseekV4MoE(vllm_config, layer_id, f"{prefix}.ffn")
        self.attn_norm = RBLNRMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RBLNRMSNorm(config.hidden_size, self.norm_eps)
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
                nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False),
            )
        # Per-mix expansions of hc_*_scale, filled by finalize_hc_scales() after loading.
        for name in ("hc_attn_scale_vec", "hc_ffn_scale_vec"):
            self.register_buffer(name, torch.empty(mix_hc, dtype=torch.float32), persistent=False)

    def finalize_hc_scales(self) -> None:
        self.hc_attn_scale_vec = _hc_scale_vec(self.hc_attn_scale, self.hc_mult)
        self.hc_ffn_scale_vec = _hc_scale_vec(self.hc_ffn_scale, self.hc_mult)

    def _hc_pre(self, x: torch.Tensor, fn, scale, base):
        # x [B, L, hc, H] -> y [B, L, H], post [B, L, hc], comb [B, L, hc, hc]
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes, scale, base, self.hc_mult, self.hc_iters, self.hc_eps
        )
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)
        return y.to(x.dtype), post, comb

    @staticmethod
    def _hc_post(x, residual, post, comb):
        # x [B, L, H], residual [B, L, hc, H] -> [B, L, hc, H]
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + (
            comb.unsqueeze(-1) * residual.float().unsqueeze(-2)
        ).sum(dim=2)
        return y.to(x.dtype)

    def forward(
        self, positions: torch.Tensor, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        h, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale_vec, self.hc_attn_base)
        h = self.attn(positions, self.attn_norm(h))
        x = self._hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale_vec, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(h), input_ids)
        return self._hc_post(h, residual, post, comb)


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
        if get_pp_group().world_size > 1:
            raise NotImplementedError("RBLN DeepSeek-V4 does not support PP yet")

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNDeepseekV4DecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RBLNRMSNorm(config.hidden_size, self.norm_eps)
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

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return (pre.unsqueeze(-1) * x.float()).sum(dim=2).to(x.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, L, hc, H]
        for layer in self.layers[self.start_layer : self.end_layer]:
            h = layer(positions, h, input_ids)
        return self.norm(self._hc_head(h))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
        n_local_heads = self.config.num_attention_heads // tp_size
        params_dict = dict(self.named_parameters())
        expert_mapping = self.get_expert_mapping()
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
                    if is_pp_missing_parameter(mapped, self) or mapped not in params_dict:
                        continue
                    param = params_dict[mapped]
                    param.weight_loader(
                        param, loaded_weight, mapped, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded_params.add(mapped)
                    break
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self) or mapped not in params_dict:
                    break
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if is_pp_missing_parameter(name, self) or name not in params_dict:
                    continue
                param = params_dict[name]
                if name.endswith("attn_sink"):
                    start = tp_rank * n_local_heads
                    param.data.copy_(loaded_weight[start : start + n_local_heads])
                elif name.endswith("tid2eid"):
                    param.data.zero_()
                    param.data[:, : loaded_weight.shape[1]].copy_(loaded_weight.to(torch.bfloat16))
                else:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                loaded_params.add(name)
        for layer in self.layers:
            if hasattr(layer, "finalize_hc_scales"):
                layer.finalize_hc_scales()
            if hasattr(layer, "attn"):
                layer.attn.finalize_wo_a()
        return loaded_params


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
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = RBLNDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

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
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["model.mtp.", "mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
