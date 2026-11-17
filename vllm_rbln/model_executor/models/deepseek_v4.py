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
from rebel.ops.torch_custom_ops.mhc import mhc_post, mhc_pre
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
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.deepseek_v4 import (
    DSV4_INDEX_HEAD_DIM,
    DSV4_KV8_ROW_BYTES,
    RBLNDeepseekV4RingBackend,
    RBLNDeepseekV4StateSpec,
    dsv4_paged_backend,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec
from vllm_rbln.v1.worker.utils import pipeline_adjusted_layer_index

# Cache names reserve six slots per decoder layer. The runner assigns compact
# layer_index values when binding the caches actually present on each PP stage.
SLOT_SWA, SLOT_CMP, SLOT_IDX_K, SLOT_IDX_SCALE, SLOT_STATE, SLOT_IDX_STATE = range(6)
DSV4_NUM_CACHE_SLOTS = 6


def _compress_ratio(config: PretrainedConfig, layer_id: int) -> int:
    ratios = config.compress_ratios
    num_mtp = getattr(config, "num_nextn_predict_layers", 0) or 0
    if num_mtp and layer_id >= config.num_hidden_layers:
        # Truncating target layers must not move MTP ratios away from the tail.
        layer_id = len(ratios) - num_mtp + (layer_id - config.num_hidden_layers)
    return int(ratios[layer_id]) if 0 <= layer_id < len(ratios) else 0


def _register_layer(module: nn.Module, prefix: str) -> None:
    vllm_config = get_current_vllm_config()
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")
    compilation_config.static_forward_context[prefix] = module
    module.layer_index = pipeline_adjusted_layer_index(
        prefix,
        vllm_config.model_config,
        vllm_config.parallel_config,
        DSV4_NUM_CACHE_SLOTS,
    )


def _metadata(prefix: str):
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return attn_metadata[prefix]
    return attn_metadata


class RBLNDeepseekV4RingCache(nn.Module, AttentionLayerBase):
    """A per-request ring: ``[slot, window, width]`` (the SWA ring or a compressor
    state)."""

    def __init__(
        self,
        prefix: str,
        window: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.window = window
        self.width = width
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register_layer(self, prefix)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        speculative = vllm_config.speculative_config
        drafts = speculative.num_speculative_tokens if speculative is not None else 0
        capacity = 1 << (self.window + drafts - 1).bit_length()
        return RBLNDeepseekV4StateSpec(
            num_state_slots=vllm_config.scheduler_config.max_num_seqs + 1,
            block_size=capacity,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[RBLNDeepseekV4RingBackend]:
        return RBLNDeepseekV4RingBackend


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


class RBLNDeepseekV4PagedCache(nn.Module, AttentionLayerBase):
    """Compressed KV8 entries or fp8 indexer keys, grouped by token block size."""

    def __init__(
        self,
        prefix: str,
        compress_ratio: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.width = width
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register_layer(self, prefix)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = _paged_block_size(vllm_config, self.compress_ratio)
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=self.width,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self):
        return dsv4_paged_backend(self.compress_ratio)


class RBLNDeepseekV4AuxCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        prefix: str,
        owner: RBLNDeepseekV4PagedCache,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.owner_prefix = owner.prefix
        self.owner_ratio = owner.compress_ratio
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        _register_layer(self, prefix)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None

    def get_aux_cache_spec(self, vllm_config: VllmConfig) -> RBLNBlockAuxSpec:
        rows = _paged_block_size(vllm_config, self.owner_ratio) // self.owner_ratio
        return RBLNBlockAuxSpec(self.owner_prefix, (rows,), self.dtype)

    def forward(self) -> None: ...

    def get_attn_backend(self):
        return dsv4_paged_backend(self.owner_ratio)

    def inputs(self) -> torch.Tensor:
        return _resolve_kv_cache(_metadata(self.owner_prefix), self.layer_index)


def _ring_inputs(cache: RBLNDeepseekV4RingCache) -> tuple[torch.Tensor, ...]:
    """(ring tensor [N, window, width], slot [B, 1], query_len [B, 1]) of a ring
    cache.
    """
    metadata = _metadata(cache.prefix)
    ring = _resolve_kv_cache(metadata, cache.layer_index)
    slots = metadata.block_tables.reshape(-1, 1).to(torch.int32)
    query_len = metadata.cache_offsets - metadata.cache_seq_lens
    return ring, slots, query_len.to(torch.int32)


def _paged_inputs(cache: RBLNDeepseekV4PagedCache) -> tuple[torch.Tensor, ...]:
    metadata = _metadata(cache.prefix)
    return _resolve_kv_cache(metadata, cache.layer_index), metadata.block_tables


class RBLNRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(
            xf.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (self.weight * xf).to(x.dtype)


class RBLNDeepseekV4Rope(nn.Module):
    """Upstream RoPE tables with rank-4 forward/inverse rotations for RBLN."""

    def __init__(
        self, config: PretrainedConfig, compress_ratio: int, max_pos: int
    ) -> None:
        super().__init__()
        self.rope_dim = config.qk_rope_head_dim
        device = torch.get_default_device()
        # Use upstream's YaRN parameters and frequency calculation for both legacy
        # flat and per-layer-type RoPE configurations.
        with torch.device("cpu"):
            rope = build_deepseek_v4_rope(
                config,
                head_dim=config.head_dim,
                rope_head_dim=self.rope_dim,
                max_position_embeddings=max_pos,
                compress_ratio=compress_ratio,
            )
            cos, sin = rope.cos_sin_cache[:max_pos].chunk(2, dim=-1)
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cache", cos.to(device), persistent=False)
        self.register_buffer("sin_cache", sin.to(device), persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """x [B, L, ..., D]; positions [B, L]. Rotates x[..., -rope_dim:]."""
        if x.dim() == 3:
            # The compiler fuses a rotation over a 4- or 5-D tensor only; a head-less
            # [B, L, D] one (kv, compressed entries) would leave its channel shuffle
            # on the host.
            return self.forward(x.unsqueeze(2), positions, inverse).squeeze(2)
        batch, seq_len = positions.shape[0], positions.shape[1]
        flat = positions.flatten()
        cos = self.cos_cache.index_select(0, flat).view(batch, seq_len, -1)
        sin = self.sin_cache.index_select(0, flat).view(batch, seq_len, -1)
        for _ in range(x.dim() - 3):
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        nope, rope = x[..., : -self.rope_dim], x[..., -self.rope_dim :].float()
        if inverse:
            rot = rope * cos - rotate_gptj(rope) * sin
        else:
            rot = rope * cos + rotate_gptj(rope) * sin
        return torch.cat([nope, rot.to(x.dtype)], dim=-1)


_TID2EID_LANES = 64


def _hc_scale_vec(scale: torch.Tensor, hc: int) -> torch.Tensor:
    """[3] (pre, post, comb) scales -> [(2 + hc) * hc], one per mix; built on the
    host.
    """
    s = scale.detach().to("cpu", torch.float32)
    vec = torch.cat([s[0].expand(hc), s[1].expand(hc), s[2].expand(hc * hc)])
    return vec.to(scale.device)


class RBLNDeepseekV4Compressor(nn.Module):
    """Gated pooling of ``compress_ratio`` tokens into one entry, then RMSNorm + RoPE
    at the block's first position. The pooling and its per-request ring live in
    ``rbln_custom_ops.deepseek_v4_compressor``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        compress_ratio: int,
        head_dim: int,
        state_slot: int,
        rope: RBLNDeepseekV4Rope,
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
        self.norm = RBLNRMSNorm(head_dim, config.rms_norm_eps)
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
        # Padded candidates can extend beyond max_model_len; their rotation is unused.
        positions = positions.minimum((seq_idx + query_len - 1).clamp_min(0))
        return self.rope(entries, positions.long())


class RBLNDeepseekV4Indexer(nn.Module):
    """Lightning indexer of a CSA layer over its own compressed keys (fp8, V3.2
    format).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        rope: RBLNDeepseekV4Rope,
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
        q = self.rope(q.view(batch, seq_len, self.n_heads, self.head_dim), positions)
        k_cur = self.compressor(x, seq_idx)
        weights, _ = self.weights_proj(x)
        weights = weights.float() * self.weight_scale
        k_cache, block_table = _paged_inputs(self.k_cache)
        k_scale = self.k_scale.inputs()  # [NB, E] f16: one scale a compressed key
        # GLM-5.3's kpool indexer has the V4 compressed-key indexer's contract (store
        # the step's candidates, pick among e < (p + 1) // 4, ascending, -1 tail).
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
        ropes: dict[bool, RBLNDeepseekV4Rope],
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
        self.q_norm = RBLNRMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RBLNRMSNorm(self.head_dim, self.eps)
        # Grouped: o [.., g, H * D / g] x wo_a[g] -> [.., g, o_lora]; the weight is
        # read dequantized and multiplied per group (a bmm).
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

        max_pos = vllm_config.model_config.max_model_len
        # Share tables within this model, without retaining another model's device.
        compressed = bool(self.compress_ratio)
        if compressed not in ropes:
            ropes[compressed] = RBLNDeepseekV4Rope(config, self.compress_ratio, max_pos)
        self.rope = ropes[compressed]
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
        """wo_a as the [1, g, d, r] bf16 batched-matmul weight, dequantized once on
        the host after loading (as MLA's W_UV): an einsum lowers to a host op, and
        an in-graph fp8 dequant + transpose in front of the matmul fails the fp8
        weight-format annotation.
        """
        weight = self.wo_a.weight.detach().to("cpu")
        method = self.wo_a.quant_method
        if hasattr(method, "dequantized_weight"):
            block_n, block_k = (int(v) for v in method.weight_block_size)
            n, k = weight.shape
            if hasattr(self.wo_a, "weight_scale_t"):  # folded already: [K / 128, N]
                scale = self.wo_a.weight_scale_t.detach().to("cpu", torch.float32).t()
            else:
                # Before folding: the checkpoint's [N / 128, K / 128].
                from vllm_rbln.model_executor.layers.quantization.deepseek_v4 import (
                    e8m0_to_bf16,
                )

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
        q = self.rope(q, positions)
        kv = self.rope(self.kv_norm(kv), positions)  # [B, L, 512]

        swa, swa_slot, query_len = _ring_inputs(self.swa_cache)
        seq_idx = _metadata(self.swa_cache.prefix).cache_seq_lens.to(torch.int32)
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
            cmp_cache, cmp_block_table = _paged_inputs(self.cmp_cache)
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
        o = self.rope(o.transpose(1, 2), positions, inverse=True)  # [B, L, H, D]
        o = o.reshape(batch, seq_len, self.n_local_groups, -1)
        o = torch.matmul(o.transpose(1, 2), self.wo_a_t.to(o.dtype))
        out, _ = self.wo_b(o.transpose(1, 2).flatten(2))
        return out


# ------------------------------------------------------------------------------
# MoE
# ------------------------------------------------------------------------------


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
            # bf16 rows padded to 64 lanes: looked up as a device embedding (an int32
            # table gather lowers to one host index per token). Expert ids < 256 are
            # exact in bf16.
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

    def _router(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        # Hash layers: the expert ids are a bf16 table lookup of the local tokens;
        # under DP the runner carries them through its hidden-state gather
        # (`gather_extra`) and hands the gathered rows back as `extra`.
        eids_local = (
            F.embedding(input_ids.reshape(-1), self.gate.tid2eid)
            if self.is_hash
            else None
        )

        def route(x: torch.Tensor, extra: torch.Tensor | None = None):
            logits, _ = self.gate(x.float())
            if not self.is_hash:
                return logits
            eids = eids_local if extra is None else extra
            # bf16 ids as they are: the router masks with a compare, no int cast
            return logits, eids[:, : self.top_k]

        if self.is_hash:
            route.gather_extra = eids_local  # type: ignore[attr-defined]
        return route

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        out = self.experts(
            hidden_states=hidden_states, router=self._router(hidden_states, input_ids)
        )
        out = out * self.routed_scaling_factor + self.shared_experts(hidden_states)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.to(hidden_states.dtype)


# ------------------------------------------------------------------------------
# decoder
# ------------------------------------------------------------------------------


class RBLNDeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        ropes: dict[bool, RBLNDeepseekV4Rope],
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        layer_id = int(prefix.split(sep=".")[-1])
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = RBLNDeepseekV4Attention(
            vllm_config, layer_id, f"{prefix}.attn", ropes
        )
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
        self.hc_attn_scale_vec = _hc_scale_vec(self.hc_attn_scale, self.hc_mult)
        self.hc_ffn_scale_vec = _hc_scale_vec(self.hc_ffn_scale, self.hc_mult)

    def _sublayer(
        self, streams, fn, scale, base, norm, sublayer, *args
    ) -> torch.Tensor:
        """Run a sublayer with token-cut mHC for prefill chunks."""
        x, state = mhc_pre(
            streams,
            fn,
            scale,
            base,
            rms_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            iters=self.hc_iters,
        )
        x = sublayer(norm(x.to(streams.dtype)), *args)
        return mhc_post(
            x, state, post_mult=2.0, iters=self.hc_iters, hc_eps=self.hc_eps
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
        if hasattr(layer, "finalize_hc_scales"):
            layer.finalize_hc_scales()
        if hasattr(layer, "attn"):
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
        ropes: dict[bool, RBLNDeepseekV4Rope] = {}
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNDeepseekV4DecoderLayer(
                vllm_config=vllm_config, prefix=prefix, ropes=ropes
            ),
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
        # Slicing ModuleList during tracing lifts weights into graph inputs.
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(positions, streams, input_ids)
        # PP and MTP consume pre-hc-head streams; compute_logits applies the head.
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
        # the MTP head's weights are RBLNDeepseekV4MTP's
        weights = (
            (n, w) for n, w in weights if not n.startswith(("model.mtp.", "mtp."))
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# ------------------------------------------------------------------------------
# MTP (multi-token prediction) draft head
# ------------------------------------------------------------------------------


class RBLNDeepseekV4MTPLayer(RBLNDeepseekV4DecoderLayer):
    """Apply the checkpoint's SWA MTP layer to projected streams and embeddings."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        ropes: dict[bool, RBLNDeepseekV4Rope],
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, ropes=ropes)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        hidden = config.hidden_size
        self.enorm = RBLNRMSNorm(hidden, self.norm_eps)
        self.hnorm = RBLNRMSNorm(hidden, self.norm_eps)
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
        self.shared_head.norm = RBLNRMSNorm(hidden, self.norm_eps)

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
        streams = self.forward(positions, streams, input_ids)
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
        ropes: dict[bool, RBLNDeepseekV4Rope] = {}
        self.layers = nn.ModuleDict(
            {
                str(idx): RBLNDeepseekV4MTPLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                    ropes=ropes,
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
        # The draft wrapper maps MTP names to layers.{num_hidden_layers + i}.*.
        return _load_deepseek_v4_weights(
            self, self.config, weights, list(self.layers.values())
        )


class RBLNDeepseekV4MTP(nn.Module):
    """MTP draft model sharing the target lm_head and non-PP embeddings.

    Previous hidden states are the target's flat pre-hc-head streams.
    """

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
        # Upstream expects one tensor: flat streams serve both the logits and the next
        # drafting step's previous hidden states.
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
            # The target's embedding lives on the first PP stage, so the drafter on
            # the last stage loads its own copy. The lm_head remains shared.
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
