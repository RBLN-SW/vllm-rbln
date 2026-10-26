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

import math
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from rebel.ops.torch_custom_ops.kda import CHUNK as KDA_CHUNK
from rebel.ops.torch_custom_ops.kda import CONV_LANES
from rebel.ops.torch_custom_ops.sparse_attn_glm5 import hadamard_128
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
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
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)
from vllm.models.glm5next.common.multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, MLAAttentionSpec

from vllm_rbln.model_executor.models.interfaces import SupportsRBLNMultiModal
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)
from vllm_rbln.v1.attention.backends.glm5_next import (
    RBLNGlm5NextCacheBackend,
    RBLNGlm5NextIndexerBackend,
    RBLNGlm5NextStateMetadata,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec
from vllm_rbln.v1.worker.utils import (
    num_attn_module as rbln_num_attn_module,
)
from vllm_rbln.v1.worker.utils import pipeline_adjusted_layer_index

SLOT_DSA_LATENT, SLOT_DSA_INDEX_KEY, SLOT_DSA_INDEX_SCALE, SLOT_DSA_INDEX_TAIL = range(
    4
)
# The 3 taps before a step's first token, and room for one MTP draft.
KDA_CONV_RING = 4


class _RBLNGlm5NextCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        model_config = vllm_config.model_config
        self.layer_index = pipeline_adjusted_layer_index(
            prefix,
            model_config,
            vllm_config.parallel_config,
            rbln_num_attn_module(model_config, vllm_config.cache_config.cache_dtype),
        )

    def forward(self) -> None: ...


class RBLNGlm5NextStateCache(_RBLNGlm5NextCache):
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # TODO(RBLN): prefix caching. A state block of max_model_len tokens is never
        # cached; it needs mamba_cache_mode "align", a state checkpoint a block.
        # The extent of an fp8 MLA latent block: [block_size, kv_lora_rank] bytes.
        latent_block = (
            vllm_config.cache_config.block_size,
            vllm_config.model_config.hf_text_config.kv_lora_rank,
        )
        return MambaSpec(
            block_size=vllm_config.model_config.max_model_len,
            shapes=(latent_block,),
            dtypes=(torch.uint8,),
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        )

    def get_attn_backend(self) -> type[RBLNGlm5NextCacheBackend]:
        return RBLNGlm5NextCacheBackend

    def inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RBLNGlm5NextStateMetadata]:
        metadata = get_forward_context().attn_metadata[self.prefix]
        assert isinstance(metadata, RBLNGlm5NextStateMetadata), type(metadata)
        assert metadata.load_blocks is not None and metadata.store_blocks is not None
        state = _resolve_kv_cache(metadata, self.layer_index)
        return (
            state,
            metadata.load_blocks.view(-1),
            metadata.store_blocks.view(-1),
            metadata,
        )


class RBLNGlm5NextPagedCache(_RBLNGlm5NextCache):
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
        metadata = get_forward_context().attn_metadata[self.prefix]
        return (
            _resolve_kv_cache(metadata, self.layer_index),
            metadata.block_tables,
            metadata,
        )


# TODO(RBLN): PD. The NIXL connector refuses Mamba layers; the KDA states and the aux
# caches have to be registered as regions and moved with their blocks.
class RBLNGlm5NextAuxCache(_RBLNGlm5NextCache):
    def __init__(
        self,
        prefix: str,
        owner: str,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(prefix)
        self.owner = owner
        self.block_shape = block_shape
        self.dtype = dtype

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None

    def get_aux_cache_spec(self, vllm_config: VllmConfig) -> RBLNBlockAuxSpec:
        return RBLNBlockAuxSpec(self.owner, self.block_shape, self.dtype)

    def get_attn_backend(self) -> type[RBLNGlm5NextCacheBackend]:
        return RBLNGlm5NextCacheBackend

    def inputs(self) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata[self.owner]
        return _resolve_kv_cache(metadata, self.layer_index)


class _ConvWeight(nn.Module):
    def __init__(self, channels: int, taps: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, taps), requires_grad=False)
        self.weight.weight_loader = self._load

    @staticmethod
    def _load(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight.reshape(param.shape))


def _swiglu_clamped(
    gate: torch.Tensor, up: torch.Tensor, limit: float | None
) -> torch.Tensor:
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class RBLNGlm5NextMLP(nn.Module):
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
        gate, up = gate_up.chunk(2, dim=-1)
        x, _ = self.down_proj(_swiglu_clamped(gate, up, self.swiglu_limit))
        return x


class RBLNGlm5NextMoE(nn.Module):
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
        self.shared_experts = RBLNGlm5NextMLP(
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


class RBLNGlm5NextKDA(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
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
        self.register_buffer(
            "f_weight", torch.empty(proj, hidden_size), persistent=False
        )
        self.register_buffer(
            "g_weight", torch.empty(proj, hidden_size), persistent=False
        )
        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            proj, hidden_size, bias=False, prefix=f"{prefix}.o_proj"
        )
        # forward takes o_proj's columns a head group at a time: whole, unquantized.
        assert self.o_proj.tp_size == 1

        vllm_config = get_current_vllm_config()
        dtype = vllm_config.model_config.dtype
        # g head groups, each group's state filling one fp8 MLA latent block.
        block_size = vllm_config.cache_config.block_size
        latent_bytes = block_size * config.kv_lora_rank
        state_bytes = self.head_dim * self.head_dim * dtype.itemsize
        if latent_bytes % state_bytes or self.num_heads % (latent_bytes // state_bytes):
            raise ValueError(
                f"GLM-5.3: block_size {block_size} does not split the "
                f"{self.num_heads} KDA heads into states of a latent block "
                f"({latent_bytes} B, {state_bytes} B a head)"
            )
        self.group_heads = latent_bytes // state_bytes
        self.num_groups = self.num_heads // self.group_heads
        self.register_buffer(
            "conv_weight",
            torch.empty(3 * self.num_groups, proj // self.num_groups, taps),
            persistent=False,
        )
        self.states = nn.ModuleList(
            RBLNGlm5NextStateCache(f"{prefix}.attn.{j}") for j in range(self.num_groups)
        )
        self.positions_from = self.states[0].prefix
        self.conv_ring = RBLNGlm5NextAuxCache(
            f"{prefix}.attn.{self.num_groups}",
            self.states[0].prefix,
            (
                KDA_CONV_RING,
                3 * self.num_groups,
                proj // self.num_groups // CONV_LANES,
                CONV_LANES,
            ),
            dtype,
        )
        spec = vllm_config.speculative_config
        self.num_drafts = spec.num_speculative_tokens if spec is not None else 0
        self.drafts = None
        if self.num_drafts:
            assert taps - 1 + self.num_drafts <= KDA_CONV_RING, self.num_drafts
            self.drafts = nn.ModuleList(
                RBLNGlm5NextAuxCache(
                    f"{prefix}.attn.{self.num_groups + 1 + j}",
                    state.prefix,
                    (self.group_heads, self.num_drafts, 4, self.head_dim),
                    dtype,
                )
                for j, state in enumerate(self.states)
            )

    @torch.no_grad()
    def prepare_weights(self) -> None:
        for fused, a, b in (
            (self.f_weight, self.f_a_proj, self.f_b_proj),
            (self.g_weight, self.g_a_proj, self.g_b_proj),
        ):
            fused.copy_(
                (b.weight.cpu().float() @ a.weight.cpu().float()).to(fused.dtype)
            )
        q, k, v = (
            conv.weight.cpu().view(self.num_groups, -1, conv.weight.shape[-1])
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        stacked = torch.cat([torch.stack((q, k), dim=1).flatten(0, 1), v])
        self.conv_weight.copy_(stacked.to(self.conv_weight.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)

        group_states = [state.inputs() for state in self.states]
        metadata = group_states[0][3]
        positions = get_forward_context().attn_metadata[self.positions_from]
        seq_idx = positions.seq_lens.to(torch.int32)
        query_len = (positions.cache_offsets - positions.cache_seq_lens).to(torch.int32)
        has_initial_state = (seq_idx > 0).to(torch.int32).view(-1)
        conv_load, conv_store = group_states[0][1:3]

        qkv = tuple(
            proj(hidden_states)[0] for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        group_shape = (batch, tokens, self.group_heads, self.head_dim)
        group_channels = self.group_heads * self.head_dim
        q, k, v = qkv
        group = [
            slice(j * group_channels, (j + 1) * group_channels)
            for j in range(self.num_groups)
        ]
        inputs = [x[..., c] for c in group for x in (q, k)] + [v[..., c] for c in group]
        convolved = torch.ops.rbln_custom_ops.kda_conv(
            torch.stack(inputs, dim=2),
            self.conv_weight,
            self.conv_ring.inputs(),
            conv_load,
            seq_idx.view(-1),
            query_len.view(-1),
            has_initial_state,
            KDA_CONV_RING,
            2 * self.num_groups,
            conv_store,
        ).unbind(2)
        group_qkv = [
            tuple(
                x.view(group_shape).transpose(1, 2)
                for x in (
                    convolved[2 * j],
                    convolved[2 * j + 1],
                    convolved[2 * self.num_groups + j],
                )
            )
            for j in range(self.num_groups)
        ]

        f = F.linear(hidden_states, self.f_weight).view(shape)
        decay = self.A_log.exp().view(-1, 1) * (
            f.float() + self.dt_bias.view(shape[2:])
        )
        g = self.lower_bound * torch.sigmoid(decay)  # [B, T, H, D] f32
        beta = torch.sigmoid(self.b_proj(hidden_states)[0].float())  # [B, T, H]

        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2).to(hidden_states.dtype)
        pad = -tokens % KDA_CHUNK if metadata.is_prefill else 0
        if metadata.is_prefill:
            # Padding tokens carry the state through unchanged: no decay, no update.
            # TODO(RBLN): query_len - t > 0 goes through int16 (exact: query_len is
            # at most a prefill chunk) because the compiler runs an int32 broadcast or
            # a compare with a scalar int32 on the host only.
            ahead = query_len.to(torch.int16).view(-1, 1, 1).expand(-1, 1, tokens)
            ahead = ahead.to(torch.int32) - torch.arange(
                tokens, dtype=torch.int32, device=g.device
            ).view(1, 1, -1)
            valid = ahead.to(torch.int16) > 0
            g = g * valid.unsqueeze(-1)
            beta = beta * valid

        gate = F.linear(hidden_states, self.g_weight).view(shape)
        outs = []
        for j, ((state, load, store, _), (qj, kj, vj)) in enumerate(
            zip(group_states, group_qkv)
        ):
            channels = slice(j * group_channels, (j + 1) * group_channels)
            heads = slice(j * self.group_heads, (j + 1) * self.group_heads)
            gj, bj = g[:, heads], beta[:, heads]
            if pad:
                qj, kj, vj, gj = (F.pad(x, (0, 0, 0, pad)) for x in (qj, kj, vj, gj))
                bj = F.pad(bj, (0, pad))
            if metadata.is_prefill:
                o = torch.ops.rbln_custom_ops.kda_prefill(
                    qj, kj, vj, gj, bj, state, load, has_initial_state, None, store
                )[:, :, :tokens]
            elif self.drafts is None:
                o = torch.ops.rbln_custom_ops.kda_decode(
                    qj, kj, vj, gj, bj, state, load, has_initial_state, None, store
                )
            else:
                assert tokens == self.num_drafts + 1, tokens
                o = torch.ops.rbln_custom_ops.kda_verify(
                    qj,
                    kj,
                    vj,
                    gj,
                    bj,
                    state,
                    self.drafts[j].inputs(),
                    load,
                    metadata.num_accepted.view(-1),
                    has_initial_state,
                    None,
                    load,
                    store,
                    store,
                )
            o = o.transpose(1, 2)  # [B, T, h, D]
            o = self.o_norm(o) * torch.sigmoid(gate[:, :, heads].float()).to(o.dtype)
            outs.append(
                F.linear(o.reshape(batch, tokens, -1), self.o_proj.weight[:, channels])
            )
        return sum(outs[1:], outs[0])


class RBLNGlm5NextIndexer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        # The key cache holds a row a pool.
        assert config.index_kpool == RBLNGlm5NextIndexerBackend.compress_ratio
        kpool = config.index_kpool
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk // kpool
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
            torch.empty(kpool, self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )

        self.block_size = get_current_vllm_config().cache_config.block_size
        attn_prefix = prefix.rsplit(".indexer", 1)[0]
        self.key_cache = RBLNGlm5NextPagedCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_KEY}",
            self.head_dim,
            torch.float8_e4m3fn,
            kpool,
            RBLNGlm5NextIndexerBackend,
        )
        self.scale_cache = RBLNGlm5NextAuxCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_SCALE}",
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_KEY}",
            (self.block_size // kpool,),
            torch.float16,
        )
        self.tail_cache = RBLNGlm5NextAuxCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_TAIL}",
            f"{attn_prefix}.attn.{SLOT_DSA_LATENT}",
            (kpool, 2 * self.head_dim),
            torch.bfloat16,
        )

    @torch.no_grad()
    def fold_hadamard(self) -> None:
        weight = self.wq_b.weight
        w = weight.cpu().float().view(self.n_head, self.head_dim, -1)
        rotated = torch.einsum("ij,hjk->hik", hadamard_128().float(), w)
        weight.copy_(rotated.reshape(weight.shape).to(weight.dtype))

    def forward(self, hidden_states: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        q = self.wq_b(qr)[0].view(batch, tokens, self.n_head, self.head_dim)
        key = self.wk(hidden_states)[0]
        key = F.layer_norm(
            key.float(),
            (self.head_dim,),
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.eps,
        ).to(hidden_states.dtype)
        gate = F.linear(hidden_states, self.index_kpool_compress_gate)
        weights = (
            F.linear(hidden_states.float(), self.weights_proj.weight.float())
            * self.weights_scale
        )

        key_cache, key_table, metadata = self.key_cache.inputs()
        assert metadata.cache_offsets is not None
        assert metadata.cache_seq_lens is not None
        seq_idx = metadata.seq_lens.to(torch.int32)
        query_len = (metadata.cache_offsets - metadata.cache_seq_lens).to(torch.int32)
        k_cur = torch.ops.rbln_custom_ops.glm5_kpool_compress(
            key,
            gate,
            self.index_kpool_compress_ape,
            self.tail_cache.inputs(),
            seq_idx,
            query_len,
            key_table.view(batch, -1),
            self.block_size,
        )
        return torch.ops.rbln_custom_ops.sparse_attn_glm5_indexer(
            q.transpose(1, 2),
            k_cur,
            key_cache,
            self.scale_cache.inputs(),
            weights,
            seq_idx,
            query_len,
            key_table,
            self.topk,
        )


class RBLNGlm5NextDSA(nn.Module):
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
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
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
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
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
        self.indexer = RBLNGlm5NextIndexer(config, f"{prefix}.indexer")
        self.register_buffer(
            "_k_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False
        )
        self.latent_cache = RBLNGlm5NextPagedCache(
            f"{prefix}.attn.{SLOT_DSA_LATENT}",
            self.kv_lora_rank,
            torch.uint8,
            1,
            RBLNGlm5NextCacheBackend,
        )
        self.scale = self.qk_nope_head_dim**-0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        qr = self.q_a_layernorm(self.q_a_proj(hidden_states)[0])
        q = self.q_b_proj(qr)[0].view(
            batch, tokens, self.num_heads, self.qk_nope_head_dim
        )
        kv_c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states)[0])
        topk = self.indexer(hidden_states, qr)

        kv_b = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_uk, w_uv = kv_b.unsqueeze(0).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=2
        )
        q_mla = torch.matmul(q.transpose(1, 2), w_uk)  # [B, H, T, L]
        latent_cache, block_table, metadata = self.latent_cache.inputs()
        o = torch.ops.rbln_custom_ops.sparse_attn_glm5_mla(
            q_mla,
            kv_c,
            latent_cache,
            self.scale,
            metadata.seq_lens.to(torch.int32),
            block_table,
            topk,
            k_scale=self._k_scale,
            cache_dtype=torch.float8_e4m3fn,
            is_prompt_sorted=metadata.is_prompt_sorted,
        )
        o = torch.matmul(o, w_uv.transpose(2, 3))  # [B, H, T, V]
        o = o.transpose(1, 2).reshape(batch, tokens, -1)
        return self.o_proj(o)[0]


class RBLNGlm5NextDecoderLayer(nn.Module):
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
            self.self_attn: nn.Module = RBLNGlm5NextKDA(config, f"{prefix}.self_attn")
        else:
            assert config.layer_types[layer_idx] == "deepseek_sparse_attention"
            self.self_attn = RBLNGlm5NextDSA(
                config, quant_config, f"{prefix}.self_attn"
            )
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp: nn.Module = RBLNGlm5NextMoE(config, quant_config, f"{prefix}.mlp")
        else:
            self.mlp = RBLNGlm5NextMLP(
                config.hidden_size,
                config.intermediate_size,
                config.swiglu_limit,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
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
            # TODO(RBLN): the 3 scales spread over the mix's lanes at load
            # (expand_hc_scales) because the compiler runs a slice of a small weight as
            # a host op whose constant gets no buffer slot.
            self.register_buffer(
                f"hc_{kind}_scale_lanes",
                torch.empty(mix, dtype=torch.float32),
                persistent=False,
            )

    @torch.no_grad()
    def expand_hc_scales(self) -> None:
        n = self.n
        lanes = torch.tensor([n, n, n * n])
        for kind in ("attn", "ffn"):
            scale = getattr(self, f"hc_{kind}_scale").cpu()
            getattr(self, f"hc_{kind}_scale_lanes").copy_(
                scale.repeat_interleave(lanes)
            )

    def _sublayer(self, streams, fn, scale, base, norm, sublayer) -> torch.Tensor:
        batch, n, tokens, hidden = streams.shape
        flat = streams.transpose(1, 2).reshape(batch, tokens, n * hidden).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        z = F.linear(flat, fn) * scale + base
        z_pre, z_post = z[..., :n], z[..., n : 2 * n]
        comb = torch.ops.rbln_custom_ops.sinkhorn_normalize(
            z[..., 2 * n :],
            iters=self.sinkhorn_iters,
            eps=self.hc_eps,
        )
        streams_f32 = streams.float()
        x = torch.ops.rbln_custom_ops.hc_stream_reduce(
            streams_f32, z_pre, eps=self.hc_eps
        ).to(streams.dtype)
        x = sublayer(norm(x))
        return torch.ops.rbln_custom_ops.hc_residual_add(
            x.float(), streams_f32, z_post, comb, post_mult=self.post_mult
        ).to(streams.dtype)

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


def _load_glm5_next_weights(
    module: nn.Module,
    config: PretrainedConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
    keep,
) -> set[str]:
    stacked_params_mapping = [
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    expert_params_mapping = fused_moe_make_expert_params_mapping(
        module,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=config.n_routed_experts,
    )
    params_dict = dict(module.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if not keep(name):
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
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    return loaded_params


class RBLNGlm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        assert config.mhc
        quant_config = vllm_config.quant_config
        self.n = config.mhc_num_residual_streams

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNGlm5NextDecoderLayer(
                config, int(prefix.rsplit(".", 1)[1]), quant_config, prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        kda = [
            layer.self_attn
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if isinstance(layer.self_attn, RBLNGlm5NextKDA)
        ]
        for attn in kda:
            attn.positions_from = kda[0].states[0].prefix

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
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(streams)
        return self.norm(streams.mean(dim=1))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_glm5_next_weights(
            self,
            self.config,
            weights,
            lambda name: (
                not name.startswith("layers.")
                or self.start_layer <= int(name.split(".", 2)[1]) < self.end_layer
            ),
        )


class RBLNGlm5NextForCausalLM(nn.Module):
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
        self.model = RBLNGlm5NextModel(
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
        loaded_params = AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
        model = self.model
        for layer in islice(model.layers, model.start_layer, model.end_layer):
            layer.expand_hc_scales()
            if isinstance(layer.self_attn, RBLNGlm5NextDSA):
                layer.self_attn.indexer.fold_hadamard()
            else:
                layer.self_attn.prepare_weights()
        return loaded_params


# Patches per image the compiled vision encoder takes (64 x 64 grid, 896 x 896 px,
# 1024 image tokens).
# TODO(RBLN): image size buckets and video inputs.
VISION_PATCH_BUCKET = 4096


class _RBLNGlm5NextVisionEncoder(nn.Module):
    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.tower = tower
        attn = tower.blocks[0].attn
        self.num_heads = attn.num_attention_heads_per_partition
        self.head_dim = attn.head_dim
        self.swiglu_limit = tower.blocks[0].mlp.act_fn.swiglu_limit
        half = self.head_dim // 2
        rot = torch.zeros(self.head_dim, self.head_dim)
        for j in range(half):
            rot[j + half, j] = -1.0
            rot[j, j + half] = 1.0
        self.rot = rot.to(device=tower.device, dtype=tower.dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_bias: torch.Tensor,
    ) -> torch.Tensor:
        tower = self.tower
        n = pixel_values.shape[0]
        w = tower.patch_embed.proj.weight
        x = F.linear(
            pixel_values, w.reshape(w.shape[0], -1), tower.patch_embed.proj.bias
        )
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        scale = self.head_dim**-0.5
        for block in tower.blocks:
            attn = block.attn
            qkv = attn.qkv(block.norm1(x))[0]
            qkv = qkv.view(n, 3, self.num_heads, self.head_dim)
            q = attn.q_norm(qkv[:, 0])
            k = attn.k_norm(qkv[:, 1])
            v = qkv[:, 2]  # [N, heads, D]
            q = q * cos + torch.matmul(q, self.rot) * sin
            k = k * cos + torch.matmul(k, self.rot) * sin
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            scores = torch.matmul(q, k.transpose(1, 2)) * scale + key_bias
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
            x = x + attn.proj(out.transpose(0, 1).reshape(n, -1))[0]
            mlp = block.mlp
            gate, up = mlp.gate_up_proj(block.norm2(x))[0].chunk(2, dim=-1)
            x = x + mlp.down_proj(_swiglu_clamped(gate, up, self.swiglu_limit))[0]
        x = tower.post_layernorm(x)
        merge2 = tower.spatial_merge_size**2
        d = tower.downsample.weight
        x = F.linear(
            x.reshape(n // merge2, merge2 * x.shape[-1]),
            d.permute(0, 2, 3, 1).reshape(d.shape[0], -1),
            tower.downsample.bias,
        )
        merger = tower.merger
        x = F.gelu(merger.post_projection_norm(merger.proj(x)[0]))
        gate, up = merger.gate_up_proj(x)[0].chunk(2, dim=-1)
        return merger.down_proj(_swiglu_clamped(gate, up, self.swiglu_limit))[0]


class RBLNGlm5NextProcessingInfo(Glm5NextProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        processor = super().get_hf_processor(**kwargs)
        image_processor = processor.image_processor
        merge = image_processor.merge_size
        image_processor.max_image_tokens = min(
            image_processor.max_image_tokens, VISION_PATCH_BUCKET // (merge * merge)
        )
        return processor


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=RBLNGlm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class RBLNGlm5NextForConditionalGeneration(
    RBLNGlm5NextForCausalLM, SupportsMultiModal, SupportsRBLNMultiModal
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "model.visual.": "visual.",
        }
    )

    get_placeholder_str = Glm4vForConditionalGeneration.get_placeholder_str

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        with self._mark_language_model(vllm_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        hf_config = vllm_config.model_config.hf_config
        # fp32 tower (dlfp16 on the device, as MiniMax-M3's). Its checkpoint weights
        # are bf16, outside the fp8 quantization.
        with (
            self._mark_tower_model(vllm_config, {"image", "video"}),
            set_default_torch_dtype(torch.float32),
        ):
            self.visual = Glm5NextVisionTransformer(
                hf_config.text_config,
                hf_config.vision_config,
                norm_eps=hf_config.vision_config.rms_norm_eps,
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        self.config.image_token_index = hf_config.image_token_id
        object.__setattr__(self, "_mm_encoder", None)

    def get_language_model(self) -> nn.Module:
        return self

    def compile_mm_encoder(self, compile_fn) -> None:
        encoder = _RBLNGlm5NextVisionEncoder(self.visual).eval()
        object.__setattr__(self, "_mm_encoder", compile_fn(encoder))

    def _encoder_inputs(
        self, pixel_values: torch.Tensor, grid_thw: list[int]
    ) -> list[torch.Tensor]:
        visual = self.visual
        bucket = VISION_PATCH_BUCKET
        num_patches = math.prod(grid_thw)
        assert num_patches <= bucket, (
            f"an image of {num_patches} patches (grid {grid_thw}) exceeds the "
            f"vision encoder's {bucket}-patch bucket"
        )
        dtype = visual.dtype
        patch_dim = visual.patch_embed.proj.weight[0].numel()
        padded = torch.zeros(bucket, patch_dim, dtype=dtype)
        padded[:num_patches] = pixel_values.to(dtype)

        t, h, w = grid_thw
        merge = visual.spatial_merge_size
        order = (
            torch.arange(h * w)
            .view(h // merge, merge, w // merge, merge)
            .permute(0, 2, 1, 3)
            .flatten()
            .repeat(t)
        )
        rope = visual.rotary_pos_emb
        exponents = torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / rope.base ** (exponents / rope.rotary_dim)
        freqs_h = torch.outer((order // w).float(), inv_freq)
        freqs_w = torch.outer((order % w).float(), inv_freq)
        angles = torch.cat([freqs_h, freqs_w, freqs_h, freqs_w], dim=-1)
        head_dim = angles.shape[-1]
        cos = torch.ones(bucket, head_dim)
        sin = torch.zeros(bucket, head_dim)
        cos[:num_patches], sin[:num_patches] = angles.cos(), angles.sin()

        key_bias = torch.zeros(1, 1, bucket)
        key_bias[..., num_patches:] = -30000.0
        return [
            x.to(device=visual.device, dtype=dtype)
            for x in (padded, cos, sin, key_bias)
        ]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if "pixel_values_videos" in kwargs:
            raise NotImplementedError("GLM-5.3 video inputs are not supported on RBLN")
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return ()
        encoder = self._mm_encoder
        if encoder is None:
            encoder = _RBLNGlm5NextVisionEncoder(self.visual).eval()
            object.__setattr__(self, "_mm_encoder", encoder)
        image_grid_thw = kwargs["image_grid_thw"]
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_grid_thw, torch.Tensor)
        grid_thw = image_grid_thw.tolist()
        sizes = [math.prod(g) for g in grid_thw]
        merge2 = self.visual.spatial_merge_size**2
        with torch.inference_mode():
            return tuple(
                encoder(*self._encoder_inputs(pv, g))[: n // merge2].cpu()
                for pv, g, n in zip(pixel_values.split(sizes), grid_thw, sizes)
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
    ) -> torch.Tensor:
        if mm_embeds is not None and inputs_embeds is None:
            assert mm_mask is not None
            text_embeds = self.embed_input_ids(input_ids)
            inputs_embeds = torch.where(mm_mask > 0.5, mm_embeds, text_embeds)
            input_ids = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)


class RBLNGlm5NextMTPLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.enorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # bf16 in the checkpoint, so no quant_config.
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size, hidden_size, bias=False, prefix=f"{prefix}.eh_proj"
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = RBLNGlm5NextDSA(config, quant_config, f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.mlp = RBLNGlm5NextMoE(config, quant_config, f"{prefix}.mlp")
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, inputs_embeds: torch.Tensor, previous_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # NOTE(RBLN): upstream zeroes the embedding at position 0; dropped, as the
        # DeepSeek MTP patch does (the compiler rejects the position-conditional
        # select under device tensor). Only the drafter's position 0 (its hidden and
        # cache entries) differs, which can move the acceptance rate, not the output.
        x = torch.cat(
            [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
        )
        x = self.eh_proj(x)[0]
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head.norm(x)


class RBLNGlm5NextMTPModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): RBLNGlm5NextMTPLayer(
                    config, quant_config, f"{prefix}.layers.{idx}"
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
        prefixes = tuple(f"layers.{idx}." for idx in self.layers)
        return _load_glm5_next_weights(
            self, self.config, weights, lambda name: name.startswith(prefixes)
        )


class RBLNGlm5NextMTP(nn.Module):
    has_own_lm_head = False
    hf_to_vllm_mapper = RBLNGlm5NextForCausalLM.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = RBLNGlm5NextMTPModel(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        layer = self._mtp_layers[spec_step_index % len(self._mtp_layers)]
        out = layer(inputs_embeds, hidden_states)
        return out, out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = ((n, w) for n, w in weights if not n.startswith("lm_head."))
        loaded_params = AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
        # Once every weight is in, as the target's (fold_hadamard rotates in place).
        for layer in self._mtp_layers:
            layer.self_attn.indexer.fold_hadamard()
        return loaded_params
