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

Upstream's ``vllm.models.glm5next.nvidia`` binds FLA / TileLang / DeepGEMM kernels and the
hybrid mamba state manager. This keeps its module tree and weight names and runs:

* mHC (4 residual streams) through ``hc_stream_reduce`` / ``hc_residual_add`` /
  ``sinkhorn_normalize``;
* KDA layers through ``kda_conv`` and ``kda_prefill`` / ``kda_decode``, the recurrent
  state split into g head groups, each a ``MambaSpec`` block that shares R1's bytes
  with a DSA layer's latent block;
* DSA layers (NoPE MLA) through ``glm5_kpool_compress``, ``sparse_attn_glm5_indexer`` and
  ``sparse_attn_glm5_mla`` over paged latent / indexer key / indexer scale caches.

The cache layout is region by region (SoA, design README 2.7; see
``v1/attention/backends/glm5_next.py``). A decoder layer's caches are named
``{prefix}.attn.{slot}``. Text only: the vision tower is not modeled and its weights are
skipped. TP 1.
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
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed import utils as dist_utils
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
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, MLAAttentionSpec

from vllm_rbln.config import get_rbln_config
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)
from vllm_rbln.v1.attention.backends.glm5_next import (
    RBLNGlm5IndexerBackend,
    RBLNGlm5LatentBackend,
    RBLNGlm5StateBackend,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec

# A KDA layer's caches by the slot in their names: the g head groups' states at
# 0 .. g-1, then the q / k / v conv rings, then (MTP) the g head groups' drafts.
KDA_CONV_SLOTS = 3
# The stride of a layer's cache slots (num_attn_module), past the most a layer owns.
CACHES_PER_LAYER = 16
SLOT_DSA_LATENT, SLOT_DSA_INDEX_KEY, SLOT_DSA_INDEX_SCALE, SLOT_DSA_INDEX_TAIL = range(
    4
)
CACHES_PER_DSA_LAYER = 4

KDA_HEAD_DIM = 128
# R1 (a KDA state shares an fp8 MLA latent block): rows of the latent's width.
R1_ROW_BYTES = _rbln_kda_ops.R1_ROW_BYTES
# The conv ring's width: the 3 taps before the current token (no MTP verify tokens).
KDA_CONV_RING = 4
# kda_prefill runs whole chunks of the compiler's size (64 tokens).
KDA_CHUNK = _rbln_kda_ops.CHUNK
INDEX_KPOOL = _rbln_glm5_ops.KPOOL
INDEX_HEAD_DIM = _rbln_glm5_ops.INDEX_HEAD_DIM


def _set_from_host(dst: torch.Tensor, value: torch.Tensor) -> None:
    """Write a load-time weight transform computed on the host into `dst` whole. The
    transforms run on the CPU so that only a host-to-device copy touches the device: an
    eager op on a device tensor (the einsum, a copy into a view) cannot run without an
    NPU, which compile-only (RBLN_DUMMY_DEVICE) has not."""
    dst.data.copy_(value.to(dst.dtype).reshape(dst.shape).contiguous())


def kda_head_groups(vllm_config: VllmConfig) -> int:
    """g: a KDA layer's heads split into g head groups, so that one group's state is
    exactly an MLA latent block, b * 512 bytes (R1): h = b / 64 heads of a bf16
    [128, 128] state each."""
    config = vllm_config.model_config.hf_text_config
    block_size = vllm_config.cache_config.block_size
    state_bytes = KDA_HEAD_DIM * KDA_HEAD_DIM * vllm_config.model_config.dtype.itemsize
    latent_bytes = block_size * config.kv_lora_rank  # fp8
    num_heads = config.linear_num_heads
    if latent_bytes % state_bytes or num_heads % (latent_bytes // state_bytes):
        raise ValueError(
            f"GLM-5.3: block_size {block_size} does not split the {num_heads} KDA "
            f"heads into states of a latent block ({latent_bytes} B, {state_bytes} B "
            "a head)"
        )
    return num_heads // (latent_bytes // state_bytes)


def _layer_index_of(prefix: str) -> int:
    """The cache's position in the runner's KV cache list, which holds the caches
    of this rank's layers in name order (then the MTP layers'), compacted: a KDA
    layer owns g states + 3 conv rings (+ g draft buffers with MTP), a DSA layer 4."""
    vllm_config = get_current_vllm_config()
    config = vllm_config.model_config.hf_text_config
    spec = vllm_config.speculative_config
    groups = kda_head_groups(vllm_config)
    kda_caches = groups + KDA_CONV_SLOTS
    if spec is not None and spec.num_speculative_tokens:
        kda_caches += groups

    def caches(layer: int) -> int:
        if config.layer_types[layer] == "linear_attention":
            return kda_caches
        return CACHES_PER_DSA_LAYER

    ints = [int(name) for name in prefix.split(".") if name.isdigit()]
    layer, slot = ints[0], ints[-1]
    num_layers = config.num_hidden_layers
    # Looked up per call: vllm-rbln patches it (VLLM_RBLN_NUM_HIDDEN_LAYERS).
    start, end = dist_utils.get_pp_indices(
        num_layers,
        get_pp_group().rank_in_group,
        vllm_config.parallel_config.pipeline_parallel_size,
    )
    if layer >= num_layers:  # MTP layers (DSA), after this rank's layers
        return (
            sum(caches(i) for i in range(start, end))
            + CACHES_PER_DSA_LAYER * (layer - num_layers)
            + slot
        )
    assert start <= layer < end, f"{prefix} is not on this PP rank [{start}, {end})"
    return sum(caches(i) for i in range(start, layer)) + slot


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


class RBLNGlm5StateCache(_RBLNGlm5Cache):
    """A KDA head group's recurrent state, one block a request (``MambaSpec``,
    block_size = max_model_len): R1's bytes, uint8 ``[rows, 512]``, the size of an MLA
    latent block, whose extent it shares (the KDA ops' ``_r1`` kernels reinterpret
    them)."""

    def __init__(self, prefix: str, rows: int) -> None:
        super().__init__(prefix)
        self.rows = rows

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MambaSpec(
            block_size=vllm_config.model_config.max_model_len,
            shapes=((self.rows, R1_ROW_BYTES),),
            dtypes=(torch.uint8,),
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        )

    def get_attn_backend(self) -> type[RBLNGlm5StateBackend]:
        return RBLNGlm5StateBackend

    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, RBLNFlashAttentionMetadata]:
        """(state [N, rows, 512], block [B, 1], metadata). Read only the metadata fields
        an op takes: a graph input nothing uses is pruned from the compiled module, and
        the runtime then binds the inputs out of step."""
        metadata = _metadata(self.prefix)
        state = _resolve_kv_cache(metadata, self.layer_index)
        return state, metadata.block_tables.reshape(-1, 1).to(torch.int32), metadata


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


class RBLNGlm5AuxCache(_RBLNGlm5Cache):
    """A per-block region of `owner`'s blocks (``RBLNBlockAuxSpec``),
    ``[N, *block_shape]`` in its own dtype: not a KV cache layer the manager sees (no
    spec, no group, no block table), addressed by the owner's block ids."""

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

    def get_attn_backend(self) -> type[RBLNGlm5StateBackend]:
        return RBLNGlm5StateBackend

    def inputs(self) -> torch.Tensor:
        return _resolve_kv_cache(_metadata(self.owner), self.layer_index)


class _ConvWeight(nn.Module):
    """A depthwise conv's taps as ``[channels, taps]`` (the checkpoint's ``[C, 1, taps]``)."""

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
    """Sigmoid-routed (noaux_tc) MoE with a shared expert. The RBLN ``MoERunner`` takes the
    router as a callable and applies neither ``routed_scaling_factor`` nor the shared
    expert, so both are applied here, as the DeepSeek-V2 RBLN patch does."""

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

        vllm_config = get_current_vllm_config()
        dtype = vllm_config.model_config.dtype
        # The heads split into g head groups (their recurrences are independent), each
        # a state the size of an MLA latent block (R1); the ops run once a group.
        self.num_groups = kda_head_groups(vllm_config)
        self.group_heads = self.num_heads // self.num_groups
        rows = vllm_config.cache_config.block_size
        self.states = nn.ModuleList(
            RBLNGlm5StateCache(f"{prefix}.attn.{j}", rows)
            for j in range(self.num_groups)
        )
        # The q / k / v conv rings, whole layer's channels, on head group 0's block.
        conv_rows = proj * KDA_CONV_RING // 64
        self.conv_q, self.conv_k, self.conv_v = (
            RBLNGlm5AuxCache(
                f"{prefix}.attn.{self.num_groups + i}",
                self.states[0].prefix,
                (conv_rows, 64),
                dtype,
            )
            for i in range(KDA_CONV_SLOTS)
        )
        # MTP: a verify step (1 + K tokens) keeps the state after token 0 and the K
        # drafts' (k, v, exp(g), beta) in this buffer; the next step replays the
        # accepted ones (kda_verify, design README 2.5 method 2). One a head group, on
        # that group's block.
        spec = vllm_config.speculative_config
        self.num_drafts = spec.num_speculative_tokens if spec is not None else 0
        self.drafts = None
        if self.num_drafts:
            # The conv ring indexes by position: it must hold the 3 taps before a
            # verify step's tokens.
            assert self.num_drafts <= KDA_CONV_RING - 3, (
                f"KDA conv ring {KDA_CONV_RING} holds {KDA_CONV_RING - 3} draft(s)"
            )
            self.drafts = nn.ModuleList(
                RBLNGlm5AuxCache(
                    f"{prefix}.attn.{self.num_groups + KDA_CONV_SLOTS + j}",
                    state.prefix,
                    (self.group_heads, self.num_drafts, 4, self.head_dim),
                    dtype,
                )
                for j, state in enumerate(self.states)
            )

    @torch.no_grad()
    def fuse_low_rank(self) -> None:
        for fused, a, b in (
            (self.f_weight, self.f_a_proj, self.f_b_proj),
            (self.g_weight, self.g_a_proj, self.g_b_proj),
        ):
            _set_from_host(fused, b.weight.cpu().float() @ a.weight.cpu().float())

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)

        group_states = [state.inputs() for state in self.states]
        metadata = group_states[0][2]
        seq_idx = metadata.seq_lens.to(torch.int32)
        query_len = (metadata.cache_offsets - metadata.cache_seq_lens).to(torch.int32)
        has_initial_state = (seq_idx > 0).to(torch.int32).view(-1)
        conv_slot = group_states[0][1].view(-1)

        def conv(x: torch.Tensor, cache: RBLNGlm5AuxCache, weight: torch.Tensor):
            return torch.ops.rbln_custom_ops.kda_conv(
                x,
                weight,
                cache.inputs(),
                conv_slot,
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

        q, k, v, g = (x.transpose(1, 2) for x in (q, k, v, g))
        beta = beta.transpose(1, 2).to(hidden_states.dtype)
        if metadata.is_prefill:
            # Padding tokens carry the state through unchanged: no decay, no update.
            # query_len - t > 0 on the device: an int32 broadcast or a compare with a
            # scalar int32 runs on the host only, so the length spreads over the tokens
            # as int16 and the compare is int16 (query_len - t is within a chunk).
            ahead = query_len.to(torch.int16).view(-1, 1, 1).expand(-1, 1, tokens)
            ahead = ahead.to(torch.int32) - torch.arange(
                tokens, dtype=torch.int32, device=g.device
            ).view(1, 1, -1)
            valid = ahead.to(torch.int16) > 0
            g = g * valid.unsqueeze(-1)
            beta = beta * valid
            pad = -tokens % KDA_CHUNK
            if pad:
                q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
                beta = F.pad(beta, (0, pad))

        outs = []
        for j, (state, slot, _) in enumerate(group_states):
            heads = slice(j * self.group_heads, (j + 1) * self.group_heads)
            qj, kj, vj, gj, bj = (x[:, heads] for x in (q, k, v, g, beta))
            slot = slot.view(-1)
            if metadata.is_prefill:
                o = torch.ops.rbln_custom_ops.kda_prefill(
                    qj, kj, vj, gj, bj, state, slot, has_initial_state
                )[:, :, :tokens]
            elif tokens > 1 and self.drafts is not None:
                # An MTP verify step: token 0 always accepted, tokens 1 .. K the drafts.
                o = torch.ops.rbln_custom_ops.kda_verify(
                    qj,
                    kj,
                    vj,
                    gj,
                    bj,
                    state,
                    self.drafts[j].inputs(),
                    slot,
                    metadata.num_accepted.view(-1),
                    None,
                    slot,
                )
            else:
                o = torch.ops.rbln_custom_ops.kda_decode(
                    qj, kj, vj, gj, bj, state, slot
                )
            outs.append(o)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

        o = o.transpose(1, 2)  # [B, T, H, D]
        gate = F.linear(hidden_states, self.g_weight).view(shape)
        o = self.o_norm(o) * torch.sigmoid(gate.float()).to(o.dtype)
        return self.o_proj(o.reshape(batch, tokens, -1))[0]


class RBLNGlm5Indexer(nn.Module):
    """The kpool lightning indexer: 4-token pools, Hadamard-rotated fp8 keys. The rotation
    is folded into ``wq_b`` at load (``fold_hadamard``)."""

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
        # One f16 scale a pool, a region of the key's blocks addressed by the key's
        # (the latent's) block table: [N, b / 4], the [blocks, pools] the op reads, so the
        # graph input needs no reshape (one is a host op, refused for a dynamic KV input).
        self.scale_cache = RBLNGlm5AuxCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_SCALE}",
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_KEY}",
            (get_current_vllm_config().cache_config.block_size // INDEX_KPOOL,),
            torch.float16,
        )
        # The raw (key | gate) rows of the pool a step ended inside, in the block that
        # holds it (design README 2.6): a block's region beside its latent rows. A
        # block size is a multiple of 4, so a pool never straddles two blocks.
        self.tail_cache = RBLNGlm5AuxCache(
            f"{attn_prefix}.attn.{SLOT_DSA_INDEX_TAIL}",
            f"{attn_prefix}.attn.{SLOT_DSA_LATENT}",
            (INDEX_KPOOL, 2 * self.head_dim),
            torch.bfloat16,
        )
        self.block_size = get_current_vllm_config().cache_config.block_size
        assert self.block_size % INDEX_KPOOL == 0

    @torch.no_grad()
    def fold_hadamard(self) -> None:
        weight = self.wq_b.weight
        w = weight.cpu().float().view(self.n_head, self.head_dim, -1)
        rot = _rbln_glm5_ops.hadamard_128().float()
        _set_from_host(
            weight, torch.einsum("ij,hjk->hik", rot, w).reshape(weight.shape)
        )

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

        key_cache, key_table, metadata = self.key_cache.inputs()
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
        vllm_config = get_current_vllm_config()
        # KV8 (--kv-cache-dtype fp8): [fp8 512] latent rows in a uint8 container under
        # one per-tensor k_scale; the op routes a uint8 cache to its fp8 kernel. The
        # checkpoint carries no KV scale, so k_scale is 1.0, held as vLLM's Attention
        # does (a rank-0 fp32 buffer a checkpoint scale could fill in place). The
        # indexer and the KDA / tail-ring caches keep their own dtypes.
        if not vllm_config.cache_config.cache_dtype.startswith("fp8"):
            raise ValueError(
                "GLM-5.3 on RBLN runs with an fp8 KV cache only "
                f"(--kv-cache-dtype fp8), got {vllm_config.cache_config.cache_dtype!r}"
            )
        self.register_buffer(
            "_k_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False
        )
        self.latent_cache = RBLNGlm5PagedCache(
            f"{prefix}.attn.{SLOT_DSA_LATENT}",
            self.kv_lora_rank,
            torch.uint8,
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
        lanes = torch.tensor([n, n, n * n])
        for kind in ("attn", "ffn"):
            scale = getattr(self, f"hc_{kind}_scale").cpu()
            _set_from_host(
                getattr(self, f"hc_{kind}_scale_lanes"), scale.repeat_interleave(lanes)
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
        # Normalize the streams first, then mix (the reference's weightless RMSNorm,
        # then the linear). Scaling the 24 mix lanes by a per-token rsqrt instead left a
        # [B, T, 1] rsqrt and its repeat to 24 lanes on the host between every two
        # sublayers; the [B, T, n * hidden] form stays on the device.
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, fn)
        # The gate / mix logits stay fp32, as upstream's mHC pre (mhc_pre_torch); the
        # mHC ops run fp32 operands as dlf16 and compute in fp32.
        z = mixes * scale + base
        z_pre, z_post = z[..., :n], z[..., n : 2 * n]
        comb = torch.ops.rbln_custom_ops.sinkhorn_normalize(
            z[..., 2 * n :],
            iters=self.sinkhorn_iters,
            eps=self.hc_eps,
        )
        return z_pre, z_post, comb

    def _sublayer(self, streams, fn, scale, base, norm, sublayer) -> torch.Tensor:
        z_pre, z_post, comb = self._hc_mix(streams, fn, scale, base)
        # Upstream collapses and adds the streams in fp32 and casts back to the
        # streams' dtype (mhc_pre_torch / mhc_post_torch). An mHC op takes one dtype
        # across its tensors (a mix runs on the host), so the bf16 streams go in as
        # fp32 too: the device holds them as dlf16, which bf16 values fit exactly.
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


def _load_glm5_weights(
    module: nn.Module,
    config: PretrainedConfig,
    weights: Iterable[tuple[str, torch.Tensor]],
    keep,
) -> set[str]:
    """Load GLM-5.3 checkpoint weights (the dense MLP's stacked gate / up, the routed
    experts' fused params, everything else by name) into `module`, skipping a name
    `keep` refuses."""
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
        # Layers not built here: the MTP head, or past a VLLM_RBLN_NUM_HIDDEN_LAYERS cut.
        loaded_params = _load_glm5_weights(
            self,
            self.config,
            weights,
            lambda name: (
                not name.startswith("layers.")
                or self.start_layer <= int(name.split(".", 2)[1]) < self.end_layer
            ),
        )
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


class RBLNGlm5MTPLayer(nn.Module):
    """GLM-5.3's MTP layer: enorm(embed) | hnorm(previous hidden) -> eh_proj, then a
    DSA + MoE block with a plain residual (no mHC), then the shared head's norm."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.enorm = RBLNRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RBLNRMSNorm(hidden_size, eps=config.rms_norm_eps)
        # bf16 in the checkpoint, so no quant_config.
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size, hidden_size, bias=False, prefix=f"{prefix}.eh_proj"
        )
        self.input_layernorm = RBLNRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = RBLNGlm5DSA(config, quant_config, f"{prefix}.self_attn")
        self.post_attention_layernorm = RBLNRMSNorm(
            hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = RBLNGlm5MoE(config, quant_config, f"{prefix}.mlp")
        self.shared_head = nn.Module()
        self.shared_head.norm = RBLNRMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, inputs_embeds: torch.Tensor, previous_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # NOTE(RBLN): upstream zeroes the embedding at position 0; dropped, as the
        # DeepSeek MTP patch does (the position-conditional select is rejected
        # under device tensor). Position 0 is never a draft step.
        x = torch.cat(
            [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
        )
        x = self.eh_proj(x)[0]
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head.norm(x)


class RBLNGlm5MTPModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        quant_config = vllm_config.quant_config
        # The MTP layers sit past the target's (layer num_hidden_layers onward), as
        # their checkpoint names and the KV cache layer index both expect.
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): RBLNGlm5MTPLayer(
                    config, quant_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        # Replaced by the target's (the proposer shares an MTP draft's embedding).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefixes = tuple(f"layers.{idx}." for idx in self.layers)
        loaded_params = _load_glm5_weights(
            self, self.config, weights, lambda name: name.startswith(prefixes)
        )
        for layer in self.layers.values():
            layer.self_attn.indexer.fold_hadamard()
        return loaded_params


class RBLNGlm5NextMTP(nn.Module):
    """The GLM-5.3 MTP drafter (``Glm5NextMTPModel``), one draft step a layer. The
    embedding and the lm_head are the target's: the checkpoint's MTP layer carries
    neither, only the shared head's norm."""

    # The proposer then shares the target's lm_head.
    has_own_lm_head = False
    hf_to_vllm_mapper = RBLNGlm5NextForCausalLM.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = RBLNGlm5MTPModel(
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
        # Post-norm, for both the draft logits and the next step's hidden.
        return out, out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The lm_head is the target's (shared by the proposer).
        weights = ((n, w) for n, w in weights if not n.startswith("lm_head."))
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
