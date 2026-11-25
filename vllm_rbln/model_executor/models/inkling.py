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
from rebel.ops.torch_custom_ops.kda import CONV_LANES
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.utils import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
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
    WeightsMapper,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.v1.attention.backends.inkling import (
    RBLNInklingAttentionBackend,
    RBLNInklingAttentionMetadata,
)
from vllm_rbln.v1.attention.kv_cache_bindings import resolve_kv_cache
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec

_NVFP4_INPUT_SCALE_DENOMINATOR = torch.finfo(torch.float8_e4m3fn).max * 6.0


class _RBLNInklingCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[RBLNInklingAttentionBackend]:
        return RBLNInklingAttentionBackend


class RBLNInklingKVCache(_RBLNInklingCache):
    def __init__(self, prefix: str, num_kv_heads: int, head_size: int) -> None:
        super().__init__(prefix)
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=torch.bfloat16,
        )

    def inputs(self) -> tuple[torch.Tensor, RBLNInklingAttentionMetadata]:
        metadata = get_forward_context().attn_metadata[self.prefix]
        assert isinstance(metadata, RBLNInklingAttentionMetadata), type(metadata)
        return resolve_kv_cache(metadata, self.prefix), metadata


class RBLNInklingConvRing(_RBLNInklingCache):
    def __init__(
        self,
        prefix: str,
        owner: str,
        num_inputs: int,
        channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__(prefix)
        assert channels % CONV_LANES == 0, (
            f"kda_conv takes channels in blocks of {CONV_LANES}, got {channels}"
        )
        self.owner = owner
        self.width = kernel_size - 1
        self.block_shape = (
            self.width,
            2 * num_inputs,
            channels // CONV_LANES,
            CONV_LANES,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None

    def get_aux_cache_spec(self, vllm_config: VllmConfig) -> RBLNBlockAuxSpec:
        return RBLNBlockAuxSpec(self.owner, self.block_shape, torch.bfloat16)

    def convolve(
        self, xs: list[torch.Tensor], weights: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """`x + conv(x)` for each `[B, T, C]` stream; taps `[C, K]`, the last on
        the current token."""
        metadata = get_forward_context().attn_metadata[self.owner]
        assert isinstance(metadata, RBLNInklingAttentionMetadata), type(metadata)
        signed = torch.stack([y for x in xs for y in (x, -x)], dim=2)
        taps = torch.stack([w for w in weights for _ in range(2)], dim=0)
        out = torch.ops.rbln_custom_ops.kda_conv(
            signed,
            taps,
            resolve_kv_cache(metadata, self.prefix),
            metadata.conv_load_blocks,
            metadata.conv_start_positions,
            metadata.conv_query_lens,
            metadata.has_initial_state,
            self.width,
            store_index=metadata.conv_store_blocks,
        ).float()
        return [
            (x.float() + out[:, :, 2 * i] - out[:, :, 2 * i + 1]).to(x.dtype)
            for i, x in enumerate(xs)
        ]


class RBLNInklingAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        is_local: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if is_local:
            num_heads = config.swa_num_attention_heads
            num_kv_heads = config.swa_num_key_value_heads
            head_dim = config.swa_head_dim
        else:
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = config.head_dim
        self.is_local = is_local
        self.num_heads = divide(num_heads, tp_size)
        self.num_kv_heads = divide(num_kv_heads, tp_size)
        self.head_dim = head_dim
        self.d_rel = config.d_rel
        self.extent = config.sliding_window_size if is_local else config.rel_extent
        self.log_scaling_n_floor = None if is_local else config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha

        def linear(name: str, out_size: int) -> ColumnParallelLinear:
            return ColumnParallelLinear(
                config.hidden_size,
                out_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.{name}",
            )

        self.q_proj = linear("q_proj", num_heads * head_dim)
        self.k_proj = linear("k_proj", num_kv_heads * head_dim)
        self.v_proj = linear("v_proj", num_kv_heads * head_dim)
        self.r_proj = linear("r_proj", num_heads * config.d_rel)
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        kv_size = self.num_kv_heads * head_dim
        kernel_size = config.sconv_kernel_size
        self.k_sconv = nn.Parameter(torch.empty(kv_size, kernel_size))
        self.v_sconv = nn.Parameter(torch.empty(kv_size, kernel_size))
        # Reversed along the distance at load time, the order rel_bias takes.
        self.rel_proj = nn.Parameter(torch.empty(config.d_rel, self.extent))

        self.kv_cache = RBLNInklingKVCache(prefix, self.num_kv_heads, head_dim)
        self.kv_ring = RBLNInklingConvRing(
            f"{prefix}.kv_ring", prefix, 2, kv_size, kernel_size
        )
        self.scale = torch.tensor(
            1.0 / head_dim,
            dtype=torch.float32,
            device=get_current_vllm_config().device_config.device,
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        heads, kv_heads, dim = self.num_heads, self.num_kv_heads, self.head_dim
        groups = heads // kv_heads

        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)
        r, _ = self.r_proj(x)
        k, v = self.kv_ring.convolve([k, v], [self.k_sconv, self.v_sconv])
        q = self.q_norm(q.view(batch, seq_len, heads, dim))
        k = self.k_norm(k.view(batch, seq_len, kv_heads, dim))
        rel = torch.matmul(r.view(batch, seq_len, heads, self.d_rel), self.rel_proj)
        if self.log_scaling_n_floor is not None:
            tau = 1.0 + self.log_scaling_alpha * torch.log(
                ((positions + 1).float() / self.log_scaling_n_floor).clamp(min=1.0)
            )
            q = (q.float() * tau[..., None, None]).to(q.dtype)
            rel = (rel.float() * tau[..., None, None]).to(rel.dtype)

        def heads_first(t: torch.Tensor, n: int, g: int) -> torch.Tensor:
            # [B, L, n * g, X] -> [B, n, g, L, X], the kernels' layout
            t = t.reshape(batch, seq_len, n, g, t.shape[-1])
            return t.permute(0, 2, 3, 1, 4).contiguous()

        kv_cache, metadata = self.kv_cache.inputs()
        # A prefill's table is its one request's row.
        block_tables = metadata.block_tables
        if block_tables.dim() == 1:
            block_tables = block_tables[None]
        args = (
            heads_first(q, kv_heads, groups),
            heads_first(k, kv_heads, 1),
            heads_first(v.view(batch, seq_len, kv_heads, dim), kv_heads, 1),
            kv_cache,
            metadata.seq_lens,
            self.scale,
            block_tables,
        )
        rel_bias = heads_first(rel, kv_heads, groups)
        if self.is_local:
            out = torch.ops.rbln_custom_ops.sliding_window_attention_v1(
                *args,
                self.extent,
                is_prompt_sorted=metadata.is_prompt_sorted,
                rel_bias=rel_bias,
            )
        else:
            out = torch.ops.rbln_custom_ops.gqa(
                *args,
                is_prompt_sorted=metadata.is_prompt_sorted,
                null_block_id=NULL_BLOCK_ID,
                rel_bias=rel_bias,
            )
        out = out.view(batch, heads, seq_len, dim).transpose(1, 2)
        out, _ = self.o_proj(out.reshape(batch, seq_len, heads * dim))
        return out


class RBLNInklingMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        global_scale: bool,
        quant_config: QuantizationConfig | None,
        reduce_results: bool,
        prefix: str,
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
        self.global_scale = nn.Parameter(torch.ones(1)) if global_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x, _ = self.down_proj(F.silu(gate) * up)
        if self.global_scale is not None:
            x = x * self.global_scale.to(x.dtype)
        return x


class RBLNInklingMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if not (
            config.gate_activation == "sigmoid"
            and config.norm_after_topk
            and config.shared_expert_sink
            and config.use_gate_bias
            and config.inference_moe_w13_interleaved
        ):
            raise NotImplementedError(
                "Inkling on RBLN implements the sigmoid, gate-bias, sink-shared "
                "routing and interleaved gate/up rows of the released checkpoints"
            )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_routed = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts + config.n_shared_experts,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.gate.global_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        self.shared_experts = nn.ModuleList(
            RBLNInklingMLP(
                config.hidden_size,
                config.intermediate_size,
                global_scale=False,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts.{i}",
            )
            for i in range(config.n_shared_experts)
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.gate(tokens.float())[0])
        # Expert-major, as the MoE runner selects: a token-major gather over
        # 256 experts does not compile for one token.
        routed_scores = scores[:, : self.num_routed].transpose(0, 1)
        _, selected = torch.topk(
            routed_scores + self.gate.e_score_correction_bias[:, None],
            self.top_k,
            dim=0,
        )
        topk_sum = routed_scores.gather(0, selected).sum(0)[:, None]
        shared_scores = scores[:, self.num_routed :]
        scale = (
            self.route_scale
            * self.gate.global_scale
            / (topk_sum + shared_scores.sum(-1, keepdim=True))
        )

        routed = self.experts(
            hidden_states=hidden_states,
            router=lambda x: self.gate(x.float())[0][..., : self.num_routed],
        ).reshape(tokens.shape)
        out = routed * (topk_sum * scale).to(routed.dtype)
        for i, expert in enumerate(self.shared_experts):
            weight = (shared_scores[:, i : i + 1] * scale).to(routed.dtype)
            out = out + expert(tokens) * weight
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.reshape(shape)


class RBLNInklingStreamConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, owner: str, prefix: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))
        self.ring = RBLNInklingConvRing(
            f"{prefix}.ring", owner, 1, hidden_size, kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ring.convolve([x], [self.weight])[0]


class RBLNInklingDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = int(prefix.split(".")[-1])
        kernel_size = config.sconv_kernel_size
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = RBLNInklingAttention(
            config,
            is_local=layer_id in set(config.local_layer_ids),
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        owner = self.attn.kv_cache.prefix
        self.attn_sconv = RBLNInklingStreamConv(
            config.hidden_size, kernel_size, owner, f"{prefix}.attn_sconv"
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp: nn.Module
        if layer_id < config.dense_mlp_idx:
            self.mlp = RBLNInklingMLP(
                config.hidden_size,
                config.dense_intermediate_size,
                global_scale=config.use_global_scale,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = RBLNInklingMoE(config, quant_config, prefix=f"{prefix}.mlp")
        self.mlp_sconv = RBLNInklingStreamConv(
            config.hidden_size, kernel_size, owner, f"{prefix}.mlp_sconv"
        )

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn_sconv(self.attn(positions, self.attn_norm(hidden)))
        return hidden + self.mlp_sconv(self.mlp(self.mlp_norm(hidden)))


class RBLNInklingModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RBLNInklingDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden = layer(positions, hidden)
        return self.norm(hidden)


class RBLNInklingForCausalLM(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".attn.wq_du": ".attn.q_proj",
            ".attn.wk_dv": ".attn.k_proj",
            ".attn.wv_dv": ".attn.v_proj",
            ".attn.wr_du": ".attn.r_proj",
            ".attn.wo_ud": ".attn.o_proj",
            ".attn.k_sconv.weight": ".attn.k_sconv",
            ".attn.v_sconv.weight": ".attn.v_sconv",
            ".attn.rel_logits_proj.proj": ".attn.rel_proj",
            ".mlp.w13_dn": ".mlp.gate_up_proj",
            ".mlp.w2_md": ".mlp.down_proj",
            ".mlp.gate.bias": ".mlp.gate.e_score_correction_bias",
            ".mlp.experts.": ".mlp.experts.routed_experts.",
        },
        orig_to_new_prefix={
            "model.llm.layers.": "model.layers.",
            "model.llm.embed_norm": "model.embed_norm",
            "model.llm.embed": "model.embed_tokens",
            "model.llm.norm": "model.norm",
            "model.llm.unembed": "lm_head",
            "model.audio.": None,
            "model.visual.": None,
            "model.mtp.": None,
        },
        orig_to_new_suffix={
            "_weight.scale": "_weight_scale",
            "_weight.scale2": "_weight_scale_2",
            "_weight.input_amax": "_input_scale",
            ".original_shape": None,
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if get_pp_group().world_size != 1:
            raise NotImplementedError("Inkling on RBLN does not support PP")
        if (
            not config.use_sconv
            or not config.use_embed_norm
            or config.q_bias
            or config.o_bias
            or config.final_logit_softcapping is not None
        ):
            raise NotImplementedError(
                "Inkling on RBLN implements the released checkpoints: short "
                "convolutions, normed embedding, bias-free q/o, no logit softcap"
            )
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "the relative-bias attention kernels read a 16-bit KV cache"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "Inkling on RBLN stores a short-convolution ring only in each "
                "step's last block, so a prefix hit could resume a stale ring"
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Inkling on RBLN does not support speculative decoding: a "
                "short-convolution ring stores the state after every scheduled "
                "token, so it would keep the rejected drafts"
            )
        self.config = config
        self.model = RBLNInklingModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.padded_vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        scale = 1.0
        if config.logits_mup_width_multiplier:
            scale = 1.0 / config.logits_mup_width_multiplier
        self.logits_processor = LogitsProcessor(
            config.padded_vocab_size, org_vocab_size=config.vocab_size, scale=scale
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
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        tp_rank = get_tensor_model_parallel_rank()
        loaded: set[str] = set()

        def load(name: str, weight: torch.Tensor, *args) -> None:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight, *args)
            loaded.add(name)

        def load_experts(name: str, weight: torch.Tensor) -> None:
            if name.endswith("input_scale"):
                weight = (weight.float() / _NVFP4_INPUT_SCALE_DENOMINATOR).expand(
                    self.config.n_routed_experts
                )
            # Per-expert scalars take no row selection.
            shards: list[tuple[str, slice | None]]
            if name.endswith(("_scale_2", "input_scale")):
                ids = ("w1", "w3") if ".w13_" in name else ("w2",)
                shards = [(shard_id, None) for shard_id in ids]
            elif ".w13_" in name:
                shards = [("w1", slice(0, None, 2)), ("w3", slice(1, None, 2))]
            else:
                shards = [("w2", slice(None))]
            param = params[name]
            for expert_id in range(weight.shape[0]):
                for shard_id, rows in shards:
                    part = weight[expert_id]
                    param.weight_loader(
                        param,
                        part if rows is None else part[rows],
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
            loaded.add(name)

        gate_up = ((0, slice(0, None, 2)), (1, slice(1, None, 2)))
        for name, weight in self.hf_to_vllm_mapper.apply(weights):
            if is_pp_missing_parameter(name, self):
                continue
            if name.endswith((".attn.k_sconv", ".attn.v_sconv")):
                rows = params[name].shape[0]
                load(name, weight.squeeze(1).narrow(0, tp_rank * rows, rows))
            elif name.endswith("_sconv.weight"):
                load(name, weight.squeeze(1))
            elif name.endswith(".attn.rel_proj"):
                load(name, weight.flip(-1))
            elif name.endswith(".mlp.gate_up_proj.weight"):
                for shard_id, rows in gate_up:
                    load(name, weight[rows], shard_id)
            elif ".routed_experts." in name:
                load_experts(name, weight)
            elif ".shared_experts.shared_w" in name:
                kind = "gate_up_proj" if "shared_w13" in name else "down_proj"
                base = name.split(".shared_experts.")[0] + ".shared_experts"
                for i in range(weight.shape[0]):
                    target = f"{base}.{i}.{kind}.weight"
                    if kind == "gate_up_proj":
                        for shard_id, rows in gate_up:
                            load(target, weight[i, rows], shard_id)
                    else:
                        load(target, weight[i])
            else:
                load(name, weight)
        return loaded
