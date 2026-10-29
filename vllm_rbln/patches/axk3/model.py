# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""A.X K3 on top of the vendored A.X K2 model.

K3 keeps K2's gated MLA (fused output gate in q_b_proj), DSA indexer and MoE,
and changes two things, both following the checkpoint's ``modeling_axk3.py``:

- Gated Residual replaces both pre-norms and ``model.norm``: the residual is
  ``num_streams`` copies of the hidden state, each block reads a gated mean of
  them and writes its output back with a per-stream gate.
- Index-share: only every ``index_topk_freq``-th layer owns an indexer; the
  others reuse the top-k of their source layer from the same forward.

The indexer reads the post-``q_a_layernorm`` q latent, as in K2. The
checkpoint's ``modeling_axk3.py`` feeds it the pre-norm one, which the trained
weights do not match.
"""

import contextlib
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_rbln import envs
from vllm_rbln.patches.axk2 import model as _axk2_model

# Loaded at import time from the frozen copy, so mypy cannot see its classes
# (nor `Indexer`, which `_without_indexer` swaps on the module).
_k2: Any = _axk2_model._module
_AXK2DecoderLayer: Any = _k2.AXK2DecoderLayer
_AXK2ForCausalLM: Any = _k2.AXK2ForCausalLM


class AXK3GatedResidual(nn.Module):
    """Gated Residual read/write over ``num_streams`` residual branches.

    Numerics follow ``AXK3GatedResidual`` in ``modeling_axk3.py``: fp32 RMSNorm
    statistics, the read gate applied in fp32, the write gate kept in fp32.
    """

    def __init__(
        self,
        hidden_size: int,
        num_streams: int,
        rank: int,
        eps: float,
        has_write: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        wide = num_streams * hidden_size
        self.gamma = nn.Parameter(torch.ones(wide), requires_grad=False)
        self.norm = RMSNorm(hidden_size, eps=eps, has_weight=False)
        self.read_down = ReplicatedLinear(
            wide, rank, bias=False, quant_config=None, prefix=f"{prefix}.read_down"
        )
        self.read_up = ReplicatedLinear(
            rank, wide, bias=False, quant_config=None, prefix=f"{prefix}.read_up"
        )
        if has_write:
            self.write_proj = ReplicatedLinear(
                wide,
                num_streams,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.write_proj",
            )
            self.write_bias = nn.Parameter(
                torch.zeros(num_streams, dtype=torch.float32)
            )
        else:
            self.write_proj = None
            self.write_bias = None

    def _streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.num_streams, self.hidden_size)

    def read(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """[..., n*h] -> (block input [..., h], write gate [..., n] fp32 or None)."""
        n = self.num_streams
        # vLLM's RMSNorm over the h axis of [..., n, h], so the norm lowers like
        # every other RMSNorm; a hand-written [..., n, 1] statistic left a host
        # rsqrt between the device functions of every decode layer. It is fed
        # fp32 so it returns fp32, which keeps the per-stream gain on the
        # unrounded normalize and leaves the one cast the reference has.
        normed = self.norm(self._streams(residual).float())
        u = normed * self.gamma.view(n, self.hidden_size).float()
        u = u.to(residual.dtype).view_as(residual)

        z = F.silu(self.read_down(u)[0] / n)
        gate = torch.sigmoid(self.read_up(z)[0].float())
        block_input = self._streams(gate * u.float()).mean(dim=-2).to(residual.dtype)

        write_gate = None
        if self.write_proj is not None:
            write_gate = 2.0 * torch.sigmoid(
                (self.write_proj(u)[0] / n).float() + self.write_bias
            )
        return block_input, write_gate

    def write(
        self, residual: torch.Tensor, y: torch.Tensor, write_gate: torch.Tensor
    ) -> torch.Tensor:
        """R'_i = R_i + s_i * y."""
        scaled = (write_gate.unsqueeze(-1) * y.float().unsqueeze(-2)).to(residual.dtype)
        return (self._streams(residual) + scaled).view_as(residual)


@contextlib.contextmanager
def _without_indexer():
    """Build a K2 attention layer with no indexer of its own.

    AXK2MLAAttention gives every DSA layer an indexer; an index-share skip layer
    must not own one (the checkpoint has no weights for it, and its cache would
    be allocated for nothing). The layer stays sparse and reads the source
    layer's top-k instead.
    """
    original = _k2.Indexer
    _k2.Indexer = lambda *args, **kwargs: None
    try:
        yield
    finally:
        _k2.Indexer = original


def _gated_mla_forward(
    mla: nn.Module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """K2's gated MLA forward, returning the top-k it attended with.

    ``mla`` is the AXK2GatedMultiHeadLatentAttentionWrapper. A layer that owns
    an indexer computes the top-k from the post-norm q latent; a skip layer is
    handed its source layer's top-k.
    """
    batch_size, seq_len, _ = hidden_states.shape

    qkv_lora = mla.fused_qkv_a_proj(hidden_states)[0]
    q_c_prenorm, kv_lora = qkv_lora.split(
        [mla.q_lora_rank, mla.kv_lora_rank + mla.qk_rope_head_dim], dim=-1
    )
    q_c = mla.q_a_layernorm(q_c_prenorm)

    # q_b_proj takes [post-norm | pre-norm] and emits per head [q | gate].
    q_b_out = mla.q_b_proj(torch.cat([q_c, q_c_prenorm], dim=-1))[0]
    q_gate = q_b_out.view(
        batch_size, seq_len, mla.num_heads, mla.qk_head_dim + mla.v_head_dim
    )
    q, gate = q_gate.split([mla.qk_head_dim, mla.v_head_dim], dim=-1)
    attn_gate = gate.reshape(batch_size, seq_len, mla.num_heads * mla.v_head_dim)

    kv_c, k_pe = kv_lora.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    kv_c_normed = mla.kv_a_layernorm(kv_c)

    q_nope, q_pe = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
    q_pe, k_pe = mla.rotary_emb(positions, q_pe, k_pe.unsqueeze(2))
    q = torch.cat([q_nope, q_pe], dim=-1)
    k_pe = k_pe.squeeze(2)

    if mla.indexer is not None:
        topk_indices = mla.indexer(hidden_states, q_c, positions, mla.indexer_rope_emb)

    attn_out = mla.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(batch_size, seq_len, mla.num_heads * mla.v_head_dim),
        topk_indices=topk_indices,
    )
    attn_out = attn_out * torch.sigmoid(attn_gate)
    return mla.o_proj(attn_out)[0], topk_indices


class AXK3DecoderLayer(_AXK2DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        if config is None:
            config = vllm_config.model_config.hf_config
        layer_idx = int(prefix.split(sep=".")[-1])
        skip_topk = config.is_index_skip_layer(layer_idx)
        with _without_indexer() if skip_topk else contextlib.nullcontext():
            super().__init__(
                vllm_config,
                prefix,
                config=config,
                topk_indices_buffer=topk_indices_buffer,
            )
        assert not self.self_attn.dsv3_pad, "axk3 has the DSv3 MLA layout"
        self.skip_topk = skip_topk
        self.index_source_layer = config.index_source_layer(layer_idx)

        # Gated Residual replaces both pre-norms.
        del self.input_layernorm
        del self.post_attention_layernorm
        gr_kwargs = dict(
            hidden_size=config.hidden_size,
            num_streams=config.gated_residual_num_streams,
            rank=config.gated_residual_rank,
            eps=config.rms_norm_eps,
        )
        self.attn_gated_residual = AXK3GatedResidual(
            **gr_kwargs, prefix=f"{prefix}.attn_gated_residual"
        )
        self.mlp_gated_residual = AXK3GatedResidual(
            **gr_kwargs, prefix=f"{prefix}.mlp_gated_residual"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_by_layer: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        x, write_gate = self.attn_gated_residual.read(hidden_states)
        x, topk_indices = _gated_mla_forward(
            self.self_attn.mla_attn,
            positions,
            x,
            topk_by_layer[self.index_source_layer] if self.skip_topk else None,
        )
        hidden_states = self.attn_gated_residual.write(residual, x, write_gate)

        residual = hidden_states
        x, write_gate = self.mlp_gated_residual.read(hidden_states)
        x = self.mlp(x)
        hidden_states = self.mlp_gated_residual.write(residual, x, write_gate)
        return hidden_states, None if self.skip_topk else topk_indices

    if envs.VLLM_RBLN_USE_NESTED_COMPILE_REGION:
        # The region must not write topk_by_layer; forward returns its top-k instead.
        forward = torch.compiler.nested_compile_region(forward)


@support_torch_compile
class AXK3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        if get_pp_group().world_size > 1:
            raise NotImplementedError("axk3 does not support pipeline parallel yet")
        if not (
            config.gated_residual
            and config.gated_residual_first_layer <= 1
            and config.gated_residual_final == "gated_read"
        ):
            raise NotImplementedError(
                "axk3 supports only a Gated Residual on every layer with a gated "
                "final read"
            )
        self.num_streams = config.gated_residual_num_streams

        topk_indices_buffer = None
        if hasattr(config, "index_topk"):
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AXK3DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )
        self.final_gated_residual = AXK3GatedResidual(
            hidden_size=config.hidden_size,
            num_streams=self.num_streams,
            rank=config.gated_residual_rank,
            eps=config.rms_norm_eps,
            has_write=False,
            prefix=f"{prefix}.final_gated_residual",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * self.num_streams
        )
        self.aux_hidden_state_layers = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Every stream starts as the embedding.
        hidden_states = torch.cat([inputs_embeds] * self.num_streams, dim=-1)

        topk_by_layer: dict[int, torch.Tensor] = {}
        # `make_layers` pads the list to the full depth with `PPMissingLayer`,
        # which returns its first argument; walk only the layers this rank built.
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, topk_indices = layer(positions, hidden_states, topk_by_layer)
            if topk_indices is not None:
                topk_by_layer[layer.layer_idx] = topk_indices

        hidden_states, _ = self.final_gated_residual.read(hidden_states)
        return hidden_states


_NO_EAGLE3 = (
    "axk3 does not support EAGLE3 yet: its forward produces no auxiliary hidden states"
)


class AXK3ForCausalLM(_AXK2ForCausalLM):
    model_cls = AXK3Model

    # The K2 base advertises EAGLE3, so the runner goes on to resolve the aux
    # layers, from the speculative config or from the default below, and hands
    # them to the setter. Both paths end here; refusing on either keeps a draft
    # from being paired with a target that never returns the states it reads.
    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        raise NotImplementedError(_NO_EAGLE3)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        raise NotImplementedError(_NO_EAGLE3)


__all__ = ["AXK3ForCausalLM", "AXK3Model"]
