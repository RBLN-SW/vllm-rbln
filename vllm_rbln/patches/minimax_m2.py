# Copyright 2025 Rebellions Inc. All rights reserved.
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

from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
    MiniMaxText01RMSNormTP,
)
from vllm.model_executor.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2Model,
    MiniMaxM2MoE,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2MoE.forward",
    reason=(
        "Replace MiniMaxM2MoE.forward with an RBLN-friendly form. "
        "(1) Keep the 3D [B, L, H] hidden_states instead of unpacking a 2D "
        "shape (upstream's `num_tokens, hidden_dim = hidden_states.shape` "
        "raises on RBLN's 3D layout). "
        "(2) Pass `self.gate` as a router callable instead of precomputed "
        "router_logits, so routing runs after RBLN DP multicast. "
        "(3) Explicitly all-reduce TP outputs (upstream has no explicit "
        "all-reduce in this forward)."
    ),
)
def patched_minimax_m2_moe_forward(
    self: MiniMaxM2MoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    # router_logits: (num_tokens, n_experts)
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x.to(torch.float32))[0]
    )
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    return final_hidden_states.to(hidden_states.dtype)


def _forward_qk(
    q_norm: MiniMaxText01RMSNormTP,
    k_norm: MiniMaxText01RMSNormTP,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    if q_norm.tp_world > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + q_norm.variance_epsilon) * q_norm.weight
    k = k * torch.rsqrt(k_var + k_norm.variance_epsilon) * k_norm.weight
    q = q.to(orig_dtype)
    k = k.to(orig_dtype)
    return q, k


@register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2Attention.forward",
    reason=(
        "Reimplement MiniMaxM2Attention.forward exactly as vLLM 0.22 to keep the "
        "hidden states in RBLN's native 3D [B, L, H] layout end-to-end. Upstream "
        "0.24 routes QK-norm through MiniMaxText01RMSNormTP.forward_qkv, which "
        "(a) asserts qkv.ndim == 2 and (b) for tp_size > 1 dispatches to the TRT-LLM "
        "fused all-reduce+RMSNorm kernel / Triton fallback wired in #43410 -- neither "
        "runnable on RBLN. Crucially, the obvious workaround of flattening qkv to 2D "
        "([B*L, H]) around forward_qkv and reshaping back, although a numerical no-op "
        "in eager, corrupts attention on RBLN: the flatten/unflatten roundtrip "
        "perturbs the [B, L, H] layout that the compiled rotary + flash-attention path "
        "(and VLLM_RBLN_BATCH_ATTN_OPT / VLLM_RBLN_SORT_BATCH) is built around. "
        "Instead we split qkv on the last dim (no reshape) and apply a locally "
        "reimplemented forward_qk -- plain RMSNorm over dim=-1, which is "
        "layout-invariant -- so q/k/v reach rotary/attn in the exact layout 0.22 used. "
        "The norm is reimplemented locally rather than calling "
        "MiniMaxText01RMSNormTP.forward_qk because #43410 also stripped the cross-TP "
        "variance all-reduce out of that method (it now lives only in the fused "
        "kernel); the local copy restores 0.22's all-reduce so the result stays "
        "correct for tp_size > 1 as well, not just tp_size == 1."
    ),
)
def patched_minimax_m2_attention_forward(
    self: MiniMaxM2Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = _forward_qk(self.q_norm, self.k_norm, q, k)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


_AUX_SLOT = "aux_hidden_states_"


def _aux_slots_received(model: MiniMaxM2Model) -> tuple[int, ...]:
    """Global aux layer indices that reach this stage from upstream stages.

    A capture at index ``i`` is the input to layer ``i``, so this stage receives
    every requested index at or before its first owned layer. On the first stage
    the set is empty: index ``start_layer == 0`` is captured locally from the
    embedding output.
    """
    if get_pp_group().is_first_rank:
        return ()
    return tuple(i for i in model.aux_hidden_state_layers if i <= model.start_layer)


@register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2Model._set_aux_hidden_state_layers",
    reason=(
        "Size the pipeline handoff to carry the EAGLE3 aux hidden states. The "
        "handoff key set is fixed in __init__, but the aux layers are only known "
        "once the model runner calls this setter, so the factory has to be rebuilt "
        "here. Upstream has no hook between the two. "
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
    self.aux_hidden_state_layers = layers
    keys = ["hidden_states", "residual"]
    keys += [f"{_AUX_SLOT}{i}" for i in _aux_slots_received(self)]
    self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
        keys, self.config.hidden_size
    )


@register_patch(
    target="vllm.model_executor.models.minimax_m2.MiniMaxM2Model.forward",
    reason=(
        "Collect EAGLE3 aux hidden states correctly under pipeline parallelism. "
        "Upstream indexes the capture with `enumerate(islice(...))`, which restarts "
        "at zero on every stage, so a non-first stage matches a requested global "
        "layer index against a stage-local one and harvests the wrong layer. It "
        "then drops `aux_hidden_states` entirely on non-last stages, while the "
        "drafter runs on the last one. Carry them in the existing IntermediateTensors "
        "handoff under global-index slots and reassemble in layer order; no new "
        "collective is introduced. "
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    incoming: dict[int, torch.Tensor] = {}
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
        incoming = {
            i: intermediate_tensors[f"{_AUX_SLOT}{i}"]
            for i in _aux_slots_received(self)
        }

    # Index i means "input to layer i". The pre-loop capture is first-rank only:
    # for a later stage that index is the previous stage's final capture, and
    # taking it again would duplicate a boundary layer. Layer 31 of the 62-layer
    # [15,16,16,15] split is exactly such a boundary.
    aux: dict[int, torch.Tensor] = {}

    def capture(index: int) -> None:
        if index in self.aux_hidden_state_layers:
            aux[index] = (
                hidden_states + residual if residual is not None else hidden_states
            )

    if get_pp_group().is_first_rank:
        capture(0)
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(positions, hidden_states, residual)
        capture(self.start_layer + idx + 1)

    # A received index is at or before start_layer and a local one is past it, so
    # the two sets never collide.
    aux.update(incoming)

    if not get_pp_group().is_last_rank:
        tensors = {"hidden_states": hidden_states, "residual": residual}
        for i, value in aux.items():
            tensors[f"{_AUX_SLOT}{i}"] = value
        return IntermediateTensors(tensors)

    hidden_states, _ = self.norm(hidden_states, residual)

    if not self.aux_hidden_state_layers:
        return hidden_states

    missing = set(self.aux_hidden_state_layers) - aux.keys()
    assert not missing, (
        f"EAGLE3 aux hidden states missing for layers {sorted(missing)}; "
        f"requested {self.aux_hidden_state_layers}, arrived {sorted(aux)}"
    )
    return hidden_states, [aux[i] for i in sorted(aux)]
