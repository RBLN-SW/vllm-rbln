# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_rbln.v1.attention.backends.gdn_attn import RBLNGDNAttentionMetadata

# The delta rule solves (I - A)^-1 per chunk of this many tokens.
CHUNK = 128


def _l2norm(x: torch.Tensor, prefill: bool) -> torch.Tensor:
    # A sum over the innermost axis of a one-token tensor lowers to ~0 on RBLN;
    # a matmul computes the same sum of squares.
    sum_squares = (x * x).sum(-1, keepdim=True) if prefill else x @ x.transpose(-1, -2)
    return x * torch.rsqrt(sum_squares + 1e-6)


class RBLNQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """Qwen3-Next / Qwen3.5 Gated DeltaNet on RBLN.

    Hidden states arrive as ``[batch, seq_len, hidden]`` and the whole layer is
    plain torch: prefill runs the chunked delta rule one 128-token chunk at a
    time, decode the recurrence. A request reads its state row by gather and
    writes it back with ``rbln_cache_update``; the runner zeroes the row
    before the request's first step. The conv state is ``[K - 1, conv_dim]``
    and the recurrent state ``[num_v_heads, Dk, Dv]``.

    A pooling model reads a prompt in one prefill and never decodes, so its
    layers start from zero and keep no state.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keeps_state = self.model_config.runner_type != "pooling"

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        assert self.tp_size == 1 and self.num_spec == 0
        return (
            (self.conv_kernel_size - 1, self.conv_dim),
            (self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata[self.prefix]
        assert isinstance(attn_metadata, RBLNGDNAttentionMetadata)
        batch, seq_len, _ = hidden_states.shape

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        mixed_ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkv, z = mixed_qkvz.split([self.conv_dim, self.value_dim], dim=-1)
        b, a = mixed_ba.chunk(2, dim=-1)

        mixed_qkv = self._causal_conv(mixed_qkv, attn_metadata)
        query, key, value = mixed_qkv.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        beta = b.float().sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if attn_metadata.is_prefill:
            # Right padding must leave the state alone: g = 0 keeps it
            # undecayed and beta = 0 writes nothing.
            assert attn_metadata.valid_mask is not None
            g = g * attn_metadata.valid_mask.unsqueeze(-1)
            beta = beta * attn_metadata.valid_mask.unsqueeze(-1)

        core = self._delta_rule(query, key, value, g, beta, attn_metadata)
        core = core.to(hidden_states.dtype)
        z = z.reshape(batch, seq_len, -1, self.head_v_dim)
        core = self.norm(core, z)
        output, _ = self.out_proj(core.flatten(-2))
        return output

    def _write_state(
        self,
        cache: torch.Tensor,
        new_state: torch.Tensor,
        attn_metadata: RBLNGDNAttentionMetadata,
    ) -> None:
        """Write each request's row back at its slot."""
        axis = torch.tensor(0, dtype=torch.int16)
        new_state = new_state.to(cache.dtype)
        for row in range(new_state.shape[0]):
            attn_metadata.state_updates.append(
                torch.ops.rbln_custom_ops.rbln_cache_update(
                    cache,
                    new_state[row : row + 1],
                    attn_metadata.state_slots[row].to(torch.int16),
                    axis,
                )
            )

    def _causal_conv(
        self,
        mixed_qkv: torch.Tensor,
        attn_metadata: RBLNGDNAttentionMetadata,
    ) -> torch.Tensor:
        """Depthwise causal conv over ``[batch, seq_len, conv_dim]``."""
        seq_len = mixed_qkv.shape[1]
        width = self.conv_kernel_size
        assert attn_metadata.conv_states is not None
        conv_state = attn_metadata.conv_states[self.prefix]
        if self.keeps_state:
            previous = conv_state[attn_metadata.state_slots].to(mixed_qkv.dtype)
        else:
            previous = mixed_qkv.new_zeros(
                (mixed_qkv.shape[0], width - 1, self.conv_dim)
            )
        window = torch.cat([previous, mixed_qkv], dim=1)

        weight = self.conv1d.weight.view(self.conv_dim, width)
        out = window[:, :seq_len] * weight[:, 0]
        for tap in range(1, width):
            out = out + window[:, tap : tap + seq_len] * weight[:, tap]
        if self.conv1d.bias is not None:
            out = out + self.conv1d.bias

        if self.keeps_state:
            if attn_metadata.is_prefill:
                # The last K - 1 inputs of the real tokens, not of the padding.
                new_state = window.index_select(1, attn_metadata.conv_state_index)
            else:
                new_state = window[:, 1:]
            self._write_state(conv_state, new_state, attn_metadata)
        return F.silu(out)

    def _delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        attn_metadata: RBLNGDNAttentionMetadata,
    ) -> torch.Tensor:
        """Gated delta rule; returns ``[batch, seq_len, num_v_heads, head_v_dim]``."""
        batch, seq_len, _ = query.shape
        prefill = attn_metadata.is_prefill
        ratio = self.num_v_heads // self.num_k_heads

        def heads(x: torch.Tensor, dim: int) -> torch.Tensor:
            return x.view(batch, seq_len, -1, dim).transpose(1, 2).float()

        query = _l2norm(heads(query, self.head_k_dim), prefill)
        key = _l2norm(heads(key, self.head_k_dim), prefill)
        if ratio > 1:
            query = query.repeat_interleave(ratio, dim=1)
            key = key.repeat_interleave(ratio, dim=1)
        query = query * self.head_k_dim**-0.5
        value = heads(value, self.head_v_dim)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)

        assert attn_metadata.ssm_states is not None
        ssm_state = attn_metadata.ssm_states[self.prefix]
        if self.keeps_state:
            state = ssm_state[attn_metadata.state_slots].float()
        else:
            state = query.new_zeros(
                (batch, self.num_v_heads, self.head_k_dim, self.head_v_dim)
            )
        if prefill:
            # The chunk masks multiply, so a non-finite padding row would reach
            # the real rows through 0 * inf; the padding's hidden states come
            # from attention over cache blocks nothing has written.
            assert attn_metadata.valid_mask is not None
            valid = attn_metadata.valid_mask.view(1, 1, -1, 1) > 0
            query, key, value = (
                torch.where(valid, x, torch.zeros_like(x)) for x in (query, key, value)
            )
            core, state = self._chunked(query, key, value, g, beta, state)
        else:
            state = state * g.exp().unsqueeze(-1)
            delta = (value - key @ state) * beta.unsqueeze(-1)
            state = state + key.transpose(-1, -2) @ delta
            core = query @ state
        if self.keeps_state:
            self._write_state(ssm_state, state, attn_metadata)
        return core.transpose(1, 2)

    @staticmethod
    def _chunked(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked delta rule over ``[batch, heads, seq_len, dim]``.

        One chunk per iteration: batching the chunks into one tensor, as
        upstream does, fails to place on a multi-chiplet REBEL device.
        """
        incl = torch.ones(CHUNK, CHUNK, device=query.device).tril()
        strict = incl.tril(-1)
        outputs = []
        for start in range(0, query.shape[2], CHUNK):
            chunk = slice(start, start + CHUNK)
            q, k, v = query[:, :, chunk], key[:, :, chunk], value[:, :, chunk]
            increment, b = g[:, :, chunk], beta[:, :, chunk].unsqueeze(-1)
            beta_value, beta_key = v * b, k * b
            decay_sum = increment.cumsum(-1)
            decay = (
                (decay_sum.unsqueeze(-1) - decay_sum.unsqueeze(-2)) * incl
            ).exp() * incl
            attn = -((beta_key @ k.transpose(-1, -2)) * decay) * strict
            attn = torch.ops.rbln.tri_recur_update(attn) + (incl - strict)
            values = attn @ beta_value
            key_decay = attn @ (beta_key * decay_sum.exp().unsqueeze(-1))
            intra = (q @ k.transpose(-1, -2)) * decay
            v_new = values - key_decay @ state
            inter = (q * decay_sum.exp().unsqueeze(-1)) @ state
            outputs.append(inter + intra @ v_new)
            state = (
                state * increment.sum(-1, keepdim=True).unsqueeze(-1).exp()
                + (k * (increment @ strict).exp().unsqueeze(-1)).transpose(-1, -2)
                @ v_new
            )
        return torch.cat(outputs, dim=2), state
