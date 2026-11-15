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

"""The Qwen3-Next attention patch on RBLN."""

from types import SimpleNamespace

import pytest
import torch


class TestProjectQkvGate:
    """q, k, v and the gate all keep the leading dims of `qkv`, so the gated
    product never broadcasts [tokens, hidden] against [batch, 1, hidden]."""

    HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8

    def _layer(self, gate: bool):
        from vllm_rbln.patches.qwen3_next import patched_qwen3_next_project_qkv_gate

        layer = SimpleNamespace(
            attn_output_gate=gate,
            num_heads=self.HEADS,
            num_kv_heads=self.KV_HEADS,
            head_dim=self.HEAD_DIM,
            q_size=self.HEADS * self.HEAD_DIM,
            kv_size=self.KV_HEADS * self.HEAD_DIM,
            q_norm=lambda x: x * 2.0,
            k_norm=lambda x: x * 3.0,
            rotary_emb=lambda positions, q, k: (q + 1.0, k - 1.0),
        )
        return layer, patched_qwen3_next_project_qkv_gate

    @pytest.mark.parametrize("gate", [True, False])
    def test_leading_dims_are_kept(self, gate):
        layer, project = self._layer(gate)
        width = layer.q_size * (2 if gate else 1) + 2 * layer.kv_size
        torch.manual_seed(0)
        qkv = torch.randn(2, 3, width)
        positions = torch.zeros(2, 3, dtype=torch.long)
        q, k, v, g = project(layer, qkv, positions)
        assert q.shape == (2, 3, layer.q_size)
        assert k.shape == (2, 3, layer.kv_size)
        assert v.shape == (2, 3, layer.kv_size)
        assert (g is None) is not gate
        if gate:
            assert g.shape == (2, 3, layer.q_size)
        # The same math as the flattened [tokens, ...] form upstream computes.
        q_flat, k_flat, v_flat, g_flat = project(
            layer, qkv.reshape(6, width), positions.reshape(6)
        )
        assert torch.equal(q.reshape(6, -1), q_flat)
        assert torch.equal(k.reshape(6, -1), k_flat)
        assert torch.equal(v.reshape(6, -1), v_flat)
        if gate:
            assert torch.equal(g.reshape(6, -1), g_flat)

    def test_gate_splits_per_head(self):
        # q_gate interleaves [q | gate] per head; the chunk must follow that layout.
        layer, project = self._layer(True)
        qkv = torch.zeros(1, 1, layer.q_size * 2 + 2 * layer.kv_size)
        per_head = qkv[..., : layer.q_size * 2].view(
            1, 1, self.HEADS, 2 * self.HEAD_DIM
        )
        per_head[..., : self.HEAD_DIM] = 1.0  # q half of every head
        per_head[..., self.HEAD_DIM :] = 5.0  # gate half
        q, _, _, g = project(layer, qkv, torch.zeros(1, 1, dtype=torch.long))
        assert torch.all(q == 1.0 * 2.0 + 1.0)  # q_norm then rotary
        assert torch.all(g == 5.0)
