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

import pytest
import torch
import torch.nn.functional as F

from vllm_rbln.model_executor.layers.gdn import (
    CHUNK,
    RBLNQwenGatedDeltaNetAttention,
    _l2norm,
)

# The chunked gated delta rule must equal the token-by-token recurrence it
# reorganizes, including the state carried across chunk boundaries.


def _recurrence(query, key, value, g, beta, state):
    outputs = []
    for t in range(query.shape[2]):
        state = state * g[:, :, t].exp()[..., None, None]
        k_t = key[:, :, t : t + 1]
        delta = (value[:, :, t : t + 1] - k_t @ state) * beta[:, :, t, None, None]
        state = state + k_t.transpose(-1, -2) @ delta
        outputs.append(query[:, :, t : t + 1] @ state)
    return torch.cat(outputs, dim=2), state


@pytest.mark.parametrize("seq_len", [CHUNK, 2 * CHUNK])
def test_chunked_matches_recurrence(seq_len):
    torch.manual_seed(0)
    batch, heads, dk, dv = 1, 2, 16, 8
    query = F.normalize(torch.randn(batch, heads, seq_len, dk), dim=-1) * dk**-0.5
    key = F.normalize(torch.randn(batch, heads, seq_len, dk), dim=-1)
    value = torch.randn(batch, heads, seq_len, dv)
    g = -torch.rand(batch, heads, seq_len) * 0.1
    beta = torch.rand(batch, heads, seq_len)
    state = torch.randn(batch, heads, dk, dv) * 0.1

    core, final = RBLNQwenGatedDeltaNetAttention._chunked(
        query, key, value, g, beta, state
    )
    ref_core, ref_final = _recurrence(query, key, value, g, beta, state)

    torch.testing.assert_close(core, ref_core, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(final, ref_final, atol=1e-4, rtol=1e-4)


def test_chunked_padding_leaves_state_alone():
    # Right padding has g = 0, beta = 0 and zero q/k/v, as forward() prepares it.
    torch.manual_seed(0)
    real = CHUNK + 5
    shape = (1, 2, 2 * CHUNK)
    query = F.normalize(torch.randn(*shape, 16), dim=-1)
    key = F.normalize(torch.randn(*shape, 16), dim=-1)
    value = torch.randn(*shape, 8)
    g = -torch.rand(shape) * 0.1
    beta = torch.rand(shape)
    for x in (query, key, value, g, beta):
        x[:, :, real:] = 0
    state = torch.zeros(1, 2, 16, 8)

    _, final = RBLNQwenGatedDeltaNetAttention._chunked(
        query, key, value, g, beta, state
    )
    _, ref_final = _recurrence(
        query[:, :, :real],
        key[:, :, :real],
        value[:, :, :real],
        g[:, :, :real],
        beta[:, :, :real],
        state,
    )

    torch.testing.assert_close(final, ref_final, atol=1e-4, rtol=1e-4)


def test_l2norm_forms_agree():
    x = torch.randn(1, 2, 1, 16)
    torch.testing.assert_close(_l2norm(x, prefill=True), _l2norm(x, prefill=False))
    torch.testing.assert_close(
        _l2norm(x, prefill=True), F.normalize(x, dim=-1), atol=1e-5, rtol=1e-4
    )
