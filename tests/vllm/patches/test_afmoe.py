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

"""The patched AfmoeMoE block against the Trinity (AfMoE) routing it serves.

A real upstream AfmoeMoE is built on CPU, so the router upstream's factory picks
for this config and the RBLN MoE runner and expert kernel are the ones the model
runs. The reference follows the HF modeling code: sigmoid scores, the expert bias
only in the selection, renormalised top-k weights times route_scale, plus the
shared expert.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

HIDDEN, INTER, EXPERTS, TOP_K, ROUTE_SCALE = 64, 32, 8, 2, 2.448


@pytest.fixture
def afmoe_block(tmp_path):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.models.afmoe import AfmoeMoE
    from vllm.transformers_utils.configs.afmoe import AfmoeConfig

    from tests.vllm.vllm_config import make_vllm_config

    config = AfmoeConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        num_shared_experts=1,
        score_func="sigmoid",
        route_norm=True,
        route_scale=ROUTE_SCALE,
    )
    # A MoE model, or ensure_model_parallel_initialized creates no EP group.
    vllm_config = make_vllm_config(model="Qwen/Qwen3-30B-A3B", dtype="float32")
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path}/dist",
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
        try:
            block = AfmoeMoE(config, prefix="model.layers.6.mlp").eval()
            g = torch.Generator().manual_seed(0)
            for param in block.parameters():
                param.data = torch.randn(param.shape, generator=g) * 0.2
            block.expert_bias.data = torch.randn(EXPERTS, generator=g)
            yield block
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _set_tokens(monkeypatch, num_tokens: int) -> None:
    import vllm_rbln.model_executor.layers.fused_moe.utils as fused_moe_utils

    monkeypatch.setattr(
        fused_moe_utils,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor([num_tokens]),
                max_pads_across_dp=None,
            )
        ),
    )


def _reference(block, x: torch.Tensor) -> torch.Tensor:
    """HF AfmoeMoE on [T, H]: per-token loop over the selected experts."""
    routed = block.experts.routed_experts
    w13, w2 = routed.w13_weight, routed.w2_weight
    scores = torch.sigmoid(x @ block.gate.weight.T)
    _, selected = torch.topk(scores + block.expert_bias, TOP_K, dim=-1)
    weights = scores.gather(-1, selected)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE

    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOP_K):
            e = selected[t, j]
            gate, up = (x[t] @ w13[e].T).split(INTER)
            out[t] += weights[t, j] * ((F.silu(gate) * up) @ w2[e].T)

    shared = block.shared_experts
    gate, up = (x @ shared.gate_up_proj.weight.T).split(INTER, dim=-1)
    return out + (F.silu(gate) * up) @ shared.down_proj.weight.T


@pytest.mark.parametrize("batch, seq", [(1, 5), (2, 3)])
def test_patched_forward_matches_hf_routing(afmoe_block, monkeypatch, batch, seq):
    _set_tokens(monkeypatch, batch * seq)
    x = torch.randn(batch, seq, HIDDEN, generator=torch.Generator().manual_seed(1))

    with torch.inference_mode():
        out = afmoe_block(x)
        expected = _reference(afmoe_block, x.reshape(-1, HIDDEN))

    assert out.shape == x.shape
    torch.testing.assert_close(out.reshape(-1, HIDDEN), expected, atol=1e-4, rtol=1e-4)
