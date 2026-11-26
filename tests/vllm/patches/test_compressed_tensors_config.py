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
"""compressed-tensors experts through the registered RBLN config.

A real upstream AfmoeMoE is built on CPU from the config vLLM resolves for
"compressed-tensors", with the schemes of arcee-ai/Trinity-Large-Thinking-FP8-Block
(fp8 weights in 128x128 blocks, dynamic per-128-group fp8 activations) and
RedHatAI/Trinity-Large-Thinking-NVFP4 (NVFP4 weights and activations, group 16).
The reference dequantises the same weights and follows the HF routing.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

HIDDEN, INTER, EXPERTS, TOP_K, ROUTE_SCALE, BLOCK = 128, 128, 8, 2, 2.448, 128
FP4_GROUP = 16


def _quant_config(weight_strategy: str = "block") -> dict:
    fmt = "float-quantized"
    if weight_strategy == "nvfp4":
        fmt = "nvfp4-pack-quantized"
        weights = {"num_bits": 4, "strategy": "tensor_group", "group_size": FP4_GROUP}
        activations = dict(weights, dynamic="local")
        weights["dynamic"] = False
    elif weight_strategy == "block":
        weights = {
            "num_bits": 8,
            "strategy": "block",
            "block_structure": [BLOCK, BLOCK],
        }
        activations = {"num_bits": 8, "strategy": "group", "group_size": BLOCK}
    else:
        weights = {"num_bits": 8, "strategy": "channel"}
        activations = {"num_bits": 8, "strategy": "token"}
    weights |= {"type": "float", "symmetric": True, "dynamic": False}
    activations |= {"type": "float", "symmetric": True}
    activations.setdefault("dynamic", True)
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": activations,
                "format": fmt,
            }
        },
        "ignore": [],
    }


@contextmanager
def _build_block(tmp_path, weight_strategy="block", use_w8a8=False, shared=0):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.models.afmoe import AfmoeMoE
    from vllm.transformers_utils.configs.afmoe import AfmoeConfig

    from tests.vllm.vllm_config import make_vllm_config

    config = AfmoeConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        num_shared_experts=shared,
        score_func="sigmoid",
        route_norm=True,
        route_scale=ROUTE_SCALE,
    )
    quant_config = get_quantization_config("compressed-tensors").from_config(
        _quant_config(weight_strategy)
    )
    vllm_config = make_vllm_config(
        model="Qwen/Qwen3-30B-A3B",
        dtype="float32",
        additional_config={"use_w8a8": use_w8a8},
    )
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
            yield AfmoeMoE(
                config, quant_config=quant_config, prefix="model.layers.6.mlp"
            )
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _load_random_weights(block) -> None:
    g = torch.Generator().manual_seed(0)
    routed = block.experts.routed_experts
    for name in ("w13_weight", "w2_weight"):
        param = getattr(routed, name)
        param.data = (torch.randn(param.shape, generator=g) * 0.5).to(param.dtype)
    for name in ("w13_weight_scale", "w2_weight_scale"):
        param = getattr(routed, name)
        param.data = torch.rand(param.shape, generator=g) * 0.2 + 0.1
    block.gate.weight.data = torch.randn(block.gate.weight.shape, generator=g) * 0.2
    block.expert_bias.data = torch.randn(EXPERTS, generator=g)
    routed.quant_method.process_weights_after_loading(routed)


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


def _routed(block, x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor):
    """HF AfmoeMoE routed experts on [T, H] with the given dequantised weights."""
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
    return out


def _reference(block, x: torch.Tensor) -> torch.Tensor:
    """HF AfmoeMoE on [T, H] with the block-dequantised experts."""
    routed = block.experts.routed_experts

    def dequant(w, s):
        return w.float() * s.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)

    w13 = dequant(routed.w13_weight, routed.w13_weight_scale)
    w2 = dequant(routed.w2_weight, routed.w2_weight_scale)
    return _routed(block, x, w13, w2)


@pytest.mark.parametrize("use_w8a8", [False, True])
def test_block_fp8_experts_match_dequantised_reference(tmp_path, monkeypatch, use_w8a8):
    with _build_block(tmp_path, use_w8a8=use_w8a8) as block:
        assert block.experts.routed_experts.w13_weight.dtype == torch.float8_e4m3fn
        _load_random_weights(block)
        _set_tokens(monkeypatch, 6)
        x = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(1))

        with torch.inference_mode():
            out = block(x).reshape(-1, HIDDEN)
            expected = _reference(block, x.reshape(-1, HIDDEN))

        if use_w8a8:
            # fp8 activations: near the reference, but not the W8A16 result.
            err = (out - expected).norm() / expected.norm()
            assert 0 < err < 0.05
        else:
            torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


def test_non_block_fp8_experts_keep_upstream_method(tmp_path):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
        CompressedTensorsW8A8Fp8MoEMethod,
    )

    with _build_block(tmp_path, weight_strategy="channel") as block:
        assert type(block.experts.routed_experts.quant_method) is (
            CompressedTensorsW8A8Fp8MoEMethod
        )


def _load_random_nvfp4(block) -> dict[str, torch.Tensor]:
    """Fill the checkpoint-format NVFP4 tensors and return their dense values.

    Gate and up get different global scales: the RBLN path keeps both.
    """
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

    g = torch.Generator().manual_seed(0)

    def fill(packed, scale, global_scale):
        packed.data = torch.randint(
            0, 256, packed.shape, dtype=torch.uint8, generator=g
        )
        scale.data = (torch.rand(scale.shape, generator=g) * 0.05 + 0.02).to(
            torch.float8_e4m3fn
        )
        global_scale.data = torch.rand(global_scale.shape, generator=g) + 1.0

    def dense(packed, scale, global_scale):
        cols = packed.shape[-1] * 2
        w = unpack_fp4_from_uint8(
            packed.reshape(-1, cols // 2), -1, cols, torch.float32
        )
        w = w.reshape(*packed.shape[:-1], cols)
        return w * scale.float().repeat_interleave(FP4_GROUP, -1) / global_scale

    routed = block.experts.routed_experts
    for w in ("w13", "w2"):
        fill(
            getattr(routed, f"{w}_weight_packed"),
            getattr(routed, f"{w}_weight_scale"),
            getattr(routed, f"{w}_weight_global_scale"),
        )
    gate_up_global = routed.w13_weight_global_scale.data.repeat_interleave(INTER, 1)
    dense_weights = {
        "w13": dense(
            routed.w13_weight_packed, routed.w13_weight_scale, gate_up_global[..., None]
        ),
        "w2": dense(
            routed.w2_weight_packed,
            routed.w2_weight_scale,
            routed.w2_weight_global_scale.data[:, None, None],
        ),
    }
    for name in ("gate_up_proj", "down_proj"):
        linear = getattr(block.shared_experts, name)
        fill(linear.weight_packed, linear.weight_scale, linear.weight_global_scale)
        # Checkpoints share one global scale across a fused layer's shards.
        linear.weight_global_scale.data.fill_(1.5)
        dense_weights[name] = dense(linear.weight_packed, linear.weight_scale, 1.5)
        linear.input_global_scale.data.fill_(1.0)
        linear.quant_method.process_weights_after_loading(linear)

    block.gate.weight.data = torch.randn(block.gate.weight.shape, generator=g) * 0.2
    block.expert_bias.data = torch.randn(EXPERTS, generator=g)
    routed.quant_method.process_weights_after_loading(routed)
    return dense_weights


def test_nvfp4_experts_and_shared_linear_match_dequantised_reference(
    tmp_path, monkeypatch
):
    with _build_block(tmp_path, weight_strategy="nvfp4", shared=1) as block:
        dense_weights = _load_random_nvfp4(block)
        _set_tokens(monkeypatch, 6)
        x = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(1))

        with torch.inference_mode():
            out = block(x).reshape(-1, HIDDEN)
            x2d = x.reshape(-1, HIDDEN)
            gate, up = (x2d @ dense_weights["gate_up_proj"].T).split(INTER, dim=-1)
            shared = (F.silu(gate) * up) @ dense_weights["down_proj"].T
            expected = (
                _routed(block, x2d, dense_weights["w13"], dense_weights["w2"]) + shared
            )

        # W4A16: the activations are not quantized, so the match is tight.
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)
