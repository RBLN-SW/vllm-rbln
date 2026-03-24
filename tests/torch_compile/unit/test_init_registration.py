from collections.abc import Iterable
from unittest.mock import Mock

import torch

import vllm_rbln


def test_register_returns_rbln_platform_path():
    assert vllm_rbln.register() == "vllm_rbln.platform.RblnPlatform"


def test_register_ops_exposes_triton_kernel_families(monkeypatch):
    monkeypatch.setattr(vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", True, raising=False)

    vllm_rbln.register_ops()

    expected_ops: Iterable[str] = (
        "attention_naive_prefill",
        "causal_attention_naive_prefill",
        "flash_attention_naive_prefill",
        "flash_causal_attention_naive_prefill",
        "sliding_window_attention_naive_prefill",
    )

    for op_name in expected_ops:
        assert hasattr(torch.ops.rbln_triton_ops, op_name), (
            f"Expected torch.ops.rbln_triton_ops.{op_name} to be registered"
        )


def test_register_model_registers_expected_optimum_models(monkeypatch):
    registry = Mock()

    monkeypatch.setattr(
        vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", False, raising=False
    )
    monkeypatch.setattr("vllm.ModelRegistry.register_model", registry)

    vllm_rbln.register_model()

    assert [call.args for call in registry.call_args_list] == [
        (
            "T5WithLMHeadModel",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        ),
        (
            "T5ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.t5:RBLNT5ForConditionalGeneration",
        ),
        (
            "T5EncoderModel",
            "vllm_rbln.model_executor.models.optimum.encoder:RBLNOptimumForEncoderModel",
        ),
        (
            "Gemma3ForConditionalGeneration",
            "vllm_rbln.model_executor.models.optimum.gemma3:RBLNOptimumGemma3ForConditionalGeneration",
        ),
    ]


def test_register_model_skips_registration_in_vllm_model_mode(monkeypatch):
    registry = Mock()

    monkeypatch.setattr(vllm_rbln.envs, "VLLM_RBLN_USE_VLLM_MODEL", True, raising=False)
    monkeypatch.setattr("vllm.ModelRegistry.register_model", registry)

    vllm_rbln.register_model()

    registry.assert_not_called()
