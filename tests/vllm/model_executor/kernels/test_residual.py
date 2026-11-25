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
from vllm import ir
from vllm.config import set_current_vllm_config

import vllm_rbln.model_executor.kernels.residual as residual_kernels
from tests.vllm.vllm_config import make_vllm_config


@pytest.fixture
def calls(monkeypatch):
    """Record the residual ops the rbln implementation calls, running the real ones."""
    ops = torch.ops.rbln_custom_ops
    calls = []

    def recording(name, op):
        def call(*args):
            calls.append(name)
            return op(*args)

        return call

    for name, op_name in (
        ("write", "residual_write"),
        ("read", "residual_read"),
        ("zero_centered_read", "zero_centered_residual_read"),
    ):
        monkeypatch.setattr(ops, op_name, recording(name, getattr(ops, op_name)))
    return calls


def _fused_add_rms_norm(monkeypatch, x, residual, weight, prefill=True):
    monkeypatch.setattr(residual_kernels, "is_prefill_step", lambda: prefill)
    # The vllm model path traces with torch wrapping off, so the op dispatches here.
    with (
        ir.enable_torch_wrap(False),
        ir.ops.fused_add_rms_norm.set_priority(["rbln", "native"]),
    ):
        return ir.ops.fused_add_rms_norm(x, residual, weight, 1e-6)


@pytest.mark.parametrize("prefill, ops", [(True, ["write", "read"]), (False, [])])
def test_prefill_splits_the_boundary_and_decode_stays_native(
    calls, monkeypatch, prefill, ops
):
    x, residual, weight = (
        torch.randn(2, 3, 64),
        torch.randn(2, 3, 64),
        torch.randn(64),
    )
    out, new_residual = _fused_add_rms_norm(monkeypatch, x, residual, weight, prefill)

    native_out, native_residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x, residual, weight, 1e-6
    )
    assert calls == ops
    torch.testing.assert_close(new_residual, native_residual)
    torch.testing.assert_close(out, native_out)


def test_rows_without_a_batch_axis_stay_native(calls, monkeypatch):
    x, residual, weight = (torch.randn(3, 64), torch.randn(3, 64), torch.randn(64))
    _fused_add_rms_norm(monkeypatch, x, residual, weight)

    assert calls == []


def _gemma_norm(cuts_residual):
    with set_current_vllm_config(make_vllm_config()):
        norm = residual_kernels.RBLNGemmaRMSNorm(64)
    norm.weight.data.normal_(std=0.3)
    norm.cuts_residual = cuts_residual
    return norm


@pytest.mark.parametrize(
    "prefill, ops",
    [(True, ["write", "zero_centered_read"]), (False, ["zero_centered_read"])],
)
def test_gemma_norm_reads_the_stored_weight_on_the_residual_ops(
    calls, monkeypatch, prefill, ops
):
    monkeypatch.setattr(residual_kernels, "is_prefill_step", lambda: prefill)
    norm = _gemma_norm(cuts_residual=True)
    x, residual = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    out, new_residual = norm(x, residual)

    native_out, native_residual = norm.forward_native(x, residual)
    assert calls == ops
    torch.testing.assert_close(new_residual, native_residual)
    torch.testing.assert_close(out, native_out)


def test_gemma_norm_of_other_models_stays_upstream(calls):
    norm = _gemma_norm(cuts_residual=False)
    norm(torch.randn(2, 3, 64), torch.randn(2, 3, 64))

    assert calls == []


def test_gemma_norm_is_registered_as_oot():
    from vllm.model_executor.custom_op import maybe_get_oot_by_class
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    assert maybe_get_oot_by_class(GemmaRMSNorm) is residual_kernels.RBLNGemmaRMSNorm
