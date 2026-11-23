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

import vllm_rbln.model_executor.kernels.residual as residual_kernels


@pytest.fixture
def calls(monkeypatch):
    """Record the residual ops the rbln implementation calls, running the real ones."""
    ops = torch.ops.rbln_custom_ops
    write, read = ops.residual_write, ops.residual_read
    calls = []

    def recording_write(*args):
        calls.append("write")
        return write(*args)

    def recording_read(*args):
        calls.append("read")
        return read(*args)

    monkeypatch.setattr(ops, "residual_write", recording_write)
    monkeypatch.setattr(ops, "residual_read", recording_read)
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
