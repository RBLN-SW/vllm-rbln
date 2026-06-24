# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.spec_decode.medusa import MedusaProposer

import vllm_rbln.v1.spec_decode.medusa as medusa_module
from vllm_rbln.v1.spec_decode.medusa import RBLNMedusaProposer

DEVICE = torch.device(current_platform.device_type)


class FakeMedusaModel:
    """Draft model double that records the forward / compute_logits order."""

    def __init__(self):
        self.events: list[tuple[str, torch.Tensor]] = []

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append(("forward", hidden_states.clone()))
        return hidden_states + 100

    def compute_logits(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        self.events.append(("compute_logits", hidden_states.clone()))
        return [hidden_states + 1, hidden_states + 2]


def make_vllm_config(*, max_num_seqs: int = 4, enforce_eager: bool = True):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=SimpleNamespace(enforce_eager=enforce_eager),
    )


def make_proposer_stub(
    *,
    max_num_seqs: int = 4,
    hidden_size: int = 4,
    dtype: torch.dtype = torch.float32,
    enforce_eager: bool = True,
    compile_context: object | None = None,
) -> RBLNMedusaProposer:
    """Build a proposer without running the heavy base __init__.

    Used by load_model/propose/dummy_run tests, which only read attributes
    rather than exercise construction.
    """
    stub = object.__new__(RBLNMedusaProposer)
    stub.max_num_seqs = max_num_seqs
    stub.hidden_size = hidden_size
    stub.dtype = dtype
    stub.device = DEVICE
    stub.hidden_states = torch.zeros(
        max_num_seqs, hidden_size, device=DEVICE, dtype=dtype
    )
    stub.compile_context = object() if compile_context is None else compile_context
    stub.vllm_config = make_vllm_config(
        max_num_seqs=max_num_seqs, enforce_eager=enforce_eager
    )
    return stub


def patch_base_init(
    monkeypatch,
    *,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Stand in for MedusaProposer.__init__ so real __init__ can run on RBLN."""

    def fake_super_init(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.hidden_size = hidden_size
        self.dtype = dtype

    monkeypatch.setattr(MedusaProposer, "__init__", fake_super_init)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


# Verifies __init__ preallocates a persistent buffer padded to max_num_seqs.
def test_init_preallocates_padded_hidden_states(monkeypatch):
    patch_base_init(monkeypatch, hidden_size=6, dtype=torch.float16)
    monkeypatch.setattr(
        medusa_module, "create_compile_context", Mock(return_value=object())
    )
    vllm_config = make_vllm_config(max_num_seqs=4)

    proposer = RBLNMedusaProposer(vllm_config, DEVICE)

    assert proposer.hidden_states.shape == (4, 6)
    assert proposer.hidden_states.dtype == torch.float16
    assert proposer.hidden_states.device == DEVICE
    assert torch.count_nonzero(proposer.hidden_states) == 0


# Verifies an injected compile context is used verbatim and no new one is made.
def test_init_uses_injected_compile_context(monkeypatch):
    patch_base_init(monkeypatch, hidden_size=4, dtype=torch.float32)
    create_compile_context = Mock()
    monkeypatch.setattr(medusa_module, "create_compile_context", create_compile_context)
    injected = object()

    proposer = RBLNMedusaProposer(make_vllm_config(), DEVICE, compile_context=injected)

    assert proposer.compile_context is injected
    create_compile_context.assert_not_called()


# Verifies __init__ builds a weight-sharing compile context when none is given.
def test_init_creates_compile_context_when_not_provided(monkeypatch):
    patch_base_init(monkeypatch, hidden_size=4, dtype=torch.float32)
    sentinel = object()
    create_compile_context = Mock(return_value=sentinel)
    monkeypatch.setattr(medusa_module, "create_compile_context", create_compile_context)

    proposer = RBLNMedusaProposer(make_vllm_config(), DEVICE)

    create_compile_context.assert_called_once_with(use_weight_sharing=True)
    assert proposer.compile_context is sentinel


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


# Verifies either eager trigger installs the direct wrapper and skips compile.
@pytest.mark.parametrize(
    ("enforce_eager", "compile_model"),
    [
        (True, True),  # enforce_eager arm
        (False, False),  # VLLM_RBLN_COMPILE_MODEL arm
    ],
)
def test_load_model_uses_eager_wrapper(monkeypatch, enforce_eager, compile_model):
    fake = make_proposer_stub(enforce_eager=enforce_eager)
    fake_model = FakeMedusaModel()

    def fake_super_load_model(self, target_model):
        self.model = fake_model

    compile_mock = Mock(side_effect=AssertionError("compile must not run"))
    monkeypatch.setattr(MedusaProposer, "load_model", fake_super_load_model)
    monkeypatch.setattr(medusa_module, "compile", compile_mock)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_MODEL", compile_model)

    RBLNMedusaProposer.load_model(fake, target_model=object())

    compile_mock.assert_not_called()

    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    logits = fake.model_executable(hidden_states)

    # Wrapper composes forward(...) then compute_logits(...) in that order.
    assert [name for name, _ in fake_model.events] == ["forward", "compute_logits"]
    torch.testing.assert_close(fake_model.events[0][1], hidden_states)
    torch.testing.assert_close(fake_model.events[1][1], hidden_states + 100)
    torch.testing.assert_close(logits[0], hidden_states + 101)
    torch.testing.assert_close(logits[1], hidden_states + 102)


# Verifies the compile path routes the wrapper through compile() with the
# proposer's contract args, and mode tracks the strict-mode env flag.
@pytest.mark.parametrize("strict", [True, False])
def test_load_model_compiles_wrapper(monkeypatch, strict):
    fake = make_proposer_stub(enforce_eager=False)
    fake_model = FakeMedusaModel()
    compiled_sentinel = object()
    process_group_sentinel = object()
    captured: dict[str, object] = {}

    def fake_super_load_model(self, target_model):
        self.model = fake_model

    def fake_compile(target, **kwargs):
        captured["target"] = target
        captured.update(kwargs)
        return compiled_sentinel

    monkeypatch.setattr(MedusaProposer, "load_model", fake_super_load_model)
    monkeypatch.setattr(medusa_module, "compile", fake_compile)
    monkeypatch.setattr(
        medusa_module, "build_process_group_dict", lambda: process_group_sentinel
    )
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", 8)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_STRICT_MODE", strict)

    RBLNMedusaProposer.load_model(fake, target_model=object())

    assert fake.model_executable is compiled_sentinel
    assert captured["dynamic"] is False
    assert captured["fullgraph"] is True
    assert captured["compile_context"] is fake.compile_context
    assert captured["tensor_parallel_size"] == 8
    assert captured["process_group_dict"] is process_group_sentinel
    assert captured["guard_filter_fn"] is torch.compiler.keep_tensor_guards_unsafe
    assert captured["mode"] == ("strict" if strict else "")

    # The compiled target is the wrapper composing forward then compute_logits.
    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    target = cast(Callable[[torch.Tensor], list[torch.Tensor]], captured["target"])
    logits = target(hidden_states)
    assert [name for name, _ in fake_model.events] == ["forward", "compute_logits"]
    torch.testing.assert_close(logits[0], hidden_states + 101)
    torch.testing.assert_close(logits[1], hidden_states + 102)


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


# Verifies propose writes the active rows into the persistent buffer in place,
# preserves padding rows, and hands the whole buffer to the executable.
def test_propose_uses_padded_hidden_state_buffer():
    fake = make_proposer_stub(max_num_seqs=4, hidden_size=4)
    sentinel = torch.full((2, 4), -999.0, dtype=torch.float32)
    fake.hidden_states[2:] = sentinel
    captured: dict[str, object] = {}

    def model_executable(hidden_states: torch.Tensor) -> list[torch.Tensor]:
        captured["buffer"] = hidden_states
        return [torch.zeros((4, 3), dtype=torch.float32)]

    fake.model_executable = model_executable
    target_hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)

    RBLNMedusaProposer.propose(
        fake, target_hidden_states=target_hidden_states, sampling_metadata=None
    )

    # Same object passed through (static address matters for compile).
    assert captured["buffer"] is fake.hidden_states
    torch.testing.assert_close(fake.hidden_states[:2], target_hidden_states)
    torch.testing.assert_close(fake.hidden_states[2:], sentinel)


# Verifies per-head argmax is stacked into [batch, num_heads] for the active
# batch, ignoring the padded tail rows of the logits.
def test_propose_returns_headwise_argmax_for_active_batch():
    fake = make_proposer_stub(max_num_seqs=4, hidden_size=4)
    pad = [9.0, 9.0, 9.0]
    fake.model_executable = lambda hidden_states: [
        torch.tensor([[0.0, 4.0, 1.0], [5.0, 1.0, 0.0], pad, pad]),
        torch.tensor([[7.0, 1.0, 0.0], [0.0, 2.0, 6.0], pad, pad]),
        torch.tensor([[0.0, 1.0, 8.0], [9.0, 1.0, 0.0], pad, pad]),
    ]

    output = RBLNMedusaProposer.propose(
        fake,
        target_hidden_states=torch.zeros((2, 4), dtype=torch.float32),
        sampling_metadata=None,
    )

    assert output.shape == (2, 3)
    torch.testing.assert_close(
        output, torch.tensor([[1, 0, 2], [0, 2, 0]], dtype=torch.int64)
    )


# ---------------------------------------------------------------------------
# dummy_run
# ---------------------------------------------------------------------------


# Verifies dummy_run executes the persistent buffer rather than a fresh tensor.
def test_dummy_run_executes_persistent_hidden_states():
    fake = make_proposer_stub(max_num_seqs=3, hidden_size=6)
    calls: list[torch.Tensor] = []

    def model_executable(hidden_states: torch.Tensor) -> None:
        calls.append(hidden_states)

    fake.model_executable = model_executable

    RBLNMedusaProposer.dummy_run(fake)

    assert len(calls) == 1
    assert calls[0] is fake.hidden_states
