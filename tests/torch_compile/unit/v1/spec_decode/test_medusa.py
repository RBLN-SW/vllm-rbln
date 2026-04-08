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

import sys
import types
from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.spec_decode.medusa import MedusaProposer

import vllm_rbln.v1.spec_decoding.medusa as medusa_module
from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer


class FakeMedusaModel:
    def __init__(self):
        self.forward_inputs: list[torch.Tensor] = []
        self.logit_inputs: list[torch.Tensor] = []

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.forward_inputs.append(hidden_states.clone())
        return hidden_states + 100

    def compute_logits(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        self.logit_inputs.append(hidden_states.clone())
        return [
            hidden_states + 1,
            hidden_states + 2,
        ]


class CaptureModel:
    def __init__(self):
        self.calls: list[torch.Tensor] = []

    def __call__(self, hidden_states: torch.Tensor) -> None:
        self.calls.append(hidden_states.clone())


def make_fake_proposer(
    *,
    hidden_size: int = 4,
    dtype: torch.dtype = torch.float32,
) -> RBLNMedusaProposer:
    fake = object.__new__(RBLNMedusaProposer)
    fake.hidden_size = hidden_size
    fake.dtype = dtype
    fake.device = torch.device("cpu")
    fake.compile_context = object()
    fake.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=True)
    )
    return fake


def make_fake_group(group_name: str, ranks: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        device_group=SimpleNamespace(group_name=f"{group_name}_device"),
        cpu_group=SimpleNamespace(group_name=f"{group_name}_cpu"),
        ranks=ranks,
    )


# Verifies that __init__ delegates to the base proposer and creates a compile context.
def test_init_creates_compile_context_with_weight_sharing(monkeypatch):
    super_init = Mock()
    compile_context_instance = object()
    compile_context_ctor = Mock(return_value=compile_context_instance)

    def fake_super_init(self, vllm_config, device):
        super_init(vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device

    rebel_module = types.ModuleType("rebel")
    compile_context_module = types.ModuleType("rebel.compile_context")
    compile_context_module.CompileContext = compile_context_ctor  # type: ignore[attr-defined]
    rebel_module.compile_context = compile_context_module  # type: ignore[attr-defined]

    monkeypatch.setattr(MedusaProposer, "__init__", fake_super_init)
    monkeypatch.setitem(sys.modules, "rebel", rebel_module)
    monkeypatch.setitem(sys.modules, "rebel.compile_context", compile_context_module)

    vllm_config = object()
    device = torch.device("cpu")

    proposer = RBLNMedusaProposer(vllm_config, device)

    super_init.assert_called_once_with(vllm_config, device)
    compile_context_ctor.assert_called_once_with(use_weight_sharing=True)
    assert proposer.compile_context is compile_context_instance


# Verifies that Medusa stacks per-head argmax outputs into [batch, num_heads].
def test_propose_returns_headwise_argmax_stack():
    fake = make_fake_proposer()
    fake.model_executable = lambda hidden_states: [
        torch.tensor(
            [[0.0, 4.0, 1.0], [5.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[7.0, 1.0, 0.0], [0.0, 2.0, 6.0]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.0, 1.0, 8.0], [9.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    ]

    output = RBLNMedusaProposer.propose(
        fake,
        target_hidden_states=torch.zeros((2, fake.hidden_size), dtype=torch.float32),
        sampling_metadata=None,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[1, 0, 2], [0, 2, 0]], dtype=torch.int64),
    )


# Verifies that eager mode installs the direct model wrapper and skips compile.
def test_load_model_uses_eager_wrapper_when_enforce_eager(monkeypatch):
    fake = make_fake_proposer()
    fake.vllm_config.speculative_config.enforce_eager = True
    fake_model = FakeMedusaModel()

    def fake_super_load_model(self, target_model):
        self.model = fake_model

    monkeypatch.setattr(MedusaProposer, "load_model", fake_super_load_model)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    fake._compile_model = Mock(
        side_effect=AssertionError("_compile_model should not be used")
    )

    RBLNMedusaProposer.load_model(fake, target_model=object())

    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    logits = fake.model_executable(hidden_states)

    fake._compile_model.assert_not_called()
    torch.testing.assert_close(fake_model.forward_inputs[0], hidden_states)
    torch.testing.assert_close(fake_model.logit_inputs[0], hidden_states + 100)
    torch.testing.assert_close(logits[0], hidden_states + 101)
    torch.testing.assert_close(logits[1], hidden_states + 102)


# Verifies that compile mode routes the wrapper through _compile_model.
def test_load_model_uses_compile_wrapper_when_enabled(monkeypatch):
    fake = make_fake_proposer()
    fake.vllm_config.speculative_config.enforce_eager = False
    fake_model = FakeMedusaModel()
    compiled_sentinel = object()
    captured_wrapper: Callable[[torch.Tensor], list[torch.Tensor]] | None = None

    def fake_super_load_model(self, target_model):
        self.model = fake_model

    def fake_compile_model(
        wrapper: Callable[[torch.Tensor], list[torch.Tensor]],
    ):
        nonlocal captured_wrapper
        captured_wrapper = wrapper
        return compiled_sentinel

    monkeypatch.setattr(MedusaProposer, "load_model", fake_super_load_model)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
    fake._compile_model = Mock(side_effect=fake_compile_model)

    RBLNMedusaProposer.load_model(fake, target_model=object())

    assert fake.model_executable is compiled_sentinel
    fake._compile_model.assert_called_once()

    hidden_states = torch.arange(8, dtype=torch.float32).view(2, 4)
    assert captured_wrapper is not None
    logits = captured_wrapper(hidden_states)

    torch.testing.assert_close(fake_model.forward_inputs[0], hidden_states)
    torch.testing.assert_close(fake_model.logit_inputs[0], hidden_states + 100)
    torch.testing.assert_close(logits[0], hidden_states + 101)
    torch.testing.assert_close(logits[1], hidden_states + 102)


# Verifies that RBLN compile options include process groups and optional cache_dir.
@pytest.mark.parametrize(
    ("disable_compile_cache", "expect_cache_dir"),
    [
        (False, True),
        (True, False),
    ],
)
def test_compile_model_builds_expected_rbln_compile_options(
    monkeypatch, disable_compile_cache: bool, expect_cache_dir: bool
):
    fake = make_fake_proposer()
    tp_group = make_fake_group("tp", [0, 1])
    pp_group = make_fake_group("pp", [0])
    dp_group = make_fake_group("dp", [0, 2])
    captured: dict[str, Any] = {}
    compiled_sentinel = object()

    monkeypatch.setattr(medusa_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(medusa_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(medusa_module, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_TP_SIZE", 8)
    monkeypatch.setattr(
        medusa_module.envs, "VLLM_DISABLE_COMPILE_CACHE", disable_compile_cache
    )
    monkeypatch.setattr(medusa_module.envs, "VLLM_CACHE_ROOT", "/tmp/test-cache")

    def fake_torch_compile(model, backend, options, dynamic):
        captured["model"] = model
        captured["backend"] = backend
        captured["options"] = options
        captured["dynamic"] = dynamic
        return compiled_sentinel

    monkeypatch.setattr(medusa_module.torch, "compile", fake_torch_compile)

    model = Mock()
    output = RBLNMedusaProposer._compile_model(fake, model)

    assert output is compiled_sentinel
    assert captured["model"] is model
    assert captured["backend"] == "rbln"
    assert captured["dynamic"] is False

    options = cast(dict[str, Any], captured["options"])
    assert options["compile_context"] is fake.compile_context
    assert options["tensor_parallel_size"] == 8
    assert options["guard_filter_fn"] is torch.compiler.keep_tensor_guards_unsafe
    assert options["mode"] == "strict"
    assert options["process_group_dict"] == {
        "tp_device": [0, 1],
        "tp_cpu": [0, 1],
        "pp_device": [0],
        "pp_cpu": [0],
        "dp_device": [0, 2],
        "dp_cpu": [0, 2],
    }
    if expect_cache_dir:
        assert options["cache_dir"] == "/tmp/test-cache/rbln"
    else:
        assert "cache_dir" not in options


# Verifies that dummy_run uses the requested batch size in both context and input shape.
def test_dummy_run_uses_batch_sized_hidden_states_and_forward_context(monkeypatch):
    fake = make_fake_proposer(hidden_size=6, dtype=torch.float16)
    fake.model = CaptureModel()
    captured: dict[str, object] = {}

    @contextmanager
    def fake_forward_context(attn_metadata, vllm_config, num_tokens):
        captured["attn_metadata"] = attn_metadata
        captured["vllm_config"] = vllm_config
        captured["num_tokens"] = num_tokens
        yield

    monkeypatch.setattr(medusa_module, "set_forward_context", fake_forward_context)

    RBLNMedusaProposer.dummy_run(fake, batch_size=3)

    assert captured["attn_metadata"] is None
    assert captured["vllm_config"] is fake.vllm_config
    assert captured["num_tokens"] == 3
    assert len(fake.model.calls) == 1
    assert fake.model.calls[0].shape == (3, 6)
    assert fake.model.calls[0].dtype == torch.float16
