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

"""Tests for RBLNMedusaProposer methods (spec_decode/medusa.py), built through
its real constructor from vLLM's tiny random medusa pair with the compiled model
left unset (upstream has no medusa unit test, so scenarios are from our source)."""

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.v1.spec_decode.medusa as medusa_module
from tests.native.v1.spec_decode.utils import make_medusa_proposer

pytestmark = pytest.mark.maybe_use_device


class TestInit:
    def test_preallocates_hidden_states_buffer(self):
        proposer = make_medusa_proposer()
        assert proposer.hidden_states.shape == (
            proposer.max_num_seqs,
            proposer.hidden_size,
        )

    def test_no_compile_context_when_device_tensors_on(self, monkeypatch):
        # With device tensors the model runs on-device, so no compile context.
        monkeypatch.setattr(medusa_module, "USE_DEVICE_TENSOR", True)
        proposer = make_medusa_proposer(compile_context=object())
        assert proposer.compile_context is None

    def test_uses_injected_compile_context_when_device_tensors_off(self, monkeypatch):
        monkeypatch.setattr(medusa_module, "USE_DEVICE_TENSOR", False)
        sentinel = object()
        proposer = make_medusa_proposer(compile_context=sentinel)
        assert proposer.compile_context is sentinel

    def test_creates_compile_context_when_absent(self, monkeypatch):
        # Without device tensors and no injected context, one is created.
        monkeypatch.setattr(medusa_module, "USE_DEVICE_TENSOR", False)
        marker = object()
        monkeypatch.setattr(
            medusa_module, "create_compile_context", lambda **kwargs: marker
        )
        proposer = make_medusa_proposer(compile_context=None)
        assert proposer.compile_context is marker


class TestLoadModel:
    @staticmethod
    def _stub_super_load_model(monkeypatch):
        from vllm.v1.spec_decode.medusa import MedusaProposer

        monkeypatch.setattr(
            MedusaProposer,
            "load_model",
            lambda self, target_model: setattr(self, "model", SimpleNamespace()),
        )

    def test_eager_wrapper_when_enforce_eager(self, monkeypatch):
        self._stub_super_load_model(monkeypatch)
        proposer = make_medusa_proposer()
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", True
        )
        proposer.load_model(target_model=object())
        assert proposer.model_executable.__name__ == "model_wrapper"

    def test_compiles_wrapper_when_not_eager(self, monkeypatch):
        self._stub_super_load_model(monkeypatch)
        sentinel = object()
        captured: dict = {}
        monkeypatch.setattr(
            medusa_module, "compile", lambda fn, **kw: captured.update(kw) or sentinel
        )
        monkeypatch.setattr(medusa_module, "build_process_group_dict", lambda: {})
        monkeypatch.setattr(medusa_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        proposer = make_medusa_proposer()
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", False
        )
        proposer.load_model(target_model=object())
        assert proposer.model_executable is sentinel
        assert captured["fullgraph"] is True
        assert "compile_context" in captured

    def test_eager_wrapper_composes_forward(self, monkeypatch):
        # The eager wrapper runs the draft model then returns its compute_logits.
        from vllm.v1.spec_decode.medusa import MedusaProposer

        class _FakeModel:
            def __call__(self, target_hidden_states):
                return target_hidden_states + 100

            def compute_logits(self, hidden_states):
                return [hidden_states + 1]  # a per-head logits list

        monkeypatch.setattr(
            MedusaProposer,
            "load_model",
            lambda self, target_model: setattr(self, "model", _FakeModel()),
        )
        proposer = make_medusa_proposer()
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", True
        )
        proposer.load_model(target_model=object())

        out = proposer.model_executable(torch.tensor([[1.0, 2.0]]))
        # compute_logits(model(t)) = [(t + 100) + 1]
        assert len(out) == 1
        assert out[0].cpu().tolist() == [[102.0, 103.0]]


class TestPropose:
    def test_stacks_headwise_argmax(self):
        # Each medusa head contributes one column; propose argmaxes each head over
        # the active batch and stacks them into [batch, num_heads].
        proposer = make_medusa_proposer(num_speculative_tokens=3)

        def executable(hidden_states):
            outputs = []
            for head_tokens in ([10, 11], [20, 21], [30, 31]):
                logits = torch.full((proposer.max_num_seqs, 128), -10.0)
                for i, tok in enumerate(head_tokens):
                    logits[i, tok] = 10.0
                outputs.append(logits)
            return outputs

        proposer.model_executable = executable
        draft = proposer.propose(
            torch.zeros(2, proposer.hidden_size), sampling_metadata=None
        )
        assert draft.shape == (2, 3)
        assert draft.cpu().tolist() == [[10, 20, 30], [11, 21, 31]]

    def test_stages_target_into_hidden_buffer(self):
        # propose writes the target into the persistent hidden-states buffer
        # before invoking the model.
        proposer = make_medusa_proposer(num_speculative_tokens=3)
        captured = {}

        def executable(hidden_states):
            captured["input"] = hidden_states.clone()
            return [torch.zeros((proposer.max_num_seqs, 128)) for _ in range(3)]

        proposer.model_executable = executable
        target = torch.arange(
            2 * proposer.hidden_size, dtype=proposer.hidden_states.dtype
        ).reshape(2, proposer.hidden_size)
        proposer.propose(target, sampling_metadata=None)
        assert torch.equal(captured["input"][:2].cpu(), target.cpu())


class TestDummyRun:
    def test_runs_and_invokes_model(self):
        proposer = make_medusa_proposer()
        calls = []

        def executable(hidden_states):
            calls.append(1)
            return [torch.zeros((proposer.max_num_seqs, 128))]

        proposer.model_executable = executable
        assert proposer.dummy_run() is None
        assert calls
