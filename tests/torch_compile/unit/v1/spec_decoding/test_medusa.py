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

"""Unit tests for RBLNMedusaProposer (vllm_rbln/v1/spec_decoding/medusa.py)."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestMedusaPropose:
    """Test the propose() method of RBLNMedusaProposer.

    The propose() method:
    1. Calls model_executable(target_hidden_states) to get logits
    2. Stacks argmax of each logit head into draft_tokens
    """

    def test_propose_single_head(self):
        """Single medusa head produces [batch_size, 1] draft tokens."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        batch_size = 4
        vocab_size = 32000

        # Mock model_executable returning list of 1 logits tensor
        logits = [torch.randn(batch_size, vocab_size)]
        proposer.model_executable = MagicMock(return_value=logits)

        hidden_states = torch.randn(batch_size, 2048)
        sampling_metadata = MagicMock()

        result = proposer.propose(hidden_states, sampling_metadata)

        assert result.shape == (batch_size, 1)
        # Verify argmax correctness
        expected = logits[0].argmax(dim=-1)
        torch.testing.assert_close(result[:, 0], expected)

    def test_propose_multiple_heads(self):
        """Multiple medusa heads produce [batch_size, num_heads] draft tokens."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        batch_size = 4
        vocab_size = 32000
        num_heads = 3

        logits = [torch.randn(batch_size, vocab_size) for _ in range(num_heads)]
        proposer.model_executable = MagicMock(return_value=logits)

        hidden_states = torch.randn(batch_size, 2048)
        sampling_metadata = MagicMock()

        result = proposer.propose(hidden_states, sampling_metadata)

        assert result.shape == (batch_size, num_heads)
        for head_idx in range(num_heads):
            expected = logits[head_idx].argmax(dim=-1)
            torch.testing.assert_close(result[:, head_idx], expected)

    def test_propose_batch_size_one(self):
        """Edge case: single-request batch."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        num_heads = 5

        logits = [torch.randn(1, 100) for _ in range(num_heads)]
        proposer.model_executable = MagicMock(return_value=logits)

        result = proposer.propose(torch.randn(1, 64), MagicMock())

        assert result.shape == (1, num_heads)

    def test_propose_deterministic_argmax(self):
        """Argmax should be deterministic - same input gives same output."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        num_heads = 3

        # Use fixed logits where argmax is unambiguous
        logits = []
        for h in range(num_heads):
            t = torch.zeros(2, 100)
            t[0, h * 10] = 100.0  # Clear winner
            t[1, h * 10 + 5] = 100.0
            logits.append(t)
        proposer.model_executable = MagicMock(return_value=logits)

        result = proposer.propose(torch.randn(2, 64), MagicMock())

        assert result[0, 0].item() == 0
        assert result[0, 1].item() == 10
        assert result[0, 2].item() == 20
        assert result[1, 0].item() == 5
        assert result[1, 1].item() == 15
        assert result[1, 2].item() == 25


class TestMedusaDummyRun:
    """Test the dummy_run() method of RBLNMedusaProposer."""

    def test_dummy_run_creates_correct_tensor(self):
        """dummy_run creates zero tensor with correct shape and calls model."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        proposer.hidden_size = 2048
        proposer.dtype = torch.float16
        proposer.device = torch.device("cpu")
        proposer.vllm_config = MagicMock()
        proposer.model = MagicMock(return_value=torch.randn(4, 2048))

        with patch("vllm_rbln.v1.spec_decoding.medusa.set_forward_context"):
            proposer.dummy_run(batch_size=4)

        proposer.model.assert_called_once()
        call_args = proposer.model.call_args[0][0]
        assert call_args.shape == (4, 2048)
        assert call_args.dtype == torch.float16
        assert torch.all(call_args == 0)


class TestMedusaInit:
    """Test __init__ (lines 29-34): rebel import + CompileContext."""

    def test_init_sets_compile_context(self):
        """__init__ creates CompileContext with use_weight_sharing=True."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        mock_compile_ctx = MagicMock()
        with (
            patch.object(
                RBLNMedusaProposer.__bases__[0], "__init__", return_value=None
            ),
            # CompileContext is imported locally inside __init__
            patch(
                "rebel.compile_context.CompileContext",
                return_value=mock_compile_ctx,
            ) as mock_cc_cls,
        ):
            proposer = RBLNMedusaProposer(MagicMock(), torch.device("cpu"))

        mock_cc_cls.assert_called_once_with(use_weight_sharing=True)
        assert proposer.compile_context is mock_compile_ctx


class TestMedusaLoadModel:
    """Test load_model() branching: eager vs compile."""

    def test_load_model_eager_when_enforce_eager(self):
        """When enforce_eager=True, model_executable is the wrapper (not compiled).
        Also covers lines 39-42 (model_wrapper body) by invoking it.
        """
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        mock_model = MagicMock()
        mock_hidden = torch.randn(2, 64)
        mock_logits = torch.randn(2, 100)
        mock_model.return_value = mock_hidden
        mock_model.compute_logits = MagicMock(return_value=mock_logits)
        proposer.model = mock_model
        proposer.vllm_config = MagicMock()
        proposer.vllm_config.speculative_config.enforce_eager = True

        with patch.object(RBLNMedusaProposer.__bases__[0], "load_model"):
            proposer.load_model(MagicMock())

        assert proposer.model_executable is not None

        # Invoke model_executable to cover lines 40-42
        result = proposer.model_executable(torch.randn(2, 64))
        mock_model.assert_called_once()
        mock_model.compute_logits.assert_called_once_with(mock_hidden)
        assert torch.equal(result, mock_logits)

    def test_load_model_eager_when_compile_disabled(self):
        """When VLLM_RBLN_COMPILE_MODEL=False, use eager path."""
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        proposer.model = MagicMock()
        proposer.vllm_config = MagicMock()
        proposer.vllm_config.speculative_config.enforce_eager = False

        with (
            patch.object(RBLNMedusaProposer.__bases__[0], "load_model"),
            patch("vllm_rbln.v1.spec_decoding.medusa.envs") as mock_envs,
        ):
            mock_envs.VLLM_RBLN_COMPILE_MODEL = False
            proposer.load_model(MagicMock())

        assert proposer.model_executable is not None

    def test_load_model_compile_path(self):
        """When enforce_eager=False and COMPILE_MODEL=True, _compile_model is called.
        Covers line 50 and lines 53-80 (_compile_model).
        """
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        proposer.model = MagicMock()
        proposer.vllm_config = MagicMock()
        proposer.vllm_config.speculative_config.enforce_eager = False
        proposer.compile_context = MagicMock()

        mock_compiled = MagicMock(name="compiled_model")

        with (
            patch.object(RBLNMedusaProposer.__bases__[0], "load_model"),
            patch("vllm_rbln.v1.spec_decoding.medusa.envs") as mock_envs,
            patch("torch.compile", return_value=mock_compiled) as mock_torch_compile,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_tp_group") as mock_tp,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_pp_group") as mock_pp,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_dp_group") as mock_dp,
        ):
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True
            mock_envs.VLLM_RBLN_TP_SIZE = 1
            mock_envs.VLLM_DISABLE_COMPILE_CACHE = True

            # Setup mock process groups
            for mock_group in (mock_tp.return_value, mock_pp.return_value, mock_dp.return_value):
                mock_group.device_group.group_name = f"device_{id(mock_group)}"
                mock_group.cpu_group.group_name = f"cpu_{id(mock_group)}"
                mock_group.ranks = [0]

            proposer.load_model(MagicMock())

        # torch.compile should have been called with rbln backend
        mock_torch_compile.assert_called_once()
        call_kwargs = mock_torch_compile.call_args[1]
        assert call_kwargs["backend"] == "rbln"
        assert call_kwargs["dynamic"] is False
        # options should contain compile_context and strict mode
        options = call_kwargs["options"]
        assert options["compile_context"] is proposer.compile_context
        assert options["tensor_parallel_size"] == 1
        assert options["mode"] == "strict"

        # model_executable should be the compiled model
        assert proposer.model_executable is mock_compiled

    def test_compile_model_with_cache(self):
        """When VLLM_DISABLE_COMPILE_CACHE=False, cache_dir is set in options.
        Covers lines 72-73.
        """
        from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer

        proposer = object.__new__(RBLNMedusaProposer)
        proposer.model = MagicMock()
        proposer.vllm_config = MagicMock()
        proposer.vllm_config.speculative_config.enforce_eager = False
        proposer.compile_context = MagicMock()

        with (
            patch.object(RBLNMedusaProposer.__bases__[0], "load_model"),
            patch("vllm_rbln.v1.spec_decoding.medusa.envs") as mock_envs,
            patch("torch.compile", return_value=MagicMock()) as mock_compile,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_tp_group") as mock_tp,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_pp_group") as mock_pp,
            patch("vllm_rbln.v1.spec_decoding.medusa.get_dp_group") as mock_dp,
        ):
            mock_envs.VLLM_RBLN_COMPILE_MODEL = True
            mock_envs.VLLM_RBLN_TP_SIZE = 1
            mock_envs.VLLM_DISABLE_COMPILE_CACHE = False
            mock_envs.VLLM_CACHE_ROOT = "/tmp/test_cache"

            for mock_group in (mock_tp.return_value, mock_pp.return_value, mock_dp.return_value):
                mock_group.device_group.group_name = f"device_{id(mock_group)}"
                mock_group.cpu_group.group_name = f"cpu_{id(mock_group)}"
                mock_group.ranks = [0]

            proposer.load_model(MagicMock())

        options = mock_compile.call_args[1]["options"]
        assert "cache_dir" in options
        assert options["cache_dir"] == "/tmp/test_cache/rbln"


# NOTE: TestCalcSpecDecodeMetadata, TestTakeDraftTokenIds, and
# TestMedusaHiddenStateSelection are tested in
# tests/torch_compile/unit/v1/spec_decoding/test_model_runner.py
