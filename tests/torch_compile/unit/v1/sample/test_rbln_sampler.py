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

"""Unit tests for RBLNSampler, helper functions, and penalties."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample.rbln_sampler import (
    WARM_UP_CONFIGS,
    _SAMPLING_EPS,
    apply_top_k_top_p,
    random_sample,
)


# ===========================================================================
# Tests: random_sample
# ===========================================================================


class TestRandomSample:
    def test_basic(self):
        """Basic sampling returns one token per row."""
        probs = torch.softmax(torch.randn(4, 10), dim=-1)
        result = random_sample(probs, {})
        assert result.shape == (4,)
        assert result.dtype == torch.int64

    def test_with_generators(self):
        """Per-request generators produce deterministic results."""
        probs = torch.softmax(torch.randn(3, 10), dim=-1)
        gen = {0: torch.Generator().manual_seed(42)}
        r1 = random_sample(probs.clone(), {0: torch.Generator().manual_seed(42)})
        r2 = random_sample(probs.clone(), {0: torch.Generator().manual_seed(42)})
        assert r1[0] == r2[0]

    def test_all_generators(self):
        """When all rows have generators, the branch len(generators) != shape[0] is False."""
        probs = torch.softmax(torch.randn(2, 10), dim=-1)
        gens = {
            0: torch.Generator().manual_seed(0),
            1: torch.Generator().manual_seed(1),
        }
        result = random_sample(probs, gens)
        assert result.shape == (2,)

    def test_empty_generators(self):
        probs = torch.softmax(torch.randn(2, 5), dim=-1)
        result = random_sample(probs, {})
        assert result.shape == (2,)

    def test_single_row(self):
        probs = torch.softmax(torch.randn(1, 100), dim=-1)
        result = random_sample(probs, {})
        assert result.shape == (1,)
        assert 0 <= result.item() < 100


# ===========================================================================
# Tests: apply_top_k_top_p
# ===========================================================================


class TestApplyTopKTopP:
    def test_greedy_no_k_no_p(self):
        """Without k and p, should return argmax."""
        logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 1.0, 2.0]])
        result = apply_top_k_top_p(logits, None, None)
        assert result.tolist() == [1, 0]

    def test_top_k_only(self):
        torch.manual_seed(0)
        logits = torch.randn(10, 100)
        k = torch.full((10,), 5, dtype=torch.int64)
        result = apply_top_k_top_p(logits, k, None)
        assert result.shape == (10,)
        # All results should be within top-k indices
        for i in range(10):
            topk_indices = logits[i].topk(5).indices
            assert result[i] in topk_indices

    def test_top_p_only(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 20)
        p = torch.full((5,), 0.9, dtype=torch.float32)
        result = apply_top_k_top_p(logits, None, p)
        assert result.shape == (5,)

    def test_top_k_and_top_p(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 50)
        k = torch.full((3,), 10, dtype=torch.int64)
        p = torch.full((3,), 0.9, dtype=torch.float32)
        result = apply_top_k_top_p(logits, k, p)
        assert result.shape == (3,)

    def test_top_k_equals_1(self):
        """top_k=1 should be equivalent to greedy."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        k = torch.tensor([1], dtype=torch.int64)
        result = apply_top_k_top_p(logits, k, None)
        assert result.item() == 1  # index of 5.0


# ===========================================================================
# Tests: RBLNSampler
# ===========================================================================


class TestRBLNSampler:
    @pytest.fixture
    def sampler(self):
        """Create an RBLNSampler with mocked compilation."""
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext") as mock_cc, \
             patch("torch.compile") as mock_compile, \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", False):
            mock_cc.return_value = MagicMock()
            mock_compile.return_value = MagicMock()
            from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler
            s = RBLNSampler(logprobs_mode="raw_logprobs", seed=42)
        return s

    @pytest.fixture
    def sampler_with_logits_mode(self):
        """Create an RBLNSampler with raw_logits mode."""
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext") as mock_cc, \
             patch("torch.compile") as mock_compile, \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", False):
            mock_cc.return_value = MagicMock()
            mock_compile.return_value = MagicMock()
            from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler
            s = RBLNSampler(logprobs_mode="raw_logits", seed=42)
        return s

    def test_unsupported_logprobs_mode(self):
        """processed_logits mode should fall back to native sampler."""
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext"), \
             patch("torch.compile"), \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", False):
            from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler
            # Should not raise, just log warning
            s = RBLNSampler(logprobs_mode="processed_logits", seed=42)

    def test_apply_penalties_no_penalties(self, sampler):
        logits = torch.randn(2, 10)
        metadata = MagicMock()
        metadata.no_penalties = True
        result = sampler.apply_penalties(logits, metadata, [[1, 2], [3]])
        assert torch.equal(result, logits)

    def test_apply_penalties_with_penalties(self, sampler):
        logits = torch.randn(2, 10)
        prompt_ids = torch.tensor([[1, 2, 0], [3, 4, 0]])
        metadata = MagicMock()
        metadata.no_penalties = False
        metadata.prompt_token_ids = prompt_ids
        metadata.presence_penalties = torch.tensor([0.1, 0.1])
        metadata.frequency_penalties = torch.tensor([0.1, 0.1])
        metadata.repetition_penalties = torch.tensor([1.0, 1.0])

        with patch(
            "vllm_rbln.v1.sample.rbln_sampler.rbln_apply_all_penalties",
            return_value=logits,
        ) as mock_apply:
            result = sampler.apply_penalties(logits, metadata, [[1], [2]])
        mock_apply.assert_called_once()

    def test_greedy_sample(self, sampler):
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        # nn.Module __setattr__ blocks non-Module assignment, use __dict__
        orig = sampler.__dict__.get("topk_topp_sampler")
        mock_fn = MagicMock(return_value=(torch.tensor([1]), None))
        sampler.__dict__["topk_topp_sampler"] = mock_fn
        try:
            result = sampler.greedy_sample(logits)
            assert result.item() == 1
            mock_fn.assert_called_once()
        finally:
            if orig is not None:
                sampler.__dict__["topk_topp_sampler"] = orig

    def test_apply_temperature_all_random(self, sampler):
        logits = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        temp = torch.tensor([2.0, 0.5])
        result = sampler.apply_temperature(logits, temp, all_random=True)
        expected = logits / temp.unsqueeze(1)
        assert torch.allclose(result, expected)

    def test_apply_temperature_not_all_random(self, sampler):
        """When not all_random, zero temps are replaced with 1.0."""
        logits = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        temp = torch.tensor([0.0, 0.5])  # first request is greedy
        result = sampler.apply_temperature(logits, temp, all_random=False)
        # temp[0] = 0.0 < eps => becomes 1.0
        expected_temp = torch.tensor([1.0, 0.5])
        expected = logits / expected_temp.unsqueeze(1)
        assert torch.allclose(result, expected)

    def test_forward_no_logprobs(self, sampler):
        """Forward without logprobs."""
        logits = torch.randn(2, 10)
        metadata = MagicMock()
        metadata.max_num_logprobs = None
        metadata.no_penalties = True

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), None)
        )

        output = sampler.forward(logits, metadata)
        assert output.sampled_token_ids.shape == (2, 1)
        assert output.logprobs_tensors is None

    def test_forward_with_logprobs_raw_logprobs(self, sampler):
        """Forward with raw_logprobs mode and max_num_logprobs set."""
        logits = torch.randn(2, 10)
        metadata = MagicMock()
        metadata.max_num_logprobs = 3

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), None)
        )
        sampler.compute_logprobs = MagicMock(
            return_value=torch.randn(2, 10)
        )
        sampler.gather_logprobs = MagicMock(
            return_value=MagicMock()
        )

        output = sampler.forward(logits, metadata)
        sampler.compute_logprobs.assert_called_once()
        sampler.gather_logprobs.assert_called_once()

    def test_forward_with_logprobs_raw_logits(self, sampler_with_logits_mode):
        """Forward with raw_logits mode."""
        sampler = sampler_with_logits_mode
        logits = torch.randn(2, 10, dtype=torch.float32)
        metadata = MagicMock()
        metadata.max_num_logprobs = 3

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), None)
        )
        sampler.gather_logprobs = MagicMock(return_value=MagicMock())

        output = sampler.forward(logits, metadata)
        sampler.gather_logprobs.assert_called_once()

    def test_forward_with_logprobs_raw_logits_non_fp32(self, sampler_with_logits_mode):
        """When logits are not fp32, they should be cast."""
        sampler = sampler_with_logits_mode
        logits = torch.randn(2, 10, dtype=torch.bfloat16)
        metadata = MagicMock()
        metadata.max_num_logprobs = 3

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), None)
        )
        sampler.gather_logprobs = MagicMock(return_value=MagicMock())

        output = sampler.forward(logits, metadata)
        sampler.gather_logprobs.assert_called_once()

    def test_forward_full_logprobs(self, sampler):
        """max_num_logprobs == -1 returns full logprobs."""
        logits = torch.randn(2, 10)
        metadata = MagicMock()
        metadata.max_num_logprobs = -1

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), None)
        )
        sampler.compute_logprobs = MagicMock(return_value=torch.randn(2, 10))

        output = sampler.forward(logits, metadata)
        assert output.logprobs_tensors is not None

    def test_forward_processed_logprobs(self, sampler):
        """When sample returns processed_logprobs, use them as raw_logprobs."""
        logits = torch.randn(2, 10)
        metadata = MagicMock()
        metadata.max_num_logprobs = 3
        processed = torch.randn(2, 10)

        sampler.apply_logits_processors = MagicMock(return_value=logits)
        sampler.sample = MagicMock(
            return_value=(torch.tensor([1, 2], dtype=torch.long), processed)
        )
        sampler.compute_logprobs = MagicMock(return_value=torch.randn(2, 10))
        sampler.gather_logprobs = MagicMock(return_value=MagicMock())

        output = sampler.forward(logits, metadata)
        # gather_logprobs should receive processed logprobs
        call_args = sampler.gather_logprobs.call_args
        assert torch.equal(call_args[0][0], processed)


# ===========================================================================
# Tests: RBLNTopKTopPSampler
# ===========================================================================


class TestRBLNTopKTopPSampler:
    @pytest.fixture
    def topk_sampler(self):
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext") as mock_cc, \
             patch("torch.compile") as mock_compile, \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", False):
            mock_cc.return_value = MagicMock()
            mock_compile.return_value = MagicMock(return_value=torch.tensor([1]))
            from vllm_rbln.v1.sample.rbln_sampler import RBLNTopKTopPSampler
            s = RBLNTopKTopPSampler(logprobs_mode="raw_logprobs", seed=42)
        return s

    def test_forward_rbln(self, topk_sampler):
        logits = torch.randn(2, 10)
        k = torch.tensor([5, 5])
        p = torch.tensor([0.9, 0.9])
        result, logprobs = topk_sampler.forward_rbln(logits, {}, k, p)
        assert logprobs is None

    def test_forward_rbln_with_generators_logs_warning(self, topk_sampler):
        logits = torch.randn(2, 10)
        gens = {0: torch.Generator().manual_seed(0)}
        result, logprobs = topk_sampler.forward_rbln(logits, gens, None, None)
        assert logprobs is None

    def test_strict_mode(self):
        """Strict mode adds mode='strict' to compile options."""
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext") as mock_cc, \
             patch("torch.compile") as mock_compile, \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", True):
            mock_cc.return_value = MagicMock()
            mock_compile.return_value = MagicMock()
            from vllm_rbln.v1.sample.rbln_sampler import RBLNTopKTopPSampler
            s = RBLNTopKTopPSampler(logprobs_mode="raw_logprobs", seed=42)
        call_kwargs = mock_compile.call_args
        assert call_kwargs.kwargs.get("options", {}).get("mode") == "strict"

    def test_with_torch_rbln(self):
        """When has_torch_rbln is True, use_global_ctx and device options are set."""
        with patch("rebel.manual_seed"), \
             patch("rebel.CompileContext") as mock_cc, \
             patch("torch.compile") as mock_compile, \
             patch("vllm_rbln.v1.sample.rbln_sampler.envs.VLLM_RBLN_COMPILE_STRICT_MODE", False), \
             patch("vllm_rbln.v1.sample.rbln_sampler.has_torch_rbln", True):
            mock_cc.return_value = MagicMock()
            mock_compile.return_value = MagicMock()
            from vllm_rbln.v1.sample.rbln_sampler import RBLNTopKTopPSampler
            s = RBLNTopKTopPSampler(logprobs_mode="raw_logprobs", seed=42)
        opts = mock_compile.call_args.kwargs.get("options", {})
        assert opts.get("use_global_ctx") is True
        assert opts.get("global_device_id") == 0


# ===========================================================================
# Tests: penalties
# ===========================================================================


class TestPenalties:
    def test_apply_all_penalties(self):
        from vllm_rbln.v1.sample.ops.penalties import apply_all_penalties

        logits = torch.randn(2, 10)
        prompt_ids = torch.tensor([[1, 2], [3, 4]])
        presence = torch.tensor([0.0, 0.0])
        frequency = torch.tensor([0.0, 0.0])
        repetition = torch.tensor([1.0, 1.0])
        output_ids = [[5], [6]]

        result = apply_all_penalties(
            logits, prompt_ids, presence, frequency, repetition, output_ids
        )
        assert result.shape == logits.shape

    def test_apply_all_penalties_with_negative_ids(self):
        """Placeholder -1 token ids should be replaced."""
        from vllm_rbln.v1.sample.ops.penalties import apply_all_penalties

        logits = torch.randn(1, 10)
        prompt_ids = torch.tensor([[1, 2]])
        presence = torch.tensor([0.1])
        frequency = torch.tensor([0.1])
        repetition = torch.tensor([1.0])
        output_ids = [[-1, 3]]

        result = apply_all_penalties(
            logits, prompt_ids, presence, frequency, repetition, output_ids
        )
        assert result.shape == logits.shape

    def test_convert_to_tensors(self):
        from vllm_rbln.v1.sample.ops.penalties import _convert_to_tensors

        output_ids = [[1, 2], [3]]
        result = _convert_to_tensors(output_ids, vocab_size=10, device=torch.device("cpu"))
        assert result.shape[0] == 2
        assert result.dtype == torch.int64

    def test_convert_to_tensors_empty(self):
        from vllm_rbln.v1.sample.ops.penalties import _convert_to_tensors

        output_ids = [[], []]
        result = _convert_to_tensors(output_ids, vocab_size=10, device=torch.device("cpu"))
        assert result.shape[0] == 2


# ===========================================================================
# Tests: WARM_UP_CONFIGS
# ===========================================================================


class TestWarmUpConfigs:
    def test_configs_exist(self):
        assert len(WARM_UP_CONFIGS) == 8

    def test_each_config_has_required_keys(self):
        required = {"name", "no_penalties", "all_greedy", "all_random", "temperature"}
        for cfg in WARM_UP_CONFIGS:
            assert required.issubset(cfg.keys()), f"Config {cfg['name']} missing keys"

    def test_penalty_configs_have_penalty_params(self):
        for cfg in WARM_UP_CONFIGS:
            if not cfg["no_penalties"]:
                assert "frequency_penalties" in cfg
                assert "presence_penalties" in cfg
                assert "repetition_penalties" in cfg
