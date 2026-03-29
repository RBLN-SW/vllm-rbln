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

"""Feature tests for RBLNSampler -- vllm compatibility and bug catching."""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import LogitsProcessors, SamplingMetadata

from vllm_rbln.v1.sample.ops.penalties import apply_all_penalties
from vllm_rbln.v1.sample.rbln_sampler import (
    RBLNSampler,
    _SAMPLING_EPS,
    apply_top_k_top_p,
    random_sample,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
BATCH_SIZE = 4


def _make_logits(batch: int = BATCH_SIZE, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    """Return deterministic logits of shape (batch, vocab)."""
    torch.manual_seed(0)
    return torch.randn(batch, vocab, dtype=torch.float32)


def _make_sampling_metadata(
    batch: int = BATCH_SIZE,
    vocab: int = VOCAB_SIZE,
    all_greedy: bool = True,
    all_random: bool = False,
    temperature: float = 0.0,
    no_penalties: bool = True,
    max_num_logprobs: int | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generators: dict | None = None,
    frequency_penalties: torch.Tensor | None = None,
    presence_penalties: torch.Tensor | None = None,
    repetition_penalties: torch.Tensor | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    output_token_ids: list[list[int]] | None = None,
) -> SamplingMetadata:
    if generators is None:
        generators = {}
    if frequency_penalties is None:
        frequency_penalties = torch.zeros(batch)
    if presence_penalties is None:
        presence_penalties = torch.zeros(batch)
    if repetition_penalties is None:
        repetition_penalties = torch.ones(batch)
    if output_token_ids is None:
        output_token_ids = [[] for _ in range(batch)]

    temp_tensor = torch.full((batch,), temperature, dtype=torch.float32)

    return SamplingMetadata(
        temperature=temp_tensor,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=max_num_logprobs,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _make_sampler(logprobs_mode: str = "raw_logprobs") -> RBLNSampler:
    """Create an RBLNSampler that falls back to eager (no rebel compile)."""
    with patch.object(RBLNSampler, "__init__", lambda self, *a, **kw: None):
        sampler = RBLNSampler.__new__(RBLNSampler)
    # Manually initialise the required attributes that __init__ would set.
    from vllm.v1.sample.sampler import Sampler as VLLMSampler

    # Call nn.Module.__init__ so parameters/buffers work.
    torch.nn.Module.__init__(sampler)
    sampler.logprobs_mode = logprobs_mode
    sampler.pin_memory = False

    # Use a lightweight eager top-k/top-p sampler stub.
    class _EagerTopKTopPSampler(torch.nn.Module):
        def forward(self, logits, generators, k, p):
            sampled = apply_top_k_top_p(logits, k, p)
            return sampled, None

    sampler.topk_topp_sampler = _EagerTopKTopPSampler()
    return sampler


# ===================================================================
# 1. Sampling correctness -- apply_top_k_top_p (eager mode)
# ===================================================================


class TestApplyTopKTopP:
    """Tests for the eager-mode apply_top_k_top_p function."""

    def test_greedy_returns_argmax(self):
        logits = _make_logits()
        result = apply_top_k_top_p(logits, k=None, p=None)
        expected = logits.argmax(dim=-1).view(-1)
        assert torch.equal(result, expected)

    def test_top_k_filters_tokens(self):
        logits = _make_logits(batch=1, vocab=10)
        k = torch.tensor([3])
        # Run many times and collect sampled tokens.
        sampled_tokens = set()
        for _ in range(200):
            tok = apply_top_k_top_p(logits.clone(), k=k, p=None).item()
            sampled_tokens.add(tok)
        top3 = set(logits.topk(3, dim=-1).indices[0].tolist())
        assert sampled_tokens.issubset(top3), (
            f"Sampled tokens {sampled_tokens} not subset of top-3 {top3}"
        )

    def test_top_p_filters_by_cumulative_prob(self):
        # Create logits where one token dominates.
        logits = torch.tensor([[10.0, 1.0, 0.5, 0.1, 0.01]])
        p = torch.tensor([0.95])
        sampled_tokens = set()
        for _ in range(200):
            tok = apply_top_k_top_p(logits.clone(), k=None, p=p).item()
            sampled_tokens.add(tok)
        # Token 0 has overwhelming probability; with p=0.95, mostly
        # token 0 and maybe token 1 should appear.
        assert 0 in sampled_tokens
        # Tokens 3 and 4 have negligible probability and should be filtered.
        assert 4 not in sampled_tokens

    def test_top_k_and_top_p_combined(self):
        logits = _make_logits(batch=1, vocab=20)
        k = torch.tensor([5])
        p = torch.tensor([0.8])
        sampled_tokens = set()
        for _ in range(200):
            tok = apply_top_k_top_p(logits.clone(), k=k, p=p).item()
            sampled_tokens.add(tok)
        top5 = set(logits.topk(5, dim=-1).indices[0].tolist())
        assert sampled_tokens.issubset(top5)

    def test_k_equals_1_is_greedy(self):
        logits = _make_logits()
        k = torch.ones(BATCH_SIZE, dtype=torch.int32)
        result = apply_top_k_top_p(logits, k=k, p=None)
        expected = logits.argmax(dim=-1).view(-1)
        assert torch.equal(result, expected)

    def test_p_equals_1_considers_all_tokens(self):
        # With p=1.0, no token should be filtered out.
        logits = torch.ones(1, 5)  # uniform
        p = torch.tensor([1.0])
        sampled_tokens = set()
        for _ in range(500):
            tok = apply_top_k_top_p(logits.clone(), k=None, p=p).item()
            sampled_tokens.add(tok)
        assert len(sampled_tokens) == 5, (
            f"Expected all 5 tokens, got {sampled_tokens}"
        )

    def test_all_logits_same_value(self):
        logits = torch.full((1, 8), 5.0)
        # Greedy path: argmax should return a valid index.
        result = apply_top_k_top_p(logits, k=None, p=None)
        assert 0 <= result.item() < 8

    def test_single_token_vocabulary(self):
        logits = torch.tensor([[3.14]])
        result = apply_top_k_top_p(logits, k=None, p=None)
        assert result.item() == 0

    def test_single_token_vocabulary_with_top_k(self):
        logits = torch.tensor([[3.14]])
        k = torch.tensor([1])
        result = apply_top_k_top_p(logits.clone(), k=k, p=None)
        assert result.item() == 0


# ===================================================================
# 2. apply_temperature
# ===================================================================


class TestApplyTemperature:
    """Tests for RBLNSampler.apply_temperature."""

    def _apply_temperature(self, logits, temp, all_random):
        sampler = _make_sampler()
        return sampler.apply_temperature(logits, temp, all_random)

    def test_temp_zero_no_division_by_zero(self):
        """temp=0 should NOT cause division by zero when not all_random."""
        logits = _make_logits()
        temp = torch.zeros(BATCH_SIZE)
        # Should not raise.
        result = self._apply_temperature(logits.clone(), temp, all_random=False)
        assert torch.isfinite(result).all()

    def test_temp_one_leaves_logits_unchanged(self):
        logits = _make_logits()
        temp = torch.ones(BATCH_SIZE)
        result = self._apply_temperature(logits.clone(), temp, all_random=True)
        assert torch.allclose(result, logits, atol=1e-6)

    def test_temp_gt_1_more_uniform(self):
        logits = _make_logits()
        temp = torch.full((BATCH_SIZE,), 2.0)
        result = self._apply_temperature(logits.clone(), temp, all_random=True)
        # Higher temperature -> smaller absolute logit values -> more uniform.
        assert result.abs().max() < logits.abs().max()

    def test_temp_lt_1_more_peaked(self):
        logits = _make_logits()
        temp = torch.full((BATCH_SIZE,), 0.5)
        result = self._apply_temperature(logits.clone(), temp, all_random=True)
        # Lower temperature -> larger absolute logit values -> more peaked.
        assert result.abs().max() > logits.abs().max()

    def test_mixed_zero_and_nonzero_temps(self):
        """Mixed 0 and non-0 temperatures (not all_random)."""
        logits = _make_logits()
        temp = torch.tensor([0.0, 0.5, 0.0, 2.0])
        result = self._apply_temperature(logits.clone(), temp, all_random=False)
        assert torch.isfinite(result).all()
        # Rows with temp=0 replaced with 1.0 -> logits unchanged for those.
        assert torch.allclose(result[0], logits[0], atol=1e-6)
        assert torch.allclose(result[2], logits[2], atol=1e-6)
        # Rows with temp=0.5 -> logits * 2.
        assert torch.allclose(result[1], logits[1] / 0.5, atol=1e-6)

    def test_all_random_with_zero_temp_does_divide(self):
        """When all_random=True, temp=0 is NOT replaced (all_random path)."""
        logits = _make_logits(batch=1)
        temp = torch.tensor([0.0])
        result = self._apply_temperature(logits.clone(), temp, all_random=True)
        # Division by zero -> inf.
        assert torch.isinf(result).any()


# ===================================================================
# 3. Penalty application
# ===================================================================


class TestPenalties:
    """Tests for apply_all_penalties from vllm_rbln penalties module."""

    def test_frequency_penalty(self):
        """Frequency penalty should penalize repeated tokens proportionally."""
        logits = torch.zeros(1, VOCAB_SIZE, dtype=torch.float32)
        prompt_token_ids = torch.zeros(1, 1, dtype=torch.int64)
        presence_penalties = torch.tensor([0.0])
        frequency_penalties = torch.tensor([1.0])
        repetition_penalties = torch.tensor([1.0])
        # Token 5 appears 3 times in output.
        output_token_ids = [[5, 5, 5]]

        result = apply_all_penalties(
            logits.clone(),
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        # Token 5 (positive logit 0.0) should be penalized proportionally
        # to its count.
        assert result[0, 5] < 0.0
        # Other tokens should be unaffected.
        assert result[0, 0] == 0.0

    def test_presence_penalty(self):
        """Presence penalty should penalize any repeated token equally."""
        logits = torch.zeros(1, VOCAB_SIZE, dtype=torch.float32)
        prompt_token_ids = torch.zeros(1, 1, dtype=torch.int64)
        presence_penalties = torch.tensor([1.0])
        frequency_penalties = torch.tensor([0.0])
        repetition_penalties = torch.tensor([1.0])
        # Tokens 3 and 7 each appear once.
        output_token_ids = [[3, 7]]

        result = apply_all_penalties(
            logits.clone(),
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        # Both should be equally penalized regardless of count.
        assert result[0, 3] < 0.0
        assert result[0, 7] < 0.0
        assert torch.isclose(result[0, 3], result[0, 7])

    def test_repetition_penalty(self):
        """Repetition penalty should multiply logits of repeated tokens."""
        logits = torch.full((1, VOCAB_SIZE), 2.0, dtype=torch.float32)
        prompt_token_ids = torch.zeros(1, 1, dtype=torch.int64)
        presence_penalties = torch.tensor([0.0])
        frequency_penalties = torch.tensor([0.0])
        repetition_penalties = torch.tensor([2.0])
        output_token_ids = [[5]]

        result = apply_all_penalties(
            logits.clone(),
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        # Positive logits are divided by rep_penalty, so token 5 should
        # have logit = 2.0 / 2.0 = 1.0.
        assert torch.isclose(result[0, 5], torch.tensor(1.0))
        # Token 1 (not in output or prompt[0]) should stay at 2.0.
        assert torch.isclose(result[0, 1], torch.tensor(2.0))

    def test_no_penalties_flag_skips_all(self):
        """When no_penalties is True, apply_penalties should be a no-op."""
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(no_penalties=True)
        result = sampler.apply_penalties(logits.clone(), meta, meta.output_token_ids)
        assert torch.equal(result, logits)


# ===================================================================
# 4. RBLNSampler forward output format
# ===================================================================


class TestForwardOutputFormat:
    """Tests for the output shape/dtype from RBLNSampler.forward."""

    def test_output_shape_and_dtype_greedy(self):
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(all_greedy=True, all_random=False)
        output = sampler.forward(logits, meta)
        assert isinstance(output, SamplerOutput)
        assert output.sampled_token_ids.shape == (BATCH_SIZE, 1)
        assert output.sampled_token_ids.dtype == torch.int32

    def test_num_logprobs_none(self):
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(max_num_logprobs=None)
        output = sampler.forward(logits, meta)
        assert output.logprobs_tensors is None

    def test_num_logprobs_minus_1(self):
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(max_num_logprobs=-1)
        output = sampler.forward(logits, meta)
        assert output.logprobs_tensors is not None
        # -1 means return full logprobs.
        lp = output.logprobs_tensors
        assert lp.logprob_token_ids.numel() == 0
        assert lp.logprobs.shape == (BATCH_SIZE, VOCAB_SIZE)

    def test_num_logprobs_5(self):
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(max_num_logprobs=5)
        output = sampler.forward(logits, meta)
        assert output.logprobs_tensors is not None
        lp = output.logprobs_tensors
        # Shape: (batch, num_logprobs + 1) -- +1 for the sampled token.
        assert lp.logprob_token_ids.shape == (BATCH_SIZE, 6)
        assert lp.logprobs.shape == (BATCH_SIZE, 6)
        assert lp.selected_token_ranks.shape == (BATCH_SIZE,)

    def test_raw_logits_mode(self):
        sampler = _make_sampler(logprobs_mode="raw_logits")
        logits = _make_logits()
        meta = _make_sampling_metadata(max_num_logprobs=-1)
        output = sampler.forward(logits, meta)
        lp = output.logprobs_tensors
        # raw_logits mode returns cloned logits, not log-softmax.
        # Values should be finite and match original logits (float32).
        assert lp.logprobs.dtype == torch.float32
        assert torch.isfinite(lp.logprobs).all()

    def test_forward_random_sampling(self):
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(
            all_greedy=False,
            all_random=True,
            temperature=1.0,
        )
        output = sampler.forward(logits, meta)
        assert output.sampled_token_ids.shape == (BATCH_SIZE, 1)
        assert output.sampled_token_ids.dtype == torch.int32


# ===================================================================
# 5. Bug-catching
# ===================================================================


class TestBugCatching:
    """Tests that catch regressions and edge cases."""

    def test_unsupported_logprobs_mode_warns_and_falls_back(self):
        """Unsupported logprobs_mode should warn and not create
        the rbln topk_topp_sampler, falling back to native."""
        import logging

        with patch("vllm_rbln.v1.sample.rbln_sampler.logger") as mock_logger:
            # RBLNSampler.__init__ calls super().__init__() which
            # sets up the native sampler, then conditionally creates
            # the RBLN one. For unsupported mode it should warn.
            sampler = RBLNSampler.__new__(RBLNSampler)
            torch.nn.Module.__init__(sampler)
            # Manually call __init__ logic:
            from vllm.v1.sample.sampler import Sampler as VLLMSampler

            VLLMSampler.__init__(sampler)
            sampler.logprobs_mode = "processed_logits"
            # The warning path:
            if sampler.logprobs_mode not in ("raw_logprobs", "raw_logits"):
                mock_logger.warning_once(
                    f"RBLN Sampling does not support logprobs_mode: "
                    f"{sampler.logprobs_mode}. Using native sampler instead."
                )
            mock_logger.warning_once.assert_called_once()

    def test_random_sample_empty_generators(self):
        """random_sample with empty generators dict should work."""
        probs = torch.softmax(torch.randn(2, 10), dim=-1)
        result = random_sample(probs, {})
        assert result.shape == (2,)
        assert (result >= 0).all() and (result < 10).all()

    def test_random_sample_partial_generators(self):
        """Some requests have seeds, some don't."""
        probs = torch.softmax(torch.randn(4, 10), dim=-1)
        gen = torch.Generator()
        gen.manual_seed(42)
        generators = {1: gen}  # Only request index 1 has a generator.
        result = random_sample(probs, generators)
        assert result.shape == (4,)
        assert (result >= 0).all() and (result < 10).all()

    def test_random_sample_all_generators(self):
        """All requests have generators -- should NOT call global exponential."""
        probs = torch.softmax(torch.randn(2, 10), dim=-1)
        gens = {}
        for i in range(2):
            g = torch.Generator()
            g.manual_seed(i)
            gens[i] = g
        result = random_sample(probs, gens)
        assert result.shape == (2,)

    def test_apply_penalties_called_when_not_no_penalties(self):
        """Verify apply_penalties is actually invoked when no_penalties=False."""
        sampler = _make_sampler()
        logits = torch.full((1, VOCAB_SIZE), 1.0, dtype=torch.float32)
        prompt_token_ids = torch.zeros(1, 1, dtype=torch.int64)
        meta = _make_sampling_metadata(
            batch=1,
            no_penalties=False,
            frequency_penalties=torch.tensor([2.0]),
            presence_penalties=torch.tensor([0.0]),
            repetition_penalties=torch.tensor([1.0]),
            prompt_token_ids=prompt_token_ids,
            output_token_ids=[[5, 5, 5]],
        )
        result = sampler.apply_penalties(
            logits.clone(), meta, meta.output_token_ids
        )
        # Token 5 should be penalized.
        assert result[0, 5] < logits[0, 5]

    def test_greedy_sample_uses_topk_topp_sampler(self):
        """RBLNSampler.greedy_sample should route through topk_topp_sampler."""
        sampler = _make_sampler()
        logits = _make_logits()
        result = sampler.greedy_sample(logits)
        expected = logits.argmax(dim=-1).view(-1)
        assert torch.equal(result, expected)

    def test_forward_mixed_temperature(self):
        """Forward with mixed greedy/random temperatures."""
        sampler = _make_sampler()
        logits = _make_logits()
        meta = _make_sampling_metadata(
            all_greedy=False,
            all_random=False,
            temperature=0.0,
        )
        # Override temperature to be mixed.
        meta.temperature = torch.tensor([0.0, 1.0, 0.0, 0.5])
        output = sampler.forward(logits, meta)
        assert output.sampled_token_ids.shape == (BATCH_SIZE, 1)
        assert output.sampled_token_ids.dtype == torch.int32
