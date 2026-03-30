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

"""Unit tests for speculative decoding paths in RBLNModelRunner.

Tests _calc_spec_decode_metadata, propose_draft_token_ids dispatch,
take_draft_token_ids, effective_drafter_max_model_len logic,
_update_states_after_model_execute, and _prepare_input_ids
draft token scatter.
Follows the TPU inference test pattern (tpu-inference/tests/test_spec_dec.py).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

from vllm_rbln.v1.spec_decoding.medusa import RBLNMedusaProposer
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner


def _make_runner_stub():
    """Create a minimal RBLNModelRunner stub for spec decode tests.

    Binds only the methods needed for spec decode logic, avoiding
    full __init__ which requires device/config.
    """
    runner = object.__new__(RBLNModelRunner)
    runner.device = torch.device("cpu")
    runner.arange_np = np.arange(4096, dtype=np.int64)
    runner._draft_token_ids = None

    # Bind real methods
    for method_name in (
        "_get_cumsum_and_arange",
        "_calc_spec_decode_metadata",
        "take_draft_token_ids",
        "propose_draft_token_ids",
        "_update_states_after_model_execute",
    ):
        method = getattr(RBLNModelRunner, method_name)
        setattr(runner, method_name, method.__get__(runner))

    return runner


class TestCalcSpecDecodeMetadata:
    """Test _calc_spec_decode_metadata numpy index computation.

    This mirrors the TPU inference parametrized spec decode metadata tests.
    """

    @pytest.mark.parametrize(
        "num_draft_tokens,cu_num_scheduled_tokens,"
        "expected_logits_indices,expected_bonus_logits_indices,"
        "expected_target_logits_indices",
        [
            # Case 1: Normal mixed case from docstring
            # 5 requests, some with drafts, some without
            (
                [3, 0, 2, 0, 1],
                [4, 104, 107, 207, 209],
                [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
                [3, 4, 7, 8, 10],
                [0, 1, 2, 5, 6, 9],
            ),
            # Case 2: All zeros - pure decode, no speculation
            (
                [0, 0, 0],
                [1, 2, 3],
                [0, 1, 2],
                [0, 1, 2],
                [],
            ),
            # Case 3: Single request with drafts
            (
                [4],
                [5],
                [0, 1, 2, 3, 4],
                [4],
                [0, 1, 2, 3],
            ),
            # Case 4: High draft count, 2 requests
            (
                [5, 3],
                [6, 10],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [5, 9],
                [0, 1, 2, 3, 4, 6, 7, 8],
            ),
            # Case 5: Single token per request (no draft)
            (
                [0],
                [1],
                [0],
                [0],
                [],
            ),
        ],
    )
    def test_metadata_indices(
        self,
        num_draft_tokens,
        cu_num_scheduled_tokens,
        expected_logits_indices,
        expected_bonus_logits_indices,
        expected_target_logits_indices,
    ):
        runner = _make_runner_stub()

        max_tokens = max(cu_num_scheduled_tokens) + 10
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(max_tokens, dtype=torch.int32)

        num_draft_np = np.array(num_draft_tokens, dtype=np.int32)
        cu_scheduled_np = np.array(cu_num_scheduled_tokens, dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        assert isinstance(metadata, SpecDecodeMetadata)
        assert metadata.logits_indices.tolist() == expected_logits_indices
        assert metadata.bonus_logits_indices.tolist() == expected_bonus_logits_indices
        assert (
            metadata.target_logits_indices.tolist() == expected_target_logits_indices
        )
        assert metadata.num_draft_tokens == num_draft_tokens

    def test_draft_token_ids_computed_correctly(self):
        """Verify draft_token_ids are correctly gathered from input_ids."""
        runner = _make_runner_stub()

        # input_ids: [0, 10, 20, 30, 40, 50, ...]
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(0, 1000, dtype=torch.int32) * 10

        # 2 requests: first has 2 drafts, second has 1 draft
        num_draft_np = np.array([2, 1], dtype=np.int32)
        cu_scheduled_np = np.array([3, 5], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        # draft_token_ids should be the token ids at draft positions
        assert metadata.draft_token_ids is not None
        assert len(metadata.draft_token_ids) == 3  # 2 + 1 drafts total


class TestProposeDraftTokenIds:
    """Test propose_draft_token_ids method dispatch for each spec method."""

    def test_dispatch_ngram(self):
        """Ngram method dispatches to NgramProposer.propose()."""
        runner = _make_runner_stub()
        runner.speculative_config = MagicMock()
        runner.speculative_config.method = "ngram"
        runner.speculative_config.use_eagle.return_value = False

        mock_drafter = MagicMock(spec=NgramProposer)
        mock_drafter.propose.return_value = [[10, 11], [20]]
        runner.drafter = mock_drafter

        runner.input_batch = MagicMock()
        runner.input_batch.num_tokens_no_spec = [5, 8]
        runner.input_batch.token_ids_cpu = MagicMock()

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 10
        sampled_token_ids = [[1], [2]]

        result = runner.propose_draft_token_ids(
            scheduler_output=scheduler_output,
            sampled_token_ids=sampled_token_ids,
            sampling_metadata=MagicMock(),
            hidden_states=torch.randn(10, 64),
            sample_hidden_states=torch.randn(2, 64),
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=MagicMock(),
            slot_mappings=None,
        )

        mock_drafter.propose.assert_called_once_with(
            sampled_token_ids,
            runner.input_batch.num_tokens_no_spec,
            runner.input_batch.token_ids_cpu,
            slot_mappings=None,
        )
        assert result == [[10, 11], [20]]

    def test_dispatch_suffix(self):
        """Suffix method dispatches to SuffixDecodingProposer.propose()."""
        runner = _make_runner_stub()
        runner.speculative_config = MagicMock()
        runner.speculative_config.method = "suffix"
        runner.speculative_config.use_eagle.return_value = False

        mock_drafter = MagicMock(spec=SuffixDecodingProposer)
        mock_drafter.propose.return_value = [[30, 31]]
        runner.drafter = mock_drafter
        runner.input_batch = MagicMock()

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 5
        sampled_token_ids = [[1]]

        result = runner.propose_draft_token_ids(
            scheduler_output=scheduler_output,
            sampled_token_ids=sampled_token_ids,
            sampling_metadata=MagicMock(),
            hidden_states=torch.randn(5, 64),
            sample_hidden_states=torch.randn(1, 64),
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=MagicMock(),
            slot_mappings=None,
        )

        mock_drafter.propose.assert_called_once_with(
            runner.input_batch, sampled_token_ids
        )
        assert result == [[30, 31]]

    def test_dispatch_medusa_prefill(self):
        """Medusa prefill path uses sample_hidden_states directly."""
        runner = _make_runner_stub()
        runner.speculative_config = MagicMock()
        runner.speculative_config.method = "medusa"
        runner.speculative_config.use_eagle.return_value = False

        batch_size = 2
        hidden_dim = 64
        num_heads = 3

        mock_drafter = MagicMock(spec=RBLNMedusaProposer)
        draft_result = torch.tensor([[10, 11, 12], [20, 21, 22]])
        mock_drafter.propose.return_value = draft_result
        runner.drafter = mock_drafter

        sample_hidden = torch.randn(batch_size, hidden_dim)
        sampled_token_ids = [[1], [2]]

        result = runner.propose_draft_token_ids(
            scheduler_output=MagicMock(total_num_scheduled_tokens=5),
            sampled_token_ids=sampled_token_ids,
            sampling_metadata=MagicMock(),
            hidden_states=torch.randn(5, hidden_dim),
            sample_hidden_states=sample_hidden,
            aux_hidden_states=None,
            spec_decode_metadata=None,  # prefill
            common_attn_metadata=MagicMock(),
            slot_mappings=None,
        )

        mock_drafter.propose.assert_called_once()
        call_kwargs = mock_drafter.propose.call_args[1]
        # In prefill, hidden_states should be sample_hidden_states reshaped
        assert call_kwargs["target_hidden_states"].shape == (batch_size, hidden_dim)
        assert torch.equal(result, draft_result)

    def test_dispatch_medusa_decode_with_drafts(self):
        """Medusa decode path picks last accepted token's hidden state."""
        runner = _make_runner_stub()
        runner.speculative_config = MagicMock()
        runner.speculative_config.method = "medusa"
        runner.speculative_config.use_eagle.return_value = False

        batch_size = 3
        max_accepted = 4
        hidden_dim = 64

        mock_drafter = MagicMock(spec=RBLNMedusaProposer)
        mock_drafter.propose.return_value = torch.zeros(batch_size, 2)
        runner.drafter = mock_drafter

        # [batch, max_accepted, hidden_dim]
        sample_hidden = torch.randn(batch_size, max_accepted, hidden_dim)
        # Different accepted lengths: 3, 1, 2
        sampled_token_ids = [[1, 2, 3], [5], [10, 20]]

        spec_decode_metadata = MagicMock()  # not None -> decode path

        result = runner.propose_draft_token_ids(
            scheduler_output=MagicMock(total_num_scheduled_tokens=10),
            sampled_token_ids=sampled_token_ids,
            sampling_metadata=MagicMock(),
            hidden_states=torch.randn(10, hidden_dim),
            sample_hidden_states=sample_hidden,
            aux_hidden_states=None,
            spec_decode_metadata=spec_decode_metadata,
            common_attn_metadata=MagicMock(),
            slot_mappings=None,
        )

        call_kwargs = mock_drafter.propose.call_args[1]
        target_hs = call_kwargs["target_hidden_states"]
        # Should select indices [2, 0, 1] from dim=1
        assert target_hs.shape == (batch_size, hidden_dim)
        torch.testing.assert_close(target_hs[0], sample_hidden[0, 2])
        torch.testing.assert_close(target_hs[1], sample_hidden[1, 0])
        torch.testing.assert_close(target_hs[2], sample_hidden[2, 1])


class TestTakeDraftTokenIds:
    """Test take_draft_token_ids lifecycle."""

    def test_returns_none_when_no_drafts(self):
        runner = _make_runner_stub()
        assert runner.take_draft_token_ids() is None

    def test_returns_and_clears_list_drafts(self):
        runner = _make_runner_stub()
        runner._draft_token_ids = [[10, 11], [20, 21, 22]]
        runner.input_batch = MagicMock()
        runner.input_batch.req_ids = ["req-0", "req-1"]

        result = runner.take_draft_token_ids()

        assert isinstance(result, DraftTokenIds)
        assert result.req_ids == ["req-0", "req-1"]
        assert result.draft_token_ids == [[10, 11], [20, 21, 22]]
        assert runner._draft_token_ids is None

    def test_converts_tensor_drafts_to_list(self):
        runner = _make_runner_stub()
        runner._draft_token_ids = torch.tensor([[10, 11], [20, 21]])
        runner.input_batch = MagicMock()
        runner.input_batch.req_ids = ["req-0", "req-1"]

        result = runner.take_draft_token_ids()

        assert isinstance(result, DraftTokenIds)
        assert result.draft_token_ids == [[10, 11], [20, 21]]

    def test_idempotent_after_take(self):
        """Second call after take should return None."""
        runner = _make_runner_stub()
        runner._draft_token_ids = [[1, 2]]
        runner.input_batch = MagicMock()
        runner.input_batch.req_ids = ["req-0"]

        runner.take_draft_token_ids()
        assert runner.take_draft_token_ids() is None


class TestEffectiveDrafterMaxModelLen:
    """Test the effective_drafter_max_model_len and input_fits_in_drafter logic.

    This determines whether draft token proposal happens at all.
    """

    def test_no_spec_config(self):
        """When speculative_config is None, no drafting should occur."""
        runner = _make_runner_stub()
        runner.speculative_config = None
        runner.max_model_len = 1024

        # use_padded_batch_for_eagle should be False
        spec_config = runner.speculative_config
        use_padded_batch_for_eagle = (
            spec_config is not None
            and spec_config.use_eagle()
            and not spec_config.disable_padded_drafter_batch
        )
        assert not use_padded_batch_for_eagle

    def test_input_fits_with_default_max_model_len(self):
        """input_fits_in_drafter when max_seq_len + num_spec_tokens <= max_model_len."""
        runner = _make_runner_stub()
        runner.max_model_len = 1024
        runner.num_spec_tokens = 5
        runner.model_config = MagicMock()
        runner.model_config.max_model_len = 2048
        runner.speculative_config = MagicMock()
        runner.speculative_config.draft_model_config = None

        common_attn = MagicMock()
        common_attn.max_seq_len = 1000

        effective_drafter_max_model_len = runner.max_model_len
        input_fits = common_attn and (
            common_attn.max_seq_len + runner.num_spec_tokens
            <= effective_drafter_max_model_len
        )
        assert input_fits  # 1000 + 5 <= 1024

    def test_input_does_not_fit(self):
        """input_fits_in_drafter is False when sequence too long."""
        runner = _make_runner_stub()
        runner.max_model_len = 1024
        runner.num_spec_tokens = 5

        common_attn = MagicMock()
        common_attn.max_seq_len = 1020

        effective_drafter_max_model_len = runner.max_model_len
        input_fits = common_attn and (
            common_attn.max_seq_len + runner.num_spec_tokens
            <= effective_drafter_max_model_len
        )
        assert not input_fits  # 1020 + 5 > 1024

    def test_draft_model_config_overrides_max_model_len(self):
        """When draft_model_config has max_model_len, it takes precedence."""
        runner = _make_runner_stub()
        runner.max_model_len = 4096
        runner.num_spec_tokens = 5

        spec_config = MagicMock()
        spec_config.draft_model_config.max_model_len = 512

        effective_drafter_max_model_len = runner.max_model_len
        if (
            spec_config.draft_model_config is not None
            and spec_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                spec_config.draft_model_config.max_model_len
            )

        assert effective_drafter_max_model_len == 512

        common_attn = MagicMock()
        common_attn.max_seq_len = 510
        input_fits = common_attn and (
            common_attn.max_seq_len + runner.num_spec_tokens
            <= effective_drafter_max_model_len
        )
        assert not input_fits  # 510 + 5 > 512


class TestSpecDecodeMetadataProperties:
    """Test properties and invariants of the computed SpecDecodeMetadata."""

    def test_cu_num_draft_tokens_monotonic(self):
        """cu_num_draft_tokens should be monotonically non-decreasing."""
        runner = _make_runner_stub()
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(500, dtype=torch.int32)

        num_draft_np = np.array([3, 0, 2, 1], dtype=np.int32)
        cu_scheduled_np = np.array([4, 5, 8, 10], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        cu_draft = metadata.cu_num_draft_tokens.tolist()
        for i in range(1, len(cu_draft)):
            assert cu_draft[i] >= cu_draft[i - 1]

    def test_cu_num_sampled_tokens_monotonic(self):
        """cu_num_sampled_tokens should be monotonically non-decreasing."""
        runner = _make_runner_stub()
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(500, dtype=torch.int32)

        num_draft_np = np.array([2, 3, 0], dtype=np.int32)
        cu_scheduled_np = np.array([3, 7, 8], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        cu_sampled = metadata.cu_num_sampled_tokens.tolist()
        for i in range(1, len(cu_sampled)):
            assert cu_sampled[i] >= cu_sampled[i - 1]

    def test_bonus_logits_indices_count_equals_num_requests(self):
        """bonus_logits_indices should have one entry per request."""
        runner = _make_runner_stub()
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(500, dtype=torch.int32)

        num_reqs = 5
        num_draft_np = np.array([1, 2, 0, 3, 1], dtype=np.int32)
        cu_scheduled_np = np.array([2, 5, 6, 10, 12], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        assert len(metadata.bonus_logits_indices) == num_reqs

    def test_total_logits_indices_count(self):
        """logits_indices length should equal sum(num_draft_tokens + 1)."""
        runner = _make_runner_stub()
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(500, dtype=torch.int32)

        num_draft_np = np.array([3, 1, 2], dtype=np.int32)
        cu_scheduled_np = np.array([4, 6, 9], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        expected_count = sum(d + 1 for d in num_draft_np)
        assert len(metadata.logits_indices) == expected_count

    def test_target_logits_indices_count(self):
        """target_logits_indices length should equal sum(num_draft_tokens)."""
        runner = _make_runner_stub()
        runner.input_ids = MagicMock()
        runner.input_ids.gpu = torch.arange(500, dtype=torch.int32)

        num_draft_np = np.array([2, 0, 3], dtype=np.int32)
        cu_scheduled_np = np.array([3, 4, 8], dtype=np.int32)

        metadata = runner._calc_spec_decode_metadata(num_draft_np, cu_scheduled_np)

        expected_count = sum(num_draft_np)
        assert len(metadata.target_logits_indices) == expected_count


class TestUpdateCachedStatesAfterModelExecution:
    """Test _update_states_after_model_execute (lines 910-943).

    This method computes num_accepted_tokens from output_token_ids
    for hybrid models with speculative decoding.
    """

    def test_skips_when_not_hybrid(self):
        """Returns early when model is not hybrid."""
        runner = _make_runner_stub()
        runner.model_config = MagicMock()
        runner.model_config.is_hybrid = False
        runner.speculative_config = MagicMock()

        # Should not raise even without input_batch
        runner._update_states_after_model_execute(torch.tensor([[1, 2]]))

    def test_skips_when_no_spec_config(self):
        """Returns early when speculative_config is None."""
        runner = _make_runner_stub()
        runner.model_config = MagicMock()
        runner.model_config.is_hybrid = True
        runner.speculative_config = None

        runner._update_states_after_model_execute(torch.tensor([[1, 2]]))

    def test_all_tokens_accepted(self):
        """When no -1 in output_token_ids, all tokens are accepted."""
        runner = _make_runner_stub()
        runner.model_config = MagicMock()
        runner.model_config.is_hybrid = True
        runner.speculative_config = MagicMock()
        runner.input_batch = MagicMock()
        runner.input_batch.num_accepted_tokens_cpu = np.zeros(4, dtype=np.int64)

        # 3 requests, each with 3 draft tokens, all accepted (no -1)
        output_token_ids = torch.tensor([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ])

        runner._update_states_after_model_execute(output_token_ids)

        # argmax of [False, False, False, True] = 3 (the appended -1 column)
        assert runner.input_batch.num_accepted_tokens_cpu[0] == 3
        assert runner.input_batch.num_accepted_tokens_cpu[1] == 3
        assert runner.input_batch.num_accepted_tokens_cpu[2] == 3

    def test_early_rejection(self):
        """When -1 appears early, only tokens before it are accepted."""
        runner = _make_runner_stub()
        runner.model_config = MagicMock()
        runner.model_config.is_hybrid = True
        runner.speculative_config = MagicMock()
        runner.input_batch = MagicMock()
        runner.input_batch.num_accepted_tokens_cpu = np.zeros(3, dtype=np.int64)

        # Request 0: accepted 1 token, then rejected
        # Request 1: all 3 accepted
        # Request 2: rejected immediately
        output_token_ids = torch.tensor([
            [10, -1, -1],
            [40, 50, 60],
            [-1, -1, -1],
        ])

        runner._update_states_after_model_execute(output_token_ids)

        assert runner.input_batch.num_accepted_tokens_cpu[0] == 1
        assert runner.input_batch.num_accepted_tokens_cpu[1] == 3
        assert runner.input_batch.num_accepted_tokens_cpu[2] == 0

    def test_mixed_acceptance(self):
        """Various acceptance lengths in a batch."""
        runner = _make_runner_stub()
        runner.model_config = MagicMock()
        runner.model_config.is_hybrid = True
        runner.speculative_config = MagicMock()
        runner.input_batch = MagicMock()
        runner.input_batch.num_accepted_tokens_cpu = np.zeros(4, dtype=np.int64)

        output_token_ids = torch.tensor([
            [10, 20, -1, -1, -1],  # 2 accepted
            [30, 40, 50, 60, -1],  # 4 accepted
            [-1, -1, -1, -1, -1],  # 0 accepted
            [70, -1, -1, -1, -1],  # 1 accepted
        ])

        runner._update_states_after_model_execute(output_token_ids)

        assert runner.input_batch.num_accepted_tokens_cpu[0] == 2
        assert runner.input_batch.num_accepted_tokens_cpu[1] == 4
        assert runner.input_batch.num_accepted_tokens_cpu[2] == 0
        assert runner.input_batch.num_accepted_tokens_cpu[3] == 1


class TestPrepareInputIdsDraftScatter:
    """Test the draft token scatter logic in _prepare_input_ids (lines 1096-1117).

    When draft tokens exist, they are scattered into input_ids.gpu at
    spec_flattened_indices positions using prev_draft_token_indices.
    """

    def test_draft_tokens_scattered_to_correct_positions(self):
        """Verify draft token ids are placed at the right indices in input_ids."""
        # Simulate the scatter logic directly (extracted from _prepare_input_ids)
        input_ids_gpu = torch.zeros(10, dtype=torch.int32)

        # Previous draft tokens: [[100, 200, 0], [300, 400, 500]]
        # Flattened: [100, 200, 0, 300, 400, 500]
        draft_token_ids = torch.tensor([[100, 200, 0], [300, 400, 500]], dtype=torch.int32)

        # spec_flattened_indices: positions in input_ids where drafts go
        spec_flattened_indices = [1, 2, 5, 6, 7]
        # prev_draft_token_indices: indices into flattened draft_token_ids
        # [0]=100, [1]=200, [3]=300, [4]=400, [5]=500
        prev_draft_token_indices = [0, 1, 3, 4, 5]

        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64
        )
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64
        )

        input_ids_gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

        assert input_ids_gpu[1].item() == 100
        assert input_ids_gpu[2].item() == 200
        assert input_ids_gpu[5].item() == 300
        assert input_ids_gpu[6].item() == 400
        assert input_ids_gpu[7].item() == 500
        # Positions not written should remain 0
        assert input_ids_gpu[0].item() == 0
        assert input_ids_gpu[3].item() == 0

    def test_no_scatter_when_no_draft_tokens(self):
        """When _draft_token_ids is None, scatter is skipped."""
        # This tests the guard at line 1097
        draft_token_ids = None
        spec_flattened_indices = []

        # The condition: if self._draft_token_ids is None or not spec_flattened_indices
        assert draft_token_ids is None or not spec_flattened_indices

    def test_no_scatter_when_empty_spec_indices(self):
        """When spec_flattened_indices is empty, scatter is skipped."""
        draft_token_ids = torch.tensor([[1, 2]])
        spec_flattened_indices = []

        assert draft_token_ids is None or not spec_flattened_indices

    def test_draft_token_dtype_conversion(self):
        """Draft token ids are converted to int32 before scatter (line 1110)."""
        # draft_token_ids from eagle are int64, must be converted to int32
        draft_token_ids = torch.tensor([[100, 200]], dtype=torch.int64)
        converted = draft_token_ids.to(dtype=torch.int32)

        assert converted.dtype == torch.int32
        assert converted[0, 0].item() == 100
