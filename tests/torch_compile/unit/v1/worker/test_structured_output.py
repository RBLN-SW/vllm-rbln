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

"""Unit tests for structured output (grammar bitmask) in RBLNModelRunner.

Tests the grammar bitmask application logic in sample_tokens() and the
apply_grammar_bitmask utility, following the TPU inference test pattern.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.core.sched.output import GrammarOutput

from vllm_rbln.v1.worker.rbln_model_runner import ExecuteModelState, RBLNModelRunner


def _make_runner_stub():
    """Create a minimal RBLNModelRunner stub for structured output tests."""
    runner = object.__new__(RBLNModelRunner)
    runner.device = torch.device("cpu")
    runner.execute_model_state = None
    runner.kv_connector_output = None
    runner.use_async_scheduling = False
    runner.input_batch = MagicMock()
    runner.input_batch.req_ids = []
    runner.speculative_config = None
    runner.max_model_len = 4096
    runner.model_config = MagicMock()
    runner.num_spec_tokens = 0
    runner.e2e_performance_tracker = None
    runner._draft_token_ids = None
    return runner


class TestGrammarBitmaskApplication:
    """Test the grammar bitmask application logic in sample_tokens().

    In sample_tokens (lines 3047-3056), when grammar_output is not None:
    1. Convert logits to float32 (required by xgrammar for CPU tensors)
    2. Call apply_grammar_bitmask()
    3. Convert logits back to original dtype
    """

    def test_bitmask_applied_when_grammar_output_present(self):
        """Grammar bitmask is applied when grammar_output is not None."""
        runner = _make_runner_stub()

        logits = torch.ones(3, 64, dtype=torch.float16)
        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = 3

        runner.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(3, 128),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )

        grammar_output = GrammarOutput(
            structured_output_request_ids=["req-0"],
            grammar_bitmask=np.zeros((1, 2), dtype=np.int32),
        )

        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.apply_grammar_bitmask"
            ) as mock_apply,
            patch.object(runner, "_sample", return_value=MagicMock()),
            patch.object(
                runner,
                "_bookkeeping_sync",
                return_value=(
                    0,
                    None,
                    [[1]],
                    {},
                    ["req-0"],
                    {"req-0": 0},
                    [],
                ),
            ),
            patch.object(runner, "is_prefills", return_value=[False]),
        ):
            runner.sample_tokens(grammar_output)

        mock_apply.assert_called_once()
        # Verify logits were converted to float32 before apply
        call_args = mock_apply.call_args
        applied_logits = call_args[0][3]  # 4th positional arg
        assert applied_logits.dtype == torch.float32

    def test_bitmask_skipped_when_grammar_output_none(self):
        """Grammar bitmask is NOT applied when grammar_output is None."""
        runner = _make_runner_stub()

        logits = torch.ones(3, 64, dtype=torch.float16)
        scheduler_output = MagicMock()

        runner.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(3, 128),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )

        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.apply_grammar_bitmask"
            ) as mock_apply,
            patch.object(runner, "_sample", return_value=MagicMock()),
            patch.object(
                runner,
                "_bookkeeping_sync",
                return_value=(
                    0,
                    None,
                    [[1]],
                    {},
                    ["req-0"],
                    {"req-0": 0},
                    [],
                ),
            ),
            patch.object(runner, "is_prefills", return_value=[False]),
        ):
            runner.sample_tokens(None)

        mock_apply.assert_not_called()

    def test_bitmask_skipped_when_logits_empty(self):
        """Grammar bitmask is NOT applied when logits has 0 rows."""
        runner = _make_runner_stub()

        logits = torch.ones(0, 64, dtype=torch.float16)
        scheduler_output = MagicMock()

        runner.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(0, 128),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )

        grammar_output = GrammarOutput(
            structured_output_request_ids=["req-0"],
            grammar_bitmask=np.zeros((1, 2), dtype=np.int32),
        )

        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.apply_grammar_bitmask"
            ) as mock_apply,
            patch.object(runner, "_sample", return_value=MagicMock()),
            patch.object(
                runner,
                "_bookkeeping_sync",
                return_value=(
                    0,
                    None,
                    [],
                    {},
                    [],
                    {},
                    [],
                ),
            ),
            patch.object(runner, "is_prefills", return_value=[False]),
        ):
            runner.sample_tokens(grammar_output)

        mock_apply.assert_not_called()


class TestApplyGrammarBitmask:
    """Test apply_grammar_bitmask utility function directly.

    This tests the bitmask reordering and application logic from
    vllm.v1.structured_output.utils.apply_grammar_bitmask.
    """

    def test_bitmask_reordering_matches_batch_order(self):
        """Bitmask is reordered to match the order of requests in the batch.

        Verifies that the sorted_bitmask aligns masks to batch indices,
        filling non-structured-output rows with -1 (all bits set = allow all).
        """
        vocab_size = 64
        num_int32_per_row = vocab_size // 32  # 2

        input_batch = MagicMock()
        input_batch.req_ids = ["req-0", "req-1", "req-2"]

        scheduler_output = MagicMock()
        scheduler_output.scheduled_spec_decode_tokens = {}

        mask_req0 = np.array([-1, 0], dtype=np.int32)
        mask_req2 = np.array([0, -1], dtype=np.int32)

        grammar_bitmask = np.array([mask_req0, mask_req2], dtype=np.int32)
        grammar_output = GrammarOutput(
            structured_output_request_ids=["req-0", "req-2"],
            grammar_bitmask=grammar_bitmask,
        )

        logits = torch.ones(3, vocab_size, dtype=torch.float32)

        # Patch both xgr and pin_memory (not available on CPU-only systems)
        with (
            patch("vllm.v1.structured_output.utils.xgr") as mock_xgr,
            patch(
                "torch.tensor",
                wraps=lambda *a, **kw: torch.as_tensor(
                    *a, **{k: v for k, v in kw.items() if k != "pin_memory"}
                ),
            ),
        ):
            from vllm.v1.structured_output.utils import apply_grammar_bitmask

            apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits)

        mock_xgr.apply_token_bitmask_inplace.assert_called_once()
        call_args = mock_xgr.apply_token_bitmask_inplace.call_args

        bitmask_tensor = call_args[0][1]
        assert bitmask_tensor.shape == (3, num_int32_per_row)
        np.testing.assert_array_equal(bitmask_tensor[0].numpy(), mask_req0)
        np.testing.assert_array_equal(bitmask_tensor[1].numpy(), [-1, -1])
        np.testing.assert_array_equal(bitmask_tensor[2].numpy(), mask_req2)

    def test_bitmask_with_spec_decode_offsets(self):
        """Bitmask handles speculative decoding token offsets correctly."""
        vocab_size = 64
        num_int32_per_row = 2

        input_batch = MagicMock()
        input_batch.req_ids = ["req-0", "req-1"]

        scheduler_output = MagicMock()
        scheduler_output.scheduled_spec_decode_tokens = {
            "req-0": [10, 20],
        }

        mask0_base = np.array([-1, 0], dtype=np.int32)
        mask0_spec1 = np.array([0, -1], dtype=np.int32)
        mask0_spec2 = np.array([-1, -1], dtype=np.int32)

        grammar_bitmask = np.array(
            [mask0_base, mask0_spec1, mask0_spec2], dtype=np.int32
        )
        grammar_output = GrammarOutput(
            structured_output_request_ids=["req-0"],
            grammar_bitmask=grammar_bitmask,
        )

        logits = torch.ones(4, vocab_size, dtype=torch.float32)

        with (
            patch("vllm.v1.structured_output.utils.xgr") as mock_xgr,
            patch(
                "torch.tensor",
                wraps=lambda *a, **kw: torch.as_tensor(
                    *a, **{k: v for k, v in kw.items() if k != "pin_memory"}
                ),
            ),
        ):
            from vllm.v1.structured_output.utils import apply_grammar_bitmask

            apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits)

        mock_xgr.apply_token_bitmask_inplace.assert_called_once()
        call_args = mock_xgr.apply_token_bitmask_inplace.call_args
        bitmask_tensor = call_args[0][1]

        assert bitmask_tensor.shape == (4, num_int32_per_row)
        np.testing.assert_array_equal(bitmask_tensor[0].numpy(), mask0_base)
        np.testing.assert_array_equal(bitmask_tensor[1].numpy(), mask0_spec1)
        np.testing.assert_array_equal(bitmask_tensor[2].numpy(), mask0_spec2)
        np.testing.assert_array_equal(bitmask_tensor[3].numpy(), [-1, -1])


class TestGrammarOutputCreation:
    """Test GrammarOutput dataclass creation and properties."""

    def test_grammar_output_basic(self):
        """GrammarOutput stores request IDs and bitmask."""
        bitmask = np.array([[0, -1], [-1, 0]], dtype=np.int32)
        output = GrammarOutput(
            structured_output_request_ids=["req-a", "req-b"],
            grammar_bitmask=bitmask,
        )

        assert output.structured_output_request_ids == ["req-a", "req-b"]
        assert output.grammar_bitmask.shape == (2, 2)
        np.testing.assert_array_equal(output.grammar_bitmask, bitmask)

    def test_grammar_output_empty(self):
        """GrammarOutput with no requests."""
        output = GrammarOutput(
            structured_output_request_ids=[],
            grammar_bitmask=np.zeros((0, 2), dtype=np.int32),
        )

        assert len(output.structured_output_request_ids) == 0
        assert output.grammar_bitmask.shape[0] == 0


class TestSampleTokensEarlyReturn:
    """Test sample_tokens early return paths."""

    def test_returns_none_when_no_state_and_no_kv_output(self):
        """When execute_model_state is None and no kv_connector_output."""
        runner = _make_runner_stub()
        runner.execute_model_state = None
        runner.kv_connector_output = None

        result = runner.sample_tokens(None)
        assert result is None

    def test_returns_none_when_no_state_with_async_scheduling(self):
        """When execute_model_state is None in async scheduling mode."""
        runner = _make_runner_stub()
        runner.execute_model_state = None
        runner.kv_connector_output = None
        runner.use_async_scheduling = True

        with patch("vllm_rbln.v1.worker.rbln_model_runner.get_pp_group") as mock_pp:
            mock_pp.return_value.world_size = 1
            result = runner.sample_tokens(None)

        assert result is None


class TestLogitsDtypeConversion:
    """Test that logits dtype conversion is correct for grammar bitmask.

    The RBLN-specific note (line 3049-3050) states:
    xgr.apply_token_bitmask_inplace requires logits to be float32 dtype
    for CPU tensors.
    """

    def test_float16_converted_to_float32_and_back(self):
        """float16 logits are converted to float32 for bitmask, then back."""
        runner = _make_runner_stub()

        original_logits = torch.ones(2, 32, dtype=torch.float16)
        scheduler_output = MagicMock()

        runner.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=original_logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(2, 64),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )

        grammar_output = GrammarOutput(
            structured_output_request_ids=["req-0"],
            grammar_bitmask=np.zeros((1, 1), dtype=np.int32),
        )

        applied_dtypes = []

        def capture_dtype(so, go, ib, logits):
            applied_dtypes.append(logits.dtype)

        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.apply_grammar_bitmask",
                side_effect=capture_dtype,
            ),
            patch.object(runner, "_sample", return_value=MagicMock()) as mock_sample,
            patch.object(
                runner,
                "_bookkeeping_sync",
                return_value=(
                    0,
                    None,
                    [[1]],
                    {},
                    ["req-0"],
                    {"req-0": 0},
                    [],
                ),
            ),
            patch.object(runner, "is_prefills", return_value=[False]),
        ):
            runner.sample_tokens(grammar_output)

        # apply_grammar_bitmask received float32 logits
        assert applied_dtypes[0] == torch.float32

        # _sample received logits converted back to original dtype
        sample_logits = mock_sample.call_args[0][0]
        assert sample_logits.dtype == torch.float16
