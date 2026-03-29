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

"""Unit tests for RBLNModelRunner and module-level functions."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput

from vllm_rbln.v1.worker.metrics import PerformanceTracker
from vllm_rbln.v1.worker.rbln_model_runner import (
    AsyncRBLNModelRunnerOutput,
    DummyRunState,
    ExecuteModelState,
    create_lora_mask,
    create_sampler_indices_padded,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_runner_stub(**overrides):
    """Create a lightweight stub mimicking RBLNModelRunner attributes."""
    defaults = dict(
        model=MagicMock(),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                tensor_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sequence_parallelism=False)
            ),
        ),
        model_config=SimpleNamespace(
            runner_type="generate",
            logprobs_mode="raw_logprobs",
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
        ),
        lora_config=None,
        speculative_config=None,
        input_batch=MagicMock(),
        arange_np=np.arange(10000, dtype=np.int64),
        intermediate_tensors=None,
        bucketing_manager=MagicMock(),
        max_num_batched_tokens=256,
        specialized_moe_decode=False,
        sampler=MagicMock(),
        rejection_sampler=MagicMock(),
        performance_tracker=None,
        sampler_performance_tracker=None,
        e2e_performance_tracker=None,
        uses_mrope=False,
        positions=MagicMock(),
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    stub = SimpleNamespace(**defaults)
    return stub


# ===========================================================================
# Tests: ExecuteModelState / DummyRunState
# ===========================================================================


class TestNamedTuples:
    def test_execute_model_state(self):
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            kv_connector_output=None,
            slot_mappings=None,
        )
        assert state.spec_decode_metadata is None
        assert isinstance(state.logits, torch.Tensor)

    def test_dummy_run_state(self):
        state = DummyRunState(
            attn_metadata={},
            num_input_tokens=10,
            input_ids={},
            positions={},
        )
        assert state.num_input_tokens == 10


# ===========================================================================
# Tests: AsyncRBLNModelRunnerOutput
# ===========================================================================


class TestAsyncRBLNModelRunnerOutput:
    def test_init(self):
        mro = MagicMock(spec=ModelRunnerOutput)
        sampled = torch.tensor([[1], [2], [3]])
        stream = MagicMock()
        output = AsyncRBLNModelRunnerOutput(
            model_runner_output=mro,
            sampled_token_ids=sampled,
            invalid_req_indices=[1],
            async_output_copy_stream=stream,
        )
        assert output._model_runner_output is mro
        assert output._invalid_req_indices == [1]
        assert torch.equal(output._sampled_token_ids, sampled)


# ===========================================================================
# Tests: _get_cumsum_and_arange (bound to runner)
# ===========================================================================


class TestGetCumsumAndArange:
    def _call(self, num_tokens):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        return RBLNModelRunner._get_cumsum_and_arange(stub, num_tokens)

    def test_basic(self):
        arr = np.array([2, 5, 3])
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [2, 7, 10])
        np.testing.assert_array_equal(arange, [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])

    def test_single_element(self):
        arr = np.array([4])
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [4])
        np.testing.assert_array_equal(arange, [0, 1, 2, 3])

    def test_ones(self):
        arr = np.array([1, 1, 1])
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [1, 2, 3])
        np.testing.assert_array_equal(arange, [0, 0, 0])

    def test_with_cumsum_dtype(self):
        arr = np.array([2, 3])
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [2, 5])
        assert len(arange) == 5


# ===========================================================================
# Tests: _enable_performance_tracker
# ===========================================================================


class TestEnablePerformanceTracker:
    def test_metrics_enabled(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_METRICS", True):
            RBLNModelRunner._enable_performance_tracker(stub)
        assert isinstance(stub.performance_tracker, PerformanceTracker)
        assert isinstance(stub.sampler_performance_tracker, PerformanceTracker)
        assert isinstance(stub.e2e_performance_tracker, PerformanceTracker)

    def test_metrics_disabled(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_METRICS", False):
            RBLNModelRunner._enable_performance_tracker(stub)
        assert stub.performance_tracker is None


# ===========================================================================
# Tests: get_supported_tasks
# ===========================================================================


class TestGetSupportedTasks:
    def _bind(self, stub, method_name):
        """Bind an RBLNModelRunner unbound method to a stub."""
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        method = getattr(RBLNModelRunner, method_name)
        import types
        return types.MethodType(method, stub)

    def _make_stub_with_methods(self, **kw):
        stub = _make_runner_stub(**kw)
        stub.get_model = self._bind(stub, "get_model")
        stub.get_supported_generation_tasks = self._bind(
            stub, "get_supported_generation_tasks"
        )
        stub.get_supported_pooling_tasks = self._bind(
            stub, "get_supported_pooling_tasks"
        )
        return stub

    def test_generation_text(self):
        stub = self._make_stub_with_methods()
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
            return_value=True,
        ), patch(
            "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
            return_value=False,
        ):
            result = stub.get_supported_generation_tasks()
        assert "generate" in result

    def test_generation_transcription_only(self):
        model = MagicMock()
        model.supports_transcription_only = True
        stub = self._make_stub_with_methods(model=model)
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
            return_value=False,
        ), patch(
            "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
            return_value=True,
        ):
            result = stub.get_supported_generation_tasks()
        assert result == ["transcription"]

    def test_pooling_basic(self):
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed", "classify"]
        stub = self._make_stub_with_methods(model=model)
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            result = stub.get_supported_pooling_tasks()
        assert "embed" in result

    def test_pooling_not_pooling_model(self):
        stub = self._make_stub_with_methods()
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=False,
        ):
            result = stub.get_supported_pooling_tasks()
        assert result == []

    def test_pooling_score_removed_for_multi_label(self):
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed", "score"]
        stub = self._make_stub_with_methods(model=model)
        stub.model_config.hf_config = SimpleNamespace(num_labels=3)
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            result = stub.get_supported_pooling_tasks()
        assert "score" not in result

    def test_pooling_chunked_prefill_removes_encode(self):
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed", "encode"]
        stub = self._make_stub_with_methods(model=model)
        stub.scheduler_config.enable_chunked_prefill = True
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            result = stub.get_supported_pooling_tasks()
        assert "encode" not in result


# ===========================================================================
# Tests: compute_logits
# ===========================================================================


class TestComputeLogits:
    def test_delegates(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        hidden = torch.randn(2, 10)
        stub.model.compute_logits.return_value = torch.randn(2, 100)
        result = RBLNModelRunner.compute_logits(stub, hidden)
        stub.model.compute_logits.assert_called_once_with(hidden)


# ===========================================================================
# Tests: collect_metrics
# ===========================================================================


class TestCollectMetrics:
    def test_prefill(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        tracker = PerformanceTracker("test")
        stub = _make_runner_stub()
        RBLNModelRunner.collect_metrics(
            stub,
            tracker,
            is_prefill=True,
            start_time=0.0,
            end_time=0.1,
            reports=[{"total_host": 100, "total_device": 200, "total_ccl": 50}],
            token_count=10,
        )
        assert tracker.prefill_metrics.get_call_counts() == 1

    def test_decode(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        tracker = PerformanceTracker("test")
        stub = _make_runner_stub()
        RBLNModelRunner.collect_metrics(
            stub,
            tracker,
            is_prefill=False,
            start_time=0.0,
            end_time=0.05,
            reports=[],
            token_count=5,
        )
        assert tracker.decode_metrics.get_call_counts() == 1

    def test_no_reports(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        tracker = PerformanceTracker("test")
        stub = _make_runner_stub()
        RBLNModelRunner.collect_metrics(
            stub,
            tracker,
            is_prefill=True,
            start_time=0.0,
            end_time=0.1,
            reports=None,
            token_count=0,
        )
        assert tracker.prefill_metrics.get_call_counts() == 1


# ===========================================================================
# Tests: use_wrapped_compute_logits
# ===========================================================================


class TestUseWrappedComputeLogits:
    def test_default(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub(lora_config=None, speculative_config=None)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_with_lora(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub(lora_config=MagicMock())
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False

    def test_with_eagle_spec(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        spec_cfg = SimpleNamespace(method="eagle")
        stub = _make_runner_stub(speculative_config=spec_cfg)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False

    def test_with_ngram_spec(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        spec_cfg = SimpleNamespace(method="ngram")
        stub = _make_runner_stub(speculative_config=spec_cfg)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True


# ===========================================================================
# Tests: get_dp_padding
# ===========================================================================


class TestGetDpPadding:
    def test_single_dp(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        stub.vllm_config.parallel_config.data_parallel_size = 1
        result = RBLNModelRunner.get_dp_padding(
            stub, num_tokens=10, batch_bucket_size=32
        )
        assert result == (32, None, None)

    def test_single_dp_with_padded_raises(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        stub.vllm_config.parallel_config.data_parallel_size = 1
        with pytest.raises(AssertionError, match="num_padded_tokens should not"):
            RBLNModelRunner.get_dp_padding(
                stub, num_tokens=10, batch_bucket_size=32, num_padded_tokens=64
            )

    def test_dp_with_padded_tokens(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub(
            specialized_moe_decode=True,
            max_num_batched_tokens=256,
        )
        stub.vllm_config.parallel_config.data_parallel_size = 2
        stub.vllm_config.parallel_config.data_parallel_rank = 0

        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.RBLNDPMetadata"
        ) as mock_dp:
            mock_dp.num_tokens_across_dp.return_value = torch.tensor([10, 12])
            result = RBLNModelRunner.get_dp_padding(
                stub, num_tokens=10, batch_bucket_size=32, num_padded_tokens=256
            )
        assert result[1] == 256


# ===========================================================================
# Tests: sync_and_slice_intermediate_tensors
# ===========================================================================


class TestSyncAndSlice:
    def _make_stub(self):
        stub = _make_runner_stub()
        stub.compilation_config = SimpleNamespace(
            pass_config=SimpleNamespace(enable_sequence_parallelism=False)
        )
        return stub

    def test_from_dummy(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = self._make_stub()
        stub.intermediate_tensors = {
            "hidden": torch.zeros(20, 10),
        }
        result = RBLNModelRunner.sync_and_slice_intermediate_tensors(
            stub, batch_size=2, seq_len=10, intermediate_tensors=None, sync_self=False
        )
        assert result.tensors["hidden"].shape == (2, 10, 10)

    def test_from_input(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = self._make_stub()
        it = IntermediateTensors({"hidden": torch.randn(4, 10)})
        result = RBLNModelRunner.sync_and_slice_intermediate_tensors(
            stub, batch_size=-1, seq_len=-1, intermediate_tensors=it, sync_self=True
        )
        assert torch.equal(result.tensors["hidden"], it.tensors["hidden"])


# ===========================================================================
# Tests: _sample
# ===========================================================================


class TestSample:
    def test_empty_logits(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        logits = torch.empty(0, 100)
        result = RBLNModelRunner._sample(stub, logits, None)
        assert result.sampled_token_ids.shape == (0, 1)

    def test_normal_sampling(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        logits = torch.randn(2, 100)
        expected = SamplerOutput(
            sampled_token_ids=torch.tensor([[1], [2]]),
            logprobs_tensors=None,
        )
        stub.sampler.return_value = expected
        stub.input_batch.sampling_metadata = MagicMock()

        # Make hasattr(rebel, "capture_reports") return False
        import rebel
        capture_reports_backup = getattr(rebel, "capture_reports", None)
        if hasattr(rebel, "capture_reports"):
            delattr(rebel, "capture_reports")
        try:
            with patch(
                "vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_METRICS", False
            ):
                result = RBLNModelRunner._sample(stub, logits, None)
        finally:
            if capture_reports_backup is not None:
                rebel.capture_reports = capture_reports_backup
        assert result is expected

    def test_spec_decode_sampling(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        stub._update_states_after_model_execute = MagicMock()
        logits = torch.randn(4, 100)
        spec_meta = MagicMock()
        expected = SamplerOutput(
            sampled_token_ids=torch.tensor([[1], [2]]),
            logprobs_tensors=None,
        )
        stub.rejection_sampler.return_value = expected
        stub.input_batch.sampling_metadata = MagicMock()

        import rebel
        capture_reports_backup = getattr(rebel, "capture_reports", None)
        if hasattr(rebel, "capture_reports"):
            delattr(rebel, "capture_reports")
        try:
            with patch(
                "vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_METRICS", False
            ):
                result = RBLNModelRunner._sample(stub, logits, spec_meta)
        finally:
            if capture_reports_backup is not None:
                rebel.capture_reports = capture_reports_backup
        stub.rejection_sampler.assert_called_once()


# ===========================================================================
# Tests: _may_reorder_batch
# ===========================================================================


class TestMayReorderBatch:
    def test_disabled(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        sched_output = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_SORT_BATCH", False
        ):
            RBLNModelRunner._may_reorder_batch(stub, sched_output)
        # Should return early without accessing kv_cache_config
        stub.input_batch.swap_states.assert_not_called()

    def test_no_kv_groups(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        stub.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        sched_output = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RBLN_SORT_BATCH", True
        ):
            RBLNModelRunner._may_reorder_batch(stub, sched_output)


# ===========================================================================
# Tests: _get_positions
# ===========================================================================


class TestGetPositions:
    def test_int_no_mrope(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub(uses_mrope=False)
        stub.positions = SimpleNamespace(gpu=torch.arange(100))
        result = RBLNModelRunner._get_positions(stub, 10)
        assert result.shape == (10,)

    def test_tensor_no_mrope(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub(uses_mrope=False)
        stub.positions = SimpleNamespace(gpu=torch.arange(100))
        indices = torch.tensor([0, 5, 10])
        result = RBLNModelRunner._get_positions(stub, indices)
        assert result.shape == (3,)


# ===========================================================================
# Tests: _init_device_properties / _sync_device
# ===========================================================================


class TestDeviceMethods:
    def test_init_device_properties_noop(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        RBLNModelRunner._init_device_properties(stub)

    def test_sync_device_noop(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        RBLNModelRunner._sync_device(stub)


# ===========================================================================
# Tests: get_model
# ===========================================================================


class TestGetModel:
    def test_returns_model(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        model = MagicMock()
        stub = _make_runner_stub(model=model)
        assert RBLNModelRunner.get_model(stub) is model


# ===========================================================================
# Tests: maybe_randomize_inputs
# ===========================================================================


class TestMaybeRandomizeInputs:
    def test_no_randomize(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        stub.vllm_config.parallel_config.data_parallel_size = 1
        input_ids = torch.zeros(10, dtype=torch.int32)
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS",
            False,
        ):
            with RBLNModelRunner.maybe_randomize_inputs(stub, input_ids):
                pass
        # Input should remain zeros
        assert (input_ids == 0).all()


# ===========================================================================
# Tests: create_lora_mask
# ===========================================================================


class TestCreateLoraMask:
    def test_no_active_lora(self):
        input_ids = torch.zeros(2, 4, dtype=torch.int64)
        mask = create_lora_mask(
            input_ids,
            lora_ids=[0, 0],
            lora_index_to_id=[0, 1],
            max_loras=2,
            max_lora_rank=4,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert mask.shape == (8, 8)  # (2*4, 2*4)
        assert (mask == 0).all()

    def test_single_active_lora(self):
        input_ids = torch.zeros(2, 3, dtype=torch.int64)
        mask = create_lora_mask(
            input_ids,
            lora_ids=[1, 0],
            lora_index_to_id=[0, 1],
            max_loras=2,
            max_lora_rank=4,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert mask.shape == (6, 8)
        # First request should have ones in lora_index=1 block
        assert mask[0:3, 4:8].sum() == 3 * 4

    def test_multiple_active_loras(self):
        input_ids = torch.zeros(3, 2, dtype=torch.int64)
        mask = create_lora_mask(
            input_ids,
            lora_ids=[1, 2, 0],
            lora_index_to_id=[0, 1, 2],
            max_loras=3,
            max_lora_rank=2,
            lora_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert mask.shape == (6, 6)
        # Request 0 (lora_id=1 -> index 1): rows 0-1, cols 2-3
        assert mask[0:2, 2:4].sum() == 2 * 2
        # Request 1 (lora_id=2 -> index 2): rows 2-3, cols 4-5
        assert mask[2:4, 4:6].sum() == 2 * 2
        # Request 2 (lora_id=0): no mask
        assert mask[4:6, :].sum() == 0


# ===========================================================================
# Tests: create_sampler_indices_padded
# ===========================================================================


class TestCreateSamplerIndicesPadded:
    def test_prefill_single_lora(self):
        result = create_sampler_indices_padded(
            lora_ids=[1],
            lora_index_to_id=[0, 1],
            max_num_seqs=4,
            is_prefill=True,
            max_loras=2,
            device=torch.device("cpu"),
        )
        assert result.shape == (1,)

    def test_decode_multiple_loras(self):
        result = create_sampler_indices_padded(
            lora_ids=[1, 2, 0],
            lora_index_to_id=[0, 1, 2],
            max_num_seqs=4,
            is_prefill=False,
            max_loras=3,
            device=torch.device("cpu"),
        )
        assert result.shape == (4,)

    def test_prefill_multiple_loras_raises(self):
        with pytest.raises(AssertionError, match="Only single LoRA"):
            create_sampler_indices_padded(
                lora_ids=[1, 2],
                lora_index_to_id=[0, 1, 2],
                max_num_seqs=4,
                is_prefill=True,
                max_loras=3,
                device=torch.device("cpu"),
            )

    def test_no_active_lora(self):
        result = create_sampler_indices_padded(
            lora_ids=[0, 0],
            lora_index_to_id=[0, 1],
            max_num_seqs=4,
            is_prefill=False,
            max_loras=2,
            device=torch.device("cpu"),
        )
        assert result.shape == (4,)


# ===========================================================================
# Tests: select_common_block_size
# ===========================================================================


class TestSelectCommonBlockSize:
    def test_direct_match(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [16, 32, 64]
        group = SimpleNamespace(backend=backend)
        result = RBLNModelRunner.select_common_block_size(32, [group])
        assert result == 32

    def test_fallback_to_int_size(self):
        from vllm.v1.attention.backend import MultipleOf
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [16, 32]
        group = SimpleNamespace(backend=backend)
        # kv_manager_block_size=64, 64%32==0, 32 is supported
        result = RBLNModelRunner.select_common_block_size(64, [group])
        assert result == 32

    def test_no_common_raises(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [7]
        group = SimpleNamespace(backend=backend)
        with pytest.raises(ValueError, match="No common block size"):
            RBLNModelRunner.select_common_block_size(32, [group])

    def test_multiple_of_support(self):
        from vllm.v1.attention.backend import MultipleOf
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [MultipleOf(16)]
        group = SimpleNamespace(backend=backend)
        result = RBLNModelRunner.select_common_block_size(64, [group])
        assert result == 64  # 64 % 16 == 0

    def test_multiple_backends(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        b1 = MagicMock()
        b1.get_supported_kernel_block_sizes.return_value = [16, 32, 64]
        b2 = MagicMock()
        b2.get_supported_kernel_block_sizes.return_value = [8, 16, 32]
        groups = [SimpleNamespace(backend=b1), SimpleNamespace(backend=b2)]
        result = RBLNModelRunner.select_common_block_size(64, groups)
        assert result == 32  # largest common factor of 64


# ===========================================================================
# Tests: calculate_reorder_batch_threshold
# ===========================================================================


class TestCalculateReorderBatchThreshold:
    def test_noop(self):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        stub = _make_runner_stub()
        RBLNModelRunner.calculate_reorder_batch_threshold(stub)
