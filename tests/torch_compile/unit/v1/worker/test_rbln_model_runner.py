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

"""Feature tests for RBLNModelRunner: mixin compliance, async output,
named tuples, get_supported_tasks integration, and bug-catching scenarios."""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import MoveDirectionality
from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm_rbln.v1.worker.rbln_model_runner as model_runner_module
from vllm_rbln.v1.worker.rbln_model_runner import (
    AsyncRBLNModelRunnerOutput,
    DummyRunState,
    ExecuteModelState,
    RBLNModelRunner,
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
    return SimpleNamespace(**defaults)


def _bind(stub, method_name):
    """Bind an RBLNModelRunner unbound method to a stub."""
    method = getattr(RBLNModelRunner, method_name)
    return types.MethodType(method, stub)


# ===========================================================================
# 1. Mixin interface compliance
# ===========================================================================


class TestMixinInterfaceCompliance:
    """Verify that RBLNModelRunner inherits the expected mixin methods."""

    def test_inherits_lora_mixin(self):
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        assert issubclass(RBLNModelRunner, LoRAModelRunnerMixin)

    def test_inherits_kv_connector_mixin(self):
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )

        assert issubclass(RBLNModelRunner, KVConnectorModelRunnerMixin)

    @pytest.mark.parametrize(
        "method_name",
        ["add_lora", "remove_lora", "list_loras", "pin_lora"],
    )
    def test_lora_mixin_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing LoRA mixin method: {method_name}"
        )
        assert callable(getattr(RBLNModelRunner, method_name))

    @pytest.mark.parametrize(
        "method_name",
        [
            "load_lora_model",
            "set_active_loras",
            "maybe_remove_all_loras",
            "maybe_setup_dummy_loras",
            "maybe_select_dummy_loras",
            "maybe_dummy_run_with_lora",
        ],
    )
    def test_lora_mixin_extended_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name)

    @pytest.mark.parametrize(
        "method_name",
        [
            "allocate_uniform_kv_caches",
            "ensure_kv_transfer_shutdown",
            "finalize_kv_connector",
            "kv_connector_no_forward",
            "maybe_get_kv_connector_output",
            "use_uniform_kv_cache",
        ],
    )
    def test_kv_connector_mixin_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing KV connector mixin method: {method_name}"
        )

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_kv_cache_spec",
            "load_model",
            "execute_model",
            "sample_tokens",
            "get_supported_tasks",
            "get_model",
            "warm_up_model",
            "dummy_run",
            "initialize_attn_backend",
            "initialize_kv_cache",
        ],
    )
    def test_expected_public_methods_exist(self, method_name):
        assert hasattr(RBLNModelRunner, method_name), (
            f"RBLNModelRunner missing public method: {method_name}"
        )


# ===========================================================================
# 2. AsyncRBLNModelRunnerOutput
# ===========================================================================


class TestAsyncRBLNModelRunnerOutputFeature:
    def _make_output(self, num_reqs=3, invalid_indices=None):
        """Helper to create an AsyncRBLNModelRunnerOutput with controllable state."""
        mro = MagicMock(spec=ModelRunnerOutput)
        sampled = torch.tensor([[10], [20], [30]][:num_reqs])
        stream = MagicMock()
        output = AsyncRBLNModelRunnerOutput(
            model_runner_output=mro,
            sampled_token_ids=sampled,
            invalid_req_indices=invalid_indices or [],
            async_output_copy_stream=stream,
        )
        return output, mro

    def test_invalid_req_indices_clears_sampled_tokens(self):
        """Verify that invalid_req_indices correctly clears the corresponding
        sampled token entries when get_output() is called."""
        output, mro = self._make_output(num_reqs=3, invalid_indices=[0, 2])

        # Simulate what would happen after the async copy completes:
        # Manually set the internal state that get_output() reads.
        output._sampled_token_ids_cpu = torch.tensor([[10], [20], [30]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()

        # Index 0 and 2 should be cleared (empty lists)
        assert result.sampled_token_ids[0] == []
        assert result.sampled_token_ids[1] == [20]  # index 1 is not invalid, preserved
        assert result.sampled_token_ids[2] == []

    def test_get_output_returns_model_runner_output(self):
        """Verify get_output() returns the underlying ModelRunnerOutput."""
        output, mro = self._make_output(num_reqs=2, invalid_indices=[])

        output._sampled_token_ids_cpu = torch.tensor([[10], [20]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result is mro
        assert result.sampled_token_ids == [[10], [20]]

    def test_get_output_synchronizes_event(self):
        """Verify get_output() calls synchronize on the copy event."""
        output, mro = self._make_output(num_reqs=1, invalid_indices=[])
        mock_event = MagicMock()
        output._async_copy_ready_event = mock_event
        output._sampled_token_ids_cpu = torch.tensor([[42]])

        output.get_output()
        mock_event.synchronize.assert_called_once()

    def test_get_output_deletes_device_tensor(self):
        """After get_output(), the device tensor reference should be released."""
        output, _ = self._make_output(num_reqs=1, invalid_indices=[])
        output._async_copy_ready_event = MagicMock()
        output._sampled_token_ids_cpu = torch.tensor([[1]])

        output.get_output()
        assert not hasattr(output, "_sampled_token_ids")

    def test_all_invalid_indices(self):
        """When all indices are invalid, all sampled tokens should be cleared."""
        output, mro = self._make_output(num_reqs=3, invalid_indices=[0, 1, 2])
        output._sampled_token_ids_cpu = torch.tensor([[1], [2], [3]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == [[], [], []]

    def test_no_invalid_indices(self):
        """When no indices are invalid, all tokens should be preserved."""
        output, mro = self._make_output(num_reqs=2, invalid_indices=[])
        output._sampled_token_ids_cpu = torch.tensor([[5], [6]])
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == [[5], [6]]


# ===========================================================================
# 3. ExecuteModelState / DummyRunState named tuples
# ===========================================================================


class TestExecuteModelStateFeature:
    def test_field_names(self):
        expected_fields = (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "slot_mappings",
        )
        assert ExecuteModelState._fields == expected_fields

    def test_field_count(self):
        assert len(ExecuteModelState._fields) == 8

    def test_is_named_tuple(self):
        assert issubclass(ExecuteModelState, tuple)

    def test_construct_with_all_none_optionals(self):
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(2, 10),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.ones(2, 10),
            sample_hidden_states=None,
            aux_hidden_states=None,
            slot_mappings=None,
        )
        assert state.spec_decode_metadata is None
        assert isinstance(state.logits, torch.Tensor)
        assert isinstance(state.hidden_states, torch.Tensor)

    def test_slot_mappings_dict(self):
        """slot_mappings can be a dict of tensors."""
        mappings = {"layer_0": torch.tensor([0, 1, 2])}
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            slot_mappings=mappings,
        )
        assert "layer_0" in state.slot_mappings

    def test_slot_mappings_list_of_dicts(self):
        """slot_mappings can also be a list of dicts."""
        mappings = [
            {"layer_0": torch.tensor([0])},
            {"layer_1": torch.tensor([1])},
        ]
        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.zeros(1),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.zeros(1),
            sample_hidden_states=None,
            aux_hidden_states=None,
            slot_mappings=mappings,
        )
        assert len(state.slot_mappings) == 2


class TestDummyRunStateFeature:
    def test_field_names(self):
        expected_fields = (
            "attn_metadata",
            "num_input_tokens",
            "input_ids",
            "positions",
        )
        assert DummyRunState._fields == expected_fields

    def test_field_count(self):
        assert len(DummyRunState._fields) == 4

    def test_is_named_tuple(self):
        assert issubclass(DummyRunState, tuple)

    def test_typical_construction(self):
        state = DummyRunState(
            attn_metadata={0: {"key": "value"}},
            num_input_tokens=32,
            input_ids={0: torch.zeros(32, dtype=torch.long)},
            positions={0: torch.arange(32)},
        )
        assert state.num_input_tokens == 32
        assert isinstance(state.attn_metadata, dict)
        assert isinstance(state.input_ids, dict)
        assert isinstance(state.positions, dict)

    def test_empty_dicts(self):
        state = DummyRunState(
            attn_metadata={},
            num_input_tokens=0,
            input_ids={},
            positions={},
        )
        assert state.num_input_tokens == 0
        assert len(state.attn_metadata) == 0


# ===========================================================================
# 4. get_supported_tasks integration
# ===========================================================================


class TestGetSupportedTasksFeature:
    def _make_stub_with_task_methods(self, **kw):
        stub = _make_runner_stub(**kw)
        stub.get_model = _bind(stub, "get_model")
        stub.get_supported_generation_tasks = _bind(
            stub, "get_supported_generation_tasks"
        )
        stub.get_supported_pooling_tasks = _bind(stub, "get_supported_pooling_tasks")
        stub.get_supported_tasks = _bind(stub, "get_supported_tasks")
        return stub

    def test_text_generation_model_returns_generate_task(self):
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
                return_value=True,
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
                return_value=False,
            ),
        ):
            tasks = stub.get_supported_tasks()
        task_names = [t if isinstance(t, str) else t for t in tasks]
        assert "generate" in task_names

    def test_pooling_model_returns_pooling_tasks(self):
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed", "classify"]
        stub = self._make_stub_with_task_methods(model=model)
        stub.model_config.runner_type = "pooling"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            tasks = stub.get_supported_tasks()
        task_names = list(tasks)
        assert "embed" in task_names
        assert "classify" in task_names

    def test_generate_runner_returns_no_pooling_tasks(self):
        """A generate runner should not include pooling tasks."""
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
                return_value=True,
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
                return_value=False,
            ),
        ):
            tasks = stub.get_supported_tasks()
        # Should only contain generation tasks, not pooling
        for t in tasks:
            assert t not in ("embed", "classify", "score", "encode")

    def test_pooling_runner_returns_no_generate_tasks(self):
        """A pooling runner should not include generate tasks."""
        model = MagicMock()
        model.pooler.get_supported_tasks.return_value = ["embed"]
        stub = self._make_stub_with_task_methods(model=model)
        stub.model_config.runner_type = "pooling"
        with patch(
            "vllm_rbln.v1.worker.rbln_model_runner.is_pooling_model",
            return_value=True,
        ):
            tasks = stub.get_supported_tasks()
        assert "generate" not in tasks

    def test_returns_tuple(self):
        stub = self._make_stub_with_task_methods()
        stub.model_config.runner_type = "generate"
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.is_text_generation_model",
                return_value=True,
            ),
            patch(
                "vllm_rbln.v1.worker.rbln_model_runner.supports_transcription",
                return_value=False,
            ),
        ):
            tasks = stub.get_supported_tasks()
        assert isinstance(tasks, tuple)


# ===========================================================================
# 5. Bug-catching: edge cases and integration
# ===========================================================================


class TestEdgeCases:
    """Bug-catching tests for edge conditions."""

    def test_async_output_with_empty_sampled_tokens(self):
        """AsyncRBLNModelRunnerOutput should handle zero requests gracefully."""
        mro = MagicMock(spec=ModelRunnerOutput)
        sampled = torch.zeros(0, 1, dtype=torch.long)
        output = AsyncRBLNModelRunnerOutput(
            model_runner_output=mro,
            sampled_token_ids=sampled,
            invalid_req_indices=[],
            async_output_copy_stream=MagicMock(),
        )
        output._sampled_token_ids_cpu = sampled
        output._async_copy_ready_event = MagicMock()

        result = output.get_output()
        assert result.sampled_token_ids == []

    def test_get_model_returns_model_attribute(self):
        """get_model should return the model attribute directly."""
        model = MagicMock()
        stub = _make_runner_stub(model=model)
        result = RBLNModelRunner.get_model(stub)
        assert result is model

    def test_compute_logits_delegates_to_model(self):
        """compute_logits should delegate to model.compute_logits."""
        stub = _make_runner_stub()
        hidden = torch.randn(2, 10)
        expected = torch.randn(2, 100)
        stub.model.compute_logits.return_value = expected
        assert RBLNModelRunner.compute_logits(stub, hidden) is expected
        stub.model.compute_logits.assert_called_once_with(hidden)

    def test_use_wrapped_compute_logits_default_true(self):
        """By default (no lora, no spec_decode), use_wrapped_compute_logits is True."""
        stub = _make_runner_stub(lora_config=None, speculative_config=None)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_use_wrapped_compute_logits_false_with_lora(self):
        """With lora_config, use_wrapped_compute_logits is False."""
        stub = _make_runner_stub(lora_config=MagicMock())
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False

    def test_mixin_mro_order(self):
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin,
        )
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        mro = RBLNModelRunner.__mro__
        lora_idx = mro.index(LoRAModelRunnerMixin)
        kv_idx = mro.index(KVConnectorModelRunnerMixin)
        assert lora_idx < kv_idx, (
            "LoRAModelRunnerMixin should precede KVConnectorModelRunnerMixin in MRO"
        )

    def test_constructor_signature(self):
        """RBLNModelRunner.__init__ should accept (self, vllm_config, device)."""
        import inspect

        sig = inspect.signature(RBLNModelRunner.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "vllm_config" in params
        assert "device" in params


# ===========================================================================
# 6. _get_cumsum_and_arange – REAL code path
# ===========================================================================


class TestGetCumsumAndArange:
    """Test RBLNModelRunner._get_cumsum_and_arange with real numpy arrays."""

    def _call(self, num_tokens, cumsum_dtype=None, arange_size=10000):
        stub = SimpleNamespace(arange_np=np.arange(arange_size, dtype=np.int64))
        bound = types.MethodType(RBLNModelRunner._get_cumsum_and_arange, stub)
        return bound(num_tokens, cumsum_dtype=cumsum_dtype)

    def test_basic_example(self):
        """Docstring example: [2, 5, 3] -> ([2, 7, 10], [0,1,0,1,2,3,4,0,1,2])."""
        arr = np.array([2, 5, 3], dtype=np.int64)
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [2, 7, 10])
        np.testing.assert_array_equal(arange, [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])

    def test_single_element(self):
        arr = np.array([4], dtype=np.int64)
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [4])
        np.testing.assert_array_equal(arange, [0, 1, 2, 3])

    def test_all_ones(self):
        arr = np.array([1, 1, 1], dtype=np.int64)
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [1, 2, 3])
        np.testing.assert_array_equal(arange, [0, 0, 0])

    def test_cumsum_dtype(self):
        arr = np.array([2, 3], dtype=np.int64)
        cu, arange = self._call(arr, cumsum_dtype=np.int32)
        assert cu.dtype == np.int32
        np.testing.assert_array_equal(cu, [2, 5])

    def test_large_values(self):
        arr = np.array([100, 200], dtype=np.int64)
        cu, arange = self._call(arr)
        assert cu[-1] == 300
        assert len(arange) == 300
        # First segment: 0..99, second segment: 0..199
        assert arange[0] == 0
        assert arange[99] == 99
        assert arange[100] == 0
        assert arange[299] == 199

    def test_two_elements(self):
        arr = np.array([3, 2], dtype=np.int64)
        cu, arange = self._call(arr)
        np.testing.assert_array_equal(cu, [3, 5])
        np.testing.assert_array_equal(arange, [0, 1, 2, 0, 1])


# ===========================================================================
# 7. is_prefills – REAL code path
# ===========================================================================


class TestIsPrefills:
    """Test RBLNModelRunner.is_prefills with real numpy arrays."""

    def _call(self, num_computed, num_tokens_no_spec):
        batch = MagicMock(spec=InputBatch)
        batch.num_computed_tokens_cpu = np.array(num_computed, dtype=np.int64)
        batch.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)
        stub = SimpleNamespace(input_batch=batch)
        bound = types.MethodType(RBLNModelRunner.is_prefills, stub)
        return bound()

    def test_all_prefill(self):
        # computed < total - 1 means prefill
        result = self._call([0, 0, 0], [10, 20, 30])
        np.testing.assert_array_equal(result, [True, True, True])

    def test_all_decode(self):
        # computed >= total - 1 means decode
        result = self._call([9, 19, 29], [10, 20, 30])
        np.testing.assert_array_equal(result, [False, False, False])

    def test_mixed(self):
        result = self._call([0, 19], [10, 20])
        np.testing.assert_array_equal(result, [True, False])

    def test_boundary(self):
        # num_computed == num_tokens - 2 => True (prefill)
        # num_computed == num_tokens - 1 => False (decode)
        result = self._call([8, 9], [10, 10])
        np.testing.assert_array_equal(result, [True, False])


# ===========================================================================
# 8. use_wrapped_compute_logits – REAL code path
# ===========================================================================


class TestUseWrappedComputeLogits:
    """Test RBLNModelRunner.use_wrapped_compute_logits with real logic."""

    def test_no_lora_no_spec(self):
        stub = SimpleNamespace(lora_config=None, speculative_config=None)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_with_lora(self):
        stub = SimpleNamespace(lora_config=MagicMock(), speculative_config=None)
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False

    def test_with_eagle_spec(self):
        stub = SimpleNamespace(
            lora_config=None,
            speculative_config=SimpleNamespace(method="eagle"),
        )
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_with_eagle3_spec(self):
        stub = SimpleNamespace(
            lora_config=None,
            speculative_config=SimpleNamespace(method="eagle3"),
        )
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_with_non_eagle_spec(self):
        stub = SimpleNamespace(
            lora_config=None,
            speculative_config=SimpleNamespace(method="ngram"),
        )
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is True

    def test_lora_and_spec_both_set(self):
        stub = SimpleNamespace(
            lora_config=MagicMock(),
            speculative_config=SimpleNamespace(method="eagle"),
        )
        assert RBLNModelRunner.use_wrapped_compute_logits(stub) is False


class TestUpdateStatesAfterExecute:
    def test_non_hybrid_is_noop(self):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=False),
            speculative_config=None,
        )
        RBLNModelRunner._update_states_after_model_execute(
            runner, torch.tensor([[1, 2, 3]])
        )

    def test_hybrid_without_spec_is_noop(self):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=True),
            speculative_config=None,
        )
        RBLNModelRunner._update_states_after_model_execute(
            runner, torch.tensor([[1, 2, 3]])
        )

    def test_hybrid_with_spec_counts_accepted(self):
        num_accepted = np.zeros(3, dtype=np.int64)
        batch = MagicMock(spec=InputBatch)
        batch.num_accepted_tokens_cpu = num_accepted
        runner = SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=True),
            speculative_config=SimpleNamespace(),
            input_batch=batch,
        )

        output = torch.tensor(
            [
                [10, 20, -1],
                [30, -1, -1],
                [40, 50, 60],
            ]
        )
        RBLNModelRunner._update_states_after_model_execute(runner, output)

        np.testing.assert_array_equal(num_accepted, [2, 1, 3])


class TestProposeDraftTokenIdsIntermediatePrefill:
    def test_medusa_returns_empty_drafts_without_calling_drafter(self, monkeypatch):
        class DummyMedusaProposer:
            def propose(self, **kwargs):
                raise AssertionError("intermediate prefill must not propose drafts")

        monkeypatch.setattr(
            model_runner_module, "RBLNMedusaProposer", DummyMedusaProposer
        )

        runner = SimpleNamespace(
            speculative_config=SimpleNamespace(method="medusa"),
            drafter=DummyMedusaProposer(),
            input_batch=SimpleNamespace(req_ids=["req-0", "req-1"]),
        )

        draft_token_ids = RBLNModelRunner.propose_draft_token_ids(
            runner,
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=4),
            sampled_token_ids=[],
            sampling_metadata=MagicMock(),
            hidden_states=torch.empty(4, 8),
            sample_hidden_states=None,
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=MagicMock(),
            slot_mappings=None,
            is_intermediate_prefill=True,
        )

        assert draft_token_ids == [[], []]

    def test_eagle_runs_prefill_only_without_copying_sampled_count(self, monkeypatch):
        class DummyEagleProposer:
            def __init__(self):
                self.prepare_next_token_ids_padded = MagicMock(
                    side_effect=self._prepare_next_token_ids_padded
                )
                self.prefill_only = MagicMock()
                self.propose = MagicMock(
                    side_effect=AssertionError(
                        "intermediate prefill must not propose drafts"
                    )
                )

            def _prepare_next_token_ids_padded(
                self,
                common_attn_metadata,
                sampled_token_ids,
                requests,
                input_batch,
                discard_request_mask,
            ):
                torch.testing.assert_close(
                    sampled_token_ids,
                    torch.full((2, 1), -1, dtype=torch.int32),
                )
                torch.testing.assert_close(
                    discard_request_mask,
                    torch.ones(2, dtype=torch.bool),
                )
                assert input_batch.num_reqs == 2
                return (
                    torch.tensor([11, 22], dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                )

        monkeypatch.setattr(
            model_runner_module, "RBLNEagleProposer", DummyEagleProposer
        )

        drafter = DummyEagleProposer()
        copy_valid_sampled_token_count = MagicMock()
        common_attn_metadata = MagicMock()
        hidden_states = torch.randn(4, 8)
        runner = SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="eagle",
                disable_padded_drafter_batch=False,
                use_eagle=lambda: True,
            ),
            drafter=drafter,
            input_batch=SimpleNamespace(
                num_reqs=2,
                req_ids=["req-0", "req-1"],
                sampling_metadata=MagicMock(),
            ),
            requests={"req-0": MagicMock(), "req-1": MagicMock()},
            discard_request_mask=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.bool)),
            input_ids=SimpleNamespace(gpu=torch.arange(8, dtype=torch.int32)),
            device=torch.device("cpu"),
            use_aux_hidden_state_outputs=False,
            supports_mm_inputs=False,
            _get_positions=lambda num_tokens: torch.arange(
                num_tokens, dtype=torch.int64
            ),
            _copy_valid_sampled_token_count=copy_valid_sampled_token_count,
        )

        draft_token_ids = RBLNModelRunner.propose_draft_token_ids(
            runner,
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=4),
            sampled_token_ids=torch.empty(0, 1, dtype=torch.int32),
            sampling_metadata=MagicMock(),
            hidden_states=hidden_states,
            sample_hidden_states=None,
            aux_hidden_states=None,
            spec_decode_metadata=None,
            common_attn_metadata=common_attn_metadata,
            slot_mappings=None,
            is_intermediate_prefill=True,
        )

        assert draft_token_ids is None
        copy_valid_sampled_token_count.assert_not_called()
        drafter.prepare_next_token_ids_padded.assert_called_once()
        drafter.propose.assert_not_called()

        drafter.prefill_only.assert_called_once()
        prefill_kwargs = drafter.prefill_only.call_args.kwargs
        torch.testing.assert_close(
            prefill_kwargs["target_token_ids"],
            torch.arange(4, dtype=torch.int32),
        )
        torch.testing.assert_close(
            prefill_kwargs["target_positions"],
            torch.arange(4, dtype=torch.int64),
        )
        torch.testing.assert_close(
            prefill_kwargs["target_hidden_states"],
            hidden_states,
        )
        torch.testing.assert_close(
            prefill_kwargs["next_token_ids"],
            torch.tensor([11, 22], dtype=torch.int32),
        )
        assert prefill_kwargs["common_attn_metadata"] is common_attn_metadata
        assert prefill_kwargs["mm_embed_inputs"] is None


# ===========================================================================
# 9. _to_list – REAL code path
# ===========================================================================


class TestToList:
    """Test RBLNModelRunner._to_list with real tensors."""

    def _call(self, sampled_token_ids, pinned_size=16):
        pinned = torch.zeros(pinned_size, 1, dtype=torch.long)
        stub = SimpleNamespace(
            sampled_token_ids_pinned_cpu=pinned,
        )
        bound = types.MethodType(RBLNModelRunner._to_list, stub)
        return bound(sampled_token_ids)

    def test_basic(self):
        t = torch.tensor([[5], [10], [15]])
        result = self._call(t)
        assert result == [[5], [10], [15]]

    def test_single(self):
        t = torch.tensor([[42]])
        result = self._call(t)
        assert result == [[42]]

    def test_preserves_values(self):
        t = torch.tensor([[0], [1], [9999]])
        result = self._call(t, pinned_size=8)
        assert result == [[0], [1], [9999]]


# ===========================================================================
# 15. _may_reorder_batch – REAL code path with env override
# ===========================================================================


class TestMayReorderBatch:
    """Test RBLNModelRunner._may_reorder_batch with REAL sorting logic.

    Uses MagicMock(spec=InputBatch) so that accessing a removed/renamed
    attribute raises AttributeError immediately — no silent drift when
    upstream changes the InputBatch interface.
    """

    @staticmethod
    def _make_input_batch_mock(req_ids, num_tokens_no_spec, swap_fn=None):
        """Spec-bound InputBatch mock for the early-return paths (env off,
        no kv groups, already-sorted) that don't reach the reindex."""
        batch = MagicMock(spec=InputBatch)
        batch.req_ids = req_ids
        batch.num_tokens_no_spec = np.array(num_tokens_no_spec)
        if swap_fn is not None:
            batch.swap_states = swap_fn
        return batch

    # --- real InputBatch helpers for the vectorized-reorder (B) tests ---

    @staticmethod
    def _real_input_batch(
        num_tokens,
        rng,
        max_num_reqs=8,
        max_model_len=64,
        vocab_size=100,
        block_size=16,
        is_pooling_model=False,
    ):
        """Build and populate a real InputBatch with distinct per-slot data
        in every field that swap_states / _reorder_input_batch touches."""
        n = len(num_tokens)
        ib = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=512,
            device=torch.device("cpu"),
            pin_memory=False,
            vocab_size=vocab_size,
            block_sizes=[block_size],
            kernel_block_sizes=[block_size],
            max_num_blocks_per_req=[max_model_len // block_size],
            logitsprocs=None,
            is_spec_decode=False,
            is_pooling_model=is_pooling_model,
        )
        # _req_ids length == num_reqs (the runner uses len(req_ids) as n).
        ib._req_ids = [f"r{i}" for i in range(n)]
        ib.req_id_to_index = {f"r{i}": i for i in range(n)}
        ib.req_output_token_ids[:n] = [[i, i * 10] for i in range(n)]
        ib.spec_token_ids[:n] = [[i + 1] for i in range(n)]
        ib.num_tokens_no_spec[:n] = np.asarray(
            num_tokens, dtype=ib.num_tokens_no_spec.dtype
        )
        for name in (
            "num_prompt_tokens",
            "num_computed_tokens_cpu",
            "temperature_cpu",
            "top_p_cpu",
            "top_k_cpu",
            "frequency_penalties_cpu",
            "presence_penalties_cpu",
            "repetition_penalties_cpu",
            "num_accepted_tokens_cpu",
            "request_lora_mapping",
        ):
            arr = getattr(ib, name)
            arr[:n] = rng.integers(1, 999, size=n).astype(arr.dtype)
        ib.token_ids_cpu[:n] = rng.integers(0, vocab_size, size=(n, max_model_len))
        ib.is_token_ids[:n] = rng.integers(0, 2, size=(n, max_model_len)).astype(bool)
        bt = ib.block_table.block_tables[0]
        bt.num_blocks_per_row[:n] = rng.integers(0, max_model_len // block_size, size=n)
        bt.block_table.np[:n] = rng.integers(0, 500, size=bt.block_table.np[:n].shape)
        # sparse index-keyed dicts (deterministic so ref/ours start identical)
        ib.generators = {}
        for i in range(0, n, 2):
            g = torch.Generator()
            g.manual_seed(1000 + i)
            ib.generators[i] = g
        ib.bad_words_token_ids = {
            i: [[int(x) for x in rng.integers(0, vocab_size, size=2)]]
            for i in range(1, n, 3)
        }
        ib.req_prompt_embeds = {
            i: torch.from_numpy(rng.random(4).astype(np.float32))
            for i in range(0, n, 4)
        }
        return ib, n

    @staticmethod
    def _cycle_pairs(sorted_order):
        """Pairwise swaps realizing `sorted_order` (mirrors _may_reorder_batch)."""
        n = len(sorted_order)
        orig = np.arange(n)
        m = {int(s): int(d) for s, d in zip(orig[sorted_order], orig)}
        pairs = []
        for src in list(m):
            dst = m[src]
            while src != dst:
                pairs.append((src, dst))
                next_dst = m.get(dst, dst)
                m[dst] = dst
                dst = next_dst
        return pairs

    @staticmethod
    def _assert_input_batch_equal(a, b, n):
        assert a._req_ids[:n] == b._req_ids[:n]
        assert a.req_id_to_index == b.req_id_to_index
        assert a.req_output_token_ids[:n] == b.req_output_token_ids[:n]
        assert a.spec_token_ids[:n] == b.spec_token_ids[:n]
        for name in (
            "num_tokens_no_spec",
            "num_prompt_tokens",
            "num_computed_tokens_cpu",
            "temperature_cpu",
            "top_p_cpu",
            "top_k_cpu",
            "frequency_penalties_cpu",
            "presence_penalties_cpu",
            "repetition_penalties_cpu",
            "num_accepted_tokens_cpu",
            "request_lora_mapping",
        ):
            np.testing.assert_array_equal(
                getattr(a, name)[:n], getattr(b, name)[:n], err_msg=name
            )
        # token matrices: compare each row only over its meaningful extent
        # (num_tokens + spec). The vectorized reindex narrows to valid columns,
        # so the don't-care tail beyond a row's extent need not match
        # swap_states' full-row copy. Independent of the production valid_w
        # formula — a too-small valid_w would corrupt some row's [:ext].
        for k in range(n):
            ext = int(b.num_tokens_no_spec[k]) + len(b.spec_token_ids[k])
            np.testing.assert_array_equal(
                a.token_ids_cpu[k, :ext],
                b.token_ids_cpu[k, :ext],
                err_msg="token_ids_cpu",
            )
            np.testing.assert_array_equal(
                a.is_token_ids[k, :ext], b.is_token_ids[k, :ext], err_msg="is_token_ids"
            )
        bta = a.block_table.block_tables[0]
        btb = b.block_table.block_tables[0]
        np.testing.assert_array_equal(
            bta.num_blocks_per_row[:n], btb.num_blocks_per_row[:n]
        )
        np.testing.assert_array_equal(bta.block_table.np[:n], btb.block_table.np[:n])
        assert a.batch_update_builder.moved == b.batch_update_builder.moved
        # index-keyed dicts (generators compared by seed; full-dict replacement
        # must match swap_dict_values for keys in [0, n))
        assert {k: g.initial_seed() for k, g in a.generators.items()} == {
            k: g.initial_seed() for k, g in b.generators.items()
        }
        assert a.bad_words_token_ids == b.bad_words_token_ids
        assert set(a.req_prompt_embeds) == set(b.req_prompt_embeds)
        for k in a.req_prompt_embeds:
            assert torch.equal(a.req_prompt_embeds[k], b.req_prompt_embeds[k])

    def test_no_reorder_when_env_disabled(self):
        """When VLLM_RBLN_SORT_BATCH is False, no reordering occurs."""
        batch = self._make_input_batch_mock(["a", "b", "c"], [10, 30, 20])
        stub = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
            input_batch=batch,
        )
        bound = types.MethodType(RBLNModelRunner._may_reorder_batch, stub)
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_SORT_BATCH = False
            bound(MagicMock())
        np.testing.assert_array_equal(batch.num_tokens_no_spec, [10, 30, 20])

    def test_no_reorder_when_no_kv_cache_groups(self):
        """When kv_cache_groups is empty, no reordering occurs."""
        batch = self._make_input_batch_mock(["a", "b"], [5, 10])
        stub = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
            input_batch=batch,
        )
        bound = types.MethodType(RBLNModelRunner._may_reorder_batch, stub)
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_SORT_BATCH = True
            bound(MagicMock())
        np.testing.assert_array_equal(batch.num_tokens_no_spec, [5, 10])

    def test_reorder_sorts_descending(self):
        """End-to-end: _may_reorder_batch sorts a real batch by descending
        num_tokens_no_spec, carries per-slot state along, and emits move
        records."""
        ib, n = self._real_input_batch([10, 30, 20], np.random.default_rng(0))
        # Tag token rows so we can verify rows follow the permutation.
        ib.token_ids_cpu[:n, 0] = [100, 101, 102]
        runner = object.__new__(RBLNModelRunner)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[1])
        runner.input_batch = ib
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_SORT_BATCH = True
            runner._may_reorder_batch(MagicMock())

        np.testing.assert_array_equal(ib.num_tokens_no_spec[:n], [30, 20, 10])
        # idx1(30)->slot0, idx2(20)->slot1, idx0(10)->slot2
        np.testing.assert_array_equal(ib.token_ids_cpu[:n, 0], [101, 102, 100])
        assert ib.req_id_to_index == {"r1": 0, "r2": 1, "r0": 2}
        assert ib._req_ids[:n] == ["r1", "r2", "r0"]
        assert ib.batch_update_builder.moved, "expected emitted move records"

    def test_already_sorted_no_swaps(self):
        """If already sorted descending, no swaps needed."""
        swap_log = []

        def mock_swap(src, dst):
            swap_log.append((src, dst))

        batch = self._make_input_batch_mock(
            ["a", "b", "c"],
            [30, 20, 10],
            swap_fn=mock_swap,
        )
        stub = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
            input_batch=batch,
        )
        bound = types.MethodType(RBLNModelRunner._may_reorder_batch, stub)
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_SORT_BATCH = True
            bound(MagicMock())
        assert len(swap_log) == 0

    def test_reorder_matches_swap_states(self):
        """_reorder_input_batch is equivalent to N-1 swap_states calls
        across random permutations and every per-request field."""
        runner = object.__new__(RBLNModelRunner)
        for trial in range(20):
            seed_rng = np.random.default_rng(trial)
            n = int(seed_rng.integers(2, 8))
            toks = [int(t) for t in seed_rng.integers(1, 50, size=n)]
            ref, _ = self._real_input_batch(toks, np.random.default_rng(100 + trial))
            ours, _ = self._real_input_batch(toks, np.random.default_rng(100 + trial))
            perm = np.argsort(np.asarray(toks) * (-1), kind="stable")
            pairs = self._cycle_pairs(perm)
            # reference: realize the permutation via swap_states
            for i1, i2 in pairs:
                ref.swap_states(i1, i2)
            # ours: emit the same move records, then one vectorized reindex
            for i1, i2 in pairs:
                ours.batch_update_builder.moved.append(
                    (i1, i2, MoveDirectionality.SWAP)
                )
            runner._reorder_input_batch(ours, perm)
            self._assert_input_batch_equal(ref, ours, n)

    def test_reorder_allowed_token_ids_mask_gather(self):
        """(B) allowed_token_ids_mask is permuted correctly (a true gather;
        upstream swap_states' row tuple-swap is buggy here, so we don't
        mirror it)."""
        runner = object.__new__(RBLNModelRunner)
        ib, n = self._real_input_batch([10, 30, 20], np.random.default_rng(0))
        rng = np.random.default_rng(7)
        ib.allowed_token_ids_mask_cpu_tensor = torch.from_numpy(
            rng.integers(0, 2, size=(8, 100)).astype(bool)
        )
        orig = ib.allowed_token_ids_mask_cpu_tensor[:n].clone()
        perm = np.argsort(np.asarray([10, 30, 20]) * (-1), kind="stable")
        runner._reorder_input_batch(ib, perm)
        assert torch.equal(ib.allowed_token_ids_mask_cpu_tensor[:n], orig[perm])

    def test_reorder_narrows_to_valid_width(self):
        """With num_tokens (+spec) << max_model_len, only the valid token
        columns are permuted; the don't-care tail (columns >= num_tokens,
        never read) is left untouched. _reorder_input_batch derives the valid
        width itself from num_tokens_no_spec + spec length."""
        runner = object.__new__(RBLNModelRunner)
        toks = [2, 4, 3]
        ib, n = self._real_input_batch(toks, np.random.default_rng(0))
        # _real_input_batch sets one spec token per slot -> valid_w = max + 1.
        valid_w = max(toks) + 1
        full_w = ib.token_ids_cpu.shape[1]
        assert valid_w < full_w
        ib.token_ids_cpu[:n, valid_w:] = -7  # sentinel in the don't-care tail
        orig = ib.token_ids_cpu.copy()
        orig_is = ib.is_token_ids.copy()
        perm = np.argsort(np.asarray(toks) * (-1), kind="stable")
        runner._reorder_input_batch(ib, perm)
        # valid columns are gathered by the permutation ...
        np.testing.assert_array_equal(
            ib.token_ids_cpu[:n, :valid_w], orig[perm][:, :valid_w]
        )
        np.testing.assert_array_equal(
            ib.is_token_ids[:n, :valid_w], orig_is[perm][:, :valid_w]
        )
        # ... and the tail is untouched (token_ids still the sentinel, and
        # is_token_ids unchanged — neither is permuted).
        np.testing.assert_array_equal(
            ib.token_ids_cpu[:n, valid_w:], np.full((n, full_w - valid_w), -7)
        )
        np.testing.assert_array_equal(
            ib.is_token_ids[:n, valid_w:], orig_is[:n, valid_w:]
        )

    def test_reorder_pooling_emits_no_move_records(self):
        """Pooling models carry no sampling/logits state, so the reorder must
        not emit move records (matching upstream swap_states' early return)."""
        ib, n = self._real_input_batch(
            [10, 30, 20], np.random.default_rng(0), is_pooling_model=True
        )
        runner = object.__new__(RBLNModelRunner)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[1])
        runner.input_batch = ib
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_RBLN_SORT_BATCH = True
            runner._may_reorder_batch(MagicMock())
        # Non-sampling state is still reordered ...
        np.testing.assert_array_equal(ib.num_tokens_no_spec[:n], [30, 20, 10])
        # ... but no logits-processor move records are emitted.
        assert ib.batch_update_builder.moved == []
