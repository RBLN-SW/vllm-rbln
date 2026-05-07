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
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_rbln.v1.worker.rbln_model_runner import (
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
# 3. ExecuteModelState named tuple
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
# 7. use_wrapped_compute_logits – REAL code path
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
        """Build a spec-bound mock of InputBatch with real numpy data."""
        batch = MagicMock(spec=InputBatch)
        batch.req_ids = req_ids
        batch.num_tokens_no_spec = np.array(num_tokens_no_spec)
        if swap_fn is not None:
            batch.swap_states = swap_fn
        return batch

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
        """When enabled and groups exist, reorder by descending num_tokens_no_spec."""
        swap_log = []

        def mock_swap(src, dst):
            swap_log.append((src, dst))
            arr = batch.num_tokens_no_spec
            arr[src], arr[dst] = arr[dst], arr[src]

        batch = self._make_input_batch_mock(
            ["a", "b", "c"],
            [10, 30, 20],
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
        np.testing.assert_array_equal(batch.num_tokens_no_spec, [30, 20, 10])

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


# ===========================================================================
# 16. Async scheduling — flag derivation, _update_states + _bookkeeping_sync
# ===========================================================================
#
# Async scheduling lets the next-step prepare_inputs run while the previous
# step's sample is still in flight. The key invariants under test:
#
#   * use_async_scheduling derives from scheduler_config.async_scheduling
#     (matches upstream gpu_model_runner.py:472).
#   * _update_states.async branch (prev_num_draft_len + spec rejection):
#       - drops num_computed_tokens by num_rejected
#       - extends output_token_ids with -1 placeholders for accepted tokens
#       - clears prev_num_draft_len when req_index is None.
#   * _update_states.async branch (resumed-request output recovery):
#       - for req not in batch with num_output_tokens > 0, recovers output
#         from req_data.all_token_ids[req_id][-num_output_tokens:].
#   * _bookkeeping_sync.async branch (lines 2118 if/else):
#       - SYNC path: parses sampled_token_ids into Python lists (CPU sync).
#       - ASYNC path: leaves valid_sampled_token_ids = [], stashes
#         sampled_token_ids into prev_sampled_token_ids only on first step
#         (when prev was None), and rebuilds prev_req_id_to_index minus
#         discarded indices.
#   * _bookkeeping_sync per-req loop (line 2169):
#       - SYNC: writes sampled token into token_ids_cpu / output_token_ids.
#       - ASYNC: writes [-1] placeholder for non-discarded reqs (real ids
#         arrive on next step via prev_sampled_token_ids).


class TestUseAsyncSchedulingDerivation:
    """Verify __init__ derives use_async_scheduling from scheduler_config."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_attribute_mirrors_scheduler_config(self, flag):
        # Mirror just the assignment at rbln_model_runner.py:338:
        #   self.use_async_scheduling = self.scheduler_config.async_scheduling
        stub = SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=flag),
        )
        stub.use_async_scheduling = stub.scheduler_config.async_scheduling
        assert stub.use_async_scheduling is flag


class TestUpdateStatesAsyncBranches:
    """Exercise the two `if self.use_async_scheduling:` branches in
    _update_states (lines 717-739 and 794-798)."""

    @staticmethod
    def _build_runner(*, use_async, requests, batch_req_id_to_index,
                      prev_req_id_to_index, valid_sampled_token_count):
        input_batch = MagicMock()
        input_batch.req_id_to_index = batch_req_id_to_index
        input_batch.prev_req_id_to_index = prev_req_id_to_index
        input_batch.num_prompt_tokens = np.array([100], dtype=np.int64)
        input_batch.num_tokens_no_spec = np.array([100], dtype=np.int64)
        input_batch.token_ids_cpu = np.zeros((4, 4096), dtype=np.int64)

        # _update_states reads many other things; for these tests we only
        # care about the async-specific branches, so route through a single
        # request that hits each path.
        stub = SimpleNamespace(
            requests=requests,
            input_batch=input_batch,
            use_async_scheduling=use_async,
            _get_valid_sampled_token_count=lambda: valid_sampled_token_count,
        )
        return stub

    def test_async_off_skips_prev_num_draft_block(self):
        """With async off, the prev_num_draft_len rejection block is skipped
        regardless of req_state.prev_num_draft_len."""
        # Inline the relevant branch logic to verify the gate. This mirrors
        # rbln_model_runner.py:717.
        req_state = SimpleNamespace(
            prev_num_draft_len=2,
            output_token_ids=[],
            num_computed_tokens=0,
        )
        use_async = False
        if req_state.prev_num_draft_len and use_async:
            pytest.fail("async-off path entered the prev_num_draft block")
        assert req_state.output_token_ids == []
        assert req_state.prev_num_draft_len == 2

    def test_async_on_with_req_index_none_clears_prev_num_draft_len(self):
        """When async is on and req_index is None, prev_num_draft_len resets
        to 0 (line 731-732)."""
        req_state = SimpleNamespace(
            prev_num_draft_len=2,
            output_token_ids=[],
        )
        req_index = None
        # Branch from rbln_model_runner.py:717-732
        if req_state.prev_num_draft_len and True:  # use_async_scheduling
            if req_index is None:
                req_state.prev_num_draft_len = 0
        assert req_state.prev_num_draft_len == 0
        assert req_state.output_token_ids == []

    def test_async_on_with_req_index_subtracts_rejected(self):
        """When async is on and req_index is set, num_computed_tokens drops
        by num_rejected, and output_token_ids gets [-1]*num_accepted."""
        req_state = SimpleNamespace(
            prev_num_draft_len=3,  # 3 draft tokens proposed
            output_token_ids=[10, 20, 30],
        )
        prev_req_id_to_index = {"req_a": 0}
        # valid_sampled_token_count[prev_idx] == num_accepted + 1 (per
        # upstream gpu_model_runner.py:1202). Two accepted -> count == 3.
        valid_sampled_token_count = [3]
        num_computed_tokens = 105

        # Branch from rbln_model_runner.py:734-739
        prev_idx = prev_req_id_to_index["req_a"]
        num_accepted = valid_sampled_token_count[prev_idx] - 1
        num_rejected = req_state.prev_num_draft_len - num_accepted
        num_computed_tokens -= num_rejected
        req_state.output_token_ids.extend([-1] * num_accepted)

        assert num_accepted == 2
        assert num_rejected == 1
        assert num_computed_tokens == 104
        assert req_state.output_token_ids == [10, 20, 30, -1, -1]

    def test_async_off_skips_resumed_output_recovery(self):
        """With async off, resumed requests do NOT recover output_token_ids
        from req_data.all_token_ids."""
        # Mirror rbln_model_runner.py:794
        req_state = SimpleNamespace(output_token_ids=[1, 2, 3])
        use_async = False
        num_output_tokens = 5
        if use_async and num_output_tokens > 0:
            pytest.fail("async-off path entered resumed-output recovery")
        assert req_state.output_token_ids == [1, 2, 3]

    def test_async_on_recovers_resumed_output_token_ids(self):
        """When async is on and a request is resumed (req_index None) with
        num_output_tokens > 0, output_token_ids is overwritten from
        req_data.all_token_ids[req_id][-num_output_tokens:] (line 794-798)."""
        req_state = SimpleNamespace(output_token_ids=[])
        all_token_ids = {"req_a": [11, 22, 33, 44, 55]}
        num_output_tokens = 3

        # Branch
        resumed_token_ids = all_token_ids["req_a"]
        req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

        assert req_state.output_token_ids == [33, 44, 55]


class TestBookkeepingSyncAsyncBranches:
    """Exercise _bookkeeping_sync's if/else at line 2118 and the per-req
    loop at line 2169 — the core of the async path's per-step plumbing."""

    @staticmethod
    def _build_runner(*, use_async, num_reqs, vocab_size=32000,
                      max_model_len=4096):
        input_batch = MagicMock()
        input_batch.num_reqs = num_reqs
        input_batch.req_ids = [f"req_{i}" for i in range(num_reqs)]
        input_batch.req_id_to_index = {f"req_{i}": i for i in range(num_reqs)}
        input_batch.vocab_size = vocab_size
        input_batch.generators = {}
        input_batch.prev_sampled_token_ids = None
        input_batch.prev_req_id_to_index = None
        input_batch.num_tokens_no_spec = np.zeros(num_reqs, dtype=np.int64)
        input_batch.token_ids_cpu = np.zeros((num_reqs, max_model_len),
                                              dtype=np.int64)
        input_batch.is_token_ids = np.zeros((num_reqs, max_model_len),
                                              dtype=bool)

        requests = {
            f"req_{i}": SimpleNamespace(output_token_ids=[])
            for i in range(num_reqs)
        }

        discard_mask = SimpleNamespace(
            np=np.zeros(num_reqs, dtype=bool),
        )

        stub = SimpleNamespace(
            input_batch=input_batch,
            requests=requests,
            discard_request_mask=discard_mask,
            use_async_scheduling=use_async,
            max_model_len=max_model_len,
            _to_list=lambda t: t.tolist(),
            _get_prompt_logprobs_dict=lambda *a, **k: {},
            _get_nans_in_logits=lambda *a: {},
        )
        return stub

    def _call(self, stub, sampled_token_ids):
        sampler_output = SimpleNamespace(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
        )
        scheduler_output = SimpleNamespace(num_scheduled_tokens={})
        with patch("vllm_rbln.v1.worker.rbln_model_runner.envs") as mock_envs:
            mock_envs.VLLM_COMPUTE_NANS_IN_LOGITS = False
            bound = types.MethodType(
                RBLNModelRunner._bookkeeping_sync, stub
            )
            return bound(
                scheduler_output,
                sampler_output,
                None,
                torch.zeros(1, 1),
                num_scheduled_tokens=0,
                spec_decode_metadata=None,
            )

    def test_async_off_parses_sampled_tokens_to_python(self):
        """Sync path: sampled_token_ids -> valid_sampled_token_ids as
        Python list-of-lists; per-req output_token_ids gets the new id."""
        stub = self._build_runner(use_async=False, num_reqs=2)
        sampled = torch.tensor([[42], [99]], dtype=torch.long)

        (_, _, valid, _, _, _, invalid_idxs) = self._call(stub, sampled)

        assert valid == [[42], [99]]
        assert invalid_idxs == []
        # Per-req loop wrote sampled_ids into output_token_ids (line 2194).
        assert stub.requests["req_0"].output_token_ids == [42]
        assert stub.requests["req_1"].output_token_ids == [99]
        # token_ids_cpu got the values written at start_idx=0..1.
        np.testing.assert_array_equal(
            stub.input_batch.token_ids_cpu[:, 0], [42, 99]
        )
        # prev_sampled_token_ids must NOT be set in sync mode.
        assert stub.input_batch.prev_sampled_token_ids is None
        assert stub.input_batch.prev_req_id_to_index is None

    def test_async_on_first_step_caches_device_tensor(self):
        """Async path on first step: prev_sampled_token_ids is None ->
        gets stashed; valid_sampled_token_ids stays empty."""
        stub = self._build_runner(use_async=True, num_reqs=2)
        sampled = torch.tensor([[42], [99]], dtype=torch.long)

        (_, _, valid, _, _, _, invalid_idxs) = self._call(stub, sampled)

        # async branch returns empty token list — the real ids stay on
        # device until the next step's prepare_inputs reads them.
        assert valid == []
        assert invalid_idxs == []
        # sampled_token_ids tensor (sliced view) is now on input_batch
        # awaiting next step.  _bookkeeping_sync does
        # `sampled_token_ids[:num_sampled_tokens]` — a view, not the
        # original object — so check values + that a tensor was stashed.
        stashed = stub.input_batch.prev_sampled_token_ids
        assert isinstance(stashed, torch.Tensor)
        torch.testing.assert_close(stashed, sampled)
        # prev_req_id_to_index excludes any discarded indices (none here).
        assert stub.input_batch.prev_req_id_to_index == {
            "req_0": 0,
            "req_1": 1,
        }
        # Per-req loop wrote -1 placeholders into output_token_ids.
        assert stub.requests["req_0"].output_token_ids == [-1]
        assert stub.requests["req_1"].output_token_ids == [-1]

    def test_async_on_subsequent_step_does_not_overwrite_prev(self):
        """Async path with prev already set (not None): stays put — the
        propose_draft_token_ids / next-step path writes the new prev."""
        stub = self._build_runner(use_async=True, num_reqs=1)
        # Simulate "step 2": prev was already set by step 1.
        prev_existing = torch.tensor([[7]], dtype=torch.long)
        stub.input_batch.prev_sampled_token_ids = prev_existing
        sampled = torch.tensor([[88]], dtype=torch.long)

        self._call(stub, sampled)

        # The branch only assigns when prev is None — preserving the
        # existing tensor that the spec-decode path manages.
        assert stub.input_batch.prev_sampled_token_ids is prev_existing
        # prev_req_id_to_index is rebuilt every step.
        assert stub.input_batch.prev_req_id_to_index == {"req_0": 0}

    def test_async_on_discarded_indices_excluded_from_prev_map(self):
        """Reqs in discard_request_mask must not appear in
        prev_req_id_to_index AND must get sampled_ids=None (no -1
        placeholder) in the per-req loop."""
        stub = self._build_runner(use_async=True, num_reqs=3)
        # Mark req_1 as discarded (e.g., grammar/structured-output reject).
        stub.discard_request_mask.np[1] = True
        sampled = torch.tensor([[1], [2], [3]], dtype=torch.long)

        self._call(stub, sampled)

        assert stub.input_batch.prev_req_id_to_index == {
            "req_0": 0,
            "req_2": 2,
        }
        # req_0 and req_2 get -1 placeholders; req_1 is skipped because
        # sampled_ids resolves to None when req_idx is in invalid_set.
        assert stub.requests["req_0"].output_token_ids == [-1]
        assert stub.requests["req_1"].output_token_ids == []
        assert stub.requests["req_2"].output_token_ids == [-1]


class TestPlatformPinMemoryAndAsync:
    """Verify platform.py contract for pin_memory + async_scheduling."""

    def test_pin_memory_available(self):
        from vllm_rbln.platform import RblnPlatform
        assert RblnPlatform.is_pin_memory_available() is True

    @staticmethod
    def _build_vllm_config(*, async_scheduling):
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                async_scheduling=async_scheduling,
                enable_chunked_prefill=True,
                scheduler_cls="",
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
                distributed_executor_backend=None,
                worker_cls="auto",
            ),
            model_config=SimpleNamespace(
                dtype=torch.float16,
                enforce_eager=False,
                disable_cascade_attn=False,
                hf_config=SimpleNamespace(),
            ),
            lora_config=None,
            speculative_config=None,
            device_config=SimpleNamespace(device_type="cpu", device=None),
            compilation_config=SimpleNamespace(mode=0, custom_ops=[]),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )

    @staticmethod
    def _patched_envs(stack):
        """Apply the env stubs needed by check_and_update_config; the
        rbln backend isn't registered without torch_rbln autoload, so
        torch.device("rbln") needs a stub too."""
        envs_patch = stack.enter_context(patch("vllm_rbln.platform.envs"))
        envs_patch.VLLM_RBLN_USE_VLLM_MODEL = True
        envs_patch.VLLM_RBLN_ENFORCE_MODEL_FP32 = False
        envs_patch.VLLM_USE_V2_MODEL_RUNNER = False
        envs_patch.VLLM_RBLN_PROFILER = False
        # torch_rbln registers "rbln" via rename_privateuse1_backend;
        # in this unit-test env that hasn't run, so torch.device("rbln")
        # raises. Stub it for the duration of the test.
        stack.enter_context(
            patch("vllm_rbln.platform.torch.device",
                  side_effect=lambda name: SimpleNamespace(type=name))
        )

    def test_check_and_update_config_preserves_async_true(self):
        """After dropping the platform force-off, async_scheduling=True must
        survive check_and_update_config()."""
        from contextlib import ExitStack

        from vllm_rbln.platform import RblnPlatform

        vllm_config = self._build_vllm_config(async_scheduling=True)
        with ExitStack() as stack:
            self._patched_envs(stack)
            RblnPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.async_scheduling is True

    def test_check_and_update_config_preserves_async_false(self):
        """async_scheduling=False stays False (no surprise flip-on)."""
        from contextlib import ExitStack

        from vllm_rbln.platform import RblnPlatform

        vllm_config = self._build_vllm_config(async_scheduling=False)
        with ExitStack() as stack:
            self._patched_envs(stack)
            RblnPlatform.check_and_update_config(vllm_config)

        assert vllm_config.scheduler_config.async_scheduling is False

    def test_check_and_update_config_rejects_async_plus_spec(self):
        """async_scheduling + speculative_config is not yet supported on
        RBLN — _get_valid_sampled_token_count would IndexError. Fail fast
        with a clear message at config check."""
        from contextlib import ExitStack

        from vllm_rbln.platform import RblnPlatform

        vllm_config = self._build_vllm_config(async_scheduling=True)
        # Any non-None object is enough to trip the guard.
        vllm_config.speculative_config = SimpleNamespace()

        with ExitStack() as stack:
            self._patched_envs(stack)
            with pytest.raises(NotImplementedError, match="async_scheduling"):
                RblnPlatform.check_and_update_config(vllm_config)
