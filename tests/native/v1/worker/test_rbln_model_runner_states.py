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

# _update_states on a real runner: the block table, the CPU token buffers and the
# pipeline-parallel branches, none of which a bare stub can reach.

from types import SimpleNamespace

import pytest

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.native.v1.worker.utils import (
    block_table_matches_state,
    is_in_batch,
    make_scheduler_output,
    schedule_cached,
    schedule_new,
)

pytestmark = pytest.mark.maybe_use_device


def _last_rank(monkeypatch, is_last: bool = True) -> None:
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=is_last)
    )


class TestBlockTableMaintenance:
    def test_appends_new_blocks_to_running_request(
        self, make_model_runner, monkeypatch
    ):
        # A growing running request gets its new blocks appended, and the block
        # table row must still agree with the cached state.
        _last_rank(monkeypatch)
        runner = make_model_runner()
        runner._update_states(schedule_new("a", block_ids=[([0],)]))
        # block_ids keeps the per-KV-cache-group tuple shape it arrived with.
        assert runner.requests["a"].block_ids == ([0],)

        runner._update_states(
            schedule_cached(
                req_ids=["a"],
                num_computed_tokens=[3],
                new_block_ids=[([1, 2],)],
            )
        )

        assert runner.requests["a"].block_ids == ([0, 1, 2],)
        assert block_table_matches_state(runner, "a")

    def test_resumed_from_preemption_replaces_block_table(
        self, make_model_runner, monkeypatch
    ):
        # Resuming re-allocates, so block ids are replaced; appending would leave
        # freed blocks in the table and read another request's KV.
        _last_rank(monkeypatch)
        runner = make_model_runner()
        runner._update_states(schedule_new("a", block_ids=[([0, 1],)]))
        assert runner.requests["a"].block_ids == ([0, 1],)

        # Drop it from the batch first: the resumed path asserts it is not
        # currently batched.
        runner._update_states(make_scheduler_output(num_scheduled_tokens={}))
        assert not is_in_batch(runner, "a")

        runner._update_states(
            schedule_cached(
                req_ids=["a"],
                num_computed_tokens=[3],
                new_block_ids=[([7, 8, 9],)],
                resumed_req_ids={"a"},
            )
        )

        assert runner.requests["a"].block_ids == ([7, 8, 9],)
        assert is_in_batch(runner, "a")
        assert block_table_matches_state(runner, "a")

    def test_condenses_batch_after_a_gap(self, make_model_runner, monkeypatch):
        # Finishing a middle request leaves a hole; rows must be compacted since
        # the forward pass indexes 0..num_reqs-1.
        _last_rank(monkeypatch)
        runner = make_model_runner()
        runner._update_states(schedule_new("a", "b", "c"))
        assert runner.input_batch.req_id_to_index == {"a": 0, "b": 1, "c": 2}

        runner._update_states(
            make_scheduler_output(
                num_scheduled_tokens={"a": 1, "c": 1}, finished_req_ids={"b"}
            )
        )

        assert "b" not in runner.requests
        assert set(runner.input_batch.req_id_to_index) == {"a", "c"}
        assert sorted(runner.input_batch.req_id_to_index.values()) == [0, 1]
        for req_id in ("a", "c"):
            assert block_table_matches_state(runner, req_id)


class TestOutputTokenAlignment:
    def test_discarded_output_tokens_are_truncated(
        self, make_model_runner, monkeypatch
    ):
        # A sync-KV-load failure reports fewer output tokens than the runner
        # cached; the extras must go or the next step samples an unwritten slot.
        _last_rank(monkeypatch)
        runner = make_model_runner()
        runner._update_states(schedule_new("a"))

        state = runner.requests["a"]
        req_index = runner.input_batch.req_id_to_index["a"]
        state.output_token_ids.extend([101, 102])
        runner.input_batch.num_tokens_no_spec[req_index] = (
            runner.input_batch.num_prompt_tokens[req_index] + 2
        )

        runner._update_states(
            schedule_cached(
                req_ids=["a"], num_computed_tokens=[3], num_output_tokens=[1]
            )
        )

        assert state.output_token_ids == [101]
        assert (
            runner.input_batch.num_tokens_no_spec[req_index]
            == runner.input_batch.num_prompt_tokens[req_index] + 1
        )


class TestPipelineParallel:
    def test_non_last_rank_folds_in_scheduler_supplied_tokens(
        self, make_model_runner, monkeypatch
    ):
        # A non-last rank never samples, so the scheduler ships the sampled token
        # back; it must land in both the request state and the CPU buffer.
        _last_rank(monkeypatch, is_last=False)
        runner = make_model_runner()
        runner._update_states(schedule_new("a"))
        req_index = runner.input_batch.req_id_to_index["a"]

        runner._update_states(
            schedule_cached(
                req_ids=["a"],
                num_computed_tokens=[3],
                new_token_ids=[[101]],
                num_output_tokens=[1],
            )
        )

        assert runner.requests["a"].output_token_ids == [101]
        assert runner.input_batch.token_ids_cpu[req_index, 3] == 101
        assert runner.input_batch.num_tokens_no_spec[req_index] == 4

    def test_last_rank_keeps_its_own_cached_tokens(
        self, make_model_runner, monkeypatch
    ):
        # The last rank samples itself and gets no token ids back, so it must
        # leave its buffer alone; the non-last path would rewind the cursor.
        _last_rank(monkeypatch, is_last=True)
        runner = make_model_runner()
        runner._update_states(schedule_new("a"))

        req_index = runner.input_batch.req_id_to_index["a"]
        runner.input_batch.token_ids_cpu[req_index, 3] = 101
        runner.input_batch.is_token_ids[req_index, 3] = True
        runner.input_batch.num_tokens_no_spec[req_index] = 4
        runner.requests["a"].output_token_ids.append(101)

        runner._update_states(
            schedule_cached(
                req_ids=["a"],
                num_computed_tokens=[3],
                new_token_ids=[[]],  # what the scheduler sends the last rank
                num_output_tokens=[1],
            )
        )

        assert runner.input_batch.num_tokens_no_spec[req_index] == 4
        assert runner.input_batch.token_ids_cpu[req_index, 3] == 101
        assert runner.requests["a"].output_token_ids == [101]


class TestSampleTokensWithoutPendingState:
    def test_surfaces_stashed_kv_connector_output(self, make_model_runner):
        # With no pending state sample_tokens must still forward the stashed
        # KV-connector output, else finished send/recv notices are lost.
        runner = make_model_runner()
        runner.execute_model_state = None
        sentinel = object()
        runner.kv_connector_output = sentinel

        output = runner.sample_tokens(grammar_output=None)

        assert output.kv_connector_output is sentinel
        # Consumed, so a second call does not re-report it.
        assert runner.kv_connector_output is None
        assert runner.sample_tokens(grammar_output=None).kv_connector_output is None
