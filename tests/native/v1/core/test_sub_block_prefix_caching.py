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

# RBLNScheduler's side of sub-block prefix caching: draining the manager's copy
# ops into RBLNSchedulerOutput and releasing them, plus arbitration against a KV
# connector. Matching itself is in test_rbln_kv_cache_manager.py.

from tests.native.v1.core.utils import (
    MockKVConfig,
    _drain,
    create_rbln_scheduler,
    make_model_runner_output,
    make_request,
)

BLOCK_SIZE = 16
SUB_BLOCK_SIZE = 4


class TestScheduleSubBlockCopyOps:
    # The scheduler's own role: draining the manager's copy ops and releasing
    # them.
    @staticmethod
    def _scheduler_with_indexed_block():
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            max_num_batched_tokens=128,
            num_blocks=10000,
        )
        # req0 prefills a full block; update runs do_pending_indexing so its
        # sub-blocks become matchable (req0 stays running).
        req0 = make_request("0", list(range(16)), 16, max_tokens=20)
        sched.add_request(req0)
        out0 = sched.schedule()
        sched.update_from_output(out0, make_model_runner_output(out0, 0))
        return sched

    def test_copy_ops_populated_on_match(self):
        # A request sharing req0's first sub-block yields a drained copy op.
        sched = self._scheduler_with_indexed_block()
        req1 = make_request("1", list(range(8)) + [100 + i for i in range(16)], 16)
        sched.add_request(req1)
        out = sched.schedule()
        assert len(out.kv_cache_copy_ops) == 1
        assert out.kv_cache_copy_ops[0].num_tokens == 8

    def test_update_from_output_releases_copy_ops(self):
        # The next update_from_output releases the copy ops' source refs.
        sched = self._scheduler_with_indexed_block()
        req1 = make_request("1", list(range(8)) + [100 + i for i in range(16)], 16)
        sched.add_request(req1)
        out = sched.schedule()
        src = sched.kv_cache_manager.block_pool.blocks[
            out.kv_cache_copy_ops[0].src_block_id
        ]
        ref_before = src.ref_cnt
        sched.update_from_output(out, make_model_runner_output(out, 0))
        assert src.ref_cnt == ref_before - 1

    def test_no_copy_ops_on_first_request(self):
        # A cold request has nothing to match -> empty copy ops.
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            max_num_batched_tokens=128,
            num_blocks=10000,
        )
        req = make_request("0", list(range(16)), 16)
        sched.add_request(req)
        out = sched.schedule()
        assert out.kv_cache_copy_ops == []


class TestSubBlockVersusKVConnector:
    # Whoever covers more wins, sub-block on ties since a local copy beats an
    # RDMA fetch. None of that code runs without a connector, hence the mock.
    MAX_LEN = BLOCK_SIZE * 100

    def _scheduler(self, matched_tokens: int):
        return create_rbln_scheduler(
            block_size=BLOCK_SIZE,
            num_blocks=100,
            max_num_batched_tokens=self.MAX_LEN,
            max_model_len=self.MAX_LEN,
            enable_prefix_caching=True,
            sub_block_size=SUB_BLOCK_SIZE,
            use_kv_connector=MockKVConfig(matched_tokens=matched_tokens),
        )

    @staticmethod
    def _cache_one_block(sched, tokens):
        req = make_request("0", tokens, BLOCK_SIZE, max_tokens=1)
        sched.add_request(req)
        out = sched.schedule()
        sched.update_from_output(out, make_model_runner_output(out, 0))

    @staticmethod
    def _schedule_query(sched, tokens, *, remote_prefill=False):
        req = make_request("1", tokens, BLOCK_SIZE, max_tokens=1)
        if remote_prefill:
            req.kv_transfer_params = {"do_remote_prefill": True}
        sched.add_request(req)
        return sched.schedule()

    def test_sub_block_wins_a_tie(self):
        # Equal coverage: the local copy is cheaper than an RDMA fetch, so the
        # sub-block match takes it and the connector's offer is cancelled.
        sched = self._scheduler(matched_tokens=SUB_BLOCK_SIZE)
        self._cache_one_block(sched, [0] * BLOCK_SIZE)

        tokens = [0] * SUB_BLOCK_SIZE + [900 + i for i in range(BLOCK_SIZE)]
        out = self._schedule_query(sched, tokens, remote_prefill=True)

        assert len(out.kv_cache_copy_ops) == 1
        assert out.kv_cache_copy_ops[0].num_tokens == SUB_BLOCK_SIZE
        assert out.num_scheduled_tokens["1"] == len(tokens) - SUB_BLOCK_SIZE

    def test_sub_block_wins_when_it_covers_more(self):
        # 3 sub-blocks (12 tokens) locally vs 4 tokens remotely: the connector's
        # offer is cancelled, so the request computes everything past the copy.
        num_shared = 3 * SUB_BLOCK_SIZE
        sched = self._scheduler(matched_tokens=SUB_BLOCK_SIZE)
        self._cache_one_block(
            sched,
            list(range(num_shared)) + [800 + i for i in range(BLOCK_SIZE - num_shared)],
        )

        tokens = list(range(num_shared)) + [900 + i for i in range(BLOCK_SIZE)]
        out = self._schedule_query(sched, tokens, remote_prefill=True)

        assert len(out.kv_cache_copy_ops) == 1
        assert out.kv_cache_copy_ops[0].num_tokens == num_shared
        assert out.num_scheduled_tokens["1"] == len(tokens) - num_shared

    def test_connector_wins_when_it_covers_more(self):
        # A whole block remotely vs one sub-block locally: the match is released
        # and no copy op is emitted.
        sched = self._scheduler(matched_tokens=BLOCK_SIZE)
        self._cache_one_block(sched, [0] * BLOCK_SIZE)

        tokens = [0] * SUB_BLOCK_SIZE + [900 + i for i in range(2 * BLOCK_SIZE)]
        out = self._schedule_query(sched, tokens, remote_prefill=True)

        assert out.kv_cache_copy_ops == []
        assert out.num_scheduled_tokens["1"] == len(tokens) - BLOCK_SIZE

    def test_connector_is_asked_with_a_block_aligned_count(self):
        # The connector is queried before the sub-block match, with the
        # full-block count only, or it would fetch from the wrong offset.
        sched = self._scheduler(matched_tokens=0)
        seen: list[int] = []
        original = sched.connector.get_num_new_matched_tokens

        def recording(request, num_computed_tokens):
            seen.append(num_computed_tokens)
            return original(request, num_computed_tokens)

        sched.connector.get_num_new_matched_tokens = recording

        self._cache_one_block(sched, [0] * BLOCK_SIZE)
        tokens = [0] * SUB_BLOCK_SIZE + [900 + i for i in range(BLOCK_SIZE)]
        out = self._schedule_query(sched, tokens, remote_prefill=True)

        # The sub-block match did happen, so the count could have been inflated.
        assert len(out.kv_cache_copy_ops) == 1
        assert seen[-1] == 0


class TestSubBlockPrefixHitRun:
    # A full run over the sub-block copy path: req1 shares req0's first sub-block
    # (yielding a copy op), and both requests still generate to completion.
    def test_shared_prefix_second_request_completes(self):
        sched = create_rbln_scheduler(
            enable_prefix_caching=True,
            block_size=16,
            sub_block_size=8,
            max_num_batched_tokens=128,
            num_blocks=10000,
        )
        req0 = make_request("0", list(range(16)), 16, max_tokens=3)
        sched.add_request(req0)
        out0 = sched.schedule()
        sched.update_from_output(out0, make_model_runner_output(out0, 0))

        # req1 shares req0's first 8-token sub-block, then diverges.
        req1_tokens = list(range(8)) + [100 + i for i in range(16)]
        req1 = make_request("1", req1_tokens, 16, max_tokens=3)
        sched.add_request(req1)
        out1 = sched.schedule()
        assert len(out1.kv_cache_copy_ops) == 1, "expected a sub-block copy op"
        sched.update_from_output(out1, make_model_runner_output(out1, 0))

        _drain(sched)
        assert req0.is_finished()
        assert req1.is_finished()
