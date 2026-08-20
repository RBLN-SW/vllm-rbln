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

# The scheduler side of P/D disaggregation: a request whose KV lives on a remote
# prefill engine waits, then rejoins as decode (full match) or as a remainder
# prefill (partial). Only reachable with a KV connector attached.

from vllm.v1.request import RequestStatus

from tests.native.v1.core.utils import (
    MockKVConfig,
    create_rbln_scheduler,
    make_model_runner_output,
    make_request,
)
from vllm_rbln.v1.core.utils import step_is_prefill

BLOCK_SIZE = 16
PROMPT_LEN = 2 * BLOCK_SIZE
# 99, not 100: one block of the pool is BlockPool's reserved null block.
MAX_LEN = BLOCK_SIZE * 99


def _scheduler(matched_tokens: int, *, is_async: bool = True, **kwargs):
    return create_rbln_scheduler(
        block_size=BLOCK_SIZE,
        num_blocks=100,
        max_num_batched_tokens=MAX_LEN,
        max_model_len=MAX_LEN,
        enable_prefix_caching=True,
        use_kv_connector=MockKVConfig(matched_tokens=matched_tokens, is_async=is_async),
        **kwargs,
    )


def _remote_request(req_id: str, num_tokens: int = PROMPT_LEN, *, base: int = 0):
    req = make_request(
        req_id, [base + i for i in range(num_tokens)], BLOCK_SIZE, max_tokens=8
    )
    req.kv_transfer_params = {"do_remote_prefill": True}
    return req


def _local_request(req_id: str, num_tokens: int = PROMPT_LEN, *, base: int = 500):
    return make_request(
        req_id, [base + i for i in range(num_tokens)], BLOCK_SIZE, max_tokens=8
    )


def _park_awaiting_remote_kv(sched, req):
    sched.add_request(req)
    out = sched.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return out


def _signal_kv_arrived(sched, out, *req_ids: str, token: int | None = None):
    sched.update_from_output(
        out, make_model_runner_output(out, token, finished_recving=set(req_ids))
    )


def _run_local_decode(sched, req):
    """Prefill a connector-less request and advance it into decode."""
    sched.add_request(req)
    out = sched.schedule()
    sched.update_from_output(out, make_model_runner_output(out, 0))
    return out


class TestAwaitingRemoteKV:
    def test_async_load_parks_the_request_without_scheduling_it(self):
        # An async remote match must not consume budget: no tokens, and the
        # request moves to the skipped queue so it is retried, not re-examined.
        sched = _scheduler(matched_tokens=BLOCK_SIZE)
        req = _remote_request("0")
        out = _park_awaiting_remote_kv(sched, req)

        assert out.num_scheduled_tokens == {}
        assert len(sched.waiting) == 0
        assert len(sched.skipped_waiting) == 1
        # Set even though the KV has not landed yet, so the eventual promotion
        # knows where to resume.
        assert req.num_computed_tokens == BLOCK_SIZE

    def test_promotion_waits_for_the_worker_signal(self):
        # Without finished_recving the request stays parked: promotion is driven
        # by the worker's connector output, not by time passing.
        sched = _scheduler(matched_tokens=BLOCK_SIZE)
        req = _remote_request("0")
        out = _park_awaiting_remote_kv(sched, req)

        sched.update_from_output(out, make_model_runner_output(out))
        out2 = sched.schedule()

        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert out2.num_scheduled_tokens == {}


class TestPartialMatchRemainder:
    def test_remainder_runs_alone_evicting_decodes_then_reaches_decode(self):
        # The remainder is a real prefill, so RBLN's no-mixed-batching rule
        # applies: the running decode is dropped from this step's output.
        sched = _scheduler(matched_tokens=BLOCK_SIZE)
        running = _local_request("R", num_tokens=BLOCK_SIZE)
        _run_local_decode(sched, running)

        req = _remote_request("0")
        sched.add_request(req)
        out = sched.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

        _signal_kv_arrived(sched, out, "0", token=0)
        out2 = sched.schedule()

        assert out2.num_scheduled_tokens["0"] == PROMPT_LEN - BLOCK_SIZE
        assert "R" not in out2.num_scheduled_tokens

        # Only after that remainder step does the request reach decode.
        sched.update_from_output(out2, make_model_runner_output(out2, 0))
        out3 = sched.schedule()
        assert out3.num_scheduled_tokens["0"] == 1


class TestCoexistenceWithDecodes:
    def test_full_match_does_not_evict_running_decodes(self):
        # A decode-ready promotion is not a prefill, so it joins the batch
        # instead of taking the lone-prefill path.
        sched = _scheduler(matched_tokens=PROMPT_LEN - 1)
        running = _local_request("R", num_tokens=BLOCK_SIZE)
        _run_local_decode(sched, running)

        req = _remote_request("0")
        sched.add_request(req)
        out = sched.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

        _signal_kv_arrived(sched, out, "0", token=0)
        out2 = sched.schedule()

        assert out2.num_scheduled_tokens["0"] == 1
        assert out2.num_scheduled_tokens["R"] == 1
        # The promoted request brought its whole prompt with it, so the step the
        # runner sees is a plain decode step.
        assert step_is_prefill(out2) is False

    def test_local_prefill_is_deferred_behind_a_promoted_request(self):
        # A promotion was admitted this step, so a local prefill must wait:
        # running it would mix it with decode reqs the eviction cannot reach.
        sched = _scheduler(matched_tokens=PROMPT_LEN - 1)
        remote = _remote_request("0")
        out = _park_awaiting_remote_kv(sched, remote)

        local = _local_request("L")
        sched.add_request(local)
        _signal_kv_arrived(sched, out, "0")
        out2 = sched.schedule()

        assert out2.num_scheduled_tokens["0"] == 1
        assert "L" not in out2.num_scheduled_tokens
        assert local.status == RequestStatus.WAITING
