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

# The suite's own runners (currently AsyncVllmRunner only): rank pinning,
# concurrent submission, result ordering, the per-request timeout, shutdown --
# with the engine faked at AsyncLLM.from_engine_args. A fault here would not fail
# the DP e2e lane, it would make it answer wrongly.

from __future__ import annotations

import asyncio

import pytest

import tests.native.runners as runners
from tests.native.runners import AsyncVllmRunner, DPRequest

MODEL = "fake/model-for-args-only"


class _FakeCompletion:
    def __init__(self, token_ids: list[int], text: str) -> None:
        self.token_ids = token_ids
        self.text = text


class _FakeRequestOutput:
    def __init__(self, token_ids: list[int], text: str) -> None:
        self.outputs = [_FakeCompletion(token_ids, text)]


class _FakeEngine:
    """Stands in for AsyncLLM: records submissions, yields streams the test
    scripts. ``gate`` holds every request until all of them have arrived."""

    def __init__(self) -> None:
        self.submissions: list[dict] = []
        self.captured: dict = {}  # AsyncEngineArgs and the loop at build time
        self.shutdown_calls = 0
        self.gate: asyncio.Event | None = None
        self.expected_in_flight: int | None = None
        self.never_finish: set[str] = set()
        self.yield_nothing = False
        self.steps = 2

    async def _stream(self, request_id: str, max_tokens: int):
        if self.gate is not None:
            if len(self.submissions) == self.expected_in_flight:
                self.gate.set()
            await self.gate.wait()
        if request_id in self.never_finish:
            await asyncio.Event().wait()  # pending forever
        if self.yield_nothing:
            return
        # Later requests finish first, so ordering cannot follow completion.
        await asyncio.sleep(0.01 * (len(self.submissions) - self._index(request_id)))
        token_ids = list(range(max_tokens))
        for step in range(self.steps):
            yield _FakeRequestOutput(token_ids, f"{request_id}-step{step}")

    def _index(self, request_id: str) -> int:
        ids = [sub["request_id"] for sub in self.submissions]
        return ids.index(request_id)

    def generate(
        self,
        prompt,
        sampling_params,
        request_id: str,
        *,
        data_parallel_rank: int | None = None,
        **kwargs,
    ):
        self.submissions.append(
            dict(
                prompt=prompt,
                request_id=request_id,
                dp_rank=data_parallel_rank,
                temperature=sampling_params.temperature,
                max_tokens=sampling_params.max_tokens,
            )
        )
        return self._stream(request_id, sampling_params.max_tokens)

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def fake_engine(monkeypatch):
    """Replaces the AsyncLLM the runner builds with a scriptable stand-in."""
    engine = _FakeEngine()

    class _FakeAsyncLLM:
        @staticmethod
        def from_engine_args(args, *rest, **kwargs):
            engine.captured["args"] = args
            # Raises unless the engine really is built inside a loop.
            engine.captured["build_loop"] = asyncio.get_running_loop()
            return engine

    monkeypatch.setattr(runners, "AsyncLLM", _FakeAsyncLLM)
    return engine


def test_native_defaults_apply_and_kwargs_override(fake_engine):
    with AsyncVllmRunner(MODEL, data_parallel_size=4, block_size=512):
        pass
    args = fake_engine.captured["args"]
    assert (args.model, args.data_parallel_size) == (MODEL, 4)
    assert args.max_num_batched_tokens == 128
    assert args.enable_chunked_prefill is True
    assert args.block_size == 512


class TestSubmission:
    def test_requests_reach_their_pinned_rank(self, fake_engine):
        requests = [
            DPRequest("a", 4, dp_rank=0),
            DPRequest("b", 8, dp_rank=7),
            DPRequest("c", 2),  # unpinned: the balancer places it
        ]
        with AsyncVllmRunner(MODEL) as runner:
            runner.generate_greedy(requests)

        submitted = {sub["prompt"]: sub for sub in fake_engine.submissions}
        assert submitted["a"]["dp_rank"] == 0
        assert submitted["b"]["dp_rank"] == 7
        assert submitted["c"]["dp_rank"] is None

    def test_sampling_is_greedy_with_per_request_length(self, fake_engine):
        with AsyncVllmRunner(MODEL) as runner:
            runner.generate_greedy([DPRequest("a", 4, 0), DPRequest("b", 32, 1)])

        submitted = fake_engine.submissions
        assert [sub["temperature"] for sub in submitted] == [0.0, 0.0]
        assert {sub["prompt"]: sub["max_tokens"] for sub in submitted} == {
            "a": 4,
            "b": 32,
        }

    def test_requests_are_in_flight_together(self, fake_engine):
        # DP ranks only interact while several have work at once, so a runner
        # that awaited requests one by one would deadlock on the gate.
        fake_engine.gate = asyncio.Event()
        fake_engine.expected_in_flight = 3
        requests = [DPRequest(str(i), 2, dp_rank=i) for i in range(3)]

        with AsyncVllmRunner(MODEL, request_timeout_s=5.0) as runner:
            outputs = runner.generate_greedy(requests)

        assert len(outputs) == 3


class TestResults:
    def test_order_follows_the_request_list_not_completion(self, fake_engine):
        # The fake finishes the last request first (see _stream).
        requests = [DPRequest("first", 3, 0), DPRequest("second", 5, 1)]
        with AsyncVllmRunner(MODEL) as runner:
            outputs = runner.generate_greedy(requests)

        assert [ids for ids, _text in outputs] == [[0, 1, 2], [0, 1, 2, 3, 4]]

    def test_last_streamed_output_wins(self, fake_engine):
        # A streaming engine yields a growing output; the last one is the answer.
        fake_engine.steps = 3
        with AsyncVllmRunner(MODEL) as runner:
            ((_ids, text),) = runner.generate_greedy([DPRequest("a", 2, 0)])

        assert text.endswith("-step2")


class TestFailureModes:
    def test_timeout_names_the_rank(self, fake_engine):
        fake_engine.never_finish = {"dp-req-0"}
        with (
            AsyncVllmRunner(MODEL, request_timeout_s=0.05) as runner,
            pytest.raises(TimeoutError, match="dp_rank=3"),
        ):
            runner.generate_greedy([DPRequest("a", 2, dp_rank=3)])

    def test_stream_without_output_is_not_a_type_error(self, fake_engine):
        # A rank whose engine died can close the stream having produced nothing.
        fake_engine.yield_nothing = True
        with (
            AsyncVllmRunner(MODEL) as runner,
            pytest.raises(AssertionError, match="without producing any output"),
        ):
            runner.generate_greedy([DPRequest("a", 2, dp_rank=1)])


class TestLifecycle:
    def test_exit_shuts_the_engine_down_once(self, fake_engine):
        with AsyncVllmRunner(MODEL) as runner:
            runner.generate_greedy([DPRequest("a", 2, 0)])
        assert fake_engine.shutdown_calls == 1

    def test_a_failed_build_closes_the_loop(self, monkeypatch):
        # __init__ raising means __exit__ never runs, so the loop has to be closed
        # on the way out or a ResourceWarning lands next to the real failure.
        created: list = []
        real_new_event_loop = asyncio.new_event_loop

        def record_loop():
            loop = real_new_event_loop()
            created.append(loop)
            return loop

        monkeypatch.setattr(asyncio, "new_event_loop", record_loop)

        class _Exploding:
            @staticmethod
            def from_engine_args(args, *rest, **kwargs):
                raise RuntimeError("engine build failed")

        monkeypatch.setattr(runners, "AsyncLLM", _Exploding)

        with pytest.raises(RuntimeError, match="engine build failed"):
            AsyncVllmRunner(MODEL)

        assert created and created[0].is_closed()

    def test_exit_closes_the_loop(self, fake_engine):
        with AsyncVllmRunner(MODEL) as runner:
            loop = runner._loop
        assert loop.is_closed()

    def test_engine_is_built_on_the_runners_loop(self, fake_engine):
        # Its output handler must land on the loop later generate calls use.
        with AsyncVllmRunner(MODEL) as runner:
            assert fake_engine.captured["build_loop"] is runner._loop
