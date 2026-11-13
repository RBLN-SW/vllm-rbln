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

# Output handling and runner feedback share an immutable CPU token snapshot.

import json
import os
import subprocess
import sys
import textwrap
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput

import vllm_rbln.v1.worker.utils as worker_utils
from vllm_rbln.v1.sample.rbln_sampler import compile_sampler, rbln_greedy_sample
from vllm_rbln.v1.worker.async_output import AsyncRBLNModelRunnerOutput

pytestmark = pytest.mark.maybe_use_device


def _output(req_ids):
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=None,
    )


def _async_output(
    tokens: torch.Tensor,
    *,
    req_ids=None,
    placeholder_pos=None,
    invalid_req_indices=(),
    fail_fast=False,
    rank=None,
    dp_rank=None,
    device_input_ids=None,
):
    if req_ids is None:
        req_ids = [f"r{i}" for i in range(len(tokens))]
    async_out = AsyncRBLNModelRunnerOutput(
        model_runner_output=_output(req_ids),
        sampled_token_ids=tokens,
        invalid_req_indices=list(invalid_req_indices),
        req_ids=req_ids,
        placeholder_pos=(
            {req_id: 3 for req_id in req_ids}
            if placeholder_pos is None
            else placeholder_pos
        ),
        logprobs_tensors=None,
        fail_fast=fail_fast,
        rank=rank,
        dp_rank=dp_rank,
        device_input_ids=device_input_ids,
    )
    return async_out


class TestGetOutput:
    def test_retains_device_inputs_until_consumer_readback_finishes(self):
        input_ids = torch.tensor([[7]], dtype=torch.int32)
        input_ref = weakref.ref(input_ids)
        producer = _async_output(input_ids)
        consumer = _async_output(
            torch.tensor([[8]], dtype=torch.int32), device_input_ids=input_ids
        )
        completed = []

        def finish_readback():
            assert input_ref() is not None
            completed.append(True)

        consumer._copy_ready_event = SimpleNamespace(synchronize=finish_readback)
        del input_ids
        assert producer.get_sampled_token_ids_cpu().tolist() == [[7]]
        assert input_ref() is not None
        assert consumer.get_sampled_token_ids_cpu().tolist() == [[8]]
        assert completed == [True]
        assert input_ref() is None

    @pytest.mark.parametrize("failure_stage", ["copy", "logprobs"])
    @pytest.mark.parametrize(
        ("fail_fast", "disabled", "should_exit"),
        [
            (True, "0", True),
            (True, "1", False),
            (False, "0", False),
        ],
    )
    def test_output_thread_failure_uses_worker_exit_policy(
        self, monkeypatch, failure_stage, fail_fast, disabled, should_exit
    ):
        async_out = _async_output(
            torch.tensor([[7]], dtype=torch.int32), fail_fast=fail_fast
        )
        error = RuntimeError("device output failed")
        if failure_stage == "copy":
            monkeypatch.setattr(
                async_out, "get_sampled_token_ids_cpu", Mock(side_effect=error)
            )
        else:
            async_out._logprobs_tensors = SimpleNamespace(
                tolists=Mock(side_effect=error)
            )
        monkeypatch.setenv("VLLM_RBLN_DISABLE_WORKER_FAIL_FAST", disabled)
        exit_codes = []

        def fake_exit(code):
            exit_codes.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(worker_utils.os, "_exit", fake_exit)
        expected = SystemExit if should_exit else RuntimeError
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(expected) as excinfo,
        ):
            pool.submit(async_out.get_output).result(timeout=5)
        if should_exit:
            assert exit_codes == [70]
            assert excinfo.value.code == 70
        else:
            assert exit_codes == []
            assert excinfo.value is error

    def test_output_thread_failure_names_its_rank(self, monkeypatch, capfd):
        # The decorator's receiver here is the output, not the worker, so the
        # ranks must travel with the output for the event to name them.
        async_out = _async_output(
            torch.tensor([[7]], dtype=torch.int32), fail_fast=True, rank=2, dp_rank=1
        )
        monkeypatch.setattr(
            async_out,
            "get_sampled_token_ids_cpu",
            Mock(side_effect=RuntimeError("device output failed")),
        )
        monkeypatch.setenv("VLLM_RBLN_DISABLE_WORKER_FAIL_FAST", "0")
        monkeypatch.setattr(worker_utils.os, "_exit", Mock(side_effect=SystemExit(70)))
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(SystemExit):
            pool.submit(async_out.get_output).result(timeout=5)
        (event,) = [
            json.loads(line)
            for line in capfd.readouterr().err.splitlines()
            if line.startswith('{"event":"rbln.worker.fatal"')
        ]
        assert event["where"] == "AsyncRBLNModelRunnerOutput.get_output"
        assert (event["rank"], event["dp_rank"]) == (2, 1)

    def test_output_thread_failure_terminates_the_process(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""
                    from threading import Thread
                    from unittest.mock import Mock

                    import torch

                    from tests.vllm.v1.worker.test_async_output import _async_output

                    tokens = torch.tensor([[7]], dtype=torch.int32)
                    output = _async_output(tokens, fail_fast=True)
                    output.get_sampled_token_ids_cpu = Mock(
                        side_effect=RuntimeError("subprocess output copy failed")
                    )
                    thread = Thread(target=output.get_output, daemon=True)
                    thread.start()
                    thread.join(timeout=10)
                    raise SystemExit(1)
                """),
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(sys.path),
                "VLLM_RBLN_DISABLE_WORKER_FAIL_FAST": "0",
            },
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 70, result.stdout + result.stderr
        assert "subprocess output copy failed" in result.stdout + result.stderr

    def test_returns_the_sampled_tokens(self):
        async_out = _async_output(torch.tensor([[7], [8]], dtype=torch.int32))
        assert async_out.get_output().sampled_token_ids == [[7], [8]]

    @pytest.mark.use_device
    @pytest.mark.parametrize("first_reader", ["feedback", "output"])
    def test_device_tokens_are_copied_behind_the_sampler(self, first_reader):
        # The D2H is async only if the source lives on the device. A sampler
        # output does, hence the compiled sampler; a tensor fresh from .to("rbln")
        # is still host-resident.
        greedy = compile_sampler(rbln_greedy_sample, None)
        logits = torch.tensor([[0.0, 5.0, 1.0], [9.0, 0.0, 2.0]], dtype=torch.float16)
        src = greedy(logits.to("rbln")).unsqueeze(-1)
        async_out = _async_output(src)
        # A copy issued at construction must preserve the original tokens
        # even if the device source changes before output handling.
        src.copy_(torch.tensor([[5], [6]], dtype=src.dtype).to("rbln"))
        source_ref = weakref.ref(src)
        del src
        assert source_ref() is not None
        copy_ready_event = async_out._copy_ready_event
        if first_reader == "feedback":
            assert async_out.get_sampled_token_ids_cpu().tolist() == [[1], [0]]
        else:
            assert async_out.get_output().sampled_token_ids == [[1], [0]]
        assert source_ref() is None
        assert async_out._copy_ready_event is copy_ready_event
        assert async_out.get_output().sampled_token_ids == [[1], [0]]
        assert async_out.get_sampled_token_ids_cpu().tolist() == [[1], [0]]

    def test_clears_the_rows_the_sampler_discarded(self):
        async_out = _async_output(
            torch.tensor([[7], [8]], dtype=torch.int32), invalid_req_indices=[1]
        )
        assert async_out.get_output().sampled_token_ids == [[7], []]

    def test_feedback_can_read_the_snapshot_before_get_output(self):
        source = torch.tensor([[7], [8]], dtype=torch.int32)
        async_out = _async_output(source)
        source.fill_(99)
        snapshot = async_out.get_sampled_token_ids_cpu()
        assert snapshot.tolist() == [[7], [8]]
        assert async_out.get_output().sampled_token_ids == [[7], [8]]
        assert async_out.get_sampled_token_ids_cpu() is snapshot

    def test_snapshot_survives_an_in_place_trim_of_the_output(self):
        async_out = _async_output(torch.tensor([[7], [8]], dtype=torch.int32))
        output = async_out.get_output()
        for ids in output.sampled_token_ids:
            ids.clear()
        assert output.sampled_token_ids == [[], []]
        assert async_out.get_sampled_token_ids_cpu().tolist() == [[7], [8]]
