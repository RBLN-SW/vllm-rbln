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

# The speculative-decode halves of _prepare_inputs and _bookkeeping_sync: the
# runner builds the query the scheduler assumed, and reads the rejection
# sampler's output back.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import SamplerOutput

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.native.v1.worker.utils import (
    make_scheduler_output,
    make_speculative_config,
    schedule_new,
)

pytestmark = pytest.mark.maybe_use_device


def _decode_ready(
    runner,
    monkeypatch,
    *,
    num_spec_tokens: int,
    num_computed: int = 3,
    fixed_window: bool = True,
) -> None:
    """One request past its prompt in the decode phase, so the spec branch is
    reachable. The phase comes from the scheduler output, so it is set through
    _is_prefill_step rather than derived from input_batch."""
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    runner._update_states(schedule_new("a"))
    runner.input_batch.num_computed_tokens_cpu[0] = num_computed
    runner.input_batch.num_tokens_no_spec[0] = num_computed
    # Patched rather than configured: a real speculative_config would pull in a
    # drafter, and the only thing the arithmetic reads off it is whether the
    # drafter is model-based, which decides the fixed decode window.
    monkeypatch.setattr(runner, "num_spec_tokens", num_spec_tokens)
    monkeypatch.setattr(
        runner,
        "speculative_config",
        make_speculative_config("mtp" if fixed_window else "ngram"),
    )
    runner._is_prefill_step = False
    assert runner.is_prefill is False


class TestPrepareInputsSpecDecode:
    def test_pads_query_to_full_spec_behind_the_scheduled_tokens(
        self, make_model_runner, monkeypatch
    ):
        # 1 real + 1 draft = 2 logical tokens, but the decode query is fixed at
        # num_spec_tokens + 1 = 3, so one slot is padded. Away from the sequence's
        # end the padding goes behind, leaving the scheduled tokens at the front.
        runner = make_model_runner()
        _decode_ready(runner, monkeypatch, num_spec_tokens=2)

        logits_indices, spec_md, query_lengths, total = runner._prepare_inputs(
            make_scheduler_output(
                num_scheduled_tokens={"a": 2}, spec_decode_tokens={"a": [11]}
            ),
            np.array([2], dtype=np.int32),
        )

        assert query_lengths.tolist() == [3]
        assert total == 3
        # The scheduled tokens land in slots 0 and 1; slot 2 is the back padding.
        assert runner.positions[:3].tolist() == [3, 4, 5]
        # seq_lens follows the logical count (3 + 2), not the padded query
        # length, or attention would read a KV slot this step never wrote.
        assert runner.seq_lens[:1].tolist() == [5]

        assert spec_md is not None
        assert spec_md.num_draft_tokens == [1]
        assert logits_indices.tolist() == spec_md.logits_indices.tolist()

    def test_drafter_runs_the_logical_length(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        _decode_ready(runner, monkeypatch, num_spec_tokens=2, fixed_window=False)

        logits_indices, spec_md, query_lengths, total = runner._prepare_inputs(
            make_scheduler_output(num_scheduled_tokens={"a": 1}),
            np.array([1], dtype=np.int32),
        )

        assert spec_md is None
        assert query_lengths.tolist() == [1]
        assert total == 1
        assert runner.positions[:1].tolist() == [3]
        assert runner.seq_lens[:1].tolist() == [4]
        assert logits_indices.tolist() == [0]


class TestPrepareInputsFixedWindow:
    # A decode with a model-based drafter always stages num_spec_tokens + 1
    # slots and puts the slack behind the scheduled token, so everything
    # downstream has to follow the token, not the window.
    BLOCK = 1024
    NUM_SPEC = 2

    @pytest.mark.parametrize(
        "num_computed",
        [
            3,
            BLOCK - 1,
            BLOCK,
        ],
    )
    def test_the_window_is_fixed_and_starts_at_the_scheduled_token(
        self, make_model_runner, monkeypatch, num_computed
    ):
        runner = make_model_runner()
        _decode_ready(
            runner,
            monkeypatch,
            num_spec_tokens=self.NUM_SPEC,
            num_computed=num_computed,
        )
        window = self.NUM_SPEC + 1

        logits_indices, spec_md, query_lengths, total = runner._prepare_inputs(
            make_scheduler_output(num_scheduled_tokens={"a": 1}),
            np.array([1], dtype=np.int32),
        )

        assert query_lengths.tolist() == [window]
        positions = runner.positions[:window].tolist()
        assert positions == list(range(num_computed, num_computed + window))
        # The scheduled token is the first slot, whatever the window crosses.
        assert logits_indices.tolist() == [0]
        assert runner.seq_lens[:1].tolist() == [num_computed + 1]

    def test_a_window_past_max_model_len_moves_the_overshoot_in_front(
        self, make_model_runner, monkeypatch
    ):
        runner = make_model_runner()
        window = self.NUM_SPEC + 1
        num_computed = runner.max_model_len - window + 1
        _decode_ready(
            runner,
            monkeypatch,
            num_spec_tokens=self.NUM_SPEC,
            num_computed=num_computed,
        )

        logits_indices, _, query_lengths, _ = runner._prepare_inputs(
            make_scheduler_output(num_scheduled_tokens={"a": 1}),
            np.array([1], dtype=np.int32),
        )

        assert query_lengths.tolist() == [window]
        positions = runner.positions[:window].tolist()
        assert positions[-1] == runner.max_model_len - 1
        assert positions == [num_computed - 1, num_computed, num_computed + 1]
        assert logits_indices.tolist() == [1]
        assert runner.decode_back_pad_np[0] == 1

    def test_kept_drafts_at_a_block_start_sample_before_the_back_padding(
        self, make_model_runner, monkeypatch
    ):
        runner = make_model_runner()
        _decode_ready(
            runner, monkeypatch, num_spec_tokens=self.NUM_SPEC, num_computed=self.BLOCK
        )
        window = self.NUM_SPEC + 1

        logits_indices, spec_md, query_lengths, total = runner._prepare_inputs(
            make_scheduler_output(
                num_scheduled_tokens={"a": 2}, spec_decode_tokens={"a": [11]}
            ),
            np.array([2], dtype=np.int32),
        )

        assert query_lengths.tolist() == [window]
        assert total == window
        positions = runner.positions[:window].tolist()
        assert positions == [self.BLOCK, self.BLOCK + 1, self.BLOCK + 2]
        assert runner.decode_back_pad_np[0] == 1

        assert spec_md is not None
        assert spec_md.num_draft_tokens == [1]
        assert spec_md.logits_indices.tolist() == [0, 1]
        assert logits_indices.tolist() == [0, 1]
        assert runner.seq_lens[:1].tolist() == [self.BLOCK + 2]


class TestBookkeepingSyncSpecDecode:
    def test_parses_accepted_tokens_and_drops_placeholders(
        self, make_model_runner, monkeypatch
    ):
        # The rejection sampler pads rows with -1; only the accepted prefix may
        # be cached, since a surviving -1 would be emitted as a real token id.
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        # This class covers the synchronous bookkeeping path, and vLLM now
        # resolves an unset --async-scheduling to enabled, so pin it -- through
        # EngineArgs, so a rename of the runner attribute cannot silently put
        # this back on the async path.
        runner = make_model_runner(async_scheduling=False)
        runner._update_states(schedule_new("req_0", "req_1"))
        batch = runner.input_batch
        batch.num_tokens_no_spec[:2] = [3, 3]
        runner.discard_request_mask[:2] = False

        sampler_output = SamplerOutput(
            sampled_token_ids=torch.tensor(
                [[101, 102, -1, -1], [201, -1, -1, -1]],
                dtype=torch.int32,
                device=runner.device,
            ),
            logprobs_tensors=None,
        )
        hidden_states = torch.zeros(
            (6, runner.model_config.get_hidden_size()), dtype=runner.dtype
        )

        # The async path added invalid_req_indices to the tail of this tuple.
        _, _, valid_sampled_token_ids, *_ = runner._bookkeeping_sync(
            scheduler_output=make_scheduler_output(
                num_scheduled_tokens={"req_0": 3, "req_1": 3}
            ),
            sampler_output=sampler_output,
            logits=None,
            hidden_states=hidden_states,
            num_scheduled_tokens=6,
        )

        assert valid_sampled_token_ids == [[101, 102], [201]]

        # Accepted tokens are appended from the old cursor and it advances by
        # exactly the accepted count.
        assert batch.token_ids_cpu[0, 3:5].tolist() == [101, 102]
        assert batch.num_tokens_no_spec[0] == 5
        assert batch.token_ids_cpu[1, 3] == 201
        assert batch.num_tokens_no_spec[1] == 4

        assert runner.requests["req_0"].output_token_ids == [101, 102]
        assert runner.requests["req_1"].output_token_ids == [201]
