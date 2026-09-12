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
from tests.native.v1.worker.utils import make_scheduler_output, schedule_new

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
        SimpleNamespace(method="mtp" if fixed_window else "ngram"),
    )
    runner._is_prefill_step = False
    assert runner.is_prefill is False


class TestPrepareInputsSpecDecode:
    def test_pads_query_to_full_spec_and_backfills_positions(
        self, make_model_runner, monkeypatch
    ):
        # 1 real + 1 draft = 2 logical tokens, but the decode query is fixed at
        # num_spec_tokens + 1 = 3, so one slot is padded. Mid-block the used tail
        # absorbs it, so the window re-runs the token before the scheduled ones.
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
        # Slot 0 re-runs position 2; the scheduled tokens land in slots 1 and 2.
        assert runner.positions[:3].tolist() == [2, 3, 4]
        # seq_lens follows the logical count (3 + 2), not the padded query
        # length, or attention would read a KV slot this step never wrote.
        assert runner.seq_lens[:1].tolist() == [5]

        assert spec_md is not None
        assert spec_md.num_draft_tokens == [1]
        assert logits_indices.tolist() == spec_md.logits_indices.tolist()

    def test_an_ngram_style_drafter_runs_the_logical_length(
        self, make_model_runner, monkeypatch
    ):
        # Without a model-based drafter the window is not fixed: an ngram-style
        # proposer misses often, and padding every miss out to the full window
        # would cost more than the extra compiled shape.
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
    # slots, spending the slack on tokens already computed in this block and
    # putting whatever the block's used tail cannot absorb behind the scheduled
    # token. Everything downstream has to follow the token, not the window.
    BLOCK = 1024
    NUM_SPEC = 2

    @pytest.mark.parametrize(
        "num_computed,window_start,sample_slot",
        [
            # Mid-block: the used tail holds the whole slack, so the window
            # re-runs the two tokens before the scheduled one.
            (3, 1, 2),
            # At a block start there is no tail to re-run, so the slack has
            # nowhere to go but behind the scheduled token.
            (BLOCK, BLOCK, 0),
        ],
    )
    def test_window_is_fixed_and_stays_in_one_block(
        self, make_model_runner, monkeypatch, num_computed, window_start, sample_slot
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
        assert positions == list(range(window_start, window_start + window))
        # The invariant the split exists for: one write range, one block.
        assert window_start // self.BLOCK == positions[-1] // self.BLOCK
        # The sampled slot is the scheduled token, wherever the padding put it.
        assert logits_indices.tolist() == [sample_slot]
        # seq_lens stays the logical length; padding must not inflate it.
        assert runner.seq_lens[:1].tolist() == [num_computed + 1]


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
