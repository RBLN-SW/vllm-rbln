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

"""Tests for RBLNEagleProposer methods (spec_decode/eagle.py), built through its
real constructor from upstream's eagle model pair with the compiled model left
unset; scenarios mirror upstream's tests/v1/spec_decode/test_eagle.py.

The draft->target id mapping (`_to_target_token_ids`, and the `logits_processor`
branch in the load_model wrapper) has no upstream counterpart -- upstream widens
the draft logits inside compute_logits, which this platform cannot compile.
Those tests pin the RBLN detour, so do not resolve a divergence from upstream by
deleting them."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

import vllm_rbln.v1.spec_decode.eagle as eagle_module
import vllm_rbln.v1.worker.dp_utils as dp_utils
from tests.native.v1.spec_decode.utils import make_cad, make_eagle_proposer

pytestmark = pytest.mark.maybe_use_device


def _neutralize(monkeypatch):
    """No-op the heavy forward-context / kv-cache plumbing so propose/dummy_run
    orchestration can run without a real runtime."""
    monkeypatch.setattr(
        eagle_module, "set_forward_context", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(eagle_module, "attach_kv_cache_bindings", lambda *a, **k: None)
    monkeypatch.setattr(
        eagle_module, "build_kv_cache_forward_context_kwargs", lambda *a, **k: {}
    )


def _wire_runner(proposer, *, num_reqs):
    proposer.runner = SimpleNamespace(
        # propose runs in the decode phase, which is the step phase it reads.
        is_prefill=False,
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        kv_caches=[],
        kv_cache_bases=[],
        kv_cache_view_infos=[],
        bucketing_manager=SimpleNamespace(find_decode_batch_bucket=lambda n: n),
    )
    proposer.draft_attn_groups = [
        SimpleNamespace(
            get_metadata_builder=lambda: SimpleNamespace(build=lambda **k: object()),
            layer_names=["draft.layer"],
        )
    ]


def _wire_dummy_runner(proposer):
    """Wire the runner/attention plumbing dummy_run touches and return the list
    the fake model appends to, so a test can count forward passes."""
    proposer.runner = SimpleNamespace(
        is_prefill=False,
        input_batch=SimpleNamespace(
            num_reqs=2,
            block_table=[
                SimpleNamespace(
                    get_cpu_tensor=lambda: torch.zeros((8, 4), dtype=torch.int32)
                )
            ],
        ),
        kv_caches=[],
        kv_cache_bases=[],
        kv_cache_view_infos=[],
        bucketing_manager=SimpleNamespace(find_decode_batch_bucket=lambda n: n),
        _get_cumsum_and_arange=lambda nt, cumsum_dtype=None: (
            np.cumsum(nt, dtype=cumsum_dtype),
            None,
        ),
    )
    proposer.draft_attn_groups = [
        SimpleNamespace(
            get_metadata_builder=lambda: SimpleNamespace(build=lambda **k: object()),
            layer_names=["draft.layer"],
        )
    ]
    calls = []

    def executable(**kwargs):
        calls.append(1)
        return kwargs["hidden_states"].reshape(-1, proposer.hidden_size), torch.zeros(
            (8, 128)
        )

    proposer.model_executable = executable
    return calls


def _fake_model_exec(argmax_tokens, hidden_size):
    # The load_model wrapper contract: (hidden [n, H], logits) with a
    # controllable per-row argmax, reshaped so the multi-step loop can feed it back.
    def executable(
        *, input_ids, positions, hidden_states, inputs_embeds, token_indices_to_sample
    ):
        logits = torch.full((8, 128), -10.0)
        for i, tok in enumerate(argmax_tokens):
            logits[i, tok] = 10.0
        # Filled on the host but returned on the device the real executable
        # returns, so the argmax and any d2t gather run where production runs them.
        return hidden_states.reshape(-1, hidden_size), logits.to(hidden_states.device)

    return executable


def _echo_model_exec(hidden_size):
    # argmax = first input id + 1, so consecutive columns are the observable
    # signature that the loop feeds each step's output forward.
    def executable(
        *, input_ids, positions, hidden_states, inputs_embeds, token_indices_to_sample
    ):
        rows = input_ids.reshape(input_ids.shape[0], -1)[:, 0].tolist()
        logits = torch.full((8, 128), -10.0)
        for i, val in enumerate(rows):
            logits[i, (int(val) + 1) % 128] = 10.0
        return hidden_states.reshape(-1, hidden_size), logits

    return executable


def _call_propose(proposer):
    return proposer.propose(
        target_token_ids=torch.arange(4, dtype=torch.int32),
        target_positions=torch.arange(4, dtype=torch.int64),
        target_hidden_states=torch.zeros(4, proposer.hidden_size),
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=make_cad([0, 2, 4], [10, 11]),
    )


class TestSetInputsFirstPass:
    def test_default_eagle_shifts_tokens_and_inserts_next(self):
        # Three requests, query_lens [3, 2, 4]. The target ids shift left by one
        # (drop the first) and each request's next token lands at its last slot.
        proposer = make_eagle_proposer()
        cad = make_cad([0, 3, 5, 9], [3, 5, 4])
        target_token_ids = torch.arange(11, 20, dtype=torch.int32)  # 9 tokens
        next_token_ids = torch.tensor([100, 200, 300], dtype=torch.int32)
        positions = torch.arange(9, dtype=torch.int64)
        hidden = torch.zeros(9, proposer.hidden_size)

        num_tokens, token_indices = proposer.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=positions,
            target_hidden_states=hidden,
            token_indices_to_sample=None,  # defaults to query_start_loc[1:] - 1
            cad=cad,
        )
        assert num_tokens == 9
        assert token_indices.cpu().tolist() == [2, 4, 8]
        # shifted target [12..19] with next tokens overwritten at [2, 4, 8].
        assert proposer.input_ids[:9].cpu().tolist() == [
            12,
            13,
            100,
            15,
            200,
            17,
            18,
            19,
            300,
        ]
        assert proposer.positions[:9].cpu().tolist() == list(range(9))

    def test_uses_explicit_token_indices_verbatim(self):
        # A given token_indices_to_sample is used as-is (not recomputed from
        # query_start_loc); the next tokens land at exactly those slots.
        proposer = make_eagle_proposer()
        num_tokens, token_indices = proposer.set_inputs_first_pass(
            target_token_ids=torch.arange(11, 20, dtype=torch.int32),
            next_token_ids=torch.tensor([100, 200, 300], dtype=torch.int32),
            target_positions=torch.arange(9, dtype=torch.int64),
            target_hidden_states=torch.zeros(9, proposer.hidden_size),
            token_indices_to_sample=torch.tensor([0, 3, 8], dtype=torch.int64),
            cad=make_cad([0, 3, 5, 9], [3, 5, 4]),
        )
        assert num_tokens == 9
        assert token_indices.cpu().tolist() == [0, 3, 8]
        assert proposer.input_ids[:9].cpu().tolist() == [
            100,
            13,
            14,
            200,
            16,
            17,
            18,
            19,
            300,
        ]

    def test_rejects_extra_input_slots(self):
        # Draft-model / parallel-drafting / dflash paths (needs_extra_input_slots)
        # are unsupported; this guard trips when one is later enabled.
        proposer = make_eagle_proposer()
        proposer.needs_extra_input_slots = True
        with pytest.raises(NotImplementedError):
            proposer.set_inputs_first_pass(
                target_token_ids=torch.arange(4, dtype=torch.int32),
                next_token_ids=torch.tensor([1], dtype=torch.int32),
                target_positions=torch.arange(4, dtype=torch.int64),
                target_hidden_states=torch.zeros(4, proposer.hidden_size),
                token_indices_to_sample=None,
                cad=make_cad([0, 4], [4]),
            )


class TestPreprocess:
    def test_prefill_views_full_buffers(self):
        # Prefill reshapes the whole (max_num_tokens,) buffer to [num_reqs, -1];
        # token indices pass through unpadded (num_reqs already the padded width).
        proposer = make_eagle_proposer()
        n = proposer.max_num_tokens
        proposer.input_ids[:] = torch.arange(n, dtype=torch.int32)
        token_indices = torch.tensor([0, 32, 64, 96], dtype=torch.int32)

        input_ids, positions, hidden, tip = proposer._preprocess(
            4, 4, n, token_indices, True
        )
        assert input_ids.shape == (4, n // 4)
        assert positions.shape == (4, n // 4)
        assert hidden.shape == (4, n // 4, proposer.hidden_size)
        assert tip.cpu().tolist() == [0, 32, 64, 96]

    def test_decode_slices_and_pads_to_bucket(self):
        # Decode slices the used tokens, reshapes to [num_reqs, -1], then pads
        # rows and token indices up to the padded batch (here 2 -> 4) with zeros.
        proposer = make_eagle_proposer()
        proposer.input_ids[:] = -1
        proposer.input_ids[:4] = torch.tensor([10, 11, 20, 21], dtype=torch.int32)

        input_ids, _, hidden, tip = proposer._preprocess(
            2, 4, 4, torch.tensor([1, 3], dtype=torch.int32), False
        )
        assert input_ids.cpu().tolist() == [[10, 11], [20, 21], [0, 0], [0, 0]]
        assert hidden.shape == (4, 2, proposer.hidden_size)
        assert tip.cpu().tolist() == [1, 3, 0, 0]


class TestPrepareInputsPadded:
    def test_builds_spec_metadata_and_delegates(self):
        # Delegates the rejected/index math to eagle_prepare_inputs_padded and
        # assembles the padded spec-decode CommonAttentionMetadata around it.
        proposer = make_eagle_proposer()
        cad = make_cad([0, 3, 6, 9], [8, 8, 8])
        spec_md = SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([2, 4, 6], dtype=torch.int32)
        )
        valid_count = torch.tensor([2, 3, 1], dtype=torch.int32)

        spec_cad, token_indices, num_rejected = proposer.prepare_inputs_padded(
            common_attn_metadata=cad,
            spec_decode_metadata=spec_md,
            valid_sampled_tokens_count=valid_count,
        )
        assert num_rejected.cpu().tolist() == [1, 0, 2]
        assert token_indices.cpu().tolist() == [1, 5, 6]
        assert spec_cad.num_actual_tokens == 9
        assert spec_cad.max_query_len == 3
        assert spec_cad.max_seq_len == 8
        assert spec_cad.causal is True


class TestPrepareNextTokenIdsPadded:
    def test_builds_backup_from_requests_and_delegates(self):
        # Backup tokens come from each request's state; the discarded request (1)
        # and the all-rejected request (2) fall back to their backup with count 0.
        proposer = make_eagle_proposer()
        cad = make_cad([0, 1, 2, 3], [5, 6, 7])
        requests = {
            f"r{i}": mock.MagicMock(**{"get_token_id.return_value": 100 + i})
            for i in range(3)
        }
        gpu_input_batch = SimpleNamespace(
            num_reqs=3, req_ids=["r0", "r1", "r2"], vocab_size=50
        )
        sampled = torch.tensor([[5, -1], [6, 7], [-1, -1]], dtype=torch.int32)
        discard = torch.tensor([False, True, False])

        next_ids, valid_count = proposer.prepare_next_token_ids_padded(
            common_attn_metadata=cad,
            sampled_token_ids=sampled,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_mask=discard,
        )
        assert next_ids.cpu().tolist() == [5, 101, 102]
        assert valid_count.cpu().tolist() == [1, 0, 0]


def _dp_aggregates(*, num_tokens, num_reqs, is_prefill=None):
    """What synchronize_draft_dp_ranks returns: the cross-rank aggregates the
    draft's own shape rule reads."""
    return (
        torch.tensor(num_tokens, dtype=torch.int32),
        any(is_prefill) if is_prefill is not None else False,
        max(num_reqs),
        max(t // r for t, r in zip(num_tokens, num_reqs)),
    )


class TestDetermineDraftBatchPadding:
    def test_dp1_prefill_keeps_num_reqs_decode_buckets(self):
        # With data_parallel_size == 1 the dp fields stay None; prefill keeps the
        # request count, decode rounds up to the runner's decode bucket.
        proposer = make_eagle_proposer()
        proposer.runner = SimpleNamespace(
            bucketing_manager=SimpleNamespace(find_decode_batch_bucket=lambda n: 8),
            specialized_moe_decode=False,
        )
        assert proposer._determine_draft_batch_padding(3, 10, True) == (3, None, None)
        assert proposer._determine_draft_batch_padding(3, 3, False) == (8, None, None)

    def test_dp_greater_than_one_specialized_moe(self, monkeypatch):
        # The per-DP counts drive the padding: batch bucket from max reqs across
        # DP, token pad from bucket * max tokens-per-req.
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(
            proposer.vllm_config.parallel_config, "data_parallel_size", 2
        )
        proposer.dp_rank = 0
        proposer.runner = SimpleNamespace(
            specialized_moe_decode=True,
            bucketing_manager=SimpleNamespace(
                find_decode_batch_bucket=lambda n: 8,
                decode_batch_buckets=[1, 2, 4, 8],
            ),
        )
        monkeypatch.setattr(
            dp_utils,
            "synchronize_draft_dp_ranks",
            lambda *a: _dp_aggregates(num_tokens=(16, 16), num_reqs=(4, 2)),
        )
        num_reqs_padded, num_tokens_padded, across = (
            proposer._determine_draft_batch_padding(3, 6, False)
        )
        # bucket(max(4, 2)) = 8; max(16//4, 16//2) = 8; tokens = 8 * 8 = 64.
        assert num_reqs_padded == 8
        assert num_tokens_padded == 64
        assert across.cpu().tolist() == [16, 16]

    def test_dp_greater_than_one_with_a_prefilling_rank(self, monkeypatch):
        # A rank in prefill sends the draft to the largest decode bucket and the
        # full token budget.
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(
            proposer.vllm_config.parallel_config, "data_parallel_size", 2
        )
        proposer.dp_rank = 0
        proposer.runner = SimpleNamespace(
            specialized_moe_decode=True,
            bucketing_manager=SimpleNamespace(
                find_decode_batch_bucket=lambda n: 8,
                decode_batch_buckets=[1, 2, 4, 8],
            ),
        )
        monkeypatch.setattr(
            dp_utils,
            "synchronize_draft_dp_ranks",
            lambda *a: _dp_aggregates(
                num_tokens=(16, 16), num_reqs=(4, 2), is_prefill=(False, True)
            ),
        )
        num_reqs_padded, num_tokens_padded, _ = proposer._determine_draft_batch_padding(
            3, 6, False
        )
        assert num_reqs_padded == 8  # decode_batch_buckets[-1]
        assert num_tokens_padded == proposer.max_num_tokens


class TestInitGuards:
    def test_rejects_multimodal_inputs(self, monkeypatch):
        # Multimodal targets are unsupported; the guard trips at construction.
        from vllm.multimodal import MULTIMODAL_REGISTRY

        monkeypatch.setattr(
            MULTIMODAL_REGISTRY, "supports_multimodal_inputs", lambda cfg: True
        )
        with pytest.raises(NotImplementedError):
            make_eagle_proposer()


class TestToTargetTokenIds:
    def test_applies_d2t_as_an_offset(self):
        # d2t holds offsets, not absolute target ids: id -> id + d2t[id]. Both
        # operands sit on the proposer's device, so this is the gather production
        # runs -- the reason the mapping moved out of the compiled graph.
        proposer = make_eagle_proposer()
        proposer.draft_id_to_target_id = torch.tensor(
            [0, 2, 3, 5, 8], dtype=torch.long, device=proposer.device
        )
        draft_ids = torch.tensor(
            [0, 1, 4, 2], dtype=torch.int64, device=proposer.device
        )

        out = proposer._to_target_token_ids(draft_ids)

        assert out.cpu().tolist() == [0, 3, 12, 5]

    def test_matches_full_vocab_scatter_then_argmax(self):
        # Upstream widens the draft logits into the target vocabulary and
        # argmaxes there. Dropping that scatter is only safe because
        # argmax-then-map picks the same id for a monotonic mapping.
        target_vocab_size = 20
        d2t = torch.tensor([0, 2, 3, 5, 8, 11], dtype=torch.long)
        target_ids = torch.arange(d2t.shape[0], dtype=torch.long) + d2t
        # The two premises of that equivalence; without them the reference below
        # would agree only by accident.
        assert bool((target_ids.diff() > 0).all())
        assert int(target_ids.max()) < target_vocab_size
        draft_logits = torch.tensor(
            [
                [0.0, 9.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 1.0, 0.0, 0.0, 7.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],
            ]
        )
        full_logits = torch.full(
            (draft_logits.shape[0], target_vocab_size), float("-inf")
        )
        full_logits[:, target_ids] = draft_logits
        proposer = make_eagle_proposer()
        proposer.draft_id_to_target_id = d2t.to(proposer.device)

        out = proposer._to_target_token_ids(
            draft_logits.argmax(dim=-1).to(proposer.device)
        )

        assert out.cpu().tolist() == full_logits.argmax(dim=-1).tolist()


class TestPropose:
    def test_single_step_early_exit_argmaxes_once(self, monkeypatch):
        # num_speculative_tokens == 1 takes the early-exit branch: one model pass,
        # per-request argmax, shaped [num_reqs, 1].
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle", num_speculative_tokens=1)
        _wire_runner(proposer, num_reqs=2)
        proposer.model_executable = _fake_model_exec([42, 60], proposer.hidden_size)

        out = _call_propose(proposer)
        assert out.shape == (2, 1)
        assert out.cpu().tolist() == [[42], [60]]

    def test_multi_step_feeds_previous_draft_forward(self, monkeypatch):
        # With the echo model each draft is the previous + 1, so consecutive
        # columns prove the loop feeds outputs forward (a broken one repeats).
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle", num_speculative_tokens=3)
        _wire_runner(proposer, num_reqs=2)
        proposer.model_executable = _echo_model_exec(proposer.hidden_size)

        out = proposer.propose(
            target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            target_positions=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            target_hidden_states=torch.zeros(4, proposer.hidden_size),
            next_token_ids=torch.tensor([50, 60], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
            common_attn_metadata=make_cad([0, 2, 4], [10, 11]),
        )
        assert out.shape == (2, 3)
        cols = out.cpu()
        assert torch.equal(cols[:, 1:], cols[:, :-1] + 1)

    def test_multi_step_handles_rejected_and_capped_positions(self, monkeypatch):
        # Drives the loop's seq_len adjustments (num_rejected_tokens, positions
        # hitting max_model_len). They feed a stubbed builder, so only the loop
        # still running and feeding drafts forward is observable here.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle", num_speculative_tokens=2)
        _wire_runner(proposer, num_reqs=1)
        proposer.max_model_len = 3  # positions hit the cap on the second step
        proposer.model_executable = _echo_model_exec(proposer.hidden_size)

        out = proposer.propose(
            target_token_ids=torch.tensor([7], dtype=torch.int32),
            target_positions=torch.tensor([2], dtype=torch.int64),
            target_hidden_states=torch.zeros(1, proposer.hidden_size),
            next_token_ids=torch.tensor([9], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([0], dtype=torch.int32),
            common_attn_metadata=make_cad([0, 1], [3]),
            num_rejected_tokens=torch.tensor([1], dtype=torch.int32),
        )
        assert out.shape == (1, 2)
        assert torch.equal(out.cpu()[:, 1:], out.cpu()[:, :-1] + 1)

    def test_eagle3_rejects_wrong_combined_hidden_size(self, monkeypatch):
        # eagle3 combines the target hidden states first and asserts the width;
        # a mismatch fails before any draft pass.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle3", num_speculative_tokens=1)
        _wire_runner(proposer, num_reqs=2)

        class _FakeEagle3(Eagle3LlamaForCausalLM):
            def __init__(self, combined):
                self.combined = combined

            def combine_hidden_states(self, target_hidden_states):
                return self.combined

        proposer.model = _FakeEagle3(torch.zeros((4, proposer.hidden_size + 1)))
        with pytest.raises(AssertionError):
            _call_propose(proposer)

    def test_single_step_maps_draft_ids_into_target_space(self, monkeypatch):
        # The early-exit branch maps too; an unmapped draft id names a different
        # token in the target vocabulary.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle", num_speculative_tokens=1)
        _wire_runner(proposer, num_reqs=2)
        proposer.model_executable = _fake_model_exec([42, 60], proposer.hidden_size)
        d2t = torch.zeros(128, dtype=torch.long)
        d2t[42], d2t[60] = 5, 7
        proposer.draft_id_to_target_id = d2t.to(proposer.device)

        out = _call_propose(proposer)

        assert out.cpu().tolist() == [[47], [67]]

    def test_multi_step_feeds_the_mapped_id_forward(self, monkeypatch):
        # Each draft is mapped before it becomes the next step's input: the draft
        # head's input embedding is in target space even when its output head is
        # not.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(method="eagle", num_speculative_tokens=2)
        _wire_runner(proposer, num_reqs=2)
        d2t = torch.zeros(128, dtype=torch.long)
        d2t[1], d2t[2], d2t[3], d2t[4] = 2, 3, 5, 8
        proposer.draft_id_to_target_id = d2t.to(proposer.device)
        argmax_per_call = [[1, 3], [2, 4]]
        seen: list[torch.Tensor] = []

        def executable(
            *,
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            token_indices_to_sample,
        ):
            logits = torch.full((8, 128), -10.0)
            for row, tok in enumerate(argmax_per_call[len(seen)]):
                logits[row, tok] = 10.0
            seen.append(input_ids.clone())
            return (
                hidden_states.reshape(-1, proposer.hidden_size),
                logits.to(hidden_states.device),
            )

        proposer.model_executable = executable

        out = proposer.propose(
            target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            target_positions=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
            target_hidden_states=torch.zeros(4, proposer.hidden_size),
            next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
            common_attn_metadata=make_cad([0, 2, 4], [10, 11]),
        )

        # step 1 argmax [1, 3] -> [3, 8], which is what step 2 must be fed.
        assert seen[1][:, 0].cpu().tolist() == [3, 8]
        # step 2 argmax [2, 4] -> [5, 12].
        assert out.cpu().tolist() == [[3, 5], [8, 12]]


class TestLoadModel:
    class _FakeMappedDraft(Eagle3LlamaForCausalLM):
        """An EAGLE3 draft head whose output stays in draft-vocab space: it
        carries a d2t mapping and a logits_processor. compute_logits only counts
        calls -- reaching its full-vocab scatter is the defect under test."""

        def __init__(self, d2t):
            self.draft_id_to_target_id = d2t
            self.lm_head = object()
            self.logits_processor_calls: list[tuple[object, torch.Tensor]] = []
            self.compute_logits_calls = 0

        def __call__(self, *, input_ids, positions, hidden_states, inputs_embeds):
            return hidden_states + 1000, hidden_states + 2000

        def logits_processor(self, lm_head, hidden_states):
            self.logits_processor_calls.append((lm_head, hidden_states.clone()))
            return hidden_states + 1

        def compute_logits(self, sample_hidden_states):
            self.compute_logits_calls += 1
            return sample_hidden_states + 1

    @staticmethod
    def _stub_super_load_model(monkeypatch, model=None):
        from vllm.v1.spec_decode.eagle import EagleProposer

        if model is None:
            model = SimpleNamespace(compute_logits=lambda h: h)
        monkeypatch.setattr(
            EagleProposer,
            "load_model",
            lambda self, target_model: setattr(self, "model", model),
        )

    def test_eager_wrapper_when_enforce_eager(self, monkeypatch):
        self._stub_super_load_model(monkeypatch)
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", True
        )
        proposer.load_model(target_model=object())
        # eager path leaves the executable as the uncompiled wrapper closure.
        assert proposer.model_executable.__name__ == "model_wrapper"

    def test_compiles_wrapper_when_not_eager(self, monkeypatch):
        self._stub_super_load_model(monkeypatch)
        sentinel = object()
        captured: dict = {}
        monkeypatch.setattr(
            eagle_module, "compile", lambda fn, **kw: captured.update(kw) or sentinel
        )
        monkeypatch.setattr(eagle_module, "build_process_group_dict", lambda: {})
        monkeypatch.setattr(eagle_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", False
        )
        proposer.runner = SimpleNamespace(compile_context=object())

        proposer.load_model(target_model=object())
        assert proposer.model_executable is sentinel
        assert captured["fullgraph"] is True
        assert "compile_context" in captured

    def test_eager_wrapper_composes_model_forward(self, monkeypatch):
        # The eager wrapper runs the draft model, reshapes the hidden states,
        # keeps the sampled positions, and returns (hidden, compute_logits).
        from vllm.v1.spec_decode.eagle import EagleProposer

        class _FakeModel:
            def __call__(self, *, input_ids, positions, hidden_states, inputs_embeds):
                return hidden_states  # a single tensor -> model_returns_tuple False

            def compute_logits(self, sample_hidden_states):
                return sample_hidden_states + 1

            def modules(self):
                return []  # dense draft: no fused-MoE layers

        monkeypatch.setattr(
            EagleProposer,
            "load_model",
            lambda self, target_model: setattr(self, "model", _FakeModel()),
        )
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(proposer, "model_returns_tuple", lambda: False)
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", True
        )
        proposer.load_model(target_model=object())

        h = proposer.hidden_size
        hidden = torch.arange(4, dtype=torch.float32).reshape(4, 1).repeat(1, h)
        out_hidden, out_logits = proposer.model_executable(
            input_ids=torch.zeros(4, dtype=torch.int32),
            positions=torch.zeros(4, dtype=torch.int64),
            hidden_states=hidden,
            token_indices_to_sample=torch.tensor([0, 2], dtype=torch.int64),
        )
        # rows 0 and 2 survive the index_select; compute_logits adds one.
        assert out_hidden.shape == (2, h)
        assert out_hidden[:, 0].cpu().tolist() == [0.0, 2.0]
        assert out_logits[:, 0].cpu().tolist() == [1.0, 3.0]

    def test_mapped_wrapper_skips_the_full_vocab_scatter(self, monkeypatch):
        # A mapped head must reach logits_processor instead: compute_logits
        # scatters into an -inf target-vocab row, and that index is
        # input-independent, so it folds into a constant whose out-of-bounds
        # write is the SIGSEGV in KV warm-up. What is checked here is the branch
        # selection -- only a real compile on the device reaches the SIGSEGV, so
        # this guards against someone routing back through compute_logits.
        d2t = torch.tensor([0, 2, 3], dtype=torch.long)
        model = self._FakeMappedDraft(d2t)
        self._stub_super_load_model(monkeypatch, model)
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        monkeypatch.setattr(
            proposer.vllm_config.speculative_config, "enforce_eager", True
        )
        proposer.load_model(target_model=object())

        # Taking that branch at all depends on load_model lifting d2t off the
        # draft model, so this covers the capture too.
        assert proposer.draft_id_to_target_id is d2t

        h = proposer.hidden_size
        hidden = torch.arange(2 * 3 * h, dtype=torch.float32).view(2, 3, h)
        _, logits = proposer.model_executable(
            input_ids=torch.zeros((2, 3), dtype=torch.int32),
            positions=torch.zeros((2, 3), dtype=torch.int64),
            hidden_states=hidden,
            token_indices_to_sample=None,
        )

        assert model.compute_logits_calls == 0
        assert len(model.logits_processor_calls) == 1
        lm_head_arg, hidden_arg = model.logits_processor_calls[0]
        assert lm_head_arg is model.lm_head
        # model_returns_tuple() is True for eagle, so the wrapper samples from
        # the first stream.
        expected_sample = (hidden + 1000).view(-1, h)
        assert torch.equal(hidden_arg, expected_sample)
        assert torch.equal(logits, expected_sample + 1)


class TestDummyRun:
    def test_runs_and_invokes_model(self, monkeypatch):
        # dummy_run is a warm-up: it drives the same orchestration and returns
        # nothing, invoking the model at least once.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        calls = _wire_dummy_runner(proposer)

        assert (
            proposer.dummy_run(num_reqs=2, num_tokens_per_req=4, is_prefill=True)
            is None
        )
        assert calls

    def test_multi_step_runs_second_loop(self, monkeypatch):
        # num_speculative_tokens > 1 warms up the extra draft loop too, so the
        # model runs once for the first pass plus once per extra step.
        _neutralize(monkeypatch)
        proposer = make_eagle_proposer(num_speculative_tokens=2)
        calls = _wire_dummy_runner(proposer)
        proposer.dummy_run(num_reqs=2, num_tokens_per_req=4, is_prefill=True)
        # first pass (1) + one extra draft step (num_speculative_tokens - 1 = 1).
        assert len(calls) == 2

    @staticmethod
    def _padded_tokens_per_pass(monkeypatch, proposer, **kwargs):
        seen: list[int | None] = []

        def record(*a, num_padded_tokens=None, **k):
            seen.append(num_padded_tokens)
            return nullcontext()

        monkeypatch.setattr(eagle_module, "set_forward_context", record)
        monkeypatch.setattr(
            eagle_module, "attach_kv_cache_bindings", lambda *a, **k: None
        )
        monkeypatch.setattr(
            eagle_module, "build_kv_cache_forward_context_kwargs", lambda *a, **k: {}
        )
        _wire_dummy_runner(proposer)
        proposer.dummy_run(num_reqs=2, num_tokens_per_req=4, is_prefill=True, **kwargs)
        return seen

    def test_warmup_override_pins_every_pass(self, monkeypatch):
        # Warm-up compiles for a caller-pinned token dimension, so the override
        # has to reach the extra draft step too -- not just the first pass.
        proposer = make_eagle_proposer(num_speculative_tokens=2)
        assert self._padded_tokens_per_pass(
            monkeypatch, proposer, num_padded_tokens=64
        ) == [64, 64]

    def test_without_override_falls_back_to_the_decided_padding(self, monkeypatch):
        # Single-DP decides no token padding, and no override must not turn that
        # None into a dimension of its own.
        proposer = make_eagle_proposer(num_speculative_tokens=2)
        assert self._padded_tokens_per_pass(monkeypatch, proposer) == [None, None]


class TestBuildDummyAttnMetadata:
    def test_computes_cumsum_and_shapes(self):
        proposer = make_eagle_proposer(num_speculative_tokens=1)
        proposer.runner = SimpleNamespace(
            is_prefill=False,
            input_batch=SimpleNamespace(
                block_table=[
                    SimpleNamespace(
                        get_cpu_tensor=lambda: torch.zeros((8, 4), dtype=torch.int32)
                    )
                ]
            ),
            _get_cumsum_and_arange=lambda nt, cumsum_dtype=None: (
                np.cumsum(nt, dtype=cumsum_dtype),
                None,
            ),
        )
        cad = proposer._build_dummy_attn_metadata(num_reqs=3, num_tokens_per_req=2)
        assert cad.query_start_loc.cpu().tolist() == [0, 2, 4, 6]
        assert cad.seq_lens.cpu().tolist() == [2, 2, 2]
        assert cad.num_actual_tokens == 6
        assert cad.max_query_len == 2
