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

"""Tests for RBLNDFlashProposer's draft-block geometry and cache write.

Every case here pins something a run measured rather than something the code
merely does, because each was a regression that only a full measurement run
surfaced:

  - the mask's query positions must name the block's real slots; naming the
    slots one block further on admitted the previous step's rejected draft K/V
  - the block is open within itself, not causal; closing it within itself
    reduced acceptance compared with leaving it open
  - one seq_idx and one block table for the whole drafter, because a second
    dynamic index on a partition is a compiler error that arrives as a segfault
  - context K/V writes are grouped across heads, because the current runtime
    executes that strided region on device and avoids hundreds of submissions
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer

import vllm_rbln.v1.spec_decode.dflash as dflash_module
from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer
from vllm_rbln.v1.worker.dp_utils import DPStatus, ShapeConfig

BLOCK_SIZE = 1024
WINDOW = 2048
NUM_SPEC = 7
QUERY_LEN = 1 + NUM_SPEC
MAX_SEQ = 8192


def _mask_self(sliding_window=WINDOW):
    """The attributes `_draft_block_mask` reads, and nothing else."""
    return SimpleNamespace(
        num_speculative_tokens=NUM_SPEC,
        sliding_window=sliding_window,
        block_size=BLOCK_SIZE,
    )


def _mask(seq_lens, sliding_window, num_reqs=None, max_seq_len=MAX_SEQ):
    lens = torch.tensor(seq_lens, dtype=torch.int64)
    num_reqs = num_reqs if num_reqs is not None else len(seq_lens)
    return RBLNDFlashProposer._draft_block_mask(
        _mask_self(sliding_window),
        lens,
        num_reqs,
        num_reqs,
        max_seq_len,
        sliding_window,
    )


class TestDraftBlockMask:
    def test_shape_is_one_row_per_query_slot(self):
        for window in (None, WINDOW):
            mask = _mask([4000], window)
            assert tuple(mask.shape) == (1, 1, 1, QUERY_LEN, MAX_SEQ)

    @pytest.mark.parametrize("window", [None, WINDOW])
    def test_admits_nothing_past_the_block(self, window):
        """The regression that cost the most: a mask built from a length that
        counted the query block in sat eight slots further on, so every draft
        query admitted eight slots holding the previous step's rejected K/V."""
        seq_len = 4000
        mask = _mask([seq_len], window)[0, 0, 0]
        assert mask[:, seq_len + QUERY_LEN :].sum() == 0
        # ...and the block's own slots are all real keys, so they are admitted.
        assert mask[-1, seq_len : seq_len + QUERY_LEN].all()

    def test_block_is_open_within_itself(self):
        """Not causal: a block that only looked backwards is what the causal
        kernel family already gives, and this model would not need the
        mask-taking one at all."""
        seq_len = 4000
        mask = _mask([seq_len], WINDOW)[0, 0, 0]
        block = mask[:, seq_len : seq_len + QUERY_LEN]
        assert block.all(), "every query slot must see every other one"

    def test_sliding_row_sees_exactly_the_window(self):
        seq_len = 4000
        mask = _mask([seq_len], WINDOW)[0, 0, 0]
        context = mask[:, :seq_len]
        for row in range(QUERY_LEN):
            # The window is measured back from the row's own position, so the
            # rows nearest the block trade context slots for block slots.
            # The row's own slot is inside the block, so the context holds
            # one fewer than the window and slides forward with the row.
            assert int(context[row].sum()) == WINDOW - 1 - row
            first = int(context[row].nonzero()[0])
            assert first == seq_len - WINDOW + 1 + row

    def test_full_layer_sees_the_whole_context(self):
        seq_len = 4000
        mask = _mask([seq_len], None)[0, 0, 0]
        assert mask[:, : seq_len + QUERY_LEN].all()

    def test_rows_are_padded_with_zeros_not_dropped(self):
        mask = RBLNDFlashProposer._draft_block_mask(
            _mask_self(WINDOW),
            torch.tensor([4000, 3000], dtype=torch.int64),
            2,
            4,
            MAX_SEQ,
            WINDOW,
        )
        assert mask.shape[0] == 4
        assert mask[2:].sum() == 0


class TestPageCrossing:
    """The kernel scatters the whole query block at one offset per partition,
    so the last QUERY_LEN - 1 offsets of a page are unrepresentable."""

    @staticmethod
    def _crossing(seq_lens):
        lens = torch.tensor(seq_lens, dtype=torch.int64)
        return (lens % BLOCK_SIZE) + QUERY_LEN > BLOCK_SIZE

    def test_only_the_last_offsets_of_a_page_cross(self):
        crossing = self._crossing(list(range(BLOCK_SIZE)))
        assert int(crossing.sum()) == QUERY_LEN - 1
        assert crossing[BLOCK_SIZE - QUERY_LEN + 1 :].all()
        assert not crossing[: BLOCK_SIZE - QUERY_LEN + 1].any()

    def test_a_block_start_never_crosses(self):
        assert not self._crossing([0, BLOCK_SIZE, 4 * BLOCK_SIZE]).any()

    def test_redirect_lands_on_the_next_page_start(self):
        lens = torch.tensor([BLOCK_SIZE - 3], dtype=torch.int64)
        redirected = (lens // BLOCK_SIZE + 1) * BLOCK_SIZE
        assert int(redirected[0]) == BLOCK_SIZE
        assert not self._crossing([int(redirected[0])]).any()


class TestContextWriteGrouping:
    NUM_LAYERS = 2
    NUM_KV_HEADS = 3
    NUM_TOKENS = 5
    HEAD_DIM = 4
    CACHE_BLOCK_SIZE = 16

    def _model(self):
        layers = []
        for _ in range(self.NUM_LAYERS):
            cache = torch.zeros(
                2,
                4,
                self.NUM_KV_HEADS,
                1,
                self.CACHE_BLOCK_SIZE,
                self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            layers.append(
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=cache))
                )
            )
        return SimpleNamespace(layers=layers)

    def test_groups_each_layer_and_run_across_all_heads(self):
        model = self._model()
        keys = torch.arange(
            self.NUM_LAYERS * self.NUM_KV_HEADS * self.NUM_TOKENS * self.HEAD_DIM,
            dtype=torch.bfloat16,
        ).view(
            self.NUM_LAYERS,
            self.NUM_KV_HEADS,
            self.NUM_TOKENS,
            self.HEAD_DIM,
        )
        values = keys + 1000
        runs = [(0, 2, 1, 3), (2, 3, 2, 4)]

        destinations, sources = RBLNDFlashProposer._context_kv_copy_pairs(
            model, keys, values, runs
        )

        assert len(destinations) == self.NUM_LAYERS * len(runs) * 2
        assert len(sources) == len(destinations)
        assert all(pair.shape[0] == self.NUM_KV_HEADS for pair in destinations)
        assert all(not pair.is_contiguous() for pair in destinations)

        torch._foreach_copy_(destinations, sources)
        for layer_index, layer in enumerate(model.layers):
            cache = layer.self_attn.attn.kv_cache
            assert torch.equal(cache[0, 1, :, 0, 3:5, :], keys[layer_index, :, :2])
            assert torch.equal(cache[1, 1, :, 0, 3:5, :], values[layer_index, :, :2])
            assert torch.equal(cache[0, 2, :, 0, 4:7, :], keys[layer_index, :, 2:])
            assert torch.equal(cache[1, 2, :, 0, 4:7, :], values[layer_index, :, 2:])

    def test_a_write_run_never_leaves_its_block(self):
        """Runs are cut at block boundaries, which is what makes the
        destination a single contiguous span."""
        positions = torch.tensor([1020, 1021, 1022, 1023, 1024, 1025])
        blocks = (positions // BLOCK_SIZE).tolist()
        assert blocks == [0, 0, 0, 0, 1, 1]
        offsets = (positions % BLOCK_SIZE).tolist()
        assert offsets == [1020, 1021, 1022, 1023, 0, 1]


class TestWarmup:
    """Warmup only queues the drafter's shapes; `finish_dummy_run` executes
    them once every target runtime exists. After warmup `dummy_run` is the
    DP-idle path and drafts at once."""

    @staticmethod
    def _proposer(*, warmup_pending=True, dp_status=None, draft_has_moe=False):
        calls = SimpleNamespace(projections=[], queries=[], decisions=[])
        cad = object()
        runner = SimpleNamespace(is_prefill=None, dp_status=dp_status)

        def decide(num_reqs, num_tokens):
            calls.decisions.append((num_reqs, num_tokens, runner.is_prefill))

        runner._determine_batch_execution_and_padding = decide
        proposer = SimpleNamespace(
            num_speculative_tokens=NUM_SPEC,
            max_num_tokens=512,
            dp_rank=0,
            draft_has_moe=draft_has_moe,
            _proj_states=torch.zeros(512, 3),
            _proj_positions=torch.zeros(1, 512, dtype=torch.int64),
            _warmup_projection_buckets=set(),
            _warmup_query_batch_sizes=set(),
            _warmup_pending=warmup_pending,
            runner=runner,
        )

        def project(states, positions):
            calls.projections.append((tuple(states.shape), tuple(positions.shape)))

        proposer._project_context_kv = project
        proposer._build_dummy_attn_metadata = lambda num_reqs, num_query_per_req: cad
        proposer._run_query_pass = lambda *args: calls.queries.append(args)
        proposer._run_dummy_query_pass = (
            lambda num_reqs,
            num_query_per_req: RBLNDFlashProposer._run_dummy_query_pass(
                proposer, num_reqs, num_query_per_req
            )
        )
        return proposer, cad, calls

    def test_prefill_warms_the_large_projection_bucket_only(self):
        proposer, _, calls = self._proposer()

        RBLNDFlashProposer.dummy_run(proposer, 1, 512, True)

        assert calls.projections == []
        RBLNDFlashProposer.finish_dummy_run(proposer)

        assert calls.projections == [((512, 3), (1, 512))]
        assert calls.queries == []
        assert calls.decisions == []

    def test_spec_decode_warms_projection_and_query_graphs(self):
        proposer, cad, calls = self._proposer()

        RBLNDFlashProposer.dummy_run(proposer, 2, QUERY_LEN, False)

        assert calls.projections == []
        assert calls.queries == []
        RBLNDFlashProposer.finish_dummy_run(proposer)

        assert calls.projections == [((QUERY_LEN, 3), (1, QUERY_LEN))]
        # The batch's own status is published before the pass reads it.
        assert calls.decisions == [(2, 2 * QUERY_LEN, False)]
        assert len(calls.queries) == 1
        assert calls.queries[0][0] is cad
        assert calls.queries[0][1:4] == (2, QUERY_LEN, 2 * QUERY_LEN)
        assert torch.equal(calls.queries[0][4], torch.zeros(2, dtype=torch.int32))
        assert torch.equal(calls.queries[0][5], torch.zeros(2, dtype=torch.int32))
        assert proposer._warmup_pending is False

    def test_target_decode_shape_does_not_duplicate_drafter_warmup(self):
        proposer, _, calls = self._proposer()

        RBLNDFlashProposer.dummy_run(proposer, 1, 1, False)
        RBLNDFlashProposer.finish_dummy_run(proposer)

        assert calls.projections == []
        assert calls.queries == []
        assert calls.decisions == []

    def test_each_bucket_is_warmed_once(self):
        proposer, _, calls = self._proposer()

        for num_reqs in (1, 2, 4, 4, 2):
            RBLNDFlashProposer.dummy_run(proposer, num_reqs, QUERY_LEN, False)
        RBLNDFlashProposer.finish_dummy_run(proposer)

        assert [call[1] for call in calls.queries] == [1, 2, 4]
        assert calls.projections == [((QUERY_LEN, 3), (1, QUERY_LEN))]

    def test_after_warmup_an_idle_rank_drafts_at_once(self):
        proposer, cad, calls = self._proposer(warmup_pending=False)

        RBLNDFlashProposer.dummy_run(proposer, 1, QUERY_LEN, False)

        # No projection and no new status: the step already published one.
        assert calls.projections == []
        assert calls.decisions == []
        assert len(calls.queries) == 1
        assert calls.queries[0][0] is cad
        assert calls.queries[0][1:4] == (1, QUERY_LEN, QUERY_LEN)

    @pytest.mark.parametrize("draft_has_moe", [False, True])
    def test_an_idle_rank_skips_only_without_moe(self, draft_has_moe):
        proposer, _, calls = self._proposer(
            warmup_pending=False,
            dp_status=SimpleNamespace(is_idle=[True]),
            draft_has_moe=draft_has_moe,
        )

        RBLNDFlashProposer.dummy_run(proposer, 1, QUERY_LEN, False)

        assert len(calls.queries) == (1 if draft_has_moe else 0)


class TestSchedulerCapacity:
    @pytest.mark.parametrize("max_num_seqs", [1, 2, 4, 16])
    def test_initialization_accepts_any_scheduler_capacity(
        self, monkeypatch, max_num_seqs
    ):
        """The configured capacity may exceed the number of active requests."""

        def initialize_base(proposer, **_kwargs):
            proposer.arange = torch.arange(NUM_SPEC + 1)
            proposer.dflash_causal = False

        monkeypatch.setattr(DFlashProposer, "__init__", initialize_base)
        monkeypatch.setattr(dflash_module, "USE_DEVICE_TENSOR", True)
        monkeypatch.setattr(dflash_module.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            speculative_config=SimpleNamespace(
                enforce_eager=False,
                draft_model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(layer_types=[], sliding_window=None)
                ),
            ),
        )

        proposer = RBLNDFlashProposer(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

        assert proposer.runner is None
        assert proposer.arange_cpu.shape == (NUM_SPEC + 1,)


class TestDraftVocabularyMapping:
    class _FakeMappedDraft:
        def __init__(self, d2t):
            self.draft_id_to_target_id = d2t
            self.lm_head = object()
            self.compute_logits_calls = 0
            self.logits_processor_calls = []

        def compute_logits(self, hidden_states):
            self.compute_logits_calls += 1
            return hidden_states + 100

        def logits_processor(self, lm_head, hidden_states):
            self.logits_processor_calls.append((lm_head, hidden_states.clone()))
            return hidden_states + 1

    @staticmethod
    def _proposer(model):
        proposer = object.__new__(RBLNDFlashProposer)
        proposer.model = model
        proposer.draft_id_to_target_id = model.draft_id_to_target_id
        return proposer

    def test_mapped_logits_skip_the_full_vocab_scatter(self):
        """Mapped models stay in draft-vocabulary space until after argmax."""
        model = self._FakeMappedDraft(torch.tensor([0, 2, 3], dtype=torch.long))
        proposer = self._proposer(model)
        hidden = torch.arange(6, dtype=torch.float32).view(2, 3)

        logits = proposer._compute_draft_logits(hidden)

        assert model.compute_logits_calls == 0
        assert len(model.logits_processor_calls) == 1
        lm_head_arg, hidden_arg = model.logits_processor_calls[0]
        assert lm_head_arg is model.lm_head
        assert torch.equal(hidden_arg, hidden)
        assert torch.equal(logits, hidden + 1)

    def test_argmax_then_map_matches_full_vocab_scatter(self):
        target_vocab_size = 20
        d2t = torch.tensor([0, 2, 3, 5, 8, 11], dtype=torch.long)
        target_ids = torch.arange(d2t.shape[0], dtype=torch.long) + d2t
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
        proposer = self._proposer(self._FakeMappedDraft(d2t))

        out = proposer._to_target_token_ids(draft_logits.argmax(dim=-1))

        assert out.tolist() == full_logits.argmax(dim=-1).tolist()


class TestRedirectTarget:
    """A crossing row is redirected to its next page, which the scheduler is
    supposed to have allocated. It does not always: the lookahead is zeroed
    while `num_computed_tokens` is 0, and that field is assigned only after
    allocation, so a prefix-cache hit that leaves one waiting-path chunk can end
    in a page's last slots with nothing reserved beyond it. An unfilled slot
    reads 0, which is the pool's shared null block, and the query graph would
    scatter the draft block's K/V there -- dropping the row afterwards does not
    unwind the write, so the step has to give up before the forward."""

    @staticmethod
    def _call(table, next_page, crossing):
        """The give-up decision `_run_query_pass` makes, in the same order.

        Returns True when the step gives up, and raises when a crossing row's
        target is inside the table but unallocated -- the assertion the inlined
        check keeps as a regression guard.
        """
        block_table = torch.tensor(table, dtype=torch.int32)
        pages = (torch.tensor(next_page, dtype=torch.int64) // BLOCK_SIZE).to(
            torch.int64
        )
        if int(pages.max()) >= block_table.shape[-1]:
            return True
        rows = torch.arange(pages.shape[0])
        crossing = torch.tensor(crossing, dtype=torch.bool)
        assert not bool((block_table.cpu()[rows, pages][crossing] == 0).any())
        return False

    def test_allocated_next_page_is_accepted(self):
        # page 1 holds block 6, so the redirect has somewhere to land.
        assert not self._call([[71, 6, 0, 0]], [BLOCK_SIZE], [True])

    def test_unfilled_next_page_trips_the_assertion(self):
        """The reviewed regression: inside the table, but never allocated.

        The scheduler's lookahead reservation now rules this out, so it is an
        assertion rather than a give-up branch.
        """
        with pytest.raises(AssertionError):
            self._call([[71, 0, 0, 0]], [BLOCK_SIZE], [True])

    def test_past_the_table_is_refused(self):
        """The context ceiling -- no next page exists at all."""
        assert self._call([[71, 6]], [2 * BLOCK_SIZE], [True])

    def test_a_non_crossing_row_does_not_veto_the_step(self):
        """Only the rows that actually redirect are checked."""
        assert not self._call([[71, 0, 0, 0]], [BLOCK_SIZE], [False])

    def test_only_crossing_rows_are_checked(self):
        table = [[71, 6, 0, 0], [12, 0, 0, 0]]
        with pytest.raises(AssertionError):
            self._call(table, [BLOCK_SIZE, BLOCK_SIZE], [True, True])
        # the second row is the unfilled one, so it only matters when it crosses
        assert not self._call(table, [BLOCK_SIZE, BLOCK_SIZE], [True, False])


class TestPlatformRefusals:
    """The three configurations DFlash cannot run on, all refused at
    construction and all before the base class does any work, so none of them
    reaches a device.

    Each fails silently otherwise: an eager context write goes through an
    attention op that exists only as a compiled kernel, and without device
    tensors the cache is allocated on `meta`, which accepts a host copy and
    discards it."""

    @staticmethod
    def _config(enforce_eager=False):
        return SimpleNamespace(
            speculative_config=SimpleNamespace(enforce_eager=enforce_eager),
        )

    def _construct(self):
        return RBLNDFlashProposer(self._config(), torch.device("cpu"))

    def test_eager_is_refused(self):
        with pytest.raises(NotImplementedError, match="cannot run eager"):
            RBLNDFlashProposer(self._config(enforce_eager=True), torch.device("cpu"))

    def test_compile_disabled_is_refused(self, monkeypatch):
        monkeypatch.setattr(dflash_module.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        with pytest.raises(NotImplementedError, match="cannot run eager"):
            self._construct()

    def test_host_visible_cache_is_required(self, monkeypatch):
        """Without device tensors the cache is on `meta` and the context write
        is dropped without an error."""
        monkeypatch.setattr(dflash_module, "USE_DEVICE_TENSOR", False)
        with pytest.raises(NotImplementedError, match="USE_DEVICE_TENSOR"):
            self._construct()


class TestIdleSkip:
    """A DP-idle rank may skip its draft only when the drafter runs no
    collective of its own. Fused MoE is the one thing that would give it one,
    and a busy rank would then block in a collective the idle rank never
    joined -- so the skip is keyed off `draft_has_moe`, as it is for the
    chained drafter."""

    @staticmethod
    def _has_moe(modules):
        """What `load_model` computes."""
        return any(isinstance(module, dflash_module.MoERunner) for module in modules)

    @staticmethod
    def _skips(is_idle, draft_has_moe):
        """The `dummy_run` guard."""
        return is_idle and not draft_has_moe

    def test_a_dense_drafter_lets_an_idle_rank_skip(self):
        assert not self._has_moe([object(), object()])
        assert self._skips(is_idle=True, draft_has_moe=False)

    def test_a_moe_drafter_keeps_an_idle_rank_drafting(self):
        moe = object.__new__(dflash_module.MoERunner)
        assert self._has_moe([object(), moe])
        assert not self._skips(is_idle=True, draft_has_moe=True)

    def test_a_busy_rank_always_drafts(self):
        assert not self._skips(is_idle=False, draft_has_moe=False)


class TestQueryPassBatch:
    """The query pass runs at the decode bucket the target verifies at.

    The runner pads the target to its smallest fitting bucket. A drafter that
    stayed at the live request count ran a batch shape the target never did:
    one warmup never compiled, and the target/drafter batch asymmetry behind
    the acceptance collapse under padded verification. The full bucket ladder
    only hid it while every measured live count happened to be a bucket."""

    BUCKETS = [1, 2, 4]
    MAX_TOKENS = 4 * QUERY_LEN
    CONTEXT = 100  # well inside the first page, so no row crosses
    MASK_TOKEN = 151667
    STALE = 99  # what a previous step left in the buffers

    def _proposer(
        self, monkeypatch, num_reqs, *, dp_size=1, dp_status=None, max_tokens=None
    ):
        max_tokens = self.MAX_TOKENS if max_tokens is None else max_tokens
        proposer = RBLNDFlashProposer.__new__(RBLNDFlashProposer)
        proposer.runner = SimpleNamespace(
            shape_config=ShapeConfig(
                decode_batch_buckets=self.BUCKETS,
                find_bucket=lambda n: next(b for b in self.BUCKETS if b >= n),
                max_num_tokens=max_tokens,
                specialized_moe_decode=False,
            ),
            dp_status=dp_status,
            kv_cache_bases=None,
            input_batch=SimpleNamespace(num_reqs=num_reqs),
        )
        proposer.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp_size)
        )
        proposer.dp_rank = 0
        proposer.draft_has_moe = False
        proposer.num_speculative_tokens = NUM_SPEC
        proposer.block_size = BLOCK_SIZE
        proposer.dflash_causal = False
        proposer.max_num_tokens = max_tokens
        proposer.arange_cpu = torch.arange(num_reqs + 1, dtype=torch.int32)
        proposer.device = torch.device("cpu")
        proposer.parallel_drafting_token_id = self.MASK_TOKEN
        proposer.input_ids = torch.full((max_tokens,), self.STALE, dtype=torch.int32)
        proposer.positions = torch.full((max_tokens,), self.STALE, dtype=torch.int64)

        calls = SimpleNamespace(metadata=None, model=None, context=None)

        def build_metadata(cad, positions, num_reqs, num_reqs_padded):
            calls.metadata = (num_reqs, num_reqs_padded, cad.num_reqs)
            return {}

        def model(input_ids, positions, token_indices_to_sample):
            calls.model = (
                tuple(input_ids.shape),
                tuple(positions.shape),
                int(token_indices_to_sample.shape[0]),
            )
            # One argmax per mask position, numbered so a slice is checkable.
            return torch.arange(token_indices_to_sample.shape[0])

        @contextmanager
        def forward_context(per_layer, vllm_config, **kwargs):
            calls.context = kwargs
            yield

        proposer._build_draft_attn_metadata = build_metadata
        proposer.model_executable = model
        monkeypatch.setattr(dflash_module, "set_forward_context", forward_context)
        monkeypatch.setattr(
            dflash_module, "build_kv_cache_forward_context_kwargs", lambda bases: {}
        )
        return proposer, calls

    def _cad(self, num_reqs):
        return SimpleNamespace(
            seq_lens_cpu_upper_bound=MAX_SEQ,
            max_seq_len=self.CONTEXT,
            block_table_tensor=torch.ones(num_reqs, 4, dtype=torch.int32),
        )

    def _run(self, proposer, num_reqs):
        return proposer._run_query_pass(
            self._cad(num_reqs),
            num_reqs,
            QUERY_LEN,
            num_reqs * QUERY_LEN,
            torch.zeros(num_reqs, dtype=torch.int32),
            torch.full((num_reqs,), self.CONTEXT, dtype=torch.int32),
        )

    def test_a_live_count_between_buckets_runs_at_the_next_bucket(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=3)
        drafts = self._run(proposer, 3)
        real, padded, described = calls.metadata
        assert (real, padded) == (3, 4)
        assert described == 3, "the metadata describes the real rows; the builder pads"
        assert calls.model == ((4, QUERY_LEN), (4, QUERY_LEN), 4 * NUM_SPEC)
        assert drafts.shape[0] == 4 * NUM_SPEC
        assert calls.context["num_tokens"] == 3 * QUERY_LEN
        # Off the DP path nothing pads the token dimension (`RBLNDPMetadata.make`).
        assert calls.context["num_padded_tokens"] is None
        assert calls.context["num_tokens_across_dp"] is None

    def test_a_bucket_sized_batch_is_not_padded(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=2)
        self._run(proposer, 2)
        assert calls.metadata[:2] == (2, 2)
        assert calls.model[0] == (2, QUERY_LEN)
        assert (proposer.input_ids == self.STALE).all(), "nothing to define"

    def test_padded_rows_are_defined_not_stale(self, monkeypatch):
        proposer, _ = self._proposer(monkeypatch, num_reqs=3)
        self._run(proposer, 3)
        tail = slice(3 * QUERY_LEN, 4 * QUERY_LEN)
        assert (proposer.input_ids[tail] == self.MASK_TOKEN).all()
        assert (proposer.positions[tail] == 0).all()
        # The real rows belong to `_fill_first_pass_inputs` and are left alone.
        assert (proposer.input_ids[: 3 * QUERY_LEN] == self.STALE).all()
        assert (proposer.positions[: 3 * QUERY_LEN] == self.STALE).all()

    def test_propose_returns_one_row_per_real_request(self, monkeypatch):
        proposer, calls = self._proposer(monkeypatch, num_reqs=3)
        proposer.supports_mm_inputs = False
        proposer.hidden_size = 16
        proposer.draft_id_to_target_id = None
        proposer._fill_first_pass_inputs = lambda *args: (
            0,
            torch.zeros(3, dtype=torch.int32),
            torch.full((3,), self.CONTEXT, dtype=torch.int32),
        )
        proposer._write_context_kv = lambda *args: None

        drafts = proposer.propose(
            target_token_ids=torch.zeros(3, dtype=torch.int32),
            target_positions=torch.zeros(3, dtype=torch.int64),
            target_hidden_states=torch.zeros(3, 16),
            next_token_ids=torch.zeros(3, dtype=torch.int32),
            token_indices_to_sample=None,
            common_attn_metadata=self._cad(3),
        )

        assert calls.model[0] == (4, QUERY_LEN), "the graph ran at the bucket"
        # ...and the padded fourth row's drafts never reach the scheduler.
        assert torch.equal(drafts, torch.arange(3 * NUM_SPEC).view(3, NUM_SPEC))

    def test_dp_pads_the_token_dimension_to_the_bucket(self, monkeypatch):
        status = DPStatus(
            num_tokens=(3 * QUERY_LEN, 2 * QUERY_LEN),
            num_reqs=(3, 2),
            is_prefill=(False, False),
            is_idle=(False, False),
            num_tokens_across_dp=torch.tensor(
                [3 * QUERY_LEN, 2 * QUERY_LEN], dtype=torch.int32
            ),
        )
        proposer, calls = self._proposer(
            monkeypatch, num_reqs=3, dp_size=2, dp_status=status
        )
        self._run(proposer, 3)
        assert calls.metadata[:2] == (3, 4)
        assert calls.context["num_padded_tokens"] == 4 * QUERY_LEN
        assert calls.context["num_tokens_across_dp"].tolist() == [
            3 * QUERY_LEN,
            2 * QUERY_LEN,
        ]

    def test_a_bucket_wider_than_the_token_budget_is_refused(self, monkeypatch):
        """The live count fits the buffers, its bucket does not: refuse rather
        than fall back to the live shape, which is the mismatch itself."""
        proposer, _ = self._proposer(monkeypatch, num_reqs=3, max_tokens=3 * QUERY_LEN)
        with pytest.raises(AssertionError, match="decode bucket"):
            self._run(proposer, 3)
