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

# RBLNModelRunner's small pure helpers, on a bare object.__new__ stub with only
# what each method reads. Methods that need the real runner's buffers live in
# test_rbln_model_runner_states / _inputs / _kv_cache. The one exception is
# TestShapeConfigWiring: what it checks is produced by __init__, so a stub could
# only repeat itself -- it builds a real runner and carries its own device marker.

import contextlib
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)

import vllm_rbln.v1.worker.dp_utils as dp_utils
import vllm_rbln.v1.worker.rbln_model_runner as mr
from vllm_rbln.v1.core.rbln_kv_cache_manager import KVCacheCopyOp
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer
from vllm_rbln.v1.spec_decode.utils import eagle_prepare_inputs_padded
from vllm_rbln.v1.worker.bucketing.exponential_bucketing_manager import (
    ExponentialBucketingManager,
)
from vllm_rbln.v1.worker.dp_utils import (
    BatchDescriptor,
    BatchRoute,
    DPStatus,
    ShapeConfig,
)
from vllm_rbln.v1.worker.rbln_model_runner import (
    ExecuteModelState,
    RBLNModelRunner,
    _depad_sampler_output,
    _pad_rows,
    _pad_sampling_metadata,
    _pad_spec_decode_metadata,
)


def _resolved_batch(
    *, num_reqs_padded, query_len, num_tokens_padded, route=BatchRoute.AGREED
):
    """What _determine_batch_execution_and_padding returns, for tests that drive
    _dummy_run without a DP group."""
    return (
        BatchDescriptor(
            num_reqs_padded=num_reqs_padded,
            query_len=query_len,
            num_tokens_padded=num_tokens_padded,
        ),
        route,
        torch.tensor([0, 0], dtype=torch.int32),
    )


def _make_runner_stub(**attrs):
    # A bare RBLNModelRunner (no __init__); set only the attributes the method
    # under test reads. dp_status is the exception: __init__ always sets it, and
    # the dummy step reads it before anything publishes one.
    runner = object.__new__(RBLNModelRunner)
    runner.dp_status = None
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def _sampling_metadata(n, *, no_penalties=True, spec_token_ids=None):
    return SamplingMetadata(
        temperature=torch.ones(n),
        all_greedy=False,
        all_random=True,
        top_p=torch.ones(n),
        top_k=torch.zeros(n, dtype=torch.long),
        generators={},
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(n),
        presence_penalties=torch.zeros(n),
        repetition_penalties=torch.ones(n),
        output_token_ids=[[] for _ in range(n)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
        logprob_token_ids=None,
        spec_token_ids=spec_token_ids,
        thinking_budget_state_holder=None,
    )


def _spec_decode_metadata(num_draft_tokens: list[int]) -> SpecDecodeMetadata:
    num_sampled_tokens = [n + 1 for n in num_draft_tokens]
    cu_sampled = torch.tensor(num_sampled_tokens, dtype=torch.int32).cumsum(0)
    cu_draft = torch.tensor(num_draft_tokens, dtype=torch.int32).cumsum(0)
    total_draft = int(cu_draft[-1])
    return SpecDecodeMetadata(
        draft_token_ids=torch.zeros(total_draft, dtype=torch.int32),
        num_draft_tokens=list(num_draft_tokens),
        cu_num_draft_tokens=cu_draft,
        cu_num_sampled_tokens=cu_sampled,
        target_logits_indices=torch.arange(total_draft, dtype=torch.int32),
        bonus_logits_indices=cu_sampled - 1,
        logits_indices=torch.arange(int(cu_sampled[-1]), dtype=torch.int32),
    )


def _input_batch(num_reqs=0):
    ib = InputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=1000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
    )
    for i in range(num_reqs):
        ib.add_request(
            _cached_state(f"r{i}", prompt_len=3 + i, block_ids=([i, i + 1],))
        )
    return ib


def _cached_state(req_id, *, prompt_len=3, num_computed=0, block_ids=None):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=SamplingParams(temperature=1.0),
        pooling_params=None,
        generator=None,
        block_ids=block_ids or ([0, 1],),
        num_computed_tokens=num_computed,
        output_token_ids=[],
    )


class TestGetCumsumAndArange:
    @staticmethod
    def _runner():
        return _make_runner_stub(arange_np=np.arange(64))

    def test_basic_example(self):
        cu, ar = self._runner()._get_cumsum_and_arange(np.array([2, 5, 3]))
        assert cu.tolist() == [2, 7, 10]
        assert ar.tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]

    def test_single_element(self):
        cu, ar = self._runner()._get_cumsum_and_arange(np.array([4]))
        assert cu.tolist() == [4]
        assert ar.tolist() == [0, 1, 2, 3]

    def test_two_elements(self):
        cu, ar = self._runner()._get_cumsum_and_arange(np.array([3, 2]))
        assert cu.tolist() == [3, 5]
        assert ar.tolist() == [0, 1, 2, 0, 1]

    def test_all_ones(self):
        cu, ar = self._runner()._get_cumsum_and_arange(np.array([1, 1, 1]))
        assert cu.tolist() == [1, 2, 3]
        assert ar.tolist() == [0, 0, 0]

    def test_cumsum_dtype_respected(self):
        cu, _ = self._runner()._get_cumsum_and_arange(
            np.array([2, 3]), cumsum_dtype=np.int32
        )
        assert cu.dtype == np.int32


class TestPadDepad:
    def test_pad_rows_repeats_last_row(self):
        t = torch.arange(6).reshape(2, 3)
        out = _pad_rows(t, 4)
        assert out.shape == (4, 3)
        assert torch.equal(out[:2], t)
        assert torch.equal(out[2], t[1])
        assert torch.equal(out[3], t[1])

    def test_pad_rows_reuses_storage_when_at_bucket(self):
        # A batch already at the bucket is handed through untouched -- reuse is
        # what keeps the sampler graph's input address fixed across steps.
        t = torch.arange(6).reshape(2, 3)
        out = _pad_rows(t, 2)
        assert torch.equal(out, t)
        assert out is t

    def test_pad_rows_none_passthrough(self):
        assert _pad_rows(None, 4) is None

    def test_pad_sampling_metadata_pads_rows_and_lists(self):
        # With penalties on, penalty rows + prompt/output lists are padded too.
        p = _pad_sampling_metadata(_sampling_metadata(2, no_penalties=False), 4)
        assert p.temperature.shape[0] == 4
        assert p.top_p.shape[0] == 4
        assert p.top_k.shape[0] == 4
        assert p.frequency_penalties.shape[0] == 4
        assert len(p.output_token_ids) == 4
        assert p.output_token_ids[2] == []

    def test_depad_sampler_output_trims_to_num_reqs(self):
        out = SamplerOutput(
            sampled_token_ids=torch.arange(4).reshape(4, 1), logprobs_tensors=None
        )
        assert _depad_sampler_output(out, 2).sampled_token_ids.shape == (2, 1)

    def test_depad_sampler_output_keeps_every_speculative_logprob_position(self):
        batch_size = 4
        output_width = 3
        out = SamplerOutput(
            sampled_token_ids=torch.zeros(
                (batch_size, output_width), dtype=torch.int32
            ),
            logprobs_tensors=LogprobsTensors(
                torch.zeros((batch_size * output_width, 2), dtype=torch.int32),
                torch.zeros((batch_size * output_width, 2)),
                torch.zeros(batch_size * output_width, dtype=torch.int32),
            ),
        )

        depadded = _depad_sampler_output(out, 2)

        assert depadded.sampled_token_ids.shape == (2, output_width)
        assert depadded.logprobs_tensors is not None
        assert depadded.logprobs_tensors.logprobs.shape[0] == 2 * output_width

    def test_pad_spec_decode_metadata_preserves_packed_token_axes(self):
        original = _spec_decode_metadata([1, 1])

        padded = _pad_spec_decode_metadata(original, 4)

        assert padded.num_draft_tokens == [1, 1, 0, 0]
        assert padded.cu_num_draft_tokens.shape[0] == 4
        assert padded.cu_num_sampled_tokens.shape[0] == 4
        assert padded.bonus_logits_indices.shape[0] == 4
        assert padded.max_spec_len == original.max_spec_len
        assert torch.equal(padded.target_logits_indices, original.target_logits_indices)
        assert torch.equal(padded.logits_indices, original.logits_indices)

    def test_padded_metadata_breaks_the_eagle_reader_it_would_reach(self):
        # Why the pad lives in _sample, not in the producer: this reader
        # differences cu_num_draft_tokens against num_reqs-sized tensors.
        padded = _pad_spec_decode_metadata(_spec_decode_metadata([1, 1]), 4)

        with pytest.raises(RuntimeError):
            eagle_prepare_inputs_padded(
                padded.cu_num_draft_tokens,
                torch.tensor([2, 2], dtype=torch.int32),
                torch.tensor([0, 2, 4], dtype=torch.int32),
            )


class TestSamplePadding:
    @staticmethod
    def _runner(rejection_output: SamplerOutput):
        rejection_sampler = MagicMock(return_value=rejection_output)
        runner = _make_runner_stub(
            _is_prefill_step=False,
            use_async_scheduling=False,
            input_batch=SimpleNamespace(
                num_reqs=2,
                sampling_metadata=_sampling_metadata(2, spec_token_ids=[[], []]),
            ),
            bucketing_manager=SimpleNamespace(
                decode_batch_buckets=[2, 4], max_batch_size=4
            ),
            max_num_reqs=8,
            rejection_sampler=rejection_sampler,
        )
        return runner, rejection_sampler

    def test_compiled_rejection_sampler_uses_per_stage_batch_bound(self, monkeypatch):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_SAMPLER", True)
        output = SamplerOutput(
            sampled_token_ids=torch.zeros((4, 3), dtype=torch.int32),
            logprobs_tensors=None,
        )
        runner, rejection_sampler = self._runner(output)

        runner._sample(torch.zeros((4, 10)), _spec_decode_metadata([1, 1]))

        padded_metadata = rejection_sampler.call_args.args[0]
        assert len(padded_metadata.num_draft_tokens) == 4

    def test_torch_rejection_sampler_keeps_live_batch_metadata(self, monkeypatch):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_SAMPLER", False)
        output = SamplerOutput(
            sampled_token_ids=torch.zeros((2, 3), dtype=torch.int32),
            logprobs_tensors=None,
        )
        runner, rejection_sampler = self._runner(output)
        spec_decode_metadata = _spec_decode_metadata([1, 1])
        sampling_metadata = runner.input_batch.sampling_metadata

        runner._sample(torch.zeros((4, 10)), spec_decode_metadata)

        assert rejection_sampler.call_args.args[0] is spec_decode_metadata
        assert rejection_sampler.call_args.args[3] is sampling_metadata


def test_rejection_sampler_warmup_uses_per_stage_batch_bound(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RBLN_SAMPLER", True)
    rejection_sample = MagicMock()
    runner = _make_runner_stub(
        speculative_config=object(),
        num_spec_tokens=2,
        is_pooling_model=False,
        model_config=SimpleNamespace(get_vocab_size=lambda: 10),
        device=torch.device("cpu"),
        bucketing_manager=SimpleNamespace(
            decode_batch_buckets=[2, 4], max_batch_size=4
        ),
        max_num_reqs=8,
        rejection_sampler=SimpleNamespace(
            impl=SimpleNamespace(rejection_sample=rejection_sample)
        ),
    )

    runner._warmup_sampler_decode_batches()

    assert rejection_sample.call_count == 1
    assert len(rejection_sample.call_args.args[1]) == 4


class TestPredicates:
    def test_is_prefill_is_written_only_through_the_setter(self):
        # The is_prefill setter is the sole write path; the getter -- the
        # single source of truth -- returns it, so all PP ranks agree.
        r = _make_runner_stub()
        r.is_prefill = True
        assert r.is_prefill is True
        r.is_prefill = False
        assert r.is_prefill is False

    def test_is_intermediate_chunked_prefill(self):
        # is_prefill (the step phase) AND discard_request_mask[0].
        r = _make_runner_stub(
            _is_prefill_step=True,
            discard_request_mask=np.array([True]),
        )
        assert r.is_intermediate_chunked_prefill is True
        r.discard_request_mask = np.array([False])
        assert r.is_intermediate_chunked_prefill is False

    def test_use_wrapped_compute_logits_is_not_pooling(self):
        assert _make_runner_stub(is_pooling_model=False).use_wrapped_compute_logits
        assert not _make_runner_stub(is_pooling_model=True).use_wrapped_compute_logits


class TestExecuteModelState:
    def test_field_names(self):
        assert ExecuteModelState._fields == (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "combined_hidden_states",
        )

    def test_is_named_tuple(self):
        s = ExecuteModelState(1, 2, 3, 4, 5, 6, 7)
        assert isinstance(s, tuple)
        assert tuple(s) == (1, 2, 3, 4, 5, 6, 7)


class TestGetNansInLogits:
    @staticmethod
    def _runner():
        return _make_runner_stub(
            input_batch=SimpleNamespace(
                req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}
            )
        )

    def test_none_logits_returns_all_zero(self):
        assert self._runner()._get_nans_in_logits(None) == {"a": 0, "b": 0}

    def test_counts_nan_rows_per_request(self):
        logits = torch.zeros(2, 4)
        logits[1, 0] = float("nan")
        assert self._runner()._get_nans_in_logits(logits) == {"a": 0, "b": 1}


class TestSelectCanonicalKvLayersPerPool:
    @staticmethod
    def _full():
        # Only isinstance(spec, FullAttentionSpec) matters; skip the ctor.
        return object.__new__(FullAttentionSpec)

    @staticmethod
    def _group(layer_names, spec):
        return SimpleNamespace(layer_names=layer_names, kv_cache_spec=spec)

    def _runner(self, groups):
        r = _make_runner_stub()
        r._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
        return r

    @staticmethod
    def _cfg(*pools):
        return SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(shared_by=list(p)) for p in pools]
        )

    def test_prefers_full_attention_layer(self):
        groups = [
            self._group(["sw0"], SimpleNamespace()),
            self._group(["full0"], self._full()),
        ]
        r = self._runner(groups)
        assert r._select_canonical_kv_layers_per_pool(self._cfg(["sw0", "full0"])) == {
            "full0"
        }

    def test_falls_back_to_first_layer(self):
        # No full-attention layer in the pool -> shared_by[0].
        r = self._runner([self._group(["sw0", "sw1"], SimpleNamespace())])
        assert r._select_canonical_kv_layers_per_pool(self._cfg(["sw0", "sw1"])) == {
            "sw0"
        }

    def test_skips_empty_shared_by(self):
        r = self._runner([self._group(["full0"], self._full())])
        assert r._select_canonical_kv_layers_per_pool(self._cfg([])) == set()

    def test_one_canonical_layer_per_pool(self):
        groups = [
            self._group(["full0"], self._full()),
            self._group(["full1"], self._full()),
        ]
        r = self._runner(groups)
        assert r._select_canonical_kv_layers_per_pool(
            self._cfg(["full0"], ["full1"])
        ) == {"full0", "full1"}


class TestGetSupportedTasks:
    # Tests the runner_type dispatch; the underlying task-detection lives in
    # the (model-dependent) sub-methods, which are stubbed here.
    @staticmethod
    def _runner(runner_type):
        r = _make_runner_stub(model_config=SimpleNamespace(runner_type=runner_type))
        r.get_supported_generation_tasks = lambda: ["generate"]
        r.get_supported_pooling_tasks = lambda: ["encode"]
        return r

    def test_generate_routing(self):
        assert self._runner("generate").get_supported_tasks() == ("generate",)

    def test_pooling_routing(self):
        assert self._runner("pooling").get_supported_tasks() == ("encode",)

    def test_neither_runner_type_is_empty(self):
        # A runner_type matching neither branch (e.g. "draft") -> empty tuple.
        assert self._runner("draft").get_supported_tasks() == ()


class TestResolveBatchDescriptor:
    # data_parallel_size == 1 is the covered path (multi-DP needs RBLNDPMetadata
    # collectives). The phase is driven via _is_prefill_step (is_prefill), not
    # the runner's input_batch.
    @staticmethod
    def _runner(*, is_prefill, bucket=8):
        computed = 0 if is_prefill else 9
        return _make_runner_stub(
            _is_prefill_step=is_prefill,
            shape_config=ShapeConfig(
                decode_batch_buckets=(1, 2, 4, 8),
                find_bucket=lambda n: bucket,
                max_num_tokens=512,
                specialized_moe_decode=False,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            input_batch=SimpleNamespace(
                num_computed_tokens_cpu=np.array([computed]),
                num_tokens_no_spec=np.array([10]),
            ),
        )

    def test_decode_pads_to_bucket(self):
        runner = self._runner(is_prefill=False, bucket=8)
        batch_desc, route, _across = runner._determine_batch_execution_and_padding(
            3, 30
        )
        assert route is BatchRoute.LOCAL
        assert batch_desc.num_reqs_padded == 8
        assert batch_desc.query_len == 10  # 30 tokens over 3 requests

    def test_prefill_uses_unpadded(self):
        batch_desc, _route, _across = self._runner(
            is_prefill=True
        )._determine_batch_execution_and_padding(3, 30)
        assert batch_desc.num_reqs_padded == 3

    def test_single_dp_returns_no_token_padding(self):
        # Nothing to agree with, so the caller pads neither tokens nor the context.
        batch_desc, _route, across = self._runner(
            is_prefill=False
        )._determine_batch_execution_and_padding(3, 30)
        assert batch_desc.num_tokens_padded is None
        assert across is None


class TestShapeConfigWiring:
    # ShapeConfig is built once in __init__ and every route reads it. The tests
    # around here hand-build one, which leaves the construction unpinned: a field
    # wired to the wrong source would keep every one of them green.

    @pytest.mark.maybe_use_device
    def test_a_real_runners_shapes_reach_the_decision(self, make_model_runner):
        # Drive the rule with this runner's own config and a two-rank status, so
        # every field shows up in the answer: the bucket rule (3 sits between two
        # buckets, so a pass-through would keep 3), the token dimension a padded
        # route uses, and specialization -- off at a single rank, which is why this
        # is the unspecialized answer rather than a peer-driven one.
        runner = make_model_runner()
        desc, route = dp_utils.determine_batch_execution_and_padding(
            cfg=runner.shape_config,
            num_reqs=3,
            num_tokens=3,
            is_prefill=False,
            status=DPStatus(
                num_tokens=(3, 3),
                num_reqs=(3, 3),
                is_prefill=(False, False),
                is_idle=(False, False),
                num_tokens_across_dp=torch.tensor([3, 3], dtype=torch.int32),
            ),
        )
        assert route is BatchRoute.UNSPECIALIZED
        assert (
            desc.num_reqs_padded == runner.bucketing_manager.find_decode_batch_bucket(3)
        )
        assert desc.num_tokens_padded == runner.max_num_tokens

    @pytest.mark.maybe_use_device
    def test_the_bucket_list_reaches_the_routes_that_read_it(self, make_model_runner):
        # decode_batch_buckets is read by the two routes that pick an end of it, and
        # both need data parallelism to specialize at all -- so this runner is built
        # with a peer and a ladder of buckets, or the top and the first would be the
        # same entry and the answers indistinguishable.
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT", "4")
        runner = make_model_runner(data_parallel_size=2, max_num_seqs=8)
        monkeypatch.undo()
        buckets = runner.bucketing_manager.decode_batch_buckets
        assert len(buckets) > 1, buckets

        def decide(status):
            return dp_utils.determine_batch_execution_and_padding(
                cfg=runner.shape_config,
                num_reqs=2,
                num_tokens=2,
                is_prefill=False,
                status=status,
            )

        desc, route = decide(
            DPStatus(
                num_tokens=(2, 128),
                num_reqs=(2, 1),
                is_prefill=(False, True),
                is_idle=(False, False),
                num_tokens_across_dp=torch.tensor([2, 128], dtype=torch.int32),
            )
        )
        assert (route, desc.num_reqs_padded) == (BatchRoute.ANY_PREFILL, buckets[-1])

        # The other end is read where the busy ranks disagree on a query length, so
        # this one comes from a status that says so.
        desc, route = decide(
            DPStatus(
                num_tokens=(2, 6),
                num_reqs=(2, 2),
                is_prefill=(False, False),
                is_idle=(False, False),
                num_tokens_across_dp=torch.tensor([2, 6], dtype=torch.int32),
            )
        )
        assert (route, desc.num_reqs_padded) == (BatchRoute.QLEN_ASYM, buckets[-1])


class TestDummyRunPadding:
    # _dummy_run under DP: the decode layout must carry the group-agreed bucket,
    # not this rank's own count. #894 was the layout keeping num_reqs while the
    # attention metadata already used num_reqs_padded.
    @staticmethod
    def _runner(
        monkeypatch,
        *,
        reqs_across_dp,
        tokens_across_dp=None,
        specialized=True,
        peers_idle=False,
    ):
        captured: dict = {}
        # One token per request unless a case needs the peers on a longer query.
        tokens = list(tokens_across_dp or reqs_across_dp)

        def fake_gather(
            num_tokens, num_reqs, dp_size, dp_rank, is_prefill, is_idle=False
        ):
            # This rank's flags are the ones it passed in -- a fake that hardcodes
            # them cannot tell whether the caller forwarded the phase or the idle
            # bit at all.
            return DPStatus(
                num_tokens=tuple(tokens),
                num_reqs=tuple(reqs_across_dp),
                is_prefill=(is_prefill,) + (False,) * (len(reqs_across_dp) - 1),
                is_idle=(is_idle,) + (peers_idle,) * (len(reqs_across_dp) - 1),
                num_tokens_across_dp=torch.tensor(tokens, dtype=torch.int32),
            )

        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
        )

        def forward_context(*a, **kw):
            captured["forward"] = kw
            return contextlib.nullcontext()

        monkeypatch.setattr(mr, "set_forward_context", forward_context)
        monkeypatch.setattr(
            mr, "build_kv_cache_forward_context_kwargs", lambda *a, **kw: {}
        )
        monkeypatch.setattr(dp_utils, "_synchronize_dp_ranks", fake_gather)

        def stage(**kwargs):
            captured["layout"] = kwargs["layout"]
            return SimpleNamespace(as_kwargs=lambda: {})

        # Two buckets, so a padded count can differ from a raw one at all.
        bucketing = ExponentialBucketingManager(
            max_batch_size=2, min_batch_size=1, limit=2, step=2
        )
        runner = _make_runner_stub(
            bucketing_manager=bucketing,
            shape_config=ShapeConfig(
                decode_batch_buckets=bucketing.decode_batch_buckets,
                find_bucket=bucketing.find_decode_batch_bucket,
                max_num_tokens=128,
                specialized_moe_decode=specialized,
            ),
            parallel_config=SimpleNamespace(data_parallel_size=4, data_parallel_rank=0),
            input_batch=SimpleNamespace(
                num_tokens_no_spec=np.zeros(8, dtype=np.int32),
                num_computed_tokens_cpu=np.zeros(8, dtype=np.int32),
            ),
            max_num_tokens=128,
            max_num_reqs=8,
            seq_lens_np=np.zeros(8, dtype=np.int32),
            query_start_loc_np=np.zeros(16, dtype=np.int32),
            arange_np=np.arange(128),
            input_ids=torch.zeros(128, dtype=torch.int32),
            positions=torch.zeros(128, dtype=torch.int32),
            # use_wrapped_compute_logits is a property over this.
            is_pooling_model=True,
            speculative_config=None,
            # Read by the batch decision to floor the MoE dispatch pad at the
            # spec width; __init__ always sets it.
            use_aux_hidden_state_outputs=False,
            # Gates the drafter's dummy run; __init__ always sets it.
            drafter=None,
            kv_cache_bases=None,
            vllm_config=None,
            input_stager=SimpleNamespace(stage=stage),
            model_executable=lambda **kwargs: None,
            _build_attention_metadata=lambda **kwargs: (
                captured.setdefault("attn", kwargs),
                None,
            ),
        )
        return runner, captured

    def test_decode_layout_takes_the_group_bucket(self, monkeypatch):
        runner, captured = self._runner(monkeypatch, reqs_across_dp=[2, 1, 1, 1])
        runner._dummy_run(1, 1, is_prefill=False)

        assert captured["layout"].num_reqs == 1
        assert captured["layout"].num_reqs_padded == 2
        # Both halves must agree; they disagreeing is the shape error.
        assert captured["attn"]["num_reqs_padded"] == 2

    def test_prefill_layout_keeps_the_raw_count(self, monkeypatch):
        runner, captured = self._runner(monkeypatch, reqs_across_dp=[2, 1, 1, 1])
        runner._dummy_run(1, 4, is_prefill=True)

        assert captured["layout"].num_reqs_padded == 1

    def test_warmup_pin_is_the_token_dimension(self, monkeypatch):
        # Warm-up dictates the dimension it wants compiled -- the group agreement
        # would give a smaller one -- and nothing downstream recomputes it, so the
        # forward context (the only reader of the token dimension) gets the pin.
        runner, captured = self._runner(monkeypatch, reqs_across_dp=[2, 1, 1, 1])
        runner._dummy_run(1, 1, is_prefill=False, num_tokens_padded_override=64)

        assert captured["forward"]["num_padded_tokens"] == 64
        # The pin only dictates the token dimension; the batch stays this rank's.
        assert captured["layout"].num_reqs_padded == 1

    def test_serving_idle_step_is_excluded_from_the_agreement(self, monkeypatch):
        # warmup=False is the serving DP-idle step. This rank reports a minimal
        # (1 req, 1 token) so the peers do not block, but it must be marked idle:
        # counted as busy it disagrees with their query length, and the group falls
        # to the asymmetric graph where this rank stages its own length instead of
        # theirs. Every other case here is warm-up, where nothing is idle.
        runner, captured = self._runner(
            monkeypatch, reqs_across_dp=[1, 2, 2, 2], tokens_across_dp=[1, 8, 8, 8]
        )
        runner._dummy_run(1, 1, is_prefill=False, warmup=False)

        # The busy ranks' shape: bucket for 2 requests, their query length 8/2.
        assert captured["layout"].num_reqs_padded == 2
        assert captured["layout"].query_len == 4

    @pytest.mark.parametrize("specialized", [True, False])
    def test_a_fully_drained_group_runs_nothing(self, monkeypatch, specialized):
        # Every rank read the same status, so they all stop: no peer is waiting on
        # this rank inside a forward, and the output of a step nobody asked for is
        # discarded anyway. The status says so whatever the MoE configuration is,
        # which a shape route could not -- the configurations answer differently.
        runner, captured = self._runner(
            monkeypatch,
            reqs_across_dp=[1, 1, 1, 1],
            peers_idle=True,
            specialized=specialized,
        )
        runner._dummy_run(1, 1, is_prefill=False, warmup=False)

        assert captured == {}

    def test_without_specialised_decode_no_group_agreement(self, monkeypatch):
        runner, captured = self._runner(
            monkeypatch, reqs_across_dp=[2, 1, 1, 1], specialized=False
        )
        runner._dummy_run(1, 1, is_prefill=False)

        # Each rank keeps its own bucket, so a peer at bucket 2 disagrees.
        assert captured["layout"].num_reqs_padded == 1


class TestProcessKvCacheCopyOps:
    # Path selection: use_runtime = not USE_DEVICE_TENSOR and not enforce_eager
    # and VLLM_RBLN_COMPILE_MODEL. Forced deterministically via monkeypatch.
    def test_eager_copy_non_mla(self, monkeypatch):
        monkeypatch.setattr(mr, "USE_DEVICE_TENSOR", True)  # -> eager path
        # non-MLA layout: (2, num_blocks, heads, 1, block_tokens, dim).
        kv = torch.zeros(2, 4, 1, 1, 8, 2)
        kv[:, 1, :, :, :, :] = 5.0  # source = block 1
        r = _make_runner_stub(
            kv_caches=[kv],
            model_config=SimpleNamespace(use_mla=False, enforce_eager=True),
            runtime_holder=[None],
        )
        r._process_kv_cache_copy_ops([KVCacheCopyOp(0, 1, 2, 3)])
        # First 3 token slots of dst block 2 now match src; the rest stay 0.
        assert torch.equal(kv[:, 2, :, :, :3, :].cpu(), kv[:, 1, :, :, :3, :].cpu())
        assert (kv[:, 2, :, :, :3, :] == 5.0).all()
        assert (kv[:, 2, :, :, 3:, :] == 0.0).all()

    def test_eager_copy_mla(self, monkeypatch):
        monkeypatch.setattr(mr, "USE_DEVICE_TENSOR", True)
        kv = torch.zeros(4, 8, 2)  # (num_blocks, block_tokens, dim)
        kv[1] = 7.0
        r = _make_runner_stub(
            kv_caches=[kv],
            model_config=SimpleNamespace(use_mla=True, enforce_eager=True),
            runtime_holder=[None],
        )
        r._process_kv_cache_copy_ops([KVCacheCopyOp(0, 1, 2, 3)])
        assert (kv[2, :3, :] == 7.0).all()
        assert (kv[2, 3:, :] == 0.0).all()

    def test_runtime_copy_when_compiled_non_device_tensor(self, monkeypatch):
        monkeypatch.setattr(mr, "USE_DEVICE_TENSOR", False)
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        calls = []
        runtime = SimpleNamespace(
            _copy_kv_cache=lambda src, dst, nt: calls.append((src, dst, nt))
        )
        r = _make_runner_stub(
            kv_caches=[],
            model_config=SimpleNamespace(use_mla=False, enforce_eager=False),
            runtime_holder=[runtime],
        )
        r._process_kv_cache_copy_ops([KVCacheCopyOp(0, 5, 6, 4)])
        assert calls == [(5, 6, 4)]


def _empty_cached():
    return SimpleNamespace(
        resumed_req_ids=set(),
        req_ids=[],
        num_computed_tokens=[],
        new_block_ids=[],
        num_output_tokens=[],
        new_token_ids=[],
    )


def _new_req(req_id, *, prompt_logprobs=None):
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        mm_features=None,
        sampling_params=SamplingParams(
            temperature=1.0, prompt_logprobs=prompt_logprobs
        ),
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _sched(*, new=(), finished=(), scheduled=None, cached=None, spec=None):
    return SimpleNamespace(
        finished_req_ids=list(finished),
        num_scheduled_tokens=scheduled if scheduled is not None else {},
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached or _empty_cached(),
        scheduled_spec_decode_tokens=spec or {},
    )


class TestUpdateStates:
    # Request-state bookkeeping on a real InputBatch; scheduler_output is
    # duck-typed since every access is an attribute or index read.
    @staticmethod
    def _runner(monkeypatch, *, input_batch, requests=None):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        return _make_runner_stub(
            input_batch=input_batch,
            requests=requests if requests is not None else {},
            num_prompt_logprobs={},
            is_pooling_model=False,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        )

    def test_new_request_added(self, monkeypatch):
        r = self._runner(monkeypatch, input_batch=_input_batch(0))
        r._update_states(_sched(new=[_new_req("n0")], scheduled={"n0": 3}))
        assert "n0" in r.requests
        assert "n0" in r.input_batch.req_id_to_index

    def test_new_request_prompt_logprobs_recorded(self, monkeypatch):
        r = self._runner(monkeypatch, input_batch=_input_batch(0))
        r._update_states(
            _sched(new=[_new_req("n0", prompt_logprobs=5)], scheduled={"n0": 3})
        )
        assert r.num_prompt_logprobs["n0"] == 5

    def test_finished_request_removed(self, monkeypatch):
        r = self._runner(
            monkeypatch,
            input_batch=_input_batch(1),
            requests={"r0": _cached_state("r0")},
        )
        r._update_states(_sched(finished=["r0"]))
        assert "r0" not in r.input_batch.req_id_to_index
        assert "r0" not in r.requests

    def test_unscheduled_request_removed(self, monkeypatch):
        # r0 is cached but not scheduled (and not resumed) -> dropped.
        r = self._runner(monkeypatch, input_batch=_input_batch(1))
        r._update_states(_sched(scheduled={}))
        assert "r0" not in r.input_batch.req_id_to_index

    def test_resumed_request_readded(self, monkeypatch):
        # A preempted request: present in requests, absent from the batch, and
        # flagged resumed -> re-admitted to the persistent batch.
        state = _cached_state("p0", num_computed=3)
        r = self._runner(
            monkeypatch, input_batch=_input_batch(0), requests={"p0": state}
        )
        cached = SimpleNamespace(
            resumed_req_ids={"p0"},
            req_ids=["p0"],
            num_computed_tokens=[5],
            new_block_ids=[([0, 1],)],
            num_output_tokens=[0],
            new_token_ids=[[]],
        )
        r._update_states(_sched(scheduled={"p0": 1}, cached=cached))
        assert "p0" in r.input_batch.req_id_to_index

    def test_cached_request_num_computed_tokens_updated(self, monkeypatch):
        r = self._runner(
            monkeypatch,
            input_batch=_input_batch(1),
            requests={"r0": _cached_state("r0")},
        )
        cached = SimpleNamespace(
            resumed_req_ids=set(),
            req_ids=["r0"],
            num_computed_tokens=[7],
            new_block_ids=[None],
            num_output_tokens=[0],
            new_token_ids=[[]],
        )
        r._update_states(_sched(scheduled={"r0": 1}, cached=cached))
        idx = r.input_batch.req_id_to_index["r0"]
        assert r.input_batch.num_computed_tokens_cpu[idx] == 7


class TestCalcSpecDecodeMetadata:
    # Pure index math; input_ids just needs to be indexable by logits_indices.
    @staticmethod
    def _runner():
        return _make_runner_stub(
            arange_np=np.arange(512),
            input_ids=torch.arange(512),
            device=torch.device("cpu"),
        )

    def test_logits_indices_layout(self):
        # The worked example from the source docstring.
        md = self._runner()._calc_spec_decode_metadata(
            np.array([3, 0, 2, 0, 1]), np.array([4, 104, 107, 207, 209])
        )
        logits = [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        assert md.logits_indices.tolist() == logits
        assert md.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
        assert md.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10]
        assert md.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6]
        assert md.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11]
        # With input_ids = arange, draft_token_ids == the docstring's
        # draft_token_indices (extracted at target_logits_indices + 1).
        assert md.draft_token_ids.tolist() == [1, 2, 3, 105, 106, 208]

    def test_no_draft_tokens(self):
        # Decode-only: one sampled (bonus) token per request, no draft targets.
        md = self._runner()._calc_spec_decode_metadata(
            np.array([0, 0]), np.array([1, 2])
        )
        assert md.logits_indices.tolist() == [0, 1]
        assert md.target_logits_indices.tolist() == []
        assert md.bonus_logits_indices.tolist() == [0, 1]


class TestMayReorderBatch:
    # Stable descending sort by num_tokens_no_spec, applied in place. Uses a real
    # InputBatch; scheduler_output is unused by the sort path, so None is passed.
    @staticmethod
    def _runner(monkeypatch, ib, *, sort=True, groups=1):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_SORT_BATCH", sort)
        return _make_runner_stub(
            input_batch=ib,
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()] * groups),
        )

    @staticmethod
    def _batch(tokens):
        ib = _input_batch(len(tokens))
        ib.num_tokens_no_spec[: len(tokens)] = tokens
        return ib

    def test_noop_when_sort_disabled(self, monkeypatch):
        r = self._runner(monkeypatch, self._batch([1, 3, 2, 4]), sort=False)
        r._may_reorder_batch(None)
        assert r.input_batch.req_ids == ["r0", "r1", "r2", "r3"]

    def test_noop_when_no_kv_cache_groups(self, monkeypatch):
        r = self._runner(monkeypatch, self._batch([1, 3, 2, 4]), groups=0)
        r._may_reorder_batch(None)
        assert r.input_batch.req_ids == ["r0", "r1", "r2", "r3"]

    def test_already_sorted_skips(self, monkeypatch):
        r = self._runner(monkeypatch, self._batch([4, 3, 2, 1]))
        r._may_reorder_batch(None)
        assert r.input_batch.req_ids == ["r0", "r1", "r2", "r3"]
        assert r.input_batch.batch_update_builder.moved == []

    def test_sorts_descending_by_num_tokens(self, monkeypatch):
        r = self._runner(monkeypatch, self._batch([1, 3, 2, 4]))
        r._may_reorder_batch(None)
        assert r.input_batch.req_ids == ["r3", "r1", "r2", "r0"]
        assert r.input_batch.num_tokens_no_spec[:4].tolist() == [4, 3, 2, 1]

    def test_emits_swap_records_for_non_pooling(self, monkeypatch):
        r = self._runner(monkeypatch, self._batch([1, 3, 2, 4]))
        assert not r.input_batch.is_pooling_model
        r._may_reorder_batch(None)
        # Non-pooling models replay pairwise swaps into the logits-proc builder.
        assert r.input_batch.batch_update_builder.moved != []


class TestAllocateKvCacheTensors:
    # Device selection: "cpu" if not compiling, else self.device if device-tensor,
    # else "meta". The mapping/validation logic is exercised on CPU.
    @staticmethod
    def _cfg():
        return SimpleNamespace(
            kv_cache_tensors=[
                SimpleNamespace(size=64, shared_by=["l0", "l1"]),
                SimpleNamespace(size=32, shared_by=["l2"]),
            ],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["l0", "l1"]),
                SimpleNamespace(layer_names=["l2"]),
            ],
        )

    def _runner(self):
        return _make_runner_stub(
            device=torch.device("cpu"), runner_only_attn_layers=set()
        )

    def test_cpu_when_not_compiling(self, monkeypatch):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_COMPILE_MODEL", False)
        raw = self._runner()._allocate_kv_cache_tensors(self._cfg())
        assert set(raw) == {"l0", "l1", "l2"}
        assert raw["l0"].device.type == "cpu"
        # Layers sharing a pool share the same buffer object.
        assert raw["l0"] is raw["l1"]
        assert raw["l0"] is not raw["l2"]

    def test_meta_when_compiling_without_device_tensor(self, monkeypatch):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        monkeypatch.setattr(mr, "USE_DEVICE_TENSOR", False)
        raw = self._runner()._allocate_kv_cache_tensors(self._cfg())
        assert raw["l0"].device.type == "meta"

    def test_self_device_when_compiling_with_device_tensor(self, monkeypatch):
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_COMPILE_MODEL", True)
        monkeypatch.setattr(mr, "USE_DEVICE_TENSOR", True)
        raw = self._runner()._allocate_kv_cache_tensors(self._cfg())
        assert raw["l0"].device.type == "cpu"  # self.device is cpu here


class TestRepairStagedInputIds:
    # The scheduler stages -1 where this step's input token belongs; the real
    # token is still on the device in the previous step's ring slot. The repair
    # has to land each one on the row the request occupies *now*.
    @staticmethod
    def _runner(*, prev_index, prev_tokens, is_prefill=False):
        return _make_runner_stub(
            is_prefill=is_prefill,
            _prev_token_host_buffer=None,
            input_batch=SimpleNamespace(
                prev_sampled_token_ids=(
                    None
                    if prev_tokens is None
                    else torch.tensor(prev_tokens, dtype=torch.int32).unsqueeze(1)
                ),
                prev_req_id_to_index=prev_index,
            ),
        )

    @staticmethod
    def _staged(num_rows):
        # -1 is what the scheduler left behind.
        return SimpleNamespace(
            input_ids=torch.full((num_rows, 1), -1, dtype=torch.int32)
        )

    def test_repairs_every_row_when_the_batch_is_unchanged(self):
        r = self._runner(prev_index={"a": 0, "b": 1}, prev_tokens=[10, 11])
        staged = self._staged(2)
        r._repair_staged_input_ids(staged, ["a", "b"])
        assert staged.input_ids[:, 0].tolist() == [10, 11]

    def test_follows_the_request_when_rows_are_reordered(self):
        r = self._runner(prev_index={"a": 0, "b": 1}, prev_tokens=[10, 11])
        staged = self._staged(2)
        r._repair_staged_input_ids(staged, ["b", "a"])
        assert staged.input_ids[:, 0].tolist() == [11, 10]

    def test_does_not_shift_rows_up_when_a_request_is_skipped(self):
        # The row-crossing case: "new" has no previous row, so the requests that
        # do have one sit at rows 1 and 2. Their previous rows are also 1 and 2,
        # so the row lists match -- but writing contiguously would still start at
        # row 0 and hand row 1 the token belonging to row 0.
        r = self._runner(prev_index={"a": 0, "b": 1, "c": 2}, prev_tokens=[10, 11, 12])
        staged = self._staged(3)
        r._repair_staged_input_ids(staged, ["new", "b", "c"])
        assert staged.input_ids[:, 0].tolist() == [-1, 11, 12]

    def test_leaves_a_request_absent_from_the_previous_batch(self):
        # _apply_pending_token_writeback repairs that one a step later.
        r = self._runner(prev_index={"a": 0}, prev_tokens=[10])
        staged = self._staged(2)
        r._repair_staged_input_ids(staged, ["a", "new"])
        assert staged.input_ids[:, 0].tolist() == [10, -1]

    def test_does_nothing_when_no_request_carries_over(self):
        r = self._runner(prev_index={"gone": 0}, prev_tokens=[10])
        staged = self._staged(1)
        r._repair_staged_input_ids(staged, ["new"])
        assert staged.input_ids[:, 0].tolist() == [-1]

    def test_does_nothing_on_a_prefill_step(self):
        # Prefill has no previous sampled token to feed back.
        r = self._runner(prev_index={"a": 0}, prev_tokens=[10], is_prefill=True)
        staged = self._staged(1)
        r._repair_staged_input_ids(staged, ["a"])
        assert staged.input_ids[:, 0].tolist() == [-1]

    def test_does_nothing_before_the_first_sampled_token_exists(self):
        r = self._runner(prev_index={"a": 0}, prev_tokens=None)
        staged = self._staged(1)
        r._repair_staged_input_ids(staged, ["a"])
        assert staged.input_ids[:, 0].tolist() == [-1]


class TestRepairAsyncOutputTokenIds:
    # The logits processors read output_token_ids in the same step, before
    # _apply_pending_token_writeback runs, so the -1 tail has to go now.
    @staticmethod
    def _runner(*, req_ids, output_token_ids, prev_index, prev_tokens):
        return _make_runner_stub(
            input_batch=SimpleNamespace(
                req_ids=req_ids,
                sampling_metadata=SimpleNamespace(output_token_ids=output_token_ids),
                prev_sampled_token_ids=(
                    None
                    if prev_tokens is None
                    else torch.tensor(prev_tokens, dtype=torch.int32).unsqueeze(1)
                ),
                prev_req_id_to_index=prev_index,
            ),
        )

    def test_replaces_the_placeholder_tail(self):
        ids = [[7, -1], [8, -1]]
        r = self._runner(
            req_ids=["a", "b"],
            output_token_ids=ids,
            prev_index={"a": 0, "b": 1},
            prev_tokens=[10, 11],
        )
        r._repair_async_output_token_ids()
        assert ids == [[7, 10], [8, 11]]

    def test_reads_the_row_the_request_had_last_step(self):
        ids = [[7, -1], [8, -1]]
        r = self._runner(
            req_ids=["b", "a"],
            output_token_ids=ids,
            prev_index={"a": 0, "b": 1},
            prev_tokens=[10, 11],
        )
        r._repair_async_output_token_ids()
        assert ids == [[7, 11], [8, 10]]

    def test_leaves_a_tail_that_is_not_a_placeholder(self):
        ids = [[7, 9]]
        r = self._runner(
            req_ids=["a"],
            output_token_ids=ids,
            prev_index={"a": 0},
            prev_tokens=[10],
        )
        r._repair_async_output_token_ids()
        assert ids == [[7, 9]]

    def test_leaves_a_request_absent_from_the_previous_batch(self):
        ids = [[7, -1]]
        r = self._runner(
            req_ids=["new"],
            output_token_ids=ids,
            prev_index={"a": 0},
            prev_tokens=[10],
        )
        r._repair_async_output_token_ids()
        assert ids == [[7, -1]]

    def test_leaves_a_request_with_no_output_yet(self):
        ids = [[], [8, -1]]
        r = self._runner(
            req_ids=["a", "b"],
            output_token_ids=ids,
            prev_index={"a": 0, "b": 1},
            prev_tokens=[10, 11],
        )
        r._repair_async_output_token_ids()
        assert ids == [[], [8, 11]]


class TestApplyPendingTokenWriteback:
    # The async repair has to move both stores together. token_ids_cpu is rebuilt
    # from output_token_ids when a request re-enters the batch, so writing the
    # real token to one and not the other leaves a value that comes back wrong.
    PROMPT_LEN = 3
    START = PROMPT_LEN  # where staging put the placeholder

    @classmethod
    def _runner(cls, *, output_token_ids, keep_request_state=True):
        ib = _input_batch(0)
        state = _cached_state("r0", prompt_len=cls.PROMPT_LEN)
        ib.add_request(state)
        # What the async path left behind: a -1 in both stores.
        state.output_token_ids[:] = output_token_ids
        ib.token_ids_cpu[0, cls.START] = -1
        runner = _make_runner_stub(
            input_batch=ib,
            requests={"r0": state} if keep_request_state else {},
            _pending_token_writeback=deque([(["r0"], [[42]], {"r0": cls.START})]),
        )
        return runner, ib, state

    def test_repairs_both_stores(self):
        runner, ib, state = self._runner(output_token_ids=[-1])
        runner._apply_pending_token_writeback()
        assert state.output_token_ids == [42]
        assert ib.token_ids_cpu[0, self.START] == 42

    def test_leaves_both_stores_when_the_offset_is_out_of_bounds(self):
        # The request rolled back past this step, so output_token_ids no longer
        # has room for the placeholder.
        runner, ib, state = self._runner(output_token_ids=[])
        runner._apply_pending_token_writeback()
        assert state.output_token_ids == []
        assert ib.token_ids_cpu[0, self.START] == -1

    def test_leaves_both_stores_when_the_request_state_is_gone(self):
        runner, ib, _ = self._runner(output_token_ids=[-1], keep_request_state=False)
        runner._apply_pending_token_writeback()
        assert ib.token_ids_cpu[0, self.START] == -1


class TestMixinConformance:
    def test_inherits_kv_connector_mixin(self):
        assert issubclass(RBLNModelRunner, KVConnectorModelRunnerMixin)

    def test_expected_public_methods_exist(self):
        for name in (
            "execute_model",
            "sample_tokens",
            "get_kv_cache_spec",
            "load_model",
        ):
            assert callable(getattr(RBLNModelRunner, name, None)), name


class TestDummyRunDraftParticipation:
    # On a serving DP-idle step (warmup=False) the rank still runs the draft dummy,
    # so a draft whose forward joins a DP all-gather keeps this rank in it -- on the
    # length the step decided, which is what the group's dimension was sized for.
    NUM_SPEC = 2

    @classmethod
    def _runner(cls, monkeypatch, *, has_drafter):
        # has_drafter=False simulates a non-last PP rank: the drafter is None
        # there, and the draft leg has to skip cleanly.
        attrs = dict(
            max_num_tokens=64,
            max_num_reqs=8,
            speculative_config=SimpleNamespace(),
            num_spec_tokens=cls.NUM_SPEC,
            query_start_loc_np=np.zeros(16, dtype=np.int32),
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            input_batch=SimpleNamespace(num_tokens_no_spec=np.zeros(8, dtype=np.int32)),
            seq_lens_np=np.zeros(8, dtype=np.int32),
            model_config=SimpleNamespace(dtype=torch.float16),
            device=torch.device("cpu"),
            vllm_config=SimpleNamespace(),
            kv_cache_bases=None,
            input_stager=SimpleNamespace(
                stage=lambda **k: SimpleNamespace(as_kwargs=lambda: {})
            ),
            model_executable=lambda **k: None,
        )
        drafter = MagicMock(spec=RBLNEagleProposer) if has_drafter else None
        attrs["drafter"] = drafter
        runner = _make_runner_stub(**attrs)
        # use_wrapped_compute_logits is a property (no setter); override on the
        # class. is_prefill=False here so it only needs to not raise.
        monkeypatch.setattr(RBLNModelRunner, "use_wrapped_compute_logits", False)
        # Stub the model body so _dummy_run reaches the drafter leg with no NPU.
        # The decided shape: bucket 2 at this step's own query length (nothing here
        # is idle-adopted), so 8 padded tokens.
        # The idle case decides query length 1 -- what a rank beside a prefilling
        # peer gets, its own -- which is not the speculative length, so a draft run
        # on anything but the decision is visible here.
        monkeypatch.setattr(
            runner,
            "_determine_batch_execution_and_padding",
            lambda nr, nt, idle, pinned_num_tokens_padded=None: _resolved_batch(
                num_reqs_padded=2, query_len=nt // nr, num_tokens_padded=8
            ),
        )
        monkeypatch.setattr(
            runner,
            "_get_cumsum_and_arange",
            lambda x: (np.cumsum(x, dtype=np.int32), None),
        )
        monkeypatch.setattr(
            runner, "_build_attention_metadata", lambda **k: (object(), None)
        )
        monkeypatch.setattr(mr, "set_forward_context", lambda *a, **k: nullcontext())
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
        )
        monkeypatch.setattr(mr, "build_kv_cache_forward_context_kwargs", lambda b: {})
        return runner, drafter

    def test_idle_draft_runs_the_decided_length(self, monkeypatch):
        # Beside a prefilling peer the step decides this rank's own single token,
        # and the group's token dimension is sized for that. Running the draft at
        # the speculative length instead would stage past it.
        runner, drafter = self._runner(monkeypatch, has_drafter=True)
        runner._dummy_run(1, 1, is_prefill=False, warmup=False)
        drafter.dummy_run.assert_called_once_with(1, 1, False)

    @pytest.mark.parametrize("query_len", [1, 1 + NUM_SPEC])
    def test_warmup_compiles_the_draft_at_every_query_length(
        self, monkeypatch, query_len
    ):
        # Both decode lengths reach the draft. Query length 1 is the one a step
        # forced to no-spec runs, and compiling only the spec length leaves that
        # step to compile its own graph while it serves.
        runner, drafter = self._runner(monkeypatch, has_drafter=True)
        runner._dummy_run(2, query_len, is_prefill=False, warmup=True)
        # warmup path keeps the num_padded_tokens kwarg (draft's own pad target).
        drafter.dummy_run.assert_called_once_with(
            2, query_len, False, num_padded_tokens=None
        )

    def test_no_drafter_skips_cleanly(self, monkeypatch):
        # A non-last PP rank reaches here with self.drafter None, and the draft leg
        # has to skip rather than treat it as a proposer.
        runner, drafter = self._runner(monkeypatch, has_drafter=False)
        assert drafter is None
        runner._dummy_run(1, 1, is_prefill=False, warmup=False)  # must not raise


class TestDummyRunPPIntermediateTensors:
    # A non-first PP rank builds empty intermediate tensors for the dummy step at
    # the group-bucket batch (num_reqs_padded), since the stager passes them
    # through unpadded -- num_reqs would undersize them when a peer forced the
    # bucket up.
    HIDDEN = 4

    def _runner(self, monkeypatch, *, num_reqs_padded, query_len):
        captured: dict = {}

        def make_empty(batch_size, dtype, device):
            return {"h": torch.zeros(batch_size, self.HIDDEN, dtype=dtype)}

        def stage(**kwargs):
            captured["intermediate_tensors"] = kwargs["intermediate_tensors"]
            captured["layout"] = kwargs["layout"]
            return SimpleNamespace(as_kwargs=lambda: {})

        runner = _make_runner_stub(
            model=SimpleNamespace(make_empty_intermediate_tensors=make_empty),
            model_config=SimpleNamespace(dtype=torch.float16),
            device=torch.device("cpu"),
            max_num_tokens=64,
            max_num_reqs=8,
            speculative_config=None,
            # Gates the drafter leg; __init__ always sets it.
            drafter=None,
            query_start_loc_np=np.zeros(16, dtype=np.int32),
            input_ids=torch.zeros(64, dtype=torch.int32),
            positions=torch.zeros(64, dtype=torch.int64),
            input_batch=SimpleNamespace(num_tokens_no_spec=np.zeros(8, dtype=np.int32)),
            seq_lens_np=np.zeros(8, dtype=np.int32),
            kv_cache_bases=None,
            vllm_config=SimpleNamespace(),
            input_stager=SimpleNamespace(stage=stage),
            model_executable=lambda **k: None,
        )
        monkeypatch.setattr(RBLNModelRunner, "use_wrapped_compute_logits", False)
        monkeypatch.setattr(
            runner,
            "_determine_batch_execution_and_padding",
            lambda nr, nt, idle, pinned_num_tokens_padded=None: _resolved_batch(
                num_reqs_padded=num_reqs_padded,
                query_len=query_len,
                num_tokens_padded=num_reqs_padded * query_len,
            ),
        )
        monkeypatch.setattr(
            runner,
            "_get_cumsum_and_arange",
            lambda x: (np.cumsum(x, dtype=np.int32), None),
        )
        monkeypatch.setattr(
            runner, "_build_attention_metadata", lambda **k: (object(), None)
        )
        monkeypatch.setattr(mr, "set_forward_context", lambda *a, **k: nullcontext())
        # Non-first PP rank -> the else branch builds intermediate tensors.
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False)
        )
        monkeypatch.setattr(mr, "build_kv_cache_forward_context_kwargs", lambda b: {})
        return runner, captured

    def test_idle_intermediate_tensors_use_group_bucket(self, monkeypatch):
        # DP-idle (warmup=False): this rank stages num_reqs=1 but a peer forced
        # bucket 8, so the empty intermediate tensors must be (8, qlen, hidden) --
        # num_reqs_padded, not num_reqs=1 -- to match the compiled PP graph.
        runner, captured = self._runner(monkeypatch, num_reqs_padded=8, query_len=1)
        runner._dummy_run(1, 1, is_prefill=False, warmup=False)
        assert captured["intermediate_tensors"]["h"].shape == (8, 1, self.HIDDEN)
        assert captured["layout"].num_reqs == 1
        assert captured["layout"].num_reqs_padded == 8
