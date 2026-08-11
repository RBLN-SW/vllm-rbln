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
# test_rbln_model_runner_states / _inputs / _kv_cache.

import contextlib
from types import SimpleNamespace

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
)

import vllm_rbln.v1.worker.rbln_model_runner as mr
from vllm_rbln.v1.core.rbln_kv_cache_manager import KVCacheCopyOp
from vllm_rbln.v1.worker.bucketing.exponential_bucketing_manager import (
    ExponentialBucketingManager,
)
from vllm_rbln.v1.worker.rbln_model_runner import (
    ExecuteModelState,
    RBLNModelRunner,
    _depad_sampler_output,
    _pad_rows,
    _pad_sampling_metadata,
)


def _make_runner_stub(**attrs):
    # A bare RBLNModelRunner (no __init__); set only the attributes the method
    # under test reads.
    runner = object.__new__(RBLNModelRunner)
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

    def test_pad_rows_clone_when_at_bucket(self):
        t = torch.arange(6).reshape(2, 3)
        out = _pad_rows(t, 2)
        assert torch.equal(out, t)
        assert out is not t

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


class TestPredicates:
    def test_is_prefill_step_reads_scheduler_stamp(self):
        # The is_prefill_step setter is the sole write path; the getter -- the
        # single source of truth -- returns it, so all PP ranks agree.
        r = _make_runner_stub()
        r.is_prefill_step = True
        assert r.is_prefill_step is True
        r.is_prefill_step = False
        assert r.is_prefill_step is False

    def test_is_intermediate_chunked_prefill(self):
        # is_prefill_step (scheduler-stamped step phase) AND
        # discard_request_mask[0].
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
            "aux_hidden_states",
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


class TestDetermineBatchPadding:
    # data_parallel_size == 1 is the covered path (multi-DP needs RBLNDPMetadata
    # collectives -> e2e). The phase is driven via the scheduler-stamped
    # _is_prefill_step (is_prefill_step), not the runner's input_batch.
    @staticmethod
    def _runner(*, is_prefill, bucket=8):
        computed = 0 if is_prefill else 9
        return _make_runner_stub(
            _is_prefill_step=is_prefill,
            bucketing_manager=SimpleNamespace(
                find_decode_batch_bucket=lambda n: bucket
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            specialized_moe_decode=False,
            input_batch=SimpleNamespace(
                num_computed_tokens_cpu=np.array([computed]),
                num_tokens_no_spec=np.array([10]),
            ),
        )

    def test_decode_pads_to_bucket(self):
        r = self._runner(is_prefill=False, bucket=8)
        padded, tok, across = r._determine_batch_padding(3, 30)
        assert padded == 8
        assert tok is None and across is None

    def test_prefill_uses_unpadded(self):
        padded, _, _ = self._runner(is_prefill=True)._determine_batch_padding(3, 30)
        assert padded == 3

    def test_single_dp_returns_no_token_padding(self):
        _, tok, across = self._runner(is_prefill=False)._determine_batch_padding(3, 30)
        assert tok is None and across is None


class TestDummyRunPadding:
    # _dummy_run under DP: the decode layout must carry the group-agreed bucket,
    # not this rank's own count. #894 was the layout keeping num_reqs while the
    # attention metadata already used num_reqs_padded.
    @staticmethod
    def _runner(monkeypatch, *, reqs_across_dp, specialized=True):
        captured: dict = {}

        def fake_across_dp(num_tokens, num_reqs, dp_size, dp_rank, is_prefill):
            reqs = torch.tensor(reqs_across_dp, dtype=torch.int32)
            # The real one returns no per-rank counts while any rank prefills.
            return reqs.clone(), None if is_prefill else reqs

        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
        )
        monkeypatch.setattr(
            mr, "set_forward_context", lambda *a, **kw: contextlib.nullcontext()
        )
        monkeypatch.setattr(
            mr, "build_kv_cache_forward_context_kwargs", lambda *a, **kw: {}
        )
        monkeypatch.setattr(
            mr.RBLNDPMetadata,
            "num_tokens_and_reqs_across_dp",
            staticmethod(fake_across_dp),
        )

        def stage(**kwargs):
            captured["layout"] = kwargs["layout"]
            return SimpleNamespace(as_kwargs=lambda: {})

        runner = _make_runner_stub(
            # Two buckets, so a padded count can differ from a raw one at all.
            bucketing_manager=ExponentialBucketingManager(
                max_batch_size=2, min_batch_size=1, limit=2, step=2
            ),
            parallel_config=SimpleNamespace(data_parallel_size=4, data_parallel_rank=0),
            specialized_moe_decode=specialized,
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
