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
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import ECTransferConfig, ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.vllm.v1.worker.utils import (
    MAX_MODEL_LEN,
    make_scheduler_output,
    make_speculative_config,
    schedule_cached,
    schedule_new,
)

pytestmark = pytest.mark.maybe_use_device


class _InGraphMergeModel(torch.nn.Module):
    merges_mm_embeds_in_graph = True


class TestPreprocessBeforeStaging:
    @pytest.mark.parametrize("is_prefill", [False, True])
    @pytest.mark.parametrize("is_first_rank", [False, True])
    def test_prepared_inputs_remain_cpu_views(
        self, make_model_runner, monkeypatch, is_prefill, is_first_rank
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=is_first_rank)
        )
        runner = make_model_runner()
        runner.is_prefill = is_prefill
        runner.input_ids[:3] = torch.tensor([1, 2, 3], dtype=runner.input_ids.dtype)
        runner.positions[:3] = torch.tensor([0, 1, 2], dtype=runner.positions.dtype)

        intermediate = IntermediateTensors({"hidden_states": torch.ones(1, 3, 4)})
        prepared = runner._preprocess(
            schedule_new("a"),
            1,
            2,
            3,
            torch.tensor([2], dtype=torch.int32),
            intermediate,
        )[0]
        runner.input_ids[0] = 42

        assert prepared.input_ids.tolist() == [[42, 2, 3]]
        assert prepared.input_ids.device.type == "cpu"
        assert prepared.positions.tolist() == [[0, 1, 2]]
        assert (prepared.token_indices is not None) == is_prefill
        assert prepared.intermediate_tensors is (
            None if is_first_rank else intermediate
        )

    def test_decode_repairs_cpu_input_before_staging(
        self, make_model_runner, monkeypatch
    ):
        from tests.vllm.v1.worker.test_async_output import _async_output

        monkeypatch.setattr(
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        runner = make_model_runner()
        runner.use_async_scheduling = True
        runner._update_states(schedule_new("a"))
        state = runner.requests["a"]
        n = state.num_prompt_tokens
        state.output_token_ids.append(-1)
        runner.input_batch.token_ids_cpu[0, n] = -1
        runner.input_batch.num_tokens_no_spec[0] = n + 1
        previous = _async_output(
            torch.tensor([[42]], dtype=torch.int32),
            req_ids=["a"],
            placeholder_pos={"a": n},
        )
        runner._prev_async_output = previous
        order = []
        original_kwargs = runner._init_model_kwargs
        original_readback = previous.get_sampled_token_ids_cpu
        original_stage = runner.input_stager.stage
        original_metadata = runner._build_attention_metadata

        def kwargs():
            order.append("cpu_kwargs")
            return original_kwargs()

        def readback():
            order.append("readback")
            return original_readback()

        def stage(**values):
            order.append("stage")
            assert values["input_ids"].tolist() == [[42]]
            return original_stage(**values)

        def metadata(**values):
            order.append("metadata")
            return original_metadata(**values)

        def forward(**values):
            order.append("forward")
            assert values["input_ids"][0, 0] == 42
            return torch.zeros((1, 1, 4)), torch.zeros((1, 1, 1000)), None

        monkeypatch.setattr(runner, "_init_model_kwargs", kwargs)
        monkeypatch.setattr(previous, "get_sampled_token_ids_cpu", readback)
        monkeypatch.setattr(runner.input_stager, "stage", stage)
        monkeypatch.setattr(runner, "_build_attention_metadata", metadata)
        monkeypatch.setattr(runner, "model_executable", forward, raising=False)

        runner.execute_model(
            schedule_cached(
                req_ids=["a"], num_computed_tokens=[n], num_output_tokens=[1]
            )
        )

        assert order == ["metadata", "cpu_kwargs", "readback", "stage", "forward"]
        assert state.output_token_ids == [42]

    @staticmethod
    def _device_feedback_step(make_model_runner, monkeypatch, num_reqs=4):
        from tests.vllm.v1.worker.test_async_output import _async_output

        monkeypatch.setattr(
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        runner = make_model_runner(async_scheduling=True)
        req_ids = list("abcd")[:num_reqs]
        runner._update_states(
            schedule_new(
                *req_ids, sampling_params=[SamplingParams(temperature=0)] * num_reqs
            )
        )
        positions = {}
        for row, req_id in enumerate(req_ids):
            state = runner.requests[req_id]
            n = positions[req_id] = state.num_prompt_tokens
            state.output_token_ids.append(-1)
            runner.input_batch.token_ids_cpu[row, n] = -1
            runner.input_batch.num_tokens_no_spec[row] = n + 1
        tokens = torch.tensor(
            [[42], [65536], [7], [9]], dtype=torch.int32, device="rbln"
        )
        previous = _async_output(
            tokens[:num_reqs], req_ids=req_ids, placeholder_pos=positions
        )
        runner._prev_async_output = previous
        runner._prev_device_token_ids = tokens
        scheduled = schedule_cached(
            req_ids=req_ids,
            num_computed_tokens=list(positions.values()),
            num_output_tokens=[1] * num_reqs,
        )
        return runner, previous, tokens, scheduled

    @pytest.mark.use_device
    @pytest.mark.parametrize("num_reqs", [3, 4])
    @pytest.mark.parametrize("consume_output_first", [False, True])
    def test_device_feedback_submits_forward_before_readback(
        self, make_model_runner, monkeypatch, consume_output_first, num_reqs
    ):
        runner, previous, tokens, scheduled = self._device_feedback_step(
            make_model_runner, monkeypatch, num_reqs
        )
        expected = [[42], [65536], [7], [9]][:num_reqs]
        if consume_output_first:
            assert previous.get_output().sampled_token_ids == expected
            assert previous._sampled_token_ids is None
        order = []
        original_readback = previous.get_sampled_token_ids_cpu

        def readback():
            order.append("readback")
            return original_readback()

        def forward(**values):
            order.append("forward")
            assert values["input_ids"] is tokens
            for state in runner.requests.values():
                assert state.output_token_ids == [-1]
            return torch.zeros((4, 1, 4)), torch.zeros((4, 1, 1000)), None

        monkeypatch.setattr(previous, "get_sampled_token_ids_cpu", readback)
        monkeypatch.setattr(runner, "model_executable", forward, raising=False)
        runner.execute_model(scheduled)

        assert order == ["forward", "readback"]
        assert [
            state.output_token_ids for state in runner.requests.values()
        ] == expected
        assert runner._device_input_ids is tokens

    @pytest.mark.use_device
    @pytest.mark.parametrize(
        "mismatch", ["reordered", "padding", "missing_placeholder", "rollback"]
    )
    def test_device_feedback_repairs_before_forward_when_ineligible(
        self, make_model_runner, monkeypatch, mismatch
    ):
        runner, previous, tokens, scheduled = self._device_feedback_step(
            make_model_runner, monkeypatch
        )
        if mismatch == "reordered":
            previous.req_ids.reverse()
        elif mismatch == "padding":
            runner._prev_device_token_ids = torch.cat(
                [tokens, tokens.new_zeros((2, 1))]
            )
        elif mismatch == "missing_placeholder":
            previous.placeholder_pos.pop("a")
        else:
            scheduled.scheduled_cached_reqs.num_computed_tokens[0] -= 1
            scheduled.scheduled_cached_reqs.num_output_tokens[0] = 0
        order = []
        original_readback = previous.get_sampled_token_ids_cpu

        def readback():
            order.append("readback")
            return original_readback()

        def forward(**values):
            order.append("forward")
            return torch.zeros((4, 1, 4)), torch.zeros((4, 1, 1000)), None

        monkeypatch.setattr(previous, "get_sampled_token_ids_cpu", readback)
        monkeypatch.setattr(runner, "model_executable", forward, raising=False)
        runner.execute_model(scheduled)

        assert order == ["readback", "forward"]
        assert runner._device_input_ids is None


class TestResolveMultiModalSupport:
    @staticmethod
    def _multimodal(monkeypatch, value: bool) -> None:
        monkeypatch.setattr(
            ModelConfig, "supports_multimodal_inputs", property(lambda self: value)
        )

    def test_a_text_model_stays_off_the_path(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        self._multimodal(monkeypatch, False)
        runner.model = _InGraphMergeModel()

        assert runner._resolve_mm_support() is False

    def test_a_model_that_merges_in_graph_opts_in(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        self._multimodal(monkeypatch, True)
        runner.model = _InGraphMergeModel()

        assert runner._resolve_mm_support() is True

    def test_any_other_multimodal_model_is_refused(
        self, make_model_runner, monkeypatch
    ):
        runner = make_model_runner()
        self._multimodal(monkeypatch, True)
        runner.model = SimpleNamespace(embed_multimodal=lambda **kw: ())

        with pytest.raises(ValueError, match="SupportsRBLNMultiModal"):
            runner._resolve_mm_support()


class TestGatherMultimodalEmbeddings:
    def test_scheduled_images_are_encoded_and_cached_by_hash(
        self, make_model_runner, monkeypatch
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        runner = make_model_runner()
        step = schedule_new("images")
        pixels = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        elems = MultiModalFieldConfig.batched("image").build_elems(
            "pixel_values", pixels
        )
        step.scheduled_new_reqs[0].mm_features = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem({"pixel_values": elem}),
                modality="image",
                identifier=f"image-{i}",
                mm_position=PlaceholderRange(offset=i, length=1),
            )
            for i, elem in enumerate(elems)
        ]
        step.scheduled_encoder_inputs = {"images": [1, 0]}
        runner._update_states(step)
        calls = []

        def embed_multimodal(*, pixel_values):
            calls.append(pixel_values.clone())
            return tuple(pixel_values + 10)

        runner.model = SimpleNamespace(embed_multimodal=embed_multimodal)
        runner._execute_mm_encoder(step)

        assert len(calls) == 1
        torch.testing.assert_close(calls[0], pixels.flip(0))
        torch.testing.assert_close(runner.encoder_cache["image-0"], pixels[0] + 10)
        torch.testing.assert_close(runner.encoder_cache["image-1"], pixels[1] + 10)

    @pytest.mark.parametrize(
        "computed, with_text_holes, values",
        [
            (0, True, [0, 0, 11]),
            (2, True, [11, 0, 22]),
            (3, True, [0, 22, 33]),
            (6, True, [0, 0, 0]),
            (0, False, [0, 0, 11]),
            (3, False, [22, 33, 0]),
        ],
    )
    def test_chunk_boundaries_preserve_text_holes_and_request_order(
        self, make_model_runner, monkeypatch, computed, with_text_holes, values
    ):
        monkeypatch.setattr(
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_last_rank=True, is_first_rank=True),
        )
        runner = make_model_runner()
        step = schedule_new(
            "text",
            "image",
            prompt_token_ids=[[1] * 12, [1] * 12],
            num_computed_tokens=[0, computed],
            num_scheduled_tokens=[3, 3],
        )
        step.scheduled_new_reqs[1].mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-hash",
                mm_position=PlaceholderRange(
                    offset=2,
                    length=5 if with_text_holes else 3,
                    is_embed=(
                        torch.tensor([True, False, True, True, False])
                        if with_text_holes
                        else None
                    ),
                ),
            )
        ]
        runner._update_states(step)
        hidden = runner.model_config.get_hidden_size()
        runner.encoder_cache["image-hash"] = (
            torch.tensor([11.0, 22.0, 33.0]).unsqueeze(1).expand(-1, hidden)
        )

        runner.is_prefill = True
        runner.supports_mm_inputs = True
        _, model_kwargs, layout, _ = runner._preprocess(
            step, 2, 2, 6, torch.tensor([2, 5], dtype=torch.int32)
        )
        mask = model_kwargs["mm_mask"].reshape(6, 1)
        embeds, _ = runner.input_stager.stage_mm_embeds(
            model_kwargs["mm_embeds"],
            model_kwargs["mm_mask"],
            hidden,
            layout,
        )

        expected = torch.tensor([0, 0, 0] + values, dtype=runner.dtype)
        torch.testing.assert_close(
            embeds.cpu()[:, :3].reshape(6, hidden), expected[:, None].expand(-1, hidden)
        )
        torch.testing.assert_close(mask[:, 0], (expected != 0).to(runner.dtype))


class TestWarmupMultimodalEncoder:
    @pytest.mark.parametrize("limits", [{"image": 1, "video": 0}, {"image": 0}])
    def test_warms_the_models_encoder_graphs(
        self, make_model_runner, monkeypatch, limits
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
        )
        runner = make_model_runner()
        runner.supports_mm_inputs = True
        runner.model_config.multimodal_config = MultiModalConfig(
            limit_per_prompt=limits
        )
        warmup = Mock()
        runner.model = SimpleNamespace(warmup_mm_encoder=warmup)

        runner._warmup_mm_encoder()

        warmup.assert_called_once_with()

    @pytest.mark.parametrize(
        "reason",
        ["text_only_model", "other_pp_rank", "no_image_or_video", "ec_consumer"],
    )
    def test_skips_when_nothing_can_reach_the_encoder(
        self, make_model_runner, monkeypatch, reason
    ):
        monkeypatch.setattr(
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=reason != "other_pp_rank"),
        )
        runner = make_model_runner()
        runner.supports_mm_inputs = reason != "text_only_model"
        if reason == "ec_consumer":
            monkeypatch.setattr(
                runner.vllm_config,
                "ec_transfer_config",
                ECTransferConfig(
                    ec_connector="RblnECNixlConnector", ec_role="ec_consumer"
                ),
            )
        limit = 0 if reason == "no_image_or_video" else 1
        runner.model_config.multimodal_config = MultiModalConfig(
            limit_per_prompt={"image": limit, "video": limit}
        )
        warmup = Mock()
        runner.model = SimpleNamespace(warmup_mm_encoder=warmup)

        runner._warmup_mm_encoder()

        warmup.assert_not_called()


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


class TestPrepareInputsReplaysAcceptedDrafts:
    @pytest.mark.parametrize("num_spec_tokens", [2, 3, 5])
    def test_context_end_reports_unsupported_replay_window(
        self, make_model_runner, monkeypatch, num_spec_tokens
    ):
        runner = make_model_runner()
        _decode_ready(
            runner,
            monkeypatch,
            num_spec_tokens=num_spec_tokens,
            num_computed=runner.max_model_len - 2,
        )
        runner.replays_accepted_drafts = True

        with pytest.raises(
            NotImplementedError, match="recurrent-state replay.*max_model_len"
        ) as exc:
            runner._prepare_inputs(
                make_scheduler_output(num_scheduled_tokens={"a": 1}),
                np.array([1], dtype=np.int32),
            )
        assert f"num_speculative_tokens={num_spec_tokens}" in str(exc.value)
        assert f"max_model_len={runner.max_model_len}" in str(exc.value)

    def test_verify_resumes_from_the_previous_steps_state(
        self, make_model_runner, monkeypatch
    ):
        runner = make_model_runner()
        _decode_ready(runner, monkeypatch, num_spec_tokens=2)
        runner.replays_accepted_drafts = True
        verify = make_scheduler_output(
            num_scheduled_tokens={"a": 3}, spec_decode_tokens={"a": [11, 12]}
        )
        runner._prepare_inputs(verify, np.array([3], dtype=np.int32))
        assert runner.state_load_pos[0] == 2

        runner.input_batch.num_computed_tokens_cpu[0] = 5
        runner.input_batch.num_tokens_no_spec[0] = 5
        runner.input_batch.num_accepted_tokens_cpu[0] = 2
        runner._prepare_inputs(verify, np.array([3], dtype=np.int32))
        assert runner.state_load_pos[0] == 5

        attn_metadata, _ = runner._build_attention_metadata(
            num_tokens=3, num_reqs=1, max_query_len=3, num_reqs_padded=2
        )
        assert attn_metadata["layer.0"].num_accepted.view(-1).tolist() == [1, 0]


class TestPrepareInputsFixedWindow:
    # A decode with a model-based drafter always stages num_spec_tokens + 1
    # slots and puts the slack behind the scheduled token, so everything
    # downstream has to follow the token, not the window.
    BLOCK = 1024
    NUM_SPEC = 2

    @pytest.mark.parametrize(
        "num_computed, front_pad",
        [
            # The formula switches on `max_model_len`, not on the block: the
            # window only shifts once it would run past the last KV slot the
            # request will ever own.
            (3, 0),
            (MAX_MODEL_LEN - NUM_SPEC - 1, 0),
            (MAX_MODEL_LEN - NUM_SPEC, 1),
        ],
        ids=["mid_sequence", "last_placement_that_fits", "overshoots_by_one"],
    )
    def test_the_window_is_fixed_and_shifts_only_at_max_model_len(
        self, make_model_runner, monkeypatch, num_computed, front_pad
    ):
        runner = make_model_runner()
        _decode_ready(
            runner,
            monkeypatch,
            num_spec_tokens=self.NUM_SPEC,
            num_computed=num_computed,
        )
        window = self.NUM_SPEC + 1

        logits_indices, _, query_lengths, _ = runner._prepare_inputs(
            make_scheduler_output(num_scheduled_tokens={"a": 1}),
            np.array([1], dtype=np.int32),
        )

        assert query_lengths.tolist() == [window]
        start = num_computed - front_pad
        assert runner.positions[:window].tolist() == list(range(start, start + window))
        assert runner.positions[window - 1] < runner.max_model_len
        assert logits_indices.tolist() == [front_pad]
        assert runner.decode_back_pad_np[0] == self.NUM_SPEC - front_pad
        assert runner.seq_lens[:1].tolist() == [num_computed + 1]

    def test_only_the_request_near_the_end_shifts(self, make_model_runner, monkeypatch):
        """front_pad is per request, so a request at the end of its context must
        not move the rest of the batch."""
        runner = make_model_runner()
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        runner._update_states(schedule_new("a", "b"))
        near_end = MAX_MODEL_LEN - self.NUM_SPEC
        for i, num_computed in enumerate((10, near_end)):
            runner.input_batch.num_computed_tokens_cpu[i] = num_computed
            runner.input_batch.num_tokens_no_spec[i] = num_computed
        monkeypatch.setattr(runner, "num_spec_tokens", self.NUM_SPEC)
        monkeypatch.setattr(
            runner, "speculative_config", make_speculative_config("mtp")
        )
        runner._is_prefill_step = False
        window = self.NUM_SPEC + 1

        _, _, query_lengths, total = runner._prepare_inputs(
            make_scheduler_output(num_scheduled_tokens={"a": 1, "b": 1}),
            np.array([1, 1], dtype=np.int32),
        )

        assert query_lengths.tolist() == [window, window]
        assert runner.positions[:total].tolist() == [10, 11, 12] + list(
            range(near_end - 1, near_end + 2)
        )
        assert runner.decode_back_pad_np[:2].tolist() == [self.NUM_SPEC, 1]

    def test_kept_drafts_sample_before_the_back_padding(
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

    def test_recounts_the_accepted_tokens_every_step(
        self, make_model_runner, monkeypatch
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        runner = make_model_runner(async_scheduling=False)
        runner.replays_accepted_drafts = True
        runner._update_states(schedule_new("req_0", "req_1"))
        batch = runner.input_batch
        batch.num_tokens_no_spec[:2] = [3, 3]
        runner.discard_request_mask[:2] = False

        def step(sampled_token_ids: list[list[int]]) -> list[int]:
            num_tokens = 2 * len(sampled_token_ids[0])
            runner._bookkeeping_sync(
                scheduler_output=make_scheduler_output(
                    num_scheduled_tokens={"req_0": num_tokens // 2, "req_1": 1}
                ),
                sampler_output=SamplerOutput(
                    sampled_token_ids=torch.tensor(
                        sampled_token_ids, dtype=torch.int32, device=runner.device
                    ),
                    logprobs_tensors=None,
                ),
                logits=None,
                hidden_states=torch.zeros(
                    (num_tokens, runner.model_config.get_hidden_size()),
                    dtype=runner.dtype,
                ),
                num_scheduled_tokens=num_tokens,
            )
            assert [
                runner.num_accepted_tokens_by_req[r] for r in ("req_0", "req_1")
            ] == batch.num_accepted_tokens_cpu[:2].tolist()
            return batch.num_accepted_tokens_cpu[:2].tolist()

        assert step([[101, 102], [201, -1]]) == [2, 1]
        assert step([[103], [202]]) == [1, 1]
