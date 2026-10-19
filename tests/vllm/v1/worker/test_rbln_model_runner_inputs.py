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
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.outputs import SamplerOutput

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.vllm.v1.worker.utils import (
    MAX_MODEL_LEN,
    make_scheduler_output,
    make_speculative_config,
    schedule_new,
)

pytestmark = pytest.mark.maybe_use_device


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
        "computed, values",
        [(0, [0, 0, 11]), (3, [0, 22, 33]), (6, [0, 0, 0])],
    )
    def test_chunk_boundaries_preserve_text_holes_and_request_order(
        self, make_model_runner, monkeypatch, computed, values
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        runner = make_model_runner()
        step = schedule_new(
            "image",
            "text",
            prompt_token_ids=[[1] * 12, [1] * 12],
            num_computed_tokens=[computed, 0],
            num_scheduled_tokens=[3, 3],
        )
        step.scheduled_new_reqs[0].mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-hash",
                mm_position=PlaceholderRange(
                    offset=2,
                    length=5,
                    is_embed=torch.tensor([True, False, True, True, False]),
                ),
            )
        ]
        runner._update_states(step)
        hidden = runner.model_config.get_hidden_size()
        runner.encoder_cache["image-hash"] = (
            torch.tensor([11.0, 22.0, 33.0]).unsqueeze(1).expand(-1, hidden)
        )

        embeds, mask = runner._gather_mm_embeddings(step, num_tokens=6)

        expected = torch.tensor(values + [0, 0, 0], dtype=runner.dtype)
        torch.testing.assert_close(embeds, expected[:, None].expand(-1, hidden))
        torch.testing.assert_close(mask[:, 0], (expected != 0).to(runner.dtype))


class TestWarmupMultimodalEncoder:
    @staticmethod
    def _fake_processor(monkeypatch, pixels, *, image_limit=1):
        """Stand in for the registry's processor: one dummy image of `pixels`."""
        elems = MultiModalFieldConfig.batched("image").build_elems(
            "pixel_values", pixels
        )
        dummy_calls = []

        def get_dummy_mm_inputs(mm_counts, *, scheduler_config=None):
            dummy_calls.append(dict(mm_counts))
            return {
                "mm_kwargs": {
                    "image": [MultiModalKwargsItem({"pixel_values": elems[0]})]
                }
            }

        processor = SimpleNamespace(
            info=SimpleNamespace(allowed_mm_limits={"image": image_limit, "video": 1}),
            get_dummy_mm_inputs=get_dummy_mm_inputs,
        )
        monkeypatch.setattr(
            mr.MULTIMODAL_REGISTRY, "create_processor", lambda model_config: processor
        )
        return dummy_calls

    def test_encodes_the_processors_dummy_image_once(
        self, make_model_runner, monkeypatch
    ):
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
        )
        runner = make_model_runner()
        runner.supports_mm_inputs = True
        pixels = torch.tensor([[[1.0, 2.0]]])
        dummy_calls = self._fake_processor(monkeypatch, pixels)
        encoded = []

        def embed_multimodal(*, pixel_values):
            encoded.append(pixel_values.clone())
            return (pixel_values[0],)

        runner.model = SimpleNamespace(embed_multimodal=embed_multimodal)

        runner._warmup_mm_encoder()

        # One image, the processor's own dummy, through the serving path.
        assert dummy_calls == [{"image": 1}]
        assert len(encoded) == 1
        torch.testing.assert_close(encoded[0], pixels)

    @pytest.mark.parametrize(
        "reason", ["text_only_model", "other_pp_rank", "no_image_limit"]
    )
    def test_skips_when_no_image_can_reach_the_encoder(
        self, make_model_runner, monkeypatch, reason
    ):
        monkeypatch.setattr(
            mr,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=reason != "other_pp_rank"),
        )
        runner = make_model_runner()
        runner.supports_mm_inputs = reason != "text_only_model"
        self._fake_processor(
            monkeypatch,
            torch.ones(1, 1, 2),
            image_limit=0 if reason == "no_image_limit" else 1,
        )

        def embed_multimodal(**kwargs):
            raise AssertionError("nothing should be encoded")

        runner.model = SimpleNamespace(embed_multimodal=embed_multimodal)

        runner._warmup_mm_encoder()


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
