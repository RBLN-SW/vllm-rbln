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

"""Tests for the input staging buffers (input_stager.py): pad-and-copy into
reused, key-cached buffers, compared on .cpu() (device from current_platform)."""

import pytest
import torch
from vllm.platforms import current_platform

from vllm_rbln.v1.worker.input_stager import (
    InputLayout,
    InputStager,
    StagedModelInputs,
)


def _stager() -> InputStager:
    return InputStager(torch.device(current_platform.device_type))


def _layout(
    *,
    num_reqs: int = 2,
    num_reqs_padded: int = 4,
    query_len: int = 3,
    query_len_padded: int = 8,
    **kwargs,
) -> InputLayout:
    return InputLayout(
        num_reqs=num_reqs,
        num_reqs_padded=num_reqs_padded,
        query_len=query_len,
        query_len_padded=query_len_padded,
        **kwargs,
    )


def _ids(rows: int, cols: int, base: int = 1) -> torch.Tensor:
    # Non-zero, distinct values so they never collide with the pad value.
    return torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols) + base


class TestInputLayout:
    def test_shape_is_padded_dims(self):
        layout = _layout(num_reqs=2, num_reqs_padded=4, query_len=3, query_len_padded=8)
        assert layout.shape == (4, 8)


class TestStagedModelInputs:
    def test_as_kwargs_maps_all_fields(self):
        a, b, c, d, e = (object() for _ in range(5))
        staged = StagedModelInputs(
            input_ids=a,
            positions=b,
            intermediate_tensors=c,
            inputs_embeds=d,
            token_indices=e,
        )
        assert staged.as_kwargs() == {
            "input_ids": a,
            "positions": b,
            "intermediate_tensors": c,
            "inputs_embeds": d,
            "token_indices": e,
        }


@pytest.mark.maybe_use_device
class TestStage:
    def test_copies_into_top_left_and_pads_rest(self):
        stager = _stager()
        layout = _layout(num_reqs=2, num_reqs_padded=4, query_len=3, query_len_padded=8)
        input_ids = _ids(2, 3, base=1)
        positions = _ids(2, 3, base=100)
        staged = stager.stage(input_ids=input_ids, positions=positions, layout=layout)

        expected_ids = torch.zeros(4, 8, dtype=torch.int64)
        expected_ids[:2, :3] = input_ids
        expected_pos = torch.zeros(4, 8, dtype=torch.int64)
        expected_pos[:2, :3] = positions
        assert staged.input_ids.shape == (4, 8)
        assert torch.equal(staged.input_ids.cpu(), expected_ids)
        assert torch.equal(staged.positions.cpu(), expected_pos)

    def test_custom_pad_values(self):
        stager = _stager()
        layout = _layout(input_pad_value=7, position_pad_value=9)
        input_ids = _ids(2, 3, base=1)
        positions = _ids(2, 3, base=100)
        staged = stager.stage(input_ids=input_ids, positions=positions, layout=layout)

        ids = staged.input_ids.cpu()
        pos = staged.positions.cpu()
        # A padding cell holds the custom pad value; the data region is intact.
        assert ids[3, 7] == 7
        assert pos[3, 7] == 9
        assert torch.equal(ids[:2, :3], input_ids)
        assert torch.equal(pos[:2, :3], positions)

    def test_no_padding_when_full(self):
        stager = _stager()
        layout = _layout(num_reqs=4, num_reqs_padded=4, query_len=8, query_len_padded=8)
        input_ids = _ids(4, 8, base=1)
        positions = _ids(4, 8, base=100)
        staged = stager.stage(input_ids=input_ids, positions=positions, layout=layout)
        assert torch.equal(staged.input_ids.cpu(), input_ids)
        assert torch.equal(staged.positions.cpu(), positions)

    def test_buffers_live_on_configured_device(self):
        stager = _stager()
        staged = stager.stage(
            input_ids=_ids(2, 3),
            positions=_ids(2, 3),
            layout=_layout(),
        )
        assert staged.input_ids.device.type == current_platform.device_type
        assert staged.positions.device.type == current_platform.device_type


@pytest.mark.maybe_use_device
class TestBufferReuse:
    def test_same_layout_and_dtype_reuses_buffer(self):
        stager = _stager()
        layout = _layout()
        s1 = stager.stage(input_ids=_ids(2, 3), positions=_ids(2, 3), layout=layout)
        s2 = stager.stage(input_ids=_ids(2, 3), positions=_ids(2, 3), layout=layout)
        assert s1.input_ids is s2.input_ids
        assert s1.positions is s2.positions

    def test_reuse_clears_stale_data(self):
        stager = _stager()
        # Same padded shape (4, 8) and dtypes -> same buffer is reused.
        big = _layout(num_reqs=3, num_reqs_padded=4, query_len=6, query_len_padded=8)
        small = _layout(num_reqs=1, num_reqs_padded=4, query_len=2, query_len_padded=8)
        stager.stage(
            input_ids=torch.full((3, 6), 5, dtype=torch.int64),
            positions=torch.full((3, 6), 5, dtype=torch.int64),
            layout=big,
        )
        staged = stager.stage(
            input_ids=torch.full((1, 2), 9, dtype=torch.int64),
            positions=torch.full((1, 2), 9, dtype=torch.int64),
            layout=small,
        )
        # No stale value from call 1 survives: the whole buffer is the new data
        # over pad (e.g. cell [2, 5] was 5 before, must be 0 now).
        expected = torch.zeros(4, 8, dtype=torch.int64)
        expected[:1, :2] = 9
        assert torch.equal(staged.input_ids.cpu(), expected)

    def test_different_shape_creates_new_buffer(self):
        stager = _stager()
        s1 = stager.stage(
            input_ids=_ids(2, 3),
            positions=_ids(2, 3),
            layout=_layout(query_len_padded=8),
        )
        s2 = stager.stage(
            input_ids=_ids(2, 3),
            positions=_ids(2, 3),
            layout=_layout(query_len_padded=16),
        )
        assert s1.input_ids is not s2.input_ids

    def test_different_input_dtype_creates_new_buffer(self):
        stager = _stager()
        layout = _layout()
        s1 = stager.stage(
            input_ids=torch.ones(2, 3, dtype=torch.int64),
            positions=torch.ones(2, 3, dtype=torch.int64),
            layout=layout,
        )
        s2 = stager.stage(
            input_ids=torch.ones(2, 3, dtype=torch.int32),
            positions=torch.ones(2, 3, dtype=torch.int64),
            layout=layout,
        )
        assert s1.input_ids is not s2.input_ids
        # The buffer honors the input dtype (not a hardcoded one).
        assert s1.input_ids.dtype == torch.int64
        assert s2.input_ids.dtype == torch.int32

    def test_different_positions_dtype_creates_new_buffer(self):
        stager = _stager()
        layout = _layout()
        s1 = stager.stage(
            input_ids=torch.ones(2, 3, dtype=torch.int64),
            positions=torch.ones(2, 3, dtype=torch.int64),
            layout=layout,
        )
        s2 = stager.stage(
            input_ids=torch.ones(2, 3, dtype=torch.int64),
            positions=torch.ones(2, 3, dtype=torch.int32),
            layout=layout,
        )
        # positions.dtype is a distinct component of the buffer key.
        assert s1.positions is not s2.positions
        assert s2.positions.dtype == torch.int32


@pytest.mark.maybe_use_device
class TestTokenIndices:
    @staticmethod
    def _base_kwargs():
        return {
            "input_ids": torch.ones(2, 3, dtype=torch.int64),
            "positions": torch.ones(2, 3, dtype=torch.int64),
            "layout": _layout(),
        }

    def test_none_returns_none(self):
        stager = _stager()
        staged = stager.stage(**self._base_kwargs(), token_indices=None)
        assert staged.token_indices is None

    def test_stages_and_copies_contents(self):
        stager = _stager()
        ti = torch.tensor([3, 1, 4, 1], dtype=torch.int64)
        staged = stager.stage(**self._base_kwargs(), token_indices=ti)
        assert staged.token_indices is not None
        assert staged.token_indices.device.type == current_platform.device_type
        assert torch.equal(staged.token_indices.cpu(), ti)

    def test_reuses_buffer_and_pads_to_padded_batch(self):
        stager = _stager()
        # Same padded batch, four live requests and then two.
        s1 = stager.stage(
            input_ids=torch.ones(4, 3, dtype=torch.int64),
            positions=torch.ones(4, 3, dtype=torch.int64),
            layout=_layout(num_reqs=4),
            token_indices=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        )
        s2 = stager.stage(
            **self._base_kwargs(),
            token_indices=torch.tensor([5, 6], dtype=torch.int64),
        )
        # Same (dtype, num_reqs_padded) -> same buffer object, refreshed to the
        # new values; the shrunk tail is pad, never the last call's indices.
        assert s1.token_indices is s2.token_indices
        assert torch.equal(
            s2.token_indices.cpu(), torch.tensor([5, 6, 0, 0], dtype=torch.int64)
        )

    def test_new_buffer_for_different_padded_batch(self):
        stager = _stager()
        s1 = stager.stage(
            **self._base_kwargs(),
            token_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        s2 = stager.stage(
            **{**self._base_kwargs(), "layout": _layout(num_reqs_padded=8)},
            token_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        assert s1.token_indices is not s2.token_indices
        assert s2.token_indices.shape == (8,)

    def test_new_buffer_for_different_dtype(self):
        stager = _stager()
        s1 = stager.stage(
            **self._base_kwargs(),
            token_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        s2 = stager.stage(
            **self._base_kwargs(),
            token_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        )
        # dtype is a distinct component of the token-indices buffer key.
        assert s1.token_indices is not s2.token_indices


@pytest.mark.maybe_use_device
class TestHiddenStates:
    # The eagle3 drafter's third input, staged into the same padded layout.
    @staticmethod
    def _base_kwargs():
        return {
            "input_ids": torch.ones(2, 3, dtype=torch.int64),
            "positions": torch.ones(2, 3, dtype=torch.int64),
            "layout": _layout(),
        }

    def test_copies_into_top_left_and_reuses_buffer(self):
        stager = _stager()
        hidden = torch.full((2, 3, 4), 5.0)
        s1 = stager.stage(**self._base_kwargs(), hidden_states=hidden)
        expected = torch.zeros(4, 8, 4)
        expected[:2, :3] = hidden
        assert s1.hidden_states.shape == (4, 8, 4)
        assert torch.equal(s1.hidden_states.cpu(), expected)

        # Reused across steps: the pad region must not keep the stale values.
        s2 = stager.stage(
            input_ids=torch.ones(1, 2, dtype=torch.int64),
            positions=torch.ones(1, 2, dtype=torch.int64),
            layout=_layout(num_reqs=1, query_len=2),
            hidden_states=torch.full((1, 2, 4), 9.0),
        )
        assert s2.hidden_states is s1.hidden_states
        expected = torch.zeros(4, 8, 4)
        expected[:1, :2] = 9.0
        assert torch.equal(s2.hidden_states.cpu(), expected)


@pytest.mark.maybe_use_device
class TestPassthrough:
    def test_intermediate_and_embeds_passed_through(self):
        stager = _stager()
        inter = object()
        emb = object()
        staged = stager.stage(
            input_ids=_ids(2, 3),
            positions=_ids(2, 3),
            layout=_layout(),
            intermediate_tensors=inter,
            inputs_embeds=emb,
        )
        assert staged.intermediate_tensors is inter
        assert staged.inputs_embeds is emb
