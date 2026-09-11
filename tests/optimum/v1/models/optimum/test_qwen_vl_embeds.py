# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.base import ModelInputForRBLN
from vllm_rbln.model_executor.models.optimum.model_base import (
    RBLNOptimumMultimodalMixin,
)
from vllm_rbln.model_executor.models.optimum.qwen2_vl import (
    RBLNOptimumQwen2_5_VLForConditionalGeneration as Qwen2_5VL,
)
from vllm_rbln.model_executor.models.optimum.qwen2_vl import split_by_grid_thw
from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNOptimumQwen3VLForConditionalGeneration as Qwen3VL,
)

HIDDEN = 4
IMAGE_TOKEN = 7
# Two images of 1x4x4 and 1x2x4 patches, merged 2x2: 4 and 2 tokens.
GRID = torch.tensor([[1, 4, 4], [1, 2, 4]])
COUNTS = [4, 2]
# 24 patches of 8 values; the schema only checks the rank.
PIXELS = torch.zeros(24, 8)


def _rows(n, base=0.0):
    return torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, HIDDEN) + base


class TestSplitByGridThw:
    def test_cuts_at_the_item_boundaries(self):
        items = split_by_grid_thw(_rows(6), GRID)
        assert [t.shape[0] for t in items] == COUNTS
        assert items[1][0, 0] == 4  # the second item starts where the first ends

    def test_merge_unit_is_read_off_the_output(self):
        # 3x3 merge: 18 + 9 patches become 2 + 1 tokens.
        grid = torch.tensor([[1, 3, 6], [1, 3, 3]])
        assert [t.shape[0] for t in split_by_grid_thw(_rows(3), grid)] == [2, 1]


def _qwen(cls, visual, *, deepstack_layers=0, dtype=torch.float32):
    """Uninitialised model wrapper with only what the embed path touches."""
    obj = cls.__new__(cls)
    obj.model = types.SimpleNamespace(
        visual=visual,
        embed_tokens=lambda ids: torch.zeros(*ids.shape, HIDDEN),
        config=types.SimpleNamespace(
            image_token_id=IMAGE_TOKEN,
            video_token_id=8,
            vision_config=types.SimpleNamespace(
                deepstack_visual_indexes=list(range(deepstack_layers))
            ),
        ),
        rbln_config=types.SimpleNamespace(dtype=dtype),
    )
    return obj


class TestEmbedMultimodal:
    def test_qwen2_5_returns_one_tensor_per_image(self):
        model = _qwen(Qwen2_5VL, visual=lambda px, grid_thw: _rows(6))
        items = model.embed_multimodal(pixel_values=PIXELS, image_grid_thw=GRID)
        assert [t.shape for t in items] == [(4, HIDDEN), (2, HIDDEN)]

    def test_qwen3_packs_deepstack_after_each_items_embeddings(self):
        main = _rows(6)
        deepstack = [_rows(6, 100.0), _rows(6, 200.0)]
        model = _qwen(
            Qwen3VL, visual=lambda px, grid_thw: (main, deepstack), deepstack_layers=2
        )
        items = model.embed_multimodal(pixel_values=PIXELS, image_grid_thw=GRID)
        assert [t.shape for t in items] == [(4, 3 * HIDDEN), (2, 3 * HIDDEN)]
        # item 1, token 0 is main row 4 followed by its two deepstack rows.
        assert (
            items[1][0].tolist() == [4.0] * HIDDEN + [104.0] * HIDDEN + [204.0] * HIDDEN
        )

    def test_no_inputs_means_no_items(self):
        model = _qwen(Qwen2_5VL, visual=None)
        assert model.embed_multimodal() == []


def _prefill_input(tokens, mm_embeds, is_mm_embed):
    return ModelInputForRBLN(
        input_tokens=torch.tensor([tokens]),
        input_positions=torch.arange(len(tokens), dtype=torch.int32).unsqueeze(0),
        block_tables=torch.tensor([0], dtype=torch.int16),
        running_requests_ids=["r0"],
        padded_batch_size=1,
        is_prompt=True,
        mm_embeds=mm_embeds,
        is_mm_embed=torch.tensor([is_mm_embed]),
    )


class TestQwen3Deepstack:
    TOKENS = [1, IMAGE_TOKEN, IMAGE_TOKEN, 2, IMAGE_TOKEN, 3]
    MASK = [False, True, True, False, True, False]

    def _items(self):
        # One 2-token item and one 1-token item, each [n, hidden * (1 + 2 layers)].
        packed = torch.cat([_rows(3), _rows(3, 100.0), _rows(3, 200.0)], dim=-1)
        return [packed[:2], packed[2:]]

    def test_embed_input_ids_scatters_only_the_leading_hidden_block(self):
        model = _qwen(Qwen3VL, visual=None, deepstack_layers=2)
        embeds = model.embed_input_ids(
            torch.tensor([self.TOKENS]),
            self._items(),
            is_multimodal=torch.tensor([self.MASK]),
        )
        assert embeds.shape == (1, 6, HIDDEN)
        assert embeds[0, 1, 0] == 0 and embeds[0, 2, 0] == 1 and embeds[0, 4, 0] == 2
        assert embeds[0, 3].abs().sum() == 0  # text position untouched

    def test_pack_deepstack_places_each_layer_at_the_item_positions(self):
        model = _qwen(Qwen3VL, visual=None, deepstack_layers=2)
        mask, deepstack = model._pack_deepstack(
            _prefill_input(self.TOKENS, self._items(), self.MASK)
        )
        assert mask.tolist() == [self.MASK]
        assert deepstack.shape == (2, 6, HIDDEN)
        assert deepstack[0, 1, 0] == 100 and deepstack[1, 1, 0] == 200
        assert deepstack[0, 4, 0] == 102  # third multimodal token, layer 0
        assert deepstack[:, 3].abs().sum() == 0

    def test_text_only_prefill_has_no_deepstack(self):
        model = _qwen(Qwen3VL, visual=None, deepstack_layers=2)
        assert model._pack_deepstack(_prefill_input([1, 2], [], [False, False])) == (
            None,
            None,
        )


class _RecordingMixin(RBLNOptimumMultimodalMixin):
    """The model must build the prefill from the runner's gathered embeddings
    and never touch the vision encoder itself."""

    def __init__(self):
        self.scattered: tuple | None = None

    def embed_multimodal(self, **kwargs):
        pytest.fail("the model must not run the vision encoder during prefill")

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, **kwargs):
        self.scattered = (multimodal_embeddings, kwargs["is_multimodal"])
        return torch.zeros(input_ids.shape[1], HIDDEN)


def test_prefill_scatters_the_gathered_embeddings():
    model = _RecordingMixin()
    mm_embeds = [_rows(2)]
    mask = [False, True, True]
    out = model.build_prefill_forward_inputs(
        _prefill_input([1, IMAGE_TOKEN, IMAGE_TOKEN], mm_embeds, mask)
    )
    assert model.scattered is not None
    assert model.scattered[0] is mm_embeds
    assert model.scattered[1].tolist() == [mask]
    assert out.inputs_embeds.shape == (3, HIDDEN)


def _feature(modality, offset, grid, second_per_grid_ts=None):
    data = {f"{modality}_grid_thw": types.SimpleNamespace(data=torch.tensor(grid))}
    if second_per_grid_ts is not None:
        data["second_per_grid_ts"] = types.SimpleNamespace(
            data=torch.tensor(second_per_grid_ts)
        )
    return types.SimpleNamespace(
        modality=modality, mm_position=types.SimpleNamespace(offset=offset), data=data
    )


def _mrope_model(cls):
    model = _qwen(cls, visual=None)
    calls = []

    def rope_index(input_ids, mm_token_type_ids, **kwargs):
        calls.append((mm_token_type_ids, kwargs))
        n = input_ids.shape[1]
        return torch.arange(n).expand(3, 1, n), torch.tensor([[5]])

    model.model._get_rope_index_func = rope_index
    return model, calls


class TestMropeInputPositions:
    TOKENS = [1, 8, 8, 2, IMAGE_TOKEN, 3]  # a video item, then an image item

    def test_grids_follow_prompt_order_and_token_types_mark_each_modality(self):
        model, calls = _mrope_model(Qwen2_5VL)
        features = [
            _feature("image", 4, [1, 2, 2]),
            _feature("video", 1, [2, 2, 2], second_per_grid_ts=0.5),
        ]
        positions, delta = model.get_mrope_input_positions(self.TOKENS, features)

        assert positions.shape == (3, 6) and delta == 5
        ((token_types, kwargs),) = calls
        assert token_types.tolist() == [[0, 2, 2, 0, 1, 0]]
        assert kwargs["image_grid_thw"].tolist() == [[1, 2, 2]]
        assert kwargs["video_grid_thw"].tolist() == [[2, 2, 2]]
        assert kwargs["second_per_grid_ts"].tolist() == [0.5]

    def test_qwen3_passes_no_second_per_grid_ts(self):
        model, calls = _mrope_model(Qwen3VL)
        model.get_mrope_input_positions(
            self.TOKENS, [_feature("video", 1, [2, 2, 2], second_per_grid_ts=0.5)]
        )
        ((_, kwargs),) = calls
        assert "second_per_grid_ts" not in kwargs
        assert kwargs["image_grid_thw"] is None


def test_position_embed_zeroes_the_padding_rows():
    model = _qwen(Qwen2_5VL, visual=None)
    model.model._get_position_embeddings = lambda x, positions: torch.ones(
        2, positions.shape[1], 1, positions.shape[2], HIDDEN
    )
    model_input = ModelInputForRBLN(
        input_tokens=torch.zeros(4, 1, dtype=torch.int64),
        input_positions=torch.zeros(4, 1, dtype=torch.int32),
        block_tables=torch.zeros(4, 1, dtype=torch.int16),
        running_requests_ids=["r0", "r1"],
        padded_batch_size=4,
        batch_rows=torch.tensor([3, 0]),
        mrope_positions=torch.zeros(3, 4, 1, dtype=torch.int64),
    )
    embed = model._position_embed(model_input)
    assert embed.shape == (2, 4, 1, 1, HIDDEN)
    assert embed[:, [0, 3]].sum() == 2 * 2 * HIDDEN
    assert embed[:, [1, 2]].sum() == 0
