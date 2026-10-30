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

"""The RBLN Gemma4 model's own parts, on CPU and without a checkpoint: the
vision encoder against HF's, the mm prefix query lengths, and the in-graph
merge of image embeddings."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import Gemma4VisionConfig, Gemma4VisionModel
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

import vllm_rbln.model_executor.models.gemma4 as gemma4
from vllm_rbln.model_executor.models.gemma4 import RBLNGemma4ForConditionalGeneration

POOL = 3


class _Identity(nn.Module):
    def forward(self, inputs_embeds):
        return inputs_embeds


def _bare_model(**attrs) -> RBLNGemma4ForConditionalGeneration:
    model = RBLNGemma4ForConditionalGeneration.__new__(
        RBLNGemma4ForConditionalGeneration
    )
    nn.Module.__init__(model)
    object.__setattr__(model, "_mm_encoder", None)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _vision_tower() -> Gemma4VisionModel:
    torch.manual_seed(0)
    config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=2,
        pooling_kernel_size=POOL,
        position_embedding_size=16,
        standardize=True,
    )
    tower = Gemma4VisionModel(config).eval()
    with torch.no_grad():
        for param in tower.parameters():
            param.normal_(std=0.2)
        tower.std_bias.normal_()
        tower.std_scale.uniform_(0.5, 1.5)
    return tower


def _image(width: int, height: int, num_patches: int):
    """A row-major patch grid, padded the way the processor pads it."""
    grid = [(x, y) for y in range(height) for x in range(width)]
    positions = torch.full((num_patches, 2), -1)
    positions[: len(grid)] = torch.tensor(grid)
    pixels = torch.rand(num_patches, 3 * 2 * 2)
    pixels[len(grid) :] = 0
    return pixels, positions


@pytest.mark.parametrize("width, height", [(6, 3), (3, 9), (9, 9)])
def test_the_vision_encoder_matches_hfs(width, height):
    # The encoder pads every image to its bucket and runs attention spelled out;
    # what comes back must be HF's soft tokens for the image alone.
    tower = _vision_tower()
    model = _bare_model(
        vision_tower=tower,
        embed_vision=_Identity(),
        model_dtype=torch.float32,
        mm_encoder_num_patches=90,
    )
    pixels, positions = _image(width, height, num_patches=81)

    with torch.no_grad():
        got = model._encode_vision(pixels, positions)
        expected = tower(
            pixel_values=pixels[None], pixel_position_ids=positions[None]
        ).last_hidden_state

    assert got.shape == (width * height // POOL**2, 32)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_an_image_beyond_the_bucket_is_refused():
    model = _bare_model(
        vision_tower=_vision_tower(),
        embed_vision=_Identity(),
        model_dtype=torch.float32,
        mm_encoder_num_patches=9,
    )
    pixels, positions = _image(6, 3, num_patches=18)

    with pytest.raises(AssertionError, match="9-patch bucket"):
        model._encode_vision(pixels, positions)


@pytest.mark.parametrize(
    "cap, query_lens",
    [(280, [128, 256, 384]), (1120, [128, 256, 384, 640, 1152]), (70, [128])],
)
def test_one_query_len_per_soft_token_budget(cap, query_lens):
    model = _bare_model(mm_soft_token_cap=cap)
    assert model.get_mm_prefix_query_lens() == query_lens


def _processing_info(engine_kwargs, processor_default, default_output_length):
    """Upstream's merge: the request's kwargs over the engine's."""
    image_processor = SimpleNamespace(max_soft_tokens=processor_default)
    vision_config = SimpleNamespace(default_output_length=default_output_length)
    return SimpleNamespace(
        ctx=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(vision_config=vision_config)
            ),
            get_merged_mm_kwargs=lambda kwargs: {**engine_kwargs, **kwargs},
        ),
        get_hf_processor=lambda: SimpleNamespace(image_processor=image_processor),
    )


@pytest.mark.parametrize(
    "engine_kwargs, cap",
    [
        ({}, 280),
        ({"max_soft_tokens": 70}, 70),
        ({"images_kwargs": {"max_soft_tokens": 1120}}, 1120),
    ],
)
def test_the_cap_is_the_engines_budget_else_the_processors(engine_kwargs, cap):
    info = _processing_info(engine_kwargs, 280, 280)
    assert gemma4.mm_soft_token_cap(info) == cap


def test_a_default_output_length_the_processor_ignores_is_refused():
    # HF would cut a 280-token image that upstream counts as 70.
    info = _processing_info({}, 280, 70)
    with pytest.raises(ValueError, match="--mm-processor-kwargs"):
        gemma4.mm_soft_token_cap(info)
    info = _processing_info({"max_soft_tokens": 70}, 280, 70)
    assert gemma4.mm_soft_token_cap(info) == 70


def test_an_unsupported_budget_is_refused():
    info = _processing_info({"max_soft_tokens": 100}, 280, 280)
    with pytest.raises(ValueError, match="not one of"):
        gemma4.mm_soft_token_cap(info)


@pytest.mark.parametrize(
    "num_images, request_kwargs, refused",
    [
        (1, {"max_soft_tokens": 560}, True),
        (1, {"max_soft_tokens": 280}, False),
        (1, {}, False),
        # Video frames take their own budget whatever the request says.
        (0, {"max_soft_tokens": 560}, False),
    ],
)
def test_a_request_budget_above_the_cap_is_refused(
    monkeypatch, num_images, request_kwargs, refused
):
    upstream = Mock(return_value="processed")
    monkeypatch.setattr(
        gemma4.Gemma4MultiModalProcessor, "_apply_hf_processor_main", upstream
    )
    processor = gemma4.RBLNGemma4MultiModalProcessor.__new__(
        gemma4.RBLNGemma4MultiModalProcessor
    )
    processor.info = _processing_info({}, 280, 280)
    items = SimpleNamespace(get_count=lambda modality, strict: num_images)

    if refused:
        with pytest.raises(ValueError, match="--mm-processor-kwargs"):
            processor._apply_hf_processor_main(items, request_kwargs)
        upstream.assert_not_called()
    else:
        assert processor._apply_hf_processor_main(items, request_kwargs) == "processed"


def test_video_frames_are_regrouped_per_video(monkeypatch):
    model = _bare_model()
    monkeypatch.setattr(
        model, "_encode_vision", lambda pixels, positions: pixels[:1].clone()
    )
    frames = torch.arange(5.0).view(5, 1, 1).expand(5, 2, 1)

    videos = model._process_video_input(
        {
            "pixel_values_videos": frames,
            "pixel_position_ids_videos": torch.zeros(5, 2, 2),
            "video_frame_counts": torch.tensor([2, 3]),
        }
    )

    assert [v.flatten().tolist() for v in videos] == [[0.0, 1.0], [2.0, 3.0, 4.0]]


@pytest.mark.parametrize(
    "first_rank, limits, compiled",
    [
        (True, {"image": 1, "video": 0}, True),
        (True, {"image": 0, "video": 1}, True),
        (True, {"image": 0, "video": 0, "audio": 1}, False),
        (False, {"image": 1, "video": 1}, False),
    ],
)
def test_the_encoder_compiles_only_where_the_tower_lives(
    monkeypatch, first_rank, limits, compiled
):
    monkeypatch.setattr(
        gemma4, "get_pp_group", lambda: SimpleNamespace(is_first_rank=first_rank)
    )
    built = nn.Identity()
    monkeypatch.setattr(
        gemma4, "_RBLNGemma4VisionEncoder", lambda tower, embed_vision: built
    )
    model = _bare_model(
        vision_tower=nn.Identity(),
        embed_vision=nn.Identity(),
        multimodal_config=SimpleNamespace(
            get_limit_per_prompt=lambda modality: limits.get(modality, 0)
        ),
    )
    runtime = Mock()
    compile_fn = Mock(return_value=runtime)

    model.compile_mm_encoder(compile_fn)

    if compiled:
        compile_fn.assert_called_once_with(built)
        assert model._mm_encoder is runtime
    else:
        compile_fn.assert_not_called()
        assert model._mm_encoder is None


def test_image_embeddings_replace_their_tokens_in_the_graph(monkeypatch):
    monkeypatch.setattr(
        gemma4, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True)
    )
    seen: dict[str, Any] = {}

    def upstream_forward(self, input_ids, positions, intermediate_tensors, embeds):
        seen.update(input_ids=input_ids, inputs_embeds=embeds)
        return "HIDDEN"

    monkeypatch.setattr(Gemma4ForConditionalGeneration, "forward", upstream_forward)
    model = _bare_model(
        language_model=SimpleNamespace(
            embed_input_ids=lambda ids: ids[..., None].float().expand(*ids.shape, 2)
        )
    )
    input_ids = torch.tensor([[1, 2, 3]])
    mm_embeds = torch.full((1, 3, 2), 9.0)
    mm_mask = torch.tensor([[[0.0], [1.0], [0.0]]])

    out = model(input_ids, None, mm_embeds=mm_embeds, mm_mask=mm_mask)

    assert out == "HIDDEN"
    assert seen["input_ids"] is None
    assert seen["inputs_embeds"][0, :, 0].tolist() == [1.0, 9.0, 3.0]
