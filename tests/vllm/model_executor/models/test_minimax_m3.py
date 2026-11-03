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


import math
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("has_tower", [False, True])
def test_minimax_m3_encoder_compile_requires_local_tower(monkeypatch, has_tower):
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    model = m3.RBLNMiniMaxM3SparseForConditionalGeneration.__new__(
        m3.RBLNMiniMaxM3SparseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    # Non-first PP ranks and text-only configurations own no vision tower.
    model.vision_tower = torch.nn.Identity() if has_tower else None
    encoder = torch.nn.Identity()
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(m3, "_RBLNVisionEncoder", build_encoder)
    runtime = Mock()
    compile_fn = Mock(return_value=runtime)
    model._mm_encoder = None
    model.mm_embeds_dtype = torch.bfloat16

    model.compile_mm_encoder(compile_fn)

    if has_tower:
        build_encoder.assert_called_once_with(model.vision_tower, torch.bfloat16)
        compile_fn.assert_called_once_with(encoder)
        assert model._mm_encoder is runtime
    else:
        build_encoder.assert_not_called()
        compile_fn.assert_not_called()
        assert model._mm_encoder is None


@pytest.mark.parametrize("aux_layers", [(), (0, 2)])
def test_minimax_m3_target_returns_requested_aux_states(monkeypatch, aux_layers):
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    class Decoder(torch.nn.Module):
        def forward(self, positions, hidden_states, residual):
            return hidden_states + 1, torch.full_like(hidden_states, 10)

    backbone = m3.RBLNMiniMaxM3Model.__new__(m3.RBLNMiniMaxM3Model)
    torch.nn.Module.__init__(backbone)
    backbone.layers = torch.nn.ModuleList([Decoder(), Decoder()])
    backbone.start_layer, backbone.end_layer = 0, 2
    backbone.norm = lambda hidden, residual: (hidden + residual, None)
    target = m3.RBLNMiniMaxM3SparseForCausalLM.__new__(
        m3.RBLNMiniMaxM3SparseForCausalLM
    )
    torch.nn.Module.__init__(target)
    target.model = backbone
    target.set_aux_hidden_state_layers(aux_layers)
    monkeypatch.setattr(
        m3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    inputs = torch.zeros(1, 2, 4)

    result = target(
        input_ids=None, positions=torch.tensor([[0, 1]]), inputs_embeds=inputs
    )

    if aux_layers:
        hidden, aux = result
        assert len(aux) == 2
        torch.testing.assert_close(aux[0], inputs)
        torch.testing.assert_close(aux[1], torch.full_like(inputs, 12))
    else:
        hidden = result
    torch.testing.assert_close(hidden, torch.full_like(inputs, 12))


@pytest.mark.parametrize("compiled", [False, True])
def test_image_encoding_reuses_eager_or_compiled_encoder(monkeypatch, compiled):
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    model = m3.RBLNMiniMaxM3SparseForConditionalGeneration.__new__(
        m3.RBLNMiniMaxM3SparseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.vision_tower = torch.nn.Linear(2, 2, bias=False)
    model.vision_tower.spatial_merge_size = 1
    model.vision_tower.vision_model = SimpleNamespace(vision_segment_max_frames=4)
    encoder = torch.nn.Sequential(model.vision_tower)
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(m3, "_RBLNVisionEncoder", build_encoder)
    object.__setattr__(model, "_mm_encoder", encoder if compiled else None)
    model.mm_embeds_dtype = torch.float32
    model._encoder_inputs = Mock(side_effect=lambda pixels, grid, modality: [pixels])
    pixels = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grids = torch.tensor([[1, 1, 1], [1, 1, 2]])

    for _ in range(2):
        outputs = model.embed_multimodal(pixel_values=pixels, image_grid_thw=grids)
        assert len(outputs) == 2
        torch.testing.assert_close(outputs[0], model.vision_tower(pixels[:1]))
        torch.testing.assert_close(outputs[1], model.vision_tower(pixels[1:]))

    if compiled:
        build_encoder.assert_not_called()
    else:
        build_encoder.assert_called_once_with(model.vision_tower, torch.float32)
        assert not encoder.training
    assert model._mm_encoder is encoder
    assert "_mm_encoder" not in model._modules
    assert list(model.state_dict()) == ["vision_tower.weight"]


def test_fp32_vision_encoder_hands_out_model_dtype_embeddings():
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    torch.manual_seed(0)
    patch_weight = torch.randn(4, 1, 2, 3)
    vision_model = SimpleNamespace(
        encoder=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(num_heads_per_partition=1, head_dim=4)
                )
            ]
        ),
        embeddings=SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=patch_weight)
        ),
        t_dim=2,
        h_dim=1,
        w_dim=1,
        pre_layrnorm=torch.nn.LayerNorm(4),
    )
    tower = torch.nn.Module()
    tower.vision_model = vision_model
    tower.multi_modal_projector = torch.nn.Linear(4, 4)
    tower.patch_merge_mlp = torch.nn.Linear(4, 4)
    encoder = m3._RBLNVisionEncoder(tower, torch.bfloat16)
    # No encoder layers: only the dtype the embeddings leave in is under test.
    vision_model.encoder.layers = []
    pixels = torch.randn(3, 6)

    out = encoder(pixels, torch.ones(3, 4), torch.zeros(3, 4), torch.zeros(1, 1, 1, 3))

    hidden = vision_model.pre_layrnorm(pixels @ patch_weight.reshape(4, -1).T)
    expected = tower.patch_merge_mlp(tower.multi_modal_projector(hidden))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected.to(torch.bfloat16))


def _image_info(video_kwargs=None, **kwargs):
    from vllm.transformers_utils.processors.minimax_m3 import (
        MiniMaxM3VLImageProcessor,
        MiniMaxM3VLVideoProcessor,
    )

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    processor = SimpleNamespace(
        image_processor=MiniMaxM3VLImageProcessor(**kwargs),
        video_processor=MiniMaxM3VLVideoProcessor(**(video_kwargs or {})),
    )
    ctx = SimpleNamespace(
        get_hf_processor=lambda *args, **kw: processor,
        get_merged_mm_kwargs=lambda kwargs, **kw: kwargs,
    )
    return m3.RBLNMiniMaxM3SparseForConditionalGeneration._processor_factory.info(ctx)


def _build_with_patch_buckets(
    monkeypatch,
    buckets,
    *,
    max_side=672,
    info=None,
    limits=None,
    segment_frames=4,
    is_first_rank=True,
):
    import vllm_rbln.model_executor.models.minimax_m3 as m3
    from vllm_rbln.config import RBLNConfig

    cls = m3.RBLNMiniMaxM3SparseForConditionalGeneration
    info = info or _image_info(max_pixels=max_side**2)
    limits = limits or {"image": 1, "video": 0}
    monkeypatch.setattr(
        m3.MULTIMODAL_REGISTRY,
        "create_processor",
        lambda model_config: SimpleNamespace(info=info),
    )
    monkeypatch.setattr(
        m3, "get_pp_group", lambda: SimpleNamespace(is_first_rank=is_first_rank)
    )

    def tower(**kw):
        module = torch.nn.Module()
        module.vision_model = SimpleNamespace(vision_segment_max_frames=segment_frames)
        return module

    monkeypatch.setattr(m3._vision_tower, "MiniMaxVLVisionModel", tower)
    monkeypatch.setattr(
        m3,
        "init_vllm_registered_model",
        lambda **kw: SimpleNamespace(make_empty_intermediate_tensors=None),
    )
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, c: nullcontext())
    config = SimpleNamespace(
        vision_config={"vision_segment_max_frames": segment_frames},
        text_config=SimpleNamespace(hidden_size=8),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
            max_model_len=4096,
            multimodal_config=SimpleNamespace(get_limit_per_prompt=limits.get),
        ),
        quant_config=None,
        additional_config=RBLNConfig(mm_encoder_patch_buckets=buckets),
    )
    return cls(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("configured", "buckets"),
    [
        # Unset: one bucket, the processor's largest image (672 px: 48 x 48).
        ({}, [2304]),
        ({"image": []}, [2304]),
        ({"image": [2304, 1024]}, [1024, 2304]),
        ({"image": [1024, 2400]}, [1024, 2400]),
    ],
)
def test_patch_buckets_default_to_the_processors_largest_image(
    monkeypatch, configured, buckets
):
    model = _build_with_patch_buckets(monkeypatch, configured)

    assert model.mm_encoder_patch_buckets == {"image": buckets}


@pytest.mark.parametrize(
    ("limits", "buckets"),
    [
        # A video segment: 4 temporal patches of a 602,112 px frame (3072 each).
        ({"image": 0, "video": 1}, {"video": [12288]}),
        ({"image": 1, "video": 1}, {"image": [2304], "video": [12288]}),
    ],
)
def test_patch_buckets_default_to_one_per_enabled_modality(
    monkeypatch, limits, buckets
):
    model = _build_with_patch_buckets(monkeypatch, {}, limits=limits)

    assert model.mm_encoder_patch_buckets == buckets


def test_patch_buckets_must_hold_a_video_segment(monkeypatch):
    with pytest.raises(ValueError, match="must reach 12288 patches"):
        _build_with_patch_buckets(
            monkeypatch, {"video": [2304, 3072]}, limits={"image": 1, "video": 1}
        )


def test_video_buckets_do_not_change_image_selection(monkeypatch, vision_encoder_model):
    model = _build_with_patch_buckets(
        monkeypatch,
        {"image": [1024, 2304], "video": [512, 12288]},
        limits={"image": 1, "video": 1},
    )
    model.vision_tower = vision_encoder_model.vision_tower
    pixels = torch.ones(16, 6)

    image, *_ = model._encoder_inputs(pixels, [1, 4, 4], "image")
    video, *_ = model._encoder_inputs(pixels, [1, 4, 4], "video")

    assert image.shape[0] == 1024
    assert video.shape[0] == 512


def test_image_buckets_allow_default_video_buckets(monkeypatch):
    model = _build_with_patch_buckets(
        monkeypatch, {"image": [1024, 2304]}, limits={"image": 1, "video": 1}
    )
    assert model.mm_encoder_patch_buckets == {"image": [1024, 2304], "video": [12288]}


@pytest.mark.parametrize("image_limit", [0, 1])
@pytest.mark.parametrize("is_first_rank", [False, True])
def test_unsegmented_video_is_rejected_at_startup(
    monkeypatch, image_limit, is_first_rank
):
    with pytest.raises(ValueError, match="vision_segment_max_frames=None"):
        _build_with_patch_buckets(
            monkeypatch,
            {},
            limits={"image": image_limit, "video": 1},
            segment_frames=None,
            is_first_rank=is_first_rank,
        )


def test_image_only_allows_no_frame_limit(monkeypatch):
    model = _build_with_patch_buckets(
        monkeypatch, {}, limits={"image": 1, "video": 0}, segment_frames=None
    )
    assert model.mm_encoder_patch_buckets == {"image": [2304]}


@pytest.mark.parametrize(
    ("limits", "is_first_rank"),
    [({"image": 1, "video": 1}, False), ({"image": 0, "video": 0}, True)],
    ids=["later-pp-rank", "text-only"],
)
def test_a_rank_without_an_encoder_has_no_patch_buckets(
    monkeypatch, limits, is_first_rank
):
    model = _build_with_patch_buckets(
        monkeypatch, {}, limits=limits, is_first_rank=is_first_rank
    )
    assert model.vision_tower is None
    assert model.mm_encoder_patch_buckets == {}


def test_video_budget_can_schedule_a_video_enlarged_by_min_pixels():
    from vllm.config import SchedulerConfig
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.encoder_cache_manager import (
        EncoderCacheManager,
        compute_mm_encoder_budget,
    )
    from vllm.v1.request import Request

    from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler

    # 25 temporal patches of a frame enlarged to 1008 x 1008 (1296 tokens).
    seq_len, num_embeds = 32768, 32400
    info = _image_info(video_kwargs={"min_pixels": 1_000_000})
    config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        max_model_len=seq_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    budget, cache_size = compute_mm_encoder_budget(
        config, {"video": info.get_max_video_tokens(seq_len, {"video": 1})}
    )
    request = Request(
        "video",
        [1] * num_embeds,
        SamplingParams(max_tokens=1),
        None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="video",
                identifier="video",
                mm_position=PlaceholderRange(offset=0, length=num_embeds),
            )
        ],
    )
    scheduler = SimpleNamespace(
        is_encoder_decoder=False,
        scheduler_config=config,
        ec_connector=None,
        encoder_cache_manager=EncoderCacheManager(cache_size),
    )

    inputs, num_tokens, remaining, _ = RBLNScheduler._try_schedule_encoder_inputs(
        scheduler, request, 0, 2048, budget
    )

    assert inputs == [0]
    assert num_tokens == 2048
    assert remaining == budget - num_embeds


@pytest.mark.parametrize(
    ("seq_len", "batch_tokens", "disable_mm_chunking"),
    [(4096, 2048, False), (4096, 3840, True), (1048576, 2048, False)],
)
def test_video_budget_preserves_upstream_estimate(
    seq_len, batch_tokens, disable_mm_chunking
):
    from vllm.config import SchedulerConfig
    from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    info = _image_info()
    upstream = m3.MiniMaxM3VLProcessingInfo(info.ctx)
    config = SchedulerConfig(
        max_num_batched_tokens=batch_tokens,
        max_num_seqs=1,
        max_model_len=seq_len,
        enable_chunked_prefill=True,
        disable_chunked_mm_input=disable_mm_chunking,
        is_encoder_decoder=False,
    )
    expected = compute_mm_encoder_budget(
        config, {"video": upstream.get_max_video_tokens(seq_len, {"video": 1})}
    )
    actual = compute_mm_encoder_budget(
        config, {"video": info.get_max_video_tokens(seq_len, {"video": 1})}
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("configured", "max_side", "error"),
    [
        ({"image": [1022, 2304]}, 672, r"\[1022\] must be multiples of 4"),
        ({"image": [1024]}, 672, "must reach 2304 patches"),
        ({"image": [2304, 5184]}, 672, r"\[5184\] would never be used"),
    ],
)
def test_unusable_patch_buckets_are_rejected_at_startup(
    monkeypatch, configured, max_side, error
):
    with pytest.raises(ValueError, match=error):
        _build_with_patch_buckets(monkeypatch, configured, max_side=max_side)


@pytest.fixture
def vision_encoder_model():
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    model = m3.RBLNMiniMaxM3SparseForConditionalGeneration.__new__(
        m3.RBLNMiniMaxM3SparseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    tower = torch.nn.Module()
    tower.anchor = torch.nn.Parameter(torch.zeros(1))
    tower.dtype = torch.float32
    tower.spatial_merge_size = 2
    tower.vision_model = SimpleNamespace(
        embeddings=SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=torch.zeros(8, 6))
        ),
        inv_freq_t=torch.ones(2),
        inv_freq_h=torch.ones(2),
        inv_freq_w=torch.ones(2),
        encoder=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(head_dim=16))]
        ),
        vision_segment_max_frames=2,
    )
    model.vision_tower = tower
    model.mm_encoder_patch_buckets = {"image": [16, 64], "video": [16, 64]}
    return model


def test_a_video_is_encoded_in_segments_of_temporal_patches(vision_encoder_model):
    model = vision_encoder_model
    calls = []

    def encode(pixels, cos, sin, bias):
        calls.append((pixels, cos))
        # One token per 2 x 2 merge: the patch index of its first patch.
        return pixels[::4, :1]

    object.__setattr__(model, "_mm_encoder", encode)
    pixels = torch.arange(20.0).unsqueeze(1).expand(-1, 6)

    (embeds,) = model.embed_multimodal(
        pixel_values_videos=pixels, video_grid_thw=torch.tensor([[5, 2, 2]])
    )

    torch.testing.assert_close(embeds, torch.arange(0.0, 20.0, 4).unsqueeze(1))
    starts = [0, 8, 16]
    for (seg_pixels, cos), start, n in zip(calls, starts, [8, 8, 4], strict=True):
        torch.testing.assert_close(seg_pixels[:n], pixels[start : start + n])
        # The temporal position restarts in each segment (inv_freq_t is 1).
        t_pos = torch.arange(n) // 4
        torch.testing.assert_close(cos[:n, 0], t_pos.float().cos())


@pytest.mark.maybe_use_device
@pytest.mark.parametrize("num_frames, frame_limit", [(1, None), (5, 2)])
def test_encoder_cache_survives_static_output_reuse(
    vision_encoder_model, num_frames, frame_limit
):
    from vllm.platforms import current_platform

    model = vision_encoder_model
    device = torch.device(current_platform.device_type)
    model.vision_tower.to(device)
    model.vision_tower.vision_model.vision_segment_max_frames = frame_limit
    output_buffer = torch.empty(4, 1, device=device)

    def encode(pixels, cos, sin, bias):
        output_buffer.copy_(pixels[::4, :1])
        return output_buffer

    object.__setattr__(model, "_mm_encoder", encode)
    pixels = torch.arange(num_frames * 4.0).unsqueeze(1).expand(-1, 6)
    (embeds,) = model.embed_multimodal(
        pixel_values=pixels, image_grid_thw=torch.tensor([[num_frames, 2, 2]])
    )
    output_buffer.fill_(-1)

    assert embeds.device == output_buffer.device
    torch.testing.assert_close(embeds.cpu(), pixels[::4, :1])


def test_shared_bucket_is_warmed_once(vision_encoder_model):
    model = vision_encoder_model
    model.mm_encoder_patch_buckets = {"image": [16, 64], "video": [64, 128]}
    encoder = Mock()
    object.__setattr__(model, "_mm_encoder", encoder)

    model.warmup_mm_encoder()

    assert [call.args[0].shape[0] for call in encoder.call_args_list] == [16, 64, 128]


@pytest.mark.parametrize("video_first", [False, True])
def test_mixed_modalities_preserve_input_order(vision_encoder_model, video_first):
    model = vision_encoder_model
    object.__setattr__(model, "_mm_encoder", lambda pixels, *args: pixels[::4, :1])
    image = {
        "pixel_values": torch.full((4, 6), 3.0),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    video = {
        "pixel_values_videos": torch.full((12, 6), 7.0),
        "video_grid_thw": torch.tensor([[3, 2, 2]]),
    }
    inputs = {**video, **image} if video_first else {**image, **video}

    outputs = model.embed_multimodal(**inputs)

    expected = [torch.full((1, 1), 3.0), torch.full((3, 1), 7.0)]
    if video_first:
        expected.reverse()
    assert len(outputs) == 2
    for output, target in zip(outputs, expected):
        torch.testing.assert_close(output, target)


def test_an_image_runs_in_the_smallest_bucket_that_holds_it(vision_encoder_model):
    model = vision_encoder_model

    for grid, bucket in (([1, 4, 4], 16), ([1, 4, 6], 64)):
        pixels = torch.ones(math.prod(grid), 6)

        padded, cos, sin, key_bias = model._encoder_inputs(pixels, grid, "image")

        assert padded.shape == (bucket, 6)
        assert cos.shape == sin.shape == (bucket, 16)
        assert (key_bias[..., : math.prod(grid)] == 0).all()
        assert (key_bias[..., math.prod(grid) :] < 0).all()


@pytest.mark.parametrize(
    ("buckets", "max_side"),
    [
        ([2240, 2300, 2304], 672),
        ([2208, 2240, 2304], 672),
        ([2304, 2308, 5184], 1008),
    ],
)
def test_warmup_runs_every_patch_bucket(
    monkeypatch, vision_encoder_model, buckets, max_side
):
    model = _build_with_patch_buckets(
        monkeypatch, {"image": buckets}, max_side=max_side
    )
    model.vision_tower = vision_encoder_model.vision_tower
    warmed = []

    def encode(pixels, cos, sin, bias):
        n = pixels.shape[0]
        assert pixels.shape == (n, 6)
        assert cos.shape == sin.shape == (n, 16)
        assert bias.shape == (1, 1, 1, n)
        assert all(x.dtype == torch.float32 for x in (pixels, cos, sin, bias))
        assert torch.is_inference_mode_enabled()
        warmed.append(n)
        return torch.zeros(n // 4, 8)

    object.__setattr__(model, "_mm_encoder", encode)
    model.warmup_mm_encoder()

    assert warmed == buckets
    # The non-square request must use an already warmed shape, even though the
    # upstream dummy builder clips its width from 700 to 672.
    outputs = model.embed_multimodal(
        pixel_values=torch.ones(2300, 6), image_grid_thw=torch.tensor([[1, 46, 50]])
    )
    assert warmed[-1] in warmed[:-1]
    assert outputs[0].shape == (575, 8)


def test_warmup_without_a_local_vision_tower(vision_encoder_model):
    model = vision_encoder_model
    model.vision_tower = None
    model._mm_encoder = Mock()

    model.warmup_mm_encoder()

    model._mm_encoder.assert_not_called()


@pytest.mark.parametrize(
    ("modality", "frames", "side", "error"),
    [
        ("image", 1, 48, None),
        ("image", 1, 50, "2500 patches a frame, more than the 2304"),
        # A video frame is bounded by the video processor's 602,112 px.
        ("video", 6, 55, None),
        ("video", 6, 56, "3136 patches a frame, more than the 3072"),
    ],
)
def test_a_request_may_not_enlarge_a_frame_past_the_compiled_encoder(
    monkeypatch, modality, frames, side, error
):
    from transformers import BatchFeature

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    grid = torch.tensor([[frames, 32, 32], [frames, side, side]])
    monkeypatch.setattr(
        m3._mm_preprocess.MiniMaxM3VLMultiModalProcessor,
        "_call_hf_processor",
        lambda self, hf_data, hf_kwargs: BatchFeature({f"{modality}_grid_thw": grid}),
    )
    processor = m3._RBLNMiniMaxM3VLMultiModalProcessor.__new__(
        m3._RBLNMiniMaxM3VLMultiModalProcessor
    )
    processor.info = _image_info(max_pixels=672**2)

    if error:
        with pytest.raises(ValueError, match=error):
            processor._call_hf_processor({}, {"max_pixels": 800 * 800})
    else:
        outputs = processor._call_hf_processor({}, {})
        torch.testing.assert_close(outputs[f"{modality}_grid_thw"], grid)


@pytest.mark.parametrize(
    ("kwargs", "size"),
    [
        ({"max_pixels": 640000}, (812, 784)),
        ({"max_long_side_pixel": 1000}, (1000, 1000)),
        ({"max_long_side_pixel": 1000}, (200, 1)),
        ({"max_pixels": 3136}, (97, 1)),
    ],
)
def test_image_bound_covers_the_real_resize(monkeypatch, kwargs, size):
    from PIL import Image

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    info = _image_info(**kwargs)
    outputs = info.get_image_processor()(Image.new("RGB", size), return_tensors="pt")
    patches = int(outputs["image_grid_thw"].prod())
    model = _build_with_patch_buckets(monkeypatch, {}, info=info)
    assert model.mm_encoder_patch_buckets["image"][-1] >= patches
    assert info.get_max_image_tokens() >= patches // 4

    monkeypatch.setattr(
        m3._mm_preprocess.MiniMaxM3VLMultiModalProcessor,
        "_call_hf_processor",
        lambda self, hf_data, hf_kwargs: outputs,
    )
    processor = m3._RBLNMiniMaxM3VLMultiModalProcessor.__new__(
        m3._RBLNMiniMaxM3VLMultiModalProcessor
    )
    processor.info = info
    assert processor._call_hf_processor({}, {}) is outputs


@pytest.mark.parametrize(
    ("kwargs", "size"),
    [
        ({"max_pixels": 640000}, (812, 784)),
        ({"max_long_side_pixel": 1000}, (200, 1)),
        # The video processor enlarges to its own min_pixels, unlike images.
        ({"min_pixels": 1_000_000}, (100, 100)),
    ],
)
def test_video_bound_covers_the_real_resize(monkeypatch, kwargs, size):
    import numpy as np

    info = _image_info(video_kwargs=kwargs)
    frames = np.zeros((10, size[1], size[0], 3), dtype=np.uint8)
    outputs = info.get_video_processor()(
        videos=[frames],
        do_resize=True,
        return_tensors="pt",
        input_data_format="channels_last",
    )
    t, h, w = outputs["video_grid_thw"][0].tolist()
    assert info.get_max_video_frame_tokens() >= h * w // 4
    assert (
        info.get_num_video_tokens(
            image_width=size[0],
            image_height=size[1],
            num_frames=10,
            image_processor=info.get_video_processor(),
            mm_kwargs={},
        )
        == t * h * w // 4
    )

    model = _build_with_patch_buckets(
        monkeypatch, {}, info=info, limits={"image": 0, "video": 1}
    )
    assert model.mm_encoder_patch_buckets["video"][-1] >= 4 * h * w


def test_patch_bucket_accepts_a_non_square_image(monkeypatch):
    from PIL import Image

    info = _image_info()
    outputs = info.get_image_processor()(
        Image.new("RGB", (784, 560)), return_tensors="pt"
    )
    assert int(outputs["image_grid_thw"].prod()) == 2240
    model = _build_with_patch_buckets(
        monkeypatch, {"image": [2208, 2240, 2304]}, info=info
    )
    assert model.mm_encoder_patch_buckets == {"image": [2208, 2240, 2304]}


@pytest.mark.parametrize(
    ("modality", "kwargs"),
    [
        ("image", {"max_pixels": 640000}),
        ("image", {"images_kwargs": {"max_pixels": 640000}}),
        ("image", {"images_kwargs": {"max_long_side_pixel": 1000}}),
        ("video", {"videos_kwargs": {"max_pixels": 640000}}),
    ],
)
def test_startup_limits_match_the_full_processor(tmp_path, modality, kwargs):
    import numpy as np
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing.context import InputProcessingContext
    from vllm.transformers_utils.processors.minimax_m3 import (
        MiniMaxM3VLImageProcessor,
        MiniMaxM3VLVideoProcessor,
    )

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    tokens = [
        "[UNK]",
        "]<]image[>[",
        "]<]video[>[",
        "]<]start of image[>[",
        "]<]end of image[>[",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({s: i for i, s in enumerate(tokens)}, unk_token=tokens[0])
        ),
        unk_token=tokens[0],
        additional_special_tokens=tokens[1:],
    )
    tokenizer.save_pretrained(tmp_path)
    MiniMaxM3VLImageProcessor().save_pretrained(tmp_path)
    MiniMaxM3VLVideoProcessor().save_pretrained(tmp_path)
    mm_config = MultiModalConfig(
        mm_processor_kwargs=kwargs, mm_device_do_normalize=False
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        trust_remote_code=False,
        dtype=torch.float32,
        get_multimodal_config=lambda: mm_config,
    )
    ctx = InputProcessingContext(model_config, tokenizer)
    info = m3.RBLNMiniMaxM3SparseForConditionalGeneration._processor_factory.info(ctx)
    processor = m3._RBLNMiniMaxM3VLMultiModalProcessor.__new__(
        m3._RBLNMiniMaxM3VLMultiModalProcessor
    )
    processor.info = info

    if modality == "image":
        data = {"images": [Image.new("RGB", (812, 784))], "text": tokens[1]}
        max_frame_tokens = info.get_max_image_tokens
    else:
        frames = np.zeros((2, 784, 812, 3), dtype=np.uint8)
        data = {"videos": [frames], "text": tokens[2]}
        max_frame_tokens = info.get_max_video_frame_tokens

    outputs = processor._call_hf_processor(data, {})

    assert outputs[f"{modality}_grid_thw"].tolist() == [[1, 56, 58]]
    assert max_frame_tokens() >= 812


def test_text_config_loads_from_local_checkpoint(tmp_path):
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    config = MiniMaxM3TextConfig(
        architectures=["MiniMaxM3SparseForCausalLM"],
        hidden_size=512,
        num_hidden_layers=2,
        vocab_size=256,
        num_mtp_modules=0,
    )
    config.save_pretrained(tmp_path)

    loaded = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(loaded, MiniMaxM3TextConfig)
    assert loaded.hidden_size == 512
    assert loaded.num_hidden_layers == 2
    assert loaded.vocab_size == 256
    assert loaded.architectures == ["MiniMaxM3SparseForCausalLM"]


def test_pp_ranks_report_the_same_kv_cache_layouts():
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts

    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionBackend,
    )
    from vllm_rbln.v1.attention.backends.minimax_m3 import (
        RBLNMiniMaxM3IndexerBackend,
        RBLNMiniMaxM3SparseBackend,
    )

    msa = [RBLNMiniMaxM3SparseBackend, RBLNMiniMaxM3IndexerBackend]
    assert get_supported_kv_cache_layouts(msa) == get_supported_kv_cache_layouts(
        [RBLNFlashAttentionBackend, *msa]
    )
