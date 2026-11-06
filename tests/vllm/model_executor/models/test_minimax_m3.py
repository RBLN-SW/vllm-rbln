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


# The stages of a pipeline split, run one after another in this process with stub
# layers that fill their output with their own global index plus one, so what
# reaches the last stage reads back as the layers actually captured. The band
# comes from the real split, unaffected by the suite's layer truncation.
_M3_LAYERS = 60
_M3_HIDDEN = 8
# Upstream's default for a 60-layer target when the draft config names no aux
# layers. 30 is where the PP4 split puts stage 2's first layer: stage 1's last
# capture and stage 2's incoming hidden state, so it is the index that can be
# taken twice.
_M3_DEFAULT_AUX = (2, _M3_LAYERS // 2, _M3_LAYERS - 3)
# Every interior PP4 band boundary ([0,15) [15,30) [30,45) [45,60)).
_M3_BOUNDARY_AUX = (15, 30, 45)
# A later stage taking index 0 too would send its own incoming hidden state as
# layer 0.
_M3_EMBEDDING_AUX = (0, 30, 57)


def _m3_stage(rank, pp_size, monkeypatch):
    import vllm_rbln.model_executor.models.minimax_m3 as m3
    import vllm_rbln.v1.spec_decode.eagle3_pp as eagle3_pp
    from vllm_rbln.patches.distributed_utils import original_get_pp_indices

    class Stamp(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.out = float(index + 1)

        def forward(self, positions, hidden_states, residual):
            return torch.zeros_like(hidden_states), torch.full_like(
                hidden_states, self.out
            )

    class Unowned(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError("a stage ran a layer outside its own band")

    start, end = original_get_pp_indices(_M3_LAYERS, rank, pp_size)
    model = m3.RBLNMiniMaxM3Model.__new__(m3.RBLNMiniMaxM3Model)
    torch.nn.Module.__init__(model)
    model.start_layer, model.end_layer = start, end
    model.layers = torch.nn.ModuleList(
        [Stamp(i) if start <= i < end else Unowned() for i in range(_M3_LAYERS)]
    )
    model.embed_input_ids = lambda input_ids: torch.zeros(2, _M3_HIDDEN)
    model.norm = lambda hidden, residual: (hidden + residual, None)
    group = SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == pp_size - 1)
    monkeypatch.setattr(m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle3_pp, "get_pp_group", lambda: group)
    return model


def _m3_layers_in(aux):
    if aux is None:
        return []
    return [
        int(aux[..., block * _M3_HIDDEN].flatten()[0].item())
        for block in range(aux.shape[-1] // _M3_HIDDEN)
    ]


def _m3_pipeline(pp_size, aux_layers, monkeypatch):
    import vllm_rbln.v1.spec_decode.eagle3_pp as eagle3_pp

    carried, sent, keys = None, [], set()
    for rank in range(pp_size):
        model = _m3_stage(rank, pp_size, monkeypatch)
        model._set_aux_hidden_state_layers(aux_layers)
        out = model.forward(
            input_ids=torch.zeros(2, dtype=torch.long),
            positions=torch.zeros(2, dtype=torch.long),
            intermediate_tensors=carried,
        )
        if rank < pp_size - 1:
            sent.append(_m3_layers_in(out.tensors.get(eagle3_pp.AUX_COMBINED)))
            keys |= set(out.tensors)
            carried = out
    return out, sent, keys


@pytest.mark.parametrize("pp_size", [2, 4])
@pytest.mark.parametrize(
    "aux_layers", [_M3_DEFAULT_AUX, _M3_BOUNDARY_AUX, _M3_EMBEDDING_AUX]
)
def test_minimax_m3_last_stage_gets_every_aux_layer_once(
    monkeypatch, pp_size, aux_layers
):
    out, _, _ = _m3_pipeline(pp_size, aux_layers, monkeypatch)

    _, aux = out
    assert _m3_layers_in(torch.cat(aux, dim=-1)) == list(aux_layers)


@pytest.mark.parametrize(
    "pp_size, expected", [(2, [[2, 30]]), (4, [[2], [2, 30], [2, 30]])]
)
def test_minimax_m3_stages_hand_on_what_they_received(monkeypatch, pp_size, expected):
    # A stage that captures none of 2, 30 or 57 (stage 2 at PP4) still has to pass
    # on what it received; the stage just before layer 30 sends 30 as its own capture.
    _, sent, _ = _m3_pipeline(pp_size, _M3_DEFAULT_AUX, monkeypatch)

    assert sent == expected


def test_minimax_m3_last_stage_refuses_a_layer_no_stage_holds(monkeypatch):
    # No stage captures an index past the last layer, so without the check the
    # drafter would read one aux state short.
    with pytest.raises(AssertionError, match="expected 3 aux hidden states"):
        _m3_pipeline(4, (2, 30, _M3_LAYERS + 1), monkeypatch)


@pytest.mark.parametrize("pp_size", [2, 4])
def test_minimax_m3_without_a_draft_sends_no_aux(monkeypatch, pp_size):
    # The receiving stage's placeholder has no aux slot then, and its receive
    # checks the key set exactly.
    _, _, keys = _m3_pipeline(pp_size, (), monkeypatch)

    assert keys == {"hidden_states", "residual"}


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_image_encoding_reuses_eager_or_compiled_encoder(
    monkeypatch, compiled, out_dtype
):
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    model = m3.RBLNMiniMaxM3SparseForConditionalGeneration.__new__(
        m3.RBLNMiniMaxM3SparseForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.vision_tower = torch.nn.Linear(2, 2, bias=False)
    model.vision_tower.spatial_merge_size = 1
    model.mm_embeds_dtype = out_dtype
    model.vision_tower.vision_model = SimpleNamespace(vision_segment_max_frames=4)
    encoder = torch.nn.Sequential(model.vision_tower)
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(m3, "_RBLNVisionEncoder", build_encoder)
    object.__setattr__(model, "_mm_encoder", encoder if compiled else None)
    model._encoder_inputs = Mock(side_effect=lambda pixels, grid, modality: [pixels])
    pixels = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grids = torch.tensor([[1, 1, 1], [1, 1, 2]])

    for _ in range(2):
        outputs = model.embed_multimodal(pixel_values=pixels, image_grid_thw=grids)
        assert len(outputs) == 2
        torch.testing.assert_close(
            outputs[0], model.vision_tower(pixels[:1]).to(out_dtype)
        )
        torch.testing.assert_close(
            outputs[1], model.vision_tower(pixels[1:]).to(out_dtype)
        )
        assert model.vision_tower.weight.dtype == torch.float32

    if compiled:
        build_encoder.assert_not_called()
    else:
        build_encoder.assert_called_once_with(model.vision_tower, out_dtype)
        assert not encoder.training
    assert model._mm_encoder is encoder
    assert "_mm_encoder" not in model._modules
    assert list(model.state_dict()) == ["vision_tower.weight"]


@pytest.mark.parametrize("ec_consumer", [False, True])
def test_an_image_reaches_an_ec_consumer_as_its_grid(monkeypatch, ec_consumer):
    import vllm_rbln.model_executor.models.minimax_m3 as m3

    # The registered processing info, without a checkpoint behind it.
    info_cls = m3.RBLNMiniMaxM3SparseForConditionalGeneration._processor_factory.info
    monkeypatch.setattr(
        info_cls, "get_image_processor", lambda self: SimpleNamespace(merge_size=2)
    )
    monkeypatch.setattr(info_cls, "_get_expected_hidden_size", lambda self: None)
    monkeypatch.setattr(
        info_cls, "allow_missing_mm_embeddings", property(lambda self: ec_consumer)
    )
    parser = info_cls.__new__(info_cls).get_data_parser()

    # What an EC producer publishes beside the encoder output.
    assert parser.placeholder_metadata_fields("image") == {"image_grid_thw"}

    grid = torch.tensor([[1, 4, 6]])
    embeds = parser.parse_mm_data(
        {"image": {"image_grid_thw": grid, "image_embeds": torch.zeros(6, 8)}}
    )["image"]
    assert embeds.get(0)["image_embeds"].shape == (6, 8)

    grid_only = {"image": {"image_grid_thw": grid}}
    if ec_consumer:
        assert parser.parse_mm_data(grid_only)["image"].get_count() == 1
    else:
        with pytest.raises(ValueError, match="image_embeds"):
            parser.parse_mm_data(grid_only)

    # The chat parser asks for the placeholder by the content type's modality.
    model_cls = m3.RBLNMiniMaxM3SparseForConditionalGeneration
    assert model_cls.get_placeholder_str(
        "image_embeds", 1
    ) == model_cls.get_placeholder_str("image", 1)


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


def _build_with_token_budgets(
    monkeypatch,
    budgets,
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
        additional_config=RBLNConfig(encoder_token_budgets=budgets),
    )
    return cls(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("configured", "merge_size", "buckets"),
    [
        # Unset: one bucket, the processor's largest image (672 px: 48 x 48).
        ([], 2, [2304]),
        ([576, 256], 2, [1024, 2304]),
        ([256, 600], 2, [1024, 2400]),
        ([16, 600], 3, [144, 5400]),
    ],
)
def test_token_budgets_resolve_to_patch_buckets(
    monkeypatch, configured, merge_size, buckets
):
    info = _image_info(merge_size=merge_size)
    model = _build_with_token_budgets(monkeypatch, configured, info=info)

    assert model.mm_encoder_patch_buckets == buckets


@pytest.mark.parametrize(
    ("limits", "buckets"),
    [
        # A video segment: 4 temporal patches of a 602,112 px frame (3072 each).
        ({"image": 0, "video": 1}, [12288]),
        ({"image": 1, "video": 1}, [2304, 12288]),
    ],
)
def test_default_buckets_cover_enabled_modalities(monkeypatch, limits, buckets):
    model = _build_with_token_budgets(monkeypatch, [], limits=limits)

    assert model.mm_encoder_patch_buckets == buckets


def test_token_budgets_must_hold_a_video_segment(monkeypatch):
    with pytest.raises(ValueError, match="must reach 3072 output tokens"):
        _build_with_token_budgets(
            monkeypatch, [576, 768], limits={"image": 1, "video": 1}
        )


@pytest.mark.parametrize(
    ("grid", "expected_bucket"),
    [([1, 4, 4], 1024), ([1, 32, 32], 1024), ([1, 24, 48], 2304)],
)
def test_images_and_video_segments_share_buckets(
    monkeypatch, vision_encoder_model, grid, expected_bucket
):
    model = _build_with_token_budgets(
        monkeypatch,
        [256, 576, 3072],
        limits={"image": 1, "video": 1},
    )
    model.vision_tower = vision_encoder_model.vision_tower
    pixels = torch.ones(math.prod(grid), 6)

    image, *_ = model._encoder_inputs(pixels, grid, "image")
    video, *_ = model._encoder_inputs(pixels, grid, "video")

    assert image.shape[0] == video.shape[0] == expected_bucket


def test_default_buckets_deduplicate_equal_modality_bounds(monkeypatch):
    info = _image_info()
    monkeypatch.setattr(info, "get_max_video_frame_tokens", lambda: 144)
    model = _build_with_token_budgets(
        monkeypatch, [], info=info, limits={"image": 1, "video": 1}
    )
    assert model.mm_encoder_patch_buckets == [2304]


@pytest.mark.parametrize("image_limit", [0, 1])
@pytest.mark.parametrize("is_first_rank", [False, True])
def test_unsegmented_video_is_rejected_at_startup(
    monkeypatch, image_limit, is_first_rank
):
    with pytest.raises(ValueError, match="vision_segment_max_frames=None"):
        _build_with_token_budgets(
            monkeypatch,
            [],
            limits={"image": image_limit, "video": 1},
            segment_frames=None,
            is_first_rank=is_first_rank,
        )


def test_image_only_allows_no_frame_limit(monkeypatch):
    model = _build_with_token_budgets(
        monkeypatch, [], limits={"image": 1, "video": 0}, segment_frames=None
    )
    assert model.mm_encoder_patch_buckets == [2304]


@pytest.mark.parametrize(
    ("limits", "is_first_rank"),
    [({"image": 1, "video": 1}, False), ({"image": 0, "video": 0}, True)],
    ids=["later-pp-rank", "text-only"],
)
def test_a_rank_without_an_encoder_has_no_patch_buckets(
    monkeypatch, limits, is_first_rank
):
    model = _build_with_token_budgets(
        monkeypatch, [], limits=limits, is_first_rank=is_first_rank
    )
    assert model.vision_tower is None
    assert model.mm_encoder_patch_buckets == []


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
    scheduler = object.__new__(RBLNScheduler)
    scheduler.__dict__.update(
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


def test_token_budgets_must_hold_an_image(monkeypatch):
    with pytest.raises(ValueError, match="must reach 576 output tokens"):
        _build_with_token_budgets(monkeypatch, [256])


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
    model.mm_embeds_dtype = torch.float32
    model.mm_encoder_patch_buckets = [16, 64]
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


def test_no_frame_limit_keeps_image_output_storage(vision_encoder_model):
    model = vision_encoder_model
    model.vision_tower.vision_model.vision_segment_max_frames = None
    outputs = []

    def encode(pixels, cos, sin, bias):
        outputs.append(pixels[::4, :1])
        return outputs[-1]

    object.__setattr__(model, "_mm_encoder", encode)
    pixels = torch.arange(4.0).unsqueeze(1).expand(-1, 6)
    (embeds,) = model.embed_multimodal(
        pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]])
    )

    assert len(outputs) == 1
    assert embeds.data_ptr() == outputs[0].data_ptr()
    torch.testing.assert_close(embeds, pixels[::4, :1])


def test_shared_bucket_is_warmed_once(vision_encoder_model):
    model = vision_encoder_model
    model.mm_encoder_patch_buckets = [16, 64, 128]
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
    ("buckets", "max_side", "video_limit"),
    [
        ([2240, 2300, 2304], 672, 0),
        ([2208, 2240, 2304], 672, 0),
        ([2304, 2308, 5184], 1008, 0),
        ([2304, 2308, 5184], 672, 0),
        ([1024, 2304, 12288], 672, 1),
        ([1024, 2304, 12288], 672, 0),
    ],
)
def test_warmup_runs_every_patch_bucket(
    monkeypatch, vision_encoder_model, buckets, max_side, video_limit
):
    model = _build_with_token_budgets(
        monkeypatch,
        [b // 4 for b in buckets],
        max_side=max_side,
        limits={"image": 1, "video": video_limit},
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
    model = _build_with_token_budgets(monkeypatch, [], info=info)
    assert model.mm_encoder_patch_buckets[-1] >= patches
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

    model = _build_with_token_budgets(
        monkeypatch, [], info=info, limits={"image": 0, "video": 1}
    )
    assert model.mm_encoder_patch_buckets[-1] >= 4 * h * w


def test_patch_bucket_accepts_a_non_square_image(monkeypatch):
    from PIL import Image

    info = _image_info()
    outputs = info.get_image_processor()(
        Image.new("RGB", (784, 560)), return_tensors="pt"
    )
    assert int(outputs["image_grid_thw"].prod()) == 2240
    model = _build_with_token_budgets(monkeypatch, [552, 560, 576], info=info)
    assert model.mm_encoder_patch_buckets == [2208, 2240, 2304]


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
