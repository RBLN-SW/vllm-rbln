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
    encoder = torch.nn.Sequential(model.vision_tower)
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(m3, "_RBLNVisionEncoder", build_encoder)
    object.__setattr__(model, "_mm_encoder", encoder if compiled else None)
    model.mm_embeds_dtype = torch.float32
    model._encoder_inputs = Mock(side_effect=lambda pixels, grid: [pixels])
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


def _image_info(**kwargs):
    from vllm.transformers_utils.processors.minimax_m3 import MiniMaxM3VLImageProcessor

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    processor = SimpleNamespace(image_processor=MiniMaxM3VLImageProcessor(**kwargs))
    ctx = SimpleNamespace(
        get_hf_processor=lambda *args, **kw: processor,
        get_merged_mm_kwargs=lambda kwargs, **kw: kwargs,
    )
    return m3.RBLNMiniMaxM3SparseForConditionalGeneration._processor_factory.info(ctx)


def _build_with_patch_buckets(monkeypatch, buckets, *, max_side=672, info=None):
    import vllm_rbln.model_executor.models.minimax_m3 as m3
    from vllm_rbln.config import RBLNConfig

    cls = m3.RBLNMiniMaxM3SparseForConditionalGeneration
    info = info or _image_info(max_pixels=max_side**2)
    monkeypatch.setattr(
        m3.MULTIMODAL_REGISTRY,
        "create_processor",
        lambda model_config: SimpleNamespace(info=info),
    )
    monkeypatch.setattr(m3, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    monkeypatch.setattr(
        m3._vision_tower, "MiniMaxVLVisionModel", lambda **kw: torch.nn.Identity()
    )
    monkeypatch.setattr(
        m3,
        "init_vllm_registered_model",
        lambda **kw: SimpleNamespace(make_empty_intermediate_tensors=None),
    )
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, c: nullcontext())
    config = SimpleNamespace(
        vision_config={}, text_config=SimpleNamespace(hidden_size=8)
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda m: 1),
        ),
        quant_config=None,
        additional_config=RBLNConfig(mm_encoder_patch_buckets=buckets),
    )
    return cls(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("configured", "buckets"),
    [
        # Unset: one bucket, the processor's largest image (672 px: 48 x 48).
        ([], [2304]),
        ([2304, 1024], [1024, 2304]),
        ([1024, 2400], [1024, 2400]),
    ],
)
def test_patch_buckets_default_to_the_processors_largest_image(
    monkeypatch, configured, buckets
):
    model = _build_with_patch_buckets(monkeypatch, configured)

    assert model.mm_encoder_patch_buckets == buckets


@pytest.mark.parametrize(
    ("configured", "max_side", "error"),
    [
        ([1022, 2304], 672, r"\[1022\] must be multiples of 4"),
        ([1024], 672, "must reach 2304 patches"),
        ([2304, 5184], 672, r"\[5184\] would never be used"),
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
    )
    model.vision_tower = tower
    model.mm_encoder_patch_buckets = [16, 64]
    return model


def test_an_image_runs_in_the_smallest_bucket_that_holds_it(vision_encoder_model):
    model = vision_encoder_model

    for grid, bucket in (([1, 4, 4], 16), ([1, 4, 6], 64)):
        pixels = torch.ones(math.prod(grid), 6)

        padded, cos, sin, key_bias = model._encoder_inputs(pixels, grid)

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
    model = _build_with_patch_buckets(monkeypatch, buckets, max_side=max_side)
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


@pytest.mark.parametrize(("side", "rejected"), [(48, False), (50, True)])
def test_a_request_may_not_enlarge_an_image_past_the_compiled_encoder(
    monkeypatch, side, rejected
):
    from transformers import BatchFeature

    import vllm_rbln.model_executor.models.minimax_m3 as m3

    grid = torch.tensor([[1, 32, 32], [1, side, side]])
    monkeypatch.setattr(
        m3._mm_preprocess.MiniMaxM3VLMultiModalProcessor,
        "_call_hf_processor",
        lambda self, hf_data, hf_kwargs: BatchFeature({"image_grid_thw": grid}),
    )
    processor = m3._RBLNMiniMaxM3VLMultiModalProcessor.__new__(
        m3._RBLNMiniMaxM3VLMultiModalProcessor
    )
    processor.info = _image_info(max_pixels=672**2)

    if rejected:
        with pytest.raises(ValueError, match="2500 patches .* than the 2304"):
            processor._call_hf_processor({}, {"max_pixels": 700 * 700})
    else:
        outputs = processor._call_hf_processor({}, {})
        torch.testing.assert_close(outputs["image_grid_thw"], grid)


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
    model = _build_with_patch_buckets(monkeypatch, [], info=info)
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


def test_patch_bucket_accepts_a_non_square_image(monkeypatch):
    from PIL import Image

    info = _image_info()
    outputs = info.get_image_processor()(
        Image.new("RGB", (784, 560)), return_tensors="pt"
    )
    assert int(outputs["image_grid_thw"].prod()) == 2240
    model = _build_with_patch_buckets(monkeypatch, [2208, 2240, 2304], info=info)
    assert model.mm_encoder_patch_buckets == [2208, 2240, 2304]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixels": 640000},
        {"images_kwargs": {"max_pixels": 640000}},
        {"images_kwargs": {"max_long_side_pixel": 1000}},
    ],
)
def test_startup_image_limits_match_the_full_processor(tmp_path, kwargs):
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

    outputs = processor._call_hf_processor(
        {"images": [Image.new("RGB", (812, 784))], "text": tokens[1]}, {}
    )

    assert outputs["image_grid_thw"].tolist() == [[1, 56, 58]]
    assert info.get_max_image_tokens() >= 812


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
