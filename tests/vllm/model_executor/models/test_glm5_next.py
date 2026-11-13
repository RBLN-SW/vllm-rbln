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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _bare(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def test_load_time_transforms_run_once_after_every_weight(monkeypatch):
    import vllm_rbln.model_executor.models.glm5_next as glm

    calls: list[tuple[int, str]] = []
    layers = []
    for i, attn_cls in enumerate((glm.RBLNGlm5NextKDA, glm.RBLNGlm5NextDSA)):
        attn = _bare(attn_cls)
        attn.prepare_weights = lambda i=i: calls.append((i, "prepare_weights"))
        attn.indexer = SimpleNamespace(
            fold_hadamard=lambda i=i: calls.append((i, "fold_hadamard"))
        )
        layer = _bare(glm.RBLNGlm5NextDecoderLayer)
        layer.self_attn = attn
        layer.expand_hc_scales = lambda i=i: calls.append((i, "expand_hc_scales"))
        layers.append(layer)
    model = _bare(glm.RBLNGlm5NextModel)
    model.config = SimpleNamespace()
    model.layers = torch.nn.ModuleList(layers)
    model.start_layer, model.end_layer = 0, 2
    runs: list[list[str]] = []

    def load(module, config, weights, keep):
        runs.append([name for name, _ in weights if keep(name)])
        return set(runs[-1])

    monkeypatch.setattr(glm, "_load_glm5_next_weights", load)
    causal = _bare(glm.RBLNGlm5NextForCausalLM)
    causal.model = model
    causal.lm_head = torch.nn.Linear(1, 1, bias=False)

    causal.load_weights(
        [
            ("model.language_model.layers.0.a", torch.zeros(1)),
            ("lm_head.weight", torch.zeros(1, 1)),
            ("model.language_model.layers.1.b", torch.zeros(1)),
        ]
    )

    assert runs == [["layers.0.a"], ["layers.1.b"]]
    assert calls == [
        (0, "expand_hc_scales"),
        (0, "prepare_weights"),
        (1, "expand_hc_scales"),
        (1, "fold_hadamard"),
    ]


@pytest.mark.parametrize("compiled", [False, True])
def test_image_encoding_reuses_eager_or_compiled_encoder(monkeypatch, compiled):
    import vllm_rbln.model_executor.models.glm5_next as glm

    cls = glm.RBLNGlm5NextForConditionalGeneration
    model = _bare(cls)
    model.visual = torch.nn.Linear(2, 2, bias=False)
    model.visual.spatial_merge_size = 1
    model.mm_embeds_dtype = torch.float32
    encoder = torch.nn.Sequential(model.visual)
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(glm, "_RBLNGlm5NextVisionEncoder", build_encoder)
    object.__setattr__(model, "_mm_encoder", None)
    if compiled:
        model.compile_mm_encoder(lambda module: module)
    model._encoder_inputs = Mock(side_effect=lambda pixels, grid: [pixels])
    pixels = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    grids = torch.tensor([[1, 1, 1], [1, 1, 2]])

    for _ in range(2):
        outputs = model.embed_multimodal(pixel_values=pixels, image_grid_thw=grids)
        assert len(outputs) == 2
        torch.testing.assert_close(outputs[0], model.visual(pixels[:1]))
        torch.testing.assert_close(outputs[1], model.visual(pixels[1:]))

    build_encoder.assert_called_once_with(model.visual, torch.float32)
    assert not encoder.training
    assert model._mm_encoder is encoder
    assert "_mm_encoder" not in model._modules


def _build_with_token_budgets(monkeypatch, budgets, limits, is_first_rank=True):
    import vllm_rbln.model_executor.models.glm5_next as glm
    from vllm_rbln.config import RBLNConfig

    cls = glm.RBLNGlm5NextForConditionalGeneration

    def init_language_model(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace()

    monkeypatch.setattr(glm.RBLNGlm5NextForCausalLM, "__init__", init_language_model)
    monkeypatch.setattr(cls, "_mark_language_model", lambda *args: nullcontext())
    monkeypatch.setattr(cls, "_mark_tower_model", lambda *args: nullcontext())
    monkeypatch.setattr(
        glm, "get_pp_group", lambda: SimpleNamespace(is_first_rank=is_first_rank)
    )
    visual = SimpleNamespace(
        dtype=torch.float32,
        device=torch.device("cpu"),
        spatial_merge_size=2,
        patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(3, 5))),
        blocks=[SimpleNamespace(attn=SimpleNamespace(head_dim=4))],
    )
    monkeypatch.setattr(glm, "Glm5NextVisionTransformer", lambda *args, **kw: visual)
    monkeypatch.setattr(
        glm.MULTIMODAL_REGISTRY,
        "create_processor",
        lambda model_config: SimpleNamespace(
            info=SimpleNamespace(
                get_max_frame_tokens={"image": 4, "video": 16}.__getitem__
            )
        ),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            multimodal_config=SimpleNamespace(get_limit_per_prompt=limits.get),
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(),
                vision_config=SimpleNamespace(rms_norm_eps=1e-6),
                image_token_id=1,
            ),
        ),
        additional_config=RBLNConfig(encoder_token_budgets=budgets),
    )
    return cls(vllm_config=config)


@pytest.mark.parametrize("is_first_rank", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "limits,budgets,buckets",
    [
        ({"image": 1, "video": 0}, [], [16]),
        ({"image": 0, "video": 1}, [], [64]),
        ({"image": 1, "video": 1}, [], [16, 64]),
        ({"image": 0, "video": 0}, [], []),
        ({"image": 1, "video": 1}, [16, 4, 8, 32], [16, 32, 64, 128]),
    ],
)
def test_encoder_buckets_are_initialized_and_warmed(
    monkeypatch, is_first_rank, compiled, limits, budgets, buckets
):
    import vllm_rbln.model_executor.models.glm5_next as glm

    model = _build_with_token_budgets(monkeypatch, budgets, limits, is_first_rank)
    encoder = Mock()
    encoder.eval.return_value = encoder
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(glm, "_RBLNGlm5NextVisionEncoder", build_encoder)
    compile_fn = Mock(side_effect=lambda module: module)
    if compiled:
        model.compile_mm_encoder(compile_fn)

    model.warmup_mm_encoder()

    if not is_first_rank or not buckets:
        assert model.mm_encoder_patch_buckets == []
        build_encoder.assert_not_called()
        compile_fn.assert_not_called()
        encoder.assert_not_called()
        return
    assert model.mm_encoder_patch_buckets == buckets
    build_encoder.assert_called_once_with(model.visual, torch.bfloat16)
    if compiled:
        compile_fn.assert_called_once_with(encoder)
    assert encoder.call_count == len(buckets)
    for call, bucket in zip(encoder.call_args_list, buckets, strict=True):
        for actual, expected in zip(
            call.args,
            (
                torch.zeros(bucket, 5),
                torch.ones(bucket, 4),
                torch.zeros(bucket, 4),
                torch.zeros(1, 1, bucket),
            ),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("modality,limit", [("image", 4), ("video", 16)])
def test_encoder_budgets_must_cover_the_processor_bound(monkeypatch, modality, limit):
    limits = {"image": int(modality == "image"), "video": int(modality == "video")}
    with pytest.raises(ValueError, match=f"must reach {limit} output tokens"):
        _build_with_token_budgets(monkeypatch, [limit - 1], limits)


def test_vision_encoder_builds_its_inputs_from_upstream_rope(monkeypatch):
    import vllm_rbln.model_executor.models.glm5_next as glm

    visual = SimpleNamespace(
        dtype=torch.float32,
        device=torch.device("cpu"),
        spatial_merge_size=2,
        patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(3, 5))),
        rotary_pos_emb=SimpleNamespace(rotary_dim=4, base=100.0),
    )
    pixels = torch.arange(40, dtype=torch.float32).view(8, 5)

    encoder_inputs = glm.RBLNGlm5NextForConditionalGeneration._encoder_inputs
    padded, cos, sin, key_bias = encoder_inputs(
        SimpleNamespace(visual=visual, mm_encoder_patch_buckets=[16]),
        pixels,
        [1, 2, 4],
    )

    torch.testing.assert_close(padded[:8], pixels)
    assert not padded[8:].any()
    inv_freq = 1.0 / 100.0 ** (torch.tensor([0.0, 2.0]) / 4)
    # (h, w) of each row: the 2 x 4 grid a 2 x 2 block at a time.
    rows = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
        dtype=torch.float32,
    )
    angles = torch.cat(
        [torch.outer(rows[:, 0], inv_freq), torch.outer(rows[:, 1], inv_freq)] * 2,
        dim=-1,
    )
    torch.testing.assert_close(cos[:8], angles.cos())
    torch.testing.assert_close(sin[:8], angles.sin())
    assert (cos[8:] == 1).all() and not sin[8:].any()
    assert not key_bias[..., :8].any() and (key_bias[..., 8:] < -1e4).all()


def test_only_scratch_aux_caches_are_left_out_of_prefix_caching(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import set_current_vllm_config

    import vllm_rbln.model_executor.models.glm5_next as glm
    from tests.vllm.vllm_config import make_vllm_config

    config = make_vllm_config(block_size=128)
    text = config.model_config.hf_text_config
    text.linear_num_heads = 4
    text.linear_head_dim = 128
    text.linear_lower_bound = -5.0
    text.linear_conv_kernel_dim = 4
    text.kv_lora_rank = 512
    text.index_kpool = glm.RBLNGlm5NextIndexerBackend.compress_ratio
    text.index_n_heads = 2
    text.index_head_dim = 32
    text.index_topk = 4 * text.index_kpool
    text.q_lora_rank = 64
    config.model_config.dtype = torch.bfloat16
    config.cache_config.cache_dtype = "fp8"
    config.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    with set_current_vllm_config(config):
        glm.RBLNGlm5NextKDA(text, "model.layers.0.self_attn")
        glm.RBLNGlm5NextIndexer(text, "model.layers.1.self_attn.indexer")

    context = config.compilation_config.static_forward_context
    flags = {
        name: module.get_aux_cache_spec(config).prefix_cacheable
        for name, module in context.items()
        if hasattr(module, "get_aux_cache_spec")
    }
    assert flags == {
        # KDA: two state groups, then the conv ring and a draft cache per group.
        "model.layers.0.self_attn.attn.2": True,
        "model.layers.0.self_attn.attn.3": False,
        "model.layers.0.self_attn.attn.4": False,
        # DSA indexer: the key scales, then the tail ring.
        "model.layers.1.self_attn.attn.2": True,
        "model.layers.1.self_attn.attn.3": False,
    }


@pytest.mark.parametrize("video_first", [False, True])
def test_video_frames_reuse_image_buckets_without_cross_frame_attention(video_first):
    import vllm_rbln.model_executor.models.glm5_next as glm

    model = _bare(glm.RBLNGlm5NextForConditionalGeneration)
    model.visual = SimpleNamespace(
        dtype=torch.float32,
        device=torch.device("cpu"),
        spatial_merge_size=1,
        patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(4, 4))),
        rotary_pos_emb=SimpleNamespace(rotary_dim=2, base=100.0),
    )
    model.mm_encoder_patch_buckets = [4, 8]
    model.mm_embeds_dtype = torch.bfloat16
    calls = []

    def encoder(pixels, cos, sin, bias):
        calls.append(pixels.shape[0])
        weights = (pixels @ pixels.T + bias).softmax(-1)
        return (weights @ pixels).squeeze(0)

    object.__setattr__(model, "_mm_encoder", encoder)
    image = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    video = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 10
    image_inputs = dict(pixel_values=image, image_grid_thw=torch.tensor([[1, 1, 2]]))
    video_inputs = dict(
        pixel_values_videos=video, video_grid_thw=torch.tensor([[2, 1, 5]])
    )
    inputs = video_inputs | image_inputs if video_first else image_inputs | video_inputs

    outputs = model.embed_multimodal(**inputs)

    def reference(pixels):
        return ((pixels @ pixels.T).softmax(-1) @ pixels).to(torch.bfloat16)

    expected = [reference(image), torch.cat([reference(p) for p in video.split(5)])]
    if video_first:
        expected.reverse()
    assert calls == ([8, 8, 4] if video_first else [4, 8, 8])
    for actual, wanted in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize(
    "kwargs,limits",
    [
        ({}, (8000, 30000)),
        ({"max_image_tokens": 64}, (64, 64)),
        (
            {
                "images_kwargs": {"max_image_tokens": 64},
                "videos_kwargs": {"max_image_tokens": 128},
            },
            (64, 128),
        ),
    ],
)
def test_processor_keeps_checkpoint_limits_and_resolves_modality_overrides(
    kwargs, limits
):
    from PIL import Image
    from vllm.config import MultiModalConfig
    from vllm.multimodal.processing import InputProcessingContext
    from vllm.transformers_utils.processors.glm5next import (
        Glm5NextImageProcessor,
        Glm5NextVideoProcessor,
    )

    import vllm_rbln.model_executor.models.glm5_next as glm

    processor = SimpleNamespace(
        image_processor=Glm5NextImageProcessor(max_image_tokens=8000),
        video_processor=Glm5NextVideoProcessor(max_image_tokens=30000),
    )
    mm_config = MultiModalConfig(mm_processor_kwargs=kwargs)
    ctx = InputProcessingContext(
        SimpleNamespace(get_multimodal_config=lambda: mm_config), tokenizer=None
    )
    info = glm.RBLNGlm5NextProcessingInfo(ctx)
    info._glm5_hf_processor = processor

    assert info.get_hf_processor() is processor
    assert processor.image_processor.max_image_tokens == 8000
    assert processor.video_processor.max_image_tokens == 30000
    assert info.get_max_image_tokens() == limits[0]
    for modality, limit in zip(("image", "video"), limits, strict=True):
        assert info.get_max_frame_tokens(modality) == limit
        proc = getattr(processor, f"{modality}_processor")
        pixels = getattr(info, f"_get_{modality}_max_pixels")()
        assert pixels == limit * proc.temporal_patch_size * (proc.patch_size * 2) ** 2
        if modality == "image":
            data = Image.new("RGB", (517, 317))
            call_kwargs = {}
        else:
            data = torch.zeros(4, 3, 317, 517, dtype=torch.uint8)
            call_kwargs = {"do_sample_frames": False}
        merged = ctx.get_merged_mm_kwargs({}, modality=modality)
        if "max_image_tokens" in merged:
            call_kwargs["max_image_tokens"] = merged["max_image_tokens"]
        outputs = proc(data, **call_kwargs, return_tensors="pt")
        _, h, w = outputs[f"{modality}_grid_thw"][0].tolist()
        assert h * w // 4 <= limit


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("width", [4, 8])
def test_request_frame_size_cannot_exceed_startup_bound(monkeypatch, modality, width):
    from transformers import BatchFeature

    import vllm_rbln.model_executor.models.glm5_next as glm

    pixels = "pixel_values" if modality == "image" else "pixel_values_videos"
    frames = 1 if modality == "image" else 10
    outputs = BatchFeature(
        {
            pixels: torch.zeros(frames * 4 * width, 3),
            f"{modality}_grid_thw": torch.tensor([[frames, 4, width]]),
        }
    )
    monkeypatch.setattr(
        glm.Glm5NextMultiModalProcessor,
        "_apply_hf_processor_main",
        lambda *args: outputs,
    )
    processor = glm._RBLNGlm5NextMultiModalProcessor.__new__(
        glm._RBLNGlm5NextMultiModalProcessor
    )
    processor.info = SimpleNamespace(
        get_image_processor=lambda: SimpleNamespace(merge_size=2),
        get_max_frame_tokens=lambda modality: 4,
    )
    if width > 4:
        with pytest.raises(ValueError, match="more than the 16"):
            processor._apply_hf_processor_main(None, {})
    else:
        assert processor._apply_hf_processor_main(None, {}) is outputs


@pytest.mark.parametrize(
    "sample,indices,kwargs,timestamps",
    [
        (True, list(range(30)), {}, [0, 1, 2]),
        (True, list(range(30)), {"fps": 1.0}, [0, 2]),
        (False, list(range(5)), {}, [0, 0, 0]),
        (False, [0, 0, 1, 1, 2], {}, [0, 0, 0]),
    ],
)
def test_video_placeholders_follow_processed_frames(
    sample, indices, kwargs, timestamps
):
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PretrainedConfig, PreTrainedTokenizerFast
    from vllm.config import MultiModalConfig
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.processing import InputProcessingContext
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.transformers_utils.processors.glm5next import (
        Glm5NextImageProcessor,
        Glm5NextProcessor,
        Glm5NextVideoProcessor,
    )

    import vllm_rbln.model_executor.models.glm5_next as glm

    tokens = [
        "[UNK]",
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({s: i for i, s in enumerate(tokens)}, unk_token=tokens[0])
        ),
        unk_token=tokens[0],
        additional_special_tokens=tokens[1:],
    )
    mm_config = MultiModalConfig(mm_processor_kwargs={"max_image_tokens": 128})
    hf_config = PretrainedConfig(
        image_token_id=1,
        video_token_id=2,
        image_start_token_id=3,
        image_end_token_id=4,
        video_start_token_id=5,
        video_end_token_id=6,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    ctx = InputProcessingContext(
        SimpleNamespace(
            model="glm5-video-test",
            hf_config=hf_config,
            dtype=torch.float32,
            multimodal_config=mm_config,
            get_multimodal_config=lambda: mm_config,
        ),
        tokenizer,
    )
    info = glm.RBLNGlm5NextProcessingInfo(ctx)
    info._glm5_hf_processor = Glm5NextProcessor(
        Glm5NextImageProcessor(), tokenizer, Glm5NextVideoProcessor()
    )
    processor = glm._RBLNGlm5NextMultiModalProcessor(
        info, glm.Glm4vDummyInputsBuilder(info)
    )
    frames = len(indices)
    video = np.zeros((frames, 112, 112, 3), dtype=np.uint8)
    metadata = dict(
        total_num_frames=frames,
        fps=10.0,
        duration=frames / 10,
        video_backend="opencv",
        frames_indices=indices,
        do_sample_frames=sample,
    )
    inputs = ProcessorInputs(
        prompt=[5, 2, 6],
        mm_data_items=processor.data_parser.parse_mm_data(
            {"video": [(video, metadata)]}
        ),
        hf_processor_mm_kwargs=kwargs,
        cache=MultiModalProcessorOnlyCache(ctx.model_config),
    )

    for _ in range(2):
        output = processor.apply(inputs, TimingContext(enabled=False))
        item = output["mm_kwargs"]["video"][0]
        grid = item["video_grid_thw"].data
        assert output["prompt_token_ids"].count(1) == int(grid.prod()) // 4
        assert output["prompt_token_ids"].count(3) == int(grid[0])
        assert item["timestamps"].data.tolist() == timestamps
    assert inputs.cache.make_stats().hits == 1
