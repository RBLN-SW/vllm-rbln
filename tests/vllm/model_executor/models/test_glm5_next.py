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

    build_encoder.assert_called_once_with(model.visual)
    assert not encoder.training
    assert model._mm_encoder is encoder
    assert "_mm_encoder" not in model._modules


@pytest.mark.parametrize("is_first_rank", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_encoder_buckets_are_initialized_and_warmed(
    monkeypatch, is_first_rank, compiled
):
    import vllm_rbln.model_executor.models.glm5_next as glm

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
    monkeypatch.setattr(glm, "VISION_PATCH_BUCKET", 16)
    visual = SimpleNamespace(
        dtype=torch.float32,
        device=torch.device("cpu"),
        patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(3, 5))),
        blocks=[SimpleNamespace(attn=SimpleNamespace(head_dim=4))],
    )
    monkeypatch.setattr(glm, "Glm5NextVisionTransformer", lambda *args, **kw: visual)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(),
                vision_config=SimpleNamespace(rms_norm_eps=1e-6),
                image_token_id=1,
            )
        )
    )
    model = cls(vllm_config=config)
    encoder = Mock()
    encoder.eval.return_value = encoder
    build_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(glm, "_RBLNGlm5NextVisionEncoder", build_encoder)
    compile_fn = Mock(side_effect=lambda module: module)
    if compiled:
        model.compile_mm_encoder(compile_fn)

    model.warmup_mm_encoder()

    if not is_first_rank:
        assert model.mm_encoder_patch_buckets == []
        build_encoder.assert_not_called()
        compile_fn.assert_not_called()
        encoder.assert_not_called()
        return
    assert model.mm_encoder_patch_buckets == [16]
    build_encoder.assert_called_once_with(visual)
    if compiled:
        compile_fn.assert_called_once_with(encoder)
    encoder.assert_called_once()
    pixels, cos, sin, key_bias = encoder.call_args.args
    for actual, expected in zip(
        (pixels, cos, sin, key_bias),
        (
            torch.zeros(16, 5),
            torch.ones(16, 4),
            torch.zeros(16, 4),
            torch.zeros(1, 1, 16),
        ),
    ):
        torch.testing.assert_close(actual, expected)


def test_vision_encoder_builds_its_inputs_from_upstream_rope(monkeypatch):
    import vllm_rbln.model_executor.models.glm5_next as glm

    monkeypatch.setattr(glm, "VISION_PATCH_BUCKET", 16)
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
        SimpleNamespace(visual=visual), pixels, [1, 2, 4]
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
