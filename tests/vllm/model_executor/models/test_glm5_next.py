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
