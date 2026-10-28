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

    model.compile_mm_encoder(compile_fn)

    if has_tower:
        build_encoder.assert_called_once_with(model.vision_tower)
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
        build_encoder.assert_called_once_with(model.vision_tower)
        assert not encoder.training
    assert model._mm_encoder is encoder
    assert "_mm_encoder" not in model._modules
    assert list(model.state_dict()) == ["vision_tower.weight"]


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
