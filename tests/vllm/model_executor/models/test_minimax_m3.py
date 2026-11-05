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
