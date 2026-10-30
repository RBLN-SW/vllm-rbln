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

"""MiniMax-M3's EAGLE3 aux hidden states survive the pipeline split.

Same harness as tests/vllm/patches/test_minimax_m2.py: the stages run in order
in one process with stub layers that stamp their own global index into what
they return, so the values reaching the last stage name the layers actually
harvested.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.model_executor.models.minimax_m3 as m3
import vllm_rbln.v1.spec_decode.eagle3_pp as eagle3_pp
from vllm_rbln.patches.distributed_utils import (
    original_get_pp_indices as get_pp_indices,
)

NUM_LAYERS = 60
HIDDEN = 8
BATCH = 2
# get_eagle3_default_aux_hidden_state_layers() for 60 layers. 30 is the PP4
# boundary between stages 1 and 2 ([0,15) [15,30) [30,45) [45,60)).
AUX_LAYERS = (2, NUM_LAYERS // 2, NUM_LAYERS - 3)
# Every index on a PP4 boundary, so an off-by-one double-counts all three.
BOUNDARY_AUX_LAYERS = (15, 30, 45)
# The embedding output: captured on the first stage only.
EMBEDDING_AUX_LAYERS = (0, 31)


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self.rank_in_group = rank
        self._size = size

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self._size - 1


class _StampingLayer(torch.nn.Module):
    """Return a (hidden, residual) pair that sums to this layer's output index."""

    def __init__(self, global_idx: int) -> None:
        super().__init__()
        self.out_index = global_idx + 1

    def forward(self, positions, hidden_states, residual):
        return (
            torch.zeros_like(hidden_states),
            torch.full_like(hidden_states, float(self.out_index)),
        )


class _UnownedLayer(torch.nn.Module):
    """Stands in for PPMissingLayer: calling it means the stage band is wrong."""

    def forward(self, *args, **kwargs):
        raise AssertionError("a stage ran a layer outside its own band")


def _build_stage(rank: int, pp_size: int, monkeypatch) -> m3.RBLNMiniMaxM3Model:
    model = m3.RBLNMiniMaxM3Model.__new__(m3.RBLNMiniMaxM3Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=NUM_LAYERS)
    start, end = get_pp_indices(NUM_LAYERS, rank, pp_size)
    model.start_layer, model.end_layer = start, end
    model.layers = torch.nn.ModuleList(
        [
            _StampingLayer(i) if start <= i < end else _UnownedLayer()
            for i in range(NUM_LAYERS)
        ]
    )
    # The embedding output stamps 0: the input to layer 0.
    model.embed_input_ids = lambda input_ids: torch.zeros(BATCH, HIDDEN)
    model.norm = lambda hidden, residual: (
        hidden if residual is None else hidden + residual,
        None,
    )
    model.aux_hidden_state_layers = ()
    group = _Group(rank, pp_size)
    monkeypatch.setattr(m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle3_pp, "get_pp_group", lambda: group)
    return model


def _stamped_layers(aux: torch.Tensor | None) -> list[int]:
    if aux is None:
        return []
    return [
        int(aux[..., block * HIDDEN].flatten()[0].item())
        for block in range(aux.shape[-1] // HIDDEN)
    ]


def _run_pipeline(pp_size: int, aux_layers: tuple[int, ...], monkeypatch):
    """Drive every stage in order; return the last output and what each sent."""
    carried = None
    layers_sent = []
    for rank in range(pp_size):
        model = _build_stage(rank, pp_size, monkeypatch)
        model._set_aux_hidden_state_layers(aux_layers)
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        if rank < pp_size - 1:
            layers_sent.append(_stamped_layers(out.tensors.get(eagle3_pp.AUX_COMBINED)))
            carried = out
    return out, layers_sent


@pytest.mark.parametrize("pp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "aux_layers", [AUX_LAYERS, BOUNDARY_AUX_LAYERS, EMBEDDING_AUX_LAYERS]
)
def test_last_stage_receives_every_aux_layer_once_in_order(
    pp_size, aux_layers, monkeypatch
):
    out, _ = _run_pipeline(pp_size, aux_layers, monkeypatch)

    _, aux = out
    assert _stamped_layers(torch.cat(aux, dim=-1)) == list(aux_layers)


def test_a_stage_forwards_what_it_received(monkeypatch):
    # Stage 2 of PP4 spans [30, 45) and captures nothing of (2, 30, 57) past its
    # incoming state; it still has to pass on what stages 0 and 1 sent.
    _, layers_sent = _run_pipeline(4, AUX_LAYERS, monkeypatch)

    assert layers_sent == [[2], [2, 30], [2, 30]]


def test_the_handoff_carries_no_aux_slot_without_eagle3(monkeypatch):
    carried = None
    for rank in range(4):
        model = _build_stage(rank, 4, monkeypatch)
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        if rank < 3:
            assert set(out.tensors) == {"hidden_states", "residual"}
        carried = out

    assert isinstance(out, torch.Tensor)
