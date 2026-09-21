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

"""EAGLE3 aux hidden states survive the pipeline split on an A.X-K2 target.

The stages run sequentially in one process with stub layers, so this needs no
checkpoint and no device. Each stub layer stamps its own global index into the
tensor it returns, which is what makes a stage-local index visible: the values
that reach the last stage are the layer numbers actually harvested.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.patches.axk2.model as patch_module
import vllm_rbln.v1.spec_decode.eagle3_pp as eagle3_pp

# The suite truncates models via VLLM_RBLN_NUM_HIDDEN_LAYERS, and the patched
# get_pp_indices honors it. These stages are stubs, so the truncation has nothing
# to shrink -- what the split must be is the real 61-layer one.
from vllm_rbln.patches.distributed_utils import (  # noqa: E402
    original_get_pp_indices as get_pp_indices,
)

NUM_LAYERS = 61
HIDDEN = 8
BATCH = 2
# The variants ICR-49 is about run pp8; bands are [0,7) [7,14) [14,22) [22,30)
# [30,38) [38,46) [46,54) [54,61).
PP_SIZE = 8
# What skt/A.X-K2-EAGLE3 asks for in its eagle_config, which for this depth is also
# what get_eagle3_default_aux_hidden_state_layers() returns. 30 is stage 4's
# start_layer, the boundary an off-by-one in the split double-counts or drops.
CHECKPOINT_AUX_LAYERS = (2, 30, 58)
# Every index on a band boundary, so a widened capture range double-counts all
# three at once -- and compiles, and passes a short probe.
BOUNDARY_AUX_LAYERS = (7, 30, 54)
# None on a boundary, and one index in a stage that owns no other.
INTERIOR_AUX_LAYERS = (2, 31, 58)


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
    """Return a (hidden, residual) pair whose sum is this layer's output index.

    `forward` records `hidden_states + residual`, so making the two sum to
    `global_idx + 1` lets a caller read back which layer a captured tensor came
    from.
    """

    def __init__(self, global_idx: int) -> None:
        super().__init__()
        self.out_index = global_idx + 1

    def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        return (
            torch.zeros_like(hidden_states),
            torch.full_like(hidden_states, float(self.out_index)),
        )


class _UnownedLayer(torch.nn.Module):
    """Stands in for PPMissingLayer: calling it means the stage band is wrong."""

    def forward(self, *args, **kwargs):
        raise AssertionError("a stage ran a layer outside its own band")


def _build_stage(rank: int, pp_size: int, monkeypatch):
    """Assemble one stage's state with stub layers, real bands.

    The band comes from the real `get_pp_indices`, so a stage that reads outside
    its own band calls an `_UnownedLayer` rather than producing a tensor that still
    happens to fit.
    """
    model = patch_module.AXK2Model.__new__(patch_module.AXK2Model)
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
    model.embed_input_ids = lambda input_ids: torch.zeros(BATCH, HIDDEN)
    model.norm = lambda hidden, residual: (
        hidden if residual is None else hidden + residual,
        None,
    )
    # What AXK2ForCausalLM.set_aux_hidden_state_layers writes.
    model.aux_hidden_state_layers = ()

    # The forward reads the group directly; the shared slot helpers read their own
    # import of it.
    group = _Group(rank, pp_size)
    monkeypatch.setattr(patch_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle3_pp, "get_pp_group", lambda: group)
    return model


def _stamped_layers(aux: torch.Tensor | None) -> list[int]:
    """Which layer each hidden-wide block of a combined aux tensor came from.

    Every stub layer fills its whole output with its own index, so block `k` of the
    concatenation reads back the layer it was captured from.
    """
    if aux is None:
        return []
    return [
        int(aux[..., block * HIDDEN].flatten()[0].item())
        for block in range(aux.shape[-1] // HIDDEN)
    ]


def _run_pipeline(pp_size: int, aux_layers: tuple[int, ...], monkeypatch):
    """Drive every stage in order; return the last output and what each stage sent."""
    carried = None
    layers_sent = []
    for rank in range(pp_size):
        model = _build_stage(rank, pp_size, monkeypatch)
        model.aux_hidden_state_layers = aux_layers
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        if rank < pp_size - 1:
            layers_sent.append(_stamped_layers(out.tensors.get(eagle3_pp.AUX_COMBINED)))
            carried = out
    return out, layers_sent


@pytest.mark.parametrize("pp_size", [2, PP_SIZE])
@pytest.mark.parametrize(
    "aux_layers",
    [CHECKPOINT_AUX_LAYERS, BOUNDARY_AUX_LAYERS, INTERIOR_AUX_LAYERS],
)
def test_last_stage_receives_every_aux_layer_in_order(pp_size, aux_layers, monkeypatch):
    out, _ = _run_pipeline(pp_size, aux_layers, monkeypatch)

    _, aux = out
    assert _stamped_layers(torch.cat(aux, dim=-1)) == list(aux_layers)


def test_a_boundary_layer_is_captured_once(monkeypatch):
    # 30 is both stage 3's last capture and stage 4's incoming hidden state on the
    # pp8 split. Capturing at both would duplicate it.
    assert get_pp_indices(NUM_LAYERS, 4, PP_SIZE)[0] == 30

    out, layers_sent = _run_pipeline(PP_SIZE, CHECKPOINT_AUX_LAYERS, monkeypatch)

    assert layers_sent == [[2], [2], [2], [2, 30], [2, 30], [2, 30], [2, 30]]
    _, aux = out
    assert len(_stamped_layers(torch.cat(aux, dim=-1))) == len(CHECKPOINT_AUX_LAYERS)


def test_a_stage_owning_no_aux_layer_forwards_what_it_received(monkeypatch):
    # Stages 5 and 6 span [38, 46) and [46, 54) and own none of (2, 31, 58); they
    # must still pass on what stage 4 sent, or the last stage comes up short.
    _, layers_sent = _run_pipeline(PP_SIZE, INTERIOR_AUX_LAYERS, monkeypatch)

    assert layers_sent[6] == layers_sent[5] == layers_sent[4] == [2, 31]


def test_handoff_carries_no_aux_slots_when_eagle3_is_off(monkeypatch):
    model = _build_stage(0, PP_SIZE, monkeypatch)

    out = model.forward(
        input_ids=torch.zeros(BATCH, dtype=torch.long),
        positions=torch.zeros(BATCH, dtype=torch.long),
        intermediate_tensors=None,
    )

    assert set(out.tensors) == {"hidden_states", "residual"}


def test_last_stage_returns_a_bare_tensor_when_eagle3_is_off(monkeypatch):
    carried = None
    for rank in range(PP_SIZE):
        model = _build_stage(rank, PP_SIZE, monkeypatch)
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        carried = out

    assert isinstance(out, torch.Tensor)
