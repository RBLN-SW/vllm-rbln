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

"""EAGLE3 aux hidden states survive the pipeline split.

The stages run sequentially in one process with stub layers, so this needs no
checkpoint and no device. Each stub layer stamps its own global index into the
tensor it returns, which is what makes a stage-local index visible: the values
that reach the last stage are the layer numbers actually harvested.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model

import vllm_rbln.patches.minimax_m2 as patch_module
import vllm_rbln.v1.spec_decode.eagle3_pp as eagle3_pp

# The suite truncates models via VLLM_RBLN_NUM_HIDDEN_LAYERS, and the patched
# get_pp_indices honors it. These stages are stubs, so the truncation has nothing
# to shrink -- what the split must be is the real 62-layer one.
from vllm_rbln.patches.distributed_utils import (  # noqa: E402
    original_get_pp_indices as get_pp_indices,
)

NUM_LAYERS = 62
HIDDEN = 8
BATCH = 2
# What get_eagle3_default_aux_hidden_state_layers() returns for this depth. 31 is a
# stage boundary of the PP4 split, which is the case that can double-count.
AUX_LAYERS = (2, NUM_LAYERS // 2, NUM_LAYERS - 3)
# What MiniMax-M2.5-Eagle3 asks for in its eagle_config, one off the default. It
# lands one index in each of stages 0, 1 and 3, so the carry has to work for a set
# the model did not pick itself.
CHECKPOINT_AUX_LAYERS = (1, 30, 58)
# Every index sits exactly on a PP4 band boundary ([0,15) [15,31) [31,47) [47,62)).
# A boundary index is both one stage's last capture and the next stage's incoming
# hidden state, so an off-by-one in the split double-counts all three at once --
# and compiles, and passes a short probe, and only shows up as wrong output.
BOUNDARY_AUX_LAYERS = (15, 31, 47)


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

    def forward(self, positions, hidden_states, residual):
        return (
            torch.zeros_like(hidden_states),
            torch.full_like(hidden_states, float(self.out_index)),
        )


class _UnownedLayer(torch.nn.Module):
    """Stands in for PPMissingLayer: calling it means the stage band is wrong."""

    def forward(self, *args, **kwargs):
        raise AssertionError("a stage ran a layer outside its own band")


def _build_stage(rank: int, pp_size: int, monkeypatch) -> MiniMaxM2Model:
    """Assemble one stage's state with stub layers, real bands.

    The band comes from the real `get_pp_indices`, so a stage that reads outside
    its own band calls an `_UnownedLayer` rather than producing a tensor that still
    happens to fit.
    """
    model = MiniMaxM2Model.__new__(MiniMaxM2Model)
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
    model.aux_hidden_state_layers = ()

    # The forward reads the group directly; the shared slot helpers read their own
    # import of it.
    group = _Group(rank, pp_size)
    monkeypatch.setattr(patch_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle3_pp, "get_pp_group", lambda: group)
    return model


def _slot_indices(keys) -> list[int]:
    return sorted(
        int(key.rsplit("_", 1)[1]) for key in keys if key.startswith(eagle3_pp.AUX_SLOT)
    )


def _run_pipeline(pp_size: int, aux_layers: tuple[int, ...], monkeypatch):
    """Drive every stage in order; return the last output and what each stage sent."""
    carried = None
    slots_sent = []
    for rank in range(pp_size):
        model = _build_stage(rank, pp_size, monkeypatch)
        model._set_aux_hidden_state_layers(aux_layers)
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        if rank < pp_size - 1:
            slots_sent.append(_slot_indices(out.tensors))
            carried = out
    return out, slots_sent


@pytest.mark.parametrize("pp_size", [2, 4])
@pytest.mark.parametrize(
    "aux_layers", [AUX_LAYERS, CHECKPOINT_AUX_LAYERS, BOUNDARY_AUX_LAYERS]
)
def test_last_stage_receives_every_aux_layer_in_order(pp_size, aux_layers, monkeypatch):
    out, _ = _run_pipeline(pp_size, aux_layers, monkeypatch)

    _, aux = out
    stamped = [int(tensor.flatten()[0].item()) for tensor in aux]
    assert stamped == list(aux_layers)


def test_a_boundary_layer_is_captured_once(monkeypatch):
    # Layer 31 of the 62-layer PP4 split ([15, 16, 16, 15]) is both stage 1's
    # last capture and stage 2's incoming hidden state. Capturing at both would
    # duplicate it.
    assert get_pp_indices(NUM_LAYERS, 2, 4)[0] == 31

    out, slots_sent = _run_pipeline(4, AUX_LAYERS, monkeypatch)

    assert slots_sent == [[2], [2, 31], [2, 31]]
    _, aux = out
    assert len(aux) == len(AUX_LAYERS)


def test_a_stage_owning_no_aux_layer_forwards_the_slots(monkeypatch):
    # Stage 2 spans [31, 47) and owns none of (1, 30, 58); it must still pass what
    # stage 1 sent, or the last stage comes up short.
    _, slots_sent = _run_pipeline(4, CHECKPOINT_AUX_LAYERS, monkeypatch)

    assert slots_sent[2] == slots_sent[1] == [1, 30]


def test_the_handoff_placeholder_follows_a_later_setter_call(monkeypatch):
    # ForCausalLM.__init__ copies this callable onto itself and the runner calls
    # that copy, so the installer binds it on the outer object and reads the key set
    # at call time. A callable that baked the key set in when it was bound leaves the
    # copy a slot short, which the receiving stage only finds out about as a KeyError.
    inner = _build_stage(1, 4, monkeypatch)
    outer = SimpleNamespace(model=inner)
    eagle3_pp.install_aux_handoff_slots(outer)
    snapshot = outer.make_empty_intermediate_tensors

    inner._set_aux_hidden_state_layers(CHECKPOINT_AUX_LAYERS)

    tensors = snapshot(batch_size=BATCH, dtype=torch.float32, device="cpu")
    assert _slot_indices(tensors.tensors) == [1]


def test_handoff_carries_no_aux_slots_when_eagle3_is_off(monkeypatch):
    model = _build_stage(0, 4, monkeypatch)

    out = model.forward(
        input_ids=torch.zeros(BATCH, dtype=torch.long),
        positions=torch.zeros(BATCH, dtype=torch.long),
        intermediate_tensors=None,
    )

    assert set(out.tensors) == {"hidden_states", "residual"}


def test_last_stage_returns_a_bare_tensor_when_eagle3_is_off(monkeypatch):
    carried = None
    for rank in range(4):
        model = _build_stage(rank, 4, monkeypatch)
        out = model.forward(
            input_ids=torch.zeros(BATCH, dtype=torch.long),
            positions=torch.zeros(BATCH, dtype=torch.long),
            intermediate_tensors=carried,
        )
        carried = out

    assert isinstance(out, torch.Tensor)


def test_the_received_slot_set_follows_the_stage_band(monkeypatch):
    # The handoff placeholder is built from start_layer, so each stage has to ask
    # for exactly the indices an earlier stage can have captured.
    expected = {0: (), 1: (1,), 2: (1, 30), 3: (1, 30)}
    for rank, want in expected.items():
        model = _build_stage(rank, 4, monkeypatch)
        model._set_aux_hidden_state_layers(CHECKPOINT_AUX_LAYERS)

        assert eagle3_pp.aux_slots_received(model) == want
