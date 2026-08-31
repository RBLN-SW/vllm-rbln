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
"""Carrying EAGLE3 aux hidden states across a pipeline split.

Everything here is architecture-independent: the slot naming, which indices a
stage receives versus captures, and the handoff placeholder that has to advertise
them. Only the capture itself is not, because it lives inside a model's `forward`
and the tensors have to be graph outputs of that forward -- see
`vllm_rbln/patches/minimax_m2.py` for the one architecture that has it.

TODO(vllm-project/vllm#50514): delete once that lands and is released.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import SpeculativeConfig
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

# Target architectures whose `forward` this plugin has patched to carry the aux
# hidden states across the split. Upstream's own forward indexes the capture with
# a stage-local `enumerate` and then drops the list on every non-last stage, so an
# architecture absent from here harvests the wrong layers and comes up short at
# the drafter. `RblnPlatform.check_and_update_config` rejects the combination
# rather than letting it fail mid-compile.
EAGLE3_PP_TARGET_ARCHS = frozenset({"MiniMaxM2ForCausalLM"})

AUX_SLOT = "aux_hidden_states_"


def eagle3_aux_hidden_states_enabled(
    speculative_config: SpeculativeConfig | None,
) -> bool:
    """Whether an EAGLE3 draft consumes the target's aux hidden states.

    A draft can turn them off in its `eagle_config`, and then nothing is captured
    anywhere: `aux_hidden_state_layers` stays empty, so upstream's unpatched
    forward is harmless under a pipeline split. The model runner and the startup
    guard both have to agree on this, hence a single reader.

    Read from the config rather than from the drafter, which exists only on the
    last rank. Every stage has to know: the aux tensors are captured across the
    stages and consumed only on the last one, so a non-last stage that thinks
    EAGLE3 is off captures nothing and the last stage comes up short.
    """
    if speculative_config is None or speculative_config.method != "eagle3":
        return False
    eagle_config = getattr(
        speculative_config.draft_model_config.hf_config, "eagle_config", None
    )
    if not isinstance(eagle_config, dict):
        return True
    return bool(eagle_config.get("use_aux_hidden_state", True))


def aux_capture_module(model: nn.Module) -> nn.Module:
    """The module that owns the capture state, resolved as upstream resolves it.

    Mirrors `SupportsEagle3.set_aux_hidden_state_layers`, so the state this module
    reads is the state the runner's setter wrote.
    """
    parent = model
    if hasattr(model, "get_language_model"):
        parent = model.get_language_model()
    elif hasattr(model, "language_model"):
        parent = model.language_model
    return parent.model


def aux_slots_received(model: nn.Module) -> tuple[int, ...]:
    """Global aux layer indices that reach this stage from upstream stages.

    A capture at index ``i`` is the input to layer ``i``, so this stage receives
    every requested index at or before its first owned layer. On the first stage
    the set is empty: index ``start_layer == 0`` is captured locally from the
    embedding output.
    """
    if get_pp_group().is_first_rank:
        return ()
    return tuple(
        sorted(i for i in model.aux_hidden_state_layers if i <= model.start_layer)
    )


def aux_slots_captured(model: nn.Module) -> tuple[int, ...]:
    """Global aux layer indices this stage produces, in the order it produces them.

    The loop captures at ``start_layer + idx + 1`` for each owned layer, so the
    reachable indices are ``start_layer + 1 .. end_layer``; the first stage adds
    index 0 from the embedding output. Every index is a compile-time constant, so
    deriving the order here keeps it out of the traced graph -- the capture itself
    appends to a list and never keys a dict inside the loop.

    Together with `aux_slots_received` this partitions the requested set: received
    indices are at or before ``start_layer``, captured ones past it. Received then
    captured is therefore already ascending, which is the order the drafter's `fc`
    expects its concatenated blocks in.
    """
    captured = [
        i
        for i in sorted(model.aux_hidden_state_layers)
        if model.start_layer < i <= model.end_layer
    ]
    if get_pp_group().is_first_rank and 0 in model.aux_hidden_state_layers:
        captured.insert(0, 0)
    return tuple(captured)


def install_aux_handoff_slots(model: nn.Module) -> None:
    """Size the pipeline handoff to carry the aux hidden states.

    Rebind rather than patch a model's `__init__`: the aux layers are only known
    once the runner calls `set_aux_hidden_state_layers`, and the key set has to be
    read at call time regardless, because a model's `__init__` builds this callable
    from a key list fixed at construction. Bind it on the object the runner calls
    -- `ForCausalLM.__init__` copies the inner model's callable onto itself, and
    the runner calls that copy.
    """
    inner = aux_capture_module(model)
    hidden_size = inner.config.hidden_size

    def make_empty_intermediate_tensors(
        batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        keys = ["hidden_states", "residual"]
        keys += [f"{AUX_SLOT}{i}" for i in aux_slots_received(inner)]
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    model.make_empty_intermediate_tensors = make_empty_intermediate_tensors
