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

from dataclasses import dataclass
from typing import Any

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.utils import record_function_or_nullcontext


@dataclass(frozen=True)
class InputLayout:
    num_reqs: int
    num_reqs_padded: int
    query_len: int
    query_len_padded: int
    input_pad_value: int = 0
    position_pad_value: int = 0
    hidden_state_pad_value: float = 0.0
    token_index_pad_value: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_reqs_padded, self.query_len_padded)


@dataclass
class InputBuffer:
    input_ids: torch.Tensor
    positions: torch.Tensor


@dataclass(slots=True)
class StagedModelInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    intermediate_tensors: IntermediateTensors | None
    inputs_embeds: torch.Tensor | None
    token_indices: torch.Tensor | None
    # For Eagle3 drafter
    hidden_states: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "positions": self.positions,
            "intermediate_tensors": self.intermediate_tensors,
            "inputs_embeds": self.inputs_embeds,
            "token_indices": self.token_indices,
        }


class InputStager:
    def __init__(self, device: torch.device):
        self.device = device
        self._buffers: dict[tuple, tuple[InputBuffer, InputBuffer]] = {}
        self._hidden_state_buffers: dict[tuple, torch.Tensor] = {}
        self._token_indices_buffers: dict[tuple[torch.dtype, int], torch.Tensor] = {}
        self._mm_embeds_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def stage(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        layout: InputLayout,
    ) -> StagedModelInputs:
        with record_function_or_nullcontext("input_stage:get_buffer"):
            buf, host_buf = self._get_or_create_buffer(layout, input_ids, positions)

        assert buf.input_ids.shape == layout.shape
        with record_function_or_nullcontext("input_stage:ids_host_fill"):
            host_buf.input_ids.fill_(layout.input_pad_value)
        with record_function_or_nullcontext("input_stage:ids_host_copy"):
            host_buf.input_ids[: layout.num_reqs, : layout.query_len].copy_(input_ids)
        with record_function_or_nullcontext("input_stage:ids_copy"):
            buf.input_ids.copy_(host_buf.input_ids, non_blocking=True)

        with record_function_or_nullcontext("input_stage:positions_host_copy"):
            host_buf.positions.fill_(layout.position_pad_value)
            host_buf.positions[: layout.num_reqs, : layout.query_len].copy_(positions)
        with record_function_or_nullcontext("input_stage:positions_copy"):
            buf.positions.copy_(host_buf.positions, non_blocking=True)

        return StagedModelInputs(
            input_ids=buf.input_ids,
            positions=buf.positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=self._stage_token_indices(token_indices, layout),
            hidden_states=self._stage_hidden_states(hidden_states, layout),
        )

    def _get_or_create_buffer(
        self,
        layout: InputLayout,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[InputBuffer, InputBuffer]:
        key = (
            layout.shape,
            input_ids.dtype,
            positions.dtype,
        )
        if key in self._buffers:
            return self._buffers[key]

        buf = InputBuffer(
            input_ids=torch.empty(
                layout.shape,
                dtype=input_ids.dtype,
                device=self.device,
            ),
            positions=torch.empty(
                layout.shape,
                dtype=positions.dtype,
                device=self.device,
            ),
        )
        host_buf = InputBuffer(
            input_ids=torch.empty(
                layout.shape, dtype=input_ids.dtype, device="cpu", pin_memory=True
            ),
            positions=torch.empty(
                layout.shape, dtype=positions.dtype, device="cpu", pin_memory=True
            ),
        )
        self._buffers[key] = (buf, host_buf)
        return buf, host_buf

    def _stage_hidden_states(
        self,
        hidden_states: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if hidden_states is None:
            return None

        key = (layout.shape, hidden_states.dtype, hidden_states.shape[-1])
        if (buf := self._hidden_state_buffers.get(key)) is None:
            buf = torch.empty(
                (*layout.shape, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=self.device,
            )
            self._hidden_state_buffers[key] = buf

        buf[layout.num_reqs :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, layout.query_len :].fill_(layout.hidden_state_pad_value)
        buf[: layout.num_reqs, : layout.query_len].copy_(
            hidden_states,
            non_blocking=True,
        )
        return buf

    def stage_mm_embeds(
        self,
        mm_rows: list[tuple[int, int, torch.Tensor]],
        mm_mask: torch.Tensor,
        hidden_size: int,
        layout: InputLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a multimodal model's 0 / 1 mask [R, L, 1] to the layout as text
        (mask 0), and copy each ``(request, first token, [N, H] rows)`` run of
        vision embeddings into a [R, L, H] buffer in the mask's dtype."""
        key = (layout.shape, mm_mask.dtype, hidden_size)
        if (bufs := self._mm_embeds_buffers.get(key)) is None:
            bufs = (
                torch.zeros(
                    (*layout.shape, hidden_size),
                    dtype=mm_mask.dtype,
                    device=self.device,
                ),
                torch.empty(
                    (*layout.shape, 1), dtype=mm_mask.dtype, device=self.device
                ),
            )
            self._mm_embeds_buffers[key] = bufs

        embeds, mask = bufs
        mask.zero_()
        mask[: layout.num_reqs, : layout.query_len].copy_(mm_mask, non_blocking=True)
        for req_index, start, rows in mm_rows:
            assert req_index < layout.num_reqs
            assert start + rows.shape[0] <= layout.query_len
            embeds[req_index, start : start + rows.shape[0]].copy_(
                rows, non_blocking=True
            )
        return bufs

    def _stage_token_indices(
        self,
        token_indices: torch.Tensor | None,
        layout: InputLayout,
    ) -> torch.Tensor | None:
        if token_indices is None:
            return None

        num_indices = token_indices.shape[0]
        assert num_indices <= layout.num_reqs_padded

        key = (token_indices.dtype, layout.num_reqs_padded)
        if (buf := self._token_indices_buffers.get(key)) is None:
            buf = torch.empty(
                layout.num_reqs_padded,
                dtype=token_indices.dtype,
                device=self.device,
            )
            self._token_indices_buffers[key] = buf

        buf.fill_(layout.token_index_pad_value)
        buf[:num_indices].copy_(token_indices, non_blocking=True)
        return buf
