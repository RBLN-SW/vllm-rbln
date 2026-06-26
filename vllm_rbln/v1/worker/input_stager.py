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


@dataclass(frozen=True)
class InputLayout:
    num_reqs: int
    num_reqs_padded: int
    query_len: int
    query_len_padded: int
    input_pad_value: int = 0
    position_pad_value: int = 0

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
    intermediate_tensors: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    token_indices: torch.Tensor | None

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
        self._buffers: dict[tuple, InputBuffer] = {}
        self._token_indices_buffers: dict[tuple[torch.dtype, int], torch.Tensor] = {}

    def stage(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        layout: InputLayout,
    ) -> StagedModelInputs:
        buf = self._get_or_create_buffer(layout, input_ids, positions)

        assert buf.input_ids.shape == layout.shape
        buf.input_ids.fill_(layout.input_pad_value)
        buf.input_ids[: layout.num_reqs, : layout.query_len].copy_(
            input_ids,
            non_blocking=True,
        )

        buf.positions.fill_(layout.position_pad_value)
        buf.positions[: layout.num_reqs, : layout.query_len].copy_(
            positions,
            non_blocking=True,
        )

        return StagedModelInputs(
            input_ids=buf.input_ids,
            positions=buf.positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=self._stage_token_indices(token_indices),
        )

    def _get_or_create_buffer(
        self,
        layout: InputLayout,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> InputBuffer:
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
        self._buffers[key] = buf
        return buf

    def _stage_token_indices(
        self,
        token_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if token_indices is None:
            return None

        key = (token_indices.dtype, token_indices.numel())
        if (buf := self._token_indices_buffers.get(key)) is None:
            buf = torch.empty(
                token_indices.shape,
                dtype=token_indices.dtype,
                device=self.device,
            )
            self._token_indices_buffers[key] = buf

        buf.copy_(token_indices, non_blocking=True)
        return buf
