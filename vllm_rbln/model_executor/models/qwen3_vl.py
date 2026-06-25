# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy
from types import MethodType
from typing import Any

import torch
import torch.nn as nn


def rbln_dense_mm_encoder_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    del cu_seqlens, max_seqlen, sequence_lengths

    if query.dim() != 4:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        reshape_output = True
    else:
        reshape_output = False

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, value)
    output = output.transpose(1, 2)

    if reshape_output:
        output = output.reshape(output.size(0), output.size(1), -1)
    return output


class RBLNQwen3VisionTail(nn.Module):
    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.blocks = visual.blocks
        self.deepstack_visual_indexes = list(visual.deepstack_visual_indexes)
        self.deepstack_merger_list = visual.deepstack_merger_list
        self.merger = visual.merger

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
        return hidden_states


def rbln_qwen3_visual_forward(
    self,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    hidden_states = self.patch_embed(hidden_states)

    if encoder_metadata is None:
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
        else:
            grid_thw_list = grid_thw.tolist()
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

    pos_embeds = encoder_metadata["pos_embeds"]
    hidden_states = hidden_states + pos_embeds
    hidden_states = hidden_states.unsqueeze(1)

    return self.rbln_compiled_tail(
        hidden_states,
        encoder_metadata["cu_seqlens"],
        encoder_metadata["rotary_pos_emb_cos"],
        encoder_metadata["rotary_pos_emb_sin"],
        encoder_metadata["max_seqlen"],
        encoder_metadata.get("sequence_lengths"),
    )


def compile_qwen3_vision_tail(
    visual: nn.Module,
    backend: Any,
    options: dict[str, Any],
) -> None:
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        return

    for block in blocks:
        mm_encoder_attn = getattr(getattr(block, "attn", None), "attn", None)
        if mm_encoder_attn is not None:
            mm_encoder_attn.forward = MethodType(
                rbln_dense_mm_encoder_attention_forward,
                mm_encoder_attn,
            )

    visual.rbln_compiled_tail = torch.compile(
        RBLNQwen3VisionTail(visual),
        backend=backend,
        options=copy(options),
        dynamic=False,
    )
    visual.forward = MethodType(rbln_qwen3_visual_forward, visual)


def preserve_or_trim_multimodal_embeddings(
    model: nn.Module,
    mm_embeds: list[torch.Tensor],
) -> list[torch.Tensor]:
    visual_dim = getattr(model, "visual_dim", None)
    if visual_dim is None:
        return mm_embeds

    if getattr(model, "use_deepstack", False):
        deepstack_num_level = getattr(model, "deepstack_num_level", 0)
        expected_dim = visual_dim * (1 + deepstack_num_level)
        for mm_embed in mm_embeds:
            if mm_embed.shape[-1] != expected_dim:
                raise ValueError(
                    "Qwen3-VL deepstack requires full multimodal embeddings "
                    f"with hidden size {expected_dim}, but got "
                    f"{mm_embed.shape[-1]}."
                )
        return mm_embeds

    return [
        x[..., :visual_dim] if x.shape[-1] > visual_dim else x for x in mm_embeds
    ]
