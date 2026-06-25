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

_DEFAULT_MAX_VISUAL_TOKENS = 4096


def _reshape_qkv_for_static_attention(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
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
    return query, key, value, reshape_output


def _get_valid_mask(
    sequence_lengths: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if sequence_lengths is None or sequence_lengths.dim() < 2:
        return None
    return sequence_lengths.to(device=device, dtype=torch.bool)


def _restore_static_attention_output(
    output: torch.Tensor,
    valid_mask: torch.Tensor | None,
    reshape_output: bool,
) -> torch.Tensor:
    output = output.transpose(1, 2)
    if valid_mask is not None:
        query_mask = valid_mask[:, :, None, None].to(dtype=output.dtype)
        output = output * query_mask
    if reshape_output:
        output = output.reshape(output.size(0), output.size(1), -1)
    return output


def rbln_dense_mm_encoder_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense attention replacement for compiling Qwen3-VL vision blocks.

    vLLM's MMEncoderAttention consumes packed metadata for variable-length
    sequences. The visual tail is compiled for one static image shape here, so
    this implementation ignores the packed sequence metadata and runs standard
    dense scaled dot-product attention with compile-friendly tensor ops. When a
    static visual-token bucket is used, ``sequence_lengths`` carries a
    fixed-shape boolean valid-token mask.
    """
    del cu_seqlens, max_seqlen

    query, key, value, reshape_output = _reshape_qkv_for_static_attention(
        self, query, key, value
    )

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
    valid_mask = _get_valid_mask(sequence_lengths, attn_scores.device)
    if valid_mask is not None:
        key_mask = valid_mask[:, None, None, :]
        mask_value = torch.tensor(
            -1.0e4,
            device=attn_scores.device,
            dtype=attn_scores.dtype,
        )
        attn_scores = torch.where(key_mask, attn_scores, mask_value)

    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, value)
    return _restore_static_attention_output(output, valid_mask, reshape_output)


class RBLNQwen3VisionTail(nn.Module):
    """Compile target for the Qwen3-VL vision transformer tail.

    The original visual module performs patch embedding and positional embedding
    before entering the transformer blocks. This wrapper starts after that point
    and compiles the block stack, the main merger, and the deepstack mergers as
    one graph so the output keeps Qwen3-VL's full multimodal embedding layout.
    """

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
        """Run vision blocks and concatenate main/deepstack merged features."""
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


def _get_qwen3_visual_merge_factor(visual: nn.Module) -> int:
    spatial_merge_size = getattr(visual, "spatial_merge_size", 2)
    return int(spatial_merge_size) * int(spatial_merge_size)


def _pad_qwen3_visual_tensor(
    tensor: torch.Tensor,
    max_visual_tokens: int,
) -> torch.Tensor:
    num_tokens = tensor.shape[0]
    if num_tokens == max_visual_tokens:
        return tensor

    pad_shape = (max_visual_tokens - num_tokens, *tensor.shape[1:])
    padding = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, padding], dim=0)


def _make_qwen3_visual_valid_mask(
    num_tokens: int,
    max_visual_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((1, max_visual_tokens), device=device, dtype=torch.bool)
    mask[:, :num_tokens] = True
    return mask


def rbln_qwen3_visual_forward(
    self,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Forward override that calls the compiled Qwen3-VL visual tail.

    Patch embedding and positional embedding remain in Python because they
    depend on the incoming image grid metadata. The block/merger tail is then
    dispatched to ``self.rbln_compiled_tail`` using precomputed encoder metadata.
    Inputs are padded to ``self.rbln_max_visual_tokens`` so different image
    sizes in the same bucket reuse the same compiled graph.
    """
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
    num_visual_tokens = hidden_states.shape[0]
    max_visual_tokens = self.rbln_max_visual_tokens
    merge_factor = self.rbln_visual_merge_factor
    if num_visual_tokens > max_visual_tokens:
        raise ValueError(
            "Qwen3-VL image has too many visual tokens for the compiled "
            f"RBLN bucket: {num_visual_tokens} > {max_visual_tokens}."
        )
    if num_visual_tokens % merge_factor != 0:
        raise ValueError(
            "Qwen3-VL visual token count must be divisible by the merge "
            f"factor {merge_factor}, but got {num_visual_tokens}."
        )

    valid_merged_tokens = num_visual_tokens // merge_factor
    hidden_states = _pad_qwen3_visual_tensor(hidden_states, max_visual_tokens)
    hidden_states = hidden_states.unsqueeze(1)
    rotary_pos_emb_cos = _pad_qwen3_visual_tensor(
        encoder_metadata["rotary_pos_emb_cos"],
        max_visual_tokens,
    )
    rotary_pos_emb_sin = _pad_qwen3_visual_tensor(
        encoder_metadata["rotary_pos_emb_sin"],
        max_visual_tokens,
    )
    valid_mask = _make_qwen3_visual_valid_mask(
        num_visual_tokens,
        max_visual_tokens,
        hidden_states.device,
    )

    hidden_states = self.rbln_compiled_tail(
        hidden_states,
        encoder_metadata["cu_seqlens"],
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
        encoder_metadata["max_seqlen"],
        valid_mask,
    )
    return hidden_states[:valid_merged_tokens]


def compile_qwen3_vision_tail(
    visual: nn.Module,
    backend: Any,
    options: dict[str, Any],
    max_visual_tokens: int = _DEFAULT_MAX_VISUAL_TOKENS,
) -> None:
    """Patch and compile a Qwen3-VL visual module for the RBLN path.

    This mutates ``visual`` in place by replacing MM encoder attention with the
    dense compile-friendly implementation, installing ``rbln_compiled_tail``,
    and binding ``visual.forward`` to the RBLN-aware forward override.
    """
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        return
    merge_factor = _get_qwen3_visual_merge_factor(visual)
    if max_visual_tokens % merge_factor != 0:
        raise ValueError(
            "Qwen3-VL max visual token bucket must be divisible by the merge "
            f"factor {merge_factor}, but got {max_visual_tokens}."
        )

    for block in blocks:
        mm_encoder_attn = getattr(getattr(block, "attn", None), "attn", None)
        if mm_encoder_attn is not None:
            mm_encoder_attn.forward = MethodType(
                rbln_dense_mm_encoder_attention_forward,
                mm_encoder_attn,
            )

    visual.rbln_max_visual_tokens = max_visual_tokens
    visual.rbln_visual_merge_factor = merge_factor
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
    """Keep Qwen3-VL deepstack embeddings intact or trim non-deepstack extras.

    Qwen3-VL with deepstack expects the full concatenated visual feature width.
    Older/non-deepstack paths may receive wider embeddings, so those are trimmed
    back to ``visual_dim`` for compatibility.
    """
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
