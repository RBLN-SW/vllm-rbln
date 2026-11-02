# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5 (Qwen3_5ForConditionalGeneration) with images on the vllm model path.

The vision tower runs as its own compiled graph at one fixed patch bucket: an
image is padded to the bucket, its rotary and learned position embeddings are
built on the host, and a key mask hides the padding. The image embeddings
reach the language model through the runner's staged `mm_embeds` / `mm_mask`
(SupportsRBLNMultiModal), merged inside the compiled graph.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import pos_embed_interpolate_native
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_rbln.model_executor.models.interfaces import SupportsRBLNMultiModal

# The largest image, in patches, the compiled vision encoder takes. Attention
# runs over the whole bucket at once, so the processor's own limit (16M pixels
# for Qwen3.5) does not fit; --mm-processor-kwargs '{"max_pixels": N}' sets it.
MAX_ENCODER_PATCHES = 16384


class _RBLNQwen3VisionEncoder(nn.Module):
    """Qwen3-VL's vision transformer on one padded image, without the cu_seqlens
    attention: full attention over the bucket with padded keys masked."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual
        attn = visual.blocks[0].attn
        self.num_heads = attn.num_attention_heads_per_partition
        self.head_dim = attn.hidden_size_per_attention_head
        proj = visual.patch_embed.proj
        self.patch_weight = proj.weight.view(proj.weight.shape[0], -1)
        self.patch_bias = proj.bias

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(
        self,
        patches: torch.Tensor,
        pos_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patches: [N, C * T * P * P] the image's patches, padded to N rows
            pos_embeds: [N, hidden] the interpolated learned position embedding
            cos, sin: [N, head_dim / 2] the 2-D rotary embedding
            key_bias: [1, 1, N] 0 on the image's patches, -30000 on the padding
        """
        n = patches.shape[0]
        x = F.linear(patches, self.patch_weight, self.patch_bias) + pos_embeds
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        scale = self.head_dim**-0.5
        for block in self.visual.blocks:
            qkv, _ = block.attn.qkv(block.norm1(x))
            qkv = qkv.view(n, 3, self.num_heads, self.head_dim)
            q = self._rotate(qkv[:, 0], cos, sin).transpose(0, 1)
            k = self._rotate(qkv[:, 1], cos, sin).transpose(0, 1)
            v = qkv[:, 2].transpose(0, 1)
            scores = torch.matmul(q, k.transpose(1, 2)) * scale + key_bias
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
            out, _ = block.attn.proj(out.transpose(0, 1).reshape(n, -1))
            x = x + out
            x = x + block.mlp(block.norm2(x))
        return self.visual.merger(x)


class RBLNQwen3_5ForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, SupportsRBLNMultiModal
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        object.__setattr__(self, "_mm_encoder", None)
        # Upstream builds no vision tower when images are off (a placeholder
        # stands in for it).
        mm_config = vllm_config.model_config.get_multimodal_config()
        self.has_vision = mm_config.get_limit_per_prompt("image") > 0
        self.encoder_patches = 0
        if self.has_vision:
            info = MULTIMODAL_REGISTRY.create_processor(vllm_config.model_config).info
            size = info.get_image_size_with_most_features()
            patch = self.visual.patch_size
            self.encoder_patches = (size.width // patch) * (size.height // patch)
            if self.encoder_patches > MAX_ENCODER_PATCHES:
                max_pixels = MAX_ENCODER_PATCHES * patch * patch
                raise ValueError(
                    f"The largest image the processor accepts has "
                    f"{self.encoder_patches} patches; the RBLN vision encoder takes "
                    f"at most {MAX_ENCODER_PATCHES}. Limit it with --mm-processor-"
                    f"kwargs '{{\"max_pixels\": {max_pixels}}}' or less, or serve "
                    "text-only with --language-model-only."
                )

    def compile_mm_encoder(self, compile_fn: Callable[[nn.Module], nn.Module]) -> None:
        if not self.has_vision:
            return
        encoder = _RBLNQwen3VisionEncoder(self.visual).eval()
        object.__setattr__(self, "_mm_encoder", compile_fn(encoder))

    def _encoder_inputs(
        self, pixel_values: torch.Tensor, grid_thw: list[int]
    ) -> list[torch.Tensor]:
        """Pad one image to the bucket and build its position embeddings and key
        mask on the host, as upstream's fast_pos_embed_interpolate and
        rot_pos_emb do on the tower's device (and with Triton when it imports)."""
        visual = self.visual
        bucket = self.encoder_patches
        num_patches = math.prod(grid_thw)
        assert num_patches <= bucket, (
            f"an image of {num_patches} patches (grid {grid_thw}) exceeds the "
            f"vision encoder's {bucket}-patch bucket"
        )
        dtype = visual.dtype
        t, h, w = grid_thw
        patches = torch.zeros(bucket, pixel_values.shape[-1], dtype=dtype)
        patches[:num_patches] = pixel_values.to(dtype)
        pos_embeds = torch.zeros(bucket, visual.hidden_size, dtype=dtype)
        pos_embeds[:num_patches] = pos_embed_interpolate_native(
            visual.pos_embed.weight.cpu(),
            t,
            h,
            w,
            visual.num_grid_per_side,
            visual.spatial_merge_size,
            dtype,
        )
        pos_ids = visual.rot_pos_ids(h, w, visual.spatial_merge_size).repeat(t, 1)
        cos_table, sin_table = visual.rotary_pos_emb.get_cos_sin(max(h, w))
        image_cos = cos_table.cpu()[pos_ids].flatten(1)
        cos = torch.ones(bucket, image_cos.shape[-1], dtype=dtype)
        sin = torch.zeros(bucket, image_cos.shape[-1], dtype=dtype)
        cos[:num_patches] = image_cos
        sin[:num_patches] = sin_table.cpu()[pos_ids].flatten(1)
        key_bias = torch.zeros(1, 1, bucket, dtype=dtype)
        key_bias[..., num_patches:] = -30000.0
        return [x.to(visual.device) for x in (patches, pos_embeds, cos, sin, key_bias)]

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return super()._process_image_input(image_input)
        encoder = self._mm_encoder
        if encoder is None:
            # enforce_eager: the runner never calls compile_mm_encoder.
            encoder = _RBLNQwen3VisionEncoder(self.visual).eval()
            object.__setattr__(self, "_mm_encoder", encoder)
        grid_thw = image_input["image_grid_thw"].tolist()
        sizes = [math.prod(g) for g in grid_thw]
        merge = self.visual.spatial_merge_size
        embeds = []
        with torch.inference_mode():
            for pixels, grid in zip(image_input["pixel_values"].split(sizes), grid_thw):
                out = encoder(*self._encoder_inputs(pixels, grid))
                embeds.append(out[: math.prod(grid) // (merge * merge)].cpu())
        return tuple(embeds)

    def _process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("Qwen3.5 video inputs are not supported on RBLN")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_embeds: torch.Tensor | None = None,
        mm_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if mm_embeds is not None and inputs_embeds is None:
            assert mm_mask is not None
            text_embeds = self.language_model.embed_input_ids(input_ids)
            inputs_embeds = torch.where(mm_mask > 0.5, mm_embeds, text_embeds)
            input_ids = None
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
