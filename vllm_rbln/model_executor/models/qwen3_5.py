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
"""Qwen3.5 (Qwen3_5ForConditionalGeneration) with images and video on the vllm
model path.

The vision tower runs as its own compiled graph at fixed patch buckets
(`encoder_token_budgets`): an image is padded to the smallest bucket that holds
it, its rotary and learned position embeddings are built on the host, and a key
mask hides the padding. A video runs one temporal patch at a time, each as an
image of its grid. The embeddings reach the language model through the runner's
staged `mm_embeds` / `mm_mask` (SupportsRBLNMultiModal), merged inside the
compiled graph.
"""

import math
from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    pos_embed_interpolate_native,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataItems
from vllm.sequence import IntermediateTensors

from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.models.interfaces import SupportsRBLNMultiModal
from vllm_rbln.platform import RBLN_DEFAULT_MAX_NUM_BATCHED_TOKENS
from vllm_rbln.v1.attention.backends.gdn_attn import grouped_qkv_rows

logger = init_logger(__name__)


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


class RBLNQwen3_5MultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Upstream's processor, refusing per request an image or a video frame
    above the engine's start-up limit, which the vision encoder buckets are
    sized to hold."""

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        outputs = super()._apply_hf_processor_main(mm_items, hf_kwargs)
        merge = self.info.get_hf_config().vision_config.spatial_merge_size
        for modality, pixels in (
            ("image", "pixel_values"),
            ("video", "pixel_values_videos"),
        ):
            # An input sent as embeddings has a grid too, but skips the encoder.
            if pixels not in outputs:
                continue
            limit = (
                self.info.get_mm_max_tokens_per_item(
                    seq_len=self.info.ctx.model_config.max_model_len,
                    mm_counts={modality: 1},
                )[modality]
                * merge**2
            )
            for grid in outputs[f"{modality}_grid_thw"].tolist():
                if grid[1] * grid[2] > limit:
                    raise ValueError(
                        f"The {modality} grid {grid} has {grid[1] * grid[2]} patches "
                        f"a frame, more than the {limit} the RBLN vision encoder is "
                        "compiled for. Lower the request's max_pixels, or raise it at "
                        "start-up in --mm-processor-kwargs (and add a budget that "
                        "reaches it if encoder_token_budgets is set)."
                    )
        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    RBLNQwen3_5MultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class RBLNQwen3_5ForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, SupportsRBLNMultiModal
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # Pruning changes the embeddings' width and a video's token count and
        # positions after the encoder, none of which the RBLN runner follows.
        mm_config = vllm_config.model_config.get_multimodal_config()
        if mm_config.is_multimodal_pruning_enabled():
            raise NotImplementedError(
                "Qwen3.5 video pruning is not supported on RBLN. Leave "
                "--video-pruning-rate unset."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Upstream's MTP proposer reads the image token by this name from a
        # target whose class name it does not list.
        self.config.image_token_index = self.config.image_token_id
        object.__setattr__(self, "_mm_encoder", None)
        # Upstream builds no vision tower when images and video are off (a
        # placeholder stands in for it).
        modalities = [
            m for m in ("image", "video") if mm_config.get_limit_per_prompt(m) > 0
        ]
        self.has_vision = bool(modalities)
        self.mm_encoder_patch_buckets: list[int] = []
        if self.has_vision:
            info = MULTIMODAL_REGISTRY.get_processing_info(vllm_config.model_config)
            merge = self.visual.spatial_merge_size
            max_tokens_per_item = info.get_mm_max_tokens_per_item(
                seq_len=vllm_config.model_config.max_model_len,
                mm_counts=dict.fromkeys(modalities, 1),
            )
            max_tokens = max(max_tokens_per_item[m] for m in modalities)
            rbln_config = vllm_config.additional_config
            # One compiled encoder shape per budget; an image is padded to the
            # smallest that holds its patches.
            budgets = sorted(rbln_config.encoder_token_budgets)
            if not budgets:
                # Unset: powers of two from the default prefill chunk counted in
                # patches up to the longest prefill chunk in output tokens (the
                # default one at least).
                default_chunk = RBLN_DEFAULT_MAX_NUM_BATCHED_TOKENS
                top = max(default_chunk, rbln_config.max_prefill_chunk_size)
                budgets = [default_chunk // merge**2]
                while budgets[-1] * 2 < top:
                    budgets.append(budgets[-1] * 2)
                budgets.append(top)
                # No budget above the largest frame, which is the last one.
                budgets = [b for b in budgets if b < max_tokens] + [max_tokens]
            if budgets[-1] < max_tokens:
                raise ValueError(
                    f"encoder_token_budgets {budgets} must reach {max_tokens} output "
                    "tokens, the processor's upper bound for an image or a video "
                    "frame. Add a budget that large, or lower max_pixels in "
                    "--mm-processor-kwargs."
                )
            self.mm_encoder_patch_buckets = [b * merge**2 for b in budgets]
            logger.info(
                "Vision encoder buckets: patches=%s",
                self.mm_encoder_patch_buckets,
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        # The checkpoint's [q | k | v] rows in key-head groups, the order the
        # GatedDeltaNet conv runs them in (RBLNQwenGatedDeltaNetAttention).
        for layer in self.language_model.model.layers:
            attn = getattr(layer, "linear_attn", None)
            if attn is None:
                continue
            rows = grouped_qkv_rows(
                attn.num_k_heads // attn.tp_size,
                attn.num_v_heads // attn.num_k_heads,
                attn.head_k_dim,
            )
            for weight in (attn.in_proj_qkvz.weight, attn.conv1d.weight):
                weight.data[: rows.numel()] = weight.data[rows]
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype]:
        return (vllm_config.model_config.dtype,)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...]]:
        model_config = vllm_config.model_config
        text_config = model_config.hf_text_config
        return (
            (
                text_config.linear_num_value_heads
                // vllm_config.parallel_config.tensor_parallel_size,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            ),
        )

    def compile_mm_encoder(self, compile_fn: Callable[[nn.Module], nn.Module]) -> None:
        if not self.has_vision:
            return
        encoder = _RBLNQwen3VisionEncoder(self.visual).eval()
        object.__setattr__(self, "_mm_encoder", compile_fn(encoder))

    def warmup_mm_encoder(self) -> None:
        if not self.has_vision:
            return
        encoder = self._mm_encoder
        if encoder is None:
            # enforce_eager: the runner never calls compile_mm_encoder.
            self.compile_mm_encoder(lambda encoder: encoder)
            encoder = self._mm_encoder
        assert encoder is not None
        visual = self.visual
        patch_dim = visual.patch_embed.proj.weight[0].numel()
        rotary_dim = visual.blocks[0].attn.hidden_size_per_attention_head // 2
        kwargs = {"dtype": visual.dtype, "device": visual.device}
        with torch.inference_mode():
            for bucket in self.mm_encoder_patch_buckets:
                encoder(
                    torch.zeros(bucket, patch_dim, **kwargs),
                    torch.zeros(bucket, visual.hidden_size, **kwargs),
                    torch.ones(bucket, rotary_dim, **kwargs),
                    torch.zeros(bucket, rotary_dim, **kwargs),
                    torch.zeros(1, 1, bucket, **kwargs),
                )

    def _encoder_inputs(
        self, pixel_values: torch.Tensor, grid_thw: list[int]
    ) -> list[torch.Tensor]:
        """Pad one image or video frame to its bucket and build its position
        embeddings and key mask on the host; upstream builds them on the tower's
        device."""
        visual = self.visual
        num_patches = math.prod(grid_thw)
        fits = [b for b in self.mm_encoder_patch_buckets if b >= num_patches]
        assert fits, (
            f"a frame of {num_patches} patches (grid {grid_thw}) exceeds the "
            f"vision encoder's {self.mm_encoder_patch_buckets[-1]}-patch bucket"
        )
        bucket = fits[0]
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

    def _encode(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """The embeddings of each image or video, one encoder run a temporal
        patch: upstream's cu_seqlens keeps attention inside each, so a video's
        frames are images of its grid."""
        encoder = self._mm_encoder
        if encoder is None:
            # enforce_eager: the runner never calls compile_mm_encoder.
            self.compile_mm_encoder(lambda encoder: encoder)
            encoder = self._mm_encoder
        assert encoder is not None
        grids = grid_thw.tolist()
        sizes = [math.prod(g) for g in grids]
        tokens = [h * w // self.visual.spatial_merge_size**2 for _, h, w in grids]
        embeds = []
        with torch.inference_mode():
            for pixels, (_, h, w), n in zip(pixel_values.split(sizes), grids, tokens):
                frames = [
                    encoder(*self._encoder_inputs(frame, [1, h, w]))[:n].cpu()
                    for frame in pixels.split(h * w)
                ]
                embeds.append(torch.cat(frames))
        return tuple(embeds)

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return super()._process_image_input(image_input)
        return self._encode(image_input["pixel_values"], image_input["image_grid_thw"])

    def _process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:
        if video_input["type"] == "video_embeds":
            return super()._process_video_input(video_input)
        return self._encode(
            video_input["pixel_values_videos"], video_input["video_grid_thw"]
        )

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


class RBLNQwen3_5MTP(Qwen3_5MTP):
    """Upstream's drafter with its layers numbered past the target's: the runner
    orders the KV caches by the number in a layer's name, one layer a number."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        layers = vllm_config.compilation_config.static_forward_context
        start = self.model.mtp_start_layer_idx
        for idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn.attn
            del layers[attn.layer_name]
            attn.layer_name = attn.layer_name.replace(
                f".layers.{idx}.", f".layers.{start + idx}."
            )
            layers[attn.layer_name] = attn
