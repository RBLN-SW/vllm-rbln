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

"""Gemma4 on the RBLN vLLM path: upstream's model, with its vision tower
compiled and its image embeddings merged inside the compiled graph."""

import math
from collections.abc import Callable, Mapping
from functools import cached_property

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature
from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.gemma4_mm import (
    _SUPPORTED_SOFT_TOKENS,
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4ImageInputs,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
    Gemma4VideoInputs,
    _get_max_soft_tokens,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataItems
from vllm.sequence import IntermediateTensors

from vllm_rbln.model_executor.models.interfaces import (
    SupportsRBLNMmEncoderWarmup,
    SupportsRBLNMultiModal,
)

# Padding seen by the encoder's softmax: far below any real score, still
# finite in half precision.
_KEY_PAD_BIAS = -30000.0


class _RBLNGemma4VisionEncoder(nn.Module):
    """Gemma4's vision tower and its projection into the text model as one
    static graph.

    HF's encoder builds its mask and rotary tables in Python and dispatches
    attention through ALL_ATTENTION_FUNCTIONS. Here the host hands in both,
    the attention is spelled out, and the patches arrive padded to one count.
    """

    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module):
        super().__init__()
        self.tower = vision_tower
        self.embed_vision = embed_vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        key_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # HF's patch embedder, with its x and y lookups made one: two lookups
        # into views of the same table fail the RBLN compiler.
        tower = self.tower
        patch_embedder = tower.patch_embedder
        dtype = patch_embedder.input_proj.weight.dtype
        hidden_states = patch_embedder.input_proj((2 * (pixel_values - 0.5)).to(dtype))
        table = patch_embedder.position_embedding_table
        index = pixel_position_ids.clamp(min=0) + table.new_tensor(
            [0, table.shape[1]], dtype=pixel_position_ids.dtype
        )
        position_embeddings = F.embedding(index, table.flatten(0, 1)).sum(dim=-2)
        hidden_states = hidden_states + torch.where(
            padding_positions.unsqueeze(-1), 0.0, position_embeddings
        )
        for layer in tower.encoder.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(hidden_states).view(shape))
            k = attn.k_norm(attn.k_proj(hidden_states).view(shape))
            v = attn.v_norm(attn.v_proj(hidden_states).view(shape))
            q = apply_multidimensional_rope(q, cos, sin, pixel_position_ids)
            k = apply_multidimensional_rope(k, cos, sin, pixel_position_ids)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
            v = v.repeat_interleave(attn.num_key_value_groups, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling + key_bias
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v).transpose(1, 2).reshape(residual.shape)
            hidden_states = residual + layer.post_attention_layernorm(attn.o_proj(out))
            residual = hidden_states
            hidden_states = layer.pre_feedforward_layernorm(hidden_states)
            hidden_states = layer.post_feedforward_layernorm(layer.mlp(hidden_states))
            hidden_states = residual + hidden_states

        output_length = pixel_values.shape[1] // tower.config.pooling_kernel_size**2
        pooled, _ = tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        # The pooler scales in float32; standardize there too, as HF does.
        if tower.config.standardize:
            pooled = (pooled - tower.std_bias.float()) * tower.std_scale.float()
        return self.embed_vision(inputs_embeds=pooled.to(dtype))


def mm_soft_token_cap(info: Gemma4ProcessingInfo) -> int:
    """The most soft tokens an image may take: the engine's max_soft_tokens,
    else the HF image processor's own. Encoder and image prefill compile to it.
    """
    model_config = info.ctx.model_config
    cap, _ = _get_max_soft_tokens(info.ctx.get_merged_mm_kwargs({}))
    if cap is None:
        cap = info.get_hf_processor().image_processor.max_soft_tokens
        # Upstream counts an image without max_soft_tokens at
        # default_output_length, while HF sizes it by its processor.
        default = model_config.hf_config.vision_config.default_output_length
        if cap != default:
            raise ValueError(
                f"vision_config.default_output_length={default} differs from "
                f"the image processor's max_soft_tokens={cap}; set the budget "
                "with --mm-processor-kwargs '{\"max_soft_tokens\": N}' instead."
            )
    if cap not in _SUPPORTED_SOFT_TOKENS:
        raise ValueError(
            f"max_soft_tokens={cap} is not one of {_SUPPORTED_SOFT_TOKENS}."
        )
    return cap


class RBLNGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    """Upstream's processor, refusing per request an image budget above the
    one the vision encoder and image prefill compile to."""

    @cached_property
    def _mm_soft_token_cap(self) -> int:
        return mm_soft_token_cap(self.info)

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Video frames take a fixed budget of their own, below any cap.
        if mm_items.get_count("image", strict=False):
            budget, _ = _get_max_soft_tokens(
                self.info.ctx.get_merged_mm_kwargs(hf_kwargs)
            )
            if isinstance(budget, int) and budget > self._mm_soft_token_cap:
                raise ValueError(
                    f"max_soft_tokens={budget} is above the engine's image budget "
                    f"of {self._mm_soft_token_cap}, which the vision encoder and "
                    "image prefill are compiled to; raise it with "
                    "--mm-processor-kwargs at start-up."
                )
        return super()._apply_hf_processor_main(mm_items, hf_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class RBLNGemma4ForConditionalGeneration(
    Gemma4ForConditionalGeneration, SupportsRBLNMultiModal, SupportsRBLNMmEncoderWarmup
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.mm_soft_token_cap = mm_soft_token_cap(
            MULTIMODAL_REGISTRY.create_processor(vllm_config.model_config).info
        )
        # One compiled encoder shape per soft token budget up to the cap; an
        # image or frame is padded to the smallest that holds its patches.
        pool = self.config.vision_config.pooling_kernel_size
        self.mm_encoder_buckets = [
            budget * pool**2
            for budget in _SUPPORTED_SOFT_TOKENS
            if budget <= self.mm_soft_token_cap
        ]
        self._mm_encoder: Callable[..., torch.Tensor] | None
        object.__setattr__(self, "_mm_encoder", None)

    def get_mm_prefix_query_lens(self) -> list[int]:
        # One length per soft token budget the processor may emit, rounded up
        # to the 128-token tiles optimum-rbln compiles image prefill at. Video
        # frames use the smallest budget.
        return sorted(
            {
                (budget + 127) // 128 * 128
                for budget in _SUPPORTED_SOFT_TOKENS
                if budget <= self.mm_soft_token_cap
            }
        )

    def compile_mm_encoder(self, compile_fn) -> None:
        # Without an image or video limit, or off the first PP rank, upstream
        # leaves the vision tower out.
        if not (
            get_pp_group().is_first_rank
            and any(
                self.multimodal_config.get_limit_per_prompt(m) > 0
                for m in ("image", "video")
            )
        ):
            return
        encoder = _RBLNGemma4VisionEncoder(self.vision_tower, self.embed_vision)
        object.__setattr__(self, "_mm_encoder", compile_fn(encoder.eval()))

    def warm_up_mm_encoder(self) -> None:
        if self._mm_encoder is None:
            return
        vision_config = self.config.vision_config
        pool = vision_config.pooling_kernel_size
        patch_pixels = 3 * vision_config.patch_size**2
        with torch.inference_mode():
            for bucket in self.mm_encoder_buckets:
                # A full grid whose sides are multiples of the pooling kernel.
                cells = bucket // pool**2
                rows = max(d for d in range(1, math.isqrt(cells) + 1) if cells % d == 0)
                ys, xs = torch.meshgrid(
                    torch.arange(rows * pool),
                    torch.arange(cells // rows * pool),
                    indexing="ij",
                )
                positions = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
                self._encode_vision(torch.zeros(bucket, patch_pixels), positions)

    def _encode_vision(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """One image or video frame, `[num_patches, patch_pixels]` with `(-1, -1)`
        positions on its padding, to its soft token embeddings."""
        encoder = self._mm_encoder
        if encoder is None:
            # enforce_eager: the runner never calls compile_mm_encoder.
            encoder = _RBLNGemma4VisionEncoder(
                self.vision_tower, self.embed_vision
            ).eval()
            object.__setattr__(self, "_mm_encoder", encoder)
        tower = self.vision_tower
        valid = (pixel_position_ids >= 0).all(dim=-1)
        num_patches = int(valid.sum())
        assert num_patches <= self.mm_encoder_buckets[-1], (
            f"an image of {num_patches} patches exceeds the vision encoder's "
            f"{self.mm_encoder_buckets[-1]}-patch bucket"
        )
        bucket = next(b for b in self.mm_encoder_buckets if b >= num_patches)
        # Fixed dtypes, so warm-up and serving reach the same compiled graphs.
        positions = torch.full((1, bucket, 2), -1, dtype=torch.long)
        positions[0, :num_patches] = pixel_position_ids[valid]
        patches = torch.zeros(1, bucket, pixel_values.shape[-1])
        patches[0, :num_patches] = pixel_values[valid]
        padding = ~(positions >= 0).all(dim=-1)
        key_bias = torch.where(padding, _KEY_PAD_BIAS, 0.0)[:, None, None, :]
        # The rotary tables only read the host tensor's device and dtype.
        cos, sin = tower.encoder.rotary_emb(
            torch.empty(0, dtype=self.model_dtype), positions
        )
        device = next(tower.parameters()).device
        embeds = encoder(
            patches.to(device),
            positions.to(device),
            padding.to(device),
            *(
                x.to(device=device, dtype=self.model_dtype)
                for x in (key_bias, cos, sin)
            ),
        )
        # The patches of a grid whose sides are multiples of the pooling kernel
        # pool into the leading cells, in order.
        num_soft_tokens = num_patches // tower.config.pooling_kernel_size**2
        return embeds[0, :num_soft_tokens].cpu()

    def _process_image_input(
        self, image_input: Gemma4ImageInputs
    ) -> list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]
        with torch.inference_mode():
            return [
                self._encode_vision(pixel_values[i], pixel_position_ids[i])
                for i in range(len(pixel_values))
            ]

    def _process_video_input(
        self, video_input: Gemma4VideoInputs
    ) -> list[torch.Tensor]:
        pixel_values = video_input["pixel_values_videos"]
        pixel_position_ids = video_input["pixel_position_ids_videos"]
        with torch.inference_mode():
            frames = [
                self._encode_vision(pixel_values[i], pixel_position_ids[i])
                for i in range(pixel_values.shape[0])
            ]
        frame_counts = video_input["video_frame_counts"]
        if isinstance(frame_counts, torch.Tensor):
            frame_counts = frame_counts.tolist()
        videos = []
        for count in frame_counts:
            videos.append(torch.cat(frames[:count]))
            frames = frames[count:]
        return videos

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_embeds: torch.Tensor | None = None,
        mm_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            mm_embeds is not None
            and inputs_embeds is None
            and get_pp_group().is_first_rank
        ):
            assert mm_mask is not None
            text_embeds = self.language_model.embed_input_ids(input_ids)
            inputs_embeds = torch.where(mm_mask > 0.5, mm_embeds, text_embeds)
            input_ids = None
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
