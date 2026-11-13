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
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature
from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.gemma4 import Gemma4Attention, Gemma4MLP
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
    SupportsRBLNMmPrefix,
    SupportsRBLNMultiModal,
)
from vllm_rbln.patches.attention import _resolve_kv_cache


def _build_mm_prefix_sliding_mask(
    row_end_offsets: torch.Tensor,
    chunk_start_position: torch.Tensor,
    sliding_window: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a mask over the kernel's block-aligned key strip.

    Positions are absolute token indices; offsets are relative to the chunk
    start. Row end offsets include the last key each query may attend to.
    """
    query_length = row_end_offsets.numel()
    num_key_blocks = (
        sliding_window + query_length - 1 + block_size - 1
    ) // block_size + 1
    chunk_start_position = chunk_start_position.to(torch.int32).reshape(1, 1)
    first_key_block_start_position = (
        torch.clamp(chunk_start_position - sliding_window + 1, min=0)
        // block_size
        * block_size
    )
    query_offsets = torch.arange(
        query_length, device=row_end_offsets.device, dtype=torch.int32
    ).view(-1, 1)
    key_offsets = (
        first_key_block_start_position
        - chunk_start_position
        + torch.arange(
            num_key_blocks * block_size,
            device=row_end_offsets.device,
            dtype=torch.int32,
        )
    )
    within_prefix = key_offsets <= row_end_offsets.to(torch.int32).view(-1, 1)
    within_window = key_offsets > query_offsets - sliding_window
    return (within_prefix & within_window).to(dtype).view(1, 1, query_length, -1)


class _RBLNGemma4VisionEncoder(nn.Module):
    """Gemma4's vision tower and its projection into the text model as one
    static graph: the host hands in the mask and patches padded to one count."""

    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module):
        super().__init__()
        self.tower = vision_tower
        self.embed_vision = embed_vision
        # HF's cos and sin for every position along one axis, side by side: a
        # patch's are the rows at its x and at its y.
        patch_embedder = vision_tower.patch_embedder
        axis = torch.arange(patch_embedder.position_embedding_table.shape[1])
        cos, sin = vision_tower.encoder.rotary_emb(
            torch.empty(0, dtype=patch_embedder.input_proj.weight.dtype),
            torch.stack([axis, axis], dim=-1)[None],
        )
        half = cos.shape[-1] // 2
        self.register_buffer(
            "rope_table",
            torch.cat([cos[0, :, :half], sin[0, :, :half]], dim=-1).to(
                patch_embedder.input_proj.weight.device
            ),
            persistent=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        key_bias: torch.Tensor,
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
        rope = F.embedding(pixel_position_ids.clamp(min=0), self.rope_table)
        half = rope.shape[-1] // 2
        cos, sin = rope[..., :half].flatten(-2), rope[..., half:].flatten(-2)
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


def split_gate_up_weights(model: nn.Module) -> set[str]:
    """Give every Gemma4MLP its gate and up weights apart, as the patched
    Gemma4MLP.forward reads them, and return their names for the loader's
    check. Each TP rank holds [gate shard; up shard], so the halves line up
    per rank."""
    names = set()
    for prefix, mlp in model.named_modules():
        if not isinstance(mlp, Gemma4MLP):
            continue
        if not isinstance(mlp.gate_up_proj.quant_method, UnquantizedLinearMethod):
            raise ValueError(
                "Gemma4 on RBLN splits the MLP gate_up weight and serves "
                "unquantized weights only."
            )
        gate, up = mlp.gate_up_proj.weight.detach().chunk(2, dim=0)
        mlp.gate_weight = nn.Parameter(gate.clone(), requires_grad=False)
        mlp.up_weight = nn.Parameter(up.clone(), requires_grad=False)
        del mlp.gate_up_proj.weight
        names |= {
            f"{prefix}.{name}".lstrip(".") for name in ("gate_weight", "up_weight")
        }
    return names


def mm_soft_token_cap(info: Gemma4ProcessingInfo) -> int:
    """The most soft tokens an image may take: the engine's max_soft_tokens,
    else the HF image processor's own."""
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
    engine's."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Upstream caches one HF processor per max_soft_tokens; build each
        # allowed budget's before serving.
        for budget in _SUPPORTED_SOFT_TOKENS:
            if budget <= self._mm_soft_token_cap:
                self.info.get_hf_processor(max_soft_tokens=budget)

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
                    f"of {self._mm_soft_token_cap}; raise it with "
                    "--mm-processor-kwargs at start-up."
                )
        return super()._apply_hf_processor_main(mm_items, hf_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    RBLNGemma4MultiModalProcessor,
    info=Gemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class RBLNGemma4ForConditionalGeneration(
    Gemma4ForConditionalGeneration, SupportsRBLNMultiModal, SupportsRBLNMmPrefix
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.data_parallel_size > 1:
            raise ValueError("Gemma4 does not support data parallel on RBLN.")
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "Gemma4 does not support speculative decoding on RBLN yet."
            )
        # Upstream fills per-layer inputs in its own embed_input_ids, which the
        # in-graph merge of mm embeddings (forward) bypasses.
        text_config = vllm_config.model_config.hf_config.text_config
        if getattr(text_config, "hidden_size_per_layer_input", None):
            raise ValueError(
                "Gemma4 variants with per-layer inputs (E2B, E4B) are not "
                "supported on RBLN yet."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.mm_prefix_attn_layers: list[nn.Module] = []
        for module in self.modules():
            if (
                isinstance(module, Gemma4Attention)
                and module.is_sliding
                and module.attn.use_mm_prefix
            ):
                self.mm_prefix_attn_layers.append(module.attn)
        cap = mm_soft_token_cap(
            MULTIMODAL_REGISTRY.create_processor(vllm_config.model_config).info
        )
        # Video frames take the smallest budget, so an image at the cap is the
        # longest range.
        self.max_mm_prefix_range_len = cap
        # One compiled encoder shape per budget; an image or frame is padded to
        # the smallest that holds its patches.
        rbln_config = vllm_config.additional_config
        budgets = sorted(rbln_config.encoder_token_budgets) or [
            budget for budget in _SUPPORTED_SOFT_TOKENS if budget <= cap
        ]
        if unsupported := sorted(set(budgets) - set(_SUPPORTED_SOFT_TOKENS)):
            raise ValueError(
                f"encoder_token_budgets {unsupported} are not Gemma4 soft token "
                f"budgets; pick from {list(_SUPPORTED_SOFT_TOKENS)}, the counts "
                "its image processor emits."
            )
        if budgets[-1] < cap:
            raise ValueError(
                f"encoder_token_budgets {budgets} must reach max_soft_tokens={cap}, "
                "the most soft tokens an image may take; add a budget that large "
                "or lower max_soft_tokens."
            )
        pool = self.config.vision_config.pooling_kernel_size
        self.mm_encoder_patch_buckets = [budget * pool**2 for budget in budgets]
        self._mm_encoder: Callable[..., torch.Tensor] | None
        object.__setattr__(self, "_mm_encoder", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Scale each expert's down projection instead of its routing weight:
        # the decode graph lowers the [E, 1] routing-weight scale incorrectly.
        # Folded on the checkpoint tensors, before they reach the device.
        def fold(weights):
            scales: dict[str, torch.Tensor] = {}
            downs: dict[str, torch.Tensor] = {}
            for name, w in weights:
                if name.endswith("router.per_expert_scale"):
                    key = name.removesuffix("router.per_expert_scale")
                    scales[key] = w
                    if key in downs:
                        down = downs.pop(key)
                        yield (
                            key + "experts.down_proj",
                            down * w.to(down.dtype).view(-1, 1, 1),
                        )
                elif name.endswith("experts.down_proj"):
                    key = name.removesuffix("experts.down_proj")
                    if key in scales:
                        w = w * scales[key].to(w.dtype).view(-1, 1, 1)
                    else:
                        downs[key] = w
                        continue
                yield name, w
            assert not downs, f"no per_expert_scale for {sorted(downs)}"

        return super().load_weights(fold(weights)) | split_gate_up_weights(self)

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

    def warmup_mm_encoder(self) -> None:
        if self._mm_encoder is None:
            return
        vision_config = self.config.vision_config
        pool = vision_config.pooling_kernel_size
        patch_pixels = 3 * vision_config.patch_size**2
        with torch.inference_mode():
            for bucket in self.mm_encoder_patch_buckets:
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
        assert num_patches <= self.mm_encoder_patch_buckets[-1], (
            f"an image of {num_patches} patches exceeds the vision encoder's "
            f"{self.mm_encoder_patch_buckets[-1]}-patch bucket"
        )
        bucket = next(b for b in self.mm_encoder_patch_buckets if b >= num_patches)
        # Fixed dtypes, so warm-up and serving reach the same compiled graphs.
        positions = torch.full((1, bucket, 2), -1, dtype=torch.long)
        positions[0, :num_patches] = pixel_position_ids[valid]
        patches = torch.zeros(1, bucket, pixel_values.shape[-1])
        patches[0, :num_patches] = pixel_values[valid]
        padding = ~(positions >= 0).all(dim=-1)
        key_bias = torch.where(padding, torch.finfo(self.model_dtype).min, 0.0)[
            :, None, None, :
        ]
        device = next(tower.parameters()).device
        embeds = encoder(
            patches.to(device),
            positions.to(device),
            padding.to(device),
            key_bias.to(device=device, dtype=self.model_dtype),
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
        saved_masks = []
        if self.mm_prefix_attn_layers:
            attn_metadata = get_forward_context().attn_metadata
            layer = self.mm_prefix_attn_layers[0]
            metadata = attn_metadata[layer.layer_name]
            if metadata.mm_prefix_row_ends is not None:
                assert metadata.is_prefill and metadata.swa_appends
                mask = _build_mm_prefix_sliding_mask(
                    metadata.mm_prefix_row_ends,
                    metadata.seq_lens,
                    layer.sliding_window,
                    _resolve_kv_cache(metadata, layer.layer_index).size(-2),
                    self.model_dtype,
                )
                # Workaround: borrow metadata without mutating Attention modules.
                saved_masks = [
                    (
                        attn_metadata[layer.layer_name],
                        attn_metadata[layer.layer_name].swa_attn_masks,
                    )
                    for layer in self.mm_prefix_attn_layers
                ]
                for metadata, _ in saved_masks:
                    metadata.swa_attn_masks = mask
        try:
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
        finally:
            for metadata, original_mask in saved_masks:
                metadata.swa_attn_masks = original_mask
