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
"""Design sketch: Qwen-VL integration assuming a factored optimum-rbln API.

This is a **proposal / reference file**. It is intentionally NOT imported by
``__init__.py`` and therefore not registered — instantiating these classes today
would fail because the assumed optimum-rbln public methods do not exist yet.

The current production implementation lives in ``qwen_vl.py`` and reimplements
optimum-rbln's ``_preprocess_prefill`` body (the MRoPE loop, deepstack prep) on
top of private attributes (``_get_rope_index_func``, ``_get_position_embeddings``,
``_prepare_deepstack``, ``config.*_token_id``). That coupling exists only because
``_preprocess_prefill`` fuses three concerns and does not accept a precomputed
``inputs_embeds``.

This file shows what the vLLM-RBLN side collapses to once optimum-rbln exposes
those concerns as composable public methods. All model-type differences
(``second_per_grid_ts``, Qwen3 temporal-chunk video-grid slicing, deepstack)
move behind the optimum-rbln API, so a single class covers Qwen2/2.5/3-VL and the
per-variant subclasses become empty registry aliases.

Assumed optimum-rbln public API on ``RBLNQwen*VLModel``
------------------------------------------------------
``encode_vision(pixel_values=None, pixel_values_videos=None, image_grid_thw=None,``
``               video_grid_thw=None, second_per_grid_ts=None) -> dict``
    Run the visual encoder and normalize outputs into a dict:
    ``{image_embeds, [deepstack_image_embeds], video_embeds,``
    `` [deepstack_video_embeds], image_grid_thw, video_grid_thw,``
    `` [second_per_grid_ts]}``. Absorbs the tuple-vs-tensor return contract.

``merge_multimodal_embeddings(input_ids, *, image_embeds=None, video_embeds=None,``
``                            inputs_embeds=None) -> inputs_embeds``
    Text-embedding lookup (when ``inputs_embeds`` is None) followed by scattering
    each modality onto its placeholder positions, with the token/feature count
    check. Owns the placeholder-token-id knowledge; scatters image and video in
    two independent passes (order-invariant).

``compute_mrope_prefill(input_ids, attention_mask, *, image_grid_thw=None,``
``                      video_grid_thw=None, second_per_grid_ts=None)``
``                      -> (position_embed, rope_deltas)``
    Per-request MRoPE. ``position_embed`` has the documented shape
    ``[2, batch, 1, seq, head_dim]``. Owns the batch loop, ``get_rope_index``,
    model-specific video-grid slicing, and ``second_per_grid_ts`` handling.

``prepare_deepstack_prefill(input_ids, *, image_embeds=None, video_embeds=None,``
``                          deepstack_image_embeds=None,``
``                          deepstack_video_embeds=None)``
``                          -> (visual_pos_mask, deepstack_embeds)``
    Qwen3-VL only. Absent on models without deepstack, so presence is the
    variant switch (``hasattr``).

``compute_mrope_decode(cache_position, rope_delta) -> position_embed``
    Single-step decode MRoPE for one request.
"""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings

from vllm_rbln.utils.optimum.bucket import select_bucket_size

from .base import ModelInputForRBLN
from .model_base import (
    RBLNOptimumDecoderMixin,
    RBLNOptimumModelBase,
    RBLNOptimumMultimodalMixin,
)


class RBLNOptimumQwenVLForConditionalGeneration(
    RBLNOptimumModelBase, RBLNOptimumMultimodalMixin, RBLNOptimumDecoderMixin
):
    """Unified Qwen2/2.5/3-VL integration.

    vLLM-RBLN only orchestrates *when* to call *what*; every model-specific
    computation lives behind the optimum-rbln public API. The prefill path is
    ``encode_vision → merge_multimodal_embeddings → compute_mrope_prefill``,
    plus ``prepare_deepstack_prefill`` when the model carries deepstack features.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config=vllm_config)
        if self._is_ec_producer_only():
            return
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

    def get_language_model(self):
        return self.model

    # ----- encode (EC producer / non-EC prefill) -------------------------------

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | dict:
        # The encoder + output normalization is owned by optimum-rbln. Here we
        # only forward the raw multimodal kwargs when at least one modality is
        # present; the empty list keeps the "no multimodal input" contract.
        if kwargs.get("pixel_values") is None and (
            kwargs.get("pixel_values_videos") is None
        ):
            return []
        return self.model.encode_vision(**kwargs)

    # ----- merge (text lookup + scatter) ---------------------------------------

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | dict | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mm = multimodal_embeddings or {}
        return self.model.merge_multimodal_embeddings(
            input_ids,
            image_embeds=mm.get("image_embeds"),
            video_embeds=mm.get("video_embeds"),
        )

    # ----- prefill (non-EC): encode -> merge -> mrope --------------------------

    def build_prefill_forward_inputs(
        self,
        model_input: ModelInputForRBLN,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
        input_ids = model_input.input_tokens
        mm = self.embed_multimodal(**(model_input.multi_modal_kwargs or {})) or {}
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = self.embed_input_ids(input_ids, mm)
        position_embed, rope_deltas = self.model.compute_mrope_prefill(
            input_ids,
            attention_mask,
            image_grid_thw=mm.get("image_grid_thw"),
            video_grid_thw=mm.get("video_grid_thw"),
            second_per_grid_ts=mm.get("second_per_grid_ts"),
        )
        return inputs_embeds, position_embed, rope_deltas.item()

    # ----- prefill (EC consumer): cached merge -> merge -> mrope (+deepstack) ---

    def build_prefill_inputs_from_cache(
        self,
        input_ids: torch.Tensor,
        cached_mm_outputs: list[dict],
        *,
        cache_position: torch.Tensor | None = None,
        running_requests_ids: list[str] | None = None,
        mrope_position_deltas: dict[str, float] | None = None,
    ) -> dict:
        mm = self._merge_cached_mm(cached_mm_outputs)
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = self.embed_input_ids(input_ids, mm)
        position_embed, rope_deltas = self.model.compute_mrope_prefill(
            input_ids,
            attention_mask,
            image_grid_thw=mm.get("image_grid_thw"),
            video_grid_thw=mm.get("video_grid_thw"),
            second_per_grid_ts=mm.get("second_per_grid_ts"),
        )
        if running_requests_ids and mrope_position_deltas is not None:
            mrope_position_deltas[running_requests_ids[0]] = rope_deltas.item()

        prefill_params = {
            "inputs_embeds": inputs_embeds,
            "position_embed": position_embed,
        }
        # Deepstack support is a model capability, not a vLLM-RBLN decision:
        # models that carry it expose prepare_deepstack_prefill, others don't.
        if hasattr(self.model, "prepare_deepstack_prefill"):
            visual_pos_mask, deepstack_embeds = self.model.prepare_deepstack_prefill(
                input_ids,
                image_embeds=mm.get("image_embeds"),
                video_embeds=mm.get("video_embeds"),
                deepstack_image_embeds=mm.get("deepstack_image_embeds"),
                deepstack_video_embeds=mm.get("deepstack_video_embeds"),
            )
            prefill_params["visual_pos_mask"] = visual_pos_mask
            prefill_params["deepstack_embeds"] = deepstack_embeds
        return prefill_params

    def _merge_cached_mm(self, cached_mm_outputs: list[dict]) -> dict:
        """Concatenate cached per-request encoder outputs into one batch dict.

        Pure tensor bookkeeping — no model knowledge. Everything downstream
        (scatter, MRoPE, deepstack) consumes this same dict shape that
        ``encode_vision`` produces.
        """
        model_dtype = self.dtype

        def _cat(caches, key, dtype=model_dtype):
            return torch.cat([c[key].to(dtype) for c in caches], dim=0)

        def _cat_deepstack(caches, key):
            present = [c for c in caches if c.get(key) is not None]
            if not present:
                return None
            num_layers = len(present[0][key])
            return [
                torch.cat([c[key][layer].to(model_dtype) for c in present], dim=0)
                for layer in range(num_layers)
            ]

        image_caches = [c for c in cached_mm_outputs if "image_embeds" in c]
        video_caches = [c for c in cached_mm_outputs if "video_embeds" in c]

        mm: dict = {}
        if image_caches:
            mm["image_embeds"] = _cat(image_caches, "image_embeds")
            mm["image_grid_thw"] = _cat(image_caches, "image_grid_thw", torch.int64)
            deepstack = _cat_deepstack(image_caches, "deepstack_image_embeds")
            if deepstack is not None:
                mm["deepstack_image_embeds"] = deepstack
        if video_caches:
            mm["video_embeds"] = _cat(video_caches, "video_embeds")
            mm["video_grid_thw"] = _cat(video_caches, "video_grid_thw", torch.int64)
            # Per-video metadata; carry the first feature's value for mixed batches.
            if "second_per_grid_ts" in video_caches[0]:
                mm["second_per_grid_ts"] = video_caches[0]["second_per_grid_ts"]
            deepstack = _cat_deepstack(video_caches, "deepstack_video_embeds")
            if deepstack is not None:
                mm["deepstack_video_embeds"] = deepstack
        return mm

    # ----- decode --------------------------------------------------------------

    def compute_decode_position_embed(
        self,
        model_input: ModelInputForRBLN,
        mrope_position_deltas: dict[str, float],
    ) -> torch.Tensor:
        # int32 mirrors the cache_position dtype the decode path expects.
        cache_position = model_input.input_positions.to(torch.int32)
        running_requests_ids = model_input.running_requests_ids
        padded_batch_size = self.decoder_batch_size
        if self.use_multiple_decoder:
            padded_batch_size = select_bucket_size(
                len(running_requests_ids), self.decoder_batch_sizes
            )

        position_embeds = [
            self.model.compute_mrope_decode(
                cache_position[b_id], mrope_position_deltas[request_id]
            )
            for b_id, request_id in enumerate(running_requests_ids)
        ]
        pad = padded_batch_size - len(running_requests_ids)
        position_embeds += [torch.zeros_like(position_embeds[0])] * pad
        return torch.cat(position_embeds, dim=1)

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        # FIXME This should be removed in the future
        # by moving the padding logic into model runner.
        assert len(model_input.running_requests_ids) == request_nums, (
            f"The number of running requests is "
            f"{len(model_input.running_requests_ids)}, "
            f"but the shape of input_ids is {input_ids.shape}"
        )

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            logits = self.model.prefill_decoder(
                inputs_embeds=model_input.inputs_embeds,
                position_embed=model_input.position_embed,
                block_tables=block_tables,
            ).logits
        else:
            padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
            self.model.decoder = self.model.decoders[padded_batch_size]
            input_ids = kwargs.pop("input_ids")
            inputs_embeds = self.model.embed_tokens(input_ids).to(self.dtype)
            logits = self.model.decoder(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_embed=model_input.position_embed,
                block_tables=block_tables,
            ).logits
        if not is_prompt:
            logits = logits[:request_nums]
        return logits


# --- registry aliases -------------------------------------------------------
# With the model-specific logic behind the optimum-rbln API, the per-variant
# classes carry no behavior; they exist only so the architecture-name registry
# in __init__.py can map each HF architecture to a class. (Not wired up here —
# this is a proposal file.)


class RBLNOptimumQwen2VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    pass


class RBLNOptimumQwen2_5_VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    pass


class RBLNOptimumQwen3VLForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    pass


class RBLNOptimumQwen3VLMoeForConditionalGeneration(
    RBLNOptimumQwenVLForConditionalGeneration
):
    pass
