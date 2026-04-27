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
"""Runner-side helpers for encoder-cache (EC) disaggregation.

Kept as a mixin so `RBLNOptimumModelRunner` keeps a stable public
surface (all call sites are `self._make_producer_output(...)` etc.) while
the EC-specific logic lives in its own module.
"""

from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_rbln.model_executor.models.optimum import ModelInputForRBLN

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class ECDisaggHelpersMixin:
    """Producer/consumer helpers for encoder-cache disaggregation.

    Expects the host class to provide:
      - self.model, self.model_config, self.encoder_cache
      - self.maybe_save_ec_to_connector (from ECConnectorModelRunnerMixin)
    """

    # Attributes and methods supplied by the host class / sibling mixins.
    # Declared here so mypy sees them when type-checking this file in isolation.
    if TYPE_CHECKING:
        model: Any
        model_config: "ModelConfig"
        encoder_cache: dict[str, Any]

        def maybe_save_ec_to_connector(
            self, encoder_cache: dict[str, Any], mm_hash: str
        ) -> None: ...

    def _make_producer_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Build a ModelRunnerOutput that tells the engine core every
        request is finished (by returning the EOS token).

        Without this, the engine keeps scheduling decode steps for a
        request that will never produce real tokens.
        """
        if not scheduler_output.num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Multimodal configs (e.g. Qwen3-VL) leave the top-level
        # hf_config.eos_token_id as None and carry the real value inside
        # text_config / generation_config. Walk the fallbacks so the
        # scheduler never sees a None token id.
        eos = None
        for cfg in (
            getattr(self.model_config, "hf_text_config", None),
            self.model_config.hf_config,
            getattr(self.model_config, "hf_generation_config", None),
        ):
            if cfg is None:
                continue
            cand = getattr(cfg, "eos_token_id", None)
            if isinstance(cand, list):
                cand = next((x for x in cand if x is not None), None)
            if cand is not None:
                eos = cand
                break
        if eos is None:
            eos = 0

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: idx for idx, rid in enumerate(req_ids)},
            sampled_token_ids=[[eos] for _ in req_ids],
        )

    def _run_encoder_and_save(
        self,
        model_input: ModelInputForRBLN,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Run the vision encoder only and save results to the EC
        connector.  Producer-only path — sends encode() output
        (image_embeds/video_embeds + grid_thw) instead of the full
        preprocess_prefill result."""
        image_input = None
        video_input = None
        if model_input.multi_modal_kwargs:
            image_input = self.model._parse_and_validate_image_input(
                **model_input.multi_modal_kwargs
            )
            video_input = self.model._parse_and_validate_video_input(
                **model_input.multi_modal_kwargs
            )

        encode_output = self.model.encode(image_input, video_input)

        mm_hash = self._get_mm_hash_for_request(scheduler_output)
        if mm_hash is not None:
            self.encoder_cache[mm_hash] = encode_output
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

    def _run_decoder_with_cached_encoder(
        self,
        model_input: ModelInputForRBLN,
        scheduler_output: "SchedulerOutput",
    ) -> torch.Tensor:
        """Consumer path: reconstruct embedding inputs from all cached
        per-mm_feature encode() outputs and run preprocess_prefill +
        prefill_decoder.

        Each image/video in the request is a separate mm_feature with its
        own identifier; the producer caches one encode() result per
        mm_feature. This walks every mm_feature of the new request in
        order, fetches its cached embeddings, and concatenates them so
        the LLM sees the full embedding sequence for the whole request.
        """
        if not scheduler_output.scheduled_new_reqs:
            raise RuntimeError("EC consumer: no scheduled_new_reqs on prefill step.")
        req = scheduler_output.scheduled_new_reqs[0]
        if not req.mm_features:
            raise RuntimeError("EC consumer: request has no mm_features.")

        model_dtype = self.model.dtype

        image_caches: list[dict] = []
        video_caches: list[dict] = []
        for feat in req.mm_features:
            mm_hash = feat.identifier
            if mm_hash not in self.encoder_cache:
                raise RuntimeError(
                    f"EC consumer cache miss: mm_hash={mm_hash}, "
                    f"encoder_cache_keys={list(self.encoder_cache.keys())[:5]}, "
                    f"mm_features={[f.identifier for f in req.mm_features]}"
                )
            cached = self.encoder_cache[mm_hash]
            modality = getattr(feat, "modality", None)
            if modality == "image" or (modality is None and "image_embeds" in cached):
                image_caches.append(cached)
            elif modality == "video" or (modality is None and "video_embeds" in cached):
                video_caches.append(cached)
            else:
                # Fallback: include into whichever modality is populated.
                if "image_embeds" in cached:
                    image_caches.append(cached)
                if "video_embeds" in cached:
                    video_caches.append(cached)

        def _concat_deepstack(
            caches: list[dict], key: str
        ) -> list[torch.Tensor] | None:
            """Concat per-layer deepstack tensors across features along dim=0."""
            present = [c for c in caches if c.get(key) is not None]
            if not present:
                return None
            num_layers = len(present[0][key])
            out: list[torch.Tensor] = []
            for layer in range(num_layers):
                out.append(
                    torch.cat([c[key][layer].to(model_dtype) for c in present], dim=0)
                )
            return out

        image_input = None
        video_input = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if image_caches:
            image_embeds = torch.cat(
                [c["image_embeds"].to(model_dtype) for c in image_caches], dim=0
            )
            image_grid_thw = torch.cat(
                [c["image_grid_thw"].to(torch.int64) for c in image_caches], dim=0
            )
            image_input = self.model._create_image_embedding_inputs(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )
            deepstack_image_embeds = _concat_deepstack(
                image_caches, "deepstack_image_embeds"
            )

        if video_caches:
            video_embeds = torch.cat(
                [c["video_embeds"].to(model_dtype) for c in video_caches], dim=0
            )
            video_grid_thw = torch.cat(
                [c["video_grid_thw"].to(torch.int64) for c in video_caches], dim=0
            )
            video_input = self.model._create_video_embedding_inputs(
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )
            # Qwen2.5-VL: second_per_grid_ts is per-video metadata; carry
            # the first feature's value as a best-effort for mixed batches.
            if "second_per_grid_ts" in video_caches[0]:
                video_input["second_per_grid_ts"] = video_caches[0][
                    "second_per_grid_ts"
                ]
            deepstack_video_embeds = _concat_deepstack(
                video_caches, "deepstack_video_embeds"
            )

        input_ids = model_input.input_tokens
        attention_mask = torch.ones_like(input_ids)
        prefill_params = self.model.preprocess_prefill(
            input_ids,
            attention_mask,
            image_input,
            video_input,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
        )

        rope_deltas = prefill_params.pop("rope_deltas", None)
        if rope_deltas is not None:
            cur_request_id = model_input.running_requests_ids[0]
            self.model.rope_deltas[cur_request_id] = rope_deltas.item()

        kwargs = self.model.preprocess_for_decoder(
            True,
            model_input.block_tables,
            model_input.input_tokens,
            model_input.input_positions,
        )
        block_tables = kwargs.pop("block_tables")
        logits = self.model.model.prefill_decoder(
            **prefill_params,
            block_tables=block_tables,
        ).logits
        return logits

    @staticmethod
    def _get_mm_hash_for_request(
        scheduler_output: "SchedulerOutput",
    ) -> str | None:
        """Get the mm_hash from the first mm_feature of the first new request."""
        if not scheduler_output.scheduled_new_reqs:
            return None
        req = scheduler_output.scheduled_new_reqs[0]
        if not req.mm_features:
            return None
        return req.mm_features[0].identifier
