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
"""Runner-side helper for encoder-cache (EC) disaggregation.

The EC-specific producer/consumer logic lives here as a collaborator object
(`ECDisaggHelper`) owned by `RBLNOptimumModelRunner` and reached via
`self.ec_disagg.<...>`. Keeping it a collaborator (rather than a mixin) makes
the delegation explicit at the call site and lets the helper read the runner's
shared state directly, instead of declaring borrowed attributes.
"""

from typing import TYPE_CHECKING

import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_rbln.model_executor.models.optimum import ModelInputForRBLN

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner


class ECDisaggHelper:
    """Producer/consumer helper for encoder-cache disaggregation.

    Owned by `RBLNOptimumModelRunner`; reads the runner's shared state
    (``model``, ``model_config``, ``encoder_cache``, ``mrope_position_deltas``)
    and its ``maybe_save_ec_to_connector`` (from ECConnectorModelRunnerMixin).
    """

    def __init__(self, runner: "RBLNOptimumModelRunner") -> None:
        self._runner = runner

    def make_producer_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput:
        """Build a ModelRunnerOutput that tells the engine core every
        request is finished (by returning the EOS token).

        Without this, the engine keeps scheduling decode steps for a
        request that will never produce real tokens.
        """
        if not scheduler_output.num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_config = self._runner.model_config
        # Multimodal configs (e.g. Qwen3-VL) leave the top-level
        # hf_config.eos_token_id as None and carry the real value inside
        # text_config / generation_config. Walk the fallbacks so the
        # scheduler never sees a None token id.
        eos = None
        for cfg in (
            getattr(model_config, "hf_text_config", None),
            model_config.hf_config,
            getattr(model_config, "hf_generation_config", None),
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

    def run_encoder_and_save(
        self,
        model_input: ModelInputForRBLN,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Producer path: run the vision encoder (model.embed_multimodal) and
        cache the result for the consumer to merge back."""
        mm_kwargs = model_input.multi_modal_kwargs or {}
        encode_output = self._runner.model.embed_multimodal(**mm_kwargs)

        mm_hash = self._get_mm_hash_for_request(scheduler_output)
        if mm_hash is not None:
            self._runner.encoder_cache[mm_hash] = encode_output
            self._runner.maybe_save_ec_to_connector(
                self._runner.encoder_cache, mm_hash
            )

    def run_prefill_with_cached_encoder(
        self,
        model_input: ModelInputForRBLN,
        scheduler_output: "SchedulerOutput",
    ) -> torch.Tensor:
        """Consumer prefill path: gather the cached encoder outputs, let the
        model merge them (model.build_prefill_inputs_from_cache), and run the
        prefill decoder (optimum-rbln's prefill runtime)."""
        if not scheduler_output.scheduled_new_reqs:
            raise RuntimeError("EC consumer: no scheduled_new_reqs on prefill step.")
        req = scheduler_output.scheduled_new_reqs[0]
        if not req.mm_features:
            raise RuntimeError("EC consumer: request has no mm_features.")

        encoder_cache = self._runner.encoder_cache
        cached_mm_outputs: list = []
        for feat in req.mm_features:
            mm_hash = feat.identifier
            if mm_hash not in encoder_cache:
                raise RuntimeError(
                    f"EC consumer cache miss: mm_hash={mm_hash}, "
                    f"encoder_cache_keys={list(encoder_cache.keys())[:5]}, "
                    f"mm_features={[f.identifier for f in req.mm_features]}"
                )
            cached_mm_outputs.append(encoder_cache[mm_hash])

        model = self._runner.model
        input_ids = model_input.input_tokens
        kwargs = model.preprocess_for_decoder(
            True,
            model_input.block_tables,
            input_ids,
            model_input.input_positions,
        )
        cache_position = kwargs.pop("cache_position")
        block_tables = kwargs.pop("block_tables")

        prefill_params = model.build_prefill_inputs_from_cache(
            input_ids,
            cached_mm_outputs,
            cache_position=cache_position,
            running_requests_ids=model_input.running_requests_ids,
            mrope_position_deltas=self._runner.mrope_position_deltas,
        )

        language_model = model.get_language_model()
        logits = language_model.prefill_decoder(
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
