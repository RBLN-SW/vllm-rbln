# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Decision 3.0 (vllm-sr/d3) on the text backbone.

Vendored from upstream vLLM #61565 (`vllm/model_executor/models/d3.py`), which
builds on `Qwen3_5ForConditionalGeneration`. The vllm model path has no
multimodal input, so this one builds on `Qwen3_5ForCausalLM` and drops the
vision tower's weights. The checkpoint is a bare `Qwen3_5Model` plus
`readout.safetensors`, one row per answer code, served as the `score` head.
"""

import itertools
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.adapters import as_seq_cls_model
from vllm.model_executor.models.config import Qwen3_5ForCausalLMConfig
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import WeightsMapper
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

READOUT_WEIGHTS = "readout.safetensors"
DECISION_CONFIG = "decision_config.json"


class D3ForSequenceClassificationConfig(Qwen3_5ForCausalLMConfig):
    """Apply the contract declared by a Decision 3.0 `decision_config.json`."""

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        decision = get_hf_file_to_dict(
            DECISION_CONFIG, model_config.model, model_config.revision
        )
        if decision is None:
            raise ValueError("Decision 3.0 checkpoints need a decision_config.json")
        if decision.get("format_version") != 1:
            raise ValueError(
                "unsupported decision_config.json format_version "
                f"{decision.get('format_version')!r}; vLLM reads version 1"
            )
        if decision.get("pooling", "last") != "last":
            raise ValueError(
                f"unsupported decision_config.json pooling {decision['pooling']!r}"
            )
        attention_mode = decision.get("attention_mode", "causal")
        if attention_mode not in ("causal", "noncausal_full_attention"):
            raise ValueError(
                f"unsupported decision_config.json attention_mode {attention_mode!r}"
            )
        codes = decision.get("codes")
        token_ids = decision.get("token_ids")
        if not codes or not isinstance(token_ids, list) or len(codes) != len(token_ids):
            raise ValueError(
                "decision_config.json must list one token id per answer code"
            )

        hf_config = model_config.hf_config
        # The full attention layers read the whole prompt in both directions;
        # the Gated DeltaNet layers stay causal.
        is_causal = attention_mode == "causal"
        for config in {
            id(config): config for config in (hf_config, model_config.hf_text_config)
        }.values():
            config.is_causal = is_causal
        # One readout row per answer code.
        hf_config.id2label = dict(enumerate(codes))
        hf_config.decision_config = decision

        # Probabilities only exist over a question's own options, so the head
        # returns raw code logits unless activation is requested.
        pooler_config = model_config.pooler_config
        assert pooler_config is not None
        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM as _D3ForSequenceClassificationBase,
    )
else:
    _D3ForSequenceClassificationBase = as_seq_cls_model(Qwen3_5ForCausalLM)


@default_pooling_type(seq_pooling_type="LAST")
class D3ForSequenceClassification(_D3ForSequenceClassificationBase):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "model.",
            "visual.": None,
            "mtp.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model_name = vllm_config.model_config.model
        self.revision = vllm_config.model_config.revision

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Upstream reads the readout through `secondary_weights`; vLLM 0.31
        # checks every source against the checkpoint's index, which does not
        # list the readout, so it is appended to the backbone's weights here.
        if os.path.isdir(self.model_name):
            path = os.path.join(self.model_name, READOUT_WEIGHTS)
        else:
            path = hf_hub_download(
                self.model_name, READOUT_WEIGHTS, revision=self.revision
            )
        readout = load_file(path)["weight"]
        return super().load_weights(
            itertools.chain(weights, [("score.weight", readout)])
        )
