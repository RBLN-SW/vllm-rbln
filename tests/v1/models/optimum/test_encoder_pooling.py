# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E check that encoder models end up with the right sequence pooling.

For each real encoder architecture supported by the RBLN optimum path a real
``VllmConfig`` is built, so vLLM's own ``ModelConfig`` resolves ``pooler_config``
exactly as it would in production. The model is then built through the real
``RBLNOptimumForEncoderModel.__init__`` code path; only the optimum-rbln model
compilation/loading (``init_model``) is faked, since that requires an NPU.

Encoder-only models (BertModel / RobertaModel / XLMRobertaModel and their
``*ForSequenceClassification`` variants) are forced to CLS pooling. Decoder-based
pooling models (Qwen3 embedding) keep their native LAST-token pooling — CLS
(token 0) would be wrong for a causal model.

We assert that:
  * the pooling type comes out as expected per architecture, and
  * the right pooler is selected per task — ``DispatchPooler`` (embed) for the
    plain encoders and ``RBLNClassifierPooler`` for the classification ones.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.pooler import DispatchPooler

from vllm_rbln.model_executor.models.optimum import encoder as encoder_mod
from vllm_rbln.model_executor.models.optimum.encoder import (
    RBLNClassifierPooler,
    RBLNOptimumForEncoderModel,
    RBLNQwen3RerankerPooler,
)
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.utils.optimum.predicates import is_qwen3_pooling, is_qwen3_reranker

# Loading the original Qwen3-Reranker as a sequence classification model, exactly
# as examples/optimum/pooling_models/qwen3_reranker_score_offline.py does.
QWEN3_RERANKER_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

# (model id, expected seq_pooling_type, expected pooler) per architecture.
# Encoder-only models are forced to CLS; the *ForSequenceClassification ones use
# the passthrough classifier pooler. Qwen3 embedding is decoder-based and keeps
# its native LAST-token pooling.
ENCODER_MODELS = [
    # arch: BertModel
    ("sentence-transformers/all-MiniLM-L6-v2", "CLS", DispatchPooler),
    # arch: XLMRobertaModel
    ("intfloat/multilingual-e5-base", "CLS", DispatchPooler),
    # arch: RobertaForSequenceClassification
    ("cross-encoder/stsb-roberta-base", "CLS", RBLNClassifierPooler),
    # arch: XLMRobertaForSequenceClassification
    ("BAAI/bge-reranker-base", "CLS", RBLNClassifierPooler),
    # arch: Qwen3ForCausalLM remapped to Qwen3Model (decoder-based embedder)
    ("Qwen/Qwen3-Embedding-0.6B", "LAST", DispatchPooler),
]


def _build_encoder(model_id: str, hf_overrides: dict | None = None):
    model_config = ModelConfig(
        model=model_id, dtype=torch.float32, seed=42, hf_overrides=hf_overrides or {}
    )
    # Mirror RBLNOptimumModelRunner: Qwen3 pooling models have their HF arch
    # (Qwen3ForCausalLM, or Qwen3ForSequenceClassification for the reranker)
    # remapped to Qwen3Model before the encoder is built.
    if is_qwen3_pooling(model_config) or is_qwen3_reranker(model_config):
        model_config.hf_config.__dict__["architectures"] = ["Qwen3Model"]
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=128,
            max_model_len=128,
            is_encoder_decoder=False,
        ),
    )

    # Fake ONLY the optimum-rbln compile/load step (the only part that needs an
    # NPU); everything else runs the real encoder __init__ / pooler setup.
    def fake_init_model(self):
        self.model = MagicMock()
        self.model.get_kvcache_num_blocks.return_value = 1

    with (
        set_current_vllm_config(vllm_config),
        patch.object(RBLNOptimumModelBase, "init_model", fake_init_model),
    ):
        return RBLNOptimumForEncoderModel(vllm_config=vllm_config)


@pytest.mark.parametrize("model_id, expected_seq_pool, expected_pooler", ENCODER_MODELS)
def test_encoder_pooling(model_id, expected_seq_pool, expected_pooler):
    model = _build_encoder(model_id)

    pooler_config = model.vllm_config.model_config.pooler_config
    assert pooler_config.seq_pooling_type == expected_seq_pool
    assert isinstance(model.pooler, expected_pooler)


def test_qwen3_reranker_pooling(monkeypatch):
    """The original Qwen3-Reranker gets the host-side 2-way-softmax classifier.

    The score head itself is stubbed out: reading it needs the checkpoint, which
    is far too large to pull into a unit test. ``test_seq_cls_head`` covers the
    extraction against a synthetic checkpoint instead.
    """
    hidden_size = 1024
    monkeypatch.setattr(
        encoder_mod,
        "load_2_way_softmax_score_weight",
        lambda model_config: torch.zeros(1, hidden_size),
    )

    model = _build_encoder("Qwen/Qwen3-Reranker-0.6B", QWEN3_RERANKER_OVERRIDES)

    assert model.vllm_config.model_config.pooler_config.seq_pooling_type == "LAST"
    assert isinstance(model.pooler, RBLNQwen3RerankerPooler)
    # score() needs the classify task; embed/token_embed must not be offered.
    assert model.pooler.get_supported_tasks() == {"classify"}
