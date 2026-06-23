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

from vllm_rbln.model_executor.models.optimum.encoder import (
    RBLNClassifierPooler,
    RBLNOptimumForEncoderModel,
)
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.utils.optimum.predicates import is_qwen3_pooling

# (model id, expected seq_pooling_type, expected pooler) per architecture.
# Encoder-only models are forced to CLS; the *ForSequenceClassification ones use
# the passthrough classifier pooler. Qwen3 embedding is decoder-based and keeps
# its native LAST-token pooling.
ENCODER_MODELS = [
    # arch: BertModel
    ("sentence-transformers/all-MiniLM-L6-v2", "CLS", DispatchPooler),
    # arch: RobertaModel
    ("sentence-transformers/nli-roberta-base-v2", "CLS", DispatchPooler),
    # arch: XLMRobertaModel
    ("intfloat/multilingual-e5-base", "CLS", DispatchPooler),
    # arch: RobertaForSequenceClassification
    ("cross-encoder/stsb-roberta-base", "CLS", RBLNClassifierPooler),
    # arch: XLMRobertaForSequenceClassification
    ("BAAI/bge-reranker-base", "CLS", RBLNClassifierPooler),
    # arch: Qwen3ForCausalLM remapped to Qwen3Model (decoder-based embedder)
    ("Qwen/Qwen3-Embedding-0.6B", "LAST", DispatchPooler),
]


def _build_encoder(model_id: str):
    model_config = ModelConfig(model=model_id, dtype=torch.float32, seed=42)
    # Mirror RBLNOptimumModelRunner: Qwen3 pooling models have their HF arch
    # (Qwen3ForCausalLM) remapped to Qwen3Model before the encoder is built.
    if is_qwen3_pooling(model_config):
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
