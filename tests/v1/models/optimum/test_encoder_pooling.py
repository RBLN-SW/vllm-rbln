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
"""E2E check that an encoder (BERT) model ends up with CLS sequence pooling.

A real ``VllmConfig`` is built for a real BERT encoder model, so vLLM's own
``ModelConfig`` resolves ``pooler_config`` exactly as it would in production
(``sentence-transformers/all-MiniLM-L6-v2`` resolves to ``seq_pooling_type =
"MEAN"``). The model is then built through the real
``RBLNOptimumForEncoderModel.__init__`` code path; only optimum-rbln model
compilation/loading (``init_model``) is faked, since that requires an NPU.

We assert solely that the pooling type comes out as ``"CLS"`` — i.e. the
encoder overrode the model default ("MEAN") as intended.
"""

from unittest.mock import MagicMock, patch

import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)

from vllm_rbln.model_executor.models.optimum.encoder import (
    RBLNOptimumForEncoderModel,
)
from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase

# Real encoder whose vLLM-resolved default pooling is MEAN (not CLS), so the
# override performed by RBLNOptimumForEncoderModel is actually observable.
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_encoder_forces_cls_pooling():
    model_config = ModelConfig(model=ENCODER_MODEL, dtype=torch.float32, seed=42)
    # Sanity: vLLM resolved the model default to something other than CLS.
    assert model_config.pooler_config.seq_pooling_type == "MEAN"

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
        model = RBLNOptimumForEncoderModel(vllm_config=vllm_config)

    assert model.vllm_config.model_config.pooler_config.seq_pooling_type == "CLS"
