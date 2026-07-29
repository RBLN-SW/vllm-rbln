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
"""Qwen3-Reranker driven through ``score()``.

The reranker reuses the *generation* graph so it can read the score off two
vocabulary logits. These tests cover the parts that do not need an NPU:

  * the HF arch is remapped to ``Qwen3ForCausalLM``, so the compile spec and
    optimum-rbln class match the generative path and share its artifact,
  * ``classifier_from_token`` resolves to the right vocabulary ids,
  * inputs are **left** padded, which the generation graph requires,
  * ``forward`` reduces the vocabulary to ``logit_yes - logit_no``, and the
    pooler's sigmoid on that equals the 2-way softmax.
"""

from types import SimpleNamespace
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

from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.model_executor.models.optimum.qwen3_reranker import (
    RBLNOptimumQwen3RerankerModel,
    RBLNQwen3RerankerPooler,
    resolve_label_token_ids,
)
from vllm_rbln.utils.optimum.converter.common import is_chunked_prefill_arch
from vllm_rbln.utils.optimum.predicates import is_qwen3_embedding, is_qwen3_reranker
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_generation_arch,
    is_pooling_arch,
)

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
# Exactly what examples/optimum/pooling_models/qwen3_reranker_score_offline.py passes.
RERANKER_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
NO_ID, YES_ID = 2152, 9693


def _model_config(**overrides):
    return ModelConfig(
        model=MODEL_ID,
        dtype=torch.float32,
        seed=42,
        runner="pooling",
        hf_overrides={**RERANKER_OVERRIDES, **overrides},
    )


def _build_reranker(max_seq_len=16, batch_size=2):
    model_config = _model_config()
    # Mirror RBLNOptimumModelRunner's remap.
    assert is_qwen3_reranker(model_config)
    model_config.hf_config.__dict__["architectures"] = ["Qwen3ForCausalLM"]
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=batch_size,
            max_num_batched_tokens=128,
            max_model_len=128,
            is_encoder_decoder=False,
        ),
    )

    # Fake ONLY the optimum-rbln compile/load step; the rest is the real path.
    def fake_init_model(self):
        self.model = MagicMock()
        self.model.get_kvcache_num_blocks.return_value = 1
        self.rbln_model_config = SimpleNamespace(max_seq_len=max_seq_len)

    with (
        set_current_vllm_config(vllm_config),
        patch.object(RBLNOptimumModelBase, "init_model", fake_init_model),
    ):
        return RBLNOptimumQwen3RerankerModel(vllm_config=vllm_config)


def test_predicate_survives_arch_remap():
    """The predicate keys on classifier_from_token, not the arch name."""
    model_config = _model_config()
    assert is_qwen3_reranker(model_config)
    model_config.hf_config.__dict__["architectures"] = ["Qwen3ForCausalLM"]
    assert is_qwen3_reranker(model_config), "must still hold after the remap"


def test_label_token_ids():
    assert resolve_label_token_ids(_model_config()) == (NO_ID, YES_ID)


def test_not_confused_with_qwen3_embedding():
    """Both load RBLNQwen3ForCausalLM under a pooling runner, but differ.

    The embedder pools hidden states; the reranker reads vocabulary logits. If
    `is_qwen3_embedding` also matched, the reranker would be remapped to
    Qwen3Model and lose lm_head.
    """
    model_config = _model_config()

    assert is_qwen3_reranker(model_config)
    assert not is_qwen3_embedding(model_config)

    embedder = ModelConfig(
        model="Qwen/Qwen3-Embedding-0.6B", dtype=torch.float32, seed=42
    )
    assert is_qwen3_embedding(embedder)
    assert not is_qwen3_reranker(embedder)


def test_prefix_caching_is_disabled():
    """It is served by a pooling runner, which cannot use prefix caching."""
    model = _build_reranker()
    assert model.vllm_config.cache_config.enable_prefix_caching in (False, None)


def test_registered_as_a_generation_arch():
    """It is served by score(), but it runs the generation graph.

    Classifying it as a pooling arch makes `max_num_batched_tokens` a
    full-prefill budget, i.e. a compiled prefill_chunk_size equal to
    max_model_len, and the decoder graph cannot be built that way.
    """
    hf_config = _model_config().hf_config

    assert is_generation_arch(hf_config)
    assert not is_pooling_arch(hf_config)
    assert is_chunked_prefill_arch(hf_config)
    assert get_rbln_model_info(hf_config) == ("qwen3", "RBLNQwen3ForCausalLM")


def _bare_config(tokens):
    """A config that skips vLLM's own validation of `classifier_from_token`.

    vLLM asserts on it in `Qwen3ForSequenceClassificationConfig`, but only when
    the architecture resolves to `Qwen3ForSequenceClassification`. Setting just
    `classifier_from_token` (which `is_qwen3_reranker` accepts) leaves the
    architecture alone, so that hook never runs and our own check is what stops
    a bad value.
    """
    hf_config = SimpleNamespace(classifier_from_token=tokens)
    hf_config.get_text_config = lambda: hf_config
    return SimpleNamespace(
        hf_config=hf_config,
        tokenizer=MODEL_ID,
        tokenizer_revision=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
    )


@pytest.mark.parametrize("tokens", [None, [], ["yes"], ["no", "maybe", "yes"]])
def test_rejects_bad_classifier_from_token(tokens):
    with pytest.raises(ValueError, match="exactly two"):
        resolve_label_token_ids(_bare_config(tokens))


def test_rejects_unknown_label_token():
    with pytest.raises(ValueError, match="not single tokens"):
        resolve_label_token_ids(_bare_config(["no", "nope"]))


def test_vllm_also_rejects_bad_classifier_from_token():
    """Belt and braces: with the arch override, vLLM catches it at config time."""
    with pytest.raises(Exception, match="Try loading the original Qwen3 Reranker"):
        _model_config(classifier_from_token=["yes"])


def test_builds_reranker_pooler():
    model = _build_reranker()

    assert isinstance(model.pooler, RBLNQwen3RerankerPooler)
    # score() needs classify; embed/token_embed must not be offered.
    assert model.pooler.get_supported_tasks() == {"classify"}
    assert (model.false_id, model.true_id) == (NO_ID, YES_ID)


def test_inputs_are_left_padded():
    """The generation graph rejects right-padded masks outright."""
    model = _build_reranker(max_seq_len=8, batch_size=2)
    input_ids = torch.arange(1, 4, dtype=torch.long).unsqueeze(0)  # one request, 3 ids

    padded_ids, mask = model.preprocess(input_ids)

    assert padded_ids.shape == (2, 8)
    assert mask.shape == (2, 8)
    # Real tokens sit at the end, padding at the front.
    assert padded_ids[0].tolist() == [0, 0, 0, 0, 0, 1, 2, 3]
    assert mask[0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    # The unused batch slot stays fully masked.
    assert mask[1].sum() == 0


def test_rejects_overlong_input():
    model = _build_reranker(max_seq_len=4)
    with pytest.raises(ValueError, match="exceeds the compiled maximum"):
        model.preprocess(torch.zeros(1, 5, dtype=torch.long))


def test_forward_reduces_to_2_way_softmax():
    """sigmoid(forward()) must equal softmax over the yes/no logits."""
    model = _build_reranker(max_seq_len=8, batch_size=2)
    vocab_size = model.model_config.hf_config.vocab_size

    torch.manual_seed(0)
    # The graph returns the final position only: [batch, 1, vocab_size].
    logits = torch.randn(2, 1, vocab_size)
    model.model.forward = MagicMock(return_value=SimpleNamespace(logits=logits))

    num_requests = 2
    model_input = SimpleNamespace(
        input_tokens=torch.zeros(num_requests, 5, dtype=torch.long)
    )
    out = model.forward(model_input)

    assert out.shape == (num_requests, 1)
    expected = torch.softmax(logits[:, 0, [NO_ID, YES_ID]], dim=-1)[:, 1]
    torch.testing.assert_close(torch.sigmoid(out.squeeze(-1)), expected)
