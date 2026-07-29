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

"""Unit tests for the host-side Qwen3-Reranker score head.

The head must equal ``w_yes - w_no`` so that ``sigmoid(h @ w)`` reproduces the
2-way softmax over the reranker's "yes"/"no" logits. Both are checked against a
synthetic checkpoint, so no real weights are downloaded.
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_rbln.model_executor.models.optimum import seq_cls_head

VOCAB_SIZE = 32
HIDDEN_SIZE = 8
FALSE_ID = 3
TRUE_ID = 7
TOKEN_IDS = {"no": FALSE_ID, "yes": TRUE_ID}


@pytest.fixture
def checkpoint(tmp_path):
    """A one-shard checkpoint with tied embeddings, like Qwen3-Reranker."""
    embed = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(
        VOCAB_SIZE, HIDDEN_SIZE
    )
    save_file({"model.embed_tokens.weight": embed}, str(tmp_path / "model.safetensors"))
    return tmp_path, embed


def _model_config(path, *, tokens=("no", "yes"), tie_word_embeddings=True):
    hf_config = SimpleNamespace(
        _name_or_path=str(path),
        classifier_from_token=list(tokens) if tokens is not None else None,
        tie_word_embeddings=tie_word_embeddings,
        is_original_qwen3_reranker=True,
    )
    hf_config.get_text_config = lambda: hf_config
    return SimpleNamespace(
        hf_config=hf_config,
        model=str(path),
        tokenizer=str(path),
        tokenizer_revision=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
    )


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    tokenizer = SimpleNamespace(convert_tokens_to_ids=TOKEN_IDS.get)
    monkeypatch.setattr(seq_cls_head, "get_tokenizer", lambda *a, **kw: tokenizer)


def test_score_weight_is_true_minus_false(checkpoint):
    path, embed = checkpoint

    weight = seq_cls_head.load_2_way_softmax_score_weight(_model_config(path))

    assert weight.shape == (1, HIDDEN_SIZE)
    assert weight.dtype == torch.float32
    torch.testing.assert_close(weight[0], embed[TRUE_ID] - embed[FALSE_ID])


def test_score_matches_2_way_softmax(checkpoint):
    """sigmoid(h @ w) must equal the 2-way softmax over the yes/no logits."""
    path, embed = checkpoint
    weight = seq_cls_head.load_2_way_softmax_score_weight(_model_config(path))

    hidden = torch.randn(4, HIDDEN_SIZE, generator=torch.manual_seed(0))
    # With tied embeddings the "lm_head" logits are hidden @ embed.T.
    logits = hidden @ embed.T
    expected = torch.softmax(logits[:, [FALSE_ID, TRUE_ID]], dim=-1)[:, 1]

    torch.testing.assert_close(torch.sigmoid(hidden @ weight[0]), expected)


def test_untied_checkpoint_reads_lm_head(tmp_path):
    embed = torch.zeros(VOCAB_SIZE, HIDDEN_SIZE)
    lm_head = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(
        VOCAB_SIZE, HIDDEN_SIZE
    )
    save_file(
        {"model.embed_tokens.weight": embed, "lm_head.weight": lm_head},
        str(tmp_path / "model.safetensors"),
    )

    weight = seq_cls_head.load_2_way_softmax_score_weight(
        _model_config(tmp_path, tie_word_embeddings=False)
    )

    torch.testing.assert_close(weight[0], lm_head[TRUE_ID] - lm_head[FALSE_ID])


@pytest.mark.parametrize("tokens", [None, [], ["yes"], ["no", "maybe", "yes"]])
def test_rejects_bad_classifier_from_token(checkpoint, tokens):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="exactly two"):
        seq_cls_head.load_2_way_softmax_score_weight(_model_config(path, tokens=tokens))


def test_rejects_unknown_label_token(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="not single tokens"):
        seq_cls_head.load_2_way_softmax_score_weight(
            _model_config(path, tokens=("no", "nope"))
        )
