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

# RBLN's split TP layout keeps token embeddings replicated (tp_size pinned to 1)
# while the LM head stays vocabulary-sharded.

from types import SimpleNamespace
from typing import Any

import torch

from vllm_rbln.model_executor.layers.vocab_parallel_embedding import (
    RBLNParallelLMHead,
    RBLNVocabParallelEmbedding,
)


def _embedding(num_embeddings=1000, embedding_dim=16):
    return RBLNVocabParallelEmbedding(
        num_embeddings=num_embeddings, embedding_dim=embedding_dim
    )


def _lm_head(*, tp_size, quant_config=None) -> Any:
    lm: Any = object.__new__(RBLNParallelLMHead)
    lm.tp_size = tp_size
    lm.quant_config = quant_config
    return lm


class TestRBLNVocabParallelEmbedding:
    def test_token_embeddings_are_replicated_not_tp_sharded(self):
        # The RBLN divergence: token embeddings stay replicated even under TP, so
        # tp_size is pinned to 1 and the whole padded vocab lives on every rank.
        emb = _embedding(1000, 16)
        assert emb.tp_size == 1
        assert emb.num_embeddings_per_partition == emb.num_embeddings_padded

    def test_vocab_dimension_is_padded(self):
        emb = _embedding(1000, 16)
        assert emb.num_embeddings_padded >= 1000
        assert emb.weight.shape == (emb.num_embeddings_padded, 16)

    def test_forward_is_a_plain_lookup_without_tp_collectives(self):
        # tp_size == 1 -> no vocab masking, no all-reduce; forward is just the
        # local embedding lookup.
        emb = _embedding(64, 8)
        with torch.no_grad():
            emb.weight.copy_(
                torch.arange(emb.weight.numel(), dtype=emb.weight.dtype).reshape(
                    emb.weight.shape
                )
            )
        ids = torch.tensor([0, 5, 63])
        assert torch.equal(emb.forward(ids), emb.weight[ids])


class TestRBLNParallelLMHeadTieWeights:
    def test_aliases_weight_when_single_rank(self):
        # Single-rank: the LM head can share the replicated embedding weight.
        lm = _lm_head(tp_size=1)
        embed = SimpleNamespace(weight=torch.ones(4, 4))
        result = lm.tie_weights(embed)
        assert result is lm
        assert lm.weight is embed.weight

    def test_does_not_alias_under_multi_rank_tp(self):
        # Multi-rank: embed is replicated but the LM head is vocab-sharded, so
        # they cannot share a Parameter -- the weight is left for the loader.
        lm = _lm_head(tp_size=2)
        lm.weight = torch.zeros(2, 2)  # sentinel; must be left untouched
        embed = SimpleNamespace(weight=torch.ones(4, 4))
        result = lm.tie_weights(embed)
        assert result is lm
        assert lm.weight is not embed.weight

    def test_gguf_returns_embed_tokens_unchanged(self):
        lm = _lm_head(tp_size=1, quant_config=SimpleNamespace(get_name=lambda: "gguf"))
        embed = SimpleNamespace(weight=torch.ones(4, 4))
        assert lm.tie_weights(embed) is embed


class TestRegistration:
    def test_registered_as_oot(self):
        from vllm.model_executor.custom_op import maybe_get_oot_by_class
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        assert (
            maybe_get_oot_by_class(VocabParallelEmbedding) is RBLNVocabParallelEmbedding
        )
        assert maybe_get_oot_by_class(ParallelLMHead) is RBLNParallelLMHead
