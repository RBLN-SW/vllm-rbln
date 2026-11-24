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

"""Qwen3 text embedding and reranking models."""

import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_embed
from vllm.model_executor.models import VllmModelForPooling
from vllm.tasks import PoolingTask
from vllm.tokenizers import get_tokenizer
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from vllm_rbln.logger import init_logger

from .base import ModelInputForRBLN, build_embedding_pooler, version_error
from .model_base import RBLNOptimumModelBase, RBLNOptimumPrefixCacheMixin

logger = init_logger(__name__)


def resolve_label_token_ids(model_config: ModelConfig) -> tuple[int, int]:
    """Return the ``(false_id, true_id)`` vocabulary ids to score against.

    The two label tokens come from ``classifier_from_token``, false label first,
    matching vLLM's ``from_2_way_softmax`` convention.
    """
    hf_config = model_config.hf_config
    text_config = hf_config.get_text_config()
    tokens = getattr(
        hf_config,
        "classifier_from_token",
        getattr(text_config, "classifier_from_token", None),
    )
    if not tokens or len(tokens) != 2:
        raise ValueError(
            "Qwen3-Reranker needs exactly two `classifier_from_token` entries, "
            f'false label first (e.g. ["no", "yes"]); got {tokens!r}.'
        )

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )
    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    if false_id is None or true_id is None:
        raise ValueError(
            f"`classifier_from_token` {tokens!r} are not single tokens of this "
            "model's vocabulary."
        )
    return false_id, true_id


class RBLNQwen3RerankerPooler(Pooler):
    """Turn the logit difference from the model into a relevance score.

    ``forward`` has already reduced the vocabulary down to a single logit per
    request, so there is nothing left to pool -- only the activation, gated by
    ``PoolingParams.use_activation`` the same way vLLM's classifier pooler does
    it. ``PoolerClassify`` reads num_labels off the logits, which are
    ``[batch, 1]``, and so applies sigmoid: the 2-way softmax.
    """

    def __init__(self) -> None:
        super().__init__()
        self.activation = PoolerClassify()

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify"}

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        flags = [p.use_activation for p in pooling_metadata.pooling_params]
        if len(set(flags)) == 1:
            return self.activation(hidden_states) if flags[0] else hidden_states
        return [
            self.activation(logits) if f else logits
            for logits, f in zip(hidden_states, flags)
        ]


class RBLNOptimumQwen3ForSequenceClassification(
    RBLNOptimumModelBase, VllmModelForPooling
):
    """Score Qwen3-Reranker through vLLM's ``score()`` API.

    The reranker is a causal LM that answers "yes" or "no". Its relevance
    score is the two-way softmax over the answer logits::

        p_yes / (p_yes + p_no) = sigmoid(logit_yes - logit_no)

    Reusing the generation graph with its lm_head lets the same compiled
    artifact serve both generation and scoring. The graph keeps only the
    final position (``logits_to_keep == 1``), where the answer would be sampled.
    """

    is_pooling_model = True
    pooler: Pooler

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config=vllm_config)
        self.false_id, self.true_id = resolve_label_token_ids(vllm_config.model_config)
        self.pooler = RBLNQwen3RerankerPooler()
        logger.info(
            "Scoring Qwen3-Reranker on logit ids %d (true) - %d (false)",
            self.true_id,
            self.false_id,
        )

    def preprocess(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-pad each request to the left, which the generation graph requires.

        Requests arrive right padded to a common width, their real lengths
        recoverable from ``positions``. The generation graph keeps a KV cache, so
        optimum-rbln rejects a right-padded mask.

        The batch is left as-is: padding it out to the compiled batch size would
        hand the model all-zero masks for the unused slots, which it refuses.
        """
        num_requests, width = input_ids.shape
        max_seq_len = self.rbln_model_config.max_seq_len
        if width > max_seq_len:
            raise ValueError(
                f"Input length ({width}) exceeds the compiled maximum ({max_seq_len})."
            )

        # positions run 0..n-1 per request, so the largest one sits at index n-1.
        lengths = torch.max(positions, dim=1).indices + 1

        padded_ids = input_ids.new_zeros((num_requests, width))
        mask = torch.zeros(num_requests, width, dtype=torch.long)
        for idx, length in enumerate(lengths.tolist()):
            padded_ids[idx, width - length :] = input_ids[idx, :length]
            mask[idx, width - length :] = 1
        return padded_ids, mask

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        num_requests = model_input.input_tokens.shape[0]
        input_ids, attention_mask = self.preprocess(
            model_input.input_tokens, model_input.input_positions
        )

        outputs = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        # [batch, 1, vocab_size] -- the final position only, which is where the
        # "yes"/"no" answer would have been sampled.
        logits = outputs.logits.reshape(num_requests, -1)

        # The fp32 head dtype keeps sigmoid scores near 1 distinguishable.
        score_logits = logits[:, self.true_id] - logits[:, self.false_id]
        return score_logits.unsqueeze(-1).to(self.model_config.head_dtype)


class RBLNOptimumQwen3Model(
    RBLNOptimumModelBase, RBLNOptimumPrefixCacheMixin, VllmModelForPooling
):
    """Qwen3-Embedding on the decoder graph that returns hidden states."""

    is_pooling_model = True
    pooler: Pooler

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config=vllm_config)
        pooler_config = self.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler(
            {
                "token_embed": pooler_for_token_embed(pooler_config),
                "embed": build_embedding_pooler(self.model_config),
            },
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        if self.model.prefill_decoder is None:
            raise version_error
        return self.model.prefill_decoder(
            input_ids=model_input.input_tokens,
            cache_position=model_input.input_positions,
            block_tables=model_input.block_tables,
        ).logits[0]
