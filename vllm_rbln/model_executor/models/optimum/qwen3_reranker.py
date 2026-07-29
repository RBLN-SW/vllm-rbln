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

"""Qwen3-Reranker scored through vLLM's ``score()`` API.

The reranker is a causal LM that answers "yes" or "no", and its relevance score
is the 2-way softmax over those two logits::

    p_yes / (p_yes + p_no) = sigmoid(logit_yes - logit_no)

So this reuses the generation graph as-is -- lm_head included -- and picks the
two label logits out of the vocabulary. That keeps one compiled artifact serving
both the generative and the score() entry points, and needs nothing from the
original checkpoint at runtime.

The graph returns only the final position (``logits_to_keep == 1``), which is
exactly the position the answer would be sampled at, so no pooling step is
needed: ``forward`` hands the pooler an already-reduced ``[batch, 1]`` logit
difference.
"""

import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.models import VllmModelForPooling
from vllm.tasks import PoolingTask
from vllm.tokenizers import get_tokenizer
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from vllm_rbln.logger import init_logger

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumModelBase

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


class RBLNOptimumQwen3RerankerModel(RBLNOptimumModelBase, VllmModelForPooling):
    """Drive the Qwen3 generation graph as a scorer."""

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

    def preprocess(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Left pad each request up to the compiled batch and sequence length.

        The generation graph keeps a KV cache, so optimum-rbln requires the
        attention mask to be left padded; the encoder path's right padding would
        be rejected outright.
        """
        num_requests, seq_len = input_ids.shape
        max_seq_len = self.rbln_model_config.max_seq_len
        if seq_len > max_seq_len:
            raise ValueError(
                f"Input length ({seq_len}) exceeds the compiled maximum "
                f"({max_seq_len})."
            )

        padded_ids = input_ids.new_zeros((self.batch_size, max_seq_len))
        mask = torch.zeros(self.batch_size, max_seq_len, dtype=torch.long)
        padded_ids[:num_requests, max_seq_len - seq_len :] = input_ids
        mask[:num_requests, max_seq_len - seq_len :] = 1
        return padded_ids, mask

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        num_requests = model_input.input_tokens.shape[0]
        input_ids, attention_mask = self.preprocess(model_input.input_tokens)

        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        # [batch, 1, vocab_size] -- the final position only, which is where the
        # "yes"/"no" answer would have been sampled.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = logits[:num_requests].reshape(num_requests, -1)

        # sigmoid(logit_yes - logit_no) == p_yes / (p_yes + p_no), so reduce to
        # one logit here and let the pooler apply the activation.
        score_logits = logits[:, self.true_id] - logits[:, self.false_id]
        return score_logits.unsqueeze(-1).float()
