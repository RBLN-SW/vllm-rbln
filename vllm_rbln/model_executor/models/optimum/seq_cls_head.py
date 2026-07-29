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

"""Host-side classifier head for the original Qwen3-Reranker.

Qwen3-Reranker is a causal LM whose relevance score is the 2-way softmax over
its "yes"/"no" logits. Written out, that score collapses to a single vector::

    p_yes / (p_yes + p_no) = sigmoid(logit_yes - logit_no)
                           = sigmoid((w_yes - w_no) @ h_last)

so the compiled model only has to produce hidden states, and the classifier is
one dot product on the host.

vLLM reaches the same place by rewriting ``lm_head`` into a ``score`` layer while
loading weights (``from_2_way_softmax`` in
``vllm/model_executor/models/adapters.py``). That hook lives in vLLM's weight
loader, which the optimum-rbln path never goes through -- the model arrives as a
precompiled artifact. We therefore read the two rows straight out of the
checkpoint instead.
"""

import glob
import json
import os
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from vllm.tokenizers import get_tokenizer

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def _checkpoint_source(model_config: "ModelConfig") -> str:
    """Return where the original HF weights live.

    ``model_config.model`` is rewritten to the RBLN compile cache once the model
    has been loaded, so it cannot be relied on here. ``_name_or_path`` keeps
    whatever the HF config was originally loaded from.
    """
    source = getattr(model_config.hf_config, "_name_or_path", None)
    return source or model_config.model


def _safetensors_paths(source: str) -> list[str]:
    """Return every safetensors shard for ``source``, local dir or HF repo id."""
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*.safetensors")))
        if not paths:
            raise ValueError(f"No .safetensors file found under {source}.")
        return paths

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        index_path = hf_hub_download(source, SAFE_WEIGHTS_INDEX_NAME)
    except EntryNotFoundError:
        # Unsharded checkpoint: there is no index to consult.
        return [hf_hub_download(source, SAFE_WEIGHTS_NAME)]

    with open(index_path) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    return [hf_hub_download(source, shard) for shard in shards]


def _read_rows(source: str, name: str, row_ids: list[int]) -> torch.Tensor:
    """Read selected rows of a checkpoint tensor, in ``row_ids`` order."""
    for path in _safetensors_paths(source):
        with safe_open(path, framework="pt") as f:
            # safe_open is not a mapping, so membership goes through keys().
            if name not in set(f.keys()):
                continue
            # get_slice avoids materializing the whole vocab x hidden matrix,
            # which for an 8B model is several GB.
            rows = f.get_slice(name)
            return torch.cat([rows[i : i + 1, :] for i in row_ids], dim=0)
    raise ValueError(f"Tensor {name!r} not found in the checkpoint at {source}.")


def load_2_way_softmax_score_weight(model_config: "ModelConfig") -> torch.Tensor:
    """Return ``w_yes - w_no`` as a ``[1, hidden_size]`` float32 tensor.

    The two rows come from ``lm_head``, or from the input embeddings when the
    checkpoint ties them (Qwen3-Reranker does), in which case no separate
    ``lm_head.weight`` exists.
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

    weight_name = (
        "model.embed_tokens.weight"
        if getattr(text_config, "tie_word_embeddings", False)
        else "lm_head.weight"
    )
    source = _checkpoint_source(model_config)
    # Order matters: true row first, so the difference is w_yes - w_no.
    rows = _read_rows(source, weight_name, [true_id, false_id]).to(torch.float32)

    logger.info(
        "Built Qwen3-Reranker score head from %s rows %d(%s) - %d(%s) of %s",
        weight_name,
        true_id,
        tokens[1],
        false_id,
        tokens[0],
        source,
    )
    return (rows[0] - rows[1]).unsqueeze(0)
