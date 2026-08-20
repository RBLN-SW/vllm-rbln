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

"""VllmConfig builders for native tests, via EngineArgs.create_engine_config()
so the result matches the config the engine hands production (after the
platform's check_and_update_config). Import lazily from a fixture."""

from __future__ import annotations

import functools
import os

from huggingface_hub import hf_hub_download
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@functools.cache
def local_model_path(model: str) -> str:
    """``model``'s hub cache directory, for configs that are built but never loaded.

    A hub id makes transformers revalidate config.json over the network on every
    single build -- five HEAD requests each, and the unit lane builds well over a
    hundred configs. Handing it the directory instead resolves everything locally,
    so the whole suite pays one lookup per model. What comes out differs only in
    the identity strings (``model``, ``tokenizer``, ``served_model_name``,
    ``_name_or_path``); every other field is identical.

    Not a stand-in for the repo id: on a cold cache the directory holds config.json
    and nothing else, so anything that loads weights or a tokenizer -- the
    ``vllm_runner`` fixtures, the whole --model-compile lane -- keeps the id.
    """
    return os.path.dirname(hf_hub_download(model, "config.json"))


def make_vllm_config(
    *,
    model: str = DEFAULT_MODEL,
    max_model_len: int = 8192,
    block_size: int = 1024,
    max_num_batched_tokens: int = 128,
    max_num_seqs: int = 4,
    enable_chunked_prefill: bool = True,
    **engine_args,
) -> VllmConfig:
    """Full-control native VllmConfig; extra EngineArgs fields pass through.

    Nothing here loads the model, so ``model`` is resolved to its cache directory
    (see local_model_path) rather than fetched over and over.
    """
    return EngineArgs(
        model=local_model_path(model),
        max_model_len=max_model_len,
        block_size=block_size,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=enable_chunked_prefill,
        **engine_args,
    ).create_engine_config()
