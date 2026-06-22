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

from collections.abc import Iterable

import torch
import vllm.model_executor.models.utils as _vllm_model_utils
import vllm.v1.worker.utils as _vllm_worker_utils
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer, logger


def __auto_weights_loader__load_module(
    self,
    base_prefix: str,
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[str]:
    if isinstance(module, PPMissingLayer):
        return

    # Avoid infinite recursion since this function is typically
    # called inside load_weights of the module itself
    if module != self.module:
        module_load_weights = getattr(module, "load_weights", None)
        if callable(module_load_weights):
            loaded_params = module_load_weights(weights)
            if loaded_params is None:
                logger.warning(
                    "Unable to collect loaded parameters for module %s", module
                )
            else:
                yield from map(
                    lambda x: self._get_qualname(base_prefix, x),
                    loaded_params,
                )

    child_modules = dict(module.named_children())
    child_params = dict(module.named_parameters(recurse=False))

    # Add missing tensors the weight loader needs to be able to load
    # that aren't registered as params, e.g., batchnorm statistics.
    self._add_loadable_non_param_tensors(module, child_params)

    EMBED_TOKENS = "embed_tokens"
    LM_HEAD = "lm_head"
    tie_word_embeddings = any(p.startswith(LM_HEAD) for p in self.skip_prefixes)
    tp_enabled = get_tensor_model_parallel_world_size() > 1

    embed_tokens: list[tuple[str, torch.Tensor]] = []

    def gen_weights(cur_weights):
        for name, weight in cur_weights:
            if name.startswith(EMBED_TOKENS):
                new_name = name.replace(EMBED_TOKENS, LM_HEAD)
                embed_tokens.append((new_name, weight))
            yield (name, weight)

    for child_prefix, child_weights in self._groupby_prefix(weights):
        prefix = self._get_qualname(base_prefix, child_prefix)

        if child_prefix in child_modules:
            if self._can_skip(prefix + "."):
                logger.debug("Skipping module %s", prefix)

                continue

            if tie_word_embeddings and tp_enabled:
                child_weights = gen_weights(child_weights)
            yield from self._load_module(
                prefix, child_modules[child_prefix], child_weights
            )
        elif child_prefix in child_params:
            if self._can_skip(prefix):
                logger.debug("Skipping param %s", prefix)

                continue

            yield from self._load_param(
                prefix, child_params[child_prefix], child_weights
            )
        else:
            can_skip_module = self._can_skip(prefix + ".")
            can_skip_param = self._can_skip(prefix)
            if can_skip_module or can_skip_param:
                logger.debug("Skipping missing %s", prefix)

                continue

            can_ignore_module = self._can_ignore_unexpected(prefix + ".")
            can_ignore_param = self._can_ignore_unexpected(prefix)
            if can_ignore_module or can_ignore_param:
                logger.debug("Ignoring missing %s", prefix)

                continue

            msg = (
                f"There is no module or parameter named '{prefix}' "
                f"in {type(self.module).__name__}"
            )
            raise ValueError(msg)

    assert len(embed_tokens) < 2
    if len(embed_tokens) != 0:
        org_skip_prefixes = self.skip_prefixes
        self.skip_prefixes = [p for p in org_skip_prefixes if not p.startswith(LM_HEAD)]

        for child_prefix, child_weights in self._groupby_prefix(embed_tokens):
            assert child_prefix == LM_HEAD
            prefix = self._get_qualname(base_prefix, child_prefix)
            if child_prefix in child_modules:
                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )

        self.skip_prefixes = org_skip_prefixes


AutoWeightsLoader._load_module = __auto_weights_loader__load_module


# layer_index splitting for models with >1 attention module per decoder layer
_original_extract_layer_index = _vllm_model_utils.extract_layer_index


def rbln_extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    if num_attn_module <= 1 or "attn" not in layer_name:
        return _original_extract_layer_index(layer_name, num_attn_module)

    int_vals: list[int] = []
    for subname in layer_name.split("."):
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert int_vals, f"layer name {layer_name} has no integer layer index"
    base = int_vals[0]
    sub = int_vals[1] if len(int_vals) >= 2 else int("indexer" in layer_name)
    return base * num_attn_module + sub


def rbln_num_attn_module(model_config) -> int:
    """Number of KV-cache-bearing attention modules per decoder layer."""
    hf_config = model_config.hf_config
    if getattr(hf_config, "model_type", None) == "longcat_flash":
        return 2
    text_config = getattr(model_config, "hf_text_config", hf_config)
    if hasattr(text_config, "index_topk") or hasattr(hf_config, "index_topk"):
        return 2
    return 1


_vllm_worker_utils.extract_layer_index = rbln_extract_layer_index
_vllm_model_utils.extract_layer_index = rbln_extract_layer_index
