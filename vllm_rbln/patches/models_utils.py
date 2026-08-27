# Copyright 2025 Rebellions Inc. All rights reserved.
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

from collections.abc import Iterable

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    logger,
    maybe_prefix,
)

from vllm_rbln.patches import register_patch


# NOTE(RBLN): Introduced in https://github.com/RBLN-SW/vllm-rbln/pull/81
@register_patch(
    target="vllm.model_executor.models.utils.AutoWeightsLoader._load_module",
    reason=(
        "In RBLN tensor parallelism, tied word embeddings cannot alias weights "
        "because token embeddings are replicated while ParallelLMHead is "
        "vocab-sharded. Replay embed_tokens weights through the normal lm_head "
        "loading path so ParallelLMHead.weight_loader can load each rank-local "
        "vocab shard. (PR#81)"
    ),
)
def patched_load_module(
    self: AutoWeightsLoader,
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
    # NOTE(RBLN): Upstream skips lm_head weights for tied embeddings because
    # lm_head.weight aliases embed_tokens.weight. In RBLN TP, the alias is
    # intertionally broken: embed_tokens is replicated and lm_head is sharded.
    # Capture embed_tokens weights so they can be replayed through
    # lm_head.weight_loader below.
    embed_tokens: list[tuple[str, torch.Tensor]] = []

    def gen_weights(cur_weights: Iterable[tuple[str, torch.Tensor]]):
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

            named_parameters = module.named_parameters(recurse=True)
            desc_param_keys = {
                maybe_prefix(base_prefix, k) for k, _ in named_parameters
            }
            msg = (
                f"There is no module or parameter named {prefix!r} "
                f"in {self.module._get_name()}. "
                f"The available parameters belonging to {base_prefix} "
                f"({module._get_name()}) are: {desc_param_keys}"
            )
            raise ValueError(msg)

    # NOTE(RBLN): Temporarily unskip lm_head and load the replayed embedding weights
    # into it. ParallelLMHead.weight_loader will select the rank-local vocab shard.
    assert len(embed_tokens) < 2
    if len(embed_tokens) == 1:
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
