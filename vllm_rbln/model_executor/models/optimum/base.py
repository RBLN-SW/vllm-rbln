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
from dataclasses import dataclass

import torch
from vllm.multimodal.inputs import BatchedTensorInputs


@dataclass(frozen=True)
class PartialPrefixInfo:
    """Inputs to rebuild the uncached tail of a partial prefix-cache hit
    (boundary may end inside an image).

    - ``full_input_tokens``: untrimmed prompt (MRoPE positions computed over it).
    - ``num_cached_tokens``: cache boundary in tokens (= tail start).
    - ``mrope_mm_kwargs``: every item's grid (incl. cached) for get_rope_index.
    - ``mm_embed_tail_starts``: per kept item, first uncached feature index.
    """

    full_input_tokens: torch.Tensor
    num_cached_tokens: int
    mrope_mm_kwargs: BatchedTensorInputs | None
    mm_embed_tail_starts: dict[str, list[int]] | None


# FIXME(eunji): In original vLLM, this dataclasss is located in model_runner.
# And it makes available to decouple the vllm logic and hf model logic
@dataclass(frozen=True)
class ModelInputForRBLN:
    """
    Used by the RBLNModelRunner.
    """

    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    block_tables: torch.Tensor
    running_requests_ids: list[str]
    finished_requests_ids: list[str]
    is_prompt: bool = False
    multi_modal_kwargs: BatchedTensorInputs | None = None
    dummy_block: int | None = None  # for prefix caching
    # Local block table ids of the running requests, in running order
    # ([1] for prefill, [num_reqs] for decode). Scheduler-allocated rows of
    # the per-sequence on-device caches (sliding-window KV, linear-attention
    # state); models without such caches ignore it.
    local_block_tables: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    position_embed: torch.Tensor | None = None
    # Qwen3-VL / Qwen3-VL-Moe prefill extras: visual token position mask and
    # deepstack features. Left None for models that don't use them.
    visual_pos_mask: torch.Tensor | None = None
    deepstack_embeds: torch.Tensor | None = None
    # Set only on a partial prefix-cache hit (see PartialPrefixInfo); None on the
    # no-hit path and for non-MRoPE models.
    partial_prefix: "PartialPrefixInfo | None" = None


version_error = RuntimeError(
    "Incompatible vLLM version detected. "
    "This vLLM version is not compatible with optimum-rbln. "
    "Please verify that you are using a supported version."
)
