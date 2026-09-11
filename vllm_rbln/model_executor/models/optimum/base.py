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


# FIXME(eunji): In original vLLM, this dataclasss is located in model_runner.
# And it makes available to decouple the vllm logic and hf model logic
@dataclass(frozen=True)
class ModelInputForRBLN:
    """Inputs of one optimum-rbln forward, laid out for the compiled graphs.

    Prefill carries one request: `input_tokens` / `input_positions` are
    `[1, seq_len]` and `block_tables` is `[num_blocks]`. Decode carries
    the running requests padded to `padded_batch_size` rows:
    `[padded_batch_size, 1]` tokens and positions,
    `[padded_batch_size, num_blocks]` block tables. Tokens are int64,
    positions int32, block tables and cache slot ids int16.
    """

    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    block_tables: torch.Tensor
    running_requests_ids: list[str]
    # Decode batch the tensors are padded to; 1 for prefill.
    padded_batch_size: int
    is_prompt: bool = False
    # Raw multimodal kwargs of this prefill's items, for models that encode
    # inside forward (Whisper). Others take their encoder output from mm_embeds.
    multi_modal_kwargs: BatchedTensorInputs | None = None
    # Prefill: encoder output of every multimodal item overlapping this
    # prefill's tokens, one 2D tensor per item in prompt order and cut to the
    # tokens being prefilled, plus the [1, seq_len] mask of the positions they
    # fill (see the runner's _gather_mm_embeddings). None on decode.
    mm_embeds: list[torch.Tensor] | None = None
    is_mm_embed: torch.Tensor | None = None
    # Block the scheduler set aside as scratch space for padding rows. None
    # when the scheduler did not set one aside; the runner then picks a block
    # no running request uses.
    dummy_block: int | None = None
    # Row of each running request in the padded decode batch, in running
    # order. None when the rows are simply [0, num_reqs): only models that pin
    # each request to a fixed row (see decode_batch_rows) get a tensor.
    batch_rows: torch.Tensor | None = None
    # Scheduler-allocated rows of the per-sequence on-device caches
    # (sliding-window KV, linear-attention state): [1] for prefill,
    # [padded_batch_size, 1] for decode with padding rows pointing at a slot
    # no running request owns. Models without such caches ignore it.
    cache_slot_ids: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    position_embed: torch.Tensor | None = None
    # Qwen3-VL / Qwen3-VL-Moe prefill extras: visual token position mask and
    # deepstack features. Left None for models that don't use them.
    visual_pos_mask: torch.Tensor | None = None
    deepstack_embeds: torch.Tensor | None = None
    # MRoPE models: the (t, h, w) positions of the tokens above as
    # [3, padded_batch_size, seq_len], 0 in the padding rows. None otherwise.
    mrope_positions: torch.Tensor | None = None


version_error = RuntimeError(
    "Incompatible vLLM version detected. "
    "This vLLM version is not compatible with optimum-rbln. "
    "Please verify that you are using a supported version."
)
