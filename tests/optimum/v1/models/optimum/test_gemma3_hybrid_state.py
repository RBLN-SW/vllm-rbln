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
"""Unit tests for Gemma3's hybrid-attention per-request state wiring.

Gemma3's slot in the sliding-window cache comes from the scheduler as
``ModelInputForRBLN.cache_slot_ids``, like the other local-cache models. What
stays model-side is the state the prefill graph produces: the padded cache
length and the attention mask, recorded at prefill and replayed (with the new
token position enabled) on every decode step. No hardware needed.
"""

import types

import torch

from vllm_rbln.model_executor.models.optimum.gemma3 import (
    HybridAttentionStateManager,
)
from vllm_rbln.model_executor.models.optimum.gemma3 import (
    RBLNOptimumGemma3ForConditionalGeneration as Gemma3,
)


def _bare_gemma3(max_batch_size: int) -> Gemma3:
    # Uninitialised instance: we only drive forward(), which needs a few plain
    # attributes (no nn.Module / hardware init).
    obj = Gemma3.__new__(Gemma3)
    obj.decoder_batch_size = max_batch_size
    obj.use_multiple_decoder = False
    obj.available_blocks = torch.arange(50, 60, dtype=torch.int16)
    obj.attention_manager = HybridAttentionStateManager()
    obj._image_token_id = lambda: 999
    return obj


def test_forward_prefill_passes_slot_and_records_graph_state():
    # Prefill forwards the scheduler-assigned slot to the graph unchanged and
    # records the graph's outputs (padded cache length, attention mask) for
    # this request's decode steps.
    obj = _bare_gemma3(max_batch_size=2)

    passed = {}
    graph_mask = torch.tensor([[1, 1, 1, 0, 0]])

    def fake_prefill(**kw):
        passed.update(kw)
        return types.SimpleNamespace(
            logits=torch.zeros(1, 1),
            attention_mask=graph_mask,
            padded_cache_lengths=2,
        )

    obj.model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(prefill_decoder=fake_prefill)
    )

    model_input = types.SimpleNamespace(
        is_prompt=True,
        running_requests_ids=["A"],
        input_tokens=torch.tensor([[11, 12, 13]]),
        input_positions=torch.tensor([[0, 1, 2]]),
        block_tables=torch.tensor([[10]], dtype=torch.int16),
        inputs_embeds=torch.zeros(1, 3, 4),
        cache_slot_ids=torch.tensor([1], dtype=torch.int16),
    )
    obj.forward(model_input)

    assert torch.equal(
        passed["local_block_tables"], torch.tensor([1], dtype=torch.int16)
    )
    # No PAD tokens in the prompt -> all-ones prompt mask.
    assert torch.equal(passed["attention_mask"], torch.ones(3, dtype=torch.int64))
    entry = obj.attention_manager.table["A"]
    assert entry.pad_len == 2
    assert torch.equal(entry.attention_mask, graph_mask)


def test_forward_decode_wires_state_by_running_order():
    # Decode looks the recorded state up in running order, offsets each
    # cache_position by the request's pad_len, enables the new token position
    # in its mask row, and writes the updated mask back for the next step.
    obj = _bare_gemma3(max_batch_size=2)
    obj.attention_manager.add(
        "A", pad_len=2, attention_mask=torch.tensor([[1, 1, 0, 0, 0, 0]])
    )
    obj.attention_manager.add(
        "B", pad_len=0, attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]])
    )

    recorded = {}

    def fake_decoder(**kw):
        recorded.update(kw)
        return types.SimpleNamespace(logits=torch.zeros(2, 1))

    obj.model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(decoders={2: fake_decoder})
    )

    model_input = types.SimpleNamespace(
        is_prompt=False,
        running_requests_ids=["B", "A"],  # reversed vs slot order
        input_tokens=torch.tensor([[201], [200]]),
        input_positions=torch.tensor([[3], [2]]),
        block_tables=torch.tensor([[10], [11]], dtype=torch.int16),
        cache_slot_ids=torch.tensor([1, 0], dtype=torch.int16),
    )
    obj.forward(model_input)

    # Slot ids in running order, padded to the decoder batch.
    assert torch.equal(
        recorded["local_block_tables"], torch.tensor([[1], [0]], dtype=torch.int16)
    )
    # position_ids are the raw positions; cache_position adds each pad_len
    # (B: 3+0, A: 2+2).
    assert torch.equal(
        recorded["position_ids"], torch.tensor([[3], [2]], dtype=torch.int32)
    )
    assert torch.equal(
        recorded["cache_position"], torch.tensor([[3], [4]], dtype=torch.int32)
    )
    # Each mask row gets its new token position enabled...
    assert torch.equal(
        recorded["attention_mask"],
        torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0]]),
    )
    # ...and is written back for the next step.
    table = obj.attention_manager.table
    assert torch.equal(table["B"].attention_mask, torch.tensor([[1, 1, 1, 1, 0, 0]]))
    assert torch.equal(table["A"].attention_mask, torch.tensor([[1, 1, 0, 0, 1, 0]]))
