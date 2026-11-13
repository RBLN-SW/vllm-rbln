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
"""Unit tests for Qwen3.5's linear-attention state slotting.

Qwen3.5's GatedDeltaNet ``linear_attention`` state is a fixed ``[max_num_seqs]``
on-device cache. The scheduler gives each request a stable cache slot for its
lifetime and ships it as ``ModelInputForRBLN.cache_slot_ids``: prefill writes
that slot (``batch_idx``), and decode names each row's slot (``state_idx``).
A decode batch the runner sorts by length keeps that order; otherwise each
request sits at ``row == slot``. The runner gathers the logits back to running
order. No hardware needed.
"""

import types

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNOptimumQwen3_5ForConditionalGeneration as Qwen3_5,
)


class TestDecodeLayoutWiring:
    """Exercise the REAL wrapper code (only the model mocked): decode names the
    cache slot of each row (``state_idx``), so every request meets its OWN
    recurrent state whatever row it sits on. Without a batch sort the request
    is pinned to the row of its slot; under ``requires_batch_sort`` it keeps the
    runner's length order. The runner gathers the logits back to running order.
    """

    @staticmethod
    def _bare_qwen3_5(
        max_batch_size: int, requires_batch_sort: bool = False
    ) -> Qwen3_5:
        # Uninitialised instance: we only drive the decode-layout methods, which
        # need a few plain attributes (no nn.Module / hardware init).
        obj = Qwen3_5.__new__(Qwen3_5)
        obj.decoder_batch_size = max_batch_size
        obj.use_multiple_decoder = False
        # ``dtype`` is a read-only property (-> self.model.rbln_config.dtype); the
        # forward tests replace the model with one that supplies it.
        obj.model = types.SimpleNamespace(
            rbln_config=types.SimpleNamespace(requires_batch_sort=requires_batch_sort)
        )
        return obj

    def test_decode_layout_pins_the_cache_slots_in_the_full_batch(self):
        # The runner asks the model where each running request sits in the
        # decode batch; Qwen3.5 pins it to its scheduler-assigned cache slot
        # and always runs the full decoder batch.
        obj = self._bare_qwen3_5(max_batch_size=4)

        padded_batch_size, rows = obj.decode_layout(
            torch.tensor([1, 0], dtype=torch.int16),
            torch.tensor([[10], [11]], dtype=torch.int16),
        )

        assert padded_batch_size == 4
        assert rows.dtype == torch.long
        assert rows.tolist() == [1, 0]

    def test_decode_layout_keeps_a_batch_sorted_by_length_in_order(self):
        # A model compiled with requires_batch_sort (RBLN-CR13+, flash_attn)
        # takes the runner's length order, padded to the full decoder batch;
        # forward then names each row's cache slot.
        obj = self._bare_qwen3_5(max_batch_size=4, requires_batch_sort=True)

        padded_batch_size, rows = obj.decode_layout(
            torch.tensor([1, 0], dtype=torch.int16),
            torch.tensor([[10], [11]], dtype=torch.int16),
        )

        assert padded_batch_size == 4
        assert rows == slice(0, 2)

    @pytest.mark.parametrize(
        "cache_slot_ids",
        [
            # Sorted [A, B, pad, pad]: A owns slot 2 and B slot 0; the padding
            # rows share free slot 1.
            [2, 0, 1, 1],
            # Pinned: A (slot 2) on row 2; the padding rows share free slot 0.
            [0, 0, 2, 0],
        ],
        ids=["sorted", "pinned"],
    )
    def test_forward_decode_names_each_rows_cache_slot(self, cache_slot_ids):
        # Each row's state is read from the slot the runner gave that row.
        obj = self._bare_qwen3_5(max_batch_size=4)

        recorded = {}

        def fake_decoder(**kw):
            recorded.update(kw)
            return types.SimpleNamespace(logits=kw["inputs_embeds"][:, 0, :])

        obj.model = types.SimpleNamespace(
            embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
            decoders={4: fake_decoder},
            rbln_config=types.SimpleNamespace(dtype=torch.float32),
        )

        model_input = types.SimpleNamespace(
            is_prompt=False,
            padded_batch_size=4,
            input_tokens=torch.zeros(4, 1, dtype=torch.int64),
            input_positions=torch.zeros(4, 1, dtype=torch.int32),
            block_tables=torch.zeros(4, 1, dtype=torch.int16),
            position_embed=torch.zeros(2, 4, 1, 1, 1),
            cache_slot_ids=torch.tensor(cache_slot_ids, dtype=torch.int16).unsqueeze(1),
        )
        obj.forward(model_input)

        assert recorded["state_idx"].tolist() == cache_slot_ids

    def test_forward_decode_pairs_each_request_with_its_own_row(self):
        # A owns row 0 and B owns row 1, but this step runs them as [B, A].
        # The runner has already placed each request on its own row, so
        # forward must pass the batch through untouched and return the logits
        # as the graph laid them out, by row.
        obj = self._bare_qwen3_5(max_batch_size=2)

        recorded = {}

        def fake_decoder(**kw):
            recorded.update(kw)
            return types.SimpleNamespace(logits=kw["inputs_embeds"][:, 0, :])

        obj.model = types.SimpleNamespace(
            embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
            decoders={2: fake_decoder},
            rbln_config=types.SimpleNamespace(dtype=torch.float32),  # -> self.dtype
        )

        model_input = types.SimpleNamespace(
            is_prompt=False,
            running_requests_ids=["B", "A"],  # reversed vs prefill order
            padded_batch_size=2,
            batch_rows=torch.tensor([1, 0]),  # rows of [B, A]
            input_tokens=torch.tensor([[200], [201]]),  # row 0 = A, row 1 = B
            input_positions=torch.tensor([[7], [5]], dtype=torch.int32),
            block_tables=torch.tensor([[11], [10]], dtype=torch.int16),
            position_embed=torch.zeros(2, 2, 1, 1, 1),
            cache_slot_ids=torch.tensor([[0], [1]], dtype=torch.int16),
        )

        logits = obj.forward(model_input)

        # The graph sees the batch BY ROW: row0 == A's row, row1 == B's row.
        assert recorded["inputs_embeds"][0, 0, 0] == 200  # A on row 0
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on row 1
        # ...and logits stay BY ROW [A, B] -> [200, 201]; the runner gathers
        # them into running order.
        assert logits[0, 0] == 200
        assert logits[1, 0] == 201
        assert recorded["state_idx"].tolist() == [0, 1]  # A's slot 0, B's slot 1

    def test_forward_decode_single_request_pads_other_rows(self):
        # Only B runs (A finished earlier and its row was freed by the
        # scheduler). The batch stays max_num_seqs wide: B on its row (1); the
        # empty row 0 is dummy padding whose output the runner drops. B is
        # unaffected (rows are independent).
        obj = self._bare_qwen3_5(max_batch_size=2)

        recorded = {}

        def fake_decoder(**kw):
            recorded.update(kw)
            return types.SimpleNamespace(logits=kw["inputs_embeds"][:, 0, :])

        obj.model = types.SimpleNamespace(
            embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
            decoders={2: fake_decoder},
            rbln_config=types.SimpleNamespace(dtype=torch.float32),
        )

        model_input = types.SimpleNamespace(
            is_prompt=False,
            running_requests_ids=["B"],
            padded_batch_size=2,
            batch_rows=torch.tensor([1]),
            input_tokens=torch.tensor([[0], [201]]),  # row 0 is padding
            input_positions=torch.tensor([[0], [9]], dtype=torch.int32),
            block_tables=torch.tensor([[50], [10]], dtype=torch.int16),
            position_embed=torch.zeros(2, 2, 1, 1, 1),
            cache_slot_ids=torch.tensor([[0], [1]], dtype=torch.int16),
        )
        logits = obj.forward(model_input)

        assert recorded["inputs_embeds"].shape[0] == 2  # still max_num_seqs wide
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on its row
        assert recorded["inputs_embeds"][0, 0, 0] == 0  # empty row -> dummy input
        assert logits.shape[0] == 2  # every row comes back, padding included
        assert logits[1, 0] == 201
        assert recorded["state_idx"].tolist() == [0, 1]  # padding row 0 on free slot 0

    def test_forward_prefill_passes_batch_idx(self):
        # Prefill passes the scheduler-assigned row as ``batch_idx`` to the
        # prefill graph (omitting it raises KeyError inside the runtime).
        obj = self._bare_qwen3_5(max_batch_size=2)

        passed = {}

        def fake_prefill(**kw):
            passed.update(kw)
            return types.SimpleNamespace(logits=torch.zeros(1, 1))

        obj.model = types.SimpleNamespace(
            prefill_decoder=fake_prefill,
            rbln_config=types.SimpleNamespace(dtype=torch.float32),
        )

        model_input = types.SimpleNamespace(
            is_prompt=True,
            running_requests_ids=["B"],
            input_tokens=torch.tensor([[1]]),
            input_positions=torch.tensor([[0]], dtype=torch.int32),
            block_tables=torch.tensor([10], dtype=torch.int16),
            inputs_embeds=torch.zeros(1, 1, 1),
            position_embed=torch.zeros(2, 1, 1, 1, 1),
            cache_slot_ids=torch.tensor([1], dtype=torch.int16),
        )
        obj.forward(model_input)

        assert passed["batch_idx"] == 1
