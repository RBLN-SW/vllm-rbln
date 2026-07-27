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
"""Unit tests for ``RBLNMambaStateIndexAllocator`` and the decode slot layout it
drives for Qwen3.5.

Qwen3.5's GatedDeltaNet ``linear_attention`` state is a fixed ``[max_num_seqs]``
on-device cache indexed by batch ROW. Each request is pinned to a stable slot for
its lifetime: prefill writes ``batch_idx = slot`` and decode places the request at
``row = slot``, then gathers logits back to running order. The allocator owns that
req_id -> slot mapping; these tests pin down its contract and the correctness of
the scatter/gather round trip that depends on it. No hardware or model needed.
"""

import types

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNMambaStateIndexAllocator,
)
from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNOptimumQwen3_5ForConditionalGeneration as Qwen3_5,
)


class TestAllocatorBasics:
    def test_allocates_lowest_free_slot_sequentially(self):
        alloc = RBLNMambaStateIndexAllocator(4)
        assert alloc.allocate("a") == 0
        assert alloc.allocate("b") == 1
        assert alloc.allocate("c") == 2

    def test_allocation_is_stable_and_idempotent(self):
        # A request's slot must not change across its lifetime (prefill wrote its
        # recurrent state there; decode must read the same row).
        alloc = RBLNMambaStateIndexAllocator(4)
        assert alloc.allocate("a") == 0
        alloc.allocate("b")  # 1
        # Re-allocating "a" returns the same slot, and does not consume a new one.
        assert alloc.allocate("a") == 0
        assert alloc.allocate("a") == 0
        assert alloc.allocate("c") == 2  # next fresh slot is still 2, not 3

    def test_indices_preserves_query_order(self):
        # ``indices`` feeds decode row placement + the logits gather, so its order
        # must follow the given running-request order, not the slot values.
        alloc = RBLNMambaStateIndexAllocator(4)
        alloc.allocate("a")  # 0
        alloc.allocate("b")  # 1
        alloc.allocate("c")  # 2
        assert alloc.indices(["c", "a", "b"]) == [2, 0, 1]
        assert alloc.indices(["b", "c", "a"]) == [1, 2, 0]


class TestFreeAndReuse:
    def test_reuses_lowest_free_slot(self):
        alloc = RBLNMambaStateIndexAllocator(4)
        for r in ("a", "b", "c"):
            alloc.allocate(r)  # 0, 1, 2
        alloc.free(["b"])  # frees 1
        # 1 is lower than the next-never-used 3, so it comes back first.
        assert alloc.allocate("d") == 1
        assert alloc.allocate("e") == 3

    def test_freed_slot_reused_by_new_request(self):
        # The correctness-critical case: a finished request's slot is handed to a
        # brand-new request. This is safe because the new request PREFILLS fresh
        # (optimum's conv/recurrent 0-mask resets the slot on window 0), so the
        # stale state left by the finished request is discarded.
        alloc = RBLNMambaStateIndexAllocator(2)
        assert alloc.allocate("old") == 0
        alloc.free(["old"])
        assert alloc.allocate("new") == 0

    def test_full_lifecycle_allocate_free_reallocate(self):
        n = 3
        alloc = RBLNMambaStateIndexAllocator(n)
        first = [alloc.allocate(f"r{i}") for i in range(n)]
        assert sorted(first) == [0, 1, 2]
        alloc.free([f"r{i}" for i in range(n)])
        second = [alloc.allocate(f"s{i}") for i in range(n)]
        assert sorted(second) == [0, 1, 2]  # all slots available again


class TestEdgeCases:
    def test_free_unknown_request_is_noop(self):
        # ``finished_requests_ids`` may include ids this wrapper never allocated
        # (e.g. requests finished before their first slotted step); must not crash
        # or corrupt the pool.
        alloc = RBLNMambaStateIndexAllocator(2)
        alloc.allocate("a")  # 0
        alloc.free(["ghost"])  # no-op
        assert alloc.allocate("b") == 1  # pool intact: next is 1, not a duplicate 0

    def test_double_free_does_not_duplicate_slot(self):
        # Freeing the same id twice must return its slot to the pool exactly once,
        # otherwise two requests could be handed the same slot.
        alloc = RBLNMambaStateIndexAllocator(3)
        alloc.allocate("a")  # 0
        alloc.allocate("b")  # 1
        alloc.free(["a"])
        alloc.free(["a"])  # second free is a no-op
        assert alloc.allocate("c") == 0
        assert alloc.allocate("d") == 2  # NOT 0 again -> slot 0 wasn't double-freed

    def test_exhaustion_raises_clear_error(self):
        # vLLM caps concurrency at max_num_seqs, so this is unreachable in practice;
        # the allocator still fails loud instead of popping an empty free list.
        alloc = RBLNMambaStateIndexAllocator(2)
        alloc.allocate("a")
        alloc.allocate("b")
        with pytest.raises(RuntimeError, match="No free linear-attention batch index"):
            alloc.allocate("c")

    def test_reallocating_existing_when_full_is_ok(self):
        # Exhaustion only blocks NEW requests; an already-slotted request can still
        # be looked up when the pool is full.
        alloc = RBLNMambaStateIndexAllocator(2)
        alloc.allocate("a")
        alloc.allocate("b")
        assert alloc.allocate("a") == 0  # no error, returns existing slot

    def test_indices_of_unallocated_request_raises(self):
        # By decode time every running request has prefilled (and been allocated).
        # A missing slot means a wiring bug, so fail loud rather than silently
        # read the wrong recurrent-state row.
        alloc = RBLNMambaStateIndexAllocator(2)
        alloc.allocate("a")
        with pytest.raises(KeyError):
            alloc.indices(["a", "ghost"])


class TestDecodeLayoutWiring:
    """These exercise the REAL code path (not a simulated scatter): decode must
    physically place each running request at its ``batch_idx`` row so it meets its
    OWN recurrent-state row, regardless of the running order vLLM hands us
    (the "Step 4" correctness). The state cache in DRAM is fixed; the INPUT is
    rearranged to match it, then logits are gathered back to running order.
    """

    @staticmethod
    def _bare_qwen3_5(max_batch_size: int) -> Qwen3_5:
        # Uninitialised instance: we only drive the decode-layout methods, which
        # need a few plain attributes (no nn.Module / hardware init).
        obj = Qwen3_5.__new__(Qwen3_5)
        obj.decoder_batch_size = max_batch_size
        # ``dtype`` is a read-only property (-> self.model.rbln_config.dtype); the
        # forward test supplies it via a mocked model instead of setting it here.
        obj.available_blocks = torch.arange(50, 60, dtype=torch.int16)
        obj._state_index_allocator = RBLNMambaStateIndexAllocator(max_batch_size)
        return obj

    def test_pad_decoder_items_scatters_to_batch_idx_rows(self):
        # Running order [B, A]; B's row is 1, A's row is 0. The real
        # pad_decoder_items must land B (running idx 0) on row 1 and A on row 0.
        obj = self._bare_qwen3_5(max_batch_size=4)
        input_ids = torch.tensor([[201], [200]])  # running order: B=201, A=200
        positions = torch.tensor([[5], [7]])
        block_tables = torch.tensor([[10], [11]], dtype=torch.int16)
        batch_indices = torch.tensor([1, 0])  # indices([B, A])

        padded_ids, padded_pos, padded_bt = obj.pad_decoder_items(
            input_ids, positions, block_tables, input_block_ids=batch_indices
        )

        assert padded_ids[1, 0] == 201 and padded_ids[0, 0] == 200  # B->1, A->0
        assert padded_pos[1, 0] == 5 and padded_pos[0, 0] == 7
        assert padded_bt[1, 0] == 10 and padded_bt[0, 0] == 11

    def test_forward_decode_pairs_each_request_with_its_own_row(self):
        # End-to-end decode through the real forward (only model.embed_tokens /
        # decoder mocked). Prefill order was A(row0), B(row1); this decode step's
        # running order is the REVERSE [B, A]. Each must still land on its own row.
        obj = self._bare_qwen3_5(max_batch_size=2)
        obj._state_index_allocator.allocate("A")  # row 0
        obj._state_index_allocator.allocate("B")  # row 1

        recorded = {}

        def fake_decoder(**kw):
            recorded.update(kw)
            # Echo each row's token id as its logit, so the gather is observable.
            return types.SimpleNamespace(logits=kw["inputs_embeds"][:, 0, :])

        obj.model = types.SimpleNamespace(
            embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
            decoders={2: fake_decoder},
            rbln_config=types.SimpleNamespace(dtype=torch.float32),  # -> self.dtype
        )

        model_input = types.SimpleNamespace(
            is_prompt=False,
            finished_requests_ids=[],
            running_requests_ids=["B", "A"],  # reversed vs prefill order
            input_tokens=torch.tensor([[201], [200]]),  # B=201, A=200
            input_positions=torch.tensor([[5], [7]]),
            block_tables=torch.tensor([[10], [11]], dtype=torch.int16),
            position_embed=torch.zeros(2, 2, 1, 1, 1),
        )

        logits = obj.forward(model_input)

        # Physical batch is laid out BY ROW: row0 == A's state row, row1 == B's.
        assert recorded["inputs_embeds"][0, 0, 0] == 200  # A on row 0 (its batch_idx)
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on row 1 (its batch_idx)
        # ...and logits come back in RUNNING order [B, A] -> [201, 200].
        assert logits[0, 0] == 201
        assert logits[1, 0] == 200

    def test_finish_free_reuse_then_decode_mixed_slots(self):
        # Full lifecycle through the REAL forward: A & B prefill (rows 0, 1) ->
        # B finishes -> C prefills and REUSES B's freed row 1 (proving the
        # free-at-start avoids a false "pool exhausted" when at capacity) ->
        # decode [A, C] lands A on row 0 and C on row 1.
        obj = self._bare_qwen3_5(max_batch_size=2)
        prefill_batch_idx = []
        recorded = {}

        def fake_prefill(**kw):
            prefill_batch_idx.append(kw["batch_idx"])
            return types.SimpleNamespace(logits=torch.zeros(1, 1))

        def fake_decoder(**kw):
            recorded.update(kw)
            return types.SimpleNamespace(logits=kw["inputs_embeds"][:, 0, :])

        obj.model = types.SimpleNamespace(
            prefill_decoder=fake_prefill,
            embed_tokens=lambda ids: ids.to(torch.float32).unsqueeze(-1),
            decoders={2: fake_decoder},
            rbln_config=types.SimpleNamespace(dtype=torch.float32),
        )

        def prefill_input(req, finished=()):
            return types.SimpleNamespace(
                is_prompt=True,
                finished_requests_ids=list(finished),
                running_requests_ids=[req],
                input_tokens=torch.tensor([[1]]),
                input_positions=torch.tensor([[0]]),
                block_tables=torch.tensor([[10]], dtype=torch.int16),
                inputs_embeds=torch.zeros(1, 1, 1),
                position_embed=torch.zeros(2, 1, 1, 1, 1),
            )

        obj.forward(prefill_input("A"))  # -> row 0
        obj.forward(prefill_input("B"))  # -> row 1 (pool now full)
        # C prefills WHILE reporting B finished; free-at-start reclaims row 1.
        obj.forward(prefill_input("C", finished=["B"]))
        assert prefill_batch_idx == [0, 1, 1]  # C reused B's row, no exhaustion
        assert obj._state_index_allocator.indices(["A", "C"]) == [0, 1]

        decode_input = types.SimpleNamespace(
            is_prompt=False,
            finished_requests_ids=[],
            running_requests_ids=["A", "C"],
            input_tokens=torch.tensor([[200], [202]]),  # A=200, C=202
            input_positions=torch.tensor([[3], [1]]),
            block_tables=torch.tensor([[10], [11]], dtype=torch.int16),
            position_embed=torch.zeros(2, 2, 1, 1, 1),
        )
        logits = obj.forward(decode_input)
        assert recorded["inputs_embeds"][0, 0, 0] == 200  # A on row 0
        assert recorded["inputs_embeds"][1, 0, 0] == 202  # C on row 1 (reused)
        assert logits[0, 0] == 200 and logits[1, 0] == 202  # running order [A, C]

    def test_forward_decode_single_request_pads_freed_row(self):
        # A & B were decoding; A finishes this step -> only B runs, no new prompt.
        # The batch stays max_num_seqs wide: B on its row (1); A's freed row (0) is
        # dummy padding whose output is discarded. B is unaffected because
        # linear-attention rows are independent.
        obj = self._bare_qwen3_5(max_batch_size=2)
        obj._state_index_allocator.allocate("A")  # row 0
        obj._state_index_allocator.allocate("B")  # row 1

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
            finished_requests_ids=["A"],  # A finished this step
            running_requests_ids=["B"],  # only B remains
            input_tokens=torch.tensor([[201]]),
            input_positions=torch.tensor([[9]]),
            block_tables=torch.tensor([[10]], dtype=torch.int16),
            position_embed=torch.zeros(2, 1, 1, 1, 1),
        )
        logits = obj.forward(model_input)

        # Batch is still 2 wide: B on row 1, A's freed row 0 is dummy (token 0).
        assert recorded["inputs_embeds"].shape[0] == 2
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on its row
        assert recorded["inputs_embeds"][0, 0, 0] == 0  # freed row -> dummy input
        # Only B's logits are returned; the padding row's output is dropped.
        assert logits.shape[0] == 1
        assert logits[0, 0] == 201
        # A's row was reclaimed for future reuse.
        assert obj._state_index_allocator.indices(["B"]) == [1]
