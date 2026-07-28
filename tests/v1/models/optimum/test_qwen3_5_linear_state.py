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
"""Unit tests for Qwen3.5's linear-attention state slotting.

Qwen3.5's GatedDeltaNet ``linear_attention`` state is a fixed ``[max_num_seqs]``
on-device cache indexed by batch ROW. Each request is pinned to a stable
``batch_idx`` for its lifetime (``LinearAttentionStrategy`` via ``AttentionManager``):
prefill writes that row, decode places the request at ``row == batch_idx`` and
gathers logits back to running order. Finished requests are freed by the model
runner via ``attention_manager.pop`` (not from ``model_input``). No hardware needed.
"""

import types

import pytest
import torch

from vllm_rbln.model_executor.models.optimum.optimum_attention import (
    AttentionManager,
    LinearAttentionStrategy,
)
from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNOptimumQwen3_5ForConditionalGeneration as Qwen3_5,
)


def _prefill(strategy: LinearAttentionStrategy, req: str, bs: int = 4) -> int:
    """Mirror the wrapper's prefill: take the lowest free row and record it."""
    batch_idx = strategy.get(True, bs, [req], [])[0]
    strategy.add(req, batch_idx)
    return batch_idx


class TestStrategyAllocation:
    def test_prefill_takes_lowest_free_row(self):
        s = LinearAttentionStrategy()
        assert _prefill(s, "a") == 0
        assert _prefill(s, "b") == 1
        assert _prefill(s, "c") == 2

    def test_decode_get_returns_recorded_rows_in_running_order(self):
        # decode looks up each running request's pinned row, in the given order
        # (this order drives the row scatter + the logits gather).
        s = LinearAttentionStrategy()
        for r in ("a", "b", "c"):
            _prefill(s, r)  # 0, 1, 2
        assert s.get(False, 4, ["c", "a", "b"], []) == [2, 0, 1]
        assert s.get(False, 4, ["b", "c", "a"], []) == [1, 2, 0]


class TestFreeAndReuse:
    def test_pop_frees_row_and_is_reused(self):
        s = LinearAttentionStrategy()
        _prefill(s, "a")  # 0
        _prefill(s, "b")  # 1
        s.pop("a")  # runner frees a finished request
        assert _prefill(s, "c") == 0  # lowest free row reused

    def test_reuses_lowest_free_row(self):
        s = LinearAttentionStrategy()
        for r in ("a", "b", "c"):
            _prefill(s, r)  # 0, 1, 2
        s.pop("b")  # frees 1
        assert _prefill(s, "d") == 1  # 1 < the never-used 3
        assert _prefill(s, "e") == 3


class TestEdgeCases:
    def test_pop_unknown_request_is_noop(self):
        # The runner may pop ids this strategy never recorded; must not corrupt.
        s = LinearAttentionStrategy()
        _prefill(s, "a")  # 0
        s.pop("ghost")  # no-op
        assert _prefill(s, "b") == 1  # pool intact -> 1, not a duplicate 0

    def test_exhaustion_on_prefill_raises(self):
        # vLLM caps concurrency at max_num_seqs, so this is unreachable in
        # practice; the framework asserts rather than handing out a bad row.
        s = LinearAttentionStrategy()
        _prefill(s, "a", bs=2)
        _prefill(s, "b", bs=2)
        with pytest.raises(AssertionError):
            s.get(True, 2, ["c"], [])

    def test_decode_lookup_of_unallocated_request_raises(self):
        # By decode time every running request has prefilled (been recorded); a
        # missing row means a wiring bug -> fail loud, don't read the wrong row.
        s = LinearAttentionStrategy()
        _prefill(s, "a")
        with pytest.raises(KeyError):
            s.get(False, 4, ["a", "ghost"], [])


class TestSorting:
    """``preprocess`` -> ``decode_row_indices`` turns the per-request rows into the
    tensor used to scatter decode inputs / gather logits.
    """

    def test_decode_row_indices_returns_running_order_rows(self):
        s = LinearAttentionStrategy()
        _prefill(s, "a")  # 0
        _prefill(s, "b")  # 1
        rows = s.decode_row_indices(s.get(False, 4, ["b", "a"], []))
        assert torch.equal(rows, torch.tensor([1, 0]))

    def test_preprocess_delegates_to_decode_row_indices(self):
        s = LinearAttentionStrategy()
        _prefill(s, "a")  # 0
        _prefill(s, "b")  # 1
        table_ids = s.get(False, 4, ["b", "a"], [])
        # cache_positions / request_nums / decoder_batch_size are ignored here.
        out = s.preprocess(table_ids, torch.zeros(2), 2, 4)
        assert torch.equal(out, torch.tensor([1, 0]))


class TestDecodeLayoutWiring:
    """Exercise the REAL wrapper code (only the model mocked): decode must place
    each running request at its ``batch_idx`` row so it meets its OWN recurrent
    state row, regardless of the running order vLLM hands us. The state cache in
    DRAM is fixed; the INPUT is rearranged to match it, then logits are gathered
    back to running order.
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
        obj.attention_manager = AttentionManager(LinearAttentionStrategy())
        return obj

    def test_pad_decoder_items_scatters_to_batch_idx_rows(self):
        # Running order [B, A]; B's row is 1, A's row is 0. The real
        # pad_decoder_items must land B (running idx 0) on row 1 and A on row 0.
        obj = self._bare_qwen3_5(max_batch_size=4)
        input_ids = torch.tensor([[201], [200]])  # running order: B=201, A=200
        positions = torch.tensor([[5], [7]])
        block_tables = torch.tensor([[10], [11]], dtype=torch.int16)
        batch_indices = torch.tensor([1, 0])  # rows of [B, A]

        padded_ids, padded_pos, padded_bt = obj.pad_decoder_items(
            input_ids, positions, block_tables, input_block_ids=batch_indices
        )

        assert padded_ids[1, 0] == 201 and padded_ids[0, 0] == 200  # B->1, A->0
        assert padded_pos[1, 0] == 5 and padded_pos[0, 0] == 7
        assert padded_bt[1, 0] == 10 and padded_bt[0, 0] == 11

    def test_forward_decode_pairs_each_request_with_its_own_row(self):
        # Prefill order was A(row0), B(row1); this decode step's running order is
        # the REVERSE [B, A]. Each must still land on its own row, and logits come
        # back in running order.
        obj = self._bare_qwen3_5(max_batch_size=2)
        obj.attention_manager.add("A", 0)
        obj.attention_manager.add("B", 1)

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
            input_tokens=torch.tensor([[201], [200]]),  # B=201, A=200
            input_positions=torch.tensor([[5], [7]]),
            block_tables=torch.tensor([[10], [11]], dtype=torch.int16),
            position_embed=torch.zeros(2, 2, 1, 1, 1),
        )

        logits = obj.forward(model_input)

        # Physical batch is laid out BY ROW: row0 == A's row, row1 == B's row.
        assert recorded["inputs_embeds"][0, 0, 0] == 200  # A on row 0
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on row 1
        # ...and logits come back in RUNNING order [B, A] -> [201, 200].
        assert logits[0, 0] == 201
        assert logits[1, 0] == 200

    def test_forward_decode_single_request_pads_other_rows(self):
        # Only B runs (A finished earlier and was popped by the runner). The batch
        # stays max_num_seqs wide: B on its row (1); the empty row 0 is dummy
        # padding whose output is dropped. B is unaffected (rows are independent).
        obj = self._bare_qwen3_5(max_batch_size=2)
        obj.attention_manager.add("B", 1)  # A already popped -> only B recorded

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
            input_tokens=torch.tensor([[201]]),
            input_positions=torch.tensor([[9]]),
            block_tables=torch.tensor([[10]], dtype=torch.int16),
            position_embed=torch.zeros(2, 1, 1, 1, 1),
        )
        logits = obj.forward(model_input)

        assert recorded["inputs_embeds"].shape[0] == 2  # still max_num_seqs wide
        assert recorded["inputs_embeds"][1, 0, 0] == 201  # B on its row
        assert recorded["inputs_embeds"][0, 0, 0] == 0  # empty row -> dummy input
        assert logits.shape[0] == 1  # only B's logits returned
        assert logits[0, 0] == 201

    def test_forward_prefill_allocates_and_passes_batch_idx(self):
        # Prefill records the request's row and passes it as ``batch_idx`` to the
        # prefill graph (omitting it raises KeyError inside the runtime).
        obj = self._bare_qwen3_5(max_batch_size=2)
        obj.attention_manager.add("A", 0)  # A already prefilled -> row 0 used

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
            running_requests_ids=["B"],  # new request
            input_tokens=torch.tensor([[1]]),
            input_positions=torch.tensor([[0]]),
            block_tables=torch.tensor([[10]], dtype=torch.int16),
            inputs_embeds=torch.zeros(1, 1, 1),
            position_embed=torch.zeros(2, 1, 1, 1, 1),
        )
        obj.forward(model_input)

        assert passed["batch_idx"] == 1  # lowest free row (0 taken by A)
        assert obj.attention_manager.get(False, 2, ["B"], []) == [1]  # recorded
