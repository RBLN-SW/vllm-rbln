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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID

import vllm_rbln.v1.spec_decode.eagle as eagle_module
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer


class FakeBackupNextTokenIds:
    def __init__(self, size: int):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = torch.zeros(size, dtype=torch.int32)

    def copy_to_gpu(self, num_reqs: int) -> None:
        self.gpu[:num_reqs] = torch.from_numpy(self.np[:num_reqs])


class FakeRequestState:
    def __init__(self, token_base: int):
        self.token_base = token_base

    def get_token_id(self, seq_len: int) -> int:
        return self.token_base + seq_len


class FakeMetadataBuilder:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.metas: list[SimpleNamespace] = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        meta = SimpleNamespace(kv_caches=None, kwargs=kwargs)
        self.metas.append(meta)
        return meta


class FakeAttentionGroup:
    def __init__(self, layer_names: list[str], builder: FakeMetadataBuilder):
        self.layer_names = layer_names
        self._builder = builder

    def get_metadata_builder(self) -> FakeMetadataBuilder:
        return self._builder


class FakeBucketingManager:
    def __init__(self, batch_bucket_size: int):
        self.batch_bucket_size = batch_bucket_size

    def find_decode_batch_bucket(self, batch_size: int) -> int:
        assert batch_size <= self.batch_bucket_size
        return self.batch_bucket_size


class FakeRunner:
    def __init__(self, batch_bucket_size: int, num_tokens_no_spec, is_prefill: bool):
        self.bucketing_manager = FakeBucketingManager(batch_bucket_size)
        self.input_batch = SimpleNamespace(num_tokens_no_spec=num_tokens_no_spec)
        self.kv_caches = [torch.tensor([11], dtype=torch.int32)]
        self._is_prefill = is_prefill

    def is_prefills(self):
        return [self._is_prefill]


def make_common_attn_metadata(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_tensor: torch.Tensor | None = None,
) -> CommonAttentionMetadata:
    total_num_tokens = query_start_loc[-1].item()
    if block_table_tensor is None:
        block_table_tensor = torch.zeros((seq_lens.shape[0], 4), dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.zeros_like(seq_lens),
        num_reqs=seq_lens.shape[0],
        num_actual_tokens=total_num_tokens,
        max_query_len=(query_start_loc[1:] - query_start_loc[:-1]).max().item(),
        max_seq_len=seq_lens.max().item(),
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.arange(total_num_tokens, dtype=torch.int64),
        causal=True,
        dcp_local_seq_lens=None,
    )


def make_fake_proposer(
    *,
    batch_bucket_size: int,
    is_prefill: bool,
    num_speculative_tokens: int,
    hidden_size: int = 4,
    max_num_tokens: int = 32,
    max_model_len: int = 32,
) -> tuple[SimpleNamespace, FakeMetadataBuilder]:
    builder = FakeMetadataBuilder()
    fake = SimpleNamespace()
    fake.method = "eagle"
    fake.hidden_size = hidden_size
    fake.supports_mm_inputs = False
    fake.runner = FakeRunner(
        batch_bucket_size=batch_bucket_size,
        num_tokens_no_spec=torch.tensor([2, 2], dtype=torch.int32),
        is_prefill=is_prefill,
    )
    fake.draft_attn_groups = [FakeAttentionGroup(["draft.layer"], builder)]
    fake.input_ids = torch.full((max_num_tokens,), -1, dtype=torch.int32)
    fake.positions = torch.full((max_num_tokens,), -1, dtype=torch.int64)
    fake.hidden_states = torch.zeros((max_num_tokens, hidden_size), dtype=torch.float32)
    fake.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=True)
    )
    fake.num_speculative_tokens = num_speculative_tokens
    fake.uses_mrope = False
    fake.allowed_attn_types = None
    fake.arange = torch.arange(max_num_tokens + 1, dtype=torch.int32)
    fake.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)
    fake.block_size = 4
    fake.max_model_len = max_model_len
    fake.compile_context = SimpleNamespace(mark_static_address=Mock())
    fake.backup_next_token_ids = FakeBackupNextTokenIds(size=8)

    def _set_positions(num_tokens: int, positions: torch.Tensor) -> None:
        fake.positions[:num_tokens] = positions

    fake._set_positions = _set_positions
    fake.set_inputs_first_pass = (
        lambda **kwargs: RBLNEagleProposer.set_inputs_first_pass(fake, **kwargs)
    )
    return fake, builder


@pytest.fixture(autouse=True)
def patch_forward_context(monkeypatch):
    monkeypatch.setattr(
        eagle_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )


# Verifies that the first-pass helper shifts tokens and infers default sample indices.
def test_set_inputs_first_pass_sets_shifted_tokens_and_default_indices():
    fake, _ = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = False
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
    )
    target_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

    num_tokens, token_indices_to_sample, returned_cad = (
        RBLNEagleProposer.set_inputs_first_pass(
            fake,
            target_token_ids=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32),
            next_token_ids=torch.tensor([99, 88], dtype=torch.int32),
            target_positions=target_positions,
            target_hidden_states=torch.zeros(
                (5, fake.hidden_size), dtype=torch.float32
            ),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )
    )

    assert num_tokens == 5
    assert returned_cad is cad
    torch.testing.assert_close(
        token_indices_to_sample,
        torch.tensor([1, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        fake.input_ids[:5],
        torch.tensor([11, 99, 21, 22, 88], dtype=torch.int32),
    )
    torch.testing.assert_close(fake.positions[:5], target_positions)


# Verifies that extra input slot mode is explicitly unsupported in RBLN EAGLE.
def test_set_inputs_first_pass_rejects_extra_input_slots():
    fake, _ = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = True
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
    )

    with pytest.raises(NotImplementedError, match="extra input slots"):
        RBLNEagleProposer.set_inputs_first_pass(
            fake,
            target_token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            next_token_ids=torch.tensor([10, 11], dtype=torch.int32),
            target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            target_hidden_states=torch.zeros(
                (4, fake.hidden_size), dtype=torch.float32
            ),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )


# Verifies that backup tokens are sourced from request state
# when sampling output is discarded or empty.
def test_prepare_next_token_ids_padded_uses_request_backup_tokens():
    fake, _ = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    common_attn_metadata = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        seq_lens=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )
    requests = {
        "req-0": FakeRequestState(100),
        "req-1": FakeRequestState(100),
        "req-2": FakeRequestState(100),
        "req-3": FakeRequestState(100),
    }
    gpu_input_batch = SimpleNamespace(
        num_reqs=4,
        req_ids=["req-0", "req-1", "req-2", "req-3"],
        vocab_size=50,
    )

    next_token_ids, valid_count = RBLNEagleProposer.prepare_next_token_ids_padded(
        fake,
        common_attn_metadata=common_attn_metadata,
        sampled_token_ids=torch.tensor(
            [
                [5, -1, -1],
                [6, 7, -1],
                [-1, -1, -1],
            ],
            dtype=torch.int32,
        ),
        requests=requests,
        gpu_input_batch=gpu_input_batch,
        discard_request_mask=torch.tensor([False, True, False, False]),
    )

    torch.testing.assert_close(
        next_token_ids,
        torch.tensor([5, 102, 103], dtype=torch.int32),
    )
    torch.testing.assert_close(
        valid_count,
        torch.tensor([1, 0, 0], dtype=torch.int32),
    )


# Verifies the decode-path proposer reshapes and pads inputs
# before calling the draft model.
def test_propose_decode_path_pads_inputs_and_hidden_states():
    fake, builder = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = False
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )
    target_positions = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    target_hidden_states = torch.arange(32, dtype=torch.float32).view(8, 4)

    def model_executable(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds,
        last_token_indices: torch.Tensor | None,
    ):
        assert inputs_embeds is None
        assert input_ids.shape == (4, 2)
        assert positions.shape == (4, 2)
        assert hidden_states.shape == (4, 2, 4)
        torch.testing.assert_close(
            last_token_indices,
            torch.tensor([1, 3, 0, 0], dtype=torch.int32),
        )
        logits = torch.tensor(
            [[0.0, 5.0, 1.0], [0.0, 1.0, 7.0]],
            dtype=torch.float32,
        )
        return hidden_states.view(-1, fake.hidden_size), logits

    fake.model_executable = model_executable

    output = RBLNEagleProposer.propose(
        fake,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
        sampling_metadata=None,
    )

    assert len(builder.calls) == 1
    assert builder.metas[0].kv_caches is fake.runner.kv_caches
    torch.testing.assert_close(output, torch.tensor([[1], [2]], dtype=torch.int64))


# Verifies that eagle3 combines hidden states before the first draft-model forward pass.
def test_propose_eagle3_combines_hidden_states_before_forward():
    fake, builder = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = False
    fake.method = "eagle3"
    raw_hidden_states = torch.arange(64, dtype=torch.float32).view(8, 8)
    combined_hidden_states = torch.arange(32, dtype=torch.float32).view(8, 4) + 200
    fake.model = SimpleNamespace(
        combine_hidden_states=Mock(return_value=combined_hidden_states)
    )
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )
    target_positions = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    def model_executable(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds,
        last_token_indices: torch.Tensor | None,
    ):
        assert inputs_embeds is None
        assert input_ids.shape == (4, 2)
        assert positions.shape == (4, 2)
        assert hidden_states.shape == (4, 2, 4)
        torch.testing.assert_close(
            hidden_states,
            combined_hidden_states.view(4, 2, 4),
        )
        torch.testing.assert_close(
            last_token_indices,
            torch.tensor([1, 3, 0, 0], dtype=torch.int32),
        )
        logits = torch.tensor(
            [[0.0, 6.0, 1.0], [0.0, 1.0, 8.0]],
            dtype=torch.float32,
        )
        return hidden_states.view(-1, fake.hidden_size), logits

    fake.model_executable = model_executable

    output = RBLNEagleProposer.propose(
        fake,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=target_positions,
        target_hidden_states=raw_hidden_states,
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
        sampling_metadata=None,
    )

    fake.model.combine_hidden_states.assert_called_once_with(raw_hidden_states)
    assert len(builder.calls) == 1
    torch.testing.assert_close(output, torch.tensor([[1], [2]], dtype=torch.int64))


# Verifies that eagle3 rejects combined hidden states
# with an unexpected final dimension.
def test_propose_eagle3_asserts_combined_hidden_size():
    fake, _ = make_fake_proposer(
        batch_bucket_size=4, is_prefill=False, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = False
    fake.method = "eagle3"
    fake.model = SimpleNamespace(
        combine_hidden_states=Mock(
            return_value=torch.zeros((8, fake.hidden_size + 1), dtype=torch.float32)
        )
    )
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    with pytest.raises(AssertionError):
        RBLNEagleProposer.propose(
            fake,
            target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
            target_positions=torch.tensor([4, 5, 6, 7], dtype=torch.int64),
            target_hidden_states=torch.zeros((8, 8), dtype=torch.float32),
            next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
            token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
            common_attn_metadata=cad,
            sampling_metadata=None,
        )


# Verifies the prefill-path proposer uses the current
# wide batch shape expected by the runner.
def test_propose_prefill_path_keeps_unpadded_batch_shape():
    fake, _ = make_fake_proposer(
        batch_bucket_size=4, is_prefill=True, num_speculative_tokens=1
    )
    fake.needs_extra_input_slots = False
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
    )

    def model_executable(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds,
        last_token_indices: torch.Tensor | None,
    ):
        assert inputs_embeds is None
        assert input_ids.shape == (2, 16)
        assert positions.shape == (2, 16)
        assert hidden_states.shape == (2, 16, 4)
        torch.testing.assert_close(
            last_token_indices,
            torch.tensor([1, 3, 0, 0], dtype=torch.int32),
        )
        logits = torch.tensor(
            [[0.0, 2.0, 1.0], [0.0, 1.0, 4.0]],
            dtype=torch.float32,
        )
        return hidden_states.view(-1, fake.hidden_size), logits

    fake.model_executable = model_executable

    output = RBLNEagleProposer.propose(
        fake,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        target_hidden_states=torch.arange(128, dtype=torch.float32).view(32, 4),
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
        sampling_metadata=None,
    )

    torch.testing.assert_close(output, torch.tensor([[1], [2]], dtype=torch.int64))


# Verifies multi-step drafting updates metadata,
# slot mapping, and attention metadata between passes.
def test_propose_multistep_updates_metadata_and_rebuilds_attention():
    fake, builder = make_fake_proposer(
        batch_bucket_size=4,
        is_prefill=False,
        num_speculative_tokens=2,
        max_model_len=9,
    )
    fake.needs_extra_input_slots = False
    cad = make_common_attn_metadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([10, 11], dtype=torch.int32),
        block_table_tensor=torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
            ],
            dtype=torch.int32,
        ),
    )
    target_positions = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
    target_hidden_states = torch.arange(32, dtype=torch.float32).view(8, 4)
    first_hidden_states = torch.arange(32, dtype=torch.float32).view(8, 4) + 100
    calls: list[dict[str, torch.Tensor | None]] = []

    def model_executable(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds,
        last_token_indices: torch.Tensor | None,
    ):
        calls.append(
            {
                "input_ids": input_ids.clone(),
                "positions": positions.clone(),
                "hidden_states": hidden_states.clone(),
                "last_token_indices": None
                if last_token_indices is None
                else last_token_indices.clone(),
            }
        )
        if len(calls) == 1:
            assert input_ids.shape == (4, 2)
            assert positions.shape == (4, 2)
            assert hidden_states.shape == (4, 2, 4)
            torch.testing.assert_close(
                last_token_indices,
                torch.tensor([1, 3, 0, 0], dtype=torch.int32),
            )
            logits = torch.tensor(
                [[0.0, 5.0, 1.0], [0.0, 1.0, 7.0]],
                dtype=torch.float32,
            )
            return first_hidden_states, logits

        assert len(calls) == 2
        assert input_ids.shape == (4, 1)
        assert positions.shape == (4, 1)
        assert hidden_states.shape == (4, 1, 4)
        assert last_token_indices is None

        expected_hidden = torch.zeros((4, 1, 4), dtype=torch.float32)
        expected_hidden[0, 0] = first_hidden_states[1]
        expected_hidden[1, 0] = first_hidden_states[3]
        torch.testing.assert_close(hidden_states, expected_hidden)

        logits = torch.tensor(
            [[0.0, 1.0, 0.0, 8.0, 0.0], [0.0, 1.0, 0.0, 0.0, 9.0]],
            dtype=torch.float32,
        )
        return torch.zeros((4, 4), dtype=torch.float32), logits

    fake.model_executable = model_executable

    output = RBLNEagleProposer.propose(
        fake,
        target_token_ids=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=torch.tensor([30, 31], dtype=torch.int32),
        token_indices_to_sample=torch.tensor([1, 3], dtype=torch.int32),
        common_attn_metadata=cad,
        sampling_metadata=None,
        num_rejected_tokens_gpu=torch.tensor([1, 0], dtype=torch.int32),
    )

    assert len(builder.calls) == 2
    assert all(meta.kv_caches is fake.runner.kv_caches for meta in builder.metas)
    assert cad._seq_lens_cpu is None
    assert cad._num_computed_tokens_cpu is None
    torch.testing.assert_close(cad.seq_lens, torch.tensor([10, 1], dtype=torch.int32))
    torch.testing.assert_close(
        cad.slot_mapping,
        torch.tensor([47, PADDING_SLOT_ID], dtype=torch.int64),
    )
    torch.testing.assert_close(
        output,
        torch.tensor([[1, 3], [2, 4]], dtype=torch.int64),
    )
