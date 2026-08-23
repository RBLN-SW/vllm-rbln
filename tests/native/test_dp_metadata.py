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

# RBLNDPMetadata's DP token/req all-reduce. The collective is faked at the
# dist.all_reduce primitive, so the real bit-packing runs and the fake only sums
# in the other ranks' values.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_rbln.forward_context as fc
from vllm_rbln.forward_context import RBLNDPMetadata


def _encode(num_tokens: int, num_reqs: int, is_prefill: bool) -> int:
    # Mirrors the source bit layout: tokens in bits 0..15, reqs in 16..29,
    # prefill flag at bit 30.
    encoded = num_tokens | (num_reqs << 16)
    if is_prefill:
        encoded |= 1 << 30
    return encoded


@pytest.fixture
def fake_dp_collective(monkeypatch):
    """Returns a setter taking ``{rank: value}`` to add into the reduce result,
    as if each remote rank had contributed ``value`` in its own slot."""
    import vllm.distributed.parallel_state as ps

    dp_group = MagicMock()
    dp_group.cpu_group = object()
    monkeypatch.setattr(ps, "get_dp_group", lambda: dp_group)

    others: dict[int, int] = {}

    def fake_all_reduce(tensor, group=None):
        assert group is dp_group.cpu_group
        for rank, value in others.items():
            tensor[rank] += value

    monkeypatch.setattr(fc.dist, "all_reduce", fake_all_reduce)

    def set_others(new_others: dict[int, int]) -> None:
        others.clear()
        others.update(new_others)

    return set_others


class TestNumTokensAndReqsAcrossDP:
    def test_mixed_across_ranks(self, fake_dp_collective):
        # Distinct per-rank counts must be surfaced separately (a consumer takes
        # the max), and tokens != reqs proves the two fields don't bleed.
        fake_dp_collective(
            {
                1: _encode(10, 5, False),
                2: _encode(6, 3, False),
                3: _encode(14, 7, False),
            }
        )
        tokens, reqs, any_prefill = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert tokens.cpu().tolist() == [8, 10, 6, 14]
        assert reqs.cpu().tolist() == [8, 5, 3, 7]
        assert any_prefill is False

    def test_any_remote_prefill_flags(self, fake_dp_collective):
        # A single prefill rank raises the flag; both count fields are still
        # extracted, so a consumer can pick what to trust.
        fake_dp_collective(
            {
                1: _encode(300, 1, True),  # prefill rank
                2: _encode(4, 4, False),
                3: _encode(6, 6, False),
            }
        )
        tokens, reqs, any_prefill = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert any_prefill is True
        assert tokens.cpu().tolist() == [8, 300, 4, 6]
        assert reqs.cpu().tolist() == [8, 1, 4, 6]

    def test_local_prefill_flags(self, fake_dp_collective):
        # The local rank being in prefill also trips any_prefill.
        fake_dp_collective({r: _encode(8, 8, False) for r in (1, 2, 3)})
        tokens, reqs, any_prefill = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
            num_tokens=512, num_reqs=1, dp_size=4, dp_rank=0, is_prefill=True
        )
        assert any_prefill is True
        assert tokens.cpu().tolist() == [512, 8, 8, 8]
        assert reqs.cpu().tolist() == [1, 8, 8, 8]

    def test_boundary_max_values_round_trip(self, fake_dp_collective):
        # The largest values each field can hold must survive pack/unpack.
        fake_dp_collective({r: _encode(0xFFFF, 0x3FFF, False) for r in (1, 2, 3)})
        tokens, reqs, _ = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
            num_tokens=0xFFFF, num_reqs=0x3FFF, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert tokens.cpu().tolist() == [0xFFFF] * 4
        assert reqs.cpu().tolist() == [0x3FFF] * 4

    def test_num_tokens_overflow_asserts(self):
        # The assert fires before any collective, so no fake is needed.
        with pytest.raises(AssertionError, match="num_tokens=65536"):
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=1 << 16, num_reqs=1, dp_size=4, dp_rank=0, is_prefill=False
            )

    def test_num_reqs_overflow_asserts(self):
        with pytest.raises(AssertionError, match="num_reqs=16384"):
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=1, num_reqs=1 << 14, dp_size=4, dp_rank=0, is_prefill=False
            )


class TestNumTokensAcrossDP:
    def test_single_rank_identity(self, fake_dp_collective):
        # world_size 1: the reduce is a no-op, so only the self slot is present.
        fake_dp_collective({})
        out = RBLNDPMetadata.num_tokens_across_dp(num_tokens=7, dp_size=1, dp_rank=0)
        assert out.dtype == torch.int32
        assert out.cpu().tolist() == [7]

    def test_places_self_at_own_rank_slot(self, fake_dp_collective):
        # A non-zero dp_rank proves the source writes at index dp_rank, not slot
        # 0; a slot-0 test could not tell placement from a constant.
        fake_dp_collective({0: 11, 1: 22, 3: 33})
        out = RBLNDPMetadata.num_tokens_across_dp(num_tokens=7, dp_size=4, dp_rank=2)
        assert out.cpu().tolist() == [11, 22, 7, 33]


class TestMake:
    def test_non_dp_builds_single_slot(self):
        parallel_config = SimpleNamespace(data_parallel_size=1)
        meta = RBLNDPMetadata.make(parallel_config, num_tokens=8)
        assert meta.num_tokens_across_dp_cpu.cpu().tolist() == [8]
        assert meta.max_pads_across_dp is None

    @pytest.mark.parametrize(
        "extra",
        [
            {"num_tokens_across_dp": torch.tensor([8], dtype=torch.int32)},
            {"num_padded_tokens": 16},
        ],
    )
    def test_non_dp_rejects_dp_only_args(self, extra):
        parallel_config = SimpleNamespace(data_parallel_size=1)
        with pytest.raises(AssertionError):
            RBLNDPMetadata.make(parallel_config, num_tokens=8, **extra)

    def test_dp_uses_provided_counts_and_pad_buffer(self):
        parallel_config = SimpleNamespace(data_parallel_size=4)
        across = torch.tensor([8, 8, 8, 8], dtype=torch.int32)
        meta = RBLNDPMetadata.make(
            parallel_config,
            num_tokens=8,
            num_tokens_across_dp=across,
            num_padded_tokens=16,
        )
        assert meta.num_tokens_across_dp_cpu.cpu().tolist() == [8, 8, 8, 8]
        assert meta.max_pads_across_dp.shape == (16,)

    @pytest.mark.parametrize(
        "extra",
        [
            {"num_padded_tokens": 16},  # missing num_tokens_across_dp
            {"num_tokens_across_dp": torch.tensor([8, 8, 8, 8], dtype=torch.int32)},
        ],
    )
    def test_dp_requires_both_args(self, extra):
        parallel_config = SimpleNamespace(data_parallel_size=4)
        with pytest.raises(AssertionError):
            RBLNDPMetadata.make(parallel_config, num_tokens=8, **extra)
