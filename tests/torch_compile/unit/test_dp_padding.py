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

from unittest.mock import patch

import pytest
import torch

from vllm_rbln.forward_context import RBLNDPMetadata


def _simulate_all_reduce(per_rank_values: list[int]) -> torch.Tensor:
    """Simulate the result tensor after dist.all_reduce SUM in
    RBLNDPMetadata.num_tokens_across_dp: each rank fills only its slot,
    others are 0, sum reproduces each rank's value in its slot.
    """
    return torch.tensor(per_rank_values, device="cpu", dtype=torch.int32)


def _patch_all_reduce(per_rank_encoded: list[int]):
    """Patch RBLNDPMetadata.num_tokens_across_dp to return the bit-packed
    per-rank tensor we expect to see post-all-reduce on rank 0.
    """
    return patch.object(
        RBLNDPMetadata,
        "num_tokens_across_dp",
        return_value=_simulate_all_reduce(per_rank_encoded),
    )


def _encode(num_tokens: int, num_reqs: int, is_prefill: bool) -> int:
    encoded = num_tokens | (num_reqs << 16)
    if is_prefill:
        encoded |= 1 << 30
    return encoded


# ----- helper tests: num_tokens_and_reqs_across_dp -----


class TestNumTokensAndReqsAcrossDP:
    def test_single_token_uniform_decode(self):
        """All 4 DP ranks each have batch=8, 1 token per req (= 8 tokens)."""
        per_rank = [_encode(8, 8, False) for _ in range(4)]
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
            )
        assert reqs_t is not None
        assert torch.equal(tokens_t, torch.tensor([8, 8, 8, 8], dtype=torch.int32))
        assert torch.equal(reqs_t, torch.tensor([8, 8, 8, 8], dtype=torch.int32))

    def test_multi_token_spec_decode_uniform(self):
        """The exact bug case: batch=8 reqs, 2 tokens/req → 16 tokens.

        Before fix: max_decode_tokens=16 → find_decode_batch_bucket(16)
        Now: num_reqs=8 carried through → consumer can do find_decode_batch_bucket(8).
        """
        per_rank = [_encode(16, 8, False) for _ in range(4)]
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=16, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
            )
        assert reqs_t is not None
        assert torch.equal(tokens_t, torch.tensor([16, 16, 16, 16], dtype=torch.int32))
        assert torch.equal(reqs_t, torch.tensor([8, 8, 8, 8], dtype=torch.int32))

    def test_mixed_batch_sizes_across_ranks(self):
        """Ranks have different batch sizes; helper must surface per-rank
        values so the consumer can take the max."""
        per_rank = [
            _encode(num_tokens=8, num_reqs=8, is_prefill=False),
            _encode(num_tokens=10, num_reqs=5, is_prefill=False),
            _encode(num_tokens=6, num_reqs=3, is_prefill=False),
            _encode(num_tokens=14, num_reqs=7, is_prefill=False),
        ]
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
            )
        assert torch.equal(tokens_t, torch.tensor([8, 10, 6, 14], dtype=torch.int32))
        assert reqs_t is not None
        assert torch.equal(reqs_t, torch.tensor([8, 5, 3, 7], dtype=torch.int32))

    def test_any_prefill_returns_none(self):
        """If any rank is in prefill, num_reqs_across_dp_cpu must be None."""
        per_rank = [
            _encode(num_tokens=8, num_reqs=8, is_prefill=False),
            _encode(num_tokens=300, num_reqs=1, is_prefill=True),  # prefill rank
            _encode(num_tokens=4, num_reqs=4, is_prefill=False),
            _encode(num_tokens=6, num_reqs=6, is_prefill=False),
        ]
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
            )
        assert reqs_t is None
        # tokens still extracted from low 16 bits
        assert torch.equal(tokens_t, torch.tensor([8, 300, 4, 6], dtype=torch.int32))

    def test_local_rank_is_prefill(self):
        """Local rank prefill, others decode → still any_prefill=True."""
        per_rank = [
            _encode(num_tokens=512, num_reqs=1, is_prefill=True),
            _encode(num_tokens=8, num_reqs=8, is_prefill=False),
            _encode(num_tokens=8, num_reqs=8, is_prefill=False),
            _encode(num_tokens=8, num_reqs=8, is_prefill=False),
        ]
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=512,
                num_reqs=1,
                dp_size=4,
                dp_rank=0,
                is_prefill=True,
            )
        assert reqs_t is None
        assert torch.equal(tokens_t, torch.tensor([512, 8, 8, 8], dtype=torch.int32))

    def test_assert_num_tokens_overflow(self):
        with pytest.raises(AssertionError, match="num_tokens=65536"):
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=1 << 16,
                num_reqs=1,
                dp_size=4,
                dp_rank=0,
                is_prefill=False,
            )

    def test_assert_num_reqs_overflow(self):
        with pytest.raises(AssertionError, match="num_reqs=16384"):
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=1,
                num_reqs=1 << 14,
                dp_size=4,
                dp_rank=0,
                is_prefill=False,
            )

    def test_boundary_max_values(self):
        """num_tokens=0xFFFF and num_reqs=0x3FFF must round-trip cleanly."""
        per_rank = [_encode(0xFFFF, 0x3FFF, False)] * 4
        with _patch_all_reduce(per_rank):
            tokens_t, reqs_t = RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens=0xFFFF,
                num_reqs=0x3FFF,
                dp_size=4,
                dp_rank=0,
                is_prefill=False,
            )
        assert reqs_t is not None
        assert tokens_t.tolist() == [0xFFFF] * 4
        assert reqs_t.tolist() == [0x3FFF] * 4


# ----- get_dp_padding tests: end-to-end on the model runner method -----


class _FakeBucketingManager:
    def __init__(self, decode_batch_buckets):
        self.decode_batch_buckets = decode_batch_buckets

    def find_decode_batch_bucket(self, batch_size: int) -> int:
        for b in self.decode_batch_buckets:
            if b >= batch_size:
                return b
        raise ValueError(
            f"No batch bucket >= {batch_size}; buckets={self.decode_batch_buckets}"
        )


class _FakeParallelConfig:
    def __init__(self, dp_size, dp_rank):
        self.data_parallel_size = dp_size
        self.data_parallel_rank = dp_rank


class _FakeVllmConfig:
    def __init__(self, dp_size, dp_rank):
        self.parallel_config = _FakeParallelConfig(dp_size, dp_rank)


class _FakeRunner:
    """Minimal stand-in for RBLNModelRunner exposing only what get_dp_padding uses."""

    def __init__(
        self,
        dp_size,
        dp_rank,
        decode_buckets,
        max_num_batched_tokens,
        specialized_moe_decode,
    ):
        self.vllm_config = _FakeVllmConfig(dp_size, dp_rank)
        self.bucketing_manager = _FakeBucketingManager(decode_buckets)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.specialized_moe_decode = specialized_moe_decode

    # Bound method bridge so we can call the real get_dp_padding logic with
    # this fake as `self`.
    def get_dp_padding(self, *args, **kwargs):
        from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

        return RBLNModelRunner.get_dp_padding(self, *args, **kwargs)


class TestGetDpPadding:
    def test_dp_size_1_short_circuits(self):
        """dp_size=1 short-circuit kept for completeness; the non-DP code
        path bypasses the all-reduce entirely."""
        runner = _FakeRunner(
            dp_size=1,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=True,
        )
        bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
            num_tokens=8,
            num_reqs=8,
            batch_bucket_size=8,
        )
        assert bucket == 8
        assert padded is None
        assert across_dp is None

    def test_single_token_decode_keeps_bucket(self):
        """Sanity: single-token DP decode still produces the right bucket."""
        runner = _FakeRunner(
            dp_size=4,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=True,
        )
        per_rank = [_encode(8, 8, False)] * 4
        with _patch_all_reduce(per_rank):
            bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
                num_tokens=8,
                num_reqs=8,
                batch_bucket_size=8,
            )
        assert bucket == 8
        # single-token → max_tokens_per_req=1 → padded == bucket
        assert padded == 8
        assert across_dp.tolist() == [8, 8, 8, 8]

    def test_multi_token_spec_decode_bug_case(self):
        """The user-reported failure case: batch=8 reqs, 2 tokens/req → 16 total.

        Before the fix find_decode_batch_bucket would be called with 16 and
        raise ValueError. With the fix it must resolve to bucket=8 and pad to
        bucket * tokens_per_req = 16.
        """
        runner = _FakeRunner(
            dp_size=4,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=True,
        )
        per_rank = [_encode(16, 8, False)] * 4
        with _patch_all_reduce(per_rank):
            bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
                num_tokens=16,
                num_reqs=8,
                batch_bucket_size=8,
            )
        assert bucket == 8, "Bucket must be looked up by num_reqs, not num_tokens"
        assert padded == 16, "Padded buffer must fit batch_bucket_size * tokens_per_req"
        assert across_dp.tolist() == [16, 16, 16, 16]

    def test_any_prefill_falls_back_to_max_num_batched_tokens(self):
        runner = _FakeRunner(
            dp_size=4,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=True,
        )
        per_rank = [
            _encode(8, 8, False),
            _encode(256, 1, True),  # prefill rank
            _encode(8, 8, False),
            _encode(8, 8, False),
        ]
        with _patch_all_reduce(per_rank):
            bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
                num_tokens=8,
                num_reqs=8,
                batch_bucket_size=8,
                is_prefill=False,
            )
        # any_prefill path: bucket left as the initial (caller-provided) value,
        # padded falls back to max_num_batched_tokens.
        assert bucket == 8  # caller's initial guess preserved on any-prefill path
        assert padded == 512

    def test_non_specialized_moe_uses_max_num_batched_tokens(self):
        runner = _FakeRunner(
            dp_size=4,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=False,
        )
        per_rank = [_encode(8, 8, False)] * 4
        with _patch_all_reduce(per_rank):
            bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
                num_tokens=8,
                num_reqs=8,
                batch_bucket_size=8,
            )
        assert padded == 512

    def test_num_padded_tokens_path_a(self):
        """Path A: explicit num_padded_tokens passed in → short-circuit."""
        runner = _FakeRunner(
            dp_size=4,
            dp_rank=0,
            decode_buckets=[1, 4, 8],
            max_num_batched_tokens=512,
            specialized_moe_decode=True,
        )
        # path A uses plain num_tokens_across_dp (no bit-packing), so patch
        # that directly.
        per_rank = [8, 8, 8, 8]
        with _patch_all_reduce(per_rank):
            bucket, padded, across_dp, _max_per_req = runner.get_dp_padding(
                num_tokens=8,
                num_reqs=8,
                batch_bucket_size=8,
                num_padded_tokens=512,
                is_prefill=False,
            )
        assert bucket == 8
        assert padded == 512
        assert across_dp.tolist() == [8, 8, 8, 8]
