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

"""Tests for the decode batch bucketing package: pure integer logic choosing
which compiled decode batch size a request lands in. Covers each strategy's
build rule, the shared base behavior, and the factory dispatch."""

import pytest

from vllm_rbln.v1.worker.bucketing import get_bucketing_manager
from vllm_rbln.v1.worker.bucketing.bucketing_manager import RBLNBucketingManager
from vllm_rbln.v1.worker.bucketing.exponential_bucketing_manager import (
    ExponentialBucketingManager,
)
from vllm_rbln.v1.worker.bucketing.linear_bucketing_manager import (
    LinearBucketingManager,
)
from vllm_rbln.v1.worker.bucketing.manual_bucketing_manager import (
    ManualBucketingManager,
)


class TestCheckConfig:
    def test_accepts_valid(self):
        # A valid config raises nothing.
        RBLNBucketingManager.check_config(
            max_batch_size=8, min_batch_size=1, limit=4, step=2
        )

    def test_rejects_max_less_than_min(self):
        with pytest.raises(ValueError):
            RBLNBucketingManager.check_config(
                max_batch_size=2, min_batch_size=4, limit=1, step=2
            )

    def test_rejects_nonpositive_limit(self):
        with pytest.raises(ValueError):
            RBLNBucketingManager.check_config(
                max_batch_size=8, min_batch_size=1, limit=0, step=2
            )

    def test_rejects_nonpositive_step(self):
        with pytest.raises(ValueError):
            RBLNBucketingManager.check_config(
                max_batch_size=8, min_batch_size=1, limit=4, step=0
            )

    def test_rejects_nonpositive_min(self):
        with pytest.raises(ValueError):
            RBLNBucketingManager.check_config(
                max_batch_size=8, min_batch_size=0, limit=4, step=2
            )


class TestExponentialBucketing:
    def test_builds_shrinking_buckets(self):
        # Halve from max until the limit count is reached.
        m = ExponentialBucketingManager(
            max_batch_size=16, min_batch_size=1, limit=5, step=2
        )
        assert m.decode_batch_buckets == [1, 2, 4, 8, 16]

    def test_stops_at_min_batch_size(self):
        # Stop once the next candidate would drop below min (limit not reached).
        m = ExponentialBucketingManager(
            max_batch_size=16, min_batch_size=4, limit=10, step=2
        )
        assert m.decode_batch_buckets == [4, 8, 16]

    def test_stops_at_limit(self):
        # Stop once limit buckets exist.
        m = ExponentialBucketingManager(
            max_batch_size=64, min_batch_size=1, limit=2, step=2
        )
        assert m.decode_batch_buckets == [32, 64]

    def test_limit_one_yields_only_max(self):
        # The default limit (1) produces a single bucket equal to max.
        m = ExponentialBucketingManager(
            max_batch_size=16, min_batch_size=1, limit=1, step=2
        )
        assert m.decode_batch_buckets == [16]

    def test_step_must_exceed_one(self):
        # step == 1 passes check_config but the exponential build rejects it.
        with pytest.raises(ValueError):
            ExponentialBucketingManager(
                max_batch_size=8, min_batch_size=1, limit=4, step=1
            )

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            ExponentialBucketingManager(
                max_batch_size=2, min_batch_size=4, limit=4, step=2
            )


class TestLinearBucketing:
    def test_builds_evenly_spaced(self):
        # Decrease from max by a fixed step.
        m = LinearBucketingManager(max_batch_size=10, min_batch_size=1, limit=5, step=2)
        assert m.decode_batch_buckets == [2, 4, 6, 8, 10]

    def test_stops_at_min_batch_size(self):
        m = LinearBucketingManager(
            max_batch_size=10, min_batch_size=6, limit=10, step=2
        )
        assert m.decode_batch_buckets == [6, 8, 10]

    def test_stops_at_limit(self):
        m = LinearBucketingManager(max_batch_size=10, min_batch_size=1, limit=2, step=3)
        assert m.decode_batch_buckets == [7, 10]

    def test_limit_one_yields_only_max(self):
        # The default limit (1) produces a single bucket equal to max.
        m = LinearBucketingManager(max_batch_size=10, min_batch_size=1, limit=1, step=2)
        assert m.decode_batch_buckets == [10]

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            LinearBucketingManager(max_batch_size=2, min_batch_size=4, limit=4, step=2)


class TestManualBucketing:
    def test_uses_sorted_unique_buckets(self):
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[4, 2, 8, 2])
        assert m.decode_batch_buckets == [2, 4, 8]

    def test_last_must_equal_max(self):
        # The largest manual bucket must equal max_batch_size.
        with pytest.raises(ValueError):
            ManualBucketingManager(max_batch_size=8, manual_buckets=[2, 4])

    def test_empty_raises(self):
        with pytest.raises(AssertionError):
            ManualBucketingManager(max_batch_size=8, manual_buckets=[])

    def test_single_bucket(self):
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[8])
        assert m.decode_batch_buckets == [8]


class TestBaseProperties:
    def test_batch_buckets_reserves_one_for_prefill(self):
        # batch_buckets == sorted({1} | decode); 1 is always present for prefill.
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[4, 8])
        assert m.decode_batch_buckets == [4, 8]
        assert m.batch_buckets == [1, 4, 8]

    def test_batch_buckets_no_duplicate_one(self):
        # A decode bucket of 1 is not duplicated by the reserved prefill bucket.
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[1, 4, 8])
        assert m.batch_buckets == [1, 4, 8]

    def test_counts(self):
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[4, 8])
        assert m.batch_buckets_count == 3
        assert m.decode_batch_buckets_count == 2

    def test_find_smallest_bucket_ge_batch_size(self):
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[2, 4, 8])
        assert m.find_decode_batch_bucket(1) == 2
        assert m.find_decode_batch_bucket(2) == 2
        assert m.find_decode_batch_bucket(3) == 4
        assert m.find_decode_batch_bucket(8) == 8

    def test_find_raises_when_larger_than_all(self):
        m = ManualBucketingManager(max_batch_size=8, manual_buckets=[2, 4, 8])
        with pytest.raises(ValueError):
            m.find_decode_batch_bucket(9)


class TestGetBucketingManagerFactory:
    def test_exponential(self):
        m = get_bucketing_manager("exponential", max_batch_size=16, limit=5, step=2)
        assert isinstance(m, ExponentialBucketingManager)
        assert m.decode_batch_buckets == [1, 2, 4, 8, 16]

    def test_linear(self):
        m = get_bucketing_manager("linear", max_batch_size=10, limit=5, step=2)
        assert isinstance(m, LinearBucketingManager)
        assert m.decode_batch_buckets == [2, 4, 6, 8, 10]

    def test_manual(self):
        m = get_bucketing_manager("manual", max_batch_size=8, manual_buckets=[4, 8])
        assert isinstance(m, ManualBucketingManager)
        assert m.decode_batch_buckets == [4, 8]

    def test_manual_without_buckets_raises(self):
        # manual_buckets defaults to [] -> ManualBucketingManager asserts non-empty.
        with pytest.raises(AssertionError):
            get_bucketing_manager("manual", max_batch_size=8)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            get_bucketing_manager("bogus", max_batch_size=8)

    def test_exp_alias_not_accepted_by_factory(self):
        # "exp" is an env-level alias normalized before this point, so the
        # factory only takes canonical strategies.
        with pytest.raises(ValueError):
            get_bucketing_manager("exp", max_batch_size=16, limit=5, step=2)
