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

import pytest

from vllm_rbln.v1.worker.bucketing import (
    ExponentialBucketingManager,
    LinearBucketingManager,
    ManualBucketingManager,
    RBLNBucketingManager,
    get_bucketing_manager,
)


class DummyManager(RBLNBucketingManager):
    def _build_decode_buckets(self):
        self.decode_batch_buckets = [2, 4, 8]


def test_base_manager_properties_and_find():
    manager = DummyManager(max_batch_size=8)
    assert manager.decode_batch_buckets == [2, 4, 8]
    assert manager.batch_buckets == [1, 2, 4, 8]
    assert manager.decode_batch_buckets_count == 3
    assert manager.batch_buckets_count == 4
    assert manager.find_decode_batch_bucket(1) == 2
    assert manager.find_decode_batch_bucket(4) == 4
    assert manager.find_decode_batch_bucket(7) == 8


def test_base_manager_find_decode_bucket_not_found():
    manager = DummyManager(max_batch_size=8)
    with pytest.raises(ValueError, match="No batch bucket found"):
        manager.find_decode_batch_bucket(9)


def test_base_manager_abstract_method_not_defined():
    class Foo(RBLNBucketingManager):
        pass

    with pytest.raises(TypeError):
        Foo(1)

    class Bar(RBLNBucketingManager):
        def _build_decode_buckets(self):
            super()._build_decode_buckets()

    with pytest.raises(
        NotImplementedError, match="Subclasses must implement this method"
    ):
        Bar(1)


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        pytest.param(
            {"max_batch_size": 1, "min_batch_size": 2, "limit": 1, "step": 1},
            "max_batch_size must be >= min_batch_size",
            id="max_lt_min",
        ),
        pytest.param(
            {"max_batch_size": 2, "min_batch_size": 1, "limit": 0, "step": 1},
            "limit must be greater than 0",
            id="non_positive_limit",
        ),
        pytest.param(
            {"max_batch_size": 2, "min_batch_size": 1, "limit": 1, "step": 0},
            "step must be greater than 0",
            id="non_positive_step",
        ),
        pytest.param(
            {"max_batch_size": 2, "min_batch_size": 0, "limit": 1, "step": 1},
            "min_batch_size must be greater than 0",
            id="non_positive_min",
        ),
    ],
)
def test_check_config_raises_for_invalid_config(kwargs, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        RBLNBucketingManager.check_config(**kwargs)


def test_check_config_allows_valid_config():
    RBLNBucketingManager.check_config(
        max_batch_size=8,
        min_batch_size=1,
        limit=4,
        step=2,
    )


def test_exponential_bucketing_manager_builds_and_stops_at_limit():
    manager = ExponentialBucketingManager(
        max_batch_size=64,
        min_batch_size=4,
        limit=4,
        step=2,
    )
    assert manager.decode_batch_buckets == [8, 16, 32, 64]
    assert manager.batch_buckets == [1, 8, 16, 32, 64]


def test_exponential_bucketing_manager_breaks_when_under_minimum():
    manager = ExponentialBucketingManager(
        max_batch_size=10,
        min_batch_size=6,
        limit=5,
        step=2,
    )
    assert manager.decode_batch_buckets == [10]


def test_exponential_bucketing_manager_requires_step_over_one():
    with pytest.raises(ValueError, match="step must be greater than 1"):
        ExponentialBucketingManager(
            max_batch_size=8,
            min_batch_size=1,
            limit=2,
            step=1,
        )


def test_linear_bucketing_manager_builds_and_stops_at_limit():
    manager = LinearBucketingManager(
        max_batch_size=10,
        min_batch_size=1,
        limit=4,
        step=3,
    )
    assert manager.decode_batch_buckets == [1, 4, 7, 10]
    assert manager.batch_buckets == [1, 4, 7, 10]


def test_linear_bucketing_manager_breaks_when_under_minimum():
    manager = LinearBucketingManager(
        max_batch_size=10,
        min_batch_size=8,
        limit=5,
        step=3,
    )
    assert manager.decode_batch_buckets == [10]


def test_manual_bucketing_manager_builds_sorted_unique_buckets():
    manager = ManualBucketingManager(
        max_batch_size=8,
        manual_buckets=[8, 2, 4, 8],
    )
    assert manager.decode_batch_buckets == [2, 4, 8]
    assert manager.batch_buckets == [1, 2, 4, 8]


def test_manual_bucketing_manager_requires_non_empty_buckets():
    with pytest.raises(AssertionError, match="manual_buckets must be non-empty"):
        ManualBucketingManager(max_batch_size=8, manual_buckets=[])
    with pytest.raises(AssertionError, match="manual_buckets must be non-empty"):
        get_bucketing_manager("manual", max_batch_size=8)


def test_manual_bucketing_manager_requires_last_bucket_to_match_max():
    with pytest.raises(ValueError, match="last manual bucket"):
        ManualBucketingManager(
            max_batch_size=8,
            manual_buckets=[2, 4, 7],
        )


def test_get_bucketing_manager_for_all_strategies():
    exp_manager = get_bucketing_manager(
        "exponential",
        max_batch_size=8,
        min_batch_size=1,
        limit=3,
        step=2,
    )
    linear_manager = get_bucketing_manager(
        "linear",
        max_batch_size=8,
        min_batch_size=1,
        limit=3,
        step=2,
    )
    manual_manager = get_bucketing_manager(
        "manual",
        max_batch_size=8,
        manual_buckets=[2, 8],
    )
    assert isinstance(exp_manager, ExponentialBucketingManager)
    assert isinstance(linear_manager, LinearBucketingManager)
    assert isinstance(manual_manager, ManualBucketingManager)


def test_get_bucketing_manager_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Invalid bucketing strategy"):
        get_bucketing_manager("unknown", max_batch_size=8)
