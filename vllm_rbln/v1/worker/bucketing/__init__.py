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

from .bucketing_manager import RBLNBucketingManager
from .exponential_bucketing_manager import ExponentialBucketingManager
from .linear_bucketing_manager import LinearBucketingManager
from .manual_bucketing_manager import ManualBucketingManager

__all__ = [
    "RBLNBucketingManager",
    "ExponentialBucketingManager",
    "LinearBucketingManager",
    "ManualBucketingManager",
    "get_bucketing_manager",
]


def get_bucketing_manager(
    strategy: str,
    *,
    max_batch_size: int,
    min_batch_size: int = 1,
    limit: int = 1,
    step: int = 2,
    manual_buckets: list[int] = [],
) -> type[RBLNBucketingManager]:
    """Create a bucketing manager for the given strategy.

    Caller can pass all possible args; only the ones required by the
    selected strategy are forwarded to the manager class.
    """
    if strategy == "exponential":
        return ExponentialBucketingManager(
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            limit=limit,
            step=step,
        )
    elif strategy == "linear":
        return LinearBucketingManager(
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            limit=limit,
            step=step,
        )
    elif strategy == "manual":
        return ManualBucketingManager(
            max_batch_size=max_batch_size,
            manual_buckets=manual_buckets,
        )
    else:
        raise ValueError(
            f"Invalid bucketing strategy: {strategy}. "
            "Valid strategies are [exponential, exp, linear, manual].",
        )
