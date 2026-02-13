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


class LinearBucketingManager(RBLNBucketingManager):
    """Generates decode buckets by decreasing size with a fixed step."""

    def __init__(
        self,
        max_batch_size: int,
        min_batch_size: int,
        limit: int,
        step: int,
    ):
        RBLNBucketingManager.check_config(
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            limit=limit,
            step=step,
        )
        self.min_batch_size = min_batch_size
        self.limit = limit
        self.step = step
        super().__init__(max_batch_size=max_batch_size)

    def _build_decode_buckets(self) -> None:
        """Fill buckets with evenly spaced sizes."""
        buckets = [self.max_batch_size]
        while len(buckets) < self.limit:
            candidate = buckets[-1] - self.step
            if candidate < self.min_batch_size:
                break
            buckets.append(candidate)
        self.decode_batch_buckets = sorted(set(buckets))