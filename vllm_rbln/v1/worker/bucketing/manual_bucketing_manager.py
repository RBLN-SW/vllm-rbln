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


class ManualBucketingManager(RBLNBucketingManager):
    """Generates decode buckets from the given manual bucket list."""

    def __init__(
        self,
        max_batch_size: int,
        manual_buckets: list[int],
    ):
        assert len(manual_buckets) > 0, "manual_buckets must be non-empty"
        self.manual_buckets = manual_buckets
        super().__init__(
            max_batch_size=max_batch_size,
        )

    def _build_decode_buckets(self) -> None:
        """Use the manual buckets."""
        buckets = sorted(set(self.manual_buckets))

        if buckets[-1] != self.max_batch_size:
            raise ValueError(
                "The last manual bucket must be equal to the max batch size, "
                f"max batch size: {self.max_batch_size}, "
                f"last manual bucket: {buckets[-1]}"
            )
        self.decode_batch_buckets = buckets


