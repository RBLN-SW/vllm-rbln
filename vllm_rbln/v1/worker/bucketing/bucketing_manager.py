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

from abc import ABC, abstractmethod


class RBLNBucketingManager(ABC):
    """Abstract base for defining decode bucket construction strategies."""

    def __init__(
        self,
        max_batch_size: int,
    ):
        self.max_batch_size = max_batch_size

        # build the decode buckets
        self.decode_batch_buckets: list[int]
        self._build_decode_buckets()

        # 1 is reserved for prefill.
        self._batch_buckets = sorted({1, *self.decode_batch_buckets})

    @abstractmethod
    def _build_decode_buckets(self):
        """Build the decode buckets."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def batch_buckets(self) -> list[int]:
        """List of supported batch sizes including the prefill and decode."""
        return self._batch_buckets

    def find_decode_batch_bucket(self, batch_size: int) -> int:
        """Return the smallest decode bucket that is >= `batch_size`."""
        for bucket in self.decode_batch_buckets:
            if bucket >= batch_size:
                return bucket
        raise ValueError(
            "No batch bucket found for batch size %d, "
            "batch buckets: %s", batch_size, self.batch_buckets)

    @staticmethod
    def check_config(
        max_batch_size: int,
        min_batch_size: int,
        limit: int,
        step: int,
    ) -> None:
        """Check if the config is valid."""
        if max_batch_size < min_batch_size:
            raise ValueError(
                "max_batch_size must be >= min_batch_size, "
                f"max_batch_size: {max_batch_size}, "
                f"min_batch_size: {min_batch_size}", )
        if limit <= 0:
            raise ValueError(
                f"limit must be greater than 0, limit: {limit}")
        if step <= 0:
            raise ValueError(f"step must be greater than 0, step: {step}")
        if min_batch_size <= 0:
            raise ValueError(
                "min_batch_size must be greater than 0, "
                f"min_batch_size: {min_batch_size}", )