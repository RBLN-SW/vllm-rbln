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

"""Unit tests for ``decode_batch_size`` -- the single source of truth for the
per-PP-stage (compiled) decode batch, shared by the runner's bucketing and the
scheduler's decode admission cap.
"""

import pytest

from vllm_rbln.v1.core.utils import decode_batch_size


@pytest.mark.parametrize(
    ("max_num_seqs", "pp_size", "expected"),
    [
        (16, 1, 16),  # non-PP: unchanged
        (16, 2, 8),
        (16, 4, 4),
        (2, 2, 1),  # small batch splits down to 1 per stage
        (4, 2, 2),
    ],
)
def test_decode_batch_size_divides_by_pp(max_num_seqs, pp_size, expected):
    assert decode_batch_size(max_num_seqs, pp_size) == expected


def test_decode_batch_size_pp1_is_identity():
    """pp_size == 1 (non-PP) returns max_num_seqs unchanged."""
    for n in (1, 2, 8, 37, 256):
        assert decode_batch_size(n, 1) == n
