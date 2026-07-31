# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The resized KV cache must still hold one max-length request.

Upstream's `check_enough_kv_cache_memory` runs against the pre-compile estimate,
inside `get_kv_cache_configs`. Nothing re-checks the number the dynamic-KV path
substitutes, so without this guard a profile returning too few blocks yields a
server that starts and then rejects every request -- where dev dies at start-up
with an actionable message.
"""

from types import SimpleNamespace

import pytest

from vllm_rbln.patches.dynamic_kv import assert_kv_cache_fits_one_request


def _config(block_size, max_model_len):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def _kv(num_blocks):
    return SimpleNamespace(num_blocks=num_blocks)


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks"),
    [
        (1024, 32768, 32),  # exactly one request
        (1024, 32768, 33),  # one to spare
        (8192, 32768, 4),  # exactly one request, larger blocks
        (1024, 32768, 1548),  # a real measured answer
        (128, 1000, 8),  # cdiv rounds up: 1000/128 -> 8
    ],
)
def test_accepts_a_pool_that_fits(block_size, max_model_len, num_blocks):
    assert_kv_cache_fits_one_request(
        _config(block_size, max_model_len), _kv(num_blocks)
    )


@pytest.mark.parametrize(
    ("block_size", "max_model_len", "num_blocks", "needed"),
    [
        (1024, 32768, 31, 32),  # one block short
        (1024, 32768, 1, 32),
        (8192, 32768, 3, 4),
        (128, 1000, 7, 8),  # the rounded-up block is required
    ],
)
def test_rejects_a_pool_that_cannot_hold_one_request(
    block_size, max_model_len, num_blocks, needed
):
    with pytest.raises(ValueError, match=f"needs {needed}") as excinfo:
        assert_kv_cache_fits_one_request(
            _config(block_size, max_model_len), _kv(num_blocks)
        )
    # The message has to be actionable, like the upstream one it restores.
    assert str(num_blocks) in str(excinfo.value)
    assert "max_model_len" in str(excinfo.value)


@pytest.mark.parametrize("block_size,max_model_len", [(0, 32768), (1024, 0), (0, 0)])
def test_degenerate_config_is_skipped_not_crashed(block_size, max_model_len):
    """A missing block_size / max_model_len is not this guard's error to raise."""
    assert_kv_cache_fits_one_request(_config(block_size, max_model_len), _kv(1))
