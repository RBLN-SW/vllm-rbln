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
import torch

from vllm_rbln.distributed.ec_transfer.ec_connector.rbln_ec_nixl_connector import (
    _DTYPE_SIZES,
    _dtype_size,
    _merge_pull_result,
)


def test_dtype_size_known_dtypes():
    assert _dtype_size("torch.float16") == 2
    assert _dtype_size("torch.bfloat16") == 2
    assert _dtype_size("torch.float32") == 4
    assert _dtype_size("torch.int64") == 8
    assert _dtype_size("torch.int32") == 4


def test_dtype_size_covers_registered_table():
    # Keep _DTYPE_SIZES and _dtype_size in lockstep — if a new dtype is
    # added to the table, this test forces the lookup to stay wired up.
    for dtype_str, expected in _DTYPE_SIZES.items():
        assert _dtype_size(dtype_str) == expected


def test_dtype_size_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _dtype_size("torch.float64")


def test_merge_pull_result_no_non_tensor_returns_tensor_copy():
    t = torch.zeros(2)
    bufs = {"image_embeds": t}

    result = _merge_pull_result(bufs, None)

    assert result == {"image_embeds": t}
    # Caller must not observe the same dict instance, otherwise later
    # registry pops would mutate the encoder cache.
    assert result is not bufs


def test_merge_pull_result_passes_primitive_non_tensor_through():
    # Qwen2.5-VL video path: second_per_grid_ts is a primitive that
    # rides alongside the encoder-output tensors.
    t = torch.zeros(1)
    bufs = {"video_embeds": t}
    non_tensor = {"second_per_grid_ts": 0.5}

    result = _merge_pull_result(bufs, non_tensor)

    assert result["video_embeds"] is t
    assert result["second_per_grid_ts"] == 0.5


def test_merge_pull_result_reconstructs_list_of_tensors():
    # Qwen3-VL deepstack path: the encoder emits a list of tensors that
    # save_caches flattens to `key.0`, `key.1`, ... with a `_seq_meta.key`
    # side-channel. _merge_pull_result must rebuild the original list.
    t0, t1, t2 = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)
    bufs = {
        "image_embeds": torch.zeros(4),
        "deepstack_image_embeds.0": t0,
        "deepstack_image_embeds.1": t1,
        "deepstack_image_embeds.2": t2,
    }
    non_tensor = {
        "_seq_meta.deepstack_image_embeds": {"length": 3, "is_tuple": False},
    }

    result = _merge_pull_result(bufs, non_tensor)

    assert "image_embeds" in result
    assert isinstance(result["deepstack_image_embeds"], list)
    assert result["deepstack_image_embeds"] == [t0, t1, t2]


def test_merge_pull_result_reconstructs_tuple_of_tensors():
    t0, t1 = torch.zeros(1), torch.ones(1)
    bufs = {
        "visual_out.0": t0,
        "visual_out.1": t1,
    }
    non_tensor = {
        "_seq_meta.visual_out": {"length": 2, "is_tuple": True},
    }

    result = _merge_pull_result(bufs, non_tensor)

    assert isinstance(result["visual_out"], tuple)
    assert result["visual_out"] == (t0, t1)
