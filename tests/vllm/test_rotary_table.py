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

import torch

from vllm_rbln.utils import RotaryTable


def _direct(positions: torch.Tensor, inv_freq: torch.Tensor):
    freqs = torch.outer(positions.float(), inv_freq)
    return freqs.cos(), freqs.sin()


def test_rows_match_the_direct_angles():
    inv_freq = 1.0 / (100.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    table = RotaryTable(inv_freq, num_positions=4)

    cos, sin = table.rows(3)
    expected_cos, expected_sin = _direct(torch.arange(3), inv_freq)

    assert cos.shape == (3, 4)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_table_grows_to_the_longest_position_asked():
    inv_freq = torch.tensor([1.0, 0.5])
    table = RotaryTable(inv_freq, num_positions=4)

    cos, sin = table.rows(10)
    expected_cos, expected_sin = _direct(torch.arange(10), inv_freq)

    assert table.cos.shape[0] >= 10
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)
    # A shorter ask after growth reads the same table.
    assert table.rows(4)[0].shape == (4, 2)


def test_indexed_rows_rebuild_a_frame_in_any_order():
    inv_freq = torch.tensor([1.0, 0.25, 0.0625])
    table = RotaryTable(inv_freq)
    ids = torch.tensor([3, 0, 7, 7, 1])

    cos, sin = table.rows(8)
    expected_cos, expected_sin = _direct(ids, inv_freq)

    torch.testing.assert_close(cos[ids], expected_cos)
    torch.testing.assert_close(sin[ids], expected_sin)


def test_tables_are_float32_on_the_host_whatever_the_source():
    inv_freq = torch.tensor([1.0, 0.5], dtype=torch.bfloat16)
    with torch.device("meta"):
        table = RotaryTable(inv_freq, num_positions=2)
        cos, sin = table.rows(3)

    assert table.inv_freq.dtype == torch.float32
    expected_cos, expected_sin = _direct(torch.arange(3), inv_freq.float())
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)
