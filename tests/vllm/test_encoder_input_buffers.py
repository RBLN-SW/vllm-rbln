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
"""The per-bucket staging buffers a vision encoder's inputs are copied into."""

import pytest
import torch

from vllm_rbln.utils import EncoderInput, EncoderInputBuffers

SPECS = [
    EncoderInput((8, 4), 0, 0.0, 0.0),
    EncoderInput((8, 2), 0, 1.0, 1.0),
    EncoderInput((1, 1, 8), 2, 0.0, -30000.0),
]


def _key_bias(num_valid):
    bias = torch.zeros(1, 1, 8)
    bias[..., num_valid:] = -30000.0
    return bias


def test_a_bucket_allocates_once_and_is_reused():
    buffers = EncoderInputBuffers("cpu", torch.bfloat16)

    first = buffers.stage(
        8, SPECS, [torch.ones(5, 4), torch.zeros(5, 2), _key_bias(5)], 5
    )
    second = buffers.stage(
        8, SPECS, [torch.ones(3, 4), torch.zeros(3, 2), _key_bias(3)], 3
    )

    assert [x.data_ptr() for x in first] == [x.data_ptr() for x in second]
    assert [x.shape for x in second] == [(8, 4), (8, 2), (1, 1, 8)]
    assert all(x.dtype == torch.bfloat16 for x in second)


def test_new_buffers_start_padded():
    buffers = EncoderInputBuffers("cpu", torch.float32)

    patches, cos, key_bias = buffers.get(8, SPECS)

    assert (patches == 0.0).all() and (cos == 1.0).all()
    assert (key_bias == -30000.0).all()


def test_staging_copies_a_prefix_and_leaves_the_padding_to_the_key_bias():
    buffers = EncoderInputBuffers("cpu", torch.float32)
    buffers.stage(
        8, SPECS, [torch.full((8, 4), 9.0), torch.full((8, 2), 9.0), _key_bias(8)], 8
    )

    patches, cos, key_bias = buffers.stage(
        8, SPECS, [torch.full((3, 4), 2.0), torch.full((3, 2), 0.5), _key_bias(3)], 3
    )

    assert (patches[:3] == 2.0).all() and (cos[:3] == 0.5).all()
    # The larger item's entries stay: finite, and masked below.
    assert (patches[3:] == 9.0).all() and (cos[3:] == 9.0).all()
    assert (key_bias[..., :3] == 0.0).all() and (key_bias[..., 3:] == -30000.0).all()


def test_a_none_value_fills_every_entry_with_the_valid_value():
    buffers = EncoderInputBuffers("cpu", torch.float32)

    patches, cos, key_bias = buffers.stage(8, SPECS, [None, None, None], 8)

    assert (patches == 0.0).all() and (cos == 1.0).all() and (key_bias == 0.0).all()


def test_a_value_of_the_wrong_length_is_refused():
    buffers = EncoderInputBuffers("cpu", torch.float32)
    with pytest.raises(AssertionError, match="4 entries"):
        buffers.stage(8, SPECS, [torch.ones(4, 4), None, None], 3)


def test_the_value_is_converted_to_the_buffer_dtype():
    buffers = EncoderInputBuffers("cpu", torch.bfloat16)

    (patches, _, _) = buffers.stage(
        8, SPECS, [torch.arange(20, dtype=torch.float32).view(5, 4), None, None], 5
    )

    assert patches.dtype == torch.bfloat16
    torch.testing.assert_close(patches[:5].float(), torch.arange(20.0).view(5, 4))


def test_buffers_are_written_in_place_inside_and_outside_inference_mode():
    buffers = EncoderInputBuffers("cpu", torch.float32)
    with torch.inference_mode():
        inside = buffers.stage(8, SPECS, [None, None, None], 8)
    outside = buffers.stage(8, SPECS, [torch.ones(2, 4), None, None], 2)
    with torch.inference_mode():
        again = buffers.stage(8, SPECS, [None, None, None], 4)

    assert inside[0].data_ptr() == outside[0].data_ptr() == again[0].data_ptr()
    assert not inside[0].is_inference()


def test_an_input_may_carry_its_own_dtype():
    buffers = EncoderInputBuffers("cpu", torch.bfloat16)
    specs = [
        EncoderInput((1, 8, 2), 1, -1.0, -1.0, torch.long),
        EncoderInput((1, 8), 1, 0.0, 1.0, torch.bool),
        EncoderInput((8, 2), 0, 0.0, 0.0),
    ]

    positions, padding, patches = buffers.get(8, specs)

    assert positions.dtype == torch.long and (positions == -1).all()
    assert padding.dtype == torch.bool and padding.all()
    assert patches.dtype == torch.bfloat16
