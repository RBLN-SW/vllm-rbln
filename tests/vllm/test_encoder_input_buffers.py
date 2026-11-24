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

from unittest.mock import Mock

import pytest
import torch

from vllm_rbln.utils import (
    EncoderInput,
    EncoderInputBuffers,
    encoder_input_buffer_sizes,
)


def _specs(bucket_size):
    return [
        EncoderInput("patches", (bucket_size, 4), axis=0, pad=0.0),
        EncoderInput("cos", (bucket_size, 2), axis=0, pad=1.0),
        EncoderInput(
            "key_bias",
            (1, 1, bucket_size),
            axis=2,
            pad=-30000.0,
            allow_prefix_copy=False,
        ),
    ]


def _key_bias(num_valid):
    bias = torch.zeros(1, 1, 8)
    bias[..., num_valid:] = -30000.0
    return bias


def test_estimated_input_sizes_match_allocations():
    def specs(bucket):
        return [
            EncoderInput(
                "patches",
                (bucket, 4),
                axis=0,
                pad=0.0,
                dtype=torch.float32,
                register_host=True,
            ),
            EncoderInput("key_bias", (1, 1, bucket), axis=2, pad=0.0),
        ]

    buckets = (8, 16)
    dtype = torch.bfloat16
    buffers = EncoderInputBuffers(
        "cpu", dtype, specs_by_bucket={bucket: specs(bucket) for bucket in buckets}
    )
    assert encoder_input_buffer_sizes(buckets, specs, dtype) == [
        tensor.nbytes
        for bucket in buckets
        for tensor in buffers.warmup_inputs(bucket, fill_overrides={})
    ]


def test_rbln_patch_buffer_is_placed_on_chiplet_zero_once(monkeypatch):
    specs = [
        EncoderInput("patches", (8, 4), axis=0, pad=2.0, register_host=True),
        EncoderInput("key_bias", (1, 1, 8), axis=2, pad=0.0),
    ]
    patches = torch.empty(8, 4)
    key_bias = torch.empty(1, 1, 8)
    full = Mock(side_effect=[patches, key_bias])
    bind = Mock()
    monkeypatch.setattr(torch, "full", full)
    monkeypatch.setattr(torch.rbln, "bind_device_memory", bind)

    device = torch.device("rbln:0")
    buffers = EncoderInputBuffers(device, torch.float32, specs_by_bucket={8: specs})
    first = buffers.warmup_inputs(8, fill_overrides={})
    second = buffers.warmup_inputs(8, fill_overrides={})

    assert first is second
    assert first[0] is patches and first[1] is key_bias
    assert torch.all(patches == 2)
    assert full.call_count == 2
    bind.assert_called_once_with(patches)


def test_a_bucket_allocates_once_and_is_reused():
    buffers = EncoderInputBuffers("cpu", torch.bfloat16, specs_by_bucket={8: _specs(8)})

    first = buffers.stage(8, [torch.ones(5, 4), torch.zeros(5, 2), _key_bias(5)], 5)
    second = buffers.stage(8, [torch.ones(3, 4), torch.zeros(3, 2), _key_bias(3)], 3)

    assert [x.data_ptr() for x in first] == [x.data_ptr() for x in second]
    assert [x.shape for x in second] == [(8, 4), (8, 2), (1, 1, 8)]
    assert all(x.dtype == torch.bfloat16 for x in second)


@pytest.mark.parametrize("registered", [True, False])
def test_staging_converts_selected_inputs_in_the_registered_buffer(
    monkeypatch, registered
):
    spec = EncoderInput("patches", (8, 4), axis=0, pad=0.0, register_host=True)
    device = torch.device("rbln:0")
    buffers = EncoderInputBuffers(device, torch.bfloat16, specs_by_bucket={8: [spec]})
    destination = Mock()
    monkeypatch.setattr(
        buffers, "_get_or_create_bucket", lambda _: ([spec], [destination])
    )
    host = torch.empty((8, 4), dtype=torch.bfloat16)
    allocation = Mock(return_value=host.view(torch.uint8).flatten())
    registration = Mock(**{"covers.return_value": registered})
    register = Mock(return_value=registration)
    monkeypatch.setattr(torch.rbln, "huge_host_empty", allocation)
    monkeypatch.setattr(torch.rbln, "host_register_memory", register)

    full = torch.arange(32, dtype=torch.float32).view(8, 4)
    buffers.stage(8, [full], 8)
    (staged,) = destination.copy_.call_args.args
    assert staged.data_ptr() == (host if registered else full).data_ptr()
    torch.testing.assert_close(staged, full.to(staged.dtype))

    prefix = torch.ones((3, 4), dtype=torch.float32)
    buffers.stage(8, [prefix], 3)
    (staged,) = destination.narrow.return_value.copy_.call_args.args
    assert staged.data_ptr() == (host if registered else prefix).data_ptr()
    torch.testing.assert_close(staged, prefix.to(staged.dtype))
    if registered:
        assert staged.dtype == torch.bfloat16
        torch.testing.assert_close(host[3:], full[3:].to(torch.bfloat16))
    else:
        registration.close.assert_called_once_with()
    allocation.assert_called_once_with(host.nbytes)
    assert register.call_count == 1
    assert register.call_args.kwargs["device"] == device


def test_new_buffers_start_padded():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})

    patches, cos, key_bias = buffers.stage(
        8, [torch.empty(0, 4), torch.empty(0, 2), _key_bias(0)], 0
    )

    assert (patches == 0.0).all() and (cos == 1.0).all()
    assert (key_bias == -30000.0).all()


def test_staging_copies_a_prefix_and_leaves_the_padding_to_the_key_bias():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})
    buffers.stage(
        8, [torch.full((8, 4), 9.0), torch.full((8, 2), 9.0), _key_bias(8)], 8
    )

    patches, cos, key_bias = buffers.stage(
        8, [torch.full((3, 4), 2.0), torch.full((3, 2), 0.5), _key_bias(3)], 3
    )

    assert (patches[:3] == 2.0).all() and (cos[:3] == 0.5).all()
    # The larger item's entries stay: finite, and masked below.
    assert (patches[3:] == 9.0).all() and (cos[3:] == 9.0).all()
    assert (key_bias[..., :3] == 0.0).all() and (key_bias[..., 3:] == -30000.0).all()


def test_warmup_restores_defaults_and_clears_the_mask_in_place():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})
    staged = buffers.stage(8, [torch.ones(3, 4), torch.zeros(3, 2), _key_bias(3)], 3)

    patches, cos, key_bias = buffers.warmup_inputs(8, fill_overrides={"key_bias": 0.0})

    assert all(
        warmed is previous
        for warmed, previous in zip((patches, cos, key_bias), staged, strict=True)
    )
    assert (patches == 0.0).all() and (cos == 1.0).all() and (key_bias == 0.0).all()


def test_staging_rejects_a_short_key_bias_for_a_partial_bucket():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})
    patches, _, key_bias = buffers.stage(
        8, [torch.ones(8, 4), torch.ones(8, 2), _key_bias(8)], 8
    )
    bias = _key_bias(3)[..., :3]

    with pytest.raises(ValueError, match="'key_bias' requires a full 8-entry tensor"):
        buffers.stage(8, [torch.zeros(3, 4), torch.zeros(3, 2), bias], 3)

    assert (patches == 1.0).all() and (key_bias == 0.0).all()


def test_a_value_of_the_wrong_length_is_refused():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})
    with pytest.raises(AssertionError, match="4 entries"):
        buffers.stage(8, [torch.ones(4, 4), torch.ones(3, 2), _key_bias(3)], 3)


def test_the_value_is_converted_to_the_buffer_dtype():
    buffers = EncoderInputBuffers("cpu", torch.bfloat16, specs_by_bucket={8: _specs(8)})

    (patches, _, _) = buffers.stage(
        8,
        [
            torch.arange(20, dtype=torch.float32).view(5, 4),
            torch.ones(5, 2),
            _key_bias(5),
        ],
        5,
    )

    assert patches.dtype == torch.bfloat16
    torch.testing.assert_close(patches[:5].float(), torch.arange(20.0).view(5, 4))


def test_buffers_are_written_in_place_inside_and_outside_inference_mode():
    buffers = EncoderInputBuffers("cpu", torch.float32, specs_by_bucket={8: _specs(8)})
    with torch.inference_mode():
        inside = buffers.warmup_inputs(8, fill_overrides={"key_bias": 0.0})
    outside = buffers.stage(8, [torch.ones(2, 4), torch.ones(2, 2), _key_bias(2)], 2)
    with torch.inference_mode():
        again = buffers.warmup_inputs(8, fill_overrides={"key_bias": 0.0})

    assert inside[0].data_ptr() == outside[0].data_ptr() == again[0].data_ptr()
    assert not inside[0].is_inference()


def test_an_input_may_carry_its_own_dtype():
    def specs(bucket_size):
        return [
            EncoderInput(
                "positions",
                (1, bucket_size, 2),
                axis=1,
                pad=-1.0,
                dtype=torch.long,
            ),
            EncoderInput(
                "padding",
                (1, bucket_size),
                axis=1,
                pad=1.0,
                dtype=torch.bool,
            ),
            EncoderInput("patches", (bucket_size, 2), axis=0, pad=0.0),
        ]

    buffers = EncoderInputBuffers("cpu", torch.bfloat16, specs_by_bucket={8: specs(8)})
    positions, padding, patches = buffers.stage(
        8, [torch.empty(1, 0, 2), torch.empty(1, 0), torch.empty(0, 2)], 0
    )

    assert positions.dtype == torch.long and (positions == -1).all()
    assert padding.dtype == torch.bool and padding.all()
    assert patches.dtype == torch.bfloat16
