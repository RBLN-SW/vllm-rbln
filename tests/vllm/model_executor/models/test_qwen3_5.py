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

"""The RBLN Qwen3.5 image path (one padded image at a time, full attention with
the padding masked) tracks upstream's vision tower on the same weights."""

from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.model_executor.models.qwen3_5 import (
    RBLNQwen3_5ForConditionalGeneration,
)

PATCH, MERGE, TEMPORAL = 16, 2, 2


@pytest.fixture
def visual(tmp_path):
    with set_current_vllm_config(make_vllm_config()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path}/dist",
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
        try:
            config = Qwen3_5VisionConfig(
                depth=2,
                hidden_size=64,
                intermediate_size=128,
                num_heads=4,
                patch_size=PATCH,
                spatial_merge_size=MERGE,
                temporal_patch_size=TEMPORAL,
                num_position_embeddings=64,
                out_hidden_size=32,
                deepstack_visual_indexes=[],
                hidden_act="gelu_pytorch_tanh",
            )
            torch.manual_seed(0)
            tower = Qwen3_VisionTransformer(config, norm_eps=1e-6).float()
            for param in tower.parameters():
                param.data.normal_(0, 0.05)
            yield tower
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_warmup_runs_the_encoder_once_over_the_whole_bucket(visual):
    # The one bucket is the processor's largest image, so the warm-up input is
    # that whole grid with no padded rows.
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.visual = visual
    model.has_vision = True
    model.encoder_grid = [1, 8, 16]
    model.encoder_patches = 128
    encoder = Mock(return_value=torch.zeros(128 // (MERGE * MERGE), 32))
    model._mm_encoder = encoder

    model.warmup_mm_encoder()

    encoder.assert_called_once()
    patches, pos_embeds, cos, sin, key_bias = encoder.call_args.args
    assert patches.shape == (128, 3 * TEMPORAL * PATCH * PATCH)
    assert pos_embeds.shape[0] == cos.shape[0] == sin.shape[0] == 128
    assert key_bias.shape == (1, 1, 128) and not key_bias.any()

    model.has_vision = False
    model.warmup_mm_encoder()
    encoder.assert_called_once()


def test_image_embeddings_match_upstream(visual):
    # Two images of different grids, each smaller than the bucket, so the
    # per-image split, the padding and the key mask all take part.
    grids = [[1, 8, 12], [1, 6, 4]]
    sizes = [t * h * w for t, h, w in grids]
    pixels = torch.randn(sum(sizes), 3 * TEMPORAL * PATCH * PATCH)
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.visual = visual
    model.encoder_patches = 128
    model._mm_encoder = None

    with torch.inference_mode():
        expected = visual(pixels, grid_thw=grids).split(
            [n // (MERGE * MERGE) for n in sizes]
        )
        embeds = model._process_image_input(
            {
                "type": "pixel_values",
                "pixel_values": pixels,
                "image_grid_thw": torch.tensor(grids),
            }
        )

    assert len(embeds) == len(expected)
    for got, want in zip(embeds, expected):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
