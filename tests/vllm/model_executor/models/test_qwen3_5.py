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

from types import SimpleNamespace
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
from vllm_rbln.config import RBLNConfig
from vllm_rbln.model_executor.models import qwen3_5
from vllm_rbln.model_executor.models.qwen3_5 import (
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5MultiModalProcessor,
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


@pytest.mark.parametrize(
    "limits", [{"image": 1}, {"image": 1, "video": 1}], ids=["unset", "set"]
)
def test_a_video_limit_above_zero_is_refused(limits):
    # An unset limit counts: upstream reads it as 999.
    vllm_config = make_vllm_config(model="Qwen/Qwen3.5-4B", limit_mm_per_prompt=limits)
    with pytest.raises(NotImplementedError, match="--limit-mm-per-prompt"):
        RBLNQwen3_5ForConditionalGeneration(vllm_config=vllm_config)


def _build_with_token_budgets(
    monkeypatch, budgets, *, max_tokens=256, prefill_chunk_size=512
):
    """The model with upstream's construction stubbed out: a largest image of
    `max_tokens` output tokens (1024 patches unless given) and the given budgets."""

    def init(self, *, vllm_config, prefix="model"):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(image_token_id=0)
        self.visual = SimpleNamespace(spatial_merge_size=MERGE)

    monkeypatch.setattr(qwen3_5.Qwen3_5ForConditionalGeneration, "__init__", init)
    info = SimpleNamespace(get_max_image_tokens=lambda: max_tokens)
    monkeypatch.setattr(
        qwen3_5.MULTIMODAL_REGISTRY, "get_processing_info", lambda model_config: info
    )
    limits = {"image": 1, "video": 0}
    mm_config = SimpleNamespace(get_limit_per_prompt=limits.get)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_multimodal_config=lambda: mm_config),
        additional_config=RBLNConfig(
            encoder_token_budgets=budgets, prefill_chunk_size=prefill_chunk_size
        ),
    )
    return RBLNQwen3_5ForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("budgets", "buckets"),
    [
        ([256, 64], [256, 1024]),
        ([64, 300], [256, 1200]),
    ],
)
def test_token_budgets_resolve_to_patch_buckets(monkeypatch, budgets, buckets):
    model = _build_with_token_budgets(monkeypatch, budgets)

    assert model.mm_encoder_patch_buckets == buckets


@pytest.mark.parametrize(
    ("prefill_chunk_size", "max_tokens", "buckets"),
    [
        # 128, 256 and 512 output tokens, then the processor's largest image.
        (128, 16384, [512, 1024, 2048, 65536]),
        # A longer prefill chunk extends the budgets to its own length.
        (1024, 16384, [512, 1024, 2048, 4096, 65536]),
        ([512, 1536], 16384, [512, 1024, 2048, 4096, 6144, 65536]),
        # Only the budgets below the largest image are kept.
        (512, 256, [512, 1024]),
        (512, 64, [256]),
    ],
)
def test_unset_token_budgets_reach_the_prefill_chunk_and_the_largest_image(
    monkeypatch, prefill_chunk_size, max_tokens, buckets
):
    model = _build_with_token_budgets(
        monkeypatch, [], max_tokens=max_tokens, prefill_chunk_size=prefill_chunk_size
    )

    assert model.mm_encoder_patch_buckets == buckets


def test_token_budgets_below_the_largest_image_are_refused(monkeypatch):
    # No bucket would hold the processor's largest image.
    with pytest.raises(ValueError, match="must reach 256 output tokens"):
        _build_with_token_budgets(monkeypatch, [64])


def test_a_request_image_above_the_start_up_limit_is_refused(monkeypatch):
    # The engine starts with a limit of 256 output tokens: 1024 patches.
    processor = RBLNQwen3_5MultiModalProcessor.__new__(RBLNQwen3_5MultiModalProcessor)
    processor.info = SimpleNamespace(
        get_max_image_tokens=lambda: 256,
        get_hf_config=lambda: SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=MERGE)
        ),
    )
    outputs = {
        "pixel_values": torch.zeros(1),
        "image_grid_thw": torch.tensor([[1, 32, 32]]),
    }
    monkeypatch.setattr(
        qwen3_5.Qwen3VLMultiModalProcessor,
        "_apply_hf_processor_main",
        lambda self, mm_items, hf_kwargs: outputs,
    )

    assert processor._apply_hf_processor_main(None, {}) is outputs

    # A request that raised max_pixels: 34 x 32 = 1088 patches.
    grids = torch.tensor([[1, 32, 32], [1, 34, 32]])
    outputs = {"pixel_values": torch.zeros(1), "image_grid_thw": grids}
    with pytest.raises(ValueError, match="1088 patches, more than the 1024"):
        processor._apply_hf_processor_main(None, {})

    # Sent as embeddings, the same grids skip the encoder and are let through.
    outputs = {"image_embeds": torch.zeros(1), "image_grid_thw": grids}
    assert processor._apply_hf_processor_main(None, {}) is outputs


def test_warmup_runs_the_encoder_once_over_each_bucket(visual):
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.visual = visual
    model.has_vision = True
    model.mm_encoder_patch_buckets = [64, 128]
    encoder = Mock(return_value=torch.zeros(1, 32))
    model._mm_encoder = encoder

    model.warmup_mm_encoder()

    assert encoder.call_count == 2
    # Warm-up compiles the shapes a request's image arrives in: the inputs it
    # builds match what _encoder_inputs returns for an image of that bucket.
    for call, grid in zip(encoder.call_args_list, [[1, 8, 8], [1, 8, 16]]):
        pixels = torch.zeros(grid[1] * grid[2], 3 * TEMPORAL * PATCH * PATCH)
        request = model._encoder_inputs(pixels, grid)
        assert [(x.shape, x.dtype, x.device) for x in call.args] == [
            (x.shape, x.dtype, x.device) for x in request
        ]
        assert not call.args[4].any()

    model.has_vision = False
    model.warmup_mm_encoder()
    assert encoder.call_count == 2


@pytest.mark.parametrize(
    ("grid", "bucket"), [([1, 6, 4], 64), ([1, 8, 8], 64), ([1, 8, 12], 128)]
)
def test_an_image_takes_the_smallest_bucket_that_holds_it(visual, grid, bucket):
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.visual = visual
    model.mm_encoder_patch_buckets = [64, 128]
    num_patches = grid[0] * grid[1] * grid[2]
    pixels = torch.zeros(num_patches, 3 * TEMPORAL * PATCH * PATCH)

    patches, _, _, _, key_bias = model._encoder_inputs(pixels, grid)

    assert patches.shape[0] == bucket
    assert int((key_bias == 0).sum()) == num_patches


def test_image_embeddings_match_upstream(visual):
    # Two images of different grids, each smaller than its bucket (128 and 64
    # patches), so the per-image split, the bucket choice, the padding and the
    # key mask all take part.
    grids = [[1, 8, 12], [1, 6, 4]]
    sizes = [t * h * w for t, h, w in grids]
    pixels = torch.randn(sum(sizes), 3 * TEMPORAL * PATCH * PATCH)
    model = RBLNQwen3_5ForConditionalGeneration.__new__(
        RBLNQwen3_5ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.visual = visual
    model.has_vision = True
    model.mm_encoder_patch_buckets = [64, 128]
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
