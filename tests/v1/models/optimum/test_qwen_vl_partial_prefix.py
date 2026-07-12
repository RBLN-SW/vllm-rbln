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

import types

import torch

from vllm_rbln.model_executor.models.optimum.qwen3_vl import (
    RBLNOptimumQwen3VLForConditionalGeneration as Qwen3VL,
)
from vllm_rbln.model_executor.models.optimum.qwen_vl import (
    RBLNOptimumQwenVLForConditionalGeneration as QwenVL,
)
from vllm_rbln.v1.worker.optimum_model_runner import RBLNOptimumModelRunner

HIDDEN = 4


def _feature(offset, length, modality="image", data="x", is_embed=None):
    """Minimal stand-in for a scheduled multimodal feature."""
    if is_embed is not None:
        is_embed = torch.tensor(is_embed, dtype=torch.bool)
    return types.SimpleNamespace(
        data=data,
        modality=modality,
        mm_position=types.SimpleNamespace(
            offset=offset, length=length, is_embed=is_embed
        ),
    )


def _scheduler_output(features):
    return types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(mm_features=features)]
    )


def _mm_embed_tail_starts(features, num_cached):
    """Call the runner method with a lightweight fake ``self``."""
    fake = types.SimpleNamespace(is_multimodal_raw_input_only_model=True)
    fake._iter_kept_mm_features = types.MethodType(
        RBLNOptimumModelRunner._iter_kept_mm_features, fake
    )
    return RBLNOptimumModelRunner._mm_embed_tail_starts(
        fake, _scheduler_output(features), num_cached
    )


def _fake_qwen(visual):
    """Fake ``self`` carrying the pieces ``_encode_and_slice_mm`` touches."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(visual=visual),
        _mm_feature_counts=QwenVL._mm_feature_counts,
        _slice_to_tail=QwenVL._slice_to_tail,
    )


class TestMmEmbedTailStarts:
    def test_split_image_and_uncached_image(self):
        # imgA pads [15, 411) is split by the boundary at 384 -> its tail starts
        # at feature 384-15=369; imgB [413, ...) is fully uncached -> starts at 0.
        starts = _mm_embed_tail_starts(
            [_feature(15, 396), _feature(413, 510)], num_cached=384
        )
        assert starts == {"image": [369, 0]}

    def test_fully_cached_item_is_dropped(self):
        # imgA ends at 15+396=411; a boundary at/after 411 fully caches it, so it
        # is not kept (no encoder run, no tail features).
        assert _mm_embed_tail_starts([_feature(15, 396)], num_cached=411) == {}

    def test_features_without_data_are_skipped(self):
        starts = _mm_embed_tail_starts(
            [_feature(15, 396, data=None), _feature(413, 510)], num_cached=384
        )
        assert starts == {"image": [0]}

    def test_is_embed_maps_token_boundary_to_feature_index(self):
        # idefics3-style block at offset 10 interleaving structural (F) and image
        # (T) tokens. A boundary 3 tokens in caches [F, T, T] -> 2 embedding
        # tokens, so the tail starts at feature index 2, not raw token offset 3.
        starts = _mm_embed_tail_starts(
            [_feature(10, 7, is_embed=[0, 1, 1, 0, 1, 1, 1])], num_cached=13
        )
        assert starts == {"image": [2]}

    def test_is_embed_leading_structural_tokens_start_at_zero(self):
        # Only a leading structural (non-embedding) token is cached, so no image
        # feature is cached yet and the whole image is re-injected from 0.
        starts = _mm_embed_tail_starts(
            [_feature(10, 7, is_embed=[0, 1, 1, 0, 1, 1, 1])], num_cached=11
        )
        assert starts == {"image": [0]}


class TestSliceToTail:
    def test_slices_each_item_to_its_tail(self):
        # Two items of 3 and 4 features; keep item0[1:] and item1[0:].
        feats = torch.arange(7 * HIDDEN, dtype=torch.float32).reshape(7, HIDDEN)
        out = QwenVL._slice_to_tail(feats, counts=[3, 4], tail_starts=[1, 0])
        assert out.shape[0] == (3 - 1) + (4 - 0)
        assert torch.equal(out[0], feats[1])  # item0's first kept feature
        assert torch.equal(out[2], feats[3])  # item1 starts right after item0


class TestMmFeatureCounts:
    def test_counts_are_prod_over_merge_squared(self):
        grid = torch.tensor([[1, 36, 44], [1, 30, 68]])
        counts = QwenVL._mm_feature_counts(grid, merge_size=2).tolist()
        assert counts == [1584 // 4, 2040 // 4]  # [396, 510]


class TestEncodeAndSliceMm:
    GRID = torch.tensor([[1, 36, 44], [1, 30, 68]])  # counts [396, 510], merge 2
    TOTAL = 396 + 510

    def _feats(self):
        return torch.arange(self.TOTAL * HIDDEN, dtype=torch.float32).reshape(
            self.TOTAL, HIDDEN
        )

    def test_base_returns_no_deepstack(self):
        feats = self._feats()
        fake = _fake_qwen(lambda pv, grid_thw=None: feats)
        tail_feats, deepstack = QwenVL._encode_and_slice_mm(
            fake, pixel_values=None, grid_thw=self.GRID, tail_starts=[369, 0], merge=2
        )
        assert deepstack is None
        assert tail_feats.shape[0] == (396 - 369) + 510
        assert torch.equal(tail_feats[0], feats[369])
        assert torch.equal(tail_feats[27], feats[396])  # imgB's first feature

    def test_qwen3_slices_deepstack_per_layer(self):
        feats = self._feats()
        deepstack_layers = [feats + (i + 1) * 100_000 for i in range(3)]
        fake = _fake_qwen(lambda pv, grid_thw=None: (feats, deepstack_layers))
        tail_feats, tail_deepstack = Qwen3VL._encode_and_slice_mm(
            fake, pixel_values=None, grid_thw=self.GRID, tail_starts=[369, 0], merge=2
        )
        expected_rows = (396 - 369) + 510
        assert tail_feats.shape[0] == expected_rows
        assert len(tail_deepstack) == 3
        assert all(layer.shape[0] == expected_rows for layer in tail_deepstack)
        # deepstack is sliced with the same boundaries as the main features.
        assert torch.equal(tail_deepstack[0][0], deepstack_layers[0][369])
