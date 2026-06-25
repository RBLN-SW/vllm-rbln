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

"""Unit tests for the gemma3/gemma4 chunked-prefill block-padding logic in
``RBLNKVCacheManager`` (``_chunked_prefill_pad`` and its helpers).

The manager's real constructor wires up a coordinator/block pool, so these tests
build a bare instance via ``object.__new__`` and set only the attributes the
padding methods read.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.v1.core.optimum_kv_cache_manager import RBLNKVCacheManager


def _make_manager(
    prefill_chunk_size: int,
    attn_block_size: int,
    image_prefill_chunk_sizes: list[int] | None = None,
) -> RBLNKVCacheManager:
    mgr = object.__new__(RBLNKVCacheManager)
    mgr.prefill_chunk_size = prefill_chunk_size
    mgr.attn_block_size = attn_block_size
    mgr.image_prefill_chunk_sizes = image_prefill_chunk_sizes
    return mgr


def _feature(offset: int, length: int, is_embed):
    """Build a fake mm_feature with a PlaceholderRange-like mm_position.

    `is_embed` is a list[bool] (becomes a bool tensor) or None.
    """
    embed = None if is_embed is None else torch.tensor(is_embed, dtype=torch.bool)
    pos = SimpleNamespace(offset=offset, length=length, is_embed=embed)
    return SimpleNamespace(mm_position=pos)


def _request(*features):
    return SimpleNamespace(mm_features=list(features))


def _image(offset: int, num_tokens: int, *, lead: int = 0, trail: int = 0):
    """An image placeholder of `num_tokens` embeds, optionally wrapped by `lead`
    and `trail` non-embed special tokens (boi/eoi/\\n\\n), mirroring gemma3/4.
    """
    mask = [False] * lead + [True] * num_tokens + [False] * trail
    return _feature(offset, len(mask), mask)


class TestImageChunkSize:
    def test_smallest_bucket_that_fits(self):
        mgr = _make_manager(128, 4096, image_prefill_chunk_sizes=[1152, 640, 384])
        assert mgr._image_chunk_size(256) == 384
        assert mgr._image_chunk_size(384) == 384
        assert mgr._image_chunk_size(385) == 640
        assert mgr._image_chunk_size(640) == 640
        assert mgr._image_chunk_size(1000) == 1152

    def test_run_exceeds_all_buckets_raises(self):
        # buckets are descending; a run larger than the largest bucket is invalid.
        mgr = _make_manager(128, 4096, image_prefill_chunk_sizes=[1152, 640, 384])
        with pytest.raises(ValueError):
            mgr._image_chunk_size(2000)

    def test_no_buckets_falls_back_to_text_chunk(self):
        mgr = _make_manager(256, 4096, image_prefill_chunk_sizes=None)
        assert mgr._image_chunk_size(256) == 256
        mgr_empty = _make_manager(256, 4096, image_prefill_chunk_sizes=[])
        assert mgr_empty._image_chunk_size(256) == 256

    def test_gemma3_single_bucket(self):
        mgr = _make_manager(256, 4096, image_prefill_chunk_sizes=[256])
        assert mgr._image_chunk_size(256) == 256


class TestImageEmbedSegments:
    def test_pure_embed_placeholder(self):
        mgr = _make_manager(256, 4096)
        req = _request(_feature(10, 4, [True, True, True, True]))
        assert mgr._image_embed_segments(req, query_len=100) == [(10, 14)]

    def test_special_tokens_excluded(self):
        # gemma3: \n\n boi [embeds] eoi \n\n -> only the embeds are the run.
        mgr = _make_manager(256, 4096)
        req = _request(_image(4, 256, lead=2, trail=2))  # placeholder [4, 264)
        assert mgr._image_embed_segments(req, query_len=1000) == [(6, 262)]

    def test_multiple_features_sorted(self):
        mgr = _make_manager(256, 4096)
        req = _request(
            _image(100, 4, lead=1, trail=1),  # embeds [101, 105)
            _image(0, 4, lead=1, trail=1),  # embeds [1, 5)
        )
        assert mgr._image_embed_segments(req, query_len=200) == [(1, 5), (101, 105)]

    def test_multiple_embed_runs_in_one_placeholder(self):
        mgr = _make_manager(256, 4096)
        req = _request(_feature(0, 5, [True, True, False, True, True]))
        assert mgr._image_embed_segments(req, query_len=100) == [(0, 2), (3, 5)]

    def test_clamped_to_query_len(self):
        mgr = _make_manager(256, 4096)
        # embeds at [2, 10); query_len cuts it at 6.
        req = _request(_feature(2, 8, [True] * 8))
        assert mgr._image_embed_segments(req, query_len=6) == [(2, 6)]

    def test_segment_starting_beyond_query_len_dropped(self):
        mgr = _make_manager(256, 4096)
        req = _request(_feature(50, 4, [True] * 4))
        assert mgr._image_embed_segments(req, query_len=10) == []

    def test_none_is_embed_rejected(self):
        # gemma3/4 placeholders always carry an is_embed mask; None is unexpected.
        mgr = _make_manager(256, 4096)
        req = _request(_feature(0, 4, None))
        with pytest.raises(AssertionError):
            mgr._image_embed_segments(req, query_len=100)


class TestChunkedPrefillPadTextOnly:
    def test_block_multiple_of_chunk_no_padding(self):
        # 4096 = 16 * 256, so chunks never straddle a block boundary.
        mgr = _make_manager(prefill_chunk_size=256, attn_block_size=4096)
        assert mgr._chunked_prefill_pad(_request(), query_len=1000) == 0

    def test_block_multiple_of_chunk_unaligned_prompt_no_padding(self):
        mgr = _make_manager(prefill_chunk_size=256, attn_block_size=4096)
        assert mgr._chunked_prefill_pad(_request(), query_len=4732) == 0

    def test_block_not_multiple_of_chunk_straddles(self):
        # block=10, chunk=4: chunks straddle the boundary repeatedly.
        # steps 0,4,8(->pad 2),12,16(->pad 2): total pad = 4.
        mgr = _make_manager(prefill_chunk_size=4, attn_block_size=10)
        assert mgr._chunked_prefill_pad(_request(), query_len=20) == 4


class TestChunkedPrefillPadWithImages:
    def test_image_aligned_no_padding(self):
        # Single image, all runs land within 4096 (multiple of 256).
        mgr = _make_manager(
            prefill_chunk_size=256,
            attn_block_size=4096,
            image_prefill_chunk_sizes=[256],
        )
        req = _request(_image(100, 256, lead=2, trail=2))
        assert mgr._chunked_prefill_pad(req, query_len=1000) == 0

    def test_interleaved_text_and_images_straddle(self):
        # Small hand-traced layout: block=16, text_chunk=4, image bucket=[4].
        # text[0,3) image[3,7) text[7,10) image[10,14) text[14,17) image[17,21)
        #  step  kind   run_len off  straddle?            pad
        #  0     text   3       0    0+4<=16              -
        #  3     image  4       3    3+4<=16              -
        #  7     text   3       7    7+4<=16              -
        #  10    image  4       10   10+4<=16             -
        #  14    text   3       14   14+4>16  -> +2       2
        #  17    image  4       3    3+4<=16              -
        mgr = _make_manager(
            prefill_chunk_size=4,
            attn_block_size=16,
            image_prefill_chunk_sizes=[4],
        )
        req = _request(
            _image(3, 4),
            _image(10, 4),
            _image(17, 4),
        )
        assert mgr._chunked_prefill_pad(req, query_len=21) == 2

    def test_large_image_bucket_forces_straddle(self):
        # text[0,2) then a 6-token image whose bucket is 8; block=8, text_chunk=2.
        #  step  kind   chunk off  straddle?       pad
        #  0     text   2     0    0+2<=8          -
        #  2     image  8     2    2+8>8  -> +6    6
        mgr = _make_manager(
            prefill_chunk_size=2,
            attn_block_size=8,
            image_prefill_chunk_sizes=[8],
        )
        req = _request(_image(2, 6))
        assert mgr._chunked_prefill_pad(req, query_len=8) == 6

    def test_bucket_selection_per_image(self):
        # Two images of different sizes pick different buckets [4, 8].
        # block=12, text_chunk=4.
        # text[0,2) img[2,5)(3 toks->bucket 4) text[5,7) img[7,13)(6->bucket 8)
        #  step  kind   chunk off  straddle?        pad
        #  0     text   4     0    0+4<=12          -
        #  2     image  4     2    2+4<=12          -    (run_len 3, advance 3)
        #  5     text   4     5    5+4<=12          -
        #  7     image  8     7    7+8>12 -> +5     5
        mgr = _make_manager(
            prefill_chunk_size=4,
            attn_block_size=12,
            image_prefill_chunk_sizes=[8, 4],  # descending, as optimum stores it
        )
        req = _request(_image(2, 3), _image(7, 6))
        assert mgr._chunked_prefill_pad(req, query_len=13) == 5
