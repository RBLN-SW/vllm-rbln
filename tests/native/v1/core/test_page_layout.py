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

"""Unit tests for the page/kernel block addressing layer."""

import pytest

from vllm_rbln.v1.core.page_layout import (
    PageLayout,
    resolve_config,
    validate_fragmentation,
)


class TestPageLayout:
    def test_ratio(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        assert geo.pages_per_kernel_block == 8
        assert not geo.is_degenerate

    def test_degenerate(self):
        assert PageLayout(page_size=512, kernel_block_size=512).is_degenerate

    @pytest.mark.parametrize(
        "page_size, kernel_block_size",
        [(0, 4096), (512, 0), (-1, 4096), (512, 4097), (513, 4096)],
    )
    def test_rejects_bad_sizes(self, page_size, kernel_block_size):
        with pytest.raises(ValueError):
            PageLayout(page_size=page_size, kernel_block_size=kernel_block_size)

    def test_validate_chunk_accepts_divisor(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        geo.validate_chunk(512)  # equality: the usual default
        geo.validate_chunk(256)  # page = 2 * chunk

    def test_validate_chunk_rejects_chunk_larger_than_page(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError, match="never spans"):
            geo.validate_chunk(1024)

    def test_validate_chunk_rejects_non_divisor(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError):
            geo.validate_chunk(300)


class TestPageLayoutConfig:
    def test_resolves_geometry_and_pool(self):
        cfg = resolve_config(page_size=512, kernel_block_size=4096, num_pages=800)
        assert cfg.enabled
        assert cfg.geometry.pages_per_kernel_block == 8
        assert cfg.num_kernel_blocks == 100

    def test_no_published_kernel_block_size_is_degenerate(self):
        # A model that does not declare an kernel block size behaves exactly as
        # upstream: one page per kernel block, layer is a no-op.
        cfg = resolve_config(page_size=512, kernel_block_size=None, num_pages=800)
        assert not cfg.enabled
        assert cfg.geometry.pages_per_kernel_block == 1

    def test_fragmentation_rejects_pool_smaller_than_concurrency(self):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        with pytest.raises(ValueError, match="kernel blocks per request"):
            validate_fragmentation(geo, max_num_seqs=8, num_kernel_blocks=8)

    def test_fragmentation_warns_when_most_of_the_pool_can_be_pinned(self, caplog):
        geo = PageLayout(page_size=512, kernel_block_size=4096)
        validate_fragmentation(geo, max_num_seqs=7, num_kernel_blocks=10)
        assert "kernel blocks can be pinned at once" in caplog.text

    def test_fragmentation_is_silent_when_degenerate(self):
        geo = PageLayout(page_size=512, kernel_block_size=512)
        validate_fragmentation(geo, max_num_seqs=1000, num_kernel_blocks=1)
