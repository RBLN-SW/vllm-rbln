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

"""Restating the worker's KV geometry from pages to extents."""

import types
from dataclasses import dataclass, field

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

import vllm_rbln.envs as envs
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

PAGE = 1024
EXTENT = 8192
BYTES_PER_PAGE = 4096


@dataclass
class FakeTensor:
    size: int
    shared_by: list = field(default_factory=lambda: ["layer0"])


@dataclass
class FakeGroup:
    kv_cache_spec: object


@dataclass
class FakeConfig:
    num_blocks: int
    kv_cache_groups: list
    kv_cache_tensors: list


def make_config(num_pages=217, page_size=PAGE):
    spec = FullAttentionSpec(
        block_size=page_size, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    return FakeConfig(
        num_blocks=num_pages,
        kv_cache_groups=[FakeGroup(spec)],
        kv_cache_tensors=[FakeTensor(size=num_pages * BYTES_PER_PAGE)],
    )


def rescale(config, extent=EXTENT, enabled=True, page=PAGE, monkeypatch=None):
    runner = RBLNModelRunner.__new__(RBLNModelRunner)
    cache_config = types.SimpleNamespace(block_size=page)
    runner.vllm_config = types.SimpleNamespace(
        additional_config={"attn_block_size": extent} if extent else {},
        cache_config=cache_config,
    )
    runner.cache_config = cache_config
    monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_EXTENT", enabled)
    RBLNModelRunner._maybe_rescale_to_extents(runner, config)
    return runner, config


def test_buffer_is_trimmed_to_whole_extents(monkeypatch):
    # 217 pages is not a whole number of 8-page extents; the leftover page
    # cannot be allocated anyway, and leaving it in trips the reshape assert.
    runner, config = rescale(make_config(217), monkeypatch=monkeypatch)
    assert config.num_blocks == 27
    assert config.kv_cache_groups[0].kv_cache_spec.block_size == EXTENT
    assert config.kv_cache_tensors[0].size == BYTES_PER_PAGE * 8 * 27
    assert config.kv_cache_tensors[0].size % (BYTES_PER_PAGE * 8) == 0


def test_exact_multiple_loses_nothing(monkeypatch):
    runner, config = rescale(make_config(216), monkeypatch=monkeypatch)
    assert config.num_blocks == 27
    assert config.kv_cache_tensors[0].size == 216 * BYTES_PER_PAGE


def test_disabled_leaves_geometry_alone(monkeypatch):
    runner, config = rescale(make_config(217), enabled=False, monkeypatch=monkeypatch)
    assert config.num_blocks == 217
    assert config.kv_cache_groups[0].kv_cache_spec.block_size == PAGE


def test_model_without_published_extent_is_untouched(monkeypatch):
    runner, config = rescale(make_config(217), extent=None, monkeypatch=monkeypatch)
    assert config.num_blocks == 217
    assert config.kv_cache_groups[0].kv_cache_spec.block_size == PAGE


@pytest.mark.parametrize("extent", [PAGE, 1536])
def test_degenerate_or_misaligned_extent_is_ignored(monkeypatch, extent):
    # extent == page is a no-op; a non-multiple is not expressible at all.
    runner, config = rescale(make_config(217), extent=extent, monkeypatch=monkeypatch)
    assert config.num_blocks == 217
    assert config.kv_cache_groups[0].kv_cache_spec.block_size == PAGE


def test_cache_config_is_restated_so_attention_sees_the_extent(monkeypatch):
    # Attention impls read their block size from cache_config, not from the KV
    # cache spec. Leaving it at the page size makes them stride a 512-token
    # block through an 8192-token tensor -- reading a sixteenth of each block,
    # which corrupts output while appearing faster.
    runner, _ = rescale(make_config(217), monkeypatch=monkeypatch)
    assert runner.cache_config.block_size == EXTENT
    assert runner.vllm_config.cache_config.block_size == EXTENT


def test_block_size_is_untouched_when_disabled(monkeypatch):
    runner, _ = rescale(make_config(217), enabled=False, monkeypatch=monkeypatch)
    assert runner.cache_config.block_size == PAGE
