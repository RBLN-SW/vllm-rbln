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

"""The KV-cache spec a sliding-window layer declares. Which one it is decides
the cache layout for the whole run, and VLLM_RBLN_USE_MULTI_BLOCK_ATTN decides
which one."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_rbln.patches.attention import patched_get_kv_cache_spec
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec


def _layer(sliding_window):
    return SimpleNamespace(
        attn_type=AttentionType.DECODER,
        sliding_window=sliding_window,
        num_kv_heads=2,
        head_size=8,
        head_size_v=8,
        kv_cache_torch_dtype=torch.float16,
    )


def _config(use_mla=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32),
        model_config=SimpleNamespace(use_mla=use_mla),
    )


@pytest.fixture
def multi_block(monkeypatch):
    def switch(on):
        monkeypatch.setenv("VLLM_RBLN_USE_MULTI_BLOCK_ATTN", "1" if on else "0")

    return switch


class TestSlidingWindowSpec:
    def test_multi_block_declares_upstreams_spec(self, multi_block):
        # Appending into an ordinary paged cache is what upstream's spec and its
        # manager already describe: several blocks, reclaimed behind the window.
        multi_block(True)
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is SlidingWindowSpec

    @pytest.mark.parametrize("explicitly_off", [True, False])
    def test_otherwise_the_rbln_spec(self, monkeypatch, multi_block, explicitly_off):
        # The shift kernel holds the window in one block, which only
        # RBLNSlidingWindowSpec and its manager allocate that way. It stays the
        # default, so an unset variable has to land here too.
        if explicitly_off:
            multi_block(False)
        else:
            monkeypatch.delenv("VLLM_RBLN_USE_MULTI_BLOCK_ATTN", raising=False)
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is RBLNSlidingWindowSpec

    def test_a_layer_without_a_window_ignores_the_flag(self, multi_block):
        # The flag names a sliding-window cache layout; full attention keeps the
        # one spec it has on either setting.
        multi_block(True)
        assert type(patched_get_kv_cache_spec(_layer(None), _config())) is (
            FullAttentionSpec
        )

    def test_mla_with_a_window_is_rejected(self, multi_block):
        multi_block(True)
        with pytest.raises(NotImplementedError, match="MLA"):
            patched_get_kv_cache_spec(_layer(16), _config(use_mla=True))
