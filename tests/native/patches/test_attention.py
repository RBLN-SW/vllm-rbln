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
the cache layout for the whole run, and the NPU decides which one."""

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform
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
def npu(monkeypatch):
    def named(name):
        monkeypatch.setattr(current_platform, "get_device_name", lambda *a: name)

    return named


class TestSlidingWindowSpec:
    def test_cr13_declares_upstreams_spec(self, npu):
        # CR13 appends into an ordinary paged cache, which upstream's spec and
        # its manager already describe: several blocks, reclaimed behind the
        # window.
        npu("RBLN-CR13")
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is SlidingWindowSpec

    def test_every_other_npu_declares_the_rbln_spec(self, npu):
        # The shift kernel holds the window in one block, which only
        # RBLNSlidingWindowSpec and its manager allocate that way.
        npu("RBLN-CA25")
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is RBLNSlidingWindowSpec

    def test_a_layer_without_a_window_asks_no_npu(self, monkeypatch):
        # Full attention is the same on every NPU, so the name is never read --
        # it is not always resolvable (a CPU-only compile host needs an env var).
        monkeypatch.setattr(
            current_platform,
            "get_device_name",
            lambda *a: pytest.fail("the NPU name is irrelevant to full attention"),
        )
        assert type(patched_get_kv_cache_spec(_layer(None), _config())) is (
            FullAttentionSpec
        )

    def test_mla_with_a_window_is_rejected(self, npu):
        npu("RBLN-CR13")
        with pytest.raises(NotImplementedError, match="MLA"):
            patched_get_kv_cache_spec(_layer(16), _config(use_mla=True))
