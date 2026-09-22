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
the cache layout for the whole run, and the device decides which one: REBEL
CR13 carries the appending kernel, every other NPU shifts one block."""

from types import SimpleNamespace

import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_rbln.patches import attention
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
def cr13(monkeypatch):
    def switch(on):
        monkeypatch.setattr(current_platform, "is_cr13", lambda: on)

    return switch


class TestSlidingWindowSpec:
    def test_cr13_declares_upstreams_spec(self, cr13):
        # Appending into an ordinary paged cache is what upstream's spec and its
        # manager already describe: several blocks, reclaimed behind the window.
        cr13(True)
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is SlidingWindowSpec

    def test_otherwise_the_rbln_spec(self, cr13):
        # The shift kernel holds the window in one block, which only
        # RBLNSlidingWindowSpec and its manager allocate that way.
        cr13(False)
        spec = patched_get_kv_cache_spec(_layer(16), _config())
        assert type(spec) is RBLNSlidingWindowSpec

    def test_a_layer_without_a_window_ignores_the_device(self, cr13):
        # The device picks a sliding-window cache layout; full attention keeps
        # the one spec it has on either NPU.
        cr13(True)
        assert type(patched_get_kv_cache_spec(_layer(None), _config())) is (
            FullAttentionSpec
        )

    def test_mla_with_a_window_is_rejected(self, cr13):
        cr13(True)
        with pytest.raises(NotImplementedError, match="MLA"):
            patched_get_kv_cache_spec(_layer(16), _config(use_mla=True))


class TestKvTransferWrap:
    """`unified_attention_with_output` is wrapped for KV-transfer connectors.

    `maybe_transfer_kv_layer` imports `get_attention_context` when it decorates,
    so the wrap has to be built after that name has been replaced. Building it
    at import instead would freeze upstream's version into the closure, and the
    connector would then read the layer's embedded KV cache -- the one thing the
    override exists to stop.
    """

    @staticmethod
    def _captured_attention_context(wrapper):
        cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()))
        return cells["get_attention_context"].cell_contents

    def test_the_wrap_closed_over_the_patched_attention_context(self):
        import vllm.model_executor.layers.attention.attention as upstream

        captured = self._captured_attention_context(
            upstream.unified_attention_with_output
        )

        assert captured is attention.patched_get_attention_context

    def test_the_wrap_is_around_our_replacement(self):
        import vllm.model_executor.layers.attention.attention as upstream

        wrapped = upstream.unified_attention_with_output.__wrapped__

        assert wrapped is attention._unified_attention_with_output

    @pytest.fixture
    def connector(self, monkeypatch):
        """Turn the wrapper's connector branch on, and record what it saves."""
        from vllm.model_executor.layers.attention import kv_transfer_utils as kvt

        saved: dict = {}

        class _Connector:
            def has_connector_metadata(self):
                return True

            def wait_for_layer_load(self, layer_name):
                saved["waited"] = layer_name

            def save_kv_layer(self, layer_name, kv_cache, attn_metadata):
                saved["layer_name"] = layer_name
                saved["kv_cache"] = kv_cache

        monkeypatch.setattr(kvt, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(kvt, "is_v1_kv_transfer_group", lambda: True)
        monkeypatch.setattr(kvt, "get_kv_transfer_group", _Connector)
        return saved

    @pytest.fixture
    def forward_context(self, monkeypatch):
        """The layer carries a KV cache the override must keep out of reach."""
        embedded = torch.full((2, 1, 1), 7.0)
        impl_saw: dict = {}

        class _Impl:
            def forward(self, layer, q, k, v, kv_cache, attn_metadata, **kwargs):
                impl_saw["kv_cache"] = kv_cache

        layer = SimpleNamespace(layer_index=0, impl=_Impl(), kv_cache=embedded)
        context = SimpleNamespace(
            attn_metadata=SimpleNamespace(kv_caches=[torch.zeros(1)]),
            no_compile_layers={"layer.0": layer},
            slot_mapping={"layer.0": torch.zeros(1)},
            additional_kwargs={},
        )
        monkeypatch.setattr(attention, "get_forward_context", lambda: context)
        # Upstream's own get_attention_context reads it from its module too, so
        # a wrap that closed over that one runs here rather than raising, and
        # the assertions below show what it hands the connector instead.
        import vllm.model_executor.layers.attention.attention as upstream

        monkeypatch.setattr(upstream, "get_forward_context", lambda: context)
        return impl_saw

    def test_the_connector_is_offered_no_kv_cache(self, connector, forward_context):
        """The override returns kv_cache=None, and this is who reads it.

        Whether the wrap closed over the patched version or upstream's decides
        what lands here: None, or the cache embedded in the layer, which Dynamo
        would bake into the graph as a constant.
        """
        import vllm.model_executor.layers.attention.attention as upstream

        upstream.unified_attention_with_output(
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), "layer.0"
        )

        assert connector["layer_name"] == "layer.0"
        assert connector["kv_cache"] is None
        # The attention op still resolves its own, from the metadata.
        assert forward_context["kv_cache"] is not None
