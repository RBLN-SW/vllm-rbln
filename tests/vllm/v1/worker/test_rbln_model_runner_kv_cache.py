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

# KV cache initialization on a real runner: the backend-driven base/view split
# and the per-group attention metadata over it. test_kv_cache_bindings.py covers
# the consumer of those recipes; this is the producer.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.vllm.v1.worker.utils import BLOCK_SIZE, make_kv_cache_config, schedule_new

pytestmark = pytest.mark.maybe_use_device


def _one_group_two_layers(make_model_runner):
    """A runner whose single KV cache group holds two layers, so they share one
    allocation -- the only shape in which the deduplicated bindings turn on."""
    runner = make_model_runner(layers=("layer.0", "layer.1"), init_kv_cache=False)
    runner.initialize_kv_cache(
        make_kv_cache_config(runner, groups=[("layer.0", "layer.1")])
    )
    return runner


class TestReshapeKVCacheTensors:
    # Technique from upstream's test_kv_cache_stride_order: drive the real
    # backend rather than a fake, so the assertions hold for the backend in use.
    def test_backend_stride_order_permutes_base_but_not_the_view(
        self, make_model_runner, monkeypatch
    ):
        runner = make_model_runner()
        config = runner.kv_cache_config
        raw = runner._allocate_kv_cache_tensors(config)
        kernel_block_sizes = runner._kernel_block_sizes

        caches, _, infos, _ = runner._reshape_kv_cache_tensors(
            config, raw, kernel_block_sizes
        )
        semantic_shape = tuple(caches["layer.0"].shape)
        identity = tuple(range(len(semantic_shape)))
        assert caches["layer.0"].is_contiguous()
        assert infos["layer.0"].permute_order == identity

        # The K/V split outermost instead of blocks. vllm 0.30.0 dropped the
        # base declaration, so only the RBLN backends that want an order define
        # one; raising=False stands this backend up as one of them.
        order = (1, 0) + identity[2:]
        monkeypatch.setattr(
            runner.attn_groups[0][0].backend,
            "get_kv_cache_stride_order",
            staticmethod(lambda *args, **kwargs: order),
            raising=False,
        )
        caches, bases, infos, _ = runner._reshape_kv_cache_tensors(
            config, raw, kernel_block_sizes
        )

        permuted_shape = tuple(semantic_shape[i] for i in order)
        cache, base, info = caches["layer.0"], bases["layer.0"], infos["layer.0"]

        # The layer still sees the semantic shape; only the allocation moved.
        assert tuple(cache.shape) == semantic_shape
        assert tuple(base.shape) == permuted_shape
        assert not cache.is_contiguous()

        # The view aliases the base, and the recorded recipe reproduces it.
        assert cache.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        assert info.view_shape == permuted_shape
        assert info.permute_order == tuple(
            order.index(i) for i in range(len(semantic_shape))
        )


class TestKVCacheBaseBindings:
    def test_disabled_when_no_layers_share_an_allocation(self, make_model_runner):
        # One base per layer means the indirection buys nothing, so the runner
        # drops it and the eager per-layer cache list is used instead.
        runner = make_model_runner()
        assert runner.kv_cache_bases == []
        assert runner.kv_cache_view_infos == []

    def test_a_shared_buffer_is_bound_in_the_full_attention_shape(
        self, make_model_runner
    ):
        # A sliding-window layer ahead of a full-attention one on one buffer, its
        # kernel on its window: the two views differ, and the graph input must
        # be the full-attention layer's, the one the static address names.
        window = BLOCK_SIZE // 4
        runner = make_model_runner(
            layers=("layer.0", "layer.1"),
            init_kv_cache=False,
            additional_config={"enable_sub_block_cache": False},
        )
        full = make_kv_cache_config(runner).kv_cache_groups[0].kv_cache_spec
        sliding = SlidingWindowSpec(
            block_size=full.block_size,
            num_kv_heads=full.num_kv_heads,
            head_size=full.head_size,
            dtype=full.dtype,
            sliding_window=window,
        )
        runner.initialize_kv_cache(
            make_kv_cache_config(
                groups=[("layer.0",), ("layer.1",)], specs=[sliding, full]
            )
        )
        window_info, full_info = runner.kv_cache_view_infos
        assert window_info.view_shape != full_info.view_shape
        assert [tuple(b.shape) for b in runner.kv_cache_bases] == [full_info.view_shape]


class TestLayerIndexFollowsCacheOrder:
    """`initialize_kv_cache_tensors` gives every layer the position of its cache
    in `kv_cache_names`. The construction-time index assumes `num_attn_module`
    caches per decoder layer, which a layer owning fewer breaks for every
    layer after it."""

    @staticmethod
    def _bind(make_model_runner, monkeypatch, names, **text_config):
        runner = make_model_runner(layers=names, init_kv_cache=False)
        # The test config is memoized, so put the attributes back afterwards.
        for key, value in text_config.items():
            monkeypatch.setattr(
                runner.model_config.hf_text_config, key, value, raising=False
            )
        ctx = runner.compilation_config.static_forward_context
        before = {name: ctx[name].layer_index for name in names}

        runner.initialize_kv_cache(make_kv_cache_config(runner, groups=[names]))

        after = {name: ctx[name].layer_index for name in names}
        # Whatever the index, it must reach the layer's own bound cache.
        for name in names:
            assert runner.kv_caches[after[name]] is ctx[name].kv_cache
        return runner, before, after

    def test_minimax_m3_dense_layers_own_no_indexer_cache(
        self, make_model_runner, monkeypatch
    ):
        names = (
            "model.layers.0.self_attn.attn",  # dense: K/V only
            "model.layers.1.self_attn.attn",  # sparse: K/V ...
            "model.layers.1.self_attn.attn.indexer",  # ... and the indexer keys
            "model.layers.2.self_attn.attn",  # dense again
        )
        runner, before, after = self._bind(
            make_model_runner,
            monkeypatch,
            names,
            sparse_attention_config={"sparse_attention_freq": [0, 1, 0]},
        )
        # Construction indexed by decoder layer, so two caches collided.
        assert (
            before["model.layers.1.self_attn.attn"]
            == before["model.layers.1.self_attn.attn.indexer"]
        )
        assert runner.kv_cache_names == list(names)
        assert after == {name: i for i, name in enumerate(names)}

    def test_dsa_fp8_keeps_three_caches_per_layer_in_order(
        self, make_model_runner, monkeypatch
    ):
        # DeepSeek-V3.2 / A.X-K3 under KV8: MLA latent, indexer keys, and the
        # indexer's fp16 scale cache, so `num_attn_module` is 3.
        monkeypatch.setattr(mr, "rbln_num_attn_module", lambda *args, **kwargs: 3)
        names = tuple(
            f"model.layers.{layer}.self_attn.{module}"
            for layer in (0, 1)
            for module in ("attn", "indexer.k_cache", "indexer.k_cache.scale_cache")
        )
        runner, _, after = self._bind(make_model_runner, monkeypatch, names)
        assert runner.kv_cache_names == list(names)
        assert after == {name: i for i, name in enumerate(names)}

    def test_mtp_layer_sits_after_the_target_layers(self, make_model_runner):
        from tests.vllm.v1.worker.utils import make_runner_config

        config = make_runner_config()
        num_layers = config.model_config.get_total_num_hidden_layers()
        names = (
            "model.layers.0.self_attn.attn",
            "model.layers.1.self_attn.attn",
            # An MTP / nextn layer is named past the target's layer count.
            f"model.layers.{num_layers}.self_attn.attn",
        )
        runner = make_model_runner(
            vllm_config=config, layers=names, init_kv_cache=False
        )
        ctx = runner.compilation_config.static_forward_context
        runner.initialize_kv_cache(make_kv_cache_config(runner, groups=[names]))
        assert runner.kv_cache_names == list(names)
        assert [ctx[name].layer_index for name in names] == [0, 1, 2]
        assert runner.kv_caches[2] is ctx[names[-1]].kv_cache


class TestHostBufferCopyOp:
    def test_the_registered_op_reads_the_axes_at_call_time(
        self, make_model_runner, monkeypatch
    ):
        # A host-staging connector moves blocks through the op registered here.
        # A dynamic-KV reallocation replaces `kv_cache_block_axes`, so an op
        # holding the dict it was registered with would keep copying by the
        # axes of a cache that no longer exists.
        captured: list = []
        seen: dict = {}
        monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(
            mr,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(
                register_kv_caches=lambda _caches: None,
                set_host_xfer_buffer_ops=captured.append,
            ),
        )
        monkeypatch.setattr(
            mr, "copy_host_device_kv_blocks", lambda *a, **kw: seen.update(kw)
        )

        runner = make_model_runner(init_kv_cache=False)
        runner.initialize_kv_cache(make_kv_cache_config(runner, groups=[("layer.0",)]))
        runner.register_kv_caches_with_connector()
        runner.kv_cache_block_axes = {"layer.0": 1}
        captured[0]({}, {}, [], [], "h2d")

        assert seen == {"block_axes": {"layer.0": 1}}


class TestRegisterKvCachesWithConnector:
    """The connector is handed the storage the runner holds when it is called.

    Each cache goes out as the runner's own tensor, or as a view of it where
    connectors need another shape. The worker registers after
    `initialize_kv_cache` on a run that keeps that cache, and again after a
    dynamic-KV resize has rebuilt every tensor.
    """

    @staticmethod
    def _registrations(monkeypatch) -> list[dict]:
        registered: list[dict] = []
        monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(
            mr,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(
                register_kv_caches=registered.append,
                set_host_xfer_buffer_ops=lambda _op: None,
            ),
        )
        return registered

    def test_the_caches_registered_are_the_ones_bound(
        self, make_model_runner, monkeypatch
    ):
        registered = self._registrations(monkeypatch)
        runner = make_model_runner(
            layers=("layer.0", "layer.1"),
            init_kv_cache=False,
            additional_config={"enable_sub_block_cache": False},
        )
        runner.initialize_kv_cache(
            make_kv_cache_config(runner, groups=[("layer.0", "layer.1")])
        )

        runner.register_kv_caches_with_connector()

        # In layer-index order: NIXL numbers its transfer regions by iteration
        # order, and the peer matches regions to layers by that same order.
        assert list(registered[0]) == runner.kv_cache_names
        assert [id(t) for t in registered[0].values()] == [
            id(t) for t in runner.kv_caches
        ]

    def test_layers_that_share_a_buffer_are_each_registered(
        self, make_model_runner, monkeypatch
    ):
        # One layer per group, so the groups overlay and both layers share one
        # buffer. A token-level connector addresses each layer through its own
        # view and group, so it needs both, not one per buffer.
        registered = self._registrations(monkeypatch)
        runner = make_model_runner(
            layers=("layer.0", "layer.1"),
            init_kv_cache=False,
            additional_config={"enable_sub_block_cache": False},
        )
        runner.initialize_kv_cache(
            make_kv_cache_config(runner, groups=[("layer.0",), ("layer.1",)])
        )
        assert runner.kv_caches[0].untyped_storage().data_ptr() == (
            runner.kv_caches[1].untyped_storage().data_ptr()
        )

        runner.register_kv_caches_with_connector()

        assert list(registered[0]) == runner.kv_cache_names
        assert [id(t) for t in registered[0].values()] == [
            id(t) for t in runner.kv_caches
        ]

    def test_a_flat_cache_is_registered_with_a_unit_trailing_dim(
        self, make_model_runner, monkeypatch
    ):
        # The DSA indexer's fp16 scale cache is allocated [num_blocks,
        # block_size]. Connectors read every per-token cache with a trailing
        # width, so they get a [num_blocks, block_size, 1] view of it; the
        # kernels keep the flat tensor, and every other cache goes out as is.
        registered = self._registrations(monkeypatch)
        runner = make_model_runner(
            layers=("layer.0", "layer.1"),
            init_kv_cache=False,
            additional_config={"enable_sub_block_cache": False},
        )
        runner.initialize_kv_cache(
            make_kv_cache_config(runner, groups=[("layer.0", "layer.1")])
        )
        flat = torch.zeros(4, 16, dtype=torch.float16)
        runner.kv_caches[1] = flat

        runner.register_kv_caches_with_connector()

        attn, scale = registered[0].values()
        assert attn is runner.kv_caches[0]
        assert tuple(scale.shape) == (4, 16, 1)
        assert scale.data_ptr() == flat.data_ptr()
        assert runner.kv_caches[1] is flat

    def test_a_rebuilt_cache_is_what_the_second_call_registers(
        self, make_model_runner, monkeypatch
    ):
        # `DynamicKvSizer.reallocate` rebuilds the tensors after warm-up. A
        # registration holding the first set would name freed memory.
        registered = self._registrations(monkeypatch)
        runner = make_model_runner()
        runner.register_kv_caches_with_connector()

        runner.initialize_kv_cache_tensors(
            runner.kv_cache_config, runner._kernel_block_sizes
        )
        runner.register_kv_caches_with_connector()

        first, second = registered
        assert all(
            before is not after
            for before, after in zip(first.values(), second.values(), strict=True)
        )
        assert [id(t) for t in second.values()] == [id(t) for t in runner.kv_caches]


class TestBuildAttentionMetadata:
    def test_one_build_per_group_shared_across_its_layers(
        self, make_model_runner, monkeypatch
    ):
        # Attention metadata is built once per group and handed to every layer in
        # it, with the deduplicated KV bindings attached.
        monkeypatch.setattr(
            mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
        )
        runner = _one_group_two_layers(make_model_runner)
        runner._update_states(schedule_new("a", "b"))

        runner.query_start_loc[:3] = torch.tensor([0, 1, 2], dtype=torch.int32)
        runner.seq_lens[:2] = torch.tensor([4, 4], dtype=torch.int32)

        attn_metadata, spec_common = runner._build_attention_metadata(
            num_tokens=2, num_reqs=2, max_query_len=1, num_reqs_padded=4
        )

        assert spec_common is None
        assert set(attn_metadata) == {"layer.0", "layer.1"}
        assert attn_metadata["layer.0"] is attn_metadata["layer.1"]

        # Each layer owns its allocation, so there is nothing to deduplicate
        # and the metadata carries the per-layer caches directly.
        metadata = attn_metadata["layer.0"]
        assert metadata.kv_caches is runner.kv_caches
        assert metadata.kv_cache_view_infos is None


class TestLayerIndexFollowsCachePosition:
    """A layer reads its cache by `layer_index` into the runner's cache list.

    Construction numbers the layers as if every decoder layer owned
    `num_attn_module` caches. A layer that owns fewer -- an index-share skip
    layer with no indexer cache -- shifts every later one, so after binding the
    runner gives each layer the position the bound list actually has for it.
    """

    # The fp8 DSA layout, three caches per decoder layer: the attention cache,
    # the indexer's key cache and its fp16 scale cache. Decoder layer 1 shares
    # layer 0's top-k and owns the attention cache alone.
    LAYERS = (
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.0.self_attn.indexer.k_cache.scale_cache",
        "model.layers.1.self_attn.attn",
        "model.layers.2.self_attn.attn",
        "model.layers.2.self_attn.indexer.k_cache",
        "model.layers.2.self_attn.indexer.k_cache.scale_cache",
    )

    @classmethod
    def _assert_each_layer_reads_its_own_cache(cls, runner) -> None:
        forward_context = runner.compilation_config.static_forward_context
        positions = []
        for name in cls.LAYERS:
            layer = forward_context[name]
            positions.append(layer.layer_index)
            assert runner.kv_cache_names[layer.layer_index] == name
            assert runner.kv_caches[layer.layer_index] is layer.kv_cache
        # One slot each: no two layers resolve to the same cache.
        assert sorted(positions) == list(range(len(cls.LAYERS)))

    def test_after_initialization_and_after_a_rebind(
        self, make_model_runner, monkeypatch
    ):
        monkeypatch.setattr(mr, "rbln_num_attn_module", lambda *args, **kwargs: 3)
        runner = make_model_runner(layers=self.LAYERS, init_kv_cache=False)

        runner.initialize_kv_cache(
            make_kv_cache_config(runner, groups=[(name,) for name in self.LAYERS])
        )
        self._assert_each_layer_reads_its_own_cache(runner)

        # A dynamic-KV resize rebuilds every tensor and binds them again.
        before = list(runner.kv_caches)
        runner.initialize_kv_cache_tensors(
            runner.kv_cache_config, runner._kernel_block_sizes
        )

        assert all(
            old is not new for old, new in zip(before, runner.kv_caches, strict=True)
        )
        self._assert_each_layer_reads_its_own_cache(runner)
