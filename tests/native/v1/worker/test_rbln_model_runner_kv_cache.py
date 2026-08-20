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

import vllm_rbln.v1.worker.rbln_model_runner as mr
from tests.native.v1.worker.utils import make_kv_cache_config, schedule_new

pytestmark = pytest.mark.maybe_use_device


def _unexpected(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return fail


def _one_group_two_layers(make_model_runner):
    """A runner whose single KV cache group holds two layers, so they share one
    allocation -- the only shape in which the deduplicated bindings turn on."""
    runner = make_model_runner(layers=("layer.0", "layer.1"), init_kv_cache=False)
    runner.initialize_kv_cache(
        make_kv_cache_config(runner, groups=[("layer.0", "layer.1")])
    )
    return runner


class TestSubBlockCacheGuard:
    def test_rejects_multi_group_before_any_side_effect(
        self, make_model_runner, monkeypatch
    ):
        # Sub-block prefix caching cannot span KV cache groups, and the refusal
        # must come before attention backends are set up.
        monkeypatch.setattr(mr.envs, "VLLM_RBLN_SUB_BLOCK_CACHE", True)
        runner = make_model_runner(layers=("layer.0", "layer.1"), init_kv_cache=False)
        monkeypatch.setattr(
            runner, "initialize_attn_backend", _unexpected("initialize_attn_backend")
        )

        config = make_kv_cache_config(runner, groups=[("layer.0",), ("layer.1",)])
        with pytest.raises(NotImplementedError, match="multi-group"):
            runner.initialize_kv_cache(config)


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

        caches, _, infos = runner._reshape_kv_cache_tensors(
            config, raw, kernel_block_sizes
        )
        semantic_shape = tuple(caches["layer.0"].shape)
        identity = tuple(range(len(semantic_shape)))
        assert caches["layer.0"].is_contiguous()
        assert infos["layer.0"].permute_order == identity

        # Blocks outermost instead of the K/V split.
        order = (1, 0) + identity[2:]
        monkeypatch.setattr(
            runner.attn_groups[0][0].backend,
            "get_kv_cache_stride_order",
            staticmethod(lambda *args, **kwargs: order),
        )
        caches, bases, infos = runner._reshape_kv_cache_tensors(
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

        # Compiled path: the per-layer cache list is dropped in favour of the
        # deduplicated view recipes.
        metadata = attn_metadata["layer.0"]
        assert metadata.kv_caches is None
        assert metadata.kv_cache_view_infos is runner.kv_cache_view_infos
