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

import re
import types

import pytest
import torch

from vllm_rbln.v1.attention.kv_cache_bindings import (
    KVCacheViewInfo,
    _storage_key,
    attach_kv_cache_bindings,
    build_kv_cache_base_bindings,
    build_kv_cache_forward_context_kwargs,
    materialize_kv_cache_view,
    validate_shared_attention_kv_cache_contiguity,
)

# A compiled graph takes every KV cache as an input, so layers sharing one
# storage collapse to a single base plus a per-layer view recipe. These pin that
# storage-identity / dedup / reconstruction pipeline.


def _layer(i: int) -> str:
    # A layer name extract_layer_index() parses to `i` (single integer, has
    # "attn"). Used as the dict keys the helpers order and group by.
    return f"model.layers.{i}.self_attn"


class TestStorageKey:
    def test_views_of_same_base_share_key(self):
        # Views over one allocation must hash equal, so dedup collapses them.
        base = torch.zeros(4, 8)
        assert _storage_key(base[0]) == _storage_key(base[1]) == _storage_key(base)

    def test_independent_tensors_have_distinct_keys(self):
        # Separately allocated tensors are never merged into one base.
        assert _storage_key(torch.zeros(4)) != _storage_key(torch.zeros(4))

    def test_meta_tensors_keyed_by_identity_not_data_ptr(self):
        # Meta tensors all report data_ptr()==0, so the meta path keys on storage
        # identity instead.
        a = torch.empty(4, 8, device="meta")
        b = torch.empty(4, 8, device="meta")
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() == 0
        assert _storage_key(a) != _storage_key(b)  # distinct via identity
        assert _storage_key(a[0]) == _storage_key(a[1])  # views still share


class TestBuildKvCacheBaseBindings:
    def test_shared_storage_is_deduplicated(self):
        # Two layers backed by views of one tensor -> a single base, both view
        # infos pointing at index 0.
        shared = torch.zeros(2, 4, 8)
        bases = {_layer(0): shared[0], _layer(1): shared[1]}
        infos = {_layer(0): KVCacheViewInfo(), _layer(1): KVCacheViewInfo()}
        base_tensors, view_infos = build_kv_cache_base_bindings(bases, infos)
        assert len(base_tensors) == 1
        assert [vi.base_index for vi in view_infos] == [0, 0]

    def test_distinct_storage_gets_incrementing_indices(self):
        # Independent buffers each become their own base, indexed in order.
        bases = {_layer(0): torch.zeros(4, 8), _layer(1): torch.zeros(4, 8)}
        infos = {_layer(0): KVCacheViewInfo(), _layer(1): KVCacheViewInfo()}
        base_tensors, view_infos = build_kv_cache_base_bindings(bases, infos)
        assert len(base_tensors) == 2
        assert [vi.base_index for vi in view_infos] == [0, 1]

    def test_view_info_fields_preserved_except_base_index(self):
        # Only base_index is filled in; the caller's reconstruction recipe
        # (dtype/shape/permute/select) is left intact.
        info = KVCacheViewInfo(view_shape=(4, 8), permute_order=(1, 0), select_index=2)
        _, view_infos = build_kv_cache_base_bindings(
            {_layer(0): torch.zeros(4, 8)}, {_layer(0): info}
        )
        got = view_infos[0]
        assert got.base_index == 0
        assert got.view_shape == (4, 8)
        assert got.permute_order == (1, 0)
        assert got.select_index == 2

    def test_mixed_shared_and_distinct_storage(self):
        # Two layers aliasing one base plus an independent third: guards the
        # dedup <-> index interaction the pure cases cannot.
        shared = torch.zeros(2, 4, 8)
        other = torch.zeros(4, 8)
        bases = {_layer(0): shared[0], _layer(1): shared[1], _layer(2): other}
        infos = {name: KVCacheViewInfo() for name in bases}
        base_tensors, view_infos = build_kv_cache_base_bindings(bases, infos)
        assert len(base_tensors) == 2
        assert [vi.base_index for vi in view_infos] == [0, 0, 1]

    def test_processing_order_follows_layer_index(self):
        # Bases are appended in layer-index order regardless of dict insertion
        # order, so kv_cache_view_infos[layer_index] lines up in the forward.
        b0, b2, b10 = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        bases = {_layer(10): b10, _layer(2): b2, _layer(0): b0}  # scrambled
        infos = {name: KVCacheViewInfo() for name in bases}
        base_tensors, _ = build_kv_cache_base_bindings(bases, infos)
        assert base_tensors[0] is b0
        assert base_tensors[1] is b2
        assert base_tensors[2] is b10

    def test_view_infos_align_with_ordered_layers(self):
        # The forward indexes view_infos by layer position, so each must line up
        # with its layer in layer-index order.
        bases = {_layer(i): torch.zeros(1) for i in (2, 0, 1)}  # scrambled
        infos = {_layer(i): KVCacheViewInfo(select_index=i) for i in (2, 0, 1)}
        _, view_infos = build_kv_cache_base_bindings(bases, infos)
        assert [vi.select_index for vi in view_infos] == [0, 1, 2]


class TestMaterializeKvCacheView:
    def test_reshape_permute_select_roundtrip(self):
        # Rebuilds the exact per-layer view, aliasing the base so KV writes
        # propagate back to the shared storage.
        flat = torch.arange(2 * 3 * 4, dtype=torch.float32)
        info = KVCacheViewInfo(
            base_index=0,
            view_shape=(2, 3, 4),
            permute_order=(1, 0, 2),
            select_index=1,
        )
        out = materialize_kv_cache_view([flat], info)
        expected = flat.view(2, 3, 4).permute(1, 0, 2).select(0, 1)
        assert torch.equal(out, expected)
        assert _storage_key(out) == _storage_key(flat)  # aliases the base

    def test_dtype_view_reinterprets_bytes(self):
        # view_dtype reinterprets the base storage as another same-width dtype
        # (used when the compiled cache dtype differs from the layer's).
        base = torch.zeros(8, dtype=torch.int32)
        out = materialize_kv_cache_view(
            [base], KVCacheViewInfo(base_index=0, view_dtype=torch.float32)
        )
        assert out.dtype == torch.float32
        assert _storage_key(out) == _storage_key(base)

    def test_dtype_then_shape_view_combined(self):
        # Production path: a raw-byte cache reinterpreted as the layer dtype and
        # then reshaped. view_dtype is applied before view_shape.
        base = torch.zeros(16, dtype=torch.int8)  # 16 bytes -> 8 float16
        info = KVCacheViewInfo(
            base_index=0, view_dtype=torch.float16, view_shape=(2, 4)
        )
        out = materialize_kv_cache_view([base], info)
        assert out.dtype == torch.float16
        assert out.shape == (2, 4)
        assert _storage_key(out) == _storage_key(base)

    def test_view_write_propagates_to_base(self):
        # Direct aliasing proof: a write through the reconstructed view lands in
        # the shared base, so KV cache updates are not silently lost to a copy.
        base = torch.zeros(2, 3, 4)
        out = materialize_kv_cache_view(
            [base], KVCacheViewInfo(base_index=0, view_shape=(2, 3, 4), select_index=1)
        )
        out.add_(5.0)
        assert torch.all(base[1] == 5.0)
        assert torch.all(base[0] == 0.0)

    def test_all_none_recipe_returns_base_unchanged(self):
        # With no transforms the base is returned as-is (identity).
        base = torch.zeros(4, 8)
        out = materialize_kv_cache_view([base], KVCacheViewInfo(base_index=0))
        assert out is base


class TestValidateSharedContiguity:
    def test_contiguous_shared_layers_pass(self):
        # Layers sharing one base are fine while their caches are contiguous.
        shared = torch.zeros(2, 4, 8)
        bases = {_layer(0): shared, _layer(1): shared}
        kv = {_layer(0): shared[0].contiguous(), _layer(1): shared[1].contiguous()}
        infos = {_layer(0): KVCacheViewInfo(), _layer(1): KVCacheViewInfo()}
        validate_shared_attention_kv_cache_contiguity(kv, bases, infos)  # no raise

    def test_noncontiguous_shared_layer_raises(self):
        # A non-contiguous cache among storage-sharing layers is rejected, with
        # the offending layer named (RBLN compile requires contiguity).
        shared = torch.zeros(4, 8)
        bases = {_layer(0): shared, _layer(1): shared}
        kv = {_layer(0): shared, _layer(1): shared.t()}  # layer 1 non-contiguous
        infos = {_layer(0): KVCacheViewInfo(), _layer(1): KVCacheViewInfo()}
        with pytest.raises(ValueError, match=re.escape(_layer(1))):
            validate_shared_attention_kv_cache_contiguity(kv, bases, infos)

    def test_single_layer_per_storage_is_skipped(self):
        # One layer per storage -> contiguity not enforced (the check targets
        # aliasing between layers, not lone caches).
        bases = {_layer(0): torch.zeros(4, 8)}
        kv = {_layer(0): torch.zeros(4, 8).t()}  # non-contiguous but alone
        infos = {_layer(0): KVCacheViewInfo()}
        validate_shared_attention_kv_cache_contiguity(kv, bases, infos)  # no raise

    def test_layers_missing_base_or_cache_are_skipped(self):
        # A layer absent from either mapping is ignored, not an error.
        shared = torch.zeros(4, 8)
        bases = {_layer(0): shared}  # layer 1 base missing
        kv = {_layer(0): shared, _layer(1): shared.t()}
        infos = {_layer(0): KVCacheViewInfo(), _layer(1): KVCacheViewInfo()}
        validate_shared_attention_kv_cache_contiguity(kv, bases, infos)  # no raise


class TestAttachKvCacheBindings:
    # attn_metadata is a real object (SimpleNamespace), not a mock.
    def test_uses_bases_and_view_infos_when_both_present(self):
        # Compile/export path: carry the deduplicated view infos, drop the
        # per-layer cache list.
        md = types.SimpleNamespace()
        infos = [KVCacheViewInfo(base_index=0)]
        attach_kv_cache_bindings(
            md,
            kv_caches=[torch.zeros(1)],
            kv_cache_bases=[torch.zeros(1)],
            kv_cache_view_infos=infos,
        )
        assert md.kv_caches is None
        assert md.kv_cache_view_infos is infos

    def test_falls_back_to_kv_caches_when_bases_absent(self):
        # Eager/fallback path: no bases -> carry the per-layer caches.
        md = types.SimpleNamespace()
        caches = [torch.zeros(1)]
        attach_kv_cache_bindings(
            md, kv_caches=caches, kv_cache_bases=None, kv_cache_view_infos=None
        )
        assert md.kv_caches is caches
        assert md.kv_cache_view_infos is None

    def test_falls_back_when_view_infos_missing(self):
        # Both bases AND view infos are required for the deduplicated path;
        # having only one falls back.
        md = types.SimpleNamespace()
        caches = [torch.zeros(1)]
        attach_kv_cache_bindings(
            md,
            kv_caches=caches,
            kv_cache_bases=[torch.zeros(1)],
            kv_cache_view_infos=None,
        )
        assert md.kv_caches is caches
        assert md.kv_cache_view_infos is None


class TestBuildForwardContextKwargs:
    @pytest.mark.parametrize("bases", [None, []])
    def test_empty_returns_empty_dict(self, bases):
        # No bases -> nothing injected into the forward context.
        assert build_kv_cache_forward_context_kwargs(bases) == {}

    def test_wraps_bases_as_tuple(self):
        # Bases reach the compiled forward as a tuple under 'kv_cache_bases'.
        t0, t1 = torch.zeros(1), torch.zeros(1)
        out = build_kv_cache_forward_context_kwargs([t0, t1])
        assert set(out) == {"kv_cache_bases"}
        value = out["kv_cache_bases"]
        assert isinstance(value, tuple)
        assert value[0] is t0 and value[1] is t1
