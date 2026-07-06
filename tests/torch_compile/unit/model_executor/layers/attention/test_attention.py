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

"""Unit tests for the RBLN attention KV-cache overrides.

These verify that the KV cache reaches the attention op as a graph input
(resolved from attention metadata) and that the connector path never reads the
layer's embedded KV cache:

* ``_resolve_kv_cache`` picks the deduplicated-base view vs. the direct
  per-layer cache.
* ``_rbln_get_attention_context`` returns the layer context with ``kv_cache``
  left as None (so the connector wrapper does not bake it into the graph).
* ``_rbln_unified_attention_with_output`` drives ``impl.forward`` with the
  metadata-resolved KV cache.
"""

from unittest.mock import MagicMock, patch

import torch

import vllm_rbln.model_executor.layers.attention.attention as attn_mod


class _FakeForwardContext:
    def __init__(
        self,
        *,
        attn_metadata=None,
        no_compile_layers=None,
        slot_mapping=None,
        additional_kwargs=None,
    ) -> None:
        self.attn_metadata = attn_metadata
        self.no_compile_layers = no_compile_layers or {}
        self.slot_mapping = {} if slot_mapping is None else slot_mapping
        self.additional_kwargs = additional_kwargs or {}


class _FakeAttnMetadata:
    def __init__(self, *, kv_caches=None, kv_cache_view_infos=None) -> None:
        self.kv_caches = kv_caches
        self.kv_cache_view_infos = kv_cache_view_infos


def _patch_forward_context(fc):
    return patch.object(attn_mod, "get_forward_context", return_value=fc)


# ---------------------------------------------------------------------------
# _resolve_kv_cache
# ---------------------------------------------------------------------------
class TestResolveKvCache:
    def test_direct_branch_indexes_metadata_kv_caches(self):
        kv0, kv1 = torch.zeros(2), torch.ones(2)
        md = _FakeAttnMetadata(kv_caches=[kv0, kv1])
        with _patch_forward_context(_FakeForwardContext()):
            assert attn_mod._resolve_kv_cache(md, 1) is kv1

    def test_empty_bases_falls_back_to_direct(self):
        kv0 = torch.zeros(2)
        md = _FakeAttnMetadata(kv_caches=[kv0], kv_cache_view_infos=["vi"])
        fc = _FakeForwardContext(additional_kwargs={"kv_cache_bases": []})
        with _patch_forward_context(fc):
            assert attn_mod._resolve_kv_cache(md, 0) is kv0

    def test_bases_branch_materializes_view_for_layer(self):
        bases = [object()]
        view_infos = ["vi0", "vi1"]
        md = _FakeAttnMetadata(kv_caches=None, kv_cache_view_infos=view_infos)
        fc = _FakeForwardContext(additional_kwargs={"kv_cache_bases": bases})
        sentinel = torch.empty(0)
        with (
            _patch_forward_context(fc),
            patch.object(
                attn_mod, "materialize_kv_cache_view", return_value=sentinel
            ) as mat,
        ):
            out = attn_mod._resolve_kv_cache(md, 1)
        assert out is sentinel
        mat.assert_called_once_with(bases, "vi1")


# ---------------------------------------------------------------------------
# _rbln_get_attention_context
# ---------------------------------------------------------------------------
class TestRblnGetAttentionContext:
    def _run(self, fc, layer_name="L0"):
        with _patch_forward_context(fc):
            return attn_mod._rbln_get_attention_context(layer_name)

    def test_returns_none_kv_cache(self):
        md, layer = _FakeAttnMetadata(), object()
        fc = _FakeForwardContext(
            attn_metadata={"L0": md},
            no_compile_layers={"L0": layer},
            slot_mapping={"L0": "sm"},
        )
        attn_metadata, attn_layer, kv_cache, slot = self._run(fc)
        assert kv_cache is None
        assert attn_metadata is md
        assert attn_layer is layer
        assert slot == "sm"

    def test_dict_metadata_indexed_by_layer(self):
        md = _FakeAttnMetadata()
        fc = _FakeForwardContext(
            attn_metadata={"L0": md}, no_compile_layers={"L0": object()}
        )
        assert self._run(fc)[0] is md

    def test_list_metadata_uses_base_model_dict(self):
        # Speculative decoding passes list[dict]; [0] is the base-model dict.
        md = _FakeAttnMetadata()
        fc = _FakeForwardContext(
            attn_metadata=[{"L0": md}], no_compile_layers={"L0": object()}
        )
        assert self._run(fc)[0] is md

    def test_plain_metadata_passed_through(self):
        md = _FakeAttnMetadata()
        fc = _FakeForwardContext(attn_metadata=md, no_compile_layers={"L0": object()})
        assert self._run(fc)[0] is md

    def test_missing_slot_mapping_entry_is_none(self):
        fc = _FakeForwardContext(
            attn_metadata=_FakeAttnMetadata(),
            no_compile_layers={"L0": object()},
            slot_mapping={},
        )
        assert self._run(fc)[3] is None


# ---------------------------------------------------------------------------
# _rbln_unified_attention_with_output
# ---------------------------------------------------------------------------
class TestRblnUnifiedAttentionWithOutput:
    def test_drives_impl_forward_with_resolved_kv_cache(self):
        q, k, v, out = (torch.empty(1) for _ in range(4))
        kv = torch.zeros(2)
        md = _FakeAttnMetadata(kv_caches=[kv])
        layer = MagicMock()
        layer.layer_index = 0
        fc = _FakeForwardContext(
            attn_metadata={"L0": md},
            no_compile_layers={"L0": layer},
            slot_mapping={"L0": None},
        )
        with _patch_forward_context(fc):
            attn_mod._rbln_unified_attention_with_output(q, k, v, out, "L0")

        layer.impl.forward.assert_called_once_with(
            layer,
            q,
            k,
            v,
            kv,
            md,
            output=out,
            output_scale=None,
            output_block_scale=None,
        )
