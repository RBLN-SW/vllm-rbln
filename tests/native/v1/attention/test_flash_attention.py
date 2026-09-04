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


from dataclasses import fields

import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_rbln.envs as envs
import vllm_rbln.v1.attention.backends.flash_attention as flash_attention
from tests.native.v1.attention.utils import (
    make_builder,
    make_common_attn_metadata,
    make_impl,
)
from tests.native.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionBackend,
    RBLNFlashAttentionImpl,
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)

# RBLN's flash-attention backend: the static contract, __post_init__ dtype
# handling, registration, and the metadata builder. build() uses a small config
# so masks stay inspectable; is_causal is read at __init__, so set the env first.

MAX_LEN = 32  # max_model_len -> attention-mask width
CHUNK = 8  # max_num_batched_tokens -> prefill chunk size


def _metadata(**overrides) -> RBLNFlashAttentionMetadata:
    # Minimal valid metadata; overrides feed the fields __post_init__ touches.
    return RBLNFlashAttentionMetadata(
        seq_lens=torch.tensor([[1]], dtype=torch.int64),
        block_tables=torch.zeros(1, dtype=torch.int64),
        is_prefill=True,
        **overrides,
    )


def _cam(*, num_reqs, query_start_loc, seq_lens, block_table):
    return make_common_attn_metadata(
        num_reqs=num_reqs,
        query_start_loc=torch.tensor(query_start_loc),
        seq_lens=torch.tensor(seq_lens),
        block_table_tensor=torch.tensor(block_table),
    )


def _unexpected(name: str):
    def called(*args):
        raise AssertionError(f"{name} must not be reached")

    return called


def _lower_triangular(n: int) -> torch.Tensor:
    # causal_mask = 1 - triu(ones, diag=1): 1 where key <= query, else 0.
    return 1 - torch.triu(torch.ones(n, n), diagonal=1)


@pytest.fixture
def on_cr13(monkeypatch):
    # The append path is CR13-only, and the impl reads the NPU name at __init__.
    monkeypatch.setattr(current_platform, "get_device_name", lambda *a: "RBLN-CR13")


@pytest.fixture
def off_cr13(monkeypatch):
    monkeypatch.setattr(current_platform, "get_device_name", lambda *a: "RBLN-CA25")


@pytest.fixture
def custom_kernel_on(monkeypatch):
    # USE_CUSTOM_KERNEL resolves from RBLN_USE_CUSTOM_KERNEL, not the
    # VLLM_RBLN_-prefixed name (pinned in test_envs).
    monkeypatch.setenv("RBLN_USE_CUSTOM_KERNEL", "1")


@pytest.fixture(scope="module")
def cfg():
    return make_vllm_config(max_model_len=MAX_LEN, max_num_batched_tokens=CHUNK)


@pytest.fixture(scope="module")
def cfg_square():
    # block_size == max_model_len, so is_normal can be True.
    return make_vllm_config(max_model_len=64, block_size=64)


class TestFlashAttentionBackendStatic:
    def test_kv_cache_shape(self):
        # Layout is (2, num_blocks, num_kv_heads, 1, block_size, head_size).
        # Distinct primes catch any argument re-ordering.
        shape = RBLNFlashAttentionBackend.get_kv_cache_shape(
            num_blocks=7, block_size=11, num_kv_heads=3, head_size=5
        )
        assert shape == (2, 7, 3, 1, 11, 5)

    def test_supported_head_sizes(self):
        # Pins the supported set; a change here is a deliberate capability shift.
        assert RBLNFlashAttentionBackend.get_supported_head_sizes() == [
            32,
            64,
            80,
            96,
            128,
            160,
            192,
            224,
            256,
        ]

    def test_get_name(self):
        # The name vLLM selects this backend by.
        assert RBLNFlashAttentionBackend.get_name() == "FLASH_ATTN"

    def test_impl_and_builder_wiring(self):
        # The backend must hand back RBLN's impl/builder, not upstream's.
        assert RBLNFlashAttentionBackend.get_impl_cls() is RBLNFlashAttentionImpl
        assert (
            RBLNFlashAttentionBackend.get_builder_cls()
            is RBLNFlashAttentionMetadataBuilder
        )


class TestBackendRegistration:
    def test_flash_attn_resolves_to_rbln_backend(self):
        # conformance: @register_backend wired RBLN's class into vLLM's registry
        # as the FLASH_ATTN override. Breaks if the registry API drifts.
        assert AttentionBackendEnum.FLASH_ATTN.is_overridden()
        assert AttentionBackendEnum.FLASH_ATTN.get_class() is RBLNFlashAttentionBackend


class TestFlashAttentionMetadataPostInit:
    def test_custom_kernel_off_leaves_dtype_untouched(self, monkeypatch):
        # Without the custom kernel, __post_init__ returns early: no casting.
        monkeypatch.delenv("RBLN_USE_CUSTOM_KERNEL", raising=False)
        assert _metadata().seq_lens.dtype == torch.int64

    def test_custom_kernel_on_casts_seq_lens_to_int32(self, custom_kernel_on):
        # The custom-kernel path casts seq_lens; absent cache tensors stay None.
        md = _metadata()
        assert md.seq_lens.dtype == torch.int32
        assert md.cache_seq_lens is None
        assert md.cache_offsets is None

    # cache_seq_lens and cache_offsets share the exact same (buggy) handling, so
    # every case runs against both to catch a fix/regression applied to only one.
    CACHE_FIELDS = ["cache_seq_lens", "cache_offsets"]

    @pytest.mark.parametrize("field", CACHE_FIELDS)
    def test_multielement_cache_field_raises_ambiguous(self, custom_kernel_on, field):
        # KNOWN BUG pinned: `if self.<field>` on a multi-element tensor raises.
        # The real SWA/decode path emits [batch, 1], so it fires at batch >= 2.
        with pytest.raises(RuntimeError, match="ambiguous"):
            _metadata(**{field: torch.tensor([[1], [2]], dtype=torch.int64)})

    @pytest.mark.parametrize("field", CACHE_FIELDS)
    def test_single_zero_cache_field_is_silently_dropped(self, custom_kernel_on, field):
        # TODO(RBLN): KNOWN BUG pinned — a 1-element tensor of value 0 is falsy,
        # so a valid all-zero value is discarded to None instead of cast.
        md = _metadata(**{field: torch.tensor([[0]], dtype=torch.int64)})
        assert getattr(md, field) is None

    @pytest.mark.parametrize("field", CACHE_FIELDS)
    def test_single_nonzero_cache_field_is_cast(self, custom_kernel_on, field):
        # The only shape that survives correctly: 1 element, != 0.
        md = _metadata(**{field: torch.tensor([[3]], dtype=torch.int64)})
        assert getattr(md, field) is not None
        assert getattr(md, field).dtype == torch.int32


class TestMetadataBuilderConstants:
    # reorder_batch / use_cascade_attention are constant False today; anchored so
    # that adding real logic later cannot slip through untested.
    def test_reorder_batch_is_false(self, cfg):
        assert make_builder(cfg).reorder_batch(None, None) is False

    def test_use_cascade_attention_is_false(self, cfg):
        assert make_builder(cfg).use_cascade_attention() is False


class TestBuildCommonIndices:
    def test_seq_idx_picks_positions_at_query_starts(self, cfg):
        # seq_idx = positions[query_start_loc[:num_reqs]]; distinct positions
        # avoid a coincidental match with the indices.
        builder = make_builder(cfg)
        positions = torch.arange(20) * 10
        md = builder.build(
            _cam(
                num_reqs=2,
                query_start_loc=[0, 3, 7],
                seq_lens=[5, 8],
                block_table=[[1], [2]],
            ),
            positions,
            batch_pad=2,
            is_prefill=False,
        )
        # positions[[0, 3]] = [0, 30]
        assert md.seq_lens.reshape(-1).tolist() == [0, 30]


class TestBuildPrefillCausal:
    def test_no_mask_and_block_tables_flattened(self, cfg):
        # Causal prefill: no attention mask is built, and block_tables is
        # flattened to 1D by taking row 0.
        builder = make_builder(cfg)  # is_causal True by default
        md = builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 4],
                seq_lens=[4],
                block_table=[[7, 8, 9]],
            ),
            torch.arange(4),
            batch_pad=1,
            is_prefill=True,
        )
        assert md.attn_masks is None
        assert md.block_tables.tolist() == [7, 8, 9]  # block_table_tensor[0], 1D
        assert md.seq_lens.reshape(-1).tolist() == [0]  # seq_idx, unpadded
        assert md.cache_seq_lens is None and md.swa_attn_masks is None


class TestBuildPrefillNonCausal:
    def _mask(self, config, monkeypatch, *, positions):
        monkeypatch.setenv("VLLM_RBLN_FLASH_CAUSAL_ATTN", "0")
        builder = make_builder(config)
        return builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 4],
                seq_lens=[4],
                block_table=[[7, 8, 9]],
            ),
            positions,
            batch_pad=1,
            is_prefill=True,
        ).attn_masks

    def test_step_below_chunk_places_triangle_only(self, cfg, monkeypatch):
        # step = positions[0] = 0 (< chunk): no cached region, triangle at [0:chunk]
        mask = self._mask(cfg, monkeypatch, positions=torch.arange(4))
        assert mask.shape == (1, 1, 1, CHUNK, MAX_LEN)
        expected = torch.zeros(1, 1, 1, CHUNK, MAX_LEN)
        expected[..., 0:CHUNK] = _lower_triangular(CHUNK)
        assert torch.equal(mask.float(), expected)

    def test_step_at_chunk_fills_cached_region(self, cfg, monkeypatch):
        # step = positions[0] = CHUNK (>= chunk): [:step] all attend, triangle after
        mask = self._mask(cfg, monkeypatch, positions=torch.arange(4) + CHUNK)
        expected = torch.zeros(1, 1, 1, CHUNK, MAX_LEN)
        expected[..., :CHUNK] = 1
        expected[..., CHUNK : 2 * CHUNK] = _lower_triangular(CHUNK)
        assert torch.equal(mask.float(), expected)

    @pytest.mark.parametrize(
        "eager, expected_dtype",
        [(False, torch.float32), (True, torch.float16)],
    )
    def test_mask_dtype_follows_enforce_eager(
        self, cfg, monkeypatch, eager, expected_dtype
    ):
        # float16 under enforce_eager, float32 otherwise. enforce_eager needs
        # device tensors, so that case is skipped on the cpu lane.
        if eager and not envs.VLLM_RBLN_USE_DEVICE_TENSOR:
            pytest.skip("enforce_eager=True requires VLLM_RBLN_USE_DEVICE_TENSOR=1")
        config = (
            make_vllm_config(
                max_model_len=MAX_LEN,
                max_num_batched_tokens=CHUNK,
                enforce_eager=True,
            )
            if eager
            else cfg
        )
        mask = self._mask(config, monkeypatch, positions=torch.arange(4))
        assert mask.dtype == expected_dtype


class TestBuildDecodeCausal:
    def test_no_mask_and_padding(self, cfg):
        # Causal decode: no mask; seq_idx and block_tables padded to batch_pad.
        builder = make_builder(cfg)  # causal
        md = builder.build(
            _cam(
                num_reqs=2,
                query_start_loc=[0, 1, 2],
                seq_lens=[3, 5],
                block_table=[[1], [2]],
            ),
            torch.arange(10),
            batch_pad=4,
            is_prefill=False,
        )
        assert md.attn_masks is None
        # seq_idx = positions[[0, 1]] = [0, 1], padded with 0 to batch_pad
        assert md.seq_lens.reshape(-1).tolist() == [0, 1, 0, 0]
        assert md.block_tables.reshape(-1).tolist() == [1, 2, 0, 0]


class TestBuildDecodeNonCausal:
    def test_per_request_attend_length(self, cfg, monkeypatch):
        # Decode mask: each batch row attends positions 0..seq_len; rows beyond
        # the request count stay zero.
        monkeypatch.setenv("VLLM_RBLN_FLASH_CAUSAL_ATTN", "0")
        builder = make_builder(cfg)
        md = builder.build(
            _cam(
                num_reqs=2,
                query_start_loc=[0, 1, 2],
                seq_lens=[3, 5],
                block_table=[[1], [2]],
            ),
            torch.arange(10),
            batch_pad=4,
            is_prefill=False,
        )
        mask = md.attn_masks
        assert mask.shape == (4, 1, 1, 1, MAX_LEN)
        expected = torch.zeros(4, 1, 1, 1, MAX_LEN)
        expected[0, ..., :4] = 1  # seq_len 3 -> attend 0..3
        expected[1, ..., :6] = 1  # seq_len 5 -> attend 0..5
        # rows 2, 3 stay 0 (loop runs over seq_lens only)
        assert torch.equal(mask.float(), expected)


class TestBuildSlidingWindowPrefill:
    @pytest.mark.parametrize(
        "seq_len, exp_cache, exp_offset",
        [(10, 4, 8), (5, 1, 5), (8, 4, 8)],  # clamped, unclamped, boundary
    )
    def test_cache_seq_lens_and_offsets(self, cfg, seq_len, exp_cache, exp_offset):
        # cache_seq_lens = clamp(num_computed, window); cache_offsets adds
        # query_lens; local_block_tables is the first block; no SWA/attn mask.
        window = 4
        builder = make_builder(cfg, sliding_window=window, appends_kv=False)
        md = builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 4],  # query_len 4 -> num_computed = seq_len - 4
                seq_lens=[seq_len],
                block_table=[[7, 8, 9]],
            ),
            torch.arange(4),
            batch_pad=1,
            is_prefill=True,
        )
        assert md.cache_seq_lens.reshape(-1).tolist() == [exp_cache]
        assert md.cache_offsets.reshape(-1).tolist() == [exp_offset]
        assert md.local_block_tables.reshape(-1).tolist() == [7]  # block[..., :1]
        assert md.swa_attn_masks is None  # prefill
        assert md.attn_masks is None  # causal

    def test_noncausal_still_sets_swa_fields(self, cfg, monkeypatch):
        # SWA block is independent of is_causal: chunked mask AND SWA fields set.
        monkeypatch.setenv("VLLM_RBLN_FLASH_CAUSAL_ATTN", "0")
        builder = make_builder(cfg, sliding_window=4, appends_kv=False)
        md = builder.build(
            _cam(num_reqs=1, query_start_loc=[0, 4], seq_lens=[10], block_table=[[7]]),
            torch.arange(4),
            batch_pad=1,
            is_prefill=True,
        )
        assert md.attn_masks is not None
        assert md.cache_seq_lens is not None

    def test_respects_num_reqs_slice(self, cfg):
        # Only seq_lens[:num_reqs] feeds the SWA fields: req0 clamps to 1 while
        # req1 would clamp to 4, so the value identifies which was used.
        builder = make_builder(cfg, sliding_window=4, appends_kv=False)
        md = builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 4, 6],  # query_seq_lens = [4, 2]
                seq_lens=[5, 99],  # num_computed = [1, 97]; second must be ignored
                block_table=[[7, 8, 9]],
            ),
            torch.arange(4),
            batch_pad=1,
            is_prefill=True,
        )
        assert md.cache_seq_lens.reshape(-1).tolist() == [1]  # clamp(1, 4), req0 only


class TestBuildSlidingWindowDecode:
    def test_padding_swa_mask_and_local_block_tables(self, cfg):
        # Decode SWA: cache_* padded to batch_pad, swa mask marks positions
        # <= cache_seq_len, local_block_tables is the first block per row.
        window = 4
        builder = make_builder(cfg, sliding_window=window, appends_kv=False)
        md = builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 1],  # query_len 1 -> num_computed = 3 - 1 = 2
                seq_lens=[3],
                block_table=[[5]],
            ),
            torch.arange(10),
            batch_pad=2,
            is_prefill=False,
        )
        # clamp(2, 4) = 2, padded to batch_pad=2 -> [2, 0]
        assert md.cache_seq_lens.reshape(-1).tolist() == [2, 0]
        # cache_offsets = cache_seq_lens + query_lens(=1) = 3, padded -> [3, 0]
        assert md.cache_offsets.reshape(-1).tolist() == [3, 0]
        assert md.swa_attn_masks.shape == (2, 1, 1, window)
        # row0 (cache=2): arange(4) <= 2 -> [1,1,1,0]; row1 (cache=0): [1,0,0,0]
        expected = torch.tensor([[[[1.0, 1, 1, 0]]], [[[1.0, 0, 0, 0]]]])
        assert torch.equal(md.swa_attn_masks.float(), expected)
        # decode block_tables padded to [[5], [0]], then [..., :1]
        assert md.local_block_tables.reshape(-1).tolist() == [5, 0]


class TestBuildSlidingWindowAppend:
    def test_builds_what_a_full_attention_group_builds(self, cfg):
        # An appended cache is upstream's SlidingWindowSpec, and the window is
        # resolved inside the op: the builder emits nothing that depends on it.
        cam = _cam(
            num_reqs=1, query_start_loc=[0, 1], seq_lens=[7], block_table=[[5, 6]]
        )
        args = (cam, torch.arange(10))
        kwargs = dict(batch_pad=2, is_prefill=False)
        swa = make_builder(cfg, sliding_window=4).build(*args, **kwargs)
        full = make_builder(cfg).build(*args, **kwargs)

        for field in fields(RBLNFlashAttentionMetadata):
            swa_value = getattr(swa, field.name)
            full_value = getattr(full, field.name)
            if isinstance(swa_value, torch.Tensor):
                assert torch.equal(swa_value, full_value)
            else:
                assert swa_value == full_value


class TestBuildOutputAssembly:
    @pytest.mark.parametrize("is_prefill", [True, False])
    def test_is_prefill_reflected(self, cfg, is_prefill):
        # build() passes is_prefill straight through onto the metadata.
        builder = make_builder(cfg)
        md = builder.build(
            _cam(
                num_reqs=1,
                query_start_loc=[0, 1],
                seq_lens=[1],
                block_table=[[1, 2, 3]],
            ),
            torch.arange(4),
            batch_pad=1,
            is_prefill=is_prefill,
        )
        assert md.is_prefill is is_prefill


class TestStage:
    def test_none_passthrough(self, cfg):
        # None in -> None out; there is nothing to stage.
        assert make_builder(cfg)._stage(None, "x") is None

    def test_copies_values_onto_self_device(self, cfg):
        # The tensor is copied into a buffer living on self.device.
        staged = make_builder(cfg)._stage(torch.tensor([1, 2, 3]), "x")
        assert staged.tolist() == [1, 2, 3]
        assert staged.device.type == "cpu"  # self.device

    def test_reuses_buffer_for_same_key(self, cfg):
        # Same (slot, shape, dtype) key returns the one buffer, overwriting it.
        builder = make_builder(cfg)
        first = builder._stage(torch.tensor([1, 2, 3]), "x")
        second = builder._stage(torch.tensor([4, 5, 6]), "x")
        assert first is second
        assert second.tolist() == [4, 5, 6]

    def test_distinct_slots_do_not_alias(self, cfg):
        # cache_seq_lens and cache_offsets share shape/dtype; the slot keeps them
        # from sharing one buffer (the _stage docstring warning).
        builder = make_builder(cfg)
        t = torch.tensor([1, 2, 3])
        assert builder._stage(t, "cache_seq_lens") is not builder._stage(
            t, "cache_offsets"
        )

    def test_different_shape_gets_new_buffer(self, cfg):
        # A different shape is a different key, so it gets its own buffer.
        builder = make_builder(cfg)
        assert builder._stage(torch.tensor([1, 2, 3]), "x") is not builder._stage(
            torch.tensor([1, 2]), "x"
        )

    def test_buffer_reused_across_builds(self, cfg):
        # The "overwritten by next build()" contract: same slot/shape/dtype across
        # builds returns the same buffer, so the earlier result is clobbered.
        builder = make_builder(cfg)

        def run(offset):
            return builder.build(
                _cam(
                    num_reqs=1,
                    query_start_loc=[0, 4],
                    seq_lens=[4],
                    block_table=[[1, 2, 3]],
                ),
                torch.arange(4) + offset,
                batch_pad=1,
                is_prefill=True,
            ).seq_lens

        first = run(0)  # seq_idx = [0]
        second = run(10)  # seq_idx = [10]
        assert first is second
        assert first.reshape(-1).tolist() == [10]


@pytest.mark.maybe_use_device
class TestFlashImplInit:
    # __init__ validation guards and derived attributes. Each guard test sets the
    # other args valid so the intended check fires (the guards are ordered).
    def test_kv_sharing_not_supported(self, cfg):
        with pytest.raises(NotImplementedError, match="KV sharing"):
            make_impl(cfg, kv_sharing_target_layer_name="model.layers.0.self_attn")

    def test_num_heads_must_divide_num_kv_heads(self, cfg):
        with pytest.raises(AssertionError):
            make_impl(cfg, num_heads=8, num_kv_heads=3)

    def test_unsupported_head_size_raises(self, cfg):
        with pytest.raises(ValueError, match="not supported"):
            make_impl(cfg, head_size=100)

    def test_non_fp8_quantized_kv_cache_not_supported(self, cfg):
        # Quantized KV cache dtypes other than fp8 are rejected; fp8 variants
        # are allowed and resolve to the real fp8 element dtype.
        with pytest.raises(NotImplementedError, match="does not support"):
            make_impl(cfg, kv_cache_dtype="nvfp4")

    @pytest.mark.parametrize(
        "kv_cache_dtype,expected",
        [
            ("auto", None),
            ("fp8", torch.float8_e4m3fn),  # upstream alias of e4m3
            ("fp8_e4m3", torch.float8_e4m3fn),
            ("fp8_e5m2", torch.float8_e5m2),
        ],
    )
    def test_fp8_cache_dtype_mapping(self, kv_cache_dtype, expected):
        # _fp8_cache_dtype resolves the real element dtype the uint8 fp8-KV
        # container holds; forward hands it to the compiled custom op as its
        # last argument (None on the non-fp8 "auto" path).
        from vllm_rbln.v1.attention.backends.flash_attention import _fp8_cache_dtype

        assert _fp8_cache_dtype(kv_cache_dtype) == expected

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
    def test_fp8_kv_cache_accepted(self, cfg, kv_cache_dtype):
        # every fp8 dtype in supported_kv_cache_dtypes passes the __init__
        # quantization guard.
        impl = make_impl(cfg, kv_cache_dtype=kv_cache_dtype)
        assert impl.kv_cache_dtype == kv_cache_dtype

    def test_fp8_with_sliding_window_raises(self, cfg):
        # forward() would route to the sliding-window ops, which take no
        # dequant scales and would read the uint8 container as raw bytes.
        with pytest.raises(NotImplementedError, match="flash causal"):
            make_impl(cfg, kv_cache_dtype="fp8", sliding_window=16)

    def test_fp8_normal_attention_raises(self, cfg_square):
        # cfg_square makes is_normal True, routing to the scale-less
        # causal_attention_naive ops.
        with pytest.raises(NotImplementedError, match="flash causal"):
            make_impl(cfg_square, kv_cache_dtype="fp8")

    def test_fp8_non_causal_raises(self, cfg, monkeypatch):
        # is_causal off routes to the plain attention ops.
        monkeypatch.setenv("VLLM_RBLN_FLASH_CAUSAL_ATTN", "0")
        with pytest.raises(NotImplementedError, match="flash causal"):
            make_impl(cfg, kv_cache_dtype="fp8")

    def test_fp8_with_custom_kernel_raises(self, cfg, custom_kernel_on):
        # The rbln_triton_ops variants drop the scales even on the flash
        # causal path.
        with pytest.raises(NotImplementedError, match="CUSTOM_KERNEL"):
            make_impl(cfg, kv_cache_dtype="fp8")

    def test_logits_soft_cap_disabled_with_warning(self, cfg, monkeypatch):
        # RBLN does not support a logits soft cap: it warns and forces it to 0.
        from vllm_rbln.v1.attention.backends import flash_attention

        recorded = []
        monkeypatch.setattr(
            flash_attention.logger,
            "warning_once",
            lambda msg, *a, **k: recorded.append(msg),
        )
        impl = make_impl(cfg, logits_soft_cap=30.0)
        assert impl.logits_soft_cap == 0
        assert recorded and "soft cap" in recorded[0].lower()

    def test_num_queries_per_kv_derived(self, cfg):
        # num_queries_per_kv = num_heads // num_kv_heads.
        assert make_impl(cfg, num_heads=8, num_kv_heads=4).num_queries_per_kv == 2

    def test_sinks_1d_reshaped_to_2d(self, cfg):
        # A 1-D sinks tensor [num_heads] gains a trailing dim -> [num_heads, 1].
        assert make_impl(cfg, sinks=torch.zeros(8)).sinks.shape == (8, 1)

    def test_sinks_2d_left_as_is(self, cfg):
        assert make_impl(cfg, sinks=torch.zeros(8, 2)).sinks.shape == (8, 2)

    def test_sinks_head_count_mismatch_asserts(self, cfg):
        # sinks.shape[0] must equal num_heads (8 by default).
        with pytest.raises(AssertionError):
            make_impl(cfg, sinks=torch.zeros(4))

    def test_is_normal_true_when_block_equals_max_and_no_sinks(self, cfg_square):
        assert make_impl(cfg_square).is_normal is True

    def test_is_normal_false_when_block_differs(self, cfg):
        # cfg has block_size (1024) != max_model_len (32).
        assert make_impl(cfg).is_normal is False

    def test_is_normal_false_when_sinks_present(self, cfg_square):
        assert make_impl(cfg_square, sinks=torch.zeros(8)).is_normal is False


@pytest.mark.maybe_use_device
class TestForwardSlidingWindow:
    """Which kernel a sliding-window layer reaches is decided by the NPU, and
    the two take different inputs."""

    WINDOW = 4
    HEADS, DIM = 8, 128  # make_impl defaults; num_queries_per_kv is 1

    def _forward(self, cfg, monkeypatch, target, metadata):
        impl = make_impl(cfg, sliding_window=self.WINDOW)
        b_size = metadata.seq_lens.shape[0]
        recorded = []

        def record(*args):
            recorded.append(args)
            return torch.zeros(b_size, self.HEADS, 1, 1, self.DIM)

        for name in (
            "sliding_window_attention_v1",
            "sliding_window_attention_naive_prefill",
            "sliding_window_attention_naive_decode",
        ):
            monkeypatch.setattr(
                flash_attention, name, record if name == target else _unexpected(name)
            )
        qkv = torch.zeros(b_size, self.HEADS, self.DIM)
        impl.forward(
            None,  # layer: only the flash causal branch reads it
            qkv,
            qkv,
            qkv,
            torch.zeros(1, 1, 1, 1, self.WINDOW, 1),  # kv_cache: last dim but one
            metadata,
            torch.zeros(b_size, self.HEADS, self.DIM),
        )
        return recorded[0]

    def _decode_metadata(self):
        return RBLNFlashAttentionMetadata(
            seq_lens=torch.tensor([[6], [0]]),
            block_tables=torch.tensor([[7, 8, 9], [0, 0, 0]]),
            is_prefill=False,
            cache_seq_lens=torch.tensor([[4], [0]]),
            cache_offsets=torch.tensor([[5], [0]]),
            local_block_tables=torch.tensor([[7], [0]]),
        )

    def test_cr13_appends_with_the_position_and_the_whole_table(
        self, cfg, monkeypatch, on_cr13
    ):
        # Neither the position nor the table is cut on the way in: the op
        # resolves the window from them itself.
        md = self._decode_metadata()
        _q, _k, _v, _cache, seq_idx, _scale, tables, window, sinks = self._forward(
            cfg, monkeypatch, "sliding_window_attention_v1", md
        )
        assert seq_idx is md.seq_lens  # absolute, not clamped to the window
        assert tables is md.block_tables  # the whole table, not its first column
        assert window == self.WINDOW
        assert sinks is None

    def test_elsewhere_the_shift_kernel_takes_the_fill_and_one_block(
        self, cfg, monkeypatch, off_cr13
    ):
        # The shift path is untouched: the clamped fill, its end, and the single
        # block the window lives in.
        md = self._decode_metadata()
        *_, cache_seq_len, cache_offset, _scale, tables, _mask, _sinks = self._forward(
            cfg, monkeypatch, "sliding_window_attention_naive_decode", md
        )
        assert cache_seq_len is md.cache_seq_lens
        assert cache_offset is md.cache_offsets
        assert tables is md.local_block_tables

    def test_prefill_on_cr13_takes_the_same_call(self, cfg, monkeypatch, on_cr13):
        # One op for both phases; its 1-D block table is reshaped by the op.
        md = RBLNFlashAttentionMetadata(
            seq_lens=torch.tensor([[6]]),
            block_tables=torch.tensor([7, 8, 9]),
            is_prefill=True,
        )
        *_, tables, window, _sinks = self._forward(
            cfg, monkeypatch, "sliding_window_attention_v1", md
        )
        assert tables.tolist() == [7, 8, 9]
        assert window == self.WINDOW
