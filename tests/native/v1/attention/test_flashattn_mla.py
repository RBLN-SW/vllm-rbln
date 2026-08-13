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

import pytest
import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.native.v1.attention.utils import make_mla_impl
from tests.native.vllm_config import make_vllm_config
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.backends.mla.flashattn_mla import (
    RBLNFlashAttnMLABackend,
    RBLNFlashAttnMLAImpl,
)


@pytest.fixture(scope="module")
def cfg():
    return make_vllm_config()


# The RBLN MLA backend: static contract, registration, V-up projection math and
# the interface stubs. The paged-MLA kernel is a compile-time stub, so forward()
# numerics are out of scope.


class TestMlaBackendStatic:
    def test_get_name(self):
        # The name vLLM selects this backend by.
        assert RBLNFlashAttnMLABackend.get_name() == "FLASH_ATTN_MLA"

    def test_kv_cache_shape(self):
        # (num_blocks, block_size, head_size); num_kv_heads is unused. Distinct
        # values catch any argument re-ordering.
        shape = RBLNFlashAttnMLABackend.get_kv_cache_shape(
            num_blocks=7, block_size=11, num_kv_heads=3, head_size=5
        )
        assert shape == (7, 11, 5)

    def test_supported_head_sizes(self):
        # MLA is fixed to the 576-wide latent head.
        assert RBLNFlashAttnMLABackend.get_supported_head_sizes() == [576]

    def test_impl_and_builder_wiring(self):
        # The backend hands back RBLN's MLA impl and the shared metadata builder.
        assert RBLNFlashAttnMLABackend.get_impl_cls() is RBLNFlashAttnMLAImpl
        assert (
            RBLNFlashAttnMLABackend.get_builder_cls()
            is RBLNFlashAttentionMetadataBuilder
        )

    def test_class_level_capabilities(self):
        # Declared dtype / kv-cache-dtype support and the no-output-buffer flag.
        assert RBLNFlashAttnMLABackend.supported_dtypes == [
            torch.float16,
            torch.bfloat16,
        ]
        assert RBLNFlashAttnMLABackend.supported_kv_cache_dtypes == ["auto"]
        assert RBLNFlashAttnMLABackend.accept_output_buffer is False


class TestMlaBackendRegistration:
    def test_flash_attn_mla_resolves_to_rbln_backend(self):
        # conformance: @register_backend wired RBLN's MLA class into vLLM's
        # registry as the FLASH_ATTN_MLA override. Breaks on registry drift.
        assert AttentionBackendEnum.FLASH_ATTN_MLA.is_overridden()
        assert (
            AttentionBackendEnum.FLASH_ATTN_MLA.get_class() is RBLNFlashAttnMLABackend
        )


class TestMlaVUpProj:
    def test_projects_and_flattens_heads(self):
        # [B, H, S, lora] @ W_UV[1, H, lora, v] -> [B, S, H*v]. Only v_head_dim
        # is read, so a bare instance sidesteps the __init__ validation.
        impl = object.__new__(RBLNFlashAttnMLAImpl)
        impl.v_head_dim = 5
        x = torch.randn(2, 3, 4, 7)  # [B, H, S, kv_lora_rank]
        w_uv = torch.randn(1, 3, 7, 5)  # [1, H, kv_lora_rank, v_head_dim]
        out = impl._v_up_proj(x, w_uv)
        expected = torch.matmul(x, w_uv).transpose(1, 2).reshape(2, 4, 3 * 5)
        assert out.shape == (2, 4, 15)
        assert torch.allclose(out, expected)


class TestMlaImplStubs:
    def test_can_return_lse_for_decode(self):
        # RBLN MLA reports it can return LSE for the decode path.
        assert RBLNFlashAttnMLAImpl.can_return_lse_for_decode is True

    def test_forward_mha_not_implemented(self):
        # RBLN MLA uses forward() directly; the MLA interface hooks are stubs.
        impl = object.__new__(RBLNFlashAttnMLAImpl)
        with pytest.raises(NotImplementedError):
            impl.forward_mha(None, None, None, None, None, None, None)

    def test_forward_mqa_not_implemented(self):
        impl = object.__new__(RBLNFlashAttnMLAImpl)
        with pytest.raises(NotImplementedError):
            impl.forward_mqa(None, None, None, None)

    def test_process_weights_after_loading_is_noop(self):
        # RBLN overrides the base MLA weight-absorption with a no-op; pin the
        # signature (accepts an act_dtype) and that it does nothing / returns None.
        impl = object.__new__(RBLNFlashAttnMLAImpl)
        assert impl.process_weights_after_loading(torch.float16) is None


@pytest.mark.maybe_use_device
class TestMlaImplInit:
    # __init__ validation guards. Valid args (head_size 576, all-None) construct;
    # each override trips one guard.
    def test_valid_construction(self, cfg):
        impl = make_mla_impl(cfg)
        assert impl.v_head_dim == 128
        assert impl.kv_lora_rank == 512

    @pytest.mark.parametrize(
        "override",
        [{"alibi_slopes": [0.1]}, {"sliding_window": 8}, {"logits_soft_cap": 1.0}],
    )
    def test_unsupported_features_rejected(self, cfg, override):
        # alibi_slopes / sliding_window / logits_soft_cap are all unsupported.
        with pytest.raises(NotImplementedError, match="does not support"):
            make_mla_impl(cfg, **override)

    def test_non_decoder_attn_type_rejected(self, cfg):
        with pytest.raises(NotImplementedError, match="decoder"):
            make_mla_impl(cfg, attn_type=AttentionType.ENCODER)

    def test_quantized_kv_cache_not_supported(self, cfg):
        with pytest.raises(NotImplementedError, match="FP8"):
            make_mla_impl(cfg, kv_cache_dtype="fp8")

    def test_kv_sharing_not_supported(self, cfg):
        with pytest.raises(NotImplementedError, match="KV sharing"):
            make_mla_impl(cfg, kv_sharing_target_layer_name="model.layers.0.self_attn")

    def test_unsupported_head_size_raises(self, cfg):
        # MLA only supports head_size 576.
        with pytest.raises(ValueError, match="not supported"):
            make_mla_impl(cfg, head_size=128)
