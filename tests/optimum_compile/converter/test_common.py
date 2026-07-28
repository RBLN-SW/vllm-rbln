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

from types import SimpleNamespace

import pytest

from vllm_rbln.utils.optimum.converter.common import (
    USER_MAX_NUM_BATCHED_TOKENS_KEY,
    apply_user_prefill_chunk_size,
    get_user_max_num_batched_tokens,
    is_chunked_prefill_arch,
    store_image_prefill_chunk_size,
    update_max_num_batched_tokens,
)
from vllm_rbln.utils.optimum.converter.params import RBLNParams

# Architectures picked from the RBLN registry model sets.
DECODER_ARCH = "LlamaForCausalLM"
ENC_DEC_ARCH = "WhisperForConditionalGeneration"
POOLING_ARCH = "BertModel"


def _vllm_config(
    *,
    arch: str,
    max_model_len: int = 8192,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 0,
    max_source_positions: int | None = None,
    additional_config: dict | None = None,
) -> SimpleNamespace:
    hf_config = SimpleNamespace(architectures=[arch])
    if max_source_positions is not None:
        hf_config.max_source_positions = max_source_positions
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        additional_config=additional_config,
    )


class TestIsChunkedPrefillArch:
    def test_decoder_is_chunked(self):
        assert is_chunked_prefill_arch(SimpleNamespace(architectures=[DECODER_ARCH]))

    def test_enc_dec_is_not_chunked(self):
        assert not is_chunked_prefill_arch(
            SimpleNamespace(architectures=[ENC_DEC_ARCH])
        )

    def test_pooling_is_not_chunked(self):
        assert not is_chunked_prefill_arch(
            SimpleNamespace(architectures=[POOLING_ARCH])
        )


class TestGetUserMaxNumBatchedTokens:
    def test_none_when_no_additional_config(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config=None)
        assert get_user_max_num_batched_tokens(cfg) is None

    def test_none_when_unset(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        assert get_user_max_num_batched_tokens(cfg) is None

    def test_returns_stashed_value(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 512},
        )
        assert get_user_max_num_batched_tokens(cfg) == 512


class TestUpdateMaxNumBatchedTokens:
    def test_decoder_uses_prefill_chunk_size(self):
        cfg = _vllm_config(arch=DECODER_ARCH, max_num_seqs=4)
        params = RBLNParams(prefill_chunk_size=256)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 256

    def test_decoder_chunk_smaller_than_batch_ok(self):
        # The scheduler's per-step budget is decoupled from max_num_batched_tokens,
        # so a prefill chunk smaller than max_num_seqs no longer throttles decode.
        cfg = _vllm_config(arch=DECODER_ARCH, max_num_seqs=512)
        params = RBLNParams(prefill_chunk_size=128)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 128

    def test_pooling_uses_full_prefill_budget(self):
        cfg = _vllm_config(arch=POOLING_ARCH, max_model_len=8192, max_num_seqs=4)
        params = RBLNParams(prefill_chunk_size=128)
        update_max_num_batched_tokens(cfg, params)
        # prefill_chunk_size is ignored; budget fits a full prefill.
        assert cfg.scheduler_config.max_num_batched_tokens == 8192

    def test_enc_dec_layers_source_positions(self):
        cfg = _vllm_config(
            arch=ENC_DEC_ARCH,
            max_model_len=448,
            max_num_seqs=4,
            max_source_positions=1500,
        )
        params = RBLNParams(prefill_chunk_size=128)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 1500


class TestApplyUserPrefillChunkSize:
    def test_noop_for_pooling(self):
        cfg = _vllm_config(
            arch=POOLING_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 512},
        )
        params = RBLNParams(prefill_chunk_size=128)
        apply_user_prefill_chunk_size(cfg, params, precompiled=False)
        assert params.prefill_chunk_size == 128

    def test_noop_when_user_unset(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        params = RBLNParams(prefill_chunk_size=128)
        apply_user_prefill_chunk_size(cfg, params, precompiled=False)
        assert params.prefill_chunk_size == 128

    def test_compile_path_folds_user_value(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 512},
        )
        params = RBLNParams(prefill_chunk_size=128)
        apply_user_prefill_chunk_size(cfg, params, precompiled=False)
        assert params.prefill_chunk_size == 512

    def test_compile_path_conflict_with_rbln_override(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 512},
        )
        params = RBLNParams(prefill_chunk_size=256)
        with pytest.raises(ValueError, match="Conflicting prefill chunk size"):
            apply_user_prefill_chunk_size(
                cfg, params, precompiled=False, override_prefill_chunk_size=256
            )

    def test_precompiled_conflict_raises(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 512},
        )
        params = RBLNParams(prefill_chunk_size=128)
        with pytest.raises(ValueError, match="conflicts with the compiled"):
            apply_user_prefill_chunk_size(cfg, params, precompiled=True)

    def test_precompiled_match_ok(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={USER_MAX_NUM_BATCHED_TOKENS_KEY: 128},
        )
        params = RBLNParams(prefill_chunk_size=128)
        apply_user_prefill_chunk_size(cfg, params, precompiled=True)
        assert params.prefill_chunk_size == 128


class TestStoreImagePrefillChunkSize:
    def test_stores_buckets(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        store_image_prefill_chunk_size(cfg, [1152, 640, 384])
        assert cfg.additional_config["image_prefill_chunk_size"] == [1152, 640, 384]

    def test_noop_when_none(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        store_image_prefill_chunk_size(cfg, None)
        assert "image_prefill_chunk_size" not in cfg.additional_config
