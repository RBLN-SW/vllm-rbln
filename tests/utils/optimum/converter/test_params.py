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

from vllm_rbln.utils.optimum.converter.params import (
    RBLNParams,
    _cfg_get,
    _cfg_get_submodule,
    _resolve_kvcache_block_size,
)

from .._fakes import make_hf_config, make_vllm_config


# Architecture sentinels selected from registry.py so we exercise the real
# is_*_arch dispatch rather than mocking it.
_DECODER_ARCH = ["LlamaForCausalLM"]
_POOLING_ARCH = ["BertModel"]
_ENC_DEC_ARCH = ["WhisperForConditionalGeneration"]
_MULTIMODAL_ARCH = ["LlavaForConditionalGeneration"]


def _vllm_config_with(architectures: list[str]):
    return make_vllm_config(hf_config=make_hf_config(architectures=architectures))


class TestParseDecoder:
    def test_picks_up_all_fields(self):
        cfg = _vllm_config_with(_DECODER_ARCH)
        rbln = {
            "batch_size": 4,
            "max_seq_len": 1024,
            "kvcache_block_size": 128,
            "kvcache_num_blocks": 16,
            "prefill_chunk_size": 256,
        }
        params = RBLNParams.from_rbln_config(cfg, rbln)
        assert params.batch_size == 4
        assert params.max_seq_len == 1024
        assert params.kvcache_block_size == 128
        assert params.num_blocks == 16
        assert params.prefill_chunk_size == 256

    def test_prefill_chunk_size_defaults_to_128(self):
        cfg = _vllm_config_with(_DECODER_ARCH)
        params = RBLNParams.from_rbln_config(cfg, {"kvcache_block_size": 64})
        assert params.prefill_chunk_size == 128


class TestParsePooling:
    def test_kvcache_block_size_equals_max_seq_len(self):
        cfg = _vllm_config_with(_POOLING_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "batch_size": 16,
                "max_seq_len": 512,
                "kvcache_num_blocks": 16,
            },
        )
        assert params.max_seq_len == 512
        assert params.kvcache_block_size == 512

    def test_num_blocks_falls_back_to_batch_size(self):
        cfg = _vllm_config_with(_POOLING_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {"batch_size": 8, "max_seq_len": 512},  # no kvcache_num_blocks
        )
        assert params.num_blocks == 8


class TestParseEncDec:
    def test_max_seq_len_from_dec_max_seq_len(self):
        cfg = _vllm_config_with(_ENC_DEC_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "batch_size": 4,
                "dec_max_seq_len": 448,
                "kvcache_num_blocks": 4,
            },
        )
        assert params.max_seq_len == 448
        assert params.kvcache_block_size == 448
        assert params.batch_size == 4
        assert params.num_blocks == 4


class TestParseMultimodal:
    def test_top_level_fields_used_when_present(self):
        cfg = _vllm_config_with(_MULTIMODAL_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "batch_size": 2,
                "max_seq_len": 4096,
                "kvcache_block_size": 256,
                "kvcache_num_blocks": 8,
            },
        )
        assert params.batch_size == 2
        assert params.max_seq_len == 4096
        assert params.kvcache_block_size == 256
        assert params.num_blocks == 8

    def test_falls_back_to_language_model_submodule(self):
        cfg = _vllm_config_with(_MULTIMODAL_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "language_model": {
                    "batch_size": 2,
                    "max_seq_len": 2048,
                    "kvcache_block_size": 128,
                    "kvcache_num_blocks": 16,
                }
            },
        )
        assert params.batch_size == 2
        assert params.max_seq_len == 2048
        assert params.kvcache_block_size == 128
        assert params.num_blocks == 16

    def test_falls_back_to_text_model_when_language_model_missing(self):
        cfg = _vllm_config_with(_MULTIMODAL_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "text_model": {
                    "batch_size": 2,
                    "max_seq_len": 1024,
                    "kvcache_block_size": 64,
                    "kvcache_num_blocks": 16,
                }
            },
        )
        assert params.kvcache_block_size == 64
        assert params.batch_size == 2

    def test_whisper_style_dec_max_seq_len_fallback(self):
        # Top-level multimodal config that exposes dec_max_seq_len instead
        # of max_seq_len (the Whisper FIXME path).
        cfg = _vllm_config_with(_MULTIMODAL_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg,
            {
                "batch_size": 2,
                "dec_max_seq_len": 448,
                "kvcache_block_size": 64,
                "kvcache_num_blocks": 4,
            },
        )
        assert params.max_seq_len == 448


class TestResolveKvcacheBlockSize:
    def test_both_absent_returns_none(self):
        assert _resolve_kvcache_block_size({}, arch="decoder") is None

    def test_only_kvcache_block_size_returned(self):
        assert (
            _resolve_kvcache_block_size({"kvcache_block_size": 128}, arch="decoder")
            == 128
        )

    def test_only_kvcache_partition_len_returned(self):
        assert (
            _resolve_kvcache_block_size(
                {"kvcache_partition_len": 128}, arch="decoder"
            )
            == 128
        )

    def test_both_equal_returns_value(self):
        cfg = {"kvcache_block_size": 128, "kvcache_partition_len": 128}
        assert _resolve_kvcache_block_size(cfg, arch="decoder") == 128

    def test_both_different_raises(self):
        cfg = {"kvcache_block_size": 128, "kvcache_partition_len": 256}
        with pytest.raises(AssertionError, match="decoder"):
            _resolve_kvcache_block_size(cfg, arch="decoder")


class TestCfgGet:
    def test_dict_access(self):
        assert _cfg_get({"a": 1}, "a") == 1
        assert _cfg_get({"a": 1}, "b", default=99) == 99

    def test_object_attribute_access(self):
        obj = SimpleNamespace(a=1)
        assert _cfg_get(obj, "a") == 1
        assert _cfg_get(obj, "b", default=99) == 99


class TestCfgGetSubmodule:
    def test_dict_access_returns_subdict(self):
        assert _cfg_get_submodule({"language_model": {"x": 1}}, "language_model") == {
            "x": 1
        }
        assert _cfg_get_submodule({}, "language_model") is None

    def test_object_attribute_access(self):
        obj = SimpleNamespace(language_model=SimpleNamespace(x=1))
        assert _cfg_get_submodule(obj, "language_model").x == 1
        assert _cfg_get_submodule(obj, "missing") is None


class TestTensorParallelSize:
    def test_propagates_from_top_level(self):
        cfg = _vllm_config_with(_DECODER_ARCH)
        params = RBLNParams.from_rbln_config(
            cfg, {"kvcache_block_size": 64, "tensor_parallel_size": 8}
        )
        assert params.tensor_parallel_size == 8

    def test_defaults_to_one(self):
        cfg = _vllm_config_with(_DECODER_ARCH)
        params = RBLNParams.from_rbln_config(cfg, {"kvcache_block_size": 64})
        assert params.tensor_parallel_size == 1
