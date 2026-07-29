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
    _num_devices_of,
    _resolve_num_devices,
)


class TestParseDecoder:
    def test_full_dict_config(self):
        cfg = {
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_block_size": 4096,
            "prefill_chunk_size": 256,
        }
        params = RBLNParams._parse_decoder(cfg)
        assert params.num_blocks == 16
        assert params.batch_size == 4
        assert params.max_seq_len == 8192
        assert params.kvcache_block_size == 4096
        assert params.prefill_chunk_size == 256
        # num_devices is populated by the caller
        # (`from_rbln_config`), not `_parse_decoder`,
        # so it is not asserted here.

    def test_none_dict_config(self):
        cfg: dict = {}
        params = RBLNParams._parse_decoder(cfg)
        assert params.num_blocks is None
        assert params.batch_size is None
        assert params.max_seq_len is None
        assert params.prefill_chunk_size == 128

    def test_default_prefill_chunk_size(self):
        cfg = {
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_block_size": 4096,
        }
        params = RBLNParams._parse_decoder(cfg)
        assert params.prefill_chunk_size == 128

    def test_kvcache_partition_len(self):
        cfg = {
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_partition_len": 4096,
        }
        params = RBLNParams._parse_decoder(cfg)
        assert params.kvcache_block_size == 4096

    def test_error_duplicated_block_size(self):
        cfg = {
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 4096,
            "kvcache_block_size": 4096,
            "kvcache_partition_len": 2048,
        }
        with pytest.raises(AssertionError, match="kvcache_partition_len"):
            RBLNParams._parse_decoder(cfg)


class TestParseEncDec:
    def test_kvcache_block_size_equals_dec_max_seq_len(self):
        cfg = {
            "kvcache_num_blocks": 4,
            "batch_size": 1,
            "dec_max_seq_len": 448,
        }
        params = RBLNParams._parse_enc_dec(cfg)
        assert params.num_blocks == 4
        assert params.batch_size == 1
        assert params.max_seq_len == 448
        assert params.kvcache_block_size == 448


class TestParsePooling:
    def test_uses_explicit_kvcache_num_blocks(self):
        cfg = {
            "max_seq_len": 512,
            "batch_size": 8,
            "kvcache_num_blocks": 16,
        }
        params = RBLNParams._parse_pooling(cfg)
        assert params.num_blocks == 16
        assert params.batch_size == 8
        assert params.max_seq_len == 512
        assert params.kvcache_block_size == 512

    def test_falls_back_num_blocks_to_batch_size(self):
        cfg = {"max_seq_len": 512, "batch_size": 8}
        params = RBLNParams._parse_pooling(cfg)
        assert params.num_blocks == 8

    def test_kvcache_block_size_equals_max_seq_len(self):
        cfg = {"max_seq_len": 512, "batch_size": 8}
        params = RBLNParams._parse_pooling(cfg)
        assert params.kvcache_block_size == 512


class TestParseMultimodal:
    def test_top_level_fields_present(self):
        cfg = {
            "kvcache_num_blocks": 32,
            "batch_size": 1,
            "max_seq_len": 4096,
            "kvcache_block_size": 128,
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.num_blocks == 32
        assert params.batch_size == 1
        assert params.max_seq_len == 4096
        assert params.kvcache_block_size == 128

    def test_uses_language_model_submodule(self):
        cfg = {
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_block_size": 64,
            },
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.num_blocks == 16
        assert params.batch_size == 2
        assert params.max_seq_len == 2048
        assert params.kvcache_block_size == 64

    def test_uses_text_model_submodule(self):
        cfg = {
            "text_model": {
                "kvcache_num_blocks": 8,
                "batch_size": 1,
                "max_seq_len": 1024,
                "kvcache_block_size": 32,
            },
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.num_blocks == 8
        assert params.batch_size == 1
        assert params.max_seq_len == 1024
        assert params.kvcache_block_size == 32

    def test_submodule_resolves_partition_len(self):
        cfg = {
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.kvcache_block_size == 128

    def test_submodule_uses_submodule_batch_size(self):
        cfg = {
            "batch_size": 1,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.batch_size == 2


class TestImagePrefillChunkSize:
    def _cfg(self, value):
        return {
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 1,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
                "image_prefill_chunk_size": value,
            },
        }

    def test_gemma3_int_normalized_to_list(self):
        params = RBLNParams._parse_multimodal(self._cfg(256))
        assert params.image_prefill_chunk_size == [256]

    def test_gemma4_list_passes_through(self):
        params = RBLNParams._parse_multimodal(self._cfg([1152, 640, 384]))
        assert params.image_prefill_chunk_size == [1152, 640, 384]

    def test_absent_is_none(self):
        cfg = {
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 1,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        params = RBLNParams._parse_multimodal(cfg)
        assert params.image_prefill_chunk_size is None

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            RBLNParams._parse_multimodal(self._cfg("256"))

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            RBLNParams._parse_multimodal(self._cfg(True))


class TestNumDevicesOf:
    def test_top_level_num_devices(self):
        assert _num_devices_of({"num_devices": 4}) == 4

    def test_falls_back_to_compile_cfgs(self):
        cfg = {"_compile_cfgs": [{"num_devices": 8}]}
        assert _num_devices_of(cfg) == 8

    def test_top_level_takes_precedence_over_compile_cfgs(self):
        cfg = {"num_devices": 2, "_compile_cfgs": [{"num_devices": 8}]}
        assert _num_devices_of(cfg) == 2

    def test_first_compile_cfg_with_num_devices_wins(self):
        cfg = {"_compile_cfgs": [{}, {"num_devices": 8}, {"num_devices": 16}]}
        assert _num_devices_of(cfg) == 8

    def test_returns_none_when_absent(self):
        assert _num_devices_of({}) is None

    def test_returns_none_when_compile_cfgs_lacks_num_devices(self):
        assert _num_devices_of({"_compile_cfgs": [{}, {}]}) is None

    def test_non_int_raises(self):
        with pytest.raises(AssertionError, match="num_devices must be an int"):
            _num_devices_of({"num_devices": "4"})

    def test_non_positive_raises(self):
        with pytest.raises(AssertionError, match="positive integer"):
            _num_devices_of({"num_devices": 0})

    def test_compile_cfgs_non_int_raises(self):
        with pytest.raises(AssertionError, match="num_devices must be an int"):
            _num_devices_of({"_compile_cfgs": [{"num_devices": 1.5}]})

    def test_compile_cfgs_non_positive_raises(self):
        with pytest.raises(AssertionError, match="positive integer"):
            _num_devices_of({"_compile_cfgs": [{"num_devices": -1}]})


class TestResolveNumDevices:
    def test_top_level(self):
        assert _resolve_num_devices({"num_devices": 4}) == 4

    def test_defaults_to_one_when_absent(self):
        assert _resolve_num_devices({}) == 1

    def test_reads_from_language_model_submodule(self):
        cfg = {"language_model": {"num_devices": 8}}
        assert _resolve_num_devices(cfg) == 8

    def test_reads_from_text_model_submodule(self):
        cfg = {"text_model": {"num_devices": 8}}
        assert _resolve_num_devices(cfg) == 8

    def test_language_model_takes_precedence_over_top_level(self):
        cfg = {"num_devices": 1, "language_model": {"num_devices": 8}}
        assert _resolve_num_devices(cfg) == 8

    def test_language_model_preferred_over_text_model(self):
        cfg = {
            "language_model": {"num_devices": 8},
            "text_model": {"num_devices": 16},
        }
        assert _resolve_num_devices(cfg) == 8

    def test_reads_from_submodule_compile_cfgs(self):
        cfg = {"language_model": {"_compile_cfgs": [{"num_devices": 8}]}}
        assert _resolve_num_devices(cfg) == 8

    def test_falls_back_to_top_level_when_submodule_lacks_num_devices(self):
        cfg = {"num_devices": 4, "language_model": {"batch_size": 2}}
        assert _resolve_num_devices(cfg) == 4


class TestFromRblnConfigDtype:
    # `additional_config["rbln_config"]` has the same shape as the compiled
    # `rbln_config.json`, so dtype is parsed identically on both sync paths.
    def _vllm_config(self):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
            )
        )

    def test_reads_dtype(self):
        cfg = {"batch_size": 4, "max_seq_len": 8192, "dtype": "bfloat16"}
        params = RBLNParams.from_rbln_config(self._vllm_config(), cfg)
        assert params.dtype == "bfloat16"

    def test_none_when_absent(self):
        cfg = {"batch_size": 4, "max_seq_len": 8192}
        params = RBLNParams.from_rbln_config(self._vllm_config(), cfg)
        assert params.dtype is None
