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


import pytest

from vllm_rbln.utils.optimum.converter.params import RBLNParams


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
        # tensor_parallel_size is populated by the caller
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
