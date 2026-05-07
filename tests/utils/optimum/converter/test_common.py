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

from vllm_rbln.utils.optimum.converter.common import (
    update_block_size,
    update_max_num_batched_tokens,
)

from .._fakes import make_hf_config


class TestUpdateBlockSizeNoPrefixCaching:
    def test_sets_block_size_and_user_specified_flag(self, vllm_config_factory):
        cfg = vllm_config_factory(
            block_size=16,
            enable_prefix_caching=False,
            user_specified_block_size=False,
        )
        update_block_size(cfg, kvcache_block_size=128, prefill_chunk_size=128)
        assert cfg.cache_config.block_size == 128
        assert cfg.cache_config.user_specified_block_size is True

    def test_noop_when_block_size_already_matches(self, vllm_config_factory):
        cfg = vllm_config_factory(block_size=128, enable_prefix_caching=False)
        update_block_size(cfg, kvcache_block_size=128, prefill_chunk_size=128)
        assert cfg.cache_config.block_size == 128


class TestUpdateBlockSizePrefixCaching:
    def test_default_prefix_block_size_is_prefill_chunk_size(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(
            enable_prefix_caching=True,
            additional_config={},
        )
        update_block_size(cfg, kvcache_block_size=256, prefill_chunk_size=64)
        assert cfg.cache_config.block_size == 64
        assert cfg.additional_config["attn_block_size"] == 256

    def test_user_supplied_prefix_block_size_when_valid(self, vllm_config_factory):
        cfg = vllm_config_factory(
            enable_prefix_caching=True,
            additional_config={"prefix_block_size": 128},
        )
        update_block_size(cfg, kvcache_block_size=256, prefill_chunk_size=64)
        assert cfg.cache_config.block_size == 128
        assert cfg.additional_config["attn_block_size"] == 256

    def test_raises_when_prefix_not_divisible_by_prefill_chunk(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(
            enable_prefix_caching=True,
            additional_config={"prefix_block_size": 130},
        )
        with pytest.raises(ValueError, match="prefix_block_size"):
            update_block_size(cfg, kvcache_block_size=256, prefill_chunk_size=128)

    def test_raises_when_prefix_greater_than_kvcache(self, vllm_config_factory):
        cfg = vllm_config_factory(
            enable_prefix_caching=True,
            additional_config={"prefix_block_size": 512},
        )
        with pytest.raises(ValueError, match="kvcache_block_size"):
            update_block_size(cfg, kvcache_block_size=256, prefill_chunk_size=128)

    def test_raises_when_kvcache_not_divisible_by_prefix(self, vllm_config_factory):
        # prefix unset -> defaults to prefill_chunk_size (=128).
        # 300 % 128 != 0 triggers the third ValueError branch.
        cfg = vllm_config_factory(
            enable_prefix_caching=True,
            additional_config={},
        )
        with pytest.raises(ValueError, match="kvcache_block_size"):
            update_block_size(cfg, kvcache_block_size=300, prefill_chunk_size=128)


class TestUpdateMaxNumBatchedTokens:
    def test_non_enc_dec_is_noop(self, vllm_config_factory):
        cfg = vllm_config_factory(
            max_num_batched_tokens=512,
            hf_config=make_hf_config(architectures=["LlamaForCausalLM"]),
        )
        update_max_num_batched_tokens(cfg, max_model_len=2048)
        assert cfg.scheduler_config.max_num_batched_tokens == 512

    def test_enc_dec_keeps_max_model_len_when_source_positions_smaller(
        self, vllm_config_factory
    ):
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=128,
        )
        cfg = vllm_config_factory(hf_config=hf, max_num_batched_tokens=64)
        update_max_num_batched_tokens(cfg, max_model_len=2048)
        # max_source_positions <= max_model_len: stays at max_model_len.
        assert cfg.scheduler_config.max_num_batched_tokens == 2048

    def test_enc_dec_lifts_to_source_positions_when_larger(
        self, vllm_config_factory
    ):
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=1500,
        )
        cfg = vllm_config_factory(hf_config=hf, max_num_batched_tokens=512)
        update_max_num_batched_tokens(cfg, max_model_len=448)
        assert cfg.scheduler_config.max_num_batched_tokens == 1500
