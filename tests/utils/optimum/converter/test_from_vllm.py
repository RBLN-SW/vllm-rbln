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

from vllm_rbln.utils.optimum.converter.from_vllm import sync_from_vllm

from .._fakes import make_hf_config


class TestSyncFromVllmBlockSizeContract:
    """The new user-facing contract: block_size must be specified one way or
    the other, or :func:`sync_from_vllm` raises :class:`ValueError`."""

    def test_raises_when_block_size_not_user_specified_and_no_overrides(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(
            user_specified_block_size=False,
            additional_config={},
        )
        with pytest.raises(ValueError, match="block_size"):
            sync_from_vllm(cfg)

    def test_raises_when_overrides_omit_kvcache_block_size(
        self, vllm_config_factory
    ):
        # batch_size / max_seq_len present but kvcache_block_size missing;
        # user_specified_block_size still defaults to False.
        cfg = vllm_config_factory(
            user_specified_block_size=False,
            additional_config={
                "rbln_config": {"batch_size": 8, "max_seq_len": 2048}
            },
        )
        with pytest.raises(ValueError, match="block_size"):
            sync_from_vllm(cfg)

    def test_passes_when_user_specified_block_size_already_true(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(
            block_size=128,
            user_specified_block_size=True,
            additional_config={},
        )
        sync_from_vllm(cfg)
        # block_size is left intact, only user_specified flag re-asserted.
        assert cfg.cache_config.block_size == 128
        assert cfg.cache_config.user_specified_block_size is True


class TestSyncFromVllmOverridePropagation:
    def test_overrides_propagate_to_vllm_config(self, vllm_config_factory):
        cfg = vllm_config_factory(
            max_model_len=512,
            max_num_seqs=2,
            block_size=16,
            user_specified_block_size=False,
            additional_config={
                "rbln_config": {
                    "batch_size": 8,
                    "max_seq_len": 1024,
                    "kvcache_block_size": 128,
                }
            },
        )
        sync_from_vllm(cfg)

        assert cfg.scheduler_config.max_num_seqs == 8
        assert cfg.model_config.max_model_len == 1024
        assert cfg.cache_config.block_size == 128
        assert cfg.cache_config.user_specified_block_size is True


class TestSyncFromVllmMaxNumBatchedTokens:
    def test_set_to_max_of_model_len_and_num_seqs(self, vllm_config_factory):
        # max_seq_len > batch_size case
        cfg = vllm_config_factory(
            additional_config={
                "rbln_config": {
                    "batch_size": 4,
                    "max_seq_len": 1024,
                    "kvcache_block_size": 128,
                }
            },
        )
        sync_from_vllm(cfg)
        assert cfg.scheduler_config.max_num_batched_tokens == 1024

    def test_set_to_max_when_batch_size_dominates(self, vllm_config_factory):
        cfg = vllm_config_factory(
            additional_config={
                "rbln_config": {
                    "batch_size": 4096,
                    "max_seq_len": 256,
                    "kvcache_block_size": 256,
                }
            },
        )
        sync_from_vllm(cfg)
        assert cfg.scheduler_config.max_num_batched_tokens == 4096

    def test_enc_dec_lifts_to_max_source_positions(self, vllm_config_factory):
        # Whisper-style enc-dec: max_source_positions > max_seq_len triggers
        # the lift inside update_max_num_batched_tokens.
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=1500,
        )
        cfg = vllm_config_factory(
            hf_config=hf,
            additional_config={
                "rbln_config": {
                    "batch_size": 4,
                    "dec_max_seq_len": 448,
                    "kvcache_num_blocks": 4,
                }
            },
        )
        sync_from_vllm(cfg)
        # _parse_enc_dec forces kvcache_block_size = dec_max_seq_len = 448,
        # so max_num_batched_tokens starts at max(448, 4) = 448 and is then
        # bumped to max_source_positions (1500).
        assert cfg.scheduler_config.max_num_batched_tokens == 1500

    def test_enc_dec_keeps_value_when_max_source_positions_smaller(
        self, vllm_config_factory
    ):
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=128,
        )
        cfg = vllm_config_factory(
            hf_config=hf,
            additional_config={
                "rbln_config": {
                    "batch_size": 4,
                    "dec_max_seq_len": 448,
                    "kvcache_num_blocks": 4,
                }
            },
        )
        sync_from_vllm(cfg)
        # No lift: max(dec_max_seq_len=448, batch_size=4) wins.
        assert cfg.scheduler_config.max_num_batched_tokens == 448
