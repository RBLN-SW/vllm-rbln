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

from vllm_rbln.utils.optimum.converter import from_optimum
from vllm_rbln.utils.optimum.converter.from_optimum import (
    _keep_only_device_keys,
    sync_from_optimum,
    update_num_blocks,
)

from .._fakes import make_hf_config


def _full_rbln_config(**overrides):
    """Return a minimal-but-complete decoder rbln_config used by sync tests."""
    base = {
        "batch_size": 8,
        "max_seq_len": 2048,
        "kvcache_block_size": 128,
        "kvcache_num_blocks": 32,
        "prefill_chunk_size": 128,
        "tensor_parallel_size": 2,
    }
    base.update(overrides)
    return base


class TestKeepOnlyDeviceKeys:
    def test_keeps_top_level_devices(self):
        assert _keep_only_device_keys({"devices": [0, 1]}) == {"devices": [0, 1]}

    def test_drops_top_level_non_device_keys(self):
        assert _keep_only_device_keys({"batch_size": 4, "max_seq_len": 1024}) == {}

    def test_keeps_nested_devices_through_named_dict(self):
        cfg = {
            "language_model": {"devices": [0, 1], "batch_size": 4},
            "vision_model": {"devices": [2, 3]},
            "noise": "drop me",
        }
        assert _keep_only_device_keys(cfg) == {
            "language_model": {"devices": [0, 1]},
            "vision_model": {"devices": [2, 3]},
        }

    def test_drops_nested_dict_with_no_devices(self):
        cfg = {"language_model": {"batch_size": 4, "max_seq_len": 1024}}
        assert _keep_only_device_keys(cfg) == {}

    def test_empty_dict_in_empty_dict_out(self):
        assert _keep_only_device_keys({}) == {}

    def test_devices_key_value_passes_through_untyped(self):
        # `devices` value is preserved verbatim regardless of type.
        for v in ([0], (0, 1), "device-id", 7):
            assert _keep_only_device_keys({"devices": v}) == {"devices": v}


class TestSyncFromOptimumAssertions:
    @pytest.mark.parametrize(
        "missing_key",
        ["batch_size", "max_seq_len", "kvcache_block_size", "kvcache_num_blocks"],
    )
    def test_asserts_on_missing_required_field(
        self, vllm_config_factory, missing_key
    ):
        cfg = vllm_config_factory(
            additional_config={},
            num_gpu_blocks_override=None,
        )
        rbln_cfg = _full_rbln_config()
        del rbln_cfg[missing_key]
        with pytest.raises(AssertionError):
            sync_from_optimum(cfg, rbln_cfg)


class TestSyncFromOptimumFieldUpdates:
    def test_updates_all_fields(self, vllm_config_factory, monkeypatch):
        # Pin env so we can detect the side-effect write deterministically.
        monkeypatch.setattr(from_optimum.envs, "VLLM_RBLN_TP_SIZE", 1)

        cfg = vllm_config_factory(
            max_model_len=512,
            max_num_seqs=2,
            block_size=16,
            additional_config={
                "rbln_config": {
                    "devices": [0, 1],
                    "batch_size": 4,  # non-device key, should be dropped
                }
            },
        )
        rbln_cfg = _full_rbln_config(
            batch_size=8,
            max_seq_len=2048,
            kvcache_block_size=128,
            kvcache_num_blocks=32,
            tensor_parallel_size=4,
        )
        sync_from_optimum(cfg, rbln_cfg)

        assert cfg.scheduler_config.max_num_seqs == 8
        assert cfg.model_config.max_model_len == 2048
        assert cfg.cache_config.block_size == 128
        # rbln_config in additional_config is reduced to device-only keys.
        assert cfg.additional_config["rbln_config"] == {"devices": [0, 1]}
        # max_num_batched_tokens uses the pre-update max_model_len (512)
        # and the freshly-updated max_num_seqs (8); max() = 512. The follow-up
        # update_max_num_batched_tokens call is a no-op for non-enc-dec.
        assert cfg.scheduler_config.max_num_batched_tokens == 512
        # tp size synced to envs.
        assert from_optimum.envs.VLLM_RBLN_TP_SIZE == 4


class TestUpdateNumBlocks:
    def test_full_block_available_returns_n_plus_1(self, vllm_config_factory):
        # block_size=128, max_model_len=1024, max_num_seqs=4 -> ideal=32.
        cfg = vllm_config_factory(
            block_size=128, max_model_len=1024, max_num_seqs=4
        )
        update_num_blocks(cfg, 32)
        assert cfg.cache_config.num_gpu_blocks == 33
        assert cfg.additional_config["num_blocks_synced"] is True

    def test_not_full_returns_n_minus_1_then_plus_1(self, vllm_config_factory):
        cfg = vllm_config_factory(
            block_size=128, max_model_len=1024, max_num_seqs=4
        )
        update_num_blocks(cfg, 31)
        # blk_ratio == 1, so (31-1)*1 + 1 == 31.
        assert cfg.cache_config.num_gpu_blocks == 31

    def test_idempotent(self, vllm_config_factory):
        cfg = vllm_config_factory(
            block_size=128, max_model_len=1024, max_num_seqs=4
        )
        update_num_blocks(cfg, 32)
        first = cfg.cache_config.num_gpu_blocks
        # Second call must be a no-op even with a wildly different value.
        update_num_blocks(cfg, 9999)
        assert cfg.cache_config.num_gpu_blocks == first

    def test_num_gpu_blocks_override_used_and_rewritten(
        self, vllm_config_factory
    ):
        cfg = vllm_config_factory(
            block_size=128,
            max_model_len=1024,
            max_num_seqs=4,
            num_gpu_blocks_override=20,
        )
        # The rbln_config-supplied num_blocks (=999) must be ignored when
        # num_gpu_blocks_override is set.
        update_num_blocks(cfg, 999)
        # ideal = 32, override = 20 < 32 -> not full -> (20-1)*1+1 = 20.
        assert cfg.cache_config.num_gpu_blocks == 20
        # Override is replaced by the adjusted value...
        assert cfg.cache_config.num_gpu_blocks_override == 20
        # ...and the original override is preserved separately.
        assert cfg.additional_config["num_blocks_override"] == 20

    def test_prefix_caching_uses_block_ratio(self, vllm_config_factory):
        # ob_size=128, ib_size=32 -> blk_ratio=4
        # max_model_len=1024, attn_block_size=128 -> blocks_per_seq=8
        # max_num_seqs=2 -> ideal=16
        cfg = vllm_config_factory(
            block_size=32,
            max_model_len=1024,
            max_num_seqs=2,
            enable_prefix_caching=True,
            additional_config={"attn_block_size": 128},
        )
        update_num_blocks(cfg, 16)  # full
        assert cfg.cache_config.num_gpu_blocks == 16 * 4 + 1

    def test_prefix_caching_not_full(self, vllm_config_factory):
        cfg = vllm_config_factory(
            block_size=32,
            max_model_len=1024,
            max_num_seqs=2,
            enable_prefix_caching=True,
            additional_config={"attn_block_size": 128},
        )
        update_num_blocks(cfg, 8)  # not full (need 16)
        assert cfg.cache_config.num_gpu_blocks == (8 - 1) * 4 + 1
