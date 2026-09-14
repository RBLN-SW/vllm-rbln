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

from unittest.mock import patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_rbln.utils.optimum.converter.from_optimum import sync_from_optimum


def _vllm_config() -> VllmConfig:
    # Force the cache-miss branch so building the config does not look for a
    # compiled artifact. The test calls sync_from_optimum itself.
    with patch(
        "vllm_rbln.utils.optimum.converter.dispatch._resolve_rbln_config",
        return_value=None,
    ):
        return VllmConfig(
            model_config=ModelConfig(model="facebook/opt-125m", dtype=torch.float),
            scheduler_config=SchedulerConfig(
                max_num_seqs=10,
                max_num_batched_tokens=128,
                max_model_len=128,
                is_encoder_decoder=False,
            ),
            cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
            additional_config={"optimum_overrides": {"prefill_chunk_size": 4}},
        )


def test_compiled_num_devices_lands_in_optimum_config(monkeypatch):
    """The compiled model decides num_devices. It must reach the config every
    worker receives, not a module attribute of the front-end process."""
    monkeypatch.setenv("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", "1")
    vllm_config = _vllm_config()
    assert vllm_config.additional_config.num_devices_per_local_rank == 1

    compiled = {
        "kvcache_num_blocks": 16,
        "batch_size": 4,
        "max_seq_len": 128,
        "kvcache_block_size": 128,
        "prefill_chunk_size": 128,
        "num_devices": 2,
    }
    sync_from_optimum(vllm_config, compiled)

    assert vllm_config.additional_config.num_devices_per_local_rank == 2
