# Copyright 2025 Rebellions Inc. All rights reserved.
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

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.configuration.runtime_config import (
    apply_runtime_config,
    get_runtime_num_blocks,
    prepare_runtime_config,
)

pytestmark = pytest.mark.usefixtures("set_npu_env_var")

# Architectures picked from the RBLN registry model sets.
DECODER_ARCH = "LlamaForCausalLM"
ENC_DEC_ARCH = "WhisperForConditionalGeneration"
POOLING_ARCH = "BertModel"


def _vllm_config(
    *, arch: str, additional_config: dict | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch]), max_model_len=8192
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=0, max_num_scheduled_tokens=None
        ),
        additional_config=build_optimum_rbln_config(additional_config),
    )


@pytest.fixture
def apply_config():
    def run(rbln_config, *, arch=DECODER_ARCH, precompiled=False):
        cfg = _vllm_config(arch=arch)
        cfg.model_config.dtype = torch.float32
        cfg.cache_config = SimpleNamespace(
            block_size=16384,
            user_specified_block_size=False,
            enable_prefix_caching=False,
            num_gpu_blocks=None,
            num_gpu_blocks_override=None,
            user_specified_mamba_block_size=False,
            mamba_block_size=None,
            gpu_memory_utilization=0.9,
        )
        prepared = prepare_runtime_config(cfg, rbln_config, precompiled=precompiled)
        apply_runtime_config(cfg, prepared)
        return cfg

    return run


class TestApplyDecoder:
    def test_full_dict_config(self, apply_config):
        cfg = {
            "num_devices": 1,
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_block_size": 4096,
            "prefill_chunk_size": 256,
        }
        synced = apply_config(cfg, arch=DECODER_ARCH)
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 16
        assert synced.scheduler_config.max_num_seqs == 4
        assert synced.model_config.max_model_len == 8192
        assert synced.cache_config.block_size == 4096
        assert synced.scheduler_config.max_num_batched_tokens == 8192
        assert synced.additional_config.prefill_chunk_size == 256

    def test_missing_required_fields_are_rejected(self, apply_config):
        with pytest.raises(AssertionError):
            apply_config({})

    def test_kvcache_partition_len(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_partition_len": 4096,
        }
        synced = apply_config(cfg, arch=DECODER_ARCH)
        assert synced.cache_config.block_size == 4096

    def test_error_duplicated_block_size(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 4096,
            "kvcache_block_size": 4096,
            "kvcache_partition_len": 2048,
        }
        with pytest.raises(AssertionError, match="kvcache_partition_len"):
            apply_config(cfg, arch=DECODER_ARCH)


class TestApplyEncDec:
    def test_kvcache_block_size_equals_dec_max_seq_len(self, apply_config):
        cfg = {
            "num_devices": 1,
            "kvcache_num_blocks": 4,
            "batch_size": 1,
            "dec_max_seq_len": 448,
            "kvcache_block_size": 448,
        }
        synced = apply_config(cfg, arch=ENC_DEC_ARCH)
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 4
        assert synced.scheduler_config.max_num_seqs == 1
        assert synced.model_config.max_model_len == 448
        assert synced.cache_config.block_size == 448


class TestApplyPooling:
    def test_reads_partition_size_from_compile_config(self, apply_config):
        synced = apply_config(
            {
                "num_devices": 1,
                "max_seq_len": 8192,
                "batch_size": 4,
                "kvcache_partition_len": 1024,
            },
            arch="Qwen3Model",
        )
        assert synced.cache_config.block_size == 1024

    def test_uses_explicit_kvcache_num_blocks(self, apply_config):
        cfg = {
            "num_devices": 1,
            "max_seq_len": 512,
            "batch_size": 8,
            "kvcache_num_blocks": 16,
            "kvcache_block_size": 512,
        }
        synced = apply_config(cfg, arch="Qwen3Model")
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 16
        assert synced.scheduler_config.max_num_seqs == 8
        assert synced.model_config.max_model_len == 512
        assert synced.cache_config.block_size == 512

    def test_encoder_pooling_batch_maps_to_vllm_blocks(self, apply_config):
        cfg = {"num_devices": 1, "max_seq_len": 512, "batch_size": 8}
        synced = apply_config(cfg, arch=POOLING_ARCH)
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 8

    def test_kvcache_block_size_equals_max_seq_len(self, apply_config):
        cfg = {"num_devices": 1, "max_seq_len": 512, "batch_size": 8}
        synced = apply_config(cfg, arch=POOLING_ARCH)
        assert synced.cache_config.block_size == 512


class TestApplyMultimodal:
    def test_reads_submodule_lengths_with_a_top_level_block_size(self, apply_config):
        synced = apply_config(
            {
                "num_devices": 1,
                "prefill_chunk_size": 128,
                "kvcache_block_size": 8192,
                "language_model": {"batch_size": 4, "max_seq_len": 8192},
            },
            arch="PaliGemmaForConditionalGeneration",
        )
        assert synced.scheduler_config.max_num_seqs == 4
        assert synced.model_config.max_model_len == 8192
        assert synced.cache_config.block_size == 8192

    def test_reads_submodule_lengths_without_a_partition(self, apply_config):
        synced = apply_config(
            {
                "num_devices": 1,
                "prefill_chunk_size": 128,
                "language_model": {"batch_size": 4, "max_seq_len": 8192},
                "kvcache_block_size": 8192,
            },
            arch="PaliGemmaForConditionalGeneration",
        )
        assert synced.scheduler_config.max_num_seqs == 4
        assert synced.model_config.max_model_len == 8192

    @pytest.mark.parametrize("as_object", [False, True])
    def test_partial_submodules_resolve_each_field_by_precedence(
        self, as_object, apply_config
    ):
        cfg: Any = {
            "num_devices": 1,
            "batch_size": 1,
            "max_seq_len": 8192,
            "kvcache_block_size": 8192,
            "kvcache_num_blocks": 8,
            "prefill_chunk_size": 128,
            "image_prefill_chunk_size": 128,
            "language_model": {"batch_size": 2, "max_seq_len": None},
            "text_model": {
                "batch_size": 3,
                "max_seq_len": 4096,
                "kvcache_partition_len": 4096,
                "prefill_chunk_size": 256,
                "image_prefill_chunk_size": [512, 256],
            },
        }
        if as_object:
            cfg = SimpleNamespace(
                **{
                    key: SimpleNamespace(**value) if isinstance(value, dict) else value
                    for key, value in cfg.items()
                }
            )
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert synced.scheduler_config.max_num_seqs == 2
        assert synced.model_config.max_model_len == 4096
        assert synced.cache_config.block_size == 4096
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 8
        assert synced.scheduler_config.max_num_batched_tokens == 4096
        assert synced.additional_config.prefill_chunk_size == 256
        assert synced.additional_config.image_prefill_chunk_size == [512, 256]

    def test_top_level_fields_present(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "kvcache_num_blocks": 32,
            "batch_size": 1,
            "max_seq_len": 4096,
            "kvcache_block_size": 128,
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 32
        assert synced.scheduler_config.max_num_seqs == 1
        assert synced.model_config.max_model_len == 4096
        assert synced.cache_config.block_size == 128

    def test_uses_language_model_submodule(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_block_size": 64,
            },
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 16
        assert synced.scheduler_config.max_num_seqs == 2
        assert synced.model_config.max_model_len == 2048
        assert synced.cache_config.block_size == 64

    def test_uses_text_model_submodule(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "text_model": {
                "kvcache_num_blocks": 8,
                "batch_size": 1,
                "max_seq_len": 1024,
                "kvcache_block_size": 32,
            },
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert get_runtime_num_blocks(synced.model_config.hf_config, cfg) == 8
        assert synced.scheduler_config.max_num_seqs == 1
        assert synced.model_config.max_model_len == 1024
        assert synced.cache_config.block_size == 32

    def test_submodule_resolves_partition_len(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert synced.cache_config.block_size == 128

    def test_submodule_uses_submodule_batch_size(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "batch_size": 1,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 2,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert synced.scheduler_config.max_num_seqs == 2


class TestImagePrefillChunkSize:
    def _cfg(self, value):
        return {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 1,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
                "image_prefill_chunk_size": value,
            },
        }

    def test_gemma3_int_normalized_to_list(self, apply_config):
        synced = apply_config(self._cfg(256), arch="PaliGemmaForConditionalGeneration")
        assert synced.additional_config.image_prefill_chunk_size == [256]

    def test_gemma4_list_passes_through(self, apply_config):
        synced = apply_config(
            self._cfg([1152, 640, 384]), arch="PaliGemmaForConditionalGeneration"
        )
        assert synced.additional_config.image_prefill_chunk_size == [1152, 640, 384]

    def test_absent_is_none(self, apply_config):
        cfg = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "language_model": {
                "kvcache_num_blocks": 16,
                "batch_size": 1,
                "max_seq_len": 2048,
                "kvcache_partition_len": 128,
            },
        }
        synced = apply_config(cfg, arch="PaliGemmaForConditionalGeneration")
        assert synced.additional_config.image_prefill_chunk_size is None

    def test_invalid_type_raises(self, apply_config):
        with pytest.raises(TypeError):
            apply_config(self._cfg("256"), arch="PaliGemmaForConditionalGeneration")

    def test_bool_rejected(self, apply_config):
        with pytest.raises(TypeError):
            apply_config(self._cfg(True), arch="PaliGemmaForConditionalGeneration")


class TestCompiledDtype:
    @pytest.mark.parametrize("dtype", ["bfloat16", None])
    def test_applies_compiled_dtype_when_present(self, apply_config, dtype):
        compiled: dict[str, Any] = {
            "num_devices": 1,
            "prefill_chunk_size": 128,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_block_size": 8192,
            "kvcache_num_blocks": 4,
        }
        if dtype is not None:
            compiled["dtype"] = dtype
        cfg = apply_config(compiled, precompiled=True)
        assert cfg.model_config.dtype == (torch.bfloat16 if dtype else torch.float32)


def _engine_config() -> VllmConfig:
    # Force the cache-miss branch so building the config does not look for a
    # compiled artifact. The test calls apply_runtime_config itself.
    with patch(
        "vllm_rbln.utils.optimum.configuration.sync.load_compiled_rbln_config",
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


def test_compiled_num_devices_lands_in_the_config(monkeypatch):
    """The compiled model decides num_devices. It has to reach the config every
    worker receives, not a module attribute of the front-end process."""
    monkeypatch.setenv("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", "1")
    vllm_config = _engine_config()
    assert vllm_config.additional_config.num_devices_per_local_rank == 1

    compiled = {
        "kvcache_num_blocks": 16,
        "batch_size": 4,
        "max_seq_len": 128,
        "kvcache_block_size": 128,
        "prefill_chunk_size": 4,
        "num_devices": 2,
    }
    prepared = prepare_runtime_config(vllm_config, compiled, precompiled=True)
    apply_runtime_config(vllm_config, prepared)

    assert vllm_config.additional_config.num_devices_per_local_rank == 2


@pytest.mark.parametrize("arch", [DECODER_ARCH, "PaliGemmaForConditionalGeneration"])
@pytest.mark.parametrize("precompiled", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_missing_prefill_chunk_size_is_rejected(
    apply_config, arch, precompiled, explicit_none
):
    selected: dict[str, Any] = {
        "num_devices": 1,
        "batch_size": 4,
        "max_seq_len": 8192,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 16,
    }
    if explicit_none:
        selected["prefill_chunk_size"] = None
    with pytest.raises(ValueError, match="prefill_chunk_size must be specified"):
        apply_config(selected, arch=arch, precompiled=precompiled)


@pytest.mark.parametrize("arch", [DECODER_ARCH, "PaliGemmaForConditionalGeneration"])
def test_apply_uses_selected_chunk_without_querying_the_npu(
    monkeypatch, apply_config, arch
):
    monkeypatch.setattr(
        "vllm_rbln.platform.RblnPlatform.get_device_name",
        lambda: pytest.fail("sync must use the selected config's prefill chunk size"),
    )
    selected = {
        "num_devices": 1,
        "batch_size": 4,
        "max_seq_len": 8192,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": 256,
    }
    cfg = apply_config(selected, arch=arch, precompiled=True)
    assert cfg.scheduler_config.max_num_batched_tokens == 8192
    assert cfg.additional_config.prefill_chunk_size == 256


@pytest.mark.parametrize(
    "arch",
    [DECODER_ARCH, ENC_DEC_ARCH, "Qwen3Model", "PaliGemmaForConditionalGeneration"],
)
@pytest.mark.parametrize("missing", ["kvcache_block_size", "num_devices"])
@pytest.mark.parametrize("precompiled", [False, True])
def test_apply_requires_resolved_compile_values(
    apply_config, arch, missing, precompiled
):
    selected = {
        "batch_size": 4,
        "max_seq_len": 8192,
        "dec_max_seq_len": 448,
        "kvcache_block_size": 448 if arch == ENC_DEC_ARCH else 8192,
        "prefill_chunk_size": 256,
        "num_devices": 2,
        "kvcache_num_blocks": 16,
    }
    del selected[missing]
    with pytest.raises(ValueError, match=missing):
        apply_config(selected, arch=arch, precompiled=precompiled)


def test_whisper_block_count_is_read_without_a_batch_size_default():
    hf_config = SimpleNamespace(architectures=[ENC_DEC_ARCH])
    assert get_runtime_num_blocks(hf_config, {"batch_size": 4}) is None
    assert (
        get_runtime_num_blocks(hf_config, {"batch_size": 4, "kvcache_num_blocks": 8})
        == 8
    )


@pytest.mark.parametrize("precompiled", [False, True])
def test_prepared_settings_are_owned_and_applied_without_reading_metadata(
    apply_config, precompiled
):
    selected = {
        "batch_size": 2,
        "max_seq_len": 4096,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": 256,
        "num_devices": 2,
        "dtype": "bfloat16",
    }
    cfg = apply_config(selected)
    cfg.scheduler_config.max_num_seqs = 4
    cfg.model_config.max_model_len = 8192
    cfg.cache_config.num_gpu_blocks_override = 16
    prepared = prepare_runtime_config(cfg, selected, precompiled=precompiled)
    assert cfg.scheduler_config.max_num_seqs == 4
    assert cfg.model_config.max_model_len == 8192
    assert cfg.cache_config.gpu_memory_utilization == 0.9
    selected.clear()

    apply_runtime_config(cfg, prepared)

    assert cfg.scheduler_config.max_num_seqs == 2
    assert cfg.model_config.max_model_len == 4096
    assert cfg.scheduler_config.max_num_batched_tokens == 4096
    assert cfg.additional_config.prefill_chunk_size == 256
    assert cfg.additional_config.num_devices_per_local_rank == 2
    assert cfg.cache_config.gpu_memory_utilization == 0.9
    assert cfg.model_config.dtype == (torch.bfloat16 if precompiled else torch.float32)


@pytest.mark.parametrize(
    "prefix_caching, override, expected_blocks",
    [(False, None, 8), (False, 4, 4), (True, None, 29), (True, 4, 13)],
)
def test_runtime_block_count_is_resolved_before_apply(
    apply_config, prefix_caching, override, expected_blocks
):
    selected = {
        "batch_size": 2,
        "max_seq_len": 8192,
        "kvcache_block_size": 1024,
        "kvcache_num_blocks": 8,
        "prefill_chunk_size": 256,
        "num_devices": 1,
    }
    cfg = apply_config(selected)
    cfg.cache_config.enable_prefix_caching = prefix_caching
    cfg.cache_config.num_gpu_blocks_override = override
    if override is not None:
        cfg.additional_config.optimum_overrides = {"kvcache_num_blocks": override}
    prepared = prepare_runtime_config(cfg, selected, precompiled=True)
    assert prepared.model_num_blocks == (8 if override is None else override)
    assert cfg.cache_config.num_gpu_blocks is None
    if override is not None:
        cfg.cache_config.num_gpu_blocks_override = override + 1

    apply_runtime_config(cfg, prepared)

    assert cfg.cache_config.num_gpu_blocks == expected_blocks
    assert cfg.additional_config.num_blocks_override == override
    repeated = prepare_runtime_config(cfg, selected, precompiled=True)
    assert repeated.model_num_blocks == prepared.model_num_blocks
    apply_runtime_config(cfg, repeated)
    assert cfg.cache_config.num_gpu_blocks == expected_blocks


@pytest.mark.parametrize(
    "sub_block_size, prefill_chunk_size, mamba_block_size, error, message",
    [
        (384, 256, None, ValueError, "not divisible.*prefill_chunk_size"),
        (2048, 256, None, ValueError, "greater than.*kvcache_block_size"),
        (768, 256, None, ValueError, "not divisible.*sub_block_size"),
        (None, 384, None, ValueError, "not divisible.*sub_block_size"),
        (256, 256, 4096, AssertionError, "mamba_block_size can only be synced"),
    ],
)
def test_block_constraints_are_validated_without_changing_config(
    apply_config, sub_block_size, prefill_chunk_size, mamba_block_size, error, message
):
    selected = {
        "batch_size": 2,
        "max_seq_len": 8192,
        "kvcache_block_size": 1024,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": prefill_chunk_size,
        "num_devices": 1,
    }
    cfg = apply_config(selected)
    cfg.cache_config.enable_prefix_caching = True
    cfg.cache_config.mamba_block_size = mamba_block_size
    cfg.additional_config.sub_block_size = sub_block_size
    cfg.additional_config.optimum_overrides = {"device": [0], "create_runtimes": False}
    cfg.scheduler_config.max_num_seqs = 4
    cfg.model_config.max_model_len = 16384
    before = deepcopy(cfg)

    with pytest.raises(error, match=message):
        prepare_runtime_config(cfg, selected, precompiled=True)

    assert cfg == before


@pytest.mark.parametrize("sub_block_size, expected_size", [(None, 256), (512, 512)])
def test_prefix_block_sizes_are_resolved_before_apply(
    apply_config, sub_block_size, expected_size
):
    selected = {
        "batch_size": 2,
        "max_seq_len": 8192,
        "kvcache_block_size": 1024,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": 256,
        "num_devices": 1,
    }
    cfg = apply_config(selected)
    cfg.cache_config.enable_prefix_caching = True
    cfg.additional_config.sub_block_size = sub_block_size
    before = deepcopy(cfg)

    prepared = prepare_runtime_config(cfg, selected, precompiled=True)

    assert cfg == before
    assert prepared.block_size == expected_size
    assert prepared.attn_block_size == 1024
    apply_runtime_config(cfg, prepared)
    assert cfg.cache_config.block_size == expected_size
    assert cfg.additional_config.attn_block_size == 1024


@pytest.mark.parametrize(
    "user_specified, precompiled, expected_size",
    [(False, True, 8192), (True, True, 4096), (False, False, 4096)],
)
def test_mamba_block_size_is_resolved_before_apply(
    apply_config, user_specified, precompiled, expected_size
):
    selected = {
        "batch_size": 2,
        "max_seq_len": 8192,
        "kvcache_block_size": 1024,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": 256,
        "num_devices": 1,
    }
    cfg = apply_config(selected)
    cfg.cache_config.mamba_block_size = 4096
    cfg.cache_config.user_specified_mamba_block_size = user_specified
    cfg.cache_config.enable_prefix_caching = user_specified
    cfg.model_config.max_model_len = 16384
    before = deepcopy(cfg)

    prepared = prepare_runtime_config(cfg, selected, precompiled=precompiled)

    assert cfg == before
    assert prepared.mamba_block_size == expected_size
    apply_runtime_config(cfg, prepared)
    assert cfg.cache_config.mamba_block_size == expected_size


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_devices",
        "prefill_conflict",
        "zero_scheduling_budget",
        "scheduling_budget",
    ],
)
def test_selected_settings_are_validated_before_config_is_changed(invalid):
    overrides = {"device": [0], "memory_budget": 0.5}
    cfg = _vllm_config(
        arch=DECODER_ARCH,
        additional_config={
            "optimum_overrides": overrides,
            "prefill_chunk_size": 512,
        },
    )
    selected = {
        "batch_size": 2,
        "max_seq_len": 4096,
        "kvcache_block_size": 4096,
        "kvcache_num_blocks": 16,
        "prefill_chunk_size": 512,
        "num_devices": 2,
    }
    if invalid == "missing_devices":
        del selected["num_devices"]
        expected_error = "num_devices"
    elif invalid == "prefill_conflict":
        selected["prefill_chunk_size"] = 256
        expected_error = "does not match the resolved model configuration"
    else:
        cfg.scheduler_config.max_num_scheduled_tokens = (
            0 if invalid == "zero_scheduling_budget" else 256
        )
        expected_error = "max_num_scheduled_tokens cannot be set"
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match=expected_error):
        prepare_runtime_config(cfg, selected, precompiled=True)
    assert cfg == before
