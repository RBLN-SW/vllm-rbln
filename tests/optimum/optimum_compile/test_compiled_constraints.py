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
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_rbln.config import build_optimum_rbln_config
from vllm_rbln.utils.optimum.compiled_constraints import (
    CompiledConstraints,
    apply_compiled_constraints,
    get_user_max_num_batched_tokens,
    is_chunked_prefill_arch,
    store_image_prefill_chunk_size,
    update_max_num_batched_tokens,
    validate_user_prefill_chunk_size,
)

pytestmark = pytest.mark.usefixtures("set_npu_env_var")

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
        additional_config=build_optimum_rbln_config(additional_config),
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
        params = CompiledConstraints._parse_decoder(cfg)
        assert params.num_blocks == 16
        assert params.batch_size == 4
        assert params.max_seq_len == 8192
        assert params.kvcache_block_size == 4096
        assert params.prefill_chunk_size == 256
        # num_devices is populated by the caller
        # (`CompiledConstraints.from_rbln_config`), not `_parse_decoder`,
        # so it is not asserted here.

    def test_none_dict_config(self):
        cfg: dict = {}
        params = CompiledConstraints._parse_decoder(cfg)
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
        params = CompiledConstraints._parse_decoder(cfg)
        assert params.prefill_chunk_size == 128

    def test_kvcache_partition_len(self):
        cfg = {
            "kvcache_num_blocks": 16,
            "batch_size": 4,
            "max_seq_len": 8192,
            "kvcache_partition_len": 4096,
        }
        params = CompiledConstraints._parse_decoder(cfg)
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
            CompiledConstraints._parse_decoder(cfg)


class TestParseEncDec:
    def test_kvcache_block_size_equals_dec_max_seq_len(self):
        cfg = {
            "kvcache_num_blocks": 4,
            "batch_size": 1,
            "dec_max_seq_len": 448,
        }
        params = CompiledConstraints._parse_enc_dec(cfg)
        assert params.num_blocks == 4
        assert params.batch_size == 1
        assert params.max_seq_len == 448
        assert params.kvcache_block_size == 448


class TestParsePooling:
    def test_reads_partition_size_from_compile_spec(self):
        params = CompiledConstraints._parse_pooling(
            {"max_seq_len": 8192, "batch_size": 4, "kvcache_partition_len": 1024}
        )
        assert params.kvcache_block_size == 1024

    def test_uses_explicit_kvcache_num_blocks(self):
        cfg = {
            "max_seq_len": 512,
            "batch_size": 8,
            "kvcache_num_blocks": 16,
        }
        params = CompiledConstraints._parse_pooling(cfg)
        assert params.num_blocks == 16
        assert params.batch_size == 8
        assert params.max_seq_len == 512
        assert params.kvcache_block_size == 512

    def test_falls_back_num_blocks_to_batch_size(self):
        cfg = {"max_seq_len": 512, "batch_size": 8}
        params = CompiledConstraints._parse_pooling(cfg)
        assert params.num_blocks == 8

    def test_kvcache_block_size_equals_max_seq_len(self):
        cfg = {"max_seq_len": 512, "batch_size": 8}
        params = CompiledConstraints._parse_pooling(cfg)
        assert params.kvcache_block_size == 512


class TestParseMultimodal:
    def test_reads_submodule_lengths_with_a_top_level_block_size(self):
        params = CompiledConstraints._parse_multimodal(
            {
                "kvcache_block_size": 8192,
                "language_model": {"batch_size": 4, "max_seq_len": 8192},
            }
        )
        assert params.batch_size == 4
        assert params.max_seq_len == 8192
        assert params.kvcache_block_size == 8192

    def test_reads_submodule_lengths_without_a_partition(self):
        params = CompiledConstraints._parse_multimodal(
            {"language_model": {"batch_size": 4, "max_seq_len": 8192}}
        )
        assert params.batch_size == 4
        assert params.max_seq_len == 8192

    @pytest.mark.parametrize("as_object", [False, True])
    def test_partial_submodules_resolve_each_field_by_precedence(self, as_object):
        cfg: Any = {
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
        params = CompiledConstraints._parse_multimodal(cfg)
        assert params.batch_size == 2
        assert params.max_seq_len == 4096
        assert params.kvcache_block_size == 4096
        assert params.num_blocks == 8
        assert params.prefill_chunk_size == 256
        assert params.image_prefill_chunk_size == [512, 256]

    def test_top_level_fields_present(self):
        cfg = {
            "kvcache_num_blocks": 32,
            "batch_size": 1,
            "max_seq_len": 4096,
            "kvcache_block_size": 128,
        }
        params = CompiledConstraints._parse_multimodal(cfg)
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
        params = CompiledConstraints._parse_multimodal(cfg)
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
        params = CompiledConstraints._parse_multimodal(cfg)
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
        params = CompiledConstraints._parse_multimodal(cfg)
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
        params = CompiledConstraints._parse_multimodal(cfg)
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
        params = CompiledConstraints._parse_multimodal(self._cfg(256))
        assert params.image_prefill_chunk_size == [256]

    def test_gemma4_list_passes_through(self):
        params = CompiledConstraints._parse_multimodal(self._cfg([1152, 640, 384]))
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
        params = CompiledConstraints._parse_multimodal(cfg)
        assert params.image_prefill_chunk_size is None

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            CompiledConstraints._parse_multimodal(self._cfg("256"))

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            CompiledConstraints._parse_multimodal(self._cfg(True))


class TestFromRblnConfigDtype:
    def _hf_config(self):
        return SimpleNamespace(architectures=["LlamaForCausalLM"])

    def test_reads_dtype(self):
        cfg = {"batch_size": 4, "max_seq_len": 8192, "dtype": "bfloat16"}
        params = CompiledConstraints.from_rbln_config(self._hf_config(), cfg)
        assert params.dtype == "bfloat16"

    def test_none_when_absent(self):
        cfg = {"batch_size": 4, "max_seq_len": 8192}
        params = CompiledConstraints.from_rbln_config(self._hf_config(), cfg)
        assert params.dtype is None


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
            additional_config={"user_max_num_batched_tokens": 512},
        )
        assert get_user_max_num_batched_tokens(cfg) == 512


class TestUpdateMaxNumBatchedTokens:
    def test_decoder_uses_prefill_chunk_size(self):
        cfg = _vllm_config(arch=DECODER_ARCH, max_num_seqs=4)
        params = CompiledConstraints(prefill_chunk_size=256)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 256

    def test_decoder_chunk_smaller_than_batch_ok(self):
        # The scheduler's per-step budget is decoupled from max_num_batched_tokens,
        # so a prefill chunk smaller than max_num_seqs no longer throttles decode.
        cfg = _vllm_config(arch=DECODER_ARCH, max_num_seqs=512)
        params = CompiledConstraints(prefill_chunk_size=128)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 128

    def test_pooling_uses_full_prefill_budget(self):
        cfg = _vllm_config(arch=POOLING_ARCH, max_model_len=8192, max_num_seqs=4)
        params = CompiledConstraints(prefill_chunk_size=128)
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
        params = CompiledConstraints(prefill_chunk_size=128)
        update_max_num_batched_tokens(cfg, params)
        assert cfg.scheduler_config.max_num_batched_tokens == 1500


class TestValidateUserPrefillChunkSize:
    def test_noop_for_pooling(self):
        cfg = _vllm_config(
            arch=POOLING_ARCH,
            additional_config={"user_max_num_batched_tokens": 512},
        )
        params = CompiledConstraints(prefill_chunk_size=128)
        validate_user_prefill_chunk_size(cfg, params.prefill_chunk_size)
        assert params.prefill_chunk_size == 128

    def test_noop_when_user_unset(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        params = CompiledConstraints(prefill_chunk_size=128)
        validate_user_prefill_chunk_size(cfg, params.prefill_chunk_size)
        assert params.prefill_chunk_size == 128

    def test_precompiled_conflict_raises(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={"user_max_num_batched_tokens": 512},
        )
        params = CompiledConstraints(prefill_chunk_size=128)
        with pytest.raises(ValueError, match="conflicts with the compiled"):
            validate_user_prefill_chunk_size(cfg, params.prefill_chunk_size)

    def test_precompiled_match_ok(self):
        cfg = _vllm_config(
            arch=DECODER_ARCH,
            additional_config={"user_max_num_batched_tokens": 128},
        )
        params = CompiledConstraints(prefill_chunk_size=128)
        validate_user_prefill_chunk_size(cfg, params.prefill_chunk_size)
        assert params.prefill_chunk_size == 128


class TestStoreImagePrefillChunkSize:
    def test_stores_buckets(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        store_image_prefill_chunk_size(cfg, [1152, 640, 384])
        assert cfg.additional_config.image_prefill_chunk_size == [1152, 640, 384]

    def test_noop_when_none(self):
        cfg = _vllm_config(arch=DECODER_ARCH, additional_config={})
        store_image_prefill_chunk_size(cfg, None)
        assert cfg.additional_config.image_prefill_chunk_size is None


def _engine_config() -> VllmConfig:
    # Force the cache-miss branch so building the config does not look for a
    # compiled artifact. The test calls apply_compiled_constraints itself.
    with patch(
        "vllm_rbln.utils.optimum.sync.load_compiled_rbln_config",
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
        "prefill_chunk_size": 128,
        "num_devices": 2,
    }
    apply_compiled_constraints(
        vllm_config,
        CompiledConstraints.from_rbln_config(
            vllm_config.model_config.hf_config, compiled
        ),
        precompiled=True,
    )

    assert vllm_config.additional_config.num_devices_per_local_rank == 2
