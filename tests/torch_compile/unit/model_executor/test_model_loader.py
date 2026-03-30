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

"""Unit tests for model loader utilities: registry, rbln_params, common.

Tests architecture detection, config parsing, parameter extraction,
and bucket size selection. Follows the TPU inference test_model_loader pattern.
"""

from unittest.mock import MagicMock

import pytest
from transformers import PretrainedConfig


# ============================================================
# Registry Tests
# ============================================================


class TestArchitectureDetection:
    """Test architecture detection functions in registry.py."""

    def test_is_generation_arch(self):
        from vllm_rbln.utils.optimum.registry import is_generation_arch

        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        assert is_generation_arch(config) is True

    def test_is_generation_arch_qwen2(self):
        from vllm_rbln.utils.optimum.registry import is_generation_arch

        config = PretrainedConfig(architectures=["Qwen2ForCausalLM"])
        assert is_generation_arch(config) is True

    def test_is_generation_arch_false(self):
        from vllm_rbln.utils.optimum.registry import is_generation_arch

        config = PretrainedConfig(architectures=["BertModel"])
        assert is_generation_arch(config) is False

    def test_is_multi_modal(self):
        from vllm_rbln.utils.optimum.registry import is_multi_modal

        config = PretrainedConfig(
            architectures=["Qwen2VLForConditionalGeneration"]
        )
        assert is_multi_modal(config) is True

    def test_is_multi_modal_false(self):
        from vllm_rbln.utils.optimum.registry import is_multi_modal

        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        assert is_multi_modal(config) is False

    def test_is_pooling_arch(self):
        from vllm_rbln.utils.optimum.registry import is_pooling_arch

        config = PretrainedConfig(architectures=["BertModel"])
        assert is_pooling_arch(config) is True

    def test_is_pooling_arch_false(self):
        from vllm_rbln.utils.optimum.registry import is_pooling_arch

        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        assert is_pooling_arch(config) is False

    def test_is_enc_dec_arch(self):
        from vllm_rbln.utils.optimum.registry import is_enc_dec_arch

        config = PretrainedConfig(
            architectures=["WhisperForConditionalGeneration"]
        )
        assert is_enc_dec_arch(config) is True

    def test_is_enc_dec_arch_false(self):
        from vllm_rbln.utils.optimum.registry import is_enc_dec_arch

        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        assert is_enc_dec_arch(config) is False

    def test_is_arch_supported_unsupported(self):
        from vllm_rbln.utils.optimum.registry import is_arch_supported

        config = PretrainedConfig(architectures=["UnsupportedModel"])
        assert is_arch_supported(config, {}) is False

    def test_is_arch_supported_empty_architectures(self):
        from vllm_rbln.utils.optimum.registry import is_arch_supported

        config = PretrainedConfig(architectures=[])
        assert is_arch_supported(config, {}) is False

    def test_is_arch_supported_none_architectures(self):
        """Regression: architectures=None should not crash."""
        from vllm_rbln.utils.optimum.registry import is_arch_supported

        config = PretrainedConfig()  # architectures defaults to None
        assert is_arch_supported(config, {}) is False


class TestGetRblnModelInfo:
    """Test get_rbln_model_info: model metadata lookup."""

    def test_supported_model(self):
        from vllm_rbln.utils.optimum.registry import get_rbln_model_info

        config = PretrainedConfig(architectures=["LlamaForCausalLM"])
        name, cls_name = get_rbln_model_info(config)
        assert name == "llama"
        assert cls_name == "RBLNLlamaForCausalLM"

    def test_supported_model_qwen3(self):
        from vllm_rbln.utils.optimum.registry import get_rbln_model_info

        config = PretrainedConfig(architectures=["Qwen3ForCausalLM"])
        name, cls_name = get_rbln_model_info(config)
        assert name == "qwen3"
        assert cls_name == "RBLNQwen3ForCausalLM"

    def test_unsupported_model_raises(self):
        from vllm_rbln.utils.optimum.registry import get_rbln_model_info

        config = PretrainedConfig(architectures=["FakeModel"])
        with pytest.raises(ValueError, match="not supported"):
            get_rbln_model_info(config)

    def test_multimodal_model(self):
        from vllm_rbln.utils.optimum.registry import get_rbln_model_info

        config = PretrainedConfig(
            architectures=["Qwen2VLForConditionalGeneration"]
        )
        name, cls_name = get_rbln_model_info(config)
        assert name == "qwen2_vl"

    def test_embedding_model(self):
        from vllm_rbln.utils.optimum.registry import get_rbln_model_info

        config = PretrainedConfig(architectures=["BertModel"])
        name, cls_name = get_rbln_model_info(config)
        assert name == "bert_model"


# ============================================================
# Config Parsing Tests (rbln_params.py)
# ============================================================


class TestCfgGet:
    """Test _cfg_get and _cfg_get_submodule helper functions."""

    def test_cfg_get_dict(self):
        from vllm_rbln.utils.optimum.rbln_params import _cfg_get

        cfg = {"batch_size": 4, "max_seq_len": 1024}
        assert _cfg_get(cfg, "batch_size") == 4
        assert _cfg_get(cfg, "max_seq_len") == 1024
        assert _cfg_get(cfg, "missing_key") is None
        assert _cfg_get(cfg, "missing_key", 42) == 42

    def test_cfg_get_object(self):
        from vllm_rbln.utils.optimum.rbln_params import _cfg_get

        cfg = MagicMock()
        cfg.batch_size = 8
        assert _cfg_get(cfg, "batch_size") == 8

    def test_cfg_get_submodule_dict(self):
        from vllm_rbln.utils.optimum.rbln_params import _cfg_get_submodule

        cfg = {
            "language_model": {"kvcache_block_size": 1024},
        }
        sub = _cfg_get_submodule(cfg, "language_model")
        assert sub == {"kvcache_block_size": 1024}
        assert _cfg_get_submodule(cfg, "nonexistent") is None

    def test_cfg_get_submodule_object(self):
        from vllm_rbln.utils.optimum.rbln_params import _cfg_get_submodule

        cfg = MagicMock()
        cfg.language_model = MagicMock()
        cfg.language_model.kvcache_block_size = 1024
        sub = _cfg_get_submodule(cfg, "language_model")
        assert sub is cfg.language_model


class TestGetRblnParams:
    """Test get_rbln_params: parameter extraction from config."""

    def _make_vllm_config(self, architectures):
        config = MagicMock()
        config.model_config.hf_config = PretrainedConfig(
            architectures=architectures
        )
        return config

    def test_decoder_model(self):
        from vllm_rbln.utils.optimum.rbln_params import get_rbln_params

        vllm_config = self._make_vllm_config(["LlamaForCausalLM"])
        rbln_config = {
            "kvcache_block_size": 1024,
            "prefill_chunk_size": 128,
            "batch_size": 4,
            "max_seq_len": 4096,
            "kvcache_num_blocks": 100,
        }

        num_blocks, batch_size, max_seq_len, kvcache_bs, prefill_cs = (
            get_rbln_params(vllm_config, rbln_config)
        )

        assert num_blocks == 100
        assert batch_size == 4
        assert max_seq_len == 4096
        assert kvcache_bs == 1024
        assert prefill_cs == 128

    def test_encoder_decoder_model(self):
        from vllm_rbln.utils.optimum.rbln_params import get_rbln_params

        vllm_config = self._make_vllm_config(
            ["WhisperForConditionalGeneration"]
        )
        rbln_config = {
            "dec_max_seq_len": 448,
            "batch_size": 1,
            "kvcache_num_blocks": 10,
        }

        num_blocks, batch_size, max_seq_len, kvcache_bs, prefill_cs = (
            get_rbln_params(vllm_config, rbln_config)
        )

        assert max_seq_len == 448
        assert kvcache_bs == 448  # for enc-dec, kvcache_block_size = max_seq_len
        assert batch_size == 1

    def test_pooling_model(self):
        from vllm_rbln.utils.optimum.rbln_params import get_rbln_params

        vllm_config = self._make_vllm_config(["BertModel"])
        rbln_config = {
            "max_seq_len": 512,
            "batch_size": 8,
            "kvcache_num_blocks": None,
        }

        num_blocks, batch_size, max_seq_len, kvcache_bs, prefill_cs = (
            get_rbln_params(vllm_config, rbln_config)
        )

        assert max_seq_len == 512
        assert kvcache_bs == 512  # for pooling, kvcache_block_size = max_seq_len
        assert batch_size == 8
        assert num_blocks == 8  # fallback: num_blocks = batch_size

    def test_multimodal_model_with_submodule(self):
        from vllm_rbln.utils.optimum.rbln_params import get_rbln_params

        vllm_config = self._make_vllm_config(
            ["Qwen2VLForConditionalGeneration"]
        )
        rbln_config = {
            "kvcache_block_size": None,
            "batch_size": None,
            "max_seq_len": None,
            "kvcache_num_blocks": None,
            "language_model": {
                "kvcache_block_size": 2048,
                "batch_size": 2,
                "max_seq_len": 8192,
                "kvcache_num_blocks": 50,
            },
        }

        num_blocks, batch_size, max_seq_len, kvcache_bs, prefill_cs = (
            get_rbln_params(vllm_config, rbln_config)
        )

        assert kvcache_bs == 2048
        assert batch_size == 2
        assert max_seq_len == 8192
        assert num_blocks == 50

    def test_missing_required_field_raises(self):
        from vllm_rbln.utils.optimum.rbln_params import get_rbln_params

        vllm_config = self._make_vllm_config(["LlamaForCausalLM"])
        rbln_config = {
            "kvcache_block_size": 1024,
            "batch_size": 4,
            "max_seq_len": 4096,
            # missing kvcache_num_blocks
        }

        with pytest.raises(AssertionError, match="num_blocks"):
            get_rbln_params(vllm_config, rbln_config)


# ============================================================
# Bucket Size Selection Tests (common.py)
# ============================================================


class TestSelectBucketSize:
    """Test select_bucket_size: cached binary search for batch bucketing."""

    def test_exact_match(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(4, (1, 2, 4, 8, 16)) == 4

    def test_round_up(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(3, (1, 2, 4, 8, 16)) == 4

    def test_smallest(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(1, (1, 2, 4, 8, 16)) == 1

    def test_largest(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(16, (1, 2, 4, 8, 16)) == 16

    def test_round_up_to_max(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(15, (1, 2, 4, 8, 16)) == 16

    def test_single_bucket(self):
        from vllm_rbln.utils.optimum.common import select_bucket_size

        assert select_bucket_size(1, (4,)) == 4
