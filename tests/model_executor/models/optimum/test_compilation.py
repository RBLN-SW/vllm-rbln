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
from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForImageTextToText,
    RBLNAutoModelForSpeechSeq2Seq,
    RBLNAutoModelForVision2Seq,
    RBLNBertModel,
    RBLNQwen3Model,
)

from vllm_rbln.model_executor.models.optimum import compilation
from vllm_rbln.model_executor.models.optimum.compilation import (
    RBLNCompileSpec,
    _deep_merge,
)


def _hf(arch: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(architectures=[arch], **extra)


class TestDeepMerge:
    def test_top_level_overwrite(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99})
        assert base == {"a": 1, "b": 99}

    def test_nested_merge_preserves_untouched_subkeys(self):
        base = {"language_model": {"batch_size": 4, "max_seq_len": 1024}}
        _deep_merge(base, {"language_model": {"max_seq_len": 2048}})
        assert base == {
            "language_model": {"batch_size": 4, "max_seq_len": 2048}
        }

    def test_nondict_overrides_dict(self):
        base = {"x": {"nested": True}}
        _deep_merge(base, {"x": "scalar"})
        assert base == {"x": "scalar"}

    def test_new_keys_added(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}

    def test_empty_overrides_is_noop(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {})
        assert base == {"a": {"b": 1}}


class TestForArchitectureDispatch:
    def test_unknown_architecture_raises(self):
        with pytest.raises(NotImplementedError):
            RBLNCompileSpec.for_architecture(
                _hf("DefinitelyNotARealArch"),
                batch_size=1,
                block_size=128,
                max_model_len=128,
                tp_size=1,
            )

    def test_generation_dispatches_to_decoder(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("LlamaForCausalLM"),
            batch_size=4,
            block_size=128,
            max_model_len=1024,
            tp_size=1,
        )
        assert spec.model_cls is RBLNAutoModelForCausalLM

    def test_pooling_dispatches_to_pooling(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("BertModel"),
            batch_size=4,
            block_size=128,
            max_model_len=128,
            tp_size=1,
        )
        assert spec.model_cls is RBLNBertModel

    def test_multimodal_dispatches_to_multimodal(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("LlavaForConditionalGeneration"),
            batch_size=2,
            block_size=128,
            max_model_len=2048,
            tp_size=1,
        )
        # LlavaForConditionalGeneration -> RBLNAutoModelForVision2Seq.
        assert spec.model_cls is RBLNAutoModelForVision2Seq

    def test_gemma3_multimodal_uses_image_text_to_text(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("Gemma3ForConditionalGeneration"),
            batch_size=2,
            block_size=128,
            max_model_len=2048,
            tp_size=1,
        )
        assert spec.model_cls is RBLNAutoModelForImageTextToText

    def test_enc_dec_dispatches_to_enc_dec(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("WhisperForConditionalGeneration", max_length=448),
            batch_size=2,
            block_size=448,
            max_model_len=448,
            tp_size=1,
        )
        assert spec.model_cls is RBLNAutoModelForSpeechSeq2Seq

    def test_rbln_overrides_are_deep_merged(self):
        spec = RBLNCompileSpec.for_architecture(
            _hf("LlamaForCausalLM"),
            batch_size=4,
            block_size=128,
            max_model_len=1024,
            tp_size=1,
            rbln_overrides={"batch_size": 9, "extra_key": "value"},
        )
        assert spec.rbln_config["batch_size"] == 9  # overridden
        assert spec.rbln_config["extra_key"] == "value"  # added
        assert spec.rbln_config["max_seq_len"] == 1024  # untouched


class TestForDecoder:
    def test_no_partition_when_block_size_equals_max_model_len(self):
        spec = RBLNCompileSpec._for_decoder(
            batch_size=4, block_size=1024, max_model_len=1024, tp_size=1
        )
        assert spec.rbln_config == {
            "tensor_parallel_size": 1,
            "batch_size": 4,
            "max_seq_len": 1024,
        }

    def test_flash_attn_when_block_size_smaller_than_max_model_len(self):
        spec = RBLNCompileSpec._for_decoder(
            batch_size=4, block_size=128, max_model_len=1024, tp_size=2
        )
        assert spec.rbln_config == {
            "tensor_parallel_size": 2,
            "batch_size": 4,
            "max_seq_len": 1024,
            "kvcache_partition_len": 128,
            "attn_impl": "flash_attn",
        }


class TestForPooling:
    def test_non_qwen3_no_flash_attn_even_when_block_size_differs(self):
        spec = RBLNCompileSpec._for_pooling(
            _hf("BertModel"),
            batch_size=4,
            block_size=128,
            max_model_len=512,
            tp_size=1,
        )
        assert spec.model_cls is RBLNBertModel
        assert "kvcache_partition_len" not in spec.rbln_config
        assert "attn_impl" not in spec.rbln_config

    def test_qwen3_model_with_smaller_block_uses_flash_attn(self):
        spec = RBLNCompileSpec._for_pooling(
            _hf("Qwen3Model"),
            batch_size=4,
            block_size=128,
            max_model_len=2048,
            tp_size=1,
        )
        assert spec.model_cls is RBLNQwen3Model
        assert spec.rbln_config["kvcache_partition_len"] == 128
        assert spec.rbln_config["attn_impl"] == "flash_attn"

    def test_qwen3_model_no_flash_attn_when_block_equals_max(self):
        spec = RBLNCompileSpec._for_pooling(
            _hf("Qwen3Model"),
            batch_size=4,
            block_size=512,
            max_model_len=512,
            tp_size=1,
        )
        assert "kvcache_partition_len" not in spec.rbln_config
        assert "attn_impl" not in spec.rbln_config


class TestForEncDec:
    def test_happy_path_produces_whisper_spec(self):
        spec = RBLNCompileSpec._for_enc_dec(
            _hf("WhisperForConditionalGeneration", max_length=448),
            batch_size=2,
            block_size=448,
            max_model_len=448,
            tp_size=1,
        )
        assert spec.model_cls is RBLNAutoModelForSpeechSeq2Seq
        assert spec.rbln_config == {
            "tensor_parallel_size": 1,
            "batch_size": 2,
            "token_timestamps": False,
        }

    def test_block_size_must_equal_max_model_len(self):
        with pytest.raises(AssertionError, match="block_size"):
            RBLNCompileSpec._for_enc_dec(
                _hf("WhisperForConditionalGeneration", max_length=448),
                batch_size=2,
                block_size=128,
                max_model_len=448,
                tp_size=1,
            )

    def test_max_model_len_must_match_hf_max_length(self):
        with pytest.raises(AssertionError, match="max_length"):
            RBLNCompileSpec._for_enc_dec(
                _hf("WhisperForConditionalGeneration", max_length=448),
                batch_size=2,
                block_size=512,
                max_model_len=512,
                tp_size=1,
            )


class TestForMultimodal:
    def test_unknown_alias_raises(self, monkeypatch):
        # Force get_rbln_model_info to return a model alias missing from
        # _COMPILE_MULTIMODAL_FNS.
        monkeypatch.setattr(
            compilation,
            "get_rbln_model_info",
            lambda config: ("definitely_unknown_alias", "RBLNDoesntMatter"),
        )
        with pytest.raises(ValueError, match="multimodal model alias"):
            RBLNCompileSpec._for_multimodal(
                _hf("LlavaForConditionalGeneration"),
                batch_size=2,
                block_size=128,
                max_model_len=2048,
                tp_size=1,
            )

    def test_dispatches_to_compile_fn_with_forwarded_args(self, monkeypatch):
        captured = {}

        def fake_compile_fn(batch_size, max_model_len, block_size, tp_size):
            captured["args"] = (batch_size, max_model_len, block_size, tp_size)
            return {"sentinel": True}

        # Patch the dispatch table on the imported module so the real fn
        # doesn't run (and so the assertion can compare without aliasing).
        monkeypatch.setitem(
            compilation._COMPILE_MULTIMODAL_FNS, "llava", fake_compile_fn
        )

        spec = RBLNCompileSpec._for_multimodal(
            _hf("LlavaForConditionalGeneration"),
            batch_size=2,
            block_size=128,
            max_model_len=2048,
            tp_size=4,
        )
        # Note the unusual argument order in `_for_multimodal`:
        # (batch_size, max_model_len, block_size, tp_size).
        assert captured["args"] == (2, 2048, 128, 4)
        assert spec.rbln_config == {"sentinel": True}
        assert spec.model_cls is RBLNAutoModelForVision2Seq
