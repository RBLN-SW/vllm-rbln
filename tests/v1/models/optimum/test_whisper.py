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
"""Unit tests for Whisper explicit language detection.

These exercise the model-side building blocks that vLLM's speech-to-text
serving layer drives when ``supports_explicit_language_detection`` is True and
a request arrives without a ``language``:

    get_language_detection_prompt  -> seed decoder with <|startoftranscript|>
    get_language_token_ids         -> constrain generation to language tokens
    parse_language_detection_output-> read back the predicted <|xx|> token

All of these are ``classmethod``s and contain no device/runtime work, so they
are tested against the class directly (no compiled model / NPU needed) using a
real Whisper tokenizer, which is what the detection path consumes in serving.
"""

import numpy as np
import pytest
from vllm.config import SpeechToTextConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.model_executor.models.whisper import ISO639_1_SUPPORTED_LANGS

from vllm_rbln.model_executor.models.optimum.whisper import (
    RBLNOptimumWhisperForConditionalGeneration as Whisper,
)

SAMPLE_RATE = 16000
TOKENIZER_ID = "openai/whisper-tiny"


@pytest.fixture(scope="module")
def tokenizer():
    """Real HF Whisper tokenizer, shared across the module.

    Skips the suite when it can't be loaded (e.g. offline with no cached
    snapshot) so these unit tests stay runnable without network access.
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.WhisperTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip, not fail
        pytest.skip(f"{TOKENIZER_ID} tokenizer unavailable: {exc}")


def _stt_params(
    language: str | None,
    *,
    request_prompt: str = "",
    task_type: str = "transcribe",
) -> SpeechToTextParams:
    return SpeechToTextParams(
        audio=np.zeros(SAMPLE_RATE, dtype=np.float32),
        stt_config=SpeechToTextConfig(sample_rate=SAMPLE_RATE),
        model_config=None,  # unused by get_generation_prompt
        language=language,
        task_type=task_type,
        request_prompt=request_prompt,
    )


class TestLanguageTokenIds:
    def test_one_id_per_supported_language(self, tokenizer):
        ids = Whisper.get_language_token_ids(tokenizer)
        assert len(ids) == len(ISO639_1_SUPPORTED_LANGS)
        assert len(set(ids)) == len(ids)  # all distinct

    def test_ids_match_language_token_lookup(self, tokenizer):
        ids = Whisper.get_language_token_ids(tokenizer)
        expected = [
            tokenizer.convert_tokens_to_ids(f"<|{code}|>")
            for code in ISO639_1_SUPPORTED_LANGS
        ]
        assert ids == expected


class TestLanguageDetectionPrompt:
    def test_decoder_seeded_with_startoftranscript_only(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        prompt = Whisper.get_language_detection_prompt(
            audio, SpeechToTextConfig(sample_rate=SAMPLE_RATE)
        )
        # Only <|startoftranscript|> => next predicted token is a language token.
        assert prompt["decoder_prompt"]["prompt"] == "<|startoftranscript|>"

    def test_audio_forwarded_on_encoder_prompt(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        prompt = Whisper.get_language_detection_prompt(
            audio, SpeechToTextConfig(sample_rate=SAMPLE_RATE)
        )
        enc = prompt["encoder_prompt"]
        assert enc["prompt"] == ""  # Whisper has no encoder text prompt
        audio_data, sr = enc["multi_modal_data"]["audio"]
        assert sr == SAMPLE_RATE
        assert audio_data is audio


class TestParseLanguageDetectionOutput:
    def test_extracts_language_code(self, tokenizer):
        token_id = tokenizer.convert_tokens_to_ids("<|de|>")
        assert Whisper.parse_language_detection_output([token_id], tokenizer) == "de"

    def test_only_first_token_is_used(self, tokenizer):
        de = tokenizer.convert_tokens_to_ids("<|de|>")
        fr = tokenizer.convert_tokens_to_ids("<|fr|>")
        assert Whisper.parse_language_detection_output([de, fr], tokenizer) == "de"

    def test_round_trip_token_ids_to_language(self, tokenizer):
        # ids produced by get_language_token_ids decode back to their lang code.
        ids = Whisper.get_language_token_ids(tokenizer)
        for code, token_id in zip(ISO639_1_SUPPORTED_LANGS, ids):
            assert (
                Whisper.parse_language_detection_output([token_id], tokenizer) == code
            )

    def test_non_language_token_rejected(self, tokenizer):
        # A regular (non-language) token decodes to plain text, not <|xx|>.
        regular_id = tokenizer.encode("hello", add_special_tokens=False)[0]
        with pytest.raises(AssertionError):
            Whisper.parse_language_detection_output([regular_id], tokenizer)

    def test_unsupported_language_token_rejected(self, tokenizer, monkeypatch):
        # Whisper has no <|zz|> token, so force the decode to model the case
        # where constrained generation somehow yields an unsupported language.
        monkeypatch.setattr(
            tokenizer, "decode", lambda ids, skip_special_tokens=False: "<|zz|>"
        )
        with pytest.raises(AssertionError):
            Whisper.parse_language_detection_output([0], tokenizer)


class TestGenerationPromptUsesGivenLanguage:
    """When a language is supplied it is used directly (no detection)."""

    def test_given_language_embedded_in_prompt(self):
        prompt = Whisper.get_generation_prompt(_stt_params("ko"))
        text = prompt["decoder_prompt"]["prompt"]
        assert text == "<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>"

    def test_task_type_embedded(self):
        prompt = Whisper.get_generation_prompt(_stt_params("en", task_type="translate"))
        assert "<|translate|>" in prompt["decoder_prompt"]["prompt"]

    def test_request_prompt_prefixed(self):
        prompt = Whisper.get_generation_prompt(
            _stt_params("en", request_prompt="hi there")
        )
        text = prompt["decoder_prompt"]["prompt"]
        assert text.startswith("<|prev|>hi there<|startoftranscript|>")

    def test_missing_language_raises(self):
        # get_generation_prompt requires a resolved language; serving must have
        # filled it in (explicitly or via detection) before this is called.
        with pytest.raises(ValueError):
            Whisper.get_generation_prompt(_stt_params(None))
