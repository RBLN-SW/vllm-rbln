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

"""End-to-end tests for speculative decoding methods.

Validates that speculative decoding (ngram, medusa, suffix) produces
outputs identical to non-speculative decoding (exact match).
"""

import random
import string

import pytest
from vllm import LLM, SamplingParams

# Default model for testing
MODEL_ID = "meta-llama/Llama-3.2-1B"
# Medusa adapter (converted to vLLM format)
MEDUSA_MODEL_ID = "MegaLearner/medusa_llama_3_2_1b_3_heads"

NUM_PROMPTS = 10


@pytest.fixture(scope="module")
def sampling_config():
    """Deterministic sampling config to ensure exact match between
    speculative and non-speculative decoding."""
    return SamplingParams(
        temperature=0,
        max_tokens=32,
        ignore_eos=True,
        repetition_penalty=1,
        frequency_penalty=0,
        presence_penalty=0,
        min_p=0,
        logprobs=None,
    )


def get_ngram_test_prompts():
    """Generate prompts that are friendly to n-gram speculation.

    Repeated characters create strong n-gram patterns that the
    n-gram proposer can easily predict.
    """
    random.seed(42)
    prompts = []
    for _ in range(NUM_PROMPTS):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Keep repeating: {w} {w} {w} {w} {w} {w} {w} {w} {w} {w}"
        )
    return prompts


def get_suffix_test_prompts():
    """Generate prompts that are friendly to suffix-based speculation.

    Repetitive sequential patterns allow the suffix proposer to find
    matching suffixes in the existing token sequence.
    """
    random.seed(42)
    prompts = []
    for _ in range(NUM_PROMPTS):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Repeat the following pattern: {w} {w} {w} {w} {w} {w} {w} {w}"
        )
    return prompts


def get_medusa_test_prompts():
    """Generate prompts suitable for Medusa speculation."""
    prompts = []
    for i in range(NUM_PROMPTS):
        prompts.append(
            f"Complete this sentence: The quick brown fox jumps over the lazy dog. "
            f"Sentence number {i}."
        )
    return prompts


def _get_prompts_for_method(method: str) -> list[str]:
    if method == "ngram":
        return get_ngram_test_prompts()
    elif method == "suffix":
        return get_suffix_test_prompts()
    elif method == "medusa":
        return get_medusa_test_prompts()
    else:
        raise NotImplementedError(f"{method} is not supported yet.")


def _create_llm(
    speculative_config: dict | None = None,
) -> LLM:
    """Create an LLM instance with optional speculative config.

    Env vars (VLLM_RBLN_USE_VLLM_MODEL, VLLM_DISABLE_COMPILE_CACHE) must be
    set by the caller before invoking this function.
    """
    kwargs = dict(
        model=MODEL_ID,
        max_model_len=2048,
        block_size=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        max_num_seqs=4,
    )
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config

    return LLM(**kwargs)


def _generate_outputs(llm: LLM, prompts: list[str], sampling_params: SamplingParams):
    """Generate outputs and return dict mapping prompt -> generated text."""
    outputs = llm.generate(prompts, sampling_params)
    return {output.prompt: output.outputs[0].text for output in outputs}


def _ensure_medusa_adapter():
    """Convert medusa adapter to vLLM format if needed.

    Returns (model_path, num_speculative_tokens).
    """
    import json
    import re
    from pathlib import Path

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import AutoConfig

    _VLLM_BLOCK_WEIGHT_RE = re.compile(r"^blocks\.\d+\.layers\.0\.weight$")
    _EXT_WEIGHT_RE = re.compile(r"^(\d+)\.0\.linear\.weight$")
    _EXT_BIAS_RE = re.compile(r"^(\d+)\.0\.linear\.bias$")
    _EXT_LM_HEAD_RE = re.compile(r"^(\d+)\.1\.weight$")

    config_path = hf_hub_download(MEDUSA_MODEL_ID, "config.json")
    with Path(config_path).open("r", encoding="utf-8") as f:
        source_config = json.load(f)

    # Check if already in vLLM format
    if source_config.get("model_type") == "medusa" and isinstance(
        source_config.get("num_heads"), int
    ):
        return MEDUSA_MODEL_ID, int(source_config["num_heads"])

    num_heads = int(source_config["medusa_num_heads"])

    lm_head_path = hf_hub_download(MEDUSA_MODEL_ID, "medusa_lm_head.pt")
    snapshot_dir = Path(lm_head_path).parent
    source_state = torch.load(lm_head_path, map_location="cpu")

    if any(_VLLM_BLOCK_WEIGHT_RE.match(key) for key in source_state):
        return MEDUSA_MODEL_ID, num_heads

    out_dir = snapshot_dir / "vllm_converted_medusa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_config_path = out_dir / "config.json"
    out_weights_path = out_dir / "model.safetensors"

    if out_config_path.exists() and out_weights_path.exists():
        return str(out_dir), num_heads

    base_config = AutoConfig.from_pretrained(MODEL_ID)
    converted: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if (m := _EXT_WEIGHT_RE.match(key)) is not None:
            converted[f"blocks.{m.group(1)}.layers.0.weight"] = value
        elif (m := _EXT_BIAS_RE.match(key)) is not None:
            converted[f"blocks.{m.group(1)}.layers.0.bias"] = value
        elif (m := _EXT_LM_HEAD_RE.match(key)) is not None:
            converted[f"lm_heads.{m.group(1)}.weight"] = value

    out_config = {
        "model_type": "medusa",
        "architectures": ["MedusaModel"],
        "dtype": "bfloat16",
        "hidden_size": int(base_config.hidden_size),
        "vocab_size": int(base_config.vocab_size),
        "num_heads": num_heads,
        "num_hidden_layers": 1,
        "truncated_vocab_size": int(base_config.vocab_size),
        "original_lm_head": False,
        "medusa_fc_bias": any("bias" in k for k in converted),
    }

    save_file(converted, str(out_weights_path))
    out_config_path.write_text(json.dumps(out_config, indent=2), encoding="utf-8")
    return str(out_dir), num_heads


class TestNgramSpecDec:
    """N-gram speculative decoding should produce identical outputs to baseline."""

    @pytest.fixture(scope="class")
    def baseline_llm(self, monkeypatch_class):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        return _create_llm()

    @pytest.fixture(scope="class")
    def spec_llm(self, monkeypatch_class):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        return _create_llm(
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 3,
            },
        )

    def test_exact_match(self, baseline_llm, spec_llm, sampling_config):
        prompts = get_ngram_test_prompts()
        baseline_outputs = _generate_outputs(baseline_llm, prompts, sampling_config)
        spec_outputs = _generate_outputs(spec_llm, prompts, sampling_config)

        for prompt in prompts:
            assert baseline_outputs[prompt] == spec_outputs[prompt], (
                f"Mismatch for prompt: {prompt!r}\n"
                f"  baseline: {baseline_outputs[prompt]!r}\n"
                f"  spec_dec: {spec_outputs[prompt]!r}"
            )


class TestSuffixSpecDec:
    """Suffix speculative decoding should produce identical outputs to baseline."""

    @pytest.fixture(scope="class")
    def baseline_llm(self, monkeypatch_class):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        return _create_llm()

    @pytest.fixture(scope="class")
    def spec_llm(self, monkeypatch_class):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        return _create_llm(
            speculative_config={
                "method": "suffix",
                "num_speculative_tokens": 3,
            },
        )

    def test_exact_match(self, baseline_llm, spec_llm, sampling_config):
        prompts = get_suffix_test_prompts()
        baseline_outputs = _generate_outputs(baseline_llm, prompts, sampling_config)
        spec_outputs = _generate_outputs(spec_llm, prompts, sampling_config)

        for prompt in prompts:
            assert baseline_outputs[prompt] == spec_outputs[prompt], (
                f"Mismatch for prompt: {prompt!r}\n"
                f"  baseline: {baseline_outputs[prompt]!r}\n"
                f"  spec_dec: {spec_outputs[prompt]!r}"
            )


class TestMedusaSpecDec:
    """Medusa speculative decoding should produce identical outputs to baseline."""

    @pytest.fixture(scope="class")
    def medusa_adapter(self):
        return _ensure_medusa_adapter()

    @pytest.fixture(scope="class")
    def baseline_llm(self, monkeypatch_class):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        return _create_llm()

    @pytest.fixture(scope="class")
    def spec_llm(self, monkeypatch_class, medusa_adapter):
        monkeypatch_class.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch_class.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        medusa_model_path, num_heads = medusa_adapter
        return _create_llm(
            speculative_config={
                "method": "medusa",
                "model": medusa_model_path,
                "num_speculative_tokens": num_heads,
            },
        )

    def test_exact_match(self, baseline_llm, spec_llm, sampling_config):
        prompts = get_medusa_test_prompts()
        baseline_outputs = _generate_outputs(baseline_llm, prompts, sampling_config)
        spec_outputs = _generate_outputs(spec_llm, prompts, sampling_config)

        for prompt in prompts:
            assert baseline_outputs[prompt] == spec_outputs[prompt], (
                f"Mismatch for prompt: {prompt!r}\n"
                f"  baseline: {baseline_outputs[prompt]!r}\n"
                f"  spec_dec: {spec_outputs[prompt]!r}"
            )
