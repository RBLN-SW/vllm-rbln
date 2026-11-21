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

# Speculative decoding is a pure optimization: drafts are verified against the
# target, so greedy output with a method ON must equal spec OFF. Covers ngram,
# suffix, eagle, eagle3, medusa and mtp.

import json
import warnings

import pytest
from vllm import SamplingParams
from vllm.config import SpeculativeConfig
from vllm.inputs import TokensPrompt

from tests.vllm.model_specs import CompileModelSpec, apply_spec_envs, spec_params
from tests.vllm.utils import TokensTextLogprobs, check_outputs_almost_equal
from tests.vllm.v1.spec_decode.utils import (
    _DRAFT,
    MEDUSA_DRAFT,
    MEDUSA_TARGET,
    TARGET_MODEL,
)
from tests.vllm.vllm_config import local_weights_path
from vllm_rbln import envs

# Ngram and suffix find drafts by matching repeated text.
_SENTENCE = "The quick brown fox jumps over the lazy dog. "
PROMPT = _SENTENCE * 20
# 251 tokens for Qwen3.5: the generated ones cross from the first 256-token
# block, where a GatedDeltaNet state lives, into the second.
_MTP_PROMPT = _SENTENCE * 25
MAX_TOKENS = 16
NUM_LOGPROBS = 5

_NGRAM_TARGET = "meta-llama/Llama-3.2-1B-Instruct"
_NGRAM_CONFIG = {
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}
SPEC_MODELS = [
    CompileModelSpec(_NGRAM_TARGET, {"speculative_config": _NGRAM_CONFIG}, id="ngram"),
    CompileModelSpec(
        _NGRAM_TARGET,
        {
            "speculative_config": {
                "method": "suffix",
                "suffix_decoding_max_spec_factor": 2.0,
                "num_speculative_tokens": 3,
            },
        },
        id="suffix",
    ),
    # The EAGLE and Medusa draft models cap at 2048 positions.
    *[
        CompileModelSpec(
            TARGET_MODEL,
            {
                "max_model_len": 2048,
                "tensor_parallel_size": 2,
                "speculative_config": {
                    "method": method,
                    "model": draft,
                    "num_speculative_tokens": 3,
                },
            },
            id=method,
        )
        for method, draft in _DRAFT.items()
    ],
    CompileModelSpec(
        MEDUSA_TARGET,
        {
            "max_model_len": 2048,
            "speculative_config": {
                "method": "medusa",
                "model": MEDUSA_DRAFT,
                "num_speculative_tokens": 3,
            },
        },
        id="medusa",
    ),
    # A GatedDeltaNet state fills one full-attention page at this block size.
    # The runner cannot size a GatedDeltaNet pool, so the engine sizes it.
    CompileModelSpec(
        "Qwen/Qwen3.5-4B",
        {
            "language_model_only": True,
            "block_size": 256,
            "num_gpu_blocks_override": None,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        },
        id="mtp",
    ),
]
SAMPLED_NGRAM = CompileModelSpec(
    "Qwen/Qwen3-0.6B",
    {"max_num_seqs": 4, "speculative_config": _NGRAM_CONFIG},
    id="ngram",
)

# Methods whose draft path should land accepted tokens on the repetitive prompt.
# medusa is excluded: its only small ungated checkpoint is random-init, so
# acceptance there would measure model quality, not the code path.
_EXPECT_ACCEPTANCE = {"ngram", "suffix", "eagle", "eagle3", "mtp"}


@pytest.fixture(autouse=True)
def _pin_layers(spec: CompileModelSpec, monkeypatch):
    apply_spec_envs(spec, monkeypatch)
    method = spec.engine_kwargs["speculative_config"]["method"]
    num_layers = envs.VLLM_RBLN_NUM_HIDDEN_LAYERS
    if method == "mtp" and 0 < num_layers < 8:
        # TODO(rebel-compiler): the compiler's layer rolling
        # (RblnTensorUnrollLayerLoop) rejects Qwen3.5 truncated to its first
        # three layers, all GatedDeltaNet, or to one period of four. Two roll.
        warnings.warn(
            f"mtp needs two periods of Qwen3.5's layers: raising "
            f"VLLM_RBLN_NUM_HIDDEN_LAYERS from {num_layers} to 8",
            stacklevel=2,
        )
        monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", "8")
    if method != "eagle3" or num_layers == 0:
        return

    if num_layers < 3:
        warnings.warn(
            f"eagle3 needs at least three aux hidden states: raising "
            f"VLLM_RBLN_NUM_HIDDEN_LAYERS from {num_layers} to 3",
            stacklevel=2,
        )
        monkeypatch.setenv("VLLM_RBLN_NUM_HIDDEN_LAYERS", "3")

    original = SpeculativeConfig.__post_init__

    def post_init(self) -> None:
        original(self)
        self.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids = [0, 1, 2]

    monkeypatch.setattr(SpeculativeConfig, "__post_init__", post_init)


@pytest.fixture(scope="module")
def reference_outputs() -> dict[tuple[str, str, int, str], list[TokensTextLogprobs]]:
    return {}


@pytest.mark.model_compile
@pytest.mark.parametrize("spec", spec_params(SPEC_MODELS))
def test_speculative_decoding_matches_reference(
    vllm_runner, spec: CompileModelSpec, whole_model: bool, reference_outputs
) -> None:
    engine_kwargs = dict(spec.engine_kwargs)
    spec_config = dict(engine_kwargs.pop("speculative_config"))
    method = spec_config["method"]
    if draft := spec_config.get("model"):
        spec_config["model"] = local_weights_path(draft)
    prompt = _MTP_PROMPT if method == "mtp" else PROMPT

    # The reference sampler, on both runs: the RBLN one compiles a graph of its
    # own, and acceptance is being compared, not sampler implementations.
    engine_kwargs["additional_config"] = {
        "use_custom_sampler": False,
        **engine_kwargs.get("additional_config", {}),
    }
    reference_key = (
        spec.model,
        prompt,
        envs.VLLM_RBLN_NUM_HIDDEN_LAYERS,
        json.dumps((engine_kwargs, spec.envs), sort_keys=True),
    )
    if reference_key not in reference_outputs:
        with vllm_runner(spec.model, **engine_kwargs) as ref_model:
            outputs = ref_model.generate_greedy_logprobs(
                [prompt], MAX_TOKENS, NUM_LOGPROBS
            )
        reference_outputs[reference_key] = outputs

    with vllm_runner(
        spec.model,
        **engine_kwargs,
        speculative_config=spec_config,
    ) as spec_model:
        spec_outputs = spec_model.generate_greedy_logprobs(
            [prompt], MAX_TOKENS, NUM_LOGPROBS
        )
        # Read while the engine is alive (__exit__ shuts down the EngineCore).
        accepted = spec_model.spec_decode_accepted_tokens()

    # Rejection sampling is lossless in exact arithmetic only: the spec run
    # verifies K+1 tokens in one target pass where the reference decodes one at
    # a time, so the two differ by rounding. Only near-tie flips are tolerated.
    check_outputs_almost_equal(
        outputs_0_lst=reference_outputs[reference_key],
        outputs_1_lst=spec_outputs,
        name_0="reference",
        name_1=f"spec:{method}",
    )
    # Equivalence alone passes even if every draft is rejected, so where drafts
    # can realistically hit, require at least one acceptance. Truncated layers
    # leave the draft's predictions unrelated to the target's, so nothing hits.
    if method in _EXPECT_ACCEPTANCE and whole_model:
        assert accepted > 0, f"spec:{method} accepted no draft tokens"


# Qwen3's generation_config.json samples with top_k=20, so a request that sets
# nothing still carries a top-k the output can be held to.
DEFAULT_TOP_K = 20


@pytest.mark.model_compile
@pytest.mark.parametrize("spec", spec_params([SAMPLED_NGRAM]))
def test_sampled_speculative_decoding_stays_in_the_top_k(
    vllm_runner, spec: CompileModelSpec, whole_model: bool
) -> None:
    """Random rows through the RBLN rejection graph, one with its own top_k and
    one with the model's defaults, must sample only within that top_k."""
    engine_kwargs = dict(spec.engine_kwargs)
    spec_config = dict(engine_kwargs.pop("speculative_config"))
    method = spec_config["method"]
    explicit = SamplingParams(
        temperature=1.0, top_k=5, max_tokens=MAX_TOKENS, ignore_eos=True
    )
    with vllm_runner(
        spec.model, **engine_kwargs, speculative_config=spec_config
    ) as spec_model:
        defaults = spec_model.llm.get_default_sampling_params()
        assert defaults.top_k == DEFAULT_TOP_K
        defaults.max_tokens = MAX_TOKENS
        params = [explicit, defaults] * 2
        outputs = spec_model.llm.generate([PROMPT] * len(params), params)
        accepted = spec_model.spec_decode_accepted_tokens()

    # A reference engine without drafts ranks every sampled token after its own
    # prefix. Rounding differs between the two, so a token at the cut may slip
    # a few ranks: allow twice the top_k.
    checks = [
        (o.prompt_token_ids + list(o.outputs[0].token_ids[:i]), token, p.top_k)
        for p, o in zip(params, outputs)
        for i, token in enumerate(o.outputs[0].token_ids)
    ]
    engine_kwargs["max_logprobs"] = 2 * DEFAULT_TOP_K
    engine_kwargs["additional_config"] = {
        **engine_kwargs.get("additional_config", {}),
        "use_custom_sampler": False,
    }
    with vllm_runner(spec.model, **engine_kwargs) as ref_model:
        ranked = ref_model.llm.generate(
            [TokensPrompt(prompt_token_ids=prefix) for prefix, _, _ in checks],
            [
                SamplingParams(temperature=0.0, max_tokens=1, logprobs=2 * top_k)
                for _, _, top_k in checks
            ],
        )

    for (_, token, top_k), out in zip(checks, ranked):
        top = out.outputs[0].logprobs[0]
        assert token in top, f"top_k={top_k} sampled {token} outside {list(top)}"
    if whole_model:
        assert accepted > 0, f"spec:{method} accepted no draft tokens"
