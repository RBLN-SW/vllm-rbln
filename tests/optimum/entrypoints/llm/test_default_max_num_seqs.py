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

import pytest
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "facebook/opt-125m"
BLOCK_SIZE = 2048

RBLN_DEFAULT_MAX_NUM_SEQS = 1
EXPLICIT_MAX_NUM_SEQS = 2
ORIGINAL_MAX_NUM_SEQS = 256
# FIXME MODE=1 is skipped for now.
MODES = ["0"]


def _load_max_num_seqs(monkeypatch, mode: str, **llm_kwargs) -> int:
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", mode)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm = LLM(model=MODEL_NAME, block_size=BLOCK_SIZE, **llm_kwargs)
    try:
        return llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
    finally:
        del llm
        try:
            cleanup_dist_env_and_memory()
        except RuntimeError as e:
            # Raised on the optimum path where no accelerator device is bound.
            if "Cannot access accelerator device when none is available" not in str(e):
                raise


@pytest.mark.parametrize("mode", MODES)
def test_llm_unset_max_num_seqs_defaults_to_one(monkeypatch, mode):
    resolved = _load_max_num_seqs(monkeypatch, mode)
    assert resolved == RBLN_DEFAULT_MAX_NUM_SEQS, (
        f"LLM(...) with VLLM_RBLN_USE_VLLM_MODEL={mode} should default an unset "
        f"max_num_seqs to {RBLN_DEFAULT_MAX_NUM_SEQS}, got {resolved}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_llm_explicit_none_max_num_seqs_defaults_to_one(monkeypatch, mode):
    """Passing max_num_seqs=None explicitly behaves like omitting it.

    The override resolves an unset value to the RBLN default; an explicit None
    must take the same path (the resolution only checks `is None`). The serve
    side has no analogue because its argparse rejects "None"/"null"/"" outright.
    """
    resolved = _load_max_num_seqs(monkeypatch, mode, max_num_seqs=None)
    assert resolved == RBLN_DEFAULT_MAX_NUM_SEQS, (
        f"LLM(max_num_seqs=None) with VLLM_RBLN_USE_VLLM_MODEL={mode} should "
        f"default to {RBLN_DEFAULT_MAX_NUM_SEQS}, got {resolved}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_llm_explicit_max_num_seqs_is_preserved(monkeypatch, mode):
    resolved = _load_max_num_seqs(monkeypatch, mode, max_num_seqs=EXPLICIT_MAX_NUM_SEQS)
    assert resolved == EXPLICIT_MAX_NUM_SEQS, (
        f"LLM(...) with VLLM_RBLN_USE_VLLM_MODEL={mode} must preserve an explicit "
        f"max_num_seqs={EXPLICIT_MAX_NUM_SEQS}, got {resolved}"
    )


def test_original_max_seqs_is_preserved_without_model():
    """Explicitly requesting the upstream default max_num_seqs (256) keeps it 256.

    The RBLN override only rewrites the default for an *unset* max_num_seqs
    (upstream's 256 -> 1); an explicitly-set value must pass through untouched.
    256 is the value that matters here: it is upstream's original implicit
    default, the exact number the override now reinterprets when max_num_seqs is
    left unset. So we confirm that asking for it explicitly still resolves to
    256, not the RBLN default of 1.

    A 256-batch model would OOM here, so instead of serving one we drive vLLM's
    own unset-default resolution directly, with the override applied and a stub
    model_config, so no model is loaded.
    """
    from types import SimpleNamespace

    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    from vllm_rbln.platform import RblnPlatform

    # Apply the override exactly as pre_register_and_update does in production.
    RblnPlatform._override_default_max_num_seqs()

    # model is never loaded (no create_model_config call); max_num_batched_tokens
    # is pinned so the stub model_config is never dereferenced.
    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_num_seqs=ORIGINAL_MAX_NUM_SEQS,
        max_num_batched_tokens=2048,
    )
    engine_args._set_default_max_num_seqs_and_batched_tokens_args(
        UsageContext.LLM_CLASS,
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(use_batched_dp_moe=False),
    )

    assert engine_args.max_num_seqs == ORIGINAL_MAX_NUM_SEQS, (
        "the RBLN override must leave an explicitly-set max_num_seqs "
        f"({ORIGINAL_MAX_NUM_SEQS}) untouched, got {engine_args.max_num_seqs}"
    )
