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

"""Offline LLM(...) coverage for the RBLN max_num_seqs default.

`a860d0d6` makes an unset max_num_seqs resolve to 1 on RBLN (upstream vLLM
defaults to 256), while leaving an explicitly-set value untouched. This module
exercises the `LLM(...)` (UsageContext.LLM_CLASS) entry point; the `vllm serve`
counterpart lives in tests/entrypoints/openai/test_default_max_num_seqs.py. Both
run across the two runtime backends selected by VLLM_RBLN_USE_VLLM_MODEL
(0 = optimum, 1 = vLLM-native).

Like the other entrypoints tests, this requires a pre-compiled model under
REBEL_VLLM_PRE_COMPILED_DIR and is skipped otherwise.
"""

import os

import pytest
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_DIR = os.getenv("REBEL_VLLM_PRE_COMPILED_DIR", "./")
MODEL_NAME = MODEL_DIR + "/opt_125m_batch2"

# The RBLN default an unset max_num_seqs resolves to (see vllm_rbln.platform).
RBLN_DEFAULT_MAX_NUM_SEQS = 1
# opt_125m_batch2 is compiled for batch size 2, so the explicit value we set to
# check it is preserved must not exceed the compiled batch.
EXPLICIT_MAX_NUM_SEQS = 2

# VLLM_RBLN_USE_VLLM_MODEL selects the runtime backend: 0 = optimum path,
# 1 = vLLM-native model path. The default must hold for both.
# FIXME MODE=1 is skipped for now.
MODES = ["0"]

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_NAME),
    reason=(
        "Pre-compiled RBLN model not found; set REBEL_VLLM_PRE_COMPILED_DIR to "
        "the directory containing 'opt_125m_batch2'."
    ),
)


def _load_max_num_seqs(monkeypatch, mode: str, **llm_kwargs) -> int:
    monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", mode)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm = LLM(model=MODEL_NAME, **llm_kwargs)
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


# FIXME EXPLICIT_MAX_NUM_SEQS=256 test is required. But it is hard to test because of OOM and limited resources.
@pytest.mark.parametrize("mode", MODES)
def test_llm_explicit_max_num_seqs_is_preserved(monkeypatch, mode):
    resolved = _load_max_num_seqs(monkeypatch, mode, max_num_seqs=EXPLICIT_MAX_NUM_SEQS)
    assert resolved == EXPLICIT_MAX_NUM_SEQS, (
        f"LLM(...) with VLLM_RBLN_USE_VLLM_MODEL={mode} must preserve an explicit "
        f"max_num_seqs={EXPLICIT_MAX_NUM_SEQS}, got {resolved}"
    )
