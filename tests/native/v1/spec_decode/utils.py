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

"""Builders for the native eagle proposer tests: a real RBLNEagleProposer from
upstream's eagle model pair (config-only fetch), with the compiled model left
unset for the caller to mock."""

from __future__ import annotations

import functools
from typing import Any

import torch
from vllm.platforms import current_platform

# Upstream's eagle test pair. Only the configs are read at construction -- no
# weights, no compile.
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_DRAFT = {
    "eagle": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "eagle3": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
}


@functools.cache
def _spec_vllm_config(method: str, num_speculative_tokens: int) -> Any:
    # Memoized: the config build (HF resolution + check_and_update_config)
    # dominates, while the proposer's own allocation is cheap.
    from tests.native.vllm_config import local_model_path, make_vllm_config

    return make_vllm_config(
        model=TARGET_MODEL,
        speculative_config={
            "method": method,
            "model": local_model_path(_DRAFT[method]),
            "num_speculative_tokens": num_speculative_tokens,
        },
    )


def make_cad(query_start_loc: list[int], seq_lens: list[int]) -> Any:
    """A CommonAttentionMetadata with the fields the eagle methods read, derived
    from the per-request query_start_loc and seq_lens."""
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    qsl = torch.tensor(query_start_loc, dtype=torch.int32)
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=sl,
        num_reqs=len(seq_lens),
        num_actual_tokens=int(qsl[-1]),
        max_query_len=int((qsl[1:] - qsl[:-1]).max()),
        max_seq_len=int(sl.max()),
        block_table_tensor=torch.zeros((len(seq_lens), 4), dtype=torch.int32),
        slot_mapping=torch.arange(int(qsl[-1]), dtype=torch.int64),
        causal=True,
        dcp_local_seq_lens=None,
    )


def make_eagle_proposer(
    *,
    method: str = "eagle",
    num_speculative_tokens: int = 3,
    runner: Any = None,
) -> Any:
    """A real RBLNEagleProposer built through its actual constructor on the
    current platform's device (cpu under --device-tensor 0, rbln under 1). The
    compiled model is left unset; attach a mock for the propose/load paths."""
    from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer

    vllm_config = _spec_vllm_config(method, num_speculative_tokens)
    device = torch.device(current_platform.device_type)
    return RBLNEagleProposer(vllm_config, device, runner)


# vLLM's tiny random medusa test pair. JackFram/llama-68m caps at 2048 positions,
# so max_model_len must be lowered from the native default.
MEDUSA_TARGET = "JackFram/llama-68m"
MEDUSA_DRAFT = "abhigoyal/vllm-medusa-llama-68m-random"

_DEFAULT_COMPILE_CONTEXT = object()


@functools.cache
def _medusa_vllm_config(num_speculative_tokens: int) -> Any:
    from tests.native.vllm_config import local_model_path, make_vllm_config

    return make_vllm_config(
        model=MEDUSA_TARGET,
        max_model_len=2048,
        speculative_config={
            "method": "medusa",
            "model": local_model_path(MEDUSA_DRAFT),
            "num_speculative_tokens": num_speculative_tokens,
        },
    )


def make_medusa_proposer(
    *,
    num_speculative_tokens: int = 3,
    compile_context: Any = _DEFAULT_COMPILE_CONTEXT,
) -> Any:
    """A real RBLNMedusaProposer. compile_context defaults to a sentinel so
    construction never calls create_compile_context; pass None for that branch."""
    from vllm_rbln.v1.spec_decode.medusa import RBLNMedusaProposer

    vllm_config = _medusa_vllm_config(num_speculative_tokens)
    device = torch.device(current_platform.device_type)
    cc = object() if compile_context is _DEFAULT_COMPILE_CONTEXT else compile_context
    return RBLNMedusaProposer(vllm_config, device, compile_context=cc)
