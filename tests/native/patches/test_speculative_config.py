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

"""The draft's parallel config stays off the target's pipeline split.

Config objects only -- no checkpoint, no device.
"""

from __future__ import annotations

import pytest
from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_rbln.patches.speculative_config import create_draft_parallel_config


@pytest.mark.parametrize("target_pp", [1, 2, 4])
def test_draft_never_inherits_target_pipeline_size(target_pp):
    target = ParallelConfig(pipeline_parallel_size=target_pp, tensor_parallel_size=1)

    draft = create_draft_parallel_config(target, 1)

    assert draft.pipeline_parallel_size == 1
    # world_size is derived in the validator, so it has to come out right at
    # construction; there is no field to correct afterwards.
    assert draft.world_size == 1


def test_the_patch_is_the_one_installed():
    assert (
        SpeculativeConfig.create_draft_parallel_config is create_draft_parallel_config
    )


def test_tensor_parallel_size_still_comes_from_the_argument():
    target = ParallelConfig(pipeline_parallel_size=4, tensor_parallel_size=1)

    draft = create_draft_parallel_config(target, 2)

    assert draft.tensor_parallel_size == 2
    assert draft.world_size == 2


def test_the_draft_head_is_what_fails_supports_pp():
    # Why the inherited pipeline size is fatal rather than merely wasteful: the
    # draft head reaches verify_with_parallel_config and cannot satisfy it. If
    # upstream ever gives the head an intermediate_tensors forward, this flips and
    # the patch has lost its reason.
    assert supports_pp(LlamaForCausalLM)
    assert not supports_pp(Eagle3LlamaForCausalLM)
