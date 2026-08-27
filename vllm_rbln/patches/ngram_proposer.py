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
"""Make the ngram proposer's numba JIT actually compile during construction.

Upstream ends `NgramProposer.__init__` with a `self.propose(...)` call whose
comment says it triggers the numba JIT. It does not: every entry of the
`sampled_token_ids` it passes is empty, `propose` skips requests with no sampled
ids, and `batch_propose` skips the numba call entirely when that leaves no valid
request. The JIT therefore fires on the first real proposal instead, inside a
serving step.

The same call with one non-empty row compiles the kernels. numba specializes on
dtype and ndim rather than length, so a single row covers the full-batch call.
"""

import functools

import numpy as np
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

_init = NgramProposer.__init__


@register_patch(
    target="vllm.v1.spec_decode.ngram_proposer.NgramProposer.__init__",
    reason=(
        "Upstream's JIT warm-up call passes only empty sampled_token_ids, so it "
        "compiles nothing and the numba JIT lands in the first serving step. "
        "TODO(vllm>0.24.0): delete once upstream's warm-up reaches the kernel."
    ),
    key="vllm_rbln.patches.ngram_proposer.__init__",
    owner_module="vllm_rbln.patches.ngram_proposer",
)
@functools.wraps(_init)
def __init__(self, vllm_config, *args, **kwargs) -> None:
    _init(self, vllm_config, *args, **kwargs)
    self.propose(
        self.k,
        [[0]],
        np.zeros(1, dtype=np.int32),
        np.zeros((1, self.max_model_len), dtype=np.int32),
    )
