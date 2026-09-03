# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer

# Proposers that load a draft model and run it through the RBLN compile path:
# they own draft attention groups, a warmup, and weights the runner has to make
# contiguous. The n-gram and suffix proposers do none of that.
DRAFT_MODEL_PROPOSERS = (RBLNEagleProposer, RBLNDFlashProposer)

__all__ = [
    "DRAFT_MODEL_PROPOSERS",
    "RBLNDFlashProposer",
    "RBLNEagleProposer",
]
