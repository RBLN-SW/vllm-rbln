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

from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod

from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    RBLNUnquantizedFusedMoEMethod,
)
from vllm_rbln.patches import add_registration

logger = init_logger(__name__)


@add_registration(
    reason="Register RBLNUnquantizedFusedMoEMethod for vLLM OOT platform."
)
def register_rbln_unquantized_fused_moe_method() -> None:
    UnquantizedFusedMoEMethod.register_oot(RBLNUnquantizedFusedMoEMethod)
