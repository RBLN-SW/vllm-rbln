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

from vllm_rbln.logger import init_logger
from vllm_rbln.patches.registry import add_registration

logger = init_logger(__name__)


def _register_oot(base_cls, rbln_cls) -> None:
    base_cls.register_oot(rbln_cls)
    logger.debug(
        "Registered RBLN OOT implementation: %s -> %s",
        base_cls.__name__,
        rbln_cls.__name__,
    )


def _register_fp8_block_kernel() -> None:
    from vllm.model_executor.kernels import linear
    from vllm.platforms import PlatformEnum

    from vllm_rbln.model_executor.kernels.linear.block_fp8 import (
        RBLNW8A16BlockFp8LinearKernel,
    )

    block_kernels = linear._POSSIBLE_FP8_BLOCK_KERNELS.setdefault(PlatformEnum.OOT, [])
    if RBLNW8A16BlockFp8LinearKernel not in block_kernels:
        block_kernels.insert(0, RBLNW8A16BlockFp8LinearKernel)

        logger.debug(
            "Registered RBLN FP8 block linear kernel for OOT platform: %s",
            RBLNW8A16BlockFp8LinearKernel.__name__,
        )


@add_registration(reason="Register RBLN OOT implementations.")
def register_rbln_oot_implementations() -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from vllm_rbln.model_executor.layers.fused_moe.layer import FusedMoE, RBLNFusedMoE
    from vllm_rbln.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        RBLNUnquantizedFusedMoEMethod,
        UnquantizedFusedMoEMethod,
    )
    from vllm_rbln.model_executor.layers.vocab_parallel_embedding import (
        RBLNParallelLMHead,
        RBLNVocabParallelEmbedding,
    )

    _register_oot(FusedMoE, RBLNFusedMoE)
    _register_oot(UnquantizedFusedMoEMethod, RBLNUnquantizedFusedMoEMethod)
    _register_oot(VocabParallelEmbedding, RBLNVocabParallelEmbedding)
    _register_oot(ParallelLMHead, RBLNParallelLMHead)
    _register_fp8_block_kernel()
