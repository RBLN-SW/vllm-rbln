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

from vllm.config import ParallelConfig
from vllm.distributed import (
    get_dp_group,
    get_pcp_group
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

@staticmethod
def __fused_moe_parallel_config_make(
        tp_size_: int,
        pcp_size_: int,
        dp_size_: int,
        vllm_parallel_config: ParallelConfig,
    ) -> "FusedMoEParallelConfig":
        """
        Determine MoE parallel configuration. Based on the input `tp_size_`,
        `dp_size_` and vllm's parallel config, determine what
        level's of parallelism to use in the fused moe layer.

        Args:
            tp_size_ (int): `tp_size` passed into the FusedMoE constructor.
            pcp_size_ (int): `pcp_size` passed into the FusedMoE constructor.
            dp_size_ (int): `dp_size` passed into the FusedMoE constructor.
            vllm_parallel_config (ParallelConfig): vLLM's parallel config
                object which contains the `enable_expert_parallel` flag.

        Examples:
            When there is no parallelism requested,
            i.e. `tp_size_` = `pcp_size_` = `dp_size_` = 1, we simply return the sizes
            unaltered and the ranks set to 0.

            Expert Parallelism is considered only when either `dp_size_`, `pcp_size_` or
            `tp_size_` is non trivial.

            Note that PCP serves the same function as DP here.

            When TP = 2, DP(PCP) = 1 and EP = False, the configuration on different
            devices:

            - device 0 : TP = {2, 0} DP = {1, 0} EP = {1, 0} //
                legend : {size, rank}
            - device 1 : TP = {2, 1} DP = {1, 0} EP = {1, 0}
            - Comment : Tensors are sharded across 2 devices.

            When TP = 1, DP(PCP) = 2 and EP = False, the configuration on different
                devices:

            - device 0 : TP = {2, 0} DP = {2, 0} EP = {1, 0}
            - device 1 : TP = {2, 1} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 2 decvices.

            When TP = 2, DP(PCP) = 2 and EP = False, the configuration on different
                devices:

            - device 0: TP = {4, 0} DP = {2, 0} EP = {1, 0}
            - device 1: TP = {4, 1} DP = {2, 0} EP = {1, 0}
            - device 2: TP = {4, 2} DP = {2, 1} EP = {1, 0}
            - device 3: TP = {4, 3} DP = {2, 1} EP = {1, 0}
            - Comment: There are 2 engine instances and the tensors are sharded
                across 4 devices.

            When, TP = 2, DP(PCP) = 1 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {1, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {1, 0} EP = {2, 1}
            - Comment: The experts are split between the 2 devices.

            When, TP = 1, DP(PCP) = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
            - device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
            - Comment: There are 2 engine instances and the experts are split
                between the 2 devices.

            When TP = 2, DP(PCP) = 2 and EP = True, the configuration on different
                devices:

            - device 0: TP = {1, 0} DP = {2, 0} EP = {4, 0}
            - device 1: TP = {1, 0} DP = {2, 0} EP = {4, 1}
            - device 2: TP = {1, 0} DP = {2, 1} EP = {4, 2}
            - device 3: TP = {1, 0} DP = {2, 1} EP = {4, 3}
            - Comment: There are 2 engine instances and the experts are split
                between the 4 devices.
        """

        use_ep = (
            dp_size_ * pcp_size_ * tp_size_ > 1
            and vllm_parallel_config.enable_expert_parallel
        )

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        pcp_size = pcp_size_
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0

        if not use_ep:
            # FIXME(RBLN) - disable flatten_tp_across_dp for data parallel w/o ep
            tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
                tp_size_, 1, 0, pcp_size_, pcp_rank
            )
            return FusedMoEParallelConfig(
                tp_size=tp_size,
                tp_rank=tp_rank,
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=1,
                ep_rank=0,
                use_ep=False,
                all2all_backend=vllm_parallel_config.all2all_backend,
            )
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size_, dp_size_, dp_rank, pcp_size_, pcp_rank
        )
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            use_ep=True,
            all2all_backend=vllm_parallel_config.all2all_backend,
        )

FusedMoEParallelConfig.make = __fused_moe_parallel_config_make
