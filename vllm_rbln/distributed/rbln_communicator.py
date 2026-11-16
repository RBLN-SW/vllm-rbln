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

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


# RBLN custom communicator (vllm/distributed/device_communicators/...)
class RblnCommunicator(DeviceCommunicatorBase):
    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if dim == -1:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()

            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            if dim == 2:
                # (0,1,2,3) -> movedim(0, 2) -> (1,2,0,3)
                output_tensor = output_tensor.permute(1, 2, 0, 3)
            elif dim == 1:
                # (0,1,2) -> movedim(0, 1) -> (1,0,2)
                output_tensor = output_tensor.permute(1, 0, 2)
            else:
                raise AssertionError("RBLN all_gather move_dim=1, 2, NYI")
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        elif dim == 0:
            pass
        else:
            # Reshape
            output_tensor = output_tensor.reshape((self.world_size,) + input_size)
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
            raise AssertionError("RBLN all_gather dim!=0, dim!=-1, NYI")

        return output_tensor
