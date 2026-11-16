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

from typing import ClassVar

import torch


class LoRAMask:
    lora_mask: ClassVar[torch.Tensor]  # [batch_size, max_loras * max_rank]

    @classmethod
    def set_lora_mask(cls, mask: torch.Tensor) -> None:
        cls.lora_mask = mask

    @classmethod
    def get_lora_mask(cls) -> torch.Tensor:
        return cls.lora_mask
