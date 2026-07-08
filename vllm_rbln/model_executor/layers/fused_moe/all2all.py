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

"""Expert-mask generation helpers for MoE expert parallelism.

These utilities build the rank/expert routing masks consumed by the CCL
all2all custom ops. The ops themselves (ccl_dispatch_send / ccl_all2all_x_kernel
/ ccl_dispatch_receive / ccl_combine_send / ccl_combine_receive) are registered
by rebel_compiler (rebel.ops.torch_custom_ops).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Mask generation helpers
# ---------------------------------------------------------------------------


def generate_expert_mask(R: int, E: int) -> np.ndarray:
    """Expert ownership mask (R, E). Local expert index or -1."""
    mask = np.ones((R, E), dtype=int) * -1
    local_cnt = E // R
    for i in range(R):
        for j in range(local_cnt):
            mask[i, j + i * local_cnt] = j
    return mask


def prepare_send_mask_matrix(R: int, E: int) -> np.ndarray:
    """(R, E) send mask — rank-independent expert-to-rank mapping.

    send_mask[dst, e] = 1 if expert e belongs to rank dst.
    """
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    return expert_binary
