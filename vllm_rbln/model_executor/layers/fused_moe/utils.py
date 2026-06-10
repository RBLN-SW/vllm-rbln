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
from vllm.forward_context import get_forward_context


def get_tokens_mask(num_tokens: int, left=1.0, right=0.0) -> torch.Tensor:
    """Real-vs-padding mask aligned with the DP multicast output layout.

    For every DP rank's slot in the multicast buffer, positions before
    that rank's actual token count get ``left`` and the reset get ``right``.
    Multiply this mask into routing weights (default ``(1.0, 0.0)``) or,
    with ``(0.0, float('-inf'))``, add it to router logits before softmax
    to suppress padded positions.

    In the DP=1 path no padding exists, so ``num_tokens`` is used as the
    pad length and the result is all ``left`` (effectively a no-op).

    Example:
        DP=2, ``max_pad=4``, rank 0 has 3 real tokens, rank 1 has 2.
        With defaults ``(left=1.0, right=0.0)``::

            rank 0 slot: [1.0, 1.0, 1.0, 0.0]
            rank 1 slot: [1.0, 1.0, 0.0, 0.0]

        Flattened return: ``[[1.0],[1.0],[1.0],[0.0],[1.0],[1.0],[0.0],[0.0]]``
        with shape ``[8, 1]``.

    Args:
        num_tokens: Used as ``max_pad`` only when DP=1 (where the metadata's
            ``max_pads_across_dp`` is ``None``); ignored otherwise
        left: Value for real-token positions.
        right: Value for padded positions.

    Returns:
        Tensor of shape ``[dp_size * max_pad, 1]``.
    """
    assert (dp_metadata := get_forward_context().dp_metadata) is not None
    num_tokens_across_dp = dp_metadata.num_tokens_across_dp_cpu.unsqueeze(1)

    max_pad = (
        num_tokens
        if num_tokens_across_dp.shape[0] == 1  # DP=1
        else dp_metadata.max_pads_across_dp.shape[0]
    )
    pos = torch.arange(max_pad, dtype=torch.int32).unsqueeze(0)

    tokens_mask = torch.where(pos < num_tokens_across_dp, left, right)
    tokens_mask = tokens_mask.reshape(-1, 1)
    return tokens_mask
