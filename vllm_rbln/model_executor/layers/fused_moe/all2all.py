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

"""All2All dispatch helpers for MoE expert parallelism.

This module implements the naive P2P all2all communication pattern
used to dispatch tokens across data-parallel ranks in MoE layers. It contains:

- Expert mask generation utilities
- CCL custom ops for dispatch_send, all2all_x, and dispatch_receive kernels
- The CCL group ID constant
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


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


# ---------------------------------------------------------------------------
# Custom op: ccl_dispatch_send
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_dispatch_send",
    mutates_args=(),
)
def ccl_dispatch_send(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor]:
    t_dim = hidden_states.shape[0]
    H_dim = hidden_states.shape[1]
    R_dim = send_mask.shape[0]

    send_logit = torch.matmul(send_mask, router_logits)  # (R, t)

    send_buffer = torch.zeros(R_dim, t_dim, H_dim, dtype=hidden_states.dtype)
    send_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = send_logit[r].nonzero(as_tuple=True)[0]
        send_buffer[r, : valid_idx.shape[0]] = hidden_states[valid_idx]
        send_sizes[r, 0] = valid_idx.shape[0]

    return send_buffer, send_sizes


@ccl_dispatch_send.register_fake
def _ccl_dispatch_send_fake(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor]:
    t_dim = hidden_states.shape[0]
    H_dim = hidden_states.shape[1]
    R_dim = send_mask.shape[0]
    return (
        torch.empty(R_dim, t_dim, H_dim, dtype=hidden_states.dtype),
        torch.empty(R_dim, 64, dtype=torch.uint16),
    )


# ---------------------------------------------------------------------------
# Custom op: ccl_all2all_x_kernel  (naive P2P — no recv_sizes)
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_all2all_x_kernel",
    mutates_args=(),
)
def ccl_all2all_x_kernel(
    send_buffer: Tensor,
    send_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    # CPU stub — returns zeros of correct shape.
    # Real communication happens on device via CCL runtime (rcclAllToAllX).
    R = ccl_world_size
    t = send_buffer.shape[1]
    H = send_buffer.shape[2]
    return torch.zeros(R, t, H, dtype=send_buffer.dtype)


@ccl_all2all_x_kernel.register_fake
def _ccl_all2all_x_kernel_fake(
    send_buffer: Tensor,
    send_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    R_dim = ccl_world_size
    t_dim = send_buffer.shape[1]
    H_dim = send_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=send_buffer.dtype)


# ---------------------------------------------------------------------------
# Custom op: ccl_dispatch_receive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_dispatch_receive",
    mutates_args=(),
)
def ccl_dispatch_receive(
    recv_buffer: Tensor,
    router_logits: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]

    # e-reduction: (e, T) → (T,) → (R, t)
    recv_logit = router_logits.sum(dim=0).reshape(R_dim, t_dim)

    # nonzero
    t_padded = (t_dim + 63) // 64 * 64
    recv_indices = torch.full((R_dim, t_padded), 65535, dtype=torch.uint16)
    recv_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = recv_logit[r].nonzero(as_tuple=True)[0].to(torch.uint16)
        recv_indices[r, : valid_idx.shape[0]] = valid_idx
        recv_sizes[r, 0] = valid_idx.shape[0]

    # scatter + replace rank_id slot
    unpacked = torch.zeros(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)
    for r in range(R_dim):
        if r == rank_id:
            unpacked[r] = hidden_states
        else:
            num_valid = int(recv_sizes[r, 0])
            valid_idx = recv_indices[r, :num_valid].long()
            unpacked[r, valid_idx] = recv_buffer[r, :num_valid]

    return unpacked


@ccl_dispatch_receive.register_fake
def _ccl_dispatch_receive_fake(
    recv_buffer: Tensor,
    router_logits: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)


# ---------------------------------------------------------------------------
# Custom op: ccl_combine_send
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_combine_send",
    mutates_args=(),
)
def ccl_combine_send(
    hidden_states: Tensor,
    router_logits: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor]:
    """Combine send: pack per-rank expert outputs for all2all exchange.

    Reverse of dispatch_receive — uses sum-reduction over local experts.

    Args:
        hidden_states: (R, t, H) — per-rank token embeddings after expert processing
        router_logits: (e, T) — local expert routing logits, e=E/R, T=R*t
        rank_id: this rank's ID

    Returns:
        send_buffer: (R, t, H) — packed hidden states per dest rank
        send_sizes: (R, 64) — uint16 valid count per dest rank
    """
    R_dim = hidden_states.shape[0]
    t_dim = hidden_states.shape[1]
    H_dim = hidden_states.shape[2]

    # e-reduction via sum: (e, T) -> (T,) -> (R, t)
    send_logit = router_logits.sum(dim=0).reshape(R_dim, t_dim)

    send_buffer = torch.zeros(R_dim, t_dim, H_dim, dtype=hidden_states.dtype)
    send_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = send_logit[r].nonzero(as_tuple=True)[0]
        send_buffer[r, : valid_idx.shape[0]] = hidden_states[r, valid_idx]
        send_sizes[r, 0] = valid_idx.shape[0]

    return send_buffer, send_sizes


@ccl_combine_send.register_fake
def _ccl_combine_send_fake(
    hidden_states: Tensor,
    router_logits: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor]:
    R_dim = hidden_states.shape[0]
    t_dim = hidden_states.shape[1]
    H_dim = hidden_states.shape[2]
    return (
        torch.empty(R_dim, t_dim, H_dim, dtype=hidden_states.dtype),
        torch.empty(R_dim, 64, dtype=torch.uint16),
    )


# ---------------------------------------------------------------------------
# Custom op: ccl_combine_receive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_combine_receive",
    mutates_args=(),
)
def ccl_combine_receive(
    recv_buffer: Tensor,
    router_logits: Tensor,
    expert_map: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    """Combine receive: unpack recv_buffer and sum-reduce over R ranks.

    Reverse of dispatch_send — uses expert_map @ router_logits matmul.

    Args:
        recv_buffer: (R, t, H) — packed received data per source rank
        router_logits: (E, t) — this rank's routing logits (full experts)
        expert_map: (R, E) — per-rank expert map (identity-based)
        hidden_states: (t, H) — local rank's hidden states
        rank_id: this rank's ID

    Returns:
        output: (t, H) — sum-reduced over R ranks
    """
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]

    # recv_logit = expert_map @ router_logits -> (R, t)
    recv_logit = torch.matmul(expert_map, router_logits)  # (R, t)

    # nonzero -> recv_indices
    t_padded = (t_dim + 63) // 64 * 64
    recv_indices = torch.full((R_dim, t_padded), 65535, dtype=torch.uint16)
    recv_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = recv_logit[r].nonzero(as_tuple=True)[0].to(torch.uint16)
        recv_indices[r, : valid_idx.shape[0]] = valid_idx
        recv_sizes[r, 0] = valid_idx.shape[0]

    # scatter + replace rank_id slot
    unpacked = torch.zeros(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)
    for r in range(R_dim):
        if r == rank_id:
            unpacked[r] = hidden_states
        else:
            num_valid = int(recv_sizes[r, 0])
            valid_idx = recv_indices[r, :num_valid].long()
            unpacked[r, valid_idx] = recv_buffer[r, :num_valid]

    # sum-reduce over R ranks
    return unpacked.sum(dim=0)


@ccl_combine_receive.register_fake
def _ccl_combine_receive_fake(
    recv_buffer: Tensor,
    router_logits: Tensor,
    expert_map: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    return torch.empty(t_dim, H_dim, dtype=recv_buffer.dtype)


# ---------------------------------------------------------------------------
# CCL All2All group ID
# ---------------------------------------------------------------------------
CCL_ALL2ALL_GROUP_ID = 42  ## may make problem in multiple layers ?
