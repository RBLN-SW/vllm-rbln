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

This module implements the recursive-doubling all2all communication pattern
used to dispatch tokens across data-parallel ranks in MoE layers. It contains:

- Hypercube mask generation utilities (send / receive routing matrices)
- CCL custom ops for send, all2all, and receive kernels
- The CCL group ID constant
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# All2All mask generation helpers
# (recursive doubling all2all implementation)
# ---------------------------------------------------------------------------


def generate_H_matrix(R: int, my_rank: int) -> np.ndarray:
    """Hypercube send-stage mask (N, R) where N=log2(R)."""
    N = int(np.log2(R))
    H = np.zeros((N, R), dtype=int)
    for s in range(N):
        b = N - 1 - s
        for d in range(R):
            if ((d >> (b + 1)) == (my_rank >> (b + 1))) and (
                ((d >> b) & 1) != ((my_rank >> b) & 1)
            ):
                H[s, d] = 1
    return H


def generate_W_matrix(R: int, my_rank: int) -> np.ndarray:
    """Receive routing matrix (R, R)."""
    W = np.zeros((R, R), dtype=int)
    for i in range(R):
        if i == my_rank:
            W[i, my_rank] = 1
        else:
            diff = i ^ my_rank
            b = diff.bit_length() - 1
            for d in range(R):
                if (d >> b) == (my_rank >> b):
                    W[i, d] = 1
    return W


def generate_expert_mask(R: int, E: int) -> np.ndarray:
    """Expert ownership mask (R, E). Local expert index or -1."""
    mask = np.ones((R, E), dtype=int) * -1
    local_cnt = E // R
    for i in range(R):
        for j in range(local_cnt):
            mask[i, j + i * local_cnt] = j
    return mask


def prepare_send_mask_matrix(R: int, my_rank: int, E: int) -> np.ndarray:
    """(N, E) send mask for each hypercube stage."""
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    return np.matmul(generate_H_matrix(R, my_rank), expert_binary)


def prepare_recv_mask_matrix(R: int, my_rank: int, E: int) -> np.ndarray:
    """(R, E) recv mask."""
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    return np.matmul(generate_W_matrix(R, my_rank), expert_binary)


# ---------------------------------------------------------------------------
# All2All mask generation helpers — naive P2P (AllToAllX)
# ---------------------------------------------------------------------------


def prepare_send_mask_matrix_p2p(R: int, my_rank: int, E: int) -> np.ndarray:
    """(R, E) naive P2P send mask — one row per destination rank.

    send_mask[dst, e] = 1 if expert e belongs to rank dst and dst != my_rank.
    """
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    send_mask = expert_binary.copy()
    send_mask[my_rank, :] = 0  # no self-send via CCL
    return send_mask


def prepare_recv_mask_matrix_p2p(R: int, my_rank: int, E: int) -> np.ndarray:
    """(R, E) naive P2P recv mask.

    recv_mask[src, e] = 1 if expert e is assigned to my_rank (I own it),
    so I expect to receive tokens for my local experts from every source.
    """
    expert_binary = np.where(generate_expert_mask(R, E) >= 0, 1, 0)
    recv_mask = np.tile(expert_binary[my_rank : my_rank + 1, :], (R, 1))
    return recv_mask


# ---------------------------------------------------------------------------
# Custom op: ccl_send_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_send_kernel",
    mutates_args=(),
)
def ccl_send_kernel(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    recv_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = hidden_states.dtype
    ltype = router_logits.dtype
    send_mask = send_mask.to(ltype)
    recv_mask = recv_mask.to(ltype)

    t_dim = hidden_states.shape[0]
    t_padded = (t_dim + 63) // 64 * 64
    H_dim = hidden_states.shape[1]
    N_dim = send_mask.shape[0]
    R_dim = recv_mask.shape[0]

    local_router = router_logits[:, rank_id, :]
    send_buffer_logit = torch.matmul(send_mask, local_router)

    send_buffer = torch.zeros(N_dim, t_dim, H_dim, dtype=hidden_states.dtype)
    send_sizes = torch.zeros(N_dim, 64, dtype=torch.uint16)
    for s in range(N_dim):
        valid_idx = send_buffer_logit[s].nonzero(as_tuple=True)[0]
        send_buffer[s, : valid_idx.shape[0]] = hidden_states[valid_idx]
        send_sizes[s, 0] = valid_idx.shape[0]

    recv_buffer_logit = torch.einsum("re,ert->rt", recv_mask, router_logits)

    recv_indices = torch.full((R_dim, t_padded), 65535, dtype=torch.uint16)
    recv_sizes = torch.zeros(R_dim, 64, dtype=torch.uint16)
    for r in range(R_dim):
        valid_idx = recv_buffer_logit[r].nonzero(as_tuple=True)[0].to(torch.uint16)
        recv_indices[r, : valid_idx.shape[0]] = valid_idx
        recv_sizes[r, 0] = valid_idx.shape[0]

    return send_buffer, recv_indices, send_sizes, recv_sizes


@ccl_send_kernel.register_fake
def _ccl_send_kernel_fake(
    hidden_states: Tensor,
    router_logits: Tensor,
    send_mask: Tensor,
    recv_mask: Tensor,
    rank_id: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    t_dim = hidden_states.shape[0]
    t_padded = (t_dim + 63) // 64 * 64
    N_dim = send_mask.shape[0]
    R_dim = recv_mask.shape[0]
    H_dim = hidden_states.shape[1]
    return (
        torch.empty(N_dim, t_dim, H_dim, dtype=hidden_states.dtype),
        torch.empty(R_dim, t_padded, dtype=torch.uint16),
        torch.empty(N_dim, 64, dtype=torch.uint16),
        torch.empty(R_dim, 64, dtype=torch.uint16),
    )


# ---------------------------------------------------------------------------
# Custom op: ccl_all2all_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_all2all_kernel",
    mutates_args=(),
)
def ccl_all2all_kernel(
    send_buffer: Tensor,
    send_sizes: Tensor,
    recv_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    # CPU stub — returns zeros of correct shape.
    # Real communication happens on device via CCL runtime.
    R = ccl_world_size
    t = send_buffer.shape[1]
    H = send_buffer.shape[2]
    return torch.zeros(R, t, H, dtype=send_buffer.dtype)


@ccl_all2all_kernel.register_fake
def _ccl_all2all_kernel_fake(
    send_buffer: Tensor,
    send_sizes: Tensor,
    recv_sizes: Tensor,
    ccl_world_size: int,
    group_id: int,
) -> Tensor:
    R_dim = recv_sizes.shape[0]
    t_dim = send_buffer.shape[1]
    H_dim = send_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=send_buffer.dtype)


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
# Custom op: ccl_receive_kernel
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "rbln_custom_ops::ccl_receive_kernel",
    mutates_args=(),
)
def ccl_receive_kernel(
    recv_buffer: Tensor,
    recv_indices: Tensor,
    recv_sizes: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    unpacked = torch.zeros(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)
    for r in range(R_dim):
        if r == rank_id:
            unpacked[r] = hidden_states
        else:
            num_valid = int(recv_sizes[r, 0])
            valid_idx = recv_indices[r, :num_valid].long()
            unpacked[r, valid_idx] = recv_buffer[r, :num_valid]
    return unpacked


@ccl_receive_kernel.register_fake
def _ccl_receive_kernel_fake(
    recv_buffer: Tensor,
    recv_indices: Tensor,
    recv_sizes: Tensor,
    hidden_states: Tensor,
    rank_id: int,
) -> Tensor:
    R_dim = recv_buffer.shape[0]
    t_dim = recv_buffer.shape[1]
    H_dim = recv_buffer.shape[2]
    return torch.empty(R_dim, t_dim, H_dim, dtype=recv_buffer.dtype)


# ---------------------------------------------------------------------------
# CCL All2All group ID
# ---------------------------------------------------------------------------
CCL_ALL2ALL_GROUP_ID = 42  ## may make problem in multiple layers ?
