# Copyright 2026 Rebellions Inc. All rights reserved.
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

# The MoE expert-parallel mask helpers: each of R ranks owns a contiguous block of
# E//R experts. Pure numpy.

from vllm_rbln.model_executor.layers.fused_moe.all2all import (
    generate_expert_mask,
    prepare_send_mask_matrix,
)


class TestGenerateExpertMask:
    def test_single_rank_owns_every_expert(self):
        # R=1: rank 0 owns all E experts, local index == global index.
        assert generate_expert_mask(1, 4).tolist() == [[0, 1, 2, 3]]

    def test_contiguous_blocks_with_local_indices(self):
        # Each rank owns E//R contiguous experts, labelled 0..(E//R-1); the rest
        # of the row is -1 (not owned by that rank).
        assert generate_expert_mask(2, 4).tolist() == [
            [0, 1, -1, -1],
            [-1, -1, 0, 1],
        ]

    def test_shape_is_ranks_by_experts(self):
        assert generate_expert_mask(4, 8).shape == (4, 8)

    def test_experts_beyond_even_split_are_unowned(self):
        # E=5 over R=2: E//R=2 experts each, so the trailing expert 4 is owned by
        # nobody (stays -1 in every row).
        mask = generate_expert_mask(2, 5)
        assert mask.tolist() == [
            [0, 1, -1, -1, -1],
            [-1, -1, 0, 1, -1],
        ]
        assert (mask[:, 4] == -1).all()


class TestPrepareSendMaskMatrix:
    def test_binary_ownership_indicator(self):
        # 1 where the expert belongs to the rank, else 0 -- the >=0 view of
        # generate_expert_mask.
        assert prepare_send_mask_matrix(2, 4).tolist() == [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]

    def test_each_owned_expert_belongs_to_exactly_one_rank(self):
        send = prepare_send_mask_matrix(4, 8)
        # Every expert is owned by exactly one rank (column sums are all 1).
        assert send.sum(axis=0).tolist() == [1] * 8

    def test_unowned_leftover_expert_column_is_all_zero(self):
        send = prepare_send_mask_matrix(2, 5)
        assert send[:, 4].tolist() == [0, 0]
