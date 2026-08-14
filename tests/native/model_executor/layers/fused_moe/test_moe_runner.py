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

# _apply_grouped_topk_torch turns router logits [T, E] into an [E, T] weight
# table: score groups by their top-2 experts, keep the top `topk_group` groups,
# then pick `top_k` within them. Pure torch, hand-verifiable inputs.

import torch
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner import (
    RBLNMoERunner,
)
from vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner import (
    _apply_grouped_topk_torch as grouped_topk,
)


class TestApplyGroupedTopkTorch:
    def test_selects_highest_scoring_group_then_expert(self):
        # groups [1,2] (score 3) and [10,3] (score 13) -> group 1 wins; within it
        # top_k=1 picks the 10-logit expert (id 2), renormalized to 1.0.
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 10.0, 3.0]]),
            top_k=1,
            num_expert_group=2,
            topk_group=1,
        )
        assert out.flatten().tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_topk_within_group_is_renormalized(self):
        # both experts of the winning group [10,3] selected -> weights 10/13, 3/13.
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 10.0, 3.0]]),
            top_k=2,
            num_expert_group=2,
            topk_group=1,
        )
        assert out.flatten().tolist() == [
            0.0,
            0.0,
            torch.tensor(10 / 13).item(),
            torch.tensor(3 / 13).item(),
        ]

    def test_output_shape_is_experts_by_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])  # [T=2,E=4]
        out = grouped_topk(logits, top_k=2, num_expert_group=2, topk_group=2)
        assert out.shape == (4, 2)  # [E, T]

    def test_exactly_top_k_experts_are_nonzero_per_token(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        out = grouped_topk(logits, top_k=2, num_expert_group=2, topk_group=2)
        # one column per token; each must have exactly top_k nonzero experts.
        assert (out != 0).sum(dim=0).tolist() == [2, 2]

    def test_experts_in_unselected_groups_stay_zero(self):
        # group 1 ([9,9]/[8,8]) dominates for both tokens -> group 0 experts (0,1)
        # are masked to zero across the batch.
        logits = torch.tensor([[1.0, 1.0, 9.0, 9.0], [2.0, 2.0, 8.0, 8.0]])
        out = grouped_topk(logits, top_k=1, num_expert_group=2, topk_group=1)
        assert out[0].tolist() == [0.0, 0.0]
        assert out[1].tolist() == [0.0, 0.0]

    def test_renormalize_makes_each_token_column_sum_to_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        out = grouped_topk(
            logits, top_k=2, num_expert_group=2, topk_group=2, renormalize=True
        )
        assert out.sum(dim=0).tolist() == [torch.tensor(1.0).item()] * 2

    def test_without_renormalize_weights_are_raw_logits(self):
        # no scoring, no renorm: the selected expert keeps its raw logit value.
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            top_k=1,
            num_expert_group=2,
            topk_group=2,
            renormalize=False,
        )
        assert out.flatten().tolist() == [0.0, 0.0, 0.0, 4.0]

    def test_sigmoid_scoring_uses_sigmoid_of_logits_as_weights(self):
        # sigmoid is applied before grouping, so the selected weight (no renorm)
        # is sigmoid(logit) of the winning expert.
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 10.0, 3.0]]),
            top_k=1,
            num_expert_group=2,
            topk_group=1,
            scoring_func="sigmoid",
            renormalize=False,
        )
        assert out.flatten().tolist() == [
            0.0,
            0.0,
            torch.sigmoid(torch.tensor(10.0)).item(),
            0.0,
        ]

    def test_bias_flips_selection_but_weight_comes_from_original_scores(self):
        # The bias steers which expert is picked, but the emitted weight comes
        # from the original scores.
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        no_bias = grouped_topk(
            logits, top_k=1, num_expert_group=2, topk_group=2, renormalize=False
        )
        biased = grouped_topk(
            logits,
            top_k=1,
            num_expert_group=2,
            topk_group=2,
            renormalize=False,
            e_score_correction_bias=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        )
        # Without bias, the 4.0-logit expert (id 3) wins with its raw weight.
        assert no_bias.flatten().tolist() == [0.0, 0.0, 0.0, 4.0]
        # The +10 bias flips selection to expert 0, but its weight is the
        # original 1.0 -- not 11.0 (which would betray using the biased score).
        assert biased.flatten().tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_sigmoid_with_bias_is_the_deepseek_path(self):
        # The production call (RBLNMoERunner.forward): sigmoid transforms the
        # weight, the bias only steers which expert is picked.
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        no_bias = grouped_topk(
            logits,
            top_k=1,
            num_expert_group=2,
            topk_group=2,
            scoring_func="sigmoid",
            renormalize=False,
        )
        biased = grouped_topk(
            logits,
            top_k=1,
            num_expert_group=2,
            topk_group=2,
            scoring_func="sigmoid",
            renormalize=False,
            e_score_correction_bias=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        )
        # Without bias, sigmoid ranking picks the largest logit (expert 3).
        assert no_bias.flatten().tolist() == [
            0.0,
            0.0,
            0.0,
            torch.sigmoid(torch.tensor(4.0)).item(),
        ]
        # The +10 bias flips selection to expert 0, but its weight is
        # sigmoid(1) -- not the raw logit, the biased score, or a softmax.
        assert biased.flatten().tolist() == [
            torch.sigmoid(torch.tensor(1.0)).item(),
            0.0,
            0.0,
            0.0,
        ]

    def test_bias_branch_renormalizes_selected_weights(self):
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            top_k=2,
            num_expert_group=2,
            topk_group=2,
            renormalize=True,
            e_score_correction_bias=torch.zeros(4),
        )
        assert out.sum(dim=0).tolist() == [torch.tensor(1.0).item()]

    def test_bias_branch_softmaxes_selected_original_scores(self):
        # In the bias branch, scoring_func="softmax" softmaxes the weights that
        # were gathered from the original scores (here experts 3 and 2 -> [4,3]).
        out = grouped_topk(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            top_k=2,
            num_expert_group=2,
            topk_group=2,
            scoring_func="softmax",
            renormalize=False,
            e_score_correction_bias=torch.zeros(4),
        )
        expected = torch.softmax(torch.tensor([4.0, 3.0]), dim=0)
        assert out.flatten().tolist() == [
            0.0,
            0.0,
            expected[1].item(),  # expert 2 <- score 3
            expected[0].item(),  # expert 3 <- score 4
        ]

    def test_softmax_post_norm_sums_to_one_while_pre_norm_leaks_mass(self):
        # renormalize=True softmaxes after topk (sums to 1); False softmaxes
        # first, so the selected subset carries less than the whole mass.
        logits = torch.tensor([[1.0, 1.0, 3.0, 1.0]])  # winning group [3,1]
        post = grouped_topk(
            logits,
            top_k=1,
            num_expert_group=2,
            topk_group=1,
            scoring_func="softmax",
            renormalize=True,
        )
        pre = grouped_topk(
            logits,
            top_k=1,
            num_expert_group=2,
            topk_group=1,
            scoring_func="softmax",
            renormalize=False,
        )
        assert post.sum().item() == torch.tensor(1.0).item()
        assert pre.sum().item() < 1.0


class TestRegistration:
    def test_moe_runner_resolves_to_rbln_oot_implementation(self):
        # The native conftest loads the general plugins before collection, so
        # RBLNMoERunner is already registered as the out-of-tree MoERunner.
        assert maybe_get_oot_by_class(MoERunner) is RBLNMoERunner
        # The factory path itself: PluggableLayer.__new__ allocates the RBLN class
        # when the base MoERunner is instantiated.
        assert type(MoERunner.__new__(MoERunner)) is RBLNMoERunner
