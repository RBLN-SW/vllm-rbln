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
# table: score each group (top-2 biased experts with a correction bias, best
# expert without one), keep the top `topk_group` groups, then pick `top_k`
# within them. Pure torch, hand-verifiable inputs.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_rbln.model_executor.layers.fused_moe.utils as fused_moe_utils
from vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner import RBLNMoERunner
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

    def test_bias_branch_weights_are_softmax_probabilities(self):
        # With a bias the softmax runs over all experts *before* the bias is
        # applied (the reference gate biases probabilities, not logits), so an
        # unrenormalized weight is that expert's full-vocabulary probability.
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = grouped_topk(
            logits,
            top_k=2,
            num_expert_group=2,
            topk_group=2,
            scoring_func="softmax",
            renormalize=False,
            e_score_correction_bias=torch.zeros(4),
        )
        probs = torch.softmax(logits, dim=-1).flatten()
        assert out.flatten().tolist() == [
            0.0,
            0.0,
            probs[2].item(),  # expert 2
            probs[3].item(),  # expert 3
        ]
        assert out.sum().item() < 1.0  # no renormalization: mass stays partial

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


def _routed_logits_dtype(
    monkeypatch,
    *,
    logits_dtype,
    bias_dtype,
    scoring_func="sigmoid",
    num_expert_group=2,
    use_moe_tokens_mask=True,
):
    """Dtype of the weights ``forward`` hands the quant method, mask included.

    Building a real RBLNMoERunner needs a MoEConfig, a quant method and an
    initialized DP group, so the instance is assembled by hand -- but ``forward``
    itself is the real one, which is what the mask dtype has to come out of.
    """
    monkeypatch.setattr(
        fused_moe_utils,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor([3]),
                max_pads_across_dp=None,  # DP=1
            )
        ),
    )

    routed: dict[str, torch.dtype] = {}

    class _CaptureQuantMethod:
        def apply(self, layer, x, router_logits):
            routed["dtype"] = router_logits.dtype
            return x

    runner = object.__new__(RBLNMoERunner)
    torch.nn.Module.__init__(runner)
    runner.use_moe_tokens_mask = use_moe_tokens_mask
    runner.routed_experts = SimpleNamespace(quant_method=_CaptureQuantMethod())
    runner.top_k = 2
    runner.moe_parallel_config = SimpleNamespace(dp_size=1, dp_rank=0)
    runner.router = SimpleNamespace(
        scoring_func=scoring_func,
        renormalize=False,
        e_score_correction_bias=torch.zeros(4, dtype=bias_dtype),
        num_expert_group=num_expert_group,
        topk_group=num_expert_group,
    )

    logits = torch.randn(1, 3, 4, dtype=logits_dtype)
    RBLNMoERunner.forward(
        runner, torch.zeros(1, 3, 8, dtype=logits_dtype), lambda _: logits
    )
    return routed["dtype"]


class TestRoutingMaskDtype:
    # The mask multiplies the routing weights AFTER the compiler has folded the
    # routing chain into one fused routing op, whose output follows the wider of
    # (scores, e_score_correction_bias). The mask has to follow the same rule or
    # the fused multiply is a dtype mismatch the compiler rejects.
    def test_an_fp32_bias_widens_the_masked_weights(self, monkeypatch):
        # A bf16 mask here is the reported failure: the multiply stays bf16 while
        # the fused routing it feeds is fp32.
        assert (
            _routed_logits_dtype(
                monkeypatch, logits_dtype=torch.bfloat16, bias_dtype=torch.float32
            )
            is torch.float32
        )

    def test_a_bias_of_equal_width_keeps_the_narrow_weights(self, monkeypatch):
        assert (
            _routed_logits_dtype(
                monkeypatch, logits_dtype=torch.bfloat16, bias_dtype=torch.bfloat16
            )
            is torch.bfloat16
        )

    def test_a_bias_the_routing_never_adds_does_not_widen(self, monkeypatch):
        # Non-grouped softmax scores without the bias, so the fused routing stays
        # bf16 and widening the mask would promote the table for nothing.
        assert (
            _routed_logits_dtype(
                monkeypatch,
                logits_dtype=torch.bfloat16,
                bias_dtype=torch.float32,
                scoring_func="softmax",
                num_expert_group=None,
            )
            is torch.bfloat16
        )

    def test_the_mask_off_leaves_the_weights_narrow(self, monkeypatch):
        # No mask, no multiply to match: the fp32 bias never reaches the weights.
        assert (
            _routed_logits_dtype(
                monkeypatch,
                logits_dtype=torch.bfloat16,
                bias_dtype=torch.float32,
                use_moe_tokens_mask=False,
            )
            is torch.bfloat16
        )


class TestRegistration:
    def test_moe_runner_resolves_to_rbln_oot_implementation(self):
        # The suite conftest loads the general plugins before collection, so
        # RBLNMoERunner is already registered as the out-of-tree MoERunner.
        assert maybe_get_oot_by_class(MoERunner) is RBLNMoERunner
        # The factory path itself: PluggableLayer.__new__ allocates the RBLN class
        # when the base MoERunner is instantiated.
        assert type(MoERunner.__new__(MoERunner)) is RBLNMoERunner


class TestSqrtSoftplusRouting:
    def test_hash_ids_select_unbiased_weights(self):
        from vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner import (
            _apply_sqrtsoftplus_topk,
        )

        logits = torch.tensor([[1.0, -2.0, 3.0, 0.5], [4.0, 1.0, -1.0, 2.0]])
        ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.bfloat16)
        expected = torch.zeros_like(logits)
        scores = torch.nn.functional.softplus(logits).sqrt()
        chosen = scores.gather(1, ids.long())
        chosen /= chosen.sum(1, keepdim=True)
        expected.scatter_(1, ids.long(), chosen)
        actual = _apply_sqrtsoftplus_topk(logits, 2, True, expert_ids=ids)
        torch.testing.assert_close(actual, expected.transpose(0, 1))

    def test_bias_changes_selection_without_changing_weights(self):
        from vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner import (
            _apply_sqrtsoftplus_topk,
        )

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        bias = torch.tensor([100.0, 0.0, 0.0, 0.0])
        actual = _apply_sqrtsoftplus_topk(logits, 1, False, bias)
        expected = torch.zeros(4, 1)
        expected[0, 0] = torch.nn.functional.softplus(logits[0, 0]).sqrt()
        torch.testing.assert_close(actual, expected)


class TestHashRoutedForward:
    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("dispatch", [False, True])
    @pytest.mark.parametrize("hash_routing", [False, True])
    def test_dp_gathers_ids_with_hidden_states_and_preserves_rank_padding(
        self, monkeypatch, rank, dispatch, hash_routing
    ):
        import vllm_rbln.model_executor.layers.fused_moe.runner.moe_runner as mr

        counts = [1, 3]
        hidden = torch.arange(2 * 3 * 64).reshape(2, 3, 64).to(torch.bfloat16)
        hidden[0, 1:] = 0
        ids = torch.zeros(2, 3, 64, dtype=torch.bfloat16)
        ids[0, 0, :2] = torch.tensor([0, 3])
        ids[1, :, :2] = torch.tensor([[1, 2], [0, 2], [1, 3]])
        context = SimpleNamespace(
            dp_metadata=SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(counts),
                max_pads_across_dp=torch.empty(3),
            )
        )
        monkeypatch.setattr(mr, "get_forward_context", lambda: context)
        monkeypatch.setattr(fused_moe_utils, "get_forward_context", lambda: context)

        def router(x):
            return x[..., :4].float() / 128

        logits = router(hidden)
        if dispatch:
            payloads = [ids[..., :2]] if hash_routing else []
            payloads.append(logits.reshape(2, -1))
        else:
            payloads = [torch.cat((hidden, ids), -1) if hash_routing else hidden]
        sends: list[torch.Tensor] = []

        def all_gather(x, dim):
            assert len(sends) < len(payloads), "Unexpected extra all-gather"
            payload = payloads[len(sends)]
            torch.testing.assert_close(x, payload[rank : rank + 1])
            sends.append(x)
            return payload.clone()

        group = SimpleNamespace(
            all_gather=all_gather,
            reduce_scatter=lambda x, dim: x[rank * 3 : (rank + 1) * 3],
            unique_name="test-hash-routing",
        )
        monkeypatch.setattr(mr, "get_dp_group", lambda: group)
        if dispatch:
            ops = torch.ops.rbln_custom_ops
            monkeypatch.setattr(ops, "ccl_dispatch_send", lambda *args: (hidden, ids))
            monkeypatch.setattr(ops, "ccl_all2all_x_kernel", lambda *args: hidden)
            monkeypatch.setattr(ops, "ccl_dispatch_receive", lambda *args: hidden)

        apply = Mock(side_effect=lambda layer, x, router_logits: x)
        runner = object.__new__(RBLNMoERunner)
        torch.nn.Module.__init__(runner)
        runner.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(apply=apply)
        )
        runner.top_k = 2
        runner.moe_parallel_config = SimpleNamespace(dp_size=2, dp_rank=rank)
        runner.use_dispatch_all2all = dispatch
        runner.use_combine_all2all = False
        runner.use_moe_tokens_mask = True
        runner.send_mask = torch.eye(4)
        runner.router = SimpleNamespace(
            scoring_func="sqrtsoftplus",
            renormalize=True,
            e_score_correction_bias=None,
            num_expert_group=None,
            topk_group=None,
        )
        local_hidden = hidden[rank : rank + 1, : counts[rank]]
        result = runner(
            local_hidden,
            router,
            expert_ids=ids[rank, : counts[rank]] if hash_routing else None,
        )

        assert len(sends) == len(payloads)
        torch.testing.assert_close(result, local_hidden)
        torch.testing.assert_close(apply.call_args.kwargs["x"], hidden.reshape(6, 64))
        weights = apply.call_args.kwargs["router_logits"].t().reshape(2, 3, 4)
        assert weights[0, 1:].count_nonzero() == 0
        for r, count in enumerate(counts):
            for t in range(count):
                chosen = (
                    ids[r, t, :2].long()
                    if hash_routing
                    else logits[r, t].topk(2).indices
                )
                scores = torch.nn.functional.softplus(logits[r, t, chosen]).sqrt()
                expected = torch.zeros(4)
                expected[chosen] = scores / scores.sum()
                torch.testing.assert_close(weights[r, t], expected)

    def test_expert_ids_pick_the_experts_the_quant_method_sees(self, monkeypatch):
        monkeypatch.setattr(
            fused_moe_utils,
            "get_forward_context",
            lambda: SimpleNamespace(
                dp_metadata=SimpleNamespace(
                    num_tokens_across_dp_cpu=torch.tensor([3]),
                    max_pads_across_dp=None,  # DP=1
                )
            ),
        )
        routed: dict[str, torch.Tensor] = {}

        class _CaptureQuantMethod:
            def apply(self, layer, x, router_logits):
                routed["weights"] = router_logits
                return x

        runner = object.__new__(RBLNMoERunner)
        torch.nn.Module.__init__(runner)
        runner.use_moe_tokens_mask = False
        runner.routed_experts = SimpleNamespace(quant_method=_CaptureQuantMethod())
        runner.top_k = 2
        runner.moe_parallel_config = SimpleNamespace(dp_size=1, dp_rank=0)
        runner.router = SimpleNamespace(
            scoring_func="sqrtsoftplus",
            renormalize=True,
            e_score_correction_bias=None,
            num_expert_group=None,
            topk_group=None,
        )
        logits = torch.randn(1, 3, 4)
        ids = torch.tensor([[0, 3], [1, 2], [3, 1]], dtype=torch.bfloat16)

        RBLNMoERunner.forward(
            runner, torch.zeros(1, 3, 8), lambda _: logits, expert_ids=ids
        )

        weights = routed["weights"].transpose(0, 1)  # [t, E]
        chosen = sorted(map(tuple, weights.nonzero().tolist()))
        assert chosen == sorted((t, int(e)) for t in range(3) for e in ids[t])
        torch.testing.assert_close(weights.sum(1), torch.ones(3))
