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

# decide_batch: the per-step decode shape decision, as a pure function of
# (config, this rank's counts, cross-DP snapshot). The golden values below are the
# contract each route has to satisfy; they pin the exact padded dimensions rather
# than a range, since a compiled graph is selected by them.

from unittest.mock import MagicMock

import pytest
import torch

import vllm_rbln.v1.worker.dp_utils as dp_utils
from vllm_rbln.v1.worker.dp_utils import (
    BatchRoute,
    DPBatchSnapshot,
    ShapeConfig,
    decide_batch,
)

# The two bucket layouts the measured configs use: a single compiled batch
# (max_num_seqs=8) and the DECODE_BUCKETING ladder.
SINGLE = (8,)
LADDER = (1, 2, 4, 8)
MAX_NUM_TOKENS = 512


def _cfg(buckets=SINGLE, *, specialized=True, max_num_tokens=MAX_NUM_TOKENS):
    def find_bucket(n: int) -> int:
        # Mirrors BucketingManager.find_decode_batch_bucket (smallest >= n).
        for b in buckets:
            if b >= n:
                return b
        raise ValueError(f"no bucket for {n} in {buckets}")

    return ShapeConfig(
        decode_batch_buckets=buckets,
        find_bucket=find_bucket,
        max_num_tokens=max_num_tokens,
        specialized_moe_decode=specialized,
    )


def _snapshot(*, num_tokens, num_reqs, is_prefill=None, is_idle=None):
    """Per-rank snapshot; is_prefill/is_idle default to all-zero of the same length."""
    zeros = [0] * len(num_tokens)
    return DPBatchSnapshot(
        num_tokens=tuple(num_tokens),
        num_reqs=tuple(num_reqs),
        is_prefill=tuple(
            bool(x) for x in (is_prefill if is_prefill is not None else zeros)
        ),
        is_idle=tuple(bool(x) for x in (is_idle if is_idle is not None else zeros)),
        num_tokens_across_dp=torch.tensor(num_tokens, dtype=torch.int32),
    )


def _decide(
    cfg,
    *,
    num_reqs,
    num_tokens,
    is_prefill=False,
    snapshot=None,
    pinned_num_tokens_padded=None,
):
    return decide_batch(
        cfg=cfg,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        is_prefill=is_prefill,
        snapshot=snapshot,
        pinned_num_tokens_padded=pinned_num_tokens_padded,
    )


class TestPinnedRoute:
    # Warm-up compiles the token dimensions only DP asymmetry produces; it dictates
    # one instead of receiving a decided one, and the batch still comes from the
    # local rule.
    def test_pin_is_the_token_dimension(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=8,
            num_tokens=8,
            snapshot=_snapshot(num_tokens=[8, 8], num_reqs=[8, 8]),
            pinned_num_tokens_padded=512,
        )
        assert route is BatchRoute.PINNED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            512,
        )

    def test_pin_wins_over_the_route_that_would_have_decided(self):
        # Without the pin these inputs agree on the small product; with it the
        # asymmetric dimension is what gets compiled.
        kwargs = dict(
            num_reqs=8,
            num_tokens=8,
            snapshot=_snapshot(num_tokens=[8, 8], num_reqs=[8, 8]),
        )
        assert _decide(_cfg(), **kwargs)[0].num_tokens_padded == 8
        assert (
            _decide(_cfg(), **kwargs, pinned_num_tokens_padded=32)[0].num_tokens_padded
            == 32
        )

    def test_spec_query_len_keeps_the_local_batch(self):
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=3,
            num_tokens=12,
            snapshot=_snapshot(num_tokens=[12, 12], num_reqs=[3, 3]),
            pinned_num_tokens_padded=32,
        )
        assert route is BatchRoute.PINNED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            4,
            32,
        )


class TestLocalRoute:
    # dp_size == 1: no cross-rank agreement, and the caller does not pad tokens.
    def test_decode_takes_the_local_bucket(self):
        desc, route = _decide(_cfg(), num_reqs=1, num_tokens=1)
        assert route is BatchRoute.LOCAL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            None,
        )

    def test_decode_of_two_reqs_still_buckets_to_eight(self):
        desc, route = _decide(_cfg(), num_reqs=2, num_tokens=2)
        assert (desc.num_reqs_padded, desc.query_len) == (8, 1)

    def test_prefill_passes_num_reqs_through(self):
        desc, route = _decide(_cfg(), num_reqs=1, num_tokens=12, is_prefill=True)
        assert route is BatchRoute.LOCAL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            12,
            None,
        )

    def test_ladder_buckets_to_the_smallest_fit(self):
        assert _decide(_cfg(LADDER), num_reqs=3, num_tokens=3)[0].num_reqs_padded == 4


class TestUnspecializedRoute:
    # DP but VLLM_RBLN_SPECIALIZE_MOE_DECODE=0: local bucket, tokens padded to the
    # prefill-sized target, no cross-rank shape agreement.
    def test_decode_pads_tokens_to_the_target(self):
        desc, route = _decide(
            _cfg(specialized=False),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(num_tokens=[1, 1, 1, 1], num_reqs=[1, 1, 1, 1]),
        )
        assert route is BatchRoute.UNSPECIALIZED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            512,
        )

    def test_two_reqs_keep_the_local_bucket(self):
        desc, route = _decide(
            _cfg(specialized=False),
            num_reqs=2,
            num_tokens=2,
            snapshot=_snapshot(num_tokens=[2, 1, 1, 1], num_reqs=[2, 1, 1, 1]),
        )
        assert (desc.num_reqs_padded, desc.num_tokens_padded) == (8, 512)

    def test_prefill_keeps_num_reqs(self):
        desc, route = _decide(
            _cfg(specialized=False),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            snapshot=_snapshot(
                num_tokens=[512, 1, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[1, 0, 0, 0],
            ),
        )
        assert route is BatchRoute.UNSPECIALIZED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            512,
            512,
        )


class TestAnyPrefillRoute:
    # Some rank is prefilling, so the decoding ranks run one padded-decode graph
    # via the top bucket and the token dimension is the prefill-sized target.
    def test_decoding_rank_takes_the_top_bucket(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        )
        assert route is BatchRoute.ANY_PREFILL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            512,
        )

    def test_top_bucket_is_not_this_ranks_own_bucket(self):
        # The one above cannot tell the rule apart from "my own bucket": with a
        # single-bucket ladder both answers are 8. On a ladder this rank's own
        # count would fit bucket 1, and taking it would put the decoding ranks on
        # a graph the prefill-sized token dimension cannot fill.
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        )
        assert route is BatchRoute.ANY_PREFILL
        assert desc.num_reqs_padded == 8  # buckets[-1], not find_bucket(1) == 1

    def test_idle_rank_joins_on_the_same_shape(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
                is_idle=[1, 0, 0, 0],
            ),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            512,
        )

    def test_spec_decode_rank_keeps_its_own_query_len(self):
        # 2 reqs x 4 tokens: the draft length stays visible in query_len even though
        # the token dimension is the padding target.
        desc, route = _decide(
            _cfg(),
            num_reqs=2,
            num_tokens=8,
            snapshot=_snapshot(
                num_tokens=[8, 512, 1, 1],
                num_reqs=[2, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            4,
            512,
        )

    def test_the_prefilling_rank_itself_keeps_num_reqs(self):
        # D1: a prefill step reports num_reqs_padded == num_reqs on every route.
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            snapshot=_snapshot(
                num_tokens=[512, 1, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[1, 0, 0, 0],
            ),
        )
        assert route is BatchRoute.ANY_PREFILL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            512,
            512,
        )


class TestAllIdleRoute:
    # Every rank drained but the engine has not synced yet (vllm only checks every
    # 32 steps), so all ranks run the cheapest identical graph.
    def test_single_bucket_layout(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1] * 4, num_reqs=[1] * 4, is_idle=[1, 1, 1, 1]
            ),
        )
        assert route is BatchRoute.ALL_IDLE
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            8,
        )

    def test_ladder_layout_takes_the_smallest_bucket(self):
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1] * 4, num_reqs=[1] * 4, is_idle=[1, 1, 1, 1]
            ),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            1,
            1,
        )


class TestQlenAsymRoute:
    # Busy ranks disagree on query_len (one drafted, another did not), so only the
    # top bucket's spec-asymmetric padding is warmed.
    def test_top_bucket_with_the_max_query_len_product(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 4, 1, 1], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
            ),
        )
        assert route is BatchRoute.QLEN_ASYM
        # 8 (top bucket) * 4 (max query_len over the busy ranks) == 32
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            32,
        )

    def test_the_drafting_rank_reports_its_own_query_len(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=4,
            snapshot=_snapshot(
                num_tokens=[4, 1, 1, 1], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
            ),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            4,
            32,
        )

    def test_a_short_rank_pads_up_to_the_graph(self):
        # This is the one route where the token dimension is deliberately larger
        # than the rank's own num_reqs_padded x query_len: the graph is warmed for
        # the longest query length and shorter ranks pad into it.
        snapshot = _snapshot(
            num_tokens=[4, 1, 1, 1], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
        )
        short, _ = _decide(_cfg(), num_reqs=1, num_tokens=1, snapshot=snapshot)
        assert short.num_tokens_padded == 32 > short.num_reqs_padded * short.query_len


class TestAgreedRoute:
    # The normal path: busy ranks share a query_len, so the shape is the bucket that
    # fits the busiest rank and the token dimension is that bucket x query_len.
    def test_bucket_from_the_busiest_rank(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 1, 1, 1], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
            ),
        )
        assert route is BatchRoute.AGREED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            8,
        )

    def test_bucket_comes_from_the_busiest_rank_not_this_one(self):
        # Every rank in the case above holds the same request count, so it cannot
        # tell the busiest rank from this one. Here the peer holds 3 and this rank
        # 1: taking this rank's bucket (1) would leave the peer's requests without
        # a slot in the graph every rank has to run.
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=4,
            snapshot=_snapshot(
                num_tokens=[4, 12, 4, 4],
                num_reqs=[1, 3, 1, 1],
            ),
        )
        assert route is BatchRoute.AGREED
        # find_bucket(max(1, 3, 1, 1)) == 4, and the token dimension follows it.
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            4,
            16,
        )

    def test_spec_step_multiplies_the_token_dimension(self):
        desc, route = _decide(
            _cfg(),
            num_reqs=8,
            num_tokens=32,
            snapshot=_snapshot(num_tokens=[32] * 4, num_reqs=[8] * 4),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            4,
            32,
        )

    def test_ladder_layout_uses_a_small_bucket(self):
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=4,
            snapshot=_snapshot(
                num_tokens=[4, 4, 1, 4], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
            ),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            4,
            4,
        )

    def test_idle_rank_adopts_the_busy_query_len(self):
        # The point of the route: an idle rank contributes (1 req, 1 token) but must
        # stage the busy ranks' query_len so it runs the same compiled graph. This is
        # what the old eff_qlen=None ("derive it from the product") encoded.
        desc, route = _decide(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 4, 4, 4],
                num_reqs=[1, 1, 1, 1],
                is_idle=[1, 0, 0, 0],
            ),
        )
        assert route is BatchRoute.AGREED
        assert desc.query_len == 4
        assert (desc.num_reqs_padded, desc.num_tokens_padded) == (1, 4)


class TestPrefillIsNormalized:
    # D1: num_reqs_padded == num_reqs on a prefill step, whatever the route. Today
    # ANY_PREFILL reports the top bucket instead and the caller overrides it back;
    # the decision owns it now, so the caller-side override can go.
    @pytest.mark.parametrize("specialized", [True, False])
    @pytest.mark.parametrize("buckets", [SINGLE, LADDER])
    def test_prefill_reports_num_reqs(self, specialized, buckets):
        desc, route = _decide(
            _cfg(buckets, specialized=specialized),
            num_reqs=1,
            num_tokens=69,
            is_prefill=True,
            snapshot=_snapshot(
                num_tokens=[69, 1, 1, 1], num_reqs=[1, 1, 1, 1], is_prefill=[1, 0, 0, 0]
            ),
        )
        assert desc.num_reqs_padded == 1
        assert desc.query_len == 69

    def test_single_dp_prefill_reports_num_reqs(self):
        desc, route = _decide(_cfg(), num_reqs=1, num_tokens=69, is_prefill=True)
        assert desc.num_reqs_padded == 1


# (route, kwargs) reaching every route, for the route-independent contract.
INVARIANT_CASES = [
    ("local-decode", dict(num_reqs=1, num_tokens=1)),
    ("local-prefill", dict(num_reqs=1, num_tokens=12, is_prefill=True)),
    (
        "unspecialized",
        dict(
            num_reqs=2,
            num_tokens=2,
            snapshot=_snapshot(num_tokens=[2, 1, 1, 1], num_reqs=[2, 1, 1, 1]),
        ),
    ),
    (
        "any-prefill",
        dict(
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        ),
    ),
    (
        "all-idle",
        dict(
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(num_tokens=[1] * 4, num_reqs=[1] * 4, is_idle=[1] * 4),
        ),
    ),
    (
        "qlen-asym",
        dict(
            num_reqs=1,
            num_tokens=4,
            snapshot=_snapshot(num_tokens=[4, 1, 1, 1], num_reqs=[1, 1, 1, 1]),
        ),
    ),
    (
        "agreed",
        dict(
            num_reqs=8,
            num_tokens=32,
            snapshot=_snapshot(num_tokens=[32] * 4, num_reqs=[8] * 4),
        ),
    ),
    (
        "agreed-idle",
        dict(
            num_reqs=1,
            num_tokens=1,
            snapshot=_snapshot(
                num_tokens=[1, 4, 4, 4], num_reqs=[1, 1, 1, 1], is_idle=[1, 0, 0, 0]
            ),
        ),
    ),
]


class TestInvariants:
    # These hold on every route, so a route added later cannot quietly break the
    # contract the callers and the compiled graphs rely on.
    @pytest.mark.parametrize("buckets", [SINGLE, LADDER])
    @pytest.mark.parametrize(
        "name,kwargs", INVARIANT_CASES, ids=[c[0] for c in INVARIANT_CASES]
    )
    def test_shape_contract(self, name, kwargs, buckets):
        specialized = name != "unspecialized"
        desc, route = _decide(_cfg(buckets, specialized=specialized), **kwargs)

        assert desc.query_len >= 1
        assert desc.num_reqs_padded >= 1
        # The staged batch fits the compiled batch dimension.
        assert desc.num_reqs_padded >= kwargs["num_reqs"]
        # What this rank stages fits in the graph's token dimension. Which exact
        # dimension each route picks is pinned by the golden cases above, so this
        # stays the one property that has to hold everywhere.
        if desc.num_tokens_padded is not None:
            assert desc.num_tokens_padded >= kwargs["num_reqs"] * desc.query_len
        # Decode picks a compiled bucket; prefill passes num_reqs through (D1).
        if kwargs.get("is_prefill"):
            assert desc.num_reqs_padded == kwargs["num_reqs"]
        else:
            assert desc.num_reqs_padded in buckets

    def test_input_query_len_must_be_uniform(self):
        # num_tokens == num_reqs * query_len is the caller's contract (RBLN compiles
        # one query length per step), so a non-divisible pair is a bug, not a shape.
        with pytest.raises((AssertionError, ValueError)):
            _decide(
                _cfg(),
                num_reqs=3,
                num_tokens=7,
                snapshot=_snapshot(num_tokens=[7, 1, 1, 1], num_reqs=[3, 1, 1, 1]),
            )

    @pytest.mark.parametrize("buckets", [SINGLE, LADDER])
    def test_every_rank_of_a_step_agrees_on_the_shape(self, buckets):
        # Same snapshot seen from each rank's point of view must yield one shape, or the
        # ranks would enter different graphs and the collective would hang.
        snapshot = _snapshot(
            num_tokens=[4, 4, 1, 4], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
        )
        per_rank = [
            _decide(
                _cfg(buckets),
                num_reqs=int(snapshot.num_reqs[i]),
                num_tokens=int(snapshot.num_tokens[i]),
                snapshot=snapshot,
            )
            for i in range(4)
        ]
        batches = {
            (d.num_reqs_padded, d.query_len, d.num_tokens_padded) for d, _ in per_rank
        }
        assert len(batches) == 1, f"ranks disagreed: {batches}"
        assert len({r for _, r in per_rank}) == 1


def _encode(
    num_tokens: int, num_reqs: int, is_prefill: bool, is_idle: bool = False
) -> int:
    # Mirrors the source bit layout: tokens in bits 0..15, reqs in 16..28,
    # prefill flag at bit 29, idle flag at bit 30.
    encoded = num_tokens | (num_reqs << 16)
    if is_prefill:
        encoded |= 1 << 29
    if is_idle:
        encoded |= 1 << 30
    return encoded


@pytest.fixture
def fake_dp_collective(monkeypatch):
    """Returns a setter taking ``{rank: value}`` to add into the reduce result,
    as if each remote rank had contributed ``value`` in its own slot."""
    import vllm.distributed.parallel_state as ps

    dp_group = MagicMock()
    dp_group.cpu_group = object()
    monkeypatch.setattr(ps, "get_dp_group", lambda: dp_group)

    others: dict[int, int] = {}

    def fake_all_reduce(tensor, group=None):
        assert group is dp_group.cpu_group
        for rank, value in others.items():
            tensor[rank] += value

    monkeypatch.setattr(dp_utils.dist, "all_reduce", fake_all_reduce)

    def set_others(new_others: dict[int, int]) -> None:
        others.clear()
        others.update(new_others)

    return set_others


class TestSynchronizeDraftDpRanks:
    # The draft's entry point: same collective, but it hands back only the four
    # aggregates the draft's own rule reads, so the snapshot stays module-local.
    def test_returns_the_cross_rank_aggregates(self, fake_dp_collective):
        fake_dp_collective(
            {
                1: _encode(10, 5, False),
                2: _encode(6, 3, False),
                3: _encode(14, 7, False),
            }
        )
        across, any_prefill, max_num_reqs, max_query_len = (
            dp_utils.synchronize_draft_dp_ranks(
                num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
            )
        )
        assert across.cpu().tolist() == [8, 10, 6, 14]
        assert any_prefill is False
        assert max_num_reqs == 8  # max(8, 5, 3, 7)
        assert max_query_len == 2  # max(8//8, 10//5, 6//3, 14//7)

    def test_a_remote_prefill_is_reported(self, fake_dp_collective):
        fake_dp_collective({1: _encode(30, 1, True)})
        _across, any_prefill, _reqs, _qlen = dp_utils.synchronize_draft_dp_ranks(
            num_tokens=1, num_reqs=1, dp_size=2, dp_rank=0, is_prefill=False
        )
        assert any_prefill is True

    def test_every_rank_counts_including_an_idle_one(self, fake_dp_collective):
        # Unlike decide_batch the draft has no idle exclusion: a rank reporting a
        # single request still raises the query length the group compiles for.
        fake_dp_collective({1: _encode(4, 1, False)})
        _across, _prefill, max_num_reqs, max_query_len = (
            dp_utils.synchronize_draft_dp_ranks(
                num_tokens=2, num_reqs=2, dp_size=2, dp_rank=0, is_prefill=False
            )
        )
        assert (max_num_reqs, max_query_len) == (2, 4)

    def test_non_uniform_query_length_asserts(self, fake_dp_collective):
        fake_dp_collective({1: _encode(5, 2, False)})
        with pytest.raises(AssertionError, match="non-uniform query length"):
            dp_utils.synchronize_draft_dp_ranks(
                num_tokens=2, num_reqs=2, dp_size=2, dp_rank=0, is_prefill=False
            )


class TestDPBatchSnapshot:
    def test_mixed_across_ranks(self, fake_dp_collective):
        # Distinct per-rank counts must be surfaced separately (a consumer takes
        # the max), and tokens != reqs proves the two fields don't bleed.
        fake_dp_collective(
            {
                1: _encode(10, 5, False),
                2: _encode(6, 3, False),
                3: _encode(14, 7, False),
            }
        )
        snapshot = dp_utils._synchronize_dp_ranks(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert snapshot.num_tokens == (8, 10, 6, 14)
        assert snapshot.num_reqs == (8, 5, 3, 7)
        assert snapshot.num_tokens_across_dp.cpu().tolist() == [8, 10, 6, 14]

    def test_a_remote_prefill_is_reported_per_rank(self, fake_dp_collective):
        # The prefill flag is data: which rank prefills is visible, and the reqs
        # of every rank stay available for the decision to use or ignore.
        fake_dp_collective(
            {
                1: _encode(300, 1, True),  # prefill rank
                2: _encode(4, 4, False),
                3: _encode(6, 6, False),
            }
        )
        snapshot = dp_utils._synchronize_dp_ranks(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert snapshot.is_prefill == (False, True, False, False)
        assert snapshot.num_tokens == (8, 300, 4, 6)
        assert snapshot.num_reqs == (8, 1, 4, 6)

    def test_the_local_prefill_flag_round_trips(self, fake_dp_collective):
        fake_dp_collective({r: _encode(8, 8, False) for r in (1, 2, 3)})
        snapshot = dp_utils._synchronize_dp_ranks(
            num_tokens=512, num_reqs=1, dp_size=4, dp_rank=0, is_prefill=True
        )
        assert snapshot.is_prefill == (True, False, False, False)
        assert snapshot.num_tokens == (512, 8, 8, 8)

    def test_boundary_max_values_round_trip(self, fake_dp_collective):
        # The largest values each field can hold must survive pack/unpack.
        fake_dp_collective({r: _encode(0xFFFF, 0x1FFF, False) for r in (1, 2, 3)})
        snapshot = dp_utils._synchronize_dp_ranks(
            num_tokens=0xFFFF, num_reqs=0x1FFF, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert snapshot.num_tokens == (0xFFFF,) * 4
        assert snapshot.num_reqs == (0x1FFF,) * 4

    def test_num_tokens_overflow_asserts(self):
        # The assert fires before any collective, so no fake is needed.
        with pytest.raises(AssertionError, match="num_tokens=65536"):
            dp_utils._synchronize_dp_ranks(
                num_tokens=1 << 16, num_reqs=1, dp_size=4, dp_rank=0, is_prefill=False
            )

    def test_num_reqs_overflow_asserts(self):
        with pytest.raises(AssertionError, match="num_reqs=8192"):
            dp_utils._synchronize_dp_ranks(
                num_tokens=1, num_reqs=1 << 13, dp_size=4, dp_rank=0, is_prefill=False
            )


class TestCoordinateBatchAcrossDp:
    # The composition: it joins the collective and hands the snapshot to
    # decide_batch. Both halves are tested above; what is only testable here is
    # what it forwards -- a rank's phase, its idle bit and warm-up's pin.

    @staticmethod
    def _peer(num_tokens, num_reqs, *, prefill=False, idle=False):
        token_bits, req_bits = 16, 13
        value = num_tokens | (num_reqs << token_bits)
        if prefill:
            value |= 1 << (token_bits + req_bits)
        if idle:
            value |= 1 << (token_bits + req_bits + 1)
        return value

    def _coordinate(self, monkeypatch, *, peer, **kwargs):
        def fake_run_ar(encoded, dp_size, dp_rank):
            arr = [0] * dp_size
            arr[dp_rank] = encoded
            arr[1 - dp_rank] = peer
            return torch.tensor(arr, dtype=torch.int32)

        monkeypatch.setattr(dp_utils, "_run_ar", fake_run_ar)
        return dp_utils.coordinate_batch_across_dp(
            cfg=_cfg(LADDER), dp_size=2, dp_rank=0, **kwargs
        )

    def test_a_prefilling_rank_is_not_reported_as_idle(self, monkeypatch):
        # is_prefill and is_idle occupy separate bits and mean opposite things: a
        # rank that prefills is the busiest rank there is. Feeding the phase in as
        # the idle bit would empty the busy set and take the all-idle route, which
        # decides the cheapest graph and discards the output.
        desc, route, across = self._coordinate(
            monkeypatch,
            peer=self._peer(1, 1),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            is_idle=False,
        )
        assert route is BatchRoute.ANY_PREFILL
        assert desc.query_len == 512
        # The tensor the caller publishes is the per-rank token count, not a
        # broadcast of this rank's own.
        assert across.tolist() == [512, 1]

    def test_an_idle_rank_adopts_the_busy_shape(self, monkeypatch):
        # The idle bit reaching the decision is what keeps this rank's minimal
        # (1 req, 1 token) from shrinking the graph the busy peer runs.
        desc, route, _across = self._coordinate(
            monkeypatch,
            peer=self._peer(12, 3),
            num_reqs=1,
            num_tokens=1,
            is_prefill=False,
            is_idle=True,
        )
        assert route is BatchRoute.AGREED
        assert (desc.num_reqs_padded, desc.query_len) == (4, 4)

    def test_warmups_pin_reaches_the_decision(self, monkeypatch):
        # Warm-up asks for a token dimension the symmetric ranks would not choose;
        # dropping it on the way through would compile a different graph.
        desc, route, _across = self._coordinate(
            monkeypatch,
            peer=self._peer(4, 1),
            num_reqs=1,
            num_tokens=4,
            is_prefill=False,
            is_idle=False,
            pinned_num_tokens_padded=MAX_NUM_TOKENS,
        )
        assert route is BatchRoute.PINNED
        assert desc.num_tokens_padded == MAX_NUM_TOKENS


class TestRunAr:
    def test_single_rank_identity(self, fake_dp_collective):
        # world_size 1: the reduce is a no-op, so only the self slot is present.
        fake_dp_collective({})
        out = dp_utils._run_ar(7, dp_size=1, dp_rank=0)
        assert out.dtype == torch.int32
        assert out.cpu().tolist() == [7]

    def test_places_self_at_own_rank_slot(self, fake_dp_collective):
        # A non-zero dp_rank proves the source writes at index dp_rank, not slot
        # 0; a slot-0 test could not tell placement from a constant.
        fake_dp_collective({0: 11, 1: 22, 3: 33})
        out = dp_utils._run_ar(7, dp_size=4, dp_rank=2)
        assert out.cpu().tolist() == [11, 22, 7, 33]


def test_synchronize_dp_ranks_bit_pack(monkeypatch):
    # Round-trip the bit-packed (num_tokens, num_reqs, is_prefill, is_idle)
    # all-reduce. Intercepts the inner all-gather so no real DP group is needed,
    # and checks every field unpacks per rank -- the transport reports what each
    # rank said and decides nothing.
    token_bits = 16

    def peer_encoded(num_tokens, num_reqs):
        return num_tokens | (num_reqs << token_bits)

    def make_fake_inner(peer_value):
        def fake_inner(encoded, dp_size, dp_rank):
            arr = [0, 0]
            arr[dp_rank] = encoded
            arr[1 - dp_rank] = peer_value
            return torch.tensor(arr, dtype=torch.int32)

        return fake_inner

    # Decode: this rank (rank0) = idle (num_tokens=1, num_reqs=1, is_idle);
    # peer (rank1) = busy (num_tokens=8, num_reqs=2).
    monkeypatch.setattr(dp_utils, "_run_ar", make_fake_inner(peer_encoded(8, 2)))
    snapshot = dp_utils._synchronize_dp_ranks(
        num_tokens=1,
        num_reqs=1,
        dp_size=2,
        dp_rank=0,
        is_prefill=False,
        is_idle=True,
    )
    assert snapshot.num_tokens == (1, 8)
    assert snapshot.num_reqs == (1, 2)
    assert snapshot.is_idle == (True, False)
    assert snapshot.is_prefill == (False, False)
    assert snapshot.num_tokens_across_dp.tolist() == [1, 8]

    # Prefill on this rank: reported as a flag, and the reqs stay available.
    monkeypatch.setattr(dp_utils, "_run_ar", make_fake_inner(peer_encoded(8, 2)))
    snapshot = dp_utils._synchronize_dp_ranks(
        num_tokens=5,
        num_reqs=1,
        dp_size=2,
        dp_rank=0,
        is_prefill=True,
        is_idle=False,
    )
    assert snapshot.num_tokens == (5, 8)
    assert snapshot.num_reqs == (1, 2)
    assert snapshot.is_prefill == (True, False)
    assert snapshot.is_idle == (False, False)
