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

# determine_batch_execution_and_padding: the per-step decode shape decision, as a
# pure function of (config, this rank's counts, cross-DP status). The golden values
# below are the contract each route has to satisfy; they pin the exact padded
# dimensions rather than a range, since a compiled graph is selected by them.

from unittest.mock import MagicMock

import pytest
import torch

import vllm_rbln.v1.worker.dp_utils as dp_utils
from vllm_rbln.v1.worker.dp_utils import (
    BatchDescriptor,
    BatchRoute,
    DPStatus,
    ShapeConfig,
    determine_batch_execution_and_padding,
    determine_draft_batch_execution_and_padding,
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


def _status(*, num_tokens, num_reqs, is_prefill=None, is_idle=None):
    """Per-rank status; is_prefill/is_idle default to all-zero of the same length."""
    zeros = [0] * len(num_tokens)
    return DPStatus(
        num_tokens=tuple(num_tokens),
        num_reqs=tuple(num_reqs),
        is_prefill=tuple(
            bool(x) for x in (is_prefill if is_prefill is not None else zeros)
        ),
        is_idle=tuple(bool(x) for x in (is_idle if is_idle is not None else zeros)),
        num_tokens_across_dp=torch.tensor(num_tokens, dtype=torch.int32),
    )


def _determine(
    cfg,
    *,
    num_reqs,
    num_tokens,
    is_prefill=False,
    status=None,
    pinned_num_tokens_padded=None,
):
    return determine_batch_execution_and_padding(
        cfg=cfg,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        is_prefill=is_prefill,
        status=status,
        pinned_num_tokens_padded=pinned_num_tokens_padded,
    )


class TestPinnedRoute:
    # Warm-up compiles the token dimensions only DP asymmetry produces; it dictates
    # one instead of receiving a decided one, and the batch still comes from the
    # local rule.
    def test_pin_is_the_token_dimension(self):
        desc, route = _determine(
            _cfg(),
            num_reqs=8,
            num_tokens=8,
            status=_status(num_tokens=[8, 8], num_reqs=[8, 8]),
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
            status=_status(num_tokens=[8, 8], num_reqs=[8, 8]),
        )
        assert _determine(_cfg(), **kwargs)[0].num_tokens_padded == 8
        assert (
            _determine(_cfg(), **kwargs, pinned_num_tokens_padded=32)[
                0
            ].num_tokens_padded
            == 32
        )

    def test_spec_query_len_keeps_the_local_batch(self):
        desc, route = _determine(
            _cfg(LADDER),
            num_reqs=3,
            num_tokens=12,
            status=_status(num_tokens=[12, 12], num_reqs=[3, 3]),
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
        desc, route = _determine(_cfg(), num_reqs=1, num_tokens=1)
        assert route is BatchRoute.LOCAL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            None,
        )

    def test_decode_of_two_reqs_still_buckets_to_eight(self):
        desc, route = _determine(_cfg(), num_reqs=2, num_tokens=2)
        assert (desc.num_reqs_padded, desc.query_len) == (8, 1)

    def test_prefill_passes_num_reqs_through(self):
        desc, route = _determine(_cfg(), num_reqs=1, num_tokens=12, is_prefill=True)
        assert route is BatchRoute.LOCAL
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            12,
            None,
        )

    def test_ladder_buckets_to_the_smallest_fit(self):
        assert (
            _determine(_cfg(LADDER), num_reqs=3, num_tokens=3)[0].num_reqs_padded == 4
        )


class TestUnspecializedRoute:
    # DP but VLLM_RBLN_SPECIALIZE_MOE_DECODE=0: local bucket, tokens padded to the
    # prefill-sized target, no cross-rank shape agreement.
    def test_decode_pads_tokens_to_the_target(self):
        desc, route = _determine(
            _cfg(specialized=False),
            num_reqs=1,
            num_tokens=1,
            status=_status(num_tokens=[1, 1, 1, 1], num_reqs=[1, 1, 1, 1]),
        )
        assert route is BatchRoute.UNSPECIALIZED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            1,
            512,
        )

    def test_two_reqs_keep_the_local_bucket(self):
        desc, route = _determine(
            _cfg(specialized=False),
            num_reqs=2,
            num_tokens=2,
            status=_status(num_tokens=[2, 1, 1, 1], num_reqs=[2, 1, 1, 1]),
        )
        assert (desc.num_reqs_padded, desc.num_tokens_padded) == (8, 512)

    def test_a_prefilling_peer_leaves_this_rank_its_own_bucket(self):
        # Specialization only decides a decode token dimension, and this route
        # already pads to the prefill-sized one -- so every bucket is compiled for
        # it and a prefilling peer changes nothing. Deciding the phase first would
        # move this rank onto the top bucket's graph and pad seven rows it does not
        # have.
        desc, route = _determine(
            _cfg(LADDER, specialized=False),
            num_reqs=1,
            num_tokens=1,
            status=_status(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        )
        assert route is BatchRoute.UNSPECIALIZED
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            1,
            MAX_NUM_TOKENS,
        )

    def test_prefill_keeps_num_reqs(self):
        desc, route = _determine(
            _cfg(specialized=False),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            status=_status(
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
        desc, route = _determine(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            status=_status(
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
        desc, route = _determine(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=1,
            status=_status(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        )
        assert route is BatchRoute.ANY_PREFILL
        assert desc.num_reqs_padded == 8  # buckets[-1], not find_bucket(1) == 1

    def test_spec_decode_rank_keeps_its_own_query_len(self):
        # 2 reqs x 4 tokens: the draft length stays visible in query_len even though
        # the token dimension is the padding target.
        desc, route = _determine(
            _cfg(),
            num_reqs=2,
            num_tokens=8,
            status=_status(
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
        desc, route = _determine(
            _cfg(),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            status=_status(
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


class TestDrainedGroup:
    # Every rank drained but the engine has not synced yet (vllm only checks every
    # 32 steps). There is no shape for a step none of them runs, so the rule says
    # so instead of naming one -- in any configuration, since a drained group is
    # answered before the split that decides token dimensions.
    @pytest.mark.parametrize("specialized", [True, False])
    def test_answers_with_no_shape(self, specialized):
        desc, route = _determine(
            _cfg(LADDER, specialized=specialized),
            num_reqs=1,
            num_tokens=1,
            status=_status(num_tokens=[1] * 4, num_reqs=[1] * 4, is_idle=[1, 1, 1, 1]),
        )
        assert (desc, route) == (None, BatchRoute.ALL_IDLE)


class TestQlenAsymRoute:
    # Busy ranks disagree on query_len (one drafted, another did not), so only the
    # top bucket's spec-asymmetric padding is warmed.
    def test_top_bucket_with_the_max_query_len_product(self):
        desc, route = _determine(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            status=_status(
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
        desc, route = _determine(
            _cfg(),
            num_reqs=1,
            num_tokens=4,
            status=_status(
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
        status = _status(
            num_tokens=[4, 1, 1, 1], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
        )
        short, _ = _determine(_cfg(), num_reqs=1, num_tokens=1, status=status)
        assert short.num_tokens_padded == 32 > short.num_reqs_padded * short.query_len


class TestAgreedRoute:
    # The normal path: busy ranks share a query_len, so the shape is the bucket that
    # fits the busiest rank and the token dimension is that bucket x query_len.
    def test_bucket_from_the_busiest_rank(self):
        desc, route = _determine(
            _cfg(),
            num_reqs=1,
            num_tokens=1,
            status=_status(
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
        desc, route = _determine(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=4,
            status=_status(
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
        desc, route = _determine(
            _cfg(),
            num_reqs=8,
            num_tokens=32,
            status=_status(num_tokens=[32] * 4, num_reqs=[8] * 4),
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            8,
            4,
            32,
        )

    def test_ladder_layout_uses_a_small_bucket(self):
        desc, route = _determine(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=4,
            status=_status(
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
        desc, route = _determine(
            _cfg(LADDER),
            num_reqs=1,
            num_tokens=1,
            status=_status(
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
        # Three requests: on a ladder the bucket rule would round up to 4, so a
        # prefill that came back padded is visible here as well as on SINGLE.
        desc, route = _determine(
            _cfg(buckets, specialized=specialized),
            num_reqs=3,
            num_tokens=69,
            is_prefill=True,
            status=_status(
                num_tokens=[69, 1, 1, 1], num_reqs=[3, 1, 1, 1], is_prefill=[1, 0, 0, 0]
            ),
        )
        assert desc.num_reqs_padded == 3
        assert desc.query_len == 23

    def test_single_dp_prefill_reports_num_reqs(self):
        desc, route = _determine(_cfg(), num_reqs=1, num_tokens=69, is_prefill=True)
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
            status=_status(num_tokens=[2, 1, 1, 1], num_reqs=[2, 1, 1, 1]),
        ),
    ),
    (
        "any-prefill",
        dict(
            num_reqs=1,
            num_tokens=1,
            status=_status(
                num_tokens=[1, 512, 1, 1],
                num_reqs=[1, 1, 1, 1],
                is_prefill=[0, 1, 0, 0],
            ),
        ),
    ),
    (
        "qlen-asym",
        dict(
            num_reqs=1,
            num_tokens=4,
            status=_status(num_tokens=[4, 1, 1, 1], num_reqs=[1, 1, 1, 1]),
        ),
    ),
    (
        "agreed",
        dict(
            num_reqs=8,
            num_tokens=32,
            status=_status(num_tokens=[32] * 4, num_reqs=[8] * 4),
        ),
    ),
    (
        "pinned",
        dict(
            num_reqs=2,
            num_tokens=2,
            status=_status(num_tokens=[2, 2], num_reqs=[2, 2]),
            pinned_num_tokens_padded=MAX_NUM_TOKENS,
        ),
    ),
    (
        "agreed-idle",
        dict(
            num_reqs=1,
            num_tokens=1,
            status=_status(
                num_tokens=[1, 4, 4, 4], num_reqs=[1, 1, 1, 1], is_idle=[1, 0, 0, 0]
            ),
        ),
    ),
]


class TestBatchDescriptor:
    def test_a_batch_that_does_not_fit_its_tokens_is_refused(self):
        # The dispatch pads the staged rows up to the token dimension and only up
        # to it, so a descriptor that does not fit would drop rows at run time
        # rather than fail here. Two requests three tokens deep need six.
        with pytest.raises(AssertionError, match="does not fit"):
            BatchDescriptor(num_reqs_padded=2, query_len=3, num_tokens_padded=5)
        BatchDescriptor(num_reqs_padded=2, query_len=3, num_tokens_padded=6)
        # Single-DP states no dimension, so there is nothing to fit inside.
        BatchDescriptor(num_reqs_padded=8, query_len=3, num_tokens_padded=None)


class TestInvariants:
    # These hold on every route, so a route added later cannot quietly break the
    # contract the callers and the compiled graphs rely on.
    @pytest.mark.parametrize("buckets", [SINGLE, LADDER])
    @pytest.mark.parametrize(
        "name,kwargs", INVARIANT_CASES, ids=[c[0] for c in INVARIANT_CASES]
    )
    def test_shape_contract(self, name, kwargs, buckets):
        specialized = name != "unspecialized"
        desc, route = _determine(_cfg(buckets, specialized=specialized), **kwargs)

        assert desc.query_len >= 1
        assert desc.num_reqs_padded >= 1
        # The staged batch fits the compiled batch dimension.
        assert desc.num_reqs_padded >= kwargs["num_reqs"]
        # The token dimension holding the staged rows is BatchDescriptor's own
        # contract, checked at construction, so every case here has already passed
        # it by getting this far.
        # Decode picks a compiled bucket; prefill passes num_reqs through (D1).
        if kwargs.get("is_prefill"):
            assert desc.num_reqs_padded == kwargs["num_reqs"]
        else:
            assert desc.num_reqs_padded in buckets

    def test_input_query_len_must_be_uniform(self):
        # num_tokens == num_reqs * query_len is the caller's contract (RBLN compiles
        # one query length per step), so a non-divisible pair is a bug, not a shape.
        with pytest.raises((AssertionError, ValueError)):
            _determine(
                _cfg(),
                num_reqs=3,
                num_tokens=7,
                status=_status(num_tokens=[7, 1, 1, 1], num_reqs=[3, 1, 1, 1]),
            )

    @pytest.mark.parametrize("buckets", [SINGLE, LADDER])
    def test_every_rank_of_a_step_agrees_on_the_shape(self, buckets):
        # Same status seen from each rank's point of view must yield one shape, or the
        # ranks would enter different graphs and the collective would hang.
        status = _status(
            num_tokens=[4, 4, 1, 4], num_reqs=[1, 1, 1, 1], is_idle=[0, 0, 1, 0]
        )
        per_rank = [
            _determine(
                _cfg(buckets),
                num_reqs=int(status.num_reqs[i]),
                num_tokens=int(status.num_tokens[i]),
                status=status,
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
    dp_group = MagicMock()
    dp_group.cpu_group = object()
    monkeypatch.setattr(dp_utils, "get_dp_group", lambda: dp_group)

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


class TestDetermineDraftBatch:
    # The draft decides inside a step that already reduced, so it reads the
    # published status. Its bounds are its own: the token dimension follows the
    # busiest rank and query_len stays what this rank stages.
    @staticmethod
    def _determine(
        status,
        *,
        num_reqs,
        num_tokens,
        is_prefill=False,
        dp_rank=0,
        draft_has_moe=True,
        **kw,
    ):
        return determine_draft_batch_execution_and_padding(
            cfg=_cfg(LADDER),
            status=status,
            dp_rank=dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=is_prefill,
            draft_has_moe=draft_has_moe,
            **kw,
        )

    def test_single_dp_pads_no_tokens(self):
        # Nothing to agree with: the batch rounds up to its bucket on decode and
        # stays as-is on prefill, and no token dimension is stated at all.
        desc, across = self._determine(None, num_reqs=3, num_tokens=3)
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            1,
            None,
        )
        assert across is None
        # Prefill runs one request, so its batch is not padded to a decode bucket.
        desc, _ = self._determine(None, num_reqs=1, num_tokens=10, is_prefill=True)
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            1,
            10,
            None,
        )

    def test_token_dimension_follows_the_busiest_rank(self):
        # bucket(max(4, 2)) = 4, and max(16 // 4, 16 // 2) = 8 tokens per request.
        desc, across = self._determine(
            _status(num_tokens=[16, 16], num_reqs=[4, 2]), num_reqs=4, num_tokens=16
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            4,
            32,
        )
        assert across.tolist() == [16, 16]

    def test_a_prefilling_rank_forces_the_top_bucket(self):
        # A peer prefilling makes the per-rank request counts unusable, so the
        # padding falls back to the largest bucket and the full token budget.
        desc, _ = self._determine(
            _status(num_tokens=[16, 512], num_reqs=[4, 1], is_prefill=[0, 1]),
            num_reqs=4,
            num_tokens=16,
        )
        assert desc.num_reqs_padded == 8
        assert desc.num_tokens_padded == MAX_NUM_TOKENS

    @pytest.mark.parametrize(
        "peer",
        [
            pytest.param(
                _status(num_tokens=[2, 16], num_reqs=[2, 4]), id="busier-peer"
            ),
            pytest.param(
                _status(num_tokens=[2, 512], num_reqs=[2, 1], is_prefill=[0, 1]),
                id="prefilling-peer",
            ),
        ],
    )
    def test_a_dense_draft_answers_from_its_own_pass(self, peer):
        # Nothing in a dense draft reads the token dimension, so no peer decides
        # anything for it: neither the busiest rank's batch nor the top bucket a
        # prefilling peer would dictate, and the dimension is the two tokens this
        # pass stages rather than a shared target.
        desc, _ = self._determine(peer, num_reqs=2, num_tokens=2, draft_has_moe=False)
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            2,
            1,
            2,
        )

    def test_the_drafting_loop_restates_the_status(self):
        # Every rank runs the loop one token per request whatever it staged for the
        # first pass, so the counts are restated at that length -- which also puts
        # the prefilling peer back in the decision instead of forcing its fallback.
        desc, across = self._determine(
            _status(num_tokens=[16, 512], num_reqs=[4, 1], is_prefill=[0, 1]),
            num_reqs=4,
            num_tokens=4,
            first_pass=False,
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            1,
            4,
        )
        assert across.tolist() == [4, 1]

    def test_parallel_drafting_scales_the_counts_by_the_query_length(self):
        # Parallel drafting stages `1 + num_speculative_tokens` queries per
        # request rather than one, so both the per-rank counts and the vector
        # the forward context consumes carry that factor.
        desc, across = self._determine(
            _status(num_tokens=[16, 512], num_reqs=[4, 1], is_prefill=[0, 1]),
            num_reqs=4,
            num_tokens=32,
            first_pass=False,
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            8,
            32,
        )
        assert across.tolist() == [32, 8]

    def test_an_idle_rank_may_run_a_length_it_did_not_report(self):
        # An idle rank reported the minimal entry so it would not drive the
        # decision, then runs the warmed decode length. Its own token count is the
        # one that cannot be checked against the status; a busy rank's still is.
        # The peer decodes four requests three tokens deep, which is the length the
        # idle rank runs too.
        idle = _status(num_tokens=[1, 12], num_reqs=[1, 4], is_idle=[1, 0])
        desc, _ = self._determine(idle, num_reqs=1, num_tokens=3)
        # The busy ranks' padding, so the all-gather buffer matches theirs.
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            3,
            12,
        )
        busy = _status(num_tokens=[1, 12], num_reqs=[1, 4])
        with pytest.raises(AssertionError, match="not on the step"):
            self._determine(busy, num_reqs=1, num_tokens=3)

    def test_a_busy_rank_at_one_token_stays_inside_the_bound(self):
        # A step the scheduler forced to no-spec: every rank drafts its own single
        # token, so what one stages is exactly the dimension they settle on. The
        # bound is tight there, which is the case worth pinning.
        desc, _ = self._determine(
            _status(num_tokens=[4, 4], num_reqs=[4, 4]), num_reqs=4, num_tokens=4
        )
        assert (desc.num_reqs_padded, desc.query_len, desc.num_tokens_padded) == (
            4,
            1,
            4,
        )

    def test_a_length_the_group_never_reported_is_rejected(self):
        # Callers reach a length by reporting it or by taking the one the ranks
        # settled on, so this is the contract rather than a reachable step: a pass
        # that got its length anywhere else stages past the dimension, and the
        # dispatch pad drops rows instead of raising.
        idle = _status(num_tokens=[1, 1], num_reqs=[1, 1], is_idle=[1, 0])
        with pytest.raises(AssertionError, match="does not fit"):
            self._determine(idle, num_reqs=1, num_tokens=3)

    def test_a_pin_replaces_only_the_token_dimension(self):
        # Warm-up asks for a token dimension the ranks would not choose; the batch
        # stays the one the status decides, which here is the peer's bucket rather
        # than this rank's own.
        desc, _ = self._determine(
            _status(num_tokens=[8, 16], num_reqs=[2, 4]),
            num_reqs=2,
            num_tokens=8,
            pinned_num_tokens_padded=MAX_NUM_TOKENS,
        )
        assert (desc.num_reqs_padded, desc.num_tokens_padded) == (4, MAX_NUM_TOKENS)


class TestDPStatus:
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
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert status.num_tokens == (8, 10, 6, 14)
        assert status.num_reqs == (8, 5, 3, 7)
        assert status.num_tokens_across_dp.cpu().tolist() == [8, 10, 6, 14]

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
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=8, num_reqs=8, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert status.is_prefill == (False, True, False, False)
        assert status.num_tokens == (8, 300, 4, 6)
        assert status.num_reqs == (8, 1, 4, 6)

    def test_the_local_prefill_flag_round_trips(self, fake_dp_collective):
        fake_dp_collective({r: _encode(8, 8, False) for r in (1, 2, 3)})
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=512, num_reqs=1, dp_size=4, dp_rank=0, is_prefill=True
        )
        assert status.is_prefill == (True, False, False, False)
        assert status.num_tokens == (512, 8, 8, 8)

    def test_the_idle_flag_round_trips(self, fake_dp_collective):
        # An idle rank reports the minimum, so the flag is what tells it apart from
        # a rank genuinely running one request one token deep.
        fake_dp_collective({1: _encode(8, 2, is_prefill=False)})
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=1,
            num_reqs=1,
            dp_size=2,
            dp_rank=0,
            is_prefill=False,
            is_idle=True,
        )
        assert status.is_idle == (True, False)
        assert status.num_tokens == (1, 8)
        assert status.num_reqs == (1, 2)
        assert status.num_tokens_across_dp.cpu().tolist() == [1, 8]

    def test_the_phase_and_idle_flags_do_not_bleed(self, fake_dp_collective):
        # Adjacent bits: a prefilling rank must not come back idle, and the peer's
        # flags must not land on this rank.
        fake_dp_collective({1: _encode(8, 2, is_prefill=False, is_idle=True)})
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=5,
            num_reqs=1,
            dp_size=2,
            dp_rank=0,
            is_prefill=True,
        )
        assert (status.is_prefill, status.is_idle) == ((True, False), (False, True))

    def test_boundary_max_values_round_trip(self, fake_dp_collective):
        # The largest values each field can hold must survive pack/unpack.
        fake_dp_collective({r: _encode(0xFFFF, 0x1FFF, False) for r in (1, 2, 3)})
        status = dp_utils._synchronize_dp_ranks(
            num_tokens=0xFFFF, num_reqs=0x1FFF, dp_size=4, dp_rank=0, is_prefill=False
        )
        assert status.num_tokens == (0xFFFF,) * 4
        assert status.num_reqs == (0x1FFF,) * 4

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
    # The composition: it joins the collective and hands the status to the rule.
    # Both halves are tested above; what is only testable here is what it
    # forwards -- a rank's phase, its idle bit and warm-up's pin.

    @staticmethod
    def _peer(num_tokens, num_reqs, *, prefill=False, idle=False):
        return _encode(num_tokens, num_reqs, prefill, idle)

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
        # answers with no shape at all and would stop the step.
        desc, route, status = self._coordinate(
            monkeypatch,
            peer=self._peer(1, 1),
            num_reqs=1,
            num_tokens=512,
            is_prefill=True,
            is_idle=False,
        )
        assert route is BatchRoute.ANY_PREFILL
        assert desc.query_len == 512
        # The status goes back with the decision: per-rank counts, not a broadcast
        # of this rank's own, and the phase the routes were decided from.
        assert status.num_tokens == (512, 1)
        assert status.num_reqs == (1, 1)
        assert status.is_prefill == (True, False)
        assert status.num_tokens_across_dp.tolist() == [512, 1]

    def test_an_idle_rank_adopts_the_busy_shape(self, monkeypatch):
        # The idle bit reaching the decision is what keeps this rank's minimal
        # (1 req, 1 token) from shrinking the graph the busy peer runs.
        desc, route, _status = self._coordinate(
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
        desc, route, _status = self._coordinate(
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
