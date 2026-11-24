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

"""DSpark query layout, Markov sampling, and local-attention cache semantics."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.config import set_current_vllm_config
from vllm.model_executor.models.qwen3_dflash import _dflash_layer_causal
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.dflash import DFlashProposer

from tests.vllm.v1.attention.utils import make_builder, make_impl
from tests.vllm.v1.spec_decode.utils import make_cad
from tests.vllm.vllm_config import make_vllm_config
from vllm_rbln.v1.spec_decode.dspark import RBLNDSparkProposer


@pytest.mark.parametrize("anchor", [True, False])
def test_anchor_layout_and_rejected_context(monkeypatch, anchor):
    hf = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=["sliding_attention"] * 3,
        sliding_window=128,
        sample_from_anchor=anchor,
    )
    spec = SimpleNamespace(
        method="dspark",
        enforce_eager=False,
        draft_model_config=SimpleNamespace(hf_config=hf),
    )
    cfg = SimpleNamespace(speculative_config=spec)

    def init_base(self, vllm_config, device, runner):
        assert vllm_config.speculative_config.method == "dflash"
        self.num_speculative_tokens = 5
        self.arange = torch.arange(32)
        self.device = device
        self.input_ids = torch.zeros(32, dtype=torch.int32)
        self.positions = torch.zeros(32, dtype=torch.int64)
        self._context_positions_buffer = torch.zeros(32, dtype=torch.int64)
        self.parallel_drafting_token_id = 99
        self.dflash_causal = _dflash_layer_causal(hf, 0)

    monkeypatch.setattr(DFlashProposer, "__init__", init_base)
    proposer = RBLNDSparkProposer(cfg, torch.device("cpu"))
    assert spec.method == "dspark"
    q = 5 if anchor else 6
    assert proposer.num_query_per_req == q
    cad = make_cad([0, 3, 6], [131, 260])
    n, starts, lengths = proposer._fill_first_pass_inputs(
        torch.tensor([7, 8]),
        torch.tensor([128, 129, 130, 257, 258, 259]),
        cad,
        torch.tensor([1, 2]),
        q,
    )
    assert (n, starts.tolist(), lengths.tolist()) == (3, [128, 257], [2, 1])
    assert proposer._ctx_gather.tolist() == [0, 1, 3]
    assert proposer.positions[: 2 * q].view(2, q)[:, 0].tolist() == [130, 258]
    ids = proposer.input_ids[: 2 * q].view(2, q)
    assert ids[:, 0].tolist() == [7, 8]
    assert (ids[:, 1:] == 99).all()
    expected = (
        [list(range(5)), list(range(5, 10))]
        if anchor
        else [list(range(1, 6)), list(range(7, 12))]
    )
    assert proposer._sample_indices(2, q).view(2, 5).tolist() == expected


def test_markov_sampling_uses_remapped_previous_tokens(monkeypatch):
    proposer = RBLNDSparkProposer.__new__(RBLNDSparkProposer)
    proposer.num_speculative_tokens = 3
    # Draft ids 0/1/2 map to target ids 2/4/6, which select different next biases.
    mapping = torch.tensor([2, 4, 6])
    bias = torch.zeros(8, 3)
    bias[1, 0] = bias[2, 1] = bias[4, 2] = bias[6, 0] = 9
    proposer.model = SimpleNamespace(
        compute_draft_logits=lambda h: h,
        markov_embed=lambda ids: bias[ids],
        markov_bias=lambda h: h,
        map_draft_to_target=lambda ids: mapping[ids],
    )
    monkeypatch.setattr(torch.ops.rbln, "argmax", lambda logits: logits.argmax(-1))
    result = proposer._draft_token_ids(torch.zeros(6, 3), torch.tensor([[1], [4]]))
    assert result.view(2, 3).tolist() == [[0, 1, 2], [2, 0, 1]]
    assert mapping[result].view(2, 3).tolist() == [[2, 4, 6], [6, 2, 4]]


@pytest.mark.parametrize("kind", ["axk2", "minimax", "full"])
def test_grouping_preserves_upstream_fallback_conditions(kind):
    cfg = make_vllm_config()
    cfg.cache_config.kv_cache_layout = "LBNHC"
    common = dict(block_size=1024, dtype=torch.bfloat16)
    if kind == "axk2":
        target = MLAAttentionSpec(**common, num_kv_heads=1, head_size=576)
        draft = SlidingWindowSpec(
            **common, num_kv_heads=64, head_size=112, sliding_window=128
        )
    else:
        target = FullAttentionSpec(**common, num_kv_heads=4, head_size=128)
        draft = SlidingWindowSpec(
            **common, num_kv_heads=8, head_size=128, sliding_window=1024
        )
        if kind == "full":
            draft = FullAttentionSpec(**common, num_kv_heads=8, head_size=128)
    groups = get_kv_cache_groups(cfg, {"target": target, "draft": draft})
    if kind in ("axk2", "full"):
        assert len(groups) == 1
        assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
        spec = groups[0].kv_cache_spec.kv_cache_specs["draft"]
        assert isinstance(spec, FullAttentionSpec)
        if kind == "axk2":
            assert spec.sliding_window == 128
            assert spec.head_size == 112
            assert spec.page_size_padded is None
            builder = make_builder(cfg, spec=spec)
            assert builder.swa_appends
    else:
        assert len(groups) == 2
        spec = next(g.kv_cache_spec for g in groups if "draft" in g.layer_names)
        assert isinstance(spec, SlidingWindowSpec)
        assert spec.page_size_padded is None


@pytest.mark.parametrize(
    "causal,head,kv_heads,q_heads,window,chunk",
    [
        (True, 112, 64, 64, 128, 5),
        (False, 128, 8, 32, 1024, 8),
    ],
    ids=["axk2", "minimax"],
)
@pytest.mark.parametrize("start", [0, 1021, 2050])
def test_window_attention_matches_dense_reference(
    causal, head, kv_heads, q_heads, window, chunk, start
):
    torch.manual_seed(0)
    block = 128 if causal else 1024
    cfg = make_vllm_config(
        max_model_len=4096, attention_config={"use_non_causal": not causal}
    )
    cfg.device_config.device = torch.device("cpu")
    spec = SlidingWindowSpec(
        block_size=block,
        num_kv_heads=kv_heads,
        head_size=head,
        sliding_window=window,
        dtype=torch.float32,
    )
    if causal:
        spec = FullAttentionSpec(
            block_size=block,
            num_kv_heads=kv_heads,
            head_size=head,
            sliding_window=window,
            dtype=torch.float32,
        )
    builder = make_builder(cfg, spec=spec)
    proposer = RBLNDSparkProposer.__new__(RBLNDSparkProposer)
    proposer.dflash_causal = causal
    proposer.num_query_per_req = chunk
    proposer.runner = SimpleNamespace(bind_kv_cache_to_metadata=lambda md: None)
    proposer.draft_attn_groups = [
        SimpleNamespace(
            kv_cache_group_id=3,
            layer_names=["draft"],
            get_metadata_builder=lambda: builder,
        )
    ]
    starts = [start, max(0, start - 3)]
    cad = make_cad([0, chunk, 2 * chunk], starts)
    columns = 4096 // block
    tables = torch.arange(1, 2 * columns + 1, dtype=torch.int32).view(2, columns)
    positions = torch.zeros(2 * chunk, dtype=torch.int64)
    positions[::chunk] = torch.tensor(starts)
    metadata = proposer._build_draft_attn_metadata(
        cad,
        positions,
        2,
        3,
        block_tables={3: tables},
    )["draft"]
    assert metadata.swa_appends
    if not causal:
        assert metadata.swa_attn_masks.shape[0] == 3
        assert not metadata.swa_attn_masks[2].any()
        assert metadata.swa_attn_masks.shape[-1] < cfg.model_config.max_model_len
    # Each request has separate physical pages; row 3 is a padded dummy request.
    cache = torch.randn(2 * columns + 1, 2, kv_heads, 1, block, head)
    reference_cache = cache.clone()
    q = torch.randn(3, chunk, q_heads * head)
    k = torch.randn(3, chunk, kv_heads * head)
    v = torch.randn_like(k)
    impl = make_impl(
        cfg,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=head,
        sliding_window=window,
        scale=head**-0.5,
    )
    actual = torch.empty_like(q)
    with set_current_vllm_config(cfg):
        impl.forward(
            SimpleNamespace(),
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cache,
            metadata,
            actual.view(-1, q_heads, head),
        )
    for row, offset in enumerate(starts):
        pages = reference_cache[tables[row].long()]
        keys = pages[:, 0, :, 0].permute(1, 0, 2, 3).reshape(kv_heads, -1, head)
        values = pages[:, 1, :, 0].permute(1, 0, 2, 3).reshape(kv_heads, -1, head)
        keys[:, offset : offset + chunk] = (
            k[row].view(chunk, kv_heads, head).transpose(0, 1)
        )
        values[:, offset : offset + chunk] = (
            v[row].view(chunk, kv_heads, head).transpose(0, 1)
        )
        key_positions = torch.arange(keys.shape[1])
        query_positions = offset + torch.arange(chunk)
        distance = query_positions[:, None] - key_positions[None, :]
        visible = (
            ((distance >= 0) & (distance < window))
            if causal
            else (distance.abs() < window)
        )
        visible &= key_positions[None, :] < offset + chunk
        expected = (
            F.scaled_dot_product_attention(
                q[row].view(chunk, q_heads, head).transpose(0, 1),
                keys.repeat_interleave(q_heads // kv_heads, 0),
                values.repeat_interleave(q_heads // kv_heads, 0),
                attn_mask=visible,
            )
            .transpose(0, 1)
            .reshape(chunk, -1)
        )
        torch.testing.assert_close(actual[row], expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("block", [512, 1024, 8192])
@pytest.mark.parametrize("prefix", [False, True])
def test_minimax_main_and_indexer_share_kernel_geometry(block, prefix):
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionBackend,
    )
    from vllm_rbln.v1.attention.backends.minimax_m3 import (
        RBLNMiniMaxM3IndexerBackend,
        RBLNMiniMaxM3SparseBackend,
    )
    from vllm_rbln.v1.worker.utils import prepare_kernel_block_sizes

    cfg = make_vllm_config(
        max_model_len=32768, block_size=block, enable_prefix_caching=prefix
    )
    cfg.cache_config.kv_cache_layout = "LBNHC"
    common = dict(block_size=block, dtype=torch.bfloat16, head_size=128)
    main = FullAttentionSpec(**common, num_kv_heads=4)
    indexer = FullAttentionSpec(**common, num_kv_heads=1, head_size_v=0)
    draft = SlidingWindowSpec(**common, num_kv_heads=8, sliding_window=1024)
    specs = {f"main.{i}": main for i in range(60)}
    specs.update({f"indexer.{i}": indexer for i in range(3, 60)})
    specs.update({f"draft.{i}": draft for i in range(6)})
    groups = get_kv_cache_groups(cfg, specs)
    backends = {
        "main": RBLNMiniMaxM3SparseBackend,
        "indexer": RBLNMiniMaxM3IndexerBackend,
        "draft": RBLNFlashAttentionBackend,
    }
    attn_groups = [
        [SimpleNamespace(backend=backends[g.layer_names[0].split(".")[0]])]
        for g in groups
    ]
    sizes = prepare_kernel_block_sizes(
        KVCacheConfig(64, [], groups),
        attn_groups,
        sub_block_size=block if prefix else None,
    )
    target_sizes = set()
    widths = set()
    for gid, group in enumerate(groups):
        spec = group.kv_cache_spec
        assert spec.page_size_padded is None
        kind = group.layer_names[0].split(".")[0]
        assert spec.block_size % sizes[gid] == 0
        if kind == "draft":
            assert isinstance(spec, SlidingWindowSpec)
            continue
        target_sizes.add(sizes[gid])
        builder = make_builder(
            cfg, spec=spec, builder_cls=backends[kind].get_builder_cls()
        )
        builder.block_size = sizes[gid]
        # A manager page's extra kernel blocks widen the physical table.
        width = (
            (32768 + spec.block_size - 1)
            // spec.block_size
            * (spec.block_size // sizes[gid])
        )
        cad = make_cad([0, 1], [1])
        cad.block_table_tensor = torch.ones(1, width, dtype=torch.int32)
        metadata = builder.build(cad, torch.tensor([0]), batch_pad=1, is_prefill=False)
        widths.add(metadata.block_tables.shape[-1])
    assert len(target_sizes) == len(widths) == 1
    assert next(iter(widths)) == 32768 // next(iter(target_sizes))


def test_context_projection_is_written_to_each_layers_own_group():
    from vllm_rbln.config import RBLNConfig

    proposer = RBLNDSparkProposer.__new__(RBLNDSparkProposer)
    caches = [torch.zeros(6, 2, 2, 1, size, 3) for size in (4, 8)]
    proposer.model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(layer_name=f"draft.{i}", kv_cache=cache)
                    )
                )
                for i, cache in enumerate(caches)
            ]
        )
    )
    proposer.vllm_config = SimpleNamespace(additional_config=RBLNConfig())
    proposer.draft_block_sizes = {2: 4, 5: 8}
    proposer.layer_kv_cache_group_ids = {"draft.0": 2, "draft.1": 5}
    proposer._ctx_positions_cpu = torch.tensor([6, 7, 8, 9])
    proposer._ctx_gather = torch.arange(4)
    proposer._context_positions_buffer = proposer._ctx_positions_cpu.clone()
    proposer.max_num_tokens = 8
    proposer._proj_buckets = (8,)
    proposer._proj_states = torch.zeros(8, 3)
    proposer._proj_positions = torch.zeros(1, 8, dtype=torch.int64)
    keys = torch.arange(2 * 2 * 8 * 3).view(2, 2, 8, 3).float()
    calls = []

    def project(states, positions):
        calls.append(positions[0, :4].tolist())
        return keys, keys + 1000

    proposer._project_context_kv = project
    tables = {2: torch.tensor([[1, 3, 5]]), 5: torch.tensor([[4, 2]])}
    proposer._write_context_kv(torch.zeros(4, 3), tables, 1, 4, torch.tensor([4]))
    assert calls == [[6, 7, 8, 9]]
    for layer, (gid, size) in enumerate(proposer.draft_block_sizes.items()):
        for token, position in enumerate([6, 7, 8, 9]):
            page = tables[gid][0, position // size]
            torch.testing.assert_close(
                caches[layer][page, 0, :, 0, position % size], keys[layer, :, token]
            )
            torch.testing.assert_close(
                caches[layer][page, 1, :, 0, position % size],
                keys[layer, :, token] + 1000,
            )
        assert not caches[layer][0].any()
