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
"""Sizing the KV cache from compiled placements and a per-chiplet snapshot.

The placement shapes are the `CompiledProgram.input_specs[i].physical_placement`
form rebel emits: ints for static dims, `("symbol", name)` for the dynamic one.
`InputSpec.shape` is the concrete shape the graph was traced with, so it carries
the dynamic dim's compile-time extent.
"""

from types import SimpleNamespace

import pytest

from vllm_rbln.v1.worker.kv_placement import (
    ChipletMemory,
    dynamic_extent,
    eval_placement_dim,
    kv_bytes_per_unit,
    kv_growth,
    max_num_blocks,
    placement_itemsize,
    select_kv_input_groups,
    snapshot_from_allocator,
    snapshot_from_driver,
)

S = ("symbol", "s0")
HINT = 4


def _shard(node, chiplet, shape):
    return SimpleNamespace(node_id=node, chiplet_id=chiplet, slice_shape=tuple(shape))


def _placement(shape, shards, dtype="dlfloat16"):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype, shards=tuple(shards))


def _spec(placement, extent=HINT, name="kv"):
    """An InputSpec traced with the dynamic dim at `extent` kernel blocks."""
    shape = tuple(extent if not isinstance(d, int) else d for d in placement.shape)
    return SimpleNamespace(name=name, shape=shape, physical_placement=placement)


def _specs(placements, extent=HINT):
    return [_spec(p, extent, name=f"kv{i}") for i, p in enumerate(placements)]


def _program(placements, name="0/0", statics=1, extent=HINT):
    specs = [
        SimpleNamespace(name=f"x{i}", shape=(1,), physical_placement=None)
        for i in range(statics)
    ]
    specs += _specs(placements, extent)
    return SimpleNamespace(
        name=name, input_specs=tuple(specs), runtime=object(), device=None
    )


# [2, n, 8, 1, 1024, 128] fp16 split 4 KV heads per chiplet over two nodes: the
# example in rebel_compiler#13555.
HEAD_SHARDED = _placement(
    [2, S, 8, 1, 1024, 128],
    [_shard(1, 0, [2, S, 4, 1, 1024, 128]), _shard(0, 0, [2, S, 4, 1, 1024, 128])],
)
# The same tensor replicated whole on every chiplet: 4x the bytes of a head split.
REPLICATED = _placement(
    [2, S, 8, 1, 1024, 128],
    [_shard(0, c, [2, S, 8, 1, 1024, 128]) for c in range(4)],
)
# A sliding-window layer: kernel blocks of 128 tokens, 8 per 1024-token block.
WINDOWED = _placement(
    [2, S, 8, 1, 128, 128],
    [_shard(0, c, [2, S, 2, 1, 128, 128]) for c in range(4)],
)


class TestPlacementDim:
    @pytest.mark.parametrize(
        ("expr", "n", "expected"),
        [
            (7, 3, 7),
            (S, 3, 3),
            (("prod", 2, S), 5, 10),
            (("sum", S, 1), 5, 6),
            (("prod", ("sum", S, S), 3), 2, 12),
        ],
    )
    def test_evaluates_with_every_symbol_bound_to_num_blocks(self, expr, n, expected):
        assert eval_placement_dim(expr, n) == expected

    @pytest.mark.parametrize("expr", [("max", S, 1), (), "s0", 1.5, True])
    def test_rejects_what_it_does_not_know(self, expr):
        with pytest.raises(ValueError):
            eval_placement_dim(expr, 1)


class TestItemsize:
    @pytest.mark.parametrize(
        ("dtype", "size"),
        [
            ("dlfloat16", 2),
            ("float16", 2),
            ("bfloat16", 2),
            ("float32", 4),
            ("int8", 1),
            ("e4m3_float8", 1),
            ("E5M2_FLOAT8", 1),
            ("int4", 1),
            ("bool", 1),
        ],
    )
    def test_reads_the_bit_width_off_the_name(self, dtype, size):
        assert placement_itemsize(dtype) == size

    def test_a_nameless_dtype_is_refused(self):
        with pytest.raises(ValueError):
            placement_itemsize("float")


class TestDynamicExtent:
    def test_reads_the_compiled_size_of_the_symbolic_dim(self):
        assert dynamic_extent(_spec(HEAD_SHARDED, extent=7)) == 7

    def test_a_placement_without_a_dynamic_dim_is_refused(self):
        static = _placement([2, 4, 8], [_shard(0, 0, [2, 4, 8])])
        with pytest.raises(ValueError, match="no dynamic dim"):
            dynamic_extent(_spec(static))


class TestBytesPerUnit:
    def test_a_head_split_charges_each_shard_s_own_extent(self):
        # 2 * n * 4 * 1024 * 128 * 2 B = n * 2 MiB per shard
        assert kv_bytes_per_unit(_specs([HEAD_SHARDED]), 3, HINT) == {
            (0, 0): 3 * 2 * 2**20,
            (1, 0): 3 * 2 * 2**20,
        }

    def test_a_replicated_tensor_is_paid_on_every_chiplet(self):
        got = kv_bytes_per_unit(_specs([REPLICATED]), 1, HINT)
        assert got == {(0, c): 4 * 2**20 for c in range(4)}

    def test_layers_on_the_same_unit_add_up(self):
        assert kv_bytes_per_unit(_specs([HEAD_SHARDED, HEAD_SHARDED]), 1, HINT) == {
            (0, 0): 4 * 2**20,
            (1, 0): 4 * 2**20,
        }

    def test_zero_blocks_is_zero_bytes(self):
        got = kv_bytes_per_unit(_specs([HEAD_SHARDED]), 0, HINT)
        assert all(v == 0 for v in got.values())

    def test_a_windowed_layer_counts_its_kernel_blocks_per_block(self):
        """gpt-oss: the sliding-window view has block_size / window kernel blocks
        per manager block, so its symbol runs 8x faster than num_blocks."""
        spec = _spec(WINDOWED, extent=8 * HINT)
        # per kernel block per chiplet: 2 * 2 * 128 * 128 * 2 B = 128 KiB; 8 of them
        assert kv_bytes_per_unit([spec], 1, HINT) == {
            (0, c): 8 * 128 * 2**10 for c in range(4)
        }

    def test_an_extent_the_hint_does_not_divide_is_refused(self):
        with pytest.raises(RuntimeError, match="not a multiple"):
            kv_bytes_per_unit([_spec(HEAD_SHARDED, extent=HINT + 1)], 1, HINT)

    def test_hint_must_be_positive(self):
        with pytest.raises(ValueError):
            kv_bytes_per_unit(_specs([HEAD_SHARDED]), 1, 0)


class TestGrowth:
    def test_per_block_is_the_slope(self):
        growth = kv_growth(_specs([HEAD_SHARDED]), hint_blocks=HINT)
        assert growth.per_block == {(0, 0): 2 * 2**20, (1, 0): 2 * 2**20}
        assert growth.hint_blocks == HINT
        assert growth.num_inputs == 1
        assert growth.bytes_at(4) == {(0, 0): 8 * 2**20, (1, 0): 8 * 2**20}

    def test_a_full_and_a_windowed_layer_add_per_block(self):
        specs = [_spec(HEAD_SHARDED), _spec(WINDOWED, extent=8 * HINT)]
        growth = kv_growth(specs, hint_blocks=HINT)
        assert growth.per_block[(0, 0)] == 2 * 2**20 + 8 * 128 * 2**10
        assert growth.per_block[(0, 1)] == 8 * 128 * 2**10
        assert growth.per_block[(1, 0)] == 2 * 2**20

    def test_an_affine_offset_breaks_the_contract(self):
        """The compiler forbids padding a dynamic dim; if it ever pads, bytes stop
        passing through the origin and the slope alone would under-size."""
        padded = _placement(
            [2, ("sum", S, 1), 8],
            [_shard(0, 0, [2, ("sum", S, 1), 8])],
        )
        with pytest.raises(RuntimeError, match="not linear through the origin"):
            kv_growth([_spec(padded)], hint_blocks=HINT)

    def test_a_unit_without_growth_is_left_out(self):
        """MiniMax puts every KV shard on chiplet 0: chiplets 1-3 are not sized."""
        on_zero = _placement([S, 8], [_shard(0, 0, [S, 8])])
        base_only = _placement([S, 8], [_shard(0, 1, [0, 8])])
        growth = kv_growth(_specs([on_zero, base_only], extent=2), 2)
        assert set(growth.per_block) == {(0, 0)}


class TestSelectKvInputGroups:
    def test_static_programs_are_skipped(self):
        logits = _program([], name="0/2")
        decode = _program([HEAD_SHARDED], name="0/1")
        [(specs, program)] = select_kv_input_groups([logits, decode])
        assert program is decode
        assert [s.physical_placement for s in specs] == [HEAD_SHARDED]

    def test_agreeing_programs_are_counted_once(self):
        """prefill and every decode bucket bind the same KV tensors."""
        prefill = _program([HEAD_SHARDED, HEAD_SHARDED], name="0/0")
        decode = _program([HEAD_SHARDED, HEAD_SHARDED], name="0/1", statics=3)
        [(specs, _)] = select_kv_input_groups([prefill, decode])
        assert len(specs) == 2

    def test_shard_order_does_not_matter(self):
        flipped = _placement(HEAD_SHARDED.shape, tuple(reversed(HEAD_SHARDED.shards)))
        groups = select_kv_input_groups([_program([HEAD_SHARDED]), _program([flipped])])
        assert len(groups) == 1

    def test_a_drafter_binding_its_own_tensors_is_a_second_group(self):
        """A speculative drafter's programs bind fewer, separate KV tensors."""
        target = _program([HEAD_SHARDED, HEAD_SHARDED], name="0/0")
        drafter = _program([HEAD_SHARDED], name="1/0")
        groups = select_kv_input_groups([target, drafter, target])
        assert [len(specs) for specs, _ in groups] == [2, 1]
        assert [program.name for _, program in groups] == ["0/0", "1/0"]

    def test_the_same_tensors_placed_two_ways_are_refused(self):
        with pytest.raises(RuntimeError, match="disagree"):
            select_kv_input_groups([_program([HEAD_SHARDED]), _program([REPLICATED])])

    def test_a_different_compiled_extent_is_another_tensor(self):
        groups = select_kv_input_groups(
            [_program([HEAD_SHARDED]), _program([HEAD_SHARDED], extent=2 * HINT)]
        )
        assert len(groups) == 2

    def test_no_dynamic_input_anywhere_names_the_cache_root(self):
        with pytest.raises(RuntimeError, match="VLLM_CACHE_ROOT"):
            select_kv_input_groups([_program([]), _program([])])
        with pytest.raises(RuntimeError, match="none of the 0"):
            select_kv_input_groups([])


class TestSnapshots:
    def test_driver_reply_pairs_total_and_used_per_chiplet(self):
        stats = {
            "npu.0.total": 100,
            "npu.0.used": 30,
            "npu.0.chiplet.0.total": 50,
            "npu.0.chiplet.0.used": 20,
            "npu.0.chiplet.0.free": 30,
            "npu.0.chiplet.0.largest_free": 30,
            "npu.0.chiplet.1.total": 50,
            "npu.0.chiplet.1.used": 10,
            "npu.0.chiplet.1.free": 40,
        }
        assert snapshot_from_driver(stats) == {
            (0, 0): ChipletMemory(total=50, used=20),
            (0, 1): ChipletMemory(total=50, used=10),
        }

    def test_driver_reply_without_chiplets_is_refused(self):
        with pytest.raises(RuntimeError, match="no per-chiplet"):
            snapshot_from_driver({"npu.0.total": 100, "npu.0.used": 30})

    def test_allocator_reply_adds_what_the_allocator_cannot_see(self):
        stats = {
            "npu.0.chiplet.0.reserved.current": 1000,
            "npu.0.chiplet.0.allocated.current": 900,
            "npu.0.chiplet.1.reserved.current": 200,
            "npu.0.chiplet.1.allocated.current": 100,
        }
        got = snapshot_from_allocator(
            stats, memory_per_chiplet=5000, foreign_card_used_bytes=401, reserve_bytes=7
        )
        # 401 of foreign usage over 2 units -> 200 each (floor).
        assert got == {
            (0, 0): ChipletMemory(total=5000, used=1000 + 200 + 7),
            (0, 1): ChipletMemory(total=5000, used=200 + 200 + 7),
        }

    def test_allocator_reply_without_a_context_is_refused(self):
        with pytest.raises(RuntimeError, match="reserved.current"):
            snapshot_from_allocator(
                {}, memory_per_chiplet=1, foreign_card_used_bytes=0, reserve_bytes=0
            )


class TestMaxNumBlocks:
    GIB = 2**30

    def _snapshot(self, used, total=35 * GIB):
        return {unit: ChipletMemory(total=total, used=u) for unit, u in used.items()}

    def _growth(self):
        return kv_growth(
            _specs([HEAD_SHARDED]), hint_blocks=HINT
        )  # 2 MiB / block / unit

    def test_the_tightest_chiplet_decides(self):
        growth = self._growth()
        snapshot = self._snapshot({(0, 0): 10 * self.GIB, (1, 0): 20 * self.GIB})
        resident = growth.bytes_at(HINT)
        n, fits = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident=resident
        )
        # (35 GiB - (20 GiB - 8 MiB)) / 2 MiB = 15 GiB / 2 MiB + 4
        assert fits[(1, 0)].num_blocks == 15 * 512 + 4
        assert fits[(0, 0)].num_blocks == 25 * 512 + 4
        assert n == 15 * 512 + 4
        assert fits[(1, 0)].base == 20 * self.GIB - 8 * 2**20

    def test_each_shard_is_counted_at_the_allocator_s_block_size(self):
        # One KV head per shard: 512 KiB per block. 2 blocks (1 MiB) take a
        # 2 MiB small block, 3 blocks (1.5 MiB) a 20 MiB medium block, and 40
        # blocks (20 MiB) round to 2 MiB.
        thin = _placement(
            [2, S, 8, 1, 1024, 128], [_shard(0, 0, [2, S, 1, 1, 1024, 128])]
        )
        growth = kv_growth(_specs([thin]), hint_blocks=HINT)
        assert growth.bytes_at(3) == {(0, 0): 3 * 512 * 2**10}
        assert growth.allocated_at(2) == {(0, 0): 2 * 2**20}
        assert growth.allocated_at(3) == {(0, 0): 20 * 2**20}
        assert growth.allocated_at(41) == {(0, 0): 22 * 2**20}
        # 2.5 MiB of room fits 5 blocks linearly; 3..40 blocks reserve 20 MiB,
        # so only 2 blocks (a 2 MiB block) actually fit.
        snapshot = self._snapshot({(0, 0): 35 * self.GIB - 5 * 512 * 2**10})
        n, fits = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident={}
        )
        assert fits[(0, 0)].num_blocks == 5
        assert n == 2

    def test_reserve_bytes_are_charged_as_base(self):
        growth = self._growth()
        snapshot = self._snapshot({(0, 0): 10 * self.GIB, (1, 0): 10 * self.GIB})
        plain, _ = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident={}
        )
        reserved, fits = max_num_blocks(
            snapshot,
            growth,
            gpu_memory_utilization=1.0,
            kv_resident={},
            reserve_bytes=64 * 2**20,
        )
        assert reserved == plain - 32
        assert fits[(0, 0)].reserve == 64 * 2**20
        assert fits[(0, 0)].base == 10 * self.GIB + 64 * 2**20

    def test_gpu_memory_utilization_scales_the_budget(self):
        growth = self._growth()
        snapshot = self._snapshot({(0, 0): 0, (1, 0): 0})
        full, _ = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident={}
        )
        half, _ = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=0.5, kv_resident={}
        )
        assert half == full // 2

    def test_not_subtracting_the_resident_cache_costs_exactly_the_hint(self):
        """TP>=2 keeps the compile-time cache, so the caller passes no resident
        bytes and the answer drops by the hint."""
        growth = self._growth()
        snapshot = self._snapshot({(0, 0): 8 * 2**20, (1, 0): 8 * 2**20})
        tp1, _ = max_num_blocks(
            snapshot,
            growth,
            gpu_memory_utilization=1.0,
            kv_resident=growth.bytes_at(HINT),
        )
        tp2, _ = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident={}
        )
        assert tp1 - tp2 == HINT

    def test_units_the_cache_does_not_grow_on_are_ignored(self):
        growth = self._growth()
        snapshot = self._snapshot(
            {(0, 0): 0, (1, 0): 0, (0, 3): 35 * self.GIB}  # chiplet 3 is full
        )
        n, fits = max_num_blocks(
            snapshot, growth, gpu_memory_utilization=1.0, kv_resident={}
        )
        assert n > 0
        assert (0, 3) not in fits

    def test_a_unit_missing_from_the_snapshot_is_refused(self):
        with pytest.raises(RuntimeError, match=r"\(1, 0\)"):
            max_num_blocks(
                self._snapshot({(0, 0): 0}),
                self._growth(),
                gpu_memory_utilization=1.0,
                kv_resident={},
            )

    def test_a_base_over_budget_answers_zero_not_negative(self):
        snapshot = self._snapshot({(0, 0): 34 * self.GIB, (1, 0): 0})
        n, fits = max_num_blocks(
            snapshot, self._growth(), gpu_memory_utilization=0.9, kv_resident={}
        )
        assert n == 0
        assert fits[(0, 0)].num_blocks == 0

    @pytest.mark.parametrize("gmu", [0, -0.1, 1.5])
    def test_gpu_memory_utilization_is_range_checked(self, gmu):
        with pytest.raises(ValueError):
            max_num_blocks(
                self._snapshot({(0, 0): 0, (1, 0): 0}),
                self._growth(),
                gpu_memory_utilization=gmu,
                kv_resident={},
            )
