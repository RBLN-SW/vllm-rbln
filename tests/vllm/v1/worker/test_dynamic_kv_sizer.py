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

"""`DynamicKvSizer`: the sizer's placement-based KV sizing, tested on
SimpleNamespace stand-ins that carry only the state each path reads."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_rbln.v1.worker.dynamic_kv_sizer as dks
from vllm_rbln.v1.worker.dynamic_kv_sizer import DynamicKvSizer

S0 = ("symbol", "s0")


def _shard(node, chiplet, shape):
    return SimpleNamespace(node_id=node, chiplet_id=chiplet, slice_shape=tuple(shape))


def _placement(shards):
    return SimpleNamespace(
        shape=(2, S0, 8, 1, 1024, 128), dtype="dlfloat16", shards=shards
    )


HEAD_SPLIT = _placement(tuple(_shard(0, c, (2, S0, 2, 1, 1024, 128)) for c in range(4)))

# The same tensor as HEAD_SPLIT, bound by a second program: dynamo names the
# symbol differently, which is what splits one tensor set into two groups.
S1 = ("symbol", "s40")
HEAD_SPLIT_S1 = SimpleNamespace(
    shape=(2, S1, 8, 1, 1024, 128),
    dtype="dlfloat16",
    shards=tuple(_shard(0, c, (2, S1, 2, 1, 1024, 128)) for c in range(4)),
)


def _program(placements, name="0/0", runtime=None, device=None, extent=4):
    specs = (SimpleNamespace(name="ids", shape=(1,), physical_placement=None),) + tuple(
        SimpleNamespace(
            name=f"kv.{i}",
            shape=tuple(extent if not isinstance(d, int) else d for d in p.shape),
            physical_placement=p,
        )
        for i, p in enumerate(placements)
    )
    return SimpleNamespace(
        name=name,
        input_specs=specs,
        runtime=runtime if runtime is not None else object(),
        device=device,
    )


def _kv_cache_tensors_for(programs):
    """One KVCacheTensor per KV input the grouping will charge; the sizer checks
    the two counts agree."""
    try:
        groups = dks.select_kv_input_groups(list(programs))
    except RuntimeError:
        return []
    return [
        SimpleNamespace(shared_by=[f"layer.{i}"])
        for i, _ in enumerate(s for g, _ in groups for s in g)
    ]


def _bind_sizing(sizer) -> None:
    """Give a SimpleNamespace sizer the real placement-sizing methods."""
    sizer.log_release_shortfall = DynamicKvSizer.log_release_shortfall
    sizer._specs_covering_tensors = DynamicKvSizer._specs_covering_tensors
    for name in (
        "_kv_growth_from_programs",
        "_size_kv_from_snapshot",
        "_size_kv_and_release",
        "_propose_kv_size",
    ):
        method = getattr(DynamicKvSizer, name)
        setattr(sizer, name, lambda *a, _m=method, **kw: _m(sizer, *a, **kw))


class TestComputeDynamicKvNumBlocks:
    """`compute_num_blocks` = placement slope x memory snapshot.

    The snapshot is stubbed at the sizer seam (`memory_snapshot`);
    `TestDynamicKvMemorySnapshot` covers the seam itself.
    """

    GIB = 2**30
    HINT = 4
    TOTAL = 35 * GIB

    @pytest.fixture(autouse=True)
    def _real_device(self):
        with patch.object(
            dks.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            yield

    def _sizer(self, *, programs, snapshot, tp_size=1, gmu=1.0):
        sizer = SimpleNamespace(
            rank=0,
            device=torch.device("cpu"),
            mode=dks.DynamicKvMode.ACTIVE,
            mode_reason=None,
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None, gpu_memory_utilization=gmu
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
            kv_blocks_before_shrink=211,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(
                    num_blocks=self.HINT,
                    kv_cache_tensors=_kv_cache_tensors_for(programs),
                )
            ),
            programs=list(programs),
            memory_snapshot=lambda device: (snapshot, "stub"),
            copy_stream_reserve_bytes=lambda: 0,
            release_kv_cache_tensors=lambda cfg: None,
        )
        _bind_sizing(sizer)
        return sizer

    def _snapshot(self, used):
        return {
            (0, c): dks.ChipletMemory(total=self.TOTAL, used=u)
            for c, u in enumerate(used)
        }

    def test_the_heaviest_chiplet_decides_after_the_compile_cache_is_released(self):
        # Two layers -> 2 MiB per block per chiplet.
        programs = [
            _program([HEAD_SPLIT, HEAD_SPLIT], name="0/0"),
            _program([HEAD_SPLIT, HEAD_SPLIT], name="0/1"),
            _program([], name="0/2"),
        ]
        used = [5 * self.GIB, 30 * self.GIB, 1 * self.GIB, 0]
        order: list = []
        sizer = self._sizer(programs=programs, snapshot=self._snapshot(used))
        sizer.release_kv_cache_tensors = lambda cfg: order.append("release")
        snapshot = sizer.memory_snapshot

        def recording_snapshot(device):
            order.append("snapshot")
            return snapshot(device)

        sizer.memory_snapshot = recording_snapshot
        n = DynamicKvSizer.compute_num_blocks(sizer)
        # chiplet 1: (35 GiB - 30 GiB) / 2 MiB = 2560 blocks, nothing subtracted
        assert n == 5 * 512
        # The first snapshot is the before-release reading the shortfall check
        # compares against; the count is sized from the one after it.
        assert order == ["snapshot", "release", "snapshot"]

    def test_a_dry_run_reports_and_resizes_nothing(self, caplog, monkeypatch):
        current = 200
        programs = [_program([HEAD_SPLIT, HEAD_SPLIT], name="0/0", extent=current)]
        resident = current * 2 * 2**20
        sizer = self._sizer(
            programs=programs,
            snapshot=self._snapshot([30 * self.GIB + resident] * 4),
        )
        sizer.mode = dks.DynamicKvMode.DRY_RUN
        sizer.cache_config.num_gpu_blocks_override = current
        sizer.kv_blocks_before_shrink = None
        sizer.model_runner.kv_cache_config.num_blocks = current
        sizer.vllm_config = SimpleNamespace()
        monkeypatch.setattr(
            dks,
            "minimum_kv_blocks",
            lambda cfg, kv: SimpleNamespace(one_request=8, decode_batch=1, needed=9),
        )
        sizer.log_dry_run = lambda *args: (DynamicKvSizer.log_dry_run(sizer, *args))
        _bind_sizing(sizer)
        with caplog.at_level("WARNING"):
            assert DynamicKvSizer.compute_num_blocks(sizer) is None
        # (35 GiB - 30 GiB) / 2 MiB = 2560 blocks if the 200 in use come back,
        # 2360 if their 400 MiB stay resident.
        assert (
            "vllm sized 200 blocks, this feature would set 2560 (+2360) if the "
            "runtime hands the current cache back, 2360 if it stays resident"
            in caplog.text
        )
        assert "needs 9 (one request 8, decode batch 1, +1 null block)" in caplog.text
        assert "would be accepted" in caplog.text
        assert "headroom=" in caplog.text
        # 2560 blocks of 2 MiB on top of the 30 GiB base fill the 35 GiB budget.
        assert (
            "at 2560 blocks: used=37580963840 budget_left=0 total_left=0" in caplog.text
        )
        # The fill is what the rollout reads: how full each chiplet ends up.
        assert "= 100.0% of budget, 100.0% of DRAM)" in caplog.text
        assert "now 86.8% of budget" in caplog.text

    def test_a_dry_run_that_cannot_size_warns_instead_of_raising(self, caplog):
        sizer = self._sizer(programs=[_program([])], snapshot=self._snapshot([0] * 4))
        sizer.mode = dks.DynamicKvMode.DRY_RUN
        sizer.kv_blocks_before_shrink = None
        with caplog.at_level("WARNING"):
            assert DynamicKvSizer.compute_num_blocks(sizer) is None
        assert "could not be computed" in caplog.text

    def test_the_copy_stream_reserve_comes_off_every_chiplet(self):
        programs = [_program([HEAD_SPLIT, HEAD_SPLIT], name="0/0")]
        sizer = self._sizer(
            programs=programs, snapshot=self._snapshot([30 * self.GIB] * 4)
        )
        sizer.copy_stream_reserve_bytes = lambda: 64 * 2**20
        # (35 GiB - 30 GiB - 64 MiB) / 2 MiB = 2560 - 32 blocks
        assert DynamicKvSizer.compute_num_blocks(sizer) == 5 * 512 - 32

    def test_gpu_memory_utilization_bounds_the_budget(self):
        programs = [_program([HEAD_SPLIT])]
        # The snapshot is taken after the compile cache is released: base 0.
        snapshot = self._snapshot([0] * 4)
        full = DynamicKvSizer.compute_num_blocks(
            self._sizer(programs=programs, snapshot=snapshot, gmu=1.0)
        )
        half = DynamicKvSizer.compute_num_blocks(
            self._sizer(programs=programs, snapshot=snapshot, gmu=0.5)
        )
        assert full == 35 * 1024
        assert half == full // 2

    def test_the_snapshot_is_taken_on_the_program_s_device(self):
        seen = []
        programs = [_program([HEAD_SPLIT], device=torch.device("cpu", 3))]
        sizer = self._sizer(programs=programs, snapshot=self._snapshot([0] * 4))

        def snapshot(device):
            seen.append(device)
            return self._snapshot([0] * 4), "stub"

        sizer.memory_snapshot = snapshot
        DynamicKvSizer.compute_num_blocks(sizer)
        # Both readings -- before the release and the one sized from.
        assert seen == [torch.device("cpu", 3)] * 2

    def test_a_base_over_budget_is_refused_with_the_breakdown(self):
        programs = [_program([HEAD_SPLIT])]
        snapshot = self._snapshot([self.TOTAL, 0, 0, 0])
        with pytest.raises(RuntimeError, match="no KV block fits") as exc:
            DynamicKvSizer.compute_num_blocks(
                self._sizer(programs=programs, snapshot=snapshot, gmu=0.9)
            )
        assert "0:0(" in str(exc.value)

    def test_a_dummy_device_keeps_the_estimate(self, caplog):
        """The executor compiles under RBLN_DUMMY_DEVICE=1: nothing to measure,
        so the pre-shrink count is restored instead of refusing the compile."""
        programs = [_program([HEAD_SPLIT])]
        sizer = self._sizer(programs=programs, snapshot=self._snapshot([0] * 4))
        with (
            patch.object(
                dks.torch,
                "rbln",
                SimpleNamespace(is_dummy_device=lambda: True),
                create=True,
            ),
            caplog.at_level("WARNING"),
        ):
            assert DynamicKvSizer.compute_num_blocks(sizer) is None
        assert "RBLN_DUMMY_DEVICE" in caplog.text

    def test_the_same_tensors_seen_twice_are_counted_once(self, caplog):
        """Prefill and decode bind the same KV tensors, but dynamo names their
        symbols differently, so they form two groups. Summing both would double
        the slope and halve the count."""
        programs = [
            _program([HEAD_SPLIT, HEAD_SPLIT], name="0/0"),
            _program([HEAD_SPLIT_S1, HEAD_SPLIT_S1], name="0/1"),
        ]
        sizer = self._sizer(
            programs=programs, snapshot=self._snapshot([30 * self.GIB] * 4)
        )
        # Two programs, two groups, but vllm allocated two tensors, not four.
        sizer.model_runner.kv_cache_config.kv_cache_tensors = [
            SimpleNamespace(shared_by=["layer.0"]),
            SimpleNamespace(shared_by=["layer.1"]),
        ]
        with caplog.at_level("INFO"):
            n = DynamicKvSizer.compute_num_blocks(sizer)
        assert "summed over 2 KV input(s) from 2 set(s)" in caplog.text
        # 2 MiB per block, not 4: (35 - 30) GiB / 2 MiB.
        assert n == 5 * 512

    def test_disjoint_sets_are_summed(self, caplog):
        """A target's and a drafter's KV are different tensors; the sum is the
        answer."""
        programs = [_program([HEAD_SPLIT, HEAD_SPLIT], name="0/0")]
        sizer = self._sizer(
            programs=programs, snapshot=self._snapshot([30 * self.GIB] * 4)
        )
        with caplog.at_level("INFO"):
            DynamicKvSizer.compute_num_blocks(sizer)
        assert "summed over 2 KV input(s) from 1 set(s)" in caplog.text
        assert "vllm allocated 2 KV cache tensor(s) for 2 layer(s)" in caplog.text

    def test_a_count_that_matches_neither_reading_is_refused(self):
        programs = [
            _program([HEAD_SPLIT, HEAD_SPLIT], name="0/0"),
            _program([HEAD_SPLIT_S1, HEAD_SPLIT_S1], name="0/1"),
        ]
        sizer = self._sizer(
            programs=programs, snapshot=self._snapshot([30 * self.GIB] * 4)
        )
        sizer.model_runner.kv_cache_config.kv_cache_tensors = [
            SimpleNamespace(shared_by=["layer.0"]),
            SimpleNamespace(shared_by=["layer.1"]),
            SimpleNamespace(shared_by=["layer.2"]),
        ]
        with pytest.raises(RuntimeError, match="neither sum to nor"):
            DynamicKvSizer.compute_num_blocks(sizer)

    def test_programs_that_disagree_on_the_layout_are_refused(self):
        other = _placement((_shard(0, 0, (2, S0, 8, 1, 1024, 128)),))
        programs = [_program([HEAD_SPLIT]), _program([other], name="0/1")]
        with pytest.raises(RuntimeError, match="disagree"):
            DynamicKvSizer.compute_num_blocks(
                self._sizer(programs=programs, snapshot=self._snapshot([0] * 4))
            )


class TestDynamicKvMemorySnapshot:
    """`memory_snapshot` prefers the driver's per-chiplet figures and
    falls back to this process's allocator with the reserve and foreign usage
    added back."""

    DRIVER = {
        "npu.0.chiplet.0.total": 100,
        "npu.0.chiplet.0.used": 40,
        "npu.0.chiplet.1.total": 100,
        "npu.0.chiplet.1.used": 10,
    }
    ALLOCATOR = {
        "npu.0.chiplet.0.reserved.current": 30,
        "npu.0.chiplet.1.reserved.current": 5,
    }

    @staticmethod
    def _sizer(foreign=0):
        return SimpleNamespace(foreign_dram_used_bytes=foreign)

    def _rbln(self, *, driver=None, driver_error=None, allocator=None, per_chiplet=100):
        calls = []
        rbln = SimpleNamespace(
            empty_cache=lambda device: calls.append(("empty_cache", device)),
            get_device_properties=lambda device: SimpleNamespace(
                memory_per_chiplet=per_chiplet
            ),
            memory_stats_per_chiplet=lambda device: dict(allocator or {}),
        )
        if driver is not None or driver_error is not None:

            def query(device):
                if driver_error is not None:
                    raise driver_error
                return dict(driver)

            rbln.mem_get_info_per_chiplet = query
        return rbln, calls

    def _snapshot(self, rbln, sizer=None):
        with patch.object(dks.torch, "rbln", rbln, create=True):
            return DynamicKvSizer.memory_snapshot(
                sizer or self._sizer(), torch.device("cpu")
            )

    def test_the_driver_wins_when_it_answers(self):
        rbln, calls = self._rbln(driver=self.DRIVER, allocator=self.ALLOCATOR)
        snapshot, source = self._snapshot(rbln)
        assert source == "driver"
        assert snapshot == {
            (0, 0): dks.ChipletMemory(total=100, used=40),
            (0, 1): dks.ChipletMemory(total=100, used=10),
        }
        assert calls == []

    def test_an_old_driver_falls_back_to_the_allocator(self, caplog):
        rbln, calls = self._rbln(
            driver_error=RuntimeError("does not provide the query"),
            allocator=self.ALLOCATOR,
            per_chiplet=100,
        )
        with caplog.at_level("WARNING"):
            snapshot, source = self._snapshot(rbln, self._sizer(foreign=20))
        assert source == "allocator"
        reserve = dks.DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES
        assert snapshot == {
            (0, 0): dks.ChipletMemory(total=100, used=30 + 10 + reserve),
            (0, 1): dks.ChipletMemory(total=100, used=5 + 10 + reserve),
        }
        # Cached-but-free blocks would otherwise count as reserved.
        assert calls == [("empty_cache", torch.device("cpu"))]
        assert "sizing from this process's allocator" in caplog.text

    def test_a_torch_rbln_without_the_query_falls_back_too(self, caplog):
        rbln, _ = self._rbln(allocator=self.ALLOCATOR)
        with caplog.at_level("WARNING"):
            _, source = self._snapshot(rbln)
        assert source == "allocator"
        assert "no mem_get_info_per_chiplet" in caplog.text


class TestModeResolution:
    """One decision at init: every input is static config or env, so the rest of
    the sizer branches on the mode instead of re-reading them."""

    @staticmethod
    def _mode(**kwargs):
        args = dict(
            use_dynamic_kv=True,
            dry_run=False,
            num_gpu_blocks_override=None,
            compile_skip_reason=None,
        )
        return dks.resolve_mode(**{**args, **kwargs})

    def test_the_flag_off_disables_everything(self):
        assert self._mode(use_dynamic_kv=False) == (dks.DynamicKvMode.DISABLED, None)

    def test_a_skipped_compile_wins_over_the_dry_run_and_the_override(self):
        assert self._mode(
            compile_skip_reason="enforce_eager is set",
            dry_run=True,
            num_gpu_blocks_override=64,
        ) == (dks.DynamicKvMode.INERT, "enforce_eager is set")

    def test_a_dry_run_still_reports_under_an_override(self):
        assert self._mode(dry_run=True, num_gpu_blocks_override=64) == (
            dks.DynamicKvMode.DRY_RUN,
            None,
        )

    def test_an_override_pins_the_count(self):
        assert self._mode(num_gpu_blocks_override=64) == (
            dks.DynamicKvMode.PINNED,
            "--num-gpu-blocks-override=64",
        )

    def test_the_flag_alone_is_active(self):
        assert self._mode() == (dks.DynamicKvMode.ACTIVE, None)


class TestWarmupCapturesPrograms:
    """The programs warm-up builds are the only handle on the KV-holding
    runtimes, so the capture has to wrap exactly the warm-up."""

    def test_off_means_no_capture(self):
        sizer = SimpleNamespace(mode=dks.DynamicKvMode.DISABLED)
        with DynamicKvSizer.capture_programs(sizer) as programs:
            pass
        assert programs is None

    def test_on_opens_torch_rbln_s_scope(self):
        recorded = ["p0", "p1"]

        @contextmanager
        def fake_capture():
            yield recorded

        sizer = SimpleNamespace(mode=dks.DynamicKvMode.ACTIVE)
        with (
            patch.object(
                dks.torch,
                "rbln",
                SimpleNamespace(capture_programs=fake_capture),
                create=True,
            ),
            DynamicKvSizer.capture_programs(sizer) as programs,
        ):
            pass
        assert programs is recorded

    def test_a_dry_run_without_capture_programs_reports_instead_of_refusing(
        self, caplog
    ):
        sizer = SimpleNamespace(mode=dks.DynamicKvMode.DRY_RUN)
        with (
            patch.object(dks.torch, "rbln", SimpleNamespace(), create=True),
            caplog.at_level("WARNING"),
            DynamicKvSizer.capture_programs(sizer) as programs,
        ):
            pass
        assert programs is None
        assert "dry run" in caplog.text

    def test_on_without_capture_programs_refuses(self):
        with (
            patch.object(dks.torch, "rbln", SimpleNamespace(), create=True),
            pytest.raises(RuntimeError, match="capture_programs"),
        ):
            DynamicKvSizer.capture_programs(
                SimpleNamespace(mode=dks.DynamicKvMode.ACTIVE)
            )

    def test_runtimes_are_deduped_across_programs(self):
        shared = object()
        sizer = SimpleNamespace(
            programs=[
                _program([HEAD_SPLIT], runtime=shared),
                _program([HEAD_SPLIT], runtime=shared),
                _program([], runtime=object()),
            ]
        )
        assert len(DynamicKvSizer.collect_runtimes(sizer)) == 2


class TestMaybeShrinkKvCacheForCompile:
    """The shrink decides the compile size and, through the latch, whether the
    resize runs at all: every branch returning the config unchanged turns the
    feature off for that run, so the branch taken and its log are the behaviour.
    """

    ESTIMATED_BLOCKS = 211
    PAGE_SIZE = 1 << 20

    @classmethod
    def _config(cls, num_blocks=None):
        blocks = cls.ESTIMATED_BLOCKS if num_blocks is None else num_blocks
        return SimpleNamespace(
            num_blocks=blocks,
            kv_cache_tensors=[
                SimpleNamespace(size=blocks * cls.PAGE_SIZE, shared_by=["layer.0"]),
                SimpleNamespace(size=blocks * cls.PAGE_SIZE, shared_by=["layer.1"]),
            ],
        )

    @staticmethod
    def _shrink(
        config, *, dynamic=True, override=None, warmup_skipped=False, dry_run=False
    ):
        mode, reason = dks.resolve_mode(
            use_dynamic_kv=dynamic,
            dry_run=dry_run,
            num_gpu_blocks_override=override,
            compile_skip_reason="enforce_eager is set" if warmup_skipped else None,
        )
        sizer = SimpleNamespace(
            mode=mode,
            mode_reason=reason,
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            kv_blocks_before_shrink=None,
        )
        out = DynamicKvSizer.shrink_for_compile(sizer, config)
        return sizer, out

    def test_the_flag_alone_shrinks_to_the_constant(self, caplog):
        config = self._config()
        with caplog.at_level("INFO"):
            sizer, out = self._shrink(config)

        assert out is not config
        assert out.num_blocks == dks.COMPILE_KV_CACHE_NUM_BLOCKS
        assert sizer.kv_blocks_before_shrink == self.ESTIMATED_BLOCKS
        # The tensors have to shrink with num_blocks or the allocation and the
        # config disagree.
        for kv_tensor in out.kv_cache_tensors:
            assert kv_tensor.size == out.num_blocks * self.PAGE_SIZE
        # The caller's config must survive: it is what the resize restores to.
        assert config.num_blocks == self.ESTIMATED_BLOCKS
        assert all(
            t.size == self.ESTIMATED_BLOCKS * self.PAGE_SIZE
            for t in config.kv_cache_tensors
        )

    def test_the_flag_off_returns_the_config_untouched_and_silently(self, caplog):
        config = self._config()
        with caplog.at_level("WARNING"):
            sizer, out = self._shrink(config, dynamic=False)
        assert out is config
        assert sizer.kv_blocks_before_shrink is None
        assert "[Dynamic KV]" not in caplog.text

    def test_a_dry_run_compiles_at_the_sized_count(self, caplog):
        config = self._config()
        with caplog.at_level("WARNING"):
            sizer, out = self._shrink(config, dry_run=True)
        assert out is config
        assert sizer.kv_blocks_before_shrink is None
        assert "dry run" in caplog.text

    def test_a_pinned_block_count_cancels_the_shrink(self, caplog):
        config = self._config()
        with caplog.at_level("WARNING"):
            sizer, out = self._shrink(config, override=64)

        assert out is config
        assert sizer.kv_blocks_before_shrink is None
        assert "num-gpu-blocks-override" in caplog.text

    def test_a_hint_that_cannot_shrink_refuses(self):
        # The estimate is free memory over the cost of one block, so a large
        # block_size can legally put it at or below the hint. Serving on there
        # would silently keep the pre-compile estimate, so it is a refusal.
        with pytest.raises(RuntimeError, match="nothing to shrink"):
            self._shrink(self._config(num_blocks=dks.COMPILE_KV_CACHE_NUM_BLOCKS))

    def test_no_warmup_means_no_shrink(self, caplog):
        """Skipping compile/warm-up has to skip the shrink too: otherwise the
        latch is set, the profile query finds no runtimes, and the restore path
        trips an assertion whose message names none of the cause.
        """
        config = self._config()
        with caplog.at_level("WARNING"):
            sizer, out = self._shrink(config, warmup_skipped=True)

        assert out is config
        assert sizer.kv_blocks_before_shrink is None
        assert "compile/warm-up is skipped" in caplog.text
        assert "does nothing for this run" in caplog.text


class TestDynamicKvLayoutGuards:
    """The layout guard is split across `initialize_kv_cache`: the attention half
    runs before it, the binding half after, and neither may drift."""

    @staticmethod
    def _layer(sliding_window=None, is_causal=True, is_normal=False):
        return SimpleNamespace(
            impl=SimpleNamespace(
                sliding_window=sliding_window,
                is_causal=is_causal,
                is_normal=is_normal,
            )
        )

    def test_a_non_paged_causal_layer_is_refused_by_name(self):
        """`block_size == max_model_len` makes is_normal True -- and is also where
        the estimate can fall below the hint, so the wrong refusal could fire."""
        sizer = SimpleNamespace(vllm_config=object(), mode=dks.DynamicKvMode.ACTIVE)
        with (
            patch(
                "vllm_rbln.v1.worker.dynamic_kv_sizer.get_layers_from_vllm_config",
                return_value={"layer.0": self._layer(is_normal=True)},
            ),
            pytest.raises(RuntimeError) as exc,
        ):
            DynamicKvSizer.assert_attention_layout(sizer)
        assert "paged causal or sliding-window naive kernel" in str(exc.value)
        assert "layer.0" in str(exc.value)
        assert "nothing to shrink" not in str(exc.value)

    def test_a_paged_causal_layer_passes(self):
        sizer = SimpleNamespace(vllm_config=object(), mode=dks.DynamicKvMode.ACTIVE)
        with patch(
            "vllm_rbln.v1.worker.dynamic_kv_sizer.get_layers_from_vllm_config",
            return_value={"layer.0": self._layer()},
        ):
            DynamicKvSizer.assert_attention_layout(sizer)

    def test_a_sliding_window_layer_passes(self):
        """gpt-oss alternates full and windowed layers; the compiler admits a
        dynamic KV input on `paged_sliding_window_attention_naive_*` too."""
        sizer = SimpleNamespace(vllm_config=object(), mode=dks.DynamicKvMode.ACTIVE)
        with patch(
            "vllm_rbln.v1.worker.dynamic_kv_sizer.get_layers_from_vllm_config",
            return_value={
                "layer.0": self._layer(),
                "layer.1": self._layer(sliding_window=128),
            },
        ):
            DynamicKvSizer.assert_attention_layout(sizer)

    def test_deduped_bases_pass(self):
        """gpt-oss shares one tensor between a full and a windowed layer; the
        compiler takes the deduped base through both views."""
        sizer = SimpleNamespace(
            mode=dks.DynamicKvMode.ACTIVE,
            model_runner=SimpleNamespace(
                kv_cache_bases=[object()], shared_kv_cache_layers={}
            ),
        )
        DynamicKvSizer.assert_cache_layout(sizer)

    def test_cross_layer_sharing_is_still_refused_after_the_split(self):
        sizer = SimpleNamespace(
            mode=dks.DynamicKvMode.ACTIVE,
            model_runner=SimpleNamespace(
                kv_cache_bases=[], shared_kv_cache_layers={"layer.1": "layer.0"}
            ),
        )
        with pytest.raises(RuntimeError, match="cross-layer KV"):
            DynamicKvSizer.assert_cache_layout(sizer)


class TestReleaseShortfallIsReported:
    """The count is sized from the snapshot after the release, so bytes the
    runtime keeps are charged to the non-KV base and silently cost blocks."""

    @staticmethod
    def _mem(used):
        return {u: dks.ChipletMemory(total=100, used=used[u]) for u in used}

    def test_a_full_release_says_nothing(self, caplog):
        with caplog.at_level("WARNING"):
            DynamicKvSizer.log_release_shortfall(
                self._mem({(0, 0): 50, (0, 1): 50}),
                self._mem({(0, 0): 10, (0, 1): 10}),
                {(0, 0): 40, (0, 1): 40},
                4,
            )
        assert caplog.text == ""

    def test_retained_bytes_are_named_per_chiplet(self, caplog):
        with caplog.at_level("WARNING"):
            DynamicKvSizer.log_release_shortfall(
                self._mem({(0, 0): 50, (0, 1): 50}),
                self._mem({(0, 0): 10, (0, 1): 35}),  # chiplet 1 kept 25
                {(0, 0): 40, (0, 1): 40},
                4,
            )
        assert "the 4-block compile cache holds" in caplog.text
        assert "{'0:1': 25}" in caplog.text


class TestDryRunOnlyObserves:
    """A dry run reports the count it would pick. It must not change whether the
    run boots or what it compiles, or it measures a different run."""

    def test_the_one_request_floor_is_only_for_the_shrunk_estimate(self, monkeypatch):
        """The floor exists because the shrink makes the estimate a placeholder.
        Every other mode serves this estimate, so raising it resizes the pool."""
        spec = SimpleNamespace(max_memory_usage_bytes=lambda cfg: 4000)
        monkeypatch.setattr(dks, "estimate_available_memory", lambda **kw: 999)

        def sizer(mode):
            return SimpleNamespace(
                mode=mode,
                vllm_config=SimpleNamespace(),
                model_runner=SimpleNamespace(
                    get_kv_cache_spec=lambda: {"a": spec, "b": spec}
                ),
            )

        with patch.object(
            dks.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: True),
            create=True,
        ):
            active = DynamicKvSizer.pre_compile_estimate(
                sizer(dks.DynamicKvMode.ACTIVE), {}
            )
            assert active == 8000
            for mode in (
                dks.DynamicKvMode.DRY_RUN,
                dks.DynamicKvMode.PINNED,
                dks.DynamicKvMode.INERT,
            ):
                assert DynamicKvSizer.pre_compile_estimate(sizer(mode), {}) == 999

    def test_the_flag_off_never_snapshots(self, monkeypatch):
        """The DISABLED early return is the only thing keeping the default path
        on the plain estimate instead of the per-chiplet snapshot."""
        seen: list = []
        monkeypatch.setattr(dks, "estimate_available_memory", lambda **kw: 777)
        sizer = SimpleNamespace(
            mode=dks.DynamicKvMode.DISABLED,
            memory_snapshot=lambda device: seen.append(device),
        )
        assert DynamicKvSizer.pre_compile_estimate(sizer, {}) == 777
        assert seen == []

    def test_a_dry_run_reports_a_refused_attention_layout(self, caplog):
        sizer = SimpleNamespace(vllm_config=object(), mode=dks.DynamicKvMode.DRY_RUN)
        layer = SimpleNamespace(impl=SimpleNamespace(is_causal=None, is_normal=True))
        with (
            patch(
                "vllm_rbln.v1.worker.dynamic_kv_sizer.get_layers_from_vllm_config",
                return_value={"layer.0": layer},
            ),
            caplog.at_level("WARNING"),
        ):
            DynamicKvSizer.assert_attention_layout(sizer)
        assert "dry run" in caplog.text
        assert "layer.0" in caplog.text

    def test_a_dry_run_reports_cross_layer_sharing(self, caplog):
        sizer = SimpleNamespace(
            mode=dks.DynamicKvMode.DRY_RUN,
            model_runner=SimpleNamespace(
                kv_cache_bases=[], shared_kv_cache_layers={"layer.1": "layer.0"}
            ),
        )
        with caplog.at_level("WARNING"):
            DynamicKvSizer.assert_cache_layout(sizer)
        assert "dry run" in caplog.text
        assert "cross-layer KV" in caplog.text


class TestDynamicKvFailuresRaise:
    """After the shrink, failing to size from the device must not boot: the run
    would serve the pre-compile estimate. The gates before it stay a quiet None."""

    @staticmethod
    def _sizer(*, shrunk=True, override=None, programs=()):
        mode, reason = dks.resolve_mode(
            use_dynamic_kv=True,
            dry_run=False,
            num_gpu_blocks_override=override,
            compile_skip_reason=None if shrunk or override else "enforce_eager is set",
        )
        sizer = SimpleNamespace(
            rank=0,
            mode=mode,
            mode_reason=reason,
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            kv_blocks_before_shrink=211 if shrunk else None,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(
                    num_blocks=211, kv_cache_tensors=_kv_cache_tensors_for(programs)
                )
            ),
            programs=list(programs),
            release_kv_cache_tensors=lambda cfg: None,
        )
        _bind_sizing(sizer)
        return sizer

    @pytest.fixture(autouse=True)
    def _real_device(self):
        with patch.object(
            dks.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            yield

    def test_no_program_after_the_shrink_raises(self):
        with pytest.raises(RuntimeError, match="none of the 0"):
            DynamicKvSizer.compute_num_blocks(self._sizer(programs=()))

    def test_no_dynamic_input_after_the_shrink_raises(self):
        """Every program static is the documented replayed-static-build case.

        It used to log an error and boot on the estimate, which is the bug this
        feature exists to remove.
        """
        with pytest.raises(RuntimeError) as exc:
            DynamicKvSizer.compute_num_blocks(
                self._sizer(programs=(_program([]), _program([], name="0/1")))
            )
        assert "dynamic-shape KV input" in str(exc.value)
        assert "VLLM_CACHE_ROOT" in str(exc.value)

    def test_the_pre_shrink_gates_still_return_none(self):
        """An override and "not shrunk" are legitimate: nothing moved."""
        assert DynamicKvSizer.compute_num_blocks(self._sizer(override=64)) is None
        # The shrink did not happen, so there is nothing to size from.
        assert DynamicKvSizer.compute_num_blocks(self._sizer(shrunk=False)) is None


class TestKvCopyStreamReserve:
    """The reserve follows the scheduler's own sub-block prefix caching predicate."""

    @staticmethod
    def _sizer(*, prefix_caching=True, sub_block_cache=True, sub_block_size=0):
        return SimpleNamespace(
            cache_config=SimpleNamespace(
                enable_prefix_caching=prefix_caching, block_size=1024
            ),
            vllm_config=SimpleNamespace(
                additional_config=SimpleNamespace(
                    enable_sub_block_cache=sub_block_cache,
                    sub_block_size=sub_block_size,
                )
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
            model_runner=SimpleNamespace(kv_cache_config=SimpleNamespace()),
        )

    @staticmethod
    def _manager(monkeypatch, eligible):
        fake = SimpleNamespace(
            RBLNKVCacheManager=SimpleNamespace(
                can_use_sub_block_caching=lambda cfg, sub_block_size: eligible
            )
        )
        monkeypatch.setitem(
            sys.modules, "vllm_rbln.v1.core.rbln_kv_cache_manager", fake
        )

    def test_reserved_when_the_scheduler_would_sub_block_cache(self, monkeypatch):
        self._manager(monkeypatch, eligible=True)
        assert (
            DynamicKvSizer.copy_stream_reserve_bytes(self._sizer())
            == dks.DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES
        )

    def test_the_configured_size_reaches_the_eligibility_check(self, monkeypatch, cr13):
        # Not the prefill chunk: RBLNConfig.sub_block_size decouples the two.
        seen: list[int] = []

        def can_use(cfg, sub_block_size):
            seen.append(sub_block_size)
            return True

        monkeypatch.setitem(
            sys.modules,
            "vllm_rbln.v1.core.rbln_kv_cache_manager",
            SimpleNamespace(
                RBLNKVCacheManager=SimpleNamespace(can_use_sub_block_caching=can_use)
            ),
        )
        DynamicKvSizer.copy_stream_reserve_bytes(self._sizer(sub_block_size=64))
        assert seen == [64]

    def test_nothing_without_prefix_caching(self, monkeypatch):
        self._manager(monkeypatch, eligible=True)
        sizer = self._sizer(prefix_caching=False)
        assert DynamicKvSizer.copy_stream_reserve_bytes(sizer) == 0

    def test_nothing_when_sub_block_cache_is_off(self, monkeypatch):
        self._manager(monkeypatch, eligible=True)
        sizer = self._sizer(sub_block_cache=False)
        assert DynamicKvSizer.copy_stream_reserve_bytes(sizer) == 0

    def test_nothing_when_the_config_is_ineligible(self, monkeypatch):
        self._manager(monkeypatch, eligible=False)
        assert DynamicKvSizer.copy_stream_reserve_bytes(self._sizer()) == 0


class TestApplyResizesThenMaterializes:
    """`apply_num_blocks` settles the latch, and any actual resize
    must be followed by the boot-time materialization: without it the first
    request pays the whole pool's physical allocation (measured 19.8 s TTFT)."""

    @staticmethod
    def _sizer(*, before_shrink=211, current=4):
        calls: list = []
        sizer = SimpleNamespace(
            kv_blocks_before_shrink=before_shrink,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=current),
                kv_caches=[object()],
            ),
            reallocate=lambda target: calls.append(("realloc", target)),
            materialize=lambda: calls.append(("materialize",)),
            expected_used={},
            log_fit_check=lambda n: calls.append(("check", n)),
        )
        return sizer, calls

    def test_a_computed_count_reallocates_then_materializes(self):
        sizer, calls = self._sizer()
        assert DynamicKvSizer.apply_num_blocks(sizer, 1368) == 1368
        assert calls == [("realloc", 1368), ("materialize",)]
        assert sizer.kv_blocks_before_shrink is None

    def test_a_computed_count_is_checked_against_the_prediction(self):
        sizer, calls = self._sizer()
        sizer.expected_used = {(0, 0): 123}
        assert DynamicKvSizer.apply_num_blocks(sizer, 1368) == 1368
        assert calls == [("realloc", 1368), ("materialize",), ("check", 1368)]

    def test_the_fit_check_reports_measured_against_expected(self, caplog):
        snapshot = {(0, 0): dks.ChipletMemory(total=1000, used=460)}
        sizer = SimpleNamespace(
            device=torch.device("cpu"),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
            expected_used={(0, 0): 450, (0, 1): 7},
            memory_snapshot=lambda device: (snapshot, "driver"),
        )
        with caplog.at_level("INFO"):
            DynamicKvSizer.log_fit_check(sizer, 58)
        assert "0:0(expected=450 measured=460 diff=+10 budget_left=+40)" in caplog.text
        assert "0:1(expected=7 measured=?)" in caplog.text

    def test_none_restores_the_pre_shrink_count(self):
        sizer, calls = self._sizer()
        with patch.object(
            dks.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            assert DynamicKvSizer.apply_num_blocks(sizer, None) == 211
        assert calls == [("realloc", 211), ("materialize",)]

    def test_none_on_a_dummy_device_keeps_the_compile_cache(self):
        sizer, calls = self._sizer()
        with patch.object(
            dks.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: True),
            create=True,
        ):
            assert DynamicKvSizer.apply_num_blocks(sizer, None) == 4
        assert calls == []
        assert sizer.kv_blocks_before_shrink is None

    def test_a_matching_count_skips_both(self):
        sizer, calls = self._sizer(before_shrink=4, current=4)
        assert DynamicKvSizer.apply_num_blocks(sizer, 4) == 4
        assert calls == []

    def test_a_matching_count_still_reallocates_a_released_cache(self):
        # Release-first sizing leaves the sizer with no KV cache; landing on
        # the compile hint must not skip the realloc, or the first forward has
        # nothing bound.
        sizer, calls = self._sizer(before_shrink=4, current=4)
        sizer.model_runner.kv_caches = []
        assert DynamicKvSizer.apply_num_blocks(sizer, 4) == 4
        assert calls == [("realloc", 4), ("materialize",)]

    def test_nothing_pending_returns_none(self):
        sizer, calls = self._sizer(before_shrink=None)
        assert DynamicKvSizer.apply_num_blocks(sizer, None) is None
        assert calls == []

    def test_materialize_runs_every_model_graph(self):
        ran: list = []
        sizer = SimpleNamespace(
            mode=dks.DynamicKvMode.ACTIVE,
            model_runner=SimpleNamespace(
                offload_context=nullcontext,
                run_model_graphs=lambda: ran.append("graphs"),
            ),
        )
        DynamicKvSizer.materialize(sizer)
        assert ran == ["graphs"]


class TestReleaseKvCacheTensors:
    def test_it_clears_every_piece_of_the_rebound_state(self, monkeypatch):
        # The rebind reassigns these together from one ordered name list, so a
        # piece left behind describes a cache that no longer exists.
        layer = SimpleNamespace(kv_cache=torch.zeros(1))
        model_runner = SimpleNamespace(
            kv_caches=[torch.zeros(1)],
            kv_cache_bases=[torch.zeros(1)],
            kv_cache_names=["l0"],
            kv_cache_block_axes={"l0": 1},
            compilation_config=SimpleNamespace(static_forward_context={"l0": layer}),
        )
        sizer = SimpleNamespace(
            model_runner=model_runner,
            allocator_state_per_chiplet=lambda: "stub",
        )
        old_cfg = SimpleNamespace(
            num_blocks=4,
            kv_cache_tensors=[SimpleNamespace(shared_by=["l0"], size=8)],
        )
        monkeypatch.setattr(dks, "empty_rbln_device_caches", lambda: False)

        DynamicKvSizer.release_kv_cache_tensors(sizer, old_cfg)

        assert model_runner.kv_caches == []
        assert model_runner.kv_cache_bases == []
        assert model_runner.kv_cache_names == []
        assert model_runner.kv_cache_block_axes == {}
        assert layer.kv_cache is None
