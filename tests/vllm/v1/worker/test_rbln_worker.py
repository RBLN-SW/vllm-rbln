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

# RBLNWorker: device-id selection, quantization-aware memory sizing, and
# compile/warmup control flow, reachable on CPU once WorkerBase.__init__ is
# patched out. Device execution stays in the e2e tier.

import inspect
import json
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm.platforms.interface as platform_interface
from torch._dynamo.exc import BackendCompilerFailed
from vllm.config import ProfilerConfig, get_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_rbln.v1.worker.dynamic_kv_sizer as dks
import vllm_rbln.v1.worker.rbln_worker as wm
import vllm_rbln.v1.worker.utils as worker_utils
from vllm_rbln.config import RBLNConfig
from vllm_rbln.platform import RblnPlatform
from vllm_rbln.v1.worker.rbln_worker import (
    RBLNWorker,
    init_worker_distributed_environment,
)


def _make_vllm_config(
    *,
    world_size=1,
    data_parallel_size=1,
    data_parallel_rank=0,
    data_parallel_rank_local=None,
    world_size_across_dp=1,
    assigned_physical_gpu_ids=None,
    quantization=None,
    enforce_eager=False,
    profiler=None,
    additional_config=None,
    backend="uni",
):
    return SimpleNamespace(
        profiler_config=SimpleNamespace(profiler=profiler),
        parallel_config=SimpleNamespace(
            distributed_executor_backend=backend,
            rank=0,
            world_size=world_size,
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_rank_local=data_parallel_rank_local,
            world_size_across_dp=world_size_across_dp,
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
            disable_custom_all_reduce=False,
        ),
        model_config=SimpleNamespace(
            quantization=quantization, enforce_eager=enforce_eager
        ),
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            num_gpu_blocks=None,
            num_gpu_blocks_override=None,
        ),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        additional_config=(
            additional_config if additional_config is not None else RBLNConfig()
        ),
    )


def _fake_super_init(
    self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False
):
    # Stand-in for WorkerBase.__init__ that skips real device setup.
    self.vllm_config = vllm_config
    self.local_rank = local_rank
    self.rank = rank
    self.distributed_init_method = distributed_init_method
    self.is_driver_worker = is_driver_worker
    self.model_config = vllm_config.model_config
    self.parallel_config = vllm_config.parallel_config
    self.cache_config = vllm_config.cache_config
    self.scheduler_config = vllm_config.scheduler_config
    self.device_config = vllm_config.device_config


@pytest.fixture(autouse=True)
def _env_cleanup(monkeypatch):
    # Save/restore the process env vars the worker touches.
    keys = [
        "RBLN_VISIBLE_DEVICES",
        "RBLN_NPUS_PER_DEVICE",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "RCCL_PORT_GEN",
        "RBLN_NUM_THREADS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _attach_sizer(worker):
    """`init_device` builds the sizer once the model runner exists; these tests
    construct the worker directly, so they have to do the same."""
    worker.dynamic_kv = dks.DynamicKvSizer(worker.vllm_config, worker.model_runner, 0)
    return worker


@pytest.fixture
def make_worker(monkeypatch):
    def _make(
        *,
        local_rank=0,
        rank=0,
        world_size=1,
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_rank_local=None,
        world_size_across_dp=None,
        assigned_physical_gpu_ids=None,
        num_devices=1,
        device_name="RBLN-CA25",
        vllm_config=None,
    ):
        # vLLM leaves dp_size out of world_size (it sizes one DP replica) and
        # multiplies it back in for world_size_across_dp.
        wsd = (
            world_size * data_parallel_size
            if world_size_across_dp is None
            else world_size_across_dp
        )
        vllm_config = vllm_config or _make_vllm_config(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_rank_local=data_parallel_rank_local,
            world_size_across_dp=wsd,
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
            additional_config=RBLNConfig(num_devices_per_local_rank=num_devices),
        )
        # MultiprocExecutor.worker_main publishes the mapping in every worker
        # process before the worker is built, so a test that supplies one must
        # too or it exercises a path the engine never takes.
        if assigned_physical_gpu_ids is not None:
            monkeypatch.setattr(
                platform_interface,
                "_assigned_physical_gpu_ids",
                assigned_physical_gpu_ids,
            )
        monkeypatch.setattr(WorkerBase, "__init__", _fake_super_init)
        monkeypatch.setattr(
            wm,
            "current_platform",
            SimpleNamespace(
                device_type="cpu",
                # Not a literal: the resolver below reads it off RblnPlatform,
                # so the two must not drift apart.
                device_control_env_var=RblnPlatform.device_control_env_var,
                dist_backend="gloo",
                get_device_name=lambda: device_name,
                # Both real resolvers: the pool semantics under test are
                # upstream's, and which of the two the worker picks is the
                # difference between reading the pool and reading the mapping.
                visible_device_id_to_physical_device_id=(
                    RblnPlatform.visible_device_id_to_physical_device_id
                ),
                device_id_to_physical_device_id=(
                    RblnPlatform.device_id_to_physical_device_id
                ),
            ),
        )
        return RBLNWorker(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=True,
        )

    return _make


class CustomMultiprocExecutor(MultiprocExecutor):
    pass


class TestWorkerFailFast:
    @pytest.mark.parametrize(
        ("backend", "disabled", "should_exit"),
        [("mp", "0", True), ("mp", "1", False), ("uni", "0", False)],
    )
    @pytest.mark.parametrize(
        "phase",
        [
            "init_device",
            "load_model",
            "determine_available_memory",
            "get_kv_cache_spec",
            "initialize_from_config",
            "compile_or_warm_up_model",
        ],
    )
    def test_startup_failure_uses_executor_policy(
        self, make_worker, monkeypatch, capfd, backend, disabled, should_exit, phase
    ):
        worker = make_worker(vllm_config=_make_vllm_config(backend=backend), rank=1)
        error = RuntimeError("startup operation failed")
        operation = Mock(side_effect=error)
        worker.model_runner = Mock()
        args: tuple[object, ...] = ()
        if phase == "init_device":
            monkeypatch.setattr(wm, "init_worker_distributed_environment", operation)
        elif phase == "load_model":
            monkeypatch.setattr(wm, "set_current_vllm_config", lambda _: nullcontext())
            worker.model_runner.load_model = operation
        elif phase == "determine_available_memory":
            worker.model_runner.model.named_parameters = operation
        elif phase == "get_kv_cache_spec":
            worker.model_runner.get_kv_cache_spec = operation
        elif phase == "initialize_from_config":
            monkeypatch.setattr(wm, "ensure_kv_transfer_initialized", operation)
            args = (SimpleNamespace(num_blocks=1),)
        else:
            monkeypatch.setattr(
                worker, "_ensure_rbln_host_threads_before_compile", Mock()
            )
            monkeypatch.setattr(
                worker, "_ensure_rbln_cpu_affinity_after_warmup", Mock()
            )
            monkeypatch.setattr(wm, "compile_and_warmup_skip_reason", lambda _: None)
            worker.dynamic_kv = dks.DynamicKvSizer(
                worker.vllm_config, worker.model_runner, 0
            )
            worker.model_runner.warmup_model = operation
        monkeypatch.setenv("VLLM_RBLN_DISABLE_WORKER_FAIL_FAST", disabled)
        exit_process = Mock(side_effect=SystemExit(70))
        monkeypatch.setattr(worker_utils.os, "_exit", exit_process)

        with pytest.raises(SystemExit if should_exit else RuntimeError) as excinfo:
            getattr(worker, phase)(*args)

        operation.assert_called_once()
        if should_exit:
            exit_process.assert_called_once_with(70)
            events = [
                json.loads(line)
                for line in capfd.readouterr().err.splitlines()
                if line.startswith('{"event":"rbln.worker.fatal"')
            ]
            assert len(events) == 1
            assert events[0]["where"] == f"RBLNWorker.{phase}"
            assert events[0]["rank"] == 1
            assert events[0]["exception_message"] == str(error)
        else:
            exit_process.assert_not_called()
            assert excinfo.value is error

    @pytest.mark.parametrize(
        ("backend", "ray_v2", "disabled", "should_exit"),
        [
            pytest.param("mp", None, "0", True, id="mp"),
            pytest.param("mp", None, "1", False, id="disabled"),
            pytest.param("uni", None, "0", False, id="uni"),
            pytest.param("external_launcher", None, "0", False, id="external"),
            pytest.param(CustomMultiprocExecutor, None, "0", True, id="custom-class"),
            pytest.param(
                f"{__name__}.CustomMultiprocExecutor",
                None,
                "0",
                True,
                id="custom-qualname",
            ),
            pytest.param("ray", None, "0", True, id="ray-default"),
            pytest.param("ray", "0", "0", False, id="ray-legacy"),
        ],
    )
    @pytest.mark.parametrize(
        ("method", "runner_method", "args"),
        [
            (
                "execute_model",
                "execute_model",
                (SimpleNamespace(total_num_scheduled_tokens=0),),
            ),
            ("sample_tokens", "sample_tokens", (None,)),
            ("execute_dummy_batch", "_dummy_run", ()),
        ],
    )
    def test_step_failure_uses_executor_policy(
        self,
        make_worker,
        monkeypatch,
        backend,
        ray_v2,
        disabled,
        should_exit,
        method,
        runner_method,
        args,
    ):
        monkeypatch.delenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", raising=False)
        if ray_v2 is not None:
            monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", ray_v2)
        if backend == "ray" and ray_v2 is None:
            # Ray is optional. Substitute its module, keeping Executor.get_class
            # and the worker's classification and exception handling real.
            monkeypatch.setitem(
                sys.modules,
                "vllm.v1.executor.ray_executor_v2",
                SimpleNamespace(RayExecutorV2=CustomMultiprocExecutor),
            )
        worker = make_worker(vllm_config=_make_vllm_config(backend=backend))
        error = RuntimeError("worker step failed")
        step = Mock(side_effect=error)
        worker.model_runner = SimpleNamespace(**{runner_method: step})
        monkeypatch.setenv("VLLM_RBLN_DISABLE_WORKER_FAIL_FAST", disabled)
        exit_process = Mock(side_effect=SystemExit(70))
        monkeypatch.setattr(worker_utils.os, "_exit", exit_process)

        with pytest.raises(SystemExit if should_exit else RuntimeError) as excinfo:
            getattr(worker, method)(*args)

        step.assert_called_once()
        if should_exit:
            exit_process.assert_called_once_with(70)
            assert isinstance(excinfo.value, SystemExit)
            assert excinfo.value.code == 70
        else:
            exit_process.assert_not_called()
            assert excinfo.value is error


class TestConformance:
    def test_extends_worker_base(self):
        assert issubclass(RBLNWorker, WorkerBase)

    def test_key_override_signatures_match_base(self):
        # The overrides stay call-compatible with WorkerBase so the engine can
        # drive RBLNWorker like any other. load_model is excluded (see below).
        for name in (
            "__init__",
            "execute_model",
            "get_kv_cache_spec",
            "compile_or_warm_up_model",
        ):
            base = inspect.signature(getattr(WorkerBase, name))
            override = inspect.signature(getattr(RBLNWorker, name))
            assert list(override.parameters) == list(base.parameters), name

    def test_every_not_implemented_method_is_overridden(self):
        # The list above cannot see a newly added method, so walk WorkerBase:
        # anything left raising NotImplementedError is implemented or a listed gap.
        expected_gaps = {
            # LoRA is rejected outright in check_and_update_config.
            "add_lora",
            "list_loras",
            "pin_lora",
            "remove_lora",
            # RBLN sizes the KV cache in determine_available_memory instead.
            "get_cache_block_size_bytes",
        }
        raising = {
            name
            for name, fn in inspect.getmembers(WorkerBase, inspect.isfunction)
            if not name.startswith("_")
            and "NotImplementedError" in inspect.getsource(fn)
        }
        assert raising, "no NotImplementedError methods found; did WorkerBase move?"

        missing = sorted(
            name
            for name in raising - expected_gaps
            if getattr(RBLNWorker, name) is getattr(WorkerBase, name)
        )
        assert missing == [], f"RBLNWorker does not override: {missing}"

    def test_load_model_signature_diverges(self):
        # TODO(RBLN): load_model(self) drops WorkerBase's `load_dummy_weights`,
        # so a generic load_model(load_dummy_weights=...) call would break.
        # Pinned until the divergence is confirmed intentional.
        base = list(inspect.signature(WorkerBase.load_model).parameters)
        override = list(inspect.signature(RBLNWorker.load_model).parameters)
        assert "load_dummy_weights" in base
        assert override == ["self"]


class TestInitDeviceEnv:
    """The env var is a pool of NPUs to index into, one entry per NPU.

    A rank takes ``VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK`` consecutive entries
    starting at its rank slot times that count. The slot spans the whole
    deployment: ``data_parallel_rank_local * world_size + local_rank``.
    """

    def test_unset_pool_is_the_whole_host(self, make_worker):
        make_worker(world_size=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "0"

    @pytest.mark.parametrize("local_rank, expected", [(0, "0"), (1, "1")])
    def test_unset_pool_indexes_by_local_rank(self, make_worker, local_rank, expected):
        make_worker(world_size=4, local_rank=local_rank)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected

    def test_pool_indexes_by_local_rank(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = "4,5,6,7"
        make_worker(world_size=4, local_rank=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "5"

    def test_pool_larger_than_needed_is_allowed(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = "0,1,2,3"
        make_worker(world_size=2, local_rank=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "1"

    def test_exported_but_empty_pool_means_no_restriction(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = ""
        make_worker(world_size=2, local_rank=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "1"

    def test_trailing_separator_tolerated(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = "3,"
        make_worker(world_size=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "3"

    def test_pool_too_small_names_the_pool(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = "0,1"
        with pytest.raises(ValueError, match="RBLN_VISIBLE_DEVICES='0,1'"):
            make_worker(world_size=4, local_rank=2)

    def test_non_integer_entry_raises(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = "a,b,c,d"
        with pytest.raises(ValueError):
            make_worker(world_size=4, local_rank=0)

    @pytest.mark.parametrize(
        "pool, world_size, num_devices, local_rank, expected",
        [
            ("1,2", 1, 2, 0, "1,2"),
            ("0,1,2,3,4,5,6,7", 2, 4, 0, "0,1,2,3"),
            ("0,1,2,3,4,5,6,7", 2, 4, 1, "4,5,6,7"),
            ("4,5,6,7", 2, 2, 1, "6,7"),
        ],
    )
    def test_rsd_takes_consecutive_entries(
        self, make_worker, pool, world_size, num_devices, local_rank, expected
    ):
        # A rank's group is enumerated, not derived by multiplying its entry,
        # which reached past the pool the job was given.
        os.environ["RBLN_VISIBLE_DEVICES"] = pool
        make_worker(
            world_size=world_size, num_devices=num_devices, local_rank=local_rank
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected

    @pytest.mark.parametrize("local_rank, expected", [(0, "0,1"), (1, "2,3")])
    def test_unset_pool_with_rsd(self, make_worker, local_rank, expected):
        make_worker(world_size=2, num_devices=2, local_rank=local_rank)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected
        assert os.environ["RBLN_NPUS_PER_DEVICE"] == "2"

    def test_single_device_skips_npus(self, make_worker):
        make_worker(world_size=1, num_devices=1)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    @pytest.mark.parametrize(
        "dp_rank, local_rank, expected",
        [(0, 0, "0"), (0, 1, "1"), (1, 0, "2"), (1, 1, "3")],
    )
    def test_dp_ranks_do_not_share(self, make_worker, dp_rank, local_rank, expected):
        make_worker(
            world_size=2,
            data_parallel_size=2,
            data_parallel_rank=dp_rank,
            data_parallel_rank_local=dp_rank,
            local_rank=local_rank,
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected

    @pytest.mark.parametrize(
        "dp_rank, local_rank, expected",
        [
            (0, 0, "0,1,2,3"),
            (0, 1, "4,5,6,7"),
            (1, 0, "8,9,10,11"),
            (1, 1, "12,13,14,15"),
        ],
    )
    def test_dp_ranks_do_not_share_with_rsd(
        self, make_worker, dp_rank, local_rank, expected
    ):
        make_worker(
            world_size=2,
            data_parallel_size=2,
            data_parallel_rank=dp_rank,
            data_parallel_rank_local=dp_rank,
            # What vLLM's slicer hands this DP rank: one entry per rank.
            assigned_physical_gpu_ids=[dp_rank * 2, dp_rank * 2 + 1],
            num_devices=4,
            local_rank=local_rank,
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected

    def test_dp_with_rsd_indexes_the_pool(self, make_worker):
        os.environ["RBLN_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(16, 32))
        make_worker(
            world_size=2,
            data_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_rank_local=1,
            num_devices=4,
            local_rank=1,
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "28,29,30,31"

    def test_non_moe_dp_offsets_by_the_local_rank(self, make_worker):
        # vLLM treats non-MoE DP ranks as independent engines: it resets
        # data_parallel_size and data_parallel_rank, and only
        # data_parallel_rank_local still tells the replicas apart.
        make_worker(
            world_size=2,
            data_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_rank_local=1,
            num_devices=2,
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "4,5"

    def test_device_ids_mapping_is_ignored(self, make_worker):
        # --device-ids leaves one entry per rank on the config, which cannot
        # express rsd, so the pool position decides instead.
        os.environ["RBLN_VISIBLE_DEVICES"] = "0,1,2,3"
        make_worker(world_size=2, assigned_physical_gpu_ids=[2, 3], local_rank=1)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "1"


def _params():
    # 100 float16 weights + 50 int8 (quantized) weights.
    return {
        "w": torch.zeros(100, dtype=torch.float16),
        "qw": torch.zeros(50, dtype=torch.int8),
    }


class TestDetermineAvailableMemory:
    # Isolates the worker's own arithmetic by capturing the kwargs it hands to
    # the already-tested estimate_available_memory. Golden values are _params().
    @staticmethod
    def _capture(
        make_worker,
        monkeypatch,
        *,
        quantization=None,
        device_name="RBLN-CA25",
        hf_config=None,
        params=None,
        specialized_moe_decode=False,
        uses_fixed_decode_window=False,
        decode_buckets=3,
        drafter=None,
        speculative_config=None,
    ):
        vcfg = _make_vllm_config(quantization=quantization)
        vcfg.model_config.hf_config = hf_config
        worker = make_worker(vllm_config=vcfg, device_name=device_name)
        worker.device = torch.device("cpu")
        captured: dict = {}

        def record(**kw):
            # The dry run calls twice; the last call is the one whose result counts.
            captured.clear()
            captured.update(kw)
            return 999

        # The worker hands the kwargs to the sizer, which is where the formula
        # is called from on both the dynamic and the default path.
        monkeypatch.setattr(dks, "estimate_available_memory", record)
        monkeypatch.setattr(wm, "estimate_model_kernel_size", lambda **kw: 111)
        # WorkerBase always carries the field; None is what no spec decode means.
        worker.speculative_config = speculative_config
        worker.model_runner = SimpleNamespace(
            model=SimpleNamespace(
                named_parameters=lambda: iter((params or _params()).items())
            ),
            specialized_moe_decode=specialized_moe_decode,
            bucketing_manager=SimpleNamespace(
                decode_batch_buckets_count=decode_buckets
            ),
            drafter=drafter,
            get_kv_cache_spec=lambda: {},
            uses_fixed_decode_window=uses_fixed_decode_window,
        )
        _attach_sizer(worker)
        worker.determine_available_memory()
        return captured

    def test_dynamic_kv_feeds_the_chiplet_snapshot(self, make_worker, monkeypatch):
        snapshot = {(0, 0): dks.ChipletMemory(total=100, used=40)}
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE", True
        )
        monkeypatch.setattr(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            raising=False,
        )
        monkeypatch.setattr(
            dks.DynamicKvSizer,
            "memory_snapshot",
            lambda self, device: (snapshot, "driver"),
        )
        cap = self._capture(make_worker, monkeypatch, device_name="RBLN-CR13")
        assert cap["chiplet_memory"] is snapshot

    def test_dynamic_kv_raises_a_short_estimate_to_one_request(
        self, make_worker, monkeypatch, caplog
    ):
        # vllm refuses a pool below one max-length request against the
        # estimate; under the flag the estimate is only the compile placeholder.
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE", True
        )
        monkeypatch.setattr(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: True),
            raising=False,
        )
        vcfg = _make_vllm_config()
        worker = make_worker(vllm_config=vcfg, device_name="RBLN-CR13")
        worker.device = torch.device("cpu")
        monkeypatch.setattr(dks, "estimate_available_memory", lambda **kw: 999)
        monkeypatch.setattr(wm, "estimate_model_kernel_size", lambda **kw: 111)
        worker.speculative_config = None
        spec = SimpleNamespace(max_memory_usage_bytes=lambda cfg: 4000)
        worker.model_runner = SimpleNamespace(
            model=SimpleNamespace(named_parameters=lambda: iter(_params().items())),
            specialized_moe_decode=False,
            bucketing_manager=SimpleNamespace(decode_batch_buckets_count=3),
            drafter=None,
            get_kv_cache_spec=lambda: {"a": spec, "b": spec},
        )
        _attach_sizer(worker)
        with caplog.at_level("WARNING"):
            assert worker.determine_available_memory() == 8000
        assert "short of one max-length request" in caplog.text

    def test_dynamic_kv_dry_run_keeps_the_formula(self, make_worker, monkeypatch):
        snapshot = {(0, 0): dks.ChipletMemory(total=100, used=40)}
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE", True
        )
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN",
            True,
        )
        monkeypatch.setattr(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            raising=False,
        )
        monkeypatch.setattr(
            dks.DynamicKvSizer,
            "memory_snapshot",
            lambda self, device: (snapshot, "driver"),
        )
        cap = self._capture(make_worker, monkeypatch, device_name="RBLN-CR13")
        assert "chiplet_memory" not in cap

    def test_dynamic_kv_skips_the_snapshot_on_a_dummy_device(
        self, make_worker, monkeypatch
    ):
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE", True
        )
        monkeypatch.setattr(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: True),
            raising=False,
        )
        cap = self._capture(make_worker, monkeypatch, device_name="RBLN-CR13")
        assert "chiplet_memory" not in cap

    def test_default_path_never_snapshots(self, make_worker, monkeypatch):
        monkeypatch.setattr(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE", False
        )
        cap = self._capture(make_worker, monkeypatch, device_name="RBLN-CR13")
        assert "chiplet_memory" not in cap

    def test_num_runtimes_from_buckets_and_moe(self, make_worker, monkeypatch):
        cap = self._capture(
            make_worker, monkeypatch, specialized_moe_decode=True, decode_buckets=3
        )
        # Non-spec: num_decode_query_lens == 1, so 1 + buckets(3)*1 = 4, plus
        # the specialized-MoE-decode fallback (+1 query length) = 5.
        assert cap["num_runtimes"] == 5

    def test_no_quant_counts_int_at_16bit(self, make_worker, monkeypatch):
        assert self._capture(make_worker, monkeypatch)["n_model_bytes"] == 300

    def test_fp8_counts_int_at_8bit(self, make_worker, monkeypatch):
        assert (
            self._capture(make_worker, monkeypatch, quantization="fp8")["n_model_bytes"]
            == 250
        )

    def test_mxfp4_atom_and_rebel_differ(self, make_worker, monkeypatch):
        atom = self._capture(
            make_worker, monkeypatch, quantization="mxfp4", device_name="RBLN-CA25"
        )
        rebel = self._capture(
            make_worker, monkeypatch, quantization="mxfp4", device_name="RBLN-CR13"
        )
        assert atom["n_model_bytes"] == 388  # bf16 + ratio 16/17, packed 2
        assert rebel["n_model_bytes"] == 250  # 4-bit, packed 2

    def test_mxfp4_unknown_device_raises(self, make_worker, monkeypatch):
        with pytest.raises(ValueError, match="invalid RBLN architecture"):
            self._capture(
                make_worker, monkeypatch, quantization="mxfp4", device_name="RBLN-XX"
            )

    def test_unsupported_quantization_raises(self, make_worker, monkeypatch):
        with pytest.raises(AssertionError):
            self._capture(make_worker, monkeypatch, quantization="bogus")

    def test_compressed_tensors_8bit_as_fp8(self, make_worker, monkeypatch):
        hf = SimpleNamespace(
            quantization_config={"config_groups": {"g": {"weights": {"num_bits": 8}}}}
        )
        cap = self._capture(
            make_worker, monkeypatch, quantization="compressed-tensors", hf_config=hf
        )
        assert cap["n_model_bytes"] == 250

    def test_compressed_tensors_mixed_bits_raises(self, make_worker, monkeypatch):
        hf = SimpleNamespace(
            quantization_config={
                "config_groups": {
                    "a": {"weights": {"num_bits": 8}},
                    "b": {"weights": {"num_bits": 4}},
                }
            }
        )
        with pytest.raises(RuntimeError, match="mixed bit-widths"):
            self._capture(
                make_worker,
                monkeypatch,
                quantization="compressed-tensors",
                hf_config=hf,
            )

    def test_draft_model_adds_kernel_size(self, make_worker, monkeypatch):
        drafter = SimpleNamespace(
            model=SimpleNamespace(
                parameters=lambda: iter([torch.zeros(20, dtype=torch.float16)])
            )
        )
        spec = SimpleNamespace(
            draft_model_config=SimpleNamespace(quantization=None),
            draft_parallel_config=None,
            method="eagle",
        )
        cap = self._capture(
            make_worker,
            monkeypatch,
            drafter=drafter,
            speculative_config=spec,
            uses_fixed_decode_window=True,
        )
        assert "kernel_size" in cap
        assert "n_model_bytes" not in cap
        # A fixed decode window compiles one decode query length: target =
        # 1 + buckets(3)*1 = 4 (no MoE); draft = 1 + buckets(3) = 4. Total 8.
        assert cap["num_runtimes"] == 8

    def test_draft_runtime_adds_specialized_moe_fallback(
        self, make_worker, monkeypatch
    ):
        # The specialized-MoE-decode fallback re-runs the top bucket at a different
        # num_padded_tokens, so it adds one draft graph.
        # Target = 1 + buckets(3)*1 + 1 = 5; draft = 1 + buckets(3) + 1 = 5;
        # total 10.
        drafter = SimpleNamespace(
            model=SimpleNamespace(
                parameters=lambda: iter([torch.zeros(20, dtype=torch.float16)])
            ),
        )
        spec = SimpleNamespace(
            draft_model_config=SimpleNamespace(quantization=None),
            draft_parallel_config=None,
            method="eagle",
        )
        cap = self._capture(
            make_worker,
            monkeypatch,
            drafter=drafter,
            speculative_config=spec,
            specialized_moe_decode=True,
            uses_fixed_decode_window=True,
        )
        assert cap["num_runtimes"] == 10

    def test_draft_quantization_rejected(self, make_worker, monkeypatch):
        drafter = SimpleNamespace(
            model=SimpleNamespace(
                parameters=lambda: iter([torch.zeros(20, dtype=torch.float16)])
            )
        )
        spec = SimpleNamespace(
            draft_model_config=SimpleNamespace(quantization="fp8"),
            draft_parallel_config=None,
            method="eagle",
        )
        with pytest.raises(ValueError, match="draft model quantization"):
            self._capture(
                make_worker,
                monkeypatch,
                drafter=drafter,
                speculative_config=spec,
                uses_fixed_decode_window=True,
            )


class TestInitializeFromConfig:
    def test_sets_num_gpu_blocks(self, make_worker, monkeypatch):
        worker = make_worker()
        monkeypatch.setattr(wm, "ensure_kv_transfer_initialized", lambda *a: None)
        init_calls = []
        worker.model_runner = SimpleNamespace(
            initialize_kv_cache=lambda cfg: init_calls.append(cfg)
        )
        _attach_sizer(worker)
        kv_cfg = SimpleNamespace(num_blocks=123)
        worker.initialize_from_config(kv_cfg)
        assert worker.cache_config.num_gpu_blocks == 123
        assert worker.cache_config.num_cpu_blocks == 123
        assert init_calls == [kv_cfg]


class TestCompileOrWarmUpModel:
    @staticmethod
    def _worker(
        make_worker,
        monkeypatch,
        *,
        enforce_eager=False,
        compile_model=True,
        warm_up=True,
        warmup_side_effect=None,
        data_parallel_size=1,
    ):
        vcfg = _make_vllm_config(
            enforce_eager=enforce_eager,
            data_parallel_size=data_parallel_size,
            additional_config=RBLNConfig(compile_model=compile_model),
        )
        vcfg.model_config.seed = 0
        worker = make_worker(vllm_config=vcfg)
        monkeypatch.setattr(wm.envs, "VLLM_RBLN_ENABLE_WARM_UP", warm_up)
        monkeypatch.setattr(wm, "has_kv_transfer_group", lambda: False)
        monkeypatch.setattr(wm, "set_random_seed", lambda s: None)
        monkeypatch.setattr(
            RBLNWorker, "_ensure_rbln_host_threads_before_compile", lambda self: None
        )
        monkeypatch.setattr(
            RBLNWorker, "_ensure_rbln_cpu_affinity_after_warmup", lambda self: None
        )
        calls = []

        def warmup():
            calls.append("warmup")
            if warmup_side_effect is not None:
                raise warmup_side_effect

        monkeypatch.setattr(wm, "get_dp_group", lambda: SimpleNamespace(cpu_group="dp"))
        monkeypatch.setattr(
            wm.dist, "barrier", lambda group: calls.append(f"barrier:{group}")
        )

        worker.model_runner = SimpleNamespace(
            warmup_model=warmup,
            kv_cache_config=SimpleNamespace(num_blocks=10),
        )
        _attach_sizer(worker)
        return worker, calls

    def test_deferred_kv_registration_runs_with_the_vllm_config_set(
        self, make_worker, monkeypatch
    ):
        # Without a scope of its own the connector raises "Current vLLM config
        # is not set" and kills the rank: warm-up runs after the executor's has
        # closed, and registration reads the config to find the block axis.
        worker, _ = self._worker(make_worker, monkeypatch)
        monkeypatch.setattr(wm, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(wm, "get_kv_transfer_group", lambda: "group")
        seen: list = []
        monkeypatch.setattr(
            wm,
            "finalize_kv_cache_registrations",
            lambda g: seen.append(get_current_vllm_config()),
        )

        worker.compile_or_warm_up_model()

        # Identity, not just "something was set": an outer scope holding a
        # different config would satisfy a bare call.
        assert seen == [worker.vllm_config]

    def test_skips_when_enforce_eager(self, make_worker, monkeypatch):
        worker, calls = self._worker(make_worker, monkeypatch, enforce_eager=True)
        worker.compile_or_warm_up_model()
        assert calls == []

    def test_skips_when_compile_disabled(self, make_worker, monkeypatch):
        worker, calls = self._worker(make_worker, monkeypatch, compile_model=False)
        worker.compile_or_warm_up_model()
        assert calls == []

    def test_skips_when_warmup_disabled(self, make_worker, monkeypatch):
        worker, calls = self._worker(make_worker, monkeypatch, warm_up=False)
        worker.compile_or_warm_up_model()
        assert calls == []

    def test_warmup_called_on_normal_path(self, make_worker, monkeypatch):
        worker, calls = self._worker(make_worker, monkeypatch)
        result = worker.compile_or_warm_up_model()
        assert calls == ["warmup"]
        assert isinstance(result, CompilationTimes)

    def test_dp_ranks_rendezvous_after_warmup(self, make_worker, monkeypatch):
        # The ranks must leave this method together: whatever skew survives it
        # lands in the first forward's DP all-reduce, where it reads as the
        # first request's prefill latency.
        worker, calls = self._worker(make_worker, monkeypatch, data_parallel_size=4)
        worker.compile_or_warm_up_model()
        assert calls == ["warmup", "barrier:dp"]

    def test_no_rendezvous_without_dp_peers(self, make_worker, monkeypatch):
        worker, calls = self._worker(make_worker, monkeypatch, data_parallel_size=1)
        worker.compile_or_warm_up_model()
        assert calls == ["warmup"]

    def test_no_rendezvous_when_warmup_skipped(self, make_worker, monkeypatch):
        # Nothing compiled, so there is no skew to absorb -- and every skip
        # reason is global config, so the ranks skip together.
        worker, calls = self._worker(
            make_worker, monkeypatch, warm_up=False, data_parallel_size=4
        )
        worker.compile_or_warm_up_model()
        assert calls == []

    @pytest.mark.parametrize(
        "msg",
        ["SYS_ENOMEM: Out of memory", "SYS_EBUSY: Lack of device memory"],
    )
    def test_oom_remapped_to_runtime_error(self, make_worker, monkeypatch, msg):
        exc = BackendCompilerFailed(lambda: 0, RuntimeError(msg), None)
        worker, _ = self._worker(make_worker, monkeypatch, warmup_side_effect=exc)
        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_non_oom_backend_error_reraised(self, make_worker, monkeypatch):
        exc = BackendCompilerFailed(lambda: 0, RuntimeError("some other error"), None)
        worker, _ = self._worker(make_worker, monkeypatch, warmup_side_effect=exc)
        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()


class TestEnsureRblnHostThreadsBeforeCompile:
    @pytest.fixture(autouse=True)
    def _restore_torch_threads(self):
        saved = torch.get_num_threads()
        yield
        torch.set_num_threads(saved)

    @staticmethod
    def _prep(monkeypatch, planned):
        captured = []
        monkeypatch.setattr(
            wm, "get_rbln_planned_affinity_cpu_count", lambda r, lr, pc: planned
        )
        monkeypatch.setattr(
            wm, "set_omp_num_threads", lambda r, lr, n: captured.append(n)
        )
        # Neutralise the numba<->torch thread juggling (global-state side effect).
        monkeypatch.setattr(
            wm,
            "numba",
            SimpleNamespace(set_num_threads=lambda n: None, get_num_threads=lambda: 1),
        )
        return captured

    @pytest.mark.parametrize("planned, expected", [(8, 4), (10, 5), (2, 2), (1, 2)])
    def test_thread_count_is_half_planned_min_2(
        self, make_worker, monkeypatch, planned, expected
    ):
        # num_threads = max(2, planned_cpu_count // 2).
        worker = make_worker()
        captured = self._prep(monkeypatch, planned)
        worker._ensure_rbln_host_threads_before_compile()
        assert captured == [expected]

    def test_idempotent(self, make_worker, monkeypatch):
        # The ready-flag guard makes a second call a no-op.
        worker = make_worker()
        captured = self._prep(monkeypatch, 8)
        worker._ensure_rbln_host_threads_before_compile()
        worker._ensure_rbln_host_threads_before_compile()
        assert captured == [4]


class TestEnsureRblnCpuAffinityAfterWarmup:
    def test_applies_affinity_once(self, make_worker, monkeypatch):
        # Applies set_cpu_affinity exactly once (idempotent guard).
        worker = make_worker()
        calls = []
        monkeypatch.setattr(
            wm, "set_cpu_affinity", lambda r, lr, pc: calls.append((r, lr))
        )
        worker._ensure_rbln_cpu_affinity_after_warmup()
        worker._ensure_rbln_cpu_affinity_after_warmup()
        assert calls == [(0, 0)]


class TestInitWorkerDistributedEnvironment:
    @staticmethod
    def _run(
        monkeypatch,
        *,
        rank=1,
        world_size=1,
        dp_size=1,
        dp_rank=0,
        world_size_across_dp=1,
        auto_port=False,
    ):
        monkeypatch.setattr(wm, "init_distributed_environment", lambda *a, **k: None)
        monkeypatch.setattr(
            wm, "ensure_model_parallel_initialized", lambda *a, **k: None
        )
        monkeypatch.setattr(wm, "set_custom_all_reduce", lambda *a, **k: None)
        monkeypatch.setattr(wm.envs, "VLLM_RBLN_AUTO_PORT", auto_port)
        vcfg = SimpleNamespace(
            parallel_config=SimpleNamespace(
                world_size=world_size,
                world_size_across_dp=world_size_across_dp,
                data_parallel_size=dp_size,
                data_parallel_rank=dp_rank,
                tensor_parallel_size=world_size,
                pipeline_parallel_size=1,
                disable_custom_all_reduce=False,
            )
        )
        init_worker_distributed_environment(vcfg, rank=rank, local_rank=rank)
        return (
            os.environ.get("LOCAL_RANK"),
            os.environ.get("WORLD_SIZE"),
            os.environ.get("RCCL_PORT_GEN"),
        )

    def test_single_dp_sets_rank_and_world(self, monkeypatch):
        lr, ws, _ = self._run(monkeypatch, rank=1, world_size=4)
        assert lr == "1"
        assert ws == "4"

    def test_multi_dp_uses_rank_across_dp(self, monkeypatch):
        # dp_rank=1, world_size=2 -> rank_across_dp = 1*2 + rank(1) = 3.
        lr, ws, _ = self._run(
            monkeypatch,
            rank=1,
            world_size=2,
            dp_size=2,
            dp_rank=1,
            world_size_across_dp=4,
        )
        assert lr == "3"
        assert ws == "4"

    def test_auto_port_sets_rccl_env(self, monkeypatch):
        _, _, rccl = self._run(monkeypatch, auto_port=True)
        assert rccl == "1"


class TestHandshakeMetadata:
    # The producer half of vllm's handshake contract: EngineCore merges these
    # per-worker dicts and hands the result to the connector.
    @staticmethod
    def _metadata(
        make_worker,
        monkeypatch,
        *,
        pp_rank=0,
        tp_rank=0,
        metadata="META",
        has_group=True,
    ):
        worker = make_worker()
        monkeypatch.setattr(wm, "has_kv_transfer_group", lambda: has_group)
        monkeypatch.setattr(
            wm,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: metadata),
        )
        monkeypatch.setattr(
            wm, "get_tp_group", lambda: SimpleNamespace(rank_in_group=tp_rank)
        )
        monkeypatch.setattr(
            wm, "get_pp_group", lambda: SimpleNamespace(rank_in_group=pp_rank)
        )
        return worker.get_kv_connector_handshake_metadata()

    def test_key_is_pp_tp_pair(self, make_worker, monkeypatch):
        assert self._metadata(make_worker, monkeypatch, pp_rank=1, tp_rank=2) == {
            (1, 2): "META"
        }

    def test_upstream_hook_takes_the_keys(self, make_worker, monkeypatch):
        # Runs vllm's own implementation over the merged dict, so a contract
        # change upstream fails here instead of at engine-core init.
        merged = self._metadata(make_worker, monkeypatch, tp_rank=1)
        received: dict = {}
        KVConnectorBase_V1.set_xfer_handshake_metadata_pp_aware(
            SimpleNamespace(set_xfer_handshake_metadata=received.update), merged
        )
        assert received == {1: "META"}

    def test_returns_none_without_kv_transfer_group(self, make_worker, monkeypatch):
        assert self._metadata(make_worker, monkeypatch, has_group=False) is None

    def test_returns_none_when_connector_has_no_metadata(
        self, make_worker, monkeypatch
    ):
        assert self._metadata(make_worker, monkeypatch, metadata=None) is None


# ---------------------------------------------------------------------------
# Dynamic KV: sizing from the compiled placement and a memory snapshot
# ---------------------------------------------------------------------------


# 1 MiB per block per chiplet, one node, four chiplets (a head split of 8 KV heads).
PER_BLOCK_PER_CHIPLET = 2**20


class TestDynamicKvLayoutGuards:
    """The attention half of the layout guard runs before the shrink, the
    binding half after `initialize_kv_cache`; neither may drift."""

    def test_the_attention_guard_runs_before_the_shrink(self):
        calls: list[str] = []

        def record(name: str, ret: object = None) -> object:
            calls.append(name)
            return ret

        config = SimpleNamespace(num_blocks=4, kv_cache_tensors=[])
        worker = SimpleNamespace(
            fail_fast=False,
            cache_config=SimpleNamespace(num_gpu_blocks=None, num_cpu_blocks=None),
            vllm_config=object(),
            model_runner=SimpleNamespace(
                initialize_kv_cache=lambda cfg: record("initialize_kv_cache")
            ),
            dynamic_kv=SimpleNamespace(
                assert_attention_layout=lambda: record("attention"),
                assert_cache_layout=lambda: record("bindings"),
                shrink_for_compile=lambda cfg: record("shrink", cfg),
            ),
        )
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                True,
            ),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
        ):
            RBLNWorker.initialize_from_config(worker, config)

        assert calls.index("attention") < calls.index("shrink")
        assert calls.index("bindings") > calls.index("initialize_kv_cache")


class TestDynamicKvBlockCountRpcs:
    def test_both_rpcs_run_with_the_vllm_config_set(self):
        # `get_kv_cache_shape` resolves the KV layout through
        # `get_current_vllm_config()`, and these two land after warm-up has
        # returned -- outside the scope WorkerWrapperBase opens around
        # `initialize_from_config`. Without one of their own, the reallocation
        # raises "Current vLLM config is not set" and kills the worker.
        vllm_config = _make_vllm_config()
        seen: list = []
        worker = SimpleNamespace(
            vllm_config=vllm_config,
            dynamic_kv=SimpleNamespace(
                compute_num_blocks=lambda: seen.append(get_current_vllm_config()),
                apply_num_blocks=lambda n: seen.append(get_current_vllm_config()),
            ),
        )

        RBLNWorker.compute_dynamic_kv_num_blocks(worker)
        RBLNWorker.apply_dynamic_kv_num_blocks(worker, 4)

        # Identity, not just "something was set": an outer scope holding a
        # different config would satisfy a bare call.
        assert seen == [vllm_config, vllm_config]


class TestProfile:
    def test_torch_profiler_keeps_the_rbln_session_to_itself(
        self, make_worker, monkeypatch, tmp_path
    ):
        calls: list[str] = []
        monkeypatch.setattr(
            wm,
            "rbln_profiler",
            SimpleNamespace(
                start=lambda: calls.append("start"),
                done=lambda: calls.append("done"),
            ),
        )
        vllm_config = _make_vllm_config()
        vllm_config.profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_dump_cuda_time_total=False,
        )
        worker = make_worker(vllm_config=vllm_config)

        worker.profile(is_start=True)
        worker.profile(is_start=False)

        assert calls == []

    def test_rbln_profiler_starts_and_flushes_at_stop(self, make_worker, monkeypatch):
        calls: list[str] = []
        monkeypatch.setenv("RBLN_PROFILER", "1")
        monkeypatch.setattr(
            wm,
            "rbln_profiler",
            SimpleNamespace(
                start=lambda: calls.append("start"),
                done=lambda: calls.append("done"),
            ),
        )
        vllm_config = _make_vllm_config()
        vllm_config.profiler_config = ProfilerConfig()
        worker = make_worker(vllm_config=vllm_config)

        worker.profile(is_start=True)
        worker.profile(is_start=False)

        assert calls == ["start", "done"]
        assert isinstance(worker.profiler, wm.RblnProfilerWrapper)
