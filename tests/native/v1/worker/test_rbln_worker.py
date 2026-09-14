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
import os
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm.platforms.interface as platform_interface
from torch._dynamo.exc import BackendCompilerFailed
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_rbln.v1.worker.rbln_worker as wm
import vllm_rbln.v1.worker.utils as worker_utils
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
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9, num_gpu_blocks=None),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        additional_config=additional_config if additional_config is not None else {},
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
        has_torch_rbln=False,
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
        monkeypatch.setattr(
            wm.envs, "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", num_devices
        )
        monkeypatch.setattr(wm, "has_torch_rbln", has_torch_rbln)
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


class TestConfigResolution:
    def test_additional_config_reaches_the_worker(self, make_worker):
        # The worker receives an already-built VllmConfig, so __init__ is the
        # only place the section can be resolved. No env var is involved.
        from vllm_rbln.config import get_rbln_config

        make_worker(vllm_config=_make_vllm_config(additional_config={"sampler": False}))
        assert get_rbln_config().sampler is False


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
        make_worker(
            world_size=2, num_devices=2, has_torch_rbln=True, local_rank=local_rank
        )
        assert os.environ["RBLN_VISIBLE_DEVICES"] == expected
        assert os.environ["RBLN_NPUS_PER_DEVICE"] == "2"

    def test_multi_device_without_torch_rbln_skips_npus(self, make_worker):
        make_worker(world_size=2, num_devices=2, has_torch_rbln=False)
        assert os.environ["RBLN_VISIBLE_DEVICES"] == "0,1"
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_single_device_skips_npus(self, make_worker):
        make_worker(world_size=1, num_devices=1, has_torch_rbln=True)
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

        monkeypatch.setattr(wm, "estimate_available_memory", record)
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
        )
        worker.determine_available_memory()
        return captured

    def test_dynamic_kv_feeds_the_chiplet_snapshot(self, make_worker, monkeypatch):
        snapshot = {(0, 0): wm.ChipletMemory(total=100, used=40)}
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
            wm.RBLNWorker,
            "_dynamic_kv_memory_snapshot",
            lambda self, device: (snapshot, "driver"),
        )
        cap = self._capture(make_worker, monkeypatch, device_name="RBLN-CR13")
        assert cap["chiplet_memory"] is snapshot

    def test_dynamic_kv_dry_run_keeps_the_formula(self, make_worker, monkeypatch):
        snapshot = {(0, 0): wm.ChipletMemory(total=100, used=40)}
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
            wm.RBLNWorker,
            "_dynamic_kv_memory_snapshot",
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
            make_worker, monkeypatch, drafter=drafter, speculative_config=spec
        )
        assert "kernel_size" in cap
        assert "n_model_bytes" not in cap
        # Spec on: target = 1 + buckets(3)*num_decode_query_lens(2) = 7 (no MoE);
        # draft = 1 + buckets(3) = 4. Total 11.
        assert cap["num_runtimes"] == 11

    def test_draft_runtime_adds_specialized_moe_fallback(
        self, make_worker, monkeypatch
    ):
        # The specialized-MoE-decode fallback re-runs the top bucket at a different
        # num_padded_tokens, so it adds one draft graph.
        # Target = 1 + buckets(3)*2 + (2 + 1) = 10; draft = 1 + buckets(3) + 1 = 5;
        # total 15.
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
        )
        assert cap["num_runtimes"] == 15

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
                make_worker, monkeypatch, drafter=drafter, speculative_config=spec
            )


class TestInitializeFromConfig:
    def test_sets_num_gpu_blocks(self, make_worker, monkeypatch):
        worker = make_worker()
        monkeypatch.setattr(wm, "ensure_kv_transfer_initialized", lambda *a: None)
        init_calls = []
        worker.model_runner = SimpleNamespace(
            initialize_kv_cache=lambda cfg: init_calls.append(cfg)
        )
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
            enforce_eager=enforce_eager, data_parallel_size=data_parallel_size
        )
        vcfg.model_config.seed = 0
        worker = make_worker(vllm_config=vcfg)
        monkeypatch.setattr(wm.envs, "VLLM_RBLN_COMPILE_MODEL", compile_model)
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
        return worker, calls

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
        has_torch_rbln=False,
    ):
        monkeypatch.setattr(wm, "init_distributed_environment", lambda *a, **k: None)
        monkeypatch.setattr(
            wm, "ensure_model_parallel_initialized", lambda *a, **k: None
        )
        monkeypatch.setattr(wm, "set_custom_all_reduce", lambda *a, **k: None)
        monkeypatch.setattr(wm.envs, "VLLM_RBLN_AUTO_PORT", auto_port)
        monkeypatch.setattr(wm, "has_torch_rbln", has_torch_rbln)
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
        _, _, rccl = self._run(monkeypatch, auto_port=True, has_torch_rbln=True)
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
S0 = ("symbol", "s0")


def _shard(node, chiplet, shape):
    return SimpleNamespace(node_id=node, chiplet_id=chiplet, slice_shape=tuple(shape))


def _placement(shards):
    return SimpleNamespace(
        shape=(2, S0, 8, 1, 1024, 128), dtype="dlfloat16", shards=shards
    )


# 1 MiB per block per chiplet, one node, four chiplets (a head split of 8 KV heads).
HEAD_SPLIT = _placement(tuple(_shard(0, c, (2, S0, 2, 1, 1024, 128)) for c in range(4)))
PER_BLOCK_PER_CHIPLET = 2**20


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


class TestComputeDynamicKvNumBlocks:
    """`compute_dynamic_kv_num_blocks` = placement slope x memory snapshot.

    The snapshot is stubbed at the worker seam (`_dynamic_kv_memory_snapshot`);
    `TestDynamicKvMemorySnapshot` covers the seam itself.
    """

    GIB = 2**30
    HINT = 4
    TOTAL = 35 * GIB

    @pytest.fixture(autouse=True)
    def _real_device(self):
        with patch.object(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            yield

    def _worker(self, *, programs, snapshot, tp_size=1, gmu=1.0):
        worker = SimpleNamespace(
            rank=0,
            device=torch.device("cpu"),
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None, gpu_memory_utilization=gmu
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
            _kv_blocks_before_shrink=211,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=self.HINT)
            ),
            _dynamic_kv_programs=list(programs),
            _dynamic_kv_memory_snapshot=lambda device: (snapshot, "stub"),
            _kv_copy_stream_reserve_bytes=lambda: 0,
            _release_kv_cache_tensors=lambda cfg: None,
        )
        worker._dynamic_kv_num_blocks_from_placement = lambda **kw: (
            RBLNWorker._dynamic_kv_num_blocks_from_placement(worker, **kw)
        )
        return worker

    def _snapshot(self, used):
        return {
            (0, c): wm.ChipletMemory(total=self.TOTAL, used=u)
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
        worker = self._worker(programs=programs, snapshot=self._snapshot(used))
        worker._release_kv_cache_tensors = lambda cfg: order.append("release")
        snapshot = worker._dynamic_kv_memory_snapshot

        def recording_snapshot(device):
            order.append("snapshot")
            return snapshot(device)

        worker._dynamic_kv_memory_snapshot = recording_snapshot
        n = RBLNWorker.compute_dynamic_kv_num_blocks(worker)
        # chiplet 1: (35 GiB - 30 GiB) / 2 MiB = 2560 blocks, nothing subtracted
        assert n == 5 * 512
        assert order == ["release", "snapshot"]

    def test_a_dry_run_reports_and_resizes_nothing(self, caplog, monkeypatch):
        current = 200
        programs = [_program([HEAD_SPLIT, HEAD_SPLIT], name="0/0", extent=current)]
        resident = current * 2 * 2**20
        worker = self._worker(
            programs=programs,
            snapshot=self._snapshot([30 * self.GIB + resident] * 4),
        )
        worker.cache_config.num_gpu_blocks_override = current
        worker._kv_blocks_before_shrink = None
        worker.model_runner.kv_cache_config.num_blocks = current
        worker.vllm_config = SimpleNamespace()
        monkeypatch.setattr(
            wm,
            "minimum_kv_blocks",
            lambda cfg, kv: SimpleNamespace(one_request=8, decode_batch=1, needed=9),
        )
        worker._log_dynamic_kv_dry_run = lambda *args: (
            RBLNWorker._log_dynamic_kv_dry_run(worker, *args)
        )
        worker._dynamic_kv_num_blocks_from_placement = lambda **kw: (
            RBLNWorker._dynamic_kv_num_blocks_from_placement(worker, **kw)
        )
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN",
                True,
            ),
            caplog.at_level("WARNING"),
        ):
            assert RBLNWorker.compute_dynamic_kv_num_blocks(worker) is None
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

    def test_a_dry_run_that_cannot_size_warns_instead_of_raising(self, caplog):
        worker = self._worker(programs=[_program([])], snapshot=self._snapshot([0] * 4))
        worker._kv_blocks_before_shrink = None
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN",
                True,
            ),
            caplog.at_level("WARNING"),
        ):
            assert RBLNWorker.compute_dynamic_kv_num_blocks(worker) is None
        assert "could not be computed" in caplog.text

    def test_the_copy_stream_reserve_comes_off_every_chiplet(self):
        programs = [_program([HEAD_SPLIT, HEAD_SPLIT], name="0/0")]
        worker = self._worker(
            programs=programs, snapshot=self._snapshot([30 * self.GIB] * 4)
        )
        worker._kv_copy_stream_reserve_bytes = lambda: 64 * 2**20
        # (35 GiB - 30 GiB - 64 MiB) / 2 MiB = 2560 - 32 blocks
        assert RBLNWorker.compute_dynamic_kv_num_blocks(worker) == 5 * 512 - 32

    def test_gpu_memory_utilization_bounds_the_budget(self):
        programs = [_program([HEAD_SPLIT])]
        # The snapshot is taken after the compile cache is released: base 0.
        snapshot = self._snapshot([0] * 4)
        full = RBLNWorker.compute_dynamic_kv_num_blocks(
            self._worker(programs=programs, snapshot=snapshot, gmu=1.0)
        )
        half = RBLNWorker.compute_dynamic_kv_num_blocks(
            self._worker(programs=programs, snapshot=snapshot, gmu=0.5)
        )
        assert full == 35 * 1024
        assert half == full // 2

    def test_the_snapshot_is_taken_on_the_program_s_device(self):
        seen = []
        programs = [_program([HEAD_SPLIT], device=torch.device("cpu", 3))]
        worker = self._worker(programs=programs, snapshot=self._snapshot([0] * 4))

        def snapshot(device):
            seen.append(device)
            return self._snapshot([0] * 4), "stub"

        worker._dynamic_kv_memory_snapshot = snapshot
        RBLNWorker.compute_dynamic_kv_num_blocks(worker)
        assert seen == [torch.device("cpu", 3)]

    def test_a_base_over_budget_is_refused_with_the_breakdown(self):
        programs = [_program([HEAD_SPLIT])]
        snapshot = self._snapshot([self.TOTAL, 0, 0, 0])
        with pytest.raises(RuntimeError, match="no KV block fits") as exc:
            RBLNWorker.compute_dynamic_kv_num_blocks(
                self._worker(programs=programs, snapshot=snapshot, gmu=0.9)
            )
        assert "0:0(" in str(exc.value)

    def test_a_dummy_device_keeps_the_estimate(self, caplog):
        """The executor compiles under RBLN_DUMMY_DEVICE=1: nothing to measure,
        so the pre-shrink count is restored instead of refusing the compile."""
        programs = [_program([HEAD_SPLIT])]
        worker = self._worker(programs=programs, snapshot=self._snapshot([0] * 4))
        with (
            patch.object(
                wm.torch,
                "rbln",
                SimpleNamespace(is_dummy_device=lambda: True),
                create=True,
            ),
            caplog.at_level("WARNING"),
        ):
            assert RBLNWorker.compute_dynamic_kv_num_blocks(worker) is None
        assert "RBLN_DUMMY_DEVICE" in caplog.text

    def test_programs_that_disagree_on_the_layout_are_refused(self):
        other = _placement((_shard(0, 0, (2, S0, 8, 1, 1024, 128)),))
        programs = [_program([HEAD_SPLIT]), _program([other], name="0/1")]
        with pytest.raises(RuntimeError, match="disagree"):
            RBLNWorker.compute_dynamic_kv_num_blocks(
                self._worker(programs=programs, snapshot=self._snapshot([0] * 4))
            )


class TestDynamicKvMemorySnapshot:
    """`_dynamic_kv_memory_snapshot` prefers the driver's per-chiplet figures and
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
    def _worker(foreign=0):
        return SimpleNamespace(_foreign_dram_used_bytes=foreign)

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

    def _snapshot(self, rbln, worker=None):
        with patch.object(wm.torch, "rbln", rbln, create=True):
            return RBLNWorker._dynamic_kv_memory_snapshot(
                worker or self._worker(), torch.device("cpu")
            )

    def test_the_driver_wins_when_it_answers(self):
        rbln, calls = self._rbln(driver=self.DRIVER, allocator=self.ALLOCATOR)
        snapshot, source = self._snapshot(rbln)
        assert source == "driver"
        assert snapshot == {
            (0, 0): wm.ChipletMemory(total=100, used=40),
            (0, 1): wm.ChipletMemory(total=100, used=10),
        }
        assert calls == []

    def test_an_old_driver_falls_back_to_the_allocator(self, caplog):
        rbln, calls = self._rbln(
            driver_error=RuntimeError("does not provide the query"),
            allocator=self.ALLOCATOR,
            per_chiplet=100,
        )
        with caplog.at_level("WARNING"):
            snapshot, source = self._snapshot(rbln, self._worker(foreign=20))
        assert source == "allocator"
        reserve = wm.DYNAMIC_KV_ALLOCATOR_RESERVE_BYTES
        assert snapshot == {
            (0, 0): wm.ChipletMemory(total=100, used=30 + 10 + reserve),
            (0, 1): wm.ChipletMemory(total=100, used=5 + 10 + reserve),
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


class TestWarmupCapturesPrograms:
    """The programs warm-up builds are the only handle on the KV-holding
    runtimes, so the capture has to wrap exactly the warm-up."""

    def test_off_means_no_capture(self):
        worker = SimpleNamespace()
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                False,
            ),
            RBLNWorker._capture_dynamic_kv_programs(worker) as programs,
        ):
            pass
        assert programs is None

    def test_on_opens_torch_rbln_s_scope(self):
        recorded = ["p0", "p1"]

        @contextmanager
        def fake_capture():
            yield recorded

        worker = SimpleNamespace()
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                True,
            ),
            patch.object(wm, "has_torch_rbln", True),
            patch.object(
                wm.torch,
                "rbln",
                SimpleNamespace(capture_programs=fake_capture),
                create=True,
            ),
            RBLNWorker._capture_dynamic_kv_programs(worker) as programs,
        ):
            pass
        assert programs is recorded

    def test_on_without_torch_rbln_refuses(self):
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                True,
            ),
            patch.object(wm, "has_torch_rbln", False),
            pytest.raises(RuntimeError, match="capture_programs"),
        ):
            RBLNWorker._capture_dynamic_kv_programs(SimpleNamespace())

    def test_runtimes_are_deduped_across_programs(self):
        shared = object()
        worker = SimpleNamespace(
            _dynamic_kv_programs=[
                _program([HEAD_SPLIT], runtime=shared),
                _program([HEAD_SPLIT], runtime=shared),
                _program([], runtime=object()),
            ]
        )
        assert len(RBLNWorker._collect_dynamic_kv_runtimes(worker)) == 2


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
    def _shrink(config, *, dynamic=True, override=None, warmup_skipped=False):
        # The flag is patched as a module attribute, not via os.environ: a
        # setattr elsewhere would leave an attribute shadowing envs.__getattr__.
        worker = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            _kv_blocks_before_shrink=None,
            _compile_and_warmup_skip_reason=lambda: (
                "enforce_eager is set" if warmup_skipped else None
            ),
        )
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
            dynamic,
        ):
            out = RBLNWorker._maybe_shrink_kv_cache_for_compile(worker, config)
        return worker, out

    def test_the_flag_alone_shrinks_to_the_constant(self, caplog):
        config = self._config()
        with caplog.at_level("INFO"):
            worker, out = self._shrink(config)

        assert out is not config
        assert out.num_blocks == wm.COMPILE_KV_CACHE_NUM_BLOCKS
        assert worker._kv_blocks_before_shrink == self.ESTIMATED_BLOCKS
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
            worker, out = self._shrink(config, dynamic=False)
        assert out is config
        assert worker._kv_blocks_before_shrink is None
        assert "[Dynamic KV]" not in caplog.text

    def test_a_dry_run_compiles_at_the_sized_count(self, caplog):
        config = self._config()
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN",
                True,
            ),
            caplog.at_level("WARNING"),
        ):
            worker, out = self._shrink(config)
        assert out is config
        assert worker._kv_blocks_before_shrink is None
        assert "dry run" in caplog.text

    def test_a_pinned_block_count_cancels_the_shrink(self, caplog):
        config = self._config()
        with caplog.at_level("WARNING"):
            worker, out = self._shrink(config, override=64)

        assert out is config
        assert worker._kv_blocks_before_shrink is None
        assert "num-gpu-blocks-override" in caplog.text

    def test_a_hint_that_cannot_shrink_refuses(self):
        # The estimate is free memory over the cost of one block, so a large
        # block_size can legally put it at or below the hint. Serving on there
        # would silently keep the pre-compile estimate, so it is a refusal.
        with pytest.raises(RuntimeError, match="nothing to shrink"):
            self._shrink(self._config(num_blocks=wm.COMPILE_KV_CACHE_NUM_BLOCKS))

    def test_no_warmup_means_no_shrink(self, caplog):
        """Skipping compile/warm-up has to skip the shrink too: otherwise the
        latch is set, the profile query finds no runtimes, and the restore path
        trips an assertion whose message names none of the cause.
        """
        config = self._config()
        with caplog.at_level("WARNING"):
            worker, out = self._shrink(config, warmup_skipped=True)

        assert out is config
        assert worker._kv_blocks_before_shrink is None
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

    def test_the_attention_guard_runs_before_the_shrink(self):
        calls: list[str] = []

        def record(name: str, ret: object = None) -> object:
            calls.append(name)
            return ret

        config = SimpleNamespace(num_blocks=4, kv_cache_tensors=[])
        worker = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=None, num_cpu_blocks=None),
            vllm_config=object(),
            model_runner=SimpleNamespace(
                initialize_kv_cache=lambda cfg: record("initialize_kv_cache")
            ),
            _assert_dynamic_kv_attention_layout=lambda: record("attention"),
            _assert_dynamic_kv_cache_layout=lambda: record("bindings"),
            _maybe_shrink_kv_cache_for_compile=lambda cfg: record("shrink", cfg),
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

    def test_a_non_paged_causal_layer_is_refused_by_name(self):
        """`block_size == max_model_len` makes is_normal True -- and is also where
        the estimate can fall below the hint, so the wrong refusal could fire."""
        worker = SimpleNamespace(vllm_config=object())
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.get_layers_from_vllm_config",
                return_value={"layer.0": self._layer(is_normal=True)},
            ),
            pytest.raises(RuntimeError) as exc,
        ):
            RBLNWorker._assert_dynamic_kv_attention_layout(worker)
        assert "paged causal or sliding-window naive kernel" in str(exc.value)
        assert "layer.0" in str(exc.value)
        assert "nothing to shrink" not in str(exc.value)

    def test_a_paged_causal_layer_passes(self):
        worker = SimpleNamespace(vllm_config=object())
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.get_layers_from_vllm_config",
            return_value={"layer.0": self._layer()},
        ):
            RBLNWorker._assert_dynamic_kv_attention_layout(worker)

    def test_a_sliding_window_layer_passes(self):
        """gpt-oss alternates full and windowed layers; the compiler admits a
        dynamic KV input on `paged_sliding_window_attention_naive_*` too."""
        worker = SimpleNamespace(vllm_config=object())
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.get_layers_from_vllm_config",
            return_value={
                "layer.0": self._layer(),
                "layer.1": self._layer(sliding_window=128),
            },
        ):
            RBLNWorker._assert_dynamic_kv_attention_layout(worker)

    def test_deduped_bases_pass(self):
        """gpt-oss shares one tensor between a full and a windowed layer; the
        compiler takes the deduped base through both views."""
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                kv_cache_bases=[object()], shared_kv_cache_layers={}
            )
        )
        RBLNWorker._assert_dynamic_kv_cache_layout(worker)

    def test_cross_layer_sharing_is_still_refused_after_the_split(self):
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                kv_cache_bases=[], shared_kv_cache_layers={"layer.1": "layer.0"}
            )
        )
        with pytest.raises(RuntimeError, match="cross-layer KV"):
            RBLNWorker._assert_dynamic_kv_cache_layout(worker)


class TestDynamicKvFailuresRaise:
    """After the shrink, failing to size from the device must not boot: the run
    would serve the pre-compile estimate. The gates before it stay a quiet None."""

    @staticmethod
    def _worker(*, shrunk=True, override=None, programs=()):
        worker = SimpleNamespace(
            rank=0,
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            _kv_blocks_before_shrink=211 if shrunk else None,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=211)
            ),
            _dynamic_kv_programs=list(programs),
            _release_kv_cache_tensors=lambda cfg: None,
        )
        worker._dynamic_kv_num_blocks_from_placement = lambda **kw: (
            RBLNWorker._dynamic_kv_num_blocks_from_placement(worker, **kw)
        )
        return worker

    @pytest.fixture(autouse=True)
    def _real_device(self):
        with patch.object(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            yield

    def test_no_program_after_the_shrink_raises(self):
        with pytest.raises(RuntimeError, match="none of the 0"):
            RBLNWorker.compute_dynamic_kv_num_blocks(self._worker(programs=()))

    def test_no_dynamic_input_after_the_shrink_raises(self):
        """Every program static is the documented replayed-static-build case.

        It used to log an error and boot on the estimate, which is the bug this
        feature exists to remove.
        """
        with pytest.raises(RuntimeError) as exc:
            RBLNWorker.compute_dynamic_kv_num_blocks(
                self._worker(programs=(_program([]), _program([], name="0/1")))
            )
        assert "dynamic-shape KV input" in str(exc.value)
        assert "VLLM_CACHE_ROOT" in str(exc.value)

    def test_the_pre_shrink_gates_still_return_none(self):
        """An override and "not shrunk" are legitimate: nothing moved."""
        assert (
            RBLNWorker.compute_dynamic_kv_num_blocks(self._worker(override=64)) is None
        )
        # The shrink did not happen, so there is nothing to size from.
        assert (
            RBLNWorker.compute_dynamic_kv_num_blocks(self._worker(shrunk=False)) is None
        )


class TestKvCopyStreamReserve:
    """The reserve follows the scheduler's own sub-block prefix caching predicate."""

    @staticmethod
    def _worker(*, prefix_caching=True, sub_block_cache=True):
        return SimpleNamespace(
            cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
            vllm_config=SimpleNamespace(
                additional_config={"sub_block_cache": sub_block_cache}
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
            RBLNWorker._kv_copy_stream_reserve_bytes(self._worker())
            == wm.DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES
        )

    def test_nothing_without_prefix_caching(self, monkeypatch):
        self._manager(monkeypatch, eligible=True)
        worker = self._worker(prefix_caching=False)
        assert RBLNWorker._kv_copy_stream_reserve_bytes(worker) == 0

    def test_nothing_when_sub_block_cache_is_off(self, monkeypatch):
        self._manager(monkeypatch, eligible=True)
        worker = self._worker(sub_block_cache=False)
        assert RBLNWorker._kv_copy_stream_reserve_bytes(worker) == 0

    def test_nothing_when_the_config_is_ineligible(self, monkeypatch):
        self._manager(monkeypatch, eligible=False)
        assert RBLNWorker._kv_copy_stream_reserve_bytes(self._worker()) == 0


class TestApplyResizesThenMaterializes:
    """`apply_dynamic_kv_num_blocks` settles the latch, and any actual resize
    must be followed by the boot-time materialization: without it the first
    request pays the whole pool's physical allocation (measured 19.8 s TTFT)."""

    @staticmethod
    def _worker(*, before_shrink=211, current=4):
        calls: list = []
        worker = SimpleNamespace(
            _kv_blocks_before_shrink=before_shrink,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=current)
            ),
            _reallocate_kv_cache=lambda target: calls.append(("realloc", target)),
            _materialize_kv_cache=lambda: calls.append(("materialize",)),
            _dynamic_kv_expected_used={},
            _log_dynamic_kv_fit_check=lambda n: calls.append(("check", n)),
        )
        return worker, calls

    def test_a_computed_count_reallocates_then_materializes(self):
        worker, calls = self._worker()
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, 1368) == 1368
        assert calls == [("realloc", 1368), ("materialize",)]
        assert worker._kv_blocks_before_shrink is None

    def test_a_computed_count_is_checked_against_the_prediction(self):
        worker, calls = self._worker()
        worker._dynamic_kv_expected_used = {(0, 0): 123}
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, 1368) == 1368
        assert calls == [("realloc", 1368), ("materialize",), ("check", 1368)]

    def test_the_fit_check_reports_measured_against_expected(self, caplog):
        snapshot = {(0, 0): wm.ChipletMemory(total=1000, used=460)}
        worker = SimpleNamespace(
            device=torch.device("cpu"),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
            _dynamic_kv_expected_used={(0, 0): 450, (0, 1): 7},
            _dynamic_kv_memory_snapshot=lambda device: (snapshot, "driver"),
        )
        with caplog.at_level("INFO"):
            RBLNWorker._log_dynamic_kv_fit_check(worker, 58)
        assert "0:0(expected=450 measured=460 diff=+10 budget_left=+40)" in caplog.text
        assert "0:1(expected=7 measured=?)" in caplog.text

    def test_none_restores_the_pre_shrink_count(self):
        worker, calls = self._worker()
        with patch.object(
            wm.torch,
            "rbln",
            SimpleNamespace(is_dummy_device=lambda: False),
            create=True,
        ):
            assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, None) == 211
        assert calls == [("realloc", 211), ("materialize",)]

    def test_none_on_a_dummy_device_keeps_the_compile_cache(self):
        worker, calls = self._worker()
        with patch.object(
            wm.torch, "rbln", SimpleNamespace(is_dummy_device=lambda: True), create=True
        ):
            assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, None) == 4
        assert calls == []
        assert worker._kv_blocks_before_shrink is None

    def test_a_matching_count_skips_both(self):
        worker, calls = self._worker(before_shrink=4, current=4)
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, 4) == 4
        assert calls == []

    def test_nothing_pending_returns_none(self):
        worker, calls = self._worker(before_shrink=None)
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, None) is None
        assert calls == []

    def test_materialize_runs_the_smallest_compiled_decode_bucket(self):
        ran: list = []
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                bucketing_manager=SimpleNamespace(decode_batch_buckets=[8, 4, 16]),
                offload_context=nullcontext,
                _dummy_run=lambda *args: ran.append(args),
            )
        )
        RBLNWorker._materialize_kv_cache(worker)
        assert ran == [(4, 1, False)]
