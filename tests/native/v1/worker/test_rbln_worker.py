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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

import vllm_rbln.v1.worker.rbln_worker as wm
from vllm_rbln.v1.worker.rbln_worker import (
    RBLNWorker,
    init_worker_distributed_environment,
)


def _make_vllm_config(
    *,
    world_size=1,
    data_parallel_size=1,
    data_parallel_rank=0,
    world_size_across_dp=1,
    assigned_physical_gpu_ids=None,
    quantization=None,
    enforce_eager=False,
    profiler=None,
):
    return SimpleNamespace(
        profiler_config=SimpleNamespace(profiler=profiler),
        parallel_config=SimpleNamespace(
            world_size=world_size,
            tensor_parallel_size=world_size,
            pipeline_parallel_size=1,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
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
def _env_cleanup():
    # Save/restore the process env vars the worker touches.
    keys = [
        "RBLN_DEVICES",
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
        world_size_across_dp=None,
        assigned_physical_gpu_ids=None,
        num_devices=1,
        num_ray_nodes=1,
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
            world_size_across_dp=wsd,
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
        )
        monkeypatch.setattr(WorkerBase, "__init__", _fake_super_init)
        monkeypatch.setattr(
            wm,
            "current_platform",
            SimpleNamespace(
                device_type="cpu",
                device_control_env_var="RBLN_DEVICES",
                dist_backend="gloo",
                get_device_name=lambda: device_name,
            ),
        )
        monkeypatch.setattr(
            wm.envs, "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", num_devices
        )
        monkeypatch.setattr(wm.envs, "VLLM_RBLN_NUM_RAY_NODES", num_ray_nodes)
        monkeypatch.setattr(wm, "has_torch_rbln", has_torch_rbln)
        return RBLNWorker(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=True,
        )

    return _make


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
    def test_auto_tp1(self, make_worker):
        make_worker(world_size=1)
        assert os.environ["RBLN_DEVICES"] == "0"

    @pytest.mark.parametrize("local_rank, expected", [(0, "0"), (1, "1")])
    def test_auto_slices_by_local_rank(self, make_worker, local_rank, expected):
        make_worker(world_size=4, local_rank=local_rank)
        assert os.environ["RBLN_DEVICES"] == expected

    def test_auto_dp_rank_offsets_range(self, make_worker):
        # dp_rank=1, tp=2 -> device range starts at total_device_count(2).
        make_worker(
            world_size=2, data_parallel_size=2, data_parallel_rank=1, local_rank=0
        )
        assert os.environ["RBLN_DEVICES"] == "2"

    def test_auto_ray_nodes_divide_world_size(self, make_worker):
        # world_size 4 // 2 ray nodes -> effective 2 devices.
        make_worker(world_size=4, num_ray_nodes=2, local_rank=0)
        assert os.environ["RBLN_DEVICES"] == "0"

    @pytest.mark.parametrize("local_rank, expected", [(0, "0,1"), (1, "2,3")])
    def test_multi_device_slices_and_sets_npus(self, make_worker, local_rank, expected):
        make_worker(
            world_size=2, num_devices=2, has_torch_rbln=True, local_rank=local_rank
        )
        assert os.environ["RBLN_DEVICES"] == expected
        assert os.environ["RBLN_NPUS_PER_DEVICE"] == "2"

    def test_multi_device_without_torch_rbln_skips_npus(self, make_worker):
        make_worker(world_size=2, num_devices=2, has_torch_rbln=False)
        assert os.environ["RBLN_DEVICES"] == "0,1"
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_single_device_skips_npus(self, make_worker):
        make_worker(world_size=1, num_devices=1, has_torch_rbln=True)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_explicit_expands_device_id_for_local_rank(self, make_worker):
        # Preset device list: local_rank 1 -> device_id 5 -> expanded to "5".
        os.environ["RBLN_DEVICES"] = "4,5,6,7"
        make_worker(world_size=4, local_rank=1)
        assert os.environ["RBLN_DEVICES"] == "5"

    def test_explicit_wrong_count_raises(self, make_worker):
        os.environ["RBLN_DEVICES"] = "0,1"
        with pytest.raises(AssertionError):
            make_worker(world_size=4)

    def test_explicit_non_int_raises(self, make_worker):
        os.environ["RBLN_DEVICES"] = "a,b,c,d"
        with pytest.raises(ValueError):
            make_worker(world_size=4, local_rank=0)

    @pytest.mark.parametrize("dp_rank, assigned", [(0, [4]), (2, [6])])
    def test_dp_mapping_wins_over_env(self, make_worker, dp_rank, assigned):
        # Under DP the env var holds the whole deployment; vLLM 0.24 puts this
        # rank's share on the config instead.
        os.environ["RBLN_DEVICES"] = "4,5,6,7"
        make_worker(
            world_size=1,
            data_parallel_size=4,
            data_parallel_rank=dp_rank,
            assigned_physical_gpu_ids=assigned,
        )
        assert os.environ["RBLN_DEVICES"] == str(assigned[0])

    def test_dp_mapping_expands_by_num_devices(self, make_worker):
        os.environ["RBLN_DEVICES"] = "0,1,2,3"
        make_worker(
            world_size=2,
            data_parallel_size=2,
            data_parallel_rank=1,
            assigned_physical_gpu_ids=[2, 3],
            num_devices=2,
            local_rank=1,
        )
        assert os.environ["RBLN_DEVICES"] == "6,7"

    def test_no_mapping_falls_back_to_env(self, make_worker):
        os.environ["RBLN_DEVICES"] = "0,1"
        make_worker(world_size=2, data_parallel_size=2, data_parallel_rank=1)
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_dp_mapping_wrong_count_raises(self, make_worker):
        os.environ["RBLN_DEVICES"] = "4,5,6,7"
        with pytest.raises(AssertionError):
            make_worker(
                world_size=1, data_parallel_size=4, assigned_physical_gpu_ids=[4, 5]
            )


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
        specialized_moe_decode=0,
        decode_buckets=3,
        drafter=None,
        speculative_config=None,
    ):
        vcfg = _make_vllm_config(quantization=quantization)
        vcfg.model_config.hf_config = hf_config
        worker = make_worker(vllm_config=vcfg, device_name=device_name)
        captured: dict = {}
        monkeypatch.setattr(
            wm, "estimate_available_memory", lambda **kw: captured.update(kw) or 999
        )
        monkeypatch.setattr(wm, "estimate_model_kernel_size", lambda **kw: 111)
        if speculative_config is not None:
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

    def test_num_runtimes_from_buckets_and_moe(self, make_worker, monkeypatch):
        cap = self._capture(
            make_worker, monkeypatch, specialized_moe_decode=2, decode_buckets=3
        )
        assert cap["num_runtimes"] == 6  # 1 + buckets(3) + moe(2)

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
        assert cap["num_runtimes"] == 8  # (1+3+0) + (1+3)

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
# Dynamic KV: the compile-time cache TP>=2 does not return
# ---------------------------------------------------------------------------
class TestRetainedCompileKvCacheCharge:
    """`_charge_retained_compile_kv_cache` keeps the budget from being spent
    twice on TP>=2, where `_release_kv_cache_tensors` does not get the
    compile-time cache back and cannot tell that it did not.
    """

    BUDGET_PER_CHIPLET = 33_772_535_808

    @staticmethod
    def _profile(regions):
        from vllm_rbln.v1.worker.kv_profile import (
            MergedKvCacheMemoryProfile,
            MergedMemoryRegion,
        )

        return MergedKvCacheMemoryProfile(
            device_regions=[MergedMemoryRegion(*r) for r in regions]
        )

    @classmethod
    def _budget(cls, num_chiplets=4):
        return {0: {c: cls.BUDGET_PER_CHIPLET for c in range(num_chiplets)}}

    @staticmethod
    def _charge(fake_self_tp, resident_blocks, budget, merged):
        # The charge reads the block count the runner actually allocated, not
        # the env var: see `_charge_retained_compile_kv_cache`.
        worker = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=fake_self_tp),
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=resident_blocks)
            ),
        )
        return RBLNWorker._charge_retained_compile_kv_cache(worker, budget, merged)

    # (node_id, chiplet_id, base_bytes, bytes_per_block, alignment)
    MINIMAX_TP4EP = [(0, 0, 0, 65_011_712, 1)]

    def test_tp1_is_not_charged_because_the_cache_comes_back(self):
        budget = self._budget()
        out = self._charge(1, 8, budget, self._profile(self.MINIMAX_TP4EP))
        assert out == budget

    def test_tp4_is_charged_on_the_chiplet_that_holds_the_growth(self):
        out = self._charge(4, 8, self._budget(), self._profile(self.MINIMAX_TP4EP))

        # The MiniMax TP4+EP profile puts every growth region on chiplet 0, so
        # only chiplet 0 loses the 8 x 62 MiB the outgoing cache still holds.
        assert out[0][0] == self.BUDGET_PER_CHIPLET - 8 * 65_011_712
        for chiplet_id in (1, 2, 3):
            assert out[0][chiplet_id] == self.BUDGET_PER_CHIPLET

    def test_an_empty_cache_has_nothing_resident_to_charge(self):
        budget = self._budget()
        out = self._charge(4, 0, budget, self._profile(self.MINIMAX_TP4EP))
        assert out == budget

    def test_a_base_only_profile_is_not_charged(self):
        budget = self._budget()
        out = self._charge(4, 8, budget, self._profile([(0, 0, 1 << 30, 0, 1)]))
        assert out == budget

    def test_alignment_is_accounted_per_region(self):
        # align_up(10059840 + 8*3000000) - align_up(10059840) at 2 MiB
        # = 35651584 - 10485760, which is 1165824 more than 8 * 3000000.
        out = self._charge(
            4,
            8,
            self._budget(),
            self._profile([(0, 0, 10_059_840, 3_000_000, 1 << 21)]),
        )
        assert out[0][0] == self.BUDGET_PER_CHIPLET - 25_165_824

    def test_a_budget_the_retained_cache_exhausts_is_refused(self):
        huge = [(0, 0, 0, self.BUDGET_PER_CHIPLET // 4, 1)]
        with pytest.raises(RuntimeError, match="leaves no budget"):
            self._charge(4, 8, self._budget(), self._profile(huge))

    def test_the_caller_s_budget_is_not_mutated(self):
        budget = self._budget()
        self._charge(4, 8, budget, self._profile(self.MINIMAX_TP4EP))
        assert budget[0][0] == self.BUDGET_PER_CHIPLET


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
        assert "paged_flash_causal_attention_naive" in str(exc.value)
        assert "layer.0" in str(exc.value)
        assert "nothing to shrink" not in str(exc.value)

    def test_a_paged_causal_layer_passes(self):
        worker = SimpleNamespace(vllm_config=object())
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.get_layers_from_vllm_config",
            return_value={"layer.0": self._layer()},
        ):
            RBLNWorker._assert_dynamic_kv_attention_layout(worker)

    def test_deduped_bases_are_still_refused_after_the_split(self):
        """Guards the split itself: moving this check earlier would make it see an
        empty list and pass, so its refusal has to stay asserted."""
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                kv_cache_bases=[object()], shared_kv_cache_layers={}
            )
        )
        with pytest.raises(RuntimeError, match="KV base deduplication"):
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
    def _worker(*, shrunk=True, override=None, runtimes=()):
        return SimpleNamespace(
            rank=0,
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            _kv_blocks_before_shrink=211 if shrunk else None,
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=211)
            ),
            _collect_dynamic_kv_runtimes=lambda: list(runtimes),
        )

    def test_no_runtime_after_the_shrink_raises(self):
        with pytest.raises(RuntimeError, match="not one of the 0"):
            RBLNWorker.compute_dynamic_kv_num_blocks(self._worker(runtimes=()))

    def test_no_profile_after_the_shrink_raises(self):
        """Every runtime refusing the query is the documented static-artifact case.

        It used to log an error and boot on the estimate, which is the bug this
        feature exists to remove.
        """
        runtime = SimpleNamespace(
            _executor=SimpleNamespace(
                kv_cache_memory_profile=lambda: (_ for _ in ()).throw(
                    RuntimeError("no dynamic-shape variable")
                )
            )
        )
        with pytest.raises(RuntimeError) as exc:
            RBLNWorker.compute_dynamic_kv_num_blocks(self._worker(runtimes=(runtime,)))
        assert "did not return" in str(exc.value) or "not one of the" in str(exc.value)
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
        )
        return worker, calls

    def test_a_computed_count_reallocates_then_materializes(self):
        worker, calls = self._worker()
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, 1368) == 1368
        assert calls == [("realloc", 1368), ("materialize",)]
        assert worker._kv_blocks_before_shrink is None

    def test_none_restores_the_pre_shrink_count(self):
        worker, calls = self._worker()
        assert RBLNWorker.apply_dynamic_kv_num_blocks(worker, None) == 211
        assert calls == [("realloc", 211), ("materialize",)]

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
