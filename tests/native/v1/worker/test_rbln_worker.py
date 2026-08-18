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
from types import SimpleNamespace

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
        specialized_moe_decode=False,
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
    ):
        vcfg = _make_vllm_config(enforce_eager=enforce_eager)
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
