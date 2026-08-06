# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for RBLNWorker: interface compliance, WorkerBase contract,
device env initialization, and behavior tests."""

import inspect
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiler_config(trace_dir=None):
    return SimpleNamespace(
        profiler="torch",
        torch_profiler_dir=trace_dir,
        torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=False,
        torch_profiler_dump_cuda_time_total=False,
        delay_iterations=0,
        max_iterations=0,
        wait_iterations=0,
        warmup_iterations=0,
        active_iterations=0,
    )


def _make_parallel_config(
    world_size=1,
    data_parallel_size=1,
    data_parallel_rank=0,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    world_size_across_dp=1,
):
    return SimpleNamespace(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        disable_custom_all_reduce=False,
        distributed_executor_backend=None,
        world_size_across_dp=world_size_across_dp,
    )


def _make_model_config(
    trust_remote_code=False,
    seed=42,
    quantization=None,
    enforce_eager=False,
    use_mla=False,
):
    return SimpleNamespace(
        trust_remote_code=trust_remote_code,
        seed=seed,
        quantization=quantization,
        enforce_eager=enforce_eager,
        use_mla=use_mla,
    )


def _make_cache_config(gpu_memory_utilization=0.9):
    return SimpleNamespace(
        gpu_memory_utilization=gpu_memory_utilization,
        num_gpu_blocks=0,
        num_cpu_blocks=0,
    )


def _make_scheduler_config(max_num_batched_tokens=256, max_num_seqs=32):
    return SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )


def _make_vllm_config(
    profiler_trace_dir=None,
    trust_remote_code=False,
    quantization=None,
    enforce_eager=False,
    data_parallel_size=1,
    data_parallel_rank=0,
    world_size=1,
    world_size_across_dp=1,
    use_mla=False,
    speculative_config=None,
):
    return SimpleNamespace(
        profiler_config=_make_profiler_config(profiler_trace_dir),
        parallel_config=_make_parallel_config(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
            world_size_across_dp=world_size_across_dp,
        ),
        model_config=_make_model_config(
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            enforce_eager=enforce_eager,
            use_mla=use_mla,
        ),
        speculative_config=speculative_config,
        cache_config=_make_cache_config(),
        scheduler_config=_make_scheduler_config(),
        device_config=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
        kv_transfer_config=None,
        instance_id="test-instance",
    )


@pytest.fixture(autouse=True)
def env_cleanup():
    """Save and restore environment variables touched by tests."""
    keys = [
        "RBLN_DEVICES",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "RBLN_NPUS_PER_DEVICE",
        "RCCL_PORT_GEN",
        "RBLN_NUM_THREADS",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Worker factory
# ---------------------------------------------------------------------------

# Patches that neutralise heavy dependencies during __init__
_INIT_PATCHES = {
    "current_platform": "vllm_rbln.v1.worker.rbln_worker.current_platform",
    "envs_num_devices": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",  # noqa: E501
    "envs_ray": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUM_RAY_NODES",
    "envs_auto_port": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT",
    "envs_compile": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_COMPILE_MODEL",
    "envs_warmup": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_ENABLE_WARM_UP",
    "envs_metrics": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS",
    "envs_numa": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUMA",
    "has_torch_rbln": "vllm_rbln.v1.worker.rbln_worker.has_torch_rbln",
}


def _fake_super_init(
    self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False
):
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


def _create_worker(
    vllm_config=None,
    local_rank=0,
    rank=0,
    is_driver_worker=True,
    *,
    num_devices=1,
    num_ray_nodes=1,
    has_torch_rbln_val=False,
    envs_overrides=None,
):
    """Instantiate RBLNWorker with mocked-out heavy dependencies."""
    from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

    if vllm_config is None:
        vllm_config = _make_vllm_config()

    defaults = {
        "envs_num_devices": num_devices,
        "envs_ray": num_ray_nodes,
        "envs_auto_port": False,
        "envs_compile": True,
        "envs_warmup": True,
        "envs_metrics": False,
        "envs_numa": False,
        "has_torch_rbln": has_torch_rbln_val,
    }
    if envs_overrides:
        defaults.update(envs_overrides)

    active = []
    try:
        # Patch WorkerBase.__init__
        p = patch.object(
            RBLNWorker.__bases__[0],
            "__init__",
            _fake_super_init,
        )
        active.append(p)
        p.start()

        # Patch current_platform
        platform_mock = MagicMock()
        platform_mock.device_type = "cpu"
        platform_mock.device_control_env_var = "RBLN_DEVICES"
        platform_mock.dist_backend = "gloo"
        platform_mock.get_device_name.return_value = "RBLN-CA25"
        p = patch(_INIT_PATCHES["current_platform"], platform_mock)
        active.append(p)
        p.start()

        # Patch scalar env values
        for key in (
            "envs_num_devices",
            "envs_ray",
            "envs_auto_port",
            "envs_compile",
            "envs_warmup",
            "envs_metrics",
            "envs_numa",
            "has_torch_rbln",
        ):
            p = patch(_INIT_PATCHES[key], defaults[key])
            active.append(p)
            p.start()

        worker = RBLNWorker(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=is_driver_worker,
        )
    finally:
        for p in active:
            p.stop()

    return worker


# ===========================================================================
# 1. Interface compliance: RBLNWorker implements all WorkerBase methods
# ===========================================================================


class TestInterfaceCompliance:
    """Verify RBLNWorker provides implementations for every method that
    WorkerBase declares (both abstract-style raise-NotImplementedError
    and regular methods)."""

    def _get_worker_base_interface_methods(self):
        """Return names of WorkerBase methods that subclasses should provide."""
        base_methods = []
        for name, obj in inspect.getmembers(WorkerBase, predicate=inspect.isfunction):
            if name.startswith("_") and name != "__init__":
                continue
            base_methods.append(name)
        return base_methods

    def _get_notimplemented_methods(self):
        """Return names of WorkerBase methods that raise NotImplementedError."""
        ni_methods = []
        for name, obj in inspect.getmembers(WorkerBase, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            src = inspect.getsource(obj)
            if "NotImplementedError" in src:
                ni_methods.append(name)
        return ni_methods

    def test_rbln_worker_extends_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert issubclass(RBLNWorker, WorkerBase)

    def test_all_not_implemented_methods_are_overridden(self):
        """Every WorkerBase method that raises NotImplementedError must be
        overridden by RBLNWorker (except known intentional gaps)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        # Methods intentionally not overridden (e.g. speculative-decoding only)
        KNOWN_GAPS = {
            "get_cache_block_size_bytes",
            "add_lora",
            "list_loras",
            "pin_lora",
            "remove_lora",
        }

        ni_methods = self._get_notimplemented_methods()
        assert len(ni_methods) > 0, "Expected some NotImplementedError methods"

        missing = []
        for name in ni_methods:
            if name in KNOWN_GAPS:
                continue
            base_method = getattr(WorkerBase, name)
            child_method = getattr(RBLNWorker, name)
            if child_method is base_method:
                missing.append(name)

        assert missing == [], (
            f"RBLNWorker does not override these WorkerBase methods: {missing}"
        )

    def test_known_gaps_documented(self):
        """Verify that get_cache_block_size_bytes is indeed not overridden
        (intentional gap for speculative decoding)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert (
            RBLNWorker.get_cache_block_size_bytes
            is WorkerBase.get_cache_block_size_bytes
        )

    def test_init_signature_matches_worker_base(self):
        """RBLNWorker.__init__ must accept the same parameters as WorkerBase."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        base_sig = inspect.signature(WorkerBase.__init__)
        child_sig = inspect.signature(RBLNWorker.__init__)

        base_params = list(base_sig.parameters.keys())
        child_params = list(child_sig.parameters.keys())

        assert base_params == child_params, (
            f"Signature mismatch: base={base_params}, child={child_params}"
        )

    def test_execute_model_signature_compatible(self):
        """execute_model must accept scheduler_output positional arg."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.execute_model)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "scheduler_output" in params

    def test_compile_or_warm_up_model_returns(self):
        """compile_or_warm_up_model must return a CompilationTimes (elapsed time)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.compile_or_warm_up_model)
        # The return annotation should be CompilationTimes
        assert (
            sig.return_annotation is CompilationTimes
            or sig.return_annotation == inspect.Parameter.empty
        )

    def test_shutdown_is_overridden(self):
        """shutdown must be overridden (not the base no-op)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.shutdown is not WorkerBase.shutdown

    def test_check_health_is_overridden(self):
        """check_health must be overridden."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.check_health is not WorkerBase.check_health


# ===========================================================================
# 2. WorkerBase contract: class hierarchy and method signatures
# ===========================================================================


class TestWorkerBaseContract:
    """Verify the class hierarchy and that method signatures match vllm
    expectations for pluggable workers."""

    def test_mro_includes_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert WorkerBase in RBLNWorker.__mro__

    def test_direct_parent_is_worker_base(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert RBLNWorker.__bases__[0] is WorkerBase

    def test_initialize_from_config_signature(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.initialize_from_config)
        params = list(sig.parameters.keys())
        assert "kv_cache_config" in params

    def test_load_model_takes_no_args(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.load_model)
        # Only self
        non_self = [p for p in sig.parameters if p != "self"]
        assert non_self == []

    def test_get_kv_cache_spec_returns_dict_annotation(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.get_kv_cache_spec)
        # Should have no positional args beyond self
        non_self = [p for p in sig.parameters if p != "self"]
        assert non_self == []


# ===========================================================================
# Tests: __init__
# ===========================================================================


class TestRBLNWorkerInit:
    def test_basic_init(self):
        worker = _create_worker()
        assert worker.profiler is None
        assert worker.parallel_config.disable_custom_all_reduce is True


# ===========================================================================
# Tests: _init_device_env
# ===========================================================================


class TestInitDeviceEnv:
    def test_auto_device_single(self):
        _create_worker()
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_tp4_single_worker(self):
        _create_worker(num_devices=4)
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_auto_device_multi(self):
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1)
        assert os.environ["RBLN_DEVICES"] == "1"

    def test_tp4_multi_worker_rank1(self):
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1, num_devices=4)
        assert os.environ["RBLN_DEVICES"] == "4,5,6,7"

    def test_multiple_ray_nodes(self):
        cfg = _make_vllm_config(world_size=8)
        _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, num_devices=1, num_ray_nodes=2
        )
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_multiple_ray_nodes_rank3(self):
        cfg = _make_vllm_config(world_size=8)
        _create_worker(
            vllm_config=cfg, local_rank=3, rank=3, num_devices=1, num_ray_nodes=2
        )
        assert os.environ["RBLN_DEVICES"] == "3"

    def test_dp_rank_offsets_device_ids(self):
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, num_devices=1)
        assert os.environ["RBLN_DEVICES"] == "2"

    def test_explicit_device_ids(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=0)
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_explicit_rbln_devices_tp_rank1(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1, num_devices=2)
        assert os.environ["RBLN_DEVICES"] == "2,3"

    def test_invalid_device_ids(self):
        os.environ["RBLN_DEVICES"] = "abc"
        with pytest.raises(ValueError, match="should be a list of integers"):
            _create_worker()

    def test_wrong_device_count(self):
        os.environ["RBLN_DEVICES"] = "0,1,2"
        cfg = _make_vllm_config(world_size=2)
        with pytest.raises(AssertionError, match="should have device count"):
            _create_worker(vllm_config=cfg)

    def test_num_devices_gt1_sets_npus_env(self):
        _create_worker(num_devices=2, has_torch_rbln_val=True)
        assert os.environ.get("RBLN_NPUS_PER_DEVICE") == "2"

    def test_tp1_no_npus_per_device(self):
        _create_worker(num_devices=1, has_torch_rbln_val=True)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_num_devices_gt1_no_torch_rbln(self):
        """Without torch_rbln, RBLN_NPUS_PER_DEVICE should not be set."""
        _create_worker(num_devices=2, has_torch_rbln_val=False)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_tp4_but_only_2_devices_explicit(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, num_devices=4)
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_device_env_with_dp_rank1_tp2(self):
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, num_devices=2)
        assert os.environ["RBLN_DEVICES"] == "4,5"


# ===========================================================================
# Tests: sleep / wake_up
# ===========================================================================


class TestSleepWakeUp:
    def test_sleep_noop(self):
        worker = _create_worker()
        worker.sleep(level=1)

    def test_sleep_default_level(self):
        worker = _create_worker()
        worker.sleep()

    def test_wake_up_noop(self):
        worker = _create_worker()
        worker.wake_up(tags=["a"])

    def test_wake_up_none_tags(self):
        worker = _create_worker()
        worker.wake_up()


# ===========================================================================
# Tests: initialize_from_config
# ===========================================================================


class TestInitializeFromConfig:
    def test_sets_blocks(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        kv_cfg = MagicMock()
        kv_cfg.num_blocks = 128
        worker.initialize_from_config(kv_cfg)
        assert worker.cache_config.num_gpu_blocks == 128
        assert worker.cache_config.num_cpu_blocks == 128
        worker.model_runner.initialize_kv_cache.assert_called_once_with(kv_cfg)

    def test_zero_blocks(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        kv_cfg = MagicMock()
        kv_cfg.num_blocks = 0
        worker.initialize_from_config(kv_cfg)
        assert worker.cache_config.num_gpu_blocks == 0


# ===========================================================================
# Tests: init_device
# ===========================================================================


class TestInitDevice:
    def _run_init_device(self, worker):
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_cpu_affinity"),
            patch("vllm_rbln.v1.worker.rbln_worker.set_omp_num_threads"),
            patch("numba.set_num_threads"),
            patch("numba.get_num_threads", return_value=2),
            patch("torch.get_num_threads", return_value=2),
            patch("torch.set_num_threads"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.init_worker_distributed_environment"
            ),
            patch("vllm.utils.torch_utils.set_random_seed"),
            patch("vllm_rbln.v1.worker.rbln_worker.RBLNModelRunner") as runner_cls,
            patch("vllm_rbln.v1.worker.rbln_worker.report_usage_stats") as report,
        ):
            runner_cls.return_value = MagicMock()
            worker.init_device()
            return runner_cls, report

    def test_init_device_driver(self):
        worker = _create_worker(rank=0, is_driver_worker=True)
        runner_cls, report = self._run_init_device(worker)
        assert worker.model_runner is runner_cls.return_value
        report.assert_called_once()

    def test_init_device_non_driver(self):
        worker = _create_worker(rank=1, is_driver_worker=False)
        _, report = self._run_init_device(worker)
        report.assert_not_called()


# ===========================================================================
# Tests: load_model
# ===========================================================================


class TestLoadModel:
    def test_delegates_to_runner(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch("vllm.config.set_current_vllm_config"):
            worker.load_model()
        worker.model_runner.load_model.assert_called_once()


# ===========================================================================
# Tests: determine_available_memory
# ===========================================================================


class TestDetermineAvailableMemory:
    def _setup(
        self,
        quantization=None,
        specialized_moe=False,
        bucket_count=1,
        quant_numel=0,
    ):
        cfg = _make_vllm_config(quantization=quantization)
        worker = _create_worker(vllm_config=cfg)

        mock_model = MagicMock()
        params = [
            ("layer.weight", torch.zeros(100, dtype=torch.bfloat16)),
            ("layer.bias", torch.zeros(50, dtype=torch.bfloat16)),
        ]
        if quant_numel > 0:
            params.append(
                ("layer.qweight", torch.zeros(quant_numel, dtype=torch.uint8))
            )
        mock_model.named_parameters.return_value = params

        runner = MagicMock()
        runner.model = mock_model
        runner.specialized_moe_decode = specialized_moe
        runner.bucketing_manager.decode_batch_buckets_count = bucket_count
        worker.model_runner = runner
        return worker

    def test_no_quantization(self):
        worker = self._setup()
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            result = worker.determine_available_memory()
        assert result == 10**9
        # bf16: 100*2 + 50*2 = 300
        assert est.call_args.kwargs["n_model_bytes"] == 300

    def test_fp8(self):
        worker = self._setup(quantization="fp8", quant_numel=64)
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        # bf16: 300; uint8 fp8: 64 * 1 * 1.0 * 8 // 8 = 64
        assert est.call_args.kwargs["n_model_bytes"] == 364

    def test_mxfp4_atom(self):
        worker = self._setup(quantization="mxfp4", quant_numel=64)
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        # bf16: 300; uint8 mxfp4-atom: int(64 * 2 * (16/17) * 16 // 8) = 240
        assert est.call_args.kwargs["n_model_bytes"] == 540

    def test_mxfp4_rebel(self):
        worker = self._setup(quantization="mxfp4", quant_numel=64)
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CR100"
            worker.determine_available_memory()
        # bf16: 300; uint8 mxfp4-rebel: 64 * 2 * 1.0 * 4 // 8 = 64
        assert est.call_args.kwargs["n_model_bytes"] == 364

    def test_mxfp4_unknown_device(self):
        worker = self._setup(quantization="mxfp4")
        with patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat:
            plat.get_device_name.return_value = "RBLN-XX99"
            with pytest.raises(ValueError, match="invalid RBLN architecture"):
                worker.determine_available_memory()

    def test_num_runtimes_with_moe(self):
        worker = self._setup(specialized_moe=True, bucket_count=2)
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        # 1 prefill + 2 normal decodes + 1 padded decode (max bucket only) = 4
        assert est.call_args.kwargs["num_runtimes"] == 4

    def test_mixed_dtype_params(self):
        """bf16 params counted as attention, non-bf16 as experts."""
        cfg = _make_vllm_config(quantization="fp8")
        worker = _create_worker(vllm_config=cfg)
        mock_model = MagicMock()
        p_bf16 = torch.zeros(100, dtype=torch.bfloat16)
        p_quant = torch.zeros(50, dtype=torch.uint8)
        mock_model.named_parameters.return_value = [
            ("attn.weight", p_bf16),
            ("mlp.weight", p_quant),
        ]
        runner = MagicMock()
        runner.model = mock_model
        runner.specialized_moe_decode = False
        runner.bucketing_manager.decode_batch_buckets_count = 1
        worker.model_runner = runner

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        # bf16: 100*2 = 200; uint8 fp8: 50 * 1 * 1.0 * 8 // 8 = 50
        assert est.call_args.kwargs["n_model_bytes"] == 250

    def test_draft_model_kernel_size_is_added(self):
        worker = self._setup()
        draft_model = MagicMock()
        draft_model.parameters.return_value = [
            torch.zeros(40, dtype=torch.bfloat16),
        ]
        worker.model_runner.drafter = SimpleNamespace(model=draft_model)
        worker.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(quantization=None),
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_model_kernel_size",
                side_effect=[400, 120],
            ) as kernel_est,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()

        assert kernel_est.call_count == 2
        assert kernel_est.call_args_list[0].kwargs["n_model_bytes"] == 300
        assert kernel_est.call_args_list[1].kwargs["n_model_bytes"] == 80
        assert est.call_args.kwargs["kernel_size"] == 520
        assert est.call_args.kwargs["num_runtimes"] == 4
        assert "n_model_bytes" not in est.call_args.kwargs

    @pytest.mark.parametrize("quantization", ["fp8", "mxfp4", "compressed-tensors"])
    def test_draft_model_quantization_is_rejected(self, quantization):
        worker = self._setup()
        draft_model = MagicMock()
        draft_model.parameters.return_value = [
            torch.zeros(100, dtype=torch.bfloat16),
            torch.zeros(50, dtype=torch.uint8),
        ]
        worker.model_runner.drafter = SimpleNamespace(model=draft_model)
        worker.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(quantization=quantization),
            draft_parallel_config=None,
        )

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_model_kernel_size",
                return_value=400,
            ),
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            with pytest.raises(
                ValueError,
                match="draft model quantization is not supported",
            ):
                worker.determine_available_memory()


# ===========================================================================
# Tests: compile_or_warm_up_model
# ===========================================================================


class TestCompileOrWarmUpModel:
    """Tests for ``compile_or_warm_up_model``.

    Thread/affinity helpers are mocked via an autouse fixture so that
    ``_ensure_rbln_host_threads_before_compile`` and
    ``_ensure_rbln_cpu_affinity_after_warmup`` never touch the real
    process-global state (os.sched_setaffinity, OMP_NUM_THREADS,
    torch/numba thread counts).
    """

    @pytest.fixture(autouse=True)
    def _mock_thread_affinity(self):
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_cpu_affinity"),
            patch("vllm_rbln.v1.worker.rbln_worker.set_omp_num_threads"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.get_rbln_planned_affinity_cpu_count",
                return_value=4,
            ),
            patch("numba.set_num_threads"),
            patch("numba.get_num_threads", return_value=2),
            patch("torch.get_num_threads", return_value=2),
            patch("torch.set_num_threads"),
        ):
            yield

    def test_skip_enforce_eager(self):
        cfg = _make_vllm_config(enforce_eager=True)
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()
        compielation_times = worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_not_called()
        worker.model_runner._enable_performance_tracker.assert_not_called()
        assert (
            compielation_times.language_model >= 0 and compielation_times.encoder >= 0
        )

    def test_skip_compile_disabled(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_COMPILE_MODEL", False
        ):
            worker.compile_or_warm_up_model()
        worker.model_runner.warmup_model.assert_not_called()

    def test_skip_warmup_disabled(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_ENABLE_WARM_UP", False
        ):
            worker.compile_or_warm_up_model()
        worker.model_runner.warmup_model.assert_not_called()

    def test_warmup_called(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.compile_or_warm_up_model()
        worker.model_runner.warmup_model.assert_called_once()

    def test_oom_enomem(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 64

        inner = RuntimeError("SYS_ENOMEM: Out of memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warmup_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_oom_ebusy(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 32

        inner = RuntimeError("SYS_EBUSY: Lack of device memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warmup_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_non_oom_backend_error(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = RuntimeError("Something else broke")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warmup_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()

    def test_non_runtime_inner_exception(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = TypeError("not a runtime error")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warmup_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()


# ===========================================================================
# Tests: execute_model
# ===========================================================================


class TestExecuteModel:
    def _make_scheduler_output(self, total_tokens=10):
        so = MagicMock()
        so.total_num_scheduled_tokens = total_tokens
        return so

    def test_basic_forward(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        output = MagicMock(spec=ModelRunnerOutput)
        worker.model_runner.execute_model.return_value = output

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = True
            result = worker.execute_model(self._make_scheduler_output())

        assert result is output

    def test_returns_none(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = None

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = True
            result = worker.execute_model(self._make_scheduler_output(0))

        assert result is None

    def test_not_first_rank_receives_tensors(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = MagicMock(
            spec=ModelRunnerOutput
        )

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = False
            pp.return_value.recv_tensor_dict.return_value = {"h": torch.zeros(1)}
            worker.execute_model(self._make_scheduler_output())

        pp.return_value.recv_tensor_dict.assert_called_once()

    def test_intermediate_tensors_sent(self):
        worker = _create_worker()
        it = IntermediateTensors({"hidden": torch.zeros(1)})
        it.kv_connector_output = None
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = it
        worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = True
            pp.return_value.is_last_rank = False
            result = worker.execute_model(self._make_scheduler_output())

        pp.return_value.send_tensor_dict.assert_called_once()
        assert result is None

    @pytest.mark.xfail(reason="unsupported feature")
    def test_kv_connector_finished(self):
        worker = _create_worker()
        kv = MagicMock()
        kv.finished_sending = True
        kv.finished_recving = False
        it = IntermediateTensors({"h": torch.zeros(1)})
        it.kv_connector_output = kv
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = it
        worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = True
            pp.return_value.is_last_rank = False
            result = worker.execute_model(self._make_scheduler_output())

        assert result.kv_connector_output is kv

    @pytest.mark.xfail(reason="unsupported feature")
    def test_kv_connector_not_finished(self):
        worker = _create_worker()
        kv = MagicMock()
        kv.finished_sending = False
        kv.finished_recving = False
        it = IntermediateTensors({"h": torch.zeros(1)})
        it.kv_connector_output = kv
        worker.model_runner = MagicMock()
        worker.model_runner.execute_model.return_value = it
        worker.vllm_config.parallel_config.distributed_executor_backend = "ray"

        with patch("vllm_rbln.v1.worker.rbln_worker.get_pp_group") as pp:
            pp.return_value.is_first_rank = True
            pp.return_value.is_last_rank = False
            result = worker.execute_model(self._make_scheduler_output())

        assert result is EMPTY_MODEL_RUNNER_OUTPUT


# ===========================================================================
# Tests: sample_tokens
# ===========================================================================


class TestSampleTokens:
    def test_delegates(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        grammar = MagicMock()
        worker.sample_tokens(grammar)
        worker.model_runner.sample_tokens.assert_called_once_with(grammar)

    def test_none_grammar(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.sample_tokens(None)
        worker.model_runner.sample_tokens.assert_called_once_with(None)


# ===========================================================================
# Tests: profile
# ===========================================================================


class TestProfile:
    def test_start(self):
        worker = _create_worker()
        worker.profiler = MagicMock()
        worker.profile(is_start=True)
        worker.profiler.start.assert_called_once()

    def test_stop_rank0(self):
        worker = _create_worker(local_rank=0)
        worker.profiler = MagicMock()
        worker.profile(is_start=False)
        worker.profiler.stop.assert_called_once()

    def test_stop_non_rank0(self):
        worker = _create_worker(local_rank=1)
        worker.profiler = MagicMock()
        worker.profile(is_start=False)
        worker.profiler.stop.assert_called_once()


# ===========================================================================
# Tests: LoRA methods
# ===========================================================================


@pytest.mark.skip("unsupported feature")
class TestLoRA:
    def test_add_lora(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.add_lora.return_value = True
        assert worker.add_lora(MagicMock()) is True

    def test_remove_lora(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.remove_lora.return_value = True
        assert worker.remove_lora(42) is True

    def test_list_loras(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.list_loras.return_value = {1, 2}
        assert worker.list_loras() == {1, 2}

    def test_pin_lora(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.pin_lora.return_value = True
        assert worker.pin_lora(7) is True


# ===========================================================================
# Tests: misc methods
# ===========================================================================


class TestMisc:
    def test_check_health(self):
        worker = _create_worker()
        assert worker.check_health() is None

    def test_get_model(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        model = MagicMock()
        worker.model_runner.get_model.return_value = model
        assert worker.get_model() is model

    def test_get_supported_tasks(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.get_supported_tasks.return_value = ("generate",)
        assert worker.get_supported_tasks() == ("generate",)

    def test_take_draft_token_ids(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.take_draft_token_ids.return_value = None
        assert worker.take_draft_token_ids() is None

    def test_get_kv_cache_spec(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        spec = {"layer": MagicMock()}
        worker.model_runner.get_kv_cache_spec.return_value = spec
        assert worker.get_kv_cache_spec() is spec


# ===========================================================================
# Tests: shutdown
# ===========================================================================


class TestShutdown:
    def test_no_metrics(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", False):
            worker.shutdown()

    def test_with_metrics(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.performance_ctx = MagicMock()
        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", True):
            worker.shutdown()
        worker.model_runner.performance_ctx.print_stats.assert_called_once()


# ===========================================================================
# Tests: init_worker_distributed_environment
# ===========================================================================


class TestInitWorkerDistributed:
    def test_single_worker(self):
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        cfg = _make_vllm_config()
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch("vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT", False),
        ):
            init_worker_distributed_environment(
                cfg, rank=0, distributed_init_method="tcp://localhost:1234"
            )

        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "1"

    def test_data_parallel(self):
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        cfg = _make_vllm_config(data_parallel_size=2, world_size=2)
        cfg.parallel_config.data_parallel_rank = 1
        cfg.parallel_config.world_size_across_dp = 4

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch("vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT", False),
        ):
            init_worker_distributed_environment(
                cfg, rank=0, distributed_init_method="tcp://localhost:1234"
            )

        # dp_rank=1, world_size=2, rank=0 => rank_across_dp = 2
        assert os.environ["LOCAL_RANK"] == "2"
        assert os.environ["WORLD_SIZE"] == "4"

    def test_auto_port_with_torch_rbln(self):
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        cfg = _make_vllm_config()
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"
            ) as mock_init,
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT", True),
            patch("vllm_rbln.v1.worker.rbln_worker.has_torch_rbln", True),
        ):
            init_worker_distributed_environment(
                cfg,
                rank=0,
                distributed_init_method="tcp://localhost:1234",
            )

        assert mock_init.call_args.kwargs["backend"] == "rbln-ccl"
        assert os.environ.get("RCCL_PORT_GEN") == "1"

    def test_auto_port_without_torch_rbln(self):
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        cfg = _make_vllm_config()
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce"),
            patch(
                "vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"
            ) as mock_init,
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT", True),
            patch("vllm_rbln.v1.worker.rbln_worker.has_torch_rbln", False),
        ):
            init_worker_distributed_environment(
                cfg,
                rank=0,
                distributed_init_method="tcp://localhost:1234",
                backend="gloo",
            )

        assert mock_init.call_args.kwargs["backend"] == "gloo"

    def test_custom_all_reduce_disabled(self):
        from vllm_rbln.v1.worker.rbln_worker import (
            init_worker_distributed_environment,
        )

        cfg = _make_vllm_config()
        cfg.parallel_config.disable_custom_all_reduce = True

        with (
            patch("vllm_rbln.v1.worker.rbln_worker.set_custom_all_reduce") as mock_car,
            patch("vllm_rbln.v1.worker.rbln_worker.init_distributed_environment"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_model_parallel_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.ensure_kv_transfer_initialized"),
            patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT", False),
        ):
            init_worker_distributed_environment(
                cfg, rank=0, distributed_init_method="tcp://localhost:1234"
            )

        mock_car.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# Dynamic KV: compiler capability probe
# ---------------------------------------------------------------------------
class TestDynamicKvCompilerSupport:
    """`_assert_dynamic_kv_compiler_support` must fire *before* the compile.

    `rebel.kv_cache.max_num_blocks` and `DynamoRuntime.reset_adaptive_buffers`
    both arrived in rebel_compiler #10678 and are only reached after warm-up, so
    without this probe an older compiler pays the whole compile and then dies on
    a bare ImportError.
    """

    @staticmethod
    def _probe():
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        return RBLNWorker._assert_dynamic_kv_compiler_support

    class _RuntimeWithReset:
        def reset_adaptive_buffers(self) -> None:
            pass

    class _RuntimeWithoutReset:
        pass

    @classmethod
    def _modules(cls, *, kv_cache: bool, reset_adaptive: bool) -> dict:
        """A fake `rebel` package with the two symbols independently present.

        `SimpleNamespace` stands in for the modules: `sys.modules` entries only
        need to answer `getattr`, and building them declaratively keeps the fakes
        readable. `rebel` itself is stubbed so the lookup never reaches a real
        installation.
        """
        mods: dict = {"rebel": SimpleNamespace()}
        if kv_cache:
            mods["rebel.kv_cache"] = SimpleNamespace(max_num_blocks=lambda *a, **k: 0)
        mods["rebel.sync_runtime"] = SimpleNamespace(
            DynamoRuntime=(
                cls._RuntimeWithReset if reset_adaptive else cls._RuntimeWithoutReset
            )
        )
        return mods

    def test_passes_when_both_symbols_exist(self):
        mods = self._modules(kv_cache=True, reset_adaptive=True)
        with patch.dict(sys.modules, mods):
            self._probe()(MagicMock())

    def test_rejects_a_compiler_without_max_num_blocks(self):
        mods = self._modules(kv_cache=False, reset_adaptive=True)
        # Block the real module too, so an installed compiler cannot mask this.
        mods["rebel.kv_cache"] = None
        with (
            patch.dict(sys.modules, mods),
            pytest.raises(RuntimeError, match="max_num_blocks"),
        ):
            self._probe()(MagicMock())

    def test_rejects_a_runtime_without_reset_adaptive_buffers(self):
        mods = self._modules(kv_cache=True, reset_adaptive=False)
        with (
            patch.dict(sys.modules, mods),
            pytest.raises(RuntimeError, match="reset_adaptive_buffers"),
        ):
            self._probe()(MagicMock())

    def test_a_missing_runtime_module_is_not_fatal(self):
        """`_assert_dynamo_runtimes` still checks the real objects later."""
        mods = self._modules(kv_cache=True, reset_adaptive=True)
        mods["rebel.sync_runtime"] = None
        with patch.dict(sys.modules, mods):
            self._probe()(MagicMock())

    def test_runs_before_the_other_dynamic_kv_guards(self):
        """Ordering is the point: it must precede anything that compiles.

        `initialize_from_config` calls it first, ahead of the scheduler-handoff
        and KV-connector guards, and the executor runs `initialize_from_config`
        on every rank before any rank's `compile_or_warm_up_model`.
        """
        source = inspect.getsource(
            __import__(
                "vllm_rbln.v1.worker.rbln_worker", fromlist=["RBLNWorker"]
            ).RBLNWorker.initialize_from_config
        )
        probe = source.index("_assert_dynamic_kv_compiler_support")
        handoff = source.index("_assert_dynamic_kv_scheduler_handoff_installed")
        assert probe < handoff


# ---------------------------------------------------------------------------
# Dynamic KV: model / feature shapes the path cannot size
# ---------------------------------------------------------------------------
class TestDynamicKvModelSupport:
    """`_assert_dynamic_kv_model_supported` must fire at construction.

    Both rejections have to land before the model loads, because for spec decode
    that load compiles the drafter. `__init__` is the only hook that precedes it.
    """

    @staticmethod
    def _guard():
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        return RBLNWorker._assert_dynamic_kv_model_supported

    @staticmethod
    def _fake_self(use_mla=False, speculative_config=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(use_mla=use_mla),
            vllm_config=SimpleNamespace(speculative_config=speculative_config),
        )

    def test_plain_model_passes(self):
        self._guard()(self._fake_self())

    def test_mla_is_rejected(self):
        with pytest.raises(RuntimeError, match="does not support MLA"):
            self._guard()(self._fake_self(use_mla=True))

    def test_speculative_decoding_is_rejected(self):
        with pytest.raises(RuntimeError, match="does not support speculative"):
            self._guard()(self._fake_self(speculative_config=SimpleNamespace()))

    def test_fires_from_init_when_the_flag_is_on(self):
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                True,
            ),
            pytest.raises(RuntimeError, match="does not support MLA"),
        ):
            _create_worker(vllm_config=_make_vllm_config(use_mla=True))

    def test_flag_off_rejects_nothing(self):
        cfg = _make_vllm_config(use_mla=True, speculative_config=SimpleNamespace())
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
            False,
        ):
            assert _create_worker(vllm_config=cfg) is not None


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
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

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
        """Drive the method with the flag patched as a module attribute.

        Not via `os.environ`: a setattr elsewhere leaves an attribute shadowing the
        getter (see conftest). The block count is a constant, not injectable.
        """
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks_override=override),
            _kv_blocks_before_shrink=None,
            # The real predicate reads model_config and two env vars; the shrink
            # only asks whether warm-up will run.
            _compile_and_warmup_skip_reason=lambda: (
                "enforce_eager is set" if warmup_skipped else None
            ),
        )
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
                dynamic,
            ),
        ):
            out = RBLNWorker._maybe_shrink_kv_cache_for_compile(worker, config)
        return worker, out

    def test_the_flag_alone_shrinks_to_the_constant(self, caplog):
        from vllm_rbln.v1.worker.rbln_worker import COMPILE_KV_CACHE_NUM_BLOCKS

        config = self._config()
        with caplog.at_level("INFO"):
            worker, out = self._shrink(config)

        assert out is not config
        assert out.num_blocks == COMPILE_KV_CACHE_NUM_BLOCKS
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
        assert caplog.text == ""

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
            self._shrink(self._config(num_blocks=4))

    def test_the_refusal_points_at_block_size_not_at_a_variable(self):
        with pytest.raises(RuntimeError) as exc:
            self._shrink(self._config(num_blocks=4))
        assert "--block-size" in str(exc.value)
        # The variable it used to name is gone; do not resurrect it in prose.
        assert "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS" not in str(exc.value)

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


class TestChargeRemedyWording:
    """What the retained-cache charge tells the operator.

    The hint is a constant, so there is nothing for them to lower: the warning has
    to say the block count is the price of the parallelism, and the exhausted-budget
    refusal has to point at the budget. Both strings are asserted nowhere else.
    """

    BUDGET = 33_772_535_808

    @staticmethod
    def _charge(*, blocks, bytes_per_block, tp=4):
        from vllm_rbln.v1.worker.kv_profile import (
            MergedKvCacheMemoryProfile,
            MergedMemoryRegion,
        )
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        merged = MergedKvCacheMemoryProfile(
            device_regions=[MergedMemoryRegion(0, 0, 0, bytes_per_block, 1)]
        )
        worker = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp),
            model_runner=SimpleNamespace(
                kv_cache_config=SimpleNamespace(num_blocks=blocks)
            ),
        )
        return RBLNWorker._charge_retained_compile_kv_cache(
            worker, {0: {0: TestChargeRemedyWording.BUDGET}}, merged
        )

    def test_the_charge_does_not_offer_a_knob_to_lower(self, caplog):
        with caplog.at_level("WARNING"):
            self._charge(blocks=8, bytes_per_block=65_011_712)
        assert "price of running this parallelism" in caplog.text
        assert "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS" not in caplog.text

    def test_a_budget_the_charge_exhausts_points_at_the_budget(self):
        with pytest.raises(RuntimeError) as exc:
            self._charge(blocks=8, bytes_per_block=self.BUDGET // 4)
        assert "gpu-memory-utilization" in str(exc.value)
        assert "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS" not in str(exc.value)


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
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        calls: list[str] = []
        config = SimpleNamespace(num_blocks=4, kv_cache_tensors=[])
        worker = SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=None, num_cpu_blocks=None),
            vllm_config=object(),
            model_runner=SimpleNamespace(
                initialize_kv_cache=lambda cfg: calls.append("initialize_kv_cache")
            ),
            _assert_dynamic_kv_compiler_support=lambda: calls.append("compiler"),
            _assert_dynamic_kv_scheduler_handoff_installed=lambda: calls.append(
                "handoff"
            ),
            _assert_dynamic_kv_transfer_absent=lambda: calls.append("transfer"),
            _assert_dynamic_kv_attention_layout=lambda: calls.append("attention"),
            _assert_dynamic_kv_cache_layout=lambda: calls.append("bindings"),
            _maybe_shrink_kv_cache_for_compile=lambda cfg: (
                calls.append("shrink") or cfg
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

    def test_a_non_paged_causal_layer_is_refused_by_name(self):
        """`block_size == max_model_len` makes is_normal True -- and is also where
        the estimate can fall below the hint, so the wrong refusal could fire."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

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
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = SimpleNamespace(vllm_config=object())
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.get_layers_from_vllm_config",
            return_value={"layer.0": self._layer()},
        ):
            RBLNWorker._assert_dynamic_kv_attention_layout(worker)

    def test_deduped_bases_are_still_refused_after_the_split(self):
        """Guards the split itself: moving this check earlier would make it see an
        empty list and pass, so its refusal has to stay asserted."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                kv_cache_bases=[object()], shared_kv_cache_layers={}
            )
        )
        with pytest.raises(RuntimeError, match="KV base deduplication"):
            RBLNWorker._assert_dynamic_kv_cache_layout(worker)

    def test_cross_layer_sharing_is_still_refused_after_the_split(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

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
            _assert_dynamo_runtimes=lambda rs: None,
        )

    @staticmethod
    def _call(worker):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
            True,
        ):
            return RBLNWorker.compute_dynamic_kv_num_blocks(worker)

    def test_no_runtime_after_the_shrink_raises(self):
        with pytest.raises(RuntimeError, match="no rbln runtime was registered"):
            self._call(self._worker(runtimes=()))

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
            self._call(self._worker(runtimes=(runtime,)))
        assert "did not return" in str(exc.value) or "not one of the" in str(exc.value)
        assert "VLLM_CACHE_ROOT" in str(exc.value)

    def test_the_pre_shrink_gates_still_return_none(self):
        """Flag off, an override, and "not shrunk" are legitimate: nothing moved."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE",
            False,
        ):
            assert RBLNWorker.compute_dynamic_kv_num_blocks(self._worker()) is None
        assert self._call(self._worker(override=64)) is None
        # The shrink did not happen, so there is nothing to size from.
        assert self._call(self._worker(shrunk=False)) is None


class TestDynamicKvDeviceTensorGuard:
    """VLLM_RBLN_USE_DEVICE_TENSOR=0 is the one configuration that could
    legitimately produce no profile, so it is refused up front rather than left to
    surface as the post-shrink raise above."""

    @staticmethod
    def _assert(*, use_device_tensor):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        worker = SimpleNamespace(
            model_config=SimpleNamespace(use_mla=False),
            vllm_config=SimpleNamespace(speculative_config=None),
        )
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.USE_DEVICE_TENSOR", use_device_tensor
        ):
            RBLNWorker._assert_dynamic_kv_model_supported(worker)

    def test_device_tensor_off_is_refused(self):
        with pytest.raises(RuntimeError, match="VLLM_RBLN_USE_DEVICE_TENSOR=1"):
            self._assert(use_device_tensor=False)

    def test_device_tensor_on_passes(self):
        self._assert(use_device_tensor=True)
