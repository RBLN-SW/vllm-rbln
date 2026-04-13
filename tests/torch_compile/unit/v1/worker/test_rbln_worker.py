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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiler_config(trace_dir=None):
    return SimpleNamespace(
        torch_profiler_dir=trace_dir,
        torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
        torch_profiler_with_flops=False,
        torch_profiler_use_gzip=False,
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
):
    return SimpleNamespace(
        trust_remote_code=trust_remote_code,
        seed=seed,
        quantization=quantization,
        enforce_eager=enforce_eager,
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
        ),
        cache_config=_make_cache_config(),
        scheduler_config=_make_scheduler_config(),
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
    "envs_tp": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_TP_SIZE",
    "envs_ray": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_NUM_RAY_NODES",
    "envs_auto_port": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_AUTO_PORT",
    "envs_compile": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_COMPILE_MODEL",
    "envs_warmup": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_ENABLE_WARM_UP",
    "envs_metrics": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS",
    "envs_dp_impl": "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DP_IMPL",
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


def _create_worker(
    vllm_config=None,
    local_rank=0,
    rank=0,
    is_driver_worker=True,
    *,
    tp_size=1,
    num_ray_nodes=1,
    has_torch_rbln_val=False,
    envs_overrides=None,
):
    """Instantiate RBLNWorker with mocked-out heavy dependencies."""
    from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

    if vllm_config is None:
        vllm_config = _make_vllm_config()

    defaults = {
        "envs_tp": tp_size,
        "envs_ray": num_ray_nodes,
        "envs_auto_port": False,
        "envs_compile": True,
        "envs_warmup": True,
        "envs_metrics": False,
        "envs_dp_impl": "padded_decode",
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
            "envs_tp",
            "envs_ray",
            "envs_auto_port",
            "envs_compile",
            "envs_warmup",
            "envs_metrics",
            "envs_dp_impl",
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
        KNOWN_GAPS = {"get_cache_block_size_bytes"}

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

    def test_compile_or_warm_up_model_returns_float(self):
        """compile_or_warm_up_model must return a float (elapsed time)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.compile_or_warm_up_model)
        # The return annotation should be float
        assert (
            sig.return_annotation is float
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

    def test_initialize_cache_signature(self):
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.initialize_cache)
        params = list(sig.parameters.keys())
        assert "num_gpu_blocks" in params
        assert "num_cpu_blocks" in params

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
        assert worker._sleep_saved_buffers == {}

    def test_init_with_profiler(self):
        cfg = _make_vllm_config(profiler_trace_dir="/tmp/test_trace")
        with (
            patch("torch.profiler.profile"),
            patch("torch.profiler.tensorboard_trace_handler"),
        ):
            worker = _create_worker(vllm_config=cfg)
        assert worker.profiler is not None

    def test_init_trust_remote_code(self):
        cfg = _make_vllm_config(trust_remote_code=True)
        mock_init_hf = MagicMock()
        with patch.dict(
            "sys.modules",
            {"vllm.utils.import_utils": MagicMock(init_cached_hf_modules=mock_init_hf)},
        ):
            _create_worker(vllm_config=cfg)
        mock_init_hf.assert_called_once()

    def test_local_world_size(self):
        cfg = _make_vllm_config(world_size=4)
        worker = _create_worker(vllm_config=cfg)
        assert worker.local_world_size == 4


# ===========================================================================
# Tests: _init_device_env
# ===========================================================================


class TestInitDeviceEnv:
    def test_auto_device_single(self):
        _create_worker()
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_tp4_single_worker(self):
        _create_worker(tp_size=4)
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_auto_device_multi(self):
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1)
        assert os.environ["RBLN_DEVICES"] == "1"

    def test_tp4_multi_worker_rank1(self):
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1, tp_size=4)
        assert os.environ["RBLN_DEVICES"] == "4,5,6,7"

    def test_multiple_ray_nodes(self):
        cfg = _make_vllm_config(world_size=8)
        worker = _create_worker(
            vllm_config=cfg, local_rank=0, rank=0, tp_size=1, num_ray_nodes=2
        )
        assert worker.local_world_size == 4
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_multiple_ray_nodes_rank3(self):
        cfg = _make_vllm_config(world_size=8)
        _create_worker(
            vllm_config=cfg, local_rank=3, rank=3, tp_size=1, num_ray_nodes=2
        )
        assert os.environ["RBLN_DEVICES"] == "3"

    def test_dp_rank_offsets_device_ids(self):
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, tp_size=1)
        assert os.environ["RBLN_DEVICES"] == "2"

    def test_explicit_device_ids(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=0)
        assert os.environ["RBLN_DEVICES"] == "0"

    def test_explicit_rbln_devices_tp_rank1(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=1, rank=1, tp_size=2)
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

    def test_tp_size_gt1_sets_npus_env(self):
        _create_worker(tp_size=2, has_torch_rbln_val=True)
        assert os.environ.get("RBLN_NPUS_PER_DEVICE") == "2"

    def test_tp1_no_npus_per_device(self):
        _create_worker(tp_size=1, has_torch_rbln_val=True)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_tp_size_gt1_no_torch_rbln(self):
        """Without torch_rbln, RBLN_NPUS_PER_DEVICE should not be set."""
        _create_worker(tp_size=2, has_torch_rbln_val=False)
        assert "RBLN_NPUS_PER_DEVICE" not in os.environ

    def test_local_world_size_not_divisible(self):
        cfg = _make_vllm_config(world_size=7)
        worker = _create_worker(vllm_config=cfg, num_ray_nodes=2)
        assert worker.local_world_size == 3

    def test_local_world_size_single_ray_node(self):
        cfg = _make_vllm_config(world_size=4)
        worker = _create_worker(vllm_config=cfg, num_ray_nodes=1)
        assert worker.local_world_size == 4

    def test_tp4_but_only_2_devices_explicit(self):
        os.environ["RBLN_DEVICES"] = "0,1"
        cfg = _make_vllm_config(world_size=2)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, tp_size=4)
        assert os.environ["RBLN_DEVICES"] == "0,1,2,3"

    def test_device_env_with_dp_rank1_tp2(self):
        cfg = _make_vllm_config(world_size=2, data_parallel_rank=1)
        _create_worker(vllm_config=cfg, local_rank=0, rank=0, tp_size=2)
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
# Tests: initialize_cache
# ===========================================================================


class TestInitializeCache:
    def test_sets_blocks(self):
        worker = _create_worker()
        worker.initialize_cache(128, 64)
        assert worker.cache_config.num_gpu_blocks == 128
        assert worker.cache_config.num_cpu_blocks == 64

    def test_zero_blocks(self):
        worker = _create_worker()
        worker.initialize_cache(0, 0)
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
    def _setup(self, quantization=None, specialized_moe=False, bucket_count=1):
        cfg = _make_vllm_config(quantization=quantization)
        worker = _create_worker(vllm_config=cfg)

        mock_model = MagicMock()
        p1 = torch.zeros(100, dtype=torch.bfloat16)
        p2 = torch.zeros(50, dtype=torch.bfloat16)
        mock_model.named_parameters.return_value = [
            ("layer.weight", p1),
            ("layer.bias", p2),
        ]

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
        assert est.call_args.kwargs["nbits_per_param"] == 16

    def test_fp8(self):
        worker = self._setup(quantization="fp8")
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        assert est.call_args.kwargs["nbits_per_param"] == 8

    def test_mxfp4_atom(self):
        worker = self._setup(quantization="mxfp4")
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CA25"
            worker.determine_available_memory()
        assert est.call_args.kwargs["nbits_per_param"] == 16

    def test_mxfp4_rebel(self):
        worker = self._setup(quantization="mxfp4")
        with (
            patch("vllm_rbln.v1.worker.rbln_worker.current_platform") as plat,
            patch(
                "vllm_rbln.v1.worker.rbln_worker.estimate_available_memory",
                return_value=10**9,
            ) as est,
        ):
            plat.get_device_name.return_value = "RBLN-CR100"
            worker.determine_available_memory()
        assert est.call_args.kwargs["nbits_per_param"] == 4

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
        # 1 + (1 + 1) * 2 = 5
        assert est.call_args.kwargs["num_runtimes"] == 5

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
        # n_model_params = 100 (bf16) + 50*1*1 (uint8 fp8 packed=1)
        assert est.call_args.kwargs["n_model_params"] == 150


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
        elapsed = worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_not_called()
        worker.model_runner._enable_performance_tracker.assert_not_called()
        assert elapsed >= 0

    def test_skip_compile_disabled(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_COMPILE_MODEL", False
        ):
            worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_not_called()

    def test_skip_warmup_disabled(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        with patch(
            "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_ENABLE_WARM_UP", False
        ):
            worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_not_called()

    def test_warmup_called(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.compile_or_warm_up_model()
        worker.model_runner.warm_up_model.assert_called_once()

    def test_oom_enomem(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 64

        inner = RuntimeError("SYS_ENOMEM: Out of memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_oom_ebusy(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.kv_cache_config.num_blocks = 32

        inner = RuntimeError("SYS_EBUSY: Lack of device memory")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(RuntimeError, match="Not enough memory"):
            worker.compile_or_warm_up_model()

    def test_non_oom_backend_error(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = RuntimeError("Something else broke")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()

    def test_non_runtime_inner_exception(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()

        inner = TypeError("not a runtime error")
        exc = BackendCompilerFailed(MagicMock(), inner, None)
        exc.inner_exception = inner
        worker.model_runner.warm_up_model.side_effect = exc

        with pytest.raises(BackendCompilerFailed):
            worker.compile_or_warm_up_model()

    def test_dp_padded_decode(self):
        cfg = _make_vllm_config(data_parallel_size=2)
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()
        worker.compile_or_warm_up_model()
        worker.model_runner.prepare_dummy_run.assert_called_once()

    def test_dp_padded_decode_not_divisible(self):
        cfg = _make_vllm_config(data_parallel_size=2)
        cfg.scheduler_config.max_num_batched_tokens = 100
        cfg.scheduler_config.max_num_seqs = 33
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()
        with pytest.raises(AssertionError, match="divisible"):
            worker.compile_or_warm_up_model()

    def test_dp_dummy_prefill_raises(self):
        cfg = _make_vllm_config(data_parallel_size=2)
        worker = _create_worker(vllm_config=cfg)
        worker.model_runner = MagicMock()
        with (
            patch(
                "vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_DP_IMPL",
                "dummy_prefill",
            ),
            pytest.raises(ValueError, match="dummy_prefill is not supported"),
        ):
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
    def test_no_profiler_raises(self):
        worker = _create_worker()
        worker.profiler = None
        with pytest.raises(RuntimeError, match="Profiler is not enabled"):
            worker.profile(is_start=True)

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
        worker.profiler.key_averages.assert_called_once()

    def test_stop_non_rank0(self):
        worker = _create_worker(local_rank=1)
        worker.profiler = MagicMock()
        worker.profile(is_start=False)
        worker.profiler.stop.assert_called_once()
        worker.profiler.key_averages.assert_not_called()


# ===========================================================================
# Tests: LoRA methods
# ===========================================================================


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

    def test_execute_dummy_batch(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.execute_dummy_batch()
        worker.model_runner.dummy_run.assert_called_once()

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

    def test_initialize_from_config(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        kv_cfg = MagicMock()
        worker.initialize_from_config(kv_cfg)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(kv_cfg)


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
        worker.model_runner.performance_tracker = MagicMock()
        worker.model_runner.sampler_performance_tracker = MagicMock()
        worker.model_runner.e2e_performance_tracker = MagicMock()
        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", True):
            worker.shutdown()
        worker.model_runner.performance_tracker.print_final_stats.assert_called_once()
        worker.model_runner.sampler_performance_tracker.print_final_stats.assert_called_once()
        worker.model_runner.e2e_performance_tracker.print_final_stats.assert_called_once()

    def test_with_metrics_none_trackers(self):
        worker = _create_worker()
        worker.model_runner = MagicMock()
        worker.model_runner.performance_tracker = None
        worker.model_runner.sampler_performance_tracker = None
        worker.model_runner.e2e_performance_tracker = None
        with patch("vllm_rbln.v1.worker.rbln_worker.envs.VLLM_RBLN_METRICS", True):
            worker.shutdown()


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
