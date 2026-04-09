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

"""Feature tests for RBLNWorker: interface compliance, WorkerBase contract,
device env initialization, and edge cases."""

import inspect
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.worker.worker_base import WorkerBase

# ---------------------------------------------------------------------------
# Helpers (mirrors the unit test factory pattern)
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

    def test_sleep_wake_up_are_defined(self):
        """sleep and wake_up must be defined on RBLNWorker."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        assert hasattr(RBLNWorker, "sleep")
        assert hasattr(RBLNWorker, "wake_up")
        assert callable(RBLNWorker.sleep)
        assert callable(RBLNWorker.wake_up)


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

    def test_determine_available_memory_returns_int(self):
        """determine_available_memory should return int (bytes)."""
        from vllm_rbln.v1.worker.rbln_worker import RBLNWorker

        sig = inspect.signature(RBLNWorker.determine_available_memory)
        ret = sig.return_annotation
        assert ret is int or ret == inspect.Parameter.empty

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
