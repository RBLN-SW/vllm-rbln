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

# RblnPlatform on the native path only (VLLM_RBLN_USE_VLLM_MODEL=1). Written
# against outcomes rather than call paths -- a real config is built so the engine's
# own entry point (VllmConfig.__post_init__ -> check_and_update_config) does the
# work, and the assertions read the resulting config, the raised error, or the
# process env. That survives splitting the optimum and native paths apart.

from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_rbln.platform as platform
from tests.native.vllm_config import local_model_path
from vllm_rbln.platform import (
    RBLN_DEFAULT_MAX_NUM_SEQS,
    RblnPlatform,
)

# Small, non-gated and already needed by the spec-decode tests; a config build
# never touches the device.
_MODEL = "JackFram/llama-68m"
_ENGINE_ARGS = dict(
    max_model_len=2048,
    block_size=1024,
    max_num_batched_tokens=128,
    enable_chunked_prefill=True,
)

_STANDALONE = "RBLN_CTX_STANDALONE"
# Means "the platform did not touch it". Set rather than deleted so monkeypatch
# has recorded the key and can roll back the platform's direct os.environ write.
_UNTOUCHED = ""


def _build(**engine_kwargs) -> VllmConfig:
    """A real config, so building it *is* running the platform hook.

    Not vllm_config.make_vllm_config: that pins max_num_seqs, which this file
    needs unset to observe the RBLN default.
    """
    return EngineArgs(
        model=local_model_path(_MODEL), **{**_ENGINE_ARGS, **engine_kwargs}
    ).create_engine_config()


@pytest.fixture(scope="module")
def configured() -> VllmConfig:
    """One plain native config, built once."""
    return _build()


@pytest.fixture
def reconfigure(configured):
    """Copy the built config, mutate it, run the hook again.

    For branches EngineArgs cannot express -- an unsupported dtype, chunked
    prefill off (upstream rejects it first), a LoRA config, more ranks. Re-running
    is safe because every field the hook writes is already at its final value.
    """

    def run(mutate) -> VllmConfig:
        config = copy.deepcopy(configured)
        mutate(config)
        RblnPlatform.check_and_update_config(config)
        return config

    return run


@pytest.fixture(autouse=True)
def isolated_standalone_env(monkeypatch):
    monkeypatch.setenv(_STANDALONE, _UNTOUCHED)


class TestPlatformIdentity:
    """Values frozen at import: the class body derives them from the env, so a
    spawned worker re-importing the module must land on the same lane."""

    def test_device_triple_moves_together(self):
        # A device_type of "rbln" with an empty dist_backend (or the reverse)
        # would half-enable the device lane.
        triple = (
            RblnPlatform.device_name,
            RblnPlatform.device_type,
            RblnPlatform.dist_backend,
        )
        if platform.USE_DEVICE_TENSOR:
            assert triple == ("rbln", "rbln", "rbln-ccl")
        else:
            assert triple == ("cpu", "cpu", "")

    def test_device_tensor_needs_both_switches(self):
        assert platform.USE_DEVICE_TENSOR is (
            platform.envs.VLLM_RBLN_USE_VLLM_MODEL
            and platform.envs.VLLM_RBLN_USE_DEVICE_TENSOR
        )

    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            # Ops dispatch on CPU even when tensors live on the device.
            ("dispatch_key", "CPU"),
            # RBLNWorker._init_device_env narrows this var per rank.
            ("device_control_env_var", "RBLN_DEVICES"),
            ("simple_compile_backend", "bypass"),
        ],
    )
    def test_pinned_attributes(self, attribute, expected):
        assert getattr(RblnPlatform, attribute) == expected

    def test_simple_compile_backend_is_registered(self):
        # Named at class scope; an unregistered name only fails once vLLM
        # compiles something.
        assert torch._dynamo.lookup_backend(RblnPlatform.simple_compile_backend)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("is_pin_memory_available", False),
            ("can_update_inplace", False),
            ("support_hybrid_kv_cache", True),
            ("get_nixl_memory_type", "DRAM"),
            (
                "get_device_communicator_cls",
                "vllm_rbln.distributed.rbln_communicator.RblnCommunicator",
            ),
        ],
    )
    def test_pinned_answers(self, method, expected):
        assert getattr(RblnPlatform, method)() == expected

    def test_nixl_supported_devices(self):
        # Upstream rejects any kv_buffer_device not listed here, so both the
        # host-bounce ("cpu") and D2D ("rbln") pairings must stay.
        assert RblnPlatform.get_nixl_supported_devices() == {
            "cpu": ("cpu", "rbln"),
            "rbln": ("rbln", "cpu"),
        }


class TestRejectedConfigs:
    def test_v2_model_runner(self, monkeypatch, reconfigure):
        monkeypatch.setattr(platform.envs, "VLLM_USE_V2_MODEL_RUNNER", True)
        with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER"):
            reconfigure(lambda config: None)

    def test_lora(self, reconfigure):
        with pytest.raises(ValueError, match="LoRA"):
            reconfigure(lambda config: setattr(config, "lora_config", object()))

    def test_chunked_prefill_off(self, reconfigure):
        with pytest.raises(ValueError, match="chunked prefill"):
            reconfigure(
                lambda config: setattr(
                    config.scheduler_config, "enable_chunked_prefill", False
                )
            )

    def test_eagle3_under_pp_needs_a_patched_target(self, reconfigure):
        # The default model is a plain LlamaForCausalLM, whose forward still
        # collects aux hidden states with a stage-local index. Asserting through
        # the hook rather than on the validator directly is the point: it is what
        # shows the guard is reached at all.
        with pytest.raises(ValueError, match="EAGLE3 with pipeline_parallel_size"):
            reconfigure(_eagle3_under_pp())

    def test_eagle3_under_pp_accepts_a_patched_target(self, reconfigure):
        reconfigure(_eagle3_under_pp(arch="MiniMaxM2ForCausalLM"))

    def test_eagle3_under_pp_accepts_a_draft_with_aux_off(self, reconfigure):
        # Nothing is captured anywhere then, so upstream's forward is harmless.
        reconfigure(_eagle3_under_pp(eagle_config={"use_aux_hidden_state": False}))

    def test_eagle3_at_pp1_is_not_gated(self, reconfigure):
        reconfigure(_eagle3_under_pp(pp_size=1))

    def test_dp_needs_a_divisible_token_budget(self, reconfigure):
        with pytest.raises(ValueError, match="divisible"):
            reconfigure(_ranks(data_parallel_size=2, max_num_seqs=5))

    @pytest.mark.parametrize("ranks", [dict(data_parallel_size=2), dict(ep=True)])
    def test_dp_and_ep_need_the_moe_tokens_mask(self, monkeypatch, reconfigure, ranks):
        monkeypatch.setattr(platform.envs, "VLLM_RBLN_USE_MOE_TOKENS_MASK", False)
        with pytest.raises(ValueError, match="VLLM_RBLN_USE_MOE_TOKENS_MASK"):
            reconfigure(_ranks(**ranks))

    def test_tp_inherits_neither_dp_rule(self, monkeypatch, reconfigure):
        # Both rules guard padding introduced by DP multicast, so TP alone must
        # pass even with the mask off and an indivisible budget.
        monkeypatch.setattr(platform.envs, "VLLM_RBLN_USE_MOE_TOKENS_MASK", False)
        reconfigure(_ranks(tensor_parallel_size=2, max_num_seqs=5))

    def test_moe_tokens_mask_defaults_on(self):
        # The error above calls 1 the default; a flipped default breaks DP.
        assert platform.envs.VLLM_RBLN_USE_MOE_TOKENS_MASK is True


def _eagle3_under_pp(*, arch: str | None = None, eagle_config=None, pp_size: int = 2):
    """A mutator that puts an EAGLE3 draft on a pipeline-parallel target.

    EngineArgs would have to resolve a real draft checkpoint to build this, so the
    speculative config is a stand-in shaped like the fields the guard reads.
    """

    def mutate(config: VllmConfig) -> None:
        config.parallel_config.pipeline_parallel_size = pp_size
        # The guard runs after the per-stage decode batch check, which would
        # otherwise raise first and mask it.
        config.scheduler_config.max_num_seqs = pp_size * 2
        if arch is not None:
            config.model_config.hf_config.architectures = [arch]
        config.speculative_config = SimpleNamespace(
            method="eagle3",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_config=eagle_config)
            ),
        )

    return mutate


def _ranks(*, ep: bool = False, max_num_seqs: int | None = None, **parallel):
    """A mutator that widens a config to more ranks."""

    def mutate(config: VllmConfig) -> None:
        for field, value in parallel.items():
            setattr(config.parallel_config, field, value)
        if ep:
            config.parallel_config.enable_expert_parallel = True
        if max_num_seqs is not None:
            config.scheduler_config.max_num_seqs = max_num_seqs

    return mutate


class TestModelParallelSideEffect:
    @pytest.mark.parametrize(
        "ranks",
        [
            dict(data_parallel_size=2),
            dict(tensor_parallel_size=2),
            # PP compiles max_num_seqs // pp_size decode slots per stage, so the
            # budget must be >= pp_size (the default of 1 would floor to 0).
            dict(pipeline_parallel_size=2, max_num_seqs=2),
            dict(ep=True),
        ],
        ids=["dp", "tp", "pp", "ep"],
    )
    def test_any_model_parallel_axis_sets_standalone_context(self, reconfigure, ranks):
        reconfigure(_ranks(**ranks))
        assert os.environ[_STANDALONE] == "1"

    def test_single_rank_leaves_it_alone(self, reconfigure):
        reconfigure(lambda config: None)
        assert os.environ[_STANDALONE] == _UNTOUCHED


class TestDtype:
    def test_supported_dtype_is_kept(self):
        assert _build(dtype="float16").model_config.dtype == torch.float16

    def test_unsupported_dtype_falls_back_to_fp32(self, reconfigure):
        config = reconfigure(
            lambda config: setattr(config.model_config, "dtype", torch.float64)
        )
        assert config.model_config.dtype == torch.float32

    def test_enforce_fp32_overrides_a_supported_dtype(self, monkeypatch, reconfigure):
        monkeypatch.setattr(platform.envs, "VLLM_RBLN_ENFORCE_MODEL_FP32", True)
        config = reconfigure(
            lambda config: setattr(config.model_config, "dtype", torch.float16)
        )
        assert config.model_config.dtype == torch.float32


class TestWorkerAndScheduler:
    def test_auto_worker_becomes_the_rbln_worker(self, configured):
        assert (
            configured.parallel_config.worker_cls
            == "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"
        )

    def test_an_explicit_worker_is_respected(self):
        assert (
            _build(worker_cls="pkg.mod.MyWorker").parallel_config.worker_cls
            == "pkg.mod.MyWorker"
        )

    def test_scheduler_is_replaced_unconditionally(self, monkeypatch, reconfigure):
        # Unlike worker_cls there is no "auto" guard: whatever was asked for is
        # overwritten. Reading the expectation back off the config under test
        # would agree with whatever the platform decided, so the carriers are
        # pinned off and the sync scheduler named outright.
        monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
        config = reconfigure(
            lambda config: setattr(
                config.scheduler_config, "scheduler_cls", "pkg.mod.MyScheduler"
            )
        )
        assert (
            config.scheduler_config.scheduler_cls
            == "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"
        )

    def test_a_plain_build_lands_on_the_async_scheduler(self, monkeypatch):
        # Nobody passes --async-scheduling here: vLLM resolves the unset flag to
        # True before this platform hook, and with both carriers on nothing
        # refuses it, so the native path selects the async scheduler. This is
        # what the async support changed, and it went unasserted.
        for name in ("VLLM_RBLN_USE_DEVICE_TENSOR", "VLLM_RBLN_SAMPLER"):
            monkeypatch.setenv(name, "1")
        config = _build()
        assert config.scheduler_config.async_scheduling is True
        assert (
            config.scheduler_config.scheduler_cls
            == "vllm_rbln.v1.core.rbln_scheduler.RBLNAsyncScheduler"
        )


class TestCompilation:
    def test_vllm_compilation_is_disabled(self, configured):
        # @support_torch_compile is unsupported; RBLN compiles in its own runner.
        assert configured.compilation_config.mode == CompilationMode.NONE

    def test_a_lone_none_custom_op_is_cleared(self):
        assert (
            _build(
                compilation_config={"custom_ops": ["none"]}
            ).compilation_config.custom_ops
            == []
        )

    def test_only_a_lone_none_is_cleared(self):
        # Two entries fall outside the guard, so "none" survives as upstream
        # meant it. (The default list is upstream's business, not ours.)
        assert _build(
            compilation_config={"custom_ops": ["none", "+rms_norm"]}
        ).compilation_config.custom_ops == ["none", "+rms_norm"]


class TestSchedulerOverrides:
    def test_async_scheduling_is_honored(self, monkeypatch):
        # The platform used to force this off unconditionally. It now follows
        # vLLM's --async-scheduling, as long as the device-side token path is
        # available (see below). Both carriers are pinned on because
        # --device-tensor 0 switches the first one off for the whole session,
        # which would land this on the negative case below.
        for name in ("VLLM_RBLN_USE_DEVICE_TENSOR", "VLLM_RBLN_SAMPLER"):
            monkeypatch.setenv(name, "1")
        assert _build(async_scheduling=True).scheduler_config.async_scheduling is True

    @pytest.mark.parametrize(
        "switched_off", ["VLLM_RBLN_USE_DEVICE_TENSOR", "VLLM_RBLN_SAMPLER"]
    )
    def test_async_scheduling_needs_the_device_token_carriers(
        self, monkeypatch, switched_off
    ):
        """Either env var off means async has no way to carry its in-flight tokens.

        VLLM_RBLN_USE_DEVICE_TENSOR gates the feedback scatter that replaces the
        scheduler's -1 placeholders; VLLM_RBLN_SAMPLER gates the ring the output
        thread reads. Without them the runner decodes from a token that was never
        sampled and returns wrong text with no error raised, so the platform
        downgrades to sync rather than run the combination.
        """
        # Both are set before one is switched off: the gate refuses on either,
        # so leaving the other to the lane lets --device-tensor 0 satisfy this
        # case with the sampler still on, and the parametrization proves nothing.
        for name in ("VLLM_RBLN_USE_DEVICE_TENSOR", "VLLM_RBLN_SAMPLER"):
            monkeypatch.setenv(name, "1")
        monkeypatch.setenv(switched_off, "0")
        config = _build(async_scheduling=True)
        assert config.scheduler_config.async_scheduling is False
        assert config.scheduler_config.scheduler_cls.endswith("RBLNScheduler")

    def test_async_scheduling_is_refused_with_speculative_decoding(self, reconfigure):
        """The async feedback carries one sampled token per step.

        _bookkeeping_sync asserts a single sampled column, which a rejection
        sampler output of shape (batch, num_spec + 1) cannot satisfy. vLLM allows
        async with eagle / ngram / draft_model, so without this the combination
        reaches the runner and dies on that assert mid-decode.
        """

        def mutate(config: VllmConfig) -> None:
            config.scheduler_config.async_scheduling = True
            # eagle is a method vLLM does allow async scheduling with.
            config.speculative_config = SimpleNamespace(method="eagle")

        config = reconfigure(mutate)
        assert config.scheduler_config.async_scheduling is False
        assert config.scheduler_config.scheduler_cls.endswith("RBLNScheduler")

    def test_async_scheduling_is_refused_under_pipeline_parallelism(self, reconfigure):
        """Under PP the scheduler stops shipping the sampled tokens.

        It expects the runner to broadcast prev_sampled_token_ids from the last
        stage instead, which this runner does not do, so a non-last rank reaches
        _update_states with no token source and dies on its assert mid-decode.
        """

        def mutate(config: VllmConfig) -> None:
            config.scheduler_config.async_scheduling = True
            _ranks(pipeline_parallel_size=2, max_num_seqs=2)(config)

        config = reconfigure(mutate)
        assert config.scheduler_config.async_scheduling is False
        assert config.scheduler_config.scheduler_cls.endswith("RBLNScheduler")

    def test_cascade_attention_is_disabled(self, configured):
        assert configured.model_config.disable_cascade_attn is True

    def test_unset_max_num_seqs_takes_the_rbln_default(self, configured):
        assert configured.scheduler_config.max_num_seqs == RBLN_DEFAULT_MAX_NUM_SEQS

    def test_an_explicit_max_num_seqs_is_respected(self):
        assert _build(max_num_seqs=8).scheduler_config.max_num_seqs == 8


class TestEnforceEager:
    def test_outcome_follows_the_device_lane(self, reconfigure):
        mutate = lambda config: setattr(  # noqa: E731
            config.model_config, "enforce_eager", True
        )
        if platform.USE_DEVICE_TENSOR:
            # Eager needs real device tensors; dtype is forced to fp16 there.
            assert reconfigure(mutate).model_config.dtype == torch.float16
        else:
            with pytest.raises(ValueError, match="VLLM_RBLN_USE_DEVICE_TENSOR"):
                reconfigure(mutate)


def _selector(*, use_mla: bool = False, use_sparse: bool = False) -> SimpleNamespace:
    return SimpleNamespace(use_mla=use_mla, use_sparse=use_sparse)


class TestAttentionBackend:
    @pytest.mark.parametrize(
        ("use_mla", "expected"),
        [
            (False, AttentionBackendEnum.FLASH_ATTN),
            (True, AttentionBackendEnum.FLASH_ATTN_MLA),
        ],
    )
    def test_unset_backend_resolves_by_mla(self, use_mla, expected):
        assert (
            RblnPlatform.get_attn_backend_cls(None, _selector(use_mla=use_mla))
            == expected.get_path()
        )

    @pytest.mark.parametrize(
        "backend",
        [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.FLASH_ATTN_MLA],
    )
    def test_an_allowed_backend_passes_through(self, backend):
        assert (
            RblnPlatform.get_attn_backend_cls(backend, _selector())
            == backend.get_path()
        )

    def test_any_other_backend_is_rejected(self):
        other = next(
            backend
            for backend in AttentionBackendEnum
            if backend
            not in (
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASH_ATTN_MLA,
            )
        )
        with pytest.raises(ValueError, match="Cannot use"):
            RblnPlatform.get_attn_backend_cls(other, _selector())

    def test_sparse_attention_is_unsupported(self):
        with pytest.raises(NotImplementedError, match="Sparse"):
            RblnPlatform.get_attn_backend_cls(
                AttentionBackendEnum.FLASH_ATTN, _selector(use_sparse=True)
            )


class TestDeviceName:
    """The NPU name drives compilation, so a compile-only host without an NPU
    must still be able to name its target."""

    @pytest.fixture(autouse=True)
    def no_host_override(self, monkeypatch):
        # Both are host passthrough vars the suite does not scrub.
        monkeypatch.delenv("RBLN_FORCE_NPU_NAME", raising=False)
        monkeypatch.delenv("RBLN_TARGET_SOC", raising=False)

    def test_the_driver_answer_wins(self, monkeypatch):
        monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-FROM-ENV")
        monkeypatch.setattr(platform.rebel, "get_npu_name", lambda *a: "RBLN-CR03")
        assert RblnPlatform.get_device_name() == "RBLN-CR03"

    @pytest.mark.parametrize("env", ["RBLN_FORCE_NPU_NAME", "RBLN_TARGET_SOC"])
    def test_env_fallbacks_when_no_npu_is_mounted(self, monkeypatch, env):
        monkeypatch.setattr(platform.rebel, "get_npu_name", lambda *a: None)
        monkeypatch.setenv(env, "RBLN-CA25")
        assert RblnPlatform.get_device_name() == "RBLN-CA25"

    def test_no_npu_and_no_override_raises(self, monkeypatch):
        monkeypatch.setattr(platform.rebel, "get_npu_name", lambda *a: None)
        with pytest.raises(RuntimeError, match="RBLN_FORCE_NPU_NAME"):
            RblnPlatform.get_device_name()


class TestAdditionalForwardContext:
    def test_kv_cache_bases_passes_through(self):
        bases = object()
        assert RblnPlatform.set_additional_forward_context(kv_cache_bases=bases) == {
            "kv_cache_bases": bases
        }

    @pytest.mark.parametrize("kwargs", [{}, {"kv_bases": 1}, {"attn_metadata": 1}])
    def test_everything_else_is_dropped(self, kwargs):
        assert RblnPlatform.set_additional_forward_context(**kwargs) == {}


class TestPreRegisterAndUpdate:
    def test_parser_gains_rbln_and_loses_block_size_choices(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--device", choices=["cuda", "cpu"])
        parser.add_argument("--block-size", dest="block_size", choices=[8, 16])

        RblnPlatform.pre_register_and_update(parser)

        device, block_size = parser._actions[1], parser._actions[2]
        assert device.choices is not None and "rbln" in device.choices
        # RBLN block sizes are not from upstream's list.
        assert block_size.choices is None

    def test_the_engine_args_patches_are_idempotent(self):
        # Applied at plugin load already; wrapping twice would stack wrappers.
        # __func__ because each attribute access rebinds the classmethod.
        before = EngineArgs.get_batch_defaults.__func__
        RblnPlatform.pre_register_and_update()
        assert EngineArgs.get_batch_defaults.__func__ is before


class TestCustomKvCacheSpecs:
    @pytest.fixture(autouse=True)
    def restore_registry(self):
        # The registry is a process global with lazy init: it populates itself
        # only while empty, so leaving it half-filled makes every later test that
        # builds a scheduler fail with "No manager registered for ...".
        from vllm.v1 import kv_cache_spec_registry as registry

        saved = dict(registry._REGISTRY_KVCACHESPEC_LIST)
        yield
        registry._REGISTRY_KVCACHESPEC_LIST.clear()
        registry._REGISTRY_KVCACHESPEC_LIST.update(saved)

    def test_the_sliding_window_manager_is_reachable(self, configured):
        # The runner looks the manager up by spec at KV-cache init; an
        # unregistered pair only fails there, on a device. Registered through
        # register_all_kvcache_specs like production does -- it fills in the
        # built-in specs first and calls the platform hook last.
        from vllm.v1.core.single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        from vllm_rbln.v1.kv_cache import (
            RBLNSlidingWindowManager,
            RBLNSlidingWindowSpec,
        )

        register_all_kvcache_specs(configured)
        spec = RBLNSlidingWindowSpec(
            block_size=1024,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.float16,
            sliding_window=512,
        )
        assert KVCacheSpecRegistry.get_manager_class(spec) is RBLNSlidingWindowManager


def test_running_the_hook_twice_changes_nothing(configured, reconfigure):
    """The premise every reconfigure() test rests on: each field the hook writes
    is already at its final value, so a second pass is a no-op. Should the hook
    start appending instead of assigning, this fails first."""
    again = reconfigure(lambda config: None)

    assert again.model_config.dtype == configured.model_config.dtype
    assert again.parallel_config.worker_cls == configured.parallel_config.worker_cls
    assert (
        again.scheduler_config.scheduler_cls
        == configured.scheduler_config.scheduler_cls
    )
    assert again.compilation_config.mode == configured.compilation_config.mode
    assert (
        again.compilation_config.custom_ops == configured.compilation_config.custom_ops
    )
    assert (
        again.cache_config.enable_prefix_caching
        == configured.cache_config.enable_prefix_caching
    )


class TestKnownGaps:
    """Behaviour pinned as-is because it looks unintended; see
    docs/test_note.log."""

    def test_sliding_window_keeps_prefix_caching_on(self):
        # disable_unsupported_prefix_caching is only called from the optimum
        # branch, so its native clause is unreachable. SWA instead surfaces much
        # later as the sub-block multi-group NotImplementedError in
        # RBLNModelRunner.initialize_kv_cache, and the workaround in use is
        # VLLM_RBLN_SUB_BLOCK_CACHE=0.
        config = _build(
            enable_prefix_caching=True, hf_overrides={"sliding_window": 512}
        )
        assert RblnPlatform._uses_sliding_window(config.model_config.hf_config)
        assert config.cache_config.enable_prefix_caching is True

    def test_a_non_mp_executor_backend_is_only_warned_about(self, configured):
        # The warning says "fallback to mp" but nothing assigns, and vLLM's own
        # default for world_size 1 is "uni" -- so it fires on every single-process
        # run and Executor.get_class still builds a UniProcExecutor.
        assert configured.parallel_config.distributed_executor_backend == "uni"


class TestDynamicKvConfig:
    """VLLM_RBLN_USE_DYNAMIC_KV_CACHE is validated at config time, before the
    model loads; the worker keeps only the checks that read runtime state."""

    @staticmethod
    def _cfg(use_mla=False, speculative_config=None, kv_transfer_config=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(use_mla=use_mla),
            speculative_config=speculative_config,
            kv_transfer_config=kv_transfer_config,
        )

    @pytest.fixture(autouse=True)
    def _native_lane(self, monkeypatch):
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "1")

    def test_a_clean_config_passes(self):
        RblnPlatform._validate_dynamic_kv_config(self._cfg())

    def test_needs_the_vllm_model_path(self, monkeypatch):
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "0")
        with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL=1"):
            RblnPlatform._validate_dynamic_kv_config(self._cfg())

    def test_mla_is_rejected(self):
        with pytest.raises(ValueError, match="MLA"):
            RblnPlatform._validate_dynamic_kv_config(self._cfg(use_mla=True))

    def test_speculative_decoding_is_rejected(self):
        with pytest.raises(ValueError, match="speculative"):
            RblnPlatform._validate_dynamic_kv_config(
                self._cfg(speculative_config=SimpleNamespace())
            )

    def test_a_kv_transfer_connector_is_rejected(self):
        with pytest.raises(ValueError, match="KV transfer"):
            RblnPlatform._validate_dynamic_kv_config(
                self._cfg(kv_transfer_config=SimpleNamespace())
            )

    def test_device_tensor_off_is_refused(self):
        with (
            patch("vllm_rbln.platform.USE_DEVICE_TENSOR", False),
            pytest.raises(ValueError, match="VLLM_RBLN_USE_DEVICE_TENSOR=1"),
        ):
            RblnPlatform._validate_dynamic_kv_config(self._cfg())

    def test_the_hook_validates_only_under_the_flag(self, monkeypatch, reconfigure):
        seen: list = []
        with patch.object(
            RblnPlatform,
            "_validate_dynamic_kv_config",
            side_effect=lambda cfg: seen.append(cfg),
        ):
            monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "0")
            reconfigure(lambda config: None)
            assert seen == []
            monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "1")
            reconfigure(lambda config: None)
            assert len(seen) == 1


class TestDflashTokenBudget:
    """DFlash reserves no drafting slots, so the auto-computed budget is the
    whole of `max_num_batched_tokens`; anything else was set explicitly, and no
    other prefill chunk lands on a KV block boundary."""

    def _mutate(self, scheduled):
        def mutate(config: VllmConfig) -> None:
            config.speculative_config = SimpleNamespace(method="dflash")
            config.scheduler_config.max_num_scheduled_tokens = scheduled

        return mutate

    def test_the_auto_computed_budget_is_accepted(self, reconfigure, configured):
        budget = configured.scheduler_config.max_num_batched_tokens
        config = reconfigure(self._mutate(budget))
        assert config.scheduler_config.max_num_scheduled_tokens == budget

    @pytest.mark.parametrize("delta", [-1, -8, 1])
    def test_any_other_budget_is_refused(self, reconfigure, configured, delta):
        budget = configured.scheduler_config.max_num_batched_tokens
        with pytest.raises(ValueError, match="auto-computed"):
            reconfigure(self._mutate(budget + delta))

    def test_only_dflash_is_gated(self, reconfigure, configured):
        def mutate(config: VllmConfig) -> None:
            config.speculative_config = SimpleNamespace(method="eagle3")
            config.scheduler_config.max_num_scheduled_tokens = 8

        assert reconfigure(mutate).scheduler_config.max_num_scheduled_tokens == 8
