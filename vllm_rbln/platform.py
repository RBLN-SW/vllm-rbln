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

import contextlib
import os
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

import rebel
from torch._dynamo import register_backend
from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum

import vllm_rbln.logger  # noqa: F401
from vllm_rbln import envs

logger = init_logger(__name__)
# Earliest point at which `vllm.envs` is guaranteed to exist, and still before
# any engine code reads a variable.
envs.publish_to_vllm_envs()

try:
    import torch.rbln  # noqa: F401

    HAS_TORCH_RBLN: bool = True
except ImportError:
    HAS_TORCH_RBLN = False

USE_DEVICE_TENSOR: bool = (
    envs.VLLM_RBLN_USE_VLLM_MODEL and envs.VLLM_RBLN_USE_DEVICE_TENSOR
)
# RBLN default for an unset max_num_seqs (upstream vLLM defaults to 256).
RBLN_DEFAULT_MAX_NUM_SEQS = 1


def bypass_backend(graph_module: torch.fx.GraphModule, example_inputs):
    return graph_module.forward


register_backend(name="bypass", compiler_fn=bypass_backend)


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT

    # Compute device_name/device_type/dist_backend once at class definition
    # from env vars so that subprocesses spawned under
    # VLLM_WORKER_MULTIPROC_METHOD=spawn (which re-import this module fresh)
    # observe identical values to the parent without any extra plumbing.
    plugin_name: str = "rbln"
    device_name: str = "rbln" if USE_DEVICE_TENSOR else "cpu"
    device_type: str = "rbln" if USE_DEVICE_TENSOR else "cpu"
    dist_backend: str = "rbln-ccl" if USE_DEVICE_TENSOR else ""
    dispatch_key: str = "CPU"
    ray_device_key: str = "RBLN"
    device_control_env_var: str = "RBLN_DEVICES"
    simple_compile_backend = "bypass"

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend is None:
            selected_backend = (
                AttentionBackendEnum.FLASH_ATTN_MLA
                if attn_selector_config.use_mla
                else AttentionBackendEnum.FLASH_ATTN
            )
        if selected_backend and selected_backend not in (
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASH_ATTN_MLA,
        ):
            raise ValueError(f"Cannot use {selected_backend} backend on RBLN.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on RBLN.")

        logger.info("Using %s Backend", selected_backend)

        return selected_backend.get_path()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # No NPU mounted (e.g., CPU-only compile worker): fall back to the env var
        # the compiler CI sets - RBLN_FORCE_NPU_NAME (RBLN_TARGET_SOC = legacy).
        device_name = (
            rebel.get_npu_name(device_id)
            or os.environ.get("RBLN_FORCE_NPU_NAME")
            or os.environ.get("RBLN_TARGET_SOC")
        )
        if not device_name:
            raise RuntimeError(
                "Could not determine the RBLN NPU name "
                f"(rebel.get_npu_name({device_id}) returned None). On a host "
                "without an NPU mounted (e.g., a CPU-only compile worker running "
                "with VLLM_RBLN_COMPILE_ONLY=1), set RBLN_FORCE_NPU_NAME to the "
                "target NPU (e.g., RBLN-CA25) so compilation can target it."
            )
        return device_name

    @staticmethod
    def inference_mode():
        return torch.no_grad()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        rebel.manual_seed(seed)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        logger.warning("set_device is not supported on RBLN.")
        pass

    @classmethod
    def _override_default_max_num_seqs(cls) -> None:
        """Default an unset max_num_seqs to RBLN_DEFAULT_MAX_NUM_SEQS.

        Wraps EngineArgs.get_batch_defaults() so RBLN's default applies to both
        `vllm serve` and `LLM(...)`. Explicit values are not None and untouched.
        """
        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_rbln_max_num_seqs_patched", False):
            return

        orig_get_batch_defaults = EngineArgs.get_batch_defaults.__func__

        def get_batch_defaults(cls_, world_size):
            from vllm.usage.usage_lib import UsageContext

            default_batched_tokens, _ = orig_get_batch_defaults(cls_, world_size)
            # Cover every usage context plus None (create_engine_config's
            # usage_context is UsageContext | None);
            # otherwise .get(ctx, DEFAULT_MAX_NUM_SEQS) falls through to 128.
            default_max_num_seqs = {
                ctx: RBLN_DEFAULT_MAX_NUM_SEQS for ctx in UsageContext
            }
            default_max_num_seqs[None] = RBLN_DEFAULT_MAX_NUM_SEQS
            return default_batched_tokens, default_max_num_seqs

        EngineArgs.get_batch_defaults = classmethod(get_batch_defaults)
        EngineArgs._rbln_max_num_seqs_patched = True

    @classmethod
    def _capture_user_max_num_batched_tokens(cls) -> None:
        """Stash the user's raw max_num_batched_tokens so the converter can read it.

        In the RBLN optimum path an explicit max_num_batched_tokens IS the
        prefill chunk size, so ``sync_from_vllm`` needs to know whether the user
        set it. By the time that runs it can no longer tell, because vLLM has
        already overwritten the value:

          1. The user passes ``max_num_batched_tokens`` (an int) or leaves it
             ``None``.
          2. ``_set_default_max_num_seqs_and_batched_tokens_args`` replaces a
             ``None`` with a throughput default and, since chunked prefill is
             off on RBLN, floors it up to ``max_model_len``.
          3. ``VllmConfig.__post_init__`` calls ``check_and_update_config`` ->
             ``sync_from_vllm``, which now sees a concrete number with no trace
             of whether it came from the user or from step 2.

        This wrapper runs at the start of step 2, before the overwrite, and
        records the raw value (``None`` if unset) into ``additional_config``,
        which flows unchanged into ``VllmConfig``. ``sync_from_vllm`` then reads
        it via ``get_user_max_num_batched_tokens``.
        """
        from vllm.engine.arg_utils import EngineArgs

        from vllm_rbln.utils.optimum.converter.common import (
            USER_MAX_NUM_BATCHED_TOKENS_KEY,
        )

        if getattr(EngineArgs, "_rbln_user_mnbt_patched", False):
            return

        orig_set_defaults = EngineArgs._set_default_max_num_seqs_and_batched_tokens_args

        def _set_default_max_num_seqs_and_batched_tokens_args(self, *args, **kwargs):
            # Runs before the value is resolved from None to its default.
            if self.additional_config is None:
                self.additional_config = {}
            self.additional_config[USER_MAX_NUM_BATCHED_TOKENS_KEY] = (
                self.max_num_batched_tokens
            )
            return orig_set_defaults(self, *args, **kwargs)

        EngineArgs._set_default_max_num_seqs_and_batched_tokens_args = (
            _set_default_max_num_seqs_and_batched_tokens_args
        )
        EngineArgs._rbln_user_mnbt_patched = True

    @classmethod
    def pre_register_and_update(
        cls, parser: "FlexibleArgumentParser | None" = None
    ) -> None:
        # Runs before max_num_seqs is resolved from None to its default.
        cls._override_default_max_num_seqs()
        cls._capture_user_max_num_batched_tokens()

        if parser is None:
            return

        for action in parser._actions:
            if action.dest == "device":
                action.choices.append("rbln")

        for action in parser._actions:
            if action.dest == "block_size":
                action.choices = None  # Override choices

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm_rbln.utils.optimum.converter import sync_vllm_and_optimum
        from vllm_rbln.utils.optimum.predicates import forces_fp32_dtype
        from vllm_rbln.utils.optimum.registry import is_pooling_arch

        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError(
                "VLLM_USE_V2_MODEL_RUNNER is not supported for RBLN backend."
            )

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.async_scheduling:
            logger.warning(
                "Asynchronous scheduling is not supported on RBLN. "
                "Overriding scheduler_config.async_scheduling to False."
            )
            scheduler_config.async_scheduling = False

        # NOTE(RBLN): checked here, not in `validate_and_setup_prerequisite` --
        # that runs only inside the vLLM-native branch below, and the optimum
        # path is exactly where an unsupported flag would go unnoticed.
        if envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE:
            cls._validate_dynamic_kv_config(vllm_config)

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if vllm_config.lora_config is not None:
                raise ValueError("LoRA is not supported on RBLN.")

            cls.validate_and_setup_prerequisite(vllm_config)

            if envs.VLLM_RBLN_ENFORCE_MODEL_FP32:
                if model_config.dtype != torch.float32:
                    # FIXME(RBLN): force model dtype into fp32 for graph compilation
                    original_dtype = model_config.dtype
                    model_config.dtype = torch.float32
                    logger.info(
                        "Overriding model_config.dtype from %s to %s.",
                        original_dtype,
                        model_config.dtype,
                    )
            else:
                if model_config.dtype not in (
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ):
                    logger.warning(
                        "Unsupported dtype for RBLN: %s. Falling back to %s. "
                        "Supported dtypes are torch.float32, torch.float16, "
                        "and torch.bfloat16.",
                        model_config.dtype,
                        torch.float32,
                    )
                    model_config.dtype = torch.float32

            logger.info("Using model_config.dtype for RBLN: %s", model_config.dtype)

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = (
                    "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"
                )
            scheduler_config.scheduler_cls = (
                "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"
            )

            # Under PP the compiled per-stage decode batch is max_num_seqs // pp_size
            # (see decode_batch_size). Fail fast on an impossible config.
            pp_size = parallel_config.pipeline_parallel_size
            if pp_size > 1:
                max_num_seqs = scheduler_config.max_num_seqs
                if max_num_seqs < pp_size:
                    raise ValueError(
                        f"pipeline_parallel_size={pp_size} requires "
                        f"max_num_seqs >= {pp_size} (got {max_num_seqs}); "
                        f"per-stage decode batch would floor to 0."
                    )
                if max_num_seqs % pp_size != 0:
                    logger.warning(
                        "max_num_seqs=%d is not a multiple of "
                        "pipeline_parallel_size=%d; %d decode slot(s) will be unused.",
                        max_num_seqs,
                        pp_size,
                        max_num_seqs % pp_size,
                    )
                logger.info_once(
                    "pipeline_parallel_size=%d, max_num_seqs=%d -> "
                    "per-stage decode batch=%d.",
                    pp_size,
                    max_num_seqs,
                    max_num_seqs // pp_size,
                )

            # FIXME(jiwoo.park) This is a temporary workaround.
            if model_config.enforce_eager:
                if not USE_DEVICE_TENSOR:
                    raise ValueError(
                        "enforce_eager=True requires VLLM_RBLN_USE_DEVICE_TENSOR=1. "
                        "Eager mode bypasses torch.compile, so ops must dispatch "
                        "to a real device='rbln' rather than the compile-backend "
                        "fake-CPU tensors used by the default vLLM model path."
                    )

                hf_config = vllm_config.model_config.hf_config
                assert not hasattr(hf_config, "sliding_window") or not getattr(
                    hf_config, "use_sliding_window", True
                )

                # RBLN(NOTE): force dtype into fp16 for eager mode
                model_config.dtype = torch.float16

            from vllm.config import CompilationMode

            if vllm_config.compilation_config.mode != CompilationMode.NONE:
                logger.info(
                    "vLLM compilation mode is not used on RBLN because "
                    "@support_torch_compile is not supported. "
                    "Overriding compilation_config.mode from %s to %s.",
                    vllm_config.compilation_config.mode,
                    CompilationMode.NONE,
                )
                vllm_config.compilation_config.mode = CompilationMode.NONE
                if (
                    len(vllm_config.compilation_config.custom_ops) == 1
                    and vllm_config.compilation_config.custom_ops[0] == "none"
                ):
                    logger.debug(
                        "Clearing compilation_config.custom_ops because "
                        "vLLM compilation mode is disabled on RBLN."
                    )
                    vllm_config.compilation_config.custom_ops = []

            if not model_config.disable_cascade_attn:
                logger.warning(
                    "Cascade attention is not supported on RBLN. "
                    "Overriding model_config.disable_cascade_attn to True."
                )
                model_config.disable_cascade_attn = True

        else:
            if forces_fp32_dtype(vllm_config.model_config):
                model_config.dtype = torch.float32

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = (
                    "vllm_rbln.v1.worker.optimum_worker.RBLNOptimumWorker"
                )
            scheduler_config.scheduler_cls = (
                "vllm_rbln.v1.core.optimum_scheduler.RBLNOptimumScheduler"
            )

            assert vllm_config.parallel_config.tensor_parallel_size == 1, (
                "Cannot set tensor_parallel_size for pre-compiled optimum-rbln models. "
                "If you want to compile with tensor parallelism in vllm-rbln, "
                "please use the `VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK` "
                "environment variable instead."
            )
            assert vllm_config.parallel_config.pipeline_parallel_size == 1, (
                "Pipeline parallelism is not supported in optimum-rbln."
            )
            assert vllm_config.speculative_config is None, (
                "Speculative decoding is not supported in optimum-rbln."
            )
            # T5EncoderModel is encoder-only but inherits T5Config which has
            # is_encoder_decoder=True. This causes vllm to route inputs
            # through the enc-dec path, prepending decoder_start_token_id and
            # breaking CLS pooling. Set it to False for pooling models.
            # ModelConfig.is_encoder_decoder is a @cached_property that's
            # already evaluated by this point, so invalidate the cache too.
            hf_config = model_config.hf_config
            if is_pooling_arch(hf_config) and getattr(
                hf_config, "is_encoder_decoder", False
            ):
                hf_config.is_encoder_decoder = False
                with contextlib.suppress(KeyError):
                    del model_config.__dict__["is_encoder_decoder"]

            cls.disable_unsupported_prefix_caching(vllm_config)
            sync_vllm_and_optimum(vllm_config)

        if (
            parallel_config.distributed_executor_backend is not None
            and parallel_config.distributed_executor_backend != "mp"
        ):
            logger.warning(
                (
                    "%s is not supported on RBLN, fallback to mp "
                    "distributed executor backend."
                ),
                parallel_config.distributed_executor_backend,
            )

    @staticmethod
    def _validate_dynamic_kv_config(vllm_config: VllmConfig) -> None:
        """Reject configurations the dynamic-KV path cannot size.

        Reasons per shape: docs/dynamic_kv_cache.md, "Unsupported
        Configurations".
        """
        if not envs.VLLM_RBLN_USE_VLLM_MODEL:
            raise ValueError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1 requires "
                "VLLM_RBLN_USE_VLLM_MODEL=1; see docs/dynamic_kv_cache.md."
            )

        if vllm_config.model_config.use_mla:
            raise ValueError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support MLA models. "
                "Run with the flag off, or with VLLM_MLA_DISABLE=1."
            )

        if vllm_config.speculative_config is not None:
            raise ValueError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE does not support speculative "
                "decoding; the merged profiles cannot be attributed per artifact."
            )

        if not USE_DEVICE_TENSOR:
            raise ValueError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE requires "
                "VLLM_RBLN_USE_DEVICE_TENSOR=1; without it the artifact carries "
                "no dynamic KV dimension."
            )

        if vllm_config.kv_transfer_config is not None:
            raise ValueError(
                "VLLM_RBLN_USE_DYNAMIC_KV_CACHE cannot be combined with a KV "
                "transfer connector; the resize invalidates its registrations."
            )

    @classmethod
    def register_custom_kv_cache_specs(cls, vllm_config: "VllmConfig") -> None:
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        from vllm_rbln.v1.kv_cache import (
            RBLNSlidingWindowManager,
            RBLNSlidingWindowSpec,
        )

        KVCacheSpecRegistry.register(
            RBLNSlidingWindowSpec,
            RBLNSlidingWindowManager,
            uniform_type_base_spec=RBLNSlidingWindowSpec,
        )

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on RBLN.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_rbln.distributed.rbln_communicator.RblnCommunicator"  # noqa

    @classmethod
    def validate_and_setup_prerequisite(cls, vllm_config: VllmConfig) -> None:
        scheduler_config = vllm_config.scheduler_config
        if not scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "Disabling chunked prefill is not supported on RBLN. "
                "Please enable chunked prefill by yourself."
            )

        if envs.VLLM_RBLN_COMPILE_ONLY:
            # Compile-only injects the compile_only torch.compile option. The
            # optimum-rbln path is not torch.compile-based, so the flag has no
            # meaning there and conflicts with that path; it only applies to the
            # vLLM-native (torch.compile) path, which VLLM_RBLN_USE_VLLM_MODEL
            # selects.
            if not envs.VLLM_RBLN_USE_VLLM_MODEL:
                raise ValueError(
                    "VLLM_RBLN_COMPILE_ONLY=1 is a torch.compile option and only "
                    "applies to the vLLM-native model path; set "
                    "VLLM_RBLN_USE_VLLM_MODEL=1 to use it. The optimum-rbln path "
                    "is not torch.compile-based, so compile-only conflicts with "
                    "it."
                )
            if envs.VLLM_DISABLE_COMPILE_CACHE:
                # Compile-only compiles each graph and writes the .rbln artifact
                # to the compile cache (the runtime is built on a dummy device
                # so no NPU is needed). With the cache disabled there is nowhere
                # to write the artifact, so the two options are mutually
                # exclusive.
                raise ValueError(
                    "VLLM_RBLN_COMPILE_ONLY=1 needs the compile cache enabled "
                    "to write compiled artifacts to disk; do not set "
                    "VLLM_DISABLE_COMPILE_CACHE=1 together with it."
                )

        parallel_config = vllm_config.parallel_config
        use_model_parallel = (
            parallel_config.tensor_parallel_size > 1
            or parallel_config.pipeline_parallel_size > 1
            or parallel_config.data_parallel_size > 1
            or parallel_config.enable_expert_parallel
        )
        if use_model_parallel:
            if (
                parallel_config.data_parallel_size > 1
                and scheduler_config.max_num_batched_tokens
                % scheduler_config.max_num_seqs
                != 0
            ):
                raise ValueError(
                    "max_num_batched_tokens must be divisible by max_num_seqs "
                    "when DP enabled."
                )

            if (
                parallel_config.data_parallel_size > 1
                or parallel_config.enable_expert_parallel
            ) and not envs.VLLM_RBLN_USE_MOE_TOKENS_MASK:
                raise ValueError(
                    "VLLM_RBLN_USE_MOE_TOKENS_MASK is required when DP or EP enabled: "
                    "the mask marks padded tokens introduced by DP multicast. "
                    "Set VLLM_RBLN_USE_MOE_TOKENS_MASK=1 (default)."
                )

            os.environ["RBLN_CTX_STANDALONE"] = "1"
            if os.environ.get("RBLN_RUNTIME_FORCE_SYNC") == "1":
                logger.warning(
                    "RBLN_RUNTIME_FORCE_SYNC=1 forces the synchronous runtime, "
                    "which may cause performance degradation "
                    "when using vLLM model parallel (TP, DP, EP, or PP)."
                )

    @classmethod
    def _disable_prefix_caching(cls, vllm_config: VllmConfig, reason: str) -> None:
        """Disable prefix caching with warning message."""
        logger.warning(
            "Prefix caching is not available for %s. "
            "It has been automatically disabled.",
            reason,
        )
        vllm_config.cache_config.enable_prefix_caching = False

    @staticmethod
    def _uses_sliding_window(hf_config) -> bool:
        """Whether any layer uses sliding-window attention. Reads the text
        sub-config (multimodal composites nest it), honors a
        ``use_sliding_window=False`` opt-out, and treats a sliding ``layer_types``
        entry as sliding. Errs toward True (disabling prefix caching is safe).
        """
        config = (
            hf_config.get_text_config()
            if hasattr(hf_config, "get_text_config")
            else hf_config
        )
        # use_sliding_window is a Qwen2-only opt-out flag; models without it
        # (Gemma/Mistral) are judged by sliding_window/layer_types, so default
        # to True (no opt-out) to avoid short-circuiting their detection.
        if not getattr(config, "use_sliding_window", True):
            return False
        if getattr(config, "sliding_window", None) is not None:
            return True
        layer_types = getattr(config, "layer_types", None) or []
        return any("sliding" in str(layer_type).lower() for layer_type in layer_types)

    @classmethod
    def disable_unsupported_prefix_caching(cls, vllm_config: VllmConfig) -> None:
        from vllm_rbln.utils.optimum.predicates import (
            is_qwen3_embedding,
            is_qwen3_reranker,
        )
        from vllm_rbln.utils.optimum.registry import (
            is_enc_dec_arch,
            is_pooling_arch,
        )

        if not vllm_config.cache_config.enable_prefix_caching:
            return
        # An EC producer runs only the (vision) encoder and never executes the
        # LLM, so it holds no KV cache. Prefix caching there is a no-op and its
        # KV-cache manager is only a placeholder, so disable it explicitly.
        ec = getattr(vllm_config, "ec_transfer_config", None)
        if ec is not None and ec.is_ec_producer and not ec.is_ec_consumer:
            cls._disable_prefix_caching(vllm_config, "EC producer (encoder-only)")
            return

        hf_config = vllm_config.model_config.hf_config
        has_sliding_window = cls._uses_sliding_window(hf_config)

        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if has_sliding_window:
                cls._disable_prefix_caching(vllm_config, "sliding window models")

        else:
            # Prefix caching is supported only for decoder-only models for now.
            model_config = vllm_config.model_config
            if is_qwen3_embedding(model_config) or is_qwen3_reranker(model_config):
                # Qwen3 pooling model does not support prefix caching for now.
                cls._disable_prefix_caching(vllm_config, "Qwen3 pooling models")
            elif is_enc_dec_arch(hf_config):
                cls._disable_prefix_caching(vllm_config, "encoder-decoder models")
            elif is_pooling_arch(hf_config):
                cls._disable_prefix_caching(vllm_config, "pooling models")
            elif has_sliding_window:
                cls._disable_prefix_caching(vllm_config, "sliding window models")

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_rbln.lora.punica_wrapper.punica_rbln.PunicaWrapperRBLN"

    @classmethod
    def can_update_inplace(cls) -> bool:
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        # kv_buffer_device "cpu" is the host-bounce path; "rbln" is the D2D
        # path (upstream NixlConnectorWorker.__init__ rejects kv_buffer_device
        # values not listed here). Listed under both device_types because
        # device_type is "rbln" only when VLLM_RBLN_USE_DEVICE_TENSOR and
        # VLLM_RBLN_USE_VLLM_MODEL are both set.
        return {
            "cpu": ("cpu", "rbln"),
            "rbln": ("rbln", "cpu"),
        }

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        return "DRAM"

    @classmethod
    def discover_numa_topology(cls) -> list[list[int]]:
        """
        Discover NUMA topology and keep the last physical core of each numa
        into one core group list for nixl start_kv_load()
        """
        return []

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Set some additional forward context for the current platform if needs.
        """
        additional_kwargs: dict[str, Any] = {}
        if "kv_cache_bases" in kwargs:
            additional_kwargs["kv_cache_bases"] = kwargs["kv_cache_bases"]

        return additional_kwargs
