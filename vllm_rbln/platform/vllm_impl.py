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

"""The vllm model path's half of the platform hooks."""

import os
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.version import __version_tuple__ as VLLM_VERSION

from vllm_rbln import envs
from vllm_rbln.platform import USE_DEVICE_TENSOR

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_rbln.config import RBLNConfig

logger = init_logger(__name__)


def patch_upstream() -> None:
    """The vllm model path replaces no upstream symbol from the platform hook."""


def check_and_update(vllm_config: "VllmConfig") -> None:
    from vllm_rbln.config import build_rbln_config

    # Everything below reads additional_config as an RBLNConfig.
    vllm_config.additional_config = build_rbln_config(vllm_config.additional_config)

    _validate(vllm_config)
    _override(vllm_config)
    _wire(vllm_config)
    _setup_runtime_env(vllm_config)


def _uses_model_parallel(parallel_config) -> bool:
    return (
        parallel_config.tensor_parallel_size > 1
        or parallel_config.pipeline_parallel_size > 1
        or parallel_config.data_parallel_size > 1
        or parallel_config.enable_expert_parallel
    )


def _validate(vllm_config: "VllmConfig") -> None:
    rbln_config: RBLNConfig = vllm_config.additional_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config

    if vllm_config.lora_config is not None:
        raise ValueError("LoRA is not supported on RBLN.")

    if not scheduler_config.enable_chunked_prefill:
        raise ValueError(
            "Disabling chunked prefill is not supported on RBLN. "
            "Please enable chunked prefill by yourself."
        )

    if envs.VLLM_RBLN_COMPILE_ONLY and envs.VLLM_DISABLE_COMPILE_CACHE:
        # Compile-only compiles each graph and writes the .rbln artifact to
        # the compile cache (the runtime is built on a dummy device so no
        # NPU is needed). With the cache disabled there is nowhere to write
        # the artifact, so the two options are mutually exclusive.
        raise ValueError(
            "VLLM_RBLN_COMPILE_ONLY=1 needs the compile cache enabled "
            "to write compiled artifacts to disk; do not set "
            "VLLM_DISABLE_COMPILE_CACHE=1 together with it."
        )

    if _uses_model_parallel(parallel_config):
        if (
            parallel_config.data_parallel_size > 1
            and scheduler_config.max_num_batched_tokens % scheduler_config.max_num_seqs
            != 0
        ):
            raise ValueError(
                "max_num_batched_tokens must be divisible by max_num_seqs "
                "when DP enabled."
            )

        if (
            parallel_config.data_parallel_size > 1
            or parallel_config.enable_expert_parallel
        ) and not rbln_config.use_moe_tokens_mask:
            raise ValueError(
                "VLLM_RBLN_USE_MOE_TOKENS_MASK is required when DP or EP enabled: "
                "the mask marks padded tokens introduced by DP multicast. "
                "Set VLLM_RBLN_USE_MOE_TOKENS_MASK=1 (default)."
            )

    if (
        vllm_config.speculative_config is not None
        and vllm_config.speculative_config.method == "dflash"
        and scheduler_config.max_num_scheduled_tokens
        != scheduler_config.max_num_batched_tokens
    ):
        # DFlash reserves no slots (see patches/speculative_config.py),
        # so the auto-computed budget is the whole of
        # max_num_batched_tokens and any other value was set explicitly.
        raise ValueError(
            "DFlash needs max_num_scheduled_tokens left auto-computed "
            f"(expected {scheduler_config.max_num_batched_tokens}, got "
            f"{scheduler_config.max_num_scheduled_tokens}): the prefill "
            "chunk has to stay at max_num_batched_tokens, which is also "
            "the compiled prefill length and the sub-block cache's "
            "granularity."
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
            "pipeline_parallel_size=%d, max_num_seqs=%d -> per-stage decode batch=%d.",
            pp_size,
            max_num_seqs,
            max_num_seqs // pp_size,
        )

        _validate_eagle3_pp_config(vllm_config)

    # FIXME(jiwoo.park) This is a temporary workaround.
    if model_config.enforce_eager:
        if not USE_DEVICE_TENSOR:
            raise ValueError(
                "enforce_eager=True requires VLLM_RBLN_USE_DEVICE_TENSOR=1. "
                "Eager mode bypasses torch.compile, so ops must dispatch "
                "to a real device='rbln' rather than the compile-backend "
                "fake-CPU tensors used by the default vLLM model path."
            )

        hf_config = model_config.hf_config
        assert not hasattr(hf_config, "sliding_window") or not getattr(
            hf_config, "use_sliding_window", True
        )


def _validate_eagle3_pp_config(vllm_config: "VllmConfig") -> None:
    """Reject an EAGLE3 target whose aux collection is not pipeline-aware.

    Called only when `pipeline_parallel_size > 1`. Upstream's model `forward`
    indexes the aux capture with a stage-local `enumerate` and drops the list
    on every non-last stage, so a target outside `EAGLE3_PP_TARGET_ARCHS`
    harvests the wrong layers and reaches the drafter short. That surfaces as a
    shape mismatch mid-compile, or -- where the counts happen to line up -- not
    at all, as a silently worse draft.

    TODO(vllm-project/vllm#50514): delete once that lands and is released.
    """
    from vllm_rbln.v1.spec_decode.eagle3_pp import (
        EAGLE3_PP_TARGET_ARCHS,
        eagle3_aux_hidden_states_enabled,
    )

    # A draft with `use_aux_hidden_state` off captures nothing, so upstream's
    # forward is harmless and the split is fine.
    if not eagle3_aux_hidden_states_enabled(vllm_config.speculative_config):
        return

    architectures = vllm_config.model_config.hf_config.architectures or []
    if set(architectures) & EAGLE3_PP_TARGET_ARCHS:
        return

    raise ValueError(
        "EAGLE3 with pipeline_parallel_size="
        f"{vllm_config.parallel_config.pipeline_parallel_size} is supported on "
        f"RBLN only for target architectures "
        f"{sorted(EAGLE3_PP_TARGET_ARCHS)}, but got {list(architectures)}. "
        "Collecting the target's auxiliary hidden states across pipeline "
        "stages needs a per-architecture patch that this target does not have "
        "yet. Run this target with pipeline_parallel_size=1, or with a draft "
        "whose eagle_config sets use_aux_hidden_state=false."
    )


def _override(vllm_config: "VllmConfig") -> None:
    rbln_config: RBLNConfig = vllm_config.additional_config
    from vllm.config import CompilationMode

    model_config = vllm_config.model_config
    spec_config = vllm_config.speculative_config

    if rbln_config.enforce_model_fp32:
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

    if (
        spec_config is not None
        and model_config.hf_text_config.model_type == "deepseek_v32"
    ):
        # TODO(vllm>=0.29.0): delete this block; vllm#52861 stops upstream
        # forcing v32 MTP eager, which leaves the reset below dead.
        assert VLLM_VERSION < (0, 29), (
            f"vLLM {VLLM_VERSION} ships vllm#52861; delete the deepseek_v32 "
            "MTP enforce_eager reset."
        )
        if not model_config.enforce_eager and spec_config.enforce_eager:
            spec_config.enforce_eager = False

    if model_config.enforce_eager:
        # RBLN(NOTE): force dtype into fp16 for eager mode
        model_config.dtype = torch.float16

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


def _wire(vllm_config: "VllmConfig") -> None:
    rbln_config: RBLNConfig = vllm_config.additional_config
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config

    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = "vllm_rbln.v1.worker.rbln_worker.RBLNWorker"

    # The async refusals have to precede the scheduler_cls assignment below,
    # which reads the flag they clear.
    if scheduler_config.async_scheduling and not (
        envs.VLLM_RBLN_USE_DEVICE_TENSOR and rbln_config.use_custom_sampler
    ):
        logger.warning(
            "Disabling asynchronous scheduling: it requires "
            "VLLM_RBLN_USE_DEVICE_TENSOR=1 (got %s), which carries the "
            "in-flight sampled tokens, and VLLM_RBLN_SAMPLER=1 (got %s), "
            "which puts the sampler on the device so those tokens never "
            "reach the host mid-step. Running synchronously.",
            int(envs.VLLM_RBLN_USE_DEVICE_TENSOR),
            int(rbln_config.use_custom_sampler),
        )
        scheduler_config.async_scheduling = False

    # TODO(yskim): Support speculative decoding on RBLN under async scheduling.
    if scheduler_config.async_scheduling and vllm_config.speculative_config is not None:
        logger.warning(
            "Disabling asynchronous scheduling: speculative decoding is "
            "not supported on RBLN under async scheduling, because the "
            "async path feeds one sampled token per step back into the "
            "next. Running synchronously."
        )
        scheduler_config.async_scheduling = False

    # TODO(yskim): Support PP on RBLN under async scheduling.
    if scheduler_config.async_scheduling and parallel_config.pipeline_parallel_size > 1:
        logger.warning(
            "Disabling asynchronous scheduling: pipeline parallelism is "
            "not supported on RBLN under async scheduling. Under PP the "
            "scheduler stops propagating sampled tokens and expects the "
            "runner to broadcast prev_sampled_token_ids from the last "
            "stage, which this runner does not do. Running synchronously."
        )
        scheduler_config.async_scheduling = False

    if scheduler_config.async_scheduling:
        # Only RBLNAsyncScheduler bumps num_output_placeholders at
        # schedule time, which is what lets the batch_queue fill.
        scheduler_config.scheduler_cls = (
            "vllm_rbln.v1.core.rbln_scheduler.RBLNAsyncScheduler"
        )
    else:
        scheduler_config.scheduler_cls = (
            "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"
        )


def _setup_runtime_env(vllm_config: "VllmConfig") -> None:
    if not _uses_model_parallel(vllm_config.parallel_config):
        return

    os.environ["RBLN_CTX_STANDALONE"] = "1"
    if os.environ.get("RBLN_RUNTIME_FORCE_SYNC") == "1":
        logger.warning(
            "RBLN_RUNTIME_FORCE_SYNC=1 forces the synchronous runtime, "
            "which may cause performance degradation "
            "when using vLLM model parallel (TP, DP, EP, or PP)."
        )
