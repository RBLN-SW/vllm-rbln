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

"""The optimum-rbln model path's half of the platform hooks."""

import contextlib
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def patch_upstream() -> None:
    # Only sync_from_vllm reads the key the first one writes, and on the vllm
    # path it would be an unknown field of RBLNConfig.
    _capture_user_max_num_batched_tokens()
    _allow_gemma4_global_per_layer_attribute_access()


def check_and_update(vllm_config: "VllmConfig") -> None:
    from vllm_rbln.utils.optimum.converter import sync_vllm_and_optimum
    from vllm_rbln.utils.optimum.predicates import forces_fp32_dtype
    from vllm_rbln.utils.optimum.registry import is_pooling_arch

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config

    if forces_fp32_dtype(model_config):
        model_config.dtype = torch.float32

    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = (
            "vllm_rbln.v1.worker.optimum_worker.RBLNOptimumWorker"
        )
    scheduler_config.scheduler_cls = (
        "vllm_rbln.v1.core.optimum_scheduler.RBLNOptimumScheduler"
    )
    # Optimum model runner doesn't support async scheduling.
    if scheduler_config.async_scheduling:
        logger.warning(
            "Disabling asynchronous scheduling: the optimum model runner "
            "does not support it. Running synchronously. Set "
            "VLLM_RBLN_USE_VLLM_MODEL=1 to use the runner that does."
        )
    scheduler_config.async_scheduling = False

    assert parallel_config.tensor_parallel_size == 1, (
        "Cannot set tensor_parallel_size for pre-compiled optimum-rbln models. "
        "If you want to compile with tensor parallelism in vllm-rbln, "
        "please use the `VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK` "
        "environment variable instead."
    )
    assert parallel_config.pipeline_parallel_size == 1, (
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
    if is_pooling_arch(hf_config) and getattr(hf_config, "is_encoder_decoder", False):
        hf_config.is_encoder_decoder = False
        with contextlib.suppress(KeyError):
            del model_config.__dict__["is_encoder_decoder"]

    disable_unsupported_prefix_caching(vllm_config)
    sync_vllm_and_optimum(vllm_config)


def _capture_user_max_num_batched_tokens() -> None:
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


def _allow_gemma4_global_per_layer_attribute_access() -> None:
    """Let vLLM's gemma4 config convertor read ``head_dim`` on transformers 5.15.

    transformers 5.15 makes it per-layer and raises on a top-level read.
    Fixed upstream in vllm-project/vllm#49797. TODO(vllm>=0.28.0): delete.
    """
    from vllm.config import model as vllm_model_config

    if getattr(vllm_model_config, "_rbln_gemma4_get_config_patched", False):
        return

    orig_get_config = vllm_model_config.get_config

    def get_config(*args, **kwargs):
        config = orig_get_config(*args, **kwargs)
        if config.model_type == "gemma4":
            config.text_config.allow_global_per_layer_attribute_access = True
        return config

    vllm_model_config.get_config = get_config
    vllm_model_config._rbln_gemma4_get_config_patched = True


def _disable_prefix_caching(vllm_config: "VllmConfig", reason: str) -> None:
    """Disable prefix caching with warning message."""
    logger.warning(
        "Prefix caching is not available for %s. It has been automatically disabled.",
        reason,
    )
    vllm_config.cache_config.enable_prefix_caching = False


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


def disable_unsupported_prefix_caching(vllm_config: "VllmConfig") -> None:
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
        _disable_prefix_caching(vllm_config, "EC producer (encoder-only)")
        return

    model_config = vllm_config.model_config
    hf_config = model_config.hf_config

    # Prefix caching is supported only for decoder-only models for now.
    if is_qwen3_embedding(model_config) or is_qwen3_reranker(model_config):
        _disable_prefix_caching(vllm_config, "Qwen3 pooling models")
    elif is_enc_dec_arch(hf_config):
        _disable_prefix_caching(vllm_config, "encoder-decoder models")
    elif is_pooling_arch(hf_config):
        _disable_prefix_caching(vllm_config, "pooling models")
    elif _uses_sliding_window(hf_config):
        _disable_prefix_caching(vllm_config, "sliding window models")
