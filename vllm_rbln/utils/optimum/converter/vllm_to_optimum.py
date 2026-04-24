from vllm_rbln.logger import init_logger
from typing import TYPE_CHECKING

from vllm_rbln.utils.optimum.params import (
    RBLNParams,
)
from vllm_rbln.utils.optimum.registry import (
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)
from vllm_rbln.utils.optimum.converter.utils import (
    update_max_num_batched_tokens,
    update_block_size,
)
if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def _set_default_block_size(vllm_config: VllmConfig) -> None:
    """Set a default block_size in cache_config if not already set by the user."""
    cache_config = vllm_config.cache_config
    if not cache_config.user_specified_block_size:
        cache_config.block_size = vllm_config.model_config.max_model_len

def _validate_block_size(vllm_config: VllmConfig) -> None:
    cache_config = vllm_config.cache_config
    hf_config = vllm_config.model_config.hf_config
    max_model_len = vllm_config.model_config.max_model_len

    if is_multi_modal(hf_config) or is_generation_arch(hf_config):
        assert cache_config.block_size >= 4096, (
            "block_size must be at least 4096 for compilation."
        )
    if is_pooling_arch(hf_config):
        assert cache_config.block_size == max_model_len, (
            "For pooling models, block_size must be equal to max_model_len."
        )

def sync_from_vllm(vllm_config: VllmConfig) -> None:
    """
    Sync vLLM configuration from RBLN parameters.
    1. Parse RBLNParams from vllm_config.additional_config["rbln_config"].
    2. Update vllm_config based on the parsed RBLNParams to ensure consistency between vLLM and RBLN configurations.
    3. Validate the updated block size
    """
    rbln_config = vllm_config.additional_config.get("rbln_config", {})
    params = RBLNParams.from_rbln_config(vllm_config, rbln_config)

    if params.batch_size is not None:
        logger.info(
            "Setting max_num_seqs to %d based on rbln_config in additional_config",
            params.batch_size,
        )
        vllm_config.scheduler_config.max_num_seqs = params.batch_size
    if params.max_seq_len is not None:
        logger.info(
            "Setting max_model_len to %d based on rbln_config in additional_config",
            params.max_seq_len,
        )
        vllm_config.model_config.max_model_len = params.max_seq_len
    if params.kvcache_block_size is not None:
        logger.info(
            "Setting block_size to %d based on rbln_config in additional_config",
            params.kvcache_block_size,
        )
        vllm_config.cache_config.block_size = params.kvcache_block_size
        vllm_config.cache_config.user_specified_block_size = params.kvcache_block_size

    _set_default_block_size(vllm_config)
    _validate_block_size(vllm_config)
    update_block_size(
        vllm_config, vllm_config.cache_config.block_size, prefill_chunk_size=128
    )

    # max_num_batched_tokens must fit both a full-length prefill and a full
    # batch dispatch; update_max_num_batched_tokens layers the enc-dec
    # max_source_positions constraint on top.
    vllm_config.scheduler_config.max_num_batched_tokens = max(
        vllm_config.model_config.max_model_len,
        vllm_config.scheduler_config.max_num_seqs,
    )
    update_max_num_batched_tokens(
        vllm_config, vllm_config.scheduler_config.max_num_batched_tokens
    )

    logger.info("Prepared vLLM config for compilation: %s", vllm_config)

