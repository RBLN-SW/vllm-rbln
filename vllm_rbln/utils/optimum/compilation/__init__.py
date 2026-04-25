from typing import Any

import optimum.rbln
from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSpeechSeq2Seq,
)
from transformers import PretrainedConfig

from vllm_rbln.utils.optimum.multimodal import (
    _COMPILE_MULTIMODAL_FNS,
    get_multimodal_cls,
)
from vllm_rbln.utils.optimum.registry import (
    get_rbln_model_info,
    is_enc_dec_arch,
    is_generation_arch,
    is_multi_modal,
    is_pooling_arch,
)


def compile_model(
    hf_model_name: str,
    config: PretrainedConfig,
    batch_size: int,
    block_size: int,
    max_model_len: int,
    tp_size: int,
    model_path: str,
    additional_config: dict[str, Any] | None = None,
) -> Any:
    architectures = getattr(config, "architectures", [])
    model_name, model_cls_name = get_rbln_model_info(
        config
    )  # check if the model is supported and get model info
    default_param: dict[str, Any] = {
        "tensor_parallel_size": tp_size,
    }

    if is_generation_arch(config):
        default_param["batch_size"] = batch_size
        default_param["max_seq_len"] = max_model_len
        if block_size != max_model_len:
            default_param["kvcache_partition_len"] = block_size
            default_param["attn_impl"] = "flash_attn"
        model_cls = RBLNAutoModelForCausalLM
    elif is_pooling_arch(config):
        model_cls = getattr(optimum.rbln, model_cls_name)
        assert model_cls is not None
        default_param["batch_size"] = batch_size
        default_param["max_seq_len"] = max_model_len
        # FIXME: We need a more generalized logic to specify block sizes
        # as the number of supported models continues to grow.
        if architectures[0] == "Qwen3Model" and block_size != max_model_len:
            default_param["kvcache_partition_len"] = block_size
            default_param["attn_impl"] = "flash_attn"
    elif is_multi_modal(config):
        model_cls = get_multimodal_cls(architectures[0])
        compile_fn = _COMPILE_MULTIMODAL_FNS.get(model_name)
        if compile_fn is None:
            raise ValueError(
                f"Unknown multimodal model alias: {model_name}. "
                f"Supported aliases: {sorted(_COMPILE_MULTIMODAL_FNS.keys())}"
            )
        default_param = compile_fn(batch_size, max_model_len, block_size, tp_size)
    elif is_enc_dec_arch(config):
        assert architectures[0] == "WhisperForConditionalGeneration"
        # Whisper model does not require max_model_len and block_size
        assert block_size == max_model_len, (
            "block_size must be equal to max_model_len for Whisper models."
        )  # noqa: E501
        assert max_model_len == config.max_length, (
            f"max_model_len ({max_model_len}) must match the Whisper model's "
            f"max_length ({config.max_length}) from the HuggingFace config."
        )
        default_param["batch_size"] = batch_size
        default_param["token_timestamps"] = False
        model_cls = RBLNAutoModelForSpeechSeq2Seq
    else:
        raise NotImplementedError(
            f"Compilation is not implemented for architecture {architectures[0]}"
        )
    if additional_config:
        default_param.update(additional_config)
    # FIXME:
    # Check conflict between default_param and additional_config,
    # and raise error if conflict exists, to avoid silent bug.
    model = model_cls.from_pretrained(
        hf_model_name,
        rbln_config=default_param,
    )

    model.save_pretrained(model_path)
    return model
