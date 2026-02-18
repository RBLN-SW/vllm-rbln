# Gemma3ForConditionalGeneration
from optimum.rbln import RBLNAutoModelForVision2Seq, RBLNAutoModelForImageTextToText

def get_language_model_config(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    attn_impl = "flash_attn" if block_size != max_model_len else "eager"
    return {
        "use_inputs_embeds": True,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
        "kvcache_partition_len": block_size,
        "tensor_parallel_size": tp_size,
        "attn_impl": attn_impl,
    }