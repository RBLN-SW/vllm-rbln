# Blip2ForConditionalGeneration

# Gemma3ForConditionalGeneration
# PaliGemmaForConditionalGeneration
from .common import get_language_model_config

def get_param_paligemma(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    language_model_config = get_language_model_config(batch_size, max_model_len, block_size, tp_size)
    language_model_config["prefill_chunk_size"] = 8192
    param = {
        "language_model": language_model_config
    }
    return param