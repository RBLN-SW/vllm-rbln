from .common import get_language_model_config

def get_param_gemma3(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    language_model_config = get_language_model_config(batch_size, max_model_len, block_size, tp_size)
    return {
        "language_model": language_model_config
    }