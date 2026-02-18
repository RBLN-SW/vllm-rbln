from .common import get_language_model_config
# Idefics3ForConditionalGeneration
def get_param_idefics3(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    return get_language_model_config(batch_size, max_model_len, block_size, tp_size)