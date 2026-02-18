


# LlavaNextForConditionalGeneration
# LlavaForConditionalGeneration
from .common import get_language_model_config

def get_param_llava(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    param = {
        "vision_tower": {"output_hidden_states": True},
        "language_model": get_language_model_config(batch_size, max_model_len, block_size, tp_size)
    }
    return param

def get_param_llava_next(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    return get_language_model_config(batch_size, max_model_len, block_size, tp_size)