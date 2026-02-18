# Qwen2VLForConditionalGeneration
# Qwen2_5_VLForConditionalGeneration

def get_param_qwen2_vl(batch_size: int, max_model_len: int, block_size: int, tp_size: int) -> dict:
    param = {
        "visual": {
            # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image.
            # Example: For a 224x224 pixel image with patch size 14,
            # this produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_lens must be at least 256.
            # RBLN optimization processes inference per image or video frame, so set max_seq_lens to
            # match the maximum expected resolution to optimize computation.
            "max_seq_lens": 6400,
        },
        "tensor_parallel_size": tp_size,
        "max_seq_len": max_model_len,
        "kvcache_block_size": block_size,
        "batch_size": batch_size,
    }
    return param


get_param_qwen2_5_vl = get_param_qwen2_vl