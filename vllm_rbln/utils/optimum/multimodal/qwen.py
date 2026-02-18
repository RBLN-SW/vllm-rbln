# SPDX-License-Identifier: Apache-2.0
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


def get_param_qwen2_vl(
    batch_size: int, max_model_len: int, block_size: int, tp_size: int
) -> dict:
    # Max sequence length for Vision Transformer (ViT), representing the number of patches in an image. # noqa: E501
    # Example: For a 224x224 pixel image with patch size 14,
    # this produces 256 patches [(224/14) * (224/14)]. Thus, max_seq_lens must be at least 256. # noqa: E501
    # RBLN optimization processes inference per image or video frame, so set max_seq_lens to # noqa: E501
    # match the maximum expected resolution to optimize computation.
    param = {
        "visual": {
            "max_seq_lens": 6400,
        },
        "tensor_parallel_size": tp_size,
        "max_seq_len": max_model_len,
        "kvcache_block_size": block_size,
        "batch_size": batch_size,
    }
    return param


get_param_qwen2_5_vl = get_param_qwen2_vl
