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


def get_language_model_config(
    batch_size: int, max_model_len: int, block_size: int, tp_size: int
) -> dict:
    attn_impl = "flash_attn" if block_size != max_model_len else "eager"
    return {
        "use_inputs_embeds": True,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
        "kvcache_partition_len": block_size,
        "tensor_parallel_size": tp_size,
        "attn_impl": attn_impl,
    }
