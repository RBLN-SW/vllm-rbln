# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def get_param_exaone4_5(
    batch_size: int,
    max_model_len: int,
    block_size: int,
    num_devices: int,
    prefill_chunk_size: int | None = None,
) -> dict:
    param = {
        "visual": {
            # if tensor_parallel_size of submodule is not specified,
            # it inherits tensor_parallel_size of main module.
            "max_seq_len": 6400,
        },
        "num_devices": num_devices,
        "max_seq_len": max_model_len,
        "batch_size": batch_size,
        "use_inputs_embeds": True,
    }
    if block_size != max_model_len:
        attn_impl = "flash_attn" if block_size != max_model_len else "eager"
        param["kvcache_partition_len"] = block_size
        param["attn_impl"] = attn_impl
    # Pin prefill_chunk_size so the compiled model stays in sync with the value
    # used for KV-cache block padding.
    if prefill_chunk_size is not None:
        param["prefill_chunk_size"] = prefill_chunk_size
    return param
