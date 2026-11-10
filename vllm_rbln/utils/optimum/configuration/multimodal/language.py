# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def build_language_model_config(
    batch_size: int,
    max_model_len: int,
    block_size: int,
    num_devices: int,
    memory_budget: float,
    prefill_chunk_size: int | None = None,
) -> dict:
    rbln_config: dict = {
        "use_inputs_embeds": True,
        "batch_size": batch_size,
        "max_seq_len": max_model_len,
        "num_devices": num_devices,
        "memory_budget": memory_budget,
    }
    if block_size != max_model_len:
        attn_impl = "flash_attn" if block_size != max_model_len else "eager"
        rbln_config["attn_impl"] = attn_impl
        rbln_config["kvcache_partition_len"] = block_size
    # Pin prefill_chunk_size so the compiled model stays in sync with the value
    # used for KV-cache block padding.
    if prefill_chunk_size is not None:
        rbln_config["prefill_chunk_size"] = prefill_chunk_size
    return rbln_config
