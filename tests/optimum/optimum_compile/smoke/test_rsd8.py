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

"""Optimum smoke tests for the RSD-8 (multi-device-class) runner.

Run with ``pytest tests/optimum_compile/test_rsd8.py``.
"""

from test_base import MultimodalSmoke, requires_npu

pytestmark = requires_npu


class TestQwen25VL(MultimodalSmoke.Test):
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    HF_OVERRIDES = {
        "text_config.num_hidden_layers": 1,
        "vision_config.depth": 2,
        "vision_config.fullatt_block_indexes": [1],
    }
    NUM_DEVICES = 1
    LLM_KWARGS = {
        "block_size": 1024,
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "additional_config": {"rbln_config": {"visual": {"max_seq_len": [512]}}},
    }
    # Shrink the image so its vision-token count fits the visual max_seq_len.
    MM_PROCESSOR_KWARGS = {"min_pixels": 64 * 14 * 14, "max_pixels": 64 * 14 * 14}
