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

import pytest

from vllm_rbln.utils.optimum.converter.common import (
    update_block_size,
    update_max_num_batched_tokens,
)

from .._fakes import make_hf_config


class TestUpdateMaxNumBatchedTokens:
    def test_non_enc_dec_is_noop(self, vllm_config_factory):
        cfg = vllm_config_factory(
            max_num_batched_tokens=512,
            hf_config=make_hf_config(architectures=["LlamaForCausalLM"]),
        )
        update_max_num_batched_tokens(cfg, max_model_len=2048)
        assert cfg.scheduler_config.max_num_batched_tokens == 512

    def test_enc_dec_keeps_max_model_len_when_source_positions_smaller(
        self, vllm_config_factory
    ):
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=128,
        )
        cfg = vllm_config_factory(hf_config=hf, max_num_batched_tokens=64)
        update_max_num_batched_tokens(cfg, max_model_len=2048)
        # max_source_positions <= max_model_len: stays at max_model_len.
        assert cfg.scheduler_config.max_num_batched_tokens == 2048

    def test_enc_dec_lifts_to_source_positions_when_larger(
        self, vllm_config_factory
    ):
        hf = make_hf_config(
            architectures=["WhisperForConditionalGeneration"],
            max_source_positions=1500,
        )
        cfg = vllm_config_factory(hf_config=hf, max_num_batched_tokens=512)
        update_max_num_batched_tokens(cfg, max_model_len=448)
        assert cfg.scheduler_config.max_num_batched_tokens == 1500
