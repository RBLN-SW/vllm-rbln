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

from types import SimpleNamespace

import pytest

from vllm_rbln.utils.optimum.converter.from_vllm import sync_from_vllm

DECODER_ARCH = "LlamaForCausalLM"


def _vllm_config(rbln_config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[DECODER_ARCH])
        ),
        additional_config={"rbln_config": rbln_config},
    )


class TestDtypeOverrideRejected:
    def test_raises_and_points_at_vllm_dtype(self):
        # On the compile path dtype is vLLM's own argument: optimum-rbln takes
        # it as a load argument and ignores the rbln_config field, so honouring
        # it here would mean silently ranking two knobs that mean the same thing.
        cfg = _vllm_config({"batch_size": 4, "dtype": "bfloat16"})
        with pytest.raises(ValueError, match="`dtype` cannot be set"):
            sync_from_vllm(cfg)
