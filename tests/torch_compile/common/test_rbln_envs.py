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

import vllm_rbln.rbln_envs as rbln_envs


def test_rbln_envs():
    # check default values
    assert rbln_envs.VLLM_RBLN_COMPILE_MODEL, (
        f"Expected VLLM_RBLN_COMPILE_MODEL to be True, \
        got {rbln_envs.VLLM_RBLN_COMPILE_MODEL}"
    )

    assert not rbln_envs.VLLM_RBLN_COMPILE_STRICT_MODE, (
        f"Expected VLLM_RBLN_COMPILE_STRICT_MODE to be False, \
        got {rbln_envs.VLLM_RBLN_COMPILE_STRICT_MODE}"
    )

    assert rbln_envs.VLLM_RBLN_TP_SIZE == 1, (
        f"Expected VLLM_RBLN_TP_SIZE to be 1, \
        got {rbln_envs.VLLM_RBLN_TP_SIZE}"
    )

    assert rbln_envs.VLLM_RBLN_ENABLE_WARM_UP, (
        f"Expected VLLM_RBLN_ENABLE_WARM_UP to be True, \
        got {rbln_envs.VLLM_RBLN_ENABLE_WARM_UP}"
    )

    assert rbln_envs.VLLM_RBLN_USE_VLLM_MODEL, (
        f"Expected VLLM_RBLN_USE_VLLM_MODEL to be True, \
        got {rbln_envs.VLLM_RBLN_USE_VLLM_MODEL}"
    )

    assert rbln_envs.VLLM_RBLN_FLASH_CAUSAL_ATTN, (
        f"Expected VLLM_RBLN_FLASH_CAUSAL_ATTN to be True, \
        got {rbln_envs.VLLM_RBLN_FLASH_CAUSAL_ATTN}"
    )

    assert not rbln_envs.VLLM_RBLN_DISABLE_MM, (
        f"Expected VLLM_RBLN_DISABLE_MM to be False, \
        got {rbln_envs.VLLM_RBLN_DISABLE_MM}"
    )

    assert rbln_envs.VLLM_RBLN_DP_IMPL == "padded_decode", (
        f"Expected VLLM_RBLN_DP_IMPL to be padded_decode, \
        got {rbln_envs.VLLM_RBLN_DP_IMPL}"
    )

    assert not rbln_envs.VLLM_RBLN_ENFORCE_MODEL_FP32, (
        f"Expected VLLM_RBLN_ENFORCE_MODEL_FP32 to be False, \
        got {rbln_envs.VLLM_RBLN_ENFORCE_MODEL_FP32}"
    )

    assert rbln_envs.VLLM_RBLN_MOE_CUSTOM_KERNEL, (
        f"Expected VLLM_RBLN_MOE_CUSTOM_KERNEL to be True, \
        got {rbln_envs.VLLM_RBLN_MOE_CUSTOM_KERNEL}"
    )

    assert rbln_envs.VLLM_RBLN_DP_INPUT_ALL_GATHER, (
        f"Expected VLLM_RBLN_DP_INPUT_ALL_GATHER to be True, \
        got {rbln_envs.VLLM_RBLN_DP_INPUT_ALL_GATHER}"
    )

    assert rbln_envs.VLLM_RBLN_LOGITS_ALL_GATHER, (
        f"Expected VLLM_RBLN_LOGITS_ALL_GATHER to be True, \
        got {rbln_envs.VLLM_RBLN_LOGITS_ALL_GATHER}"
    )

    assert rbln_envs.VLLM_RBLN_NUM_RAY_NODES == 1, (
        f"Expected VLLM_RBLN_NUM_RAY_NODES to be 1, \
        got {rbln_envs.VLLM_RBLN_NUM_RAY_NODES}"
    )

    assert not rbln_envs.VLLM_RBLN_METRICS, (
        f"Expected VLLM_RBLN_METRICS to be False, \
        got {rbln_envs.VLLM_RBLN_METRICS}"
    )

    assert not rbln_envs.VLLM_RBLN_AUTO_PORT, (
        f"Expected VLLM_RBLN_AUTO_PORT to be False, \
        got {rbln_envs.VLLM_RBLN_AUTO_PORT}"
    )
