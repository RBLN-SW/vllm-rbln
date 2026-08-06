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

import vllm_rbln.envs as envs


def test_rbln_envs():
    # check default values
    assert envs.VLLM_RBLN_COMPILE_MODEL, (
        f"Expected VLLM_RBLN_COMPILE_MODEL to be True, \
        got {envs.VLLM_RBLN_COMPILE_MODEL}"
    )

    assert not envs.VLLM_RBLN_COMPILE_STRICT_MODE, (
        f"Expected VLLM_RBLN_COMPILE_STRICT_MODE to be False, \
        got {envs.VLLM_RBLN_COMPILE_STRICT_MODE}"
    )

    assert envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK == 1, (
        f"Expected VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK to be 1, \
        got {envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK}"
    )

    assert envs.VLLM_RBLN_SAMPLER, (
        f"Expected VLLM_RBLN_SAMPLER to be True, \
        got {envs.VLLM_RBLN_SAMPLER}"
    )

    assert envs.VLLM_RBLN_ENABLE_WARM_UP, (
        f"Expected VLLM_RBLN_ENABLE_WARM_UP to be True, \
        got {envs.VLLM_RBLN_ENABLE_WARM_UP}"
    )

    assert envs.VLLM_RBLN_USE_VLLM_MODEL, (
        f"Expected VLLM_RBLN_USE_VLLM_MODEL to be True, \
        got {envs.VLLM_RBLN_USE_VLLM_MODEL}"
    )

    assert envs.VLLM_RBLN_FLASH_CAUSAL_ATTN, (
        f"Expected VLLM_RBLN_FLASH_CAUSAL_ATTN to be True, \
        got {envs.VLLM_RBLN_FLASH_CAUSAL_ATTN}"
    )

    assert not envs.VLLM_RBLN_ENFORCE_MODEL_FP32, (
        f"Expected VLLM_RBLN_ENFORCE_MODEL_FP32 to be False, \
        got {envs.VLLM_RBLN_ENFORCE_MODEL_FP32}"
    )

    assert envs.VLLM_RBLN_NUM_RAY_NODES == 1, (
        f"Expected VLLM_RBLN_NUM_RAY_NODES to be 1, \
        got {envs.VLLM_RBLN_NUM_RAY_NODES}"
    )

    assert not envs.VLLM_RBLN_METRICS, (
        f"Expected VLLM_RBLN_METRICS to be False, \
        got {envs.VLLM_RBLN_METRICS}"
    )

    assert envs.VLLM_RBLN_METRICS_FILE == "", (
        f"Expected VLLM_RBLN_METRICS_FILE to be empty by default, \
        got {envs.VLLM_RBLN_METRICS_FILE}"
    )

    assert envs.VLLM_RBLN_AUTO_PORT, (
        f"Expected VLLM_RBLN_AUTO_PORT to be True, \
        got {envs.VLLM_RBLN_AUTO_PORT}"
    )


def test_dynamic_kv_cache_envs_default_off():
    """The dynamic KV cache path must be opt-in behind ONE switch.

    `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` is read before / during the compile, so a
    non-False default would change the compiled artifact for every existing
    deployment. It is also the only variable the feature has: the compile-time
    hint is a module constant, so there is no way to reach the feature in a
    half-enabled state (flag on, no shrink, no resize, mark_dynamic still logged).
    """
    assert "VLLM_RBLN_USE_DYNAMIC_KV_CACHE" in envs.environment_variables
    assert not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE, (
        f"Expected VLLM_RBLN_USE_DYNAMIC_KV_CACHE to be False, \
        got {envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE}"
    )


def test_dynamic_kv_cache_envs_are_read_from_os_environ(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "1")
    assert envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE is True


def test_the_unprofiled_reserve_is_not_an_env_var():
    """It is a constant, and it has to stay above what the measurement found.

    48 MiB covers the 41,968,576 B of device memory the measured artifacts hold
    with no matching profile region, and on both configs measured on device it
    costs zero blocks. Not an env var: `_dynamic_kv_chiplet_budget` is its only
    reader, and lowering it is the one "remedy" that would re-create the overrun
    the reserve exists to prevent.
    """
    from vllm_rbln.v1.worker.rbln_worker import DYNAMIC_KV_UNPROFILED_RESERVE_BYTES

    assert (
        "VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES"
        not in envs.environment_variables
    )
    assert DYNAMIC_KV_UNPROFILED_RESERVE_BYTES == 48 * 1024 * 1024
    assert DYNAMIC_KV_UNPROFILED_RESERVE_BYTES > 41_968_576
