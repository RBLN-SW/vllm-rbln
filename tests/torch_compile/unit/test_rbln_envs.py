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

import os

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
    deployment. The block count is the opposite case: it is only consulted from
    inside the dynamic-KV path, so its non-zero default cannot be seen by a
    static deployment, and having one keeps the feature from being reachable in a
    half-enabled state (flag on, no shrink, no resize, mark_dynamic still
    logged).
    """
    assert "VLLM_RBLN_USE_DYNAMIC_KV_CACHE" in envs.environment_variables
    assert "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS" in envs.environment_variables
    assert not envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE, (
        f"Expected VLLM_RBLN_USE_DYNAMIC_KV_CACHE to be False, \
        got {envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE}"
    )
    assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 8, (
        f"Expected VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS to default to 8, \
        got {envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS}"
    )
    assert envs.DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS == 8


def test_compile_kv_cache_num_blocks_default_is_not_derived_from_the_flag(
    monkeypatch,
):
    """The default must not be spelled in terms of the other variable."""
    monkeypatch.delenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", raising=False)
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", flag)
        assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 8


def test_blank_compile_kv_cache_num_blocks_is_the_default_not_a_crash(monkeypatch):
    """`FOO=` must not raise: `enable_envs_cache` calls every getter once."""
    monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", "")
    assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 8
    monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", "   ")
    assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 8


def test_explicit_zero_survives_as_an_opt_out(monkeypatch):
    """0 has to stay distinguishable from unset, or the escape hatch is gone."""
    monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", "0")
    assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 0
    assert envs.compile_kv_cache_num_blocks_is_explicit()


def test_provenance_and_value_agree_on_a_blank_value(monkeypatch):
    """A blank value must not read as a choice the operator made."""
    for raw in ("", "   "):
        monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", raw)
        assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 8
        assert not envs.compile_kv_cache_num_blocks_is_explicit()
        assert "VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS" in os.environ

    monkeypatch.delenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", raising=False)
    assert not envs.compile_kv_cache_num_blocks_is_explicit()
    monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", "16")
    assert envs.compile_kv_cache_num_blocks_is_explicit()


def test_dynamic_kv_cache_envs_are_read_from_os_environ(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS", "64")
    assert envs.VLLM_RBLN_USE_DYNAMIC_KV_CACHE is True
    assert envs.VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS == 64


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
