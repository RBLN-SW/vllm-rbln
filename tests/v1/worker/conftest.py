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
import shutil
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(autouse=True)
def fresh_inductor_cache_per_test(monkeypatch):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "root")
    cache_dir = f"/tmp/torchinductor_{worker_id}_{os.getpid()}"
    shutil.rmtree(cache_dir, ignore_errors=True)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    torch._dynamo.reset()
    yield


@pytest.fixture(autouse=True)
def pin_npu_per_worker(monkeypatch):
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    idx = int(worker[2:]) if worker[2:].isdigit() else 0
    monkeypatch.setenv("RBLN_DEVICES", str(idx))


@pytest.fixture(autouse=True)
def skip_sync_vllm_and_optimum():
    # Force `compiled_rbln_config = None` so sync_vllm_and_optimum() takes the
    # sync_from_vllm() path instead of resolving a real compiled config.
    with patch(
        "vllm_rbln.utils.optimum.converter.dispatch._resolve_rbln_config",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def set_npu_env_var(monkeypatch):
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", "RBLN-CA25")
