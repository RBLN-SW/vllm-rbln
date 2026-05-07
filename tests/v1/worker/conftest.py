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
    cache_dir = f"/tmp/torchinductor_{worker_id}"
    shutil.rmtree(cache_dir, ignore_errors=True)

    torch._dynamo.reset()

    yield


@pytest.fixture(autouse=True)
def skip_sync_vllm_and_optimum():
    with patch("vllm_rbln.platform.sync_vllm_and_optimum"):
        yield
