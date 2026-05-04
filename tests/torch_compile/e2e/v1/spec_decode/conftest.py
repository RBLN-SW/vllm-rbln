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


@pytest.fixture(scope="module", autouse=True)
def common_specdec_env(monkeypatch_module):
    monkeypatch_module.setenv("RBLN_USE_CUSTOM_KERNEL", "1")
    monkeypatch_module.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch_module.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch_module.setenv("VLLM_RBLN_ENABLE_WARM_UP", "1")
    monkeypatch_module.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
