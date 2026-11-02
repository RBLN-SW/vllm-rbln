# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""Tests for vllm_rbln.compilation.backends: the conformance surface (it wraps
rebel's torch.compile backend, so a drift in that dependency must surface here).
The log/format helpers and the actual NPU compile are not unit-tested."""

import vllm_rbln.compilation.backends as backends
from vllm_rbln.compilation import rbln_backend


class TestBackendConformance:
    def test_rebel_backend_importable_and_callable(self):
        # backends.py imports this at module load; the drift alarm if rebel
        # moves or renames its torch.compile backend.
        from rebel.core.torch_compile import rbln_backend as rebel_backend

        assert callable(rebel_backend)

    def test_rbln_backend_exposed_and_callable(self):
        # The package re-exports backends.rbln_backend as the Dynamo backend.
        assert callable(rbln_backend)
        assert backends.rbln_backend is rbln_backend
