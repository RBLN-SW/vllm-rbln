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

# Placeholder: KV sharing is unsupported on RBLN (both attention impls raise on
# kv_sharing_target_layer_name), so there is nothing to exercise yet. See NOTE.md.

import pytest

pytestmark = pytest.mark.skip(reason="placeholder -- not yet implemented")


@pytest.mark.model_compile
def test_fast_prefill_matches_baseline(vllm_runner) -> None:
    # ON vs OFF on a KV-sharing model must give identical greedy token ids.
    ...
