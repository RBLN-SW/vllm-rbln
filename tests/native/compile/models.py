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

"""Models the compile-and-run smoke exercises (spec type in model_specs)."""

from __future__ import annotations

from tests.native.model_specs import CompileModelSpec

MODELS: list[CompileModelSpec] = [
    CompileModelSpec("Qwen/Qwen3-30B-A3B", {"tensor_parallel_size": 8}),
    CompileModelSpec("Qwen/Qwen1.5-MoE-A2.7B", {"tensor_parallel_size": 8}),
    CompileModelSpec(
        "openai/gpt-oss-20b", {"tensor_parallel_size": 8}, num_hidden_layers=4
    ),
]
