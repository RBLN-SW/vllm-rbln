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

"""Per-request Perfetto tracing for vLLM v1 engine.

This subpackage applies runtime monkey-patches to vLLM's `EngineCore`,
`AsyncLLM`, and OpenAI API server so that per-request timestamps
(arrival, first scheduled, prefill, decode, finish) are emitted as
Chrome Trace JSON events. Trace lifecycle is controlled via the
`/v1/trace/start` and `/v1/trace/stop` HTTP endpoints.

Patches are applied automatically when `patches` is imported, which
is wired into `vllm_rbln.register_ops()`.
"""
