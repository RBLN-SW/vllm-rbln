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
import importlib.util

import torch
from vllm.profiler.wrapper import TorchProfilerActivityMap

from vllm_rbln.patches import register_patch

# RBLN activity alias recognized by vLLM's TorchProfilerWrapper.
RBLN_PROFILER_ACTIVITY = "RBLN"

# The replacement is built at import time, so it must be safe when PrivateUse1
# is absent (condition is only evaluated at apply time). Add the key only when
# PrivateUse1 exists.
_PRIVATEUSE1 = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
_PATCHED_ACTIVITY_MAP = dict(TorchProfilerActivityMap)
if _PRIVATEUSE1 is not None:
    _PATCHED_ACTIVITY_MAP[RBLN_PROFILER_ACTIVITY] = _PRIVATEUSE1


def _rbln_profiler_activity_available() -> bool:
    """True only when torch-rbln is installed and torch supports the PrivateUse1
    activity. Uses find_spec to avoid the import side effect."""
    return (
        _PRIVATEUSE1 is not None and importlib.util.find_spec("torch_rbln") is not None
    )


register_patch(
    target="vllm.profiler.wrapper.TorchProfilerActivityMap",
    reason=(
        "torch-rbln exposes the NPU as a PrivateUse1 device with a libkineto "
        "ActivityProfiler; register an 'RBLN' activity alias so vLLM's "
        "TorchProfilerWrapper can record RBLN device activity."
    ),
    key="vllm_rbln.patches.profiler.TorchProfilerActivityMap",
    owner_module="vllm_rbln.patches.profiler",
    condition=_rbln_profiler_activity_available,
)(_PATCHED_ACTIVITY_MAP)
