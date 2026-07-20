# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect

import pytest

import optimum.rbln
from vllm_rbln.utils.optimum.registry import _RBLN_SUPPORTED_MODELS


@pytest.mark.parametrize(
    ("arch", "model_cls_name"),
    [(arch, cls_name) for arch, (_, cls_name) in _RBLN_SUPPORTED_MODELS.items()],
    ids=list(_RBLN_SUPPORTED_MODELS.keys()),
)
def test_registered_class_exists_in_optimum(arch: str, model_cls_name: str):
    model_cls = getattr(optimum.rbln, model_cls_name, None)
    assert model_cls is not None, (
        f"{arch} maps to '{model_cls_name}', which is not exported by "
        f"optimum.rbln {optimum.rbln.__version__}. The registry entry is stale."
    )
    assert inspect.isclass(model_cls), (
        f"optimum.rbln.{model_cls_name} is not a class."
    )
