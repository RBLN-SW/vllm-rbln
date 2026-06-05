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

# TODO: drop these once the LoRA and KV-connector e2e suites are stable
# under the nightly FSW matrix.
collect_ignore = ["v1/lora", "v1/kv_connector"]


@pytest.fixture(
    autouse=True,
    params=[
        False,
        # weight-free compilation is not yet correctness-complete across all
        # models/features, so the `on` leg is expected to fail. Non-strict so a
        # test that already works under weight-free reports XPASS instead of
        # breaking CI; flip to strict=True once the path is fully supported to
        # be alerted when these can be promoted to real assertions.
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="RBLN_WEIGHT_FREE compile path not yet fully supported",
                strict=False,
            ),
        ),
    ],
    ids=["wf_off", "wf_on"],
)
def rbln_weight_free(request, monkeypatch):
    """Run every e2e test once with weight-free compilation off and once on.

    ``RBLN_WEIGHT_FREE`` is consumed by the rebel compiler (not vllm-rbln): when
    set, ``compile_from_torch`` strips weights from the ``.rbln`` artifact and the
    runtime re-applies them from host tensors (``_apply_weight_free_weights``).
    The flag is read at *compile* time, inside the freshly ``spawn``-ed EngineCore
    worker, which inherits this process's env — so setting it here before each
    test builds its ``LLM`` is enough; no separate test process is needed.

    The compile-cache hash (``TorchModelHasher.get_model_hash``) does NOT include
    the weight-free mode, so off and on collide on the same ``{hash}.rbln`` and the
    second run would silently cache-hit the first's artifact. Disable the cache so
    each parametrization forces a real compile down its own path.
    """
    monkeypatch.setenv("RBLN_WEIGHT_FREE", "1" if request.param else "0")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    yield request.param
