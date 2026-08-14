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

# A real RBLNModelRunner (worker-side, unlike the parent conftest's vllm_runner /
# hf_runner which build a whole engine). Shaped like upstream's
# test_gpu_model_runner.py::model_runner fixture, minus its dist_init.
#
# Every vllm import below is function-local: a conftest is imported before the
# native conftest's pytest_configure sets VLLM_RBLN_USE_VLLM_MODEL=1, and
# resolving RblnPlatform early would pin device_type to "cpu" for the session.

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch


@pytest.fixture
def make_model_runner():
    """Build a real RBLNModelRunner; call it from the test body."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.platforms import current_platform

    from tests.native.v1.worker.utils import make_kv_cache_config, make_runner_config
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

    # A factory, not a plain fixture: spawning is decided in pytest_pyfunc_call
    # (call phase), so building here would open the device in the parent.
    stack = contextlib.ExitStack()

    def _make(
        *,
        vllm_config=None,
        layers: tuple[str, ...] = ("layer.0",),
        kv_sharing: dict[str, str] | None = None,
        kv_cache_config=None,
        init_kv_cache: bool = True,
        **config_overrides: Any,
    ) -> Any:
        config = vllm_config or make_runner_config(**config_overrides)
        # Left open until the test ends so later get_current_vllm_config() calls
        # still see this config.
        stack.enter_context(set_current_vllm_config(config))

        # Passing kv heads as num_heads mirrors upstream's fixture; only the
        # resulting shapes matter here, not the head ratio.
        num_heads = config.model_config.get_num_kv_heads(config.parallel_config)
        head_size = config.model_config.get_head_size()
        for name in layers:
            # Registered in order, so a kv_sharing target must be an earlier
            # layer -- Attention.__init__ validates exactly that.
            config.compilation_config.static_forward_context[name] = Attention(
                num_heads,
                head_size,
                scale=0.1,
                prefix=name,
                kv_sharing_target_layer_name=(kv_sharing or {}).get(name),
            )

        runner = RBLNModelRunner(config, torch.device(current_platform.device_type))
        # On by default: __init__ leaves a placeholder input_batch that
        # initialize_kv_cache may replace, and production never uses it.
        if init_kv_cache:
            runner.initialize_kv_cache(
                kv_cache_config
                or make_kv_cache_config(runner, groups=[(name,) for name in layers])
            )
        return runner

    try:
        yield _make
    finally:
        stack.close()
