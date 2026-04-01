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

import multiprocessing
from collections.abc import Callable
from typing import Any

import pytest


def patch_and_run(
    monkeypatch: pytest.MonkeyPatch,
    env: dict,
    target_func: Callable,
    *target_func_args: Any,
    **target_func_kwargs: dict[str, Any],
):
    with monkeypatch.context() as m:
        for k, i in env.items():
            m.setenv(k, i)

        # rebellions SDK somewhat requires LLM instance to be
        # instantiated in separated process
        p = multiprocessing.Process(
            target=target_func,
            args=target_func_args,
            kwargs=target_func_kwargs,
        )
        p.start()
        p.join()

        assert not p.exitcode, f"Process exited with code {p.exitcode}"
