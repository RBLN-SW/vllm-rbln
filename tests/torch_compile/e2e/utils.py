# Copyright 2025 Rebellions Inc. All rights reserved.
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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from vllm import LLM


@contextmanager
def managed_llm(
    monkeypatch: pytest.MonkeyPatch,
    env: dict | None = None,
    **llm_kwargs: Any,
) -> Iterator[LLM]:
    """Set env vars, build an ``LLM``, and deterministically free it on exit.

    vLLM tears its EngineCore subprocess down only through a GC-lazy
    ``weakref.finalize``. Relying on that both leaks device memory across tests
    until GC happens to run (cross-test OOM) and can block the process at
    interpreter exit (CI hang). Calling the explicit, bounded shutdown
    (``terminate -> join(5s) -> SIGKILL``) kills EngineCore immediately, so the
    kernel driver reclaims the device before the next test builds its engine.

    ``monkeypatch`` is the test's fixture; the env vars are reverted at test
    teardown. ``env`` may be omitted when the env is set elsewhere (e.g. a
    module-scoped fixture); the deterministic teardown still applies.
    """
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    llm = LLM(**llm_kwargs)
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()
