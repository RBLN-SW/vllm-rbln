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

import logging

import pytest
import torch

from vllm_rbln.torch_compile_backend import (
    logged_rbln_backend,
    set_warmup_active,
)

LOGGER_NAME = "vllm.vllm_rbln.torch_compile_backend"


@pytest.fixture
def caplog_vllm(caplog):
    """vllm-rbln's "vllm" parent logger has propagate=False; re-enable it so
    caplog (handler attached at root) sees records during the test."""
    parent = logging.getLogger("vllm")
    saved = parent.propagate
    parent.propagate = True
    try:
        yield caplog
    finally:
        parent.propagate = saved


@pytest.fixture
def reset_warmup_flag():
    yield
    set_warmup_active(False)


@pytest.mark.parametrize(
    "warmup, expected_phase, expected_level",
    [
        (True, "warm-up", logging.INFO),
        (False, "HOT PATH", logging.WARNING),
    ],
)
def test_logged_rbln_backend(
    warmup,
    expected_phase,
    expected_level,
    monkeypatch,
    caplog_vllm,
    reset_warmup_flag,
):
    caplog = caplog_vllm
    """Single end-to-end test for the backend wrapper.

    Covers: phase-dependent log level + label, input-shape summary, call-chain
    derivation, args/kwargs passthrough, and return-value passthrough.
    """
    captured = {}

    def fake_rbln_backend(graph_module, inputs, **kwargs):
        captured["gm"] = graph_module
        captured["inputs"] = inputs
        captured["kwargs"] = kwargs
        return "RUNTIME_SENTINEL"

    monkeypatch.setattr(
        "vllm_rbln.torch_compile_backend._rbln_backend",
        fake_rbln_backend,
    )

    set_warmup_active(warmup)

    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    inputs = [
        torch.empty((2, 3), dtype=torch.bfloat16),
        torch.empty((4,), dtype=torch.int32),
    ]
    options = {"cache_dir": "/tmp/x"}

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        out = logged_rbln_backend(gm, inputs, options=options)

    # Return value and arguments forwarded verbatim.
    assert out == "RUNTIME_SENTINEL"
    assert captured["gm"] is gm
    assert captured["inputs"] is inputs
    assert captured["kwargs"] == {"options": options}

    records = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert len(records) == 2, f"expected 2 log lines, got {records}"
    before, done = records

    # Phase-dependent log level.
    assert before.levelno == expected_level
    assert done.levelno == expected_level

    # First line carries phase label, input summary, and call chain.
    msg = before.getMessage()
    assert f"[{expected_phase}]" in msg
    assert "(2, 3):torch.bfloat16" in msg
    assert "(4,):torch.int32" in msg
    # Call chain points back to this test file (and not torch / rebel internals).
    assert "test_torch_compile_backend.py" in msg
    assert "/torch/" not in msg
    assert "/rebel/" not in msg

    # Second line is the timing line.
    assert "done:" in done.getMessage()
