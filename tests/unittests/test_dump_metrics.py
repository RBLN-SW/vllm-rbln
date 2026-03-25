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

"""This writes metrics dump files when enabled.

Requires RBLN NPU, network (HF model), and opt-in:
  VLLM_RBLN_DUMP_METRICS=1 pytest ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Smallest public Qwen2 causal LM in the Qwen2 family (~0.5B params).
QWEN2_SMALL_MODEL = "Qwen/Qwen2-0.5B-Instruct"

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_INFERENCE_BASIC = REPO_ROOT / "examples" / "experimental" / "offline_inference_basic.py"


pytestmark = pytest.mark.skipif(
    os.environ.get("VLLM_RBLN_DUMP_METRICS", "").lower()
    not in ("1", "true", "yes"),
    reason="Set VLLM_RBLN_DUMP_METRICS=1 to run (needs RBLN + HF).",
)


def test_dump_metrics_files_from_offline_inference_basic(tmp_path: Path) -> None:
    """Run offline_inference_basic with metrics + dump; expect *_metrics.txt in cwd."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["VLLM_RBLN_METRICS"] = "1"
    env["VLLM_RBLN_DUMP_METRICS"] = "1"

    cmd = [
        sys.executable,
        str(OFFLINE_INFERENCE_BASIC),
        "--model",
        QWEN2_SMALL_MODEL,
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "4",
        "--block-size",
        "128",
        "--tensor-parallel-size",
        "1",
    ]
    subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=7200,
    )

    dumped = list(tmp_path.glob("*_metrics.txt"))
    assert dumped, (
        f"Expected at least one *_metrics.txt under {tmp_path}, "
        f"got {list(tmp_path.iterdir())}"
    )
    for path in dumped:
        text = path.read_text()
        assert "METRICS" in text, f"{path} should contain METRICS section, got {text!r}"
