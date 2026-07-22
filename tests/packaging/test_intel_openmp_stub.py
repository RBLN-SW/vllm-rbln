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
"""Verify intel-openmp resolves to the vendored empty stub, not Intel's wheel."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path


def test_installed_intel_openmp_is_stub() -> None:
    (wheel,) = (Path(__file__).resolve().parents[2] / "vendor").glob(
        "intel_openmp-*+rbln.stub-*.whl"
    )
    match = re.match(r"intel_openmp-(.+?)-py3-none-any\.whl$", wheel.name)
    assert match, wheel.name
    stub_version = match.group(1)

    dist = importlib.metadata.distribution("intel-openmp")
    assert dist.version == stub_version, (
        f"env has intel-openmp {dist.version}; expected {stub_version}. "
        "Run `uv sync --locked` to pick up the vendored stub."
    )

    files = list(dist.files or [])
    binaries = [str(f) for f in files if f.suffix in {".so", ".dll", ".dylib"}]
    assert not binaries, binaries
