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

"""Every `*rbln*` requirement in pyproject must be capped to its floor's patch line.

A floor alone lets a published wheel resolve to whatever is newest at install time,
which is how old vllm-rbln releases came to install today's optimum-rbln.
"""

from itertools import chain
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version


def _rbln() -> list[Requirement]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps = chain(
        project.get("dependencies", []),
        *project.get("optional-dependencies", {}).values(),
        *data.get("dependency-groups", {}).values(),
    )
    return [req for dep in deps if "rbln" in (req := Requirement(dep)).name.lower()]


def test_rbln_requirements_are_discovered() -> None:
    assert _rbln(), "no `*rbln*` requirement found in pyproject.toml"


@pytest.mark.parametrize("req", [pytest.param(r, id=r.name) for r in _rbln()])
def test_capped_to_patch_line(req: Requirement) -> None:
    floors = [
        Version(spec.version.rstrip(".*"))
        for spec in req.specifier
        if spec.operator in {">=", "==", "~="}
    ]
    assert floors, f"`{req}` declares no lower bound"

    floor = max(floors)
    ceiling = Version(f"{floor.major}.{floor.minor}.{floor.micro + 1}")
    assert ceiling not in req.specifier, (
        f"`{req}` admits {ceiling}; write `>={floor},<{ceiling}`"
    )
