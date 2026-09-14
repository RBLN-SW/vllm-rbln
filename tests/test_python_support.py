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

from pathlib import Path
from urllib.parse import unquote

import pytest
import tomllib
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename


def _load(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / name
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _minors() -> list[int]:
    requires = SpecifierSet(_load("pyproject.toml")["project"]["requires-python"])
    return [minor for minor in range(30) if f"3.{minor}.0" in requires]


def _installable(package: dict, minor: int) -> bool:
    if "sdist" in package:
        return True
    supported = {
        (tag.interpreter, tag.abi)
        for tags in (
            cpython_tags((3, minor), platforms=["any"]),
            compatible_tags((3, minor), platforms=["any"]),
        )
        for tag in tags
    }
    return any(
        (tag.interpreter, tag.abi) in supported
        for wheel in package["wheels"]
        for tag in parse_wheel_filename(unquote(wheel["url"].rsplit("/", 1)[-1]))[3]
    )


def test_minors_are_discovered() -> None:
    assert len(_minors()) > 1, "`requires-python` admits fewer than two CPython minors"


@pytest.mark.parametrize("minor", _minors(), ids=lambda minor: f"3.{minor}")
def test_lock_covers_every_admitted_minor(minor: int) -> None:
    gaps = [
        f"{package['name']}=={package['version']}"
        for package in _load("uv.lock")["package"]
        if package.get("wheels") and not _installable(package, minor)
    ]
    assert not gaps, (
        f"`requires-python` admits 3.{minor}, but uv.lock has no distribution "
        f"installable on it for: {', '.join(gaps)}"
    )
