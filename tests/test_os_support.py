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

# uv answers, rather than a reimplementation of wheel-tag matching: --python-platform
# and --python model the OS's glibc and interpreter, --no-build turns "would build from
# source" into a failure, and --dry-run downloads nothing. Hand-rolled tag math gets
# dependency markers and multi-version lock entries wrong in both directions.

import os
import subprocess
from pathlib import Path

import pytest
import tomllib
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]

# glibc per OS release, and the CPython that release ships within `requires-python`.
# RHEL 9's default python3 is 3.9, under our floor, so its 3.11 module is the lowest
# usable one. uv caps --python-platform at manylinux_2_40, so a newer glibc is checked
# at that cap -- sound, because a higher glibc admits strictly more wheels.
OS_MATRIX = {
    "ubuntu-22.04": ("x86_64-manylinux_2_35", "3.10"),
    "ubuntu-24.04": ("x86_64-manylinux_2_39", "3.12"),
    "ubuntu-26.04": ("x86_64-manylinux_2_40", "3.14"),
    "rhel-9": ("x86_64-manylinux_2_34", "3.11"),
    "rhel-10": ("x86_64-manylinux_2_39", "3.12"),
}

UNRELEASED = {
    "ubuntu-26.04": "numpy 2.2.6 publishes no cp314 wheel, and 26.04 is unreleased so "
    "its glibc and CPython are provisional. Drop this mark once the lock carries a "
    "3.14-installable numpy."
}


def _targets(expected_to_fail: bool = True):
    return [
        pytest.param(
            *OS_MATRIX[name],
            id=name,
            marks=(
                [pytest.mark.xfail(strict=True, reason=UNRELEASED[name])]
                if expected_to_fail and name in UNRELEASED
                else []
            ),
        )
        for name in OS_MATRIX
    ]


@pytest.mark.parametrize("platform, python", _targets())
@pytest.mark.parametrize(
    "extras", [(), ("kv_connectors",)], ids=["base", "kv_connectors"]
)
def test_locked_stack_installs(platform, python, extras, tmp_path) -> None:
    command = [
        "uv",
        "sync",
        "--locked",
        "--python",
        python,
        "--python-platform",
        platform,
        "--no-install-project",
        "--no-build",
        "--dry-run",
    ]
    for extra in extras:
        command += ["--extra", extra]

    done = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(tmp_path / "env")},
    )
    assert done.returncode == 0, (
        f"uv.lock has no installable distribution on {platform} / CPython {python}"
        f"{', extras ' + ','.join(extras) if extras else ''}:\n{done.stderr.strip()}"
    )


@pytest.mark.parametrize("platform, python", _targets(expected_to_fail=False))
def test_matrix_python_is_admitted(platform, python) -> None:
    requires = SpecifierSet(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "requires-python"
        ]
    )
    assert f"{python}.0" in requires, (
        f"the OS matrix targets CPython {python}, which `requires-python` excludes"
    )
