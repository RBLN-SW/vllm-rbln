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

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class _Buf:
    def __init__(self):
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, s):
        self.text += s


_written = {}


def _stub_open(name, mode="r"):
    buf = _Buf()
    _written[name] = buf
    return buf


sys.modules["mkdocs_gen_files"] = types.SimpleNamespace(open=_stub_open)  # type: ignore[assignment]

_spec = importlib.util.spec_from_file_location(
    "gen_env_vars", _ROOT / "docs" / "gen_env_vars.py"
)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)  # runs main() -> writes to _written via stub


def test_render_active_section():
    out = gen.render(
        [
            {
                "name": "VLLM_RBLN_A",
                "description": "Desc A.",
                "default": True,
                "type": "bool",
                "deprecated": "",
            },
        ]
    )
    assert "# RBLN Environment Variables" in out
    assert "## Boolean variables" in out
    assert "### VLLM_RBLN_A" in out
    assert "Desc A." in out
    assert "Defaults to `True`." in out
    assert "## Deprecated variables" not in out


def test_render_none_default_omits_defaults_line():
    out = gen.render(
        [
            {
                "name": "VLLM_RBLN_B",
                "description": "Desc B.",
                "default": None,
                "type": "bool",
                "deprecated": "",
            },
        ]
    )
    assert "### VLLM_RBLN_B" in out
    assert "Defaults to" not in out


def test_render_groups_by_type():
    out = gen.render(
        [
            {
                "name": "VLLM_RBLN_B1",
                "description": "b.",
                "default": True,
                "type": "bool",
                "deprecated": "",
            },
            {
                "name": "VLLM_RBLN_N1",
                "description": "n.",
                "default": 1,
                "type": "int",
                "deprecated": "",
            },
            {
                "name": "VLLM_RBLN_S1",
                "description": "s.",
                "default": "x",
                "type": "str",
                "deprecated": "",
            },
        ]
    )
    assert "## Boolean variables" in out
    assert "## Numeric variables" in out
    assert "## String and list variables" in out


def test_render_deprecated_section():
    out = gen.render(
        [
            {
                "name": "VLLM_RBLN_OLD",
                "description": "Old one.",
                "default": None,
                "type": "bool",
                "deprecated": "use VLLM_RBLN_NEW",
            },
        ]
    )
    assert "## Deprecated variables" in out
    assert "### VLLM_RBLN_OLD" in out
    assert "**Deprecated.** use VLLM_RBLN_NEW" in out


def test_read_metadata_real_file():
    src = (_ROOT / "vllm_rbln" / "rbln_envs.py").read_text(encoding="utf-8")
    recs = gen.read_metadata(src)
    names = {r["name"] for r in recs}
    # Change-detector: bump when ENV_METADATA gains/loses an entry.
    assert len(recs) == 35
    assert "VLLM_RBLN_COMPILE_MODEL" in names
    auto_port = next(r for r in recs if r["name"] == "VLLM_RBLN_AUTO_PORT")
    assert auto_port["default"] is None  # conditional default


def test_module_main_wrote_env_vars_page():
    # main() ran at import time via the stub
    assert "env_vars.md" in _written
    assert "VLLM_RBLN_COMPILE_MODEL" in _written["env_vars.md"].text
