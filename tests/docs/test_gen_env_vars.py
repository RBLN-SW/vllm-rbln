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


def _rec(name, **kw):
    rec = {
        "name": name,
        "description": "Desc.",
        "default": None,
        "type": "bool",
        "deprecated": "",
        "category": "Miscellaneous",
    }
    rec.update(kw)
    return rec


def test_render_entry_heading_and_meta_line():
    out = gen.render(
        [
            _rec(
                "VLLM_RBLN_A",
                description="Desc A.",
                default=True,
                type="bool",
                category="Attention",
            )
        ]
    )
    assert "# RBLN Environment Variables" in out
    assert "## Attention" in out
    # monospace heading + type/default meta line
    assert "### `VLLM_RBLN_A`" in out
    assert "`bool` · default: `True`" in out
    assert "Desc A." in out


def test_render_none_default_omits_default():
    out = gen.render([_rec("VLLM_RBLN_B", default=None, type="bool")])
    assert "### `VLLM_RBLN_B`" in out
    assert "`bool`" in out
    assert "default:" not in out


def test_render_groups_by_category_in_order():
    out = gen.render(
        [
            _rec("VLLM_RBLN_M1", category="Miscellaneous"),
            _rec("VLLM_RBLN_C1", category="Compilation & model loading"),
            _rec("VLLM_RBLN_A1", category="Attention"),
        ]
    )
    assert "## Compilation & model loading" in out
    assert "## Attention" in out
    assert "## Miscellaneous" in out
    # canonical order: Compilation before Attention before Miscellaneous
    assert (
        out.index("## Compilation & model loading")
        < out.index("## Attention")
        < out.index("## Miscellaneous")
    )


def test_render_untagged_category_falls_back_to_misc():
    out = gen.render([_rec("VLLM_RBLN_X", category="")])
    assert "## Miscellaneous" in out


def test_render_deprecated_admonition():
    out = gen.render([_rec("VLLM_RBLN_OLD", deprecated="use VLLM_RBLN_NEW")])
    assert "### `VLLM_RBLN_OLD`" in out
    assert '!!! warning "Deprecated"' in out
    assert "    use VLLM_RBLN_NEW" in out


def test_read_metadata_real_file():
    src = (_ROOT / "vllm_rbln" / "rbln_envs.py").read_text(encoding="utf-8")
    recs = gen.read_metadata(src)
    names = {r["name"] for r in recs}
    # Change-detector: bump when ENV_METADATA gains/loses an entry.
    assert len(recs) == 35
    assert "VLLM_RBLN_COMPILE_MODEL" in names
    auto_port = next(r for r in recs if r["name"] == "VLLM_RBLN_AUTO_PORT")
    assert auto_port["default"] is None  # conditional default
    # every entry is tagged with a functional category
    assert all(r["category"] for r in recs)


def test_module_main_wrote_env_vars_page():
    # main() ran at import time via the stub
    assert "env_vars.md" in _written
    assert "VLLM_RBLN_COMPILE_MODEL" in _written["env_vars.md"].text
