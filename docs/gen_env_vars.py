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

from __future__ import annotations

import ast
from pathlib import Path

ENVS_PATH = Path(__file__).resolve().parent.parent / "vllm_rbln" / "rbln_envs.py"

_FIELDS = ("description", "default", "type", "deprecated")


def _find_metadata_dict(tree: ast.Module) -> ast.Dict | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id == "ENV_METADATA"
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    return None


def read_metadata(src: str) -> list[dict]:
    """Parse ENV_METADATA from rbln_envs.py source (no import)."""
    tree = ast.parse(src)
    meta_dict = _find_metadata_dict(tree)
    if meta_dict is None:
        raise ValueError("ENV_METADATA dict not found")

    records = []
    for key, value in zip(meta_dict.keys, meta_dict.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        rec = {
            "name": key.value,
            "description": "",
            "default": None,
            "type": "",
            "deprecated": "",
        }
        if isinstance(value, ast.Call):
            if value.args:
                rec["description"] = ast.literal_eval(value.args[0])
            for kw in value.keywords:
                if kw.arg in _FIELDS:
                    rec[kw.arg] = ast.literal_eval(kw.value)
        records.append(rec)
    return records


_BOOL_NOTE = (
    "These variables are read as **true** when set to `1` or `true` "
    "(case-insensitive). Any other value, or leaving them unset, is **false**."
)


def _entry(rec: dict) -> str:
    """Render a single variable as a section: heading, description, default."""
    lines = [f"### {rec['name']}", "", rec["description"]]
    if rec["default"] is not None:
        lines += ["", f"Defaults to `{rec['default']!r}`."]
    if rec["deprecated"]:
        lines += ["", f"**Deprecated.** {rec['deprecated']}"]
    return "\n".join(lines)


def render(records: list[dict]) -> str:
    active = [r for r in records if not r["deprecated"]]
    deprecated = sorted(
        (r for r in records if r["deprecated"]), key=lambda r: r["name"]
    )

    # Group active vars by value kind so booleans can share a parsing note.
    groups = [
        ("Boolean variables", [r for r in active if r["type"] == "bool"], _BOOL_NOTE),
        ("Numeric variables", [r for r in active if r["type"] == "int"], ""),
        (
            "String and list variables",
            [r for r in active if r["type"] not in ("bool", "int")],
            "",
        ),
    ]

    parts = [
        "# RBLN Environment Variables",
        "",
        "Environment variables specific to vllm-rbln. All are prefixed with "
        "`VLLM_RBLN_`. This page is generated from `ENV_METADATA` in "
        "`vllm_rbln/rbln_envs.py`.",
    ]
    for title, items, note in groups:
        if not items:
            continue
        parts += ["", f"## {title}"]
        if note:
            parts += ["", note]
        for rec in sorted(items, key=lambda r: r["name"]):
            parts += ["", _entry(rec)]
    if deprecated:
        parts += ["", "## Deprecated variables"]
        for rec in deprecated:
            parts += ["", _entry(rec)]
    return "\n".join(parts) + "\n"


def main() -> None:
    import mkdocs_gen_files

    records = read_metadata(ENVS_PATH.read_text(encoding="utf-8"))
    with mkdocs_gen_files.open("env_vars.md", "w") as f:
        f.write(render(records))


# mkdocs-gen-files executes this module as a script, so main() runs at import.
main()
