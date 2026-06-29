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
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "ENV_METADATA" and isinstance(node.value, ast.Dict):
            return node.value
        if isinstance(node, ast.Assign) \
                and any(isinstance(t, ast.Name) and t.id == "ENV_METADATA" for t in node.targets) \
                and isinstance(node.value, ast.Dict):
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
        rec = {"name": key.value, "description": "", "default": None,
               "type": "", "deprecated": ""}
        if isinstance(value, ast.Call):
            if value.args:
                rec["description"] = ast.literal_eval(value.args[0])
            for kw in value.keywords:
                if kw.arg in _FIELDS:
                    rec[kw.arg] = ast.literal_eval(kw.value)
        records.append(rec)
    return records


def _fmt_default(default) -> str:
    return "" if default is None else f"`{default!r}`"


def _table(rows: list[dict]) -> str:
    lines = ["| Name | Type | Default | Description |",
             "|------|------|---------|-------------|"]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['type']} | {_fmt_default(r['default'])} "
            f"| {r['description']} |")
    return "\n".join(lines)


def render(records: list[dict]) -> str:
    active = sorted((r for r in records if not r["deprecated"]),
                    key=lambda r: r["name"])
    deprecated = sorted((r for r in records if r["deprecated"]),
                        key=lambda r: r["name"])
    parts = [
        "# RBLN Environment Variables",
        "",
        "Environment variables specific to vllm-rbln. All are prefixed with "
        "`VLLM_RBLN_`. This page is generated from `ENV_METADATA` in "
        "`vllm_rbln/rbln_envs.py`.",
        "",
        _table(active),
    ]
    if deprecated:
        parts += ["", "## Deprecated", "", _table(deprecated)]
    return "\n".join(parts) + "\n"


def main() -> None:
    import mkdocs_gen_files

    records = read_metadata(ENVS_PATH.read_text(encoding="utf-8"))
    with mkdocs_gen_files.open("env_vars.md", "w") as f:
        f.write(render(records))


# mkdocs-gen-files executes this module as a script, so main() runs at import.
main()
