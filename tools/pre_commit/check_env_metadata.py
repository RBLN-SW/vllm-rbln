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

"""Lint that every VLLM_RBLN_* env var has a non-trivial ENV_METADATA entry.

Parses rbln_envs.py with the ast module (no import) so it stays free of the
vllm runtime dependency and runs in pre-commit's isolated environment.
"""

import argparse
import ast
import sys

ENVS_PATH = "vllm_rbln/rbln_envs.py"
PREFIX = "VLLM_RBLN_"
MIN_DESC_LEN = 10


def _string_keys(node: ast.Dict) -> list[str]:
    keys = []
    for key in node.keys:
        # key is None for ** unpacking (e.g. **vllm_envs); skip it.
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


def _description_of(value: ast.expr) -> str | None:
    # Expect EnvMeta("...", ...). Description is the first positional arg
    # or the `description` keyword. Return None if not a string literal.
    if not isinstance(value, ast.Call):
        return None
    if value.args and isinstance(value.args[0], ast.Constant) \
            and isinstance(value.args[0].value, str):
        return value.args[0].value
    for kw in value.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _find_dict(tree: ast.Module, name: str) -> ast.Dict | None:
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    target = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name:
            target = node.value
        if isinstance(target, ast.Dict):
            return target
    return None


def parse_source(src: str) -> tuple[set[str], dict[str, str]]:
    """Return (VLLM_RBLN_* getter keys, {meta key: description})."""
    tree = ast.parse(src)

    env_dict = _find_dict(tree, "environment_variables")
    meta_dict = _find_dict(tree, "ENV_METADATA")
    if env_dict is None:
        raise ValueError("environment_variables dict not found")
    if meta_dict is None:
        raise ValueError("ENV_METADATA dict not found")

    env_keys = {k for k in _string_keys(env_dict) if k.startswith(PREFIX)}

    meta: dict[str, str] = {}
    for key, value in zip(meta_dict.keys, meta_dict.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            meta[key.value] = _description_of(value) or ""
    return env_keys, meta


def check(env_keys: set[str], meta: dict[str, str]) -> list[str]:
    """Return a list of human-readable error strings (empty == ok)."""
    errors = []
    for key in sorted(env_keys - meta.keys()):
        errors.append(f"{key}: missing ENV_METADATA entry")
    for key in sorted(meta.keys() - env_keys):
        errors.append(
            f"{key}: orphan ENV_METADATA entry (no VLLM_RBLN_* getter)")
    for key in sorted(env_keys & meta.keys()):
        desc = meta[key].strip()
        if len(desc) < MIN_DESC_LEN:
            errors.append(
                f"{key}: description too short (min {MIN_DESC_LEN} chars)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=ENVS_PATH)
    args = parser.parse_args(argv)

    with open(args.path, encoding="utf-8") as f:
        env_keys, meta = parse_source(f.read())

    errors = check(env_keys, meta)
    if errors:
        print(f"ENV_METADATA check failed in {args.path}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
