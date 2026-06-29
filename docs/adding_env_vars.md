# Adding an Environment Variable

All `VLLM_RBLN_*` environment variables live in
[`vllm_rbln/rbln_envs.py`](https://github.com/RBLN-SW/vllm-rbln/blob/main/vllm_rbln/rbln_envs.py).
That file is the **single source of truth**: the runtime reads the value, the
docs on the [Environment Variables](env_vars.md) page are generated from it, and
a pre-commit lint makes sure nothing is left undocumented.

Adding a variable is two steps.

## 1. Add the runtime getter

Add an entry to the `environment_variables` dict. This is the code that actually
reads and parses the value at runtime.

```python
environment_variables = {
    **vllm_envs,
    ...
    "VLLM_RBLN_MY_FLAG": (
        lambda: os.environ.get("VLLM_RBLN_MY_FLAG", "False").lower() in ("true", "1")
    ),
}
```

If the variable needs validation or a non-trivial default, write a small helper
function (see `get_dp_impl` for an example) and reference it here.

## 2. Add the documentation metadata

Add a matching entry to `ENV_METADATA`. The key **must** be identical to the
getter key, and the description must be non-empty — the lint enforces both.

```python
ENV_METADATA: dict[str, EnvMeta] = {
    ...
    "VLLM_RBLN_MY_FLAG": EnvMeta(
        "Short sentence describing what this flag does and when to set it.",
        default=False,
        type="bool",
        category="Miscellaneous",
    ),
}
```

`EnvMeta` fields:

| Field         | Required | Purpose                                                        |
| ------------- | -------- | -------------------------------------------------------------- |
| `description` | yes      | One or two sentences. Drives the docs; lint rejects if empty.  |
| `default`     | no       | Shown in the docs. Use `None` when the default is conditional. |
| `type`        | no       | `"bool"`, `"int"`, `"str"`, `"list[int]"`, …                   |
| `category`    | no       | Functional group heading on the docs page (see below).         |
| `deprecated`  | no       | Non-empty string marks the variable deprecated (with the note).|
| `choices`     | no       | Tuple of valid values; the SSOT for validation **and** docs.   |

### Variables with a fixed set of values

Put the valid values in `choices` and read them back in the getter — do **not**
hardcode the set in the getter.

```python
"VLLM_RBLN_MY_MODE": EnvMeta(
    "Selects the operating mode.",
    default="fast",
    type="str",
    category="Miscellaneous",
    choices=("fast", "accurate"),
),

def get_my_mode() -> str:
    value = os.environ.get("VLLM_RBLN_MY_MODE", "fast").lower()
    choices = set(ENV_METADATA["VLLM_RBLN_MY_MODE"].choices)
    if value not in choices:
        raise ValueError(f"Invalid VLLM_RBLN_MY_MODE: {value}, choices: {choices}")
    return value
```

The docs page renders the choices automatically as a *Possible values* line.

## Categories

The docs group variables by `category`. Reuse an existing category string when
one fits. To add a **new** category, also add it (in the order you want it
rendered) to `_CATEGORY_ORDER` and `_CATEGORY_INTRO` in
[`docs/gen_env_vars.py`](https://github.com/RBLN-SW/vllm-rbln/blob/main/docs/gen_env_vars.py).
Untagged variables fall back to *Miscellaneous*.

## What happens automatically

- **Lint** — `tools/pre_commit/check_env_metadata.py` (pre-commit hook
  `check-env-metadata`) fails if a `VLLM_RBLN_*` getter has no `ENV_METADATA`
  entry, if there is an orphan metadata entry, or if a description is missing.
- **Docs** — the [Environment Variables](env_vars.md) page is regenerated from
  `ENV_METADATA` on every build (`docs/gen_env_vars.py`). You never edit the env
  var page by hand.
- **Cross-links** — mentioning another `VLLM_RBLN_*` variable in a description
  automatically links to its section.

## Checklist

- [ ] Getter added to `environment_variables`.
- [ ] `ENV_METADATA` entry with a non-empty `description`.
- [ ] `choices` set on `EnvMeta` (not hardcoded) if the value is constrained.
- [ ] `pre-commit run check-env-metadata --all-files` passes.
- [ ] `mkdocs build` shows the variable on the Environment Variables page.
