# 🛠️ Development setup (uv)

Requirements:
- Linux x86_64 with access to the internal network (Nexus / internal PyPI)
- Python **3.12** for this dev workflow (`rebel-compiler` nightly wheels are currently cp312-only; the package itself targets 3.10-3.13)
- uv **>= 0.11.25** (`uv self update`) — enforced via `required-version` in `pyproject.toml`. Older uv writes `uv.lock` in a different serialization (repeats the `tool.uv.environments` marker on every dependency), so re-locking with it produces a ~900-line noise diff.

Internal indexes require your **LDAP account** credentials (set once, e.g. in your shell profile):

```bash
export UV_INDEX_RBLN_NEXUS_NIGHTLY_USERNAME=<ldap-username>
export UV_INDEX_RBLN_NEXUS_NIGHTLY_PASSWORD=<ldap-password>
export UV_INDEX_REBELLIONS_USERNAME=<ldap-username>
export UV_INDEX_REBELLIONS_PASSWORD=<ldap-password>
```

> `rbln-release` (pypi.rbln.ai) uses a separate account system; credentials for it are only needed if the lock ever resolves packages from that index.

Then install the locked, team-identical environment with a single command:

```bash
uv sync --extra runtime
```

> `rebel-compiler` is separated into the `runtime` extra — it is required for NPU execution but excluded from the base dependency set. `--extra runtime` installs it alongside the rest of the locked environment.

## pyproject.toml vs uv.lock — what to edit when

| File | Role | Edit by hand? |
|---|---|---|
| `pyproject.toml` | Declares dependency **ranges** (what we're compatible with). Ships in the wheel as `Requires-Dist`. | Yes |
| `uv.lock` | Pins the **exact** versions + hashes of the full graph (what dev/CI actually installs). Never shipped. | **No** — only via `uv` commands |

Common tasks:

```bash
# Add / remove / change a dependency range:
#   edit pyproject.toml, then refresh the lock and commit BOTH files
uv lock

# Bump one package to the latest version allowed by pyproject (lock-only change):
uv lock --upgrade-package rebel-compiler

# Reproduce the CI-identical environment (never resolves, only installs the lock):
uv sync --extra runtime
```

Rules of thumb:
- If you edited `pyproject.toml`, always run `uv lock` and commit `uv.lock` in the same PR — pre-commit rejects drift between the two.
- CI installs with `uv sync --locked`, which fails if `uv.lock` doesn't match `pyproject.toml`.
- Never edit `uv.lock` by hand.

To bump `rebel-compiler` to the latest nightly (do not edit `pyproject.toml`):

```bash
uv lock --upgrade-package rebel-compiler
# or pin a specific version:
uv lock --upgrade-package rebel-compiler==0.11.1.dev200
# commit the updated uv.lock
```

Available versions can be checked by browsing the index directly
(e.g. <https://nexus.mgmt.rbln.in/repository/pypi-group-nightly/simple/rebel-compiler/>,
LDAP login required), or:

```bash
curl -s -u "$UV_INDEX_RBLN_NEXUS_NIGHTLY_USERNAME:$UV_INDEX_RBLN_NEXUS_NIGHTLY_PASSWORD" \
  https://nexus.mgmt.rbln.in/repository/pypi-group-nightly/simple/rebel-compiler/ \
  | grep -oE 'rebel_compiler-[0-9][A-Za-z0-9.+]*' | sort -uV | tail -10
```

## External contributors (no Rebellions internal network)

`uv.lock` pins packages to Rebellions-internal indexes, so `uv sync` only works
inside the internal network. Internal nightly builds of `rebel-compiler` are
not published externally. External contributors instead:

1. Request an **external LDAP account** via the
   [Request RBLN SDK/Portal Access](https://rebellions.ai/request-form-rbln-sdk/).
2. Install `rebel-compiler` with that account, following the
   [official guide](https://docs.rbln.ai/latest/getting_started/installation_guide.html).
3. Install `vllm-rbln` on top — everything else resolves from public indexes:

```bash
# 1. Create a venv and install rebel-compiler with your external LDAP account
#    (per the official guide above):
uv venv --python 3.12
source .venv/bin/activate
# ... install rebel-compiler into this venv per the official instructions ...

# 2. Install vllm-rbln (rebel-compiler is an optional 'runtime' extra,
#    so your separately installed copy is left untouched):
uv pip install -e ".[test]"
```

`rebel-compiler` is an optional `runtime` extra and is not pulled in by the
base `test` install, so the resolver never conflicts with your separately
installed copy. Everything else resolves from public indexes (PyPI,
`wheels.vllm.ai`, `download.pytorch.org`); internal indexes are skipped
automatically.

> Note: this environment is **not** what CI reproduces. CI always builds from
> the committed `uv.lock` (internal nightly `rebel-compiler` included), so unless
> `uv.lock` is updated, your PR is tested against the existing lock — not against
> the environment you built above. With a released SDK version some recent
> features may not work locally — CI is the source of truth for compatibility.
