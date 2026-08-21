# Agent instructions for vllm-rbln

vllm-rbln is a vLLM platform plugin for Rebellions NPUs. Source lives in `vllm_rbln/`, tests in `tests/`.

Two documents own their subjects; read them rather than duplicating them here:

- Environment setup, dependency and lockfile policy: `DEVELOPMENT.md`
- PR process and issue labels: `CONTRIBUTING.md`

## Commands

Run everything through `uv`. Never use the system `python3` or a bare `pip`.

```bash
uv run --no-sync pytest tests/native/v1/worker/test_rbln_worker.py -x
uvx pre-commit run --files <every file you changed>
```

Pick the narrowest test target that covers the change. `pre-commit` is not a project dependency, so run it through `uvx`. Passing `--files` explicitly keeps the run independent of what happens to be staged.

A new `.py` file needs the Apache header that every other file carries; `check-license-header` rejects it otherwise. Copy the header from a neighbouring file.

`tests/native/` defines three session options (see its `conftest.py`):

- `--model-compile` — opt into whole-model compiles, minutes per test
- `--device-tensor {0,1}` — session-wide, cannot be parametrized per test
- `--num-hidden-layers N` — build only the first N decoder layers

They do not exist in `tests/optimum/`.

## Terms

The codebase already has a word for each of these. Use it, and do not reach for a synonym because the sentence reads better.

- **model path** — which model implementation runs. The two are the **optimum-rbln path** and the **vLLM-native path**; `optimum` and `native` are the short forms. Not "backend", not "mode". `torch.compile` describes how the native path works and is not its name.
- **suite** — a top-level test tree: `tests/native/`, `tests/optimum/`.
- **lane** — a slice of a suite selected by a flag, a mark, or the device mode: the default lane, the `--model-compile` lane, the cpu and device lanes.

## Two model paths

`VLLM_RBLN_USE_VLLM_MODEL` selects the model path at startup: unset or `0` is optimum-rbln, `1` is vLLM-native.

| Path         | Owns                                                                        |
| ------------ | --------------------------------------------------------------------------- |
| optimum-rbln | `model_executor/models/optimum/`, `utils/optimum/`, `v1/worker/optimum_*.py` |
| vLLM-native  | `patches/`, `compilation/`, `v1/worker/rbln_*.py`                            |
| shared       | everything else                                                              |

**`envs.py` defines the flag; only `__init__.py` and `platform.py` branch on it.** Do not branch on `VLLM_RBLN_USE_VLLM_MODEL` anywhere else. Path-specific code belongs in the module that path owns.

- Say which path or paths you changed in the PR description.
- A change to one path must not alter the other. If it appears to need both, stop and ask before writing code.
- A new env var goes in three places in `envs.py`: the `TYPE_CHECKING` block, the `environment_variables` dict, and either `RBLN_COMPILE_ENV` or `RBLN_NON_COMPILE_ENV`. The two sets partition the mega-cache bundle key, and `test_mega_cache.py` asserts they cover every variable.
- Suites carry the path: `tests/native/` sets it to `1` in its conftest and scrubs `VLLM_RBLN_*`; `tests/optimum/` has no suite-level conftest and takes the default. An exported `VLLM_RBLN_USE_VLLM_MODEL` therefore changes what `tests/optimum/` exercises without failing.
- Do not set `VLLM_RBLN_USE_VLLM_MODEL` inside a test to escape its suite.

## Patching upstream vLLM

`vllm_rbln/patches/` adapts upstream vLLM for RBLN. Both mechanisms live in `patches/registry.py`, both take a required `reason`, and `register_ops()` applies registrations before patches — on the native path only.

**Use upstream's own extension points first.** `@add_registration` wraps a callback that registers through a vLLM API: `base_cls.register_oot(...)`, a `PlatformEnum.OOT` entry in a kernel registry, and so on. `patches/oot.py` is the worked example. A registration survives an upstream refactor; a replaced symbol does not.

**`@register_patch` overwrites an upstream symbol and is the last resort.** Use it only when no extension point exists, and make `reason` say what upstream cannot express — not what the replacement does. When you are unsure whether an extension point exists, ask instead of defaulting to a patch.

These are the cases that legitimately reach a patch:

- Upstream's shape cannot express an RBLN constraint, such as a KV cache that has to enter the compiled graph as an input, or a kernel RBLN does not provide.
- An upstream module needs adapting to an RBLN interface, and replacing one method beats reimplementing the model.
- An upstream bug, or a fix that is not released yet. Cite the upstream issue and give the version that removes the need, such as `TODO(vllm>=0.26.0): delete`. A temporary patch with no removal condition becomes a permanent one.

The registry rules:

- Never `setattr` an upstream symbol directly. Every replacement goes through the registry, which verifies that it took.
- A new module under `patches/` must be added to the import list in `patches/__init__.py`. A decorator in a module nobody imports registers nothing.
- Duplicate keys and duplicate targets raise. Two patches may share a target only when their `condition` predicates are mutually exclusive.
- `priority` applies `0` first and `100` last, default `50`. `apply_immediately` patches at import time, for targets that import-time code snapshots before `apply_registered_patches()` runs; it cannot be combined with an explicit `priority`.
- Pass `verify` when "the attribute is now our object" does not prove the patch took effect.

## Language

Everything in the repository is written in English: code, comments, docstrings, log and error messages, test names, docs, commit messages, PR titles and bodies. This extends the English requirement in `CONTRIBUTING.md`.

Conversation with the user is not a repository artifact. Reply in the language the user writes in.

## What not to publish

This repository is public. Absolute measurements stay out of it: throughput, latency, memory footprint, and accuracy from an internal run do not belong in code, comments, tests, docs, commit messages, or PR descriptions. Benchmark scripts live here; their results do not.

A relative change is fine. Say that something got a third faster, not what the two numbers were, and keep the raw figures to an internal channel.

Upstream vLLM asks for eval results in the PR description. That convention does not apply here.

## Scope

- Look for an existing utility or mechanism before writing a significant amount of new code.
- Do not create a helper that is called once. Inline it.
- Do not build an abstraction for a single use. Three similar lines beat a premature abstraction.
- Do not add error handling for states that cannot occur.
- Do not create files nobody asked for — docs, examples, scripts, changelogs.
- Delete the scratch files you made while iterating.
- Prefer fixing the underlying problem over a local workaround, even when that means a larger refactor. If the refactor is out of scope, say so and ask.

## Failure handling

Code either succeeds or fails with a clear error.

- Do not catch `Exception` or use a bare `except` to get past a problem.
- `except: pass` and `except: continue` hide the failure. Do not write them.
- Do not route a case that cannot happen into an `else`. Use `assert` or `raise`. An `if` is for two paths that both really occur.
- Do not add a fallback, a default, or a silent recovery that was not asked for.
- Delete removed code completely: no renamed `_var` leftovers, no dead re-exports, no `# removed` comments.

## Comments

- Do not restate what the code says. If a comment narrates the next line, delete it and let the name carry the meaning.
- Do not describe your changes or address the reader in a comment. The commit message and the PR body are for that.
- Do not add or edit comments in code you did not otherwise change.
- Comments are for invariants, non-obvious constraints, and why an unusual approach was taken. These may be long; explain them properly. The module comments in `vllm_rbln/__init__.py` and `vllm_rbln/envs.py` are the intended level.
- Assume the reader knows vLLM and RBLN hardware.

## Tests

- A test must fail without the change it covers. Verify that; do not assume it.
- Cover what the change touched. Do not test pre-existing logic, third-party functions, or statically defined values.
- Extend an existing file, `conftest.py` fixture, or `utils.py` helper before adding a new file.
- Do not add a test that is skipped as a placeholder. A test that never runs covers nothing while reading as coverage that exists.
- `--strict-markers` turns a misspelled mark into a collection error, but a mark you leave off is silent. Forgetting `model_compile` is how a whole-model compile ends up in the lane that runs on every PR.
- The `model_compile`, `use_device`, and `maybe_use_device` marks exist in `tests/native/` only.
- `tests/optimum/optimum_correctness/` holds `fire` CLI scripts, not pytest tests. Do not add `test_*.py` there.

## Reporting

- If you cannot finish, do not produce a plausible partial result. Stop and say what blocked you.
- End every task by saying what you did not do: tests you could not run, hardware you do not have, assumptions you could not check.
- Do not state a hypothesis as a fact.
- Tests that break after your change are your regressions. Debug them. Do not stash or revert to check whether they also fail on `main`.

## When a rule is in the way

Some of the rules above cost something to follow. When you believe a case is a real exception — a comment no naming can replace, a workaround whose root fix is out of scope, a change that must touch both model paths — stop and ask before writing the code. Do not decide it alone, and do not work around it quietly.

## Commits and PRs

Follow the existing convention: `type(scope): summary`, for example `fix(mega-cache): key the bundle on the warm-up graph set`.

Explain intent. Summarise what changed in a line or two, then spend the space on what the diff cannot show — the reason behind an unusual approach, a constraint that forced the shape, a tradeoff that was weighed. Do not walk through the diff file by file; that is the part a reader can already see.

## Skills

Read and follow the matching skill at the point it applies:

- Adding or changing tests under `tests/native/`: `.claude/skills/writing-tests/SKILL.md`
- Investigating a bug, a test failure, or unexpected behavior: `.claude/skills/debugging/SKILL.md`
- Before claiming a change is done: `.claude/skills/finishing-a-change/SKILL.md`
