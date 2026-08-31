---
name: finishing-a-change
description: Use before claiming a change is done, fixed, ready, or passing, and before you commit, open a PR, or run gh pr create — runs the checks, reviews the diff against the repository rules, fills in the PR template, and reports what was not done.
---

# Finishing a change

By the time you get here the session is long and the instructions you read at the start have faded. Work through this list rather than from memory.

## 1. Run the hooks

```bash
uvx pre-commit run --files <every file you changed>
```

`pre-commit` is not a project dependency; `uvx` is how it runs here. Read the output. Passing means you saw it pass.

A new `.py` file must carry the Apache header, or `check-license-header` fails.

## 2. Run the tests

Run the tests covering what you changed, and include the command and its result in your reply. "Should pass" is not a result. If you could not run them, say that instead of implying you did.

## 3. Read your own diff

```bash
git diff
```

Look for each of these:

- **Files nobody asked for** — docs, examples, scripts, changelogs
- **Scratch files** left over from iterating
- **Single-use helpers** that should be inlined
- **Abstractions** built for one call site
- **Comments that narrate the code**, describe your changes, or address the reader
- **Comments added to code you did not otherwise change**
- **Comment blocks over 5 lines, a `reason=` over 400 characters, docstring prose over 15 lines** — the surplus belongs in the PR description, not the file, and not the commit message
- **Swallowed failures** — `except Exception`, bare `except`, `except: pass`, an unrequested fallback or default
- **`else` branches for cases that cannot happen** — should be `assert` or `raise`
- **Dead leftovers** — renamed `_var`, unused re-exports, `# removed` comments
- **Anything not in English**
- **Absolute measurements** — throughput, latency, memory, or accuracy from an internal run. Take them out; this repository is public. A relative change is fine.

Repository-specific checks:

- No new branch on `VLLM_RBLN_USE_VLLM_MODEL` outside `__init__.py` and `platform.py`.
- A new env var must appear in **three** places in `envs.py`: the `TYPE_CHECKING` block, the `environment_variables` dict, and either `RBLN_COMPILE_ENV` or `RBLN_NON_COMPILE_ENV`.
- No `model_compile`, `use_device`, or `maybe_use_device` mark under `tests/optimum/` — nothing gates them there.
- No `test_*.py` under `tests/optimum/optimum_correctness/` — those are `fire` scripts.
- No test that is skipped as a placeholder.

## 4. Report what you did not do

State it explicitly, every time:

- Tests you could not run, and why
- Hardware or models you did not have
- Assumptions you could not verify
- Parts of the request you left out

An empty list is a claim. Only write it if it is true.

## 5. Write the commit and PR

- `type(scope): summary`, matching the existing history
- Summarise what changed in a line or two, then explain what the diff cannot show. That explanation goes in the PR description; the squash merge discards the commit body
- Say which model path or paths the change affects
- English
- A perf change may give a relative change, never an absolute measurement
- Fill in `.github/PULL_REQUEST_TEMPLATE.md`. `gh pr create --body` does not apply it, so read the file and follow its sections rather than writing free-form prose.
