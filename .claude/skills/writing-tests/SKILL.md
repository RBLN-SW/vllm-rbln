---
name: writing-tests
description: Use when adding or modifying tests under tests/native/ — covers placement, the model_compile/use_device/maybe_use_device marks, the session options, and proving a new test fails before the change it covers.
---

# Writing tests in the native suite

## Scope

This skill covers `tests/native/`, the vLLM-native path.

For `tests/optimum/` the conventions are not settled. Do not carry the rules below into that suite — the marks and session options gate nothing there. Follow the patterns in the nearest existing directory, and ask before introducing a new one.

## 1. Design before writing

Answer these four before you write a line. If you cannot, ask instead of guessing.

1. What is the module under test for?
2. What is its input/output contract?
3. What failure is this test guarding against?
4. What is the cheapest level that catches that failure? Prefer unit over integration over whole-model.

## 2. Where the file goes

Mirror the source tree:

```
vllm_rbln/<path>/<module>.py  ->  tests/native/<path>/test_<module>.py
```

Root modules (`envs.py`, `platform.py`, …) get root-level files in `tests/native/`.

Two directories are not mirrors. They hold whole-model tests, split by the question the test asks:

- `compile/` — does it compile and run at all? Smoke over large models, no reference output.
- `basic_correctness/` — is the output right? Greedy generation compared against an HF reference on small models.

`compilation/` is the mirror of `vllm_rbln/compilation/` and holds unit tests only — option building, backend conformance, no NPU. A test that actually compiles a model belongs in one of the two directories above, not here.

## 3. Marks

Marks decide whether an item runs at all and whether it runs in this process or a fresh one. Getting one wrong is expensive, and the filename does not help — `conftest.py` reads only the mark.

| Mark               | Effect                                                                          |
| ------------------ | ------------------------------------------------------------------------------- |
| `model_compile`    | Whole-model compile, minutes. Skipped unless `--model-compile`. Always spawned.  |
| `use_device`       | Always opens the NPU. Always spawned.                                            |
| `maybe_use_device` | Opens the NPU when `device_type != "cpu"`, and is spawned in that case.         |
| none               | Runs in this process.                                                            |

Spawning is not an optimization. A test that opens the device in the parent process pins it for the rest of the session, so `use_device` and `maybe_use_device` exist to force a fresh process.

Compiling an individual op is deliberately left unmarked so it stays in the default lane. Only a whole-model compile takes `model_compile`.

`--strict-markers` turns a misspelled mark into a collection error, so a wrong name fails loudly. A mark you leave off does not: forgetting `model_compile` puts a multi-minute test into the lane that runs on every PR.

## 4. Naming end-to-end tests

Name a file `*_e2e.py` when every test in it is a whole-model compile. Skip the suffix where the directory already says it — `basic_correctness/`, `compile/`.

There is no `e2e/` directory today; four such files sit in four different subsystems, and one-file directories would be worse. When a single directory reaches three of them, move them into an `e2e/` subdirectory there, matching upstream's `tests/v1/e2e/`.

## 5. Session options

Read `tests/native/conftest.py` for the full text; the constraints that bite:

- `--num-hidden-layers N` builds only the first N decoder layers to cut compile time, and `hf_runner` truncates to the same N so comparisons stay like-for-like. Default is 3; `0` means the whole model.
- The suite scrubs exported `VLLM_RBLN_*` variables. These options are the way in — do not read the environment directly to get around them.

## 6. Reuse what is there

Check before adding anything new:

- `tests/native/conftest.py`, and the local `conftest.py` in your subdirectory
- `tests/native/utils.py`, `runners.py`, `model_specs.py`, `vllm_config.py`
- the `utils.py` beside your target, e.g. `v1/worker/utils.py`

Shared helpers go in the `utils.py` nearest to their users. Do not add a new `helpers_*.py`.

A new file needs the Apache header every other file carries, and a package directory needs an `__init__.py`. Copy both from a sibling.

## 7. Prove the test fails

A test that passes without the change covers nothing. Before you commit:

```bash
git stash push -- <production files only>
uv run --no-sync pytest <your new test> -x   # must fail
git stash pop
```

Watch it fail and read the failure — a test can fail for the wrong reason. If running it is impossible here, say so explicitly rather than assuming it is red.

## 8. Reject these

- Asserting a statically defined value against itself
- Testing wiring with no behavior behind it
- A negative test for logic that was removed
- Duplicating the implementation's logic inside the test
- Mocking what a real object or a temporary directory would do
- A placeholder test that is skipped
