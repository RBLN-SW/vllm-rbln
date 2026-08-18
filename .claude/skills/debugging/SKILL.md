---
name: debugging
description: Use when investigating a bug, a failing test, or unexpected behavior in vllm-rbln — the environment and model-path traps that make a reproduction lie, and the evidence rules that keep a wrong cause from looking confirmed.
---

# Debugging vllm-rbln

Two things make a debugging session go wrong here: reproducing something other than the reported bug, and confirming a cause that is not the cause. This skill is about those. It does not restate general debugging method.

## 1. Before you trust the reproduction

Get a reproducer before changing anything, and narrow it to the smallest input and configuration that still fails.

**If you cannot reproduce it, stop and report that.** Do not fix a bug you have not seen. An untested fix for an unverified report costs a reviewer more than no patch.

Check these before concluding anything, because each one makes a reproduction silently exercise something other than what you think:

- **`tests/native/` scrubs `VLLM_RBLN_*`.** Exported values do not reach it; the session options are the only way in. A knob you "set" may never have applied.
- **`--device-tensor` is session-wide.** `platform.py` resolves it at module scope, so a per-test attempt to change it does nothing and the test still passes.
- **Marks decide the process.** `use_device` and `model_compile` items run in a spawned child. A failure that only appears in one of the two contexts is about the process, not the logic.

## 2. Make the evidence discriminate

Evidence that fits your hypothesis is easy to find and proves nothing — a wrong cause has confirming evidence too. Force the evidence to choose between explanations: an observation that every candidate predicts has narrowed nothing.

- **State what would be true if you are wrong**, and check that specifically.
- **Predict the output before you run.** Write down what you expect. Matching means your model of the system is right; a surprise means it is wrong, and that is the finding.
- **The cause must explain every symptom**, not just the loudest one. A leftover unexplained detail usually means a second cause or the wrong one.

Report which parts you verified and which you inferred. Never state a hypothesis as a fact.

## 3. Fix the cause, not its surroundings

Change the thing that is wrong — not a caller, not a wrapper, not a guard around the symptom.

A workaround is allowed when the real fix is out of reach, as long as it is declared: name the cause, say why the fix is out of reach, and get agreement. A patch under `patches/` is that shape, with the `reason` recording what upstream cannot express and review deciding whether to accept it. What is not allowed is a guard added because you could not find the cause.

## 4. If the fix does not work, discard it

Revert it and form another hypothesis. Do not stack a second patch on a failed first one.

A failed fix is information: the cause was probably wrong. Go back to step 2 rather than making the current theory work harder.

Never widen an exception, add a default, or skip a branch to make the failure go away. Reaching for that means you are fixing without a cause.

## 5. Verify against the reproducer

Run the reproducer again, plus the tests around the code you touched, and read the output. A fix that has not been run is not a fix. Say so plainly if you could not run it.

## 6. Check both model paths

- Which path does the bug live on?
- Does the same defect exist on the other path?
- Does the fix change behavior on the other path?

Only `__init__.py` and `platform.py` branch on the flag. If the fix seems to need a new `if envs.VLLM_RBLN_USE_VLLM_MODEL` anywhere else, the shape is wrong — stop and ask.

## 7. Report the gaps

State explicitly what you could not build, run, or reproduce; hardware or models you did not have; and anything you inferred rather than verified.

Tests that broke after your change are your regressions. Debug them. Do not stash or revert to check whether they also fail on `main`.
