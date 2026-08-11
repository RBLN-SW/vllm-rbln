#!/usr/bin/env python3
"""Report the golden Pearsonr and the determinism verdict per arm.

Local counterpart of the CI's judge.py, extended to several arms at once:
ab_throughput.sh writes one <arm>_r<N>.result.json per run, and both numbers
we care about land in the same place - the rbln-run task's
`additional_results`:

  Pearsonr   worst per-step correlation against the cached native golden
             (CI gates at 0.99; its own passing runs sit around 0.9965)
  run-iter   "<N> - pass|fail|error" from --run-iter, i.e. whether N repeats
             of a 6-token full-vocab-logits pass were bit-identical

They answer different questions and neither subsumes the other. Pearsonr says
"are RBLN's numerics close to native torch"; run-iter says "does this build
return the same logits twice". The async bug moves the second, not the first.

Usage:  judge_pearson.py RESULT.json [RESULT.json ...] [--threshold 0.99]
"""

import json
import sys
from pathlib import Path


def extract(path):
    """Return (pearson, run_iter_verdict) from a result.json, either may be None."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable ({e.__class__.__name__})"
    pearson = verdict = None
    for entry in data.values() if isinstance(data, dict) else []:
        if not isinstance(entry, dict):
            continue
        extra = entry.get("additional_results") or {}
        if "Pearsonr" in extra:
            pearson = extra["Pearsonr"]
        if "run-iter" in extra:
            verdict = extra["run-iter"]
    return pearson, verdict


def main(argv):
    threshold = 0.99
    files = []
    it = iter(argv)
    for a in it:
        if a == "--threshold":
            threshold = float(next(it))
        else:
            files.append(a)
    if not files:
        sys.exit(__doc__)

    print(f"{'arm':<28}{'Pearsonr':>12}{'vs thr':>9}   run-iter")
    failed = []
    for f in files:
        pearson, verdict = extract(f)
        name = Path(f).stem.replace(".result", "")
        if pearson is None:
            print(f"{name:<28}{'-':>12}{'-':>9}   {verdict or 'no golden compared'}")
            failed.append(name)
            continue
        ok = float(pearson) >= threshold
        if not ok or (verdict and verdict.endswith("fail")):
            failed.append(name)
        print(
            f"{name:<28}{float(pearson):>12.6f}{'PASS' if ok else 'FAIL':>9}"
            f"   {verdict or '(--run-iter 1, not checked)'}"
        )

    print(f"\nthreshold >= {threshold}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
