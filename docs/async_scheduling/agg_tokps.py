#!/usr/bin/env python3
"""Aggregate warm tok/s samples from vllm-rbln-exec rbln-run logs.

The readings come from vLLM's own tqdm line, so this parser is independent of
which executor drove the run.

One reading is emitted per rank per generate call, at the "Processed prompts:
100%" completion - tqdm rewrites the line continuously, so any earlier reading
is partial and the last line in the file is not special. Reading a single value
(e.g. `grep | tail -1`) produced several wrong conclusions on this branch; the
spread between samples of the *same* build is 1-3%.

The first generate call of each rank is cold and lands in a clearly separate low
cluster (~20% below), so it is dropped by a relative threshold rather than by
position - the ranks interleave in the log and position is unreliable.

Usage:  agg_tokps.py LOG [LOG ...]
        agg_tokps.py --pool async=_ab/asyncsched_*.log --pool sync=_ab/sync_*.log
"""

import re
import statistics
import sys
from pathlib import Path

DONE = re.compile(
    r"Processed prompts: 100%\|[^|]*\|\s*\d+/\d+[^\n]*?"
    r"est\. speed input: [0-9.]+ toks/s, output: ([0-9.]+) toks/s"
)
WARM_FRACTION = 0.8


def warm_samples(path):
    vals = [
        float(m.group(1)) for m in DONE.finditer(Path(path).read_text(errors="ignore"))
    ]
    if not vals:
        return []
    hi = max(vals)
    return [v for v in vals if v >= WARM_FRACTION * hi]


def describe(label, v):
    if not v:
        print(f"{label:<34}{'-':>5}  no warm samples")
        return None
    med, lo, hi = statistics.median(v), min(v), max(v)
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [med, med, med]
    print(
        f"{label:<34}{len(v):>5}{med:>10.1f}{lo:>10.1f}{hi:>10.1f}"
        f"{q[0]:>10.1f}{q[2]:>10.1f}{(hi - lo) / med * 100:>9.1f}%"
    )
    return med


def main(argv):
    header = (
        f"{'run / pool':<34}{'n':>5}{'median':>10}{'min':>10}{'max':>10}"
        f"{'p25':>10}{'p75':>10}{'spread':>10}"
    )
    pools, files = {}, []
    for a in argv:
        if a.startswith("--pool"):
            continue
        if "=" in a and not Path(a).exists():
            name, _, pat = a.partition("=")
            pools.setdefault(name, []).extend(sorted(Path().glob(pat)))
        else:
            files.append(a)

    print(header)
    for f in files:
        describe(f"{Path(f).parent.name}/{Path(f).stem}", warm_samples(f))
    meds = {}
    for name, fs in pools.items():
        v = [x for f in fs for x in warm_samples(f)]
        meds[name] = describe(f"[pool] {name}", v)
    if len(meds) == 2 and all(meds.values()):
        (na, a), (nb, b) = meds.items()
        print(f"\n{na}/{nb} median ratio = {a / b:.3f}")
    print(
        "\nreminder: differences below ~3% are inside run-to-run noise "
        "(docs/async_scheduling.md section 6)."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
