#!/usr/bin/env python3
"""Compare arms by generated text: same prompt in, same text out?

One of two correctness checks worth running on async scheduling, and the one
that needs no reference run: same prompt in, same text out. The other is the
executor's native golden, which async does reach now that it carries logprobs
through the deferred output.

Neither replaces the other. Comparing the arms to each other cannot see a defect
in the sampler they share - measured: a bad runtime patch left both arms corrupt
and agreeing 32/32, while the text had moved 30k characters from a known-good
build. Compare against something that did not run the same code.

Reads the outputs.json that ab_throughput.sh lifts out of the run cache
(<arm>_r<N>.outputs.json). Entries are joined on the prompt string, never on
list position: a positional join silently compares different requests whenever
batch composition shifts.

Usage:  cmp_texts.py A.outputs.json B.outputs.json [C.outputs.json ...]
        cmp_texts.py --show 3 A.outputs.json B.outputs.json
"""

import json
import sys
from itertools import combinations
from pathlib import Path


def load(path):
    """Return {prompt: text}; refuses to guess when a prompt repeats."""
    entries = json.loads(Path(path).read_text())
    by_prompt = {}
    for entry in entries:
        prompt = entry["prompt"]
        if prompt in by_prompt and by_prompt[prompt] != entry.get("text", ""):
            raise SystemExit(f"{path}: duplicate prompt with differing text; cannot join")
        by_prompt[prompt] = entry.get("text", "")
    return by_prompt


def first_diff(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main(argv):
    show = 0
    files = []
    it = iter(argv)
    for arg in it:
        if arg == "--show":
            show = int(next(it))
        else:
            files.append(arg)
    if len(files) < 2:
        sys.exit(__doc__)

    arms = {Path(f).name.replace(".outputs.json", ""): load(f) for f in files}
    for name, texts in arms.items():
        print(f"{name}: {len(texts)} prompts")

    print(f"\n{'pair':<44}{'identical':>12}{'differ':>8}{'median 1st diff':>18}")
    worst = 0
    for (na, a), (nb, b) in combinations(arms.items(), 2):
        shared = [p for p in a if p in b]
        if not shared:
            print(f"{na} vs {nb}: no shared prompts")
            continue
        diffs = [p for p in shared if a[p] != b[p]]
        cuts = sorted(first_diff(a[p], b[p]) for p in diffs)
        median = cuts[len(cuts) // 2] if cuts else "-"
        print(
            f"{na + ' vs ' + nb:<44}{len(shared) - len(diffs):>7}/{len(shared):<4}"
            f"{len(diffs):>8}{str(median):>18}"
        )
        worst = max(worst, len(diffs))
        for p in diffs[:show]:
            cut = first_diff(a[p], b[p])
            print(f"    prompt {p[-60:]!r}")
            print(f"      diverges at char {cut}")
            print(f"      {na}: ...{a[p][cut:cut + 70]!r}")
            print(f"      {nb}: ...{b[p][cut:cut + 70]!r}")

    print("\nA differing pair means the two arms answered the same prompt")
    print("differently. Greedy decoding (temperature 0) makes that a defect,")
    print("not noise - one flipped token cascades through the rest of the text.")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
