# SPDX-License-Identifier: Apache-2.0
"""Byte-diff two greedy probes. Any difference in the cached rounds is a bug."""

import json
import os
import pathlib
import sys

SP = pathlib.Path(os.environ.get("OUT_DIR", "results"))


def main() -> None:
    a_name, b_name = sys.argv[1], sys.argv[2]
    with open(SP / f"probe_{a_name}.json") as handle:
        a = json.load(handle)
    with open(SP / f"probe_{b_name}.json") as handle:
        b = json.load(handle)

    bad = 0
    for round_name in a:
        for i, (ta, tb) in enumerate(zip(a[round_name], b[round_name])):
            same = ta == tb
            bad += not same
            print(f"{'OK  ' if same else 'DIFF'} {round_name}/{i}")
            if not same:
                print(f"  {a_name}: {ta[:200]!r}")
                print(f"  {b_name}: {tb[:200]!r}")
    print(f"\n{bad} differing completions")


if __name__ == "__main__":
    main()
