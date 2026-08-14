# SPDX-License-Identifier: Apache-2.0
"""Paired analysis of the controlled multi-turn suite.

Pairs runs by (workload, seed) so machine drift and workload variation cancel,
reports the paired per-run deltas, and separately pools per-request samples for
a Welch t.

The "workload equivalence" table checks the two modes saw the same work. Note
what it does *not* check: `approx_cached_percent` is computed client-side from
the dataset (history tokens / input tokens), so it says nothing about whether
the server hit its cache. For that, read `vllm:prefix_cache_hits_total` from
/metrics -- mistaking the two cost a day here.

Usage: analyze.py [results_dir]
"""

import json
import math
import os
import pathlib
import re
import statistics as st
import sys

SP = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
MODES = ("subblock", "pagelayout")
WORKLOADS = ("short", "long")
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "1 2 3").split())
METRICS = ("ttft_ms", "tpot_ms", "latency_ms")


def load(mode, wl, seed):
    p = SP / f"st_{mode}_{wl}_{seed}.json"
    return json.load(open(p)) if p.exists() else None


def runtime(mode, wl, seed):
    p = SP / f"run_{mode}_{wl}_{seed}.log"
    if not p.exists():
        return None, None
    txt = p.read_text()
    rt = re.search(r"benchmark runtime: ([0-9.]+) sec", txt)
    rp = re.search(r"requests per second: ([0-9.]+)", txt)
    return (float(rt.group(1)) if rt else None, float(rp.group(1)) if rp else None)


def mean(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return st.mean(v) if v else float("nan")


def ci95(xs):
    if len(xs) < 2:
        return float("nan")
    return 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def main():
    print("== workload equivalence (should match within a pair) ==")
    print(f"{'wl/seed':10} {'reqs A/B':>10} {'in_tok A/B':>17} {'cached% A/B':>15}")
    for wl in WORKLOADS:
        for s in SEEDS:
            a, b = load("subblock", wl, s), load("pagelayout", wl, s)
            if not a or not b:
                continue
            tok_a, tok_b = mean(a, "input_num_tokens"), mean(b, "input_num_tokens")
            pct_a = mean(a, "approx_cached_percent")
            pct_b = mean(b, "approx_cached_percent")
            print(
                f"{wl}/{s:<7} {len(a):>4}/{len(b):<5} "
                f"{tok_a:>8.0f}/{tok_b:<8.0f} {pct_a:>6.2f}/{pct_b:<7.2f}"
            )

    print("\n== paired per-run deltas (negative = page layout better) ==")
    for wl in WORKLOADS:
        print(f"\n-- workload: {wl} --")
        header = f"{'seed':6}" + "".join(f"{m.replace('_ms', ''):>26}" for m in METRICS)
        print(header + f"{'req/s A->B':>22}")
        deltas = {m: [] for m in METRICS}
        rps_d = []
        for s in SEEDS:
            a, b = load("subblock", wl, s), load("pagelayout", wl, s)
            if not a or not b:
                continue
            row = f"{s:<6}"
            for m in METRICS:
                ma, mb = mean(a, m), mean(b, m)
                d = 100 * (mb - ma) / ma
                deltas[m].append(d)
                row += f"{ma:>9.1f}->{mb:<8.1f}{d:>+6.1f}%"
            _, ra = runtime("subblock", wl, s)
            _, rb = runtime("pagelayout", wl, s)
            if ra and rb:
                rps_d.append(100 * (rb - ra) / ra)
                row += f"{ra:>7.3f}->{rb:<6.3f}{100 * (rb - ra) / ra:>+6.1f}%"
            print(row)
        print(f"{'mean':6}", end="")
        for m in METRICS:
            d = deltas[m]
            print(
                f"{'':>18}{st.mean(d):>+6.1f}%" if d else f"{'':>25}",
                end="",
            )
        if rps_d:
            print(f"{'':>15}{st.mean(rps_d):>+6.1f}%", end="")
        print()
        for m in METRICS:
            d = deltas[m]
            if len(d) > 1:
                print(
                    f"       {m:12} paired delta {st.mean(d):+.1f}% "
                    f"+/- {ci95(d):.1f} (95% CI, n={len(d)})"
                )

    print("\n== pooled per-request Welch t (all seeds, per workload) ==")
    for wl in WORKLOADS:
        for m in METRICS:
            A = [
                r[m] for s in SEEDS for r in (load("subblock", wl, s) or []) if r.get(m)
            ]
            B = [
                r[m]
                for s in SEEDS
                for r in (load("pagelayout", wl, s) or [])
                if r.get(m)
            ]
            if len(A) < 2 or len(B) < 2:
                continue
            sea = st.stdev(A) / math.sqrt(len(A))
            seb = st.stdev(B) / math.sqrt(len(B))
            t = (st.mean(A) - st.mean(B)) / math.sqrt(sea**2 + seb**2)
            print(
                f"{wl:6} {m:12} A={st.mean(A):8.2f} B={st.mean(B):8.2f} "
                f"delta={100 * (st.mean(B) - st.mean(A)) / st.mean(A):+6.1f}%  "
                f"t={t:5.2f}  nA={len(A)} nB={len(B)}"
            )


if __name__ == "__main__":
    main()
