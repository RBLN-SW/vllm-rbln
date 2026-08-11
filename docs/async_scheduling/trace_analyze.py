#!/usr/bin/env python3
"""Analyse an RBLN vLLM torch trace (.pt.trace.json.gz).

    trace_analyze.py TRACE               # threads + span census
    trace_analyze.py TRACE --step        # one steady decode step, marker timeline
    trace_analyze.py TRACE --overlap     # gloo collective vs main-thread work
    trace_analyze.py TRACE --window 4314.825   # dump everything around a time (ms)
    trace_analyze.py TRACE --gap         # census of what fills a chosen sub-window

The trace has no NPU track, so device occupancy is NOT observable. What IS
observable: which thread each span runs on, and wall-clock overlap between
threads (gloo runs on pt_gloo_runloop, get_output on the async output thread).

Pitfall that produced two wrong conclusions on this branch: do NOT filter spans
by duration before concluding "nothing runs here", and remember that enclosing
frames (execute_model, worker_busy_loop) make naive union-coverage 100%. Count
spans that *start* in the window, unfiltered.
"""

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict

MARKERS = [
    "worker_base.py(340): execute_model",
    "sample_tokens",
    "distributed_c10d.py(2978): all_reduce",
    "gloo:all_reduce",
    "sync_runtime.py(208): run",
    "nn.Module: RBLNSampler_0",
    "_bookkeeping_sync",
    "get_output",
    "aten::copy_",
    "shm_broadcast.py(765): dequeue",
    "multiproc_executor.py(916): handle_output",
]


def load(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    ev = d["traceEvents"]
    tname = {
        e["tid"]: e["args"]["name"]
        for e in ev
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
    return X, tname


def pick(X, needle, tid=None):
    return sorted(
        (e["ts"], e["ts"] + e["dur"], e["tid"])
        for e in X
        if needle in e["name"] and (tid is None or e["tid"] == tid)
    )


def cmd_census(X, tname):
    print(f"events: {len(X)}\n=== threads ===")
    per = defaultdict(Counter)
    for e in X:
        per[e["tid"]][e["name"][:58]] += 1
    for tid in sorted(per, key=lambda t: -sum(per[t].values())):
        print(f"\ntid={tid} [{tname.get(tid, '?')}] spans={sum(per[tid].values())}")
        for n, c in per[tid].most_common(5):
            print(f"   {c:>7}  {n}")
    print("\n=== markers (n / median ms / max ms / tids) ===")
    for m in MARKERS:
        hits = [e for e in X if m in e["name"]]
        if not hits:
            continue
        d = sorted(e["dur"] / 1000.0 for e in hits)
        tids = Counter(e["tid"] for e in hits)
        print(
            f"  {m[:44]:<46} n={len(d):<5} med={statistics.median(d):8.3f} "
            f"max={d[-1]:8.3f}  tids={dict(tids)}"
        )


def steady_steps(X):
    """execute_model spans in the decode range (prefill ones are far longer)."""
    return [
        (a, b, t)
        for a, b, t in pick(X, "worker_base.py(340): execute_model")
        if 10_000 < (b - a) < 20_000
    ]


def cmd_step(X, tname):
    em = steady_steps(X)
    if not em:
        print("no steady decode execute_model found")
        return
    print(
        f"steady decode execute_model: n={len(em)} "
        f"median={statistics.median([(b - a) / 1000 for a, b, _ in em]):.2f} ms"
    )
    a0, b0, tid = em[len(em) // 2]
    nxt = next((a for a, _, _ in em if a > b0), b0 + 20_000)
    rows = []
    for m in MARKERS:
        for a, b, t in pick(X, m):
            if a0 - 200 <= a <= nxt:
                rows.append((a, b, m, t))
    rows.sort()
    print(f"\n=== one decode step (execute_model {(b0 - a0) / 1000:.2f} ms) ===")
    for a, b, m, t in rows:
        print(
            f"  {(a - a0) / 1000:9.3f} -> {(b - a0) / 1000:9.3f}  "
            f"({(b - a) / 1000:6.3f})  [{tname.get(t, t)[:18]:<18}] {m}"
        )


def cmd_overlap(X, tname):
    gloo = pick(X, "gloo:all_reduce")
    if not gloo:
        print("no gloo:all_reduce spans")
        return
    total = sum(b - a for a, b, _ in gloo) / 1000.0
    print(f"gloo total = {total:.1f} ms over {len(gloo)} collectives\n")

    def ov(spans, label):
        tot, hit = 0.0, 0
        for ga, gb, _ in gloo:
            o = sum(max(0, min(gb, b) - max(ga, a)) for a, b, _ in spans) / 1000.0
            tot += o
            hit += o > 0.01
        print(
            f"  gloo n {label:<26} {tot:8.1f} ms "
            f"({tot / total * 100:5.1f}% of gloo, {hit}/{len(gloo)} collectives)"
        )

    ov(pick(X, "sync_runtime.py(208): run"), "sync_runtime.run")
    ov(pick(X, "nn.Module: RBLNSampler_0"), "RBLNSampler_0")
    ov(pick(X, "worker_base.py(340): execute_model"), "execute_model")


def cmd_window(X, tname, at_ms, span_ms):
    t0 = min(e["ts"] for e in X)
    T = t0 + at_ms * 1000.0
    w = sorted(
        (e for e in X if T <= e["ts"] <= T + span_ms * 1000.0),
        key=lambda e: (e["ts"], -e["dur"]),
    )
    print(f"trace t0={t0:.0f} us; window +{at_ms} ms .. +{at_ms + span_ms} ms")
    print(f"spans STARTING in window: {len(w)}")
    for e in w[:60]:
        print(
            f"  {(e['ts'] - T) / 1000:9.3f} +{e['dur'] / 1000:8.3f} ms "
            f"[{tname.get(e['tid'], e['tid'])[:20]:<20}] {e['name'][:66]}"
        )


def cmd_gap(X, tname):
    """Per-step census of the region after the forward dispatch returns."""
    res, n = defaultdict(lambda: [0.0, 0]), 0
    for a0, b0, tid in steady_steps(X):
        runs = [
            e
            for e in X
            if e["tid"] == tid
            and "sync_runtime.py(208): run" in e["name"]
            and a0 <= e["ts"] <= b0
        ]
        if not runs:
            continue
        fend = max(e["ts"] + e["dur"] for e in runs)
        if b0 - fend < 500:
            continue
        n += 1
        ins = sorted(
            (
                e
                for e in X
                if e["tid"] == tid and fend <= e["ts"] and e["ts"] + e["dur"] <= b0
            ),
            key=lambda e: (e["ts"], -e["dur"]),
        )
        last = -1
        for e in ins:
            if e["ts"] >= last:
                r = res[e["name"][:66]]
                r[0] += e["dur"] / 1000.0
                r[1] += 1
                last = e["ts"] + e["dur"]
    print(f"forward dispatch end -> execute_model end, top-level spans ({n} steps)")
    print(f"{'per-step ms':>12}{'n':>7}  name")
    for k, (t, c) in sorted(res.items(), key=lambda x: -x[1][0])[:15]:
        print(f"{t / max(n, 1):>12.3f}{c:>7}  {k}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--step", action="store_true")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--gap", action="store_true")
    p.add_argument("--window", type=float, help="offset in ms from trace start")
    p.add_argument("--span", type=float, default=3.0, help="window width in ms")
    a = p.parse_args()

    X, tname = load(a.trace)
    if a.step:
        cmd_step(X, tname)
    elif a.overlap:
        cmd_overlap(X, tname)
    elif a.gap:
        cmd_gap(X, tname)
    elif a.window is not None:
        cmd_window(X, tname, a.window, a.span)
    else:
        cmd_census(X, tname)


if __name__ == "__main__":
    main()
