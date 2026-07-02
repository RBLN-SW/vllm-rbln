#!/usr/bin/env python3
"""Quantify all_reduce(N+1) <-> forward(N) overlap from host perf_counter spans.

Feed it a log captured with VLLM_RBLN_SPAN_LOG=1 (async run). The log has, per
decode step and per rank (pid):
    SPAN fwd       <pid> <t0> <t1>     # forward, on the device executor thread
    SPAN allreduce <pid> <t0> <t1>     # DP num_tokens_across_dp all_reduce, main

Why host spans and NOT torch.profile/Perfetto: on RBLN the deferred forward runs
on a raw worker thread that torch.profiler never entered, so record_function
there emits 0 events and kineto mis-attributes the RBLN runtime op to the main
tid -> Perfetto shows a false 0% overlap. perf_counter is a process-wide clock,
so these spans are directly comparable across threads.

Usage:  python3 overlap_from_spans.py /path/to/span.log
"""
import re, sys
from collections import defaultdict

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/L18_rf.log"
fwd = defaultdict(list); ar = defaultdict(list)
for line in open(src):
    m = re.search(r"SPAN (fwd|allreduce) (\d+) ([0-9.]+) ([0-9.]+)", line)
    if not m:
        continue
    kind, pid, a, b = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    (fwd if kind == "fwd" else ar)[pid].append((a, b))

def union(iv):
    if not iv:
        return []
    iv = sorted(iv); o = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= o[-1][1]:
            o[-1][1] = max(o[-1][1], b)
        else:
            o.append([a, b])
    return o

def dur(iv):
    return sum(b - a for a, b in union(iv))

def isect(a, b):
    a = union(a); b = union(b); i = j = 0; t = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t

# Decode-only: drop >10ms spans (prefill forwards + straggler all_reduce WAITs
# that dilute the full-run intersection and hide the real decode overlap).
DECODE_MAX = 0.010
TALL = TCOV = 0.0
print("per-rank(pid):  #ar  ar_dur   overlap  (%%ar)   spans_inside/total")
for pid in sorted(fwd):
    fd = [(a, b) for a, b in fwd[pid] if (b - a) < DECODE_MAX]
    ad = [(a, b) for a, b in ar.get(pid, []) if (b - a) < DECODE_MAX]
    ov = isect(fd, ad); TALL += dur(ad); TCOV += ov
    inside = sum(1 for (a, b) in ad if any(a < fb and fa < b for fa, fb in fd))
    print(f"  {pid}: n={len(ad):3d} ar={dur(ad)*1000:6.1f}ms overlap={ov*1000:6.1f}ms "
          f"({100*ov/max(dur(ad),1e-9):4.1f}%)  {inside}/{len(ad)}")
print(f"\n>>> decode all_reduce overlapped by forward = "
      f"{TCOV*1000:.1f}ms / {TALL*1000:.1f}ms = {100*TCOV/max(TALL,1e-9):.1f}%")
