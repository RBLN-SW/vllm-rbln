#!/usr/bin/env python3
"""Main-thread SELF-time breakdown of the decode forward host-walk.

Self time = dur - sum(direct children dur). Aggregated by op name over the
decode region (main thread, excluding the first/warmup execute_model). Reveals
which host op consumes the ~68%-GIL forward-walk -> the (A) lightening target.
"""
import gzip, json, sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "ts" in e and "dur" in e]
main_tid = next(e["tid"] for e in X if "execute_model" in e.get("name", ""))
M = sorted((e for e in X if e["tid"] == main_tid), key=lambda e: (e["ts"], -e["dur"]))

# decode window: from the 2nd distinct execute_model start to the last
# sample_tokens end (skip step0 warmup + the long startup all_reduce).
em_ts = sorted({round(e["ts"]) for e in M if "execute_model" in e["name"]})
sm_ends = [e["ts"] + e["dur"] for e in M if "sample_tokens" in e["name"]]
if len(em_ts) < 2 or not sm_ends:
    print(f"not enough events: em_starts={len(em_ts)} sample_ends={len(sm_ends)}"); sys.exit()
win_lo = em_ts[1]   # skip step0 -> start at 2nd decode step
win_hi = max(sm_ends)
W = [e for e in M if e["ts"] >= win_lo and e["ts"] + e["dur"] <= win_hi + 1]
print(f"main_tid={main_tid} decode-window events={len(W)}  span={(win_hi-win_lo)/1000:.1f}ms")

# self time via stack sweep (events already sorted by ts asc, dur desc => parent before child)
self_by_name = defaultdict(float)
cnt = defaultdict(int)
stack = []  # (end, name)
for e in W:
    s, d, n = e["ts"], e["dur"], e["name"]
    end = s + d
    while stack and stack[-1][0] <= s:
        stack.pop()
    # subtract this event's dur from parent's self (parent is stack[-1])
    if stack:
        self_by_name[stack[-1][1]] -= d  # child time removed from parent self
    self_by_name[n] += d
    cnt[n] += 1
    stack.append((end, n))

# report top self-time ops
rows = sorted(self_by_name.items(), key=lambda x: -x[1])
tot = sum(v for _, v in rows if v > 0)
print(f"\ntotal positive self-time in window = {tot/1000:.1f}ms")
print(f"\n=== top 30 ops by SELF time (ms) ===")
for n, v in rows[:30]:
    if v <= 0: continue
    short = n
    print(f"  {v/1000:8.1f}ms  ({100*v/tot:4.1f}%)  x{cnt[n]:<5d} {short[:88]}")
