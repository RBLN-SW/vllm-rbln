#!/usr/bin/env python3
"""Precise forward<->all_reduce overlap analysis, per-thread aware."""
import gzip, json, sys
from collections import Counter, defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]

tname = {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tname[e.get("tid")] = e["args"].get("name", "")

X = [e for e in ev if e.get("ph") == "X" and "ts" in e and "dur" in e]

# Which thread issues execute_model / sample_tokens? -> the "main" compute thread.
def has(e, s): return s in e.get("name", "")
main_tid = None
for e in X:
    if "execute_model" in e.get("name", "") or "sample_tokens" in e.get("name", ""):
        main_tid = e["tid"]; break
print(f"main compute tid = {main_tid} ({tname.get(main_tid)})")

# gloo runloop tid
gloo_tid = next((t for t, n in tname.items() if "gloo" in n.lower()), None)
print(f"gloo runloop tid = {gloo_tid} ({tname.get(gloo_tid)})")

# On the MAIN thread, find the forward op and the all_reduce op names.
main = [e for e in X if e["tid"] == main_tid]
print(f"\nmain-thread X events: {len(main)}")

# Candidate forward op: the RBLN compiled-graph exec. Look for top-level names.
cand_fwd = Counter()
cand_ar = Counter()
for e in main:
    n = e["name"]
    if any(k in n for k in ["model_executable", "CompiledFunction", "dynamo", "rbln", "OptimizedModule", "call_impl", "forward"]):
        cand_fwd[n] += 1
    if any(k in n for k in ["all_reduce", "allreduce", "num_tokens_across_dp", "forward_context"]):
        cand_ar[n] += 1
print("\n=== main-thread forward-ish op names (top 12) ===")
for n, c in cand_fwd.most_common(12): print(f"  x{c:<4d} {n[:85]}")
print("\n=== main-thread all_reduce-ish op names (top 12) ===")
for n, c in cand_ar.most_common(12): print(f"  x{c:<4d} {n[:85]}")

# Pick concrete intervals:
def intervals(es, key):
    return sorted((e["ts"], e["ts"] + e["dur"]) for e in es if key in e["name"])

# forward = the outermost per-step compiled forward on main thread.
# all_reduce = the c10d all_reduce python frame on main thread (the blocking call).
FWD_KEY = "model_executable" if any("model_executable" in e["name"] for e in main) else None
AR_KEY = "distributed_c10d.py(2978): all_reduce"
fwd_iv = intervals(main, FWD_KEY) if FWD_KEY else []
ar_iv_main = [ (e["ts"], e["ts"]+e["dur"]) for e in main if "all_reduce" in e["name"] and "distributed" in e["name"] ]
ar_iv_main.sort()
print(f"\nFWD_KEY={FWD_KEY!r}  main fwd intervals={len(fwd_iv)}  main all_reduce intervals={len(ar_iv_main)}")

def total(iv): return sum(b-a for a,b in iv)
def isect(A, B):
    A=sorted(A); B=sorted(B); i=j=0; t=0.0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]); hi=min(A[i][1],B[j][1])
        if hi>lo: t+=hi-lo
        if A[i][1]<B[j][1]: i+=1
        else: j+=1
    return t

if fwd_iv and ar_iv_main:
    ov = isect(fwd_iv, ar_iv_main)
    print(f"[main fwd vs main all_reduce] fwd_tot={total(fwd_iv)/1000:.1f}ms "
          f"ar_tot={total(ar_iv_main)/1000:.1f}ms overlap={ov/1000:.2f}ms "
          f"({100*ov/max(total(ar_iv_main),1e-9):.1f}% of ar)")

# gloo runloop all_reduce vs main-thread forward
gloo = [e for e in X if e["tid"] == gloo_tid] if gloo_tid else []
gloo_ar = sorted((e["ts"], e["ts"]+e["dur"]) for e in gloo if "all_reduce" in e["name"] or "allreduce" in e["name"])
print(f"\ngloo-runloop all_reduce intervals={len(gloo_ar)} tot={total(gloo_ar)/1000:.1f}ms")
if fwd_iv and gloo_ar:
    ov = isect(fwd_iv, gloo_ar)
    print(f"[main fwd vs GLOO-runloop all_reduce] overlap={ov/1000:.2f}ms "
          f"({100*ov/max(total(gloo_ar),1e-9):.1f}% of gloo_ar)")

# Print a sample sequence: main-thread execute_model / model_executable / all_reduce / sample events in time order (window)
seq = []
for e in main:
    n = e["name"]
    if any(k in n for k in ["execute_model", "model_executable", "distributed_c10d.py(2978): all_reduce", "sample_tokens", "num_tokens_across_dp"]):
        seq.append((e["ts"], e["dur"], n))
seq.sort()
print(f"\n=== main-thread step sequence (first 24 of {len(seq)}) [ts_ms, dur_ms, name] ===")
t0 = seq[0][0] if seq else 0
for ts, dur, n in seq[:24]:
    short = n.split("): ")[-1] if "): " in n else n
    print(f"  t={ (ts-t0)/1000:8.2f}ms  dur={dur/1000:7.2f}ms  {short[:60]}")
