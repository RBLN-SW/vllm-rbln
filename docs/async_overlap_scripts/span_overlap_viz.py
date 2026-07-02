#!/usr/bin/env python3
"""SPAN-based overlap visualization: forward(N) vs prefetched all_reduce(N+1).

Uses perf_counter SPANs (process-wide clock). The prefetched collective's
in-flight window is [ar_issue_start, ar_issue_start + that step's real
all_reduce duration] -- the real all_reduce dur is a faithful proxy for how
long the gloo network takes. Overlaid on the forward span it shows the
collective running concurrently with the forward.
"""
import re, sys, json, statistics as st
log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_run.log"
pick = sys.argv[2] if len(sys.argv) > 2 else None
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/span_overlap.json"
pat = re.compile(r"SPAN (allreduce|ar_issue|ar_collect|fwd) (\d+) ([0-9.]+) ([0-9.]+)")
from collections import defaultdict
by = defaultdict(lambda: defaultdict(list))
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if m:
        k,p,a,b = m.group(1),m.group(2),float(m.group(3)),float(m.group(4))
        by[p][k].append((a,b))
pids = sorted(by)
if pick is None:  # pick pid with the largest median real all_reduce (most visible)
    pick = max(pids, key=lambda p: st.median([b-a for a,b in by[p]["allreduce"][2:]]) if len(by[p]["allreduce"])>2 else 0)
d = by[pick]
fwd = d["fwd"]; issue = d["ar_issue"]; ar = d["allreduce"]
# steady-state network width proxy: median real all_reduce EXCLUDING startup barrier (>100ms)
steady = [b-a for a,b in ar if (b-a) < 0.1]
ar_med = st.median(steady[2:]) if len(steady) > 2 else 0.001
# only consider prefetch issues within the forward time window (drops warmup calls)
fmin = min(a for a,_ in fwd); fmax = max(b for _,b in fwd)
pref = [(a, a+ar_med) for a,b in issue if fmin <= a <= fmax]
# overlap of prefetch-inflight with forward spans
def isect(A,B):
    A=sorted(A);B=sorted(B);i=j=0;t=0.0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],B[j][0]);hi=min(A[i][1],B[j][1])
        if hi>lo:t+=hi-lo
        if A[i][1]<B[j][1]:i+=1
        else:j+=1
    return t
tot_pref=sum(b-a for a,b in pref)
ov=isect(pref,fwd)
print(f"pid={pick}  fwd spans={len(fwd)} prefetch spans={len(pref)}")
print(f"prefetch all_reduce in-flight total={tot_pref*1000:.1f}ms  overlapped by forward={ov*1000:.1f}ms "
      f"({100*ov/max(tot_pref,1e-9):.1f}%)")
# perfetto
t0=min([a for a,_ in fwd]+[a for a,_ in pref])
pe=[{"ph":"M","name":"process_name","pid":0,"args":{"name":f"DP rank pid {pick} (OVERLAP_PROBE)"}},
    {"ph":"M","name":"thread_name","pid":0,"tid":1,"args":{"name":"forward(N) (main thread)"}},
    {"ph":"M","name":"thread_name","pid":0,"tid":2,"args":{"name":"all_reduce(N+1) prefetch (gloo, in-flight)"}}]
for a,b in fwd: pe.append({"name":"forward","ph":"X","ts":(a-t0)*1e6,"dur":(b-a)*1e6,"pid":0,"tid":1})
for a,b in pref: pe.append({"name":"all_reduce(N+1)","ph":"X","ts":(a-t0)*1e6,"dur":(b-a)*1e6,"pid":0,"tid":2})
json.dump({"traceEvents":pe,"displayTimeUnit":"ms"},open(out,"w"))
print(f"wrote {out} ({len(pe)} events) -> ui.perfetto.dev")
