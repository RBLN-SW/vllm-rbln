#!/usr/bin/env python3
"""Distill main-thread execute_model / all_reduce / sample_tokens into a small
Perfetto trace + print steady-state timing and forward<->all_reduce overlap."""
import gzip, json, sys, statistics as st

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rbln_overlap_distilled.json"
with gzip.open(src, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]
tname = {e.get("tid"): e["args"].get("name","") for e in ev
         if e.get("ph")=="M" and e.get("name")=="thread_name"}
X = [e for e in ev if e.get("ph")=="X" and "ts" in e and "dur" in e]
main_tid = next(e["tid"] for e in X if "execute_model" in e.get("name",""))

def top_intervals(key):
    """Outermost (longest, dedup by ts) intervals whose name contains key, on main thread."""
    es = [(e["ts"], e["dur"]) for e in X if e["tid"]==main_tid and key in e["name"]]
    es.sort(key=lambda x:(x[0], -x[1]))
    out=[]; last_end=-1
    for ts,dur in es:
        if ts >= last_end:  # outermost only
            out.append((ts,dur)); last_end=ts+dur
    return out

em = top_intervals("execute_model")
ar = top_intervals("distributed_c10d.py(2978): all_reduce")
sm = top_intervals("sample_tokens")
print(f"main_tid={main_tid}  execute_model={len(em)} all_reduce={len(ar)} sample_tokens={len(sm)}")

# steady-state = drop step 0 (startup barrier). Report medians.
def med(iv, skip=1):
    d=[x[1] for x in iv[skip:]]
    return (st.median(d)/1000 if d else 0, min(d)/1000 if d else 0, max(d)/1000 if d else 0)
print(f"\nsteady-state (skip step0) durations [median/min/max ms]:")
print(f"  execute_model: {med(em)}")
print(f"  all_reduce   : {med(ar)}")
print(f"  sample_tokens: {med(sm)}")

# forward portion = [all_reduce_end, sample_start] within each step -> proxy for forward+post host time
# pair each all_reduce with the next sample_tokens
def overlap(A,B):
    A=sorted(A); B=sorted(B); i=j=0; t=0.0
    while i<len(A) and j<len(B):
        lo=max(A[i][0],A and B[j][0]); hi=min(A[i][0]+A[i][1], B[j][0]+B[j][1])
        if hi>lo: t+=hi-lo
        if A[i][0]+A[i][1] < B[j][0]+B[j][1]: i+=1
        else: j+=1
    return t
ov = overlap(ar, sm)  # all_reduce vs sample_tokens (should be 0 -> serial)
print(f"\noverlap(all_reduce, sample_tokens) = {ov/1000:.2f}ms  (0 => strictly serial)")

# emit distilled perfetto: 3 tracks on one 'rank' process
TID={"execute_model":1,"all_reduce":2,"sample_tokens":3}
TN={"execute_model":"execute_model (host)","all_reduce":"DP all_reduce (gloo, host)","sample_tokens":"sample_tokens (host)"}
t0=em[0][0]
pe=[{"ph":"M","name":"process_name","pid":0,"args":{"name":f"DP rank (tid {main_tid})"}}]
for k,t in TID.items():
    pe.append({"ph":"M","name":"thread_name","pid":0,"tid":t,"args":{"name":TN[k]}})
for k,iv in (("execute_model",em),("all_reduce",ar),("sample_tokens",sm)):
    for ts,dur in iv:
        pe.append({"name":k,"ph":"X","ts":(ts-t0),"dur":dur,"pid":0,"tid":TID[k]})
json.dump({"traceEvents":pe,"displayTimeUnit":"ms"}, open(out,"w"))
print(f"\nwrote distilled perfetto: {out} ({len(pe)} events) -> ui.perfetto.dev")
