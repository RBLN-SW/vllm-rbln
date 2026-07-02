#!/usr/bin/env python3
"""Per-step steady-state timing: all_reduce vs step total -> realistic overlap ceiling."""
import gzip, json, sys, statistics as st
path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
X = [e for e in data["traceEvents"] if e.get("ph")=="X" and "ts" in e and "dur" in e]
mt = next(e["tid"] for e in X if "execute_model" in e.get("name",""))
M = [e for e in X if e["tid"]==mt]

def outer(substr):
    es = sorted(((e["ts"], e["dur"]) for e in M if substr in e["name"]), key=lambda x:(x[0],-x[1]))
    out=[]; last=-1
    for ts,d in es:
        if ts>=last: out.append((ts,d)); last=ts+d
    return out

em = outer("execute_model")           # per step (contains all_reduce + forward)
ar = outer("distributed_c10d.py(2978): all_reduce")
sm = outer("sample_tokens")
wait = sorted((e["ts"], e["dur"]) for e in M if "method wait" in e["name"])

# steady state = skip step 0
def stats(iv, skip=1):
    d=[x[1]/1000 for x in iv[skip:]]
    return len(d), st.median(d), st.mean(d), max(d)
for nm, iv in [("execute_model",em),("all_reduce",ar),("sample_tokens",sm)]:
    n,md,mn,mx = stats(iv)
    print(f"{nm:16s} n={n:3d} median={md:7.3f}ms mean={mn:7.3f}ms max={mx:8.2f}ms")

# step total = execute_model + sample_tokens (they're serial, back to back)
steps = min(len(em), len(sm))
# pair each execute_model with following sample; approximate step = em.dur + sm.dur
em_d = [d for _,d in em[1:]]; sm_d = [d for _,d in sm[1:]]; ar_d=[d for _,d in ar[1:]]
k = min(len(em_d), len(sm_d), len(ar_d))
step_tot = [ (em_d[i]+sm_d[i])/1000 for i in range(k) ]
ar_ms   = [ ar_d[i]/1000 for i in range(k) ]
print(f"\nsteady-state per-step (n={k}):")
print(f"  step total (em+sample): median={st.median(step_tot):.3f}ms")
print(f"  all_reduce            : median={st.median(ar_ms):.3f}ms")
print(f"  all_reduce / step     : median={100*st.median(ar_ms)/st.median(step_tot):.1f}%  "
      f"(=> max step-latency win if all_reduce fully hidden)")
tot_ar = sum(ar_ms); tot_step = sum(step_tot)
print(f"  totals: all_reduce={tot_ar:.1f}ms  step={tot_step:.1f}ms  ratio={100*tot_ar/tot_step:.1f}%")

# wait distribution (barrier)
wd = sorted(d/1000 for _,d in wait)
big = [x for x in wd if x>100]
small = [x for x in wd if x<=100]
print(f"\nwait events: total={len(wd)}  >100ms(startup barrier)={len(big)} sum={sum(big):.0f}ms  "
      f"<=100ms(steady) n={len(small)} median={st.median(small) if small else 0:.3f}ms sum={sum(small):.1f}ms")
