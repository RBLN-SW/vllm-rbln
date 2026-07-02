import re, json, sys

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/L18_rf.log"
spans = []  # (pid, kind, t0, t1)
for line in open(src):
    m = re.search(r"SPAN (fwd|allreduce) (\d+) ([0-9.]+) ([0-9.]+)", line)
    if m:
        spans.append((m.group(2), m.group(1), float(m.group(3)), float(m.group(4))))
if not spans:
    print("no spans"); raise SystemExit

# common time origin, seconds -> microseconds
t0 = min(s[2] for s in spans)
pids = sorted(set(s[0] for s in spans))
rank = {p: i for i, p in enumerate(pids)}
TID = {"fwd": 1, "allreduce": 2}
TNAME = {"fwd": "forward (device thread)", "allreduce": "DP all_reduce (main)"}

ev = []
for p in pids:
    ev.append({"ph": "M", "name": "process_name", "pid": rank[p],
               "args": {"name": f"DP rank {rank[p]} (pid {p})"}})
    for k, t in TID.items():
        ev.append({"ph": "M", "name": "thread_name", "pid": rank[p], "tid": t,
                   "args": {"name": TNAME[k]}})
for pid, kind, a, b in spans:
    ev.append({"name": ("forward" if kind == "fwd" else "all_reduce"),
               "ph": "X", "ts": (a - t0) * 1e6, "dur": (b - a) * 1e6,
               "pid": rank[pid], "tid": TID[kind]})

out = "/tmp/c9_overlap_perfetto.json"
json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, open(out, "w"))
nf = sum(1 for s in spans if s[1] == "fwd")
na = sum(1 for s in spans if s[1] == "allreduce")
print(f"wrote {out}  ({nf} forward + {na} all_reduce spans, {len(pids)} ranks)")
print("open in https://ui.perfetto.dev  (drag-drop the file)")
