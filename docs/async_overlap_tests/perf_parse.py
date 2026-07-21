#!/usr/bin/env python3
"""Parse PERF_GEN lines into an honest throughput table.

Metric: total_output_tokens / generate wall-clock, WARM runs only (call>=2;
call=1 is the cold/compile run and is dropped). DP ranks run in lockstep, so a
"wave" = one generate call across all rank processes: global_tok_s(wave) =
sum_ranks(out_tokens) / max_ranks(wall_s). Report the median over warm waves.
"""
import os
import re
import sys
import glob
import statistics as st

LOGDIR = sys.argv[1]
DP = int(sys.argv[2]) if len(sys.argv) > 2 else 4

PAT = re.compile(
    r"PERF_GEN pid=(\d+) call=(\d+) wall_s=([\d.]+) "
    r"n_prompts=(\d+) out_tokens=(\d+) tok_s=([\d.]+)"
)


def parse_log(path):
    # rows: list of (pid, call, wall, nprompts, ntok, toks)
    rows = []
    with open(path, errors="ignore") as f:
        for ln in f:
            m = PAT.search(ln)
            if m:
                pid, call, wall, npr, ntok, toks = m.groups()
                rows.append((int(pid), int(call), float(wall), int(npr), int(ntok), float(toks)))
    return rows


def analyze(rows):
    # warm only
    warm = [r for r in rows if r[1] >= 2]
    if not warm:
        return None
    pids = sorted({r[0] for r in warm})
    calls = sorted({r[1] for r in warm})
    wave_tok_s = []
    wave_detail = []
    for c in calls:
        wave = [r for r in warm if r[1] == c]
        if len(wave) < len(pids):  # incomplete wave
            continue
        total_tok = sum(r[4] for r in wave)
        wall = max(r[2] for r in wave)
        if wall <= 0:
            continue
        wave_tok_s.append(total_tok / wall)
        wave_detail.append((c, total_tok, wall))
    if not wave_tok_s:
        return None
    return {
        "n_ranks": len(pids),
        "n_warm_waves": len(wave_tok_s),
        "global_tok_s": st.median(wave_tok_s),
        "tok_s_all": [round(x, 1) for x in wave_tok_s],
        "sample_total_tok": wave_detail[0][1],
        "sample_wall_s": round(st.median([w for _, _, w in wave_detail]), 3),
    }


def cfg_batch(fname):
    m = re.match(r"(async|sync)_b(\d+)\.log", os.path.basename(fname))
    return (m.group(1), int(m.group(2))) if m else (os.path.basename(fname), -1)


results = {}
for path in glob.glob(os.path.join(LOGDIR, "*.log")):
    cfg, b = cfg_batch(path)
    rows = parse_log(path)
    a = analyze(rows)
    results[(cfg, b)] = a

batches = sorted({b for (_, b) in results if b >= 0})
print(f"{'batch':>6} | {'async tok/s':>12} | {'sync tok/s':>11} | {'async/sync':>10} | detail")
print("-" * 80)
for b in batches:
    a = results.get(("async", b))
    s = results.get(("sync", b))
    at = a["global_tok_s"] if a else None
    stk = s["global_tok_s"] if s else None
    ratio = (at / stk) if (at and stk) else None
    astr = f"{at:12.1f}" if at else f"{'NA':>12}"
    sstr = f"{stk:11.1f}" if stk else f"{'NA':>11}"
    rstr = f"{ratio:9.2f}x" if ratio else f"{'NA':>10}"
    det = ""
    if a:
        det += f"async[ranks={a['n_ranks']},waves={a['n_warm_waves']},tok={a['sample_total_tok']},wall={a['sample_wall_s']}s,samples={a['tok_s_all']}] "
    if s:
        det += f"sync[wall={s['sample_wall_s']}s,samples={s['tok_s_all']}]"
    print(f"{b:>6} | {astr} | {sstr} | {rstr} | {det}")
