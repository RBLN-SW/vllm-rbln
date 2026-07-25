#!/usr/bin/env python3
"""model_forward vs context length (<=4k), 1x4 subplot (batch size별, y축 공유).
조건별 median 라인 + IQR(25-75%) band. decode_all per-step 데이터 사용.
출력: PNG.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "itl-ctx-results"
OUT = sys.argv[1] if len(sys.argv) > 1 else "ctx_model_forward.png"
XMAX = 4000
STEP = 200
MINCOUNT = 15

CONDS = [
    ("specoff", "off", "spec off", "#2a78d6"),
    ("ns1", "on", "num_spec=1", "#eb6834"),
    ("ns3", "on", "num_spec=3", "#1baf7a"),
]
MNS = [1, 4, 8, 16]  # max-num-seqs=2 excluded from the figure
# static-perf reference (model_forward ms) per --max-num-seqs
STATIC = {1: 30.1, 4: 38.9, 8: 45.5, 16: 54.5}
STATIC_COLOR = "#d81b1b"


def load(cond, spec, m):
    d = f"{RESULTS}/{cond}/mns={m}_spec={spec}_sampler=0"
    ctx, mf, toks = [], [], []
    for f in glob.glob(f"{d}/max-num-seqs=*.json"):
        if os.path.getsize(f) < 100:
            continue
        da = json.load(open(f))["steps"].get("decode_all", {})
        if da.get("count", 0) == 0:
            continue
        cl = (da.get("context_length") or {}).get("per_step_mean")
        m2 = da["phases"]["model_forward"].get("samples_ms")
        if cl and m2:
            n = min(len(cl), len(m2))
            ctx += cl[:n]
            mf += m2[:n]
        if da.get("tokens"):
            toks.append(da["tokens"]["avg_total"])
    return np.array(ctx), np.array(mf), (sum(toks) / len(toks) if toks else 0)


def binned(ctx, mf):
    mask = ctx <= XMAX
    ctx, mf = ctx[mask], mf[mask]
    if len(ctx) == 0:
        return None
    edges = np.arange(1200, XMAX + STEP, STEP)  # shared bins → ratios align
    xs, med, q25, q75 = [], [], [], []
    for i in range(len(edges) - 1):
        m = (ctx >= edges[i]) & (ctx < edges[i + 1])
        if m.sum() >= MINCOUNT:
            xs.append((edges[i] + edges[i + 1]) / 2)
            v = mf[m]
            med.append(np.median(v))
            q25.append(np.percentile(v, 25))
            q75.append(np.percentile(v, 75))
    if not xs:
        return None
    return np.array(xs), np.array(med), np.array(q25), np.array(q75)


plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#8a8a86",
    "figure.dpi": 150,
})

RATIO = [("ns1", "#eb6834", "num_spec=1 ÷ off"), ("ns3", "#1baf7a", "num_spec=3 ÷ off")]

# row1 = model_forward (ms), row2 = latency ratio vs spec off. cols = max-num-seqs.
fig, axes = plt.subplots(2, len(MNS), figsize=(3.35 * len(MNS) + 1.3, 7.4),
                         sharex="col", sharey="row")
handles = {}
ratio_handles = {}
y0min, y0max = float("inf"), float("-inf")
r1min, r1max = float("inf"), float("-inf")
for j, m in enumerate(MNS):
    axt, axb = axes[0][j], axes[1][j]
    subtitle = []
    med_by = {}
    for cond, spec, lab, col in CONDS:
        ctx, mf, tok = load(cond, spec, m)
        b = binned(ctx, mf)
        subtitle.append(f"{lab.split('=')[-1] if '=' in lab else 'off'}:{tok:.0f}")
        if b is None:
            continue
        xs, med, q25, q75 = b
        axt.fill_between(xs, q25, q75, color=col, alpha=0.14, linewidth=0)
        (h,) = axt.plot(xs, med, color=col, lw=2.4, marker="o", ms=3.5)
        handles[lab] = h
        med_by[cond] = dict(zip(xs, med))
        y0min, y0max = min(y0min, q25.min()), max(y0max, q75.max())
    # static-perf reference (top row only)
    if m in STATIC:
        y = STATIC[m]
        axt.axhline(y, color=STATIC_COLOR, ls="--", lw=1.6, zorder=6)
        axt.text(XMAX, y, f" static perf {y}", color=STATIC_COLOR, va="bottom",
                 ha="right", fontsize=8.5, fontweight="bold")
        y0min = min(y0min, y)
    # ratio (bottom row): num_spec median / spec-off median, matched context bin
    axb.axhline(1.0, color="#9a9a96", lw=1.0, ls=":", zorder=1)
    off = med_by.get("specoff", {})
    for cond, col, rlab in RATIO:
        d = med_by.get(cond, {})
        cxs = sorted(set(d) & set(off))
        if not cxs:
            continue
        r = [d[c] / off[c] for c in cxs]
        (rh,) = axb.plot(cxs, r, color=col, lw=2.3, marker="^", ms=3.8)
        ratio_handles[rlab] = rh
        r1min, r1max = min(r1min, min(r)), max(r1max, max(r))
    axt.set_title(f"--max-num-seqs = {m}", fontsize=12.5, fontweight="bold", pad=8)
    axb.set_xlabel("context length (tokens)", fontsize=10.5)
    axb.set_xlim(1200, XMAX)
    axt.tick_params(labelsize=9.5, labelbottom=True)  # keep x ticks on row1 too
    axb.tick_params(labelsize=9.5)
    axb.text(0.5, -0.30, "avg tok/step  " + " / ".join(subtitle),
             transform=axb.transAxes, ha="center", va="top", fontsize=8.5, color="#6a6a66")

p0 = (y0max - y0min) * 0.06
axes[0][0].set_ylim(y0min - p0, y0max + p0)
axes[1][0].set_ylim(min(0.985, r1min - 0.02), r1max + (r1max - 1.0) * 0.14)
axes[0][0].set_ylabel("model_forward per step  (ms)", fontsize=11)
axes[1][0].set_ylabel("latency ratio  vs  spec off  (×)", fontsize=11)

from matplotlib.lines import Line2D
# row-1 legend (top)
leg_handles = list(handles.values()) + [Line2D([0], [0], color=STATIC_COLOR, ls="--", lw=1.6)]
leg_labels = list(handles.keys()) + ["static perf"]
fig.legend(leg_handles, leg_labels, loc="upper center", ncol=4, frameon=False,
           fontsize=11, bbox_to_anchor=(0.5, 0.995))
# row-2 legend, sitting in the gap between the two rows
fig.legend(list(ratio_handles.values()), list(ratio_handles.keys()),
           loc="center", ncol=2, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 0.487))
fig.suptitle(
    "MiniMax-M2.5 decode  model_forward  vs  context length  (DP4+EP)",
    fontsize=13.5, fontweight="bold", y=1.045)
fig.text(0.5, 0.005,
         "top: median model_forward, shaded = IQR (25–75%)   ·   bottom: latency ratio = num_spec median ÷ spec-off median at matched context   ·   "
         "context ≤ 4k, decode_all pooled over 4 DP ranks   ·   tok/step = avg effective batch (larger max-num-seqs under-fills under the SWE-bench agent load)",
         ha="center", fontsize=8.5, color="#6a6a66")
fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.10, hspace=0.36, wspace=0.09)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
