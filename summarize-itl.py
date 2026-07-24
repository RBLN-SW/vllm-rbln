#!/usr/bin/env python3
"""itl-sweep-results/ 아래 조합별 ITL breakdown JSON 을 집계.

각 조합 디렉터리(mns=..._spec=..._sampler=...)의 DP rank 별 JSON 을 읽어
rank 평균 phase breakdown 을 만들고, 조합 간 비교 표 + 통합 JSON 을 출력한다.

사용:  ./summarize-itl.py [RESULTS_DIR]   (기본 itl-sweep-results)
출력:  stdout 표 + RESULTS_DIR/summary.json
"""
import glob
import json
import os
import re
import sys

PHASES = [
    "prepare_input",
    "model_forward",
    "postprocess",
    "sampler",
    "draft",
    "update_state",
    "others",
]
KINDS = ["prefill", "decode_all", "decode_mixed"]
DIR_RE = re.compile(r"mns=(\d+)_spec=(on|off)_sampler=(\d)")


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def load_config(cfg_dir):
    """rank JSON 들을 평균내어 kind -> {phases, avg_itl_ms, count, tokens} 반환."""
    files = sorted(glob.glob(os.path.join(cfg_dir, "max-num-seqs=*.json")))
    ranks = []
    for f in files:
        try:
            ranks.append(json.load(open(f)))
        except Exception as e:
            print(f"[warn] {f} 로드 실패: {e}", file=sys.stderr)
    if not ranks:
        return None, 0

    out = {}
    for kind in KINDS:
        # rank 별 이 kind 데이터
        per_rank = [r["steps"].get(kind, {}) for r in ranks]
        counts = [k.get("count", 0) for k in per_rank]
        # count>0 인 rank 만 평균에 사용
        valid = [k for k in per_rank if k.get("count", 0) > 0]
        if not valid:
            out[kind] = {"count": 0}
            continue
        phase_avg = {
            p: avg([k["phases"][p]["avg_ms"] for k in valid if p in k.get("phases", {})])
            for p in PHASES
        }
        itl = sum(phase_avg.values())
        # rank 별 토큰(있으면): 각 rank 의 avg_total 평균
        tok_totals = [
            k["tokens"]["avg_total"]
            for k in valid
            if k.get("tokens") and "avg_total" in k["tokens"]
        ]
        out[kind] = {
            "count": int(avg(counts)),
            "avg_itl_ms": itl,
            "phases_ms": phase_avg,
            "avg_tokens_total": avg(tok_totals) if tok_totals else None,
        }
    return out, len(ranks)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "itl-sweep-results"
    cfg_dirs = sorted(glob.glob(os.path.join(results_dir, "mns=*_spec=*_sampler=*")))
    if not cfg_dirs:
        print(f"결과 디렉터리 없음: {results_dir}/mns=*", file=sys.stderr)
        sys.exit(1)

    summary = {}
    rows = []
    for d in cfg_dirs:
        m = DIR_RE.search(os.path.basename(d))
        if not m:
            continue
        mns, spec, sampler = int(m.group(1)), m.group(2), m.group(3)
        data, nranks = load_config(d)
        key = os.path.basename(d)
        if data is None:
            summary[key] = {"status": "NO_DATA", "ranks": 0}
            rows.append((mns, spec, sampler, nranks, None))
            continue
        summary[key] = {
            "mns": mns, "spec": spec, "sampler": sampler,
            "ranks": nranks, "kinds": data,
        }
        rows.append((mns, spec, sampler, nranks, data))

    # ── 표 1: decode_all(steady-state) 비교 ──────────────────────────────
    print("=" * 118)
    print("ITL BREAKDOWN — DECODE_ALL (모든 rank decode; steady-state) | 단위 ms, () 안은 %")
    print("=" * 118)
    hdr = f"{'mns':>4} {'spec':>4} {'smp':>3} {'rk':>2} {'steps':>6} {'ITL':>8} | " + \
        " ".join(f"{p[:9]:>9}" for p in PHASES)
    print(hdr)
    print("-" * 118)
    for mns, spec, sampler, nranks, data in sorted(rows):
        if data is None or data.get("decode_all", {}).get("count", 0) == 0:
            print(f"{mns:>4} {spec:>4} {sampler:>3} {nranks:>2} {'--':>6}  (데이터 없음)")
            continue
        k = data["decode_all"]
        itl = k["avg_itl_ms"]
        cells = []
        for p in PHASES:
            ms = k["phases_ms"][p]
            pct = ms / itl * 100 if itl else 0
            cells.append(f"{ms:5.2f}({pct:2.0f})")
        print(f"{mns:>4} {spec:>4} {sampler:>3} {nranks:>2} {k['count']:>6} "
              f"{itl:>8.2f} | " + " ".join(f"{c:>9}" for c in cells))

    # ── 표 2: 조합별 kind 요약(ITL/steps) ────────────────────────────────
    print()
    print("=" * 78)
    print("kind 별 평균 ITL(ms) / steps")
    print("=" * 78)
    print(f"{'mns':>4} {'spec':>4} {'smp':>3} | "
          f"{'prefill':>16} {'decode_all':>16} {'decode_mixed':>16}")
    print("-" * 78)
    for mns, spec, sampler, nranks, data in sorted(rows):
        def cell(kind):
            if not data:
                return "--"
            k = data.get(kind, {})
            if k.get("count", 0) == 0:
                return "--"
            return f"{k['avg_itl_ms']:.1f}/{k['count']}"
        print(f"{mns:>4} {spec:>4} {sampler:>3} | "
              f"{cell('prefill'):>16} {cell('decode_all'):>16} "
              f"{cell('decode_mixed'):>16}")

    out_path = os.path.join(results_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n통합 요약 JSON → {out_path}")
    # 누락 조합 경고
    missing = [k for k, v in summary.items() if v.get("status") == "NO_DATA"]
    if missing:
        print(f"[!] 데이터 없는 조합: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
