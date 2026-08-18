# Page layout vs sub-block prefix caching

Scripts for the one comparison that decides whether
[page layout](https://github.com/RBLN-SW/vllm-rbln/issues/928) is worth keeping: does it beat
[sub-block prefix caching](../../docs/sub_block_prefix_caching.md) on multi-turn
traffic, at the same kernel block and the same prefill chunk?

Multi-turn is the deciding workload because it is the only place the two designs
differ structurally. Both match at page granularity, and both copy a partially
matched block into a private one — page layout adds exactly one case: when the
producer's kernel block is unowned and wholly covered by the match, the next turn
continues writing into it and the copy disappears (rule R3 in the design doc).
Everything else is at parity by construction, so a benchmark of independent
requests cannot separate them. `vllm bench serve` in particular sends
independent requests, so no turn ever resumes the previous turn's tokens — the
entire effect under test is absent.

## Layout

| file | what it is |
|---|---|
| `serve.sh` | launches one of the two modes; everything else held equal |
| `repeat.sh` | N replays of one fixed workload against a reset cache -- the protocol below |
| `analyze.py` | pairs the two modes by (workload, seed), reports deltas and a Welch t |
| `probe_greedy.py` | fixed prompts at temperature 0, for a correctness diff |
| `diff_probe.py` | byte-diffs two probe outputs |
| `workloads/*.json` | conversation generators for the multi-turn driver |

## Prerequisites

The multi-turn driver lives in the vLLM **source tree**, not in the wheel, so it
has to be cloned. Pin it to the tag matching the installed wheel (`vllm==0.24.0`,
see `pyproject.toml`) -- the driver is an HTTP client and mostly tolerates skew,
but a pinned checkout is the difference between a rerun that reproduces and one
that merely resembles:

```bash
cd benchmarks/page_layout
git clone --depth 1 --branch v0.24.0 https://github.com/vllm-project/vllm.git
uv pip install --python ../../.venv/bin/python \
    -r vllm/benchmarks/multi_turn/requirements.txt
```

The scripts look for `./vllm` next to themselves, so nothing has to be exported.
Set `VLLM_REPO` only to point somewhere else. If the checkout already exists,
check what it is on before trusting a measurement: `git -C vllm describe --tags`.

The generators draw their filler text from a Project Gutenberg book, resolved
relative to the results directory:

```bash
export OUT_DIR=$PWD/results
mkdir -p "$OUT_DIR" && curl -sL https://www.gutenberg.org/files/1184/1184-0.txt \
    -o "$OUT_DIR/pg1184.txt"
```

Pick the NPUs. `serve.sh` has no default and refuses to launch without this,
because a launch that silently picks its own devices is how a run ends up on
hardware another job owns, or on hardware the previous serve has not yet
released:

```bash
export RBLN_DEVICES=4,5,6,7
```

Everything else the serve needs is fixed in `serve.sh`: MiniMax-M2.5, port 8102,
`--data-parallel-size 4`, kernel block 8192, page 512, prefill chunk 512. The
comparison only means something when both modes see the same geometry, so those
are not knobs.

## Running

Both modes need the same devices, so they run one after the other. Do page layout
first, then sub-block, then compare.

### Page layout

```bash
./serve.sh pagelayout | tee "$OUT_DIR/serve_pagelayout.log"   # terminal 1

# terminal 2, once the log shows "Page/kernel block KV cache" and the port answers
./repeat.sh pagelayout 5 long 1
```

`repeat.sh` prints one line per repetition and appends them to
`$OUT_DIR/summary_pagelayout.txt`. The first three are warm-up and are marked as
such;
the rest are the measurement.

### Sub-block

Stop the first serve and wait for the devices to report **zero** used memory and
the port to be free before launching the second — see the pitfalls below, this is
not optional.

```bash
./serve.sh subblock | tee "$OUT_DIR/serve_subblock.log"       # terminal 1
./repeat.sh subblock 5 long 1                                 # terminal 2
```

The banner to wait for in this mode is `Sub-block prefix caching enabled`.

### Comparing

Take the mean of the kept lines in each `summary_*.txt`. A difference counts only
if it exceeds the instrument's resolution — about 1 point, so roughly 2 points at
five repetitions — and its confidence interval excludes zero. Decide that
threshold before running, not after seeing the numbers.

### Sweeping several workloads and seeds

The comparison above fixes one workload because that is what resolves a small
difference. To sweep instead — a coarser picture across `short` and `long` at
several seeds, paired by `analyze.py` — loop the driver directly:

```bash
MODE=pagelayout            # or subblock; must match the serve that is running
cd "$OUT_DIR"
for seed in 900 1 2 3; do                  # 900 is a discarded warm-up
  for wl in short long; do
    python vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py \
        --model MiniMaxAI/MiniMax-M2.5 --served-model-name MiniMax \
        --url http://localhost:8102 \
        --input-file ../workloads/multi_turn_$wl.json \
        --num-clients 4 --max-active-conversations 8 \
        --seed $seed --request-timeout-sec 1800 \
        --stats-json-output "st_${MODE}_${wl}_${seed}.json" \
        > "run_${MODE}_${wl}_${seed}.log" 2>&1
    echo "$MODE $wl seed=$seed $(grep -oE 'benchmark runtime: [0-9.]+ sec' \
        "run_${MODE}_${wl}_${seed}.log" | tail -1)"
  done
done
```

Run it once per mode, then `python analyze.py "$OUT_DIR"`. Note that a sweep
buys cache freshness with a fresh seed per run rather than by resetting, so its
numbers carry the between-seed spread — read it for shape, not for a few points.

## Reading the results, and four ways to be fooled

Every item here cost real time on this comparison.

**1. `approx_cached_percent` is not a hit rate.** The driver computes it
client-side as `history_tokens / input_tokens` — a property of the generated
dataset, identical for both modes by construction. It shows the two modes saw the
same *work*; it says nothing about whether the server hit its cache. For that,
read the server:

```bash
curl -s localhost:8102/metrics | grep -E 'vllm:prefix_cache_(hits|queries)_total'
```

Sum hits over queries across DP ranks. On the long workload sub-block caching
reaches ~74%; a page-layout number far below that means cache is being destroyed,
not that the workload changed.

**2. Wall clock is not comparable across sessions.** A neighbour job on the other
NPUs shifts runtime by tens of percent, and nothing in these scripts can see it.
Stamp `rbln-stat` yourself before a session if the machine is shared, compare
only runs taken close together, and treat the prefix cache hit rate — which is
internal to the engine — as the robust metric.

**3. One run per mode can report the wrong sign, and the suite is the wrong
instrument for a small difference.** Two versions of this comparison reported
opposite signs from single runs. Aggregating a suite is what makes that possible:
it mixes short with long across seeds, and hit rate swings 15.6 points between
seeds alone. Use `repeat.sh` instead, which fixes the workload and clears the
cache between repetitions:

```bash
./repeat.sh <tag> 5 long 1
```

It needs `POST /reset_prefix_cache`, which `serve.sh` enables through
`VLLM_SERVER_DEV_MODE=1`, and refuses to start without it -- a silently failed
reset would let every repetition inherit the previous one's cache. It also
discards the first three runs: a fresh serve is still warming up something
outside the prefix cache, and hit rate tracks runtime at `r = -0.892` until it
settles. Measuring inside that transient is what produced both wrong signs, since
the two modes warm up in opposite directions.

After warm-up the instrument resolves about 1 point (sd 1.02, n=7), so five
repetitions per arm separate a 2-point difference. Decide the threshold before
the run, not after.

**4. The greedy probe cannot validate correctness on a DP serve.** Wrong KV
addressing here is silent *and reads faster* — output length is fixed, so every
benchmark metric improves while the model emits garbage. A greedy diff is the
cheap detector, but on `--data-parallel-size 4` the serve is not deterministic:
requests land on different ranks between runs, batch composition changes, and
floating-point reduction order with it. Two identical probe runs against the same
serve have differed in 3 of 6 completions. For a probe that means anything, run a
single-rank serve and issue requests serially:

```bash
DP=1 RBLN_DEVICES=4 ./serve.sh pagelayout
OUT_DIR=$OUT_DIR python probe_greedy.py "$OUT_DIR/probe_pagelayout.json"
# ...then the same under `./serve.sh subblock`, and:
OUT_DIR=$OUT_DIR python diff_probe.py subblock pagelayout
```

Any difference in the cached rounds on a deterministic serve is a bug. Do this
after every addressing change.

## Workloads

Both generate 16 conversations of 12–18 turns with a 500-token shared prefix and
120–160 in / 80–120 out per turn. They differ only in the per-conversation
prefix, which is what decides how many kernel blocks a conversation spans:

| workload | per-conversation prefix | spans |
|---|---|---|
| `multi_turn_short` | lognormal, avg 1000, max 5000 | under one 8192-token kernel block |
| `multi_turn_long` | lognormal, avg 6500, max 9000 | crosses into a second kernel block |

The long one matters more: crossing a kernel block boundary is where whole-group
sharing and the copy path both come into play, and it is the case an earlier
revision crashed on.
