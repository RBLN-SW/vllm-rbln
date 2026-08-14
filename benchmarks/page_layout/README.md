# Page layout vs sub-block prefix caching

Scripts for the one comparison that decides whether
[page layout](../../docs/page_layout_kv_manager.md) is worth keeping: does it beat
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
| `run_suite.sh` | the paired suite: warmup, seeds, interleaved workloads |
| `analyze.py` | pairs the two modes by (workload, seed), reports deltas and a Welch t |
| `probe_greedy.py` | fixed prompts at temperature 0, for a correctness diff |
| `diff_probe.py` | byte-diffs two probe outputs |
| `workloads/*.json` | conversation generators for the multi-turn driver |

## Prerequisites

The multi-turn driver lives in the vLLM **source tree** (it is not in the wheel),
so point at a checkout:

```bash
export VLLM_REPO=$HOME/workspace/vllm     # must contain benchmarks/multi_turn/
pip install -r "$VLLM_REPO/benchmarks/multi_turn/requirements.txt"
```

The generators draw their filler text from a Project Gutenberg book, resolved
relative to the results directory:

```bash
export OUT_DIR=$PWD/results
mkdir -p "$OUT_DIR" && curl -sL https://www.gutenberg.org/files/1184/1184-0.txt \
    -o "$OUT_DIR/pg1184.txt"
```

## Running

One mode at a time, because the two need the same devices.

```bash
# terminal 1
RBLN_DEVICES=4,5,6,7 ./serve.sh pagelayout | tee "$OUT_DIR/serve_pagelayout.log"

# terminal 2, once "Application startup complete" appears on every DP rank
VLLM_REPO=$VLLM_REPO OUT_DIR=$OUT_DIR ./run_suite.sh pagelayout

# then stop the serve, launch `./serve.sh subblock`, and repeat:
VLLM_REPO=$VLLM_REPO OUT_DIR=$OUT_DIR ./run_suite.sh subblock

python analyze.py "$OUT_DIR"
```

Wait for the devices to report zero used memory between the two serves; launching
while the previous one is still releasing has produced worker startup failures
that look unrelated to the change under test.

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
NPUs shifts runtime by tens of percent. `run_suite.sh` stamps `rbln-stat` before
every run precisely so this is visible after the fact. Compare *within* a paired
suite, and treat the prefix cache hit rate — which is internal to the engine — as
the robust metric.

**3. One run per mode can report the wrong sign.** The first version of this
comparison did: it ran the two modes against differently composed workloads and
concluded page layout was 8.8% faster. The paired protocol reversed that. Use at
least three seeds and read the 95% CI that `analyze.py` prints.

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
