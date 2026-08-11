#!/usr/bin/env bash
# Is the worker host-bound or device-bound? Answers it from /proc alone - no build,
# no code change, no restart. Run this against a live worker BEFORE forming any
# hypothesis - it tells you whether host-side work can matter at all before you
# spend a build on instrumenting it. See async_scheduling.md section 3.4.
#
# Reads, for the worker's main thread (the one running the model runner):
#   1. utime+stime vs wall  -> how much of the step is CPU at all
#   2. wchan samples        -> what it sleeps in (dma_fence_* = device, futex_* = lock)
#   3. ctxt switches        -> voluntary (it chose to sleep) vs involuntary (CPU starved)
#
# Usage: proc_probe.sh [seconds] [step_ms]
#   step_ms only scales the per-step column; get it from tok/s (batch*dp/tokps*1000).
set -u
SECS=${1:-30}
STEP_MS=${2:-16.42}

PIDS=$(pgrep -u "$(id -u)" -f "VLLM::Worker_DP" | head -8)
[ -z "$PIDS" ] && { echo "no VLLM::Worker_DP process found"; exit 1; }
PID=$(echo "$PIDS" | head -1)
echo "workers: $(echo "$PIDS" | tr '\n' ' ')"
echo "probing main thread of pid=$PID for ${SECS}s (step=${STEP_MS}ms)"

read -r U1 S1 <<<"$(awk '{print $14, $15}' "/proc/$PID/task/$PID/stat")"
W1=$(date +%s.%N)
declare -A CS1
for p in $PIDS; do
  CS1[$p]=$(awk '/voluntary_ctxt_switches/{printf "%s ", $2}' "/proc/$p/status")
done

# Sample wchan while the clock runs. 400 samples over the window is plenty to
# separate a two-thirds share from a five-percent one.
WCHAN=$(for _ in $(seq 1 400); do cat "/proc/$PID/wchan" 2>/dev/null; echo; sleep "$(echo "$SECS/400" | bc -l)"; done)
STATE=$(for _ in $(seq 1 200); do awk '{print $3}' "/proc/$PID/stat" 2>/dev/null; done)

read -r U2 S2 <<<"$(awk '{print $14, $15}' "/proc/$PID/task/$PID/stat")"
W2=$(date +%s.%N)

python3 - "$W1" "$W2" "$U1" "$U2" "$S1" "$S2" "$STEP_MS" <<'PY'
import sys
w1, w2, u1, u2, s1, s2, step_ms = (float(x) for x in sys.argv[1:8])
wall = w2 - w1
cpu = ((u2 - u1) + (s2 - s1)) / 100.0   # USER_HZ
steps = wall / (step_ms / 1e3)
print(f"\n[1] main-thread CPU vs wall over {wall:.1f}s, ~{steps:.0f} steps")
print(f"    CPU/wall            {cpu/wall:6.3f}")
print(f"    CPU per step        {cpu/steps*1e3:6.2f} ms   of {step_ms:.2f} ms")
print(f"    SLEEPING per step   {(1-cpu/wall)*step_ms:6.2f} ms")
print("    -> a low CPU share means host-side optimization cannot pay; find what it waits on.")
PY

echo
echo "[2] where it sleeps (400 wchan samples; dma_fence_* = device, futex_* = lock, 0 = running)"
echo "$WCHAN" | sort | uniq -c | sort -rn | head -12
echo "    run/sleep state (200 samples of stat field 3):"
echo "$STATE" | sort | uniq -c | sort -rn | sed 's/^/    /'

echo
echo "[3] context switches per step (voluntary = chose to sleep, involuntary = CPU starved)"
printf "    %-12s %14s %16s\n" pid voluntary/step involuntary/step
STEPS=$(python3 -c "print($SECS/($STEP_MS/1000))")
for p in $PIDS; do
  read -r V2 N2 <<<"$(awk '/voluntary_ctxt_switches/{printf "%s ", $2}' "/proc/$p/status")"
  read -r V1 N1 <<<"${CS1[$p]}"
  python3 -c "print(f'    {$p:<12} {($V2-$V1)/$STEPS:14.2f} {($N2-$N1)/$STEPS:16.2f}')"
done
echo "    -> involuntary ~0 with idle cores rules out CPU starvation."
