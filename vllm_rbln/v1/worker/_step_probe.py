# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporary per-step probe. Not for merge.

Answers one question: does the main thread reach forward(N+1)'s dispatch while
the device is still executing forward(N)?

`blocked` is wall minus CPU time on this thread, so a region that waits on the
device shows up there and pure host work does not. If the inter-step host work
is free (blocked ~ 0) and the first transfer inside forward.dispatch is not,
the host ran ahead and the step is device-bound.

Enable with VLLM_RBLN_STEP_PROBE=<steps per report>; VLLM_RBLN_PYMARK=1 adds
microsecond breadcrumbs into rebel's own log stream so TRACE lines can be
attributed to a call site.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict


def _now() -> tuple[float, float]:
    return time.perf_counter(), time.thread_time()


class _Region:
    __slots__ = ("probe", "name", "w0", "c0")

    def __init__(self, probe: StepProbe, name: str) -> None:
        self.probe = probe
        self.name = name

    def __enter__(self) -> _Region:
        self.w0, self.c0 = _now()
        return self

    def __exit__(self, *exc) -> None:
        w1, c1 = _now()
        self.probe.add(self.name, w1 - self.w0, c1 - self.c0)


class StepProbe:
    """Per-step wall/cpu/blocked, reported every `window` steps."""

    def __init__(self) -> None:
        self.window = int(os.environ.get("VLLM_RBLN_STEP_PROBE", "0") or 0)
        self.on = self.window > 0
        self._reset()

    def _reset(self) -> None:
        self.n = 0
        self.acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        # Per-step samples, not just the window sum. A window mean cannot show a
        # tail, and the DP all_reduce is exactly the region whose tail matters:
        # hiding it under device(N) is what async scheduling is for, and a mean
        # of 1 ms is equally consistent with "always 1 ms" and with "0.9 ms most
        # steps, 5 ms sometimes". Those two say opposite things about whether
        # async is working.
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.steps: list[float] = []
        self.w0, self.c0 = _now()
        self._last_w = self.w0

    def add(self, name: str, wall: float, cpu: float) -> None:
        if not self.on:
            return
        slot = self.acc[name]
        slot[0] += wall
        slot[1] += cpu
        self.samples[name].append(wall)

    def region(self, name: str) -> _Region:
        return _Region(self, name)

    @staticmethod
    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))] * 1e3

    def tick(self) -> None:
        if not self.on:
            return
        w1, c1 = _now()
        self.steps.append(w1 - self._last_w)
        self._last_w = w1
        self.n += 1
        if self.n < self.window:
            return
        wall, cpu = w1 - self.w0, c1 - self.c0
        n = self.n
        parts = [
            f"step mean={wall / n * 1e3:7.3f} cpu={cpu / n * 1e3:7.3f} "
            f"blocked={(wall - cpu) / n * 1e3:7.3f} ms  (n={n})",
            f"    {'REGION':<20}{'mean':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}",
            f"    {'step':<20}{wall / n * 1e3:>8.3f}"
            f"{self._pct(self.steps, 0.50):>8.3f}{self._pct(self.steps, 0.90):>8.3f}"
            f"{self._pct(self.steps, 0.99):>8.3f}{self._pct(self.steps, 1.0):>8.3f}",
        ]
        for name in sorted(self.acc):
            rw, rc = self.acc[name]
            v = self.samples[name]
            parts.append(
                f"    {name:<20}{rw / n * 1e3:>8.3f}"
                f"{self._pct(v, 0.50):>8.3f}{self._pct(v, 0.90):>8.3f}"
                f"{self._pct(v, 0.99):>8.3f}{self._pct(v, 1.0):>8.3f}"
                f"   cpu={rc / n * 1e3:6.3f}"
            )
        print("[probe] " + "\n".join(parts), file=sys.stderr, flush=True)
        self._reset()


_PYMARK = os.environ.get("VLLM_RBLN_PYMARK", "0").lower() in ("1", "true")


def mark(label: str) -> None:
    """Breadcrumb in rebel's log format, so TRACE lines can be bracketed."""
    if not _PYMARK:
        return
    t = time.time()
    ms = int((t % 1) * 1e6)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    pid = os.getpid()
    print(
        f"[{stamp}.{ms:06d}] [{pid}][{pid}] [I] [pymark] {label}",
        file=sys.stderr,
        flush=True,
    )
