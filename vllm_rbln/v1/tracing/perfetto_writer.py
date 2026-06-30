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

"""Perfetto Trace Writer for vLLM EngineCore.

Collects Chrome Trace JSON events (duration / async) and flushes to file.
Designed to be used from the EngineCore process, controlled via the
/v1/trace/start and /v1/trace/stop endpoints.

Output files can be opened in https://ui.perfetto.dev/
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path


class PerfettoTraceWriter:
    """Collects Chrome Trace JSON events and writes them to a file."""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._events: list[dict] = []
        self._lock = threading.Lock()
        # Compute wall-clock offset once so monotonic timestamps can be
        # converted to absolute microseconds (required by Perfetto).
        self._wall_offset = time.time() - time.monotonic()
        self._pid = os.getpid()

    def monotonic_to_us(self, mono_ts: float) -> int:
        """Convert a monotonic timestamp to wall-clock microseconds."""
        return int((self._wall_offset + mono_ts) * 1_000_000)

    def emit_duration(
        self,
        name: str,
        ts_mono: float,
        dur_s: float,
        tid: str = "main",
        args: dict | None = None,
        cname: str | None = None,
    ) -> None:
        """Emit a complete duration event (ph=X)."""
        ev: dict = {
            "ph": "X",
            "name": name,
            "ts": self.monotonic_to_us(ts_mono),
            "dur": int(dur_s * 1_000_000),
            "pid": self._pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        if cname:
            ev["cname"] = cname
        with self._lock:
            self._events.append(ev)

    def emit_async_begin(
        self,
        name: str,
        id: str,
        ts_mono: float,
        cat: str = "lifecycle",
        args: dict | None = None,
    ) -> None:
        """Emit an async begin event (ph=b)."""
        ev: dict = {
            "ph": "b",
            "name": name,
            "id": id,
            "ts": self.monotonic_to_us(ts_mono),
            "pid": self._pid,
            "cat": cat,
        }
        if args:
            ev["args"] = args
        with self._lock:
            self._events.append(ev)

    def emit_async_end(
        self,
        name: str,
        id: str,
        ts_mono: float,
        cat: str = "lifecycle",
        args: dict | None = None,
    ) -> None:
        """Emit an async end event (ph=e)."""
        ev: dict = {
            "ph": "e",
            "name": name,
            "id": id,
            "ts": self.monotonic_to_us(ts_mono),
            "pid": self._pid,
            "cat": cat,
        }
        if args:
            ev["args"] = args
        with self._lock:
            self._events.append(ev)

    def flush(self) -> int:
        """Write events to JSON file (atomic: tmp → rename). Returns event count."""
        with self._lock:
            events = list(self._events)
        if not events:
            return 0
        payload = {"traceEvents": events}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=0)
            os.replace(tmp_path, str(self._path))
        except BaseException:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return len(events)

    def reset(self) -> None:
        """Clear all collected events (for a new session)."""
        with self._lock:
            self._events.clear()
