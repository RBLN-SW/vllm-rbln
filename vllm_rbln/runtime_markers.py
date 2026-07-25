# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in, process-local canonical runtime marker writer.

The writer deliberately depends only on the Python standard library so that it
can be used at the proxy, scheduler, connector, and model call boundaries.  It
is disabled unless ``VLLM_RBLN_RUNTIME_MARKER_DIR`` is set.  Marker failures
must never change inference behaviour.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_OUTPUT_DIR_ENV = "VLLM_RBLN_RUNTIME_MARKER_DIR"
_HOST_ID_ENV = "VLLM_RBLN_RUNTIME_MARKER_HOST_ID"
_CLOCK_DOMAIN_ID_ENV = "VLLM_RBLN_RUNTIME_MARKER_CLOCK_DOMAIN_ID"
_REQUEST_SUFFIX_ENV = "VLLM_RBLN_RUNTIME_MARKER_REQUEST_SUFFIX"
_SCHEMA_VERSION = "1.0.0"
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.:@/-]")
_ATTRIBUTE_NAME = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
)
_SENSITIVE_PARTS = (
    "address",
    "api_key",
    "authorization",
    "kv_content",
    "pointer",
    "prompt",
    "response",
    "secret",
    "tensor",
    "token_ids",
)
_KNOWN_REQUEST_PREFIXES = ("chatcmpl-", "cmpl-")
_COMPLETION_INTERNAL_SUFFIX = re.compile(
    r"^(?P<external>.+)-(?P<prompt_index>[0-9]+)-(?P<random>[0-9a-fA-F]{8})$"
)


def _safe_identifier(value: str | None, *, limit: int = 256) -> str:
    if not value:
        return ""
    return _SAFE_IDENTIFIER.sub("_", str(value))[:limit]


def _safe_text(value: str, *, limit: int = 512) -> str:
    return _HEX_ADDRESS.sub("<redacted_hex>", value)[:limit]


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "<depth_limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            safe_key = _safe_text(str(key), limit=64)
            lowered = safe_key.lower()
            if any(part in lowered for part in _SENSITIVE_PARTS):
                continue
            result[safe_key] = _safe_json_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_safe_json_value(item, depth=depth + 1) for item in value[:64]]
    return type(value).__name__


def _safe_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    result: dict[str, Any] = {}
    for key, value in attributes.items():
        name = str(key)
        lowered = name.lower()
        if not _ATTRIBUTE_NAME.fullmatch(name):
            continue
        if any(part in lowered for part in _SENSITIVE_PARTS):
            continue
        result[name] = _safe_json_value(value)
    return result


def correlation_id_from_request_id(request_id: str | None) -> str:
    """Return the proxy correlation id embedded in a vLLM request id."""
    value = str(request_id or "")
    for prefix in _KNOWN_REQUEST_PREFIXES:
        if value.startswith(prefix):
            external = value[len(prefix) :]
            if prefix == "cmpl-":
                # Completion requests are expanded by vLLM in two stages:
                # ``cmpl-<X-Request-Id>-<prompt-index>`` and then
                # ``...-<8 random hex chars>`` in InputProcessor.  The proxy
                # marker carries the original X-Request-Id, so remove only
                # that documented internal suffix before joining processes.
                match = _COMPLETION_INTERNAL_SUFFIX.fullmatch(external)
                if match is not None:
                    external = match.group("external")
            return _safe_identifier(external)
    return _safe_identifier(value)


class RuntimeMarkerSink:
    """Append-only JSONL marker sink isolated from the runtime control path."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str] | None = None,
        *,
        host_id: str | None = None,
        clock_domain_id: str | None = None,
        request_suffix_filter: str | None = None,
        monotonic_ns=time.monotonic_ns,
        register_atexit: bool = True,
    ) -> None:
        configured_dir = (
            os.environ.get(_OUTPUT_DIR_ENV) if output_dir is None else output_dir
        )
        self.output_dir = (
            Path(configured_dir) if configured_dir is not None and str(configured_dir)
            else None
        )
        self.enabled = self.output_dir is not None
        self.host_id = _safe_identifier(
            host_id
            if host_id is not None
            else os.environ.get(_HOST_ID_ENV, "localhost")
        )
        self.clock_domain_id = _safe_identifier(
            clock_domain_id
            if clock_domain_id is not None
            else os.environ.get(_CLOCK_DOMAIN_ID_ENV, "host-monotonic")
        )
        self.request_suffix_filter = str(
            os.environ.get(_REQUEST_SUFFIX_ENV, "")
            if request_suffix_filter is None
            else request_suffix_filter
        )
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._fd: int | None = None
        self._path: Path | None = None
        self._seen: set[tuple[str, str, int | None]] = set()
        self._sequence = 0
        self._records = 0
        self._bytes = 0
        self._write_total_ns = 0
        self._write_max_ns = 0
        self._dropped = 0
        self._duplicates = 0
        self._warned = False
        self._finalized = False
        if self.enabled and register_atexit:
            atexit.register(self.finalize)

    @property
    def path(self) -> Path | None:
        return self._path

    def enabled_for(self, request_id: str) -> bool:
        if not self.enabled:
            return False
        return not self.request_suffix_filter or str(request_id).endswith(
            self.request_suffix_filter
        )

    def _reset_after_fork_locked(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._pid = current_pid
        self._fd = None
        self._path = None
        self._seen.clear()
        self._sequence = 0
        self._records = 0
        self._bytes = 0
        self._write_total_ns = 0
        self._write_max_ns = 0
        self._dropped = 0
        self._duplicates = 0
        self._warned = False
        self._finalized = False

    def _open_locked(self) -> None:
        if self._fd is not None:
            return
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / f"runtime-markers-{self._pid}.jsonl"
        self._fd = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )

    def _warn_once_locked(self) -> None:
        if self._warned:
            return
        self._warned = True
        try:
            print(
                "runtime marker write failed; marker collection is degraded",
                file=sys.stderr,
            )
        except Exception:
            pass

    def _write_bytes_locked(self, payload: bytes) -> int:
        self._open_locked()
        assert self._fd is not None
        written = 0
        while written < len(payload):
            count = os.write(self._fd, payload[written:])
            if count <= 0:
                raise OSError("short runtime marker write")
            written += count
        return written

    def emit(
        self,
        event_name: str,
        request_id: str,
        *,
        phase: str,
        source: str,
        process_role: str,
        correlation_id: str | None = None,
        remote_request_id_suffix: str | None = None,
        transfer_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> bool:
        """Write one marker, returning false when disabled, duplicate, or dropped."""
        if not self.enabled_for(request_id):
            return False

        safe_request_id = _safe_identifier(request_id)
        safe_attributes = _safe_attributes(attributes)
        step_index = safe_attributes.get("decode.step_index")
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            step_index = None
        duplicate_key = (
            _safe_identifier(event_name, limit=96),
            safe_request_id,
            step_index,
        )

        with self._lock:
            self._reset_after_fork_locked()
            if duplicate_key in self._seen:
                self._duplicates += 1
                return False
            self._seen.add(duplicate_key)
            self._sequence += 1
            record: dict[str, Any] = {
                "schema_version": _SCHEMA_VERSION,
                "event_name": duplicate_key[0],
                "timestamp_ns": int(self._monotonic_ns()),
                "host_id": self.host_id,
                "clock_domain_id": self.clock_domain_id,
                "process_role": _safe_identifier(process_role, limit=96),
                "pid": self._pid,
                "thread_id": threading.get_ident(),
                "request_id": safe_request_id,
                "phase": _safe_identifier(phase, limit=96),
                "source": _safe_identifier(source, limit=192),
                "attributes": safe_attributes,
                "sequence": self._sequence,
            }
            if correlation_id:
                record["correlation_id"] = _safe_identifier(correlation_id)
            if remote_request_id_suffix:
                record["remote_request_id_suffix"] = _safe_identifier(
                    str(remote_request_id_suffix)[-64:]
                )
            if transfer_id:
                record["transfer_id"] = _safe_identifier(transfer_id)

            payload = (
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
                + b"\n"
            )
            start_ns = self._monotonic_ns()
            try:
                written = self._write_bytes_locked(payload)
            except Exception:
                self._dropped += 1
                self._seen.discard(duplicate_key)
                self._warn_once_locked()
                return False
            elapsed_ns = max(0, int(self._monotonic_ns()) - int(start_ns))
            self._records += 1
            self._bytes += written
            self._write_total_ns += elapsed_ns
            self._write_max_ns = max(self._write_max_ns, elapsed_ns)
            return True

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            average = (
                self._write_total_ns / self._records if self._records else 0.0
            )
            return {
                "records": self._records,
                "bytes": self._bytes,
                "average_write_ns": average,
                "max_write_ns": self._write_max_ns,
                "dropped": self._dropped,
                "duplicates": self._duplicates,
            }

    def finalize(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._reset_after_fork_locked()
            if self._finalized:
                return
            self._finalized = True
            average = (
                self._write_total_ns / self._records if self._records else 0.0
            )
            stats = {
                "schema_version": _SCHEMA_VERSION,
                "pid": self._pid,
                "records": self._records,
                "bytes": self._bytes,
                "average_write_ns": average,
                "max_write_ns": self._write_max_ns,
                "dropped": self._dropped,
                "duplicates": self._duplicates,
            }
            try:
                assert self.output_dir is not None
                self.output_dir.mkdir(parents=True, exist_ok=True)
                stats_path = self.output_dir / f"runtime-markers-{self._pid}.stats.json"
                flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
                stats_fd = os.open(stats_path, flags, 0o600)
                try:
                    payload = json.dumps(
                        stats,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8") + b"\n"
                    offset = 0
                    while offset < len(payload):
                        count = os.write(stats_fd, payload[offset:])
                        if count <= 0:
                            raise OSError("short runtime marker stats write")
                        offset += count
                finally:
                    os.close(stats_fd)
            except Exception:
                self._warn_once_locked()
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None


_SINK_LOCK = threading.Lock()
_SINK: RuntimeMarkerSink | None = None


def get_runtime_marker_sink() -> RuntimeMarkerSink:
    global _SINK
    with _SINK_LOCK:
        if _SINK is None:
            _SINK = RuntimeMarkerSink()
        return _SINK
