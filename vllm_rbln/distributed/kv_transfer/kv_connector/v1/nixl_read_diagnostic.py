# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import faulthandler
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_DIAGNOSTIC_ENV = "VLLM_RBLN_NIXL_READ_DIAGNOSTIC"
_REQUEST_SUFFIX_ENV = "VLLM_RBLN_NIXL_READ_DIAGNOSTIC_REQUEST_SUFFIX"
_MARKER_PREFIX = "PHASE4B_NIXL_DIAG"
_POLL_LOG_INTERVAL_NS = 1_000_000_000
_STATE_LOG_HEARTBEAT_NS = 10_000_000_000
_WATCHDOG_DELAYS_SECONDS = (5.0, 15.0, 30.0)
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _env_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_suffix(value: str | None, length: int = 12) -> str:
    if not value:
        return ""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    return safe[-length:]


def _safe_text(value: str, limit: int = 512) -> str:
    return _HEX_ADDRESS.sub("<redacted_hex>", value)[:limit]


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, Mapping):
        return {
            _safe_text(str(key), 64): _safe_value(item) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_value(item) for item in value]
    return type(value).__name__


def safe_status_fields(value: Any) -> dict[str, Any]:
    """Return status metadata without formatting a handle or native object."""
    fields: dict[str, Any] = {"return_type": type(value).__name__}
    if value is None:
        fields["return_status"] = None
        return fields
    if isinstance(value, (str, int, bool)):
        fields["return_status"] = value
        return fields

    name = getattr(value, "name", None)
    enum_value = getattr(value, "value", None)
    fields["return_status"] = _safe_value(name)
    if isinstance(enum_value, (str, int, bool)):
        fields["return_value"] = enum_value
    return fields


@dataclass(frozen=True)
class DiagnosticContext:
    request_id: str
    remote_engine_id: str = ""
    logical_block_count: int = 0
    descriptor_count: int = 0
    total_bytes: int = 0


@dataclass(frozen=True)
class PollToken:
    request_id: str
    sequence: int
    start_ns: int
    should_log: bool
    previous_status: str | None


class NixlReadDiagnostic:
    """Request-scoped structured logging for the Phase 4B NIXL READ boundary."""

    def __init__(
        self,
        logger,
        *,
        enabled: bool | None = None,
        request_suffix_filter: str | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        timer_factory: Callable[[float, Callable[[], None]], Any] = threading.Timer,
        traceback_dumper: Callable[..., None] = faulthandler.dump_traceback,
    ) -> None:
        self.logger = logger
        self.enabled = (
            _env_enabled(os.environ.get(_DIAGNOSTIC_ENV))
            if enabled is None
            else enabled
        )
        self.request_suffix_filter = (
            os.environ.get(_REQUEST_SUFFIX_ENV, "")
            if request_suffix_filter is None
            else request_suffix_filter
        )
        self._monotonic_ns = monotonic_ns
        self._timer_factory = timer_factory
        self._traceback_dumper = traceback_dumper
        self._lock = threading.Lock()
        self._poll_counts: dict[str, int] = {}
        self._poll_status: dict[str, str | None] = {}
        self._poll_last_log_ns: dict[str, int] = {}
        self._state_log: dict[tuple[str, str], tuple[str, int]] = {}
        self._transfer_return_ns: dict[str, int] = {}
        self._watchdogs: dict[str, list[Any]] = {}
        self._handle_sequence = 0
        self._handle_tokens: dict[int, str] = {}

    def enabled_for(self, request_id: str) -> bool:
        if not self.enabled:
            return False
        return not self.request_suffix_filter or request_id.endswith(
            self.request_suffix_filter
        )

    def now_ns(self) -> int:
        return self._monotonic_ns()

    def emit(
        self,
        marker: str,
        context: DiagnosticContext,
        *,
        elapsed_ns: int | None = None,
        status: str | None = None,
        exception: BaseException | None = None,
        **extra: Any,
    ) -> None:
        if not self.enabled_for(context.request_id):
            return
        record: dict[str, Any] = {
            "diagnostic_marker": _MARKER_PREFIX,
            "marker": marker,
            "monotonic_ns": self._monotonic_ns(),
            "request_suffix": _safe_suffix(context.request_id),
            "remote_engine_suffix": _safe_suffix(context.remote_engine_id),
            "logical_block_count": context.logical_block_count,
            "descriptor_count": context.descriptor_count,
            "total_bytes": context.total_bytes,
            "thread_id": threading.get_ident(),
            "pid": os.getpid(),
            "elapsed_ns": elapsed_ns,
            "status": status,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_message": _safe_text(str(exception)) if exception else None,
        }
        record.update({key: _safe_value(value) for key, value in extra.items()})
        self.logger.warning(
            "%s %s",
            _MARKER_PREFIX,
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )

    def assign_handle_token(self, handle: Any) -> str:
        key = id(handle)
        with self._lock:
            token = self._handle_tokens.get(key)
            if token is None:
                self._handle_sequence += 1
                token = f"h{self._handle_sequence}"
                self._handle_tokens[key] = token
            return token

    def handle_token(self, handle: Any) -> str:
        return self.assign_handle_token(handle)

    def forget_handle(self, handle: Any) -> None:
        with self._lock:
            self._handle_tokens.pop(id(handle), None)

    def note_transfer_return(self, request_id: str, when_ns: int | None = None) -> None:
        with self._lock:
            self._transfer_return_ns[request_id] = (
                self._monotonic_ns() if when_ns is None else when_ns
            )

    def elapsed_since_transfer_return(self, request_id: str, now_ns: int) -> int | None:
        with self._lock:
            transfer_return_ns = self._transfer_return_ns.get(request_id)
        if transfer_return_ns is None:
            return None
        return max(0, now_ns - transfer_return_ns)

    def begin_poll(
        self, context: DiagnosticContext, *, handle_present: bool
    ) -> PollToken:
        now_ns = self._monotonic_ns()
        with self._lock:
            sequence = self._poll_counts.get(context.request_id, 0) + 1
            self._poll_counts[context.request_id] = sequence
            previous_status = self._poll_status.get(context.request_id)
            last_log_ns = self._poll_last_log_ns.get(context.request_id)
            should_log = sequence == 1 or (
                last_log_ns is None or now_ns - last_log_ns >= _POLL_LOG_INTERVAL_NS
            )
            if should_log:
                self._poll_last_log_ns[context.request_id] = now_ns
        token = PollToken(
            request_id=context.request_id,
            sequence=sequence,
            start_ns=now_ns,
            should_log=should_log,
            previous_status=previous_status,
        )
        if should_log:
            self.emit(
                "status_poll_enter",
                context,
                status="enter",
                poll_sequence=sequence,
                previous_status=previous_status,
                handle_container_present=handle_present,
                elapsed_since_transfer_return_ns=self.elapsed_since_transfer_return(
                    context.request_id, now_ns
                ),
            )
        return token

    def finish_poll(
        self,
        context: DiagnosticContext,
        token: PollToken,
        status: Any,
        *,
        handle_present: bool,
    ) -> None:
        now_ns = self._monotonic_ns()
        status_fields = safe_status_fields(status)
        status_name = status_fields.get("return_status")
        status_text = None if status_name is None else str(status_name)
        changed = status_text != token.previous_status
        with self._lock:
            self._poll_status[context.request_id] = status_text
            if changed:
                self._poll_last_log_ns[context.request_id] = now_ns
        fields = {
            **status_fields,
            "poll_sequence": token.sequence,
            "previous_status": token.previous_status,
            "status_changed": changed,
            "handle_container_present": handle_present,
            "elapsed_since_transfer_return_ns": self.elapsed_since_transfer_return(
                context.request_id, now_ns
            ),
        }
        if token.should_log or changed:
            self.emit(
                "status_poll",
                context,
                elapsed_ns=now_ns - token.start_ns,
                status=status_text,
                **fields,
            )
        if changed:
            self.emit(
                "status_change",
                context,
                elapsed_ns=now_ns - token.start_ns,
                status=status_text,
                **fields,
            )

    def fail_poll(
        self,
        context: DiagnosticContext,
        token: PollToken,
        exception: BaseException,
        *,
        handle_present: bool,
    ) -> None:
        now_ns = self._monotonic_ns()
        self.emit(
            "status_poll_exception",
            context,
            elapsed_ns=now_ns - token.start_ns,
            status="exception",
            exception=exception,
            poll_sequence=token.sequence,
            previous_status=token.previous_status,
            handle_container_present=handle_present,
            elapsed_since_transfer_return_ns=self.elapsed_since_transfer_return(
                context.request_id, now_ns
            ),
        )

    def poll_count(self, request_id: str) -> int:
        with self._lock:
            return self._poll_counts.get(request_id, 0)

    def should_emit_state(
        self,
        request_id: str,
        channel: str,
        state: str,
        *,
        force: bool = False,
        heartbeat_ns: int = _STATE_LOG_HEARTBEAT_NS,
    ) -> bool:
        """Rate-limit a repeated diagnostic state without hiding transitions."""
        if not self.enabled_for(request_id):
            return False
        now_ns = self._monotonic_ns()
        key = (request_id, channel)
        with self._lock:
            previous = self._state_log.get(key)
            should_emit = (
                force
                or previous is None
                or previous[0] != state
                or now_ns - previous[1] >= heartbeat_ns
            )
            if force:
                self._state_log.pop(key, None)
            elif should_emit:
                self._state_log[key] = (state, now_ns)
        return should_emit

    def start_watchdog(
        self,
        context: DiagnosticContext,
        delays: Sequence[float] = _WATCHDOG_DELAYS_SECONDS,
    ) -> None:
        if not self.enabled_for(context.request_id):
            return
        self.cancel_watchdog(context.request_id)
        timers: list[Any] = []

        for delay in delays:

            def dump(delay_seconds: float = delay) -> None:
                self.emit(
                    "watchdog_stack",
                    context,
                    status="pending",
                    watchdog_delay_seconds=delay_seconds,
                    native_stack=False,
                )
                self._traceback_dumper(file=sys.stderr, all_threads=True)

            timer = self._timer_factory(float(delay), dump)
            timer.daemon = True
            timers.append(timer)

        with self._lock:
            self._watchdogs[context.request_id] = timers
        for timer in timers:
            timer.start()

    def cancel_watchdog(self, request_id: str) -> None:
        with self._lock:
            timers = self._watchdogs.pop(request_id, [])
        for timer in timers:
            timer.cancel()

    def finish_request(self, request_id: str) -> None:
        self.cancel_watchdog(request_id)
        with self._lock:
            self._poll_counts.pop(request_id, None)
            self._poll_status.pop(request_id, None)
            self._poll_last_log_ns.pop(request_id, None)
            self._transfer_return_ns.pop(request_id, None)
            for key in [
                key for key in self._state_log if key[0] == request_id
            ]:
                self._state_log.pop(key, None)
