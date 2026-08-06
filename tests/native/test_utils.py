# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the suite's own harness (tests/native/utils.py): a fault in the
env scrub silently makes every other test depend on the developer's shell."""

from __future__ import annotations

import os

import pytest

from .test_envs import ENV_SOURCE_ALIASES, RBLN_KEYS
from .utils import (
    HOST_ENV_PASSTHROUGH,
    NATIVE_ENV,
    SCRUBBED_EXTRA,
    SCRUBBED_PREFIXES,
    is_scrubbed,
    scrub_env,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # A host description, even though it matches the RBLN_ prefix: the
        # passthrough list takes precedence, which is the whole rule.
        ("RBLN_DEVICES", False),
        ("RBLN_FORCE_NPU_NAME", False),
        # Same prefix, but a behavior knob rather than host description.
        ("RBLN_CTX_STANDALONE", True),
        ("RBLN_RUNTIME_FORCE_SYNC", True),
        ("VLLM_RBLN_SORT_BATCH", True),
        # No RBLN prefix at all; caught only because it is listed explicitly.
        ("VLLM_USE_V2_MODEL_RUNNER", True),
        # An upstream vLLM variable that is NOT listed. Scrubbing is targeted,
        # not a blanket VLLM_ sweep -- widening it would wipe logging config.
        ("VLLM_LOGGING_LEVEL", False),
        ("VLLM_HOST_IP", False),
        ("PATH", False),
        ("HOME", False),
    ],
)
def test_is_scrubbed_classification(key, expected):
    assert is_scrubbed(key) is expected


def test_scrub_env_removes_reports_and_preserves():
    """The three observable halves of the contract, in one pass."""
    environ = {
        "VLLM_RBLN_SORT_BATCH": "1",
        "RBLN_CTX_STANDALONE": "1",
        "RBLN_DEVICES": "0,1",
        "PATH": "/usr/bin",
    }
    removed = scrub_env(environ)

    # Reported with values, so pytest_report_header can show what it took.
    assert removed == {"VLLM_RBLN_SORT_BATCH": "1", "RBLN_CTX_STANDALONE": "1"}
    # Mutated in place, and nothing else was touched.
    assert environ == {"RBLN_DEVICES": "0,1", "PATH": "/usr/bin"}


def test_scrub_env_reports_nothing_when_already_clean():
    environ = {"PATH": "/usr/bin", "RBLN_DEVICES": "0"}
    assert scrub_env(environ) == {}
    assert environ == {"PATH": "/usr/bin", "RBLN_DEVICES": "0"}


def test_scrub_env_defaults_to_the_process_environment():
    """Restored by hand: monkeypatch cannot undo a deletion it did not make, and
    scrub_env() live also removes what conftest injected."""
    saved = dict(os.environ)
    try:
        os.environ["VLLM_RBLN_PROBE"] = "1"
        removed = scrub_env()
        assert removed["VLLM_RBLN_PROBE"] == "1"
        assert "VLLM_RBLN_PROBE" not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_passthrough_entries_are_real_exceptions():
    """Every passthrough entry must be something the rule would otherwise scrub;
    a name no prefix matches is a dead entry."""
    inert = sorted(
        k for k in HOST_ENV_PASSTHROUGH if not k.startswith(SCRUBBED_PREFIXES)
    )
    assert not inert, f"{inert} are exempted from a rule that never applied to them"


def test_passthrough_and_extra_are_disjoint():
    """Listing a name in both would let passthrough silently win."""
    assert not (HOST_ENV_PASSTHROUGH & SCRUBBED_EXTRA)


def test_native_env_pins_no_host_description():
    """The suite may pin behavior, never machine identity: pinning a passthrough
    like RBLN_DEVICES would hard-code one host's topology."""
    assert not (set(NATIVE_ENV) & HOST_ENV_PASSTHROUGH)


def test_scrub_covers_every_name_the_env_table_reads():
    """A read name that survives scrubbing makes the suite host-dependent. Beyond
    the registered keys, VLLM_RBLN_TP_SIZE and the rebel compiler's custom-kernel
    flag are also read, so all must be covered."""
    names = set(RBLN_KEYS) | set(ENV_SOURCE_ALIASES.values()) | {"VLLM_RBLN_TP_SIZE"}
    uncovered = sorted(name for name in names if not is_scrubbed(name))
    assert not uncovered, (
        f"{uncovered} are read by vllm_rbln.envs but survive scrubbing, so a "
        f"value exported in the developer's shell would leak into every test"
    )
