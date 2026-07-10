# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RBLNModelRunner KV physical-layout drift guard.

``_verify_kv_physical_layout`` compares the compiler-assigned physical shape
(``torch.rbln.physical_shape``) of each bound KV cache against the shape the
framework built at init, warning by default and raising when
``VLLM_RBLN_STRICT_KV_LAYOUT=1``. These drive the method on a stub runner with a
stubbed ``torch.rbln.physical_shape`` — no device or serve needed.
"""

# Standard
import types

# Third Party
import pytest
import torch

# First Party
from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner


def _runner(expected, names, caches):
    r = RBLNModelRunner.__new__(RBLNModelRunner)
    r._kv_expected_physical_by_name = expected
    r.kv_cache_names = names
    r.kv_caches = caches
    return r


def _stub_physical_shape(monkeypatch, mapping):
    ns = types.SimpleNamespace(physical_shape=lambda kv: mapping[kv])
    monkeypatch.setattr(torch, "rbln", ns, raising=False)


def test_match_passes(monkeypatch):
    r = _runner({"l0": (2, 4), "l1": (2, 4)}, ["l0", "l1"], ["kv0", "kv1"])
    _stub_physical_shape(monkeypatch, {"kv0": (2, 4), "kv1": (2, 4)})
    r._verify_kv_physical_layout()


def test_mismatch_warns_by_default(monkeypatch):
    r = _runner({"l0": (2, 4)}, ["l0"], ["kv0"])
    _stub_physical_shape(monkeypatch, {"kv0": (2, 8)})
    monkeypatch.delenv("VLLM_RBLN_STRICT_KV_LAYOUT", raising=False)
    r._verify_kv_physical_layout()


def test_mismatch_strict_raises(monkeypatch):
    r = _runner({"l0": (2, 4)}, ["l0"], ["kv0"])
    _stub_physical_shape(monkeypatch, {"kv0": (2, 8)})
    monkeypatch.setenv("VLLM_RBLN_STRICT_KV_LAYOUT", "1")
    with pytest.raises(RuntimeError, match="physical-layout drift"):
        r._verify_kv_physical_layout()


def test_unbound_view_skipped(monkeypatch):
    # Empty physical shape (no view bound) is skipped, even in strict mode.
    r = _runner({"l0": (2, 4)}, ["l0"], ["kv0"])
    _stub_physical_shape(monkeypatch, {"kv0": ()})
    monkeypatch.setenv("VLLM_RBLN_STRICT_KV_LAYOUT", "1")
    r._verify_kv_physical_layout()


def test_getter_unavailable_noop(monkeypatch):
    # Older torch-rbln without the query: verify is a no-op.
    r = _runner({"l0": (2, 4)}, ["l0"], ["kv0"])
    monkeypatch.setattr(torch, "rbln", types.SimpleNamespace(), raising=False)
    r._verify_kv_physical_layout()
