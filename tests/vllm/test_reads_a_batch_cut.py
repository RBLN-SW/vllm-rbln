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

from __future__ import annotations

import sys
import types

import pytest

from vllm_rbln.utils import reads_a_batch_cut

_TABLE = "rebel.core.custom_converter.kernel_table.fake_residual"


def _table(monkeypatch, **kernels):
    module = types.ModuleType(_TABLE)
    for name, value in kernels.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, _TABLE, module)


def _read(*axes):
    shards = [types.SimpleNamespace(axes=(axis,)) for axis in axes]
    return types.SimpleNamespace(shard=shards)


@pytest.mark.parametrize(
    "kernels,expected",
    [
        # the read takes the batch cut itself
        ({"READ": _read("T", "B")}, True),
        # a compiler that cuts only prefill
        ({"READ": _read("T")}, False),
    ],
)
def test_reads_a_batch_cut(monkeypatch, kernels, expected):
    _table(monkeypatch, **kernels)
    assert reads_a_batch_cut("fake_residual", "READ") is expected


def test_a_compiler_without_the_table_reads_no_batch_cut():
    assert not reads_a_batch_cut("no_such_table", "READ")
