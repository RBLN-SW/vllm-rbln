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

from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_rbln.v1.attention.ops import (
    flash_attention_naive,
    flash_causal_attention_naive,
    flash_causal_mla_naive,
)


class _Recorder:
    """Stands in for torch.ops.rbln_custom_ops and records each call's kwargs."""

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def op(*args, **kwargs):
            self.calls[name] = kwargs

        return op


# (module, op, positional arguments, keyword arguments) of each wrapper; the
# tensors are never read, so None stands in for them.
_FLASH: tuple[int, dict[str, Any]] = (8, {"use_custom_kernel": False})
_MLA: tuple[int, dict[str, Any]] = (7, {})
_OPS = {
    "flash_causal_attention_naive_prefill": (flash_causal_attention_naive, *_FLASH),
    "flash_causal_attention_naive_decode": (flash_causal_attention_naive, *_FLASH),
    "flash_attention_naive_prefill": (flash_attention_naive, *_FLASH),
    "flash_attention_naive_decode": (flash_attention_naive, *_FLASH),
    "paged_flash_causal_mla_naive_prefill": (flash_causal_mla_naive, *_MLA),
    "paged_flash_causal_mla_naive_decode": (flash_causal_mla_naive, *_MLA),
}


# The dense in-memory ops' context-parallel cache store writes the rows a node
# does not own to the null block, so every rbln_custom_ops call names it.
@pytest.mark.parametrize("op", list(_OPS))
def test_dense_attention_ops_pass_null_block_id(monkeypatch, op):
    module, num_args, kwargs = _OPS[op]
    recorder = _Recorder()
    monkeypatch.setattr(
        module, "torch", SimpleNamespace(ops=SimpleNamespace(rbln_custom_ops=recorder))
    )
    getattr(module, op)(*[None] * num_args, **kwargs)
    assert recorder.calls[op]["null_block_id"] == NULL_BLOCK_ID
