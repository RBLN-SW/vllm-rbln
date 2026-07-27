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

"""Where the rbln custom ops come from at runtime.

rebel-compiler owns them; the copies in this package exist only so a build
against an older compiler still runs. These tests keep that fallback from
becoming the path everything silently takes.
"""

import importlib
import importlib.metadata

import pytest
import torch

from vllm_rbln import custom_ops

# Every module that registers rbln custom ops.
_OP_MODULES = [
    "vllm_rbln.model_executor.layers.fused_moe.all2all",
    "vllm_rbln.model_executor.layers.fused_moe.layer",
    "vllm_rbln.model_executor.layers.quantization.fp8",
    "vllm_rbln.model_executor.layers.quantization.mxfp4",
    "vllm_rbln.v1.attention.ops.attention_naive",
    "vllm_rbln.v1.attention.ops.causal_attention_naive",
    "vllm_rbln.v1.attention.ops.flash_attention_naive",
    "vllm_rbln.v1.attention.ops.flash_causal_attention_naive",
    "vllm_rbln.v1.attention.ops.flash_causal_mla_naive",
    "vllm_rbln.v1.attention.ops.sliding_window_attention_naive",
]


@pytest.fixture(scope="module", autouse=True)
def _import_op_modules():
    """Registration is an import side effect, so pull every module in first."""
    for name in _OP_MODULES:
        importlib.import_module(name)


def _compiler_version() -> str:
    try:
        return importlib.metadata.version("rebel-compiler")
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def test_every_op_is_registered():
    """Whoever supplied it, each op has to be callable through torch.ops."""
    known = custom_ops.FROM_COMPILER | custom_ops.FALLBACK
    assert known, "no rbln custom op was registered at all"
    missing = [name for name in known if custom_ops._lookup(name) is None]
    assert not missing, f"registered but not resolvable: {missing}"


def test_rebel_compiler_supplies_every_op():
    """The copies here are dead weight on a supported compiler.

    A failure means the pinned rebel-compiler predates the op move: either bump
    it, or keep the listed fallbacks until the pin catches up.
    """
    assert custom_ops.fallback_ops() == frozenset(), (
        f"rebel-compiler {_compiler_version()} does not register "
        f"{sorted(custom_ops.fallback_ops())}; vllm-rbln had to define them."
    )


def test_local_copies_match_the_compiler():
    """A copy that drifted from the compiler's schema is worse than no copy.

    It takes over on any older compiler that lacks the op, and then traces a
    graph the compiler does not lower the same way.
    """
    drifted = {
        name: (custom_ops.LOCAL_SCHEMAS[name], custom_ops._existing_schema(name))
        for name in sorted(custom_ops.FROM_COMPILER & set(custom_ops.LOCAL_SCHEMAS))
        if custom_ops.LOCAL_SCHEMAS[name] != custom_ops._existing_schema(name)
    }
    report = "\n".join(
        f"  {name}\n    rebel-compiler: {theirs}\n    vllm-rbln:      {ours}"
        for name, (ours, theirs) in drifted.items()
    )
    assert not drifted, f"stale op copies in vllm-rbln:\n{report}"


def test_compiler_wins_regardless_of_import_order():
    """Registering an op here after the compiler must not shadow it."""
    name = "rbln_custom_ops::linear"
    if custom_ops._lookup(name) is None:
        pytest.skip(f"rebel-compiler {_compiler_version()} does not register {name}")
    before = custom_ops._existing_schema(name)

    @custom_ops.custom_op(name, mutates_args=())
    def _late_copy(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.empty(input.shape[0], weight.shape[0])

    assert custom_ops._existing_schema(name) == before
