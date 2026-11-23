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
"""Activation dtype of an FP8 block-scaled checkpoint layer on RBLN.

Read from upstream's `--quantization-config` used as an activation-only
override of the checkpoint method (no `weight` key): `linear` / `moe` set it for
every layer of that kind, `targets` per layer (exact name, `re:` regex or
fnmatch pattern, which replaces `linear` / `moe`), `ignore` keeps a layer bf16.
An unset activation runs W8A16 (bf16 activation, every RBLN NPU);
`fp8_per_block_dynamic` runs W8A8 (dynamic fp8 per (token, 128) group).
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
)

# The activation the RBLN block-fp8 W8A8 kernels quantize to.
RBLN_W8A8_ACTIVATION = kFp8Dynamic128Sym

LayerKind = Literal["linear", "moe"]

# Set around Fp8LinearMethod.create_weights, where the kernel is chosen without
# the layer name. Other block-fp8 paths take no override and run W8A16.
_LINEAR_W8A8: ContextVar[bool] = ContextVar("rbln_linear_w8a8", default=False)


def _specs(args: Any) -> list[Any]:
    specs = [args.linear, args.moe]
    specs += list((args.targets or {}).values())
    return [s for s in specs if s is not None]


def validate_activation_override(args: Any, quantization: str | None) -> None:
    """Refuse a `--quantization-config` RBLN cannot run."""
    if args is None:
        return
    for spec in _specs(args):
        if isinstance(spec, str) or spec.weight is not None:
            raise ValueError(
                "--quantization-config takes only activation overrides of a "
                "pre-quantized checkpoint on RBLN (no `weight`, no online "
                f"quantization shorthand); got {spec}."
            )
        if spec.activation not in (None, RBLN_W8A8_ACTIVATION):
            raise ValueError(
                f"activation={spec.activation} is not supported on RBLN; an FP8 "
                "block checkpoint takes fp8_per_block_dynamic (W8A8) or none "
                "(W8A16)."
            )
    if quantization != "fp8":
        raise ValueError(
            "--quantization-config activation overrides apply to FP8 block "
            f"checkpoints on RBLN; this checkpoint's quantization is {quantization}."
        )


def activation_is_fp8(
    args: Any,
    prefix: str,
    kind: LayerKind,
    packed_modules_mapping: Mapping[str, list[str]] | None = None,
) -> bool:
    """Whether `prefix` runs W8A8 under the activation-override `args`."""
    if args is None:
        return False
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.layers.quantization.online.base import (
        _find_matching_targets,
    )

    fused = packed_modules_mapping or {}
    if should_ignore_layer(
        prefix, ignore=args.ignore, fused_mapping=fused, use_fnmatch=True
    ):
        return False
    if args.targets is not None:
        targets = {pattern: str(spec) for pattern, spec in args.targets.items()}
        matched = _find_matching_targets(prefix, targets, fused)
        spec = args.targets[matched[0]] if matched else None
    else:
        spec = args.linear if kind == "linear" else args.moe
    return spec is not None and spec.activation == RBLN_W8A8_ACTIVATION


def current_quantization_config() -> Any:
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config().model_config.quantization_config


@contextmanager
def linear_w8a8_scope(w8a8: bool):
    token = _LINEAR_W8A8.set(w8a8)
    try:
        yield
    finally:
        _LINEAR_W8A8.reset(token)


def linear_w8a8_in_scope() -> bool:
    return _LINEAR_W8A8.get()
