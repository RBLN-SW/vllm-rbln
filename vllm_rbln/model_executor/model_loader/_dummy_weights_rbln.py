"""RBLN-safe override of vLLM's `initialize_single_dummy_weight`.

Upstream's `initialize_single_dummy_weight` (vllm/model_executor/
model_loader/weight_utils.py) builds the Generator on
``param.data.device`` and then calls ``param.uniform_(low, high,
generator=generator)``. On RBLN this lands the Generator on the rbln
device while ``param`` may still be a CPU staging tensor — torch
raises ``RuntimeError: Expected a 'cpu' device type for generator
but found 'rbln'``.

Mirror the existing TPU branch's CPU-generator detour: build the
Generator on CPU, sample via ``torch.rand`` into a CPU buffer, then
``copy_`` into the target tensor. Skips a syscall but matches the
TPU contract that's already in upstream.

Imported during `register_ops()`; the patch is applied at import time
and is idempotent.
"""

from __future__ import annotations

import torch
from vllm.model_executor.model_loader import weight_utils
from vllm.platforms import current_platform


_orig_initialize_single_dummy_weight = weight_utils.initialize_single_dummy_weight


def _initialize_single_dummy_weight_rbln(
    param: torch.Tensor,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """RBLN-safe replacement that avoids the rbln-Generator/CPU-param
    device mismatch by sampling through a CPU generator + tensor copy.
    Falls back to the upstream implementation on every other platform."""
    if not (current_platform.device_type == "rbln" and torch.is_floating_point(param)):
        _orig_initialize_single_dummy_weight(param, low, high, seed)
        return

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if torch.finfo(param.data.dtype).bits < 16:
        # uniform_ doesn't support < 16-bit dtypes (FP8); detour through fp16.
        dtype = param.data.dtype
        tmp = torch.rand(
            param.shape,
            generator=generator,
            dtype=torch.float16,
            layout=param.layout,
            requires_grad=param.requires_grad,
            device="cpu",
        )
        tmp = ((high - low) * tmp + low).to(dtype)
        param.data.copy_(tmp)
        return

    sample = (high - low) * torch.rand(
        param.shape,
        generator=generator,
        dtype=param.dtype,
        layout=param.layout,
        requires_grad=param.requires_grad,
        device="cpu",
    ) + low
    param.copy_(sample)


# Idempotent install: don't re-wrap if we've already patched.
if weight_utils.initialize_single_dummy_weight is not _initialize_single_dummy_weight_rbln:
    weight_utils.initialize_single_dummy_weight = _initialize_single_dummy_weight_rbln
