# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Minimal reproducer: ``new_full(-inf)`` + scatter under RBLN torch.compile.

Eager semantics:

    logits = lm_head(h)                            # [N, D]  draft-vocab logits
    y = logits.new_full((N, V), -inf)              # [N, V]  every slot is -inf
    y[:, targets] = logits                         # fill D slots at fixed idx
    # => un-scattered slots stay -inf; argmax picks the top real value.

Observed under ``torch.compile(backend="rbln")``:

    * bfloat16: un-scattered slots become NaN. argmax then returns the first
      NaN index, not the top-scoring real token. SILENTLY WRONG.
    * float32:  un-scattered slots become a large finite garbage value (e.g.
      -8.58e9), not -inf. argmax is coincidentally correct here because the
      garbage is negative enough, but the `-inf` invariant is still broken.
    * float16:  behaves as specified; -inf is preserved.

This mirrors upstream ``Eagle3LlamaForCausalLM.compute_logits``, which
scatters draft-vocab logits into a full target-vocab tensor using this exact
``new_full(-inf)`` pattern. Under RBLN compile the bug collapses eagle3
speculative decoding to a single constant draft token (the first NaN index),
driving acceptance to 0%.

Workaround in vllm-rbln: return draft-vocab logits from the compiled graph
and do the draft->target index remap in eager
(see ``vllm_rbln/v1/spec_decode/eagle.py::_argmax_draft_logits``).

Run with::

    python scripts/repro/rbln_compile_neg_inf_fill.py
"""

from __future__ import annotations

import os

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")
os.environ.setdefault("VLLM_RBLN_COMPILE_STRICT_MODE", "1")

import torch
import torch.nn as nn
import rebel  # noqa: F401  # registers the "rbln" torch.compile backend
from rebel import CompileContext


N = 2
HIDDEN = 64
DRAFT_VOCAB = 32
VOCAB_SIZE = 128
DEVICE = torch.device("cpu")


class Eagle3ComputeLogits(nn.Module):
    """Stripped mirror of Eagle3LlamaForCausalLM.compute_logits."""

    def __init__(self, hidden: int, draft_vocab: int, vocab: int, dtype: torch.dtype):
        super().__init__()
        self.vocab = vocab
        self.lm_head = nn.Linear(hidden, draft_vocab, bias=False, dtype=dtype)
        # Captured as a buffer so the compiler sees it as a constant, like the
        # checkpoint's draft_id_to_target_id map.
        d2t = torch.zeros(draft_vocab, dtype=torch.long)
        d2t[10:] = torch.arange(1, draft_vocab - 10 + 1) * 3
        self.register_buffer("targets", torch.arange(draft_vocab) + d2t)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(h)                                          # [N, D]
        y = logits.new_full((logits.shape[0], self.vocab), float("-inf"))
        y[:, self.targets] = logits
        return y


def summarize(tag: str, row: torch.Tensor) -> None:
    print(
        f"  [{tag:<8}] "
        f"finite={torch.isfinite(row).sum().item():4d} "
        f"nan={torch.isnan(row).sum().item():4d} "
        f"neg_inf={(row == float('-inf')).sum().item():4d} "
        f"argmax={row.argmax().item()}"
    )


def run(dtype: torch.dtype) -> None:
    print(f"=== dtype={dtype} ===")
    torch.manual_seed(0)
    model = Eagle3ComputeLogits(HIDDEN, DRAFT_VOCAB, VOCAB_SIZE, dtype).to(DEVICE)
    h = torch.randn(N, HIDDEN, dtype=dtype, device=DEVICE)

    with torch.inference_mode():
        eager_out = model(h)
    summarize("eager", eager_out[0])

    compile_context = CompileContext(use_weight_sharing=True)
    compiled = torch.compile(
        model,
        backend="rbln",
        options={
            "compile_context": compile_context,
            "tensor_parallel_size": 1,
            "process_group_dict": {},
            "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            "mode": "strict",
        },
        dynamic=False,
    )
    with torch.inference_mode():
        compiled_out = compiled(h)
    summarize("compiled", compiled_out[0])

    un_scattered = sorted(set(range(VOCAB_SIZE)) - set(model.targets.tolist()))
    if un_scattered:
        idx = un_scattered[0]
        print(
            f"  un-scattered slot {idx}: "
            f"eager={eager_out[0, idx].item()} compiled={compiled_out[0, idx].item()}"
        )
    print(
        f"  argmax match: "
        f"eager={eager_out[0].argmax().item()} "
        f"compiled={compiled_out[0].argmax().item()} "
        f"-> {'OK' if eager_out[0].argmax().item() == compiled_out[0].argmax().item() else 'MISMATCH'}"
    )
    print()


if __name__ == "__main__":
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        run(dtype)
