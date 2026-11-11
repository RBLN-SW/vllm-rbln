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
"""min_p inside the sampling graph.

A token survives min_p when its probability is at least `min_p` times the row's
highest one. In logits that is `logit >= max_logit + log(min_p)`, which needs no
softmax. A row without min_p carries `log(0) = -inf` and keeps every token.
"""

import numpy as np
import torch
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata

_DROP = -1e4


def graph_log_min_p(
    sampling_metadata: SamplingMetadata, num_rows: int, device: torch.device
) -> torch.Tensor | None:
    """log(min_p) per row for the sampling graph, or None when the step keeps the
    eager argmax-invariant processors."""
    active = [
        p
        for p in sampling_metadata.logitsprocs.argmax_invariant
        if getattr(p, "min_p_count", 1)
    ]
    if len(active) != 1 or not isinstance(active[0], MinPLogitsProcessor):
        return None
    proc = active[0]
    n = min(proc.min_p.shape[0], num_rows)
    min_p = proc.min_p_cpu[:n]
    out = np.full(num_rows, -np.inf, np.float32)
    on = min_p > 0
    out[:n][on] = np.log(min_p[on])
    return torch.from_numpy(out).to(device)


def apply_min_p(logits: torch.Tensor, log_min_p: torch.Tensor) -> torch.Tensor:
    # Compare the gap to the max, not the logit to max + log(min_p): the gap
    # rounds relative to itself, while a threshold near the max rounds by ~6%.
    gap = logits - logits.amax(dim=-1, keepdim=True)
    keep = gap >= log_min_p.to(logits.dtype).unsqueeze(1)
    # An additive mask, not where(keep, logits, -inf): the latter fails to tile
    # next to top_k_top_p at some batch sizes. Softmax still gives 0 there.
    return logits + torch.where(keep, 0.0, _DROP).to(logits.dtype)
