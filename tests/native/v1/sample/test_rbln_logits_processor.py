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

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate

from vllm_rbln.v1.sample.rbln_logits_processor import RBLNMinPLogitsProcessor

VOCAB = 8


def _min_p_proc(num_reqs: int, min_p: float = 0.5) -> RBLNMinPLogitsProcessor:
    proc = RBLNMinPLogitsProcessor(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=8)),
        torch.device("cpu"),
        False,
    )
    proc.update_state(
        BatchUpdate(
            batch_size=num_reqs,
            removed=[],
            added=[(i, SamplingParams(min_p=min_p), None, []) for i in range(num_reqs)],
            moved=[],
        )
    )
    return proc


def _peaked_logits(num_rows: int) -> torch.Tensor:
    # One dominant token per row, so min_p=0.5 masks every other token.
    logits = torch.zeros(num_rows, VOCAB)
    logits[:, 0] = 10.0
    return logits


class TestMinPRowsMatchLogits:
    def test_decode_pad_rows_are_no_ops(self):
        proc = _min_p_proc(2)
        logits = _peaked_logits(4)

        out = proc.apply(logits)

        assert torch.isinf(out[:2, 1:]).all()
        assert torch.isfinite(out[2:]).all()

    def test_prefill_slices_to_leading_rows(self):
        proc = _min_p_proc(2)

        out = proc.apply(_peaked_logits(1))

        assert torch.isinf(out[0, 1:]).all()
        assert torch.isfinite(out[0, 0])
