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
import math
from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

from vllm_rbln.v1.sample.ops.min_p import apply_min_p, graph_log_min_p


def test_apply_min_p_matches_the_probability_rule():
    torch.manual_seed(0)
    logits = torch.randn(8, 4096) * 4
    min_p = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0])
    probs = logits.softmax(-1)
    want = probs < min_p.unsqueeze(1) * probs.amax(-1, keepdim=True)
    log_min_p = torch.where(min_p > 0, min_p.log(), torch.tensor(-math.inf))
    got = apply_min_p(logits, log_min_p)
    assert torch.equal(got < -1000, want)
    assert torch.equal(got[~want], logits[~want])
    assert torch.equal(got.softmax(-1)[want], torch.zeros(int(want.sum())))


def _min_p_proc(values):
    proc = object.__new__(MinPLogitsProcessor)
    proc.min_p_cpu = np.zeros(8, np.float32)
    proc.min_p_cpu[: len(values)] = values
    proc.min_p_count = sum(v > 0 for v in values)
    proc.min_p = torch.tensor(values).unsqueeze(1)
    return proc


def _metadata(*procs):
    return SimpleNamespace(logitsprocs=SimpleNamespace(argmax_invariant=list(procs)))


def test_log_min_p_pads_rows_without_min_p():
    md = _metadata(_min_p_proc([0.1, 0.0, 0.5]))
    got = graph_log_min_p(md, 4, torch.device("cpu"))
    want = torch.tensor([math.log(0.1), -math.inf, math.log(0.5), -math.inf])
    torch.testing.assert_close(got, want)


def test_other_argmax_invariant_processors_keep_the_eager_path():
    other = SimpleNamespace()  # no min_p_count: counted as active
    assert graph_log_min_p(_metadata(_min_p_proc([0.1]), other), 1, "cpu") is None
    assert graph_log_min_p(_metadata(_min_p_proc([0.0])), 1, "cpu") is None
