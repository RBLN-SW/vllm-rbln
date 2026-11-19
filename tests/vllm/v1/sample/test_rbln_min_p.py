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
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

from vllm_rbln.v1.sample.ops.fused_penalty import active_argmax_invariant
from vllm_rbln.v1.sample.ops.min_p import apply_min_p, graph_log_min_p
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

from .utils import build_input_batch


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
    got = graph_log_min_p(active_argmax_invariant(md), 4, torch.device("cpu"))
    want = torch.tensor([math.log(0.1), -math.inf, math.log(0.5), -math.inf])
    torch.testing.assert_close(got, want)


def test_other_argmax_invariant_processors_keep_the_eager_path():
    other = SimpleNamespace()  # no min_p_count: counted as active
    for md in (_metadata(_min_p_proc([0.1]), other), _metadata(_min_p_proc([0.0]))):
        assert graph_log_min_p(active_argmax_invariant(md), 1, "cpu") is None


def test_keep_all_buffers_live_with_the_sampler(monkeypatch):
    monkeypatch.setattr(
        "vllm_rbln.v1.sample.rbln_sampler.compile_sampler", lambda fn, ctx: fn
    )
    sampler = RBLNSampler()
    other = RBLNSampler()
    with torch.inference_mode():
        k, p = sampler._filled_top_k_top_p(None, None, 4, 128, torch.device("cpu"))
        again_k, again_p = sampler._filled_top_k_top_p(
            None, None, 4, 128, torch.device("cpu")
        )
        other_k, other_p = other._filled_top_k_top_p(
            None, None, 4, 128, torch.device("cpu")
        )
    assert k is again_k and p is again_p
    assert k is not other_k and p is not other_p
    torch.testing.assert_close(k, torch.full((4,), 128, dtype=torch.int32))
    torch.testing.assert_close(p, torch.ones(4))
    refs = weakref.ref(k), weakref.ref(p)
    del sampler, k, p, again_k, again_p
    assert all(ref() is None for ref in refs)


def test_unaligned_vocab_keeps_the_eager_min_p(monkeypatch):
    monkeypatch.setattr(
        "vllm_rbln.v1.sample.rbln_sampler.compile_sampler", lambda fn, ctx: fn
    )
    sampler = RBLNSampler()
    sampler._compiled_min_p_sample = MagicMock()
    top_k_top_p = MagicMock(return_value=(torch.zeros(2), None))
    monkeypatch.setattr(sampler.topk_topp_sampler, "forward", top_k_top_p)
    vocab = 50272  # opt-125m, not a multiple of 128
    params = SamplingParams(temperature=1.0, min_p=0.3)
    batch = build_input_batch([("0", params), ("1", params)], vocab_size=vocab)
    sampler.sample(torch.randn(2, vocab), batch.sampling_metadata)
    sampler._compiled_min_p_sample.assert_not_called()
    top_k_top_p.assert_called_once()
