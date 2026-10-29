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
from unittest.mock import Mock

import torch
from vllm.sampling_params import SamplingParams

from tests.vllm.v1.sample.utils import (
    add_requests,
    assert_samples_match,
    draw,
    expected_probs,
    make_input_batch,
    make_logits,
    min_margin_logits,
    weighted_token_ids,
)
from vllm_rbln.v1.sample import rbln_sampler as module
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler
from vllm_rbln.v1.worker.optimum_input_batch import RBLNInputBatch

BUCKET = 256
LAUNCHES = 10
WEIGHTED = weighted_token_ids(32).tolist()


def test_padded_bucket_keeps_live_rows_to_their_params(monkeypatch):
    """Neutral params on the bucket padding must not change any live row's
    distribution. The compile is mocked, so rebel's CPU body runs."""
    monkeypatch.setattr(module, "compile", lambda target, **kwargs: target)
    sampler = RBLNSampler(compile_context=Mock())
    weighted_row = make_logits(range(1, 33))
    groups = [
        (
            min_margin_logits(WEIGHTED[20], WEIGHTED[5]),
            SamplingParams(temperature=0.0),
        ),
        (weighted_row, SamplingParams(temperature=1.5, min_p=0.3)),
        (weighted_row, SamplingParams(logit_bias={WEIGHTED[0]: 4.0}, top_k=8)),
    ]
    per_group = 64
    num_reqs = per_group * len(groups)
    assert num_reqs < BUCKET

    params = [p for _, p in groups for _ in range(per_group)]
    input_batch = make_input_batch(BUCKET, cls=RBLNInputBatch, use_rbln_sampler=True)
    add_requests(input_batch, [(str(i), p) for i, p in enumerate(params)])
    input_batch.refresh_metadata_rbln(BUCKET)
    logits = torch.stack(
        [row for row, _ in groups for _ in range(per_group)]
        + [weighted_row] * (BUCKET - num_reqs)
    )
    expected = expected_probs(
        torch.stack([row for row, _ in groups]), [p for _, p in groups]
    )

    samples = draw(sampler, logits, input_batch.sampling_metadata, LAUNCHES)

    for g, probs in enumerate(expected):
        assert_samples_match(samples[:, g * per_group : (g + 1) * per_group], probs)
