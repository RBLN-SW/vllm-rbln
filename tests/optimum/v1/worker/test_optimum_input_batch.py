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

"""How the optimum path's padded sampling metadata reaches the RBLN sampling ops.

`RBLNInputBatch` pads metadata to a bucket size instead of `num_reqs`, so
`build_op_top_k_top_p` sees rows that belong to no request. Those rows must stay
valid: the fused kernel uses `top_k` as an index, and a garbage value there
faults instead of producing a discarded token.
"""

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_rbln.v1.sample.ops.top_k_top_p import (
    GREEDY_TOP_K,
    build_op_top_k_top_p,
)
from vllm_rbln.v1.worker.optimum_input_batch import RBLNInputBatch

DEVICE = torch.device("cpu")
VOCAB_SIZE = 8
BUCKET_SIZE = 4
TOP_K = 3


def build_input_batch(requests: list[tuple[str, SamplingParams]]) -> RBLNInputBatch:
    input_batch = RBLNInputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=DEVICE,
        vocab_size=VOCAB_SIZE,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
        use_rbln_sampler=True,
    )
    for req_id, params in requests:
        input_batch.add_request(
            CachedRequestState(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=None,
                sampling_params=params,
                pooling_params=None,
                generator=None,
                block_ids=([0, 1],),
                num_computed_tokens=0,
                output_token_ids=[],
            )
        )
    input_batch.refresh_metadata_rbln(BUCKET_SIZE)
    return input_batch


def test_padded_rows_reach_the_op_with_both_filters_disabled():
    # Two requests in a bucket of four: rows 2 and 3 belong to nobody. The
    # top-k request is what makes vLLM materialize the tensors for the batch.
    input_batch = build_input_batch(
        [
            ("greedy", SamplingParams(temperature=0.0)),
            ("random_top_k", SamplingParams(temperature=1.0, top_k=TOP_K)),
        ]
    )
    metadata = input_batch.sampling_metadata

    # Metadata is padded to the bucket, not cut to `num_reqs`, because the
    # sampler runs on logits padded the same way.
    assert input_batch.num_reqs == 2
    assert metadata.top_k.shape == (BUCKET_SIZE,)
    assert metadata.temperature.shape == (BUCKET_SIZE,)
    # No request uses top-p, so that one arrives as `None` and the fallback has
    # to be built at the bucket length too -- which is why the batch size comes
    # from the logits rather than from whichever tensor metadata happens to hold.
    assert metadata.top_p is None

    top_k, top_p = build_op_top_k_top_p(metadata, BUCKET_SIZE, VOCAB_SIZE, DEVICE)

    # `top_k` covers the whole bucket, the padded rows included; the unused
    # top-p is elided to `None`, which the compiler spells as a scalar 1.0.
    assert top_k.shape == (BUCKET_SIZE,)
    assert top_p is None

    greedy = input_batch.req_id_to_index["greedy"]
    random_top_k = input_batch.req_id_to_index["random_top_k"]
    assert top_k[greedy].item() == GREEDY_TOP_K
    assert top_k[random_top_k].item() == TOP_K
