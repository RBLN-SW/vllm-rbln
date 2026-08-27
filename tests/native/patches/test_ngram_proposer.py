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

import types

from vllm.v1.spec_decode import ngram_proposer as upstream

import vllm_rbln.patches.ngram_proposer as pnp


def _vllm_config(k: int = 2, max_model_len: int = 512) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            prompt_lookup_min=2,
            prompt_lookup_max=5,
            num_speculative_tokens=k,
        ),
        model_config=types.SimpleNamespace(max_model_len=max_model_len),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=types.SimpleNamespace(max_num_seqs=4),
    )


def _spy_kernel(monkeypatch) -> list[tuple]:
    """Replace the numba kernel so the assertions cost no JIT compile."""
    calls: list[tuple] = []

    def fake(valid_ngram_requests, *args, **kwargs):
        calls.append(tuple(valid_ngram_requests))

    monkeypatch.setattr(upstream, "batch_propose_numba", fake)
    return calls


def test_construction_reaches_the_numba_kernel(monkeypatch):
    calls = _spy_kernel(monkeypatch)

    pnp.__init__(upstream.NgramProposer.__new__(upstream.NgramProposer), _vllm_config())

    # Upstream's own warm-up passes only empty sampled_token_ids, which propose()
    # filters out and batch_propose() then skips: it never gets here.
    assert calls, "construction did not reach batch_propose_numba"
    assert calls[0] == (0,)


def test_upstream_warmup_alone_does_not_reach_the_kernel(monkeypatch):
    """Pins the upstream behaviour this patch exists to correct."""
    calls = _spy_kernel(monkeypatch)

    proposer = upstream.NgramProposer.__new__(upstream.NgramProposer)
    pnp._init(proposer, _vllm_config())

    assert not calls
