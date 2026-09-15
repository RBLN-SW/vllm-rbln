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

# all_gather's reshape/permute, with the collective faked at its documented
# rank-major layout and compared against torch.cat as ground truth. The real
# on-device collective belongs to the tensor-parallel e2e.

import pytest
import torch

import vllm_rbln.distributed.rbln_communicator as rc
from vllm_rbln.distributed.rbln_communicator import RblnCommunicator

_WORLD_SIZE = 3


def _make_communicator(monkeypatch, per_rank: list[torch.Tensor]) -> RblnCommunicator:
    """A communicator whose all_gather_into_tensor faithfully reproduces the
    documented layout: the output is the per-rank tensors concatenated along
    dim 0, exactly what a real world_size-rank gather would write."""
    comm = object.__new__(RblnCommunicator)
    comm.world_size = _WORLD_SIZE
    comm.device_group = object()

    def fake_all_gather_into_tensor(output, input_, group=None):
        n = input_.shape[0]
        for rank in range(_WORLD_SIZE):
            output[rank * n : (rank + 1) * n] = per_rank[rank]

    monkeypatch.setattr(rc.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    return comm


def _distinct_ranks(shape: tuple[int, ...]) -> list[torch.Tensor]:
    # Distinct, non-overlapping values per rank so a mis-placed permute is visible.
    numel = 1
    for s in shape:
        numel *= s
    return [
        torch.arange(numel).reshape(shape) + 1000 * rank for rank in range(_WORLD_SIZE)
    ]


class TestAllGatherReassembly:
    @pytest.mark.parametrize(
        ("shape", "dim"),
        [
            ((2, 4), 0),  # concat along the first dim (pass-through gather)
            ((2, 4), -1),  # 2D last dim -> permute(1, 0, 2)
            ((2, 3, 4), -1),  # 3D last dim -> permute(1, 2, 0, 3)
        ],
    )
    def test_matches_concat_reference(self, monkeypatch, shape, dim):
        # The reshape/permute must turn the dim-0 gather into a concat along dim;
        # torch.cat over the per-rank inputs is the ground truth.
        per_rank = _distinct_ranks(shape)
        comm = _make_communicator(monkeypatch, per_rank)
        out = comm.all_gather(per_rank[0], dim=dim)
        expected = torch.cat(per_rank, dim=dim)
        assert out.shape == expected.shape
        assert torch.equal(out, expected)


class TestAllGatherUnsupported:
    @pytest.mark.parametrize("dim", [1, 2])
    def test_explicit_positive_dim_rejected(self, monkeypatch, dim):
        # Only dim 0 and -1 are implemented; an explicit positive dim is NYI.
        per_rank = _distinct_ranks((2, 3, 4))
        comm = _make_communicator(monkeypatch, per_rank)
        with pytest.raises(AssertionError, match="dim!=0, dim!=-1"):
            comm.all_gather(per_rank[0], dim=dim)

    @pytest.mark.parametrize("shape", [(5,), (2, 3, 4, 5)])
    def test_last_dim_only_supports_2d_and_3d(self, monkeypatch, shape):
        # dim=-1 resolves to the last axis; only 2D and 3D inputs have a permute.
        per_rank = _distinct_ranks(shape)
        comm = _make_communicator(monkeypatch, per_rank)
        with pytest.raises(AssertionError, match="move_dim=1, 2"):
            comm.all_gather(per_rank[0], dim=-1)
