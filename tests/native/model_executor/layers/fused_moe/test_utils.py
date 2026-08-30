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

# get_tokens_mask marks, per DP rank's slot in the multicast buffer, which
# positions hold real tokens vs padding. dp_metadata is faked, so no dist init.

from types import SimpleNamespace

import pytest
import torch

import vllm_rbln.model_executor.layers.fused_moe.utils as utils


def _fake_forward_context(monkeypatch, *, num_tokens_across_dp, max_pad_len):
    dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor(num_tokens_across_dp),
        max_pads_across_dp=torch.zeros(max_pad_len),
    )
    monkeypatch.setattr(
        utils,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=dp_metadata),
    )


class TestGetTokensMask:
    def test_marks_real_vs_padded_positions_per_rank(self, monkeypatch):
        # DP=2, max_pad=4: rank 0 has 3 real tokens, rank 1 has 2. Flattened over
        # both ranks' 4-slot windows (the docstring's worked example).
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3, 2], max_pad_len=4)
        mask = utils.get_tokens_mask(999)  # num_tokens ignored when DP>1
        assert mask.shape == (8, 1)
        assert mask.flatten().tolist() == [1, 1, 1, 0, 1, 1, 0, 0]

    def test_dp1_uses_num_tokens_as_pad_and_is_all_left(self, monkeypatch):
        # DP=1 has no padding: num_tokens is the window length and every position
        # is a real token, so the mask is all `left`.
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3], max_pad_len=0)
        mask = utils.get_tokens_mask(3)
        assert mask.shape == (3, 1)
        assert mask.flatten().tolist() == [1, 1, 1]

    def test_custom_left_right_values(self, monkeypatch):
        # (left=0, right=-inf) turns the mask into an additive logit mask that
        # drives padded positions to -inf before softmax.
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3, 2], max_pad_len=4)
        mask = utils.get_tokens_mask(999, left=0.0, right=float("-inf"))
        vals = mask.flatten().tolist()
        assert vals[:3] == [0.0, 0.0, 0.0]
        assert vals[3] == float("-inf")
        assert vals[6:] == [float("-inf"), float("-inf")]

    def test_dtype_defaults_to_torch_default(self, monkeypatch):
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3, 2], max_pad_len=4)
        assert utils.get_tokens_mask(999).dtype == torch.get_default_dtype()

    def test_dtype_is_honored_so_the_product_stays_narrow(self, monkeypatch):
        # An fp32 mask would promote the bf16 routing weights it multiplies,
        # turning the whole [E, T] tensor into a precision island the compiler
        # runs in dlfp16. The mask must come back in the caller's dtype.
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3, 2], max_pad_len=4)
        mask = utils.get_tokens_mask(999, dtype=torch.bfloat16)
        assert mask.dtype == torch.bfloat16
        assert mask.flatten().tolist() == [1, 1, 1, 0, 1, 1, 0, 0]
        weights = torch.ones(2, 8, dtype=torch.bfloat16)
        assert (weights * mask.transpose(1, 0)).dtype == torch.bfloat16

    def test_dtype_is_honored_for_the_additive_logit_mask(self, monkeypatch):
        # -inf is representable in bf16, so the (0, -inf) form narrows too.
        _fake_forward_context(monkeypatch, num_tokens_across_dp=[3, 2], max_pad_len=4)
        mask = utils.get_tokens_mask(
            999, left=0.0, right=float("-inf"), dtype=torch.bfloat16
        )
        assert mask.dtype == torch.bfloat16
        vals = mask.flatten().tolist()
        assert vals[:3] == [0.0, 0.0, 0.0]
        assert vals[3] == float("-inf")

    def test_missing_dp_metadata_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            utils, "get_forward_context", lambda: SimpleNamespace(dp_metadata=None)
        )
        with pytest.raises(AssertionError):
            utils.get_tokens_mask(4)
