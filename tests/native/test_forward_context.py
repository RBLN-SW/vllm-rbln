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

# set_forward_context's RBLN-specific DP gating and context lifecycle. The heavy
# plumbing is faked, but override_forward_context is real so set/restore is genuine.

from types import SimpleNamespace

import pytest
import vllm.forward_context as vfc

import vllm_rbln.forward_context as fc


def _cfg(dp_size: int):
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size))


@pytest.fixture
def captured(monkeypatch):
    """Isolate set_forward_context: fake DP-metadata build, forward-context
    build, and the platform hook, capturing their inputs. override_forward_context
    stays real so the set/restore assertions test actual behavior."""
    calls: dict[str, list] = {"make": [], "additional": [], "create": []}

    def fake_make(parallel_config, num_tokens, num_tokens_across_dp, num_padded_tokens):
        calls["make"].append((num_tokens, num_tokens_across_dp, num_padded_tokens))
        return "DP_META"

    monkeypatch.setattr(fc.RBLNDPMetadata, "make", staticmethod(fake_make))

    def fake_create(attn, vllm_config, dp, additional_kwargs=None):
        calls["create"].append(
            {"attn": attn, "dp": dp, "additional": additional_kwargs}
        )
        return f"CTX({attn})"

    monkeypatch.setattr(fc, "create_forward_context", fake_create)

    def fake_additional(**kw):
        calls["additional"].append(kw)
        return {"echoed": kw}

    monkeypatch.setattr(
        fc.current_platform, "set_additional_forward_context", fake_additional
    )
    monkeypatch.setattr(fc.envs, "VLLM_RBLN_USE_MOE_TOKENS_MASK", False)
    return calls


class TestDpMetadataGating:
    @pytest.mark.parametrize(
        ("dp_size", "moe_mask", "attn", "num_tokens", "expect_dp"),
        [
            (1, False, "ATTN", 8, False),  # neither trigger
            (4, False, "ATTN", 8, True),  # data parallel
            (1, True, "ATTN", 8, True),  # moe tokens mask
            (4, False, None, 8, True),  # only num_tokens present (OR, not AND)
            (4, False, None, None, False),  # DP on, but no attn/num_tokens
        ],
    )
    def test_builds_dp_metadata_only_when_gated(
        self, monkeypatch, captured, dp_size, moe_mask, attn, num_tokens, expect_dp
    ):
        monkeypatch.setattr(fc.envs, "VLLM_RBLN_USE_MOE_TOKENS_MASK", moe_mask)
        with fc.set_forward_context(attn, _cfg(dp_size), num_tokens=num_tokens):
            pass
        dp_arg = captured["create"][0]["dp"]
        if expect_dp:
            assert dp_arg == "DP_META"
            assert len(captured["make"]) == 1
        else:
            assert dp_arg is None
            assert captured["make"] == []


class TestAdditionalForwardContext:
    def test_delegates_kwargs_to_platform_hook(self, captured):
        with fc.set_forward_context("ATTN", _cfg(1), num_tokens=8, kv_bases=123):
            pass
        # the extra kwarg reaches the platform hook, whose result is forwarded
        # into the built forward context.
        assert captured["additional"] == [{"kv_bases": 123}]
        assert captured["create"][0]["additional"] == {"echoed": {"kv_bases": 123}}


class TestContextLifecycle:
    def test_sets_context_during_and_restores_after(self, captured):
        before = vfc._forward_context
        with fc.set_forward_context("ATTN", _cfg(1), num_tokens=8):
            assert vfc._forward_context == "CTX(ATTN)"
        assert vfc._forward_context is before

    def test_restores_context_on_exception(self, captured):
        before = vfc._forward_context
        with (
            pytest.raises(RuntimeError, match="boom"),
            fc.set_forward_context("ATTN", _cfg(1), num_tokens=8),
        ):
            raise RuntimeError("boom")
        assert vfc._forward_context is before
