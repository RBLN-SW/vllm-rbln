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

"""Config-time MLA answers for the vendored models and their EAGLE drafts."""

from types import SimpleNamespace

import pytest
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_rbln.patches.model_arch_config import (
    EagleDraftConvertor,
    KvLoraRankMLAConvertor,
    register_eagle_draft_convertor,
    register_mla_convertor,
)


@pytest.mark.parametrize("kv_lora_rank, expected", [(512, True), (None, False)])
def test_a_vendored_model_is_mla_by_its_kv_lora_rank(kv_lora_rank, expected):
    config = SimpleNamespace(model_type="vendored", kv_lora_rank=kv_lora_rank)

    assert KvLoraRankMLAConvertor(config, config).is_deepseek_mla() is expected


class TestRegisterMlaConvertor:
    def test_registers_and_is_idempotent(self, monkeypatch):
        monkeypatch.delitem(MODEL_ARCH_CONFIG_CONVERTORS, "vendored", raising=False)

        register_mla_convertor("vendored")
        register_mla_convertor("vendored")

        assert MODEL_ARCH_CONFIG_CONVERTORS["vendored"] is KvLoraRankMLAConvertor

    def test_fails_loudly_once_upstream_owns_the_type(self, monkeypatch):
        # Shadowing an upstream convertor would hide that the vendored
        # registration is obsolete; the failure is what gets it removed.
        class UpstreamConvertor(ModelArchConfigConvertorBase):
            pass

        monkeypatch.setitem(MODEL_ARCH_CONFIG_CONVERTORS, "vendored", UpstreamConvertor)

        with pytest.raises(RuntimeError, match="upstream vLLM now ships"):
            register_mla_convertor("vendored")


class TestEagleDraftConvertor:
    @staticmethod
    def _draft(target_type: str, **target_attrs) -> SimpleNamespace:
        target = SimpleNamespace(model_type=target_type, **target_attrs)
        return SimpleNamespace(model_type="eagle", model=target)

    def test_follows_the_targets_registered_convertor(self, monkeypatch):
        monkeypatch.setitem(
            MODEL_ARCH_CONFIG_CONVERTORS, "vendored", KvLoraRankMLAConvertor
        )
        draft = self._draft("vendored", kv_lora_rank=512)

        assert EagleDraftConvertor(draft, draft).is_deepseek_mla()

    def test_a_target_without_a_convertor_keeps_upstreams_answer(self, monkeypatch):
        monkeypatch.delitem(MODEL_ARCH_CONFIG_CONVERTORS, "dense", raising=False)
        draft = self._draft("dense", kv_lora_rank=None)

        assert EagleDraftConvertor(draft, draft).is_deepseek_mla() is False

    def test_registration_takes_the_eagle_type(self, monkeypatch):
        monkeypatch.delitem(MODEL_ARCH_CONFIG_CONVERTORS, "eagle", raising=False)

        register_eagle_draft_convertor()

        assert MODEL_ARCH_CONFIG_CONVERTORS["eagle"] is EagleDraftConvertor
