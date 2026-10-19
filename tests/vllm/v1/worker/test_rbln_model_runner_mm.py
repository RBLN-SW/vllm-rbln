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

"""Which models take the runner's multimodal path, decided at model load."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import ModelConfig

pytestmark = pytest.mark.maybe_use_device


class _InGraphMergeModel(torch.nn.Module):
    merges_mm_embeds_in_graph = True


class TestResolveMultiModalSupport:
    @staticmethod
    def _multimodal(monkeypatch, value: bool) -> None:
        monkeypatch.setattr(
            ModelConfig, "supports_multimodal_inputs", property(lambda self: value)
        )

    def test_a_text_model_stays_off_the_path(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        self._multimodal(monkeypatch, False)
        runner.model = _InGraphMergeModel()

        assert runner._resolve_mm_support() is False

    def test_a_model_that_merges_in_graph_opts_in(self, make_model_runner, monkeypatch):
        runner = make_model_runner()
        self._multimodal(monkeypatch, True)
        runner.model = _InGraphMergeModel()

        assert runner._resolve_mm_support() is True

    def test_any_other_multimodal_model_is_refused(
        self, make_model_runner, monkeypatch
    ):
        # Upstream multimodal models take `mm_embeds` / `mm_mask` through
        # **kwargs and ignore them; refusing beats answering an image as text.
        runner = make_model_runner()
        self._multimodal(monkeypatch, True)
        runner.model = SimpleNamespace(embed_multimodal=lambda **kw: ())

        with pytest.raises(ValueError, match="SupportsRBLNMultiModal"):
            runner._resolve_mm_support()

    def test_the_path_is_off_until_the_model_is_loaded(self, make_model_runner):
        assert make_model_runner().supports_mm_inputs is False
