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

"""Architecture validation against upstream vLLM.

The model runner rejects architectures upstream vLLM does not know, before
remapping anything for optimum-rbln. Not every architecture vLLM accepts is
*registered*, though: an unregistered ``XForSequenceClassification`` is derived
from ``XForCausalLM`` plus the `classify` conversion, which is how the original
Qwen3-Reranker loads. The check has to follow that rule or it rejects a model
vLLM handles fine.
"""

from types import SimpleNamespace

import pytest

from vllm_rbln.utils.optimum.registry import validate_arch_supported


def _config(architectures, **kwargs):
    return SimpleNamespace(architectures=architectures, **kwargs)


def test_accepts_registered_arch():
    validate_arch_supported(_config(["Qwen3ForCausalLM"]))


def test_accepts_derived_seq_cls_arch():
    """Qwen3ForSequenceClassification is derived, not registered."""
    validate_arch_supported(
        _config(
            ["Qwen3ForSequenceClassification"],
            runner_type="pooling",
            convert_type="classify",
        )
    )


def test_rejects_unknown_arch():
    with pytest.raises(ValueError, match="not supported on upstream"):
        validate_arch_supported(_config(["TotallyMadeUpForCausalLM"]))


def test_rejects_unknown_base_behind_known_suffix():
    """A known suffix must not wave through an unknown base model."""
    with pytest.raises(ValueError, match="not supported on upstream"):
        validate_arch_supported(
            _config(
                ["TotallyMadeUpForSequenceClassification"],
                runner_type="pooling",
                convert_type="classify",
            )
        )
