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

"""Base classes for the optimum smoke suite.

A smoke test compiles a model through vLLM's optimum-rbln path -- one
``LLM(model=...)`` call = compile + inference -- and asserts it runs.

Models are shrunk *in memory* (no on-disk checkpoint), the same way
``optimum-rbln/tests/test_llm.py`` cuts ``num_hidden_layers``: vLLM's
``hf_overrides`` mutates the HF config, and vllm-rbln forwards that same object
to optimum as ``config=hf_config`` at compile time, so optimum builds the
reduced model and loads only the matching weights from the *real* checkpoint
(extra layer keys are ignored -- no ``ignore_mismatched_sizes`` needed). Using
the real checkpoint also keeps the correct architecture variant, tokenizer and
processor, avoiding the quirks of ``hf-internal-testing`` tiny randoms.

Declare ``HF_OVERRIDES`` as a flat/dotted dict of config edits, e.g.
``{"text_config.num_hidden_layers": 1}``; the base wraps it in a callable that
applies the edits defensively (vLLM first probes the callable on a bare dummy
config to detect ``model_type``, so absent sub-configs are skipped). Leave it
``None`` to run the model full-size.

Structure mirrors optimum-rbln's test suite: base ``TestCase`` classes are
*nested* inside plain namespace classes so pytest does not collect them (it
collects any module-level ``unittest.TestCase`` subclass regardless of name,
but not nested ones). Concrete per-model classes live in ``test_rsd{1,4,8}.py``.

Requires a real RBLN NPU; ``requires_npu`` skips the whole suite otherwise so
the mocked converter unit tests (``tests/optimum_compile/converter``) still run
on a non-NPU runner.
"""

from __future__ import annotations

import os
import unittest
from collections.abc import Callable
from typing import Any

import pytest
import torch
from PIL import Image
from vllm import LLM, SamplingParams

try:
    import rebel

    _NPU_NAME = rebel.get_npu_name()
except Exception:  # noqa: BLE001 - rebel import/NPU probe may fail without hw
    _NPU_NAME = None

# Attach as ``pytestmark = requires_npu`` at module level in each test_rsd*.py.
requires_npu = pytest.mark.skipif(_NPU_NAME is None, reason="requires a real RBLN NPU")

# Bundled image for multimodal inputs -- keeps the default hermetic (no
# external dataset download). Repo asset; subclasses may override get_image()
# to pull a dataset sample instead.
_ASSET_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "assets",
    "vllm-rbln-white.png",
)


def _make_hf_overrides(dotted: dict[str, Any]) -> Callable:
    """Wrap a flat/dotted config-override dict in a vLLM ``hf_overrides``
    callable. Missing sub-configs are skipped so the callable survives vLLM's
    dummy-config probe (``transformers_utils/config.py``)."""

    def apply(config):
        for key, value in dotted.items():
            obj = config
            *heads, last = key.split(".")
            for head in heads:
                obj = getattr(obj, head, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, last):
                setattr(obj, last, value)
        return config

    return apply


class SmokeBase:
    """Namespace (not collected). Holds the root smoke ``TestCase``."""

    class Model(unittest.TestCase):
        # --- declared by concrete subclasses -------------------------------
        MODEL_ID: str | None = None
        # Flat/dotted config edits applied via vLLM hf_overrides, or None to
        # run full-size. E.g. {"text_config.num_hidden_layers": 1}.
        HF_OVERRIDES: dict | None = None
        NUM_DEVICES = 1  # -> VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK
        LLM_KWARGS: dict = {}  # block_size / max_model_len / max_num_seqs / runner ...
        # -------------------------------------------------------------------
        # Populated in setUpClass (declared here so mypy sees them).
        llm: Any = None
        model_path: str = ""

        @classmethod
        def setUpClass(cls) -> None:
            if _NPU_NAME is None:
                raise unittest.SkipTest("requires a real RBLN NPU")
            assert cls.MODEL_ID, f"{cls.__name__}: MODEL_ID required"
            os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = str(cls.NUM_DEVICES)
            kwargs = dict(cls.LLM_KWARGS)
            if cls.HF_OVERRIDES:
                kwargs["hf_overrides"] = _make_hf_overrides(cls.HF_OVERRIDES)
            cls.model_path = cls.MODEL_ID
            cls.llm = LLM(model=cls.MODEL_ID, **kwargs)

        @classmethod
        def tearDownClass(cls) -> None:
            if getattr(cls, "llm", None) is not None:
                del cls.llm
                cls.llm = None


class DecoderSmoke:
    """Namespace (not collected)."""

    class Test(SmokeBase.Model):
        PROMPTS = [
            "Hello, my name is",
            "The capital of France is",
        ]
        MAX_TOKENS = 20

        def test_smoke(self) -> None:
            outputs = self.llm.generate(
                self.PROMPTS,
                SamplingParams(temperature=0, max_tokens=self.MAX_TOKENS),
            )
            self.assertEqual(len(outputs), len(self.PROMPTS))
            for out in outputs:
                # A reduced model's output is meaningless, but generation must
                # produce at least one token without erroring.
                self.assertGreater(len(out.outputs[0].token_ids), 0)


class PoolingSmoke:
    """Namespace (not collected)."""

    class Test(SmokeBase.Model):
        # query[i] is semantically closest to document[i].
        QUERIES = [
            "What is the capital of China?",
            "How do plants make food?",
        ]
        DOCUMENTS = [
            "The capital of China is Beijing.",
            "Photosynthesis lets plants convert sunlight into chemical energy.",
        ]
        # These models are kept full-size, so assert real ranking.
        ASSERT_RANKING = True

        def test_ranking(self) -> None:
            outputs = self.llm.embed(self.QUERIES + self.DOCUMENTS)
            emb = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs])
            n = len(self.QUERIES)
            scores = emb[:n] @ emb[n:].T
            print(f"scores: {scores.tolist()}")
            if self.ASSERT_RANKING:
                best = scores.argmax(dim=1)
                self.assertTrue(
                    torch.equal(best, torch.arange(n)),
                    f"ranking failed: argmax={best.tolist()} expected diagonal",
                )


class MultimodalSmoke:
    """Namespace (not collected)."""

    class Test(SmokeBase.Model):
        NUM_DEVICES = 1  # tiny multimodal models: single device
        MAX_TOKENS = 20
        # Prompt text placed after the image.
        PROMPT = "What is shown in this image?"
        USE_CHAT_TEMPLATE = True
        PROCESSOR_KWARGS: dict = {}
        # Passed through to the vLLM request as ``mm_processor_kwargs`` -- this
        # is what controls vLLM's *actual* image preprocessing (e.g. min/max
        # pixels), and hence the vision-token count.
        MM_PROCESSOR_KWARGS: dict = {}

        def get_image(self) -> Image.Image:
            """Hermetic default: a bundled repo asset. Override to use a
            dataset sample (e.g. lmms-lab/llava-bench-in-the-wild)."""
            return Image.open(_ASSET_IMAGE).convert("RGB")

        def get_inputs(self) -> list[dict]:
            """Build the vLLM multimodal request(s): a prompt string plus the
            image in ``multi_modal_data``. Chat models format the prompt via the
            processor's chat template; others use PROMPT verbatim."""
            image = self.get_image()
            if self.USE_CHAT_TEMPLATE:
                from transformers import AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    self.MODEL_ID, **self.PROCESSOR_KWARGS
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": self.PROMPT},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt = self.PROMPT
            request = {"prompt": prompt, "multi_modal_data": {"image": image}}
            if self.MM_PROCESSOR_KWARGS:
                request["mm_processor_kwargs"] = self.MM_PROCESSOR_KWARGS
            return [request]

        def test_smoke(self) -> None:
            outputs = self.llm.generate(
                self.get_inputs(),
                SamplingParams(temperature=0, max_tokens=self.MAX_TOKENS),
            )
            self.assertGreaterEqual(len(outputs), 1)
            for out in outputs:
                self.assertGreater(len(out.outputs[0].token_ids), 0)
