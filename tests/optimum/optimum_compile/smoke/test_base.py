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
``LLM(model=...)`` call = compile + inference -- and asserts it runs."""

from __future__ import annotations

import contextlib
import gc
import os
import time
import unittest
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from vllm import LLM, SamplingParams

# Seconds to wait after tearing an engine down.
_TEARDOWN_SETTLE_S = 10

# Bundled image for multimodal inputs.
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
                if last == "num_hidden_layers":
                    lt = getattr(obj, "layer_types", None)
                    if isinstance(lt, (list, tuple)) and len(lt) > value:
                        obj.layer_types = list(lt[:value])
        return config

    return apply


class SmokeBase:
    """Namespace (not collected). Holds the root smoke ``TestCase``."""

    class Model(unittest.TestCase):
        # --- declared by concrete subclasses -------------------------------
        MODEL_ID: str | None = None

        HF_OVERRIDES: dict | None = None
        NUM_DEVICES = 1
        LLM_KWARGS: dict = {}  # block_size / max_model_len / max_num_seqs / runner ...
        # -------------------------------------------------------------------
        # Populated in setUpClass (declared here so mypy sees them).
        llm: Any = None
        model_path: str = ""

        @classmethod
        def setUpClass(cls) -> None:
            # Compile once per class (the LLM(...) call is the compile step).
            assert cls.MODEL_ID, f"{cls.__name__}: MODEL_ID required"
            os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = str(cls.NUM_DEVICES)
            kwargs = dict(cls.LLM_KWARGS)
            if cls.HF_OVERRIDES:
                kwargs["hf_overrides"] = _make_hf_overrides(cls.HF_OVERRIDES)
            cls.model_path = cls.MODEL_ID
            cls.llm = LLM(model=cls.MODEL_ID, **kwargs)

        @classmethod
        def tearDownClass(cls) -> None:
            llm = getattr(cls, "llm", None)
            cls.llm = None
            if llm is None:
                return
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
            del llm
            gc.collect()
            time.sleep(_TEARDOWN_SETTLE_S)


class DecoderSmoke:
    """Namespace (not collected)."""

    class Test(SmokeBase.Model):
        PROMPTS = [
            "Hello, my name is",
            "The capital of France is",
        ]
        MAX_TOKENS = 20

        def test_generate(self) -> None:
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
        NUM_DEVICES = 1
        MAX_TOKENS = 20
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

                # setUpClass's LLM(...) already pulled this model -- processor
                # included -- into the HF cache, so resolve from disk. Re-fetching
                # re-issues an etag HEAD per file against the hub for nothing, and
                # that redundant round is a real contributor to CI rate-limit hits.
                processor = AutoProcessor.from_pretrained(
                    self.MODEL_ID, local_files_only=True, **self.PROCESSOR_KWARGS
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

        def test_generate(self) -> None:
            outputs = self.llm.generate(
                self.get_inputs(),
                SamplingParams(temperature=0, max_tokens=self.MAX_TOKENS),
            )
            self.assertGreaterEqual(len(outputs), 1)
            for out in outputs:
                self.assertGreater(len(out.outputs[0].token_ids), 0)
