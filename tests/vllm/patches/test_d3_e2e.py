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

"""d3-lite on the vllm model path against the checkpoint's own runtime.

The runtime (``trust_remote_code``) renders each question into prompt tokens and
reads its options from the 255 code logits of the readout at the last token.
vLLM's classify head must give the same option probabilities for the same
tokens. Mirrors upstream's tests/models/multimodal/pooling/test_d3.py, text only.
"""

import pytest
import torch

from tests.vllm.vllm_config import local_weights_path

MODEL = "vllm-sr/d3-lite"
ARCH = "D3ForSequenceClassification"
MAX_LEN = 1024
STATE = (
    "The order arrived damaged yesterday. The customer has a receipt and asks "
    "for a replacement today."
)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults",
        },
    },
    "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Soon", "Today"],
    },
}


def option_probs(logits: torch.Tensor, count: int) -> torch.Tensor:
    return torch.softmax(logits[:count].float(), dim=-1)


@pytest.mark.model_compile
def test_d3_matches_the_checkpoint_runtime(vllm_runner, whole_model: bool) -> None:
    from transformers import AutoModel

    reference = AutoModel.from_pretrained(
        local_weights_path(MODEL), trust_remote_code=True
    )
    runtime = reference.runtime
    text = runtime.prepare(STATE, QUESTIONS)
    keys = text.runnable
    counts = [len(text.questions[key].keys) for key in keys]
    sequences = [text.sequences[key] for key in keys]
    assert max(len(ids) for ids in sequences) <= MAX_LEN
    ref_logits = list(runtime.logits(sequences, counts).cpu())
    del reference, runtime

    # Bidirectional attention: each prompt is read whole, so no chunked prefill
    # and the token budget covers max_model_len.
    with vllm_runner(
        MODEL,
        runner="pooling",
        hf_overrides={"architectures": [ARCH]},
        max_model_len=MAX_LEN,
        block_size=MAX_LEN,
        max_num_batched_tokens=MAX_LEN,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        num_gpu_blocks_override=8,
    ) as vllm_model:
        outputs = vllm_model.llm.classify(
            [{"prompt_token_ids": ids} for ids in sequences], use_tqdm=False
        )
    # The head returns raw code logits.
    vllm_logits = [torch.tensor(output.outputs.probs) for output in outputs]
    assert all(len(logits) == 255 for logits in vllm_logits)

    # A truncated model runs, but its logits mean nothing.
    if whole_model:
        for ref, ours, count in zip(ref_logits, vllm_logits, counts, strict=True):
            torch.testing.assert_close(
                option_probs(ours, count), option_probs(ref, count), atol=0.02, rtol=0
            )
