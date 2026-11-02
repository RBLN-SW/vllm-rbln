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

"""Register a custom logits processor for controlled generation on RBLN.

Requests with SamplingParams.extra_args["target_token"] generate only that token;
the other requests in the same batch generate normally. For Llama-3.2-1B,
token IDs 364 and 1101 decode to " '" and " also", respectively.
"""

import torch
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class DummyLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        target_token = (params.extra_args or {}).get("target_token")
        if target_token is not None and not isinstance(target_token, int):
            raise ValueError(
                f"target_token value {target_token} {type(target_token)} is not int"
            )

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        self.req_info: dict[int, int] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(
            self.req_info,
            batch_update,
            lambda params, _, __: (params.extra_args or {}).get("target_token"),
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        cols = torch.tensor(
            list(self.req_info.values()), dtype=torch.long, device=logits.device
        )
        rows = torch.tensor(
            list(self.req_info.keys()), dtype=torch.long, device=logits.device
        )
        values_to_keep = logits[rows, cols].clone()
        logits[rows] = float("-inf")
        logits[rows, cols] = values_to_keep
        return logits


def main() -> None:
    llm = LLM(
        model="meta-llama/Llama-3.2-1B",
        model_impl="vllm",
        block_size=1024,
        max_num_batched_tokens=512,
        logits_processors=[DummyLogitsProcessor],
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params_list = [
        SamplingParams(temperature=0.0, extra_args={"target_token": 364}),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.0, extra_args={"target_token": 1101}),
        SamplingParams(temperature=0.0),
    ]
    outputs = llm.generate(prompts, sampling_params_list)

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
