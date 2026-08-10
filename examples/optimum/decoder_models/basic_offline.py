# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import fire
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1.0)


def main(
    model: str = "meta-llama/Llama-3.2-1B",
    max_num_seqs: int = 1,
    max_model_len: int = 4096,
    block_size: int = None,  # if None, will be set to max_model_len
    num_devices: int = 4,
    use_decoder_batch_sizes: bool = False,
):
    if num_devices is not None:
        os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] = str(num_devices)

    llm = LLM(
        model=model,
        block_size=block_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        additional_config={
            "rbln_config": {
                "decoder_batch_sizes": [1, 8] if use_decoder_batch_sizes else None,
            }
        },
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    fire.Fire(main)
