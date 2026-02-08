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
from multiprocessing import get_context

# Set VOCAB_SIZE according to the model's tokenizer vocab size
# Set EPSILON to the acceptable logprob difference threshold
# Set STEP to control the number of tokens to generate
VOCAB_SIZE = 128256
EPSILON = 1e-1 * 5
STEP = 1

llm_args = {
    "model": "meta-llama/Llama-3.2-1B",
    "max_model_len": 40 * 1024,
    "block_size": 1024,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 128,
    "max_num_seqs": 1,
    "max_logprobs": VOCAB_SIZE,
}

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def run_llm(llm, sampling_params, q):
    outputs = llm.generate(prompts, sampling_params)
    q.put(outputs)


def _worker(device, q, llm_args):
    if device == "cpu":
        os.environ["VLLM_PLUGINS"] = "cpu"
        os.environ["VLLM_USE_V1"] = "0"
    elif device == "rbln":
        os.environ.pop("VLLM_PLUGINS", None)
        os.environ["RBLN_KERNEL_MODE"] = "triton"
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["USE_VLLM_MODEL"] = "1"
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        # 1 means disable using compile cache
    else:
        raise ValueError(f"Unknown device: {device}")

    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=0.0,
                                     logprobs=VOCAB_SIZE,
                                     ignore_eos=True,
                                     max_tokens=STEP)
    llm = LLM(**llm_args)
    run_llm(llm, sampling_params, q)


if __name__ == "__main__":
    ctx = get_context("spawn")
    q = ctx.Queue()
    p1 = ctx.Process(target=_worker, args=("cpu", q, llm_args))
    p1.start()
    cpu_outputs = q.get()
    p1.join()

    p2 = ctx.Process(target=_worker, args=("rbln", q, llm_args))
    p2.start()
    rbln_outputs = q.get()
    p2.join()

    if p1.exitcode != 0 or p2.exitcode != 0:
        raise SystemExit("One of the processes worked incorrectly.")

    for cpu_output, rbln_output in zip(cpu_outputs, rbln_outputs):
        print("=========" * 10)
        cpu_logprobs = cpu_output.outputs[0].logprobs
        rbln_logprobs = rbln_output.outputs[0].logprobs
        num_outlier = 0
        for cpu_lp_token_id, cpu_lp_score in cpu_logprobs[0].items():
            cpu_logprob = cpu_lp_score.logprob
            if cpu_lp_token_id not in rbln_logprobs[0]:
                continue
            rbln_logprob = rbln_logprobs[0].get(cpu_lp_token_id).logprob
            if abs(cpu_logprob - rbln_logprob) >= EPSILON:
                num_outlier += 1
        print(f"Number of outliers: {num_outlier}")
        print(f"Prompt: {cpu_output.prompt}")
        print(f"Generated text  (CPU): {cpu_output.outputs[0].text}")
        print(f"Generated text (RBLN): {rbln_output.outputs[0].text}")