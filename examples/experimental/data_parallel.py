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

# ruff: noqa
"""
Usage:
Single node:
    - To enable padded decode and MoE tokens mask,
    set the following environment variables:
        VLLM_RBLN_USE_MOE_TOKENS_MASK=1 \
        VLLM_RBLN_DP_IMPL="padded_decode" \
        and other VLLM_RBLN_* environment variables... \
        python examples/experimental/data_parallel.py \
                --model="Qwen/Qwen1.5-MoE-A2.7B" \
                --dp-size=2 \
                --tp-size=2 --ep

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    tp_size,
    enable_ep,
    max_model_len,
    block_size,
    decode_batch,
    num_hidden_layers,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    # paralle_config.data_parallel_size = envs.sVLLM_DP_SIZE
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The vLLM is",
        "The president of the United States is",
        "The future of AI is",
    ] * dp_size

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    prompts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * prompts_per_rank
    end = start + prompts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    sampling_params = SamplingParams(temperature=0.0)

    if num_hidden_layers == 0:
        hf_overrides_kw = None
    else:
        hf_overrides_kw = {
            "num_hidden_layers": num_hidden_layers,
        }

    # Create an LLM.
    llm = LLM(
        model=model,
        hf_overrides=hf_overrides_kw,
        max_model_len=max_model_len,
        block_size=block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        max_num_seqs=decode_batch,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./profile",
        },
        # data_parallel_size=dp_size,
        # enforce_eager=True,
    )
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    # Print the outputs.
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-research/PowerMoE-3b",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--ep", action="store_true", help="vLLM enable_expert_parallel")
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Max sequence length"
    )
    parser.add_argument(
        "--block-size", type=int, default=4096, help="KV cache block size"
    )
    parser.add_argument("--decode-batch", type=int, default=1, help="decode batch size")
    parser.add_argument(
        "--num-hidden-layers", type=int, default=0, help="num hidden layers"
    )
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    args = parser.parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank
    enable_ep = args.ep
    max_model_len = args.max_model_len
    block_size = args.block_size
    decode_batch = args.decode_batch
    num_hidden_layers = args.num_hidden_layers

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                enable_ep,
                max_model_len,
                block_size,
                decode_batch,
                num_hidden_layers,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        # proc.join(timeout=3000)
        # disable timeout
        proc.join()
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
