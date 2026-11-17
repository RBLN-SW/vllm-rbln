# Copyright 2025 Rebellions Inc. All rights reserved.
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

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--attn-impl", choices=["flash_attn", "eager"], default="flash_attn"
    )

    args = parser.parse_args()

    # verifications
    if args.enable_chunked_prefill:
        assert args.max_num_batched_tokens is not None, (
            "max-num-batched-tokens is required "
        )
        "when enable-chunked-prefill is true"

    return args


def compile_optimum_rbln_model(args: argparse.Namespace) -> None:
    from optimum.rbln import RBLNAutoModelForCausalLM

    print(f"Compiling model {args.model} with the following arguments:")
    print(f"  {args.max_num_seqs=}")
    print(f"  {args.max_model_len=}")
    print(f"  {args.block_size=}")
    print(f"  {args.enable_chunked_prefill=}")
    print(f"  {args.max_num_batched_tokens=}")
    print(f"  {args.attn_impl=}")
    print(f"  {args.output_dir=}")
    model = RBLNAutoModelForCausalLM.from_pretrained(
        args.model,
        export=True,
        rbln_create_runtimes=False,
        rbln_batch_size=args.max_num_seqs,
        rbln_tensor_parallel_size=args.tensor_parallel_size,
        rbln_max_seq_len=args.max_model_len,
        rbln_kvcache_partition_len=args.block_size
        if args.attn_impl == "flash_attn"
        else None,
        rbln_kvcache_block_size=args.block_size,
        rbln_prefill_chunk_size=args.max_num_batched_tokens
        if args.enable_chunked_prefill
        else None,
        rbln_attn_impl=args.attn_impl,
    )

    model.save_pretrained(args.output_dir)
    print(f"Compiled model saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    compile_optimum_rbln_model(args)
