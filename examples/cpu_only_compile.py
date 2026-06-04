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

"""CPU-only (NPU-less) compile example.

Run this on a host that has *no* RBLN NPU mounted -- e.g. a CI build worker --
to compile every graph ahead of time and write the ``.rbln`` artifacts to the
compile cache. A real NPU host can then reuse them via cache-hit and skip
compilation entirely at serving time.

How it works
------------
``VLLM_RBLN_COMPILE_ONLY=1`` makes the rbln ``torch.compile`` backend compile +
cache each graph while building its runtime on a *dummy* device, so no NPU is
required. It is a torch.compile option, so it only has meaning on the
vLLM-native model path; the optimum-rbln path is not torch.compile-based and
the flag conflicts with it. This example therefore also sets
``VLLM_RBLN_USE_VLLM_MODEL=1`` to run the vLLM model implementation (the
torch.compile pipeline). Three things follow from that:

* ``VLLM_RBLN_USE_VLLM_MODEL=1`` selects the torch.compile path where the
  ``compile_only`` option is injected at every compile site. Leaving it unset
  (the optimum-rbln path) makes ``VLLM_RBLN_COMPILE_ONLY=1`` raise an error.
* The compile cache must stay enabled (this is where the artifacts land), so
  ``VLLM_DISABLE_COMPILE_CACHE=1`` is rejected in this mode.
* Without an NPU, ``rebel.get_npu_name()`` returns ``None`` and the target SOC
  can no longer be auto-detected, so you must set ``RBLN_TARGET_SOC`` (e.g.
  ``RBLN-CA25``) to tell the compiler what to target.

The compiled runtime lives on a dummy device, so generation is intentionally
skipped here -- the only useful output of this run is the populated cache.

Usage
-----
    # Set the SOC you ultimately want to serve on, then run:
    RBLN_TARGET_SOC=RBLN-CA25 python examples/cpu_only_compile.py \
        --model Qwen/Qwen3-0.6B

The artifacts are written under ``$VLLM_CACHE_ROOT/rbln`` (default
``~/.cache/vllm/rbln``). Copy that directory to the NPU host (or share it via a
network mount) and serving there will hit the cache instead of recompiling.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--target-soc",
        type=str,
        default=os.environ.get("RBLN_TARGET_SOC", "RBLN-CA25"),
        help="Target RBLN SOC to compile for (sets RBLN_TARGET_SOC). Required "
        "because no NPU is mounted to auto-detect it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure compile-only mode *before* the engine builds.
    # rbln_envs reads these lazily from os.environ, so setting them here is
    # enough -- you can equally export them in the shell instead.
    os.environ["VLLM_RBLN_COMPILE_ONLY"] = "1"
    # compile-only only applies to the vLLM-native (torch.compile) path, not
    # the optimum-rbln path, so select it explicitly.
    os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"
    os.environ["RBLN_TARGET_SOC"] = args.target_soc
    # The cache is where compiled artifacts are written; it must stay enabled.
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "0")

    # Imported after the env is set so the platform picks up compile-only mode.
    import vllm.envs as vllm_envs
    from vllm import LLM

    cache_dir = os.path.join(vllm_envs.VLLM_CACHE_ROOT, "rbln")
    print(
        f"[cpu-only-compile] target SOC={args.target_soc!r}, "
        f"writing artifacts to {cache_dir!r}"
    )

    # Constructing the LLM loads the model and compiles every graph for the
    # configured shapes, writing each .rbln artifact to the compile cache.
    # No NPU is touched -- the runtime is built on a dummy device.
    LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.9,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    # Generation is deliberately not run: the runtime lives on a dummy device,
    # so any output would be meaningless. The populated cache below is the
    # deliverable -- copy it to a real NPU host to serve via cache-hit.
    print("[cpu-only-compile] compilation finished. Compiled artifacts:")
    if os.path.isdir(cache_dir):
        artifacts = sorted(
            os.path.join(root, f)
            for root, _, files in os.walk(cache_dir)
            for f in files
            if f.endswith(".rbln")
        )
        if artifacts:
            for path in artifacts:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  - {path} ({size_mb:.1f} MiB)")
        else:
            print(f"  (no .rbln files found under {cache_dir})")
    else:
        print(f"  (cache dir {cache_dir} was not created)")


if __name__ == "__main__":
    main()
