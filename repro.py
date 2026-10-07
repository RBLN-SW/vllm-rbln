import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_RBLN_COMPILE_STRICT_MODE"] = "1"
os.environ["VLLM_RBLN_USE_VLLM_MODEL"] = "1"

import numpy as np
from vllm import LLM, SamplingParams

def main():
    np.random.seed(42)
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-instruct",
        max_num_seqs=4,
        block_size=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        async_scheduling=False,
    )

    llm.chat([{"role": "user", "content": "Hi, are you conscious?"}], SamplingParams(temperature=1.0, max_tokens=8192, ignore_eos=True),)

if __name__ == "__main__":
    main()
