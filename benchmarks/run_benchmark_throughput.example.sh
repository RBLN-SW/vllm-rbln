#!/bin/bash
# This script shows how to run the throughput benchmark with vLLM.

# torch.compile path
VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 RBLN_KERNEL_MODE=triton USE_VLLM_MODEL=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=0 \
python3 benchmark_throughput.py \
    --model meta-llama/Llama-3.2-1B-instruct \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 124 \
    --num-prompts 100 \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 256 \
    --warmup-requests 1 \
    --random-range-ratio 0.1 \
    --async-engine \
    --output-json torch-compile-results-async-v0.json
