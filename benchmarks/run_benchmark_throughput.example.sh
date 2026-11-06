#!/bin/bash
# This script shows how to run the throughput benchmark with vLLM.
MODEL_ID="meta-llama/Llama-3.2-1B"

# optimum-rbln path
OPTIMUM_RBLN_MODEL_PATH="./.optimum-rbln-cache"
python3 compile_optimum_rbln.py $MODEL_ID --output-dir $OPTIMUM_RBLN_MODEL_PATH \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 128 \
    --attn-impl flash_attn

VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 VLLM_RBLN_USE_VLLM_MODEL=0 \
python3 benchmark_throughput.py \
    --model $OPTIMUM_RBLN_MODEL_PATH \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 1024 \
    --num-prompts 100 \
    --max-model-len 8192 \
    --block-size 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 128 \
    --warmup-requests 1 \
    --output-json opt-rbln-results.json

rm -rf $OPTIMUM_RBLN_MODEL_PATH

# torch.compile path
VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 VLLM_RBLN_USE_VLLM_MODEL=1 VLLM_DISABLE_COMPILE_CACHE=1 \
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

