# Benchmarks

This directory used to contain vLLM-rbln's benchmark scripts and utilities for performance testing and evaluation.

## Usage

<details>
<summary> <h3> Offline Throughput Benchmarks </h3> </summary>
<div>

benchmark offline throughput with random data.

```bash
RBLN_PROFILER=0 RBLN_KERNEL_MODE=triton VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 \
python3 benchmark_throughput.py \
    --model meta-llama/Llama-3.2-1B-instruct \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 124 \
    --num-prompts 100 \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 1024 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 256 \
    --warmup-requests 1 \
    --random-range-ratio 0.1 \
    --output-json result.json
```

</div>
</details>


<details>
<summary> <h3> Online Structure Output Benchmarks </h3> </summary>
<div>
*Server Setup*

```bash
RBLN_KERNEL_MODE=triton VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 \
vllm serve meta-llama/Llama-3.2-1B-instruct \
    --tokenizer meta-llama/Llama-3.2-1B-instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --guided-decoding-backend xgrammar \
    --max-num-seqs 16 \
    --max-model-len 8192 \
    --block-size 1024 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 256
```

XGrammar Benchmark Dataset

```bash
python benchmark_serving_structured_output.py \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --model meta-llama/Llama-3.2-1B-instruct  \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset xgrammar_bench \
    --save-results \
    --result-filename structured_output_result.json \
    --output-len 512 \
    --num-prompts 100 \
    --request-rate 10
```

</div>
</details>
