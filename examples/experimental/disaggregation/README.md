# LMCache Examples for vLLM-RBLN

This directory contains experimental LMCache examples for `vllm-rbln`. The current implementation supports the RBLN KV layout through a repo-local LMCache connector running in CPU compatibility mode.

## What Is Implemented

- `RBLNLMCacheConnectorV1` is the vLLM entry point for LMCache in `vllm-rbln`.
- A repo-local tensor connector converts the RBLN paged KV layout into LMCache's `KV_2LTD` format.
- Runtime patches adapt LMCache's built-in `cuda` / `xpu` assumptions to the current CPU-compatible path.

Current scope:

- single-rank only
- synchronous load/store path
- no MLA
- no layerwise mode
- no blending
- no GPU connector V3

## Examples: `kv_both`

The `kv_both` examples run a single vLLM instance with LMCache enabled. The smoke test script performs a warm request and then a replay request so that LMCache hits can be observed on the second request.

### CPU Backend

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=lmcache-cpu-config.yaml

python run_lmcache_smoke_test.py
```

### Disk Backend

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=lmcache-disk-config.yaml

python run_lmcache_smoke_test.py
```

### Shared Filesystem Backend (`fs://`)

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=lmcache-fs-config.yaml

python run_lmcache_smoke_test.py
```

Relevant files:

- [`run_lmcache_smoke_test.py`](./run_lmcache_smoke_test.py)
- [`lmcache-cpu-config.yaml`](./lmcache-cpu-config.yaml)
- [`lmcache-disk-config.yaml`](./lmcache-disk-config.yaml)
- [`lmcache-fs-config.yaml`](./lmcache-fs-config.yaml)

## PD

In this directory, PD refers to disaggregated prefilling: one instance prepares KV cache during prefill, and another instance consumes it during decode.

### Example 1: Shared Filesystem

The currently available PD-style demo uses a shared filesystem through LMCache `RemoteBackend` with `fs://`, rather than LMCache's direct `PDBackend` transport path.

- The prefill instance runs with `kv_role="kv_producer"`.
- The decode instance runs with `kv_role="kv_consumer"`.
- A lightweight proxy sends a short request to the prefill instance first, then forwards the original request to the decode instance.
- The two instances share KV cache through LMCache `RemoteBackend` with `fs://` storage.
- Both instances use the same LMCache config file: [`lmcache-fs-config.yaml`](./lmcache-fs-config.yaml).

Start the prefill instance:

```bash
export LMCACHE_CONFIG_FILE=lmcache-fs-config.yaml
export PYTHONHASHSEED=0
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_COMPILE_MODEL=0

vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8001 \
  --block-size 1024 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"RBLNLMCacheConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

Start the decode instance:

```bash
export LMCACHE_CONFIG_FILE=lmcache-fs-config.yaml
export PYTHONHASHSEED=0
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_COMPILE_MODEL=0

vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8002 \
  --block-size 1024 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"RBLNLMCacheConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

Start the proxy:

```bash
python disagg_proxy_demo.py \
  --model Qwen/Qwen3-1.7B \
  --prefill localhost:8001 \
  --decode localhost:8002 \
  --port 8000
```

Send a request to the proxy:

```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models.",
    "max_tokens": 32,
    "temperature": 0.0,
    "stream": false
  }'
```

Relevant files:

- [`disagg_proxy_demo.py`](./disagg_proxy_demo.py)
- [`lmcache-fs-config.yaml`](./lmcache-fs-config.yaml)

### Example 2: Mooncake Shared Backend

This example uses Mooncake as the shared backend for the same producer / consumer flow. It still goes through LMCache `RemoteBackend`, but avoids the filesystem-backed `fs://` example.

- Both instances use the same LMCache config file: [`lmcache-mooncake-config.yaml`](./lmcache-mooncake-config.yaml).
- Start Mooncake master before launching vLLM.

Start Mooncake master with the embedded HTTP metadata server first:

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

Start the prefill instance:

```bash
export LMCACHE_CONFIG_FILE=lmcache-mooncake-config.yaml
export PYTHONHASHSEED=0
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_COMPILE_MODEL=0

vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8001 \
  --block-size 1024 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"RBLNLMCacheConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

Start the decode instance:

```bash
export LMCACHE_CONFIG_FILE=lmcache-mooncake-config.yaml
export PYTHONHASHSEED=0
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_COMPILE_MODEL=0

vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8002 \
  --block-size 1024 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"RBLNLMCacheConnectorV1","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

Start the proxy:

```bash
python disagg_proxy_demo.py \
  --model Qwen/Qwen3-1.7B \
  --prefill localhost:8001 \
  --decode localhost:8002 \
  --port 8000
```

Send a request to the proxy:

```bash
curl -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models.",
    "max_tokens": 32,
    "temperature": 0.0,
    "stream": false
  }'
```

Relevant files:

- [`lmcache-mooncake-config.yaml`](./lmcache-mooncake-config.yaml)
- [`disagg_proxy_demo.py`](./disagg_proxy_demo.py)

Direct LMCache `PDBackend` / transport-channel based disaggregated prefilling is reserved for future work.
