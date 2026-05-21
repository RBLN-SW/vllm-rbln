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
"""Runtime-instance-based KV cache access hook.

Sibling of `kv_cache_torch_hook.py`, but instead of moving tensors via
`.to('cpu')` / `.copy_`, it goes through rebel_compiler's runtime instance:

    runtime._fetch_kv_cache(host_t, block_idx, block_offset, size, layer_name)
    runtime._update_kv_cache(host_t, block_idx, block_offset, size, layer_name)

These thunk down to `runtime_base.cc::FetchKVCache / UpdateKVCache` →
`kv_cache.cc::SyncKVCacheBetweenHostAndDevice`.

Env vars:
  VLLM_RBLN_KV_CACHE_RT_HOOK_TRACE=1     — log first call-site reached
  VLLM_RBLN_KV_CACHE_RT_HOOK_MODE=debug  — first-layer/first-block roundtrip + log
                                  =zero  — fetch → zero → update (all layers/blocks)
                                  =random— fetch → randn → update (all layers/blocks)
                                  =bench — fetch → update, no mutation, layer-unit timer
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import torch

# Callback: (runtime, kv_caches_by_name, num_blocks, block_size, phase, step)
KVCacheRuntimeHook = Callable[
    [Any, Dict[str, torch.Tensor], int, int, str, int], None
]

_HOOK: Optional[KVCacheRuntimeHook] = None
_STEP: int = 0
_TRACE_CALLSITE: bool = (
    os.environ.get("VLLM_RBLN_KV_CACHE_RT_HOOK_TRACE", "0").lower()
    in ("1", "true")
)
_CALLSITE_LOGGED: bool = False

# Mutation hook (zero / random) cap.
# 매 forward 마다 모든 layer × 모든 block 을 덮는 비용 (≈layer당 수백 ms × 16
# layer = 수 초 / forward) 이 커서, 첫 N 회 forward 만 mutation 을 실제로 수행
# 하고 그 이후엔 즉시 return 하도록 제한.
#
# 값 의미:
#   -1 (default): 무제한 — 기존 동작 (매 forward 마다 덮어씀).
#    0          : never — hook 등록은 되지만 mutation 한 번도 안 함 (no-op).
#    N (>=1)    : step ∈ [0, N) 에서만 mutation, step >= N 부터 no-op.
#
# 주의: 이 cap 은 mutation 모드 (zero/random) 에만 적용. bench / bench_reused /
# bench_chunked / debug 는 영향 없음.
#
# Parity 관점: zero/random 으로 KV cache 를 덮으면 그 이후 forward 의 attention
# 이 corrupt 된 history 를 read 하므로 일반적으로 parity 가 깨짐. 단:
#   - 첫 prefill 이전 (= step=0 미실행) 까지만 mutation 했다면 parity 통과 가능
#   - 단순히 cap 을 작게 두는 것만으론 parity 통과 보장 안 됨. cap 이후의
#     attention 이 mutation 된 cache 를 read 하지 않는 영역만 사용해야 OK.
#   - parity check 결과로 시각적으로 확인 권장.
_MUTATE_MAX_STEPS: int = int(
    os.environ.get("VLLM_RBLN_KV_CACHE_RT_HOOK_MAX_STEPS", "-1")
)
_MUTATE_CAP_NOTIFIED: bool = False


def _mutation_capped(step: int) -> bool:
    """True 면 이 step 에서 mutation 을 skip 해야 함."""
    return _MUTATE_MAX_STEPS >= 0 and step >= _MUTATE_MAX_STEPS


def _maybe_notify_mutation_cap(step: int, mode_label: str) -> None:
    """Cap 진입 시점에 한 번만 로그를 찍어 사용자가 동작 변화를 인지하도록."""
    global _MUTATE_CAP_NOTIFIED
    if _MUTATE_CAP_NOTIFIED:
        return
    if _MUTATE_MAX_STEPS >= 0 and step == _MUTATE_MAX_STEPS:
        print(
            f"******************** [kv_cache_rt_hook][{mode_label}] mutation "
            f"cap reached at step={step} (VLLM_RBLN_KV_CACHE_RT_HOOK_MAX_STEPS"
            f"={_MUTATE_MAX_STEPS}). Subsequent forwards skip mutation.",
            flush=True,
        )
        _MUTATE_CAP_NOTIFIED = True


def register_kv_cache_runtime_hook(
    hook: Optional[KVCacheRuntimeHook],
) -> None:
    global _HOOK, _STEP
    _HOOK = hook
    _STEP = 0


def get_kv_cache_runtime_hook() -> Optional[KVCacheRuntimeHook]:
    return _HOOK


def run_kv_cache_runtime_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
) -> None:
    """Invoked once per forward by the model runner."""
    global _STEP, _CALLSITE_LOGGED
    hook = _HOOK
    if _TRACE_CALLSITE and not _CALLSITE_LOGGED:
        first_name = next(iter(kv_caches_by_name), None)
        first_t = (
            kv_caches_by_name[first_name] if first_name is not None else None
        )
        meta = (
            f"layer0_name={first_name} shape={tuple(first_t.shape)} "
            f"dtype={first_t.dtype} device={first_t.device}"
            if first_t is not None
            else "<no layers>"
        )
        print(
            "******************** [kv_cache_rt_hook][TRACE] call-site "
            f"reached: phase={phase} num_layers={len(kv_caches_by_name)} "
            f"num_blocks={num_blocks} block_size={block_size} "
            f"hook_registered={hook is not None} "
            f"runtime={type(runtime).__name__} {meta}",
            flush=True,
        )
        _CALLSITE_LOGGED = True
    if hook is None:
        return
    hook(runtime, kv_caches_by_name, num_blocks, block_size, phase, _STEP)
    _STEP += 1


def _make_host_buffer(layer_tensor: torch.Tensor) -> torch.Tensor:
    """layer tensor 전체 byte 분량의 4KB-정렬 CPU host buffer.

    runtime._fetch_kv_cache 가 host pointer를 0x1000-aligned 로 요구하므로
    rebel.kv_cache.aligned_tensor 를 사용 (fp16 단위 alloc 후 layer dtype 으로 view).
    """
    import numpy as np
    from rebel.kv_cache import aligned_tensor

    nbytes = layer_tensor.numel() * layer_tensor.element_size()
    num_fp16 = (nbytes + 1) // 2
    buf = aligned_tensor(
        num_fp16, dtype=np.float16, alignment=0x1000, tensor_type="pt"
    )
    return buf.view(layer_tensor.dtype).reshape(layer_tensor.shape)


def _log_parity_fingerprint(
    source_label: str, step: int, phase: str, layer_t: torch.Tensor
) -> None:
    """Layer 0 의 비교용 fingerprint 출력. 양쪽 path 가 *동일한 포맷*으로
    호출하므로 두 run 의 로그를 grep "PARITY/" 후 시각 비교 가능.

    layer_t 는 CPU 에 있는 bf16/fp16 tensor (layer 0 shape).
    Step 0 의 layer 0 한 번만 호출 권장 (sha1 800 MB 비용 ~1초).
    """
    import hashlib

    t = layer_t.contiguous()
    raw_bytes = t.view(torch.int8).reshape(-1).numpy().tobytes()
    sha = hashlib.sha1(raw_bytes).hexdigest()[:16]
    f32 = t.float()
    flat = t.reshape(-1)
    head = flat[:16].float().tolist()
    tail = flat[-16:].float().tolist()
    print(
        f"******************** [PARITY/{source_label}] step={step} phase={phase} "
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"n_bytes={len(raw_bytes)} sha1={sha} "
        f"min={f32.min().item():+.6f} max={f32.max().item():+.6f} "
        f"mean={f32.mean().item():+.6f}",
        flush=True,
    )
    print(
        f"******************** [PARITY/{source_label}] step={step} "
        f"first16={[f'{v:+.6f}' for v in head]}",
        flush=True,
    )
    print(
        f"******************** [PARITY/{source_label}] step={step} "
        f"last16={[f'{v:+.6f}' for v in tail]}",
        flush=True,
    )


def _runtime_debug_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """진단용. step<3만 출력.

    두 호출 형태를 시도해서 어느 쪽이 동작하는지 확인:
      (1) layer_name=None: 전체 KV cache 한 번에 fetch (block_idx=0)
      (2) layer_name=<first_layer>: 그 layer 의 모든 block 을 fetch 후
          host buffer 를 layer_t.shape 으로 reshape → torch path 와
          동일한 형태로 [PARITY/runtime] fingerprint 출력 (step=0).
    """
    if step >= 3:
        return
    import numpy as np
    from rebel.kv_cache import aligned_tensor

    first_name = next(iter(kv_caches_by_name))
    layer_t = kv_caches_by_name[first_name]

    # ── try (1) layer_name=None ─────────────────────────────────────────
    try:
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._get_kv_cache_size(None) START")
        total_bytes = runtime._get_kv_cache_size(None)
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._get_kv_cache_size(None) DONE")
        num_fp16 = (total_bytes + 1) // 2
        host_all = aligned_tensor(
            num_fp16, dtype=np.float16, alignment=0x1000, tensor_type="pt"
        )
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._fetch_kv_cache(host_all, 0, 0, block_size, None) START")
        runtime._fetch_kv_cache(host_all, 0, 0, block_size, None)
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._fetch_kv_cache(host_all, 0, 0, block_size, None) DONE")
        head = host_all.view(layer_t.dtype)[:8].float().tolist()
        print(
            f"******************** [kv_cache_rt_hook][DBG/all] step={step} "
            f"phase={phase} layer_name=None total_bytes={total_bytes} "
            f"first8={[f'{v:+.4f}' for v in head]}",
            flush=True,
        )
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._update_kv_cache(host_all, 0, 0, block_size, None) START")
        runtime._update_kv_cache(host_all, 0, 0, block_size, None)
        print(f"******************** [kv_cache_rt_hook][DBG/all] runtime._update_kv_cache(host_all, 0, 0, block_size, None) DONE")
    except Exception as e:
        print(
            f"******************** [kv_cache_rt_hook][DBG/all] step={step} "
            f"FAILED layer_name=None err={type(e).__name__}: {e}",
            flush=True,
        )

    # ── try (2) layer_name=first_name: layer 0 전체 fetch + parity ──────
    try:
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._get_kv_cache_size(first_name) START")
        layer_bytes = runtime._get_kv_cache_size(first_name)
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._get_kv_cache_size(first_name) DONE")
        num_fp16 = (layer_bytes + 1) // 2
        host_one = aligned_tensor(
            num_fp16, dtype=np.float16, alignment=0x1000, tensor_type="pt"
        )
        # Layer 0 의 *모든 block* 을 fetch — host_one 에 layer 전체가 채워짐
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._fetch_kv_cache(host_one, 0, 0, block_size, first_name) START")
        for b in range(num_blocks):
            runtime._fetch_kv_cache(host_one, b, 0, block_size, first_name)
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._fetch_kv_cache(host_one, 0, 0, block_size, first_name) DONE")
        head = host_one.view(layer_t.dtype)[:8].float().tolist()
        print(
            f"******************** [kv_cache_rt_hook][DBG/layer] step={step} "
            f"phase={phase} layer={first_name} bytes={layer_bytes} "
            f"num_blocks={num_blocks} first8={[f'{v:+.4f}' for v in head]}",
            flush=True,
        )
        # Parity fingerprint — step 0 의 layer 0 한 번만 (sha1 비용 amortize)
        if step == 0:
            layer_view = host_one.view(layer_t.dtype).reshape(layer_t.shape)
            _log_parity_fingerprint("runtime", step, phase, layer_view)
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._update_kv_cache(host_one, 0, 0, block_size, first_name) START")
        for b in range(num_blocks):
            runtime._update_kv_cache(host_one, b, 0, block_size, first_name)
        print(f"******************** [kv_cache_rt_hook][DBG/layer] runtime._update_kv_cache(host_one, 0, 0, block_size, first_name) DONE")
    except Exception as e:
        print(
            f"******************** [kv_cache_rt_hook][DBG/layer] step={step} "
            f"FAILED layer={first_name} err={type(e).__name__}: {e}",
            flush=True,
        )


def _runtime_zero_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """모든 layer × 모든 block: fetch → zero → update.

    `_MUTATE_MAX_STEPS` cap 으로 초반 N 회 forward 만 실제 mutation 수행.
    """
    if _mutation_capped(step):
        _maybe_notify_mutation_cap(step, "ZERO")
        return
    for name, layer_t in kv_caches_by_name.items():
        host = _make_host_buffer(layer_t)
        for b in range(num_blocks):
            runtime._fetch_kv_cache(host, b, 0, block_size, name)
        host.zero_()
        for b in range(num_blocks):
            runtime._update_kv_cache(host, b, 0, block_size, name)
    if step < 2:
        print(
            f"******************** [kv_cache_rt_hook][ZERO] step={step} "
            f"phase={phase} wiped all {len(kv_caches_by_name)} layers × "
            f"{num_blocks} blocks",
            flush=True,
        )


def _runtime_random_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """모든 layer × 모든 block: fetch → N(0,1) → update.

    `_MUTATE_MAX_STEPS` cap 으로 초반 N 회 forward 만 실제 mutation 수행.
    """
    if _mutation_capped(step):
        _maybe_notify_mutation_cap(step, "RAND")
        return
    for name, layer_t in kv_caches_by_name.items():
        host = _make_host_buffer(layer_t)
        for b in range(num_blocks):
            runtime._fetch_kv_cache(host, b, 0, block_size, name)
        rand_f32 = torch.randn(host.shape, dtype=torch.float32)
        host.copy_(rand_f32.to(host.dtype))
        for b in range(num_blocks):
            runtime._update_kv_cache(host, b, 0, block_size, name)
    if step < 2:
        print(
            f"******************** [kv_cache_rt_hook][RAND] step={step} "
            f"phase={phase} randomized all {len(kv_caches_by_name)} "
            f"layers × {num_blocks} blocks",
            flush=True,
        )


# ---------- bench (no mutation, per-layer timer) ------------------------

# Module-level bench stats — per-layer round-trip timings in nanoseconds.
# Each entry: (fetch_ns, update_ns, bytes, num_calls).
_BENCH_SAMPLES: List[tuple] = []
_BENCH_WARMUP_FORWARDS = 2

# Pre-allocated per-layer host buffers (lazily filled on first forward).
_RT_PINNED_HOSTS: Dict[str, torch.Tensor] = {}


def _runtime_bench_1block_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """첫 block (K + V) 만 round-trip — torch path 의 `bench_1block` 과
    동등 DMA 수. `_fetch_kv_cache(host, 0, 0, block_size, name)` 는 내부에서
    K block + V block = 2 DMA emit. 1 layer 당 fetch 1 호출, update 1 호출.

    Host buffer 는 매번 fresh alloc (`_make_host_buffer` — runtime hook 의
    `bench` 와 일치, fresh-alloc baseline).
    """
    warmup = step < _BENCH_WARMUP_FORWARDS
    if step == 0 and kv_caches_by_name:
        first_name, first_t = next(iter(kv_caches_by_name.items()))
        print(
            f"******************** [kv_cache_rt_hook][BENCH_1BLOCK/info] "
            f"step=0 phase={phase} num_layers={len(kv_caches_by_name)} "
            f"block_size={block_size} "
            f"layer0 name={first_name} shape={tuple(first_t.shape)} "
            f"dtype={first_t.dtype} device={first_t.device}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches_by_name)
    for name, layer_t in kv_caches_by_name.items():
        host = _make_host_buffer(layer_t)
        t0 = time.perf_counter_ns()
        runtime._fetch_kv_cache(host, 0, 0, block_size, name)
        t1 = time.perf_counter_ns()
        runtime._update_kv_cache(host, 0, 0, block_size, name)
        t2 = time.perf_counter_ns()
        if not warmup:
            # K block + V block bytes = layer 전체 bytes / num_blocks
            total_layer_nbytes = layer_t.numel() * layer_t.element_size()
            nb_layer_dim1 = layer_t.shape[1] if layer_t.ndim > 1 else 1
            block_nbytes = total_layer_nbytes // max(nb_layer_dim1, 1)
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, block_nbytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        first_t = next(iter(kv_caches_by_name.values()))
        total_layer_nbytes = first_t.numel() * first_t.element_size()
        nb = first_t.shape[1] if first_t.ndim > 1 else 1
        block_nbytes = total_layer_nbytes // max(nb, 1)
        print(
            f"******************** [kv_cache_rt_hook][BENCH_1BLOCK] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={block_nbytes/1e6:.2f}MB calls/layer=(1,1)",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = runtime_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)
                print("shape=", layer_t.shape)


def _runtime_bench_1block_reused_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """첫 block (K+V) 만 + reused host buffer. Fair vs torch
    bench_1block_reused: 동일 DMA 수 (1 _fetch + 1 _update = 2 DMA per dir),
    host buffer 한 번 alloc 후 reuse.
    """
    global _RT_PINNED_HOSTS
    warmup = step < _BENCH_WARMUP_FORWARDS
    if not _RT_PINNED_HOSTS and kv_caches_by_name:
        for name, layer_t in kv_caches_by_name.items():
            _RT_PINNED_HOSTS[name] = _make_host_buffer(layer_t)
        first_name = next(iter(kv_caches_by_name))
        first_t = kv_caches_by_name[first_name]
        h0 = _RT_PINNED_HOSTS[first_name]
        print(
            f"******************** [kv_cache_rt_hook][BENCH_1BLOCK_REUSED/info] "
            f"step=0 phase={phase} num_layers={len(kv_caches_by_name)} "
            f"block_size={block_size} | "
            f"layer0 shape={tuple(first_t.shape)} dtype={first_t.dtype} | "
            f"host0 shape={tuple(h0.shape)} contig={h0.is_contiguous()} "
            f"4KB_aligned={h0.data_ptr() % 0x1000 == 0}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches_by_name)
    for name, layer_t in kv_caches_by_name.items():
        host = _RT_PINNED_HOSTS[name]
        t0 = time.perf_counter_ns()
        runtime._fetch_kv_cache(host, 0, 0, block_size, name)
        t1 = time.perf_counter_ns()
        runtime._update_kv_cache(host, 0, 0, block_size, name)
        t2 = time.perf_counter_ns()
        if not warmup:
            total_layer_nbytes = layer_t.numel() * layer_t.element_size()
            nb = layer_t.shape[1] if layer_t.ndim > 1 else 1
            block_nbytes = total_layer_nbytes // max(nb, 1)
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, block_nbytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        first_t = next(iter(kv_caches_by_name.values()))
        total_layer_nbytes = first_t.numel() * first_t.element_size()
        nb = first_t.shape[1] if first_t.ndim > 1 else 1
        block_nbytes = total_layer_nbytes // max(nb, 1)
        print(
            f"******************** [kv_cache_rt_hook][BENCH_1BLOCK_REUSED] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={block_nbytes/1e6:.2f}MB calls/layer=(1,1)",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = runtime_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


def _runtime_bench_hook(
    runtime: Any,
    kv_caches_by_name: Dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    phase: str,
    step: int,
) -> None:
    """fetch then update, no mutation. layer-unit timer."""
    warmup = step < _BENCH_WARMUP_FORWARDS
    f_total = u_total = 0
    n_layers = len(kv_caches_by_name)
    for name, layer_t in kv_caches_by_name.items():
        host = _make_host_buffer(layer_t)
        t0 = time.perf_counter_ns()
        for b in range(num_blocks):
            runtime._fetch_kv_cache(host, b, 0, block_size, name)
        t1 = time.perf_counter_ns()
        for b in range(num_blocks):
            runtime._update_kv_cache(host, b, 0, block_size, name)
        t2 = time.perf_counter_ns()
        if not warmup:
            nbytes = layer_t.numel() * layer_t.element_size()
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, num_blocks))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        first_t = next(iter(kv_caches_by_name.values()))
        nbytes = first_t.numel() * first_t.element_size()
        print(
            f"******************** [kv_cache_rt_hook][BENCH] step={step} "
            f"phase={phase} per_layer fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={nbytes/1e6:.2f}MB calls/layer=({num_blocks},{num_blocks})",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = runtime_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


def runtime_bench_summary() -> Optional[str]:
    if not _BENCH_SAMPLES:
        return None
    import statistics

    fetch = [s[0] for s in _BENCH_SAMPLES]
    update = [s[1] for s in _BENCH_SAMPLES]
    total = [f + u for f, u in zip(fetch, update)]
    nbytes = _BENCH_SAMPLES[0][2]  # 같은 layer 형상 가정
    nblocks = _BENCH_SAMPLES[0][3]

    def pct(xs, p):
        s = sorted(xs)
        return s[int(len(s) * p / 100)]

    def mean(xs):
        return sum(xs) / len(xs)

    return (
        f"[kv_cache_rt_hook][BENCH] samples={len(_BENCH_SAMPLES)} "
        f"bytes_per_layer={nbytes / 1e6:.2f}MB "
        f"calls_per_layer=(fetch={nblocks},update={nblocks})\n"
        f"  fetch  µs : mean={mean(fetch)/1e3:9.2f} "
        f"median={statistics.median(fetch)/1e3:9.2f} "
        f"p99={pct(fetch,99)/1e3:9.2f}\n"
        f"  update µs : mean={mean(update)/1e3:9.2f} "
        f"median={statistics.median(update)/1e3:9.2f} "
        f"p99={pct(update,99)/1e3:9.2f}\n"
        f"  total  µs : mean={mean(total)/1e3:9.2f} "
        f"median={statistics.median(total)/1e3:9.2f} "
        f"p99={pct(total,99)/1e3:9.2f}\n"
        f"  GB/s (round-trip): "
        f"{(2 * nbytes) / (mean(total) / 1e9) / 1e9:.3f}"
    )


# ---------- env-driven default install ----------------------------------

_MODE = os.environ.get("VLLM_RBLN_KV_CACHE_RT_HOOK_MODE", "").lower()
if _MODE == "debug":
    _HOOK = _runtime_debug_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=debug → first-layer/"
        "first-block fetch+log+update",
        flush=True,
    )
elif _MODE == "zero":
    _HOOK = _runtime_zero_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=zero → wiping KV "
        "cache to 0 every forward (all layers × all blocks)",
        flush=True,
    )
elif _MODE == "random":
    _HOOK = _runtime_random_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=random → overwriting "
        "KV cache with N(0,1) every forward",
        flush=True,
    )
elif _MODE == "bench":
    _HOOK = _runtime_bench_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=bench → "
        "fetch+update (no mutation) with per-layer timer",
        flush=True,
    )
elif _MODE == "bench_1block":
    _HOOK = _runtime_bench_1block_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=bench_1block → first "
        "block (K+V) only, 1 _fetch + 1 _update / layer (torch bench_1block "
        "와 동등 DMA 수)",
        flush=True,
    )
elif _MODE == "bench_1block_reused":
    _HOOK = _runtime_bench_1block_reused_hook
    print(
        "******************** [kv_cache_rt_hook] MODE=bench_1block_reused → "
        "1-block + reused host buffer (fair vs torch bench_1block_reused)",
        flush=True,
    )


# Register an atexit hook so bench summary is printed at process end if
# benchmark mode was active.
if _MODE in ("bench", "bench_1block", "bench_1block_reused"):
    import atexit

    def _print_bench_summary() -> None:
        s = runtime_bench_summary()
        if s:
            print("******************** FINAL/LAYER " + s, flush=True)

    atexit.register(_print_bench_summary)
