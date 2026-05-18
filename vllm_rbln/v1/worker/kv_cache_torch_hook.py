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
"""User hook for inspecting / overwriting KV cache torch tensors between
decode iterations.

Usage (with VLLM_RBLN_USE_DEVICE_TENSOR=1, KV cache lives on device='rbln'):

    from vllm_rbln.v1.worker.kv_cache_torch_hook import (
        register_kv_cache_torch_hook,
    )

    def my_hook(kv_caches, phase, step):
        # kv_caches is a list of per-layer torch.Tensor on device='rbln'.
        for i, t in enumerate(kv_caches):
            cpu_t = t.to("cpu")           # pull to CPU for inspection
            # ... do anything with cpu_t (read or modify) ...
            t.copy_(cpu_t.to(t.device))   # write back to the rbln tensor

    register_kv_cache_torch_hook(my_hook)

The hook fires once per execute_model() call, *after* the forward pass has
populated the KV cache and *before* the next decode step is dispatched, so
modifications made here are visible to the next step.
"""
from __future__ import annotations

import os
import time
from typing import Callable, List, Optional, Sequence

import torch

KVCacheTorchHook = Callable[[Sequence[torch.Tensor], str, int], None]

_HOOK: Optional[KVCacheTorchHook] = None
_STEP: int = 0
# Set to True via env to print the first time the call-site is reached even
# if no hook is registered — useful to confirm the runner path is alive.
_TRACE_CALLSITE: bool = (
    os.environ.get("VLLM_RBLN_KV_CACHE_HOOK_TRACE", "0").lower() in ("1", "true")
)
_CALLSITE_LOGGED: bool = False

# Mutation hook (zero / random) cap.
# 매 forward 마다 모든 layer 를 덮는 비용이 커서, 첫 N 회 forward 만 mutation 을
# 실제 수행하고 그 이후엔 즉시 return 하도록 제한.
#
# 값 의미:
#   -1 (default): 무제한 — 기존 동작 (매 forward 마다 덮어씀).
#    0          : never — hook 등록은 되지만 mutation 한 번도 안 함 (no-op).
#    N (>=1)    : step ∈ [0, N) 에서만 mutation, step >= N 부터 no-op.
#
# 주의: 이 cap 은 mutation 모드 (zero/random) 에만 적용. bench / bench_storage /
# bench_pinned / bench_reused / bench_chunked / bench_base / debug 는 영향 없음.
#
# Parity 관점: zero/random 으로 KV cache 를 덮으면 그 이후 forward 의 attention
# 이 corrupt 된 history 를 read 하므로 일반적으로 parity 가 깨짐. cap 으로 mutation
# 횟수를 줄여도 parity 통과를 자동 보장하진 않음 — 결과 확인 필요.
_MUTATE_MAX_STEPS: int = int(
    os.environ.get("VLLM_RBLN_KV_CACHE_HOOK_MAX_STEPS", "-1")
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
            f"******************** [kv_cache_hook][{mode_label}] mutation "
            f"cap reached at step={step} (VLLM_RBLN_KV_CACHE_HOOK_MAX_STEPS"
            f"={_MUTATE_MAX_STEPS}). Subsequent forwards skip mutation.",
            flush=True,
        )
        _MUTATE_CAP_NOTIFIED = True


def register_kv_cache_torch_hook(hook: Optional[KVCacheTorchHook]) -> None:
    """Register (or clear, by passing None) the global KV-cache torch hook.

    The callable receives (kv_caches, phase, step):
      - kv_caches: list of per-layer KV cache tensors (live device views).
      - phase: "prefill" or "decode".
      - step: monotonically increasing forward-pass counter (0-based).
    """
    global _HOOK, _STEP
    _HOOK = hook
    _STEP = 0


def get_kv_cache_torch_hook() -> Optional[KVCacheTorchHook]:
    return _HOOK


def run_kv_cache_torch_hook(
    kv_caches: Sequence[torch.Tensor], phase: str
) -> None:
    """Invoke the registered hook, if any. Called by the model runner."""
    global _STEP, _CALLSITE_LOGGED
    hook = _HOOK
    if _TRACE_CALLSITE and not _CALLSITE_LOGGED:
        first = kv_caches[0] if kv_caches else None
        meta = (
            f"layer0 shape={tuple(first.shape)} dtype={first.dtype} "
            f"device={first.device}"
            if first is not None
            else "<no layers>"
        )
        print(
            f"******************** [kv_cache_hook][TRACE] call-site reached: "
            f"phase={phase} num_layers={len(kv_caches)} "
            f"hook_registered={hook is not None} {meta}",
            flush=True,
        )
        _CALLSITE_LOGGED = True
    if hook is None:
        return
    hook(kv_caches, phase, _STEP)
    _STEP += 1


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


_PARITY_LOGGED: bool = False


def _torch_debug_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Round-trip every layer through CPU and log a few stats.

    Installed automatically when VLLM_RBLN_KV_CACHE_HOOK_DEBUG=1.
    Logs every step in {0, 1, 2} and then every 50 steps, to avoid spam.

    PARITY 로그: 첫 *non-warmup* forward 의 layer 0 fingerprint 를 한 번만
    출력. Runtime hook 은 warmup 동안 호출되지 않으므로 (_STEP=0 = 첫 정식
    forward), torch path 도 같은 시점에서 비교하기 위해 warmup 동안엔 PARITY
    출력을 건너뛴다.
    """
    global _PARITY_LOGGED
    should_log = step < 3 or step % 50 == 0
    show_values = step < 2  # only the first two steps, to keep output small
    for i, t in enumerate(kv_caches):
        cpu_t = t.to("cpu")
        if should_log and i == 0:
            f32 = cpu_t.float()
            print(
                f"******************** [kv_cache_hook][DBG] step={step} "
                f"phase={phase} num_layers={len(kv_caches)} layer0 "
                f"shape={tuple(cpu_t.shape)} dtype={cpu_t.dtype} "
                f"min={f32.min().item():.4f} "
                f"max={f32.max().item():.4f} "
                f"mean={f32.mean().item():.4f}",
                flush=True,
            )
            if show_values:
                # flatten once (view), slice tiny pieces, only then cast to f32
                flat = cpu_t.reshape(-1)
                head = flat[:8].float().tolist()
                tail = flat[-8:].float().tolist()
                print(
                    "******************** [kv_cache_hook][DBG] step={s} "
                    "layer0 first8={h} last8={ta}".format(
                        s=step,
                        h=[f"{v:+.4f}" for v in head],
                        ta=[f"{v:+.4f}" for v in tail],
                    ),
                    flush=True,
                )
        # Parity fingerprint — first non-warmup forward only (step 조건 무관).
        # Runtime hook 은 warmup 동안 호출 X 라 _STEP=0 가 첫 정식 forward.
        # Torch 도 같은 시점에서 출력해야 데이터 비교 의미 있음.
        if i == 0 and not _PARITY_LOGGED:
            try:
                from vllm_rbln.torch_compile_backend import is_warmup_active
                warmup_now = is_warmup_active()
            except Exception:
                warmup_now = False
            if not warmup_now:
                _log_parity_fingerprint("torch", step, phase, cpu_t)
                _PARITY_LOGGED = True
        t.copy_(cpu_t.to(t.device))


def _log_layer0(tag: str, step: int, phase: str, cpu_t: torch.Tensor) -> None:
    if step >= 3:
        return
    f32 = cpu_t.float()
    flat = cpu_t.reshape(-1)
    head = flat[:8].float().tolist()
    print(
        "******************** [kv_cache_hook][{tag}] step={s} phase={p} "
        "layer0 shape={sh} dtype={dt} min={mn:.4f} max={mx:.4f} "
        "mean={me:.4f} first8={h}".format(
            tag=tag,
            s=step,
            p=phase,
            sh=tuple(cpu_t.shape),
            dt=cpu_t.dtype,
            mn=f32.min().item(),
            mx=f32.max().item(),
            me=f32.mean().item(),
            h=[f"{v:+.4f}" for v in head],
        ),
        flush=True,
    )


def _zero_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Wipe every KV cache layer to 0 on every forward.

    Installed via VLLM_RBLN_KV_CACHE_HOOK_MODE=zero. The whole cache buffer
    is zeroed (CPU side) and copied back to device, so the next attention
    sees no history — parity should break sharply.

    `_MUTATE_MAX_STEPS` cap 으로 초반 N 회 forward 만 실제 mutation 수행.
    """
    if _mutation_capped(step):
        _maybe_notify_mutation_cap(step, "ZERO")
        return
    for i, t in enumerate(kv_caches):
        cpu_t = t.to("cpu")
        if i == 0:
            _log_layer0("ZERO/pre", step, phase, cpu_t)
        cpu_t.zero_()
        if i == 0:
            _log_layer0("ZERO/post", step, phase, cpu_t)
        t.copy_(cpu_t.to(t.device))


def _random_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Overwrite every KV cache layer with N(0, 1) on every forward.

    Installed via VLLM_RBLN_KV_CACHE_HOOK_MODE=random. Different garbage
    each step; useful to confirm parity break is from data corruption,
    not from a specific zero pattern.

    `_MUTATE_MAX_STEPS` cap 으로 초반 N 회 forward 만 실제 mutation 수행.
    """
    if _mutation_capped(step):
        _maybe_notify_mutation_cap(step, "RAND")
        return
    for i, t in enumerate(kv_caches):
        cpu_t = t.to("cpu")
        if i == 0:
            _log_layer0("RAND/pre", step, phase, cpu_t)
        rand_f32 = torch.randn(cpu_t.shape, dtype=torch.float32)
        cpu_t.copy_(rand_f32.to(cpu_t.dtype))
        if i == 0:
            _log_layer0("RAND/post", step, phase, cpu_t)
        t.copy_(cpu_t.to(t.device))


# ---------- bench (no mutation, per-layer timer) ------------------------

# Each entry: (fetch_ns, update_ns, bytes, num_calls)
_BENCH_SAMPLES: List[tuple] = []
_BENCH_WARMUP_FORWARDS = 2

# Reference to runner's kv_cache_bases (contiguous int8 raw storages, set by
# model_runner just before invoking the hook). Empty when dedup disabled.
_BASES: List[torch.Tensor] = []


def set_kv_cache_bases(bases: Sequence[torch.Tensor]) -> None:
    global _BASES
    _BASES = list(bases)


def _torch_bench_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """per-layer round-trip latency timer. no mutation."""
    warmup = step < _BENCH_WARMUP_FORWARDS
    if step == 0 and kv_caches:
        t0 = kv_caches[0]
        s0 = t0.untyped_storage()
        print(
            f"******************** [kv_cache_hook][BENCH/info] step=0 "
            f"phase={phase} num_layers={len(kv_caches)} "
            f"layer0 shape={tuple(t0.shape)} stride={t0.stride()} "
            f"dtype={t0.dtype} device={t0.device} "
            f"contig={t0.is_contiguous()} "
            f"storage_nbytes={s0.nbytes()} "
            f"tensor_nbytes={t0.numel() * t0.element_size()} "
            f"storage_offset={t0.storage_offset()}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    for t in kv_caches:
        print(f"******************** [kv_cache_hook][BENCH] t.shape={t.shape}, t.size={t.numel()}, t.element_size={t.element_size()}, t.nbytes={t.numel() * t.element_size()}")
        # src = t[:, 0:1, :, :, :, :]
        t0 = time.perf_counter_ns()
        cpu_t = t.to("cpu")
        t1 = time.perf_counter_ns()
        t.copy_(cpu_t.to(t.device))
        t2 = time.perf_counter_ns()
        if not warmup:
            nbytes = t.numel() * t.element_size()
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        print(
            f"******************** [kv_cache_hook][BENCH] step={step} "
            f"phase={phase} per_layer fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={(kv_caches[0].numel() * kv_caches[0].element_size())/1e6:.2f}MB",
            flush=True,
        )
        # 매 10 forward 마다 누적 stats 도 같이 출력해서, atexit 가 안 불려도
        # 최신 누적 결과가 로그에 남도록 보장.
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


def torch_bench_summary() -> Optional[str]:
    if not _BENCH_SAMPLES:
        return None
    import statistics

    fetch = [s[0] for s in _BENCH_SAMPLES]
    update = [s[1] for s in _BENCH_SAMPLES]
    total = [f + u for f, u in zip(fetch, update)]
    nbytes = _BENCH_SAMPLES[0][2]
    ncalls = _BENCH_SAMPLES[0][3]

    def pct(xs, p):
        s = sorted(xs)
        return s[int(len(s) * p / 100)]

    def mean(xs):
        return sum(xs) / len(xs)

    return (
        f"[kv_cache_hook][BENCH] samples={len(_BENCH_SAMPLES)} "
        f"bytes_per_layer={nbytes / 1e6:.2f}MB "
        f"calls_per_layer=(fetch={ncalls},update={ncalls})\n"
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


def _wrap_storage_as_int8_1d(
    view_t: torch.Tensor, layer_bytes: int
) -> torch.Tensor:
    """Wrap view_t's underlying storage as a contiguous 1-D int8 tensor,
    narrowed to ``layer_bytes`` from the storage start.

    The layer view comes from a permute/reshape of a contiguous int8 raw
    buffer allocated in ``_allocate_kv_cache_tensors``. The returned tensor
    shares that same storage but is plain contiguous int8, so ``.to('cpu')``
    should hit the ``is_direct_copy()`` fast path in RBLNCopy.cpp.

    NOTE: starts at storage offset 0 (not the view's storage_offset). When
    a single storage backs multiple layers (shared_by case in dedup), this
    measures the *first* layer-sized slice. With dedup disabled in our
    config, each layer has its own storage so this is equivalent to the
    layer's own bytes.
    """
    storage = view_t.untyped_storage()
    nbytes = storage.nbytes()
    size = min(layer_bytes, nbytes)
    base = torch.empty(0, dtype=torch.int8, device=view_t.device)
    base.set_(storage, 0, (nbytes,))
    return base.narrow(0, 0, size)


def _torch_bench_storage_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Per-layer round-trip on the layer view's *underlying storage*,
    wrapped as a contiguous 1-D int8 tensor. No mutation in practice (we
    write back the same bytes we read).

    Goal: confirm/refute the hypothesis that the view path is slow because
    is_direct_copy() in RBLNCopy.cpp rejects non-contig views and falls
    through to a strided→contig re-copy. If the storage-direct path is
    fast (≈runtime path), the bottleneck is on the torch side and the fix
    is to expose a contiguous wrapper to user code.
    """
    warmup = step < _BENCH_WARMUP_FORWARDS
    if step == 0 and kv_caches:
        v = kv_caches[0]
        layer_bytes = v.numel() * v.element_size()
        s = v.untyped_storage()
        flat = _wrap_storage_as_int8_1d(v, layer_bytes)
        print(
            f"******************** [kv_cache_hook][BENCH_STORAGE/info] "
            f"step=0 phase={phase} num_layers={len(kv_caches)} | "
            f"view: shape={tuple(v.shape)} stride={v.stride()} "
            f"dtype={v.dtype} contig={v.is_contiguous()} "
            f"storage_nbytes={s.nbytes()} tensor_nbytes={layer_bytes} "
            f"storage_offset={v.storage_offset()} | "
            f"flat: shape={tuple(flat.shape)} dtype={flat.dtype} "
            f"contig={flat.is_contiguous()} device={flat.device}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    for t in kv_caches:
        layer_bytes = t.numel() * t.element_size()
        flat = _wrap_storage_as_int8_1d(t, layer_bytes)
        t0 = time.perf_counter_ns()
        cpu_t = flat.to("cpu")
        t1 = time.perf_counter_ns()
        flat.copy_(cpu_t.to(flat.device))
        t2 = time.perf_counter_ns()
        if not warmup:
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, layer_bytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        print(
            f"******************** [kv_cache_hook][BENCH_STORAGE] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={(kv_caches[0].numel() * kv_caches[0].element_size())/1e6:.2f}MB",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


# pre-allocated per-layer aligned host buffers; lazily filled on first call.
_PINNED_HOSTS: List[torch.Tensor] = []


def _torch_bench_pinned_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Per-layer round-trip using a *pre-allocated 4KB-aligned* host buffer
    (same primitive as the runtime path), reused across forwards.

    Tests whether the gap vs runtime path is explained by host-side cost
    (alloc/free + page faults + alignment), not the DMA call itself.
    """
    global _PINNED_HOSTS
    warmup = step < _BENCH_WARMUP_FORWARDS
    if not _PINNED_HOSTS and kv_caches:
        import numpy as np
        from rebel.kv_cache import aligned_tensor

        for t in kv_caches:
            nbytes = t.numel() * t.element_size()
            num_fp16 = (nbytes + 1) // 2
            buf = aligned_tensor(
                num_fp16, dtype=np.float16,
                alignment=0x1000, tensor_type="pt",
            )
            _PINNED_HOSTS.append(buf.view(t.dtype).reshape(t.shape))
        h0 = _PINNED_HOSTS[0]
        v0 = kv_caches[0]
        print(
            f"******************** [kv_cache_hook][BENCH_PINNED/info] "
            f"step={step} phase={phase} num_layers={len(kv_caches)} | "
            f"view: shape={tuple(v0.shape)} stride={v0.stride()} "
            f"dtype={v0.dtype} contig={v0.is_contiguous()} device={v0.device} | "
            f"host: shape={tuple(h0.shape)} stride={h0.stride()} "
            f"dtype={h0.dtype} contig={h0.is_contiguous()} "
            f"data_ptr=0x{h0.data_ptr():x} "
            f"4KB_aligned={h0.data_ptr() % 0x1000 == 0}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    for i, t in enumerate(kv_caches):
        host = _PINNED_HOSTS[i]
        t0 = time.perf_counter_ns()
        host.copy_(t)            # device → host (DMA into aligned reused buf)
        t1 = time.perf_counter_ns()
        t.copy_(host)            # host → device
        t2 = time.perf_counter_ns()
        if not warmup:
            nbytes = t.numel() * t.element_size()
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        print(
            f"******************** [kv_cache_hook][BENCH_PINNED] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={(kv_caches[0].numel() * kv_caches[0].element_size())/1e6:.2f}MB",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


# Chunk bench: chunk-단위 sample (layer 전체 cumulative 는 _BENCH_SAMPLES 에 들어감)
# Each entry: (fetch_ns, update_ns, chunk_bytes, num_blocks_in_chunk).
_TORCH_CHUNK_SAMPLES: List[tuple] = []
_BLOCKS_PER_CHUNK = int(
    os.environ.get("VLLM_RBLN_KV_CACHE_BLOCKS_PER_CHUNK", "64")
)


def _torch_bench_chunked_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Per-chunk round-trip: reused host buffer + 64-block slice copy_.

    Layer view shape `(2, num_blocks, n_head, 1, block_size, head_dim)`.
    dim 0 = K/V index. 두 쪽이 메모리상 396 MB gap 으로 분리되어 있어
    한 번에 묶지 못함 → K, V 분리 slice (각각 contig) 로 chunk 당 2 copy_.

    Sample 종류:
      - _TORCH_CHUNK_SAMPLES : chunk 단위 (K+V 합) timing
      - _BENCH_SAMPLES       : layer 전체 (chunk 합) timing, bench_reused 와 같은 형식
    """
    global _PINNED_HOSTS
    warmup = step < _BENCH_WARMUP_FORWARDS
    if not _PINNED_HOSTS and kv_caches:
        for t in kv_caches:
            buf = torch.empty(t.shape, dtype=t.dtype, device="cpu")
            buf.zero_()  # pre-fault all pages
            _PINNED_HOSTS.append(buf)
        v0 = kv_caches[0]
        num_blocks = v0.shape[1]
        h0 = _PINNED_HOSTS[0]
        print(
            f"******************** [kv_cache_hook][BENCH_CHUNKED/info] "
            f"step={step} phase={phase} num_layers={len(kv_caches)} "
            f"num_blocks={num_blocks} blocks_per_chunk={_BLOCKS_PER_CHUNK} | "
            f"view: shape={tuple(v0.shape)} stride={v0.stride()} "
            f"dtype={v0.dtype} contig={v0.is_contiguous()} | "
            f"k_slice[0,:64] contig={v0[0, :_BLOCKS_PER_CHUNK].is_contiguous()} | "
            f"host: shape={tuple(h0.shape)} contig={h0.is_contiguous()} "
            f"4KB_aligned={h0.data_ptr() % 0x1000 == 0}",
            flush=True,
        )
    f_layer_total = u_layer_total = 0
    n_layers = len(kv_caches)
    for i, t in enumerate(kv_caches):
        host = _PINNED_HOSTS[i]
        num_blocks = t.shape[1]
        layer_fetch_ns = layer_update_ns = 0
        for chunk_start in range(0, num_blocks, _BLOCKS_PER_CHUNK):
            chunk_end = min(chunk_start + _BLOCKS_PER_CHUNK, num_blocks)
            n_in_chunk = chunk_end - chunk_start

            k_dev = t[0, chunk_start:chunk_end]
            k_host = host[0, chunk_start:chunk_end]
            v_dev = t[1, chunk_start:chunk_end]
            v_host = host[1, chunk_start:chunk_end]

            t0 = time.perf_counter_ns()
            k_host.copy_(k_dev)
            v_host.copy_(v_dev)
            t1 = time.perf_counter_ns()
            k_dev.copy_(k_host)
            v_dev.copy_(v_host)
            t2 = time.perf_counter_ns()
            if not warmup:
                # bytes per block (K+V) — layer 전체에서 균등 분할.
                bytes_per_block = (
                    t.numel() * t.element_size() // num_blocks
                )
                chunk_bytes = n_in_chunk * bytes_per_block
                _TORCH_CHUNK_SAMPLES.append(
                    (t1 - t0, t2 - t1, chunk_bytes, n_in_chunk)
                )
            layer_fetch_ns += t1 - t0
            layer_update_ns += t2 - t1
        if not warmup:
            nbytes = t.numel() * t.element_size()
            _BENCH_SAMPLES.append(
                (layer_fetch_ns, layer_update_ns, nbytes, 1)
            )
            f_layer_total += layer_fetch_ns
            u_layer_total += layer_update_ns
    if not warmup and n_layers > 0:
        print(
            f"******************** [kv_cache_hook][BENCH_CHUNKED] step={step} "
            f"phase={phase} per_layer "
            f"fetch_mean={f_layer_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_layer_total/n_layers/1e3:.2f}µs "
            f"bytes={(kv_caches[0].numel() * kv_caches[0].element_size())/1e6:.2f}MB",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)
            cs = torch_chunk_summary()
            if cs:
                print("******************** CUMULATIVE/CHUNK\n" + cs, flush=True)


def torch_chunk_summary() -> Optional[str]:
    if not _TORCH_CHUNK_SAMPLES:
        return None
    import statistics

    full = [
        s for s in _TORCH_CHUNK_SAMPLES if s[3] == _BLOCKS_PER_CHUNK
    ]
    leftover = [
        s for s in _TORCH_CHUNK_SAMPLES if s[3] != _BLOCKS_PER_CHUNK
    ]

    def stats(samples, label):
        if not samples:
            return ""
        fetch = [s[0] for s in samples]
        update = [s[1] for s in samples]
        total = [f + u for f, u in zip(fetch, update)]
        n_in_chunk = samples[0][3]
        chunk_bytes = samples[0][2]

        def pct(xs, p):
            s = sorted(xs)
            return s[int(len(s) * p / 100)]

        def mean(xs):
            return sum(xs) / len(xs)

        return (
            f"  [{label}] samples={len(samples)} "
            f"n_blocks_per_chunk={n_in_chunk} "
            f"chunk_bytes={chunk_bytes/1e6:.2f}MB\n"
            f"    fetch  µs : mean={mean(fetch)/1e3:9.2f} "
            f"median={statistics.median(fetch)/1e3:9.2f} "
            f"p99={pct(fetch,99)/1e3:9.2f}\n"
            f"    update µs : mean={mean(update)/1e3:9.2f} "
            f"median={statistics.median(update)/1e3:9.2f} "
            f"p99={pct(update,99)/1e3:9.2f}\n"
            f"    total  µs : mean={mean(total)/1e3:9.2f} "
            f"median={statistics.median(total)/1e3:9.2f} "
            f"p99={pct(total,99)/1e3:9.2f}\n"
            f"    GB/s (round-trip): "
            f"{(2 * chunk_bytes) / (mean(total) / 1e9) / 1e9:.3f}"
        )

    parts = [
        f"[kv_cache_hook][CHUNK] total_chunks={len(_TORCH_CHUNK_SAMPLES)}"
    ]
    parts.append(stats(full, f"full({_BLOCKS_PER_CHUNK} blk)"))
    if leftover:
        parts.append(stats(leftover, f"leftover({leftover[0][3]} blk)"))
    return "\n".join(parts)


def _torch_bench_reused_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Per-layer round-trip using a plain torch.empty CPU buffer, reused
    across forwards. No special alignment / pre-fault.

    Companion to bench_pinned: tests whether the win is from *reuse alone*
    or also requires the 4KB-aligned/pre-faulted aligned_tensor.
    """
    global _PINNED_HOSTS  # reuse the slot, but fill with plain empty
    warmup = step < _BENCH_WARMUP_FORWARDS
    if not _PINNED_HOSTS and kv_caches:
        for t in kv_caches:
            buf = torch.empty(t.shape, dtype=t.dtype, device="cpu")
            _PINNED_HOSTS.append(buf)
        h0 = _PINNED_HOSTS[0]
        v0 = kv_caches[0]
        print(
            f"******************** [kv_cache_hook][BENCH_REUSED/info] "
            f"step={step} phase={phase} num_layers={len(kv_caches)} | "
            f"view: shape={tuple(v0.shape)} contig={v0.is_contiguous()} "
            f"device={v0.device} | "
            f"host: shape={tuple(h0.shape)} dtype={h0.dtype} "
            f"contig={h0.is_contiguous()} "
            f"data_ptr=0x{h0.data_ptr():x} "
            f"4KB_aligned={h0.data_ptr() % 0x1000 == 0}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    for i, t in enumerate(kv_caches):
        host = _PINNED_HOSTS[i]
        t0 = time.perf_counter_ns()
        host.copy_(t)
        t1 = time.perf_counter_ns()
        t.copy_(host)
        t2 = time.perf_counter_ns()
        if not warmup:
            nbytes = t.numel() * t.element_size()
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, 1))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        print(
            f"******************** [kv_cache_hook][BENCH_REUSED] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={(kv_caches[0].numel() * kv_caches[0].element_size())/1e6:.2f}MB",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


def _torch_bench_base_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """Round-trip a *layer-sized slice* of kv_cache_bases per forward.

    base 텐서는 dedup된 1-D int8 raw storage 라서 그 자체로 거대(uniform
    mode 면 모든 layer 합본). 측정 단위는 layer 1개 분량 (≈792 MB) — base 의
    첫 `layer_bytes` 만큼을 narrow() 로 잘라 contiguous slice 로 `.to('cpu')`
    → host→device 복사. RBLNCopy.cpp 의 `is_direct_copy()` fast path 를 타고
    view 기반 bench 에서 보였던 strided→contiguous re-copy 가 없어야 함.
    """
    if not _BASES or not kv_caches:
        if step == _BENCH_WARMUP_FORWARDS:
            print(
                "******************** [kv_cache_hook][BENCH_BASE] "
                "kv_cache_bases empty (dedup disabled) — nothing to bench. "
                "_kv_cache_layer_tensors() 는 self.kv_caches 를 그대로 반환 "
                "→ materialize_kv_cache_view 호출 안 됨",
                flush=True,
            )
        return
    warmup = step < _BENCH_WARMUP_FORWARDS
    layer_bytes = (
        kv_caches[0].numel() * kv_caches[0].element_size()
    )
    base_t = _BASES[0]
    slice_size = min(layer_bytes, base_t.numel())  # base 가 int8 라 numel==bytes
    slice_t = base_t.narrow(0, 0, slice_size)
    if step == 0:
        first_layer = kv_caches[0]
        print(
            f"******************** [kv_cache_hook][BENCH_BASE/info] "
            f"num_bases={len(_BASES)} num_layers={len(kv_caches)} "
            f"base0 shape={tuple(base_t.shape)} dtype={base_t.dtype} "
            f"device={base_t.device} numel={base_t.numel()} "
            f"contig={base_t.is_contiguous()} | "
            f"layer0 shape={tuple(first_layer.shape)} dtype={first_layer.dtype} "
            f"device={first_layer.device} contig={first_layer.is_contiguous()} "
            f"→ dedup ACTIVE, materialize_kv_cache_view 호출됨",
            flush=True,
        )
    t0 = time.perf_counter_ns()
    cpu_t = slice_t.to("cpu")
    t1 = time.perf_counter_ns()
    slice_t.copy_(cpu_t.to(slice_t.device))
    t2 = time.perf_counter_ns()
    if not warmup:
        _BENCH_SAMPLES.append((t1 - t0, t2 - t1, slice_size, 1))
        print(
            f"******************** [kv_cache_hook][BENCH_BASE] step={step} "
            f"phase={phase} slice_bytes={slice_size/1e6:.2f}MB "
            f"fetch={(t1-t0)/1e3:.2f}µs update={(t2-t1)/1e3:.2f}µs",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 5 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


# ---------- env-driven default install ----------------------------------

_MODE = os.environ.get("VLLM_RBLN_KV_CACHE_HOOK_MODE", "").lower()
_DEBUG_ENV = os.environ.get("VLLM_RBLN_KV_CACHE_HOOK_DEBUG", "0").lower() in (
    "1",
    "true",
)

# Mode dispatch — order kept consistent with kv_cache_runtime_hook.py
# (debug → mutation → bench... in increasing complexity).
if _MODE == "debug" or _DEBUG_ENV:
    _HOOK = _torch_debug_hook
    print(
        "******************** [kv_cache_hook] MODE=debug → "
        "round-trip + log per layer (lightweight inspection)",
        flush=True,
    )
elif _MODE == "zero":
    _HOOK = _zero_hook
    print(
        "******************** [kv_cache_hook] MODE=zero → "
        "wiping KV cache to 0 on every forward",
        flush=True,
    )
elif _MODE == "random":
    _HOOK = _random_hook
    print(
        "******************** [kv_cache_hook] MODE=random → "
        "overwriting KV cache with N(0,1) on every forward",
        flush=True,
    )
elif _MODE == "bench":
    _HOOK = _torch_bench_hook
    print(
        "******************** [kv_cache_hook] MODE=bench → per-layer "
        "round-trip latency timer (no mutation)",
        flush=True,
    )
elif _MODE == "bench_reused":
    _HOOK = _torch_bench_reused_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_reused → per-layer "
        "round-trip using plain torch.empty CPU buffer reused across "
        "forwards (alignment-free; isolates reuse factor)",
        flush=True,
    )
elif _MODE == "bench_chunked":
    _HOOK = _torch_bench_chunked_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_chunked → reused "
        f"host buffer + chunk-{_BLOCKS_PER_CHUNK} K/V split copy_ "
        "(per-chunk and per-layer samples both recorded)",
        flush=True,
    )
elif _MODE == "bench_storage":
    _HOOK = _torch_bench_storage_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_storage → per-layer "
        "round-trip on view.untyped_storage() wrapped as 1-D int8 contiguous "
        "(should hit is_direct_copy fast path)",
        flush=True,
    )
elif _MODE == "bench_pinned":
    _HOOK = _torch_bench_pinned_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_pinned → per-layer "
        "round-trip using pre-allocated 4KB-aligned host buffer reused "
        "across forwards (mirrors runtime path's host-side primitive)",
        flush=True,
    )
elif _MODE == "bench_base":
    _HOOK = _torch_bench_base_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_base → base tensor "
        "round-trip (contiguous fast path)",
        flush=True,
    )


# Register an atexit hook so bench summary is printed at process end if
# any benchmark mode was active.
if _MODE in (
    "bench",
    "bench_reused",
    "bench_chunked",
    "bench_storage",
    "bench_pinned",
    "bench_base",
):
    import atexit

    def _print_bench_summary() -> None:
        s = torch_bench_summary()
        if s:
            print("******************** FINAL/LAYER " + s, flush=True)
        cs = torch_chunk_summary()
        if cs:
            print("******************** FINAL/CHUNK\n" + cs, flush=True)

    atexit.register(_print_bench_summary)
