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
            t.copy_(cpu_t)                # write back (direct h2v, no
                                          # intermediate RBLN tensor)

    register_kv_cache_torch_hook(my_hook)

NOTE: prefer ``t.copy_(cpu_t)`` over ``t.copy_(cpu_t.to(t.device))`` for the
write-back. The latter allocates an intermediate RBLN tensor and does *two*
DMAs (cpu→rbln_intermediate, then rbln→rbln target) instead of a single h2v.
For 800 MB / layer this roughly doubles the update-direction cost; see
``guides/results.md`` (2026-05-18 section) for measurements.

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

# Bench hook cap. bench / bench_1block / bench_1block_reused 모드는 매 forward
# 마다 모든 layer 의 host↔device round-trip 을 수행해 total runtime 의 큰 부분
# 을 차지함. 초반 N 회 forward 만 측정/round-trip 하고 그 이후엔 즉시 return
# 해서 전체 실행 시간을 줄이기 위한 cap.
#
# 값 의미:
#   -1 (default): 무제한 — 기존 동작 (매 forward 마다 round-trip + 측정).
#    0          : never — hook 등록만 되고 round-trip 한 번도 안 함.
#    N (>=1)    : step ∈ [0, N) 에서만 round-trip + 측정, step >= N 부터 no-op.
#
# 주의: cap 이 warmup (=_BENCH_WARMUP_FORWARDS) 보다 작거나 같으면 samples 가
# 비어서 BENCH/CUMULATIVE/FINAL summary 가 안 찍힘. 유효 sample 을 얻으려면
# cap > _BENCH_WARMUP_FORWARDS 로 설정해야 함.
_BENCH_MAX_STEPS: int = int(
    os.environ.get("VLLM_RBLN_KV_CACHE_HOOK_BENCH_MAX_STEPS", "-1")
)
_BENCH_CAP_NOTIFIED: bool = False


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


def _bench_capped(step: int) -> bool:
    """True 면 이 step 에서 bench round-trip 을 skip 해야 함."""
    return _BENCH_MAX_STEPS >= 0 and step >= _BENCH_MAX_STEPS


def _maybe_notify_bench_cap(step: int, mode_label: str) -> None:
    """Cap 진입 시점에 한 번만 로그를 찍어 사용자가 동작 변화를 인지하도록.
    cap 직후 summary 도 같이 dump — 짧은 run 에서 atexit 까지 못 갈 때 대비."""
    global _BENCH_CAP_NOTIFIED
    if _BENCH_CAP_NOTIFIED:
        return
    if _BENCH_MAX_STEPS >= 0 and step == _BENCH_MAX_STEPS:
        print(
            f"******************** [kv_cache_hook][{mode_label}] bench "
            f"cap reached at step={step} "
            f"(VLLM_RBLN_KV_CACHE_HOOK_BENCH_MAX_STEPS="
            f"{_BENCH_MAX_STEPS}). Subsequent forwards skip round-trip.",
            flush=True,
        )
        s = torch_bench_summary()
        if s:
            print("******************** CAPPED/LAYER " + s, flush=True)
        _BENCH_CAP_NOTIFIED = True


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


def _torch_bench_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """per-layer round-trip latency timer. no mutation."""
    if _bench_capped(step):
        _maybe_notify_bench_cap(step, "BENCH")
        return
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



# ---------- bench_1block: K+V 1-block round-trip (runtime 등가 비교) -----


def _torch_bench_1block_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """첫 block (K + V) 만 round-trip — runtime hook 의 `_fetch_kv_cache(...,
    block_idx=0, ...)` 호출과 등가 DMA 수 (K 1 block + V 1 block = 2 DMA).

    layer view shape `(2, num_blocks, n_head, 1, block_size, head_dim)` 에서
    K = `t[0, 0:1]`, V = `t[1, 0:1]` 슬라이스. 두 슬라이스 각각 fast path
    (is_direct_copy) 통과해야 runtime 과 동등 비교가 됨.

    Update direction 은 1-copy 패턴 (`slice.copy_(cpu_slice)`) 사용 — 즉 기존
    K/V slice 에 직접 h2v.
    """
    if _bench_capped(step):
        _maybe_notify_bench_cap(step, "BENCH_1BLOCK")
        return
    warmup = step < _BENCH_WARMUP_FORWARDS
    if step == 0 and kv_caches:
        t = kv_caches[0]
        k = t[0, 0:1]
        v = t[1, 0:1]
        print(
            f"******************** [kv_cache_hook][BENCH_1BLOCK/info] step=0 "
            f"phase={phase} num_layers={len(kv_caches)} "
            f"layer shape={tuple(t.shape)} stride={t.stride()} "
            f"dtype={t.dtype} device={t.device} | "
            f"k_slice shape={tuple(k.shape)} contig={k.is_contiguous()} "
            f"nbytes={k.numel() * k.element_size()} | "
            f"v_slice shape={tuple(v.shape)} contig={v.is_contiguous()} "
            f"nbytes={v.numel() * v.element_size()}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    for t in kv_caches:
        k_slice = t[0, 0:1]
        v_slice = t[1, 0:1]
        t0 = time.perf_counter_ns()
        cpu_k = k_slice.to("cpu")
        cpu_v = v_slice.to("cpu")
        t1 = time.perf_counter_ns()
        k_slice.copy_(cpu_k)
        v_slice.copy_(cpu_v)
        t2 = time.perf_counter_ns()
        if not warmup:
            nbytes = (
                k_slice.numel() * k_slice.element_size()
                + v_slice.numel() * v_slice.element_size()
            )
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, 2))
            f_total += t1 - t0
            u_total += t2 - t1
    if not warmup and n_layers > 0:
        nbytes = (
            kv_caches[0][0, 0:1].numel() * kv_caches[0].element_size()
            + kv_caches[0][1, 0:1].numel() * kv_caches[0].element_size()
        )
        print(
            f"******************** [kv_cache_hook][BENCH_1BLOCK] step={step} "
            f"phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={nbytes/1e6:.2f}MB calls/layer=(2,2)",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
            s = torch_bench_summary()
            if s:
                print("******************** CUMULATIVE/LAYER " + s, flush=True)


# ---------- bench_1block_reused: 1-block + reused host buffer ------------

_PINNED_HOSTS_1BLOCK_K: List[torch.Tensor] = []
_PINNED_HOSTS_1BLOCK_V: List[torch.Tensor] = []


def _torch_bench_1block_reused_hook(
    kv_caches: Sequence[torch.Tensor], phase: str, step: int
) -> None:
    """1-block round-trip + reused CPU host buffer (K, V 각각 1번씩 pre-alloc).

    Fair vs runtime `bench_1block_reused` — 양쪽 다:
      - 1 block 분만 transfer (K + V = 2 DMA / direction)
      - host buffer 한 번 alloc 후 forward 마다 reuse
      - 매 step 마다 fetch + update 1 layer 당 2 호출씩.
    """
    global _PINNED_HOSTS_1BLOCK_K, _PINNED_HOSTS_1BLOCK_V
    if _bench_capped(step):
        _maybe_notify_bench_cap(step, "BENCH_1BLOCK_REUSED")
        return
    warmup = step < _BENCH_WARMUP_FORWARDS
    if not _PINNED_HOSTS_1BLOCK_K and kv_caches:
        for t in kv_caches:
            k = t[0, 0:1]
            v = t[1, 0:1]
            _PINNED_HOSTS_1BLOCK_K.append(
                torch.empty(k.shape, dtype=t.dtype, device="cpu")
            )
            _PINNED_HOSTS_1BLOCK_V.append(
                torch.empty(v.shape, dtype=t.dtype, device="cpu")
            )
        t = kv_caches[0]
        k = t[0, 0:1]
        v = t[1, 0:1]
        h_k = _PINNED_HOSTS_1BLOCK_K[0]
        print(
            f"******************** [kv_cache_hook][BENCH_1BLOCK_REUSED/info] "
            f"step=0 phase={phase} num_layers={len(kv_caches)} "
            f"layer shape={tuple(t.shape)} stride={t.stride()} | "
            f"k_slice shape={tuple(k.shape)} contig={k.is_contiguous()} "
            f"nbytes={k.numel() * k.element_size()} | "
            f"host_k shape={tuple(h_k.shape)} contig={h_k.is_contiguous()} "
            f"4KB_aligned={h_k.data_ptr() % 0x1000 == 0}",
            flush=True,
        )
    f_total = u_total = 0
    n_layers = len(kv_caches)
    # 첫 정식 step (== _BENCH_WARMUP_FORWARDS) 의 layer 별 timing 을 모아 dump
    # — forward queue contention 가설 검증용 (첫 layer 만 큰지 분포 확인).
    dump_layerwise = (step == _BENCH_WARMUP_FORWARDS)
    layer_fetch_ns = []
    layer_update_ns = []
    for i, t in enumerate(kv_caches):
        k_slice = t[0, 0:1]
        v_slice = t[1, 0:1]
        host_k = _PINNED_HOSTS_1BLOCK_K[i]
        host_v = _PINNED_HOSTS_1BLOCK_V[i]
        t0 = time.perf_counter_ns()
        host_k.copy_(k_slice)
        host_v.copy_(v_slice)
        t1 = time.perf_counter_ns()
        k_slice.copy_(host_k)
        v_slice.copy_(host_v)
        t2 = time.perf_counter_ns()
        if dump_layerwise:
            layer_fetch_ns.append(t1 - t0)
            layer_update_ns.append(t2 - t1)
        if not warmup:
            nbytes = (
                k_slice.numel() * k_slice.element_size()
                + v_slice.numel() * v_slice.element_size()
            )
            _BENCH_SAMPLES.append((t1 - t0, t2 - t1, nbytes, 2))
            f_total += t1 - t0
            u_total += t2 - t1
    if dump_layerwise:
        f_strs = ",".join(f"{n/1e3:.0f}" for n in layer_fetch_ns)
        u_strs = ",".join(f"{n/1e3:.0f}" for n in layer_update_ns)
        print(
            f"******************** [kv_cache_hook][BENCH_1BLOCK_REUSED] "
            f"step={step} phase={phase} per_layer LAYERWISE (µs)\n"
            f"  fetch  : [{f_strs}]\n"
            f"  update : [{u_strs}]",
            flush=True,
        )
    if not warmup and n_layers > 0:
        nbytes = (
            kv_caches[0][0, 0:1].numel() * kv_caches[0].element_size()
            + kv_caches[0][1, 0:1].numel() * kv_caches[0].element_size()
        )
        print(
            f"******************** [kv_cache_hook][BENCH_1BLOCK_REUSED] "
            f"step={step} phase={phase} per_layer "
            f"fetch_mean={f_total/n_layers/1e3:.2f}µs "
            f"update_mean={u_total/n_layers/1e3:.2f}µs "
            f"bytes={nbytes/1e6:.2f}MB calls/layer=(2,2)",
            flush=True,
        )
        if (step - _BENCH_WARMUP_FORWARDS) % 10 == 0:
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
elif _MODE == "bench_1block":
    _HOOK = _torch_bench_1block_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_1block → K+V "
        "1-block slice round-trip (runtime _fetch/_update 와 동등 DMA 수)",
        flush=True,
    )
elif _MODE == "bench_1block_reused":
    _HOOK = _torch_bench_1block_reused_hook
    print(
        "******************** [kv_cache_hook] MODE=bench_1block_reused → "
        "K+V 1-block slice + pre-allocated CPU host buffer (reuse)",
        flush=True,
    )


# Register an atexit hook so bench summary is printed at process end if
# any benchmark mode was active.
if _MODE in ("bench", "bench_1block", "bench_1block_reused"):
    import atexit

    def _print_bench_summary() -> None:
        s = torch_bench_summary()
        if s:
            print("******************** FINAL/LAYER " + s, flush=True)

    atexit.register(_print_bench_summary)
