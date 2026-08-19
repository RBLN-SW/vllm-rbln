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
"""RBLN path for the DFlash drafter's context-KV precomputation.

DFlash drafts a whole block in ONE non-causal forward whose keys/values are not
its own tokens but the TARGET's hidden states, projected and written into the
drafter's KV cache up front. Upstream implements that in
`DFlashQwen3Model.precompute_and_store_context_kv`, and its own docstring says
why it is eager:

    "Since the context shape is different than the query shape, we can't rely on
     the regular forward pass to apply torch.compile and CUDA graphs to this
     section. As such, this function is optimized to minimize the number of
     torch ops present: we use fused vLLM kernels for RMSNorm and RoPE..."

Eager is fine on a GPU and fatal on RBLN. Every eager op crosses `DispatchShim`,
and with `TORCH_RBLN_DEPLOY` off each one additionally drags its inputs to host
for a NaN/Inf scan. The EAGLE3 arm already lost on step cost for exactly this
class of reason (104.2 ms/step vs 52.5 ms for spec-off), so leaving a per-step
projection of the whole context in eager would measure our port rather than
DFlash. This module therefore replaces the method with compiled projection
graphs.

The cache write deliberately stays OUTSIDE those graphs. A compiled stateful
store assigns its own physical-view configuration to the cache input. The
DFlash forward assigns another one, so alternating the two graphs forces
`PrepareInputs` to rematerialise five large cache views on every proposal.
Projection-only graphs return compact K/V tensors; RBLN's native batched
device-to-device strided copy writes them into the existing DFlash cache views
without changing the forward graph's cache ABI.

The drafter mixes attention types (4 sliding_attention + 1 full_attention,
window 2048), and RBLN gives sliding-window layers their own cache geometry: one
block per request sized to the window, indexed through `local_block_tables` and
`cache_seq_lens` rather than the global slot mapping. Upstream hands the same
`context_slot_mapping` to every layer, which is correct on GPU's uniform paging
and wrong here. Slots are therefore resolved per attention group by the caller
(`RBLNDFlashProposer`), which already builds per-group metadata, and passed in
via `set_group_slots`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.rotary_embedding.common import rotate_neox
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_rbln.envs as envs
from vllm_rbln.compilation import compile as rbln_compile
from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

_ORIGINAL_SCHEDULER_INIT = Scheduler.__init__


def _scheduler_cache_drop_patch_needed() -> bool:
    return not hasattr(SpeculativeConfig, "requires_eagle_cache_drop")


@register_patch(
    target="vllm.v1.core.sched.scheduler.Scheduler.__init__",
    reason=(
        "This vLLM release classifies DFlash as EAGLE and consequently drops "
        "the final matching KV block. DFlash projects target hidden states into "
        "its own cache and must retain that block; upstream fixed this by "
        "separating use_eagle from requires_eagle_cache_drop."
    ),
    condition=_scheduler_cache_drop_patch_needed,
)
def patched_scheduler_init(self, vllm_config, *args, **kwargs):
    """Construct a DFlash scheduler without EAGLE's last-block cache drop.

    Older vLLM schedulers derive both lookahead allocation and cache-drop
    behavior from ``use_eagle()``. Present DFlash as a generic draft model only
    while that constructor runs: this preserves lookahead tokens while keeping
    the final cache block. Restore the shared config before returning so model
    runners continue to recognize the DFlash method.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_dflash():
        return _ORIGINAL_SCHEDULER_INIT(self, vllm_config, *args, **kwargs)

    original_method = spec_config.method
    spec_config.method = "draft_model"
    try:
        result = _ORIGINAL_SCHEDULER_INIT(self, vllm_config, *args, **kwargs)
    finally:
        spec_config.method = original_method

    # Keep DFlash's EAGLE execution semantics after constructing only the KV
    # cache manager as a generic draft model.  Newer vLLM releases express this
    # with a separate ``requires_eagle_cache_drop`` flag and also reserve the
    # bonus-token slot.  This older release has neither distinction, so restore
    # the scheduler flag after construction and account for DFlash's
    # [bonus + num_spec] query width explicitly.
    self.use_eagle = True
    self.num_lookahead_tokens = spec_config.num_speculative_tokens + 1
    return result


_TARGET = (
    "vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model"
    ".precompute_and_store_context_kv"
)

_WHY = (
    "Upstream runs the DFlash context-KV projection eagerly because the context "
    "shape differs from the query shape. On RBLN every eager op crosses "
    "DispatchShim, so a per-step projection of the whole context would dominate "
    "the drafter step and make the measurement reflect the port rather than the "
    "method. This replacement compiles the projection math and uses native "
    "device-to-device copies so no second compiled cache layout is installed."
)

_CACHE_PARTITION_SIZE = 1024
_DFLASH_CONTEXT_KV_MAX_RUN_LEN = 506
_DFLASH_CONTEXT_KV_DECODE_BUCKET = 8


def _context_kv_bucket_size(run_len: int) -> int:
    if not 1 <= run_len <= _DFLASH_CONTEXT_KV_MAX_RUN_LEN:
        raise ValueError(
            "DFlash projection run length must be in "
            f"[1, {_DFLASH_CONTEXT_KV_MAX_RUN_LEN}], got {run_len}"
        )
    if run_len <= _DFLASH_CONTEXT_KV_DECODE_BUCKET:
        return _DFLASH_CONTEXT_KV_DECODE_BUCKET
    return _DFLASH_CONTEXT_KV_MAX_RUN_LEN


# The third parameter of `precompute_and_store_context_kv` is upstream's
# `context_slot_mapping`. Here only its None-ness is read -- None means
# "dummy_run: project for profiling, write nothing", anything else means write.
# The coordinates themselves arrive out of band through `set_group_slots`,
# because on RBLN they are per attention group and `cad.slot_mapping` is a 0-dim
# dummy that carries no information at all (DFLASH-PORT-DESIGN.md section 17).
# Callers pass this instead of inventing a tensor whose value looks meaningful.
WRITE_CONTEXT_KV = torch.empty(0)


@dataclass(frozen=True)
class _ContextKVRun:
    """One maximal request-local write inside one physical cache block."""

    token_start: int
    token_count: int
    physical_block_id: int
    block_offset: int
    group_id: int


@dataclass(frozen=True)
class _RuntimeInputBindingPlan:
    """Minimal runtime work for one already-seen input-address profile."""

    can_reuse: bool
    prepare_indices: tuple[int, ...]
    patch_indices: tuple[int, ...]


class _RuntimeInputBindingCache:
    """Tracks which virtual addresses were safely prepared for each input port.

    A shared DFlash context runtime visits five layer-specific weight/cache
    profiles in a fixed cycle.  The normal runtime path must see every address
    once so it can install the port's device-allocation configuration.  After
    that, inputs whose *contents* change still go through ``PrepareInputs``;
    immutable inputs only need an address patch when the layer changes.
    """

    def __init__(self, dynamic_input_indices: tuple[int, ...]) -> None:
        if any(index < 0 for index in dynamic_input_indices):
            raise ValueError("dynamic input indices must be non-negative")
        if len(set(dynamic_input_indices)) != len(dynamic_input_indices):
            raise ValueError("dynamic input indices must be unique")
        self._dynamic = tuple(sorted(dynamic_input_indices))
        self._seen: tuple[set[int], ...] | None = None
        self._last: tuple[int, ...] | None = None

    def observe(self, pointers: tuple[int, ...]) -> None:
        if self._seen is None:
            self._seen = tuple(set() for _ in pointers)
        elif len(pointers) != len(self._seen):
            raise ValueError("runtime input arity changed")
        for seen, pointer in zip(self._seen, pointers):
            seen.add(pointer)
        self._last = pointers

    def plan(self, pointers: tuple[int, ...]) -> _RuntimeInputBindingPlan:
        if (
            self._seen is None
            or self._last is None
            or len(pointers) != len(self._seen)
            or any(pointer not in seen for pointer, seen in zip(pointers, self._seen))
        ):
            return _RuntimeInputBindingPlan(False, (), ())

        dynamic = set(self._dynamic)
        return _RuntimeInputBindingPlan(
            True,
            self._dynamic,
            tuple(
                index
                for index, (pointer, previous) in enumerate(zip(pointers, self._last))
                if index not in dynamic and pointer != previous
            ),
        )

    def execute(
        self,
        runtime,
        pointers: tuple[int, ...],
        get_device_addrs,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> bool:
        """Run using prior bindings; return False when a normal prepare is needed."""
        plan = self.plan(pointers)
        if not plan.can_reuse:
            return False
        if not isinstance(outputs, (list, tuple)):
            return False

        device_outputs: dict[int, int] = {}
        cpu_outputs: dict[int, int] = {}
        for output_index, output in enumerate(outputs):
            if not isinstance(output, torch.Tensor) or not output.is_contiguous():
                return False
            if output.device.type == "rbln":
                device_outputs[output_index] = output.data_ptr()
            elif output.is_cpu:
                cpu_outputs[output_index] = output.data_ptr()
            else:
                return False

        handle = runtime._runtime_handle
        handle.begin_io_patch_batch()
        try:
            handle.prepare_inputs(
                {index: pointers[index] for index in plan.prepare_indices}, {}
            )
            for index in plan.patch_indices:
                try:
                    handle.update_input_addr(index, get_device_addrs(pointers[index]))
                except RuntimeError as error:
                    # A contiguous device tensor can still span more physical
                    # allocations than this compiled input slot can relocate.
                    # Normal PrepareInputs handles that case by materialising a
                    # compatible view and can enlarge the slot profile.
                    if "addrs.size() <= rbln_slot.device_allocs().size()" not in str(
                        error
                    ):
                        raise
                    # PrepareInputs can install a larger relocation profile for
                    # this port. Do not make the fallback permanent: the next
                    # layer transition should retry the cheap address patch
                    # against that newly prepared capacity.
                    handle.prepare_inputs({index: pointers[index]}, {})
                    logger.warning(
                        "DFlash stable input %d requires PrepareInputs: "
                        "device allocation count exceeds relocation capacity",
                        index,
                    )
            # Match DynamoRuntime.run: every manual invocation must patch the
            # cached output addresses and declare their physical views as the
            # values produced by the upcoming device run.
            handle.prepare_outputs(device_outputs, cpu_outputs)
        finally:
            handle.end_io_patch_batch()
        handle.run()
        capture_reports = getattr(runtime, "_capture_reports_if_needed", None)
        if capture_reports is not None:
            capture_reports()
        self.observe(pointers)
        return True


class _StableRuntimeGraph:
    """Install context-KV binding reuse at the ordered runtime boundary.

    Dynamo export may rename and reorder placeholders. In particular, its
    generic ``args_N`` names are not the Python function's flattened argument
    order, so reconstructing the runtime input tuple from those names can patch
    the wrong port. ``DynamoRuntime.run`` is the first boundary that already
    has the executor's authoritative input order; reuse is installed there.
    """

    def __init__(
        self,
        compiled,
        runtime_holder: list,
        *,
        get_device_addrs=None,
        tensor_is_supported=None,
        dynamic_arg_indices: tuple[int, ...],
    ) -> None:
        self._compiled = compiled
        self._runtime_holder = runtime_holder
        self._get_device_addrs = get_device_addrs or self._default_get_device_addrs
        self._tensor_is_supported = (
            tensor_is_supported or self._default_tensor_is_supported
        )
        self._dynamic_arg_indices = dynamic_arg_indices
        self._runtime = None
        self._runtime_call: _StableDynamoRuntimeCall | None = None

    @staticmethod
    def _default_get_device_addrs(pointer: int):
        from rebel import _C  # imported lazily in CPU-only test environments

        return _C.vmem.get_device_addrs(pointer)

    @staticmethod
    def _default_tensor_is_supported(tensor: torch.Tensor) -> bool:
        return (
            tensor.device.type == "rbln"
            and tensor.is_contiguous()
            and tensor.data_ptr() != 0
        )

    def _dynamic_tensors(self, args: tuple) -> tuple[torch.Tensor, ...]:
        return tuple(
            args[index]
            for index in self._dynamic_arg_indices
            if index < len(args) and isinstance(args[index], torch.Tensor)
        )

    def _install_current_runtime(self) -> None:
        if len(self._runtime_holder) != 1:
            return
        runtime = self._runtime_holder[0]
        if runtime is self._runtime:
            return
        call = getattr(runtime, "_rbln_dflash_context_stable_inputs", None)
        if call is None:
            call = _StableDynamoRuntimeCall(
                runtime,
                runtime.run,
                None,
                get_device_addrs=self._get_device_addrs,
                tensor_is_supported=self._tensor_is_supported,
                log_label="context-KV",
            )
            runtime.run = call
            runtime._rbln_dflash_context_stable_inputs = call
            logger.debug(
                "DFlash context-KV runtime boundary installed: inputs=%s",
                sorted(runtime._input_name_to_index.items(), key=lambda item: item[1]),
            )
        self._runtime = runtime
        self._runtime_call = call

    def __call__(self, *args):
        if self._runtime_call is not None:
            self._runtime_call.set_dynamic_inputs(self._dynamic_tensors(args))
        output = self._compiled(*args)
        self._install_current_runtime()
        return output


class _StableDynamoRuntimeCall:
    """Reuse validated context-projection bindings and output allocations.

    Projection profiles share their input staging buffers across attention
    layers.  The wrapper keeps ordinary preparation for dynamic buffers while
    patching already-observed layer-specific weight bindings, without changing
    the compiled graph ABI.
    """

    def __init__(
        self,
        runtime,
        original_run,
        dynamic_input_indices: tuple[int, ...] | None,
        *,
        get_device_addrs,
        tensor_is_supported,
        log_label: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._original_run = original_run
        self._dynamic_input_indices = dynamic_input_indices
        self._dynamic_input_pointers: frozenset[int] = frozenset()
        self._get_device_addrs = get_device_addrs
        self._tensor_is_supported = tensor_is_supported
        self._log_label = log_label
        self._bindings: _RuntimeInputBindingCache | None = None
        self._output = None

    def set_dynamic_inputs(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self._dynamic_input_pointers = frozenset(
            tensor.data_ptr() for tensor in tensors
        )

    def _resolve_dynamic_inputs(self, inputs: list[torch.Tensor]) -> bool:
        if self._dynamic_input_indices is not None:
            return True
        if not self._dynamic_input_pointers:
            return False
        pointers = tuple(tensor.data_ptr() for tensor in inputs)
        dynamic_indices = tuple(
            index
            for index, pointer in enumerate(pointers)
            if pointer in self._dynamic_input_pointers
        )
        # Dynamo may dead-code-eliminate some Python inputs. Only executor
        # inputs need preparation; matching at this ordered boundary remains
        # exact even when not every source-level dynamic tensor survived.
        if not dynamic_indices:
            return False
        self._dynamic_input_indices = dynamic_indices
        if self._log_label is not None:
            logger.debug(
                "DFlash %s stable runtime inputs enabled: "
                "dynamic_indices=%s stable_count=%d",
                self._log_label,
                dynamic_indices,
                len(inputs) - len(dynamic_indices),
            )
        return True

    def __call__(self, *input_args, out, **input_kwargs):
        inputs = self._runtime._runtime_utils.prepare_inputs(
            *input_args, **input_kwargs
        )
        supported = (
            out is None
            and self._resolve_dynamic_inputs(inputs)
            and all(self._tensor_is_supported(tensor) for tensor in inputs)
        )
        if supported:
            pointers = tuple(tensor.data_ptr() for tensor in inputs)
            if self._bindings is not None and self._bindings.execute(
                self._runtime,
                pointers,
                self._get_device_addrs,
                self._output,
            ):
                return self._output

            output = self._original_run(*input_args, out=out, **input_kwargs)
            if self._bindings is None:
                self._bindings = _RuntimeInputBindingCache(self._dynamic_input_indices)
            self._bindings.observe(pointers)
            self._output = output
            if self._log_label is not None:
                logger.debug(
                    "DFlash %s runtime input profile installed",
                    self._log_label,
                )
            return output

        output = self._original_run(*input_args, out=out, **input_kwargs)
        return output


def _plan_context_kv_runs(
    physical_blocks: torch.Tensor,
    block_offsets: torch.Tensor,
    request_ids: torch.Tensor,
    group_id: int,
    *,
    partition_size: int = _CACHE_PARTITION_SIZE,
) -> tuple[_ContextKVRun, ...]:
    """Partition packed context tokens into maximal custom-store calls.

    The stateful prefill kernel writes a contiguous interval beginning at
    ``(block_tables[0], seq_idx[0, 0])``. Its ``slot_mapping`` argument does not
    redirect individual writes, so arbitrary paged coordinates must be split at
    request and physical-block boundaries before entering a graph.
    """
    if partition_size <= 0:
        raise ValueError(f"partition_size must be positive, got {partition_size}")
    if group_id < 0:
        raise ValueError(f"group_id must be non-negative, got {group_id}")
    tensors = (physical_blocks, block_offsets, request_ids)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("physical_blocks, block_offsets, and request_ids must be 1-D")
    if len({t.numel() for t in tensors}) != 1:
        raise ValueError("physical_blocks, block_offsets, and request_ids must align")

    blocks = [int(v) for v in physical_blocks.tolist()]
    offsets = [int(v) for v in block_offsets.tolist()]
    requests = [int(v) for v in request_ids.tolist()]
    if any(block < 0 for block in blocks):
        raise ValueError("physical block ids must be non-negative")
    if any(offset < 0 or offset >= partition_size for offset in offsets):
        raise ValueError(f"block offsets must be in [0, {partition_size})")
    if any(request < 0 for request in requests):
        raise ValueError("request ids must be non-negative")
    if not blocks:
        return ()

    runs: list[_ContextKVRun] = []
    run_start = 0
    for token_idx in range(1, len(blocks) + 1):
        continues = token_idx < len(blocks) and (
            requests[token_idx] == requests[token_idx - 1]
            and blocks[token_idx] == blocks[token_idx - 1]
            and offsets[token_idx] == offsets[token_idx - 1] + 1
        )
        if continues:
            continue

        token_count = token_idx - run_start
        run = _ContextKVRun(
            token_start=run_start,
            token_count=token_count,
            physical_block_id=blocks[run_start],
            block_offset=offsets[run_start],
            group_id=group_id,
        )
        if run.block_offset + run.token_count > partition_size:
            raise ValueError(f"run crosses partition boundary: {run}")
        if run.token_count > _DFLASH_CONTEXT_KV_MAX_RUN_LEN:
            remaining = run.token_count
            token_start = run.token_start
            block_offset = run.block_offset
            while remaining:
                token_count = min(remaining, _DFLASH_CONTEXT_KV_MAX_RUN_LEN)
                runs.append(
                    _ContextKVRun(
                        token_start=token_start,
                        token_count=token_count,
                        physical_block_id=run.physical_block_id,
                        block_offset=block_offset,
                        group_id=run.group_id,
                    )
                )
                remaining -= token_count
                token_start += token_count
                block_offset += token_count
        else:
            runs.append(run)
        run_start = token_idx

    return tuple(runs)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Traceable RMSNorm.

    `vllm._custom_ops.rms_norm` is a CUDA kernel; inside a compiled RBLN graph we
    need plain aten. Computed in fp32 like vLLM's `forward_native` so the drafter
    does not drift from the reference in bf16.
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x.to(orig_dtype) * weight).to(orig_dtype)


def _apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on K only.

    vllm-rbln already replaces the upstream cos_sin_cache with rotate-half style
    `cos_cache`/`sin_cache` (patches/rotary_embedding.py) because that layout
    suits RBLN, so we consume those directly instead of re-deriving them.
    """
    # K is [num_ctx, num_kv_heads, head_dim], while cos/sin are normally
    # [num_ctx, head_dim]. Keep a singleton head axis so rotation remains
    # independent within every KV head. This also handles already-expanded
    # inputs used by the fused CPU golden.
    if k.dim() == cos.dim() + 1:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    return k * cos + rotate_neox(k) * sin


class _ContextKVPrecompute:
    """Owns bounded projection graphs and native cache copies."""

    def __init__(self, model) -> None:
        self._model = model
        self._graphs: dict[tuple[int, int, bool], object] = {}
        self._runtime_holders: dict[tuple[int, int, bool], list] = {}
        # Persisting these eager runtimes is unsafe: a same-process cache hit
        # reconstructs its global const-buffer index and tries to allocate that
        # already-live index again in the context-wide pool.
        self._use_cache = False
        self._layer_constants: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None],
        ] = {}
        self._run_inputs: dict[
            tuple[int, torch.dtype, torch.device], tuple[torch.Tensor, ...]
        ] = {}
        # group_id -> (block_idx[num_ctx], offset[num_ctx]) for this step
        self._group_slots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._group_runs: dict[int, tuple[_ContextKVRun, ...]] = {}
        # layer index -> group id
        self._layer_group: dict[int, int] = {}
        # Shared with the runner so these graphs land in the same compile
        # context (and therefore the same weight-sharing / cache root) as the
        # target and drafter forwards.
        self._compile_context = None

    def set_compile_context(self, compile_context) -> None:
        if (
            self._graphs
            and self._compile_context is not None
            and compile_context is not self._compile_context
        ):
            raise RuntimeError(
                "DFlash context-KV compile context changed after compile"
            )
        self._compile_context = compile_context

    def set_group_slots(
        self,
        group_slots: dict[int, tuple[torch.Tensor, torch.Tensor]],
        layer_group: dict[int, int],
        request_ids: torch.Tensor | None = None,
        *,
        partition_size: int = _CACHE_PARTITION_SIZE,
    ) -> None:
        """Caller supplies per-attention-group cache coordinates.

        Sliding-window layers do not share the global slot mapping: RBLN gives
        them one window-sized block per request (`local_block_tables`) and an
        offset derived from `cache_seq_lens`. The proposer already builds that
        metadata per group, so it resolves the coordinates and hands them over
        rather than this module re-deriving them from a slot mapping that is only
        valid for the full-attention layer.
        """
        self._group_slots = group_slots
        self._layer_group = layer_group
        if request_ids is None:
            # Kept for small tests and callers that predate run planning. A real
            # scheduler batch must pass request ids so adjacent requests never
            # coalesce merely because their physical coordinates happen to be
            # consecutive.
            first = next(iter(group_slots.values()), (torch.empty(0),))[0]
            request_ids = torch.zeros(first.numel(), dtype=torch.int64)
        self._group_runs = {
            group_id: _plan_context_kv_runs(
                blocks,
                offsets,
                request_ids,
                group_id,
                partition_size=partition_size,
            )
            for group_id, (blocks, offsets) in group_slots.items()
        }
        missing_groups = set(layer_group.values()) - set(self._group_runs)
        if missing_groups:
            raise ValueError(
                f"layers refer to missing attention groups: {missing_groups}"
            )

    def _get_layer_constants(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        constants = self._layer_constants.get(layer_idx)
        if constants is not None:
            return constants

        m = self._model
        output_size = m._num_kv_heads * m._head_dim
        hidden_size = m._fused_kv_weight.shape[-1]
        weights = m._fused_kv_weight.view(
            m._num_attn_layers, 2, output_size, hidden_size
        )
        # clone() is intentional and happens once: a contiguous slice can still
        # alias the combined [2*L*kv, hidden] base. Independent storage prevents
        # lowering from reconstructing a combined K/V projection and select
        # fan-out inside the one-layer graph.
        key_weight = weights[layer_idx, 0].clone()
        value_weight = weights[layer_idx, 1].clone()
        if m._fused_kv_bias is None:
            key_bias = value_bias = None
        else:
            biases = m._fused_kv_bias.view(m._num_attn_layers, 2, output_size)
            key_bias = biases[layer_idx, 0].clone()
            value_bias = biases[layer_idx, 1].clone()
        constants = (key_weight, value_weight, key_bias, value_bias)
        self._layer_constants[layer_idx] = constants
        return constants

    def _graph_fn(self, layer_idx: int, run_len: int):
        m = self._model
        nkv, head_dim = m._num_kv_heads, m._head_dim
        key_weight, value_weight, key_bias, value_bias = self._get_layer_constants(
            layer_idx
        )
        hidden_norm = m._hidden_norm_weight
        key_norm = m._k_norm_weights[layer_idx]

        def fn(context_states, cos, sin):
            normed = _rms_norm(context_states, hidden_norm, m._rms_norm_eps)
            key = torch.nn.functional.linear(normed, key_weight, key_bias).view(
                run_len, nkv, head_dim
            )
            value = torch.nn.functional.linear(normed, value_weight, value_bias).view(
                run_len, nkv, head_dim
            )
            key = _rms_norm(key, key_norm, m._rms_norm_eps)
            key = _apply_rope(key, cos, sin)
            return (
                key.permute(1, 0, 2).contiguous(),
                value.permute(1, 0, 2).contiguous(),
            )

        # Dynamo caches by code-object identity. These closures intentionally
        # compile with different weights, shapes, and backend runtime holders;
        # sharing this lexical code object makes them look like recompilations
        # of one function and trips the 64-entry limit on chunked prompts.
        frame_name = f"dflash_context_kv_l{layer_idx}_n{run_len}"
        fn.__name__ = frame_name
        fn.__code__ = fn.__code__.replace(co_name=frame_name)
        return fn

    def _get_graph(self, layer_idx: int, run_len: int):
        # Keep each projection graph layer-local. Multi-layer stateful stores
        # violate Machine dependency constraints, while these cache-free graphs
        # can coexist safely and never change a KV physical view.
        profile_len = _context_kv_bucket_size(run_len)
        key = (layer_idx, profile_len, self._model._fused_kv_bias is not None)
        if key not in self._graphs:
            fn = self._graph_fn(layer_idx, profile_len)
            if not envs.VLLM_RBLN_COMPILE_MODEL:
                self._graphs[key] = fn
            else:
                runtime_holder: list = []
                compiled = rbln_compile(
                    fn,
                    dynamic=False,
                    fullgraph=True,
                    compile_context=self._compile_context,
                    num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                    model_trace_method="export",
                    guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                    mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                    runtime_holder=runtime_holder,
                    use_cache=self._use_cache,
                )
                self._graphs[key] = _StableRuntimeGraph(
                    compiled,
                    runtime_holder,
                    dynamic_arg_indices=(0, 1, 2),
                )
                self._runtime_holders[key] = runtime_holder
        return self._graphs[key]

    def _get_run_inputs(
        self, run_len: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stable device addresses for the verified decode/prefill buckets."""
        profile_len = _context_kv_bucket_size(run_len)
        key = (profile_len, dtype, device)
        cached = self._run_inputs.get(key)
        if cached is None:
            cached = (
                torch.zeros(
                    profile_len,
                    self._model._fused_kv_weight.shape[-1],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    profile_len,
                    self._model._head_dim,
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    profile_len,
                    self._model._head_dim,
                    dtype=dtype,
                    device=device,
                ),
            )
            self._run_inputs[key] = cached
        return cached

    def _store_projected_kv(
        self,
        layer_indices: tuple[int, ...] | list[int],
        run: _ContextKVRun,
        projected: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write one run with a single native batched D2D copy submission."""
        if len(layer_indices) != len(projected):
            raise ValueError("layer indices and projected K/V must align")
        cache_slice = slice(run.block_offset, run.block_offset + run.token_count)
        destinations: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        for layer_idx, (key, value) in zip(layer_indices, projected):
            cache = self._model._attn_layers[layer_idx].kv_cache
            expected_shape = (
                self._model._num_kv_heads,
                run.token_count,
                self._model._head_dim,
            )
            if (
                tuple(key.shape) != expected_shape
                or tuple(value.shape) != expected_shape
            ):
                raise ValueError(
                    f"projected K/V shape mismatch at layer {layer_idx}: "
                    f"{tuple(key.shape)}, {tuple(value.shape)} != {expected_shape}"
                )
            destinations.extend(
                (
                    cache[0, run.physical_block_id, :, 0, cache_slice, :],
                    cache[1, run.physical_block_id, :, 0, cache_slice, :],
                )
            )
            sources.extend((key, value))
        torch._foreach_copy_(destinations, sources)

    def run(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        m = self._model
        if context_slot_mapping is None:
            # dummy_run: upstream runs the projection for memory profiling only
            # and writes nothing. Compiling here would bake a bogus cache
            # binding, so skip entirely. Only None-ness is read here; see
            # WRITE_CONTEXT_KV.
            return
        if not self._group_runs:
            raise RuntimeError(
                "RBLNDFlashProposer must call set_group_slots() before "
                "precompute_and_store_context_kv(); sliding-window layers do "
                "not share the global slot mapping."
            )

        rope = m.layers[0].self_attn.rotary_emb
        cos = rope.cos_cache.index_select(0, context_positions).to(context_states.dtype)
        sin = rope.sin_cache.index_select(0, context_positions).to(context_states.dtype)

        num_context = context_states.shape[0]
        layers_by_group: dict[int, list[int]] = {}
        for layer_idx, group_id in self._layer_group.items():
            layers_by_group.setdefault(group_id, []).append(layer_idx)
        for layer_indices in layers_by_group.values():
            layer_indices.sort()

        for group_id, runs in self._group_runs.items():
            layer_indices = layers_by_group.get(group_id, [])
            if not layer_indices:
                raise ValueError(f"attention group {group_id} has no layers")
            partition_size = m._attn_layers[layer_indices[0]].kv_cache.shape[-2]
            for run in runs:
                if run.token_start + run.token_count > num_context:
                    raise ValueError(f"run exceeds context input: {run}")
                if run.block_offset + run.token_count > partition_size:
                    raise ValueError(f"run exceeds cache partition: {run}")
                token_slice = slice(run.token_start, run.token_start + run.token_count)
                state_input, cos_input, sin_input = self._get_run_inputs(
                    run.token_count, context_states.dtype, context_states.device
                )
                profile_len = state_input.shape[0]
                input_slice = slice(0, run.token_count)
                state_input[input_slice].copy_(context_states[token_slice])
                cos_input[input_slice].copy_(cos[token_slice])
                sin_input[input_slice].copy_(sin[token_slice])
                expected_output_shape = (
                    m._num_kv_heads,
                    profile_len,
                    m._head_dim,
                )
                projected: list[tuple[torch.Tensor, torch.Tensor]] = []
                for layer_idx in layer_indices:
                    cache = m._attn_layers[layer_idx].kv_cache
                    if cache.shape[-2] != partition_size:
                        raise ValueError(
                            f"group {group_id} cache partition mismatch at layer "
                            f"{layer_idx}: {cache.shape[-2]} != {partition_size}"
                        )
                    graph = self._get_graph(layer_idx, run.token_count)
                    output = graph(state_input, cos_input, sin_input)
                    if (
                        not isinstance(output, (tuple, list))
                        or len(output) != 2
                        or not all(
                            isinstance(tensor, torch.Tensor) for tensor in output
                        )
                    ):
                        raise RuntimeError(
                            "DFlash context projection graph must return K and V"
                        )
                    if any(
                        tuple(tensor.shape) != expected_output_shape
                        for tensor in output
                    ):
                        raise RuntimeError(
                            "DFlash context projection output shape does not match "
                            "its profile"
                        )
                    projected.append(
                        (
                            output[0][:, input_slice, :],
                            output[1][:, input_slice, :],
                        )
                    )
                self._store_projected_kv(layer_indices, run, projected)


def get_or_create_context_kv(model) -> _ContextKVPrecompute:
    """The one way to reach the precompute helper.

    It exists because the proposer has to CONFIGURE the helper
    (`set_compile_context`, `set_group_slots`) before the first
    `precompute_and_store_context_kv` call, while the helper used to be created
    lazily *inside* that call. The proposer therefore always saw None, skipped
    `set_group_slots`, and the call then built a fresh helper with no slots --
    tripping its own guard on the first real request. Creating it here, from
    either side, removes the ordering hazard instead of documenting it.
    """
    if not hasattr(model, "_num_attn_layers"):
        # Same guard as upstream: buffers are built after weight load.
        model._build_fused_kv_buffers()
    helper = getattr(model, "_rbln_ctx_kv", None)
    if helper is None:
        helper = _ContextKVPrecompute(model)
        model._rbln_ctx_kv = helper
    return helper


@register_patch(target=_TARGET, reason=_WHY)
def patched_precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    helper = get_or_create_context_kv(self)
    helper.run(context_states, context_positions, context_slot_mapping)
