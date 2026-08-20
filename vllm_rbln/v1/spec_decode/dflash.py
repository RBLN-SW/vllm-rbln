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
"""RBLN proposer for DFlash (block-diffusion) speculative decoding.

DFlash is not an EAGLE3 variant with different weights -- it drafts differently:

    EAGLE3   N sequential forwards, causal, query = previous draft token
    DFlash   ONE forward, NON-causal, query = [bonus token] + N mask tokens,
             and the keys/values are the TARGET's hidden states, projected into
             the drafter's KV cache before the forward runs

That single forward is the whole point on RBLN. handoff §10 measured EAGLE3
winning on acceptance (tok/step 1.794) and losing on step cost (104.2 ms vs
52.5 ms spec-off), for a net -9.6% at mns=1. DFlash removes the N-forward loop
that caused that loss, so it is the drafter most likely to convert acceptance
into an actual TPOT win here.

Why this subclasses RBLNEagleProposer rather than upstream DFlashProposer:
upstream splits along "which drafting algorithm", but the expensive, fragile
part on RBLN is the execution shell -- compiled graph wiring, DP rendezvous and
padding, per-group attention metadata, KV-cache bindings, batch bucketing. All
of that lives in RBLNEagleProposer and is written against SpecDecodeBaseProposer,
not against anything EAGLE3-specific (upstream's EagleProposer is a 22-line
subclass that only sets pass_hidden_states_to_model=True, which DFlash also
sets). So we keep that shell and swap the drafting algorithm on top, porting the
seven overrides from vllm/v1/spec_decode/dflash.py.

Multiple inheritance from both was rejected: DFlashProposer.__init__ forwards
`pass_hidden_states_to_model` to super(), which RBLNEagleProposer.__init__ does
not accept, so the MRO breaks at construction.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

import vllm_rbln.envs as envs
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.patches.qwen3_dflash import (
    WRITE_CONTEXT_KV,
    get_or_create_context_kv,
)
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.utils import pad
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.eagle import _DEVICE_ARGMAX, RBLNEagleProposer

logger = init_logger(__name__)

# The DFlash drafter queries exactly `1 + num_speculative_tokens` positions on
# every step, in both target phases: after a target prefill it still drafts one
# block from the bonus token, it does not re-read the prompt. So it always takes
# the decode-shaped path, and `runner.is_prefill` must not leak into it.
#
# This is not cosmetic. `_preprocess` reads the *whole* input buffer under
# is_prefill (eagle.py: `self.input_ids.view(num_reqs, -1)`), which makes the
# query width the prefill chunk instead of the block width, while the attention
# mask stays at block width -- and the flash-attention tiler rejects the pair
# with `mask_dims.at(i) == input_dims.at(i) not satisfied`. EAGLE's own unrolled
# path pins the same constant for the same reason.
_DRAFT_IS_PREFILL = False
_DFLASH_COMBINE_DECODE_BUCKET = 8
_DFLASH_COMBINE_PREFILL_BUCKET = 512


def _combine_hidden_states_bucket_size(run_len: int) -> int:
    if not 1 <= run_len <= _DFLASH_COMBINE_PREFILL_BUCKET:
        raise ValueError(
            "DFlash combine run length must be in "
            f"[1, {_DFLASH_COMBINE_PREFILL_BUCKET}], got {run_len}"
        )
    if run_len <= _DFLASH_COMBINE_DECODE_BUCKET:
        return _DFLASH_COMBINE_DECODE_BUCKET
    return _DFLASH_COMBINE_PREFILL_BUCKET


def _dflash_target_rope_is_neox_style(target_model: nn.Module) -> bool | None:
    """The target's RoPE rotation style, from its first rotary module.

    A DFlash head must rotate Q/K the way the target it was distilled against
    does, and a mismatch is silent: acceptance collapses but nothing errors.
    Draft checkpoints do not carry the style, so read it from the target.
    Returns None when the target uses no RoPE.

    Ported from newer upstream vLLM (`dflash_target_rope_is_neox_style` in
    qwen3_dflash.py); the release pinned here predates it.
    """
    language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    for module in language_model.modules():
        style = getattr(module, "is_neox_style", None)
        if isinstance(style, bool):
            return style
    return None


def _check_draft_rope_style(draft_model: nn.Module, expected_is_neox: bool) -> None:
    """Fail loudly when the built drafter cannot honor the target's RoPE style.

    The vLLM release pinned here builds `DFlashQwen3Attention` rotary modules
    without reading `hf_config.is_neox_style`, so plumbing the target's style
    into the draft config (the upstream fix) is inert on it. Proceeding with a
    mismatched drafter would silently collapse acceptance; refuse instead.
    """
    for module in draft_model.modules():
        style = getattr(module, "is_neox_style", None)
        if isinstance(style, bool):
            if style != expected_is_neox:
                raise ValueError(
                    "DFlash drafter RoPE style does not match its target: the "
                    f"target rotates {'neox' if expected_is_neox else 'gptj'}-"
                    f"style but the drafter was built {'neox' if style else 'gptj'}-"
                    "style. This vLLM release ignores hf_config.is_neox_style "
                    "for DFlash drafters; upgrade vLLM or use a neox-style "
                    "target."
                )
            return


def _validate_dflash_geometry(
    max_num_tokens: int,
    max_batch_size: int,
    num_speculative_tokens: int,
) -> None:
    """Reject configurations the port only appears to support.

    Both bounds are silent failures without this check: an oversized block
    width overruns the base class's `max_num_tokens`-sized query buffers only
    at warmup (or later), and speculation past the 8-row decode profile pads
    every context-KV and combiner decode run to the prefill profile -- a large
    unexplained step-time regression rather than an error.
    """
    num_query_per_req = 1 + num_speculative_tokens
    if num_query_per_req > _DFLASH_COMBINE_DECODE_BUCKET:
        # patches/qwen3_dflash.py pins its context-KV decode profile to the
        # same 8 rows; both would fall off the cliff together.
        raise ValueError(
            "DFlash supports num_speculative_tokens <= "
            f"{_DFLASH_COMBINE_DECODE_BUCKET - 1}, got {num_speculative_tokens}. "
            "Larger blocks pad every decode-step projection to the prefill "
            "profile; derive the decode buckets from the configuration before "
            "lifting this limit."
        )
    max_query_tokens = max_batch_size * num_query_per_req
    if max_num_tokens < max_query_tokens:
        raise ValueError(
            "DFlash query buffers reuse the base class's max_num_batched_tokens"
            f"-sized allocations, so max_num_batched_tokens ({max_num_tokens}) "
            "must cover max_batch_size * (1 + num_speculative_tokens) "
            f"({max_batch_size} * {num_query_per_req} = {max_query_tokens}). "
            "Raise max_num_batched_tokens or lower max_num_seqs."
        )
    if max_num_tokens > _DFLASH_COMBINE_PREFILL_BUCKET:
        # A step packs at most the restored token budget (max_num_batched_
        # tokens) of context rows into ONE combiner call, and dummy_run never
        # exercises the combiner, so an oversized budget only surfaces on the
        # first long prefill chunk in production.
        raise ValueError(
            "DFlash's combiner prefill profile is fixed at "
            f"{_DFLASH_COMBINE_PREFILL_BUCKET} rows, so max_num_batched_tokens "
            f"({max_num_tokens}) must not exceed it. Lower "
            "max_num_batched_tokens, or derive the combiner and context-KV "
            "profile buckets from the configuration before lifting this limit."
        )


def _get_dflash_forward_split(layer_types: list[str]) -> int | None:
    """Return the one SWA-to-full boundary that needs a graph split.

    The Machine flash-attention control pass keys dynamic indices only by
    batch and physical partition. A compiled function therefore cannot contain
    attention layers whose partition loops use different sequence indices.
    Homogeneous stacks need no split; MiniMax-M2.5-DFlash's four leading SWA
    layers and final full layer need one at index four.
    """
    if not layer_types:
        return None
    invalid = set(layer_types) - {"full_attention", "sliding_attention"}
    if invalid:
        raise ValueError(f"Invalid DFlash layer type(s): {sorted(invalid)}")
    if len(set(layer_types)) == 1:
        return None
    split_index = layer_types.index("full_attention")
    if (
        not split_index
        or any(kind != "sliding_attention" for kind in layer_types[:split_index])
        or any(kind != "full_attention" for kind in layer_types[split_index:])
    ):
        raise ValueError(
            "DFlash mixed attention layers must form contiguous sliding/full groups"
        )
    return split_index


def _dflash_page_crossing_mask(
    seq_lens: torch.Tensor,
    partition_size: int,
    query_len: int,
) -> torch.Tensor:
    """Requests whose one-block query K/V cannot fit in the current KV page.

    The RBLN naive non-causal kernel accepts one dynamic offset per physical
    partition and inserts the complete query block there. Unlike upstream's
    slot-mapping kernel, it cannot split one query across two pages. Suppress
    speculation only for the seven affected offsets in the MiniMax setup; the
    target advances normally and speculation resumes at the aligned page.
    """
    if seq_lens.ndim != 1:
        raise ValueError("DFlash sequence lengths must be one-dimensional")
    if partition_size <= 0 or query_len <= 0 or query_len > partition_size:
        raise ValueError("Invalid DFlash query/page geometry")
    offsets = torch.remainder(seq_lens.cpu(), partition_size)
    return offsets + query_len > partition_size


def _empty_drafts_for_page_crossing(
    crossing: torch.Tensor,
) -> list[list[int]] | None:
    """Skip a whole local batch before any unrepresentable KV insert runs."""
    crossing_list = [bool(value) for value in crossing.tolist()]
    if not any(crossing_list):
        return None
    # Selective execution would require rebuilding the compiled attention
    # metadata for a smaller batch. Running a crossing row and filtering only
    # its token IDs afterwards is unsafe because its unsplittable KV insert has
    # already happened. Sacrifice speculation for this one local batch instead.
    return [[] for _ in crossing_list]


class _BoundedHiddenStateCombiner:
    """Run the auxiliary-state FC through two stable input profiles.

    MiniMax supplies six target hidden states concatenated on the feature axis.
    ``combine_hidden_states`` projects those rows back to the drafter width, but
    its eager RBLN dispatch otherwise creates a runtime profile for every exact
    context length. Reuse one 8-row decode profile and the scheduler's 512-row
    prefill profile, then expose only the logical prefix to its consumer. The
    following context-KV planner independently splits that prefix at 506 rows.

    The FC is row-wise, so stale padded rows cannot affect the returned prefix.
    Keeping the staging tensors alive also makes their device addresses stable
    across requests without changing the compiled graph or FC implementation.
    """

    def __init__(self, combine) -> None:
        self._combine = combine
        self._inputs: dict[
            tuple[int, int, torch.dtype, torch.device], torch.Tensor
        ] = {}

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("DFlash auxiliary hidden states must be two-dimensional")
        run_len, width = hidden_states.shape
        bucket_len = _combine_hidden_states_bucket_size(run_len)
        key = (bucket_len, width, hidden_states.dtype, hidden_states.device)
        inputs = self._inputs.get(key)
        if inputs is None:
            inputs = torch.zeros(
                bucket_len,
                width,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._inputs[key] = inputs
        inputs[:run_len].copy_(hidden_states)
        output = self._combine(inputs)
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError(
                "DFlash hidden-state combiner must return a two-dimensional tensor"
            )
        if output.shape[0] != bucket_len:
            raise RuntimeError(
                "DFlash hidden-state combiner returned an unexpected row count: "
                f"expected {bucket_len}, got {output.shape[0]}"
            )
        return output[:run_len]


class _DFlashSplitForwardGraph:
    """Join independently compiled SWA and full-attention graph regions."""

    def __init__(self, sliding_graph, full_graph) -> None:
        self._sliding_graph = sliding_graph
        self._full_graph = full_graph

    def __call__(self, input_ids, positions, token_indices_to_sample=None):
        hidden_states, residual = self._sliding_graph(input_ids, positions)
        return self._full_graph(
            hidden_states,
            residual,
            positions,
            token_indices_to_sample,
        )


class RBLNDFlashProposer(RBLNEagleProposer):
    """DFlash drafting on the RBLN execution shell."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        self._dflash_sliding_layer_names: set[str] = set()
        self._dflash_sliding_window: int | None = None
        super().__init__(vllm_config, device, runner)

        # Only the bonus token and the mask tokens are queries; everything else
        # is context that reaches the drafter through its KV cache, so the query
        # buffers stay tiny compared with EAGLE3's.
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        _validate_dflash_geometry(
            max_num_tokens=self.max_num_tokens,
            max_batch_size=self.max_batch_size,
            num_speculative_tokens=self.num_speculative_tokens,
        )

        # Filled by set_inputs_first_pass. Upstream's fused Triton kernel uses
        # the original target positions both for RoPE and for the context slot
        # mapping. Keep a host copy for `_resolve_group_slots`: after rejection,
        # `cad.seq_lens` is intentionally rewound for the next draft query and
        # can no longer reconstruct the scheduled target rows' positions.
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._context_positions_cpu_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device="cpu"
        )

        # MUST stay None. The base class allocates this in
        # _init_parallel_drafting_params for parallel-drafting methods that embed
        # the mask slot as a HIDDEN STATE (PARD/PTD), and then load_model reads
        # `self.model.mask_hidden` to fill it. DFlash embeds the mask as an input
        # TOKEN instead, so its checkpoints carry no `mask_hidden` -- upstream
        # qwen3_dflash.load_weights even asserts the name is absent. Leaving the
        # tensor allocated makes load_model take that branch and die with
        # "'DFlashQwen3ForCausalLM' object has no attribute 'mask_hidden'".
        # Upstream DFlashProposer.__init__ ends with this same reset.
        self.parallel_drafting_hidden_state_tensor = None
        # Set per step by set_inputs_first_pass, consumed by
        # build_model_inputs_first_pass.
        self._dflash_num_context = 0
        self._dflash_hidden_states: torch.Tensor | None = None
        self._hidden_state_combiner: _BoundedHiddenStateCombiner | None = None

    # ------------------------------------------------------------------
    # Draft model configuration
    # ------------------------------------------------------------------

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Mark the DRAFTER non-causal.

        The flag is read back by vllm_rbln's attention impl
        (`_resolve_is_causal`), which is how the drafter ends up non-causal
        while the target in the same process stays causal. Without per-model
        scoping the only lever is the process-wide
        `VLLM_RBLN_FLASH_CAUSAL_ATTN`, which cannot express that split.
        """
        base = super()._create_draft_vllm_config()
        return replace(
            base,
            attention_config=replace(base.attention_config, use_non_causal=True),
        )

    def _warn_if_multimodal(self) -> None:
        # DFlash rides on Qwen3-family drafters that upstream allows with
        # multimodal targets; the base-class warning does not apply.
        pass

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is None:
            return True
        return dflash_config.get("use_aux_hidden_state", True)

    def _draft_ids(self, out: torch.Tensor, num_reqs: int) -> torch.Tensor:
        """DFlash drafters emit target-vocabulary ids directly.

        EAGLE3 narrows its logits to a draft vocabulary and needs
        `model.target_ids` to map back; DFlash shares the target vocabulary
        (200064 in both configs), so applying that map would gather with ids
        from the wrong space and collapse acceptance.

        The slice is load-bearing, exactly as in the base class: `out` carries
        one row per PADDED sample slot (`num_reqs_padded * num_spec`, from
        `token_indices_padded`), while the caller reshapes to the REAL batch.
        Without the slice, any decode step whose request count is not exactly a
        configured batch bucket dies in `view` with a shape mismatch.
        """
        out = out[:num_reqs]
        if out.dim() == 1 or out.dtype in (torch.int32, torch.int64):
            return out
        return out.argmax(dim=-1)

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Keep the drafter's metadata builders in step with its attention impl.

        These two learn their causality from different places, and for DFlash
        they would otherwise disagree:

          impl     constructed while the DRAFT config is current, so
                   _resolve_is_causal sees use_non_causal=True -> non-causal
          builder  created by the base class as
                   `attn_group.create_metadata_builders(self.vllm_config, ...)`,
                   i.e. from the TARGET config -> still causal

        The non-causal branch of RBLNFlashAttentionImpl.forward passes
        `attn_metadata.attn_masks` positionally, and the builder only fills that
        field under `if not self.is_causal`. A causal builder therefore hands the
        non-causal op a None mask, and the converter drops None arguments, so the
        graph arrives one argument short and rebel asserts
        `len(data) == num_args` -- a bare AssertionError with no mention of masks
        or causality.

        Flipping the flag here is deliberately narrower than threading the draft
        config through `create_metadata_builders`: that config also drives KV
        cache specs and layer discovery for every method, and only DFlash wants
        non-causal.
        """
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._configure_dflash_attention_layers()
        for attn_group in self.draft_attn_groups:
            builders = getattr(attn_group, "metadata_builders", None) or [
                attn_group.get_metadata_builder()
            ]
            for builder in builders:
                if getattr(builder, "is_causal", False):
                    logger.info(
                        "DFlash: forcing non-causal metadata builder for draft "
                        "layers %s",
                        sorted(attn_group.layer_names),
                    )
                builder.is_causal = False

    def _configure_dflash_attention_layers(self) -> None:
        """Map checkpoint-local layer types onto vLLM's global layer names.

        MiniMax-M2.5-DFlash has four sliding-attention layers followed by one
        full-attention layer. This vLLM release loads all five as full
        attention, so the checkpoint's ``layer_types`` never reaches the
        proposer. Keep the full KV allocation (DFlash prewrites all context
        K/V), but retain the layer split here so metadata can apply the trained
        compute semantics.
        """
        layer_names = sorted(
            (
                layer_name
                for attn_group in self.draft_attn_groups
                for layer_name in attn_group.layer_names
            ),
            key=extract_layer_index,
        )
        layer_types = getattr(self.draft_model_config.hf_config, "layer_types", None)
        if layer_types is None:
            self._dflash_sliding_layer_names = set()
            self._dflash_sliding_window = None
            return
        if len(layer_types) != len(layer_names):
            raise ValueError(
                "DFlash layer_types length does not match attention layers: "
                f"{len(layer_types)} != {len(layer_names)}"
            )
        invalid = set(layer_types) - {"full_attention", "sliding_attention"}
        if invalid:
            raise ValueError(f"Invalid DFlash layer type(s): {sorted(invalid)}")

        sliding_names = {
            layer_name
            for layer_name, layer_type in zip(layer_names, layer_types)
            if layer_type == "sliding_attention"
        }
        sliding_window = getattr(
            self.draft_model_config.hf_config, "sliding_window", None
        )
        if sliding_names and not sliding_window:
            raise ValueError("DFlash sliding layers require a sliding_window")
        self._dflash_sliding_layer_names = sliding_names
        self._dflash_sliding_window = int(sliding_window) if sliding_names else None
        logger.info(
            "DFlash attention semantics: sliding_window=%s sliding=%s full=%s",
            self._dflash_sliding_window,
            sorted(sliding_names),
            sorted(set(layer_names) - sliding_names),
        )

    def _block_start_positions(self, cad: CommonAttentionMetadata) -> torch.Tensor:
        """Positions view that puts the drafter's K/V writes at the block start.

        The metadata builder reads exactly one thing out of `positions`:

            seq_idx = positions[query_start_loc_cpu[:num_reqs]]

        and that becomes `attn_metadata.seq_lens`, the per-request cache write
        offset. Handing it the target's positions would aim the drafter's block
        K/V at the target's first scheduled token. The block lives at the end of
        the sequence -- `seq_lens` -- which is also where
        `set_inputs_first_pass` puts the block's query positions, so the two
        agree and the block can attend to itself.

        Built on the host: the builder gathers with a host index and moves the
        result across itself, so there is no reason to touch the device here --
        and on RBLN each eager device op would compile its own micro-graph every
        step.
        """
        num_reqs = cad.num_reqs
        starts = cad.query_start_loc_cpu[:num_reqs].to(torch.int64)
        probe = torch.zeros(int(starts[-1]) + 1, dtype=torch.int64)
        probe[starts] = cad._seq_lens_cpu[:num_reqs].to(torch.int64)
        return probe

    def _rebuild_block_draft_mask(
        self,
        attn_metadata,
        cad: CommonAttentionMetadata,
        num_reqs: int,
        num_reqs_padded: int,
        sliding_window: int | None = None,
        seq_lens_override: torch.Tensor | None = None,
    ) -> None:
        """Build a whole-block full-attention or causal-SWA mask.

        The shared builder writes its non-causal decode mask as
        `[batch, 1, 1, 1, max_seq_len]` -- query length one, because ordinary
        decode emits one token per request. DFlash queries `1 + num_spec`
        positions at once (bonus token plus the mask tokens), so that mask is a
        rank-matching failure inside the attention kernel:

            InferShape.cpp set_nontiled_params:
              mask_dims.at(i) == input_dims.at(i) not satisfied
            Cannot find valid tiling, op=rtosa.flash_attn_tile

        Full-attention layers let every query see the entire context and every
        query K/V in the block. Sliding-attention layers use the checkpoint's
        trained causal window: row ``i`` sees at most ``sliding_window`` keys
        ending at its own absolute position. The physical KV cache remains full
        sized in both cases because DFlash prewrites all context K/V before the
        draft forward.

        Built on the host from `_seq_lens_cpu` -- the same shadow the builder
        reads -- and moved across once, the way the builder does it. Deriving it
        from the already-uploaded mask instead would read a device tensor and
        run the arithmetic there, and every one of those eager ops becomes its
        own compiled micro-graph on RBLN, every step, in the hot path.
        """
        if getattr(attn_metadata, "attn_masks", None) is None:
            return
        num_query_per_req = 1 + self.num_speculative_tokens
        mask = attn_metadata.attn_masks

        max_seq_len = mask.shape[-1]
        seq_lens = cad._seq_lens_cpu if seq_lens_override is None else seq_lens_override
        assert seq_lens is not None, "flash-attention builder needs the host shadow"
        key_pos = torch.arange(max_seq_len)
        if sliding_window is None:
            key_end = (seq_lens[:num_reqs] + num_query_per_req).view(-1, 1)
            valid = key_pos.view(1, -1) < key_end
            new_mask = valid.view(num_reqs, 1, 1, 1, max_seq_len).expand(
                num_reqs, 1, 1, num_query_per_req, max_seq_len
            )
        else:
            query_pos = seq_lens[:num_reqs].view(-1, 1) + torch.arange(
                num_query_per_req
            ).view(1, -1)
            distance = query_pos.unsqueeze(-1) - key_pos.view(1, 1, -1)
            valid = (distance >= 0) & (distance < sliding_window)
            new_mask = valid.view(num_reqs, 1, 1, num_query_per_req, max_seq_len)
        if num_reqs_padded > num_reqs:
            new_mask = pad(new_mask, 0, num_reqs_padded)
        attn_metadata.attn_masks = new_mask.to(mask.dtype).contiguous().to(mask.device)

    def _localize_sliding_attn_metadata(
        self,
        attn_group,
        attn_metadata,
        cad: CommonAttentionMetadata,
        num_reqs: int,
        num_reqs_padded: int,
    ):
        """Bound SWA compute while preserving the shared full KV allocation.

        The non-causal flash kernel reduces one softmax state per physical
        partition. Passing the full 49K cache to a 2K sliding layer produces
        wholly-masked leading partitions once the request exceeds the window;
        those invalid partial states poison the final reduction. It also scans
        four small drafter layers over the full context on every decode step.

        Move the physical partitions that cover the union of all draft query
        windows to the front of the block table and rebase their absolute
        positions to that local view. Keep the original input shapes: mixing a
        short SWA attention shape and a full-attention shape in one compiled
        graph violates the compiler's per-partition dynamic-index invariant.
        The compiler expands the rebased raw ``[batch, 1]`` sequence position
        into per-partition lengths, making the kernel skip every unused trailing
        partition while bounding runtime work by the SWA window.
        """
        assert self._dflash_sliding_window is not None
        assert cad._seq_lens_cpu is not None
        block_size = int(attn_group.kv_cache_spec.block_size)
        num_query_per_req = 1 + self.num_speculative_tokens
        covered_tokens = self._dflash_sliding_window + num_query_per_req - 1
        # A span that straddles an aligned boundary can occupy one more block
        # than ceil(span / block_size). This exact bound is four blocks for
        # MiniMax-M2.5-DFlash: window=2048, queries=8, block_size=1024.
        num_local_blocks = (covered_tokens + block_size - 2) // block_size + 1

        absolute_seq_lens = cad._seq_lens_cpu[:num_reqs]
        window_starts = torch.clamp(
            absolute_seq_lens - self._dflash_sliding_window + 1,
            min=0,
        )
        first_blocks = torch.div(window_starts, block_size, rounding_mode="floor")
        base_positions = first_blocks * block_size
        local_seq_lens = absolute_seq_lens - base_positions

        full_block_tables = cad.block_table_tensor[:num_reqs]
        block_indices = first_blocks.view(-1, 1) + torch.arange(
            num_local_blocks, dtype=first_blocks.dtype
        ).view(1, -1)
        if (
            block_indices.numel()
            and int(block_indices.max()) >= full_block_tables.shape[1]
        ):
            raise ValueError("DFlash SWA local block view exceeds the KV block table")
        active_block_tables = torch.gather(
            full_block_tables,
            1,
            block_indices.to(full_block_tables.device),
        )
        local_block_tables = torch.zeros_like(full_block_tables)
        local_block_tables[:, :num_local_blocks] = active_block_tables
        local_block_tables = pad(local_block_tables, 0, num_reqs_padded)
        local_seq_lens = pad(local_seq_lens.view(-1, 1), 0, num_reqs_padded)

        sliding_metadata = copy(attn_metadata)
        sliding_metadata.block_tables = local_block_tables.to(
            device=attn_metadata.block_tables.device,
            dtype=attn_metadata.block_tables.dtype,
        )
        sliding_metadata.seq_lens = local_seq_lens.to(
            device=attn_metadata.seq_lens.device,
            dtype=attn_metadata.seq_lens.dtype,
        )
        sliding_metadata.attn_masks = attn_metadata.attn_masks
        self._rebuild_block_draft_mask(
            sliding_metadata,
            cad,
            num_reqs,
            num_reqs_padded,
            self._dflash_sliding_window,
            local_seq_lens[:num_reqs, 0],
        )
        return sliding_metadata

    def _specialize_layer_attn_metadata(
        self,
        attn_group,
        attn_metadata,
        cad: CommonAttentionMetadata,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> dict[str, object]:
        """Return full and SWA metadata views without changing cache geometry."""
        self._rebuild_block_draft_mask(attn_metadata, cad, num_reqs, num_reqs_padded)
        sliding_names = self._dflash_sliding_layer_names.intersection(
            attn_group.layer_names
        )
        sliding_metadata = None
        if sliding_names:
            sliding_metadata = self._localize_sliding_attn_metadata(
                attn_group,
                attn_metadata,
                cad,
                num_reqs,
                num_reqs_padded,
            )

        per_layer = {layer_name: attn_metadata for layer_name in attn_group.layer_names}
        if sliding_metadata is not None:
            for layer_name in sliding_names:
                per_layer[layer_name] = sliding_metadata
        return per_layer

    # ------------------------------------------------------------------
    # Compiled drafter graph
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        """Compile a DFlash-shaped drafter graph.

        RBLNEagleProposer.load_model wraps the drafter in a callable that
        forwards `hidden_states=` into it, because EAGLE3's drafter takes the
        target's hidden state as a forward input. DFlash's does not -- its
        `forward(input_ids, positions, inputs_embeds)` reads the target's states
        out of the KV cache that `precompute_and_store_context_kv` filled, so
        passing that kwarg fails with
        "TypeError: Unexpected keyword arguments: ['hidden_states']".

        So skip the EAGLE wrapper: `super(RBLNEagleProposer, self)` reaches the
        base weight-loading and attention-group setup directly, and we compile
        our own wrapper on top with the same options.
        """
        # The drafter must rotate K the way the target it was distilled
        # against does; a mismatch silently collapses acceptance. Newer vLLM
        # reads this from hf_config when building DFlashQwen3Attention, so
        # record it before the draft model is constructed. The release pinned
        # here ignores it -- _check_draft_rope_style below turns that into an
        # explicit error instead of a silent quality collapse.
        target_rope_is_neox = _dflash_target_rope_is_neox_style(target_model)
        if target_rope_is_neox is not None:
            # TODO(vllm-bump): inert on the pinned release (get_rope never
            # reads it for DFlash); becomes live when vLLM gains
            # dflash_target_rope_is_neox_style, and the check below can then
            # be retired.
            self.draft_model_config.hf_config.is_neox_style = target_rope_is_neox

        super(RBLNEagleProposer, self).load_model(target_model)

        if target_rope_is_neox is not None:
            _check_draft_rope_style(self.model, target_rope_is_neox)
        self._probe_dp_rendezvous_need()
        self._hidden_state_combiner = _BoundedHiddenStateCombiner(
            self.model.combine_hidden_states,
        )

        def sample_hidden_states(
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None,
        ):
            hidden_states = hidden_states.view(-1, self.hidden_size)
            if token_indices_to_sample is not None:
                # Only the mask slots are sampled. Advanced indexing rather than
                # index_select for the same reason as the EAGLE path: the index
                # arrives on CPU and index_select would reject the device mix.
                hidden_states = hidden_states[token_indices_to_sample]
            logits = self.model.compute_logits(hidden_states)
            if _DEVICE_ARGMAX:
                # Keep the reduction inside the region; eager would land on the
                # host implementation.
                ids = torch.ops.rbln.argmax(logits)
                return hidden_states, ids.to(torch.int32)
            return hidden_states, logits

        def dflash_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=None,
            )
            return sample_hidden_states(hidden_states, token_indices_to_sample)

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = dflash_wrapper
        else:

            def compile_forward(fn):
                return compile(
                    fn,
                    dynamic=False,
                    fullgraph=True,
                    compile_context=self.runner.compile_context,
                    num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                    model_trace_method="export" if USE_DEVICE_TENSOR else "",
                    process_group_dict=build_process_group_dict(),
                    guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                    mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                    use_static_output=True,
                    # Four DP workers compile the same DFlash shapes during
                    # warmup. Concurrent writes to a shared cache path can
                    # produce checksum-invalid artifacts, so drafter graphs
                    # remain process-local; target-model caching stays enabled.
                    use_cache=False,
                )

            layer_types = list(
                getattr(self.draft_model_config.hf_config, "layer_types", [])
            )
            split_index = _get_dflash_forward_split(layer_types)
            if split_index is None:
                self.model_executable = compile_forward(dflash_wrapper)
                return

            core_model = self.model.model

            def sliding_forward(input_ids: torch.Tensor, positions: torch.Tensor):
                hidden_states = core_model.embed_input_ids(input_ids)
                residual = None
                for layer in core_model.layers[:split_index]:
                    hidden_states, residual = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                    )
                return hidden_states, residual

            def full_forward(
                hidden_states: torch.Tensor,
                residual: torch.Tensor,
                positions: torch.Tensor,
                token_indices_to_sample: torch.Tensor | None = None,
            ):
                for layer in core_model.layers[split_index:]:
                    hidden_states, residual = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                    )
                hidden_states, _ = core_model.norm(hidden_states, residual)
                return sample_hidden_states(hidden_states, token_indices_to_sample)

            logger.info(
                "DFlash forward split at layer %d: SWA=%d full=%d",
                split_index,
                split_index,
                len(layer_types) - split_index,
            )
            self.model_executable = _DFlashSplitForwardGraph(
                compile_forward(sliding_forward),
                compile_forward(full_forward),
            )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        """Warm up the drafter at the shapes serving will use.

        Same skeleton as the EAGLE warmup minus the aux-projection fold (DFlash
        has no folded projection) and minus `hidden_states`. The KV-precompute
        graph is deliberately NOT warmed here: upstream's dummy_run passes
        slot_mapping=None to mean "compute, do not write", and honouring that
        keeps warmup from scribbling into a cache that is about to hold real
        context. It costs one compile on the first decode step instead.

        `num_tokens_per_req` / `is_prefill` / `num_padded_tokens` are accepted
        for signature compatibility with the runner but deliberately ignored --
        see below.
        """
        del num_tokens_per_req, is_prefill, num_padded_tokens
        # The drafter is block-shaped in every target phase: see the comment on
        # `_DRAFT_IS_PREFILL`. The runner sizes this warmup from the *target's*
        # shapes and so hands us the prefill chunk (512) on its prefill pass;
        # taking that literally would compile the drafter at a query width it
        # never runs at, and mismatch the block-width attention mask. Pin both
        # the width and the DP padding that follows from it.
        is_prefill = _DRAFT_IS_PREFILL
        num_tokens_per_req = 1 + self.num_speculative_tokens

        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens

        common_attn_metadata = self._build_dummy_attn_metadata(
            num_reqs, num_tokens_per_req
        )
        num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_tokens, is_prefill)
        )

        # Same positions view the real step uses, so warmup compiles the shape
        # serving will hit.
        block_start_positions = self._block_start_positions(common_attn_metadata)

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=block_start_positions,
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            per_layer_attn_metadata.update(
                self._specialize_layer_attn_metadata(
                    attn_group,
                    attn_metadata,
                    common_attn_metadata,
                    num_reqs,
                    num_reqs_padded,
                )
            )

        token_indices_to_sample = (
            torch.arange(num_reqs, device=self.device, dtype=torch.int32)
            * num_tokens_per_req
        )
        input_ids, positions, _, token_indices_padded = self._preprocess(
            num_reqs,
            num_reqs_padded,
            num_tokens,
            token_indices_to_sample,
            is_prefill,
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            self.model_executable(
                input_ids=input_ids,
                positions=positions,
                token_indices_to_sample=token_indices_padded,
            )

    # ------------------------------------------------------------------
    # Per-step input construction
    # ------------------------------------------------------------------

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Build the query stream: one bonus token then N mask tokens per request.

        Upstream fuses this into a Triton kernel. Here it is plain torch on the
        host: the tensors are batch x (1+num_spec) -- a few dozen elements -- so
        a kernel buys nothing, and `rebel.triton` would have to compile a graph
        for work that never touches the device.
        """
        batch_size = cad.num_reqs
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = target_token_ids.shape[0]
        # Flatten to [num_context, fused_hidden], then slice. Two separate
        # hazards make both steps necessary:
        #
        #   rank    The runner's two branches disagree. On the spec-decode path
        #           it flattens (`h.view(-1, h.shape[-1])`) giving 2-D, but on
        #           the FIRST step -- prefill, spec_decode_metadata is None -- it
        #           does `cat([h[:n] for h in aux], dim=-1)` where each `h` is
        #           [1, tokens, hidden], so the result is 3-D
        #           [1, tokens, 6*hidden]. Slicing dim 0 there cuts the BATCH.
        #   length  The spec-decode branch does NOT slice to total_num_tokens
        #           (rbln_model_runner.py:1749) while `target_token_ids` does.
        #           EAGLE survives that because `_preprocess` slices later;
        #           DFlash feeds these rows straight into the precompute graph,
        #           where the row count must match `context_positions` and the
        #           cache coordinates.
        hidden = target_hidden_states.reshape(-1, target_hidden_states.shape[-1])
        self._dflash_hidden_states = hidden[: self._dflash_num_context]

        # Context positions drive RoPE inside precompute_and_store_context_kv.
        # Upstream's kernel copies them straight from target_positions
        # (`tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos)`); the
        # context tokens are the target's scheduled tokens, packed contiguously,
        # so the copy is the whole prefix. Leaving the buffer at its zero-init
        # would rotate every context key to position 0 -- no crash, just a
        # drafter that proposes noise.
        num_context = self._dflash_num_context
        context_positions = target_positions[:num_context]
        self._context_positions_cpu_buffer[:num_context].copy_(context_positions)
        self._context_positions_buffer[:num_context].copy_(
            self._context_positions_cpu_buffer[:num_context]
        )

        # Query ids: [bonus, mask, mask, ...] per request.
        ids = self.input_ids[:num_query_total].view(batch_size, num_query_per_req)
        ids.fill_(self.parallel_drafting_token_id)
        ids[:, 0] = next_token_ids[:batch_size]

        # Query positions continue each request's sequence.
        seq_lens = cad.seq_lens[:batch_size].view(batch_size, 1)
        step = torch.arange(num_query_per_req, device=seq_lens.device).view(1, -1)
        self.positions[:num_query_total] = (seq_lens + step).reshape(-1)

        # Every mask slot is sampled; the bonus token is already known.
        token_indices = (
            torch.arange(batch_size, device=self.device).view(-1, 1) * num_query_per_req
            + torch.arange(1, num_query_per_req, device=self.device).view(1, -1)
        ).reshape(-1)

        return num_query_total, token_indices.to(torch.int32)

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        token_indices_to_sample: torch.Tensor | None,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
        """The base decode path minus the discarded hidden-states copy.

        DFlash's drafter reads the target's states out of its KV cache, so
        both call sites here drop the third return value -- but the base
        method still builds and pads `self.hidden_states` rows for it every
        step, and `pad` clones view inputs even when no padding is needed
        (the dynamo non-view contract). That is one per-step device copy
        dispatch for a value nobody reads, the same class of per-step eager
        cost this port removes everywhere else. input_ids and positions keep
        the base padding behavior exactly; they feed the compiled graph.

        DFlash always runs the decode shape (_DRAFT_IS_PREFILL); delegate a
        prefill call to the base class rather than guessing at it.
        """
        if is_prefill:
            return super()._preprocess(
                num_reqs,
                num_reqs_padded,
                num_input_tokens,
                token_indices_to_sample,
                is_prefill,
            )
        input_ids = pad(
            self.input_ids[:num_input_tokens].view(num_reqs, -1), 0, num_reqs_padded
        )
        positions = pad(
            self.positions[:num_input_tokens].view(num_reqs, -1), 0, num_reqs_padded
        )
        token_indices_to_sample_padded = (
            pad(token_indices_to_sample, 0, num_reqs_padded)
            if token_indices_to_sample is not None
            else None
        )
        return input_ids, positions, None, token_indices_to_sample_padded

    def build_per_group_and_layer_attn_metadata(self, cad, draft_index: int = 0):
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer.items():
            causal = getattr(attn_metadata, "causal", False)
            if layer_name not in self._dflash_sliding_layer_names and causal:
                raise RuntimeError(
                    f"DFlash full-attention layer {layer_name} reports causal=True. "
                    "Check the draft attention configuration."
                )
        return per_group, per_layer

    def _intermediate_prefill_drafts(self, num_reqs: int) -> torch.Tensor | None:
        """Skip the stateful draft forward when its output will be discarded.

        Chunked prefill still has to project and store every target hidden
        state into the DFlash cache.  It does not have to run the draft model
        after an intermediate chunk: the scheduler discards those proposals.
        On RBLN that unnecessary forward also writes bonus/mask query K/V into
        lookahead pages.  Avoiding those writes keeps intermediate chunks free
        of draft-only state and removes work whose output cannot be observed.

        Returning zero placeholders matches the existing Medusa intermediate
        prefill path in the runner; the scheduler drops their values.
        """
        if not getattr(self.runner, "is_intermediate_chunked_prefill", False):
            return None
        return torch.zeros(
            num_reqs,
            self.num_speculative_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Context KV precompute
    # ------------------------------------------------------------------

    def _resolve_group_slots(
        self, per_group_metadata, cad: CommonAttentionMetadata
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, int], torch.Tensor
    ]:
        """Cache coordinates for the context write, per attention group.

        **`cad.slot_mapping` is unusable here.** vllm-rbln never fills it -- it is
        `torch.tensor(0)  # dummy` in every construction site (rbln_model_runner
        :926, eagle.py:741 and :931) because the attention op derives the write
        location itself from `positions` plus the block table. Reading it gave
        `IndexError: slice() cannot be applied to a 0-dim tensor` on the first
        real request; warmup missed it because dummy_run does not precompute.

        So compute the coordinates the way upstream's fused kernel does
        (`copy_and_expand_dflash_inputs_kernel`):

            block_num = position // block_size          (clamped to the table)
            block_id  = block_table[req, block_num]
            offset    = position % block_size

        All of it on the host: the block table is already a CPU tensor here and
        the context positions are derivable from `query_start_loc_cpu` +
        `_seq_lens_cpu`, so nothing has to come back off the device.

        The sliding-window branch below is dead on this stack -- the drafter's
        five layers land in ONE full-attention KV group (`groups=1`), because
        upstream's `DFlashQwen3Attention` never passes `per_layer_sliding_window`.
        Kept for a stack where the groups do split.
        """
        block_size = self.runner.cache_config.block_size
        num_context = self._dflash_num_context
        num_reqs = cad.num_reqs

        # Per-context-token request id, and the token's original absolute
        # position. This mirrors upstream copy_and_expand_dflash_inputs_kernel,
        # which uses target_positions for both RoPE and context slot mapping.
        qsl = cad.query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
        counts = qsl[1:] - qsl[:-1]
        req_of = torch.repeat_interleave(torch.arange(num_reqs), counts)
        within = torch.arange(num_context) - qsl[:-1][req_of]
        positions = self._context_positions_cpu_buffer[:num_context].to(torch.int64)

        block_table = cad.block_table_tensor
        max_blocks = block_table.shape[-1]
        block_num = (positions // block_size).clamp(max=max_blocks - 1)
        block_id = block_table[req_of, block_num].to(torch.int64)

        group_slots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        layer_group: dict[int, int] = {}

        layer_idx = 0
        for gid, (attn_group, attn_metadata) in enumerate(per_group_metadata):
            local_tables = getattr(attn_metadata, "local_block_tables", None)
            cache_seq_lens = getattr(attn_metadata, "cache_seq_lens", None)
            if local_tables is not None and cache_seq_lens is not None:
                # Sliding-window group: one window-sized block per request, and
                # the op keeps it left-aligned (see the note above).
                blk = local_tables.cpu().view(-1)[req_of]
                off = cache_seq_lens.cpu().view(-1)[req_of] + within
            else:
                blk = block_id
                off = positions % block_size
            group_slots[gid] = (blk.to(torch.int64), off.to(torch.int64))
            for _ in attn_group.layer_names:
                layer_group[layer_idx] = gid
                layer_idx += 1

        return group_slots, layer_group, req_of

    # ------------------------------------------------------------------
    # Drafting
    # ------------------------------------------------------------------

    def _rewind_rejected_tokens(
        self,
        cad: CommonAttentionMetadata,
        num_rejected_tokens: torch.Tensor | None,
    ) -> None:
        """Strip just-rejected tokens from `seq_lens`, exactly once.

        The drafter batch arrives padded to the full speculation width, so
        `seq_lens` still counts the tokens the target just rejected. Everything
        downstream keys off it -- the block's query positions, its cache write
        offset, the mask window -- so strip the rejects first. Upstream's
        kernel does the equivalent by walking back to the last accepted
        position (`valid_ctx_end = ctx_end - num_rejected`).
        """
        if self.num_speculative_tokens <= 1 or num_rejected_tokens is None:
            return
        seq_lens = cad.seq_lens
        seq_lens_cpu = cad._seq_lens_cpu
        seq_lens -= num_rejected_tokens.to(seq_lens.device, seq_lens.dtype)
        # The flash-attention builder reads the HOST shadow, so it has to see
        # the same correction -- but on this stack the two fields are views of
        # ONE tensor: rbln_model_runner.py:920-921 slices the plain CPU
        # `self.seq_lens` twice, and prepare_inputs_padded passes both through
        # unchanged. Subtracting from both, the way eagle.py:511-521 does,
        # therefore applies the correction TWICE to the same storage. Only
        # touch the shadow when it is genuinely a separate buffer.
        if seq_lens_cpu is not None and seq_lens_cpu.data_ptr() != seq_lens.data_ptr():
            seq_lens_cpu -= num_rejected_tokens.to(
                seq_lens_cpu.device, seq_lens_cpu.dtype
            )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs=None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        """One non-causal forward produces every draft token."""
        self._rewind_rejected_tokens(common_attn_metadata, num_rejected_tokens)

        num_tokens, token_indices = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
        )

        assert self.runner is not None
        is_prefill = _DRAFT_IS_PREFILL
        num_reqs = self.runner.input_batch.num_reqs

        num_reqs_padded, num_padded_tokens, num_tokens_across_dp = (
            self._determine_draft_batch_padding(num_reqs, num_tokens, is_prefill)
        )

        block_start_positions = self._block_start_positions(common_attn_metadata)

        per_group_metadata = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=block_start_positions,
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            per_group_metadata.append((attn_group, attn_metadata))
            per_layer_attn_metadata.update(
                self._specialize_layer_attn_metadata(
                    attn_group,
                    attn_metadata,
                    common_attn_metadata,
                    num_reqs,
                    num_reqs_padded,
                )
            )

        # Project the target's hidden states into the drafter's KV cache. This
        # is a compiled graph (patches/qwen3_dflash.py): upstream runs it eager,
        # which on RBLN would put the whole context projection on the dispatch
        # path every step and swamp the single-forward advantage.
        group_slots, layer_group, request_ids = self._resolve_group_slots(
            per_group_metadata, common_attn_metadata
        )
        # Create-or-fetch, never getattr: the helper used to be built lazily
        # inside precompute_and_store_context_kv, so looking for it here always
        # found None and the slots were never handed over.
        helper = get_or_create_context_kv(self.model.model)
        helper.set_compile_context(self.runner.compile_context)
        helper.set_group_slots(
            group_slots,
            layer_group,
            request_ids,
            partition_size=self.runner.cache_config.block_size,
        )
        # Project the aux-layer concatenation down to one hidden width before
        # the precompute. With ``use_aux_hidden_state`` the runner hands over
        # six target layers concatenated to 18432 features, while the drafter's
        # RMSNorm and KV projections consume 3072. Upstream keeps this FC out of
        # ``precompute_and_store_context_kv``; retain that contract here.
        assert self._hidden_state_combiner is not None
        self.model.precompute_and_store_context_kv(
            self._hidden_state_combiner(self._dflash_hidden_states),
            self._context_positions_buffer[: self._dflash_num_context],
            WRITE_CONTEXT_KV,
        )

        intermediate_drafts = self._intermediate_prefill_drafts(num_reqs)
        if intermediate_drafts is not None:
            return intermediate_drafts

        assert common_attn_metadata._seq_lens_cpu is not None
        page_crossing = _dflash_page_crossing_mask(
            common_attn_metadata._seq_lens_cpu[:num_reqs],
            self.runner.cache_config.block_size,
            1 + self.num_speculative_tokens,
        )
        empty_drafts = _empty_drafts_for_page_crossing(page_crossing)
        if empty_drafts is not None:
            # Context K/V above is still required: this target-only step is what
            # advances the request toward the next aligned page. Do not return a
            # zero-filled tensor; the scheduler would treat those zeros as valid
            # draft token IDs.
            return empty_drafts

        # Same reshape/pad path the warmup used, so the compiled shape matches
        # and the first real step does not trigger a recompile.
        input_ids, positions, _, token_indices_padded = self._preprocess(
            num_reqs, num_reqs_padded, num_tokens, token_indices, is_prefill
        )
        # `_preprocess` pads the sampling indices to the padded BATCH, which is
        # EAGLE's shape: one sampled slot per request. DFlash samples every mask
        # token, so its padded length is num_spec times that. `pad` never
        # shrinks, so the EAGLE-shaped call above is a no-op here rather than a
        # truncation -- but it also leaves the tensor short of the padded batch.
        if token_indices_padded is not None:
            token_indices_padded = pad(
                token_indices_padded, 0, num_reqs_padded * self.num_speculative_tokens
            )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            _, out = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                token_indices_to_sample=token_indices_padded,
            )

        draft_token_ids = self._draft_ids(out, num_reqs * self.num_speculative_tokens)
        return draft_token_ids.view(num_reqs, self.num_speculative_tokens)
