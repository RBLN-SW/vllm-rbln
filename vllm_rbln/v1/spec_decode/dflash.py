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

from dataclasses import replace

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

import vllm_rbln.envs as envs
from vllm_rbln.compilation import build_process_group_dict, compile
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.patches.qwen3_dflash import (
    WRITE_CONTEXT_KV,
    _DFlashForwardGraph,
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
        super().__init__(vllm_config, device, runner)

        # Only the bonus token and the mask tokens are queries; everything else
        # is context that reaches the drafter through its KV cache, so the query
        # buffers stay tiny compared with EAGLE3's.
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)

        # Filled by set_inputs_first_pass. Upstream's fused Triton kernel also
        # emits a context slot mapping; here the equivalent coordinates are
        # recomputed in `_resolve_group_slots` from the block table, because
        # `cad.slot_mapping` is a 0-dim dummy on this stack and carries nothing
        # (DFLASH-PORT-DESIGN.md section 17).
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
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
        """
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
    ) -> None:
        """Replace the decode mask with one shaped for a whole draft block.

        The shared builder writes its non-causal decode mask as
        `[batch, 1, 1, 1, max_seq_len]` -- query length one, because ordinary
        decode emits one token per request. DFlash queries `1 + num_spec`
        positions at once (bonus token plus the mask tokens), so that mask is a
        rank-matching failure inside the attention kernel:

            InferShape.cpp set_nontiled_params:
              mask_dims.at(i) == input_dims.at(i) not satisfied
            Cannot find valid tiling, op=rtosa.flash_attn_tile

        Rebuild it at the block's query width. Every query slot sees the same
        keys, and the window is opened `num_query_per_req` past the context so
        the block's own K/V -- written into the cache by this step -- is visible
        to all of its slots, which is what makes the drafting non-causal WITHIN
        the block as DFlash requires.

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
        if mask.shape[-2] == num_query_per_req:
            return

        max_seq_len = mask.shape[-1]
        seq_lens = cad._seq_lens_cpu
        assert seq_lens is not None, "flash-attention builder needs the host shadow"
        # The builder opens keys [0, seq_len] for its single query row. The block
        # adds num_query_per_req - 1 more positions, all of which every slot in
        # the block must see -- that is what makes the drafting non-causal within
        # the block.
        window = (seq_lens[:num_reqs] + num_query_per_req).view(-1, 1)
        key_pos = torch.arange(max_seq_len).view(1, -1)
        valid = (key_pos < window).to(mask.dtype)

        new_mask = valid.view(num_reqs, 1, 1, 1, max_seq_len).expand(
            num_reqs, 1, 1, num_query_per_req, max_seq_len
        )
        if num_reqs_padded > num_reqs:
            new_mask = pad(new_mask, 0, num_reqs_padded)
        attn_metadata.attn_masks = new_mask.contiguous().to(mask.device)

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
        super(RBLNEagleProposer, self).load_model(target_model)
        self._probe_dp_rendezvous_need()

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

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = dflash_wrapper
        else:
            runtime_holder: list = []
            compiled = compile(
                dflash_wrapper,
                dynamic=False,
                fullgraph=True,
                compile_context=self.runner.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=build_process_group_dict(),
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                runtime_holder=runtime_holder,
                # Four DP workers compile the same two DFlash shapes during
                # warmup. The rebel cache writer does not serialize writes to
                # a shared cache path, so concurrent cache misses can leave a
                # checksum-invalid .rbln file. Keep this small, DFlash-only
                # graph process-local; target-model caching remains enabled.
                use_cache=False,
            )
            self.model_executable = _DFlashForwardGraph(compiled, runtime_holder)

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
            self._rebuild_block_draft_mask(
                attn_metadata, common_attn_metadata, num_reqs, num_reqs_padded
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

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
        self._context_positions_buffer[:num_context].copy_(
            target_positions[:num_context]
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

    def build_per_group_and_layer_attn_metadata(self, cad, draft_index: int = 0):
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer.items():
            causal = getattr(attn_metadata, "causal", None)
            if causal:
                raise RuntimeError(
                    f"DFlash requires non-causal attention but layer {layer_name} "
                    "reports causal=True. Check that the draft vllm_config carries "
                    "attention_config.use_non_causal=True."
                )
        return per_group, per_layer

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

        # Per-context-token request id, and the token's absolute position.
        qsl = cad.query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
        counts = qsl[1:] - qsl[:-1]
        req_of = torch.repeat_interleave(torch.arange(num_reqs), counts)
        within = torch.arange(num_context) - qsl[:-1][req_of]
        seq_lens = cad._seq_lens_cpu[:num_reqs].to(torch.int64)
        # The scheduled tokens are the tail of each request's sequence.
        positions = seq_lens[req_of] - counts[req_of] + within

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
    ) -> torch.Tensor:
        """One non-causal forward produces every draft token."""
        # The drafter batch arrives padded to the full speculation width, so
        # `seq_lens` still counts the tokens the target just rejected. Everything
        # downstream keys off it -- the block's query positions, its cache write
        # offset, the mask window -- so strip the rejects first. Upstream's
        # kernel does the equivalent by walking back to the last accepted
        # position (`valid_ctx_end = ctx_end - num_rejected`).
        if self.num_speculative_tokens > 1 and num_rejected_tokens is not None:
            seq_lens = common_attn_metadata.seq_lens
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            seq_lens -= num_rejected_tokens.to(seq_lens.device, seq_lens.dtype)
            # The flash-attention builder reads the HOST shadow, so it has to see
            # the same correction -- but on this stack the two fields are views of
            # ONE tensor: rbln_model_runner.py:920-921 slices the plain CPU
            # `self.seq_lens` twice, and prepare_inputs_padded passes both through
            # unchanged. Subtracting from both, the way eagle.py:511-521 does,
            # therefore applies the correction TWICE to the same storage. Only
            # touch the shadow when it is genuinely a separate buffer.
            if (
                seq_lens_cpu is not None
                and seq_lens_cpu.data_ptr() != seq_lens.data_ptr()
            ):
                seq_lens_cpu -= num_rejected_tokens.to(
                    seq_lens_cpu.device, seq_lens_cpu.dtype
                )

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
            self._rebuild_block_draft_mask(
                attn_metadata, common_attn_metadata, num_reqs, num_reqs_padded
            )
            per_group_metadata.append((attn_group, attn_metadata))
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

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
        # the precompute. With `use_aux_hidden_state` the runner hands over
        # `cat([h for h in aux], dim=-1)` -- 6 aux layers x 3072 = 18432 here --
        # but `precompute_and_store_context_kv` normalises with
        # `hidden_norm = RMSNorm(config.hidden_size)` and projects with
        # `_fused_kv_weight[..., hidden_size]`, both of which are 3072 wide.
        # Upstream keeps that projection OUT of the precompute, in
        # `combine_hidden_states` (qwen3_dflash.py), so the caller has to apply
        # it -- eagle.py does exactly this in three places. Without it the very
        # first real request dies in the precompute graph with
        # `mul(FakeTensor(64, 18432), FakeTensor(3072))`.
        # Left eager on purpose: this repo already dropped the aux-projection
        # graph because deploy mode makes eager faster.
        self.model.precompute_and_store_context_kv(
            self.model.combine_hidden_states(self._dflash_hidden_states),
            self._context_positions_buffer[: self._dflash_num_context],
            WRITE_CONTEXT_KV,
        )

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
