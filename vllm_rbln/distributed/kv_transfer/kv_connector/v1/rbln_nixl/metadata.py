# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the two sides settle before any of it reaches the wire.

``transfer_shape`` is a pure reduction both sides run over the same two config
objects, so neither carries a term of its own. It reads no chiplet geometry,
which is why it can run before upstream's ``__init__`` and why the scheduler,
which never sees that geometry, can run it at all. The rest is what a producer
advertises beyond upstream's ``NixlAgentMetadata``.

Peers pair by what each holds rather than by position, on two axes, and each
axis needs one thing upstream's struct does not carry: the layer names a shard
registered, and the chiplet geometry its regions expanded into. Both describe
the sender; the receiver derives its own side and matches.

Kept in a subclass so upstream's struct and its compatibility hash stay
untouched. Both ends are RBLN, so folding a private version tag into that hash
(``rbln_compat_hash``) is enough to keep peers speaking different schemas from
completing a handshake.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from vllm.config.utils import hash_factors
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqId
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_representative_spec_type,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

# Bump on any incompatible change to the RBLN metadata schema or semantics, as
# upstream does with ``NIXL_CONNECTOR_VERSION``. Folded into the NIXL compat
# hash so an RBLN peer on another schema fails the handshake cleanly -- both
# ends are RBLN; earlier bumps are `git log -L` on this line.
#   9: the unit a coverage range is counted in
RBLN_NIXL_CONNECTOR_VERSION: int = 9

# Prefix a push completion notification carries when it names the half-open
# range this write filled: ``RBLNS:<writer>:<lo>:<hi>:<per_block>:`` ahead of
# the message upstream builds, counting units of one consumer block divided
# by ``per_block`` (1 where a write never splits one). Left off where no
# single range describes the write, so a consumer must accept a bare message.
RBLN_COVERAGE_NOTIF_PREFIX: bytes = b"RBLNS:"


class KVSplitAxis(Enum):
    """Which axis the compiler cut a KV entry on across the chiplets.

    ``HEAD`` means an area holds some of the shard's KV heads over every token
    of a block; ``NON_HEAD`` means every head over some of the tokens.
    """

    HEAD = 0
    NON_HEAD = 1


@dataclass
class RblnNixlAgentMetadata(NixlAgentMetadata):
    """``NixlAgentMetadata`` + which layers and which slice of the KV cache
    this shard holds.

    New fields default to the single-shard, single-area values, so a blob decoded
    by upstream (which uses ``NixlAgentMetadata`` and ignores the extra fields)
    degrades to the shape upstream assumes.
    """

    pp_rank: int = 0
    pp_size: int = 1
    # Registered KV-cache layer names, ordered as kv_caches_base_addr / block_lens.
    registered_layer_names: list[str] = field(default_factory=list)
    # Physical areas one logical region expanded into, and how many of them are
    # DISTINCT rather than replicas (see `_slice_head_bounds`).
    kv_areas: int = 1
    kv_slices: int = 1
    # The default keeps a blob without this field meaning what versions 2 and 3
    # meant by the two counts above.
    kv_split_axis: KVSplitAxis = KVSplitAxis.HEAD
    # What a block holds (`_kv_per_block`). A process picks the layout, not the
    # build, so two peers off one build can differ and the version cannot tell
    # them apart. The default is the layout every version through 4 had.
    kv_per_block: int = 1
    # Tokens a block holds in the view the sliding-window kernel reads
    # (`_observe_swa_kernel_block`). 0 where the shard cuts no window range by
    # it, so a PP stage without such a group pairs with any peer. Two engines
    # whose runners chose differently cut a window range differently.
    swa_kernel_block: int = 0


class RblnNixlConnectorMetadata(NixlConnectorMetadata):
    """``NixlConnectorMetadata`` + the requests whose early write must be drained.

    Promoted from the instance upstream builds rather than constructed
    in its place: ``NixlBaseConnectorScheduler.build_connector_meta`` names the
    upstream type directly and offers no hook for a subclass. This struct stays
    inside one engine -- it never reaches a peer -- so it is not part of the
    handshake schema and does not move ``RBLN_NIXL_CONNECTOR_VERSION``.
    """

    def __init__(self) -> None:
        super().__init__()
        # Requests whose source blocks go back to the allocator without a lease
        # -- preempted, or finished on a non-terminal status. A write already
        # issued for them reads memory the next forward may overwrite.
        self.push_early_flush: set[ReqId] = set()
        # Blocks a streamed request will hold once its whole prompt is
        # computed. The consumer registered the tail of that, so where its
        # window begins can only be found from the total -- and the prefix
        # offered mid-stream is shorter than it.
        self.push_stream_total: dict[ReqId, int] = {}
        # Tokens of KV the offered prefix holds. Distinct from `valid_tokens`,
        # which is the request's final count and only exists once it is over:
        # an offer is a prefix of a request still being computed, and how much
        # of its last block is filled is what a write smaller than a block
        # needs to know.
        self.push_stream_tokens: dict[ReqId, int] = {}
        # Tokens of KV the offered block list holds, so a last block that is not
        # full can leave the areas above its final token behind. Absent where
        # the count is unknown, which keeps the whole block.
        self.valid_tokens: dict[ReqId, int] = {}

    @classmethod
    def promote(cls, base: NixlConnectorMetadata) -> "RblnNixlConnectorMetadata":
        meta = cls()
        meta.__dict__.update(base.__dict__)
        return meta


_T = TypeVar("_T")


def connector_option(
    vllm_config: "VllmConfig", key: str, default: _T, *, takes: type | None = None
) -> _T:
    """One of this connector's knobs, from ``--kv-transfer-config``.

    They live in ``kv_connector_extra_config`` rather than the environment
    because that is where vLLM puts a connector's own options, and because the
    environment is read for the mega-cache bundle key -- a transfer knob
    changes no compiled graph and has no business partitioning it.

    The type follows the default. That config arrives as JSON, so a bool and an
    int come through as themselves; anything else is a mistake worth naming
    here rather than coercing into a truthy string.

    A knob whose absence means something none of its values can mean passes
    ``None`` as the default and names its type in ``takes``, since the default
    no longer carries one.
    """
    value = vllm_config.kv_transfer_config.get_from_extra_config(key, default)
    if value is None and default is None:
        return value
    expected = takes or type(default)
    # `bool` is a subclass of `int`, so an int knob given `true` would pass an
    # isinstance check and then count as 1.
    wrong_type = (
        not isinstance(value, bool)
        if expected is bool
        else isinstance(value, bool) or not isinstance(value, expected)
    )
    if wrong_type:
        raise RuntimeError(
            f"RBLN NIXL: kv_connector_extra_config[{key!r}] is "
            f"{value!r}, but this knob takes a {expected.__name__}"
        )
    return value


def sliding_window_ratio(specs: list["KVCacheSpec"]) -> int | None:
    """How many windows tile a block, where a hybrid has a window to name.

    None where there is none -- no sliding window, or one as wide as the block.
    The ratio is what the window range divides a block length by, and its
    presence is what says a hybrid can be described by our own lists at all:
    without it `_compute_desc_ids` hands the whole request to upstream, whose
    list has room for neither that range nor the chunk range beside it.
    """
    ratios: set[int] = set()
    for spec in specs:
        if not isinstance(spec, SlidingWindowSpec):
            continue
        if spec.block_size % spec.sliding_window != 0:
            # Upstream's block table refuses this where the kernel addresses
            # the cache in windows; where it addresses whole blocks the engine
            # starts, and this is then the only place that sees a window no
            # granule can tile.
            raise RuntimeError(
                "RBLN NIXL: a window range cuts a block into windows, so a "
                f"{spec.sliding_window}-token window has to divide the "
                f"{spec.block_size}-token block this engine's manager leases."
            )
        ratios.add(spec.block_size // spec.sliding_window)
        if spec.block_size == spec.sliding_window:
            continue
        # Which granule the range names is read off the request's token count,
        # and that is where the window is only where it slides. This spec's
        # manager leases one block a request and the runner reads its first
        # granule, wherever the count points.
        if isinstance(spec, RBLNSlidingWindowSpec):
            raise RuntimeError(
                "RBLN NIXL: a window range needs a window that moves through "
                "its block, and this engine pins every one to the block's "
                "first kernel block. Turn off whichever of swa_window_mode, "
                "chunk_mode and push_stream asked for one."
            )
    if len(ratios) > 1:
        # The builder reads a group as windowed from its spec and then cuts it
        # by the one ratio this engine carries, so a group that tiles its block
        # differently would be named in another group's granules -- part of its
        # block, with the descriptor count unchanged.
        raise RuntimeError(
            "RBLN NIXL: every sliding-window group has to cut its block into "
            "the same number of kernel blocks, and this engine's groups cut it "
            f"{sorted(ratios)} ways."
        )
    return next((r for r in ratios if r != 1), None)


@dataclass(frozen=True)
class TransferShape:
    """What the knobs and the KV-cache groups settle, ahead of any geometry.

    Every field answers a question some branch used to ask a stand-in for, and
    the stand-in is what went wrong each time a new model arrived: how many
    groups there are was asked as "is this hybrid", whether a list carries a
    second range as "is the window ratio set". Ask the question.

    Derived by `transfer_shape` on both sides from the same two config objects,
    so the scheduler and the worker cannot disagree about any of it.
    """

    #: The knobs, as asked for.
    chunk_mode: bool
    wants_window: bool
    wants_stream: bool
    #: Whether a prefill's closed prefix leaves before the request ends.
    streams_prefix: bool
    #: How many kernel blocks tile a block, where the window can be viewed.
    window_ratio: int | None
    #: The group a token count, a chunk and a coverage range are counted in.
    #: None where the engine has no full-attention group to count in.
    counted_group: int | None
    #: Whether the model slides at all, which the parallelism guards read.
    has_swa: bool
    #: Whether every layer of every group registers the same region count,
    #: which upstream's PP refusal misreads as "hybrid".
    groups_uniform_per_layer: bool
    use_host_buffer: bool
    writes_into_peer: bool

    @property
    def has_window_range(self) -> bool:
        """Whether this engine's own descriptor list carries a second range."""
        return self.window_ratio is not None

    @property
    def owns_engine_lists(self) -> bool:
        """Whether this engine builds the whole-engine descriptor lists itself.

        Upstream names a region's block once, so a second range needs lists of
        our own. A hybrid needs them for a different reason -- a per-shard list
        names one KV group and a hybrid has two -- which is why writing part of
        a block is enough there, with or without a window range.
        """
        return self.has_window_range or (self.writes_part_of_a_block and self.has_swa)

    @property
    def writes_part_of_a_block(self) -> bool:
        """Whether a write may name less than a whole block."""
        return self.chunk_mode or self.streams_prefix

    @property
    def chunks_need_own_descriptors(self) -> bool:
        """Whether naming that part needs per-shard lists.

        A single-group engine cannot put one on the whole-engine lists, so it
        narrows per peer instead; a hybrid owns those lists and does not.
        """
        return self.writes_part_of_a_block and not self.owns_engine_lists

    @property
    def sends_token_count(self) -> bool:
        """Whether the worker needs a request's token count in the metadata.

        Chunk mode sizes the last block's chunks by it; window mode picks which
        granules of a block the window sits in. Block ids say neither.
        """
        return self.chunk_mode or self.wants_window or self.wants_stream


def transfer_shape(
    vllm_config: "VllmConfig",
    kv_cache_groups: list["KVCacheGroupSpec"],
    *,
    writes_into_peer: bool,
) -> TransferShape:
    """Settle everything the knobs and the groups decide, and nothing else.

    Runs before upstream's `__init__` on the worker and beside it on the
    scheduler, because none of its inputs is an attribute either of them sets.
    That is what lets one function answer for both sides.
    """
    specs = [g.kv_cache_spec for g in kv_cache_groups]
    has_swa = any(isinstance(spec, SlidingWindowSpec) for spec in specs)
    use_host_buffer = vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
    chunk_mode = connector_option(vllm_config, "chunk_mode", False)
    wants_window = connector_option(vllm_config, "swa_window_mode", False)
    wants_stream = connector_option(vllm_config, "push_stream", False)

    # Only the side that originates bytes into a peer sends a prefix early, so
    # the knob is inert on the other -- and a shape that let it through there
    # would put a window range on lists that never carry one. Host staging
    # holds no areas to put one on either.
    streams_prefix = bool(wants_stream) and writes_into_peer and not use_host_buffer
    # A write that names part of a block needs a list that can name it: one
    # group has the per-shard lists, and a hybrid owns the whole-engine ones
    # (`owns_engine_lists`). Groups that are neither have no list to stream in.
    if streams_prefix and len(specs) > 1 and not has_swa:
        streams_prefix = False
    # A second group is carried at a different time -- the offer takes the
    # full-attention one and the handover takes the window's block -- and
    # telling them apart on the wire needs the list a viewable window builds.
    # One group has nothing to tell apart.

    window_ratio = None
    if has_swa and wants_window:
        window_ratio = sliding_window_ratio(specs)

    # A sliding-window group holds one block whatever the prompt length, so
    # summing the groups -- or taking the first -- describes no request. One
    # group is the request's whether or not it slides.
    full = [
        g for g, spec in enumerate(specs) if not isinstance(spec, SlidingWindowSpec)
    ]
    counted_group = 0 if len(specs) == 1 else (full[0] if full else None)

    group_spec = specs[0] if len(specs) == 1 else None
    return TransferShape(
        chunk_mode=chunk_mode,
        wants_window=wants_window,
        wants_stream=wants_stream,
        streams_prefix=streams_prefix,
        window_ratio=window_ratio,
        counted_group=counted_group,
        has_swa=has_swa,
        groups_uniform_per_layer=isinstance(group_spec, UniformTypeKVCacheSpecs)
        and issubclass(get_representative_spec_type(group_spec), FullAttentionSpec),
        use_host_buffer=use_host_buffer,
        writes_into_peer=writes_into_peer,
    )


def rbln_compat_hash(
    base_hash: str,
    *,
    writes_into_peer: bool,
    speculative_config: "SpeculativeConfig | None" = None,
) -> str:
    """Fold the RBLN schema version, the transfer direction and the draft model
    into the upstream NIXL compat hash.

    An extension rather than a change to ``compute_nixl_compatibility_hash``,
    which stays upstream's. The direction belongs in it because the read and the
    write path move bytes by protocols that do not meet: a producer that writes
    into a consumer expecting to read finds a peer whose every length check
    passes. This vLLM hashes nothing that separates them -- the connector name
    is not a factor -- so this is the only place it can be settled.

    The draft model belongs in it because its attention layers are members of
    the KV cache this connector registers, while upstream's factors describe
    the target alone. Model-level values only -- the hash is compared across
    every shard, so a per-rank quantity would differ between PP stages.
    """
    factors: dict[str, object] = {
        "base": base_hash,
        "rbln_nixl_connector_version": RBLN_NIXL_CONNECTOR_VERSION,
        "rbln_writes_into_peer": writes_into_peer,
    }
    if speculative_config is not None and (
        speculative_config.use_eagle() or speculative_config.uses_draft_model()
    ):
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        factors |= {
            "rbln_spec_method": speculative_config.method,
            "rbln_draft_model": draft_model_config.model,
            "rbln_draft_revision": draft_model_config.revision,
            "rbln_draft_code_revision": draft_model_config.code_revision,
        }
    return hash_factors(factors)
