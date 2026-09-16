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

"""RBLN options for the vLLM-native model path.

On this path the config *is* `VllmConfig.additional_config`, which
`check_and_update_config` replaces with the resolved object. Being a
`VllmConfig` field is what carries it to every worker in the config pickle,
and what makes `VllmConfig.compute_hash()` call our `compute_hash`.
`platform.py` gates all of it on `VLLM_RBLN_USE_VLLM_MODEL=1`.

Resolution order, highest first:

  1. `additional_config`, an `RBLNConfig` or a dict of field names. The
     `--rbln-*` flags write into it.
  2. `VLLM_RBLN_<FIELD>`, read through `envs.py` so the parsing there still
     applies. `_ENV_PROBE` lists the names that break the pattern.
  3. the field default

`envs.py` still carries the variables and their parsing. The options that stay
there rather than move -- a patch condition reads two of them before this config
exists, and the rest are bring-up knobs -- are not fields here.
"""

import argparse
import os
from dataclasses import Field, field, fields
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from vllm.config.utils import config as vllm_config_dataclass

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

_GROUP_TITLE = "RBLNConfig"

# No `--rbln-*` flag; the field's own docstring says why. Still accepted through
# `additional_config`, which is how one of them arrives before the config exists.
_NO_FLAG = {"no_flag": True}

# Written by the config sync, from the compiled artifact or from other fields. A
# value given here is overwritten, and hashing it would key the compile cache on
# something derived from the artifact the key is looking for.
_DERIVED = {"no_flag": True, "derived": True}

DecodeBatchBucketStrategy = Literal["exponential", "linear", "manual"]


@vllm_config_dataclass
class RBLNConfigBase:
    """RBLN NPU options that are not specific to one model path."""

    num_devices_per_local_rank: int = 1
    """Number of NPU devices assigned to each local rank."""

    use_custom_sampler: bool = True
    """Use the customized RBLN sampler."""


@vllm_config_dataclass
class RBLNConfig(RBLNConfigBase):
    """RBLN NPU options for the vLLM-native model path."""

    compile_model: bool = True
    """Compile models with torch.compile. Otherwise run CPU eager mode, if
    possible."""

    enforce_model_fp32: bool = False
    """Force the model dtype to fp32 instead of model_config.dtype."""

    use_flash_causal_attn: bool = True
    """Use flash attention for causal attention."""

    use_batch_attn_opt: bool = False
    """Use the batch attention optimization for paged attention."""

    use_custom_kernel: bool = False
    """Use the custom RBLN kernels."""

    enable_sub_block_cache: bool = True
    """Enable sub-block prefix caching. The sub-block size equals
    max_num_batched_tokens (the prefill chunk size)."""

    specialize_moe_decode: bool = True
    """Specialize the case where every instance is at the decode stage."""

    use_moe_tokens_mask: bool = True
    """Apply the tokens mask to the MoE expert kernel."""

    use_all2all_dispatch: bool = False
    """Use all2all dispatch instead of all-gather for MoE DP dispatch."""

    use_all2all_combine: bool = False
    """Use all2all combine instead of reduce-scatter for MoE DP combine."""

    decode_batch_bucket_strategy: DecodeBatchBucketStrategy = "exponential"
    """How the decode batch buckets are laid out."""

    decode_batch_bucket_min: int = 1
    """Smallest decode batch bucket."""

    decode_batch_bucket_step: int = 2
    """Step between decode batch buckets."""

    decode_batch_bucket_limit: int = 1
    """Largest decode batch bucket."""

    decode_batch_bucket_manual_buckets: list[int] = field(default_factory=list)
    """Explicit decode batch sizes, used when the strategy is `manual`."""

    use_w8a8: bool = False
    """Opt in to W8A8. W8A16 runs on every RBLN NPU, W8A8 only on the ones
    whose kernels take an fp8 activation."""

    def compute_hash(self) -> str:
        """Hash of the fields that change the compiled artifact.

        `VllmConfig.compute_hash()` calls this and `mega_cache` uses that for
        its bundle key, so changing a field listed below keeps the compiled
        graphs.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors = {
            # Sampler graphs compile with use_cache=False, so they never enter
            # the bundle. Sub-block caching changes what runs, not what is built.
            "use_custom_sampler",
            "enable_sub_block_cache",
        }
        return hash_factors(get_hash_factors(self, ignored_factors))

    def __post_init__(self) -> None:
        buckets = self.decode_batch_bucket_manual_buckets
        if any(b <= 0 for b in buckets):
            raise ValueError("decode_batch_bucket_manual_buckets must all be > 0")
        if len(buckets) != len(set(buckets)):
            raise ValueError("decode_batch_bucket_manual_buckets must be unique")
        if self.decode_batch_bucket_strategy == "manual" and not buckets:
            raise ValueError(
                "decode_batch_bucket_strategy='manual' needs at least one entry "
                "in decode_batch_bucket_manual_buckets"
            )


@vllm_config_dataclass
class OptimumRBLNConfig(RBLNConfigBase):
    """RBLN NPU options for the optimum-rbln model path."""

    optimum_overrides: dict[str, Any] = field(default_factory=dict)
    """Entries for optimum-rbln's own model config, laid over what vllm-rbln
    derives from the vLLM settings when the model is compiled. With a
    pre-compiled model only the `device` entries apply."""

    prefix_block_size: int | None = None
    """Block size of the prefix cache. Defaults to the prefill chunk size."""

    # Snapshots of a vLLM field taken before it is overwritten. vLLM already has
    # the flag, so there is no `--rbln-*` one, but they stay settable: the
    # platform hook writes the first into the dict before this class exists.
    user_max_num_batched_tokens: int | None = field(default=None, metadata=_NO_FLAG)
    """`--max-num-batched-tokens` as the user gave it, before vLLM fills in its
    default. On this path it is the prefill chunk size to compile."""

    num_blocks_override: int | None = field(default=None, metadata=_NO_FLAG)
    """`--num-gpu-blocks-override` as given, before the prefix-cache block ratio
    is applied to it."""

    # Written by the sync, and carried to the processes that cannot build
    # `RBLNParams` of their own.
    attn_block_size: int | None = field(default=None, metadata=_DERIVED)
    """`RBLNParams.kvcache_block_size`, when prefix caching splits the KV-cache
    block size from `cache_config.block_size`."""

    image_prefill_chunk_size: list[int] | None = field(default=None, metadata=_DERIVED)
    """`RBLNParams.image_prefill_chunk_size`, the image-prefill buckets
    (gemma3/gemma4) the scheduler pads against."""

    cached_model_path: str | None = field(default=None, metadata=_DERIVED)
    """Where the compile cache holds, or will put, this model's artifact. Built
    from the fields above, so it cannot key the cache it names."""

    num_blocks_synced: bool = field(default=False, metadata=_DERIVED)
    """Set once num_gpu_blocks is derived from the compiled model, so the second
    run of the sync in EngineCore does not derive it again."""

    def compute_hash(self) -> str:
        """Hash of the fields that change the compiled artifact.

        `VllmConfig.compute_hash()` requires this of an `additional_config` that
        is not a dict, and `mega_cache` keys its bundle on the result.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors = {
            f.name for f in _fields_of(type(self)) if f.metadata.get("derived")
        } | {
            # Sampler graphs compile with use_cache=False, so they never enter
            # the bundle.
            "use_custom_sampler",
        }
        return hash_factors(get_hash_factors(self, ignored_factors))


# Every class a `--rbln-*` flag can belong to.
_CONFIG_CLASSES: tuple[type[RBLNConfigBase], ...] = (RBLNConfig, OptimumRBLNConfig)

# TODO(vllm-rbln>=0.12.0): delete. Former additional_config keys, still accepted
# with a warning.
_RENAMED_KEYS: dict[type[RBLNConfigBase], dict[str, str]] = {
    OptimumRBLNConfig: {"rbln_config": "optimum_overrides"},
}

_C = TypeVar("_C", bound=RBLNConfigBase)


def _fields_of(cls: type[RBLNConfigBase]) -> tuple["Field[Any]", ...]:
    # `vllm_config_dataclass` is a `dataclass_transform`, but the mypy hook runs
    # without vllm installed, so it cannot see that this makes a dataclass.
    return fields(cls)  # type: ignore[arg-type]


# Which env name means "the user set this field". It is VLLM_RBLN_<FIELD>
# unless listed here.
# The envs.py attribute a renamed field reads. Goes away with the env vars.
_ENV_NAME: dict[str, str] = {
    "use_custom_sampler": "VLLM_RBLN_SAMPLER",
    "use_flash_causal_attn": "VLLM_RBLN_FLASH_CAUSAL_ATTN",
    "use_batch_attn_opt": "VLLM_RBLN_BATCH_ATTN_OPT",
    "use_all2all_dispatch": "VLLM_RBLN_DISPATCH_ALL2ALL",
    "use_all2all_combine": "VLLM_RBLN_COMBINE_ALL2ALL",
    "enable_sub_block_cache": "VLLM_RBLN_SUB_BLOCK_CACHE",
}

_ENV_PROBE: dict[str, tuple[str, ...]] = {
    # `envs.py` still honors the deprecated VLLM_RBLN_TP_SIZE alias.
    "num_devices_per_local_rank": (
        "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",
        "VLLM_RBLN_TP_SIZE",
    ),
    "use_custom_kernel": ("RBLN_USE_CUSTOM_KERNEL",),
}


def _env_source(field_name: str) -> tuple[str, tuple[str, ...]]:
    """The envs.py attribute a field reads, and the os.environ names that mean
    the user set it."""
    attr = _ENV_NAME.get(field_name, f"VLLM_RBLN_{field_name.upper()}")
    return attr, _ENV_PROBE.get(field_name, (attr,))


def _env_overrides(cls: type[RBLNConfigBase]) -> dict[str, Any]:
    from vllm_rbln import envs

    overrides: dict[str, Any] = {}
    for f in _fields_of(cls):
        attr, probes = _env_source(f.name)
        if attr not in envs.environment_variables:
            # No variable of its own; the CLI or the default is the only source.
            continue
        for probe in probes:
            if probe in os.environ:
                overrides[f.name] = getattr(envs, attr)
                break
    return overrides


def build_rbln_config(additional_config: Any = None) -> RBLNConfig:
    return _resolve(RBLNConfig, additional_config)


def _resolve(cls: type[_C], additional_config: Any) -> _C:
    """Resolve `cls` from `additional_config` and the environment.

    A resolved config is returned unchanged, so a process that receives one
    cannot resolve it into something different.
    """
    if isinstance(additional_config, cls):
        return additional_config

    given: dict[str, Any] = additional_config or {}
    if not isinstance(given, dict):
        raise ValueError(
            f"additional_config must be a {cls.__name__} or a mapping of its "
            f"field names, got {type(given).__name__}"
        )

    for old, new in _RENAMED_KEYS.get(cls, {}).items():
        if old in given:
            logger.warning_once(
                "additional_config[%r] is deprecated and will be removed in "
                "0.12.0; use %r.",
                old,
                new,
            )
            given = {k: v for k, v in given.items() if k != old} | {new: given[old]}

    known = {f.name for f in _fields_of(cls)}
    if unknown := sorted(set(given) - known):
        # `extra="forbid"` would catch these too, but its message talks about
        # keyword arguments. Upstream's --gdn-prefill-backend arrives this way:
        # arg_utils writes it into additional_config.
        raise ValueError(
            f"additional_config takes only {cls.__name__} fields, and "
            f"{unknown} are not fields. The fields are {sorted(known)}."
        )

    overrides = _env_overrides(cls)
    shadowed = sorted(set(given) & set(overrides))
    overrides.update(given)

    if shadowed:
        logger.warning_once(
            "Both the environment and additional_config set %s; %s takes the "
            "additional_config value.",
            ", ".join(shadowed),
            cls.__name__,
        )

    resolved = cls(**overrides)

    # Upstream's `non-default args` covers what the CLI was given, but not what
    # the environment resolved to, and `VllmConfig.__str__` leaves
    # additional_config out entirely. This is the only record of the values a
    # run actually used.
    defaults = cls()
    changed = {
        f.name: getattr(resolved, f.name)
        for f in _fields_of(cls)
        if getattr(resolved, f.name) != getattr(defaults, f.name)
    }
    logger.info("RBLN config: %s", changed or "all defaults")

    return resolved


def get_rbln_config() -> RBLNConfig:
    """The RBLN section of the config the current model is being built under.

    For code that cannot reach a `vllm_config` of its own -- a free function, or
    a constructor whose signature upstream owns. Every such call site runs
    inside `set_current_vllm_config`, which upstream opens around worker start-up,
    device init and model construction. Read `vllm_config.additional_config`
    directly wherever one is in hand.
    """
    from vllm.config import get_current_vllm_config

    rbln_config = get_current_vllm_config().additional_config
    if not isinstance(rbln_config, RBLNConfig):
        raise RuntimeError(
            "additional_config is not an RBLNConfig; "
            "check_and_update_config resolves it on the vLLM-native path, so "
            f"this is the optimum-rbln path or an unbuilt config: {rbln_config!r}"
        )
    return rbln_config


# `from_cli_args` only copies dataclass fields, so a `--rbln-*` flag cannot
# have an `EngineArgs` field of its own. Each one writes into
# `additional_config` instead, which is a real field. That is why the actions
# below exist instead of plain `store`.

_OWNED = "_rbln_additional_config_owned"


def _additional_config(namespace: argparse.Namespace) -> dict[str, Any]:
    """The namespace's `additional_config`, copied so we can write into it.

    argparse seeds the namespace with the `--additional-config` action's own
    default object. A reused parser would share that object between
    `parse_args()` calls, so copy it once before writing.
    """
    if not getattr(namespace, _OWNED, False):
        current = getattr(namespace, "additional_config", None)
        namespace.additional_config = dict(current) if isinstance(current, dict) else {}
        setattr(namespace, _OWNED, True)
    return namespace.additional_config


class _StoreRbln(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _additional_config(namespace)[self.dest.removeprefix("rbln_")] = values


class _StoreRblnBool(argparse.BooleanOptionalAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            _additional_config(namespace)[
                self.dest.removeprefix("rbln_")
            ] = not option_string.startswith("--no-")


class _MergeAdditionalConfig(argparse.Action):
    """Replacement for `--additional-config`'s own action: merge, don't clobber.

    `FlexibleArgumentParser.parse_args()` rewrites `--additional-config.x v`
    into one `--additional-config <json>` and appends it at the end of argv.
    A plain store action would then drop what the `--rbln-*` flags wrote.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not isinstance(values, dict):
            # Upstream allows a bare string here, so keep that.
            namespace.additional_config = values
            setattr(namespace, _OWNED, False)
            return
        _additional_config(namespace).update(values)


def add_rbln_cli_args(parser: "FlexibleArgumentParser") -> None:
    """Add every `--rbln-*` flag to `parser`. Safe to call twice.

    `RblnPlatform.pre_register_and_update(parser)` calls this from inside
    `AsyncEngineArgs.add_cli_args()`, before `parse_args()`. That is early
    enough for `--help`, `--help=all` and `--help=rblnconfig`, and too early to
    know the model path, so both paths' fields are registered. `_resolve`
    rejects a field the selected class does not have.
    """
    if any(group.title == _GROUP_TITLE for group in parser._action_groups):
        return

    from vllm.engine.arg_utils import get_kwargs

    group = parser.add_argument_group(
        title=_GROUP_TITLE,
        description="RBLN NPU options for both model paths.",
    )
    seen: set[str] = set()
    for cls in _CONFIG_CLASSES:
        kwargs = get_kwargs(cls)
        for f in _fields_of(cls):
            if f.name in seen:
                continue
            seen.add(f.name)
            if f.metadata.get("no_flag"):
                continue
            field_kwargs = kwargs[f.name]
            is_bool = field_kwargs.pop("action", None) is argparse.BooleanOptionalAction
            group.add_argument(
                f"--rbln-{f.name.replace('_', '-')}",
                dest=f"rbln_{f.name}",
                action=_StoreRblnBool if is_bool else _StoreRbln,
                **field_kwargs,
            )

    for action in parser._actions:
        if action.dest == "additional_config":
            action.__class__ = _MergeAdditionalConfig
            break
    else:
        logger.warning(
            "--additional-config not found on the parser; --rbln-* flags may be "
            "overwritten when --additional-config is also passed."
        )
