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

Most call sites still read `envs.py` directly. They move over one subsystem
at a time.
"""

import argparse
import os
from dataclasses import field, fields
from typing import TYPE_CHECKING, Any, Literal

from vllm.config.utils import config as vllm_config_dataclass

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

_GROUP_TITLE = "RBLNConfig"

DecodeBatchBucketStrategy = Literal["exponential", "linear", "manual"]


@vllm_config_dataclass
class RBLNConfig:
    """RBLN NPU options for the vLLM-native model path."""

    num_devices_per_local_rank: int = 1
    """Number of NPU devices assigned to each local rank."""

    sampler: bool = True
    """Use the customized RBLN sampler."""

    compile_model: bool = True
    """Compile models with torch.compile. Otherwise run CPU eager mode, if
    possible."""

    compile_strict_mode: bool = False
    """Compile with torch.compile's strict mode, which fails on a graph break
    instead of falling back to eager."""

    num_hidden_layers: int = 0
    """Build only the first N decoder layers and leave the rest as
    `PPMissingLayer`, to cut compile time during bring-up. 0 disables the
    truncation."""

    enforce_model_fp32: bool = False
    """Force the model dtype to fp32 instead of model_config.dtype."""

    use_dynamic_kv_cache: bool = False
    """Size the KV cache from the compiled artifact instead of the estimate."""

    flash_causal_attn: bool = True
    """Use flash attention for causal attention."""

    batch_attn_opt: bool = False
    """Use the batch attention optimization for paged attention."""

    use_custom_kernel: bool = False
    """Use the custom RBLN kernels."""

    sort_batch: bool = False
    """Sort requests within a batch before the forward pass."""

    sub_block_cache: bool = True
    """Enable sub-block prefix caching. The sub-block size equals
    max_num_batched_tokens (the prefill chunk size)."""

    specialize_moe_decode: bool = True
    """Specialize the case where every instance is at the decode stage."""

    use_moe_tokens_mask: bool = True
    """Apply the tokens mask to the MoE expert kernel."""

    dispatch_all2all: bool = False
    """Use all2all dispatch instead of all-gather for MoE DP dispatch."""

    combine_all2all: bool = False
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

    nixl_swa_view_opt: bool = False
    """Publish a second SWA-sized descriptor range alongside the Full-sized
    range at the same NIXL base addresses, so SWA groups transfer only
    `sliding_window` bytes per block over RDMA."""

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
            # the bundle. The rest change what runs, not what is built.
            "sampler",
            "compile_strict_mode",
            "sort_batch",
            "sub_block_cache",
            "nixl_swa_view_opt",
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


# `vllm_config_dataclass` is a `dataclass_transform`, but the mypy hook runs
# without vllm installed, so it cannot see that this makes a dataclass.
_FIELDS = fields(RBLNConfig)  # type: ignore[arg-type]


# Which env name means "the user set this field". It is VLLM_RBLN_<FIELD>
# unless listed here.
_ENV_PROBE: dict[str, tuple[str, ...]] = {
    # `envs.py` still honors the deprecated VLLM_RBLN_TP_SIZE alias.
    "num_devices_per_local_rank": (
        "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",
        "VLLM_RBLN_TP_SIZE",
    ),
    "use_custom_kernel": ("RBLN_USE_CUSTOM_KERNEL",),
}


def _env_overrides() -> dict[str, Any]:
    from vllm_rbln import envs

    overrides: dict[str, Any] = {}
    for f in _FIELDS:
        env_name = f"VLLM_RBLN_{f.name.upper()}"
        for probe in _ENV_PROBE.get(f.name, (env_name,)):
            if probe in os.environ:
                overrides[f.name] = getattr(envs, env_name)
                break
    return overrides


def build_rbln_config(additional_config: Any = None) -> RBLNConfig:
    """Resolve the RBLN config from `additional_config` and the environment.

    An `RBLNConfig` is returned unchanged, so a process that receives one
    cannot resolve it into something different.
    """
    if isinstance(additional_config, RBLNConfig):
        return additional_config

    given: dict[str, Any] = additional_config or {}
    if not isinstance(given, dict):
        raise ValueError(
            "additional_config must be an RBLNConfig or a mapping of its field "
            f"names on the vLLM-native path, got {type(given).__name__}"
        )

    known = {f.name for f in _FIELDS}
    if unknown := sorted(set(given) - known):
        # `extra="forbid"` would catch these too, but its message talks about
        # keyword arguments. Upstream's --gdn-prefill-backend arrives this way:
        # arg_utils writes it into additional_config.
        raise ValueError(
            f"additional_config takes only RBLNConfig fields on the "
            f"vLLM-native path, and {unknown} are not fields. The fields are "
            f"{sorted(known)}."
        )

    overrides = _env_overrides()
    shadowed = sorted(set(given) & set(overrides))
    overrides.update(given)

    if shadowed:
        logger.warning_once(
            "Ignoring the environment variables for %s; the CLI value wins.",
            ", ".join(shadowed),
        )

    return RBLNConfig(**overrides)


_rbln_config: RBLNConfig | None = None


def set_rbln_config(config: RBLNConfig) -> None:
    """Publish the resolved config for this process.

    Each process does this at its own entry point. A worker and EngineCore
    receive an already-built `VllmConfig`, so its `__post_init__` -- where the
    platform hook runs -- does not run again there.
    """
    global _rbln_config
    _rbln_config = config

    defaults = RBLNConfig()
    changed = {
        f.name: getattr(config, f.name)
        for f in _FIELDS
        if getattr(config, f.name) != getattr(defaults, f.name)
    }
    logger.info("RBLN config: %s", changed or "all defaults")


def get_rbln_config() -> RBLNConfig:
    """The resolved RBLN config for this process.

    There is deliberately no fallback to the environment. A child process
    inherits env vars but not `--rbln-*` values, so a fallback would be right
    when the option came from the environment and wrong when it came from the
    command line.
    """
    if _rbln_config is None:
        raise RuntimeError(
            "RBLNConfig was never resolved in this process. Call "
            "set_rbln_config(build_rbln_config(vllm_config.additional_config)) "
            "from this process's entry point."
        )
    return _rbln_config


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
    """Add the `RBLNConfig` group to `parser`. Safe to call twice.

    `RblnPlatform.pre_register_and_update(parser)` calls this from inside
    `AsyncEngineArgs.add_cli_args()`, before `parse_args()`. That is early
    enough for `--help`, `--help=all` and `--help=rblnconfig`.
    """
    if any(group.title == _GROUP_TITLE for group in parser._action_groups):
        return

    from vllm.engine.arg_utils import get_kwargs

    group = parser.add_argument_group(
        title=_GROUP_TITLE,
        description=RBLNConfig.__doc__,
    )
    kwargs = get_kwargs(RBLNConfig)
    for f in _FIELDS:
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
