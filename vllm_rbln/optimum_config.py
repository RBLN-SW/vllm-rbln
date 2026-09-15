# Copyright 2026 Rebellions Inc. All rights reserved.
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

"""RBLN options for the optimum-rbln model path.

On this path the config *is* `VllmConfig.additional_config`, which
`check_and_update_config` replaces with the resolved object. Being a
`VllmConfig` field is what carries it to every worker in the config pickle,
and what makes `VllmConfig.compute_hash()` call our `compute_hash`.
`platform.py` gates all of it on `VLLM_RBLN_USE_VLLM_MODEL` being unset or 0.

`config.py` is the same thing for the vLLM-native path. Neither file imports
the other: a path takes the options it reads and nothing else.

Resolution order, highest first:

  1. `additional_config`, an `OptimumRBLNConfig` or a dict of field names. The
     `--rbln-*` flags write into it.
  2. `VLLM_RBLN_<FIELD>`, read through `envs.py` so the parsing there still
     applies, for the fields that have such a variable. `_ENV_PROBE` lists the
     names that break the pattern.
  3. the field default
"""

import argparse
import os
from dataclasses import field, fields
from typing import TYPE_CHECKING, Any

from vllm.config.utils import config as vllm_config_dataclass

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

# The flags are `--rbln-*` on either path, so the group keeps one name.
_GROUP_TITLE = "RBLNConfig"

_INTERNAL = {"internal": True}


@vllm_config_dataclass
class OptimumRBLNConfig:
    """RBLN NPU options for the optimum-rbln model path."""

    # ====================================================================
    # Given by the user
    # ====================================================================
    num_devices_per_local_rank: int = 1
    """Number of NPU devices assigned to each local rank. A pre-compiled model
    overrides this with the value it was compiled for."""

    sampler: bool = True
    """Use the customized RBLN sampler."""

    optimum_overrides: dict[str, Any] = field(default_factory=dict)
    """Entries for optimum-rbln's model config (its `rbln_config`), laid over
    what vllm-rbln derives from the vLLM settings when the model is compiled.
    With a pre-compiled model only the `device` entries apply."""

    prefix_block_size: int | None = None
    """Block size of the prefix cache. Defaults to the prefill chunk size."""

    # ====================================================================
    # Written by the platform hook and the config sync. Not user-facing, and
    # overwritten if given.
    # ====================================================================
    user_max_num_batched_tokens: int | None = field(default=None, metadata=_INTERNAL)
    """`--max-num-batched-tokens` as the user gave it, copied here by the
    platform hook before vLLM fills in its default. Under `optimum` it is the
    prefill chunk size to compile."""

    cached_model_path: str | None = field(default=None, metadata=_INTERNAL)
    """Where the compile cache holds, or will put, this model's artifact."""

    attn_block_size: int | None = field(default=None, metadata=_INTERNAL)
    """The KV-cache block size (`kvcache_block_size`), when prefix caching
    splits it from cache_config.block_size. Copied out of `rbln_config` or the
    artifact for the processes that have no RBLNParams of their own."""

    num_blocks_override: int | None = field(default=None, metadata=_INTERNAL)
    """cache_config.num_gpu_blocks_override as given, before the prefix-cache
    block ratio is applied."""

    num_blocks_synced: bool = field(default=False, metadata=_INTERNAL)
    """Set once num_gpu_blocks is derived from the compiled model, so the
    second run of the sync in EngineCore does not derive it again."""

    image_prefill_chunk_size: list[int] | None = field(default=None, metadata=_INTERNAL)
    """Image-prefill buckets (gemma3/gemma4), read by the scheduler, which has
    no RBLNParams of its own."""

    def compute_hash(self) -> str:
        """Hash of the fields that change the compiled artifact.

        `VllmConfig.compute_hash()` calls this, so a field ignored below leaves
        an already compiled model valid.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        # The sampler changes what runs, not what optimum-rbln builds.
        return hash_factors(get_hash_factors(self, {"sampler"}))


# `vllm_config_dataclass` is a `dataclass_transform`, but the mypy hook runs
# without vllm installed, so it cannot see that this makes a dataclass.
_FIELDS = fields(OptimumRBLNConfig)  # type: ignore[arg-type]


# TODO(vllm-rbln>=0.12.0): delete. Former additional_config keys, still accepted
# with a warning.
_RENAMED_KEYS = {"rbln_config": "optimum_overrides"}


# Which env name means "the user set this field". It is VLLM_RBLN_<FIELD>
# unless listed here.
_ENV_PROBE: dict[str, tuple[str, ...]] = {
    # `envs.py` still honors the deprecated VLLM_RBLN_TP_SIZE alias.
    "num_devices_per_local_rank": (
        "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",
        "VLLM_RBLN_TP_SIZE",
    ),
}


def _env_overrides() -> dict[str, Any]:
    from vllm_rbln import envs

    overrides: dict[str, Any] = {}
    for f in _FIELDS:
        env_name = f"VLLM_RBLN_{f.name.upper()}"
        if env_name not in envs.environment_variables:
            continue
        for probe in _ENV_PROBE.get(f.name, (env_name,)):
            if probe in os.environ:
                overrides[f.name] = getattr(envs, env_name)
                break
    return overrides


def build_optimum_rbln_config(additional_config: Any = None) -> OptimumRBLNConfig:
    """Resolve the RBLN config from `additional_config` and the environment.

    An `OptimumRBLNConfig` is returned unchanged, so a process that receives
    one cannot resolve it into something different.
    """
    if isinstance(additional_config, OptimumRBLNConfig):
        return additional_config

    given: dict[str, Any] = additional_config or {}
    if not isinstance(given, dict):
        raise ValueError(
            "additional_config must be an OptimumRBLNConfig or a mapping of its "
            f"field names on the optimum-rbln path, got {type(given).__name__}"
        )

    for old, new in _RENAMED_KEYS.items():
        if old in given:
            logger.warning_once(
                "additional_config[%r] is deprecated and will be removed in "
                "0.12.0; use %r.",
                old,
                new,
            )
            given = {**given, new: given[old]}
            del given[old]

    known = {f.name for f in _FIELDS}
    if unknown := sorted(set(given) - known):
        # `extra="forbid"` would catch these too, but its message talks about
        # keyword arguments. Upstream's --gdn-prefill-backend arrives this way:
        # arg_utils writes it into additional_config.
        raise ValueError(
            f"additional_config takes only OptimumRBLNConfig fields on the "
            f"optimum-rbln path, and {unknown} are not fields. The fields are "
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

    return OptimumRBLNConfig(**overrides)


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


def add_optimum_rbln_cli_args(parser: "FlexibleArgumentParser") -> None:
    """Add the `OptimumRBLNConfig` group to `parser`. Safe to call twice.

    `RblnPlatform.pre_register_and_update(parser)` calls this from inside
    `AsyncEngineArgs.add_cli_args()`, before `parse_args()`. That is early
    enough for `--help`, `--help=all` and `--help=rblnconfig`.
    """
    if any(group.title == _GROUP_TITLE for group in parser._action_groups):
        return

    from vllm.engine.arg_utils import get_kwargs

    group = parser.add_argument_group(
        title=_GROUP_TITLE,
        description=OptimumRBLNConfig.__doc__,
    )
    kwargs = get_kwargs(OptimumRBLNConfig)
    for f in _FIELDS:
        if f.metadata.get("internal"):
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
