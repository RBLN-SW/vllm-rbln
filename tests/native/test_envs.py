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

"""Tests for ``vllm_rbln.envs``: branching behavior and cross-declaration
agreement, not plain default values (which only restate the lambda)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from vllm_rbln import envs

RBLN_KEYS = sorted(
    key for key in envs.environment_variables if key.startswith("VLLM_RBLN_")
)


def read(name: str) -> Any:
    """Resolve ``name`` through the env table."""
    return envs.environment_variables[name]()


def test_manual_buckets_unset_is_empty(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", raising=False)
    assert envs.get_decode_batch_bucket_manual_buckets() == []


def test_manual_buckets_parses_in_order(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "4,1,16")
    assert envs.get_decode_batch_bucket_manual_buckets() == [4, 1, 16]


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("0,4", "greater than 0"),
        ("4,4", "unique"),
        ("four", "invalid literal"),
    ],
)
def test_manual_buckets_rejects(monkeypatch, raw, reason):
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", raw)
    with pytest.raises(ValueError) as excinfo:
        envs.get_decode_batch_bucket_manual_buckets()
    message = str(excinfo.value)
    assert reason in message
    # The message must name the variable and echo the value, or the user only
    # sees a bare "invalid literal for int()".
    assert "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS" in message
    assert raw in message


def test_strategy_unset_defaults_to_exponential(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", raising=False)
    assert envs.get_decode_batch_bucket_strategy() == "exponential"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("exponential", "exponential"),
        ("EXPONENTIAL", "exponential"),  # case folded
        ("exp", "exponential"),  # alias normalized away
        ("linear", "linear"),
    ],
)
def test_strategy_normalizes(monkeypatch, raw, expected):
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", raw)
    assert envs.get_decode_batch_bucket_strategy() == expected


def test_strategy_rejects_unknown(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "geometric")
    with pytest.raises(ValueError, match="geometric"):
        envs.get_decode_batch_bucket_strategy()


def test_manual_strategy_requires_buckets(monkeypatch):
    """``manual`` without buckets is a config error, not a silent empty list."""
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "manual")
    monkeypatch.delenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", raising=False)
    with pytest.raises(ValueError, match="at least one decode batch size"):
        envs.get_decode_batch_bucket_strategy()


def test_manual_strategy_accepted_with_buckets(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "manual")
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "1,8")
    assert envs.get_decode_batch_bucket_strategy() == "manual"


def test_manual_strategy_surfaces_invalid_buckets(monkeypatch):
    """A bucket-list typo must surface through the strategy lookup too, not
    only when the list is read on its own."""
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY", "manual")
    monkeypatch.setenv("VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS", "4,4")
    with pytest.raises(ValueError, match="unique"):
        envs.get_decode_batch_bucket_strategy()


@pytest.mark.parametrize(
    ("new", "legacy", "expected"),
    [
        (None, None, 1),
        ("4", None, 4),
        (None, "8", 8),  # deprecated VLLM_RBLN_TP_SIZE still honored
        ("4", "8", 4),  # the new variable wins when both are set
    ],
)
def test_num_devices_precedence(monkeypatch, new, legacy, expected):
    for name, value in (
        ("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", new),
        ("VLLM_RBLN_TP_SIZE", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert envs.get_num_devices_per_local_rank() == expected


@pytest.mark.parametrize(
    ("auto_port", "device_tensor", "expected"),
    [
        (None, None, True),  # follows the device-tensor default, which is on
        (None, "1", True),
        (None, "0", False),  # the coupling: no explicit auto-port, tensors off
        ("1", "0", True),  # explicit setting overrides the coupling
        ("0", "1", False),
    ],
)
def test_auto_port_follows_device_tensor(
    monkeypatch, auto_port, device_tensor, expected
):
    for name, value in (
        ("VLLM_RBLN_AUTO_PORT", auto_port),
        ("VLLM_RBLN_USE_DEVICE_TENSOR", device_tensor),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert envs.use_auto_port() is expected


# The bool lambdas come in two shapes differing only in their default; an
# unrecognized value has opposite consequences, so one of each is covered.
BOOL_DEFAULT_OFF = "VLLM_RBLN_METRICS"
BOOL_DEFAULT_ON = "VLLM_RBLN_NUMA"


@pytest.mark.parametrize("name", [BOOL_DEFAULT_OFF, BOOL_DEFAULT_ON])
@pytest.mark.parametrize("raw", ["1", "true", "TRUE"])
def test_bool_convention_accepts(monkeypatch, name, raw):
    monkeypatch.setenv(name, raw)
    assert read(name) is True


@pytest.mark.parametrize("name", [BOOL_DEFAULT_OFF, BOOL_DEFAULT_ON])
@pytest.mark.parametrize(
    "raw",
    [
        "0",  # explicit off
        "yes",  # an unrecognized word, not an error
        "",  # set but empty
        " true",  # surrounding space is not stripped
    ],
)
def test_bool_convention_rejects(monkeypatch, name, raw):
    """Only 'true'/'1' count; everything else reads as off."""
    monkeypatch.setenv(name, raw)
    assert read(name) is False


def test_unrecognized_value_disables_a_default_on_variable(monkeypatch):
    """The convention's sharp edge: on a default-on variable an unrecognized
    value (e.g. ``VLLM_RBLN_NUMA=yes``) silently disables the feature."""
    monkeypatch.delenv(BOOL_DEFAULT_ON, raising=False)
    assert read(BOOL_DEFAULT_ON) is True

    monkeypatch.setenv(BOOL_DEFAULT_ON, "yes")
    assert read(BOOL_DEFAULT_ON) is False


# Values that make a variable's resolved result differ from its default, for
# the ones a generic probe cannot guess (both are validated).
_PROBE_OVERRIDES = {
    "VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY": "linear",
    "VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS": "3,5",
}

# Keys that resolve from a differently named variable: custom kernels follow the
# rebel compiler's own flag rather than a private copy that could disagree.
ENV_SOURCE_ALIASES = {
    "VLLM_RBLN_USE_CUSTOM_KERNEL": "RBLN_USE_CUSTOM_KERNEL",
}


def _probe_value(default: Any) -> str:
    if isinstance(default, bool):  # before int -- bool is a subclass of int
        return "0" if default else "1"
    if isinstance(default, int):
        return str(default + 1)
    return "probe-value"


@pytest.mark.parametrize("name", RBLN_KEYS)
def test_each_key_reads_its_declared_source(monkeypatch, name):
    """Setting a key's declared source must change what it resolves to --
    catches a copy-pasted lambda that reads a neighbouring variable."""
    source = ENV_SOURCE_ALIASES.get(name, name)
    default = read(name)
    probe = _PROBE_OVERRIDES.get(name) or _probe_value(default)
    monkeypatch.setenv(source, probe)
    assert read(name) != default, (
        f"{source}={probe!r} left {name} at {default!r}; its lambda reads "
        f"neither {name} nor any variable declared for it"
    )


def test_custom_kernel_follows_the_compiler_flag(monkeypatch):
    """The vllm-prefixed name has no effect of its own; custom kernels follow
    the rebel compiler's own flag (honoring both would let them disagree)."""
    monkeypatch.delenv("RBLN_USE_CUSTOM_KERNEL", raising=False)
    monkeypatch.setenv("VLLM_RBLN_USE_CUSTOM_KERNEL", "1")
    assert envs.VLLM_RBLN_USE_CUSTOM_KERNEL is False

    monkeypatch.setenv("RBLN_USE_CUSTOM_KERNEL", "1")
    assert envs.VLLM_RBLN_USE_CUSTOM_KERNEL is True


def _declared_in_type_checking() -> dict[str, tuple[str, Any]]:
    """``{name: (annotation, default)}`` from the ``if TYPE_CHECKING`` block,
    which nothing consults at runtime and can therefore drift silently."""
    tree = ast.parse(Path(inspect.getfile(envs)).read_text())
    declared: dict[str, tuple[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if getattr(node.test, "id", None) != "TYPE_CHECKING":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
                continue
            # Only bare ``NAME: type = value`` declares a variable; an
            # AnnAssign target may also be an attribute or a subscript.
            if not isinstance(stmt.target, ast.Name):
                continue
            declared[stmt.target.id] = (
                ast.unparse(stmt.annotation),
                ast.literal_eval(stmt.value),
            )
    return declared


def test_type_checking_block_lists_every_variable():
    declared = _declared_in_type_checking()
    assert declared, "no annotated declarations found; did the block move?"
    assert sorted(declared) == RBLN_KEYS


@pytest.mark.parametrize("name", RBLN_KEYS)
def test_declared_default_matches_resolved(monkeypatch, name):
    """A clean environment resolves to the declared default. All RBLN vars are
    cleared, since some derive from a neighbour (use_auto_port falls back to
    VLLM_RBLN_USE_DEVICE_TENSOR) rather than a literal."""
    annotation, declared_default = _declared_in_type_checking()[name]
    for key in (
        set(RBLN_KEYS) | set(ENV_SOURCE_ALIASES.values()) | {"VLLM_RBLN_TP_SIZE"}
    ):
        monkeypatch.delenv(key, raising=False)

    resolved = read(name)
    assert resolved == declared_default
    assert type(resolved).__name__ == annotation.split("[")[0]


@pytest.mark.parametrize(
    "name", ["VLLM_USE_V2_MODEL_RUNNER", "VLLM_DISABLE_COMPILE_CACHE"]
)
def test_upstream_variables_reachable(name):
    """``platform.py`` reads these upstream names off ``vllm_rbln.envs`` via
    the ``__getattr__`` fallthrough to ``vllm.envs`` (surfaced in ``__dir__``)."""
    assert name in dir(envs)
    getattr(envs, name)


def test_rbln_variables_are_registered_with_vllm():
    """``envs.py`` pushes every RBLN name into vLLM's registry, which ray_env
    (driver->Ray copy) and collect_env read; an unregistered name silently
    never reaches a Ray worker, with no error anywhere."""
    import vllm.envs as upstream_envs

    missing = sorted(set(RBLN_KEYS) - set(upstream_envs.environment_variables))
    assert not missing, (
        f"{len(missing)} RBLN variables are not registered with vLLM and would "
        f"not be copied to Ray workers: {missing}"
    )
    # Registered *and* resolvable: ray_env only needs the name, but collect_env
    # and any direct reader go through vllm.envs' own attribute lookup.
    assert getattr(upstream_envs, "VLLM_RBLN_USE_VLLM_MODEL") is True  # noqa: B009


def test_compile_env_partition_covers_every_variable():
    """The mega-cache bundle key hashes RBLN_COMPILE_ENV and ignores
    RBLN_NON_COMPILE_ENV, so a variable in neither is a silent stale bundle
    (graph-relevant) or a silent recompile every launch (not)."""
    declared = set(envs.environment_variables)
    classified = envs.RBLN_COMPILE_ENV | envs.RBLN_NON_COMPILE_ENV
    assert classified == declared, (
        f"unclassified: {sorted(declared - classified)}, "
        f"stale: {sorted(classified - declared)}"
    )


def test_compile_env_partition_is_disjoint():
    overlap = envs.RBLN_COMPILE_ENV & envs.RBLN_NON_COMPILE_ENV
    assert not overlap, f"classified twice: {sorted(overlap)}"


def test_unknown_variable_raises():
    with pytest.raises(AttributeError, match="VLLM_RBLN_NOT_A_REAL_VARIABLE"):
        getattr(envs, "VLLM_RBLN_NOT_A_REAL_VARIABLE")  # noqa: B009


def test_dynamic_kv_cache_is_opt_in(monkeypatch):
    # The flag is read during compile, so a non-False default would change the
    # compiled artifact for every existing deployment.
    monkeypatch.delenv("VLLM_RBLN_USE_DYNAMIC_KV_CACHE", raising=False)
    assert envs.environment_variables["VLLM_RBLN_USE_DYNAMIC_KV_CACHE"]() is False
