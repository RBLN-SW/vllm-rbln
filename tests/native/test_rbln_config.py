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

"""`--rbln-*` CLI flags -> `additional_config` -> `RBLNConfig`."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_rbln.config import (
    _ENV_PROBE,
    _GROUP_TITLE,
    RBLNConfig,
    build_rbln_config,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """conftest pins VLLM_RBLN_NUM_HIDDEN_LAYERS; start from the defaults."""
    for f in dataclasses.fields(RBLNConfig):
        for name in _ENV_PROBE.get(f.name, (f"VLLM_RBLN_{f.name.upper()}",)):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def parser() -> FlexibleArgumentParser:
    # add_cli_args() ends in current_platform.pre_register_and_update(parser),
    # which is where the RBLN group is registered.
    return AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())


def resolve(parser, argv: list[str]) -> RBLNConfig:
    args = parser.parse_args(argv)
    engine_args = AsyncEngineArgs.from_cli_args(args)
    return build_rbln_config(engine_args.additional_config)


def test_group_is_registered(parser):
    assert any(g.title == _GROUP_TITLE for g in parser._action_groups)


def test_every_field_gets_a_flag(parser):
    group = next(g for g in parser._action_groups if g.title == _GROUP_TITLE)
    flags = {a.dest for a in group._group_actions}
    assert flags == {f"rbln_{f.name}" for f in dataclasses.fields(RBLNConfig)}


def test_defaults_when_nothing_is_passed(parser):
    assert resolve(parser, []) == RBLNConfig()


def test_flags_reach_the_config(parser):
    config = resolve(
        parser,
        [
            "--no-rbln-compile-model",
            "--rbln-num-devices-per-local-rank",
            "4",
            "--rbln-decode-batch-bucket-strategy",
            "manual",
            "--rbln-decode-batch-bucket-manual-buckets",
            "1",
            "4",
            "16",
        ],
    )
    assert config.compile_model is False
    assert config.num_devices_per_local_rank == 4
    assert config.decode_batch_bucket_strategy == "manual"
    assert config.decode_batch_bucket_manual_buckets == [1, 4, 16]


def test_coexists_with_additional_config(parser):
    """The dotted form is appended at the end of argv, so it must merge."""
    args = parser.parse_args(
        ["--rbln-use-w8a8", "--additional-config.decode_batch_bucket_limit", "2"]
    )
    assert args.additional_config == {
        "use_w8a8": True,
        "decode_batch_bucket_limit": 2,
    }
    config = build_rbln_config(args.additional_config)
    assert (config.use_w8a8, config.decode_batch_bucket_limit) == (True, 2)


def test_json_form_is_equivalent(parser):
    """`LLM(additional_config=...)` goes through the same code."""
    assert resolve(parser, ['--additional-config={"use_w8a8": true}']).use_w8a8 is True


def test_an_instance_passes_through(parser):
    given = RBLNConfig(use_w8a8=True)
    assert build_rbln_config(given) is given


def test_env_is_still_honored(parser, monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_W8A8", "1")
    assert resolve(parser, []).use_w8a8 is True


def test_unprefixed_custom_kernel_env_is_honored(parser, monkeypatch):
    monkeypatch.setenv("RBLN_USE_CUSTOM_KERNEL", "1")
    assert resolve(parser, []).use_custom_kernel is True


def test_cli_wins_over_env(parser, monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_W8A8", "1")
    assert resolve(parser, ["--no-rbln-use-w8a8"]).use_w8a8 is False


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="are not fields"):
        build_rbln_config({"compile_modell": False})


def test_upstream_key_is_rejected():
    """`--gdn-prefill-backend` is written into additional_config by arg_utils."""
    with pytest.raises(ValueError, match="gdn_prefill_backend"):
        build_rbln_config({"gdn_prefill_backend": "auto"})


def test_manual_strategy_needs_buckets():
    with pytest.raises(ValueError, match="needs at least one entry"):
        RBLNConfig(decode_batch_bucket_strategy="manual")


def test_get_rbln_config_needs_the_current_config_context():
    """It reads the config the model is being built under, so there has to be one."""
    from vllm_rbln.config import get_rbln_config

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        get_rbln_config()


def test_get_rbln_config_rejects_a_config_that_is_not_ours():
    """The optimum-rbln path leaves a dict there, and nothing resolves it."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config

    from vllm_rbln.config import get_rbln_config

    with (
        set_current_vllm_config(SimpleNamespace(additional_config={})),
        pytest.raises(RuntimeError, match="not an RBLNConfig"),
    ):
        get_rbln_config()


def test_json_values_are_coerced():
    """`additional_config` arrives as JSON, so the types need converting."""
    assert build_rbln_config({"use_w8a8": "false"}).use_w8a8 is False
    assert (
        build_rbln_config({"decode_batch_bucket_limit": "8"}).decode_batch_bucket_limit
        == 8
    )


def test_invalid_value_is_rejected():
    with pytest.raises(ValueError):
        build_rbln_config({"decode_batch_bucket_strategy": "garbage"})
    with pytest.raises(ValueError):
        build_rbln_config({"use_w8a8": "junk"})


def test_no_field_is_read_from_the_environment():
    """A field here is the source, so nothing on this path may read its variable.

    `envs.py` resolves the variable into the field; a reader that goes around
    that would ignore `--rbln-*` and `additional_config`. The options that stay
    in `envs.py` are not fields, so they are exempt by construction.

    `build_rbln_config` only runs on the vLLM-native path, so the optimum-rbln
    path's own readers are outside the claim.
    """
    import vllm_rbln

    root = pathlib.Path(vllm_rbln.__file__).parent
    optimum_owned = ("utils/optimum/", "model_executor/models/optimum/")
    sources = "\n".join(
        path.read_text()
        for path in root.rglob("*.py")
        if (rel := path.relative_to(root).as_posix()) not in ("envs.py", "config.py")
        and not rel.startswith(optimum_owned)
        and not path.name.startswith("optimum_")
    )
    assert not [
        f.name
        for f in dataclasses.fields(RBLNConfig)
        if f"envs.VLLM_RBLN_{f.name.upper()}" in sources
    ]


def test_only_compile_fields_change_the_hash():
    """`mega_cache` uses this for its bundle key, via VllmConfig.

    A str and a list among the values, since they have to survive
    normalize_value() to reach the key at all.
    """
    base = RBLNConfig().compute_hash()
    assert RBLNConfig(sampler=False).compute_hash() == base
    assert RBLNConfig(use_w8a8=True).compute_hash() != base
    assert RBLNConfig(decode_batch_bucket_strategy="linear").compute_hash() != base
    buckets = RBLNConfig(decode_batch_bucket_manual_buckets=[1, 2, 4])
    assert buckets.compute_hash() != base
