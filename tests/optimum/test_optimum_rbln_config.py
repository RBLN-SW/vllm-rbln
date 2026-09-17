# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`--rbln-*` CLI flags -> `additional_config` -> `OptimumRBLNConfig`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_rbln import envs
from vllm_rbln.config import _GROUP_TITLE, OptimumRBLNConfig, _resolve


@pytest.fixture(autouse=True)
def on_the_optimum_path():
    """Either name in the environment would silently move this file to the other
    path, so say so instead of testing the wrong one."""
    assert envs.model_impl_from_env() == "optimum"


@pytest.fixture(scope="module")
def parser() -> FlexibleArgumentParser:
    # add_cli_args() ends in current_platform.pre_register_and_update(parser),
    # which is where the RBLN group is registered.
    return AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())


def test_the_flags_are_registered_on_this_path(parser):
    """The group is added before the arguments are parsed, so it cannot depend
    on which path was selected."""
    assert any(g.title == _GROUP_TITLE for g in parser._action_groups)


def test_defaults_when_nothing_is_passed(parser):
    args = parser.parse_args([])
    assert _resolve(OptimumRBLNConfig, args.additional_config) == OptimumRBLNConfig()


def test_an_optimum_field_reaches_the_config(parser):
    args = parser.parse_args(["--rbln-prefix-block-size", "256"])
    config = _resolve(OptimumRBLNConfig, args.additional_config)
    assert config.prefix_block_size == 256


def test_the_overrides_arrive_as_a_mapping(parser):
    """The only dict-valued flag, so it is the one that needs the JSON path."""
    args = parser.parse_args(["--rbln-optimum-overrides", '{"device": [0, 1]}'])
    config = _resolve(OptimumRBLNConfig, args.additional_config)
    assert config.optimum_overrides == {"device": [0, 1]}


def test_a_field_of_the_other_path_is_rejected():
    with pytest.raises(ValueError, match="are not fields"):
        _resolve(OptimumRBLNConfig, {"use_w8a8": True})


def test_a_shared_field_is_taken(parser):
    args = parser.parse_args(["--rbln-num-devices-per-local-rank", "4"])
    assert _resolve(OptimumRBLNConfig, args.additional_config) == OptimumRBLNConfig(
        num_devices_per_local_rank=4
    )


def test_a_no_flag_field_has_no_flag(parser):
    group = next(g for g in parser._action_groups if g.title == _GROUP_TITLE)
    flags = {a.dest for a in group._group_actions}
    # vLLM already owns --max-num-batched-tokens, and the sync writes the other.
    assert "rbln_user_max_num_batched_tokens" not in flags
    assert "rbln_attn_block_size" not in flags


def test_a_no_flag_field_is_still_accepted():
    """`_capture_user_max_num_batched_tokens` writes this into the dict before
    the class exists, so resolution has to take it."""
    config = _resolve(OptimumRBLNConfig, {"user_max_num_batched_tokens": 512})
    assert config.user_max_num_batched_tokens == 512


def test_the_capture_writes_only_where_this_path_reads(monkeypatch):
    """`--max-num-batched-tokens` is captured where the path is already known.

    It lands in whichever shape `additional_config` arrived as, since neither a
    built config nor upstream's bare string takes item assignment. What the
    other path gets is not this suite's to say; building a native config is
    what shows that, and the whole of tests/vllm does it.
    """
    from vllm.engine.arg_utils import EngineArgs

    from vllm_rbln.platform import RblnPlatform

    # Wrapped around a stub: building a real engine config needs a compiled
    # model, and what is under test is what the wrapper writes before that.
    monkeypatch.setattr(EngineArgs, "create_engine_config", lambda self: None)
    monkeypatch.setattr(EngineArgs, "_rbln_model_impl_patched", False, raising=False)
    RblnPlatform._capture_model_impl()

    def captured(additional_config):
        args = SimpleNamespace(
            max_num_batched_tokens=512, additional_config=additional_config
        )
        EngineArgs.create_engine_config(args)
        return args.additional_config

    assert captured(None)["user_max_num_batched_tokens"] == 512
    assert captured({"prefix_block_size": 64})["user_max_num_batched_tokens"] == 512
    assert captured(OptimumRBLNConfig()).user_max_num_batched_tokens == 512
    assert captured("something") == "something"


def test_the_former_overrides_key_is_accepted():
    """TODO(vllm-rbln>=0.12.0): delete with the key."""
    config = _resolve(OptimumRBLNConfig, {"rbln_config": {"device": [0]}})
    assert config.optimum_overrides == {"device": [0]}


def test_only_compile_fields_change_the_hash():
    """A field the sync derives cannot key the artifact it was derived from."""
    base = OptimumRBLNConfig().compute_hash()
    assert OptimumRBLNConfig(attn_block_size=64).compute_hash() == base
    assert OptimumRBLNConfig(cached_model_path="/tmp/x").compute_hash() == base
    assert OptimumRBLNConfig(num_blocks_synced=True).compute_hash() == base
    assert OptimumRBLNConfig(use_custom_sampler=False).compute_hash() == base
    assert OptimumRBLNConfig(user_max_num_batched_tokens=512).compute_hash() != base
    assert OptimumRBLNConfig(prefix_block_size=256).compute_hash() != base


def test_coexists_with_additional_config(parser):
    """`add_rbln_cli_args` installs the merging action, and it now runs here too."""
    args = parser.parse_args(
        [
            "--rbln-prefix-block-size",
            "256",
            "--additional-config.num_devices_per_local_rank",
            "4",
        ]
    )
    assert args.additional_config == {
        "prefix_block_size": 256,
        "num_devices_per_local_rank": 4,
    }
