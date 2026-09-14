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

"""`additional_config` and `VLLM_RBLN_*` -> `OptimumRBLNConfig`."""

import pytest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_rbln.config import (
    _GROUP_TITLE,
    OptimumRBLNConfig,
    build_optimum_rbln_config,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "VLLM_RBLN_SAMPLER",
        "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK",
        "VLLM_RBLN_TP_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_nothing_is_set():
    assert build_optimum_rbln_config(None) == OptimumRBLNConfig()


def test_environment_is_read(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
    monkeypatch.setenv("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", "4")
    config = build_optimum_rbln_config(None)
    assert (config.sampler, config.num_devices_per_local_rank) == (False, 4)


def test_deprecated_tp_size_alias_is_read(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_TP_SIZE", "2")
    assert build_optimum_rbln_config(None).num_devices_per_local_rank == 2


def test_additional_config_wins_over_environment(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    assert build_optimum_rbln_config({"sampler": False}).sampler is False


def test_json_values_are_coerced():
    config = build_optimum_rbln_config(
        {"sampler": "false", "num_devices_per_local_rank": "2"}
    )
    assert (config.sampler, config.num_devices_per_local_rank) == (False, 2)


def test_optimum_overrides_pass_through_as_a_dict():
    overrides = {"batch_size": 4, "visual": {"max_seq_len": [512]}}
    config = build_optimum_rbln_config({"optimum_overrides": overrides})
    assert config.optimum_overrides == overrides


def test_rbln_config_key_is_still_accepted():
    """The key every example used until now. Warned about, then mapped."""
    overrides = {"batch_size": 4}
    config = build_optimum_rbln_config({"rbln_config": overrides})
    assert config.optimum_overrides == overrides


def test_fields_without_a_variable_ignore_the_environment(monkeypatch):
    """Only fields `envs.py` declares a variable for are read from it."""
    monkeypatch.setenv("VLLM_RBLN_CACHED_MODEL_PATH", "/nowhere")
    assert build_optimum_rbln_config(None).cached_model_path is None


def test_native_only_field_is_rejected():
    """`use_w8a8` is a `vllm` field. The optimum path does not take it."""
    with pytest.raises(ValueError, match="are not fields"):
        build_optimum_rbln_config({"use_w8a8": True})


def test_instance_is_returned_unchanged(monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "0")
    config = OptimumRBLNConfig(sampler=True)
    assert build_optimum_rbln_config(config) is config


def test_flags_cover_the_optimum_path_and_nothing_else():
    """No suite conftest sets VLLM_RBLN_USE_VLLM_MODEL here, so the parser is
    built for the optimum-rbln path: its user fields get a flag, the
    native-only and the internal fields do not."""
    parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
    group = next(g for g in parser._action_groups if g.title == _GROUP_TITLE)
    flags = {a.dest for a in group._group_actions}
    assert {"rbln_sampler", "rbln_optimum_overrides", "rbln_prefix_block_size"} <= flags
    assert not flags & {"rbln_use_w8a8", "rbln_cached_model_path"}
