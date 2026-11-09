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
from unittest.mock import Mock

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


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
)
def test_multimodal_normalization_stays_in_the_processor(model, monkeypatch):
    # Building the config asks the NPU for its family, and this lane has none.
    from vllm_rbln import platform

    monkeypatch.setattr(platform.rebel, "get_npu_name", lambda *a, **kw: "RBLN-CA25")
    config = AsyncEngineArgs(
        model=model, model_impl="optimum", max_model_len=4096, block_size=4096
    ).create_engine_config()

    mm_config = config.model_config.multimodal_config
    assert mm_config.mm_device_do_normalize is False
    processor_kwargs = mm_config.merge_mm_processor_kwargs({})
    assert processor_kwargs.get("do_normalize") is not False
    assert processor_kwargs.get("do_rescale") is not False


def test_an_optimum_field_reaches_the_config(parser):
    args = parser.parse_args(["--rbln-sub-block-size", "256"])
    config = _resolve(OptimumRBLNConfig, args.additional_config)
    assert config.sub_block_size == 256


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
    assert "rbln_attn_block_size" not in flags
    assert "rbln_user_max_num_batched_tokens" not in flags


@pytest.fixture
def capture_engine_config(monkeypatch):
    from vllm.engine.arg_utils import EngineArgs

    from vllm_rbln import platform

    monkeypatch.setattr(EngineArgs, "create_engine_config", lambda self: "created")
    monkeypatch.setattr(EngineArgs, "_rbln_model_impl_patched", False, raising=False)
    monkeypatch.setattr(platform.logger, "warning_once", Mock())
    platform.RblnPlatform._capture_model_impl()
    return EngineArgs.create_engine_config, platform.logger.warning_once


@pytest.mark.parametrize("budget", [128, 8192])
@pytest.mark.parametrize("model_impl", ["optimum", "transformers"])
@pytest.mark.parametrize("config_kind", ["unset", "mapping", "built"])
def test_legacy_budget_becomes_prefill_chunk_size(
    capture_engine_config, budget, model_impl, config_kind
):
    create_config, warning = capture_engine_config
    config = None
    if config_kind == "mapping":
        config = {"prefill_chunk_size": budget}
    elif config_kind == "built":
        config = OptimumRBLNConfig(prefill_chunk_size=budget)
    args = SimpleNamespace(
        model_impl=model_impl,
        additional_config=config,
        max_num_batched_tokens=budget,
        create_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        ),
    )
    assert create_config(args) == "created"
    assert args.additional_config.prefill_chunk_size == budget
    assert args.max_num_batched_tokens is None
    warning.assert_called_once()
    assert "0.14.0" in warning.call_args.args[0]


@pytest.mark.parametrize("built", [False, True])
@pytest.mark.parametrize(
    ("budget", "chunk", "error"),
    [
        (128, 256, "Conflicting prefill chunk size"),
        (0, None, "max_num_batched_tokens must be greater than 0"),
        (-1, None, "max_num_batched_tokens must be greater than 0"),
    ],
)
def test_invalid_legacy_budget_is_rejected(
    capture_engine_config, built, budget, chunk, error
):
    create_config, _ = capture_engine_config
    config = {"prefill_chunk_size": chunk}
    if built:
        config = OptimumRBLNConfig(**config)
    args = SimpleNamespace(
        model_impl="optimum",
        additional_config=config,
        max_num_batched_tokens=budget,
        create_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        ),
    )
    with pytest.raises(ValueError, match=error):
        create_config(args)


@pytest.mark.parametrize("arch", ["WhisperForConditionalGeneration", "BertModel"])
def test_legacy_budget_is_ignored_without_chunked_prefill(capture_engine_config, arch):
    create_config, warning = capture_engine_config
    args = SimpleNamespace(
        model_impl="optimum",
        additional_config=None,
        max_num_batched_tokens=128,
        create_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch])
        ),
    )
    assert create_config(args) == "created"
    assert args.additional_config is None
    assert args.max_num_batched_tokens is None
    warning.assert_called_once()


@pytest.mark.parametrize("config", [None, {}, OptimumRBLNConfig()])
def test_unset_budget_is_left_to_vllm_defaults(capture_engine_config, config):
    create_config, warning = capture_engine_config
    args = SimpleNamespace(
        model_impl="optimum", additional_config=config, max_num_batched_tokens=None
    )
    assert create_config(args) == "created"
    warning.assert_not_called()
    assert args.max_num_batched_tokens is None
    assert args.additional_config is config
    if isinstance(config, dict):
        assert config == {}


def test_prefill_chunk_size_flag(parser):
    args = parser.parse_args(["--rbln-prefill-chunk-size", "256"])
    assert _resolve(OptimumRBLNConfig, args.additional_config).prefill_chunk_size == 256


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_prefill_chunk_size_is_rejected(size):
    with pytest.raises(ValueError, match="greater_than"):
        _resolve(OptimumRBLNConfig, {"prefill_chunk_size": size})


def test_the_former_overrides_key_is_accepted():
    """TODO(vllm-rbln>=0.14.0): delete with the key."""
    config = _resolve(OptimumRBLNConfig, {"rbln_config": {"device": [0]}})
    assert config.optimum_overrides == {"device": [0]}


@pytest.mark.parametrize("size", [0, -1])
def test_a_sub_block_size_of_zero_or_less_is_rejected(size):
    """The converter divides by the size, so 0 is refused before it gets there.
    Unset is None, not 0."""
    with pytest.raises(ValueError, match="greater_than"):
        _resolve(OptimumRBLNConfig, {"sub_block_size": size})
    assert OptimumRBLNConfig().sub_block_size is None


def test_the_former_prefix_block_size_key_is_accepted():
    """TODO(vllm-rbln>=0.14.0): delete with the key."""
    config = _resolve(OptimumRBLNConfig, {"prefix_block_size": 256})
    assert config.sub_block_size == 256


def test_only_compile_fields_change_the_hash():
    """A field the sync derives cannot key the artifact it was derived from."""
    base = OptimumRBLNConfig().compute_hash()
    assert OptimumRBLNConfig(attn_block_size=64).compute_hash() == base
    assert OptimumRBLNConfig(artifact_path="/tmp/x").compute_hash() == base
    assert OptimumRBLNConfig(precompiled=True).compute_hash() == base
    assert OptimumRBLNConfig(num_blocks_synced=True).compute_hash() == base
    assert OptimumRBLNConfig(use_custom_sampler=False).compute_hash() == base
    assert OptimumRBLNConfig(prefill_chunk_size=512).compute_hash() != base
    assert OptimumRBLNConfig(sub_block_size=256).compute_hash() != base


def test_coexists_with_additional_config(parser):
    """`add_rbln_cli_args` installs the merging action, and it now runs here too."""
    args = parser.parse_args(
        [
            "--rbln-sub-block-size",
            "256",
            "--additional-config.num_devices_per_local_rank",
            "4",
        ]
    )
    assert args.additional_config == {
        "sub_block_size": 256,
        "num_devices_per_local_rank": 4,
    }
