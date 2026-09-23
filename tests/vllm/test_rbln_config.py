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
import re

import pytest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm_rbln.envs as envs
from vllm_rbln.config import (
    _GROUP_TITLE,
    OptimumRBLNConfig,
    RBLNConfig,
    _env_source,
    build_optimum_rbln_config,
    build_rbln_config,
    resolve_model_impl,
)
from vllm_rbln.envs import RESOLVED_MODEL_IMPL_ENV


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """conftest pins VLLM_RBLN_NUM_HIDDEN_LAYERS; start from the defaults."""
    for f in dataclasses.fields(RBLNConfig):
        _, probes = _env_source(f.name)
        for name in probes:
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
    # The group is built before the model path is known, so it carries both
    # classes' fields. A field on the shared base is registered once, and a
    # field the code fills in gets no flag at all.
    group = next(g for g in parser._action_groups if g.title == _GROUP_TITLE)
    flags = {a.dest for a in group._group_actions}
    assert flags == {
        f"rbln_{f.name}"
        for cls in (RBLNConfig, OptimumRBLNConfig)
        for f in dataclasses.fields(cls)
        if not f.metadata.get("no_flag")
    }


def test_a_field_of_the_other_path_is_rejected():
    """Both paths' flags are registered, so the class is what narrows them."""
    with pytest.raises(ValueError, match="are not fields"):
        build_rbln_config({"prefix_block_size": 256})


def test_defaults_when_nothing_is_passed(parser):
    assert resolve(parser, []) == RBLNConfig()


def test_flags_reach_the_config(parser):
    config = resolve(
        parser,
        [
            "--no-rbln-use-custom-sampler",
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
    assert config.use_custom_sampler is False
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


def test_dynamic_kv_cache_is_tri_state(parser):
    assert resolve(parser, []).use_dynamic_kv_cache is None
    assert resolve(parser, ["--rbln-use-dynamic-kv-cache"]).use_dynamic_kv_cache is True
    assert (
        resolve(parser, ["--no-rbln-use-dynamic-kv-cache"]).use_dynamic_kv_cache
        is False
    )


def test_cli_wins_over_env(parser, monkeypatch):
    monkeypatch.setenv("VLLM_RBLN_USE_W8A8", "1")
    assert resolve(parser, ["--no-rbln-use-w8a8"]).use_w8a8 is False


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="are not fields"):
        build_rbln_config({"use_custom_samplerr": False})


def test_a_config_of_the_other_path_is_rejected():
    """Two ways to hand a resolution the wrong class, and one message for both.

    A caller can hand either builder the other path's class, and the message
    has to read the same way round for both: what it was handed, then what is
    being resolved.
    """
    with pytest.raises(ValueError, match="belongs to one model path") as said:
        build_optimum_rbln_config(RBLNConfig())
    assert "is an RBLNConfig" in str(said.value)
    assert "OptimumRBLNConfig path is the one being resolved" in str(said.value)

    with pytest.raises(ValueError, match="belongs to one model path") as said:
        build_rbln_config(OptimumRBLNConfig())
    assert "is an OptimumRBLNConfig" in str(said.value)
    assert "RBLNConfig path is the one being resolved" in str(said.value)


def test_upstream_key_is_rejected():
    """`--gdn-prefill-backend` is written into additional_config by arg_utils."""
    with pytest.raises(ValueError, match="gdn_prefill_backend"):
        build_rbln_config({"gdn_prefill_backend": "auto"})


def test_manual_strategy_needs_buckets():
    with pytest.raises(ValueError, match="needs at least one entry"):
        RBLNConfig(decode_batch_bucket_strategy="manual")


@pytest.mark.parametrize("size", [0, -1])
def test_a_sub_block_size_of_zero_or_less_is_rejected(size):
    """`sub_block_size_in_use()` takes a size or None, so the field refuses
    the values that would mean neither. Unset is None, not 0."""
    with pytest.raises(ValueError, match="greater_than"):
        build_rbln_config({"sub_block_size": size})
    assert RBLNConfig().sub_block_size is None


def test_get_rbln_config_needs_the_current_config_context():
    """It reads the config the model is being built under, so there has to be one."""
    from vllm_rbln.config import get_rbln_config

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        get_rbln_config()


@pytest.mark.parametrize("other", [{}, OptimumRBLNConfig()], ids=["unbuilt", "optimum"])
def test_get_rbln_config_rejects_a_config_that_is_not_ours(other):
    """The two classes share a base, so `isinstance` has to reject the sibling."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config

    from vllm_rbln.config import get_rbln_config

    with (
        set_current_vllm_config(SimpleNamespace(additional_config=other)),
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
    """A field is the source, so nothing may read its variable instead.

    `envs.py` resolves the variable into the field; a reader that goes around
    that would ignore `--rbln-*` and `additional_config`. The options that stay
    in `envs.py` are not fields, so they are exempt by construction.

    Both paths resolve their own class now, so neither is excluded. The probe
    comes from `_env_source`, not from the field name, because a renamed field
    reads a variable that no longer matches it.
    """
    import vllm_rbln

    root = pathlib.Path(vllm_rbln.__file__).parent
    sources = "\n".join(
        path.read_text()
        for path in root.rglob("*.py")
        if path.relative_to(root).as_posix() not in ("envs.py", "config.py")
    )
    assert not [
        f.name
        for cls in (RBLNConfig, OptimumRBLNConfig)
        for f in dataclasses.fields(cls)
        if f"envs.{_env_source(f.name)[0]}" in sources
    ]


def test_the_model_path_keys_the_compile_cache():
    """A different model implementation is a different artifact.

    The path is not a field any more, so what keeps the two apart in the
    mega-cache bundle key is that each path hashes a class of its own. Handing a
    run the other path's artifact is what this prevents.
    """
    assert RBLNConfig().compute_hash() != OptimumRBLNConfig().compute_hash()


def test_only_compile_fields_change_the_hash():
    """`mega_cache` uses this for its bundle key, via VllmConfig.

    A str and a list among the values, since they have to survive
    normalize_value() to reach the key at all.
    """
    base = RBLNConfig().compute_hash()
    assert RBLNConfig(use_custom_sampler=False).compute_hash() == base
    assert RBLNConfig(enable_sub_block_cache=False).compute_hash() == base
    assert RBLNConfig(sub_block_size=64).compute_hash() == base
    assert RBLNConfig(use_w8a8=True).compute_hash() != base
    assert RBLNConfig(compile_dtype="float16").compute_hash() != base
    assert RBLNConfig(decode_batch_bucket_strategy="linear").compute_hash() != base
    buckets = RBLNConfig(decode_batch_bucket_manual_buckets=[1, 2, 4])
    assert buckets.compute_hash() != base
    # Unset and True resolve to the same graph; False compiles a static one.
    assert RBLNConfig(use_dynamic_kv_cache=True).compute_hash() == base
    assert RBLNConfig(use_dynamic_kv_cache=False).compute_hash() != base


# Resolving the vllm path reads the host's device now that RBLN-CA* refuses it,
# so the chip is named here rather than inherited from whatever card ran this.
@pytest.mark.usefixtures("cr13")
class TestResolveModelImpl:
    """The model path has to be readable before the config class is known.

    Which path runs picks the class, so `resolve_model_impl` reads upstream's
    `--model-impl` and the environment instead of building anything.
    """

    def test_a_built_config_states_its_own_path(self):
        """Each class defaults to the path it belongs to.

        `LLM(additional_config=RBLNConfig(...))` is a documented way in, and a
        native config that answered "optimum" here would be resolved as one and
        rejected for not being an OptimumRBLNConfig.
        """
        assert resolve_model_impl(RBLNConfig()) == "vllm"
        assert resolve_model_impl(OptimumRBLNConfig()) == "optimum"

    def test_a_disagreeing_deprecated_variable_is_rejected(self, monkeypatch):
        """Two inputs naming the path differently is a mistake, not a ladder.

        Every other field takes the additional_config value over the
        environment. This one cannot: the plugin entry points have acted on the
        variable before anything reads the flag.
        """
        # TODO(vllm-rbln>=0.14.0): delete with VLLM_RBLN_USE_VLLM_MODEL itself.
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL"):
            resolve_model_impl(model_impl="optimum")
        with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL"):
            resolve_model_impl(OptimumRBLNConfig())

        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "0")
        with pytest.raises(ValueError, match="VLLM_RBLN_USE_VLLM_MODEL"):
            resolve_model_impl(model_impl="vllm")

    def test_an_agreeing_deprecated_variable_is_not_a_conflict(self, monkeypatch):
        # TODO(vllm-rbln>=0.14.0): delete with VLLM_RBLN_USE_VLLM_MODEL itself.
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        assert resolve_model_impl(model_impl="vllm") == "vllm"
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "0")
        assert resolve_model_impl(model_impl="optimum") == "optimum"

    def test_nothing_given_is_the_default_path(self, monkeypatch):
        # Both names this suite sets are what the default is the absence of: the
        # deprecated variable, and the path a parent hands down, which the
        # conftest states for the whole session.
        monkeypatch.delenv("VLLM_RBLN_USE_VLLM_MODEL", raising=False)
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", None)
        assert resolve_model_impl() == "optimum"
        assert resolve_model_impl({}) == "optimum"
        assert resolve_model_impl(None) == "optimum"

    def test_the_deprecated_variable_still_selects_the_path(self, monkeypatch):
        # TODO(vllm-rbln>=0.14.0): delete with VLLM_RBLN_USE_VLLM_MODEL itself.
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", None)
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        assert resolve_model_impl() == "vllm"

    def test_the_inherited_path_wins_over_the_deprecated_variable(self, monkeypatch):
        # A spawned process is handed the resolved path; what the shell exported
        # has already been folded into it.
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", "optimum")
        assert resolve_model_impl() == "optimum"

    def test_the_flag_wins_over_the_inherited_path(self, monkeypatch):
        """A worker is handed the path, but an explicit flag still decides.

        Nothing relies on this today; it keeps the ladder total, so a reader
        does not have to guess which of the two wins.
        """
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", "optimum")
        assert resolve_model_impl(model_impl="vllm") == "vllm"

    @pytest.mark.parametrize(
        ("given", "resolved"),
        [("vllm", "vllm"), ("optimum", "optimum"), ("transformers", "optimum")],
    )
    def test_upstream_spellings_name_a_path(self, given, resolved):
        """`--model-impl` is the flag now, so its vocabulary is what arrives.

        `transformers` is the optimum path's name there: the models it runs come
        from optimum-rbln, which is a transformers implementation, and upstream's
        own Transformers backend does not run on RBLN.
        """
        assert resolve_model_impl(model_impl=given) == resolved

    def test_auto_leaves_the_path_to_the_ladder(self, monkeypatch):
        """Every EngineArgs carries `auto`, typed or not.

        Read as a path it would overrule what a parent handed down, so it is no
        answer at all: the process that was handed one keeps it, and the one
        that was handed nothing takes the default.
        """
        monkeypatch.delenv("VLLM_RBLN_USE_VLLM_MODEL", raising=False)
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", "vllm")
        assert resolve_model_impl(model_impl="auto") == "vllm"

        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", None)
        assert resolve_model_impl(model_impl="auto") == "optimum"

    def test_a_flag_that_disagrees_with_the_config_class_is_rejected(self):
        """The class holds one path's options and the flag names a path.

        Letting either win silently drops the other: the class would ignore what
        the caller typed, and the flag would hand the resolution a config it
        cannot read.
        """
        with pytest.raises(ValueError, match="--model-impl names the optimum"):
            resolve_model_impl(RBLNConfig(), model_impl="optimum")
        with pytest.raises(ValueError, match="--model-impl names the vllm"):
            resolve_model_impl(OptimumRBLNConfig(), model_impl="vllm")

        # The same pair agreeing is how a built config is normally passed.
        assert resolve_model_impl(RBLNConfig(), model_impl="vllm") == "vllm"
        assert resolve_model_impl(OptimumRBLNConfig(), model_impl="auto") == "optimum"

    @pytest.mark.parametrize(
        ("device_name", "model_impl", "refused"),
        [
            ("RBLN-CA25", "vllm", True),
            (" rbln-ca02 ", "vllm", True),
            ("RBLN-CA25", "optimum", False),
            ("RBLN-CR03", "vllm", False),
        ],
    )
    def test_a_disabled_path_is_rejected(
        self, monkeypatch, device_name, model_impl, refused
    ):
        """The vllm path is disabled on RBLN-CA*, and this is where paths are named.

        Refusing at resolution puts the failure on the flag the caller typed,
        before anything is built from it.
        """
        from vllm_rbln import platform

        monkeypatch.setattr(
            platform.rebel, "get_npu_name", lambda *a, **kw: device_name
        )
        if refused:
            with pytest.raises(ValueError, match=device_name.strip()):
                resolve_model_impl(model_impl=model_impl)
        else:
            assert resolve_model_impl(model_impl=model_impl) == model_impl

    @pytest.mark.parametrize(
        ("kwargs", "remedy"),
        [
            (
                {"model_impl": "vllm"},
                "--model-impl vllm selected it. Use --model-impl optimum.",
            ),
            (
                {"additional_config": RBLNConfig()},
                "an RBLNConfig additional_config selected it. Pass an "
                "OptimumRBLNConfig instead.",
            ),
        ],
    )
    def test_the_refusal_names_what_selected_the_path(
        self, monkeypatch, kwargs, remedy
    ):
        """Three inputs can name this path and each is undone somewhere else.

        A caller sent to --model-impl optimum by name lands on the refusal above
        this one whenever a config class chose the path.
        """
        from vllm_rbln import platform

        monkeypatch.setattr(
            platform.rebel, "get_npu_name", lambda *a, **kw: "RBLN-CA25"
        )
        with pytest.raises(ValueError, match=re.escape(remedy)):
            resolve_model_impl(**kwargs)

    def test_the_refusal_names_the_deprecated_variable(self, monkeypatch):
        # TODO(vllm-rbln>=0.14.0): delete with VLLM_RBLN_USE_VLLM_MODEL itself.
        from vllm_rbln import platform

        monkeypatch.setattr(
            platform.rebel, "get_npu_name", lambda *a, **kw: "RBLN-CA25"
        )
        monkeypatch.setattr(envs, "INHERITED_MODEL_IMPL", None)
        monkeypatch.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        with pytest.raises(
            ValueError,
            match="VLLM_RBLN_USE_VLLM_MODEL selected it. Unset "
            "VLLM_RBLN_USE_VLLM_MODEL.",
        ):
            resolve_model_impl()

    def test_a_host_that_cannot_name_its_npu_keeps_every_path(self, monkeypatch):
        """A compile-only worker names its target later, so nothing is refused."""
        from vllm_rbln import platform

        monkeypatch.setattr(platform.rebel, "get_npu_name", lambda *a, **kw: None)
        monkeypatch.delenv("RBLN_FORCE_NPU_NAME", raising=False)
        monkeypatch.delenv("RBLN_TARGET_SOC", raising=False)
        assert resolve_model_impl(model_impl="vllm") == "vllm"

    @pytest.mark.parametrize("value", ["terratorch", "vLLM", "", True])
    def test_an_unsupported_implementation_is_rejected(self, value):
        # `terratorch` is upstream's fourth value and has no RBLN
        # implementation. None is not here: it is the argument's own absence.
        with pytest.raises(ValueError, match="unsupported model implementation"):
            resolve_model_impl(model_impl=value)


def test_the_path_has_no_rbln_flag_of_its_own(parser):
    """Upstream's `--model-impl` is the flag, so this field does not add one.

    Registering both would give the same field two spellings that can disagree.
    """
    group = next(g for g in parser._action_groups if g.title == _GROUP_TITLE)
    assert "rbln_model_impl" not in {a.dest for a in group._group_actions}
    assert "model_impl" in {a.dest for a in parser._actions}


def test_model_impl_has_no_variable_of_its_own():
    """The path is a flag now, so it gains no `VLLM_RBLN_MODEL_IMPL`.

    Only the deprecated name keeps working, and only through
    `resolve_model_impl`.
    """
    from vllm_rbln import envs

    assert _env_source("model_impl")[0] not in envs.environment_variables


def test_only_two_modules_decide_the_model_path():
    """AGENTS.md: the selector lives in config.py, two modules branch on it.

    The whole split rests on this. A path branch anywhere else sends the two
    paths down one another's code, and reading the resolved path is what a
    branch starts from, so that is what this looks for.
    """
    import vllm_rbln

    root = pathlib.Path(vllm_rbln.__file__).parent
    allowed = {"__init__.py", "platform/__init__.py", "config.py", "envs.py"}
    readers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.relative_to(root).as_posix() not in allowed
        and (
            "model_impl_from_env" in (text := path.read_text())
            or "resolve_model_impl" in text
        )
    )
    assert not readers, f"{readers} resolve the model path; see AGENTS.md"


def test_the_deprecated_variable_survives_being_set_before_vllm_is_imported():
    """How every caller that has not migrated still starts: export, then import.

    `vllm_rbln.platform` reads the path while `vllm` is part-way through
    importing itself, so anything on that road that touches `vllm` again -- a
    logger, most easily -- raises out of the half-built module. A subprocess,
    because the failure is in import order and this one is long past it.
    """
    import os
    import subprocess
    import sys

    probe = (
        "import os;"
        "os.environ['VLLM_RBLN_USE_VLLM_MODEL'] = '1';"
        "from vllm import LLM;"
        "from vllm_rbln.config import resolve_model_impl;"
        "print('PATH' + resolve_model_impl())"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_RBLN")}
    env.pop(RESOLVED_MODEL_IMPL_ENV, None)
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )

    assert out.returncode == 0, out.stderr[-2000:]
    assert "PATHvllm" in out.stdout, out.stdout
