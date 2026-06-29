import importlib.util
from pathlib import Path

_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "pre_commit"
    / "check_env_metadata.py"
)
_spec = importlib.util.spec_from_file_location("check_env_metadata", _MOD_PATH)
assert _spec is not None and _spec.loader is not None
check_env_metadata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_env_metadata)

check = check_env_metadata.check
parse_source = check_env_metadata.parse_source


def test_check_passes_when_consistent():
    env_keys = {"VLLM_RBLN_FOO", "VLLM_RBLN_BAR"}
    meta = {
        "VLLM_RBLN_FOO": "A clear description.",
        "VLLM_RBLN_BAR": "Another clear description.",
    }
    assert check(env_keys, meta) == []


def test_check_flags_missing_metadata():
    env_keys = {"VLLM_RBLN_FOO", "VLLM_RBLN_BAR"}
    meta = {"VLLM_RBLN_FOO": "A clear description."}
    errors = check(env_keys, meta)
    assert any("VLLM_RBLN_BAR" in e and "missing" in e for e in errors)


def test_check_flags_orphan_metadata():
    env_keys = {"VLLM_RBLN_FOO"}
    meta = {"VLLM_RBLN_FOO": "A clear description.", "VLLM_RBLN_GHOST": "Stale entry."}
    errors = check(env_keys, meta)
    assert any("VLLM_RBLN_GHOST" in e and "orphan" in e for e in errors)


def test_check_flags_empty_description():
    env_keys = {"VLLM_RBLN_FOO"}
    meta = {"VLLM_RBLN_FOO": ""}
    errors = check(env_keys, meta)
    assert any("VLLM_RBLN_FOO" in e and "description" in e for e in errors)


def test_check_flags_too_short_description():
    env_keys = {"VLLM_RBLN_FOO"}
    meta = {"VLLM_RBLN_FOO": "short"}
    errors = check(env_keys, meta)
    assert any("VLLM_RBLN_FOO" in e for e in errors)


def test_parse_source_extracts_keys_and_descriptions():
    src = """
from vllm.envs import environment_variables as vllm_envs

environment_variables = {
    **vllm_envs,
    "VLLM_RBLN_FOO": lambda: True,
    "OTHER_VAR": lambda: 1,
}

ENV_METADATA = {
    "VLLM_RBLN_FOO": EnvMeta("A clear description.", default=True, type="bool"),
}
"""
    env_keys, meta = parse_source(src)
    assert env_keys == {"VLLM_RBLN_FOO"}
    assert meta == {"VLLM_RBLN_FOO": "A clear description."}


def test_parse_source_handles_concatenated_description():
    src = """
environment_variables = {
    **vllm_envs,
    "VLLM_RBLN_FOO": lambda: True,
}

ENV_METADATA = {
    "VLLM_RBLN_FOO": EnvMeta(
        "First part of the description "
        "and the second part.",
        default=True, type="bool"),
}
"""
    env_keys, meta = parse_source(src)
    assert env_keys == {"VLLM_RBLN_FOO"}
    assert meta == {
        "VLLM_RBLN_FOO": "First part of the description and the second part."
    }
