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

import contextlib
import json
import os
import warnings

import pytest

from tests.native.utils import (
    _SPAWN_CHILD_ENV,
    _SPAWN_RESULTS_ENV,
    LAYERS_PINNABLE_ENV,
    NATIVE_ENV,
    ModuleSpawn,
    read_log_tail,
    scrub_env,
)

_DEFAULT_NUM_HIDDEN_LAYERS = 3

_scrubbed: dict[str, str] = {}

# module file -> its (possibly still running) batch spawn.
_module_spawns: dict[str, ModuleSpawn] = {}

# The one spawn allowed to be live at a time; see pytest_runtest_protocol.
_active_spawn: ModuleSpawn | None = None

# Set in pytest_configure: the report (de)serialization hooks need a config, and
# the hooks that use them (logreport, warning_recorded) are not handed one. Typed
# non-optional because every reader is a hook that cannot fire before configure.
_config: pytest.Config = None  # type: ignore[assignment]


def _intra_nodeid(nodeid: str) -> str:
    # Drop the file path so parent and child (invoked with different paths) match.
    return nodeid.split("::", 1)[1] if "::" in nodeid else nodeid


def _needs_spawn(item) -> bool:
    """Whether ``item`` must run in a fresh process rather than in this one."""
    kw = item.keywords
    if "model_compile" in kw or "use_device" in kw:
        return True
    if "maybe_use_device" in kw:
        from vllm.platforms import current_platform

        return current_platform.device_type != "cpu"
    return False


def _will_be_skipped(item) -> bool:
    """Whether pytest will skip ``item`` outright -- notably the skip mark
    pytest_collection_modifyitems adds without --model-compile. Such an item must
    NOT trigger a spawn: today the skip lands in setup and no child is ever
    started, and `pytest tests/native` must keep costing nothing for them."""
    try:
        from _pytest.skipping import evaluate_skip_marks

        return evaluate_skip_marks(item) is not None
    except Exception:
        return item.get_closest_marker("skip") is not None


def _replay_warnings(spawn) -> None:
    """Re-emit the child's warnings into this session's warnings summary.

    The category class cannot cross the process boundary, so stand in a subclass
    carrying its name -- the summary renders the name, so it reads identically.
    pytest_warning_recorded is a *historic* hook, hence call_historic rather than
    a plain call. Note the child records a test's runtest warnings only after its
    teardown report (that is when pytest's catch_warnings block for the item
    exits), so they are always drained one item late -- which is why the session
    finish hook drains once more."""
    for rec in spawn.take_warnings():
        message = warnings.WarningMessage(
            message=rec["message"],
            category=type(rec["category"], (Warning,), {}),
            filename=rec["filename"],
            lineno=rec["lineno"],
        )
        with contextlib.suppress(Exception):
            _config.hook.pytest_warning_recorded.call_historic(
                kwargs=dict(
                    warning_message=message,
                    when=rec["when"],
                    nodeid=(
                        f"{spawn.nodeid_prefix}::{rec['nodeid']}"
                        if rec["nodeid"]
                        else ""
                    ),
                    location=tuple(rec["location"]) if rec["location"] else None,
                )
            )


def pytest_addoption(parser):
    parser.addoption(
        "--model-compile",
        action="store_true",
        default=False,
        help="run tests that compile a whole model on the NPU (minutes each)",
    )
    parser.addoption(
        "--device-tensor",
        choices=["0", "1"],
        default=None,
        help=(
            "VLLM_RBLN_USE_DEVICE_TENSOR for the whole session. platform.py "
            "resolves it at module scope into RblnPlatform.device_type and "
            "friends, and seven modules copy USE_DEVICE_TENSOR into their own "
            "namespace, so it cannot be parametrized per test -- run the suite "
            "once per value instead. Left unset by default so the source's own "
            "default is what gets exercised."
        ),
    )
    parser.addoption(
        "--num-hidden-layers",
        type=int,
        default=None,
        help=(
            "VLLM_RBLN_NUM_HIDDEN_LAYERS for the whole session: build only the "
            "first N decoder layers, cutting compile time in the "
            "--model-compile lane. hf_runner truncates to the same N, so "
            "correctness comparisons stay like-for-like. 0 runs the whole "
            "model. An exported value is scrubbed like every other "
            f"VLLM_RBLN_* knob; this option is the way in. Defaults to "
            f"{_DEFAULT_NUM_HIDDEN_LAYERS}, and only then does a spec's own "
            "num_hidden_layers apply."
        ),
    )


def _session_layers(config) -> int:
    """The option's value, or the default when it was left off. Not `or`: an
    explicit 0 is the whole model, not a missing value."""
    layers = config.getoption("--num-hidden-layers")
    return _DEFAULT_NUM_HIDDEN_LAYERS if layers is None else layers


def pytest_collection_modifyitems(config, items):
    """Keep whole-model compiles out of the default lane (opt-in via
    --model-compile; forgetting the flag costs nothing)."""
    if config.getoption("--model-compile"):
        return
    skip = pytest.mark.skip(reason="needs --model-compile")
    for item in items:
        if "model_compile" in item.keywords:
            item.add_marker(skip)


def _record(payload: dict) -> None:
    """Append one event to the batch child's results file. No-op outside a batch
    child (the env var is only set there)."""
    results_file = os.environ.get(_SPAWN_RESULTS_ENV)
    if not results_file:
        return
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def pytest_runtest_logreport(report):
    """In a module-batch child, hand every phase report to the parent in
    pytest's own serialized form.

    Serializing the report rather than a summary of it is the whole point: the
    parent revives it and re-emits it, so pytest renders the child's real report
    -- correct exception type in the short summary, structured traceback honoring
    --tb, skip reasons, xfail/xpass, captured sections, per-test durations. A
    hand-rolled `pytest.fail(text)` reconstruction gets all of that subtly wrong.
    All three phases go over, so a setup error or teardown error reports exactly
    as it would have in a plain run."""
    if not os.environ.get(_SPAWN_RESULTS_ENV):
        return
    data = _config.hook.pytest_report_to_serializable(config=_config, report=report)
    if data is None:
        return
    _record(
        {
            "kind": "report",
            "nodeid": _intra_nodeid(report.nodeid),
            "when": report.when,
            "report": data,
        }
    )


def pytest_warning_recorded(warning_message, when, nodeid, location):
    """Forward the child's warnings so they reach the parent's warnings summary
    instead of dying with the child."""
    if not os.environ.get(_SPAWN_RESULTS_ENV):
        return
    _record(
        {
            "kind": "warning",
            "message": str(warning_message.message),
            "category": getattr(warning_message.category, "__name__", "Warning"),
            "filename": warning_message.filename,
            "lineno": warning_message.lineno,
            "when": when,
            "nodeid": _intra_nodeid(nodeid) if nodeid else nodeid,
            "location": list(location) if location else None,
        }
    )


def _spawn_for(item) -> ModuleSpawn:
    """The batch child for ``item``'s file, started on first use."""
    global _active_spawn
    from tests.native.utils import start_module_in_spawned_process

    module_file = str(item.path)
    spawn = _module_spawns.get(module_file)
    if spawn is not None:
        return spawn

    # Exactly one child on the NPU at a time. The previous module's child can
    # still be finishing its last test -- wait it out before spawning.
    if _active_spawn is not None:
        _active_spawn.finish()

    config = item.config
    # Hand over exactly the tests this session selected from that file and that
    # must not run in-process, so -m/-k are honored and a mixed-marker file's
    # in-process tests are not run a second time in the child.
    nodeids = [
        other.nodeid
        for other in item.session.items
        if str(other.path) == module_file
        and _needs_spawn(other)
        and not _will_be_skipped(other)
    ]
    spawn = start_module_in_spawned_process(
        nodeids or [item.nodeid],
        nodeid_prefix=item.nodeid.split("::", 1)[0],
        device_tensor=config.getoption("--device-tensor"),
        model_compile=config.getoption("--model-compile"),
        num_hidden_layers=_session_layers(config),
        tb_style=config.getoption("tbstyle", None),
        maxfail=config.getoption("maxfail", 0),
        stream_output=config.getoption("capture") == "no",
    )
    _module_spawns[module_file] = spawn
    _active_spawn = spawn
    return spawn


def _crash_reports(item, spawn) -> list:
    """Stand-in reports for a test the child never reported, i.e. it died on or
    before it. The one case that cannot look like a plain pytest run: a hard
    crash (segfault, SIGKILL, os._exit) takes pytest's own capture buffer down
    with the child, so the running test's output is unrecoverable."""
    from _pytest.reports import TestReport

    text = (
        f"the module-batch child died without reporting this test.\n"
        f"Its in-test output died with it; re-run with -s to see it live:\n"
        f"    pytest {item.nodeid} -s\n\n"
        f"Tail of {spawn.log_path or 'the child output'}:\n\n"
        f"{read_log_tail(spawn.log_path)}"
    )
    common = dict(nodeid=item.nodeid, location=item.location, keywords=item.keywords)
    return [
        TestReport(**common, outcome="passed", longrepr=None, when="setup"),
        TestReport(**common, outcome="failed", longrepr=text, when="call"),
        TestReport(**common, outcome="passed", longrepr=None, when="teardown"),
    ]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Run device-touching tests in a fresh spawned process, batched per file:
    the RBLN device must not be opened in the parent (it would pin the device for
    the whole session), and one spawn per file amortizes the interpreter + plugin
    + device init over all its tests.

    @model_compile and @use_device always spawn -- they compile/run on the NPU
    regardless of --device-tensor. @maybe_use_device spawns only when device
    tensors are on (--device-tensor 1); under 0 its ops stay on CPU.

    Taking over the whole protocol (rather than just the call phase) is what
    makes a batched test indistinguishable from a plain one: the child's own
    setup/call/teardown reports are re-emitted here verbatim, so every downstream
    consumer -- terminal, -r flags, --durations, junitxml, --lf -- sees what it
    would have seen had the test run in this process."""
    # Inside a batch child this hook fires too; run normally there.
    if os.environ.get(_SPAWN_CHILD_ENV) == "1":
        return None
    # A skipped test must not start a child: without --model-compile the whole
    # model_compile lane is skipped, and that has to stay free.
    if not _needs_spawn(item) or _will_be_skipped(item):
        return None

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    spawn = _spawn_for(item)
    # Blocks only until THIS test is fully reported, so the parent's progress
    # display advances test by test instead of all at once per file.
    records = spawn.records_for(_intra_nodeid(item.nodeid))
    if records is None:
        reports = _crash_reports(item, spawn)
    else:
        reports = []
        for data in records:
            report = _config.hook.pytest_report_from_serializable(
                config=_config, data=data
            )
            # The child addressed the test by its own nodeid; re-anchor to this
            # session's item so the terminal, cache and junitxml agree.
            report.nodeid = item.nodeid
            report.location = item.location
            reports.append(report)

    _replay_warnings(spawn)
    for report in reports:
        ihook.pytest_runtest_logreport(report=report)

    # Nothing was set up for this item here, but earlier in-process items may
    # still hold higher-scope fixtures; unwind them to match nextitem exactly as
    # the standard teardown phase would have.
    with contextlib.suppress(Exception):
        item.session._setupstate.teardown_exact(nextitem)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_configure(config):
    # Must run before collection: register_ops() gates on
    # VLLM_RBLN_USE_VLLM_MODEL, and test modules capture upstream symbols at
    # import time -- so the patches have to be in place before any of them
    # execute `from vllm.xxx import yyy`.
    global _scrubbed, _config
    _config = config
    _scrubbed = scrub_env()
    os.environ.update(NATIVE_ENV)

    # Must land before the import below: platform.py reads this at module scope.
    device_tensor = config.getoption("--device-tensor")
    if device_tensor is not None:
        os.environ["VLLM_RBLN_USE_DEVICE_TENSOR"] = device_tensor

    # Also before the import below: the get_pp_indices patch conditions on this.
    os.environ["VLLM_RBLN_NUM_HIDDEN_LAYERS"] = str(_session_layers(config))

    # Parent only -- the child is handed the resolved value, so its own view of
    # the option is always "given". Cleared rather than merely left unset, or an
    # exported one would pin behind an explicit option's back.
    if os.environ.get(_SPAWN_CHILD_ENV) != "1":
        if config.getoption("--num-hidden-layers") is None:
            os.environ[LAYERS_PINNABLE_ENV] = "1"
        else:
            os.environ.pop(LAYERS_PINNABLE_ENV, None)

    # Platform plugins activate on their own when current_platform is first
    # touched, but the patches live in the general_plugins group and nothing
    # loads those implicitly. Without this the suite runs half-applied:
    # RblnPlatform is current, yet every patched symbol is still upstream's.
    from vllm.plugins import load_general_plugins

    load_general_plugins()


def pytest_sessionfinish(session):
    """Reap any child still running (a module whose last selected test finished
    before the child ran out the rest of its file), then drop the spawn logs on
    a green session -- keeping them when something failed, so the full child
    output is still there to read after the run."""
    for spawn in _module_spawns.values():
        spawn.close()
        _replay_warnings(spawn)
    if session.testsfailed:
        return
    for spawn in _module_spawns.values():
        if spawn.log_path:
            with contextlib.suppress(OSError):
                os.remove(spawn.log_path)


def pytest_report_header(config):
    # Imported by pytest_configure already; report the resolved value rather
    # than the request, since that is the lane the whole session runs in.
    from vllm_rbln.platform import RblnPlatform

    origin = "explicit" if config.getoption("--device-tensor") else "source default"
    header = [
        f"native: env {', '.join(f'{k}={v}' for k, v in NATIVE_ENV.items())}",
        f"native: device_type={RblnPlatform.device_type} ({origin})",
    ]
    num_hidden_layers = _session_layers(config)
    pinnable = (
        " (a spec may pin its own)" if os.environ.get(LAYERS_PINNABLE_ENV) else ""
    )
    header.append(
        f"native: num_hidden_layers={num_hidden_layers or 'whole model'}{pinnable}",
    )
    if _scrubbed:
        header.append(f"native: scrubbed {', '.join(sorted(_scrubbed))}")
    return header


@pytest.fixture(scope="session")
def vllm_runner():
    """VllmRunner class (lazy import so vllm stays out of module scope)."""
    from tests.native.runners import VllmRunner

    return VllmRunner


@pytest.fixture(scope="session")
def async_vllm_runner():
    """AsyncVllmRunner class -- the DP-capable runner (lazy import; see
    vllm_runner)."""
    from tests.native.runners import AsyncVllmRunner

    return AsyncVllmRunner


@pytest.fixture(scope="session")
def hf_runner():
    """HfRunner class (lazy import; see vllm_runner)."""
    from tests.native.runners import HfRunner

    return HfRunner


@pytest.fixture(scope="session")
def whole_model(pytestconfig) -> bool:
    """Whether every decoder layer is built (--num-hidden-layers 0). A truncated
    model still compiles and runs, but its logits are meaningless -- an assertion
    that depends on output quality has to gate on this."""
    return pytestconfig.getoption("--num-hidden-layers") == 0


@pytest.fixture(autouse=True)
def _drop_envs_shadows():
    """Remove any ``vllm_rbln.envs`` attribute a test left behind.

    `monkeypatch.setattr(envs, NAME, ...)` cannot clean up after itself here: envs
    serves those names through the module's ``__getattr__``, so monkeypatch records
    the value that produced and its undo puts it back as a *real* attribute -- which
    then wins over ``__getattr__`` and freezes that variable for the rest of the
    session, silently defeating any later monkeypatch.setenv on it.
    """
    from vllm_rbln import envs

    before = set(vars(envs))
    yield
    for name in set(vars(envs)) - before:
        delattr(envs, name)


@pytest.fixture(autouse=True)
def _isolate_rbln_ctx_standalone():
    """Clear the one env var the code under test writes to the process env.

    RblnPlatform.validate_and_setup_prerequisite sets RBLN_CTX_STANDALONE=1 for
    any TP/DP/PP/EP config and never clears it. The rebel runtime reads it on
    every context creation, so one test building such a config leaves every later
    test -- and every spawned child, which inherits the env -- unable to register
    a device at all. Mirrors tests/torch_compile/conftest.py."""
    os.environ.pop("RBLN_CTX_STANDALONE", None)
    yield


@pytest.fixture(autouse=True)
def _reset_memoized_env_reads():
    """Clear the one env read that sticks process-wide: get_use_w8a16() caches
    into a module global, pinning the value session-wide otherwise."""
    from vllm_rbln import envs

    envs._USE_W8A16 = None
    yield
    envs._USE_W8A16 = None
