"""Every RBLN dependency must be pinned to a single patch line.

vllm-rbln and the rest of the RBLN SDK ship as a matched set: the SDK version
matrix documents exactly one optimum-rbln for each vllm-rbln release. A
requirement carrying only a floor (``optimum-rbln>=0.11.2``) does not hold that
promise, because a *released* wheel resolves against whatever is newest at
install time, not at release time. That is how the matrix check ended up
reporting, for thirteen documented releases at once::

    vllm-rbln==0.0.6 brings optimum-rbln==0.11.2, but the matrix documents
    optimum-rbln==0.1.9

Those wheels are published and cannot be corrected; what can be corrected is
that every future one carries a ceiling. These tests fail the PR that would
publish another unbounded one.

Scope is packages whose name contains ``rbln`` -- the SDK components released in
lockstep with this project. ``rebel-compiler`` is deliberately not covered: it is
selected per build rather than per release (see the ``runtime`` extra), so it is
bounded by its own convention.
"""

import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 -- `requires-python` still admits it.
    tomllib = pytest.importorskip("tomli", reason="TOML parser unavailable on 3.10")

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# The SDK components this project releases in lockstep with. Asserted to be
# present so that renaming one cannot silently empty the parametrisation below
# and leave the guard passing while checking nothing.
EXPECTED_RBLN_REQUIREMENTS = {"optimum-rbln", "torch-rbln"}


def _iter_rbln_requirements() -> list[tuple[str, Requirement]]:
    """Every ``*rbln*`` requirement in pyproject, with where it was declared."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})

    raw: list[tuple[str, str]] = [
        ("project.dependencies", dep) for dep in project.get("dependencies", [])
    ]
    for extra, deps in project.get("optional-dependencies", {}).items():
        raw += [(f"project.optional-dependencies.{extra}", dep) for dep in deps]
    for group, deps in data.get("dependency-groups", {}).items():
        raw += [(f"dependency-groups.{group}", dep) for dep in deps]

    found = []
    for location, dep in raw:
        requirement = Requirement(dep)
        if "rbln" in requirement.name.lower():
            found.append((location, requirement))
    return found


RBLN_REQUIREMENTS = _iter_rbln_requirements()
_IDS = [f"{location}::{req.name}" for location, req in RBLN_REQUIREMENTS]


def _floor(specifier: SpecifierSet) -> Version | None:
    """The lowest version the specifier admits, or None when it has no floor."""
    lower = [
        Version(spec.version.rstrip(".*"))
        for spec in specifier
        if spec.operator in (">=", "==", "~=")
    ]
    return min(lower) if lower else None


def test_every_expected_rbln_requirement_is_declared():
    """The guard below is parametrised; keep it from silently covering nothing."""
    declared = {req.name for _, req in RBLN_REQUIREMENTS}
    missing = EXPECTED_RBLN_REQUIREMENTS - declared
    assert not missing, (
        f"{sorted(missing)} is no longer declared in {PYPROJECT.name}. If it was "
        "renamed or dropped on purpose, update EXPECTED_RBLN_REQUIREMENTS; "
        "otherwise the version-bound guard below is no longer covering it."
    )


@pytest.mark.parametrize("location, requirement", RBLN_REQUIREMENTS, ids=_IDS)
def test_rbln_requirement_declares_a_floor(location: str, requirement: Requirement):
    specifier = requirement.specifier
    assert _floor(specifier) is not None, (
        f"`{requirement}` in [{location}] declares no lower bound. An RBLN "
        "dependency needs both bounds: state the oldest release this one works "
        "with, e.g. `>=0.11.2,<0.11.3`."
    )


@pytest.mark.parametrize("location, requirement", RBLN_REQUIREMENTS, ids=_IDS)
def test_rbln_requirement_is_capped_to_its_patch_line(
    location: str, requirement: Requirement
):
    """A ceiling, and a tight one: only the floor's own patch line may resolve.

    Checked by what the specifier *admits* rather than by which operators it
    spells, so `>=X.Y.Z,<X.Y.Z+1` and `~=X.Y.Z.0` both pass while a minor-level
    cap (`<X.Y+1`) does not -- the latter still lets a release drift off the
    version the matrix documents for it.
    """
    specifier = requirement.specifier
    floor = _floor(specifier)
    if floor is None:
        pytest.skip("no floor; reported by test_rbln_requirement_declares_a_floor")

    assert specifier.contains(floor), (
        f"`{requirement}` in [{location}] excludes its own lower bound {floor}; "
        "the bounds contradict each other."
    )

    # Post-releases of the floor stay in (0.11.2.post1 fixes 0.11.2); the next
    # patch, minor and major must not.
    for label, probe in (
        ("next patch", Version(f"{floor.major}.{floor.minor}.{floor.micro + 1}")),
        ("next minor", Version(f"{floor.major}.{floor.minor + 1}.0")),
        ("next major", Version(f"{floor.major + 1}.0.0")),
    ):
        assert not specifier.contains(probe), (
            f"`{requirement}` in [{location}] still admits {probe} ({label}). "
            f"An RBLN dependency must be capped to the patch line of its floor "
            f"({floor}), so a published wheel keeps resolving to the version the "
            f"SDK matrix documents for it -- write "
            f"`>={floor},<{floor.major}.{floor.minor}.{floor.micro + 1}`."
        )
