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

"""Who owns the `rbln_custom_ops::*` torch custom ops.

rebel-compiler is the single source of truth: it registers these ops on
`import rebel`, and it is the component that lowers them, so its schema is the
one that matters. vllm-rbln still carries a copy of each definition, used only
when the installed compiler predates the move. Registering through
`custom_op` / `register_fake` here instead of `torch.library` picks the
compiler's registration whenever there is one.

Without this indirection the outcome would depend on import order and be
silent: `torch.library.custom_op` does not reject a duplicate, it replaces the
schema and the implementation. vllm-rbln's modules load after `platform.py`
does `import rebel`, so a plain decorator would quietly shadow the compiler's
definition with this package's copy.

The fallback is a transition device, not the end state. `fallback_ops()`
reports which ops it had to supply; once the minimum supported rebel-compiler
registers all of them, that set is empty everywhere and the definitions in
this package can go.
"""

from collections.abc import Callable
from typing import Any

import torch

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

try:
    # Import for the side effect: a new enough rebel-compiler registers the ops
    # here, before any of our definitions run. Doing it at this point rather
    # than relying on platform.py keeps the precedence independent of which
    # module happens to load first.
    import rebel  # noqa: F401

    _COMPILER_AVAILABLE = True
except ImportError:
    _COMPILER_AVAILABLE = False
    logger.debug(
        "rebel-compiler is not installed; vllm-rbln defines the rbln custom ops itself."
    )

#: Ops taken from rebel-compiler.
FROM_COMPILER: set[str] = set()

#: Ops this package had to define because the compiler does not provide them.
FALLBACK: set[str] = set()

#: Schema of this package's copy of each op, whether or not it was used. Lets a
#: test compare against what the compiler registered without duplicating the
#: signatures.
LOCAL_SCHEMAS: dict[str, str] = {}


def fallback_ops() -> frozenset[str]:
    """Ops vllm-rbln registered itself, because rebel-compiler lacks them.

    Empty on a supported compiler. Tests assert that, so the fallback cannot
    quietly become the path everything runs on.
    """
    return frozenset(FALLBACK)


def custom_op(name: str, **kwargs: Any) -> Callable[[Callable], Any]:
    """Define `name` unless rebel-compiler already did. Drop-in for
    torch.library.custom_op."""

    def decorator(fn: Callable) -> Any:
        local_schema = _infer_schema(fn, kwargs)
        if local_schema is not None:
            LOCAL_SCHEMAS[name] = local_schema

        existing = _lookup(name)
        if existing is not None:
            FROM_COMPILER.add(name)
            _warn_on_schema_drift(name)
            return existing

        FALLBACK.add(name)
        logger.debug(
            "%s is not provided by the installed rebel-compiler; "
            "falling back to the definition in vllm-rbln.",
            name,
        )
        return torch.library.custom_op(name, **kwargs)(fn)

    return decorator


def register_fake(name: str, **kwargs: Any) -> Callable[[Callable], Callable]:
    """Register a fake kernel for `name`, unless the compiler owns the op.

    The compiler's fake goes with the compiler's schema; overriding one and not
    the other is how the two copies would drift apart unnoticed.
    """

    def decorator(fn: Callable) -> Callable:
        if name in FROM_COMPILER:
            return fn
        torch.library.register_fake(name, **kwargs)(fn)
        return fn

    return decorator


def _lookup(qualname: str) -> Any | None:
    """Return the registered op packet for "ns::op", or None if it is unknown."""
    ns, _, op_name = qualname.partition("::")
    if not op_name:
        raise ValueError(f"op name must be 'namespace::op_name', got: {qualname}")
    namespace = getattr(torch.ops, ns, None)
    if namespace is None:
        return None
    try:
        return getattr(namespace, op_name)
    except AttributeError:
        return None


def _warn_on_schema_drift(qualname: str) -> None:
    """Report a fallback copy that no longer matches what the compiler lowers."""
    ours = LOCAL_SCHEMAS.get(qualname)
    theirs = _existing_schema(qualname)
    if ours is None or theirs is None or ours == theirs:
        return
    logger.warning(
        "%s has a different signature in vllm-rbln than in the installed "
        "rebel-compiler. The compiler's definition is used; the copy in this "
        "package is stale.\n  rebel-compiler: %s\n  vllm-rbln:      %s",
        qualname,
        theirs,
        ours,
    )


def _existing_schema(qualname: str) -> str | None:
    """Schema of the already-registered op, as "(args) -> ret"."""
    op = _lookup(qualname)
    overload = getattr(op, "default", None)
    schema = getattr(overload, "_schema", None)
    if schema is None:
        return None
    text = str(schema)
    start = text.find("(")
    return text[start:] if start != -1 else None


def _infer_schema(fn: Callable, kwargs: dict[str, Any]) -> str | None:
    """Schema torch would infer for `fn`, in the same "(args) -> ret" form."""
    declared = kwargs.get("schema")
    if declared is not None:
        return declared
    try:
        from torch._library.infer_schema import infer_schema
    except ImportError:  # private API; the comparison is best-effort
        return None
    try:
        return infer_schema(fn, mutates_args=kwargs.get("mutates_args", ()))
    except Exception:
        return None
