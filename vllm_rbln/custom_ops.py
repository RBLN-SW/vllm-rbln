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

from collections.abc import Callable
from typing import Any

# Required, and must come first: the decorators below check what it registers.
import rebel  # noqa: F401
import torch

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

#: Ops already registered when this package's copy ran, so that existing
#: definition is the one in use.
ALREADY_DEFINED: set[str] = set()

#: Schema this package declares for each op, whether or not its copy was used.
#: This is what the drift warning compares against the existing registration.
DECLARED_SCHEMAS: dict[str, str] = {}


def custom_op(name: str, **kwargs: Any) -> Callable[[Callable], Any]:
    """Define `name` unless it is already defined. Drop-in for
    torch.library.custom_op."""

    def decorator(fn: Callable) -> Any:
        declared_schema = _infer_schema(fn, kwargs)
        if declared_schema is not None:
            DECLARED_SCHEMAS[name] = declared_schema

        existing = _lookup(name)
        if existing is not None:
            ALREADY_DEFINED.add(name)
            _warn_on_schema_drift(name)
            return existing

        logger.info(
            "%s is not registered yet; falling back to the definition in vllm-rbln.",
            name,
        )
        return torch.library.custom_op(name, **kwargs)(fn)

    return decorator


def register_fake(name: str, **kwargs: Any) -> Callable[[Callable], Callable]:
    """Register a fake kernel for `name`, unless the op is already defined.

    An existing op's fake goes with its schema; overriding one and not the other
    is how the two copies would drift apart unnoticed.
    """

    def decorator(fn: Callable) -> Callable:
        if name in ALREADY_DEFINED:
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
    """Report a copy in this package that no longer matches the registered op."""
    ours = DECLARED_SCHEMAS.get(qualname)
    theirs = _existing_schema(qualname)
    if ours is None or theirs is None or ours == theirs:
        return
    logger.warning(
        "%s is already registered with a different signature than the copy in "
        "vllm-rbln. The registered definition is used; the copy in this package "
        "is stale.\n  registered: %s\n  vllm-rbln:  %s",
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
