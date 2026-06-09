# Copyright 2025 Rebellions Inc. All rights reserved.
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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)

MIN_PATCH_PRIORITY = 0
MAX_PATCH_PRIORITY = 100
DEFAULT_PATCH_PRIORITY = 50


@dataclass(frozen=True)
class RegistrationDescriptor:
    key: str
    owner_module: str
    callback: Callable[[], None]
    reason: str


@dataclass(frozen=True)
class PatchDescriptor:
    key: str
    owner_module: str
    target: str
    replacement: Any
    reason: str
    condition: Callable[[], bool] | None = None
    verify: Callable[[], None] | None = None
    priority: int = DEFAULT_PATCH_PRIORITY


_REGISTERED_REGISTRATION_DESCRIPTORS: list[RegistrationDescriptor] = []
_REGISTERED_PATCH_DESCRIPTORS: list[PatchDescriptor] = []


def add_registration(
    *,
    reason: str,
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Add a registration callback."""

    def _decorator(callback: Callable[[], None]) -> Callable[[], None]:
        callback_name = getattr(
            callback,
            "__qualname__",
            getattr(callback, "__name__", type(callback).__qualname__),
        )

        descriptor_owner_module = callback.__module__
        descriptor_key = f"{descriptor_owner_module}.{callback_name}"

        for descriptor in _REGISTERED_REGISTRATION_DESCRIPTORS:
            if descriptor.key == descriptor_key:
                return callback

        _REGISTERED_REGISTRATION_DESCRIPTORS.append(
            RegistrationDescriptor(
                key=descriptor_key,
                owner_module=descriptor_owner_module,
                callback=callback,
                reason=reason,
            )
        )
        return callback

    return _decorator


def register_patch(
    *,
    target: str,
    reason: str,
    key: str | None = None,
    owner_module: str | None = None,
    condition: Callable[[], bool] | None = None,
    verify: Callable[[], None] | None = None,
    priority: int = DEFAULT_PATCH_PRIORITY,
) -> Callable[[Any], Any]:
    """Register a replacement object as an RBLN patch descriptor.

    Args:
        target: Fully qualified upstream symbol path to replace.
        reason: Human-readable explanation for why the patch is required.
        key: Optional stable descriptor key. If omitted, the replacement's
            owner module and qualified name are used.
        owner_module: Optional descriptor owner module. Use this when the
            descriptor registration lives in a different module from the
            replacement implementation. Defaults to the replacement's module.
        condition: Optional predicate evaluated at apply time. The patch is
            skipped when the predicate returns ``False``.
        verify: Optional callback that validates the patch after assignment.
            When omitted, the registry verifies that the target is identical to
            the replacement object.
        priority: Patch application priority in the inclusive range
            `[MIN_PATCH_PRIORITY(HIGH), MAX_PATCH_PRIORITY(LOW)]`. Lower values are
            applied earlier. Descriptors with the same priority are ordered by
            key. Defaults to `DEFAULT_PATCH_PRIORITY`.

    Returns:
        A decorator that registers the replacement object and returns it
        unchanged.
    """

    def _decorator(replacement: Any) -> Any:
        replacement_name = getattr(
            replacement,
            "__qualname__",
            getattr(replacement, "__name__", type(replacement).__qualname__),
        )
        descriptor_owner_module = owner_module or replacement.__module__
        descriptor_key = key or (f"{descriptor_owner_module}.{replacement_name}")
        for descriptor in _REGISTERED_PATCH_DESCRIPTORS:
            if descriptor.key == descriptor_key:
                return replacement

        _REGISTERED_PATCH_DESCRIPTORS.append(
            PatchDescriptor(
                key=descriptor_key,
                owner_module=descriptor_owner_module,
                target=target,
                replacement=replacement,
                reason=reason,
                condition=condition,
                verify=verify,
                priority=priority,
            )
        )
        return replacement

    return _decorator


_applied_registration_keys: set[str] = set()
_applied_patch_keys: set[str] = set()


def get_registered_registration_descriptors() -> list[RegistrationDescriptor]:
    return sorted(_REGISTERED_REGISTRATION_DESCRIPTORS, key=lambda d: d.key)


def apply_registrations() -> None:
    for descriptor in get_registered_registration_descriptors():
        if descriptor.key in _applied_registration_keys:
            continue

        logger.debug(
            "Applying registration %s (owner=%s). Reason: %s",
            descriptor.key,
            descriptor.owner_module,
            descriptor.reason,
        )

        try:
            descriptor.callback()
        except Exception as exc:
            raise RuntimeError(
                "Failed to apply registration "
                f"{descriptor.key} from {descriptor.owner_module}"
            ) from exc

        _applied_registration_keys.add(descriptor.key)


def _sort_patch_descriptors(
    descriptors: Iterable[PatchDescriptor],
) -> list[PatchDescriptor]:
    return sorted(descriptors, key=lambda d: (d.priority, d.key))


def _resolve_patch_target_owner(target: str) -> tuple[object, str]:
    parts = target.split(".")
    for module_end in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:module_end])
        try:
            owner = import_module(module_name)
        except ModuleNotFoundError:
            continue

        attr_parts = parts[module_end:]
        for attr in attr_parts[:-1]:
            owner = getattr(owner, attr)
        return owner, attr_parts[-1]

    raise ValueError(f"Unable to resolve patch target: {target}")


def _apply_target_patch(descriptor: PatchDescriptor) -> None:
    owner, attr = _resolve_patch_target_owner(descriptor.target)
    setattr(owner, attr, descriptor.replacement)


def _verify_target_patch(descriptor: PatchDescriptor) -> None:
    owner, attr = _resolve_patch_target_owner(descriptor.target)
    if getattr(owner, attr) is not descriptor.replacement:
        raise RuntimeError(f"Failed to patch target: {descriptor.target}")


def _validate_patch_registry_layout() -> None:
    descriptor_keys: set[str] = set()
    descriptor_targets: set[str] = set()

    for descriptor in _REGISTERED_PATCH_DESCRIPTORS:
        if descriptor.condition is not None and not descriptor.condition():
            continue

        if not MIN_PATCH_PRIORITY <= descriptor.priority <= MAX_PATCH_PRIORITY:
            raise ValueError(
                "patch descriptor priority must be between "
                f"{MIN_PATCH_PRIORITY} and {MAX_PATCH_PRIORITY}: "
                f"{descriptor.key}={descriptor.priority}"
            )

        if descriptor.key in descriptor_keys:
            raise ValueError(f"Duplicate patch descriptor key: {descriptor.key}")
        descriptor_keys.add(descriptor.key)

        if descriptor.target in descriptor_targets:
            raise ValueError(f"Duplicate patch descriptor target: {descriptor.target}")
        descriptor_targets.add(descriptor.target)


def get_registered_patch_descriptors() -> list[PatchDescriptor]:
    _validate_patch_registry_layout()
    return _sort_patch_descriptors(_REGISTERED_PATCH_DESCRIPTORS)


def apply_registered_patches() -> None:
    descriptors = get_registered_patch_descriptors()

    for descriptor in descriptors:
        if descriptor.key in _applied_patch_keys:
            continue

        if descriptor.condition is not None and not descriptor.condition():
            continue

        _apply_target_patch(descriptor)
        logger.debug(
            "Applying custom patch %s (owner=%s, target=%s).",
            descriptor.key,
            descriptor.owner_module,
            descriptor.target,
        )

        if descriptor.verify is not None:
            descriptor.verify()
        else:
            _verify_target_patch(descriptor)
        _applied_patch_keys.add(descriptor.key)
