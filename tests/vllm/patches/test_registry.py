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

# The registry resolves upstream targets, orders patches by (priority, key),
# applies and verifies them. Its state lives in module globals the conftest has
# already populated, so tests that mutate them swap in isolated ones.

import sys
import types
from typing import Any

import pytest

import vllm_rbln.patches.registry as reg
from vllm_rbln.patches.registry import (
    MAX_PATCH_PRIORITY,
    PatchDescriptor,
    RegistrationDescriptor,
    _apply_target_patch,
    _resolve_patch_target_owner,
    _sort_patch_descriptors,
    _validate_patch_registry_layout,
    _verify_target_patch,
    add_registration,
    apply_registered_patches,
    apply_registrations,
    register_patch,
)

_TARGET = "_registry_test_target.symbol"


@pytest.fixture
def isolated(monkeypatch):
    # Swap the shared registry globals for empty ones so a test never sees or
    # mutates the real descriptors the conftest registered.
    monkeypatch.setattr(reg, "_REGISTERED_REGISTRATION_DESCRIPTORS", [])
    monkeypatch.setattr(reg, "_REGISTERED_PATCH_DESCRIPTORS", [])
    monkeypatch.setattr(reg, "_applied_registration_keys", set())
    monkeypatch.setattr(reg, "_applied_patch_keys", set())


@pytest.fixture
def fake_target(monkeypatch):
    mod: Any = types.ModuleType("_registry_test_target")
    mod.symbol = "ORIGINAL"
    monkeypatch.setitem(sys.modules, "_registry_test_target", mod)
    return mod


def _patch(
    *,
    key,
    target=_TARGET,
    replacement="R",
    priority=50,
    condition=None,
    verify=None,
    apply_immediately=False,
):
    return PatchDescriptor(
        key=key,
        owner_module="t",
        target=target,
        replacement=replacement,
        reason="r",
        condition=condition,
        verify=verify,
        priority=priority,
        apply_immediately=apply_immediately,
    )


class TestResolvePatchTargetOwner:
    def test_module_level_symbol(self):
        import json

        owner, attr = _resolve_patch_target_owner("json.dumps")
        assert owner is json and attr == "dumps"

    def test_nested_submodule(self):
        import os.path

        owner, attr = _resolve_patch_target_owner("os.path.join")
        assert owner is os.path and attr == "join"

    def test_attribute_on_a_class(self):
        import json

        owner, attr = _resolve_patch_target_owner("json.JSONEncoder.encode")
        assert owner is json.JSONEncoder and attr == "encode"

    def test_unresolvable_target_raises(self):
        with pytest.raises(ValueError, match="Unable to resolve"):
            _resolve_patch_target_owner("no_such_module_xyz.foo")


class TestSortPatchDescriptors:
    def test_orders_by_priority_then_key(self):
        descs = [
            _patch(key="b", priority=50),
            _patch(key="a", priority=50),
            _patch(key="c", priority=10),
        ]
        assert [d.key for d in _sort_patch_descriptors(descs)] == ["c", "a", "b"]


class TestApplyAndVerifyTargetPatch:
    def test_apply_sets_the_replacement(self, fake_target):
        sentinel = object()
        _apply_target_patch(_patch(key="k", replacement=sentinel))
        assert fake_target.symbol is sentinel

    def test_verify_passes_after_apply(self, fake_target):
        descriptor = _patch(key="k", replacement=object())
        _apply_target_patch(descriptor)
        _verify_target_patch(descriptor)  # must not raise

    def test_verify_fails_when_target_differs(self, fake_target):
        with pytest.raises(RuntimeError, match="Failed to patch target"):
            _verify_target_patch(_patch(key="k", replacement=object()))


class TestValidatePatchRegistryLayout:
    def test_accepts_valid_layout(self, monkeypatch):
        monkeypatch.setattr(
            reg,
            "_REGISTERED_PATCH_DESCRIPTORS",
            [
                _patch(key="a", target="m.x"),
                _patch(key="b", target="m.y"),
            ],
        )
        _validate_patch_registry_layout()  # no raise

    def test_rejects_priority_out_of_range(self, monkeypatch):
        monkeypatch.setattr(
            reg,
            "_REGISTERED_PATCH_DESCRIPTORS",
            [
                _patch(key="a", priority=MAX_PATCH_PRIORITY + 1),
            ],
        )
        with pytest.raises(ValueError, match="priority must be between"):
            _validate_patch_registry_layout()

    def test_rejects_duplicate_key(self, monkeypatch):
        monkeypatch.setattr(
            reg,
            "_REGISTERED_PATCH_DESCRIPTORS",
            [
                _patch(key="dup", target="m.x"),
                _patch(key="dup", target="m.y"),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate patch descriptor key"):
            _validate_patch_registry_layout()

    def test_rejects_duplicate_target(self, monkeypatch):
        monkeypatch.setattr(
            reg,
            "_REGISTERED_PATCH_DESCRIPTORS",
            [
                _patch(key="a", target="m.same"),
                _patch(key="b", target="m.same"),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate patch descriptor target"):
            _validate_patch_registry_layout()

    def test_condition_false_descriptors_are_skipped(self, monkeypatch):
        # A would-be duplicate target is ignored while its condition is False.
        monkeypatch.setattr(
            reg,
            "_REGISTERED_PATCH_DESCRIPTORS",
            [
                _patch(key="a", target="m.same"),
                _patch(key="b", target="m.same", condition=lambda: False),
            ],
        )
        _validate_patch_registry_layout()  # no raise


class TestAddRegistration:
    def test_registers_callback_and_returns_it(self, isolated):
        @add_registration(reason="x")
        def cb():
            pass

        descriptors = reg._REGISTERED_REGISTRATION_DESCRIPTORS
        assert len(descriptors) == 1
        assert descriptors[0].callback is cb
        assert descriptors[0].key.endswith(".cb")

    def test_same_key_is_registered_only_once(self, isolated):
        def register_one():
            @add_registration(reason="x")
            def cb():
                pass

        register_one()
        register_one()  # identical module.qualname -> same key
        assert len(reg._REGISTERED_REGISTRATION_DESCRIPTORS) == 1


class TestRegisterPatch:
    def test_apply_immediately_with_explicit_priority_raises(self):
        with pytest.raises(ValueError, match="cannot influence"):
            register_patch(
                target="m.x", reason="r", apply_immediately=True, priority=10
            )

    def test_registers_descriptor_with_given_fields(self, isolated):
        @register_patch(target="m.x", reason="r", key="mykey", priority=20)
        def replacement():
            pass

        [descriptor] = reg._REGISTERED_PATCH_DESCRIPTORS
        assert descriptor.key == "mykey"
        assert descriptor.target == "m.x"
        assert descriptor.priority == 20
        assert descriptor.replacement is replacement

    def test_duplicate_key_registered_once(self, isolated):
        register_patch(target="m.x", reason="r", key="k")(lambda: None)
        register_patch(target="m.y", reason="r", key="k")(lambda: None)
        assert len(reg._REGISTERED_PATCH_DESCRIPTORS) == 1

    def test_default_key_is_replacement_module_and_qualname(self, isolated):
        @register_patch(target="m.x", reason="r")
        def replacement():
            pass

        [descriptor] = reg._REGISTERED_PATCH_DESCRIPTORS
        assert descriptor.key == f"{replacement.__module__}.{replacement.__qualname__}"

    def test_owner_module_override_prefixes_the_key(self, isolated):
        @register_patch(target="m.x", reason="r", owner_module="custom.mod")
        def replacement():
            pass

        [descriptor] = reg._REGISTERED_PATCH_DESCRIPTORS
        assert descriptor.key == f"custom.mod.{replacement.__qualname__}"

    def test_apply_immediately_patches_at_registration(self, isolated, fake_target):
        @register_patch(target=_TARGET, reason="r", key="k", apply_immediately=True)
        def replacement():
            pass

        assert fake_target.symbol is replacement

    def test_apply_immediately_respects_false_condition(self, isolated, fake_target):
        # Registered, but a False condition keeps it from applying at import time.
        @register_patch(
            target=_TARGET,
            reason="r",
            key="k",
            apply_immediately=True,
            condition=lambda: False,
        )
        def replacement():
            pass

        assert len(reg._REGISTERED_PATCH_DESCRIPTORS) == 1
        assert fake_target.symbol == "ORIGINAL"


class TestApplyRegistrations:
    def test_runs_callbacks_in_key_order_once(self, isolated):
        calls: list = []
        reg._REGISTERED_REGISTRATION_DESCRIPTORS.extend(
            [
                RegistrationDescriptor("b", "m", lambda: calls.append("b"), "r"),
                RegistrationDescriptor("a", "m", lambda: calls.append("a"), "r"),
            ]
        )
        apply_registrations()
        apply_registrations()  # idempotent
        assert calls == ["a", "b"]

    def test_callback_failure_is_wrapped(self, isolated):
        def boom():
            raise KeyError("nope")

        reg._REGISTERED_REGISTRATION_DESCRIPTORS.append(
            RegistrationDescriptor("x", "m", boom, "r")
        )
        with pytest.raises(RuntimeError, match="Failed to apply registration"):
            apply_registrations()


class TestApplyRegisteredPatches:
    def test_applies_the_patch(self, isolated, fake_target):
        sentinel = object()
        reg._REGISTERED_PATCH_DESCRIPTORS.append(_patch(key="k", replacement=sentinel))
        apply_registered_patches()
        assert fake_target.symbol is sentinel

    def test_condition_false_is_skipped(self, isolated, fake_target):
        reg._REGISTERED_PATCH_DESCRIPTORS.append(
            _patch(key="k", replacement=object(), condition=lambda: False)
        )
        apply_registered_patches()
        assert fake_target.symbol == "ORIGINAL"

    def test_custom_verify_is_invoked(self, isolated, fake_target):
        verified: list = []
        reg._REGISTERED_PATCH_DESCRIPTORS.append(
            _patch(key="k", replacement=object(), verify=lambda: verified.append(1))
        )
        apply_registered_patches()
        assert verified == [1]

    def test_already_applied_patch_is_not_reapplied(self, isolated, fake_target):
        reg._REGISTERED_PATCH_DESCRIPTORS.append(_patch(key="k", replacement=object()))
        apply_registered_patches()
        fake_target.symbol = "CHANGED"
        apply_registered_patches()  # key already applied -> skipped
        assert fake_target.symbol == "CHANGED"

    def test_applies_multiple_patches_in_priority_order(self, isolated, fake_target):
        # Lower priority applies first; the verify hooks record the actual order.
        order: list = []
        reg._REGISTERED_PATCH_DESCRIPTORS.extend(
            [
                _patch(
                    key="a",
                    target="_registry_test_target.high",
                    priority=50,
                    verify=lambda: order.append("a"),
                ),
                _patch(
                    key="b",
                    target="_registry_test_target.low",
                    priority=10,
                    verify=lambda: order.append("b"),
                ),
            ]
        )
        apply_registered_patches()
        assert order == ["b", "a"]
