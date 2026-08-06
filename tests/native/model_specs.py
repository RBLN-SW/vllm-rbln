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

"""Shared model spec for the whole-model NPU lanes (compile smoke + correctness)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# Harness-visible env vars are promoted to fields because the harness branches
# on them; `envs` is everything the harness does NOT interpret.
_RSD_ENV = "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"
_ID_KEYS = (
    ("tp", "tensor_parallel_size", 1),
    ("pp", "pipeline_parallel_size", 1),
    ("dp", "data_parallel_size", 1),
)


@dataclass(frozen=True)
class CompileModelSpec:
    # engine_kwargs pass straight to the runner; envs are set before engine init.
    model: str
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, str] = field(default_factory=dict)
    rsd: int = 1
    id: str | None = None

    def __post_init__(self):
        if _RSD_ENV in self.envs:
            raise ValueError(f"{_RSD_ENV} is harness-visible; set rsd= instead")
        if self.rsd < 1:
            raise ValueError(f"rsd must be >= 1, got {self.rsd}")

    @property
    def test_id(self) -> str:
        if self.id:
            return self.id
        parts = [self.model.split("/")[-1]]
        parts += [
            f"{short}{self.engine_kwargs[key]}"
            for short, key, default in _ID_KEYS
            if self.engine_kwargs.get(key, default) != default
        ]
        if self.rsd != 1:
            parts.append(f"rsd{self.rsd}")
        if (
            spec_config := self.engine_kwargs.get("speculative_config", None)
        ) is not None:
            parts.append(spec_config["method"])
        return "-".join(parts)


def spec_params(specs: list[CompileModelSpec]) -> list[Any]:
    params = []
    seen: set[str] = set()
    for spec in specs:
        test_id = spec.test_id
        if test_id in seen:
            raise ValueError(f"Duplicate spec id {test_id!r}; set id= on one of them")
        seen.add(test_id)
        params.append(pytest.param(spec, id=test_id))
    return params


def apply_spec_envs(spec: CompileModelSpec, monkeypatch) -> None:
    if spec.rsd > 1:
        monkeypatch.setenv(_RSD_ENV, str(spec.rsd))
    for key, value in spec.envs.items():
        monkeypatch.setenv(key, value)
