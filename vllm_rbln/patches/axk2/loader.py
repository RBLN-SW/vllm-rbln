# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_HERE = Path(__file__).resolve().parent


def load_frozen_module(canonical_name: str, filename: str) -> ModuleType:
    existing = sys.modules.get(canonical_name)
    if existing is not None:
        return existing

    path = _HERE / filename

    spec = importlib.util.spec_from_file_location(canonical_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)

    sys.modules[canonical_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(canonical_name, None)
        raise

    parent_name, _, attr = canonical_name.rpartition(".")
    setattr(importlib.import_module(parent_name), attr, module)

    return module
