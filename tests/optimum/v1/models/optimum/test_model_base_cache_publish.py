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
"""On a cache miss the compiled model is exported into a staging directory
beside the cache entry and published by one rename.

optimum's default, a `TemporaryDirectory` under `$TMPDIR` copied into the
cache by `save_pretrained`, holds the artifact twice and leaks the `$TMPDIR`
copy when the server is stopped by a signal. Every process that misses the
cache compiles, so the staging directory is per process and a publish that
finds a sibling's artifact keeps it.
"""

import tempfile
from pathlib import Path

import pytest
from vllm.config import VllmConfig

from vllm_rbln.model_executor.models.optimum.model_base import RBLNOptimumModelBase
from vllm_rbln.utils.optimum.paths import is_compiled_dir

from .conftest import write_compiled_dir


@pytest.fixture
def cache(vllm_config: VllmConfig) -> Path:
    """The cache entry the config build staged for this miss."""
    return Path(vllm_config.additional_config.cached_model_path)


def test_export_lands_in_the_cache_without_a_temp_copy(vllm_config, fake_export, cache):
    RBLNOptimumModelBase(vllm_config)

    (kwargs,) = fake_export.calls
    staging = Path(kwargs["model_save_dir"])
    # optimum was told to write beside the cache entry, never under $TMPDIR.
    assert staging.parent == cache.parent
    assert staging.name.startswith(f"{cache.name}.export-")
    # The staging directory became the cache entry and nothing else remains.
    assert is_compiled_dir(str(cache))
    assert list(cache.parent.iterdir()) == [cache]
    assert vllm_config.model_config.model == str(cache)


def test_a_sibling_export_in_progress_is_left_alone(vllm_config, fake_export, cache):
    def sibling_starts_exporting() -> None:
        tempfile.mkdtemp(prefix=f"{cache.name}.export-", dir=cache.parent)

    fake_export.after_export = staticmethod(sibling_starts_exporting)
    RBLNOptimumModelBase(vllm_config)

    assert is_compiled_dir(str(cache))
    others = [p for p in cache.parent.iterdir() if p != cache]
    assert len(others) == 1 and others[0].name.startswith(f"{cache.name}.export-")


def test_a_sibling_that_published_first_keeps_its_artifact(
    vllm_config, fake_export, cache
):
    fake_export.after_export = staticmethod(
        lambda: write_compiled_dir(cache, marker="sibling.rbln")
    )
    RBLNOptimumModelBase(vllm_config)

    # The sibling's directory is untouched and our staging directory is gone.
    assert (cache / "sibling.rbln").exists()
    assert not (cache / "prefill.rbln").exists()
    assert list(cache.parent.iterdir()) == [cache]


def test_a_partial_cache_entry_from_a_dead_export_is_replaced(
    vllm_config, fake_export, cache
):
    # A previous run died mid-export: the cache path exists but is not a
    # compiled directory (no rbln_config.json).
    cache.mkdir(parents=True)
    (cache / "half.rbln").touch()

    RBLNOptimumModelBase(vllm_config)

    assert is_compiled_dir(str(cache))
    assert not (cache / "half.rbln").exists()
    assert list(cache.parent.iterdir()) == [cache]


def test_an_export_that_wrote_no_compiled_model_is_not_published(
    vllm_config, fake_export, cache
):
    fake_export.writes_compiled_dir = False

    with pytest.raises(RuntimeError, match="left no compiled model"):
        RBLNOptimumModelBase(vllm_config)

    assert not cache.parent.exists() or list(cache.parent.iterdir()) == []


def test_a_cache_miss_without_a_cache_path_is_an_error(vllm_config, fake_export):
    vllm_config.additional_config.cached_model_path = None

    with pytest.raises(RuntimeError, match="cache miss without a cache path"):
        RBLNOptimumModelBase(vllm_config)
