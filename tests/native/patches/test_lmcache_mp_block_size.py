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
"""LMCache mp adapters must size paged-KV work in kernel blocks under page layout.

`blocks_in_chunk = chunk // vllm_block_size` decides how many block ids a
retrieve expects. Feeding it the page while the ids (and the KV tensors) are
kernel blocks is an 8x miscount:

    ValueError: block_ids length (6) must be at least len(chunks) (6)
                * blocks_per_chunk (8)
"""

from types import SimpleNamespace

import pytest

import vllm_rbln.envs as envs
from vllm_rbln.patches import lmcache_mp_block_size as mod


def _config(*, page=512, kernel=4096, additional=True):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=page),
        model_config=object(),
        additional_config={"attn_block_size": kernel} if additional else None,
    )


class TestKernelBlockSize:
    def test_none_without_the_env(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", False)
        assert mod._kernel_block_size(_config()) is None

    def test_none_without_attn_block_size(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        assert mod._kernel_block_size(_config(additional=False)) is None

    def test_none_when_kernel_equals_page(self, monkeypatch):
        """Degenerate geometry -- page layout is a no-op, leave the value alone."""
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        assert mod._kernel_block_size(_config(page=512, kernel=512)) is None

    def test_none_when_not_a_multiple(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        assert mod._kernel_block_size(_config(page=512, kernel=700)) is None

    def test_kernel_block_when_it_applies(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        assert mod._kernel_block_size(_config()) == 4096


class TestRedirect:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.seen = {}

        def fake_original(_self, *args, **kwargs):
            self.seen.update(kwargs)

        monkeypatch.setitem(mod._ORIGINAL_INITS, "LMCacheMPWorkerAdapter", fake_original)

    def test_page_becomes_kernel_block(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        cfg = _config()
        mod._redirect(
            "LMCacheMPWorkerAdapter",
            object(),
            (),
            {"vllm_config": cfg, "vllm_block_size": 512},
        )
        assert self.seen["vllm_block_size"] == 4096

    def test_untouched_without_page_layout(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", False)
        mod._redirect(
            "LMCacheMPWorkerAdapter",
            object(),
            (),
            {"vllm_config": _config(), "vllm_block_size": 512},
        )
        assert self.seen["vllm_block_size"] == 512

    def test_finds_the_config_passed_positionally(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        cfg = _config()
        mod._redirect(
            "LMCacheMPWorkerAdapter", object(), (cfg,), {"vllm_block_size": 512}
        )
        assert self.seen["vllm_block_size"] == 4096

    def test_leaves_other_kwargs_alone(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        mod._redirect(
            "LMCacheMPWorkerAdapter",
            object(),
            (),
            {"vllm_config": _config(), "vllm_block_size": 512, "model_name": "m"},
        )
        assert self.seen["model_name"] == "m"


class TestLateApplication:
    """The patch must survive being unable to import lmcache at platform init.

    `register_ops()` imports `vllm_rbln.patches` while vLLM is still coming up;
    importing lmcache there re-enters a half-built `vllm` and fails. The registry
    evaluates each condition once, so a False there means the patch never
    applies -- measured in-cluster: descriptors registered, `_applied_patch_keys`
    empty, and the adapter still sizing in pages.
    """

    def test_condition_is_false_while_lmcache_cannot_be_imported(self, monkeypatch):
        monkeypatch.setattr(mod, "_ORIGINAL_INITS", {})
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def boom(name, *a, **k):
            if name.startswith("lmcache"):
                raise ImportError("half-built vllm")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", boom)
        assert mod._lmcache_mp_available() is False

    def test_ensure_applied_is_a_noop_without_lmcache(self, monkeypatch):
        monkeypatch.setattr(mod, "_ORIGINAL_INITS", {})
        monkeypatch.setattr(mod, "_lmcache_mp_available", lambda: False)
        mod.ensure_applied()  # must not raise, must not import the registry

    def test_connector_factory_applies_before_building(self):
        """The hook must sit on the connector factory, not on our constructors.

        The mp adapters are built inside `LMCacheMPConnector.__init__`, and the
        worker-side one is built in a different process from the scheduler-side
        one. Hooking `RBLNScheduler.__init__` / `RBLNWorker.__init__` looked
        equivalent and was not -- it still missed the worker adapter in-cluster
        (blocks_per_chunk stayed 8). The factory is the one point that is after
        vLLM is up and before any connector exists, whichever process it is.
        """
        import inspect

        src = inspect.getsource(mod.patched_create_connector)
        assert "ensure_applied()" in src
        assert src.index("ensure_applied()") < src.index("_ORIGINAL_CREATE_CONNECTOR("), (
            "the patch must be applied before the connector is constructed"
        )


class TestConfigIsNotPassedToAdapters:
    """`LMCacheMPConnector` builds its adapters **without** the VllmConfig.

    Its call site passes only server_url / context / model_name /
    vllm_block_size / parallel_strategy / extra_config. An earlier version of
    this patch looked for `vllm_config` among the adapter's own arguments, found
    nothing, and rewrote nothing -- applying cleanly while doing exactly zero
    (measured in-cluster: patch installed, blocks_per_chunk still 8). The config
    has to come from the factory hook, which runs just before in the same
    process.
    """

    def test_rewrites_from_the_stashed_config(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_RBLN_PAGE_LAYOUT", True)
        monkeypatch.setattr(mod, "_VLLM_CONFIG", _config())
        seen = {}
        monkeypatch.setitem(
            mod._ORIGINAL_INITS,
            "LMCacheMPWorkerAdapter",
            lambda _s, *a, **k: seen.update(k),
        )
        # exactly the kwargs LMCacheMPConnector uses -- no vllm_config
        mod._redirect(
            "LMCacheMPWorkerAdapter",
            object(),
            (),
            {
                "server_url": "tcp://x:5555",
                "model_name": "m",
                "vllm_block_size": 512,
                "extra_config": {},
            },
        )
        assert seen["vllm_block_size"] == 4096

    def test_factory_hook_stashes_the_config(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(mod, "_VLLM_CONFIG", None)
        monkeypatch.setattr(mod, "_ORIGINAL_CREATE_CONNECTOR", lambda *a, **k: "conn")
        monkeypatch.setattr(mod, "ensure_applied", lambda: captured.setdefault("ok", 1))
        cfg = _config()
        assert mod.patched_create_connector(cfg, "role", "kvcfg") == "conn"
        assert mod._VLLM_CONFIG is cfg
        assert captured.get("ok"), "the patch must still be applied"
