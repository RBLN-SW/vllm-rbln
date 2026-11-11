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


from types import SimpleNamespace

import pytest

from vllm_rbln.utils.optimum.configuration.readers import (
    _num_devices_of,
    has_packed_kv_cache,
    read_num_devices,
)


class TestNumDevicesOf:
    def test_top_level_num_devices(self):
        assert _num_devices_of({"num_devices": 4}) == 4

    def test_falls_back_to_compile_cfgs(self):
        cfg = {"_compile_cfgs": [{"num_devices": 8}]}
        assert _num_devices_of(cfg) == 8

    def test_top_level_takes_precedence_over_compile_cfgs(self):
        cfg = {"num_devices": 2, "_compile_cfgs": [{"num_devices": 8}]}
        assert _num_devices_of(cfg) == 2

    def test_first_compile_cfg_with_num_devices_wins(self):
        cfg = {"_compile_cfgs": [{}, {"num_devices": 8}, {"num_devices": 16}]}
        assert _num_devices_of(cfg) == 8

    def test_returns_none_when_absent(self):
        assert _num_devices_of({}) is None

    def test_returns_none_when_compile_cfgs_lacks_num_devices(self):
        assert _num_devices_of({"_compile_cfgs": [{}, {}]}) is None

    def test_non_int_raises(self):
        with pytest.raises(AssertionError, match="num_devices must be an int"):
            _num_devices_of({"num_devices": "4"})

    def test_non_positive_raises(self):
        with pytest.raises(AssertionError, match="positive integer"):
            _num_devices_of({"num_devices": 0})

    def test_compile_cfgs_non_int_raises(self):
        with pytest.raises(AssertionError, match="num_devices must be an int"):
            _num_devices_of({"_compile_cfgs": [{"num_devices": 1.5}]})

    def test_compile_cfgs_non_positive_raises(self):
        with pytest.raises(AssertionError, match="positive integer"):
            _num_devices_of({"_compile_cfgs": [{"num_devices": -1}]})


class TestReadNumDevices:
    def test_top_level(self):
        assert read_num_devices({"num_devices": 4}) == 4

    def test_returns_none_when_absent(self):
        assert read_num_devices({}) is None

    def test_reads_from_language_model_submodule(self):
        cfg = {"language_model": {"num_devices": 8}}
        assert read_num_devices(cfg) == 8

    def test_reads_from_text_model_submodule(self):
        cfg = {"text_model": {"num_devices": 8}}
        assert read_num_devices(cfg) == 8

    def test_language_model_takes_precedence_over_top_level(self):
        cfg = {"num_devices": 1, "language_model": {"num_devices": 8}}
        assert read_num_devices(cfg) == 8

    def test_language_model_preferred_over_text_model(self):
        cfg = {
            "language_model": {"num_devices": 8},
            "text_model": {"num_devices": 16},
        }
        assert read_num_devices(cfg) == 8

    def test_reads_from_submodule_compile_cfgs(self):
        cfg = {"language_model": {"_compile_cfgs": [{"num_devices": 8}]}}
        assert read_num_devices(cfg) == 8

    def test_falls_back_to_top_level_when_submodule_lacks_num_devices(self):
        cfg = {"num_devices": 4, "language_model": {"batch_size": 2}}
        assert read_num_devices(cfg) == 4


class TestHasPackedKvCache:
    def test_packed_cache(self):
        cfg = {
            "cache_metas": [
                {"name": "past_key_values_0", "shape": [4, 2, 2, 1, 256, 64]}
            ]
        }
        assert has_packed_kv_cache(cfg)

    def test_separate_k_and_v(self):
        cfg = {
            "cache_metas": [
                {"name": "past_key_values_0", "shape": [4, 2, 256, 64]},
                {"name": "past_key_values_1", "shape": [4, 2, 256, 64]},
            ]
        }
        assert not has_packed_kv_cache(cfg)

    def test_without_cache_metas(self):
        assert not has_packed_kv_cache({"kvcache_num_blocks": 4})

    def test_reads_the_language_model(self):
        cfg = {
            "language_model": {"cache_metas": [{"shape": [4, 2, 2, 1, 256, 64]}]},
            "vision_tower": {},
        }
        assert has_packed_kv_cache(cfg)

    def test_reads_config_objects(self):
        # Two blocks of a packed cache, e.g. eager attention with batch 2.
        meta = SimpleNamespace(shape=[2, 2, 2, 1, 256, 64])
        assert has_packed_kv_cache(SimpleNamespace(cache_metas=[meta]))
