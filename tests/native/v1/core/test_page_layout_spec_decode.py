# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The spec-decode backfill guard measures against the kernel block.

A decode query is one contiguous KV window that the runner pads backwards to
`num_spec_tokens + 1`. Page layout makes `block_size` the *page*, which is not
that window: pages inside a kernel block are contiguous, so only a
kernel-block boundary is a real discontinuity. Measuring against the page is
still safe -- the page divides the kernel block, so it is strictly stricter --
but it disables spec decode near every page boundary instead of every kernel
block, which at 512 vs 8192 is 16x as often.
"""

import pytest

from vllm_rbln.v1.core.rbln_scheduler import RBLNScheduler

PAGE = 512
KERNEL_BLOCK = 8192
NUM_SPEC = 3


def guard(kernel_block_size, num_computed_tokens, num_new_tokens=1):
    sched = RBLNScheduler.__new__(RBLNScheduler)
    sched.num_spec_tokens = NUM_SPEC
    sched.kernel_block_size = kernel_block_size
    return RBLNScheduler._spec_backfill_is_unsafe(
        sched, num_computed_tokens, num_new_tokens
    )


class TestBoundary:
    @pytest.mark.parametrize("offset", [0, 1, 2])
    def test_too_few_tokens_behind_the_query_is_unsafe(self, offset):
        # Fewer than num_spec tokens sit in this block, so backfilling would
        # reach into the previous one.
        assert guard(KERNEL_BLOCK, KERNEL_BLOCK + offset)

    def test_exactly_enough_is_safe(self):
        assert not guard(KERNEL_BLOCK, KERNEL_BLOCK + NUM_SPEC)

    def test_mid_block_is_safe(self):
        assert not guard(KERNEL_BLOCK, KERNEL_BLOCK + 4096)

    def test_a_longer_query_needs_less_backfill(self):
        # num_new_tokens == num_spec + 1 needs no backfill at all.
        assert not guard(KERNEL_BLOCK, KERNEL_BLOCK, num_new_tokens=NUM_SPEC + 1)


class TestKernelBlockVsPage:
    def test_page_boundaries_inside_a_kernel_block_are_not_boundaries(self):
        # The regression this guards: measured against the page, every one of
        # the 16 page boundaries in a kernel block would drop to no-spec.
        for page_index in range(1, KERNEL_BLOCK // PAGE):
            position = page_index * PAGE
            assert guard(PAGE, position), "page-sized guard fires here"
            assert not guard(KERNEL_BLOCK, position), (
                "a page boundary is contiguous inside a kernel block"
            )

    def test_kernel_block_boundary_still_fires(self):
        assert guard(KERNEL_BLOCK, 2 * KERNEL_BLOCK)

    def test_the_page_guard_is_stricter_never_looser(self):
        # Safety argument for why the old behaviour was merely wasteful: the
        # page divides the kernel block, so page-unsafe covers block-unsafe.
        for position in range(0, 4 * PAGE):
            if guard(KERNEL_BLOCK, position):
                assert guard(PAGE, position)
