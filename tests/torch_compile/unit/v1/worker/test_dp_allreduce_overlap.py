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

"""The per-step DP num_tokens all_reduce, issued async and waited later.

The runner issues this gloo collective at the top of _prepare_inputs and only
consumes it in _determine_batch_padding (VLLM_RBLN_DP_ALL_REDUCE_ASYNC selects
the async form). These tests run a real 4-rank gloo group and check that the
async form (a) decodes to exactly what the blocking form does, and (b) lets a
rank do host work between issue and wait: with one rank arriving late, the
blocking form makes every early rank sit in the collective, the async form does
not. This is a gloo-level property only - it says nothing about whether the
collective overlaps NPU work.
"""

import multiprocessing as mp
import os
import time

import pytest
import torch.distributed as dist

DP_SIZE = 4
# Simulated arrival skew of the late rank, and the host work the early ranks do
# between issuing and waiting. HOST_WORK_S > SKEW_S so a hidden collective costs
# the early ranks nothing.
SKEW_S = 0.2
HOST_WORK_S = 0.3
LATE_RANK = 0


def _rank_main(rank: int, port: int, conn) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=DP_SIZE)

    import vllm.distributed.parallel_state as ps

    from vllm_rbln.forward_context import RBLNDPMetadata

    class _FakeDPGroup:
        cpu_group = dist.group.WORLD

    ps.get_dp_group = lambda: _FakeDPGroup()

    def start(is_prefill=False, async_op=False):
        return RBLNDPMetadata.start_num_tokens_and_reqs_across_dp(
            8 + rank, 4 + rank, DP_SIZE, rank, is_prefill, async_op=async_op
        )

    try:
        # --- correctness: async_op changes when the result lands, not what it is
        blocking = start().wait()
        overlapped = start(async_op=True).wait()
        # is_prefill on any single rank must collapse num_reqs_across_dp to None.
        prefill_reqs = start(is_prefill=(rank == 2), async_op=True).wait()[1]

        # --- skew hiding, blocking form: the early ranks pay the late rank's skew
        dist.barrier()
        if rank == LATE_RANK:
            time.sleep(SKEW_S)
        t_block = time.perf_counter()
        start().wait()
        blocking_blocked_s = time.perf_counter() - t_block

        # --- skew hiding, async form: the early ranks issue, do host work, then
        # wait; the collective completed on the gloo thread meanwhile.
        dist.barrier()
        t0 = time.perf_counter()
        if rank == LATE_RANK:
            time.sleep(SKEW_S)
        async_handle = start(async_op=True)
        time.sleep(max(0.0, HOST_WORK_S - (time.perf_counter() - t0)))
        t_wait = time.perf_counter()
        async_handle.wait()
        async_blocked_s = time.perf_counter() - t_wait

        conn.send(
            {
                "rank": rank,
                "blocking": [t.tolist() if t is not None else None for t in blocking],
                "overlapped": [
                    t.tolist() if t is not None else None for t in overlapped
                ],
                "prefill_reqs": (
                    prefill_reqs.tolist() if prefill_reqs is not None else None
                ),
                "blocking_blocked_s": blocking_blocked_s,
                "async_blocked_s": async_blocked_s,
            }
        )
    except Exception as e:  # surface the failure to the parent
        conn.send({"rank": rank, "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()
        dist.destroy_process_group()


@pytest.mark.timeout(180)
def test_async_dp_allreduce_matches_blocking_and_hides_skew():
    ctx = mp.get_context("spawn")
    port = 29500 + (os.getpid() % 1000)
    procs, conns = [], []
    for rank in range(DP_SIZE):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_rank_main, args=(rank, port, child_conn))
        p.start()
        procs.append(p)
        conns.append(parent_conn)

    results = [c.recv() for c in conns]
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"rank process exited with {p.exitcode}"

    for r in results:
        assert "error" not in r, r["error"]

    expected = [[8 + r for r in range(DP_SIZE)], [4 + r for r in range(DP_SIZE)]]
    for r in results:
        assert r["blocking"] == expected
        assert r["overlapped"] == expected
        assert r["prefill_reqs"] is None

    for r in results:
        if r["rank"] == LATE_RANK:
            continue
        assert r["blocking_blocked_s"] > SKEW_S / 2, (
            f"rank {r['rank']}: blocking form only blocked "
            f"{r['blocking_blocked_s']:.3f}s - the skew was not reproduced"
        )
        assert r["async_blocked_s"] < SKEW_S / 2, (
            f"rank {r['rank']}: async form blocked {r['async_blocked_s']:.3f}s - "
            "the collective was not hidden behind host work"
        )
