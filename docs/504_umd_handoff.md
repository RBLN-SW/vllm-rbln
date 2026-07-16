# 504 page-fault under async-overlap — UMD/RCCL/KMD hand-off

## Symptom
Under `RBLN_DYNAMO_ASYNC=defer` (async-overlap decode), gpt-oss-120b **DP4 + Expert-Parallel**,
the decode intermittently aborts:

```
code=504 SYS_TASK_ABORTED (code: 3 Wait job task aborted)
KMD/FW: hw_status 0x10007 = ERR_PAGE_FAULT   (fw-expert coredump decode)
```

## Root cause (confirmed at the runtime layer)
The **main thread commits a device page-table change while an `rcclAllToAllX` is in flight on
the async worker.** Concretely:

- `DeviceCachingAllocator::AllocBlock` (a decode-time device allocation, e.g. a 2 MiB
  `kSmallBuffer`) calls `Context::Submit()` → `rblnSubmitContext(ctx_)`.
- Meanwhile the async worker has dispatched `rcclAllToAllX` (the MoE all-to-all) whose PTW is
  walking the ASID's page table.
- `rblnSubmitContext` commits/mutates the page table (it sets `RBLN_CTX_FLAG_SUBMIT_CTX_BARRIER`)
  → the in-flight collective's PTW hits a disturbed/invalid PTE → `ERR_PAGE_FAULT` → the
  collective's job aborts (all ranks' all-to-all then aborts at the barrier).

We verified the seq plumbing so `Context::Submit()`'s `default_stream_->Drain()` now waits on the
collective's **real** `seq_out` (via `rccl_extra_args`): the abort surfaces as
`[stream] Drain failed, seq=84659, rc=3` / `Failed to WaitForCompletion, seq=84659` — i.e. the
runtime *can* observe the collective's seq, but the commit still races and the collective still
faults.

## Why we cannot fix it purely in rebel_compiler
Every host-side fence we tried fails:
1. **Wait for the collective before committing (host block on the seq).** Deadlocks — the main
   thread blocking on the collective stalls the cross-rank rendezvous (and the DP `all_reduce`);
   watchdog aborts.
2. **Record the collective seq and Drain on it before the commit.** There is an unclosable
   window: `seq_out` is only returned *after* `rcclAllToAllX` has already dispatched (device PTW
   started), so a `Context::Submit()` landing in the dispatch→record gap fences the *previous*
   (retired) seq and commits into the just-launched collective.
3. **A device-mutation lock spanning the collective dispatch + the commit.** `rcclAllToAllX`
   (non-Nb, with the readiness barrier) blocks inside the call waiting for peers, so holding a
   lock across it deadlocks across ranks.

(A pre-warm that pre-allocates the pool so no `AllocBlock`→Submit happens during decode does
remove the fault, but it only *avoids triggering* the race by guessing a pool size — not a real
fix.)

## Questions / asks for UMD / RCCL / KMD

1. **Does `RBLN_CTX_FLAG_SUBMIT_CTX_BARRIER` / a successful `rblnWaitJob(ctx, seq_out)` cover the
   FULL completion (incl. RDMA/multi-rank phase) of an `rcclAllToAllX`?** We can detect its
   *abort* via `seq_out` (rc=3), but is a *successful* wait a guarantee that its PTW is finished,
   so a following `rblnSubmitContext` is safe?

2. **`rblnSubmitContext` semantics vs in-flight PTW.** The uapi header says it "can only be
   called once per context; the context becomes immutable for resource allocations after this
   call" — yet `AllocBlock` calls it repeatedly for dynamic KV-cache growth, long after jobs are
   live. (a) Is repeated `rblnSubmitContext` while other engines (rccl) are executing a supported
   contract? (b) Does it re-commit the whole ASID page table (disturbing PTEs a live collective
   is walking), or only add the new block's PTEs? (c) Is there an **incremental / atomic commit**
   mode (`RBLN_MEM_UPDATE_ON_SUBMIT`?) that installs new mappings without perturbing in-flight
   ones?

3. **Is there a device/UMD primitive to order a page-table commit *after* in-flight rccl work on
   the device, without a host barrier** (which deadlocks the cross-rank readiness barrier)? e.g.
   an rccl "quiesce", or making `rblnSubmitContext` take a `seq_in` dependency so the FW orders
   it after the collective.

4. **Confirm the deployment isn't using `RBLN_CCL_ALLTOALLX_NB`** (barrier-free variant) — with
   it the collective's readiness barrier is disabled by design, which would be an independent
   hazard.

## Repro
```
RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1 <full DP4+EP decode, gpt-oss-120b>
# ~intermittent (several / handful of runs). No RBLN_ASYNC_POOL_PREWARM.
```
Evidence files: fw-expert coredump decode (hw_status 0x10007), runtime log
`[stream] Drain failed, seq=84659, rc=3`.
