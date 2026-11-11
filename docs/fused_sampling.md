## Fused Sampling Overview

`use_fused_sampling` moves the sampling step's logits processing into the
compiled sampling graph. It covers repetition, frequency and presence penalties
and `logit_bias`.

### Fused penalties

Repetition, frequency and presence penalties and `logit_bias` only change the
logits of tokens a request has already seen or explicitly biased. Upstream vLLM
still applies them as a chain of eager ops over the whole `[batch, vocab]`
logits on every decode step. On RBLN that chain runs outside the compiled
sampler and dominates the step time once a request sets any of these
parameters.

With fused penalties enabled, the host tracks the touched tokens per request
and the compiled sampling graph rewrites only the vocab chunks that hold them,
right before temperature, softmax and top-k / top-p (or argmax).

Key components:

- `vllm_rbln.v1.sample.ops.fused_penalty` holds the per-request host state
  (`FusedPenaltyState`), which turns a step's sampling metadata into a
  `FusedPenaltyPlan`, and the device-side math (`apply_fused_penalty`).
- `RBLNSampler` asks for a plan each step, hands upstream
  `apply_logits_processors` the metadata without penalties and logit bias, and
  passes the plan to `sample()`. Its `warm_up()` compiles the fused graphs with
  the rest of the sampler.

## Enabling

| Flag | Default | Description |
| --- | --- | --- |
| `--rbln-use-fused-sampling` | `False` | Run penalties and logit bias inside the sampling graph. Requires `use_custom_sampler`. |

`LLM(...)` takes it as `additional_config={"use_fused_sampling": True}`.

A step keeps the existing path when it runs speculative decoding or a thinking
budget, when the sampler returns processed logprobs, or when the vocab size is
not a multiple of 128. A step with min_p applies the penalties in a graph of
their own first, since min_p reads the penalized logits, then continues on the
existing path.

## How It Works

### What each token gets

For a token `t` of a request, upstream applies, in this order:

```
l[t] += bias[t]
l[t]  = l[t] / rep  if l[t] > 0  else  l[t] * rep      (t in prompt or output)
l[t] -= freq * count[t] + pres                          (t in output)
```

Everything except the sign test depends only on the request's tokens and
sampling parameters, not on the logits. So the host precomputes three numbers
per token, and the device applies one expression:

```
add = bias[t]                      mul = rep if seen else 1
sub = freq * count[t] + pres if t in output else 0

v = l[t] + add
l[t] = (v / mul if v > 0 else v * mul) - sub
```

A token nobody touched has `(add, mul, sub) = (0, 1, 0)`, which leaves its logit
unchanged.

### Chunk layout

The logits are viewed as `[rows, vocab / 64, 64]`. For each row the host keeps:

- `cidx [rows, K]`: the chunk ids this row touches, in the order first seen.
- `lanes [rows, 3, K, 64]`: `(add, mul, sub)` for every lane of those chunks.

`K` is the number of chunk slots. Slots past the last used one repeat slot 0, so
the gather and scatter always see `K` chunks and duplicate targets carry the
same values.

The device graph is then:

```
v   = gather(logits_view, cidx) + add          # [rows, K, 64]
v   = where(v > 0, v / mul, v * mul) - sub
out = scatter(logits_view, cidx, v)
-> temperature -> softmax -> top-k / top-p   (or argmax)
```

### Incremental host updates

- A new request is filled from its prompt, its output so far and its
  `logit_bias`.
- After that, each step adds only the output tokens produced since the last
  step. A token is penalized once, however often it repeats; its frequency term
  is updated on every occurrence.
- Requests are keyed by request id, so new and reordered rows need no special
  handling. A request's state is dropped when it finishes, not when a step
  leaves it out: a prefill step lists only the new request. Padded rows of the
  batch bucket stay at identity.
- The prompt is folded in only with a repetition penalty. Frequency and
  presence read output tokens only, so they never take slots for the prompt.
- Each request stores only its used slots. The tail that repeats slot 0 is
  filled when the step's buffer is assembled.
- A batch of one is laid out as two half-vocab rows.

### Shapes and compilation

A fused graph is keyed by its kind (greedy, top-k / top-p, or penalties only
for min_p steps),
the number of rows, the row vocab size and `K`. Penalty values, bias values,
temperature, top-k and top-p are tensor data, so changing them never
recompiles.

`K` comes from a fixed set per row size: 256, 512, 1024 and a cap, the largest
multiple of 64 within the row's chunk count (1152 for the half rows of a batch of
one and 2368 for full rows, with a 151936 vocab). Each step uses the smallest
`K` that fits the live requests, so it drops back once a long request leaves.
Every graph key has its own code object, and warm-up compiles every `K`, so
serving never compiles a new graph.

A request that touches more chunks than the cap takes the existing path for that
step.

## Limitations

- Lane parameters are uploaded as fp16.
- The vocab size must be a multiple of 128 and at least 8192.
