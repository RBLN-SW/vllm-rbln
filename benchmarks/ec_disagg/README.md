# Gemma 4 encoder disaggregation benchmark

Run these commands from the repository root with a vLLM/RBLN runtime
environment. The scripts use `.venv` in the checkout by default and accept
an activated environment or an absolute `VENV_DIR` override. The workload
uses `google/gemma-4-26B-A4B` and the Gemma 4 chat template from
[vLLM v0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/examples/tool_chat_template_gemma4.jinja).

## Baseline

`baseline/serve.sh` starts a full-model replica on each listed device and
routes requests round-robin through the baseline proxy. Wait for the proxy
on port 8000 to respond to `/health`, then run the benchmark in another
terminal.

```bash
DEVICES=0,1,2,3 bash benchmarks/ec_disagg/baseline/serve.sh
bash benchmarks/ec_disagg/bench.sh baseline
```

## Encoder, prefill, and decode

`epd/endpoint.sh` starts the producers and consumers, waits for them to
become healthy, and starts the proxy on port 10002. The first devices in
`DEVICES` are assigned to producers. Each producer gets its own EC
side-channel port starting at 5601.

```bash
DEVICES=0,1,2,3 bash benchmarks/ec_disagg/epd/endpoint.sh 1e3pd
bash benchmarks/ec_disagg/bench.sh 1e3pd
```

Use `2e2pd` for two producers and two consumers or `1e1pd` for one of
each. The baseline and EPD layouts can use the same devices in separate
runs. Stop the launcher with Ctrl+C after each run. Use `taskset` on
either launcher to apply the same CPU affinity in both layouts.

`epd/producer.sh`, `epd/consumer.sh`, and `epd/proxy.sh` can also be
started separately. For these launches, set `RBLN_VISIBLE_DEVICES` (or
`RBLN_DEVICES`) and unique HTTP ports per process. Give each producer a
unique `VLLM_EC_SIDE_CHANNEL_PORT` and set the proxy's `PRODUCER_URLS`
and `CONSUMER_URLS` to comma-separated HTTP origins. For same-host
experiments the default UCX transports are `tcp,sm,self`; override
`UCX_TLS` and `UCX_NET_DEVICES` for a different transport.

The launcher applies `epd/proxy.patch` to a checksum-pinned upstream
vLLM v0.31.0 proxy. This preserves round-robin consumer routing and
handles requests with incomplete image metadata. The upstream script
is taken from the local cache or downloaded on first use. For a local
proxy script, set `PROXY_SCRIPT` to its absolute path.

Metadata-only image requests require a model implementation that publishes
image placeholder metadata. The proxy forwards the image when that metadata
is unavailable.

## Results

By default, `bench.sh` runs 20 measured requests per image count at a
target of 2 QPS and writes per-case JSON and `summary.csv` under
`benchmarks/ec_disagg/results/gemma4-smoke-uuid`. Pass additional
benchmark arguments after the topology, use `BASE_URL` for a different
proxy port, and give a fresh `OUTPUT_DIR` when rerunning completed cases
or changing the workload. The launchers write logs under
`benchmarks/ec_disagg/logs`.
