S=examples/vllm/features/ec_disagg
T=/home/eunji.lee/artifacts/ec-interference/trace.json
O=/home/eunji.lee/artifacts/ec-interference

python \
  "$S/benchmark_interference.py" run \
  --trace "$T" \
  --base-url http://localhost:8200 \
  --result "$O/integrated-control.json" \
  --all-short