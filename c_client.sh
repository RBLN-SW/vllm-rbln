S=examples/vllm/features/ec_disagg
T=/home/eunji.lee/artifacts/ec-interference/trace.json
O=/home/eunji.lee/artifacts/ec-interference

python \
  "$S/benchmark_interference.py" run \
  --trace "$T" \
  --base-url http://localhost:8000 \
  --result "$O/disaggregated-control.json" \
  --all-short