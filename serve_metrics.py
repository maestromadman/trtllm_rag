import json
import time
from prometheus_client import Gauge, start_http_server

THROUGHPUT = Gauge("llm_tokens_per_sec", "Tokens per second", ["backend"])
LATENCY    = Gauge("llm_avg_latency_ms", "Avg latency ms", ["backend"])
MEMORY     = Gauge("llm_peak_memory_mb", "Peak GPU memory MB", ["backend"])

for path, label in [
    ("/home/amallya0523/trtllm_rag/baseline_results.json", "huggingface_float16"),
    ("/home/amallya0523/trtllm_rag/trtllm_results.json",  "trtllm_int8"),
]:
    with open(path) as f:
        r = json.load(f)
    THROUGHPUT.labels(backend=label).set(r["avg_tokens_per_sec"])
    LATENCY.labels(backend=label).set(r["avg_latency_ms"])
    MEMORY.labels(backend=label).set(r["peak_gpu_memory_mb"])
    print(f"Loaded {label}: {r['avg_tokens_per_sec']} tok/s")

start_http_server(8000)
print("Serving metrics on port 8000 -- Ctrl+C to stop")
while True:
    time.sleep(10)
