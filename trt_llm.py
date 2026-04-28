import time
import json
import torch
from tensorrt_llm import LLM, SamplingParams

# CONFIG 

ENGINE_DIR = "/workspace/trtllm_rag/llama_int8_engine" # file path to TensorRT engine, located in the engines folder of this repo

PROMPT = (
    "Explain the difference between the transformer encoder and decoder architectures "
    "in detail, including attention mechanisms, positional encoding, and typical use cases."
)

MAX_NEW_TOKENS = 200 # same controls as baseline
NUM_TRIALS = 5
WARMUP_RUNS = 1



# LOADING ENGINE 

print("---Loading TensorRT Engine---")
llm = LLM(model=ENGINE_DIR) # engine already compiled, just loaded into GPU mem

sampling_params = SamplingParams(
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.0 # equivalent to do_sample=False in HF
)



# WARM-UP

print(f"Running {WARMUP_RUNS} warm-up pass(es) and result discarded!")
for i in range(WARMUP_RUNS):
    llm.generate([PROMPT], sampling_params)

torch.cuda.synchronize() # again, just waiting for GPU to finish 


# RESET MEMORY TRACKER
torch.cuda.reset_peak_memory_stats()
memory_before_mb = torch.cuda.memory_allocated() / (1024 ** 2) # bytes -> MB
print(f"Static engine memory footprint: {memory_before_mb:.2f} MB")


# TIMED TRIALS

latencies_ms = []
tokens_per_sec_list = []

print(f"\nRunning {NUM_TRIALS} timed trials...")
for i in range(NUM_TRIALS):
    torch.cuda.synchronize()            # ensure no async GPU work from previous, start timer right before generation
    t_start = time.perf_counter()

    outputs = llm.generate([PROMPT], sampling_params)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed_sec = t_end - t_start
    elapsed_ms  = elapsed_sec * 1000

    new_tokens = len(outputs[0].outputs[0].token_ids)   # number of new tokens generated (excluding prompt)

    tps = new_tokens / elapsed_sec

    latencies_ms.append(elapsed_ms)
    tokens_per_sec_list.append(tps)
    print(f"  Trial {i+1}: {elapsed_ms:7.1f} ms | {tps:6.1f} tok/s | {new_tokens} tokens generated")



# PEAK MEMORY 
peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) # bytes -> MB

# AGGREGATE RESULTS
avg_latency_ms = sum(latencies_ms) / NUM_TRIALS
avg_tps        = sum(tokens_per_sec_list) / NUM_TRIALS
best_latency   = min(latencies_ms)

print("\n" + "=" * 52)
print("   TRTLLM INT8 — BENCHMARK RESULTS")
print("=" * 52)
print(f"   Avg Throughput   : {avg_tps:.2f} tok/s")
print(f"   Avg Latency      : {avg_latency_ms:.1f} ms")
print(f"   Best Latency     : {best_latency:.1f} ms")
print(f"   Peak GPU Memory  : {peak_memory_mb:.1f} MB")
print("=" * 52)



# FINAL COMPARISON
print("\n--- Comparison against HuggingFace baseline ---")
try:
    with open("/workspace/trtllm_rag/baseline_results.json") as f:
        baseline = json.load(f)
    speedup = avg_tps / baseline["avg_tokens_per_sec"]
    mem_reduction = baseline["peak_gpu_memory_mb"] - peak_memory_mb
    print(f"Throughput speedup: {speedup:.2f}x")
    print(f"Memory reduction: {mem_reduction:.0f} MB saved")
except FileNotFoundError:
    print("(baseline_results.json not found — run baseline.py first)")


# SAVING RESULTS
results = {
    "backend": "trtllm_int8_weight_only",
    "engine_dir": ENGINE_DIR,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_trials": NUM_TRIALS,
    "avg_tokens_per_sec": round(avg_tps, 2),
    "avg_latency_ms": round(avg_latency_ms, 1),
    "best_latency_ms": round(best_latency, 1),
    "peak_gpu_memory_mb": round(peak_memory_mb, 1),
    "per_trial_latency_ms": [round(x, 1) for x in latencies_ms],
    "per_trial_tps": [round(x, 2) for x in tokens_per_sec_list],
}

with open("/workspace/trtllm_rag/trtllm_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to trtllm_results.json")

