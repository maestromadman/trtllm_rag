import torch
import time
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/amallya0523/models/llama-3.1-8b"

PROMPT = "Explain the difference between the transformer encoder and decoder architectures in detail, including attention mechanisms, positional encoding, and typical use cases.""

#-- Some controls --
MAX_NEW_TOKENS = 200 # fixing this for consistency
NUM_TRIALS = 5  # five generations, will take avg, smooths out noise
WARMUP_RUNS = 1 # warmup run to mitigate any initial overhead
#-------------------

# Tokenizer & Model Loading
print("---Tokenizer---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


print("---Loading model...---")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0") # places model entiely on GPU, change as needed 

model.eval() # switches from train to inf modes

print(f"Model loaded on: {next(model.parameters()).device}") # sanity check for GPU



# INPUT TOKENIZIATION

inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0") # tokenizes sentence on GPU
input_token_count = inputs["input_ids"].shape[1] # num of tokens
print(f"Prompt tokenizes to {input_token_count} tokens.")


# WARMUP

print(f"Running {WARMUP_RUNS} warmup generations but discarding results!")
with torch.no_grad(): # don't compute gradients during inf
    for i in range(WARMUP_RUNS):
        i = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, # deterministic output for consistency
        )
torch.cuda.synchronize() # Python wait until GPU done

# RESET MEM TRACKER
torch.cuda.reset_peak_memory_stats()
memory_before_mb = torch.cuda.memory_allocated() / (1024 ** 2) # to MB
print(f"Static mem footprint: {memory_before_mb:.2f} MB")



# --- BASELINE TRIALS ---

latencies_ms = []
tokens_per_sec_list = []

print(f"Running {NUM_TRIALS} trials!")
for i in range(NUM_TRIALS):
    # start and end with cuda sync and perf_counter for accuracy
    torch.cuda.syncrhonize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    
    torch.cuda.synchronize() 
    t_end = time.perf_counter()

    elapsed_sec = t_end - t_start
    elapsed_ms = elapsed_sec * 1000
    new_tokens = output_ids.shape[1] - input_token_count
    tps = new_tokens / elapsed_sec

    latencies_ms.append(elapsed_ms) # add metrics to lists
    tokens_per_sec_list.append(tps)

    print(f"Trial {i+1}: {Elapsed_ms:7.1f} ms | {TPS:7.1f} tokens/sec") | {new_tokens} tokens generated")


# PEAK MEM USAGE
peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) # to MB



# ALL TOGETHER
avg_latency_ms = sum(latencies_ms) / NUM_TRIALS
avg_tps = sum(tokens_per_sec_list) / NUM_TRIALS
best_latency_ms = min(latencies_ms)

print("\n" + "=" * 40)
print(" HUGGING FACE FLOAT16 - BASELINE")
print("=" * 52)
print(f"   Avg Throughput   : {avg_tps:.2f} tok/s")
print(f"   Avg Latency      : {avg_latency_ms:.1f} ms")
print(f"   Best Latency     : {best_latency:.1f} ms")
print(f"   Peak GPU Memory  : {peak_memory_mb:.1f} MB")
print("=" * 52)



# SAVE RESULTS

results = {
    "backend": "huggingface_float16",
    "model_path": MODEL_PATH,
    "prompt_tokens": input_token_count,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_trials": NUM_TRIALS,
    "avg_tokens_per_sec": round(avg_tps, 2),
    "avg_latency_ms": round(avg_latency_ms, 1),
    "best_latency_ms": round(best_latency, 1),
    "peak_gpu_memory_mb": round(peak_memory_mb, 1),
    "per_trial_latency_ms": [round(x, 1) for x in latencies_ms],
    "per_trial_tps": [round(x, 2) for x in tokens_per_sec_list],
}

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2) # strucuted record stored on disk for Prometheus later

print("Results saved to baseline_results.json")
