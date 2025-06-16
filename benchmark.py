# benchmark.py
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter

# --- CONFIGURATION ---
MODEL_PATH = "./out/piezo2mic_fp32.tflite" # Use your fastest model!
BLOCK_SIZE = 4096
NUM_THREADS = 4
NUM_RUNS = 100

# --- SETUP ---
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
input_details = interpreter.get_input_details()[0]
receptive_field = 7874
inference_chunk_size = receptive_field + BLOCK_SIZE

interpreter.resize_tensor_input(input_details['index'], [1, inference_chunk_size, 1])
interpreter.allocate_tensors()

# Create a dummy input tensor
dummy_input = np.random.randn(1, inference_chunk_size, 1).astype(np.float32)
if input_details['dtype'] == np.int8:
    scale, zero_point = input_details["quantization"]
    dummy_input = (dummy_input / scale + zero_point).astype(np.int8)

# --- BENCHMARK ---
print(f"Benchmarking {MODEL_PATH}...")
print(f"Input shape: {dummy_input.shape}")

# Warm-up run
interpreter.set_tensor(input_details["index"], dummy_input)
interpreter.invoke()

# Timed runs
start_time = time.perf_counter()
for _ in range(NUM_RUNS):
    interpreter.set_tensor(input_details["index"], dummy_input)
    interpreter.invoke()
end_time = time.perf_counter()

# --- RESULTS ---
total_time = end_time - start_time
avg_time_ms = (total_time / NUM_RUNS) * 1000
block_duration_ms = (BLOCK_SIZE / 48000) * 1000

print("\n--- Results ---")
print(f"Time for one block: {block_duration_ms:.2f} ms")
print(f"Average inference time: {avg_time_ms:.2f} ms")

if avg_time_ms > block_duration_ms:
    print("\n🔴 CONCLUSION: The model is TOO SLOW for real-time.")
    print(f"   It takes {avg_time_ms:.2f}ms to process a block that is only {block_duration_ms:.2f}ms long.")
else:
    print("\n🟢 CONCLUSION: The model is fast enough. The bottleneck is elsewhere (e.g., Python overhead).")