#!/usr/bin/env python3
"""
stream_audio_refactored.py

A robust, low-latency audio streaming script for real-time inference with a TFLite model.

Architecture:
1.  **Ring Buffers**: Uses two lock-free ring buffers (input/output) implemented with
    NumPy arrays to safely pass audio data between threads.
2.  **Audio Thread (Producer)**: The `sounddevice` callback runs on a high-priority
    thread. Its only job is to receive audio from the input device and place it
    into the input ring buffer. It must be extremely fast to avoid glitches.
3.  **Processing Thread (Consumer)**: A dedicated background thread that continuously:
    a. Reads a block of audio from the input ring buffer, including enough
       historical data to satisfy the model's receptive field (overlap).
    b. Runs inference using the TFLite model.
    c. Places the processed audio into the output ring buffer.
4.  **Main Thread**: Sets up the audio stream, threads, and TFLite interpreter.

This design decouples the strict timing of the audio hardware from the inference
computation, using the ring buffers to absorb any timing jitter. This prevents the
clicks, pops, and pulsing noises caused by buffer under/overruns in the original script.
"""
import sounddevice as sd
import numpy as np
import tensorflow as tf
import time
import threading
import argparse
import sys
from ai_edge_litert.interpreter import Interpreter
import os
import pandas as pd
from tabulate import tabulate

# --- Disable GPU for this script, as we want to ensure CPU performance for TFLite
tf.config.set_visible_devices([], "GPU")

# ───────────────────────── SETTINGS ───────────────────────────────────────────
parser = argparse.ArgumentParser(description="Real-time audio processing with TFLite.")
parser.add_argument("--model", default="./out/piezo2mic_fp16.tflite",
                    help="Path to the TFLite model file.")
parser.add_argument("--blocksize", type=int, default=4096,
                    help="Processing block size. Smaller values reduce latency but increase CPU load.")
parser.add_argument("--buffer", type=float, default=1,
                    help="Buffer size in seconds. Larger values increase robustness but also latency.")
parser.add_argument("--threads", type=int, default=64,
                    help="Number of threads for TFLite interpreter.")
parser.add_argument("--input-device", type=int,
                    help="ID for the audio input device")
parser.add_argument("--output-device", type=int,
                    help="ID for the audio output device")
args = parser.parse_args()

SR         = 48000
CHANNELS   = 1
MODEL_PATH = args.model
BLOCK_SIZE = args.blocksize
NUM_THREADS= args.threads
BUFFER_DURATION = args.buffer

# ───────────────── 1. LOAD TFLITE MODEL AND GET DETAILS ─────────────────────
print(f"Loading TFLite model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    sys.exit(1)

interpreter = Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
ops_details = interpreter._get_ops_details()


# get details
tensors = interpreter.get_tensor_details()
ops     = interpreter._get_ops_details()

# build rows
tensor_rows = []
for t in tensors:
    shape_str = "×".join(str(d) for d in t["shape"])
    tensor_rows.append((
        t["index"],
        t["name"],
        shape_str,
        t["dtype"].__name__,
    ))

op_rows = []
for o in ops:
    in_str  = ",".join(str(i) for i in o["inputs"])
    out_str = ",".join(str(o_) for o_ in o["outputs"])
    op_rows.append((
        o["index"],
        o["op_name"],
        in_str,
        out_str,
    ))

# print them
print(tabulate(
    tensor_rows,
    headers=["#","Name","Shape","DType"],
    tablefmt="github",
    stralign="left",
    numalign="right",
))
print()
print(tabulate(
    op_rows,
    headers=["#","Op","In","Out"],
    tablefmt="github",
    stralign="left",
    numalign="right",
))

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Check model's data type
is_int8_model = input_details['dtype'] == np.int8
if is_int8_model:
    print("Loaded INT8 quantized model.")
    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]
else:
    print(f"Loaded {input_details['dtype'].__name__} model.")

# The model expects a certain input length. For a dynamic model, this will be a placeholder.
# We will feed it chunks of (RECEPTIVE_FIELD + BLOCK_SIZE).
model_input_len = input_details['shape'][-2]

# ───────────────── 2. COMPUTE RECEPTIVE FIELD (RF) ──────────────────────────
# A simplified way to estimate RF without loading the full Keras model.
# This assumes a ResNet structure like in the training script.
# Manually set this if the estimate is wrong. A larger value is safer.
# For the provided model: kernel_size=63, dilations=(1,2,4,8,16,32,64) -> (63-1)*sum(d) = 62 * 127 = 7874
RECEPTIVE_FIELD = 7874
print(f"Estimated Receptive Field (RF): {RECEPTIVE_FIELD} samples ({RECEPTIVE_FIELD/SR*1000:.1f} ms)")
INFERENCE_CHUNK_SIZE = RECEPTIVE_FIELD + BLOCK_SIZE
print(f"Inference will be run on chunks of {INFERENCE_CHUNK_SIZE} samples.")

# Explicitly resize the input tensor to our desired dynamic shape.
# This must be done *before* allocating tensors.
print(f"Resizing model input tensor to shape: [1, {INFERENCE_CHUNK_SIZE}, 1]")
interpreter.resize_tensor_input(input_details['index'], [1, INFERENCE_CHUNK_SIZE, 1])
interpreter.allocate_tensors()



# ───────────────── 3. SETUP AUDIO DEVICES ───────────────────────────────────

if (args.input_device is not None and args.output_device is not None):
    inp_dev = args.input_device
    out_dev = args.output_device
else:
    try:
        devices = sd.query_devices()
        print("\nAvailable audio devices:\n")
        for idx, dev in enumerate(devices):
            print(f"  {idx:>2}: {dev['name']:<40}  in:{dev['max_input_channels']:<2}  out:{dev['max_output_channels']:<2}")
        
        # Let's try to find a Focusrite device automatically
        focusrite_devices = [i for i, d in enumerate(devices) if 'focusrite' in d['name'].lower()]
        default_device_idx = focusrite_devices[0] if focusrite_devices else 0

        inp_dev = int(input(f"\nSelect INPUT device index [{default_device_idx}]: ") or default_device_idx)
        out_dev = int(input(f"Select OUTPUT device index [{default_device_idx}]: ") or default_device_idx)

        print(f"\nUsing INPUT  #{inp_dev}: {devices[inp_dev]['name']}")
        print(f"Using OUTPUT #{out_dev}: {devices[out_dev]['name']}\n")
    except Exception as e:
        print(f"Error setting up audio devices: {e}")
        sys.exit(1)

# ───────────────── 4. SETUP THREAD-SAFE RING BUFFERS & STATE ────────────────
# The buffers need to be large enough to handle timing jitter between threads.
BUFFER_SAMPLES = int(SR * BUFFER_DURATION)
input_ring_buffer = np.zeros((BUFFER_SAMPLES, CHANNELS), dtype=np.float32)
output_ring_buffer = np.zeros((BUFFER_SAMPLES, CHANNELS), dtype=np.float32)

# read/write pointers for the buffers. These are advanced by different threads.
# `volatile` is not a concept in Python, but these are our shared state.
read_ptr = 0
write_ptr = 0
# A simple high-pass filter state for removing DC offset from the input.
dc_state = 0.0
HPF_ALPHA = 0.995

# Threading control
is_streaming = threading.Event()
is_streaming.set()

# ───────────────── 5. PROCESSING THREAD ─────────────────────────────────────
def processing_thread_func():
    global read_ptr, dc_state
    print("Processing thread started.")
    
    # History buffer to provide context to the model (the "overlap" part)
    history = np.zeros((RECEPTIVE_FIELD, CHANNELS), dtype=np.float32)
    
    while is_streaming.is_set():
        # Wait until there's a full block of new data available to process
        while (write_ptr - read_ptr) % BUFFER_SAMPLES < BLOCK_SIZE:
            time.sleep(0.001) # Sleep briefly to avoid busy-waiting
            if not is_streaming.is_set(): return

        # --- Grab a block of new data from the ring buffer
        # This section needs to be atomic in a real C++ implementation,
        # but Python's GIL helps us here. It's "good enough" for this.
        current_read_ptr = read_ptr
        end_ptr = current_read_ptr + BLOCK_SIZE
        
        # Handle wraparound for the ring buffer
        if end_ptr > BUFFER_SAMPLES:
            block1 = input_ring_buffer[current_read_ptr:]
            block2 = input_ring_buffer[:end_ptr % BUFFER_SAMPLES]
            new_block = np.concatenate([block1, block2])
        else:
            new_block = input_ring_buffer[current_read_ptr:end_ptr]

        # Advance the read pointer
        read_ptr = (read_ptr + BLOCK_SIZE) % BUFFER_SAMPLES

        # --- Prepare data for inference
        # Concatenate history with the new block
        model_input = np.concatenate([history, new_block])
        
        # Update history for the next iteration by rolling the buffer
        history = np.roll(history, -BLOCK_SIZE, axis=0)
        history[-BLOCK_SIZE:] = new_block

        # Add batch dimension for the model: (T, 1) -> (1, T, 1)
        model_input = model_input[np.newaxis, ...]

        # --- Run Inference
        if is_int8_model:
            # Quantize float32 input to int8
            model_input = (model_input / in_scale + in_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details["index"])

        if is_int8_model:
            # Dequantize int8 output back to float32
            y_pred = (y_pred.astype(np.float32) - out_zero_point) * out_scale

        # The model's output is of size (1, RF+BLOCK_SIZE, 1). We only want
        # the new audio, which is the last BLOCK_SIZE samples.
        output_block = y_pred[0, -BLOCK_SIZE:, :]

        # --- Write processed data to the output ring buffer
        # This logic assumes the output callback can keep up.
        # A more robust system would have a separate write pointer for output.
        current_write_pos = current_read_ptr # Write to the same "time slot"
        if current_write_pos + BLOCK_SIZE > BUFFER_SAMPLES:
            output_ring_buffer[current_write_pos:] = output_block[:BUFFER_SAMPLES - current_write_pos]
            output_ring_buffer[:(current_write_pos + BLOCK_SIZE) % BUFFER_SAMPLES] = output_block[BUFFER_SAMPLES - current_write_pos:]
        else:
            output_ring_buffer[current_write_pos:current_write_pos + BLOCK_SIZE] = output_block

    print("Processing thread finished.")

# ───────────────── 6. AUDIO CALLBACK (REAL-TIME THREAD) ─────────────────────
def audio_callback(indata, outdata, frames, time_info, status):
    global write_ptr, dc_state
    if status:
        print("Stream status:", status, file=sys.stderr)

    # This callback must be FAST. No inference, no heavy processing.
    # Just copy data into the ring buffer.
    
    # Simple IIR high-pass filter to remove DC offset, vectorized
    indata_no_dc = np.empty_like(indata)
    last_out = dc_state
    indata_no_dc[0] = indata[0,0] - last_out
    for i in range(1, frames):
        indata_no_dc[i] = indata[i,0] - (HPF_ALPHA * indata_no_dc[i-1] + (1-HPF_ALPHA)*indata[i,0])
    dc_state = indata[frames-1, 0]

    # --- Write input data to the ring buffer
    current_write_ptr = write_ptr
    end_ptr = current_write_ptr + frames
    if end_ptr > BUFFER_SAMPLES:
        # Handle wraparound
        input_ring_buffer[current_write_ptr:] = indata_no_dc[:BUFFER_SAMPLES-current_write_ptr]
        input_ring_buffer[:end_ptr % BUFFER_SAMPLES] = indata_no_dc[BUFFER_SAMPLES-current_write_ptr:]
    else:
        input_ring_buffer[current_write_ptr:end_ptr] = indata_no_dc
    
    write_ptr = (write_ptr + frames) % BUFFER_SAMPLES
    
    # --- Read processed data from the output ring buffer
    if (write_ptr - read_ptr) % BUFFER_SAMPLES < frames:
        # Not enough processed data yet, output silence to avoid stale data
        outdata[:] = 0
    else:
        out_ptr = read_ptr
        if out_ptr + frames > BUFFER_SAMPLES:
            outdata[:BUFFER_SAMPLES-out_ptr] = output_ring_buffer[out_ptr:]
            outdata[BUFFER_SAMPLES-out_ptr:] = output_ring_buffer[:(out_ptr+frames)%BUFFER_SAMPLES]
        else:
            outdata[:] = output_ring_buffer[out_ptr:out_ptr+frames]

# ───────────────── 7. START STREAM ──────────────────────────────────────────
if __name__ == "__main__":
    try:
        # Start the processing thread
        proc_thread = threading.Thread(target=processing_thread_func)
        proc_thread.daemon = True
        proc_thread.start()

        # Pre-fill the output buffer with a bit of silence
        # to ensure the callback has something to read initially.
        read_ptr = -int(SR*0.1) % BUFFER_SAMPLES

        # Start the audio stream
        stream = sd.Stream(
            samplerate=SR,
            blocksize=BLOCK_SIZE,
            device=(inp_dev, out_dev),
            channels=CHANNELS,
            dtype='float32',
            latency='low',
            callback=audio_callback
        )
        with stream:
            print(f"Streaming started... (blocksize={BLOCK_SIZE}, buffer={BUFFER_DURATION}s)")
            print("Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping stream...")
        is_streaming.clear()
        proc_thread.join()
        print("Stream stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

