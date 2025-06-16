#!/usr/bin/env python3
"""
convert_to_tflite_streaming.py

Loads the dynamic-time Keras model and converts it to three TFLite variants:
  1. float32: Baseline, largest size, best quality.
  2. float16: Half precision weights, smaller size, minor quality loss.
  3. int8: Fully integer quantized, smallest size, fastest on CPU, but
           requires careful calibration and may have quality degradation.

This version is geared for real-time streaming: it fixes preprocessing
alignment, uses fixed-length windows for calibration, and keeps I/O in float
for simpler integration in streaming applications.
"""

import os
import glob
import random
import argparse
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf

# ───────────────────────── PARAMETERS ─────────────────────────
DEFAULT_MODEL_PATH = os.path.join("./out", "piezo2mic_resnet_dynamic.keras")
OUT_DIR    = "./out"
SRC_DIR    = "./input/source"  # Calibration source .wav files
NUM_CALIB  = 200                # Increase coverage for calibration
SEG_LEN    = 8192 * 2           # Must match training segment length
WINDOW_LEN = SEG_LEN            # Streaming window size
OFFSET_PAD = 512                # Extra for filter settling
SEED       = 1234

# ───────────────────────── ARG PARSING ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   default=DEFAULT_MODEL_PATH,
                    help="Path to dynamic Keras model.")
parser.add_argument("--out_dir", default=OUT_DIR,
                    help="Where to save the TFLite files.")
parser.add_argument("--src_dir", default=SRC_DIR,
                    help="Source .wav files for calibration.")
parser.add_argument("--num_calib", type=int, default=NUM_CALIB,
                    help="# of samples for int8 calibration.")
parser.add_argument("--seg_len",  type=int, default=SEG_LEN,
                    help="Segment length for calibration.")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ───────────────────────── LOAD MODEL ─────────────────────────
if not os.path.exists(args.model):
    print(f"Error: cannot find model at {args.model}")
    sys.exit(1)
print(f"Loading model from {args.model}...")
model = tf.keras.models.load_model(args.model, compile=False)
model.summary()

# ───────────────────────── REPRESENTATIVE DATA GEN ─────────────────────────
def representative_data_gen():
    """
    Yields float32 windows of shape (1, WINDOW_LEN, 1)
    with the same preprocessing (mean removal) used at training.
    """
    files = glob.glob(os.path.join(args.src_dir, "*_source.wav"))
    if not files:
        raise RuntimeError(f"No .wav files in {args.src_dir}")
    random.shuffle(files)
    count = min(len(files), args.num_calib)
    print(f"Using {count} files for calibration...")
    for i in range(count):
        path = files[i]
        info = sf.info(path)
        read_len = args.seg_len + OFFSET_PAD
        start = random.randint(0, max(0, info.frames - read_len))
        data, _ = sf.read(path, start=start, frames=read_len, dtype='float32')
        # force mono if needed
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # trim settle region
        data = data[OFFSET_PAD:]
        # consistent preprocessing: mean removal
        data = data - np.mean(data)
        # take last WINDOW_LEN samples
        window = data[-args.seg_len:]
        inp = window.reshape(1, args.seg_len, 1)
        yield [inp]

# ─────────────────── CONVERT FLOAT32 ─────────────────────────
print("\n1) Converting float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
path32 = os.path.join(args.out_dir, "piezo2mic_fp32.tflite")
with open(path32, 'wb') as f:
    f.write(tflite_fp32)
print(f"Saved: {path32} ({len(tflite_fp32)/1024:.1f} KB)")

# ─────────────────── CONVERT FLOAT16 ─────────────────────────
print("\n2) Converting float16-quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
path16 = os.path.join(args.out_dir, "piezo2mic_fp16.tflite")
with open(path16, 'wb') as f:
    f.write(tflite_fp16)
print(f"Saved: {path16} ({len(tflite_fp16)/1024:.1f} KB)")

# ─────────────────── CONVERT INT8 ────────────────────────────
print("\n3) Converting fully integer int8 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Keep float I/O for streaming ease
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
tflite_int8 = converter.convert()
path8 = os.path.join(args.out_dir, "piezo2mic_int8.tflite")
with open(path8, 'wb') as f:
    f.write(tflite_int8)
print(f"Saved: {path8} ({len(tflite_int8)/1024:.1f} KB)")

print("\nAll conversions complete. Models saved to:", args.out_dir)
