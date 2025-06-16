#!/usr/bin/env python3
"""
convert_to_tflite_streaming.py

Loads your dynamic‐time Keras model and produces three TFLite variants:
  1) float32
  2) float16 (half‐precision weights/activations)
  3) int8 “weight‐only” (activations remain float32)
"""
import os
import argparse
import tensorflow as tf

# ───────────────────────── PARAMETERS ─────────────────────────────────────────
DEFAULT_FP32_MODEL = os.path.join("out", "piezo2mic_resnet_dynamic.keras")
DEFAULT_OUT_DIR    = "out"

# ───────────────────────── ARG PARSING ────────────────────────────────────────
p = argparse.ArgumentParser(
    description="Convert piezo2mic dynamic Keras → TFLite (fp32, fp16, int8-weights)")
p.add_argument("--model_dyn", default=DEFAULT_FP32_MODEL,
               help="Path to your FP32 dynamic‐time .keras model")
p.add_argument("--out_dir",   default=DEFAULT_OUT_DIR,
               help="Where to write the .tflite files")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# derive paths
fp32_keras = args.model_dyn
qweight_keras = fp32_keras.replace(".keras", "_qweight.keras")

# ──────────────────── 1) FLOAT32 BASELINE ────────────────────────────────────
print("1) Converting float32 TFLite from:", fp32_keras)
model_fp32 = tf.keras.models.load_model(fp32_keras, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
tflite_fp32 = converter.convert()
out32 = os.path.join(args.out_dir, "piezo2mic_fp32.tflite")
with open(out32, "wb") as f:
    f.write(tflite_fp32)
print(f"   Saved: {out32}   ({len(tflite_fp32)/1024:.1f} KB)")

# ──────────────────── 2) FLOAT16 QUANTIZATION ───────────────────────────────
print("\n2) Converting float16 TFLite from:", fp32_keras)
converter = tf.lite.TFLiteConverter.from_keras_model(model_fp32)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
out16 = os.path.join(args.out_dir, "piezo2mic_fp16.tflite")
with open(out16, "wb") as f:
    f.write(tflite_fp16)
print(f"   Saved: {out16}   ({len(tflite_fp16)/1024:.1f} KB)")

# ──────────────────── 3) INT8 WEIGHT‐ONLY QUANTIZATION ─────────────────────
if os.path.exists(qweight_keras):
    print("\n3) Converting int8‐weight‐only TFLite from:", qweight_keras)
    model_qw = tf.keras.models.load_model(qweight_keras, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_qw)
    # weight‐only quant: no representative_dataset, just Optimize.DEFAULT
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_i8 = converter.convert()
    out8 = os.path.join(args.out_dir, "piezo2mic_int8.tflite")
    with open(out8, "wb") as f:
        f.write(tflite_i8)
    print(f"   Saved: {out8}   ({len(tflite_i8)/1024:.1f} KB)")
else:
    print(f"\n⚠️  Weight‐quantized model not found at:\n   {qweight_keras}\n"
          "   Skipping int8 weight-only conversion.")

print("\nAll conversions complete. Models in:", args.out_dir)