"""
convert_to_tflite.py

Loads your trained Keras model (static shape: (SEG_LEN,1)),
produces three TFLite variants:
  1) default float32
  2) float16‐quantized
  3) full‐integer int8 (with representative dataset)

Writes them into OUT_DIR.
"""

import os
import glob
import random
import argparse
import numpy as np
import soundfile as sf
import tensorflow as tf

# ───────────────────────── PARAMETERS ─────────────────────────
SEG_LEN    = 2048          # must match your training SEG_LEN
SRC_DIR    = "./input/source"
OUT_DIR    = "./out"
NUM_CALIB  = 100           # how many segments to sample for int8 calibration
SEED       = 1234

# ───────────────────────── PARSE ARGS ─────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model", default=os.path.join(OUT_DIR, "piezo2mic_resnet.keras"),
               help="Path to your saved Keras model (static, fixed SEG_LEN).")
p.add_argument("--out_dir", default=OUT_DIR, help="Where to write .tflite files.")
p.add_argument("--seg_len", type=int, default=SEG_LEN, help="Segment length.")
p.add_argument("--src_dir", default=SRC_DIR, help="Where to read *.wav for calibration.")
p.add_argument("--num_calib", type=int, default=NUM_CALIB,
               help="Number of calibration steps for int8 quant.")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
tf.random.set_seed(SEED)
random.seed(SEED)

print("Loading Keras model from:", args.model)
model = tf.keras.models.load_model(args.model, compile=False)
print(model.summary())

# ──────────────────────── REPRESENTATIVE DATA ───────────────────────────────
def representative_data_gen():
    """
    Yields NUM_CALIB samples of shape [1, SEG_LEN, 1], float32 in the same
    distribution as your training data, for int8 quant calibration.
    """
    files = glob.glob(os.path.join(args.src_dir, "*_source.wav"))
    if not files:
        raise RuntimeError(f"No source .wav files in {args.src_dir}")
    random.shuffle(files)
    for src in files[: args.num_calib]:
        data, _ = sf.read(src, frames=args.seg_len, dtype="float32")
        if data.shape[0] < args.seg_len:
            data = np.pad(data, (0, args.seg_len - data.shape[0]), mode="constant")
        data = data - data.mean()
        # shape → [1, SEG_LEN, 1]
        inp = data.reshape(1, args.seg_len, 1)
        yield [inp]


# ─────────────────────── CONVERT FLOAT32 (baseline) ─────────────────────────
print("\n1) Converting baseline float32 TFLite model …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(args.out_dir, "piezo2mic_resnet_fp32.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("  ▶ Saved:", tflite_path,
      f"({len(tflite_model)/1024:.1f} KB)")

# ─────────────────────── CONVERT FLOAT16 ─────────────────────────────────────
print("\n2) Converting float16‐quantized TFLite model …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# this will keep activations in float32 but weights → float16
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
tflite_fp16_path = os.path.join(args.out_dir, "piezo2mic_resnet_fp16.tflite")
with open(tflite_fp16_path, "wb") as f:
    f.write(tflite_fp16)
print("  ▶ Saved:", tflite_fp16_path,
      f"({len(tflite_fp16)/1024:.1f} KB)")

# ─────────────────────── CONVERT INT8 ────────────────────────────────────────
print("\n3) Converting full‐integer int8 TFLite model …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# force full int8 inference
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8   # or tf.uint8 if you prefer [0..255]
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
tflite_int8_path = os.path.join(args.out_dir, "piezo2mic_resnet_int8.tflite")
with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8)
print("  ▶ Saved:", tflite_int8_path,
      f"({len(tflite_int8)/1024:.1f} KB)")

print("\nDone. You now have three models in", args.out_dir)
print(" – piezo2mic_resnet_fp32.tflite   (baseline)")
print(" – piezo2mic_resnet_fp16.tflite   (weights→float16)")
print(" – piezo2mic_resnet_int8.tflite   (full integer int8)")

# ─────────────────────── EXAMPLE LOADING & INFERENCE ─────────────────────────
print("""
Example usage (Python):
    import numpy as np
    import soundfile as sf
    import tensorflow as tf

    # ---- load TFLite
    interpreter = tf.lite.Interpreter(model_path="piezo2mic_resnet_int8.tflite",
                                     num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # ---- read a chunk
    data, sr = sf.read("some_source.wav", start=0, frames=2048, dtype="float32")
    data = data - data.mean()
    # if int8 model, quantize the input:
    scale, zero_point = input_details["quantization"]
    data_int8 = (data / scale + zero_point).astype(np.int8)
    data_int8 = data_int8.reshape((1,2048,1))

    interpreter.set_tensor(input_details["index"], data_int8)
    interpreter.invoke()
    out_int8 = interpreter.get_tensor(output_details["index"])
    # dequantize:
    out = (out_int8.astype(np.float32)-zero_point) * scale
    # write to disk
    sf.write("reconstructed.wav", out[0,:,0], sr)
""")