#!/usr/bin/env python3
"""
===================================
INTRODUCTION
===================================
This architecture is designed to train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal.

==================================
ARCHITECTURE
==================================

The fully Keras Serializable TensorFlow network is trained on pre-segmented audio chunks, split into pick and finger playing styles, and different
FILM (Feature-wise Linear Modulation) parameters are used to control the style of the output.

The network predicts not just the mic'd audio, but also the style of playing (pick vs finger) as a binary classification task, and
is aided by the true style labels during training.

Two models are trained in a teacher-student setup:
- An offline model, which is a large, non-causal model with look-ahead and a wide receptive field.
  It operates on 96K/24 bit audio, which provides better accuracy when predicting the sound and shape of the initial pick attack.
  This model is meant as a high-quality reference for the Realtime model, but it can also be used for offline processing, where latency is not a concern.

- A realtime model, which has much fewer parameters, is fully causal with no look-ahead, a tiny receptive field, and is designed to be ultra low-latency and has an end-to-end latency of ~7.5ms.
  It operates on 48K/24 bit audio, and distills the offline model's features to learn a more compact representation of the audio.
  The static realtime model distills based on the offline model's layers, while the dynamic realtime model is fully self-contained and explicitly copies over the weights from the offline model's layers to its own (rather than using set_weights, which would cause issues).
  This model is meant for real-time applications, where low latency is critical, such as live performance or real-time monitoring. It is small enough to run on embedded, low-power devices.
  
It is designed to be optimized for TFLite conversion and run the realtime models on embedded, low-powered ARM microcontrollers. All operations must support efficient conversion to that platform, including full support for INT8 quantization and pruning.
  
================================
INPUT DATA
================================
The input data consists of pre-segmented audio files, which are split into two styles:
- Pick style: audio recorded with a piezo pickup, where the player uses a pick to pluck the strings.
- Finger style: audio recorded with a piezo pickup, where the player uses their fingers to pluck the strings.

All audio files are recorded at 96KHz/24 bit.
At least 1 hour of audio is provided for each style, with a 50% ratio of each style in the training set.

Because the realtime model is operating on 48KHz/24 bit audio, during training, the audio within that is resampled to 48KHz, while distilling the features from the offline model at the original sample rate.

===============================
PRE-PROCESSING
===============================

- Automatic mic→piezo alignment using GCC-PHAT and fractional (sub-sample) shift (disabled by default because inputs are pre-aligned offline)
- Audio is normalized to a target RMS level of -23 dBFS, both for base model training and finetuning.
- Audio is split into contiguous non-overlapping 5-second chunks, plus an optional remainder.
- Each chunk is split into pick and finger segments, with a 50% ratio of each style.

================================
TRAINING
================================
The training is done in two stages:

1. Base model training: The offline- and realtime models are trained on the pre-segmented audio chunks with the true style labels.
   The style forcing blend is used to stabilize the learning of the heads while the classifier is still weak.
   These models are trained "from the factory" on several hours of audio, controlled and recorded by us, which allows us to have true style labels.

2. Fine-tuning: The base models can then be fine-tuned on audio we do not control neither the structure nor the length of, that may combine playing styles,
   without any true style labels. These rely on the style classification layers from the base models and freeze most of the weights during training,
   allowing the model to adapt to potentially very short amounts of foreign audio, while preserving the learned features.
   
================================
LOSS FUNCTIONS
================================
The loss functions used in the training are a combination of several components:
- Waveform L1 loss: The basic loss that measures the difference between the predicted and target audio waveforms.
- HF loss: A high-frequency loss that emphasizes the high-frequency content of the audio, which is important for capturing the initial pick attack, which is very difficult to capture
- Onset envelope loss: A loss that measures the difference between the predicted and target onset envelopes, which helps to capture the shape of the initial pick attack.
- Multi-scale STFT loss: A loss that measures the difference between the predicted and target audio in the frequency domain, using multiple scales of STFT.
- Log-Mel loss: A loss that measures the difference between the predicted and target audio in the log-Mel domain, which is a more perceptually relevant representation of audio.
- RMS envelope loss: A loss that measures the difference between the predicted and target RMS envelopes, which helps to capture the overall loudness of the audio.
- Complex STFT loss: A loss that measures the difference between the predicted and target audio in the complex STFT domain, which helps to capture the phase information of the audio.
- D2 loss: A loss that measures the difference between the predicted and target audio in the D2 domain, which is a more perceptually relevant representation of audio.
- Style classification loss: A binary cross-entropy loss that measures the difference between the predicted and true style labels, which helps to capture the style of the audio (pick vs finger).

================================
STATIC VS DYNAMIC MODELS AND
THEIR RELATION TO SAMPLE RATE
================================
- The offline model operates at 96K/24 bit audio, both in its static and dynamic modes.
- The realtime model operates internally at 48K/24 bit audio, and in its static mode,
  it has to resample to 96K when distilling features from the offline model.
  However, as the dynamic model does not rely on the offline model, it does not require resampling.

================================
CONSTRAINTS
================================
- Everything needs to be Keras-serializable
- The realtime model must be fully causal, with no look-ahead, and ultra low-latency.
"""

import os, glob, random, shutil, sys, re, math
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import minimum_phase, firwin, resample_poly
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import ( # pyright: ignore[reportMissingImports]
    TensorBoard, EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, TerminateOnNaN
)
from tensorflow.keras import regularizers, constraints  # pyright: ignore[reportMissingImports]
from tensorflow.keras.losses import Loss  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (  # pyright: ignore[reportMissingImports]
    Input, Conv1D, SeparableConv1D, ZeroPadding1D, GlobalAveragePooling1D,
    Activation, Multiply, Add, Lambda, Dense, Layer, Reshape, Minimum
)

# Determinism
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS      = 1000

# Audio sample‐rates
OFFLINE_SAMPLE_RATE = 96000
REALTIME_SAMPLE_RATE = 48000
FINETUNE_SAMPLE_RATE = 48000

#INPUT_AVG_RMS_DBFS_TARGET = -23.0
STYLE_FLAG_PROB_THRESHOLD = 0.5  # threshold for pick vs finger style flag

# NORMALIZED_PREFIX = "_normalized_"
# WRITE_NORMALIZED_TO_DISK = True

# Debugging params
DEBUG               = False

# Batch params
OFFLINE_BATCH_SIZE          = 4
REALTIME_BATCH_SIZE         = 12
FINETUNE_BATCH_SIZE         = 32

# Chunk ratios
PICK_CHUNK_RATIO = 0.5
FINGER_CHUNK_RATIO = 0.5

CALIBRATION_CHUNK_COUNT = 32

# Offline params
OFFLINE_FILTERS                = 160
OFFLINE_KERNEL_SIZE            = 51
OFFLINE_DILATIONS              = (1,2,4,8,16,32,64,128,256)
OFFLINE_PRE_BLOCKS             = 2
OFFLINE_PRE_KERNEL             = 7

# Bounded look-ahead per layer (in ms):
OFFLINE_LOOKAHEAD_MS           = 12.0
OFFLINE_HF_HP_ORDER            = 129
OFFLINE_LOOKAHEAD_SAMPLES      = int(round(OFFLINE_SAMPLE_RATE * OFFLINE_LOOKAHEAD_MS / 1000.0))
OFFLINE_RECEPTIVE_FIELD        = 1 + (OFFLINE_KERNEL_SIZE - 1) * sum(OFFLINE_DILATIONS)
OFFLINE_TRAIN_BLOCK_SIZE       = 4096
OFFLINE_SEG_LEN                = OFFLINE_RECEPTIVE_FIELD + OFFLINE_TRAIN_BLOCK_SIZE
OFFLINE_EMBED_DIM              = 32

OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ   = 14000.0
OFFLINE_GATE_THRESHOLD_DB   = -52.0
OFFLINE_GATE_RATIO          = 50
OFFLINE_GATE_ATTACK_MS      = 0.1
OFFLINE_GATE_HOLD_MS        = 4.0
OFFLINE_GATE_RELEASE_MS     = 4.0
OFFLINE_GATE_ENV_MS          = 0.50
OFFLINE_GATE_DEPTH_SMOOTH_MS = 20.0

OFFLINE_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_offline.keras"
OFFLINE_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_offline_dynamic.keras"

# Realtime params
REALTIME_FILTERS          = 48
REALTIME_KERNEL_SIZE      = 25
REALTIME_DILATIONS        = (1,2,4,8)
REALTIME_PRE_BLOCKS       = 1
REALTIME_PRE_KERNEL       = 5
REALTIME_RECEPTIVE_FIELD  = 1 + (REALTIME_KERNEL_SIZE - 1) * sum(REALTIME_DILATIONS)
REALTIME_TRAIN_BLOCK_SIZE = 4096
REALTIME_SEG_LEN          = REALTIME_RECEPTIVE_FIELD + REALTIME_TRAIN_BLOCK_SIZE
REALTIME_EMBED_DIM        = 16
REALTIME_DISTILL_ALPHA    = 5.0
REALTIME_HF_MASK_DISTILL_ALPHA = 2.0  # KD strength for the learned HF mask (teacher gate → student mask)
REALTIME_HF_HP_ORDER          = OFFLINE_HF_HP_ORDER
REALTIME_STYLE_DISTILL_ALPHA = 1.0   # KD strength for the style probability (teacher→student)

REALTIME_AUDIO_STFT_DISTILL_ALPHA   = 0.6   # KD weight for STFT KD (teacher audio_ds vs student audio)
REALTIME_STEM_PICK_DISTILL_ALPHA    = 0.5   # KD weight for pick stem
REALTIME_STEM_FINGER_DISTILL_ALPHA  = 0.5   # KD weight for finger stem
REALTIME_STEM_HF_DISTILL_ALPHA      = 0.8   # KD weight for HF stem
REALTIME_MASK_USED_DISTILL_ALPHA    = 0.3   # KD of mask_used vs teacher gate (small; avoids cap conflict)
REALTIME_FEAT_DISTILL_ALPHA         = 1.0
REALTIME_STEM_RESO_DISTILL_ALPHA    = 0.4

# Training-only penalty to suppress <~3 kHz energy in the realtime model's HF branch
REALTIME_HF_LOWBAND_PENALTY_ALPHA   = 0.8   # L1 on LPF(hf_final) with ~2.8-3.0 kHz cutoff
REALTIME_STYLE_BLEND_INIT           = 0.7
REALTIME_STYLE_BLEND_TRAINABLE      = True
REALTIME_OFFLINE_FEAT_LAYERS   = ["offline_dil_2_film_apply", "offline_dil_3_film_apply"]

REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ   = OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ
REALTIME_GATE_THRESHOLD_DB               = OFFLINE_GATE_THRESHOLD_DB
REALTIME_GATE_RATIO                      = OFFLINE_GATE_RATIO
REALTIME_GATE_ATTACK_MS                  = 0.1
REALTIME_GATE_HOLD_MS                    = 4.0
REALTIME_GATE_RELEASE_MS                 = 6.0
REALTIME_GATE_ENV_MS                     = 0.05
REALTIME_GATE_DEPTH_SMOOTH_MS            = 20.0
REALTIME_GATE_ENV_LMAX                  = 256     # keep kernels inside streaming left context
REALTIME_GATE_DEPTH_LMAX                = 256

# ─── Resonance branch params ─────────────────────────────────────────────
# OFFLINE (96k): more modes and longer IRs
OFFLINE_RESO_ENABLE           = True
OFFLINE_RESO_MODES            = 16
OFFLINE_RESO_LENGTH           = 257     # causal FIR length per mode
OFFLINE_RESO_CENTERS_HZ       = [240, 320, 420, 540, 700, 900, 1150, 1450, 1800, 2250, 2800, 3400, 4100, 4900, 5800, 6800]
OFFLINE_RESO_DECAYS_MS        = [70, 65, 60, 55, 50, 45, 40, 38, 35, 32, 30, 28, 26, 24, 22, 20]
OFFLINE_RESO_STRIKE_GAIN      = 3.0
OFFLINE_RESO_BOOST_DB         = 0.0

REALTIME_RESO_ENABLE          = True
REALTIME_RESO_MODES           = 12
REALTIME_RESO_LENGTH          = 97
REALTIME_RESO_CENTERS_HZ      = [280, 360, 460, 600, 780, 1000, 1300, 1700, 2200, 2900, 3600, 4300]
REALTIME_RESO_DECAYS_MS       = [60, 55, 50, 45, 40, 35, 32, 28, 25, 22, 20, 18]
REALTIME_RESO_STRIKE_GAIN     = 2.5
REALTIME_RESO_BOOST_DB        = 0.0

REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_realtime.keras"
REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_realtime_dynamic.keras"

# Finetune params
FINETUNE_REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_realtime.keras"
FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_realtime_dynamic.keras"

FINETUNE_REALTIME_EPOCHS      = 10
FINETUNE_REALTIME_LR          = 1e-4
FINETUNE_L1_WEIGHT           = 15.0
FINETUNE_MEL_WEIGHT          = 15.0

FINETUNE_DEMO_SEC = 60
FINETUNE_DEMO_OFFSET_SEC = 0

# HF loss shaping
HP_CUTOFF_HZ   = 3000.0
GATE_DEBUG_TONE_MULT = 1.0

HP_PRE_EMPHASIS_HF        = 0.98
HF_ONSET_GAIN             = 8.0
STFT_HIGH_BOOST           = 32.0
STFT_HF_ONSET_GAIN        = 12.0

L1_WEIGHT                 = 1.0
HF_WEIGHT                 = 5.0
STFT_WEIGHT               = 4.0
MEL_WEIGHT                = 3.0
RMS_ENV_WEIGHT            = 1.0
COMPLEX_STFT_WEIGHT       = 0.5
D2_WEIGHT                 = 0.2
LAMBDA_STYLE_RATIO        = 0.5

# ──────── TUNING KNOBS ─────────────────────────────────────────────────────────
SEGMENT_GENERATOR_MIN_AMP_THRESH         = 1e-5
#───────────────────────────────────────────────────────────────────────────────

# ─── IR branch FIR FILTER DESIGN ────────────────────────────────────────────
REALTIME_IR_DOWNSAMPLE_FACTOR = 8
REALTIME_IR_FIR_SIZE          = 129
REALTIME_IR_CUTOFF            = 1.0 / REALTIME_IR_DOWNSAMPLE_FACTOR
REALTIME_IR_KERNEL    = 33
REALTIME_IR_DILATIONS = (1,2,4,8)
REALTIME_IR_BOOST_DB = 0.0

REALTIME_LP_ORDER   = 65
REALTIME_LP_CUTOFF  = 4000

LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

OFFLINE_PICK_BOOST_DB = 22
REALTIME_PICK_BOOST_DB = 8

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

DEMO_SEC        = 10
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

LONG_DEMO_SEC        = 10
LONG_DEMO_OFFSET_SEC = 0
LONG_DEMO_PAIR_INDEX = 0

PICK_SRC_DIR   = "./input/pick/source"
PICK_TGT_DIR   = "./input/pick/target"
FINGER_SRC_DIR = "./input/finger/source"
FINGER_TGT_DIR = "./input/finger/target"

FINETUNE_SRC_DIR   = "./input/finetune"

OUT_DIR        = "./out"

CHECKPOINT_OUT_DIR = os.path.join(OUT_DIR, "checkpoints")
CHECKPOINT_OFFLINE_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "offline")
CHECKPOINT_REALTIME_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "realtime")

MODELS_OUT_DIR = os.path.join(OUT_DIR, "models")
MODELS_OFFLINE_OUT_DIR = os.path.join(MODELS_OUT_DIR, "offline")
MODELS_REALTIME_OUT_DIR = os.path.join(MODELS_OUT_DIR, "realtime")

LOGS_OUT_DIR = os.path.join(OUT_DIR, "logs")
LOGS_OFFLINE_OUT_DIR = os.path.join(LOGS_OUT_DIR, "offline")
LOGS_REALTIME_OUT_DIR = os.path.join(LOGS_OUT_DIR, "realtime")

DEMOS_OUT_DIR = os.path.join(OUT_DIR, "demos")

LONG_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "long")
LONG_DEMOS_OFFLINE_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "offline")
LONG_DEMOS_REALTIME_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "realtime")

FINETUNE_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "finetune")

DEMO_GENERATOR_OUTPUTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "generator-outputs")

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_OFFLINE_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "offline")
DEMO_CHECKPOINTS_REALTIME_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "realtime")

SAVE_WEIGHTS           = False

# ─── Minimum‑phase IR low‑pass taps for the realtime IR branch ──────────────

def make_minphase_lp_kernel(order=REALTIME_LP_ORDER,
                            cutoff_hz=REALTIME_LP_CUTOFF,
                            sr=REALTIME_SAMPLE_RATE,
                            window="hamming"):
    # Linear-phase prototype
    taps_lin = firwin(order,
                      cutoff=cutoff_hz/(sr/2.0),
                      window=window,
                      pass_zero=True).astype(np.float32)
    # Convert to minimum-phase
    taps_min = minimum_phase(taps_lin, method="homomorphic").astype(np.float32)
    # Preserve DC gain
    taps_min /= (np.sum(taps_min) + 1e-12)
    # Reverse for tf.nn.conv1d (cross-correlation) to implement causal convolution
    return tf.constant(taps_min[::-1].reshape(-1, 1, 1), dtype=tf.float32)

# Use this kernel as the default for IRLowpass
realtime_lp_taps_tf = make_minphase_lp_kernel()

for dir_path in [DEMO_GENERATOR_OUTPUTS_OUT_DIR, CHECKPOINT_OFFLINE_OUT_DIR, CHECKPOINT_REALTIME_OUT_DIR, LOGS_OFFLINE_OUT_DIR, LOGS_REALTIME_OUT_DIR, DEMO_CHECKPOINTS_OFFLINE_OUT_DIR, DEMO_CHECKPOINTS_REALTIME_OUT_DIR, LONG_DEMOS_OFFLINE_OUT_DIR, LONG_DEMOS_REALTIME_OUT_DIR, FINETUNE_DEMOS_OUT_DIR, MODELS_OFFLINE_OUT_DIR, MODELS_REALTIME_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def _extract_epoch_from_ckpt(path):
    fn = os.path.basename(path)
    m = re.search(r'epoch_(\d+)', fn)
    return int(m.group(1)) if m else 0

def gate_debug_tone_freq(sample_rate, cutoff_hz, mult=GATE_DEBUG_TONE_MULT):
    """
    Pick an audible, safe tone frequency tied to the gate HP cutoff.
    Clamps to [50 Hz, 0.45*Nyquist] to avoid aliasing and inaudible extremes.
    """
    freq = float(cutoff_hz) * float(mult)
    nyq_guard = 0.45 * float(sample_rate)
    if freq < 50.0:
        freq = 50.0
    if freq > nyq_guard:
        freq = nyq_guard
    return freq

def resample_to_model_sr(audio, audio_sample_rate: int, model_sample_rate: int):
    g = math.gcd(audio_sample_rate, model_sample_rate)
    up   = model_sample_rate // g
    down = audio_sample_rate // g
    # Ensure float32 output (SciPy may return float64)
    y = resample_poly(audio.astype(np.float32, copy=False), up, down).astype(np.float32, copy=False)
    return y

# ────────────────────────────── DATA LOADING ───────────────────────────────────
picked_pairs = []
for src in sorted(glob.glob(os.path.join(PICK_SRC_DIR, "*_source.wav"))):
    tgt = src.replace("_source.wav","_target.wav").replace(PICK_SRC_DIR, PICK_TGT_DIR)
    if os.path.exists(tgt):
        picked_pairs.append((src, tgt))
    else:
        print("Warning: no target for", src)
finger_pairs = []
for src in sorted(glob.glob(os.path.join(FINGER_SRC_DIR, "*_source.wav"))):
    tgt = src.replace("_source.wav","_target.wav").replace(FINGER_SRC_DIR, FINGER_TGT_DIR)
    if os.path.exists(tgt):
        finger_pairs.append((src, tgt))
    else:
        print("Warning: no target for", src)

all_pairs = picked_pairs + finger_pairs

# normalize files and align target to source
#target_rms = 10 ** (INPUT_AVG_RMS_DBFS_TARGET / 20.0)

audio_cache = {}
for src, tgt in all_pairs:
    data_src, _ = sf.read(src, dtype='float32', always_2d=True)
    data_tgt, _ = sf.read(tgt, dtype='float32', always_2d=True)
    info_src = sf.info(src)
    info_tgt = sf.info(tgt)
    if info_src.samplerate != info_tgt.samplerate:
        print(f"Warning: sample-rate mismatch for pair {src} vs {tgt}: {info_src.samplerate} vs {info_tgt.samplerate}")
    file_sr = info_src.samplerate

    mono_src = data_src[:,0]
    mono_tgt = data_tgt[:,0]

    audio_cache[src] = data_src
    audio_cache[tgt] = data_tgt

def build_chunks(pairs_list, seg_len, sample_rate):
    """
    Create chunks using the source file sample-rate (not the model's sample-rate).
    This prevents incorrect chunking when training the 48 kHz realtime model on 96 kHz files.
    """
    chunks = []
    for src, tgt in pairs_list:
        info = sf.info(src)
        file_sr = info.samplerate
        total = info.frames
        chunk_len = int(CHUNK_SEC * file_sr)

        n_full = total // chunk_len
        for i in range(n_full):
            chunks.append((src, tgt, i * chunk_len, chunk_len))

        rem = total - n_full * chunk_len
        # Ensure the remainder is long enough to produce a model segment after resampling
        # Minimum source samples needed to produce seg_len model samples:
        need_src_len = int(math.ceil(seg_len * file_sr / float(sample_rate)))
        if rem >= need_src_len:
            chunks.append((src, tgt, n_full * chunk_len, rem))
    return chunks
        
def assert_has_pairs ():
    if not all_pairs:
        print("ERROR: No training pairs found. Expected files named *_source.wav and *_target.wav "
            "under PICK_SRC_DIR/PICK_TGT_DIR and/or FINGER_SRC_DIR/FINGER_TGT_DIR.")
        sys.exit(1)
        
def copy_weights_by_name(dst_model, src_model, verbose=False):
    """
    Copy weights layer-by-layer by matching layer names and shapes.
    Returns the number of layers whose weights were copied.
    Leaves non-matching layers untouched.
    """
    src_layers = {l.name: l for l in src_model.layers}
    matched = 0
    for l in dst_model.layers:
        ref = src_layers.get(l.name)
        if ref is None:
            continue
        sw = ref.get_weights()
        dw = l.get_weights()
        if len(sw) != len(dw):
            continue
        # Check shape compatibility
        ok = True
        for s, d in zip(sw, dw):
            if np.shape(s) != np.shape(d):
                ok = False
                break
        if not ok:
            continue
        l.set_weights(sw)
        matched += 1
    if verbose:
        print(f"[copy_weights_by_name] copied {matched} layer(s) by name into '{dst_model.name}'")
    return matched

class DemoCallback(Callback):
    def __init__(self, sample_rate, demo_checkpoints_out_dir, build_model_fn, build_model_fn_args_dynamic={}, use_safe_weight_copy=False):
        super().__init__()
        assert_has_pairs()
        demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
        
        # if demo_src includes "pick", use the 1.0 as style_flag, otherwise use 0.0
        if "pick" in demo_src:
            self.style_true_flag = 1.0
        elif "finger" in demo_src:
            self.style_true_flag = 0.0
        else:
            # It might be a blended chunk where the styles may be intermixed
            self.style_true_flag = None
        
        self.src_path = demo_src
        self.tgt_path = demo_tgt
        self.demo_sec = DEMO_SEC
        self.offset_sec = DEMO_OFFSET_SEC
        self.build_model_fn = build_model_fn
        self.build_model_fn_args_dynamic = build_model_fn_args_dynamic
        self.sample_rate = sample_rate
        self.every = 1
        self.demos_out_dir = demo_checkpoints_out_dir
        self.use_safe_weight_copy = use_safe_weight_copy
        os.makedirs(self.demos_out_dir, exist_ok=True)

    def set_model(self, model):
        self.static_model = model
        self.dynamic_model = self.build_model_fn(
                            **self.build_model_fn_args_dynamic,
                            dynamic_time=True)
        if self.use_safe_weight_copy:
            copy_weights_by_name(self.dynamic_model, self.static_model)
        else:
            self.dynamic_model.set_weights(self.static_model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every != 0:
            return
        if self.use_safe_weight_copy:
            copy_weights_by_name(self.dynamic_model, self.static_model)
        else:
            self.dynamic_model.set_weights(self.static_model.get_weights())

        # Compute slicing in FILE sample-rate, not model sample-rate
        info_src = sf.info(self.src_path)
        file_sr = info_src.samplerate
        start_frames = int(round(self.offset_sec * file_sr))
        num_frames   = int(round(self.demo_sec   * file_sr))

        x_long, x_src_sample_rate = sf.read(self.src_path,
                            start=start_frames,
                            frames=num_frames,
                            dtype='float32',
                            always_2d=True)
        y_long, _ = sf.read(self.tgt_path,
                            start=start_frames,
                            frames=num_frames,
                            dtype='float32',
                            always_2d=True)
        x_src = x_long[:,0]
        y_src = y_long[:,0]
        if x_src_sample_rate != self.sample_rate:
            x_mod = resample_to_model_sr(x_src, x_src_sample_rate, self.sample_rate)
            y_mod = resample_to_model_sr(y_src, x_src_sample_rate, self.sample_rate)
        else:
            x_mod = x_src
            y_mod = y_src
        x_demo = x_mod[None, :, None]
        
        # First predict the style using the dynamic model, not based on the true style flag, but on the classifier itself.
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)
        y_inf_style_pred, style_pred_out = self.dynamic_model.predict([x_demo, dummy_style, force_zero], batch_size=1, verbose=0)
        y_inf_style_pred = y_inf_style_pred[0, :, 0]
        style_pred = float(style_pred_out[0, 0])
        style_pred_name = 'pick' if style_pred > STYLE_FLAG_PROB_THRESHOLD else 'finger'

        # If true style known, also infer with true style flag
        if self.style_true_flag is not None:
            style_true = np.array([[self.style_true_flag]], dtype=np.float32)
            force_one  = np.ones((1, 1), dtype=np.float32)
            y_inf_style_true, _ = self.dynamic_model.predict([x_demo, style_true, force_one], batch_size=1, verbose=0)
            y_inf_style_true = y_inf_style_true[0, :, 0]
            style_true_name = 'pick' if self.style_true_flag >= STYLE_FLAG_PROB_THRESHOLD else 'finger'
        else:
            y_inf_style_true = None
            style_true_name = None
        
        epoch_dir = os.path.join(self.demos_out_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        sf.write(os.path.join(epoch_dir,"demo_source.wav"), x_mod, self.sample_rate, subtype='FLOAT')
        sf.write(os.path.join(epoch_dir,"demo_target.wav"), y_mod, self.sample_rate, subtype='FLOAT')
        
        sf.write(os.path.join(epoch_dir,"demo_pred_style_pred.wav"),   y_inf_style_pred, self.sample_rate, subtype='FLOAT')
        if y_inf_style_true is not None:
            sf.write(os.path.join(epoch_dir,"demo_pred_style_true.wav"), y_inf_style_true, self.sample_rate, subtype='FLOAT')
        
        print(f"[DemoCallback] epoch {epoch+1}: predicted style={style_pred_name}, true style={style_true_name}, written to {epoch_dir}")

        

# ────────────────────────────── LOSSES ───────────────────────────────────────


@register_keras_serializable()
class CombinedLoss(Loss):
    def __init__(self, sample_rate, extra_small_cplx=False, name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.extra_small_cplx = bool(extra_small_cplx)
    def call(self, y_true, y_pred):
        return combined_loss(y_true, y_pred, self.sample_rate,
                             extra_small_cplx=self.extra_small_cplx)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "extra_small_cplx": self.extra_small_cplx
        })
        return config

@register_keras_serializable()
class LogMelLoss(Loss):
    def __init__(self, sample_rate, name="log_mel_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
    def call(self, y_true, y_pred):
        return log_mel_loss(y_true, y_pred, self.sample_rate)
    def get_config(self):
        config = super().get_config()
        config.update({"sample_rate": self.sample_rate})
        return config

@register_keras_serializable()
def waveform_l1_loss(y_true, y_pred):
    return tf.abs(y_true - y_pred)

@register_keras_serializable()
def hf_loss(y_true, y_pred):
    α = HP_PRE_EMPHASIS_HF
    true_hp = y_true[:,1:,:] - α*y_true[:,:-1,:]
    pred_hp = y_pred[:,1:,:] - α*y_pred[:,:-1,:]
    true_hp = tf.pad(true_hp, [[0,0],[1,0],[0,0]])
    pred_hp = tf.pad(pred_hp, [[0,0],[1,0],[0,0]])
    return tf.abs(true_hp - pred_hp)

def _build_freq_weight(n_fft, sample_rate, split_hz, high_boost):
    freqs = np.linspace(0, sample_rate/2.0, n_fft//2 + 1, dtype=np.float32)
    w = np.ones_like(freqs, dtype=np.float32)
    w[freqs >= split_hz] = high_boost
    return tf.constant(w[None, None, :], dtype=tf.float32)

@register_keras_serializable()
def multi_scale_stft_loss(y_true, y_pred):
    scales = [
        (128,   32,  128),
        (256,   64,  256),
        (512,  128,  512),
        (1024, 256, 1024),
        (2048, 512, 2048),
        (8192, 2048, 8192)
    ]
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    per_scale_losses = []
    for n_fft, hop, win in scales:
        frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
        frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
        window = tf.signal.hann_window(win, periodic=True)
        window = tf.reshape(window, (1,1,win))
        win_t = frames_t * window
        win_p = frames_p * window
        st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
        st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
        mag_t = tf.abs(st_t)
        mag_p = tf.abs(st_p)
        loss_sf = tf.reduce_mean(tf.abs(mag_t - mag_p), axis=[1,2])
        per_scale_losses.append(loss_sf)
    loss_stack = tf.stack(per_scale_losses, axis=0)
    loss_batch = tf.reduce_mean(loss_stack, axis=0)
    return loss_batch[:, None, None]

@register_keras_serializable()
def mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                   split_hz=HP_CUTOFF_HZ,
                                   high_boost=STFT_HIGH_BOOST,
                                   onset_env=None,
                                   onset_gain=STFT_HF_ONSET_GAIN):
    scales = [
        (128,   32,  128),
        (256,   64,  256),
        (512,  128,  512),
        (1024, 256, 1024),
        (2048, 512, 2048),
        (8192, 2048, 8192)
    ]
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    per_scale_losses = []
    for n_fft, hop, win in scales:
        frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
        frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
        window = tf.signal.hann_window(win, periodic=True)
        window = tf.reshape(window, (1,1,win))
        win_t = frames_t * window
        win_p = frames_p * window
        st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
        st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
        mag_t = tf.abs(st_t)
        mag_p = tf.abs(st_p)
        diff = tf.abs(mag_t - mag_p)  # (B, frames, F)
        wf = _build_freq_weight(n_fft, sample_rate, split_hz, high_boost)
        diff = diff * wf
        if onset_env is not None:
            env_f = tf.signal.frame(tf.squeeze(onset_env, axis=-1), frame_length=win, frame_step=hop, pad_end=True)
            env_w = tf.reduce_mean(env_f, axis=-1, keepdims=True)
            fw = 1.0 + onset_gain * env_w
            diff = diff * fw
        loss_sf = tf.reduce_mean(diff, axis=[1,2])
        per_scale_losses.append(loss_sf)
    loss_stack = tf.stack(per_scale_losses, axis=0)
    loss_batch = tf.reduce_mean(loss_stack, axis=0)
    return loss_batch[:, None, None]

@register_keras_serializable()
def complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=256, hop=64, win=256, onset_env=None, onset_gain=STFT_HF_ONSET_GAIN):
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
    frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
    window = tf.signal.hann_window(win, periodic=True)
    window = tf.reshape(window, (1,1,win))
    win_t = frames_t * window
    win_p = frames_p * window
    st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
    st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
    diff = tf.abs(tf.math.real(st_t - st_p)) + tf.abs(tf.math.imag(st_t - st_p))
    if onset_env is not None:
        env_f = tf.signal.frame(tf.squeeze(onset_env, axis=-1), frame_length=win, frame_step=hop, pad_end=True)
        env_w = tf.reduce_mean(env_f, axis=-1, keepdims=True)
        fw = 1.0 + onset_gain * env_w
        diff = diff * fw
    loss = tf.reduce_mean(diff, axis=[1,2])
    return loss[:, None, None]

@register_keras_serializable()
def second_diff_l1_onset(y_true, y_pred, onset_env=None):
    def d2(x):
        return x[:,2:,:] - 2.0*x[:,1:-1,:] + x[:,:-2,:]
    t2 = d2(y_true)
    p2 = d2(y_pred)
    sd = tf.abs(t2 - p2)
    sd = tf.pad(sd, [[0,0],[1,1],[0,0]])
    if onset_env is not None:
        sd = sd * (1.0 + HF_ONSET_GAIN * onset_env)
    return sd

@register_keras_serializable()
def log_mel_loss(y_true, y_pred, sample_rate):
    n_fft   = 1024
    hop     = 256
    win     = 1024
    n_mels  = 64
    fmin    = 20.0
    fmax    = float(sample_rate//2)
    t   = tf.squeeze(y_true, axis=-1)
    p   = tf.squeeze(y_pred, axis=-1)
    st_t = tf.signal.stft(t,   frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
    st_p = tf.signal.stft(p,   frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
    mag_t = tf.abs(st_t)
    mag_p = tf.abs(st_p)
    fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft//2+1,
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )
    mel_t = tf.tensordot(mag_t, fb, axes=[[2],[0]])
    mel_p = tf.tensordot(mag_p, fb, axes=[[2],[0]])
    log_t = tf.math.log(mel_t + 1e-6)
    log_p = tf.math.log(mel_p + 1e-6)
    mel_diff = tf.reduce_mean(tf.abs(log_t - log_p), axis=[1,2])
    return mel_diff[:,None,None]

@register_keras_serializable()
def rms_env_loss(y_true, y_pred):
    env_win  = 1024
    env_hop  = 512
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    ft   = tf.signal.frame(t, frame_length=env_win, frame_step=env_hop, pad_end=True)
    fp   = tf.signal.frame(p, frame_length=env_win, frame_step=env_hop, pad_end=True)
    rms_t = tf.sqrt(tf.reduce_mean(ft**2, axis=2) + 1e-12)
    rms_p = tf.sqrt(tf.reduce_mean(fp**2, axis=2) + 1e-12)
    env_diff = tf.reduce_mean(tf.abs(rms_t - rms_p), axis=1)
    return env_diff[:,None,None]


@register_keras_serializable()
def combined_loss(y_true, y_pred, sample_rate, extra_small_cplx=False):
    # Use model-appropriate gate
    gate_env = gate_weight_dynamic(y_true, sample_rate)

    td   = waveform_l1_loss(y_true, y_pred)
    hf   = hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * gate_env)
    spec = mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                          split_hz=HP_CUTOFF_HZ,
                                          high_boost=STFT_HIGH_BOOST,
                                          onset_env=gate_env, onset_gain=STFT_HF_ONSET_GAIN)
    mel  = log_mel_loss(y_true, y_pred, sample_rate)
    env  = rms_env_loss(y_true, y_pred)

    cplx_base = complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=256, hop=64, win=256, onset_env=gate_env)
    if extra_small_cplx:
        cplx_64 = complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=64, hop=16, win=64, onset_env=gate_env)
        cplx = 0.5 * (cplx_base + cplx_64)
    else:
        cplx = cplx_base

    d2   = second_diff_l1_onset(y_true, y_pred, onset_env=gate_env)
    return (L1_WEIGHT       * td
          + HF_WEIGHT       * hf
          + STFT_WEIGHT     * spec
          + MEL_WEIGHT      * mel
          + RMS_ENV_WEIGHT  * env
          + COMPLEX_STFT_WEIGHT * cplx
          + D2_WEIGHT       * d2)

# Replace mean_hf_metric with this version
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    vals = hf_loss(y_true, y_pred)  # (B, T, 1)
    # Return per-sample metric with shape (B, 1, 1), compatible with (B, T, 1) sample weights
    return tf.reduce_mean(vals, axis=[1, 2], keepdims=True)

# ────────────────────────────── DATA PIPELINE HELPERS ───────────────────────

@register_keras_serializable()
def pre_emphasis(x, alpha=HP_PRE_EMPHASIS_HF):
    """
    Simple 1st-order high-pass (pre-emphasis): y[n] = x[n] - alpha * x[n-1]
    x shape: (B, T, C)
    """
    x_shift = tf.pad(x, [[0,0],[1,0],[0,0]])[:, :-1, :]
    return x - alpha * x_shift

def segment_generator(seg_len, chunks, is_pick_chunks, model_sample_rate):
    """
    Yields (x, (y_audio, style_flag)) where x and y_audio are (seg_len, 1) at model_sample_rate.
    Handles resampling on the fly from the source file sample-rate.
    """
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = sf.info(src).samplerate

        # How many source samples are needed to produce seg_len model samples via resampling?
        need_src_len = int(math.ceil(seg_len * file_sr / float(model_sample_rate)))
        # Random offset in source-sample domain
        off = random.randrange(max(cdur - need_src_len, 1))
        rs = cstart + off

        piezo_seg_src = audio_cache[src][rs:rs + need_src_len, 0]
        mic_seg_src   = audio_cache[tgt][rs:rs + need_src_len, 0]

        # Skip too-silent windows early (on source-rate segment)
        if np.max(np.abs(piezo_seg_src)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        # Resample if needed to the model's sample-rate
        if file_sr != model_sample_rate:
            piezo_seg = resample_to_model_sr(piezo_seg_src, file_sr, model_sample_rate)
            mic_seg   = resample_to_model_sr(mic_seg_src,   file_sr, model_sample_rate)
        else:
            piezo_seg = piezo_seg_src
            mic_seg   = mic_seg_src

        # Ensure exact length seg_len at model rate
        if len(piezo_seg) < seg_len:
            p = seg_len - len(piezo_seg)
            piezo_seg = np.pad(piezo_seg, (0, p))
            mic_seg   = np.pad(mic_seg,   (0, p))
        elif len(piezo_seg) > seg_len:
            piezo_seg = piezo_seg[:seg_len]
            mic_seg   = mic_seg[:seg_len]

        x = piezo_seg.astype('float32')[:, None]
        y_audio = mic_seg.astype('float32')[:, None]
        style_flag = 1.0 if is_pick_chunks else 0.0
        yield x, (y_audio, style_flag)
        
def segment_generator_dual(seg_len_realtime,
                           chunks,
                           is_pick_chunks,
                           realtime_sample_rate=REALTIME_SAMPLE_RATE,
                           offline_sample_rate=OFFLINE_SAMPLE_RATE):
    """
    Yields ((x_realtime, x_offline), (y_audio_realtime, style_flag)).

    Supports any integer ratio offline_sample_rate : realtime_sample_rate.
    """
    if offline_sample_rate % realtime_sample_rate != 0:
        raise ValueError(f"offline_sample_rate ({offline_sample_rate}) must be an integer multiple of realtime_sample_rate ({realtime_sample_rate}).")

    factor = offline_sample_rate // realtime_sample_rate

    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = sf.info(src).samplerate

        need_src_len_realtime = int(math.ceil(seg_len_realtime * file_sr / float(realtime_sample_rate)))
        off = random.randrange(max(cdur - need_src_len_realtime, 1))
        rs = cstart + off

        piezo_seg_src = audio_cache[src][rs:rs + need_src_len_realtime, 0]
        mic_seg_src   = audio_cache[tgt][rs:rs + need_src_len_realtime, 0]

        if np.max(np.abs(piezo_seg_src)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        # realtime branch
        if file_sr != realtime_sample_rate:
            x_realtime = resample_to_model_sr(piezo_seg_src, file_sr, realtime_sample_rate)
            y_realtime = resample_to_model_sr(mic_seg_src,   file_sr, realtime_sample_rate)
        else:
            x_realtime = piezo_seg_src
            y_realtime = mic_seg_src

        # fit exact realtime length
        if len(x_realtime) < seg_len_realtime:
            p = seg_len_realtime - len(x_realtime)
            x_realtime = np.pad(x_realtime, (0, p))
            y_realtime = np.pad(y_realtime, (0, p))
        elif len(x_realtime) > seg_len_realtime:
            x_realtime = x_realtime[:seg_len_realtime]
            y_realtime = y_realtime[:seg_len_realtime]

        # offline branch = factor × seg_len_realtime
        seg_len_offline = factor * seg_len_realtime
        if file_sr != offline_sample_rate:
            x_offline = resample_to_model_sr(piezo_seg_src, file_sr, offline_sample_rate)
        else:
            x_offline = piezo_seg_src

        if len(x_offline) < seg_len_offline:
            p = seg_len_offline - len(x_offline)
            x_offline = np.pad(x_offline, (0, p))
        elif len(x_offline) > seg_len_offline:
            x_offline = x_offline[:seg_len_offline]

        x_realtime = x_realtime.astype('float32')[:, None]
        x_offline = x_offline.astype('float32')[:, None]
        y_audio_realtime = y_realtime.astype('float32')[:, None]
        style_flag = 1.0 if is_pick_chunks else 0.0

        yield (x_realtime, x_offline), (y_audio_realtime, style_flag)

def _extract_audio_io(model, batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Unexpected dataset element")
    x = batch[0]
    y = batch[1]
    if isinstance(y, (tuple, list)):
        y_true_audio = y[0]
    elif isinstance(y, dict):
        audio_key = model.output_names[0]
        y_true_audio = y[audio_key]
    else:
        y_true_audio = y
    y_pred_audio, _ = model(x, training=False)
    return y_true_audio, y_pred_audio

def analyze_loss_scales(model, batch, sample_rate):
    # Extract audio tensors from (x, y, [sw]) batch
    y_true, y_pred = _extract_audio_io(model, batch)

    # Use the model-appropriate gate (offline vs realtime) based on sample_rate
    gate_env = gate_weight_dynamic(y_true, sample_rate)

    l1   = tf.reduce_mean(waveform_l1_loss(y_true, y_pred)).numpy()
    mel  = tf.reduce_mean(log_mel_loss(y_true, y_pred, sample_rate)).numpy()
    env  = tf.reduce_mean(rms_env_loss(y_true, y_pred)).numpy()

    # Match CombinedLoss behavior: gate-weighted HF and STFT terms
    hf_on  = tf.reduce_mean(hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * gate_env)).numpy()
    stft_on = tf.reduce_mean(
        mrstft_mag_loss_weighted_onset(
            y_true, y_pred, sample_rate,
            split_hz=HP_CUTOFF_HZ,
            high_boost=STFT_HIGH_BOOST,
            onset_env=gate_env,
            onset_gain=STFT_HF_ONSET_GAIN
        )
    ).numpy()

    return {'l1': l1, 'stft': stft_on, 'mel': mel, 'env': env, 'hf': hf_on}

def calibrate_loss_weights_consistent(model, dataset, sample_rate, n_batches=5,
                                      normalize_keys=('l1','stft','mel','env','hf'),
                                      declared=None):
    if declared is None:
        declared = {'l1': L1_WEIGHT, 'stft': STFT_WEIGHT, 'mel': MEL_WEIGHT, 'env': RMS_ENV_WEIGHT, 'hf': HF_WEIGHT}
    sums = {k:0.0 for k in normalize_keys}
    count = 0
    for batch in dataset.take(n_batches):
        sc = analyze_loss_scales(model, batch, sample_rate)
        for k in normalize_keys:
            sums[k] += float(sc[k])
        count += 1
    avgs = {k: (sums[k] / max(count,1)) for k in normalize_keys}
    print("Average consistent loss scales:", {k: round(v, 6) for k,v in avgs.items()})
    ref = float(avgs['l1']) if avgs['l1'] > 0 else 1.0
    new_w = {}
    for k in normalize_keys:
        bs = float(avgs[k])
        new_w[k] = (declared[k] / bs * ref) if bs > 0 else 0.0
    return new_w, avgs

def generate_demo (model, out_dir, sample_rate, prediction_name="demo_pred", only_prediction=False):
    assert_has_pairs()
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]
    return generate_demo_from_files(
        model=model,
        out_dir=out_dir,
        sample_rate=sample_rate,
        path_src=pth_src,
        path_target=pth_tgt,
        prediction_name=prediction_name,
        only_prediction=only_prediction
    )

def _get_model_context_for_streaming(model_sample_rate: int):
    """
    Returns (left_ctx_model, right_ctx_model) in samples at the model's sample-rate.
    """
    if model_sample_rate == OFFLINE_SAMPLE_RATE:
        # Left = receptive field - 1 (causal extent), Right = bounded look-ahead per layer
        return int(OFFLINE_RECEPTIVE_FIELD - 1), int(OFFLINE_LOOKAHEAD_SAMPLES)
    elif model_sample_rate == REALTIME_SAMPLE_RATE:
        return int(REALTIME_RECEPTIVE_FIELD - 1), 0
    else:
        # Fallback: treat as causal with tiny RF (safe default)
        return 1024, 0  # adjust if you add other sample-rates
        

def generate_demo_from_files(model,
                             out_dir,
                             sample_rate,
                             path_src,
                             path_target,
                             prediction_name="demo_pred",
                             only_prediction=False,
                             offset_sec=LONG_DEMO_OFFSET_SEC,
                             sec=LONG_DEMO_SEC,
                             block_sec=12.0,
                             resample_margin_ms=50.0):
    """
    Streaming/chunked version: keeps memory flat by processing blocks with
    sufficient left/right context and a small resampling margin, and writes out
    results incrementally.

    - Handles sample-rate mismatch via resample_poly per block.
    - Uses overlap-and-trim so only the "valid" interior (free of edge effects) is written.
    - For offline (96 kHz), also applies HF boost as in your original function.

    Args:
        block_sec: length (in seconds at model SR) of the core region written per iteration.
        resample_margin_ms: extra margin (ms at file SR) around each block to absorb resampler transients.
    """
    os.makedirs(out_dir, exist_ok=True)
        
    hf_scale_layer_name = "offline_hf_scale" if sample_rate == OFFLINE_SAMPLE_RATE else "realtime_hf_scale"
    pick_boost_db = OFFLINE_PICK_BOOST_DB if sample_rate == OFFLINE_SAMPLE_RATE else REALTIME_PICK_BOOST_DB
    
    if pick_boost_db != 0.0:
        model = boost_hf_scale_in_model(model, hf_scale_layer_name, pick_boost_db)

    if sample_rate == REALTIME_SAMPLE_RATE and REALTIME_IR_BOOST_DB != 0.0:
        model = boost_hf_scale_in_model(model, "realtime_ir_scale", gain_db=REALTIME_IR_BOOST_DB)
        
    # Optional: resonance stem boost for quick A/B during demo generation
    try:
        if OFFLINE_RESO_BOOST_DB != 0.0 and sample_rate == OFFLINE_SAMPLE_RATE:
            model = boost_hf_scale_in_model(model, "offline_reso_scale", gain_db=OFFLINE_RESO_BOOST_DB)
        elif REALTIME_RESO_BOOST_DB != 0.0 and sample_rate == REALTIME_SAMPLE_RATE:
            model = boost_hf_scale_in_model(model, "realtime_reso_scale", gain_db=REALTIME_RESO_BOOST_DB)
    except Exception:
        pass  # layer may not exist (e.g., older models)

    # Determine FILE sample-rate and region to render in file-frame domain
    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    # Style flag from filename (same logic as your original function)
    if "pick" in path_src:
        style_true_flag = 1.0
    elif "finger" in path_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    # Model contexts and conversions
    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    # Outputs
    style_pred_path = os.path.join(out_dir, prediction_name + "_style_pred.wav")
    style_true_path = os.path.join(out_dir, prediction_name + "_style_true.wav")

    w_pred = sf.SoundFile(style_pred_path, mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_true = None
    if style_true_flag is not None:
        w_true = sf.SoundFile(style_true_path, mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

    w_src = None
    w_tgt = None
    if not only_prediction:
        w_src = sf.SoundFile(os.path.join(out_dir, "demo_source.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')
        w_tgt = sf.SoundFile(os.path.join(out_dir, "demo_target.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')

    # Streaming loop
    produced_model = 0
    prob_sum = 0.0
    prob_count = 0

    while produced_model < want_total_model:
        core_model = min(block_core_model, want_total_model - produced_model)

        # contexts for this block
        left_this = left_ctx_model if produced_model > 0 else 0
        right_this = right_ctx_model if (produced_model + core_model) < want_total_model else 0

        # Map model offsets → file frames
        core_start_file_off = int(round(produced_model * (file_sr / float(sample_rate))))
        core_start_file = START_FRAME + core_start_file_off

        left_ctx_file = int(math.ceil(left_this * (file_sr / float(sample_rate))))
        right_ctx_file = int(math.ceil(right_this * (file_sr / float(sample_rate))))
        core_file_len = int(math.ceil(core_model * (file_sr / float(sample_rate))))

        read_a = max(START_FRAME, core_start_file - left_ctx_file - resample_margin_file)
        read_b_desired = core_start_file + core_file_len + right_ctx_file + resample_margin_file
        read_b = min(END_FRAME, read_b_desired)
        n_read = max(0, read_b - read_a)
        if n_read <= 0:
            break

        # Read source (and target if needed) in FILE SR
        x_block_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        x_file = x_block_2d[:, 0]
        if not only_prediction:
            y_block_2d, _ = sf.read(path_target, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0]

        # Resample block to MODEL SR
        if file_sr != sample_rate:
            x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
            if not only_prediction:
                y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
        else:
            x_resamp = x_file
            if not only_prediction:
                y_resamp = y_file

        # Compute where left-context starts inside resampled array
        left_off_file = (core_start_file - left_ctx_file) - read_a
        left_off_model = int(round(left_off_file * (sample_rate / float(file_sr))))
        # Build the exact inference window: [left_ctx + core + right_ctx] in model samples
        inp_start = max(0, min(left_off_model, len(x_resamp)))
        inp_end = max(inp_start, min(inp_start + left_this + core_model + right_this, len(x_resamp)))
        x_inp = x_resamp[inp_start:inp_end]
        if x_inp.size == 0:
            break

        # Indices of "core" region inside x_inp
        core_start_in_inp = min(left_this, len(x_inp))
        core_end_in_inp = min(core_start_in_inp + core_model, len(x_inp))
        core_len = max(0, core_end_in_inp - core_start_in_inp)
        if core_len <= 0:
            break

        x_demo = x_inp[None, :, None].astype(np.float32)

        # Predicted-style pass
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)
        y_inf_pred, style_pred_out = model.predict([x_demo, dummy_style, force_zero], batch_size=1, verbose=0)
        y_inf_pred = y_inf_pred[0, :, 0]
        y_core_pred = y_inf_pred[core_start_in_inp:core_end_in_inp]
        w_pred.write(y_core_pred)

        # Accumulate style prob weighted by samples in this block's core
        prob_sum += float(style_pred_out[0, 0]) * float(core_len)
        prob_count += core_len

        # True-style pass (if known)
        if w_true is not None:
            style_true = np.array([[style_true_flag]], dtype=np.float32)
            force_one  = np.ones((1, 1), dtype=np.float32)
            y_inf_true, _ = model.predict([x_demo, style_true, force_one], batch_size=1, verbose=0)
            y_inf_true = y_inf_true[0, :, 0]
            y_core_true = y_inf_true[core_start_in_inp:core_end_in_inp]
            w_true.write(y_core_true)

        # Write source/target if requested (use the same cropping)
        if not only_prediction:
            core_start_global = inp_start + core_start_in_inp
            core_end_global = inp_start + core_end_in_inp
            x_core = x_resamp[core_start_global:core_end_global]
            y_core = y_resamp[core_start_global:core_end_global]
            w_src.write(x_core)
            w_tgt.write(y_core)

        produced_model += core_len

    # Close writers
    w_pred.close()
    if w_true is not None:
        w_true.close()
    if w_src is not None:
        w_src.close()
    if w_tgt is not None:
        w_tgt.close()

    # Final style report
    if prob_count > 0:
        p = prob_sum / float(prob_count)
        style_pred_name = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ Demo saved to {style_pred_path}. Predicted style (avg): {style_pred_name} ({p:.4f})")
        if w_true is not None:
            style_true_name = 'pick' if style_true_flag >= STYLE_FLAG_PROB_THRESHOLD else 'finger'
            print(f"✅ Demo saved to {style_true_path}. True style: {style_true_name}")
    else:
        print(f"✅ Demo saved to {style_pred_path}.")

def make_sinc_filter():
    t = np.arange(REALTIME_IR_FIR_SIZE) - (REALTIME_IR_FIR_SIZE - 1)/2.0
    h = np.sinc(2 * REALTIME_IR_CUTOFF * t)
    w = np.hamming(REALTIME_IR_FIR_SIZE)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float32)

_ir_fir     = make_sinc_filter()
h_lin = _ir_fir
h_min = minimum_phase(h_lin, method='homomorphic')
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)

# For tf.nn.conv1d (cross-correlation), use the time-reversed kernel to implement convolution
_ir_fir_kernel = h_min[::-1].reshape(-1, 1, 1).astype(np.float32)

_ir_fir_tf  = tf.constant(_ir_fir_kernel, dtype=tf.float32)  # downsampling (analysis) FIR
_ir_fir_up  = tf.constant(_ir_fir_kernel, dtype=tf.float32)  # post-LP (synthesis) FIR

# ────────────────────────────── CUSTOM LAYERS ────────────────────────────────

class KDAlphaScheduler(tf.keras.callbacks.Callback):
    """
    Linearly anneal alpha variables of KD layers and style blend over epochs.
    targets: dict of layer_name -> (start_alpha, end_alpha, start_epoch, end_epoch)
    """
    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def on_epoch_begin(self, epoch, logs=None):
        for lname, (a0, a1, e0, e1) in self.targets.items():
            layer = None
            try:
                layer = self.model.get_layer(lname)
            except:
                continue
            if layer is None:
                continue
            # progress
            if epoch <= e0:
                a = a0
            elif epoch >= e1:
                a = a1
            else:
                p = (epoch - e0) / max(1, (e1 - e0))
                a = a0 + (a1 - a0) * p
            # Set alpha if supported
            if hasattr(layer, "set_alpha"):
                layer.set_alpha(a)
            else:
                # Style blend: its "alpha" is a trainable weight[0]
                if lname == "realtime_style_blend":
                    w = layer.get_weights()
                    if w and len(w[0].shape) == 1 and w[0].shape[0] == 1:
                        w[0][0] = float(a)
                        layer.set_weights(w)

# Proper band-limited decimate-by-2 layer for 1-D sequences (applies same FIR per channel)
@register_keras_serializable()
class FIRDecimateBy2(Layer):
    def __init__(self, num_taps=127, cutoff=0.45, window="blackman", name=None, **kwargs):
        """
        Band-limited decimator-by-2. Tighter LP and more taps reduce aliasing and ripple.
        - num_taps: 127 (was 63)
        - cutoff:   0.45 (was 0.47)
        - window:   blackman (was hamming)
        """
        super().__init__(name=name, **kwargs)
        self.num_taps = int(num_taps)
        self.cutoff = float(cutoff)
        self.window = str(window)

    def build(self, input_shape):
        c = int(input_shape[-1])
        taps_np = firwin(self.num_taps, cutoff=self.cutoff, window=self.window, pass_zero=True).astype(np.float32)
        W = np.zeros((self.num_taps, c, c), dtype=np.float32)
        for ch in range(c):
            W[:, ch, ch] = taps_np
        self.kernel = self.add_weight(
            name="fir_kernel_diag",
            shape=W.shape,
            initializer=tf.keras.initializers.Constant(W),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = self.num_taps - 1
        x_pad = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
        y = tf.nn.conv1d(x_pad, filters=self.kernel, stride=2, padding='VALID')
        return y
    
    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        if t is None:
            return (b, None, c)
        out_t = t // 2
        return (b, out_t, c)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_taps": self.num_taps,
            "cutoff": self.cutoff,
            "window": self.window,
        })
        return cfg

@register_keras_serializable()
class HFHighpass(tf.keras.layers.Layer):
    def __init__(self, taps=None, sample_rate=None, cutoff_hz=HP_CUTOFF_HZ, order=OFFLINE_HF_HP_ORDER, **kwargs):
        """
        Causal min-phase FIR high-pass, implemented as conv1d with left padding.
        - If 'taps' is provided, it is used as-is (Tensor/ndarray of shape [L,1,1], already time-reversed for conv1d).
        - Otherwise, taps are generated from (sample_rate, cutoff_hz, order) via _get_minphase_hp_kernel.
        Taps are saved as non-trainable weights for TFLite export.
        """
        super().__init__(**kwargs)
        self.sample_rate = None if sample_rate is None else int(sample_rate)
        self.cutoff_hz   = float(cutoff_hz)
        self.order       = int(order)
        self._taps_init  = None if taps is None else tf.convert_to_tensor(taps, dtype=tf.float32)

    def build(self, input_shape):
        if self._taps_init is None:
            # Fallback to a known SR if none provided (should not happen if you pass sample_rate explicitly)
            sr = self.sample_rate if self.sample_rate is not None else OFFLINE_SAMPLE_RATE
            init_taps = _get_minphase_hp_kernel(sr, self.cutoff_hz, self.order)
        else:
            init_taps = self._taps_init

        self.taps = self.add_weight(
            name="hp_taps",
            shape=init_taps.shape,
            initializer=tf.keras.initializers.Constant(init_taps),
            trainable=False
        )

    def call(self, x):
        pad = tf.shape(self.taps)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "cutoff_hz":   self.cutoff_hz,
            "order":       self.order
        })
        # taps are stored as weights; no need to serialize them in config
        return cfg

@register_keras_serializable()
class RealtimeIrDownsampleFIR(Layer):
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = int(factor)
        self.kernel_size = REALTIME_IR_FIR_SIZE

    def call(self, x):
        p = x[..., 0:1]
        pad = self.kernel_size - 1
        p_pad = tf.pad(p, [[0, 0], [pad, 0], [0, 0]])
        return tf.nn.conv1d(p_pad, filters=_ir_fir_tf, stride=self.factor, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        # With left pad (L-1) and VALID conv: out_T = floor((T-1)/factor) + 1 = ceil(T/factor)
        out_T = (T + self.factor - 1) // self.factor
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'factor': self.factor})
        return cfg
    
@register_keras_serializable()
class ResizeNearest1D(tf.keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = int(factor)
    def call(self, x):
        # x: (B,T,C) -> (B,T*factor,C)
        t = tf.shape(x)[1]
        x4 = tf.expand_dims(x, axis=2)  # (B,T,1,C)
        y4 = tf.image.resize(x4, size=[t * self.factor, 1], method='nearest', antialias=False)
        return tf.squeeze(y4, axis=2)
    def compute_output_shape(self, s):
        b, t, c = s
        return (b, None if t is None else t * self.factor, c)
    def get_config(self):
        return {'factor': self.factor}

@register_keras_serializable()
class MatchTimeToRef(Layer):
    """
    Crops or pads the time axis of x to match the time length of ref.
    Always returns exactly T_ref samples (no control flow/cond needed).
    """
    def call(self, inputs):
        x, ref = inputs
        T_ref = tf.shape(ref)[1]
        pad_amt = tf.maximum(0, T_ref - tf.shape(x)[1])
        x_pad = tf.pad(x, [[0,0],[0,pad_amt],[0,0]])
        return x_pad[:, :T_ref, :]

    def compute_output_shape(self, input_shapes):
        x_shape, ref_shape = input_shapes
        return (x_shape[0], ref_shape[1], x_shape[2])
    
# ─── IRLowpass layer (unchanged API; now defaults to min‑phase taps) ────────
@register_keras_serializable()
class IRLowpass(Layer):
    def __init__(self, taps=None, **kwargs):
        super().__init__(**kwargs)
        # taps must be [L,1,1] and already reversed for conv1d (as above).
        if taps is None:
            taps = realtime_lp_taps_tf
        self._taps_init = tf.convert_to_tensor(taps, dtype=tf.float32)

    def build(self, input_shape):
        self.taps = self.add_weight(
            name="lp_taps",
            shape=self._taps_init.shape,
            initializer=tf.keras.initializers.Constant(self._taps_init),
            trainable=False,
        )

    def call(self, x):
        pad_amt = tf.shape(self.taps)[0] - 1
        x_pad = tf.pad(x, [[0, 0], [pad_amt, 0], [0, 0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class IrScale(Layer):
    def __init__(self, init=0.1, **kwargs):
        super().__init__(**kwargs)
        self.init = init
    def build(self, input_shape):
        self.scale = self.add_weight(
            name="ir_scale",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True
        )
    def call(self, x):
        return x * self.scale
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init": self.init})
        return cfg
    
@register_keras_serializable()
class BlendScalar(Layer):
    """
    Blend two scalar tensors y = (1 - alpha) * student + alpha * teacher.
    alpha in [0,1]. If trainable=False, it's a fixed convex combination.
    Inputs must be broadcast-compatible (typically shape (B,1)).
    """
    def __init__(self, init_alpha=0.5, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(init_alpha)
        self._trainable_alpha = bool(trainable)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_alpha),
            trainable=self._trainable_alpha
        )

    def call(self, inputs):
        student, teacher = inputs
        a = tf.clip_by_value(self.alpha, 0.0, 1.0)
        return (1.0 - a) * student + a * teacher

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_alpha": self.init_alpha,
            "trainable": self._trainable_alpha
        })
        return cfg

@register_keras_serializable()
class CausalMovingAverage(Layer):
    """
    Causal moving average over time with left padding of (taps-1).
    Applies a diagonal kernel → no cross-channel mixing.
    """
    def __init__(self, taps=24, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.taps = int(taps)

    def build(self, input_shape):
        c = int(input_shape[-1])
        w = np.zeros((self.taps, c, c), dtype=np.float32)
        w[:] = 0.0
        for ch in range(c):
            w[:, ch, ch] = 1.0 / float(self.taps)
        self.kernel = self.add_weight(
            name="ma_diag",
            shape=w.shape,
            initializer=tf.keras.initializers.Constant(w),
            trainable=False
        )

    def call(self, x):
        pad = self.taps - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        y = tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"taps": self.taps})
        return cfg

@register_keras_serializable()
class AdjustableAlphaMixin:
    def build_alpha(self, init_alpha, name="alpha"):
        self.alpha_var = self.add_weight(
            name=name,
            shape=(),
            initializer=tf.keras.initializers.Constant(float(init_alpha)),
            trainable=False
        )
    def set_alpha(self, new_alpha):
        self.alpha_var.assign(float(new_alpha))
    def get_config(self):
        return {}

# ───────── Training-aware KD/penalty layers (drop-in replacements) ─────────

@register_keras_serializable()
class DistillLoss(Layer, AdjustableAlphaMixin):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs, training=None):
        y_t, y_s = inputs
        if training is False:
            return y_s
        t = tf.minimum(tf.shape(y_t)[1], tf.shape(y_s)[1])
        y_t_trim = y_t[:, :t, :]
        y_s_trim = y_s[:, :t, :]
        loss = tf.reduce_mean(tf.math.squared_difference(y_t_trim, y_s_trim))
        self.add_loss(self.alpha_var * loss)
        return y_s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': float(self.init_alpha)})
        return cfg


@register_keras_serializable()
class DistillLossBestShift(Layer, AdjustableAlphaMixin):
    """
    Shift-aware MSE KD: finds best integer shift for teacher within ±max_shift samples.
    Training-only loss; returns the student unchanged.
    """
    def __init__(self, alpha, max_shift=8, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.max_shift = int(max_shift)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def _shift(self, x, s):
        T = tf.shape(x)[1]

        def shift_right():
            pad = tf.zeros_like(x[:, :s, :])
            return tf.concat([x[:, s:, :], pad], axis=1)

        def shift_left():
            s_abs = -s
            pad = tf.zeros_like(x[:, :s_abs, :])
            return tf.concat([pad, x[:, :T - s_abs, :]], axis=1)

        def no_shift():
            return x

        return tf.case(
            [(tf.greater(s, 0), shift_right),
             (tf.less(s, 0), shift_left)],
            default=no_shift
        )

    def call(self, inputs, training=None):
        y_t, y_s = inputs
        if training is False:
            return y_s

        t = tf.minimum(tf.shape(y_t)[1], tf.shape(y_s)[1])
        y_t = y_t[:, :t, :]
        y_s = y_s[:, :t, :]

        shifts = tf.range(-self.max_shift, self.max_shift + 1, dtype=tf.int32)

        def loss_for_shift(s):
            y_t_s = self._shift(y_t, s)
            return tf.reduce_mean(tf.math.squared_difference(y_t_s, y_s))

        losses = tf.map_fn(loss_for_shift, shifts, fn_output_signature=tf.float32)
        best = tf.reduce_min(losses)
        self.add_loss(self.alpha_var * best)
        return y_s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': float(self.init_alpha), 'max_shift': self.max_shift})
        return cfg


@register_keras_serializable()
class MSTFTDistillLoss(Layer, AdjustableAlphaMixin):
    """
    Multi-scale STFT magnitude L1 KD (training-only).
    """
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.scales = [
            (128,   32,  128),
            (256,   64,  256),
            (512,  128,  512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ]

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs, training=None):
        y_t, y_s = inputs
        if training is False:
            return y_s

        t = tf.minimum(tf.shape(y_t)[1], tf.shape(y_s)[1])
        t_true = tf.squeeze(y_t[:, :t, :], -1)
        t_pred = tf.squeeze(y_s[:, :t, :], -1)

        per_scale = []
        for n_fft, hop, win in self.scales:
            st_t = tf.signal.stft(
                t_true,
                frame_length=win, frame_step=hop, fft_length=n_fft,
                window_fn=tf.signal.hann_window, pad_end=True
            )
            st_p = tf.signal.stft(
                t_pred,
                frame_length=win, frame_step=hop, fft_length=n_fft,
                window_fn=tf.signal.hann_window, pad_end=True
            )
            mag_t = tf.abs(st_t)
            mag_p = tf.abs(st_p)
            per_scale.append(tf.reduce_mean(tf.abs(mag_t - mag_p)))

        loss = tf.add_n(per_scale) / float(len(self.scales))
        self.add_loss(self.alpha_var * loss)
        return y_s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': float(self.init_alpha)})
        return cfg


def _make_lpf_kernel(sample_rate, cutoff_hz=2800.0, order=129, window="hamming"):
    taps = firwin(order, cutoff_hz / (sample_rate / 2.0), window=window, pass_zero=True).astype(np.float32)
    taps = taps / np.sum(taps + 1e-12)
    # reverse for tf.nn.conv1d cross-correlation
    return tf.constant(taps[::-1].reshape(-1, 1, 1), dtype=tf.float32)

@register_keras_serializable()
class HFLowbandPenalty(Layer, AdjustableAlphaMixin):
    """
    Penalize low-band energy in the student's HF branch: L1(LPF(hf_final)).
    Returns input unchanged; training-only constraint.
    """
    def __init__(self, alpha, cutoff_hz=2800.0, order=129, sample_rate=REALTIME_SAMPLE_RATE, window="hamming", **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self.sample_rate = int(sample_rate)
        self.window = str(window)

    def build(self, input_shape):
        k = _make_lpf_kernel(self.sample_rate, self.cutoff_hz, self.order, self.window)
        self.kernel = self.add_weight(
            name="lp_taps_for_penalty",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, x, training=None):
        # Only active when training is explicitly True.
        if training is not True:
            return x
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        lp = tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')
        loss = tf.reduce_mean(tf.abs(lp))
        self.add_loss(self.alpha_var * loss)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'alpha': float(self.init_alpha),
            'cutoff_hz': self.cutoff_hz,
            'order': self.order,
            'sample_rate': self.sample_rate,
            'window': self.window
        })
        return cfg

@register_keras_serializable()
class TileStyle(Layer):
    def call(self, inputs):
        style_scalar, ref = inputs
        # ref has shape (B, T, C); we need to tile a (B,1,1) scalar across time T
        T = tf.shape(ref)[1]
        # reshape with -1 for the (possibly dynamic) batch dim
        s = tf.reshape(style_scalar, (-1, 1, 1))
        multiples = tf.stack([1, T, 1])  # build a Tensor, not a Python tuple with a Tensor inside
        return tf.tile(s, multiples)

    def compute_output_shape(self, input_shapes):
        # (batch, time, channels) -> channels=1 for style mask
        return (input_shapes[1][0], input_shapes[1][1], 1)

@register_keras_serializable()
class InvertStyle(Layer):
    def call(self, style_time):
        return 1.0 - style_time
    def compute_output_shape(self, input_shape):
        return input_shape

# ─── Gate sidechain HPF (min-phase, causal) cache ────────────────────────────
_gate_hp_kernel_cache = {}

def _get_minphase_hp_kernel(sample_rate, cutoff_hz, order=129):
    key = (int(sample_rate), float(cutoff_hz), int(order))
    k_np = _gate_hp_kernel_cache.get(key)
    if k_np is None:
        taps = firwin(
        order,
        cutoff=cutoff_hz / (sample_rate / 2.0),
        window="hamming",
        pass_zero=False
        ).astype(np.float32)
        taps_min = minimum_phase(taps, method='homomorphic').astype(np.float32)
        k_np = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        _gate_hp_kernel_cache[key] = k_np
    # Always materialize a new constant in the current graph.
    return tf.constant(k_np, dtype=tf.float32)

def _causal_conv1d(x, kernel):
    pad = tf.shape(kernel)[0] - 1
    x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
    return tf.nn.conv1d(x_pad, filters=kernel, stride=1, padding='VALID')





# ─── Gate smoothing kernels (causal FIR) ─────────────────────────────────────
_gate_smooth_cache = {}  # (tau_samples, scale, Lmax) -> kernel np.array
_gate_ones_cache   = {}  # (L,) -> kernel np.array

@register_keras_serializable()
def gate_onset(g):
    # g: (B,T,1) ∈ [0,1]; return rising edges as positive envelope
    g_prev = tf.pad(g, [[0,0],[1,0],[0,0]])[:, :-1, :]
    return tf.nn.relu(g - g_prev)

def _make_modal_kernels(centers_hz, decays_ms, length, sample_rate):
    L = int(length)
    t = np.arange(L, dtype=np.float32)
    kernels = []
    for f, d_ms in zip(centers_hz, decays_ms):
        tau_s = max(1.0, (float(d_ms) / 1000.0) * float(sample_rate))
        env = np.exp(-t / tau_s).astype(np.float32)
        s   = np.sin(2.0 * np.pi * float(f) * t / float(sample_rate)).astype(np.float32)
        h = env * s
        h /= np.maximum(np.sqrt(np.sum(h*h, dtype=np.float32)), 1e-6)
        kernels.append(h)
    K = np.stack(kernels, axis=1)      # (L, M)
    K = K[::-1, :]                     # reverse for conv1d (cross-correlation → convolution)
    return K.reshape(L, 1, -1).astype(np.float32)

@register_keras_serializable()
class ModalBank(tf.keras.layers.Layer):
    """
    Fixed bank of causal decaying-sine FIRs (one output channel per mode).
    - Inputs: (B,T,1), outputs: (B,T,M)
    - Implementation: left-pad by (L-1), tf.nn.conv1d with fixed kernels.
    """
    def __init__(self, centers_hz, decays_ms, length, sample_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.centers_hz = [float(x) for x in centers_hz]
        self.decays_ms  = [float(x) for x in decays_ms]
        self.length     = int(length)
        self.sample_rate= int(sample_rate)

    def build(self, input_shape):
        K = _make_modal_kernels(self.centers_hz, self.decays_ms, self.length, self.sample_rate)
        self.kernel = self.add_weight(
            name="modal_kernels",
            shape=K.shape,
            initializer=tf.keras.initializers.Constant(K),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = self.length - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        return (b, t, len(self.centers_hz))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "centers_hz": self.centers_hz,
            "decays_ms":  self.decays_ms,
            "length":     self.length,
            "sample_rate":self.sample_rate
        })
        return cfg

@register_keras_serializable()
def strike_boost(x, gain=2.5):
    # x: (B,T,1), returns (B,T,1) = 1 + gain*x
    return 1.0 + tf.constant(float(gain), tf.float32) * x

def _exp_smoothing_kernel(samples, scale=4.0, Lmax=2048):
    """
    Causal exponential FIR approximation of an EMA with time-constant 'samples'.
    Kernel length L ≈ scale * samples, clipped to [8, Lmax].
    Sum is normalized to 1. Returned reversed (for conv1d cross-correlation).
    """
    s = max(1, int(samples))
    L = int(np.clip(np.ceil(scale * s), 8, Lmax))
    key = (s, float(scale), int(Lmax))
    k_np = _gate_smooth_cache.get(key)
    if k_np is None:
        a = np.exp(-1.0 / float(s))
        taps = (1.0 - a) * (a ** np.arange(L, dtype=np.float32))
        taps = taps / np.sum(taps)  # normalize exactly
        k_np = taps[::-1].reshape(-1, 1, 1).astype(np.float32)  # reverse for tf.nn.conv1d
        _gate_smooth_cache[key] = k_np
    return tf.constant(k_np, dtype=tf.float32)

def _ones_kernel(length):
    """
    Causal moving-sum kernel of length 'length' (reversed for conv1d).
    """
    L = max(1, int(length))
    k_np = _gate_ones_cache.get(L)
    if k_np is None:
        taps = np.ones((L,), dtype=np.float32)
        k_np = taps[::-1].reshape(-1, 1, 1).astype(np.float32)
        _gate_ones_cache[L] = k_np
    return tf.constant(k_np, dtype=tf.float32)

@register_keras_serializable()
def gate_gain_signal(
    x, sample_rate,
    threshold_db,
    ratio,
    attack_ms,
    hold_ms,
    release_ms,
    cutoff_hz,
    hp_order=129,
    min_gain_db=-60.0,
    env_ms=None,
    depth_smooth_ms=None,
    # NEW: allow per-call Lmax
    env_lmax=None,
    depth_lmax=None
):
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    kernel_hp = _get_minphase_hp_kernel(sample_rate, cutoff_hz, hp_order)
    sc = _causal_conv1d(x, kernel_hp)

    det_raw = tf.abs(sc) + 1e-12
    if env_ms is not None and env_ms > 0.0:
        env_smpl = max(1, int(round((float(env_ms) / 1000.0) * float(sample_rate))))
        k_env = _exp_smoothing_kernel(env_smpl, scale=2.0, Lmax=(env_lmax or 512))
        det = _causal_conv1d(det_raw, k_env)
    else:
        det = det_raw

    LN10 = tf.math.log(tf.constant(10.0, dtype=tf.float32))
    level_db = 20.0 / LN10 * tf.math.log(det)

    atk_s = max(1, int(round((float(attack_ms)  / 1000.0) * float(sample_rate))))
    rel_s = max(1, int(round((float(release_ms) / 1000.0) * float(sample_rate))))
    hld_s = max(0, int(round((float(hold_ms)    / 1000.0) * float(sample_rate))))

    thr_db  = tf.constant(float(threshold_db), dtype=tf.float32)
    ratio_f = tf.constant(float(ratio),        dtype=tf.float32)

    above = tf.cast(level_db >= thr_db, tf.float32)
    if hld_s > 0:
        k_hold = _ones_kernel(hld_s)
        pad = hld_s - 1
        above_pad = tf.pad(above, [[0,0],[pad,0],[0,0]])
        hold_sum = tf.nn.conv1d(above_pad, filters=k_hold, stride=1, padding='VALID')
        open_or_hold = tf.cast(hold_sum > 0.0, tf.float32)
    else:
        open_or_hold = above

    delta_db = tf.nn.relu(thr_db - level_db[...,0])
    attn_db  = tf.minimum(ratio_f * delta_db, tf.constant(120.0, tf.float32))
    ALPHA = tf.constant(np.log(10.0)/20.0, dtype=tf.float32)
    g_close_raw = tf.exp(-ALPHA * attn_db)[:, :, None]
    if min_gain_db is not None:
        floor = tf.constant(10.0 ** (float(min_gain_db)/20.0), dtype=tf.float32)
        g_close_raw = tf.maximum(g_close_raw, floor)

    if depth_smooth_ms is not None and depth_smooth_ms > 0.0:
        d_smpl = max(1, int(round((float(depth_smooth_ms) / 1000.0) * float(sample_rate))))
        k_depth = _exp_smoothing_kernel(d_smpl, scale=2.0, Lmax=(depth_lmax or 512))
        g_close_s = _causal_conv1d(g_close_raw, k_depth)
    else:
        g_close_s = g_close_raw

    k_att = _exp_smoothing_kernel(atk_s, scale=4.0, Lmax=2048)
    k_rel = _exp_smoothing_kernel(rel_s, scale=4.0, Lmax=2048)
    g_att_open = _causal_conv1d(open_or_hold, k_att)
    g_rel_open = _causal_conv1d(open_or_hold, k_rel)

    g_prev_open = tf.pad(open_or_hold, [[0,0],[1,0],[0,0]])[:, :-1, :]
    inc_open = tf.cast(open_or_hold > g_prev_open, tf.float32)
    g_env = inc_open * g_att_open + (1.0 - inc_open) * g_rel_open
    g_env = tf.clip_by_value(g_env, 0.0, 1.0)

    g = g_env + (1.0 - g_env) * g_close_s
    g = tf.clip_by_value(g, 0.0, 1.0)
    return g


@register_keras_serializable()
class Gate(tf.keras.layers.Layer):
    """
    Gate with HPF sidechain and downward expander, returning per-sample gain [0,1].
    Now supports per-instance env_ms/depth_smooth_ms and per-instance Lmax caps.
    """
    def __init__(self, sample_rate,
                 threshold_db,
                 ratio,
                 attack_ms,
                 hold_ms,
                 release_ms,
                 cutoff_hz,
                 hp_order,
                 env_ms=OFFLINE_GATE_ENV_MS,
                 depth_smooth_ms=OFFLINE_GATE_DEPTH_SMOOTH_MS,
                 env_lmax=None,
                 depth_lmax=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.hold_ms = float(hold_ms)
        self.release_ms = float(release_ms)
        self.cutoff_hz = float(cutoff_hz)
        self.hp_order = int(hp_order)
        self.env_ms = None if env_ms is None else float(env_ms)
        self.depth_smooth_ms = None if depth_smooth_ms is None else float(depth_smooth_ms)
        self.env_lmax = None if env_lmax is None else int(env_lmax)
        self.depth_lmax = None if depth_lmax is None else int(depth_lmax)

    def call(self, x):
        return gate_gain_signal(
            x,
            sample_rate=self.sample_rate,
            threshold_db=self.threshold_db,
            ratio=self.ratio,
            attack_ms=self.attack_ms,
            hold_ms=self.hold_ms,
            release_ms=self.release_ms,
            cutoff_hz=self.cutoff_hz,
            hp_order=self.hp_order,
            env_ms=self.env_ms,
            depth_smooth_ms=self.depth_smooth_ms,
            env_lmax=self.env_lmax,
            depth_lmax=self.depth_lmax
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "hold_ms": self.hold_ms,
            "release_ms": self.release_ms,
            "cutoff_hz": self.cutoff_hz,
            "hp_order": self.hp_order,
            "env_ms": self.env_ms,
            "depth_smooth_ms": self.depth_smooth_ms,
            "env_lmax": self.env_lmax,
            "depth_lmax": self.depth_lmax
        })
        return cfg

@register_keras_serializable()
class GateInt8Linear(tf.keras.layers.Layer):
    """
    INT8/TFLM-friendly gate with the same semantics as the original dB-domain Gate:

      - Sidechain: HPF (causal min-phase) → rectifier → optional env smoothing
      - above = (env >= thr_lin) with thr_lin = 10^(threshold_db/20)
      - Hold: if hold_ms>0, keep 'open' for a causal window using ones FIR (like original)
      - Attack/Release smoothing on open_or_hold (identical mixer logic)
      - Close depth: g_close_raw = min(1, (env/thr_lin)^ratio) with integer-power via repeated squaring
      - Floor: max(min_gain_db, 10^(-120/20)) ≈ max(min_gain, 1e-6)
      - Depth smoothing (optional)
      - Final: g = g_env + (1 - g_env) * g_close_s, clipped to [0,1]

    All ops are add/mul/abs/conv/compare/cast/clip — no log/exp.
    """
    def __init__(self, sample_rate,
                 threshold_db,
                 ratio,
                 attack_ms,
                 hold_ms,
                 release_ms,
                 cutoff_hz,
                 hp_order,
                 env_ms=None,
                 depth_smooth_ms=None,
                 env_lmax=None,
                 depth_lmax=None,
                 min_gain_db=-60.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = int(sample_rate)
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.hold_ms = float(hold_ms)
        self.release_ms = float(release_ms)
        self.cutoff_hz = float(cutoff_hz)
        self.hp_order = int(hp_order)
        self.env_ms = None if env_ms is None else float(env_ms)
        self.depth_smooth_ms = None if depth_smooth_ms is None else float(depth_smooth_ms)
        self.env_lmax = None if env_lmax is None else int(env_lmax)
        self.depth_lmax = None if depth_lmax is None else int(depth_lmax)

        # Precompute linear threshold/floors
        self.thr_lin = float(10.0 ** (self.threshold_db / 20.0))
        self.min_gain = float(10.0 ** (float(min_gain_db) / 20.0))
        self.min_gain_120db = float(10.0 ** (-120.0 / 20.0))  # -120 dB floor from original clamp

        # Integer power exponent (approx the ratio, as in dB domain)
        self.ratio_int = int(max(1, round(self.ratio)))

        # Kernels (set in build)
        self.hp_kernel = None
        self.k_env = None
        self.k_att = None
        self.k_rel = None
        self.k_depth = None
        self.k_hold = None
        self.pad_hold = 0

    def build(self, input_shape):
        # HPF sidechain
        hp_k = _get_minphase_hp_kernel(self.sample_rate, self.cutoff_hz, self.hp_order)
        self.hp_kernel = self.add_weight(
            name="hp_taps_linear",
            shape=hp_k.shape,
            initializer=tf.keras.initializers.Constant(hp_k),
            trainable=False
        )

        # Optional envelope smoothing
        if self.env_ms is not None and self.env_ms > 0.0:
            env_smpl = max(1, int(round((self.env_ms / 1000.0) * self.sample_rate)))
            k_env = _exp_smoothing_kernel(env_smpl, scale=2.0, Lmax=(self.env_lmax or 512))
            self.k_env = self.add_weight(
                name="env_smooth_taps",
                shape=k_env.shape,
                initializer=tf.keras.initializers.Constant(k_env),
                trainable=False
            )

        # Attack/Release smoothing
        atk_s = max(1, int(round((self.attack_ms  / 1000.0) * self.sample_rate)))
        rel_s = max(1, int(round((self.release_ms / 1000.0) * self.sample_rate)))
        k_att = _exp_smoothing_kernel(atk_s, scale=4.0, Lmax=2048)
        k_rel = _exp_smoothing_kernel(rel_s, scale=4.0, Lmax=2048)
        self.k_att = self.add_weight(
            name="open_attack_taps",
            shape=k_att.shape,
            initializer=tf.keras.initializers.Constant(k_att),
            trainable=False
        )
        self.k_rel = self.add_weight(
            name="open_release_taps",
            shape=k_rel.shape,
            initializer=tf.keras.initializers.Constant(k_rel),
            trainable=False
        )

        # Optional depth smoothing of close gain
        if self.depth_smooth_ms is not None and self.depth_smooth_ms > 0.0:
            d_smpl = max(1, int(round((self.depth_smooth_ms / 1000.0) * self.sample_rate)))
            k_depth = _exp_smoothing_kernel(d_smpl, scale=2.0, Lmax=(self.depth_lmax or 512))
            self.k_depth = self.add_weight(
                name="depth_smooth_taps",
                shape=k_depth.shape,
                initializer=tf.keras.initializers.Constant(k_depth),
                trainable=False
            )

        # Hold kernel (like original)
        if self.hold_ms > 0.0:
            hld_s = max(1, int(round((self.hold_ms / 1000.0) * self.sample_rate)))
            k_hold = _ones_kernel(hld_s)
            self.k_hold = self.add_weight(
                name="hold_ones_taps",
                shape=k_hold.shape,
                initializer=tf.keras.initializers.Constant(k_hold),
                trainable=False
            )
            self.pad_hold = int(hld_s - 1)
        else:
            self.k_hold = None
            self.pad_hold = 0

        super().build(input_shape)

    def _powi_ratio(self, x):
        """
        Compute x^ratio_int by repeated squaring (ratio_int is a small positive int).
        Uses only muls; no log/exp/pow ops.
        """
        n = int(self.ratio_int)
        # Early outs
        if n <= 1:
            return x

        result = tf.ones_like(x)
        base = x
        m = n
        while m > 0:
            if (m & 1) == 1:
                result = result * base
            base = base * base
            m >>= 1
        return result

    def call(self, x):
        thr = tf.constant(self.thr_lin, tf.float32)

        # Sidechain env
        sc = _causal_conv1d(x, self.hp_kernel)
        env = tf.abs(sc) + 1e-12
        if self.k_env is not None:
            env = _causal_conv1d(env, self.k_env)

        # above and optional hold
        above = tf.cast(tf.greater_equal(env, thr), tf.float32)  # 0/1 float
        if self.k_hold is not None:
            above_pad = tf.pad(above, [[0, 0], [self.pad_hold, 0], [0, 0]])
            hold_sum = tf.nn.conv1d(above_pad, filters=self.k_hold, stride=1, padding='VALID')
            open_or_hold = tf.cast(tf.greater(hold_sum, 0.0), tf.float32)  # 0/1 exactly like original
        else:
            open_or_hold = above

        # Attack/Release smoothing on open_or_hold
        g_prev_open = tf.pad(open_or_hold, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        inc_open = tf.cast(tf.greater(open_or_hold, g_prev_open), tf.float32)
        g_att_open = _causal_conv1d(open_or_hold, self.k_att)
        g_rel_open = _causal_conv1d(open_or_hold, self.k_rel)
        g_env = inc_open * g_att_open + (1.0 - inc_open) * g_rel_open
        g_env = tf.clip_by_value(g_env, 0.0, 1.0)

        # Close gain in linear domain: (env/thr)^ratio, clamped to [floor, 1]
        env_over_thr = env / thr
        g_close_raw = self._powi_ratio(env_over_thr)
        g_close_raw = tf.minimum(g_close_raw, 1.0)

        floor = tf.constant(max(self.min_gain, self.min_gain_120db), tf.float32)
        g_close_raw = tf.maximum(g_close_raw, floor)

        # Optional depth smoothing
        if self.k_depth is not None:
            g_close_s = _causal_conv1d(g_close_raw, self.k_depth)
            g_close_s = tf.clip_by_value(g_close_s, floor, 1.0)
        else:
            g_close_s = g_close_raw

        # Final gain (0..1)
        g = g_env + (1.0 - g_env) * g_close_s
        return tf.clip_by_value(g, 0.0, 1.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "hold_ms": self.hold_ms,
            "release_ms": self.release_ms,
            "cutoff_hz": self.cutoff_hz,
            "hp_order": self.hp_order,
            "env_ms": self.env_ms,
            "depth_smooth_ms": self.depth_smooth_ms,
            "env_lmax": self.env_lmax,
            "depth_lmax": self.depth_lmax,
            "min_gain_db": 20.0 * np.log10(max(self.min_gain, 1e-12)),
            "ratio_int": self.ratio_int
        })
        return cfg

@register_keras_serializable()
def gate_weight_offline(y_true):
    return gate_gain_signal(
        y_true,
        sample_rate=OFFLINE_SAMPLE_RATE,
        threshold_db=OFFLINE_GATE_THRESHOLD_DB,
        ratio=OFFLINE_GATE_RATIO,
        attack_ms=OFFLINE_GATE_ATTACK_MS,
        hold_ms=OFFLINE_GATE_HOLD_MS,
        release_ms=OFFLINE_GATE_RELEASE_MS,
        cutoff_hz=OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ
    )

@register_keras_serializable()
def gate_weight_realtime(y_true):
    return gate_gain_signal(
        y_true,
        sample_rate=REALTIME_SAMPLE_RATE,
        threshold_db=REALTIME_GATE_THRESHOLD_DB,
        ratio=REALTIME_GATE_RATIO,
        attack_ms=REALTIME_GATE_ATTACK_MS,
        hold_ms=REALTIME_GATE_HOLD_MS,
        release_ms=REALTIME_GATE_RELEASE_MS,
        cutoff_hz=REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ
    )

@register_keras_serializable()
def gate_weight_dynamic(y_true, sample_rate):
    sr = int(sample_rate)
    if sr == OFFLINE_SAMPLE_RATE:
        return gate_weight_offline(y_true)
    elif sr == REALTIME_SAMPLE_RATE:
        return gate_weight_realtime(y_true)
    else:
        # default to realtime params if an unknown SR is used
        return gate_weight_realtime(y_true)

# Slice helpers
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

@register_keras_serializable()
def slice_film_gamma(x, filters):
    return x[..., :filters]

@register_keras_serializable()
def slice_film_beta(x, filters):
    return x[..., filters:]

@register_keras_serializable()
def invert(x):
    return 1.0 - x

# Stabilized FiLM
@register_keras_serializable()
def film_gamma_stable(raw):
    return 1.0 + 0.1 * tf.tanh(raw)

@register_keras_serializable()
def film_beta_stable(raw):
    return 0.1 * tf.tanh(raw)

def compute_pads(kernel, dilation, lookahead_samples):
    P_tot = (kernel - 1) * dilation
    pad_right = min(int(lookahead_samples), P_tot)
    pad_left  = P_tot - pad_right
    return pad_left, pad_right

def gated_sep_block(x, filters, kernel, dilation, prefix, lookahead=0):
    res = x
    pad_left, pad_right = compute_pads(kernel=kernel, dilation=dilation, lookahead_samples=lookahead)
    x   = ZeroPadding1D((pad_left, pad_right), name=f"{prefix}_pad")(x)
    g   = SeparableConv1D(2*filters, kernel, dilation_rate=dilation, padding="valid", name=f"{prefix}_sep")(x)
    c   = Lambda(slice_content, arguments={"filters": filters}, name=f"{prefix}_slice_c")(g)
    g2  = Lambda(slice_gate,    arguments={"filters": filters}, name=f"{prefix}_slice_g")(g)
    x   = Multiply(name=f"{prefix}_gate_mul")([Activation("tanh", name=f"{prefix}_tanh")(c),
                                               Activation("sigmoid", name=f"{prefix}_sigmoid")(g2)])
    x   = Conv1D(filters, 1, padding="same", name=f"{prefix}_pw")(x)
    x   = Add(name=f"{prefix}_res_add")([x, res])
    return x

def film_apply(x, style_embed, filters, prefix):
    film_params = Dense(filters*2, name=f"{prefix}_film_dense")(style_embed)
    gamma_raw = Lambda(slice_film_gamma, arguments={'filters': filters}, name=f"{prefix}_film_gamma_raw")(film_params)
    beta_raw  = Lambda(slice_film_beta,  arguments={'filters': filters}, name=f"{prefix}_film_beta_raw")(film_params)
    gamma = Lambda(film_gamma_stable, name=f"{prefix}_film_gamma")(gamma_raw)
    beta  = Lambda(film_beta_stable,  name=f"{prefix}_film_beta")(beta_raw)
    
    
    # Broadcast FiLM parameters across time: [B, F] -> [B, 1, F]
    gamma3 = Reshape((1, filters), name=f"{prefix}_film_gamma_expand")(gamma)
    beta3  = Reshape((1, filters), name=f"{prefix}_film_beta_expand")(beta)
    return Add(name=f"{prefix}_film_apply")([
        Multiply(name=f"{prefix}_film_scale")([x, gamma3]),
        beta3
    ])

def _add_resonance_branch_offline(inp, style_embed, gate):
    if not OFFLINE_RESO_ENABLE:
        return None
    # 1) strike (onset) envelope
    strike = Lambda(gate_onset, name="offline_strike_onrise")(gate)
    strike = CausalMovingAverage(taps=96, name="offline_strike_smooth")(strike)  # ~1 ms at 96k

    # 2) excitation (HP + gate + mild nonlinearity)
    exc = Lambda(pre_emphasis, arguments={'alpha': 0.95}, name="offline_reso_pre_emph")(inp)
    exc = Multiply(name="offline_reso_exc_gate")([exc, gate])
    exc = Activation('tanh', name="offline_reso_drive")(exc)

    # 3) modal bank
    modes = ModalBank(
        centers_hz=OFFLINE_RESO_CENTERS_HZ,
        decays_ms=OFFLINE_RESO_DECAYS_MS,
        length=OFFLINE_RESO_LENGTH,
        sample_rate=OFFLINE_SAMPLE_RATE,
        name="offline_reso_modal"
    )(exc)  # (B,T,M)

    # 4) per-mode gains (style) × (strike boost)
    gains = Dense(OFFLINE_RESO_MODES, activation="softplus", name="offline_reso_gains")(style_embed)   # (B,M)
    gains = Reshape((1, OFFLINE_RESO_MODES), name="offline_reso_gains_expand")(gains)                  # (B,1,M)
    sboost= Lambda(strike_boost, arguments={'gain': OFFLINE_RESO_STRIKE_GAIN}, name="offline_strike_boost")(strike)  # (B,T,1)
    gains_t = Multiply(name="offline_reso_gains_time")([gains, sboost])                                # broadcast to (B,T,M)

    m_scaled = Multiply(name="offline_reso_modes_scaled")([modes, gains_t])
    reso = Conv1D(1, 1, padding="same", name="offline_reso_mix")(m_scaled)
    reso = IrScale(init=1.0, name="offline_reso_scale")(reso)
    return reso

def _add_resonance_branch_realtime(inp, style_embed, gate):
    if not REALTIME_RESO_ENABLE:
        return None
    # 1) strike env
    strike = Lambda(gate_onset, name="realtime_strike_onrise")(gate)
    strike = CausalMovingAverage(taps=48, name="realtime_strike_smooth")(strike)  # ~1 ms at 48k

    # 2) excitation
    exc = Lambda(pre_emphasis, arguments={'alpha': 0.95}, name="realtime_reso_pre_emph")(inp)
    exc = Multiply(name="realtime_reso_exc_gate")([exc, gate])
    exc = Activation('tanh', name="realtime_reso_drive")(exc)

    # 3) modal bank
    modes = ModalBank(
        centers_hz=REALTIME_RESO_CENTERS_HZ,
        decays_ms=REALTIME_RESO_DECAYS_MS,
        length=REALTIME_RESO_LENGTH,
        sample_rate=REALTIME_SAMPLE_RATE,
        name="realtime_reso_modal"
    )(exc)

    # 4) per-mode gains (style) × (strike boost)
    gains = Dense(REALTIME_RESO_MODES, activation="relu", name="realtime_reso_gains")(style_embed)
    gains = Reshape((1, REALTIME_RESO_MODES), name="realtime_reso_gains_expand")(gains)
    sboost= Lambda(strike_boost, arguments={'gain': REALTIME_RESO_STRIKE_GAIN}, name="realtime_strike_boost")(strike)
    gains_t = Multiply(name="realtime_reso_gains_time")([gains, sboost])

    m_scaled = Multiply(name="realtime_reso_modes_scaled")([modes, gains_t])
    reso = Conv1D(1, 1, padding="same", name="realtime_reso_mix")(m_scaled)
    reso = IrScale(init=1.0, name="realtime_reso_scale")(reso)
    return reso

# ────────────────────────────── MODEL BUILDERS ───────────────────────────────

@register_keras_serializable()
def build_offline_model_causal(offline_src, dynamic_time=True):
    """
    Build a causalized offline model (right-lookahead=0) and copy weights by name
    from the provided non-causal offline model. Returned model is frozen.
    """
    offline_causal = build_offline_model(dynamic_time=dynamic_time, lookahead_samples=0, name="piezo2mic_offline_causal")
    copy_weights_by_name(offline_causal, offline_src, verbose=True)
    for l in offline_causal.layers:
        l.trainable = False
    return offline_causal

@register_keras_serializable()
def build_offline_model(dynamic_time=False, lookahead_samples=OFFLINE_LOOKAHEAD_SAMPLES, name="piezo2mic_offline"):
    T   = None if dynamic_time else OFFLINE_SEG_LEN
    inp = Input(shape=(T,1), name="offline_piezo_input")
        
    # The known, true style of the input (may or may not be known)
    style_true_in  = Input(shape=(1,), name="offline_style_true_input")
    # Whether to use the style prediction or the true style for training
    style_force_in = Input(shape=(1,), name="offline_style_force_input")  # 0=predict, 1=ground-truth
    
    # Style head
    cls         = Conv1D(16,3, padding="same", activation="relu", name="offline_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="offline_style_gap")(cls)
    style_logit = Dense(1, name="offline_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="offline_style_prob")(style_logit)
    
    one_minus_force = Lambda(invert, name="offline_one_minus_force")(style_force_in)
    # style_used_scalar = (1 - force)pred + forcetrue  → shape (B,1)
    style_used_scalar = Add(name="offline_style_used_scalar")([
        Multiply(name="offline_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="offline_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time_used = TileStyle(name="offline_style_time_used")([style_used_scalar, inp])
    style_embed = Dense(OFFLINE_EMBED_DIM, activation="linear", name="offline_style_embed")(style_used_scalar)

    x = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_init_pw")(inp)

    # Pre-blocks with configurable look-ahead
    for i in range(OFFLINE_PRE_BLOCKS):
        x = gated_sep_block(x, OFFLINE_FILTERS, OFFLINE_PRE_KERNEL, 1, prefix=f"offline_pre_{i}", lookahead=lookahead_samples)
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_pre_{i}")

    # Dilated stack with configurable look-ahead
    for idx, d in enumerate(OFFLINE_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=OFFLINE_KERNEL_SIZE, dilation=d, lookahead_samples=lookahead_samples)
        x = ZeroPadding1D((pad_left, pad_right), name=f"pad_{idx}")(x)
        g = SeparableConv1D(OFFLINE_FILTERS*2, OFFLINE_KERNEL_SIZE, dilation_rate=d, padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':OFFLINE_FILTERS}, name=f"slice_content_{idx}")(g)
        g2 = Lambda(slice_gate, arguments={'filters':OFFLINE_FILTERS}, name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([Activation('tanh', name=f"tanh_{idx}")(c),
                                               Activation('sigmoid', name=f"sigmoid_{idx}")(g2)])
        x = Conv1D(OFFLINE_FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_dil_{idx}")
        
    # Final pick/finger outputs
    main_pick   = Conv1D(1,1,padding="same",name="offline_final_pick")(x)
    main_finger = Conv1D(1,1,padding="same",name="offline_final_finger")(x)
 
    inv_style  = InvertStyle(name="offline_invert_style")(style_time_used)
    pick_c     = Multiply(name="offline_pick_contrib")([style_time_used, main_pick])
    finger_c   = Multiply(name="offline_finger_contrib")([inv_style,     main_finger])
    
    # ─── Gate gain (0..1) computed from the input ────────────────────────────
    offline_gate = Gate(
        sample_rate=OFFLINE_SAMPLE_RATE,
        threshold_db=OFFLINE_GATE_THRESHOLD_DB,
        ratio=OFFLINE_GATE_RATIO,
        attack_ms=OFFLINE_GATE_ATTACK_MS,
        hold_ms=OFFLINE_GATE_HOLD_MS,
        release_ms=OFFLINE_GATE_RELEASE_MS,
        cutoff_hz=OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ,
        depth_smooth_ms=OFFLINE_GATE_DEPTH_SMOOTH_MS,
        env_ms=OFFLINE_GATE_ENV_MS,
        hp_order=OFFLINE_HF_HP_ORDER,
        env_lmax=None,
        depth_lmax=None,
        name="offline_gate"
    )(inp)

    # HF branch (same topology), but gate BEFORE the high-pass
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(inp)
    hf    = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_hf_init")(hf_in)
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b0", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b0")
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b1", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b1")
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=3, dilation=1, prefix="offline_hf_b2", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b2")
    hf    = Conv1D(1,1,padding="same", name="offline_hf_out")(hf)

    # Move the gate multiply BEFORE the HPF
    hf_gated = Multiply(name="offline_hf_mult_gate")([hf, offline_gate])
    
    hf_hp    = HFHighpass(sample_rate=OFFLINE_SAMPLE_RATE,
                      cutoff_hz=HP_CUTOFF_HZ,
                      order=OFFLINE_HF_HP_ORDER,
                      name="offline_hf_highpass")(hf_gated)
    
    hf_final = IrScale(init=8.0, name="offline_hf_scale")(hf_hp)
    
    offline_reso = _add_resonance_branch_offline(inp=inp, style_embed=style_embed, gate=offline_gate)

    # Final sum
    if offline_reso is None:
        out = Add(name="offline_mix_main")([pick_c, finger_c, hf_final])
    else:
        out = Add(name="offline_mix_main")([pick_c, finger_c, hf_final, offline_reso])

    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in], outputs=[out, style_prob], name=name)

def summarize_offline_hf_usage(offline_dyn, src_path, tgt_path, sample_rate, style_true_flag=1.0):
    """
    Summarize how the offline model uses its HF branch with the new Gate.
    - Prints gate stats and HF branch RMS on high/low gate regions
    - Reports high-band STFT energy above GATE_HP_CUTOFF_HZ for target vs prediction

    Args:
        offline_dyn: dynamic offline model (built with build_offline_model(dynamic_time=True))
        src_path: path to source (piezo) file
        tgt_path: path to target (mic) file
        sample_rate: model SR (should be OFFLINE_SAMPLE_RATE)
        style_true_flag: 1.0 for pick / 0.0 for finger (forced)

    Requirements:
        - Model must expose layers named:
            "offline_gate" (per-sample gain 0..1),
            "offline_hf_out" (HF branch pre-HPF),
            "offline_hf_mult_gate" (HF high-passed and multiplied by gate),
            "offline_hf_scale" (scaled HF branch that gets mixed),
            "offline_pick_contrib", "offline_finger_contrib", "offline_mix_main"
        - Constants available: GATE_HP_CUTOFF_HZ, HP_PRE_EMPHASIS_HF
        - Utility available: resample_to_model_sr
    """

    def _build_offline_debug_model(offline):
        outs = {
            "gate":      offline.get_layer("offline_gate").output,            # gate gain (0..1)
            "hf_pre":    offline.get_layer("offline_hf_out").output,          # HF branch before HPF
            "hf_gated":  offline.get_layer("offline_hf_mult_gate").output,    # post-HPF × gate, pre-scale
            "hf_scaled": offline.get_layer("offline_hf_scale").output,        # final HF branch to mix
            "pick_c":    offline.get_layer("offline_pick_contrib").output,
            "finger_c":  offline.get_layer("offline_finger_contrib").output,
            "y":         offline.get_layer("offline_mix_main").output,        # final mix
        }
        return tf.keras.Model(inputs=offline.inputs, outputs=outs, name="offline_debug_model")

    def _hp(x, alpha=HP_PRE_EMPHASIS_HF):
        # pre-emphasis HP similar to your hf_loss
        x1 = x[:, 1:, :] - alpha * x[:, :-1, :]
        return tf.pad(x1, [[0, 0], [1, 0], [0, 0]])

    def _band_energy(x, sr, fmin):
        # crude HF energy via STFT mask above fmin
        t = tf.squeeze(x, -1)  # (B,T)
        n_fft = 1024
        hop   = 256
        win   = 1024
        st = tf.signal.stft(t, frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
        mag = tf.abs(st)  # (B, frames, bins)
        freqs = np.linspace(0.0, sr/2.0, n_fft//2 + 1, dtype=np.float32)
        mask = tf.cast(freqs[None, None, :] >= float(fmin), tf.float32)
        hf_mag = mag * mask
        return tf.reduce_mean(hf_mag)  # scalar

    def _read_pair_2s(path_src, path_tgt, model_sr):
        info = sf.info(path_src)
        file_sr = info.samplerate
        frames = int(2.0 * file_sr)  # 2 seconds
        x_long, _ = sf.read(path_src, start=0, frames=frames, dtype='float32', always_2d=True)
        y_long, _ = sf.read(path_tgt, start=0, frames=frames, dtype='float32', always_2d=True)
        x = x_long[:, 0]
        y = y_long[:, 0]
        if file_sr != model_sr:
            x = resample_to_model_sr(x, file_sr, model_sr)
            y = resample_to_model_sr(y, file_sr, model_sr)
        return x.astype(np.float32), y.astype(np.float32)

    def _rms(a, mask):
        a = a[mask]
        return float(np.sqrt(np.mean(a * a) + 1e-12)) if a.size > 0 else 0.0

    dbg = _build_offline_debug_model(offline_dyn)

    # Read a short segment
    x, y = _read_pair_2s(src_path, tgt_path, sample_rate)
    x_b = x[None, :, None]
    y_b = y[None, :, None]

    # Force true style for stable comparison
    style_true = np.array([[style_true_flag]], dtype=np.float32)
    style_force = np.ones((1, 1), dtype=np.float32)

    outs = dbg.predict([x_b, style_true, style_force], verbose=0)

    # Fetch arrays
    gate      = np.asarray(outs["gate"][0, :, 0], dtype=np.float32)        # 0..1
    hf_scaled = np.asarray(outs["hf_scaled"][0, :, 0], dtype=np.float32)
    y_pred    = np.asarray(outs["y"][0, :, 0], dtype=np.float32)

    # Pre-emphasis of target and prediction (to inspect high-frequency transient similarity)
    hp_true = _hp(tf.convert_to_tensor(y_b)).numpy()[0, :, 0]
    hp_pred = _hp(tf.convert_to_tensor(y_pred[None, :, None])).numpy()[0, :, 0]

    # Gate stats
    print(f"Gate: mean={gate.mean():.3f}, p95={np.percentile(gate, 95):.3f}, p99={np.percentile(gate, 99):.3f}, max={gate.max():.3f}")
    print(f"HF out (scaled) RMS={np.sqrt(np.mean(hf_scaled**2) + 1e-12):.6f}")

    # Onset vs non-onset comparison using gate as a mask (top 3% as onsets, bottom 50% as off)
    p_on  = float(np.percentile(gate, 97.0))
    p_off = float(np.percentile(gate, 50.0))
    on_mask  = gate >= p_on
    off_mask = gate <= p_off

    print(f"RMS hp_true onsets={_rms(hp_true, on_mask):.6f} off={_rms(hp_true, off_mask):.6f}")
    print(f"RMS hp_pred onsets={_rms(hp_pred, on_mask):.6f} off={_rms(hp_pred, off_mask):.6f}")
    print(f"RMS hf_branch onsets={_rms(hf_scaled, on_mask):.6f} off={_rms(hf_scaled, off_mask):.6f}")

    # High-band STFT energy above the gate HP cutoff
    t_hf_true = float(_band_energy(tf.convert_to_tensor(y_b), sample_rate, fmin=HP_CUTOFF_HZ).numpy())
    t_hf_pred = float(_band_energy(tf.convert_to_tensor(y_pred[None, :, None]), sample_rate, fmin=HP_CUTOFF_HZ).numpy())
    ratio = t_hf_pred / (t_hf_true + 1e-12)
    print(f"Avg |STFT| > {HP_CUTOFF_HZ/1000:.1f} kHz: target={t_hf_true:.6f} pred={t_hf_pred:.6f} (ratio={ratio:.3f})")

@register_keras_serializable()
def build_realtime_model(offline=None, dynamic_time=False):
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None,1), name="offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    style_pred_source = style_prob
    if offline is not None:
        teacher = offline
        teacher.trainable = False
        offline_audio, offline_style_prob = teacher([offline_inp, style_true_in, style_force_in])

        style_prob_3d          = Reshape((1,1), name="realtime_style_prob_3d")(style_prob)
        offline_style_prob_3d  = Reshape((1,1), name="offline_style_prob_3d")(offline_style_prob)
        _ = DistillLoss(alpha=REALTIME_STYLE_DISTILL_ALPHA, name="style_prob_distill")([offline_style_prob_3d, style_prob_3d])

        style_pred_source = BlendScalar(
            init_alpha=REALTIME_STYLE_BLEND_INIT,
            trainable=REALTIME_STYLE_BLEND_TRAINABLE,
            name="realtime_style_blend"
        )([style_prob, offline_style_prob])

    one_minus_force = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_pred_source]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    
    style_scalar_b3 = Reshape((1, 1), name="realtime_style_scalar_broadcast")(style_used_scalar)
    style_embed = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_style_embed")(style_used_scalar)

    x = Conv1D(REALTIME_FILTERS, kernel_size=1, padding="same", name="realtime_init_pw")(inp)

    # Pre-blocks (causal)
    for i in range(REALTIME_PRE_BLOCKS):
        x = gated_sep_block(x, REALTIME_FILTERS, REALTIME_PRE_KERNEL, 1, prefix=f"realtime_pre_{i}", lookahead=0)
        x = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}")

    # Dilated stack (causal)
    realtime_feats = []
    for i, d in enumerate(REALTIME_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=REALTIME_KERNEL_SIZE, dilation=d, lookahead_samples=0)
        x   = ZeroPadding1D((pad_left, pad_right), name=f"realtime_pad_{i}")(x)
        g   = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid", name=f"realtime_sep_{i}")(x)
        c   = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2  = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        x   = Multiply(name=f"realtime_gate_mul_{i}")([Activation("tanh", name=f"realtime_tanh_{i}")(c),
                                                       Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        x   = Conv1D(REALTIME_FILTERS, kernel_size=1, padding="same", name=f"realtime_pw_{i}")(x)
        x   = Add(name=f"realtime_res_add_{i}")([x, res])
        x   = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}")

        offline_feat_name = f"offline_dil_{i}_film_apply"
        if REALTIME_OFFLINE_FEAT_LAYERS and offline_feat_name in REALTIME_OFFLINE_FEAT_LAYERS:
            realtime_feats.append(x)

    # Style-gated pick/finger blend (raw stems then style gating)
    out_pick   = Conv1D(1,1,padding="same", name="realtime_final_pick")(x)
    out_finger = Conv1D(1,1,padding="same", name="realtime_final_finger")(x)
    # Style-gated pick/finger stems via broadcasting
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_realtime = Add(name="realtime_main")([pick_c, finger_c])

    # Low-rate IR Branch (causal)
        # Low-rate IR Branch (causal, intrinsically aligned; no Align needed)
    ir = RealtimeIrDownsampleFIR(name="realtime_ir_downsample")(inp)
    for i, d in enumerate(REALTIME_IR_DILATIONS):
        pad = (REALTIME_IR_KERNEL - 1) * d
        ir = ZeroPadding1D((pad, 0), name=f"realtime_ir_pad_{i}")(ir)
        ir = SeparableConv1D(
            filters=1,
            kernel_size=REALTIME_IR_KERNEL,
            dilation_rate=d,
            padding="valid",
            depthwise_regularizer=regularizers.l2(1e-4),
            pointwise_regularizer=regularizers.l2(1e-4),
            depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
            pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
            name=f"realtime_ir_sep_{i}"
        )(ir)

    ir = IrScale(init=0.6, name="realtime_ir_scale")(ir)

    # Upsample by repeat (ZOH) → single causal min-phase LPF → match length to input
    ir = ResizeNearest1D(factor=REALTIME_IR_DOWNSAMPLE_FACTOR, name="realtime_ir_upsample")(ir)
    ir = IRLowpass(name="realtime_ir_lowpass")(ir)
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir, inp])

    # Gate for HF mask safety cap
    realtime_gate = GateInt8Linear(
        sample_rate=REALTIME_SAMPLE_RATE,
        threshold_db=REALTIME_GATE_THRESHOLD_DB,
        ratio=REALTIME_GATE_RATIO,
        attack_ms=REALTIME_GATE_ATTACK_MS,
        hold_ms=REALTIME_GATE_HOLD_MS,
        release_ms=REALTIME_GATE_RELEASE_MS,
        cutoff_hz=REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ,
        hp_order=REALTIME_HF_HP_ORDER,
        env_ms=REALTIME_GATE_ENV_MS,
        depth_smooth_ms=REALTIME_GATE_DEPTH_SMOOTH_MS,
        env_lmax=REALTIME_GATE_ENV_LMAX,
        depth_lmax=REALTIME_GATE_DEPTH_LMAX,
        min_gain_db=-60.0,  # keep your original default/floor
        name="realtime_gate"
    )(inp)

    # HF branch with learned mask
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(inp)
    hf    = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_hf_init")(hf_in)
    hf    = gated_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b0", lookahead=0)
    hf    = gated_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b1", lookahead=0)
    hf    = gated_sep_block(hf, REALTIME_FILTERS, kernel=3, dilation=1, prefix="realtime_hf_b2", lookahead=0)
    hf    = Conv1D(1,1,padding="same", name="realtime_hf_out")(hf)

    m = SeparableConv1D(8, 5, dilation_rate=1, padding="same", activation="relu", name="realtime_hf_mask_c1")(hf_in)
    m = SeparableConv1D(8, 5, dilation_rate=2, padding="same", activation="relu", name="realtime_hf_mask_c2")(m)
    m = SeparableConv1D(1, 3, dilation_rate=1, padding="same", name="realtime_hf_mask_logits")(m)
    mask_pred = Activation("sigmoid", name="realtime_hf_mask")(m)
    mask_smoothed = CausalMovingAverage(taps=96, name="realtime_hf_mask_smooth")(mask_pred)
    mask_used = Minimum(name="realtime_hf_mask_used")([mask_smoothed, realtime_gate])

    hf_gated = Multiply(name="realtime_hf_mult_gate")([hf, mask_used])
    hf_hp    = HFHighpass(sample_rate=REALTIME_SAMPLE_RATE,
                      cutoff_hz=HP_CUTOFF_HZ,
                      order=REALTIME_HF_HP_ORDER,
                      name="realtime_hf_highpass")(hf_gated)
    hf_final = IrScale(init=8.0, name="realtime_hf_scale")(hf_hp)

    # Training-only constraint (no runtime cost): penalize LF content in hf_final
    hf_final = HFLowbandPenalty(
        alpha=REALTIME_HF_LOWBAND_PENALTY_ALPHA,
        cutoff_hz=2800.0,
        order=129,
        sample_rate=REALTIME_SAMPLE_RATE,
        name="realtime_hf_lowband_penalty"
    )(hf_final)

    y_realtime = Add(name="realtime_with_ir")([y_realtime, ir, hf_final])
    
    # ─── Resonance branch (realtime) ──────────────────────────────────────
    realtime_reso = _add_resonance_branch_realtime(inp=inp, style_embed=style_embed, gate=realtime_gate)

    y_realtime = Add(name="realtime_with_ir_reso")([y_realtime, realtime_reso])
        

    # ─── Distillation for resonance stem from teacher ─────────────────────
    if offline is not None and realtime_reso is not None:
        try:
            offline_reso_scaled = teacher.get_layer("offline_reso_scale").output  # (B, T96k, 1)
            offline_reso_ds     = FIRDecimateBy2(name="offline_reso_ds2")(offline_reso_scaled)
            _ = DistillLoss(alpha=REALTIME_STEM_RESO_DISTILL_ALPHA, name="stem_reso_distill")([offline_reso_ds, realtime_reso])
        except:
            # If training with a teacher that has no resonance branch yet
            pass

    # ─── Distillation (teacher must be causal here) ────────────────────────────
    if offline is not None:
        # Teacher feature model
        T_outs = []
        T_gate = None
        if REALTIME_OFFLINE_FEAT_LAYERS:
            offline_feat_model = tf.keras.models.Model(
                inputs=teacher.inputs,
                outputs=[teacher.get_layer(n).output for n in REALTIME_OFFLINE_FEAT_LAYERS] + [teacher.get_layer("offline_gate").output],
                name="offline_feat_model"
            )
            T_outs = offline_feat_model([offline_inp, style_true_in, style_force_in])
            if not isinstance(T_outs, list):
                T_outs = [T_outs]
            T_feats = T_outs[:-1]
            T_gate  = T_outs[-1]
        else:
            T_gate = teacher.get_layer("offline_gate").output

        # Downsample teacher features/gate
        T_feats_ds = [FIRDecimateBy2(name=f"offline_feat_ds2_{idx}")(t) for idx, t in enumerate(T_outs[:-1])] if T_outs else []
        T_gate_ds  = FIRDecimateBy2(name="offline_gate_ds2")(T_gate)

        # KD for features
        for idx, (T_f_ds, S_f) in enumerate(zip(T_feats_ds, realtime_feats)):
            chan = int(T_f_ds.shape[-1])
            proj = Conv1D(chan, 1, padding="same", name=f"realtime_proj_feat_{idx}")(S_f)
            _ = DistillLoss(alpha=REALTIME_FEAT_DISTILL_ALPHA, name=f"feat_distill_{idx}")([T_f_ds, proj])

        # KD for HF mask: teacher gate (96k→48k) vs student mask_pred
        _ = DistillLoss(alpha=REALTIME_HF_MASK_DISTILL_ALPHA, name="hf_mask_distill")([T_gate_ds, mask_pred])
        # Optional: also KD on mask_used to avoid fighting the cap
        _ = DistillLoss(alpha=REALTIME_MASK_USED_DISTILL_ALPHA, name="mask_used_distill")([T_gate_ds, mask_used])

        # KD for teacher audio (96k→48k) vs student audio: shift-aware + STFT
        offline_audio_ds = FIRDecimateBy2(name="offline_audio_ds2")(offline_audio)
        _ = DistillLossBestShift(alpha=REALTIME_DISTILL_ALPHA, max_shift=8, name="audio_distill")([offline_audio_ds, y_realtime])
        _ = MSTFTDistillLoss(alpha=REALTIME_AUDIO_STFT_DISTILL_ALPHA, name="audio_stft_distill")([offline_audio_ds, y_realtime])

        # KD for stems: pick/finger/HF
        offline_pick   = teacher.get_layer("offline_final_pick").output
        offline_finger = teacher.get_layer("offline_final_finger").output
        offline_hf_scaled = teacher.get_layer("offline_hf_scale").output  # post-HPF + scale at 96k

        offline_pick_ds   = FIRDecimateBy2(name="offline_pick_ds2")(offline_pick)
        offline_finger_ds = FIRDecimateBy2(name="offline_finger_ds2")(offline_finger)
        offline_hf_ds     = FIRDecimateBy2(name="offline_hf_ds2")(offline_hf_scaled)
        
        # Band-match the teacher HF KD target to the student's band by re-HPF at 48 kHz, 3 kHz
        offline_hf_ds_hp48 = HFHighpass(
            sample_rate=REALTIME_SAMPLE_RATE,
            cutoff_hz=HP_CUTOFF_HZ,
            order=REALTIME_HF_HP_ORDER,
            name="offline_hf_ds_hp48"
        )(offline_hf_ds)

        # KD with shift-aware MSE for pick/finger stems
        _ = DistillLossBestShift(alpha=REALTIME_STEM_PICK_DISTILL_ALPHA,   max_shift=8, name="stem_pick_distill")([offline_pick_ds,   out_pick])
        _ = DistillLossBestShift(alpha=REALTIME_STEM_FINGER_DISTILL_ALPHA, max_shift=8, name="stem_finger_distill")([offline_finger_ds, out_finger])

        # HF stem KD: use the 48 kHz, 3 kHz high-pass matched target
        _ = DistillLoss(alpha=REALTIME_STEM_HF_DISTILL_ALPHA, name="stem_hf_distill")([offline_hf_ds_hp48, hf_final])

    inputs_list = [inp, style_true_in, style_force_in]
    if offline_inp is not None:
        inputs_list = [inp, offline_inp, style_true_in, style_force_in]

    return tf.keras.models.Model(inputs=inputs_list, outputs=[y_realtime, style_prob], name="piezo2mic_realtime")

def approx_steps_from_chunks(chunks, seg_len, model_sample_rate, batch_size):
    """
    Estimate steps/epoch given chunks defined in file-rate samples while model uses model_sample_rate.
    Uses 50% overlap.
    """
    total_windows = 0
    for src, _, _, cdur in chunks:
        file_sr = sf.info(src).samplerate
        need_src_len = int(math.ceil(seg_len * file_sr / float(model_sample_rate)))
        hop_src = max(1, need_src_len // 2)
        num = max((cdur - need_src_len) // hop_src + 1, 0)
        total_windows += num
    return 10 if DEBUG else max(total_windows // batch_size, 1)

# ────────────────────────────── TRAINING ─────────────────────────────────────

@register_keras_serializable()
def add_sw_offline(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    # 50% chance to force truth
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask  # 1 or 0
    inputs = {
    'offline_piezo_input': x,
    'offline_style_true_input': style_true,
    'offline_style_force_input': style_force,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    
    # 50% chance to force truth
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask  # 1 or 0
    
    inputs = {
    'realtime_piezo_input': x,
    'realtime_style_true_input': style_true,
    'realtime_style_force_input': style_force,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime_dual(x, y):
    """
    x = (x_realtime, x_offline96k)
    y = (y_audio_realtime, style_flag)
    Produces a dict of model inputs with both realtime and offline inputs for distillation.
    Returns sample_weights as a tuple matching the model's output order for maximum compatibility.

    Note: style_force is randomized (50%) so the network sometimes uses the predicted path
    (where the teacher-student blended style is applied) during training.
    """
    x_realtime, x_offline = x
    y_audio, style_flag = y
    b = tf.shape(x_realtime)[0]; T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)

    # 50% chance to force ground-truth style vs. predicted (blended) style
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask  # 1 or 0

    inputs = {
        'realtime_piezo_input':        x_realtime,
        'offline_piezo_input':         x_offline,
        'realtime_style_true_input':   style_true,
        'realtime_style_force_input':  style_force,
    }
    outputs = (y_audio, style_true)

    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)

    sample_weights = (sw_audio, sw_style)
    return inputs, outputs, sample_weights

@register_keras_serializable()
def add_sw_realtime_finetune_nf(x, y):
    # y = (y_audio, style_flag) but finetune uses no true style labels
    y_audio, _ = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]

    # Dummy style label for the style head (B,1)
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    # Force the model to use its own classifier during finetune
    style_force = tf.zeros((b, 1), dtype=tf.float32)  # 0 = use prediction

    # Provide the full input signature to keep the model interface unchanged
    inputs = {
        'realtime_piezo_input':        x,
        'realtime_style_true_input':   style_true,
        'realtime_style_force_input':  style_force,
    }

    # Targets: audio target plus a dummy style target (won't be used)
    outputs = (y_audio, style_true)

    # Sample weights: train audio only, disable style BCE
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)

    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def build_offline_debug_model(offline):
    outs = {
        "gate":      offline.get_layer("offline_gate").output,
        "hf_pre":    offline.get_layer("offline_hf_out").output,
        "hf_gated":  offline.get_layer("offline_hf_mult_gate").output,
        "hf_scaled": offline.get_layer("offline_hf_scale").output,
        "pick_c":    offline.get_layer("offline_pick_contrib").output,
        "finger_c":  offline.get_layer("offline_finger_contrib").output,
        "y_pred":    offline.get_layer("offline_mix_main").output,
    }
    
    outs.update({
        "reso":     offline.get_layer("offline_reso_scale").output if "offline_reso_scale" in [l.name for l in offline.layers] else None
    })
    
    return tf.keras.Model(inputs=offline.inputs, outputs=outs, name="offline_debug_model")


@register_keras_serializable()
def build_realtime_debug_model(realtime):
    
    outs = {
        "gate":      realtime.get_layer("realtime_gate").output,
        "hf_pre":    realtime.get_layer("realtime_hf_out").output,
        "hf_gated":  realtime.get_layer("realtime_hf_mult_gate").output,
        "hf_scaled": realtime.get_layer("realtime_hf_scale").output,
        "mask_pred": realtime.get_layer("realtime_hf_mask").output,
        "mask_used": realtime.get_layer("realtime_hf_mask_used").output,
        "pick_c":    realtime.get_layer("realtime_pick_contrib").output,
        "finger_c":  realtime.get_layer("realtime_finger_contrib").output,
        "y_pred":    realtime.get_layer("realtime_with_ir_reso").output,
        "reso":     realtime.get_layer("realtime_reso_scale").output,
        "ir_only":  realtime.get_layer("realtime_ir_lowpass").output
    }
    
    # Optional bypass exports if you added them
    try:
        outs["hf_scaled_gate_only"] = realtime.get_layer("realtime_hf_scale_gate_only").output
    except:
        pass
    try:
        outs["hf_mask_smooth"] = realtime.get_layer("realtime_hf_mask_smooth").output
    except:
        pass
    return tf.keras.Model(inputs=realtime.inputs, outputs=outs, name="realtime_debug_model")

def boost_hf_scale_in_model(model, layer_name, gain_db):
    """
    Multiplies the IrScale scalar in the specified layer by 10^(gain_db/20).
    layer_name: "offline_hf_scale" or "realtime_hf_scale"
    """
    layer = model.get_layer(layer_name)
    w = layer.get_weights()  # [array([scale], dtype=float32)]
    if not w:
        raise ValueError(f"No weights in layer {layer_name}")
    s = float(w[0][0])
    g = 10.0 ** (gain_db / 20.0)
    w[0][0] = s * g
    layer.set_weights(w)
    print(f"{layer_name}: {s:.4f} → {w[0][0]:.4f} (+{gain_db} dB)")
    return model

def audible_gate_debug(debug_model,
                                 path_src,
                                 path_tgt,
                                 out_dir,
                                 sample_rate,
                                 sec=10.0,
                                 offset_sec=0.0,
                                 style_mode="true",
                                 style_true_flag=1.0,
                                 block_sec=10.0,
                                 resample_margin_ms=50.0):

    os.makedirs(out_dir, exist_ok=True)

    hf_scale_layer_name = "offline_hf_scale" if sample_rate == OFFLINE_SAMPLE_RATE else "realtime_hf_scale"
    pick_boost_db = OFFLINE_PICK_BOOST_DB if sample_rate == OFFLINE_SAMPLE_RATE else REALTIME_PICK_BOOST_DB
    
    if pick_boost_db != 0.0:
        debug_model = boost_hf_scale_in_model(debug_model, hf_scale_layer_name, pick_boost_db)

    if REALTIME_IR_BOOST_DB != 0.0 and sample_rate == REALTIME_SAMPLE_RATE:
        debug_model = boost_hf_scale_in_model(debug_model, "realtime_ir_scale", gain_db=REALTIME_IR_BOOST_DB)
        
    # Optional: resonance stem boost for quick A/B during demo generation
    try:
        if OFFLINE_RESO_BOOST_DB != 0.0 and sample_rate == OFFLINE_SAMPLE_RATE:
            debug_model = boost_hf_scale_in_model(debug_model, "offline_reso_scale", gain_db=OFFLINE_RESO_BOOST_DB)
        elif REALTIME_RESO_BOOST_DB != 0.0 and sample_rate == REALTIME_SAMPLE_RATE:
            debug_model = boost_hf_scale_in_model(debug_model, "realtime_reso_scale", gain_db=REALTIME_RESO_BOOST_DB)
    except Exception:
        pass  # layer may not exist (e.g., older models)

    # Input region in file-sample domain
    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))

    # Style inputs
    if style_mode == "true":
        style_in  = np.array([[style_true_flag]], dtype=np.float32)
        force_in  = np.ones((1, 1), dtype=np.float32)
    else:
        style_in  = np.zeros((1, 1), dtype=np.float32)
        force_in  = np.zeros((1, 1), dtype=np.float32)

    # Output writers
    w_src  = sf.SoundFile(os.path.join(out_dir, "src.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_tgt  = sf.SoundFile(os.path.join(out_dir, "tgt.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_full = sf.SoundFile(os.path.join(out_dir, "pred_full.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_nohf = sf.SoundFile(os.path.join(out_dir, "pred_nohf.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_hf   = sf.SoundFile(os.path.join(out_dir, "hf_only.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_gr   = sf.SoundFile(os.path.join(out_dir, "gate_raw.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_gt   = sf.SoundFile(os.path.join(out_dir, "gate_tone.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_gh   = sf.SoundFile(os.path.join(out_dir, "gate_hiss.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_reso = sf.SoundFile(os.path.join(out_dir, "reso_only.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_ir_only = sf.SoundFile(os.path.join(out_dir, "ir_only.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

    # Tone state (continuous phase across blocks)
    tone_freq = gate_debug_tone_freq(sample_rate, HP_CUTOFF_HZ)
    omega = 2.0 * np.pi * (tone_freq / float(sample_rate))
    phase = 0.0

    produced_model = 0

    while produced_model < want_total_model:
        core_model = min(block_core_model, want_total_model - produced_model)
        left_this = left_ctx_model if produced_model > 0 else 0
        right_this = right_ctx_model if (produced_model + core_model) < want_total_model else 0

        core_start_file_off = int(round(produced_model * (file_sr / float(sample_rate))))
        core_start_file = START_FRAME + core_start_file_off

        left_ctx_file = int(math.ceil(left_this * (file_sr / float(sample_rate))))
        right_ctx_file = int(math.ceil(right_this * (file_sr / float(sample_rate))))
        core_file_len = int(math.ceil(core_model * (file_sr / float(sample_rate))))

        read_a = max(START_FRAME, core_start_file - left_ctx_file - resample_margin_file)
        read_b_desired = core_start_file + core_file_len + right_ctx_file + resample_margin_file
        read_b = min(END_FRAME, read_b_desired)
        n_read = max(0, read_b - read_a)
        if n_read <= 0:
            break

        x_blk_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        y_blk_2d, _ = sf.read(path_tgt, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        x_file = x_blk_2d[:, 0]
        y_file = y_blk_2d[:, 0]

        if file_sr != sample_rate:
            x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
            y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
        else:
            x_resamp = x_file
            y_resamp = y_file

        left_off_file = (core_start_file - left_ctx_file) - read_a
        left_off_model = int(round(left_off_file * (sample_rate / float(file_sr))))
        inp_start = max(0, min(left_off_model, len(x_resamp)))
        inp_end = max(inp_start, min(inp_start + left_this + core_model + right_this, len(x_resamp)))
        x_inp = x_resamp[inp_start:inp_end]
        if x_inp.size == 0:
            break

        core_start_in_inp = min(left_this, len(x_inp))
        core_end_in_inp = min(core_start_in_inp + core_model, len(x_inp))
        core_len = max(0, core_end_in_inp - core_start_in_inp)
        if core_len <= 0:
            break

        x_b = x_inp[None, :, None].astype(np.float32)
        outs = debug_model.predict([x_b, style_in, force_in], verbose=0)
        # outs is a dict with keys: "gate", "hf_scaled", "pick_c", "finger_c", "y_pred"
        def _core_of(key):
            arr = np.asarray(outs[key][0, :, 0], dtype=np.float32)
            return arr[core_start_in_inp:core_end_in_inp]

        gate_core   = np.clip(_core_of("gate"), 0.0, 1.0)
        y_pred_core = _core_of("y_pred")
        pick_c_core = _core_of("pick_c")
        fins_c_core = _core_of("finger_c")
        hf_s_core   = _core_of("hf_scaled")
        y_nohf_core = pick_c_core + fins_c_core

        # Write model-derived signals
        w_full.write(y_pred_core)
        w_nohf.write(y_nohf_core)
        w_hf.write(hf_s_core)
        w_reso.write(_core_of("reso"))
        w_ir_only.write(_core_of("ir_only"))

        # Write src/tgt (same cropping)
        core_start_global = inp_start + core_start_in_inp
        core_end_global   = inp_start + core_end_in_inp
        w_src.write(x_resamp[core_start_global:core_end_global])
        w_tgt.write(y_resamp[core_start_global:core_end_global])

        # gate_raw [-0.8..+0.8]
        gate_raw = 0.8 * (2.0 * (gate_core - 0.5))
        w_gr.write(gate_raw)

        # gate_tone and gate_hiss (streaming-safe)
        t = np.arange(core_len, dtype=np.float32)
        tone = np.sin(phase + omega * t).astype(np.float32)
        phase = (phase + omega * float(core_len)) % (2.0 * np.pi)
        w_gt.write(0.8 * gate_core * tone)

        # simple HP hiss, scaled
        # quick HP via 1st-order pre-emphasis (consistent with your other code)
        hiss = np.random.randn(core_len).astype(np.float32)
        if core_len > 1:
            hiss[1:] = hiss[1:] - HP_PRE_EMPHASIS_HF * hiss[:-1]
        w_gh.write(0.8 * gate_core * hiss)

        produced_model += core_len

    # Close all writers
    for w in [w_src, w_tgt, w_full, w_nohf, w_hf, w_gr, w_gt, w_gh, w_reso, w_ir_only]:
        try:
            w.close()
        except:
            pass

    print(f"Saved streaming audible gate debug ({style_mode}) to {out_dir}")

def train_shared(
    kind,
    seg_len,
    sample_rate,
    batch_size,
    build_model_fn,
    checkpoint_out_dir,
    logs_out_dir,
    demo_checkpoints_out_dir,
    static_model_filepath,
    dynamic_model_filepath,
    resume=False,
    resume_ckpt_path=None,
    build_model_fn_args_static={},
    build_model_fn_args_dynamic={},
    dual_offline=False
):
    global L1_WEIGHT, HF_WEIGHT, STFT_WEIGHT, MEL_WEIGHT, RMS_ENV_WEIGHT

    assert_has_pairs()
    all_chunks = build_chunks(pairs_list=all_pairs,    seg_len=seg_len, sample_rate=sample_rate)
    pick_chunks = build_chunks(pairs_list=picked_pairs, seg_len=seg_len, sample_rate=sample_rate)
    finger_chunks = build_chunks(pairs_list=finger_pairs, seg_len=seg_len, sample_rate=sample_rate)

    idx = list(range(len(all_chunks)))
    if len(idx) > 1:
        tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=SEED)
    else:
        raise RuntimeError("Not enough data to split into train/val sets.")

    train_chunks = [all_chunks[i] for i in tr_i]
    val_chunks   = [all_chunks[i] for i in va_i]
    train_pick_chunks   = [c for c in train_chunks if c in pick_chunks]
    train_finger_chunks = [c for c in train_chunks if c in finger_chunks]
    val_pick_chunks     = [c for c in val_chunks   if c in pick_chunks]
    val_finger_chunks   = [c for c in val_chunks   if c in finger_chunks]

    steps_per_epoch  = approx_steps_from_chunks(chunks=train_chunks, seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    validation_steps = approx_steps_from_chunks(chunks=val_chunks,   seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    sig_y = (
        tf.TensorSpec((seg_len, 1), tf.float32),
        tf.TensorSpec((), tf.float32)
    )
    
    if kind == 'realtime' and dual_offline:
        # offline:realtime SR = 96k:48k = 2:1
        offline_factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE
        assert offline_factor == 2, "This code assumes 96k offline and 48k realtime."

        sig_x = (
            tf.TensorSpec((seg_len, 1), tf.float32),                 # realtime 48k
            tf.TensorSpec((seg_len * offline_factor, 1), tf.float32) # offline 96k
        )

        pick_ds   = tf.data.Dataset.from_generator(
            lambda: segment_generator_dual(seg_len_realtime=seg_len, chunks=train_pick_chunks, is_pick_chunks=True,
                                           realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE),
            output_signature=(sig_x, sig_y)
        )
        finger_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator_dual(seg_len_realtime=seg_len, chunks=train_finger_chunks, is_pick_chunks=False,
                                           realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE),
            output_signature=(sig_x, sig_y)
        )

        val_pick_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator_dual(seg_len_realtime=seg_len, chunks=val_pick_chunks, is_pick_chunks=True,
                                           realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE),
            output_signature=(sig_x, sig_y)
        )
        val_finger_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator_dual(seg_len_realtime=seg_len, chunks=val_finger_chunks, is_pick_chunks=False,
                                           realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE),
            output_signature=(sig_x, sig_y)
        )

        sw_map_fn = add_sw_realtime_dual
    else:
        # Original single-branch 48 kHz dataset
        sig_x = tf.TensorSpec((seg_len, 1), tf.float32)

        pick_ds   = tf.data.Dataset.from_generator(
            lambda: segment_generator(seg_len=seg_len, chunks=train_pick_chunks, is_pick_chunks=True,  model_sample_rate=sample_rate),
            output_signature=(sig_x, sig_y)
        )
        finger_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator(seg_len=seg_len, chunks=train_finger_chunks, is_pick_chunks=False, model_sample_rate=sample_rate),
            output_signature=(sig_x, sig_y)
        )

        val_pick_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator(seg_len=seg_len, chunks=val_pick_chunks, is_pick_chunks=True,  model_sample_rate=sample_rate),
            output_signature=(sig_x, sig_y)
        )
        val_finger_ds = tf.data.Dataset.from_generator(
            lambda: segment_generator(seg_len=seg_len, chunks=val_finger_chunks, is_pick_chunks=False, model_sample_rate=sample_rate),
            output_signature=(sig_x, sig_y)
        )

        sw_map_fn = add_sw_offline if kind == 'offline' else add_sw_realtime

    train_ds = tf.data.Dataset.sample_from_datasets(
        [pick_ds, finger_ds],
        weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO]
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(batch_size)

    train_ds = train_ds.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.sample_from_datasets(
        [val_pick_ds, val_finger_ds],
        weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO],
        seed=SEED
    ).batch(batch_size=batch_size)
    val_ds = val_ds.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    if resume and resume_ckpt_path:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        static_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        initial_epoch = 0
  
    new_w, _ = calibrate_loss_weights_consistent(
        model=static_model, dataset=val_ds, sample_rate=sample_rate, n_batches=5
    )
    L1_WEIGHT      = new_w['l1']
    STFT_WEIGHT    = new_w['stft']
    MEL_WEIGHT     = new_w['mel']
    RMS_ENV_WEIGHT = new_w['env']
    HF_WEIGHT      = new_w['hf']
    print(f"→ normalized ({kind}) weights:"
        f" L1={L1_WEIGHT:.6f}"
        f" STFT={STFT_WEIGHT:.6f}"
        f" MEL={MEL_WEIGHT:.6f}"
        f" ENV={RMS_ENV_WEIGHT:.6f}"
        f" HF={HF_WEIGHT:.6f}")

    # Instantiate the loss with the per-model smoothing
    _combined_loss = CombinedLoss(sample_rate, extra_small_cplx=(kind == 'offline'))
    _log_mel_loss  = LogMelLoss(sample_rate)

    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=[ _combined_loss, "binary_crossentropy" ],
        loss_weights = [ 1.0,  LAMBDA_STYLE_RATIO ],
        metrics = [
            [ 'mae', _log_mel_loss, mean_hf_metric, multi_scale_stft_loss, rms_env_loss ],
            [ 'accuracy' ]
        ]
    )
    static_model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=LR_PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=MIN_LR, mode='min', verbose=1)
    cp = ModelCheckpoint(filepath=os.path.join(checkpoint_out_dir, "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"),
                         monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    
    terminate_on_nan = TerminateOnNaN()
    
    demo_cb = DemoCallback(sample_rate=sample_rate,
                           build_model_fn=build_model_fn,
                           build_model_fn_args_dynamic=build_model_fn_args_dynamic,
                           demo_checkpoints_out_dir=demo_checkpoints_out_dir,
                           use_safe_weight_copy=(kind != 'offline')
    )
    tb = TensorBoard(log_dir=logs_out_dir, write_graph=False)

    cbs = [es, cp, reduce_lr, terminate_on_nan, tb, demo_cb]

    if kind == 'realtime':
        kd_targets = {
            # audio KD
            "audio_distill":       (REALTIME_DISTILL_ALPHA,        2.0,  0,  80),   # 8→2 example if you set REALTIME_DISTILL_ALPHA=8; adjust as needed
            "audio_stft_distill":  (REALTIME_AUDIO_STFT_DISTILL_ALPHA, 0.2,  0,  80),
            # feature/stem KD
            "feat_distill_0":      (REALTIME_FEAT_DISTILL_ALPHA,   0.5,  0,  80),
            "feat_distill_1":      (REALTIME_FEAT_DISTILL_ALPHA,   0.5,  0,  80),
            "stem_pick_distill":   (REALTIME_STEM_PICK_DISTILL_ALPHA,   0.2,  0,  80),
            "stem_finger_distill": (REALTIME_STEM_FINGER_DISTILL_ALPHA, 0.2,  0,  80),
            "stem_hf_distill":     (REALTIME_STEM_HF_DISTILL_ALPHA,     0.2,  0,  80),
            # mask/gate KD
            "hf_mask_distill":     (REALTIME_HF_MASK_DISTILL_ALPHA,     1.0,  0,  80),
            "mask_used_distill":   (REALTIME_MASK_USED_DISTILL_ALPHA,   0.1,  0,  80),
            # style blend schedule (not a KD layer, but exposes a scalar weight)
            "realtime_style_blend":(REALTIME_STYLE_BLEND_INIT,       0.3,  0, 100),
            "stem_reso_distill":   (REALTIME_STEM_RESO_DISTILL_ALPHA, 0.2, 0, 80)
        }
        cbs.append(KDAlphaScheduler(kd_targets))

    static_model.fit(train_ds,
                     initial_epoch=initial_epoch,
                     epochs=1 if DEBUG else EPOCHS,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=val_ds,
                     validation_steps=validation_steps,
                     callbacks=cbs)

    static_model.save(static_model_filepath)
    print("✅ Saved FP32 static model:", static_model_filepath)

    dynamic_model = build_model_fn(**build_model_fn_args_dynamic, dynamic_time=True)
    
    if kind == 'offline':
        dynamic_model.set_weights(static_model.get_weights())
    else:
        copy_weights_by_name(dynamic_model, static_model)
    
    dynamic_model.save(dynamic_model_filepath)
    print("✅ Saved FP32 dynamic model:", dynamic_model_filepath)

    return static_model, dynamic_model

def train_offline(resume=False, resume_ckpt_path=None):
    train_shared(
        kind='offline',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=OFFLINE_SEG_LEN,
        sample_rate=OFFLINE_SAMPLE_RATE,
        batch_size=OFFLINE_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_offline_model,
        checkpoint_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
        logs_out_dir=LOGS_OFFLINE_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR
    )

def train_realtime(resume=False, resume_ckpt_path=None):
    offline_dynamic_model_filepath = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(offline_dynamic_model_filepath):
        print("Offline model not found, run --train-offline first")
        sys.exit(1)

    # Load the original (non-causal) offline model
    offline_nc = tf.keras.models.load_model(offline_dynamic_model_filepath, compile=False)
    print("Loaded offline (non-causal) from", offline_dynamic_model_filepath)

    # Build a causalized teacher clone (no right-lookahead) and copy weights
    offline_causal = build_offline_model_causal(offline_nc, dynamic_time=True)
    print("Built causal teacher clone for KD (right-lookahead=0).")

    train_shared(
        kind='realtime',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=REALTIME_SEG_LEN,
        sample_rate=REALTIME_SAMPLE_RATE,
        batch_size=REALTIME_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_realtime_model,
        build_model_fn_args_static={ 'offline': offline_causal },
        build_model_fn_args_dynamic={ },
        checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
        logs_out_dir=LOGS_REALTIME_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
        dual_offline=True  # keep dual-rate dataset (48k student + 96k teacher input)
    )

def generate_fine_tune_demo(model):
    return generate_demo_from_files(
        model=model,
        sample_rate     = FINETUNE_SAMPLE_RATE,
        offset_sec      = FINETUNE_DEMO_OFFSET_SEC,
        sec             = FINETUNE_DEMO_SEC,
        path_src        = os.path.join(FINETUNE_SRC_DIR, "source.wav"),
        path_target     = os.path.join(FINETUNE_SRC_DIR, "target.wav"),
        out_dir         = FINETUNE_DEMOS_OUT_DIR,
        prediction_name = "demo_pred"
    )

def fine_tune():
    global picked_pairs, finger_pairs, all_pairs, audio_cache
    global L1_WEIGHT, HF_WEIGHT, STFT_WEIGHT, MEL_WEIGHT, RMS_ENV_WEIGHT

    corrected_weights = {
        'l1': FINETUNE_L1_WEIGHT,
        'hf': HF_WEIGHT,
        'stft': STFT_WEIGHT,
        'mel': FINETUNE_MEL_WEIGHT,
        'env': RMS_ENV_WEIGHT
    }

    src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
    tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    if not os.path.exists(src) or not os.path.exists(tgt):
        print("ERROR: finetune source/target not found in", FINETUNE_SRC_DIR)
        sys.exit(1)

    x_full, sr_x = sf.read(src, dtype='float32', always_2d=True)
    y_full, sr_y = sf.read(tgt, dtype='float32', always_2d=True)
    assert sr_x == sr_y == FINETUNE_SAMPLE_RATE, f"Sample rate mismatch: {sr_x} vs {sr_y} vs {FINETUNE_SAMPLE_RATE}"

    rms_x = compute_rms(x_full[:,0])
    rms_y = compute_rms(y_full[:,0])
    print(f"[fine_tune] input RMS: source={rms_x:.6f}, target={rms_y:.6f}")

    #target_rms = 10 ** ( INPUT_AVG_RMS_DBFS_TARGET / 20.0 )
    #gain_x     = target_rms / (rms_x + 1e-12)
    #gain_y     = target_rms / (rms_y + 1e-12)
    #print(f"[fine_tune] applying gains: source×{gain_x:.3f}, target×{gain_y:.3f}")

    x_norm = x_full # * gain_x
    y_norm = y_full # * gain_y

    rms_x2 = compute_rms(x_norm[:,0])
    rms_y2 = compute_rms(y_norm[:,0])
    print(f"[fine_tune] post‐norm RMS: source={rms_x2:.6f} ({20*np.log10(rms_x2):.2f} dB), target={rms_y2:.6f} ({20*np.log10(rms_y2):.2f} dB)")

    picked_pairs = [(src, tgt)]
    finger_pairs = [(src, tgt)]
    all_pairs    = picked_pairs + finger_pairs
    
    assert_has_pairs()

    audio_cache.clear()
    audio_cache[src] = x_norm
    audio_cache[tgt] = y_norm

    def make_mixed_ds(seg_len):
        chunks = build_chunks(pairs_list=all_pairs, seg_len=seg_len, sample_rate=FINETUNE_SAMPLE_RATE)
        spec_x       = tf.TensorSpec((seg_len,1), tf.float32)
        spec_y_audio = tf.TensorSpec((seg_len,1), tf.float32)
        spec_y_style = tf.TensorSpec((),       tf.float32)
        spec_y       = (spec_y_audio, spec_y_style)

        def pu_gen():
            for x, (y_audio, style_flag) in segment_generator(seg_len=seg_len, chunks=chunks, is_pick_chunks=True, model_sample_rate=FINETUNE_SAMPLE_RATE):
                yield x, (y_audio, style_flag)

        def fu_gen():
            for x, (y_audio, style_flag) in segment_generator(seg_len=seg_len, chunks=chunks, is_pick_chunks=False, model_sample_rate=FINETUNE_SAMPLE_RATE):
                yield x, (y_audio, style_flag)

        ds_pu = tf.data.Dataset.from_generator(pu_gen, output_signature=(spec_x, spec_y))
        ds_fu = tf.data.Dataset.from_generator(fu_gen, output_signature=(spec_x, spec_y))
        mixed = tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO], seed=SEED)
        return mixed.batch(FINETUNE_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("\n🔧 Fine-tuning REALTIME on new pair…")
    base_realtime_full = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME), compile=False)
    finetuned_realtime_small = build_realtime_model(offline=None, dynamic_time=False)

    for layer in finetuned_realtime_small.layers:
        if layer.name.startswith("realtime_"):
            try:
                layer.set_weights(base_realtime_full.get_layer(layer.name).get_weights())
            except ValueError:
                pass

    LAST_REALTIME_DILATION = len(REALTIME_DILATIONS) - 1
    for layer in finetuned_realtime_small.layers:
        name = layer.name
        layer.trainable = \
            "film_dense" in name \
            or name.startswith("realtime_final_") \
            or name == "realtime_ir_scale" \
            or name == f"realtime_sep_{LAST_REALTIME_DILATION}" \
            or name == f"realtime_ir_sep_{LAST_REALTIME_DILATION}" \
            or name == f"realtime_pw_{LAST_REALTIME_DILATION}" \
            or name.startswith("realtime_hf_")

    dsS_val = make_mixed_ds(REALTIME_SEG_LEN)
    dsS_val = dsS_val.map(add_sw_realtime_finetune_nf)
    dsS_val = dsS_val.take(CALIBRATION_CHUNK_COUNT)

    # Normalize weights consistently with the actual loss shape
    new_w, _ = calibrate_loss_weights_consistent(
        model=finetuned_realtime_small, dataset=dsS_val, sample_rate=FINETUNE_SAMPLE_RATE, n_batches=5
    )
    
    L1_WEIGHT      = new_w['l1']
    STFT_WEIGHT    = new_w['stft']
    MEL_WEIGHT     = new_w['mel']
    RMS_ENV_WEIGHT = new_w['env']
    HF_WEIGHT      = new_w['hf']
    print(f"→ normalized finetune weights:"
        f" L1={L1_WEIGHT:.6f}"
        f" STFT={STFT_WEIGHT:.6f}"
        f" MEL={MEL_WEIGHT:.6f}"
        f" ENV={RMS_ENV_WEIGHT:.6f}"
        f" HF={HF_WEIGHT:.6f}")

    # Instantiate the loss with the per-model smoothing
    _combined_loss = CombinedLoss(FINETUNE_SAMPLE_RATE, extra_small_cplx=False)

    finetuned_realtime_small.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_REALTIME_LR),
        loss=[_combined_loss, "binary_crossentropy"],
        loss_weights=[1.0, 0.0],   # disable BCE during finetune
        metrics=[["mae"], []]
    )

    dsS = make_mixed_ds(REALTIME_SEG_LEN)
    dsS = dsS.map(add_sw_realtime_finetune_nf)
    ft_chunks_T = build_chunks(pairs_list=[(src, tgt)], seg_len=REALTIME_SEG_LEN, sample_rate=FINETUNE_SAMPLE_RATE)

    stepsS = max(approx_steps_from_chunks(ft_chunks_T, REALTIME_SEG_LEN, FINETUNE_SAMPLE_RATE, FINETUNE_BATCH_SIZE), 1)
    print("Finetune realtime model summary:")
    finetuned_realtime_small.summary()
    finetuned_realtime_small.fit(dsS, epochs=FINETUNE_REALTIME_EPOCHS, steps_per_epoch=stepsS, verbose=1)

    ft_s_static = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    finetuned_realtime_small.save(ft_s_static)
    finetuned_realtime_dyn = build_realtime_model(offline=None, dynamic_time=True)
    finetuned_realtime_dyn.set_weights(finetuned_realtime_small.get_weights())
    ft_s_dyn = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
    finetuned_realtime_dyn.save(ft_s_dyn)
    return finetuned_realtime_dyn

def compute_rms(sig):
    sig64 = sig.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(sig64 * sig64) + 1e-12))

def cleanup_and_prepare_train (logs_out_dir, checkpoints_out_dir, demo_checkpoints_out_dir, demo_generator_outputs_out_dir):
    for dir_path in [logs_out_dir, demo_checkpoints_out_dir, demo_generator_outputs_out_dir]:
        for sub in os.listdir(dir_path):
            sub_path = os.path.join(dir_path, sub)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            elif os.path.isdir(sub_path):
                shutil.rmtree(sub_path)
    ckpts = sorted(glob.glob(os.path.join(checkpoints_out_dir, "*.keras")))
    resume = len(ckpts) > 0
    resume_ckpt = ckpts[-1] if resume else None
    if resume:
        ep = _extract_epoch_from_ckpt(resume_ckpt)
        print(f"→ Found checkpoint, will resume from epoch {ep}: {resume_ckpt}")
        return (resume, resume_ckpt)
    else:
        return (False, None)

def prepare_and_generate_demo (sample_rate, dynamic_model_filepath, checkpoint_out_dir, demo_out_dir, build_model_fn, build_model_fn_args_dynamic={}, use_safe_weight_copy=False):
    dpath = dynamic_model_filepath
    if not os.path.exists(dpath):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_out_dir, "*.keras")))
        if checkpoints:
            dpath = checkpoints[-1]
            print("Loading model from checkpoint:", dpath)
            static_model = tf.keras.models.load_model(dpath, compile=False)
            dynamic_model = build_model_fn(**build_model_fn_args_dynamic, dynamic_time=True)
            if use_safe_weight_copy:
                copy_weights_by_name(dynamic_model, static_model)
            else:
                dynamic_model.set_weights(static_model.get_weights())
        else:
            dpath = None
        if dpath is None:
            print("Dynamic model not found. run: python piezo2mic.py --train-offline and/or python piezo2mic.py --train-realtime")
            sys.exit(1)
    else:
        print("Loading dynamic model from:", dpath)
        dynamic_model = tf.keras.models.load_model(dpath, compile=False)
    generate_demo(model=dynamic_model, out_dir=demo_out_dir, sample_rate=sample_rate)

def main ():
    if len(sys.argv)>1 and sys.argv[1]=="--train-offline":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_OFFLINE_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )
        if resume:
            train_offline(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_offline()
    elif len(sys.argv)>1 and sys.argv[1]=="--train-realtime":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_REALTIME_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )
        if resume:
            train_realtime(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_realtime()
    elif (len(sys.argv)>1 and sys.argv[1]=="--demo"):
        print("Generating demo audio files...")
        if (len(sys.argv) <= 2 or sys.argv[2] !="--realtime-only"):
            prepare_and_generate_demo(
                sample_rate=OFFLINE_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_OFFLINE_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
                build_model_fn=build_offline_model,
                use_safe_weight_copy=False
            )
        if (len(sys.argv) <= 2 or sys.argv[2] !="--offline-only"):
            prepare_and_generate_demo(
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_REALTIME_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
                build_model_fn=build_realtime_model,
                use_safe_weight_copy=True
            )
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        realtime_dyn = fine_tune()
        generate_fine_tune_demo(model=realtime_dyn)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune-demo":
        realtime_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(realtime_dyn_path):
            print("Fine-tuned dynamic model not found, run --fine-tune first")
            sys.exit(1)
        realtime_dyn = None if not os.path.exists(realtime_dyn_path) else tf.keras.models.load_model(realtime_dyn_path, compile=False)
        generate_fine_tune_demo(model=realtime_dyn)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]== "--summarize-offline-hf-usage":
        resume, resume_ckpt_path = cleanup_and_prepare_train(
            logs_out_dir=LOGS_OFFLINE_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )
        if (resume and resume_ckpt_path is not None):
            static_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)
            dynamic_model = build_offline_model(dynamic_time=True)
            dynamic_model.set_weights(static_model.get_weights())
        elif os.path.exists(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)):
            dynamic_model = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)
        else:
            static_model = build_offline_model(dynamic_time=False)
            dynamic_model = build_offline_model(dynamic_time=True)
            dynamic_model.set_weights(static_model.get_weights())
            
        demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
            
        summarize_offline_hf_usage(
            src_path=demo_src,
            tgt_path=demo_tgt,
            style_true_flag=1.0,
            offline_dyn=dynamic_model,
            sample_rate=OFFLINE_SAMPLE_RATE
        )
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--audible-gate-offline":
        # Load offline dynamic or last checkpoint → dynamic
        if os.path.exists(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)):
            offline_dyn = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)
        else:
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_OFFLINE_OUT_DIR, "*.keras")))
            
            if ckpts:
                print("Loading latest checkpoint:", ckpts[-1])
                static = tf.keras.models.load_model(ckpts[-1], compile=False)
                offline_dyn = build_offline_model(dynamic_time=True)
                offline_dyn.set_weights(static.get_weights())
                offline_dyn.save(ckpts[-1] + "_foo.keras")
            else:
                static = build_offline_model(dynamic_time=False)
                offline_dyn = build_offline_model(dynamic_time=True)
                offline_dyn.set_weights(static.get_weights())

        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]
        out_dir = os.path.join(DEMOS_OUT_DIR, "audible_gate_offline")
        
        debug_model = build_offline_debug_model(offline_dyn)

        # debug_model = retune_offline_gate_inplace(
        #     model=debug_model,
        #     attack_ms=GATE_ATTACK_MS,
        #     cutoff_hz=GATE_HP_SIDECHAIN_LOW_CUT_HZ,
        #     hold_ms=GATE_HOLD_MS,
        #     release_ms=GATE_RELEASE_MS,
        #     hp_order=OFFLINE_HF_HP_ORDER,
        #     ratio=GATE_RATIO,
        #     threshold_db=GATE_THRESHOLD_DB
        # )
        
        audible_gate_debug(debug_model=debug_model, path_src=src, path_tgt=tgt, out_dir=out_dir, sample_rate=OFFLINE_SAMPLE_RATE, sec=LONG_DEMO_SEC, offset_sec=LONG_DEMO_OFFSET_SEC, style_mode="true")
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--audible-gate-realtime":
        if os.path.exists(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)):
            realtime_dyn = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME), compile=False)
        else:
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_REALTIME_OUT_DIR, "*.keras")))
            if ckpts:
                print("Loading latest checkpoint:", ckpts[-1])
                static = tf.keras.models.load_model(ckpts[-1], compile=False)
                realtime_dyn = build_realtime_model(dynamic_time=True)
                copy_weights_by_name(realtime_dyn, static)
            else:
                static = build_realtime_model(dynamic_time=False)
                realtime_dyn = build_realtime_model(dynamic_time=True)
                copy_weights_by_name(realtime_dyn, static)
                
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]
        out_dir = os.path.join(DEMOS_OUT_DIR, "audible_gate_realtime")
        debug_model = build_realtime_debug_model(realtime_dyn)
        audible_gate_debug(debug_model=debug_model, path_src=src, path_tgt=tgt, out_dir=out_dir, sample_rate=REALTIME_SAMPLE_RATE, sec=LONG_DEMO_SEC, offset_sec=LONG_DEMO_OFFSET_SEC, style_mode="true")
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()