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
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, ZeroPadding1D, GlobalAveragePooling1D, AveragePooling1D,
    Activation, Multiply, Add, Lambda, Dense, Layer, UpSampling1D, Reshape
)

# Turn off GPU acceleration for inference in demo mode (more reliable on many systems)
if len(sys.argv)>1 and sys.argv[1]=="--demo":
    tf.config.set_visible_devices([], 'GPU')

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

INPUT_AVG_RMS_DBFS_TARGET = -23.0
STYLE_FLAG_PROB_THRESHOLD = 0.5  # threshold for pick vs finger style flag

# Alignment (piezo→mic) settings
ALIGN_ENABLE            = False
ALIGN_MAX_MS            = 5.0     # search range ± this many ms
ALIGN_PREEMPH_ALPHA     = 0.98    # emphasize HF before GCC-PHAT
ALIGN_WIN_SEC           = 1.0     # seconds used for alignment estimation
ALIGN_USE_FRACTIONAL    = True    # apply sub-sample shift via FFT phase ramp

# Debugging params
DEBUG               = True

# Batch params
BATCH_SIZE          = 8

# Chunk ratios
PICK_CHUNK_RATIO = 0.5
FINGER_CHUNK_RATIO = 0.5

CALIBRATION_CHUNK_COUNT = 32

# Offline params
OFFLINE_FILTERS                = 128
OFFLINE_KERNEL_SIZE            = 51
OFFLINE_DILATIONS              = (1,2,4,8,16,32,64,128)
OFFLINE_PRE_BLOCKS             = 2
OFFLINE_PRE_KERNEL             = 7

# Bounded look-ahead per layer (in ms):
OFFLINE_LOOKAHEAD_MS           = 8.0
OFFLINE_LOSS_ONSET_SMOOTH_MS   = 3.0
OFFLINE_GATE_ONSET_SMOOTH_MS   = 5.0
OFFLINE_LOOKAHEAD_SAMPLES      = int(round(OFFLINE_SAMPLE_RATE * OFFLINE_LOOKAHEAD_MS / 1000.0))
OFFLINE_RECEPTIVE_FIELD        = 1 + (OFFLINE_KERNEL_SIZE - 1) * sum(OFFLINE_DILATIONS)
OFFLINE_TRAIN_BLOCK_SIZE       = 4096
OFFLINE_SEG_LEN                = OFFLINE_RECEPTIVE_FIELD + OFFLINE_TRAIN_BLOCK_SIZE
OFFLINE_EMBED_DIM              = 32
OFFLINE_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_offline.keras"
OFFLINE_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_offline_dynamic.keras"

# Realtime params
REALTIME_FILTERS          = 48
REALTIME_KERNEL_SIZE      = 25
REALTIME_DILATIONS        = (1,2,4,8)
REALTIME_PRE_BLOCKS       = 1
REALTIME_PRE_KERNEL       = 5
REALTIME_LOSS_ONSET_SMOOTH_MS = 6.0
REALTIME_GATE_ONSET_SMOOTH_MS  = 6.0
REALTIME_RECEPTIVE_FIELD  = 1 + (REALTIME_KERNEL_SIZE - 1) * sum(REALTIME_DILATIONS)
REALTIME_TRAIN_BLOCK_SIZE = 4096
REALTIME_SEG_LEN          = REALTIME_RECEPTIVE_FIELD + REALTIME_TRAIN_BLOCK_SIZE
REALTIME_EMBED_DIM        = 16
REALTIME_DISTILL_ALPHA    = 5.0

REALTIME_OFFLINE_FEAT_LAYERS   = ["offline_dil_2_film_apply", "offline_dil_3_film_apply"]
REALTIME_FEAT_DISTILL_ALPHA    = 1.0
REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_realtime.keras"
REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_realtime_dynamic.keras"

# Finetune params
FINETUNE_REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_realtime.keras"
FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_realtime_dynamic.keras"

FINETUNE_REALTIME_EPOCHS      = 10
FINETUNE_REALTIME_LR          = 1e-4
FINETUNE_L1_WEIGHT           = 15.0
FINETUNE_MEL_WEIGHT          = 15.0
FINETUNE_LOSS_ONSET_SMOOTH_MS = 6.0

FINETUNE_DEMO_SEC = 60
FINETUNE_DEMO_OFFSET_SEC = 0

# Onset/HF loss shaping
HP_PRE_EMPHASIS           = 0.8
HP_PRE_EMPHASIS_HF        = 0.995
HF_ONSET_GAIN             = 8.0
STFT_HF_ONSET_GAIN        = 5.0
STFT_HIGH_SPLIT_HZ        = 3500.0
STFT_HIGH_BOOST           = 12.0

L1_WEIGHT                 = 1.0
HF_WEIGHT                 = 2.0
STFT_WEIGHT               = 3.0
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
REALTIME_IR_ALIGN_OFFSET_CORRECTION = -64
REALTIME_TOTAL_EXPECTED_IR_DELAY = (REALTIME_IR_FIR_SIZE - 1) + ((REALTIME_IR_FIR_SIZE - 1)//2)

coarse_steps = math.ceil((REALTIME_IR_FIR_SIZE - 1) / REALTIME_IR_DOWNSAMPLE_FACTOR)
IR_PREDELAY = coarse_steps * REALTIME_IR_DOWNSAMPLE_FACTOR

REALTIME_LP_ORDER   = 65
REALTIME_LP_CUTOFF  = 4000

realtime_lp_taps_np = firwin(REALTIME_LP_ORDER,
                     cutoff=REALTIME_LP_CUTOFF/(REALTIME_SAMPLE_RATE/2),
                     window="hamming",
                     pass_zero=True).astype(np.float32)
realtime_lp_taps_tf = tf.constant(realtime_lp_taps_np.reshape(-1,1,1), dtype=tf.float32)

LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

DEMO_SEC        = 10
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

LONG_DEMO_SEC        = 30
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

for dir_path in [DEMO_GENERATOR_OUTPUTS_OUT_DIR, CHECKPOINT_OFFLINE_OUT_DIR, CHECKPOINT_REALTIME_OUT_DIR, LOGS_OFFLINE_OUT_DIR, LOGS_REALTIME_OUT_DIR, DEMO_CHECKPOINTS_OFFLINE_OUT_DIR, DEMO_CHECKPOINTS_REALTIME_OUT_DIR, LONG_DEMOS_OFFLINE_OUT_DIR, LONG_DEMOS_REALTIME_OUT_DIR, FINETUNE_DEMOS_OUT_DIR, MODELS_OFFLINE_OUT_DIR, MODELS_REALTIME_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ────────────────────────────── UTILITIES ───────────────────────────────────

def _extract_epoch_from_ckpt(path):
    fn = os.path.basename(path)
    m = re.search(r'epoch_(\d+)', fn)
    return int(m.group(1)) if m else 0

def pre_emphasis_np(x, α=0.98):
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - α * x[:-1]
    return y

def next_pow2(n):
    return 1<<(int(n-1).bit_length())

def gcc_phat_delay(sig, ref, sr, max_ms=5.0):
    """
    Estimate delay (in samples) where 'ref' lags 'sig' by GCC-PHAT.
    Positive return means ref occurs later (so ref should be advanced).
    """
    n = len(sig)
    m = len(ref)
    L = min(n, m)
    sig = sig[:L]
    ref = ref[:L]

    sig = pre_emphasis_np(sig, ALIGN_PREEMPH_ALPHA)
    ref = pre_emphasis_np(ref, ALIGN_PREEMPH_ALPHA)

    nfft = int(2 ** np.ceil(np.log2(2*L - 1)))
    SIG = np.fft.rfft(sig, nfft)
    REF = np.fft.rfft(ref, nfft)
    R = SIG * np.conj(REF)
    denom = np.abs(R) + 1e-12
    R /= denom

    cc = np.fft.irfft(R, nfft)
    cc = np.concatenate((cc[-(nfft//2):], cc[:(nfft//2)]))
    lags = np.arange(-nfft//2, nfft//2)

    max_lag = int(sr * max_ms / 1000.0)
    mask = (lags >= -max_lag) & (lags <= max_lag)
    cc_sel = cc[mask]
    lags_sel = lags[mask]
    k = np.argmax(np.abs(cc_sel))
    delay = lags_sel[k]

    if 1 <= k < len(cc_sel)-1:
        y0, y1, y2 = cc_sel[k-1], cc_sel[k], cc_sel[k+1]
        denom2 = (y0 - 2*y1 + y2)
        if np.abs(denom2) > 1e-12:
            peak_adj = 0.5*(y0 - y2)/denom2
            delay = delay + peak_adj

    return float(delay)

def shift_signal_int(sig, shift):
    """
    Shift 1-D signal by integer samples. Positive shift advances (moves earlier).
    """
    L = sig.shape[0]
    s = int(np.round(shift))
    if s > 0:
        out = np.empty_like(sig)
        out[:-s] = sig[s:]
        out[-s:] = 0.0
        return out
    elif s < 0:
        s = -s
        out = np.empty_like(sig)
        out[s:] = sig[:-s]
        out[:s] = 0.0
        return out
    else:
        return sig

def shift_signal_frac(sig, delay):
    """
    Fractional shift using FFT phase ramp.
    Positive delay advances the signal (earlier) by 'delay' samples.
    """
    if abs(delay) < 1e-8:
        return sig.copy()
    L = len(sig)
    N = next_pow2(L + 2048)
    X = np.fft.rfft(sig, N)
    k = np.arange(0, len(X), dtype=np.float64)
    phase = np.exp(+1j * 2.0 * np.pi * k * delay / N)
    Y = X * phase
    y = np.fft.irfft(Y, N)
    return y[:L].astype(sig.dtype)

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
target_rms = 10 ** (INPUT_AVG_RMS_DBFS_TARGET / 20.0)

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
    rms_src = np.sqrt(np.mean(mono_src**2) + 1e-12)
    rms_tgt = np.sqrt(np.mean(mono_tgt**2) + 1e-12)
    gain_src = target_rms / rms_src
    gain_tgt = target_rms / rms_tgt
    data_src = data_src * gain_src
    data_tgt = data_tgt * gain_tgt

    if ALIGN_ENABLE:
        L = data_src.shape[0]
        M = min(L, int(ALIGN_WIN_SEC * file_sr))
        if M > 2048:
            delay_samples = gcc_phat_delay(data_src[:M,0], data_tgt[:M,0], file_sr, max_ms=ALIGN_MAX_MS)
            data_tgt_shifted = data_tgt.copy()
            if ALIGN_USE_FRACTIONAL:
                for c in range(data_tgt.shape[1]):
                    data_tgt_shifted[:,c] = shift_signal_frac(data_tgt[:,c], delay_samples)
            else:
                for c in range(data_tgt.shape[1]):
                    data_tgt_shifted[:,c] = shift_signal_int(data_tgt[:,c], delay_samples)
            data_tgt = data_tgt_shifted
            if not DEBUG:
                print(f"[ALIGN] {os.path.basename(src)}: shift target by {delay_samples:.3f} samples ({delay_samples/file_sr*1000:.3f} ms)")
        else:
            print(f"[ALIGN] {os.path.basename(src)}: too short for alignment (L={L}), skipping.")

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

def validate_ir_delay(dynamic_realtime_model):
    fir_down = dynamic_realtime_model.get_layer("realtime_ir_downsample")
    fir_post = dynamic_realtime_model.get_layer("realtime_ir_post_lp")
    align   = dynamic_realtime_model.get_layer("realtime_ir_align")
    expected = int(align.offset)
    factor = REALTIME_IR_DOWNSAMPLE_FACTOR
    N = expected + factor
    N = int(math.ceil(N / factor) * factor)
    x = np.zeros((1, N, 1), dtype=np.float32)
    x[0, 0, 0] = 1.0
    y_down = fir_down(x)
    y_up   = UpSampling1D(size=factor)(y_down)
    y_fir  = fir_post(y_up)
    ir     = y_fir.numpy()[0, :, 0]
    thresh = 1e-6
    nz     = np.where(np.abs(ir) > thresh)[0]
    measured = int(nz[0]) if len(nz) else None
    print(f"FIR-only → expected delay {expected}, measured {measured}")
    if measured != expected:
        print("⚠️  MISMATCH!  adjust RealtimeIrAlign.offset to", measured)
    else:
        print("✅ FIR-delay lines up.")
        
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
class GateFloorProduct(Layer):
    def __init__(self, floor=0.30, gain=0.70, **kwargs):
        super().__init__(**kwargs)
        self.floor = float(floor)
        self.gain  = float(gain)

    def call(self, inputs):
        gate_env, gate_feat = inputs
        return self.floor + self.gain * gate_env * gate_feat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"floor": self.floor, "gain": self.gain})
        return cfg

@register_keras_serializable()
class CombinedLoss(Loss):
    def __init__(self, sample_rate, extra_small_cplx=False, onset_smooth_ms=3.0, name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.extra_small_cplx = bool(extra_small_cplx)
        self.onset_smooth_ms = float(onset_smooth_ms)
    def call(self, y_true, y_pred):
        return combined_loss(y_true, y_pred, self.sample_rate,
                             extra_small_cplx=self.extra_small_cplx,
                             onset_smooth_ms=self.onset_smooth_ms)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "extra_small_cplx": self.extra_small_cplx,
            "onset_smooth_ms": self.onset_smooth_ms
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
                                   split_hz=STFT_HIGH_SPLIT_HZ,
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
def combined_loss(y_true, y_pred, sample_rate, extra_small_cplx=False, onset_smooth_ms=3.0):
    onset_env = onset_envelope(y_true,
                               sample_rate=sample_rate,
                               pre_emph_alpha=HP_PRE_EMPHASIS_HF,
                               smooth_ms=onset_smooth_ms)
    td   = waveform_l1_loss(y_true, y_pred)
    hf   = hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * onset_env)
    spec = mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                          split_hz=STFT_HIGH_SPLIT_HZ,
                                          high_boost=STFT_HIGH_BOOST,
                                          onset_env=onset_env, onset_gain=STFT_HF_ONSET_GAIN)
    mel  = log_mel_loss(y_true, y_pred, sample_rate)
    env  = rms_env_loss(y_true, y_pred)

    cplx_base = complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=256, hop=64, win=256, onset_env=onset_env)
    if extra_small_cplx:
        cplx_64 = complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=64, hop=16, win=64, onset_env=onset_env)
        cplx = 0.5 * (cplx_base + cplx_64)
    else:
        cplx = cplx_base

    d2   = second_diff_l1_onset(y_true, y_pred, onset_env=onset_env)
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

def hp_diff_env(window, α=0.8):
    hp = np.empty_like(window)
    hp[0] = window[0]
    hp[1:] = window[1:] - α * window[:-1]
    return np.abs(np.diff(hp))

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

def analyze_loss_scales(model, dataset, sample_rate):
    # Take one batch from your validation dataset
    batch = next(iter(dataset))
    # Unpack batch: supports (x, y) or (x, y, sw)
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) >= 3:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Unsupported dataset element structure for analyze_loss_scales.")
    else:
        raise ValueError("Unsupported dataset element type for analyze_loss_scales.")

    # Extract the audio target
    if isinstance(y, dict):
        audio_key = model.output_names[0]
        y_true_audio = y[audio_key]
    else:
        y_true_audio = y[0]

    y_pred_audio, _ = model(x, training=False)
    l1_val   = tf.reduce_mean(waveform_l1_loss(y_true_audio, y_pred_audio)).numpy()
    stft_val = tf.reduce_mean(multi_scale_stft_loss(y_true_audio, y_pred_audio)).numpy()
    mel_val  = tf.reduce_mean(log_mel_loss(y_true_audio, y_pred_audio, sample_rate)).numpy()
    env_val  = tf.reduce_mean(rms_env_loss(y_true_audio, y_pred_audio)).numpy()
    hf_val   = tf.reduce_mean(hf_loss(y_true_audio, y_pred_audio)).numpy()
    return {'l1':l1_val, 'stft':stft_val, 'mel':mel_val, 'env':env_val, 'hf':hf_val}



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

def analyze_loss_scales_consistent_batch(model, batch, sample_rate, onset_smooth_ms):
    y_true, y_pred = _extract_audio_io(model, batch)
    onset_env = onset_envelope(
        y_true,
        sample_rate=sample_rate,
        pre_emph_alpha=HP_PRE_EMPHASIS_HF,
        smooth_ms=onset_smooth_ms
    )
    l1   = tf.reduce_mean(waveform_l1_loss(y_true, y_pred)).numpy()
    mel  = tf.reduce_mean(log_mel_loss(y_true, y_pred, sample_rate)).numpy()
    env  = tf.reduce_mean(rms_env_loss(y_true, y_pred)).numpy()
    hf_on  = tf.reduce_mean(hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * onset_env)).numpy()
    stft_on = tf.reduce_mean(
        mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                       split_hz=STFT_HIGH_SPLIT_HZ,
                                       high_boost=STFT_HIGH_BOOST,
                                       onset_env=onset_env,
                                       onset_gain=STFT_HF_ONSET_GAIN)
    ).numpy()
    return {'l1': l1, 'stft': stft_on, 'mel': mel, 'env': env, 'hf': hf_on}

def calibrate_loss_weights_consistent(model, dataset, sample_rate, n_batches=5,
                                      onset_smooth_ms=3.0,
                                      normalize_keys=('l1','stft','mel','env','hf'),
                                      declared=None):
    if declared is None:
        declared = {'l1': L1_WEIGHT, 'stft': STFT_WEIGHT, 'mel': MEL_WEIGHT, 'env': RMS_ENV_WEIGHT, 'hf': HF_WEIGHT}
    sums = {k:0.0 for k in normalize_keys}
    count = 0
    for batch in dataset.take(n_batches):
        sc = analyze_loss_scales_consistent_batch(model, batch, sample_rate, onset_smooth_ms)
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

def generate_demo_from_files (model, out_dir, sample_rate, path_src, path_target, prediction_name="demo_pred", only_prediction=False, offset_sec=LONG_DEMO_OFFSET_SEC, sec=LONG_DEMO_SEC):
    print("\nGenerating demo…")

    # Determine frames in FILE sample-rate, not model SR
    file_sr = sf.info(path_src).samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    FRAMES      = int(round(sec * file_sr))
    
    # if demo_src includes "pick", use the 1.0 as style_flag, otherwise use 0.0
    if "pick" in path_src:
        style_true_flag = 1.0
    elif "finger" in path_src:
        style_true_flag = 0.0
    else:
        # It might be a blended chunk where the styles may be intermixed
        style_true_flag = None
    
    x_long,x_src_sample_rate = sf.read(path_src, start=START_FRAME, frames=FRAMES, dtype='float32', always_2d=True)
    y_long,_  = sf.read(path_target, start=START_FRAME, frames=FRAMES, dtype='float32', always_2d=True)
    x_src = x_long[:,0]
    y_src = y_long[:,0]
    if x_src_sample_rate != sample_rate:
        x_mod = resample_to_model_sr(x_src, x_src_sample_rate, sample_rate)
        y_mod = resample_to_model_sr(y_src, x_src_sample_rate, sample_rate)
    else:
        x_mod = x_src
        y_mod = y_src
    x_demo = x_mod[None, :, None]
    
    # First predict the style using the dynamic model, not based on the true style flag, but on the classifier itself.
    dummy_style = np.zeros((1, 1), dtype=np.float32)
    force_zero  = np.zeros((1, 1), dtype=np.float32)
    y_inf_style_pred, style_pred_out = model.predict([x_demo, dummy_style, force_zero], batch_size=1, verbose=0)
    y_inf_style_pred = y_inf_style_pred[0, :, 0]
    style_pred = float(style_pred_out[0, 0])
    style_pred_name = 'pick' if style_pred > STYLE_FLAG_PROB_THRESHOLD else 'finger'

    # Now, if the true style is known (such as is the case for the base models),
    # also infer the audio, but guided by the true style flag
    if style_true_flag is not None:
        style_true = np.array([[style_true_flag]], dtype=np.float32)
        force_one  = np.ones((1, 1), dtype=np.float32)
        y_inf_style_true, _ = model.predict([x_demo, style_true, force_one], batch_size=1, verbose=0)
        y_inf_style_true = y_inf_style_true[0, :, 0]
        style_true_name = 'pick' if style_true_flag >= STYLE_FLAG_PROB_THRESHOLD else 'finger'
    else:
        y_inf_style_true = None
        style_true_name = None    
    
    style_pred_filepath = os.path.join(out_dir, prediction_name + "_style_pred" + ".wav")
    style_true_filepath = os.path.join(out_dir, prediction_name + "_style_true" + ".wav")

    sf.write(style_pred_filepath, y_inf_style_pred, sample_rate, subtype='FLOAT')
    print(f"✅ Demo saved to {style_pred_filepath}. Predicted style: {style_pred_name}")

    if y_inf_style_true is not None:
        sf.write(style_true_filepath, y_inf_style_true, sample_rate, subtype='FLOAT')
        print(f"✅ Demo saved to {style_true_filepath}. True style: {style_true_name}")

    if (not only_prediction):
        source_filepath = os.path.join(out_dir, "demo_source" +".wav")
        target_filepath = os.path.join(out_dir, "demo_target" + ".wav")
        sf.write(source_filepath, x_mod, sample_rate, subtype='FLOAT')
        sf.write(target_filepath, y_mod, sample_rate, subtype='FLOAT')

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

# Proper band-limited decimate-by-2 layer for 1-D sequences (applies same FIR per channel)
@register_keras_serializable()
class FIRDecimateBy2(Layer):
    def __init__(self, num_taps=63, cutoff=0.47, window="hamming", name=None, **kwargs):
        """
        num_taps: length of the FIR. 63 is a good compromise for decimation-by-2
        cutoff: normalized (0..1) relative to Nyquist of the input signal.
                For decimate-by-2, use something like ~0.47 to leave a margin.
        window: window for firwin.
        """
        super().__init__(name=name, **kwargs)
        self.num_taps = int(num_taps)
        self.cutoff = float(cutoff)
        self.window = str(window)

    def build(self, input_shape):
        # Channels must be known to build a fixed diagonal kernel for tf.nn.conv1d
        c = int(input_shape[-1])
        taps_np = firwin(self.num_taps, cutoff=self.cutoff, window=self.window, pass_zero=True).astype(np.float32)

        # Build (kernel, in_channels, out_channels) so each channel gets identical FIR, no cross-channel mixing
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
        # Time-pad on the left by (num_taps - 1) to keep output aligned with input's start
        pad = self.num_taps - 1
        x_pad = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
        # Stride-by-2 valid convolution = decimation
        y = tf.nn.conv1d(x_pad, filters=self.kernel, stride=2, padding='VALID')
        return y
    
    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        if t is None:
            return (b, None, c)
        # With left pad of (num_taps - 1) and VALID conv, output length is floor(t/2)
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
class RealtimeIrAlign(Layer):
    def __init__(self, offset, **kwargs):
        super().__init__(**kwargs)
        self.offset = int(offset)
    def call(self, inputs):
        ir, main = inputs
        T = tf.shape(main)[1]
        T_ir = tf.shape(ir)[1]
        needed = self.offset + T
        pad_end = needed - T_ir
        ir = tf.cond(
            pad_end > 0,
            lambda: tf.pad(ir, [[0, 0], [0, pad_end], [0, 0]]),
            lambda: ir
        )
        return ir[:, self.offset: self.offset + T, :]
    def compute_output_shape(self, input_shape):
        # (batch, time, channels), match main input time dimension
        T = input_shape[1][1]
        return (None, T, 1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"offset": self.offset})
        return cfg

@register_keras_serializable()
class RealtimeIrDownsampleFIR(Layer):
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
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
        # With left pad of (kernel_size - 1) and stride=self.factor, result is floor(T/factor)
        out_T = T // self.factor
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'factor': self.factor})
        return cfg

@register_keras_serializable()
class RealtimeIrPostLP(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = REALTIME_IR_FIR_SIZE
    def call(self, x):
        pad = self.kernel_size - 1
        x_pad = tf.pad(x, [[0,0], [pad,0], [0,0]])
        return tf.nn.conv1d(x_pad, filters=_ir_fir_up, stride=1, padding='VALID')
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class IRLowpass(Layer):
    def __init__(self, taps=None, **kwargs):
        super().__init__(**kwargs)
        # If taps not provided on deserialization, use module-level constant.
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
        pad_amt = tf.shape(self.taps)[0] - 1  # int32
        paddings = tf.stack([[0, 0], [pad_amt, 0], [0, 0]])
        paddings = tf.cast(paddings, tf.int32)  # ensure correct dtype
        x_pad = tf.pad(x, paddings)
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
class DistillLoss(Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    def call(self, inputs):
        y_t, y_s = inputs
        sq = tf.math.squared_difference(y_t, y_s)
        loss = tf.reduce_mean(sq)
        self.add_loss(self.alpha * loss)
        return y_s
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': self.alpha})
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

# Slice helpers
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def extract_style_scalar(x):
    return x[:, 0:1, 0:1]

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

@register_keras_serializable()
def batch_zeros_like_style(x):
    # Return a (B,1) zeros tensor using batch size from a reference tensor x
    return tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)

# Stabilized FiLM
@register_keras_serializable()
def film_gamma_stable(raw):
    return 1.0 + 0.1 * tf.tanh(raw)
@register_keras_serializable()
def film_beta_stable(raw):
    return 0.1 * tf.tanh(raw)

@register_keras_serializable()
def onset_gate_shape(g, floor=0.15, gain=0.85, eps=1e-6):
    g = tf.clip_by_value(g, eps, 1.0)
    return floor + gain * tf.sqrt(g)


# 1) Replace the shaping function with a less compressive version
@register_keras_serializable()
def onset_gate_shape_linear(g, floor=0.0, gain=1.0, eps=1e-6):
    g = tf.clip_by_value(g, eps, 1.0)
    return floor + gain * g  # no sqrt compression


@register_keras_serializable()
def onset_envelope(y, sample_rate, pre_emph_alpha=0.98, smooth_ms=3.0):
    # High-pass like pre-emphasis
    pe = y[:, 1:, :] - pre_emph_alpha * y[:, :-1, :]
    pe = tf.pad(pe, [[0, 0], [1, 0], [0, 0]])
    env = tf.abs(pe)

    # Explicit smoothing window (in samples)
    win = max(1, int(round(float(smooth_ms) * float(sample_rate) / 1000.0)))
    ker = tf.ones([win, 1, 1], dtype=tf.float32) / float(win)

    env_sm = tf.nn.conv1d(env, ker, stride=1, padding='SAME')
    m = tf.reduce_max(env_sm, axis=1, keepdims=True) + 1e-6
    env_n = tf.clip_by_value(env_sm / m, 0.0, 1.0)
    return env_n

@register_keras_serializable()
class OnsetGate(Layer):
    def __init__(self, sr, alpha=0.98, smooth_ms=3.0, **kwargs):
        super().__init__(**kwargs)
        self.sr = float(sr)
        self.alpha = float(alpha)
        self.smooth_ms = float(smooth_ms)
        self._win = max(1, int(round(self.smooth_ms * self.sr / 1000.0)))

    def call(self, x):
        pe = x[:, 1:, :] - self.alpha * x[:, :-1, :]
        pe = tf.pad(pe, [[0, 0], [1, 0], [0, 0]])
        env = tf.abs(pe)
        ker = tf.ones([self._win, 1, 1], dtype=tf.float32) / float(self._win)
        env_sm = tf.nn.conv1d(env, ker, stride=1, padding='SAME')
        m = tf.reduce_max(env_sm, axis=1, keepdims=True) + 1e-6
        env_n = tf.clip_by_value(env_sm / m, 0.0, 1.0)
        return env_n

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"sr": self.sr, "alpha": self.alpha, "smooth_ms": self.smooth_ms})
        return cfg

def compute_pads(kernel, dilation, lookahead_samples):
    P_tot = (kernel - 1) * dilation
    pad_right = min(int(lookahead_samples), P_tot)
    pad_left  = P_tot - pad_right
    return pad_left, pad_right

def gated_sep_block(x, filters, kernel, dilation, prefix, lookahead=0):
    res = x
    pad_left, pad_right = compute_pads(kernel, dilation, lookahead)
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

# ────────────────────────────── MODEL BUILDERS ───────────────────────────────

def build_offline_model(dynamic_time=False):
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

    # Pre-blocks with look-ahead
    for i in range(OFFLINE_PRE_BLOCKS):
        x = gated_sep_block(x, OFFLINE_FILTERS, OFFLINE_PRE_KERNEL, 1, prefix=f"offline_pre_{i}", lookahead=OFFLINE_LOOKAHEAD_SAMPLES)
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_pre_{i}")

    # Dilated stack with bounded look-ahead
    for idx, d in enumerate(OFFLINE_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(OFFLINE_KERNEL_SIZE, d, OFFLINE_LOOKAHEAD_SAMPLES)
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
    
    gate_env = OnsetGate(sr=OFFLINE_SAMPLE_RATE, alpha=HP_PRE_EMPHASIS_HF,
                     smooth_ms=OFFLINE_GATE_ONSET_SMOOTH_MS, name="offline_onset_gate")(inp)
    gate_env = Lambda(onset_gate_shape_linear, name="offline_onset_gate_shaped")(gate_env)
    
    g_feat = Conv1D(
        32, 5, padding="same", activation="relu",
        kernel_initializer="he_normal", bias_initializer="zeros",
        name="offline_gate_c1"
    )(x)
    g_feat = Conv1D(
        1,  3, padding="same", activation="sigmoid",
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.keras.initializers.Constant(2.5),  # ≈0.924 after sigmoid
        name="offline_gate_c2"
    )(g_feat)

    gate = GateFloorProduct(floor=0.05, gain=0.95, name="offline_onset_gate_total")([gate_env, g_feat])
    
    hf = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_hf_init")(inp)
    
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(inp)
    hf    = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_hf_init")(hf_in)
    
    hf = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b0", lookahead=OFFLINE_LOOKAHEAD_SAMPLES)
    hf = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b0")
    hf = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b1", lookahead=OFFLINE_LOOKAHEAD_SAMPLES)
    hf = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b1")
    hf = gated_sep_block(hf, OFFLINE_FILTERS, kernel=3, dilation=1, prefix="offline_hf_b2", lookahead=OFFLINE_LOOKAHEAD_SAMPLES)
    hf = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b2")
    
    hf = Conv1D(1,1,padding="same", name="offline_hf_out")(hf)
    hf_hp_pre = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_hp_pre")(hf)
    
    hf_gated = Multiply(name="offline_hf_gate")([hf_hp_pre, gate])
    hf_final = IrScale(init=1.2, name="offline_hf_scale")(hf_gated)

    out = Add(name="offline_mix_main")([pick_c, finger_c, hf_final])

    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in], outputs=[out, style_prob], name="piezo2mic_offline")

def summarize_offline_hf_usage(offline_dyn, src_path, tgt_path, sample_rate, style_true_flag=1.0):
    
    def _build_offline_debug_model(offline):
        outs = {
            "gate_total": offline.get_layer("offline_onset_gate_total").output,
            "gate_feat":  offline.get_layer("offline_gate_c2").output,
            "gate_env":   offline.get_layer("offline_onset_gate").output,  # pre-shaped envelope (before shape_linear)
            "hf_pre":     offline.get_layer("offline_hf_out").output,
            "hf_gated":   offline.get_layer("offline_hf_gate").output,
            "hf_scaled":  offline.get_layer("offline_hf_scale").output,
            "pick_c":     offline.get_layer("offline_pick_contrib").output,
            "finger_c":   offline.get_layer("offline_finger_contrib").output,
            "y":          offline.get_layer("offline_mix_main").output,
        }
        return tf.keras.Model(inputs=offline.inputs, outputs=outs, name="offline_debug_model")

    def _hp(x, alpha=0.98):
        # pre-emphasis HP like your hf_loss
        x1 = x[:,1:,:] - alpha * x[:,:-1,:]
        return tf.pad(x1, [[0,0],[1,0],[0,0]])

    def _band_energy(x, sr, fmin=5000.0):
        # crude HF energy via STFT mask
        t = tf.squeeze(x, -1)  # (B,T)
        n_fft = 1024
        hop   = 256
        win   = 1024
        st = tf.signal.stft(t, frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
        mag = tf.abs(st)  # (B, frames, bins)
        freqs = np.linspace(0.0, sr/2.0, n_fft//2 + 1, dtype=np.float32)
        mask = tf.cast(freqs[None,None,:] >= fmin, tf.float32)
        hf_mag = mag * mask
        return tf.reduce_mean(hf_mag)  # scalar
    
    dbg = _build_offline_debug_model(offline_dyn)

    info = sf.info(src_path)
    file_sr = info.samplerate
    # 2 sec slice
    frames = int(2.0 * file_sr)
    x_long,_ = sf.read(src_path, start=0, frames=frames, dtype='float32', always_2d=True)
    y_long,_ = sf.read(tgt_path, start=0, frames=frames, dtype='float32', always_2d=True)
    x = x_long[:,0]; y = y_long[:,0]
    if file_sr != sample_rate:
        x = resample_to_model_sr(x, file_sr, sample_rate)
        y = resample_to_model_sr(y, file_sr, sample_rate)

    x = x[None,:,None].astype(np.float32)
    y = y[None,:,None].astype(np.float32)

    style_true = np.array([[style_true_flag]], dtype=np.float32)
    force_one  = np.ones((1,1), dtype=np.float32)
    outs = dbg.predict([x, style_true, force_one], verbose=0)
    gate = outs["gate_total"][0,:,0]
    hf   = outs["hf_scaled"][0,:,0]
    yhat = outs["y"][0,:,0]
    hp_true = _hp(tf.convert_to_tensor(y)).numpy()[0,:,0]
    hp_pred = _hp(tf.convert_to_tensor(yhat[None,:,None])).numpy()[0,:,0]

    # simple stats
    print(f"Gate_total: mean={gate.mean():.3f}, p95={np.percentile(gate,95):.3f}, p99={np.percentile(gate,99):.3f}, max={gate.max():.3f}")
    print(f"HF out scaled RMS={np.sqrt(np.mean(hf**2)+1e-12):.6f}")

    # onset vs non-onset comparison using gate as a mask
    # top 3% as onsets, bottom 50% as off)
    p_on  = np.percentile(gate, 97.0)
    p_off = np.percentile(gate, 50.0)
    on  = gate >= p_on
    off = gate <= p_off

    def rms(a, m): 
        a = a[m]
        return np.sqrt(np.mean(a*a) + 1e-12) if a.size>0 else 0.0

    print(f"RMS hp_true onsets={rms(hp_true,on):.6f} off={rms(hp_true,off):.6f}")
    print(f"RMS hp_pred onsets={rms(hp_pred,on):.6f} off={rms(hp_pred,off):.6f}")
    print(f"RMS hf_branch onsets={rms(hf,on):.6f} off={rms(hf,off):.6f}")

    # high-band energy
    t_hf_true = float(_band_energy(tf.convert_to_tensor(y), sample_rate, fmin=6000.0).numpy())
    t_hf_pred = float(_band_energy(tf.convert_to_tensor(yhat[None,:,None]), sample_rate, fmin=6000.0).numpy())
    print(f"Avg |STFT| >6kHz: target={t_hf_true:.6f} pred={t_hf_pred:.6f} (ratio={t_hf_pred/(t_hf_true+1e-12):.3f})")

@register_keras_serializable()
def build_realtime_model(offline=None, dynamic_time=False):
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    # Extra offline input only when distilling
    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None,1), name="offline_piezo_input")

    # The known, true style of the input (may or may not be known)
    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")  # 0=predict, 1=ground-truth

    # Style head (realtime side)
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    one_minus_force = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time_used = TileStyle(name="realtime_style_time_used")([style_used_scalar, inp])
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
        pad_left, pad_right = compute_pads(REALTIME_KERNEL_SIZE, d, lookahead=0)
        x   = ZeroPadding1D((pad_left, pad_right), name=f"realtime_pad_{i}")(x)
        g   = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid", name=f"realtime_sep_{i}")(x)
        c   = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2  = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        x   = Multiply(name=f"realtime_gate_mul_{i}")([Activation("tanh", name=f"realtime_tanh_{i}")(c),
                                                       Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        x   = Conv1D(REALTIME_FILTERS, kernel_size=1, padding="same", name=f"realtime_pw_{i}")(x)
        x   = Add(name=f"realtime_res_add_{i}")([x, res])
        x   = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}")

        # Keep track of realtime features that correspond to offline layers
        offline_feat_name = f"offline_dil_{i}_film_apply"
        if REALTIME_OFFLINE_FEAT_LAYERS and offline_feat_name in REALTIME_OFFLINE_FEAT_LAYERS:
            realtime_feats.append(x)

    # Style-gated pick/finger blend
    out_pick   = Conv1D(1,1,padding="same", name="realtime_final_pick")(x)
    out_finger = Conv1D(1,1,padding="same", name="realtime_final_finger")(x)
    inv_style  = InvertStyle(name="realtime_invert_style")(style_time_used)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_time_used, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_realtime = Add(name="realtime_main")([pick_c, finger_c])

    # Low-rate IR Branch (causal)
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
    ir = UpSampling1D(size=REALTIME_IR_DOWNSAMPLE_FACTOR, name="realtime_ir_upsample")(ir)
    ir = RealtimeIrPostLP(name="realtime_ir_post_lp")(ir)
    total_delay = REALTIME_TOTAL_EXPECTED_IR_DELAY
    ir = RealtimeIrAlign(offset=total_delay + REALTIME_IR_ALIGN_OFFSET_CORRECTION, name='realtime_ir_align')([ir, inp])
    ir = IRLowpass(name="realtime_ir_lowpass")(ir)

    # Onset-gated HF branch (causal)
    gate_env = OnsetGate(sr=REALTIME_SAMPLE_RATE, alpha=HP_PRE_EMPHASIS_HF,
                     smooth_ms=REALTIME_GATE_ONSET_SMOOTH_MS, name="realtime_onset_gate")(inp)

    g_feat = Conv1D(
        32, 5, padding="same", activation="relu",
        kernel_initializer="he_normal", bias_initializer="zeros",
        name="realtime_gate_c1"
    )(x)
    g_feat = Conv1D(
        1,  3, padding="same", activation="sigmoid",
        kernel_initializer="glorot_uniform",
        bias_initializer=tf.keras.initializers.Constant(2.5),
        name="realtime_gate_c2"
    )(g_feat)

    gate = GateFloorProduct(floor=0.10, gain=0.90, name="realtime_onset_gate_total")([gate_env, g_feat])
    
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(inp)
    hf    = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_hf_init")(hf_in)
    
    hf = gated_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b0", lookahead=0)
    hf = gated_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b1", lookahead=0)
    hf = gated_sep_block(hf, REALTIME_FILTERS, kernel=3, dilation=1, prefix="realtime_hf_b2", lookahead=0)
    hf = Conv1D(1,1,padding="same", name="realtime_hf_out")(hf)
    
    
    
    hf_hp_pre = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_hp_pre")(hf)
    
    hf_gated = Multiply(name="realtime_hf_gate")([hf_hp_pre, gate])
    hf_final = IrScale(init=1.2, name="realtime_hf_scale")(hf_gated)

    y_realtime = Add(name="realtime_with_ir")([y_realtime, ir, hf_final])

    # Distillation at dual rate
    if offline is not None:
        offline.trainable = False

        # Build a feature model from the offline offline (96 kHz) using the offline input
        if REALTIME_OFFLINE_FEAT_LAYERS:
            offline_feat_model = tf.keras.models.Model(
                inputs=offline.inputs,
                outputs=[offline.get_layer(n).output for n in REALTIME_OFFLINE_FEAT_LAYERS],
                name="offline_feat_model"
            )
            # offline runs at 96 kHz
            T_feats = offline_feat_model([offline_inp, style_true_in, style_force_in])
            if not isinstance(T_feats, list):
                T_feats = [T_feats]

            # Downsample offline features by 2 to match realtime timebase (band-limited)
            for idx, (T_f, S_f) in enumerate(zip(T_feats, realtime_feats)):
                T_f_ds = FIRDecimateBy2(name=f"offline_feat_ds2_{idx}")(T_f)
                chan = int(T_f.shape[-1])
                proj = Conv1D(chan, 1, padding="same", name=f"realtime_proj_feat_{idx}")(S_f)
                _ = DistillLoss(alpha=REALTIME_FEAT_DISTILL_ALPHA, name=f"feat_distill_{idx}")([T_f_ds, proj])

        # Distill from offline audio as well (band-limited decimate-by-2)
        offline_audio, _ = offline([offline_inp, style_true_in, style_force_in])
        offline_audio_ds = FIRDecimateBy2(name="offline_audio_ds2")(offline_audio)
        y_realtime = DistillLoss(alpha=REALTIME_DISTILL_ALPHA, name="distillation")([offline_audio_ds, y_realtime])

    inputs_list = [inp, style_true_in, style_force_in]
    if offline_inp is not None:
        # Keep offline input as a dedicated input only during distillation training
        inputs_list = [inp, offline_inp, style_true_in, style_force_in]

    return tf.keras.models.Model(inputs=inputs_list, outputs=[y_realtime, style_prob], name="piezo2mic_realtime")

def approx_steps(durs, seg_len):
    return 10 if DEBUG else sum(max((d-seg_len)//(seg_len//2)+1,0) for d in durs)//BATCH_SIZE

def approx_steps_from_chunks(chunks, seg_len, model_sample_rate):
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
    return 10 if DEBUG else max(total_windows // BATCH_SIZE, 1)

# ────────────────────────────── TRAINING ─────────────────────────────────────

@register_keras_serializable()
def add_sw_offline(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    style_force = tf.ones((b, 1), dtype=tf.float32)            # 1.0 = use ground-truth
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
    style_force = tf.ones((b, 1), dtype=tf.float32)            # 1.0 = use ground-truth
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
    """
    x_realtime, x_offline = x
    y_audio, style_flag = y
    b = tf.shape(x_realtime)[0]; T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    style_force = tf.ones((b, 1), dtype=tf.float32)            # 1.0 = use ground-truth

    inputs = {
        'realtime_piezo_input':        x_realtime,
        'offline_piezo_input':         x_offline,
        'realtime_style_true_input':   style_true,
        'realtime_style_force_input':  style_force,
    }
    outputs = (y_audio, style_true)

    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)

    # Return sample weights as a tuple to match outputs order [audio, style]
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




def train_shared(
    kind,
    seg_len,
    sample_rate,
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

    steps_per_epoch  = approx_steps_from_chunks(train_chunks, seg_len, sample_rate)
    validation_steps = approx_steps_from_chunks(val_chunks,   seg_len, sample_rate)
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
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    train_ds = train_ds.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.sample_from_datasets(
        [val_pick_ds, val_finger_ds],
        weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO],
        seed=SEED
    ).batch(BATCH_SIZE)
    val_ds = val_ds.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    if resume and resume_ckpt_path:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        static_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        initial_epoch = 0
  
    # Choose smoothing per model kind
    onset_ms_for_loss = OFFLINE_LOSS_ONSET_SMOOTH_MS if kind == 'offline' else REALTIME_LOSS_ONSET_SMOOTH_MS

    # Normalize weights consistently with the actual loss shape
    new_w, _ = calibrate_loss_weights_consistent(
        model=static_model, dataset=val_ds, sample_rate=sample_rate, n_batches=5, onset_smooth_ms=onset_ms_for_loss
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
    _combined_loss = CombinedLoss(sample_rate,
                                extra_small_cplx=(kind == 'offline'),
                                onset_smooth_ms=onset_ms_for_loss)
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
    demo_cb = DemoCallback(sample_rate=sample_rate,
                           build_model_fn=build_model_fn,
                           build_model_fn_args_dynamic=build_model_fn_args_dynamic,
                           demo_checkpoints_out_dir=demo_checkpoints_out_dir,
                           use_safe_weight_copy=(kind != 'offline')
    )
    tb = TensorBoard(log_dir=logs_out_dir, write_graph=False)

    static_model.fit(train_ds,
                     initial_epoch=initial_epoch,
                     epochs=1 if DEBUG else EPOCHS,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=val_ds,
                     validation_steps=validation_steps,
                     callbacks=[es, cp, reduce_lr, tb, demo_cb])

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
    offline = tf.keras.models.load_model(offline_dynamic_model_filepath, compile=False)
    print("Loaded offline from", offline_dynamic_model_filepath)

    tmp_dynamic_model = build_realtime_model(offline=None, dynamic_time=True)
    
    validate_ir_delay(tmp_dynamic_model)
    
    train_shared(
        kind='realtime',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=REALTIME_SEG_LEN,
        sample_rate=REALTIME_SAMPLE_RATE,
        static_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_realtime_model,
        build_model_fn_args_static={ 'offline': offline },
        build_model_fn_args_dynamic={ },
        checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
        logs_out_dir=LOGS_REALTIME_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
        dual_offline=True  # ← enable 96k offline during distillation
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

    target_rms = 10 ** ( INPUT_AVG_RMS_DBFS_TARGET / 20.0 )
    gain_x     = target_rms / (rms_x + 1e-12)
    gain_y     = target_rms / (rms_y + 1e-12)
    print(f"[fine_tune] applying gains: source×{gain_x:.3f}, target×{gain_y:.3f}")

    x_norm = x_full * gain_x
    y_norm = y_full * gain_y

    if ALIGN_ENABLE:
        L = x_norm.shape[0]
        M = min(L, int(ALIGN_WIN_SEC * FINETUNE_SAMPLE_RATE))
        if M > 2048:
            delay_samples = gcc_phat_delay(x_norm[:M,0], y_norm[:M,0], FINETUNE_SAMPLE_RATE, max_ms=ALIGN_MAX_MS)
            y_norm_shifted = y_norm.copy()
            if ALIGN_USE_FRACTIONAL:
                for c in range(y_norm.shape[1]):
                    y_norm_shifted[:,c] = shift_signal_frac(y_norm[:,c], delay_samples)
            else:
                for c in range(y_norm.shape[1]):
                    y_norm_shifted[:,c] = shift_signal_int(y_norm[:,c], delay_samples)
            y_norm = y_norm_shifted
            print(f"[fine_tune ALIGN] shift target by {delay_samples:.3f} samples ({delay_samples/FINETUNE_SAMPLE_RATE*1000:.3f} ms)")

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
        return mixed.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
        model=finetuned_realtime_small, dataset=dsS_val, sample_rate=FINETUNE_SAMPLE_RATE, n_batches=5, onset_smooth_ms=FINETUNE_LOSS_ONSET_SMOOTH_MS
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
    _combined_loss = CombinedLoss(FINETUNE_SAMPLE_RATE,
                                extra_small_cplx=False,
                                onset_smooth_ms=FINETUNE_LOSS_ONSET_SMOOTH_MS)

    finetuned_realtime_small.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_REALTIME_LR),
        loss=[_combined_loss, "binary_crossentropy"],
        loss_weights=[1.0, 0.0],   # disable BCE during finetune
        metrics=[["mae"], []]
    )

    dsS = make_mixed_ds(REALTIME_SEG_LEN)
    dsS = dsS.map(add_sw_realtime_finetune_nf)
    ft_chunks_T = build_chunks(pairs_list=[(src, tgt)], seg_len=REALTIME_SEG_LEN, sample_rate=FINETUNE_SAMPLE_RATE)
    stepsS = max(approx_steps_from_chunks(ft_chunks_T, REALTIME_SEG_LEN, FINETUNE_SAMPLE_RATE), 1)
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
        if not dpath:
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
            
            demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
            
            summarize_offline_hf_usage(
                src_path=demo_src,
                tgt_path=demo_tgt,
                style_true_flag=1.0,
                offline_dyn=dynamic_model,
                sample_rate=OFFLINE_SAMPLE_RATE
            )
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()