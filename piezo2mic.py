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
   The finetuned realtime model distills from base offline model, as there may not necessarily exist a finetuned offline model.
   Before training, the finetuned realtime models are normalized to the average RMS of the base training data (~25 dbFS) to ensure the gate and HF behavior is similar.
   
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

import os, glob, random, shutil, sys, re, math, json, time
import numpy as np
import soundfile as sf
import tensorflow as tf
import scipy.signal as spsig
from ai_edge_litert.interpreter import Interpreter
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
    Activation, Multiply, Add, Lambda, Dense, Layer, Reshape
)

# Determinism
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# On/off flags
DISABLE_SCALE_BOOSTS = False

EPOCHS      = 40
OFFLINE_FINETUNE_EPOCHS = 10
REALTIME_FINETUNE_EPOCHS = 3

FINETUNE_OFFLINE_STEPS_CAP       = 1000
FINETUNE_REALTIME_STEPS_CAP      = 2500

# Audio sample‐rates
OFFLINE_SAMPLE_RATE = 96000
REALTIME_SAMPLE_RATE = 48000

# We need to normalize the amplitude of the finetuned input audio to match that of the training data,
# such that the gate params such as threshold can be shared
#FINETUNE_AVG_RMS_DBFS_TARGET = -26.0
STYLE_FLAG_PROB_THRESHOLD = 0.5  # threshold for pick vs finger style flag

# Debugging params
DEBUG               = False

# Batch params
OFFLINE_BATCH_SIZE          = 4
REALTIME_BATCH_SIZE         = 12
FINETUNE_BATCH_SIZE         = 2

# Chunk ratios
PICK_CHUNK_RATIO = 0.5
FINGER_CHUNK_RATIO = 0.5

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

# Finetune mic-target denoise (preamp hiss cleanup)
FINETUNE_TARGET_DENOISE_ENABLE      = True     # set False to disable
FINETUNE_TARGET_DENOISE_HOP_MS      = 10.0     # STFT hop
FINETUNE_TARGET_DENOISE_BETA        = 1.5      # Wiener oversubtraction strength (>=1)
FINETUNE_TARGET_DENOISE_FLOOR_DB    = -20.0    # floor for attenuation mask (dB)
FINETUNE_TARGET_DENOISE_NOISE_Q     = 0.20     # use lowest 20% energy frames to estimate noise
FINETUNE_TARGET_DENOISE_MIN_FRAMES  = 8        # if fewer than this, fall back to 20th percentile noise profile

OFFLINE_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_offline.keras"
OFFLINE_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_offline_dynamic.keras"

# HF loss shaping
OFFLINE_HP_CUTOFF_HZ   = 2500.0
REALTIME_HP_CUTOFF_HZ  = 2500.0

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

REALTIME_HF_HP_ORDER          = OFFLINE_HF_HP_ORDER

REALTIME_AUDIO_STFT_DISTILL_ALPHA   = 0.6   # KD weight for STFT KD (teacher audio_ds vs student audio)
REALTIME_STEM_PICK_DISTILL_ALPHA    = 0.5   # KD weight for pick stem
REALTIME_STEM_FINGER_DISTILL_ALPHA  = 0.5   # KD weight for finger stem
REALTIME_FEAT_DISTILL_ALPHA         = 1.0

REALTIME_STYLE_BLEND_INIT           = 0.7
REALTIME_STYLE_BLEND_TRAINABLE      = True

REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ   = OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ
REALTIME_GATE_THRESHOLD_DB               = OFFLINE_GATE_THRESHOLD_DB
REALTIME_GATE_RATIO                      = OFFLINE_GATE_RATIO
REALTIME_GATE_ATTACK_MS                  = 0.02
REALTIME_GATE_HOLD_MS                    = 4.0
REALTIME_GATE_RELEASE_MS                 = 20.0

# FILM
REALTIME_FILM_GAMMA_SCALE = 0.5
REALTIME_FILM_BETA_SCALE  = 0.0
OFFLINE_FILM_GAMMA_SCALE = 1.0
OFFLINE_FILM_BETA_SCALE  = 1.0

REALTIME_FILM_GAMMA_MAG = 0.10  # gamma ∈ [1 - 0.10, 1 + 0.10]
REALTIME_FILM_BETA_MAG  = 0.00  # beta ≡ 0 for realtime

OFFLINE_FILM_GAMMA_MAG  = 0.20  # allow a bit more for offline finetune
OFFLINE_FILM_BETA_MAG   = 0.05

# ─── Resonance branch params ─────────────────────────────────────────────

# FIR orders for the band-pass (min-phase)
OFFLINE_PICK_RESO_BP_ORDER = 257
REALTIME_PICK_RESO_BP_ORDER = 257

OFFLINE_PICK_RESO_BOOST_DB    = 0.0 if DISABLE_SCALE_BOOSTS else 8.0
REALTIME_PICK_RESO_BOOST_DB   = 0.0 if DISABLE_SCALE_BOOSTS else 38.0

FINETUNE_OFFLINE_PICK_RESO_BOOST_DB    = 0.0 if DISABLE_SCALE_BOOSTS else 8.0
FINETUNE_REALTIME_PICK_RESO_BOOST_DB   = 0.0 if DISABLE_SCALE_BOOSTS else 8.0


# Frequencies chosen to cover the “weird” pick resonances roughly in 0.7–2.2 kHz.
# Realtime kernels are short to keep left context low; offline can be longer.

PICK_RESO_MODAL_CENTERS_HZ_OFFLINE = [740.0, 880.0, 1040.0, 1200.0, 1380.0, 1580.0, 1800.0, 2050.0, 2200.0]
PICK_RESO_MODAL_DECAYS_MS_OFFLINE  = [ 45.0,  40.0,   36.0,   32.0,   27.0,   24.0,   20.0,   18.0,   16.0]
PICK_RESO_MODAL_LEN_OFFLINE        = 8192

PICK_RESO_MODAL_CENTERS_HZ_REALTIME = [740.0, 880.0, 1040.0, 1200.0, 1380.0, 1580.0, 1800.0, 2050.0, 2200.0]
PICK_RESO_MODAL_DECAYS_MS_REALTIME  = [ 45.0,  40.0,   36.0,   32.0,   27.0,   24.0,   20.0,   20.0,   28.0]
PICK_RESO_MODAL_LEN_REALTIME        = 1024

# Strike envelope smoothing (controls amplitude vs pick intensity)
RESBP_STRIKE_ENV_MS_OFFLINE     = 15.0
RESBP_STRIKE_ENV_MS_REALTIME    = 4.0

# These two represent streaming friendly version of the realtime models
REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_realtime.keras"
REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_realtime_dynamic.keras"

# This represents a trainable version of the realtime model, where the entire model is trainable (e.g. non-streamable), and have the offline teacher baked into their graphs
REALTIME_TRAINABLE_MODEL_FILENAME  = "piezo2mic_baseline_trainable_realtime.keras"

# Pick
OFFLINE_PICK_MASK_ENV_FAST_MS = 0.1
REALTIME_PICK_MASK_ENV_FAST_MS = 0.1

OFFLINE_PICK_MASK_ENV_SLOW_MS = 72.0
REALTIME_PICK_MASK_ENV_SLOW_MS = 72.0

OFFLINE_PICK_MASK_SMOOTH_ATTACK_MS  = 0.1
REALTIME_PICK_MASK_SMOOTH_ATTACK_MS  = 0.1

OFFLINE_PICK_MASK_SMOOTH_RELEASE_MS = 24.0
REALTIME_PICK_MASK_SMOOTH_RELEASE_MS = 24.0

OFFLINE_PICK_MASK_OUT_GAIN = 38.0
REALTIME_PICK_MASK_OUT_GAIN = 38.0

OFFLINE_PICK_MASK_KNEE_DB = 6.0
REALTIME_PICK_MASK_KNEE_DB = 6.0

OFFLINE_PICK_MASK_THRESHOLD_DB = 26.0
REALTIME_PICK_MASK_THRESHOLD_DB = 26.0

OFFLINE_PICK_MASK_POST_COMP_P = 5.0
REALTIME_PICK_MASK_POST_COMP_P = 5.0

OFFLINE_PICK_MASK_POST_COMP_GAIN = 1.0
REALTIME_PICK_MASK_POST_COMP_GAIN = 1.0

OFFLINE_PICK_MASK_FLOOR = 0.0
REALTIME_PICK_MASK_FLOOR = 0.00

OFFLINE_PICK_ATTACK_SOFTCLIP_DRIVE = 2.0
REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE = 2.0

OFFLINE_PICK_RESO_SOFTCLIP_DRIVE = 2.0
REALTIME_PICK_RESO_SOFTCLIP_DRIVE = 2.0

# If we don't apply lowpass penalties during training, there can be excessive low-end build-up in the pick-branches high frequency output.
# So much so that we can't boost their volumes due to excessive energy below 100hz, even sometimes below 30hz
PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ = 3000.0
PICK_ATTACK_LOWBAND_PENALTY_ORDER     = 513
PICK_ATTACK_LOWBAND_PENALTY_ALPHA = 1.5

PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ = 1000.0
PICK_RESO_LOWBAND_PENALTY_ORDER     = 513
PICK_RESO_LOWBAND_PENALTY_ALPHA = 1.5

# Peak protection for pick stems (post user scale)
PICK_LIMITER_HEADROOM_DB = 1.0   # ceiling = -1 dBFS (≈ 0.891)
PICK_LIMITER_DRIVE       = 4.0   # higher = harder knee, still smoother than hard clip

# Also do this from the main mix path, to prevent build-up in ultra low frequencies
ULTRALOW_MAIN_CUTOFF_HZ            = 40.0
ULTRALOW_MAIN_ORDER        = 513
ULTRALOW_MAIN_WEIGHT               = 0.15  # keep modest; adjust by ear/plots

# Finetune params
FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME  = "piezo2mic_finetuned_trainable_realtime.keras"
FINETUNE_REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_realtime.keras"
FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_realtime_dynamic.keras"
FINETUNE_OFFLINE_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_offline.keras"
FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_offline_dynamic.keras"

OFFLINE_TFLITE_T = 40960
REALTIME_TFLITE_T = 6144 * 4
OFFLINE_TFLITE_FP32_FILENAME = f"piezo2mic_baseline_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
REALTIME_TFLITE_FP32_FILENAME = f"piezo2mic_baseline_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"
FINETUNE_OFFLINE_TFLITE_FP32_FILENAME = f"piezo2mic_finetuned_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
FINETUNE_REALTIME_TFLITE_FP32_FILENAME = f"piezo2mic_finetuned_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"

FINETUNE_OFFLINE_LR     = 1e-4
FINETUNE_REALTIME_LR     = 3e-5

FINETUNE_DEMO_SEC = 10
FINETUNE_DEMO_OFFSET_SEC = 0

HP_PRE_EMPHASIS_HF        = 0.98
HF_ONSET_GAIN             = 8.0
STFT_HIGH_BOOST           = 32.0
STFT_HF_ONSET_GAIN        = 12.0

L1_WEIGHT                 = 1.0
HF_WEIGHT                 = 5.0
RESBP_BAND_LOSS_WEIGHT    = 0.8
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

REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR = 8
REALTIME_PICK_RESO_DS_TAPS           = 33
REALTIME_PICK_RESO_INTERP_TAPS       = 65

REALTIME_IR_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 0.0
FINETUNE_REALTIME_IR_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 0.0

REALTIME_LP_ORDER   = 257 

LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

OFFLINE_PICK_ATTACK_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 38.0
REALTIME_PICK_ATTACK_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 24.0

FINETUNE_OFFLINE_PICK_ATTACK_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 32.0
FINETUNE_REALTIME_PICK_ATTACK_BOOST_DB = 0.0 if DISABLE_SCALE_BOOSTS else 32.0

KNOB_DEFAULT_WET_PCT = 100.0

# Broad bands for tone shaping. Use tighter set at 48k, extended top at 96k.

COLOREQ_BANDS_REALTIME_BODY = [
    (80.0,   220.0),
    (220.0,  600.0),
    (600.0,  1500.0),
    (1500.0, 3500.0),
    (3500.0, 8000.0),
    (8000.0, 16000.0)
]

COLOREQ_BANDS_OFFLINE_BODY = [
    (80.0,    220.0),
    (220.0,   600.0),
    (600.0,   1500.0),
    (1500.0,  4000.0),
    (4000.0,  9000.0),
    (9000.0,  18000.0),
    (18000.0, 24000.0),
]

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

DEMO_SEC        = 2
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

LONG_DEMO_SEC        = 2
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

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_OFFLINE_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "offline")
DEMO_CHECKPOINTS_REALTIME_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "realtime")

# Stems demos
STEMS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "stems")
STEMS_OFFLINE_OUT_DIR = os.path.join(STEMS_OUT_DIR, "offline")
STEMS_REALTIME_OUT_DIR = os.path.join(STEMS_OUT_DIR, "realtime")

STEMS_FINETUNE_OUT_DIR = os.path.join(STEMS_OUT_DIR, "finetune")
STEMS_FINETUNE_OFFLINE_OUT_DIR = os.path.join(STEMS_FINETUNE_OUT_DIR, "offline")
STEMS_FINETUNE_REALTIME_OUT_DIR = os.path.join(STEMS_FINETUNE_OUT_DIR, "realtime")

def _default_dryness_value():
    # Clamp to [0,1] and convert wet% -> dry fraction
    try:
        w = float(KNOB_DEFAULT_WET_PCT)
    except Exception:
        w = 100.0
    w = max(0.0, min(100.0, w))
    return float(1.0 - (w / 100.0))

def hp_cutoff_for_sr(sr):
    sr = int(sr)
    if sr == OFFLINE_SAMPLE_RATE:
        return OFFLINE_HP_CUTOFF_HZ
    elif sr == REALTIME_SAMPLE_RATE:
        return REALTIME_HP_CUTOFF_HZ

    raise ValueError(f"Unsupported sample rate: {sr}")

for dir_path in [
    CHECKPOINT_OFFLINE_OUT_DIR, CHECKPOINT_REALTIME_OUT_DIR,
    LOGS_OFFLINE_OUT_DIR, LOGS_REALTIME_OUT_DIR, DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
    DEMO_CHECKPOINTS_REALTIME_OUT_DIR, LONG_DEMOS_OFFLINE_OUT_DIR, LONG_DEMOS_REALTIME_OUT_DIR,
    FINETUNE_DEMOS_OUT_DIR, MODELS_OFFLINE_OUT_DIR, MODELS_REALTIME_OUT_DIR,
    STEMS_OFFLINE_OUT_DIR, STEMS_REALTIME_OUT_DIR,
    STEMS_FINETUNE_OFFLINE_OUT_DIR, STEMS_FINETUNE_REALTIME_OUT_DIR
]:
    os.makedirs(dir_path, exist_ok=True)

def _extract_epoch_from_ckpt(path):
    fn = os.path.basename(path)
    m = re.search(r'epoch_(\d+)', fn)
    return int(m.group(1)) if m else 0

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

@register_keras_serializable()
def apply_runtime_gain(x, knob_scalar, name):
    # knob_scalar: (B,1) → broadcast to (B,1,1) and multiply over time
    k3 = Reshape((1,1), name=f"{name}_expand")(knob_scalar)
    return Multiply(name=f"{name}_mul")([x, k3])


def _len_lr_from_full(L_full, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR):
    return (int(L_full) - 1) // int(factor) + 1

def _decimate_modal_kernels(kernel_full_rev, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR):
    """
    Given trained modal kernels (reversed for conv1d) of shape (L_full,1,M),
    build low-rate reversed kernels by taking every 'factor'-th tap along time.
    Returns shape (L_low,1,M).

    Important: we renormalize each mode to preserve its original L2 norm so the
    post-mix level doesn't collapse after decimation.
    """
    K = np.asarray(kernel_full_rev, dtype=np.float32)
    L_full = K.shape[0]
    M = K.shape[2]
    F = int(factor)
    L_low = (int(L_full) - 1) // F + 1
    idx = np.arange(L_low, dtype=np.int32) * F
    K_low = K[idx, :, :].copy()  # (L_low, 1, M)

    # Renormalize each mode to the original L2 norm
    eps = 1e-12
    for m in range(M):
        orig_l2 = float(np.sqrt(np.sum(K[:, 0, m]**2, dtype=np.float64)))
        low_l2  = float(np.sqrt(np.sum(K_low[:, 0, m]**2, dtype=np.float64)))
        if low_l2 > eps and orig_l2 > eps:
            K_low[:, 0, m] *= (orig_l2 / low_l2)

    return K_low.astype(np.float32)

def _extract_trained_modal_kernel_rt(old_model, legacy_name="realtime_pickreso_modal"):
    """
    Return the trained full-rate modal kernel from an old (legacy) realtime model
    if present; otherwise None.
    """
    try:
        L = old_model.get_layer(legacy_name)
        W = L.get_weights()
        if W and isinstance(W[0], np.ndarray):
            return W[0].astype(np.float32)
    except Exception:
        pass
    return None

@register_keras_serializable()
class StreamingLinearRamp(tf.keras.layers.Layer):
    """
    Streaming linear fade-in from 0→1 over 'ramp_len' samples, with explicit state.
    inputs: [ref (B,T,1), s_count (B,1,1)]
      - 'ref' is only used to get T; it is not modified.
    outputs: [ramp (B,T,1), s_next (B,1,1)]
    """
    def __init__(self, ramp_len, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ramp_len = int(max(1, ramp_len))

    def call(self, inputs):
        ref, s_count = inputs
        T = tf.shape(ref)[1]                         # time length this hop
        # previous count (how many samples already emitted, clamped to ramp_len)
        c0 = tf.cast(tf.squeeze(s_count, axis=[1,2]), tf.int32)  # (B,)
        c0c = tf.clip_by_value(c0, 0, self.ramp_len)

        # ramp indices for this hop: c0 + [1..T]
        idx = tf.range(T, dtype=tf.int32)[None, :] + c0c[:, None] + 1
        r = tf.clip_by_value(tf.cast(idx, ref.dtype) / float(self.ramp_len), 0.0, 1.0)
        ramp = r[:, :, None]                         # (B,T,1)

        # next state: clamp to ramp_len so it stops at 1.0 after ramp_len samples
        c_next = tf.clip_by_value(c0c + T, 0, self.ramp_len)
        s_next = tf.reshape(tf.cast(c_next, ref.dtype), (-1, 1, 1))
        return [ramp, s_next]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ramp_len": self.ramp_len})
        return cfg

@register_keras_serializable()
class StreamingScalarEMA(tf.keras.layers.Layer):
    """
    Streaming EMA for a scalar per-batch value (shape (B,1)).
    inputs: [x (B,1), s_prev (B,1)] → outputs: [y (B,1), s_next (B,1)]
    y = alpha * s_prev + (1-alpha) * x
    """
    def __init__(self, alpha=0.98, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        x, s_prev = inputs
        a = tf.constant(self.alpha, dtype=x.dtype)
        y = a * s_prev + (1.0 - a) * x
        return [y, y]  # y is also the next state

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha})
        return cfg
    
@register_keras_serializable()
class PickResoDownsampleFIR(Layer):
    """
    Decimate by integer factor with a short, causal min-phase FIR (for pickreso branch).
    """
    def __init__(self, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_DS_TAPS, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)

    def build(self, input_shape):
        cutoff = 0.45 / float(self.factor)
        taps = firwin(self.num_taps, cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
        taps_min = minimum_phase(taps, method='homomorphic').astype(np.float32)
        taps_min = taps_min / max(np.sum(taps_min), 1e-12)
        k = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        self.kernel = self.add_weight(
            name="ds_lpf_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=self.factor, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        out_T = (T + self.factor - 1) // self.factor  # ceil(T/factor)
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": self.factor, "num_taps": self.num_taps})
        return cfg

@register_keras_serializable()
class LowrateModalBank(tf.keras.layers.Layer):
    """
    Trainable low-rate modal bank: conv1d with kernel shape (L_low,1,M).
    Initialized by decimating the default modal kernels or by a provided kernel_init.
    """
    def __init__(self, num_modes, length_lr, kernel_init=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_modes  = int(num_modes)
        self.length_lr  = int(length_lr)
        self._kernel_init_np = None if kernel_init is None else np.asarray(kernel_init, dtype=np.float32)

    def build(self, input_shape):
        if self._kernel_init_np is None:
            K_full = _make_modal_kernels(PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
                                         PICK_RESO_MODAL_DECAYS_MS_REALTIME,
                                         PICK_RESO_MODAL_LEN_REALTIME,
                                         REALTIME_SAMPLE_RATE)
            K_lr = _decimate_modal_kernels(K_full, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
        else:
            K_lr = self._kernel_init_np

        # Ensure expected shape (length_lr,1,num_modes) with center-crop or symmetric pad
        K_lr = np.asarray(K_lr, dtype=np.float32)
        want = (self.length_lr, 1, self.num_modes)
        if K_lr.shape != want:
            L_have = K_lr.shape[0]
            if L_have > self.length_lr:
                # Center-crop
                drop = L_have - self.length_lr
                start = drop // 2
                end = start + self.length_lr
                K_lr = K_lr[start:end, :, :]
            elif L_have < self.length_lr:
                # Symmetric pad on time axis
                pad = self.length_lr - L_have
                pL = pad // 2
                pR = pad - pL
                K_lr = np.pad(K_lr, ((pL, pR), (0,0), (0,0)), mode='constant')
        assert K_lr.shape == want, f"LowrateModalBank kernel shape mismatch: {K_lr.shape} != {want}"

        self.kernel = self.add_weight(
            name="modal_kernels_lr",
            shape=want,
            initializer=tf.keras.initializers.Constant(K_lr.astype(np.float32)),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        return (b, t, self.num_modes)

    def get_config(self):
        return {
            "num_modes": self.num_modes,
            "length_lr": self.length_lr,
            **super().get_config()
        }

@register_keras_serializable()
class CausalPickDBMask(tf.keras.layers.Layer):
    """
    Causal pick detector in dB domain with soft-knee logistic mapping.

    Pipeline:
      x → HPF → |.| → fast EMA = env_hi
      ref = (|HPF(x)| slow EMA)  if ref_is_hp else (|x| slow EMA)
      R_db = 20*log10(env_hi+eps) - 20*log10(ref+eps)
      m0   = sigmoid(slope * (R_db - thr_db)), where slope ~ ln(9)/knee_db (so ±knee/2 spans ~10%-90%)
      asym attack/release smoothing on m0
      optional post compression: y = 1 - (1 - m)^p with makeup
      optional floor: y = floor + (1 - floor) * y

    Notes:
      - thr_db around 0 dB means mask ≈ 0.5 when HF and ref are equal.
      - knee_db controls softness; 8–14 dB tends to sound natural.
      - ref_is_hp=True measures contrast of fast HF vs slower HF (recommended).
        If False, ref is broadband slow env (more conservative).
    """
    def __init__(self,
                 sample_rate,
                 hp_cut_hz=3500.0,
                 hp_order=129,
                 env_fast_ms=0.25,
                 env_slow_ms=10.0,
                 thr_db=0.0,
                 knee_db=12.0,
                 smooth_attack_ms=0.06,
                 smooth_release_ms=6.0,
                 ref_is_hp=True,
                 post_comp_p=1.6,
                 post_comp_gain=1.15,
                 floor=0.02,
                 out_gain=1.0,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate       = int(sample_rate)
        self.hp_cut_hz         = float(hp_cut_hz)
        self.hp_order          = int(hp_order)
        self.env_fast_ms       = float(env_fast_ms)
        self.env_slow_ms       = float(env_slow_ms)
        self.thr_db            = float(thr_db)
        self.knee_db           = float(knee_db)
        self.smooth_attack_ms  = float(smooth_attack_ms)
        self.smooth_release_ms = float(smooth_release_ms)
        self.ref_is_hp         = bool(ref_is_hp)
        self.post_comp_p       = float(post_comp_p)
        self.post_comp_gain    = float(post_comp_gain)
        self.floor             = float(floor)
        self.out_gain          = float(out_gain)

    def build(self, input_shape):
        # HPF taps
        k_hp = _get_minphase_hp_kernel(self.sample_rate, self.hp_cut_hz, self.hp_order)
        self.k_hp = self.add_weight(
            name="dbmask_hp_taps", shape=k_hp.shape,
            initializer=tf.keras.initializers.Constant(k_hp), trainable=False
        )

        # EMA taps (reversed for conv1d)
        def ema_ms(ms, scale, Lmax):
            smpls = max(1, int(round((float(ms)/1000.0) * self.sample_rate)))
            return _exp_smoothing_kernel(smpls, scale=scale, Lmax=Lmax)

        k_fast = ema_ms(self.env_fast_ms, scale=3.0, Lmax=1024)
        k_slow = ema_ms(self.env_slow_ms, scale=3.0, Lmax=4096)
        k_att  = ema_ms(self.smooth_attack_ms,  scale=4.0, Lmax=2048)
        k_rel  = ema_ms(self.smooth_release_ms, scale=4.0, Lmax=2048)

        self.k_fast = self.add_weight(
            name="dbmask_env_fast", shape=k_fast.shape,
            initializer=tf.keras.initializers.Constant(k_fast), trainable=False
        )
        self.k_slow = self.add_weight(
            name="dbmask_env_slow", shape=k_slow.shape,
            initializer=tf.keras.initializers.Constant(k_slow), trainable=False
        )
        self.k_att  = self.add_weight(
            name="dbmask_att", shape=k_att.shape,
            initializer=tf.keras.initializers.Constant(k_att), trainable=False
        )
        self.k_rel  = self.add_weight(
            name="dbmask_rel", shape=k_rel.shape,
            initializer=tf.keras.initializers.Constant(k_rel), trainable=False
        )

        # Precompute sigmoid slope so that ±knee/2 → ~0.1/0.9
        # slope ≈ ln(9)/knee_db
        slope_val = float(np.log(9.0) / max(self.knee_db, 1e-3))
        self.slope = self.add_weight(
            name="dbmask_sigmoid_slope", shape=(1,),
            initializer=tf.keras.initializers.Constant([slope_val]), trainable=False
        )
        self.thr_db_w = self.add_weight(
            name="dbmask_thr_db", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.thr_db]), trainable=False
        )

        # Output gain as a weight so you can tweak via set_weights if needed
        self.gain = self.add_weight(
            name="dbmask_out_gain", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.out_gain]), trainable=False
        )

        super().build(input_shape)

    def call(self, x):
        # x: (B,T,1)
        eps = tf.constant(1e-8, dtype=x.dtype)

        # Sidechain: HPF → abs
        hp     = _causal_conv1d(x, self.k_hp)
        abs_hp = tf.abs(hp)

        # Envelopes
        env_hi = _causal_conv1d(abs_hp, self.k_fast)
        if self.ref_is_hp:
            env_lo = _causal_conv1d(abs_hp, self.k_slow)
        else:
            env_lo = _causal_conv1d(tf.abs(x), self.k_slow)

        # Contrast in dB
        # R_db = 20 * log10(env_hi+eps) - 20*log10(env_lo+eps)
        ln10 = tf.math.log(tf.constant(10.0, dtype=x.dtype))
        r_db = (20.0 / ln10) * (tf.math.log(env_hi + eps) - tf.math.log(env_lo + eps))

        # Soft-knee logistic around thr_db with slope tied to knee_db
        z  = (r_db - self.thr_db_w[0]) * self.slope[0]
        m0 = tf.math.sigmoid(z)

        # Asym attack/release smoothing
        prev   = tf.pad(m0, [[0,0],[1,0],[0,0]])[:, :-1, :]
        rising = tf.cast(m0 > prev, x.dtype)
        m_att  = _causal_conv1d(m0, self.k_att)
        m_rel  = _causal_conv1d(m0, self.k_rel)
        m1     = tf.clip_by_value(rising * m_att + (1.0 - rising) * m_rel, 0.0, 1.0)

        # Optional post compression (saturating), then floor and final gain
        if self.post_comp_p != 1.0:
            p = tf.constant(self.post_comp_p, dtype=x.dtype)
            y = 1.0 - tf.pow(1.0 - m1, p)
            if self.post_comp_gain != 1.0:
                y = tf.clip_by_value(tf.constant(self.post_comp_gain, dtype=x.dtype) * y, 0.0, 1.0)
        else:
            y = m1

        if self.floor > 0.0:
            fl = tf.constant(self.floor, dtype=x.dtype)
            y = fl + (1.0 - fl) * y

        return tf.clip_by_value(y * self.gain[0], 0.0, 1.0)

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "hp_cut_hz": self.hp_cut_hz,
            "hp_order": self.hp_order,
            "env_fast_ms": self.env_fast_ms,
            "env_slow_ms": self.env_slow_ms,
            "thr_db": self.thr_db,
            "knee_db": self.knee_db,
            "smooth_attack_ms": self.smooth_attack_ms,
            "smooth_release_ms": self.smooth_release_ms,
            "ref_is_hp": self.ref_is_hp,
            "post_comp_p": self.post_comp_p,
            "post_comp_gain": self.post_comp_gain,
            "floor": self.floor,
            "out_gain": self.out_gain,
            **super().get_config()
        }

@register_keras_serializable()
def _pick_mask_engine(inp,
                         sample_rate,
                         prefix,
                         hp_order=129,
                         env_fast_ms=0.25,
                         env_slow_ms=10.0,
                         thr_db=0.0,
                         knee_db=12.0,
                         smooth_attack_ms=0.06,
                         smooth_release_ms=6.0,
                         ref_is_hp=True,
                         post_comp_p=1.6,
                         post_comp_gain=1.15,
                         floor=0.02,
                         out_gain=1.0):
    """
    Replacement for _pick_mask_engine using dB-domain detector + soft-knee + post compression.
    """
    return CausalPickDBMask(
        sample_rate=sample_rate,
        hp_cut_hz=hp_cutoff_for_sr(sample_rate),
        hp_order=hp_order,
        env_fast_ms=env_fast_ms,
        env_slow_ms=env_slow_ms,
        thr_db=thr_db,
        knee_db=knee_db,
        smooth_attack_ms=smooth_attack_ms,
        smooth_release_ms=smooth_release_ms,
        ref_is_hp=bool(ref_is_hp),
        post_comp_p=post_comp_p,
        post_comp_gain=post_comp_gain,
        floor=floor,
        out_gain=out_gain,
        name=f"{prefix}_pick_mask_db"
    )(inp)

@register_keras_serializable()
class SoftClip1D(tf.keras.layers.Layer):
    """
    Per-sample soft clip: y = tanh(drive * x) / tanh(drive) * limit
    Use limit<=1 to bound, drive>0 controls hardness.
    """
    def __init__(self, drive=2.0, limit=0.98, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.drive = float(drive)
        self.limit = float(limit)
    def call(self, x):
        d = tf.constant(self.drive, dtype=x.dtype)
        lim = tf.constant(self.limit, dtype=x.dtype)
        return lim * tf.math.tanh(d * x) / tf.math.tanh(d)
    def get_config(self):
        return {"drive": self.drive, "limit": self.limit, **super().get_config()}

@register_keras_serializable()
class LearnedModalBank(tf.keras.layers.Layer):
    def __init__(self, centers_hz, decays_ms, length, sample_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.centers_hz = [float(x) for x in centers_hz]
        self.decays_ms  = [float(x) for x in decays_ms]
        self.length     = int(length)
        self.sample_rate= int(sample_rate)

    def build(self, input_shape):
        K = _make_modal_kernels(self.centers_hz, self.decays_ms, self.length, self.sample_rate)
        self.kernel = self.add_weight(
            name="modal_kernels_trainable",
            shape=K.shape,
            initializer=tf.keras.initializers.Constant(K),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        shp = tf.shape(x); b = shp[0]; t = shp[1]; c = tf.reduce_prod(shp[2:])
        x3 = tf.reshape(x, [b, t, c])
        pad = self.length - 1
        x_pad = tf.pad(x3, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.centers_hz))

    def get_config(self):
        return {
            "centers_hz": self.centers_hz,
            "decays_ms":  self.decays_ms,
            "length":     self.length,
            "sample_rate":self.sample_rate,
            **super().get_config()
        }

def _pick_attack(prefix,
                 hf_in,
                 mask,
                 style_time,
                 sample_rate,
                 filters=16,
                 softclip_drive=2.0,
                 lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                 lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                 lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER):
    """
    Tiny causal HF synthesis gated by the analytic pick mask, then style-gated,
    then a runtime HPF safety-net, then final scale.

    Returns:
      - y:    post-style, post-HPF, post-scale stem named '{prefix}_pickattack_scale'
      - patt: training-only lowband penalty attached to the post-scale stem
    """
    # Mask-gated input for the tiny CNN
    x = Multiply(name=f"{prefix}_picklearn_gate")([hf_in, mask])
    x = causal_sep_conv(x, filters, 7, 1, name=f"{prefix}_picklearn_c1", activation="relu")
    x = causal_sep_conv(x, filters, 7, 2, name=f"{prefix}_picklearn_c2", activation="relu")
    x = Conv1D(1, 1, padding="same", name=f"{prefix}_picklearn_out")(x)

    # Style gating
    xg = Multiply(name=f"{prefix}_pickattack_gated")([x, style_time])
    
    xg = SoftClip1D(drive=softclip_drive, limit=0.98, name=f"{prefix}_pickattack_softclip")(xg)

    # Runtime HPF safety-net (cut below penalty_cutoff_hz)
    xh = HFHighpass(sample_rate=sample_rate,
                    cutoff_hz=lowband_penalty_cutoff_hz,
                    order=lowband_penalty_order,
                    name=f"{prefix}_pickattack_post_hp")(xg)

    # Final scale (stems are POST-HPF as requested)
    y = IrScale(init=1.0, name=f"{prefix}_pickattack_scale")(xh)
    
    # Post-scale peak protection with ≥1 dB headroom
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE,
                   limit=limit_lin,
                   name=f"{prefix}_pickattack_limit")(y)

    # Training-only lowband penalty on the post-scale signal
    patt = HFLowbandPenalty(alpha=lowband_penalty_alpha,
                            cutoff_hz=lowband_penalty_cutoff_hz,
                            order=lowband_penalty_order,
                            sample_rate=sample_rate,
                            name=f"{prefix}_pickattack_lowband_penalty")(y)
    return y, patt

def _reson_learn_offline(inp, mask, style_time):
    sr = OFFLINE_SAMPLE_RATE
    onset = Lambda(gate_onset, name="offline_pickreso_onset")(mask)
    strike = CausalExpSmoother(time_const_ms=RESBP_STRIKE_ENV_MS_OFFLINE, sample_rate=sr, Lmax=4096,
                               name="offline_pickreso_onset_pulse")(onset)
    ana = BandpassFIR(low_hz=0.9*min(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE),
                      high_hz=1.1*max(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE),
                      order=OFFLINE_PICK_RESO_BP_ORDER, sample_rate=sr,
                      name="offline_pickreso_ctrl_bp")(inp)
    amp = CausalExpSmoother(time_const_ms=4.0, sample_rate=sr, Lmax=1024,
                            name="offline_pickreso_ctrl_env")(Abs1D()(ana))
    amp = Lambda(clip_01x4, name="offline_pickreso_ctrl_gain")(amp)
    exc = Multiply(name="offline_pickreso_excite")([strike, amp])

    modes = LearnedModalBank(centers_hz=PICK_RESO_MODAL_CENTERS_HZ_OFFLINE,
                             decays_ms =PICK_RESO_MODAL_DECAYS_MS_OFFLINE,
                             length    =PICK_RESO_MODAL_LEN_OFFLINE,
                             sample_rate=sr,
                             name="offline_pickreso_modal")(exc)
    mixed = Conv1D(1, 1, use_bias=True, padding="same", name="offline_pickreso_mix")(modes)

    # Style-gate first
    xg = Multiply(name="offline_pickreso_gated")([mixed, style_time])

    # Then runtime HPF safety-net at the requested cutoff
    xh = HFHighpass(sample_rate=sr,
                    cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                    order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                    name="offline_pickreso_post_hp")(xg)
    
    xh = SoftClip1D(drive=OFFLINE_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="offline_pickreso_softclip")(xh)

    # Final scale (POST-HPF)
    y = IrScale(init=1.0, name="offline_pickreso_scale")(xh)
    
    # Post-scale peak protection with ≥1 dB headroom
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE,
                   limit=limit_lin,
                   name="offline_pickreso_limit")(y)

    # Training-only lowband penalty on the post-scale stem
    patt = HFLowbandPenalty(alpha=PICK_RESO_LOWBAND_PENALTY_ALPHA,
                            cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                            order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                            sample_rate=OFFLINE_SAMPLE_RATE,
                            name="offline_pickreso_lowband_penalty")(y)
    return y, patt

def _reson_learn_realtime(inp, mask, style_time):
    sr = REALTIME_SAMPLE_RATE
    onset = Lambda(gate_onset, name="realtime_pickreso_onset")(mask)
    strike = CausalExpSmoother(time_const_ms=RESBP_STRIKE_ENV_MS_REALTIME, sample_rate=sr, Lmax=512,
                               name="realtime_pickreso_onset_pulse")(onset)
    ana = BandpassFIR(low_hz=0.9*min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME),
                      high_hz=1.1*max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME),
                      order=REALTIME_PICK_RESO_BP_ORDER, sample_rate=sr,
                      name="realtime_pickreso_ctrl_bp")(inp)
    amp = CausalExpSmoother(time_const_ms=3.0, sample_rate=sr, Lmax=256,
                            name="realtime_pickreso_ctrl_env")(Abs1D()(ana))
    amp = Lambda(clip_01x4, name="realtime_pickreso_ctrl_gain")(amp)
    exc = Multiply(name="realtime_pickreso_excite")([strike, amp])

    # DS → low-rate modal bank → 1x1 mix → upsample → match T
    ds = PickResoDownsampleFIR(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_DS_TAPS,
                               name="realtime_pickreso_ds8")(exc)
    num_modes = len(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    L_lr = _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    modes_lr = LowrateModalBank(num_modes=num_modes, length_lr=L_lr,
                                name="realtime_pickreso_modal_lr")(ds)
    mixed_lr = Conv1D(1, 1, use_bias=True, padding="same", name="realtime_pickreso_mix")(modes_lr)

    up = FIRInterpByN(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_INTERP_TAPS,
                      cutoff=0.45/REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, window=('kaiser', 9.0),
                      name="realtime_pickreso_interp")(mixed_lr)
    mixed = MatchTimeToRef(name="realtime_pickreso_matchlen")([up, inp])

    xg = Multiply(name="realtime_pickreso_gated")([mixed, style_time])
    xh = HFHighpass(sample_rate=sr,
                    cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                    order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                    name="realtime_pickreso_post_hp")(xg)
    xh = SoftClip1D(drive=REALTIME_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickreso_softclip")(xh)
    y = IrScale(init=1.0, name="realtime_pickreso_scale")(xh)

    # Post-scale peak protection
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=limit_lin, name="realtime_pickreso_limit")(y)

    # NEW: calibration scale (migration will set its weight so knob dB → same loudness as legacy)
    y = IrScale(init=1.0, name="realtime_pickreso_calib")(y)

    patt = HFLowbandPenalty(alpha=PICK_RESO_LOWBAND_PENALTY_ALPHA,
                            cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                            order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                            sample_rate=REALTIME_SAMPLE_RATE,
                            name="realtime_pickreso_lowband_penalty")(y)
    return y, patt

@register_keras_serializable()
class ZeroBias(constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)
    def get_config(self):
        return {}
    
    
# ─────────────────────────────────────────────────────────────────────────────
# NEW helper ops/layers for streaming, Keras-serializable, TFLite-friendly
# ─────────────────────────────────────────────────────────────────────────────

@register_keras_serializable()
def _take_last(x, n):
    return x[:, -int(n):, :]

@register_keras_serializable()
class ConcatTime(tf.keras.layers.Layer):
    """Concat [state, x] along time axis."""
    def call(self, inputs):
        s, x = inputs
        return tf.concat([s, x], axis=1)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class TakeLastN(tf.keras.layers.Layer):
    def __init__(self, n, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n = int(n)
    def call(self, x):
        return _take_last(x, self.n)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n": self.n})
        return cfg

@register_keras_serializable()
class StreamFixedFIR(tf.keras.layers.Layer):
    """
    Generic streaming fixed FIR with explicit state.
    Inputs: [x, state] where state has shape (L-1, C_in)
    Kernel shape: (L, 1, C_out) (time-reversed taps for conv1d)
    Returns: [y (B,T,C_out), state_next (B,L-1,C_in)]
    If taps_init is provided (np.array or tf.Tensor of shape (L,1,C_out)), it is used
    as the initializer instead of zeros. This avoids 'silent branch' when no copy-by-name occurs.
    """
    def __init__(self, kernel_len, out_ch, weight_name, trainable_kernel=False, taps_init=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kernel_len = int(kernel_len)
        self.out_ch = int(out_ch)
        self.weight_name = str(weight_name)
        self.trainable_kernel = bool(trainable_kernel)
        # Keep init only in memory; do not serialize to JSON config
        if taps_init is None:
            self._taps_init_np = None
        else:
            if tf.is_tensor(taps_init):
                self._taps_init_np = taps_init.numpy().astype(np.float32)
            else:
                self._taps_init_np = np.asarray(taps_init, dtype=np.float32)

    def build(self, input_shape):
        if self._taps_init_np is not None:
            k = self._taps_init_np
        else:
            k = np.zeros((self.kernel_len, 1, self.out_ch), dtype=np.float32)

        # Safety: enforce expected shape if out_ch==1 and a scalar kernel provided
        if k.shape != (self.kernel_len, 1, self.out_ch):
            # Try to reshape/broadcast if possible
            try:
                k = np.reshape(k, (self.kernel_len, 1, self.out_ch)).astype(np.float32)
            except Exception:
                # fallback to zeros with warning
                k = np.zeros((self.kernel_len, 1, self.out_ch), dtype=np.float32)

        self.kernel = self.add_weight(
            name=self.weight_name,
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=self.trainable_kernel
        )
        super().build(input_shape)

    def call(self, inputs):
        x, state = inputs  # x: (B,T,C_in), state: (B,L-1,C_in)
        T = tf.shape(x)[1]
        x_cat = tf.concat([state, x], axis=1)  # (B, L-1+T, C_in)
        y_full = tf.nn.conv1d(x_cat, filters=self.kernel, stride=1, padding='VALID')  # ideally (B, T, C_out)
        y = y_full[:, -T:, :]
        s_next = x_cat[:, - (self.kernel_len - 1):, :]
        return [y, s_next]

    def compute_output_shape(self, input_shapes):
        x_shp, _ = input_shapes
        b, t, _ = x_shp
        return [(b, t, self.out_ch), (b, self.kernel_len - 1, x_shp[2])]

    def get_config(self):
        # do not serialize taps_init (weights persist in the saved model anyway)
        return {
            "kernel_len": self.kernel_len,
            "out_ch": self.out_ch,
            "weight_name": self.weight_name,
            "trainable_kernel": self.trainable_kernel,
            **super().get_config()
        }

@register_keras_serializable()
class StreamingRealtimeIrDownsampleFIR(tf.keras.layers.Layer):
    """
    Streaming decimate-by-N with causal FIR.
    If use_ir_prototype=True, use the exact same taps as the non-streaming IR branch
    (global _ir_fir_kernel), guaranteeing equivalence for the IR path.
      inputs: [x (B,T,1), state (B,L-1,1)]
      outputs: [y_ds (B, floor(T/N), 1), state_next (B,L-1,1)]
    """
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, kernel_size=REALTIME_IR_FIR_SIZE, use_ir_prototype=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.kernel_size = int(kernel_size)
        self.use_ir_prototype = bool(use_ir_prototype)

    def build(self, input_shape):
        if self.use_ir_prototype:
            k = _ir_fir_kernel.astype(np.float32)  # (L,1,1), already reversed for conv1d
        else:
            cutoff = 0.45 / float(self.factor)
            taps = firwin(self.kernel_size, cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
            taps_min = minimum_phase(taps, method='homomorphic').astype(np.float32)
            taps_min = taps_min / max(np.sum(taps_min), 1e-12)
            k = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)

        self.kernel = self.add_weight(
            name="ds_lpf_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        x, state = inputs  # x: (B,T,1), state: (B,L-1,1)
        x_cat = tf.concat([state, x], axis=1)  # (B, L-1+T, 1)
        y = tf.nn.conv1d(x_cat, filters=self.kernel, stride=self.factor, padding='VALID')  # (B, floor(T/N), 1)
        s_next = x_cat[:, - (tf.shape(self.kernel)[0] - 1):, :]
        return [y, s_next]

    def get_config(self):
        return {"factor": self.factor, "kernel_size": self.kernel_size, "use_ir_prototype": self.use_ir_prototype, **super().get_config()}

@register_keras_serializable()
class StreamingFIRInterpByN(tf.keras.layers.Layer):
    """
    Streaming version of FIRInterpByN:
      inputs: [x_low (B,Tl,C), state_low (B,L-1,C_low=1)]
      outputs: [y_up (B,Tl*F,C), state_next_low (B,L-1,C_low)]
    It zero-inserts at low rate, filters, and discards the first F*(L-1) samples,
    returning exactly Tl*F samples (aligned with new low-rate input).
    """
    def __init__(self, factor, num_taps=257, cutoff=None, window=('kaiser', 9.0), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)
        self.cutoff = float(0.45 / self.factor) if cutoff is None else float(cutoff)
        self.window = window

    def build(self, input_shape):
        win = self.window
        try:
            if not isinstance(win, (str, tuple)):
                win = tuple(win)
        except TypeError:
            win = 'hann'
        try:
            taps = firwin(self.num_taps, cutoff=self.cutoff, window=win, pass_zero=True).astype(np.float32)
        except Exception:
            taps = firwin(self.num_taps, cutoff=self.cutoff, window=('kaiser', 9.0), pass_zero=True).astype(np.float32)
        taps = taps / np.maximum(np.sum(taps), 1e-12)
        k = taps[::-1].reshape(-1, 1, 1).astype(np.float32)
        self.kernel = self.add_weight(
            name="fir_interp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        x_low, state_low = inputs
        # concat low-rate state
        x_cat = tf.concat([state_low, x_low], axis=1)  # (B, L-1 + Tl, 1)

        # Zero insertion (upsample) at factor
        b = tf.shape(x_cat)[0]
        t = tf.shape(x_cat)[1]
        c = tf.shape(x_cat)[2]
        x4 = tf.reshape(x_cat, [b, t, 1, c])                        # (B,t,1,C)
        pad_zeros = tf.zeros([b, t, self.factor - 1, c], x_cat.dtype)
        xz = tf.concat([x4, pad_zeros], axis=2)                     # (B,t,F,C)
        xu = tf.reshape(xz, [b, t * self.factor, c])                # (B, t*F, C)

        # Valid conv with left alignment → upsampled
        L = self.num_taps
        xu_pad = tf.pad(xu, [[0,0],[L-1,0],[0,0]])
        yu = tf.nn.conv1d(xu_pad, filters=self.kernel, stride=1, padding='VALID')  # (B, t*F, C)

        # Discard first factor * (L-1) samples (they correspond to the prepended low-rate state)
        left_drop = self.factor * (L - 1)
        yu_eff = yu[:, left_drop:, :]

        # Keep exactly new low-rate contribution length = Tl*F
        Tl = tf.shape(x_low)[1]
        out_len = Tl * self.factor
        yu_eff = yu_eff[:, :out_len, :]

        # Next low-rate state = last (L-1) low-rate samples from x_cat
        s_next = x_cat[:, - (L - 1):, :]

        return [yu_eff, s_next]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "factor": self.factor,
            "num_taps": self.num_taps,
            "cutoff": self.cutoff,
            "window": self.window
        })
        return cfg

@register_keras_serializable()
class StreamingPickDBMask(tf.keras.layers.Layer):
    """
    Streaming analytic pick detector in dB domain with soft-knee + post compression.
    Now includes a 1-sample continuity state for 'prev(m0)' so rising/decay selection
    is seamless across hops.

    Inputs:
      [ x, s_hp, s_fast, s_slow, s_att, s_rel, s_prev_m0 ]
        - x: (B,T,1)
        - s_*: streaming FIR states (B,L-1,1)
        - s_prev_m0: (B,1,1) last m0 sample from previous hop

    Outputs:
      [ y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_m0_n ]
        - y: (B,T,1) mask in [0,1]
        - s_prev_m0_n: (B,1,1) last m0 of this hop
    """
    def __init__(self,
                 sample_rate,
                 hp_cut_hz=3500.0,
                 hp_order=129,
                 env_fast_ms=0.25,
                 env_slow_ms=10.0,
                 thr_db=0.0,
                 knee_db=12.0,
                 smooth_attack_ms=0.06,
                 smooth_release_ms=6.0,
                 ref_is_hp=True,
                 post_comp_p=1.6,
                 post_comp_gain=1.15,
                 floor=0.02,
                 out_gain=1.0,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate       = int(sample_rate)
        self.hp_cut_hz         = float(hp_cut_hz)
        self.hp_order          = int(hp_order)
        self.env_fast_ms       = float(env_fast_ms)
        self.env_slow_ms       = float(env_slow_ms)
        self.thr_db            = float(thr_db)
        self.knee_db           = float(knee_db)
        self.smooth_attack_ms  = float(smooth_attack_ms)
        self.smooth_release_ms = float(smooth_release_ms)
        self.ref_is_hp         = bool(ref_is_hp)
        self.post_comp_p       = float(post_comp_p)
        self.post_comp_gain    = float(post_comp_gain)
        self.floor             = float(floor)
        self.out_gain          = float(out_gain)

    def build(self, input_shape):
        k_hp = _get_minphase_hp_kernel(self.sample_rate, self.hp_cut_hz, self.hp_order)
        self.k_hp = self.add_weight(
            name="dbmask_hp_taps", shape=k_hp.shape,
            initializer=tf.keras.initializers.Constant(k_hp), trainable=False
        )
        def ema_ms(ms, scale, Lmax):
            smpls = max(1, int(round((float(ms)/1000.0) * self.sample_rate)))
            return _exp_smoothing_kernel(smpls, scale=scale, Lmax=Lmax)
        k_fast = ema_ms(self.env_fast_ms, scale=3.0, Lmax=1024)
        k_slow = ema_ms(self.env_slow_ms, scale=3.0, Lmax=4096)
        k_att  = ema_ms(self.smooth_attack_ms,  scale=4.0, Lmax=2048)
        k_rel  = ema_ms(self.smooth_release_ms, scale=4.0, Lmax=2048)

        self.k_fast = self.add_weight(
            name="dbmask_env_fast", shape=k_fast.shape,
            initializer=tf.keras.initializers.Constant(k_fast), trainable=False
        )
        self.k_slow = self.add_weight(
            name="dbmask_env_slow", shape=k_slow.shape,
            initializer=tf.keras.initializers.Constant(k_slow), trainable=False
        )
        self.k_att  = self.add_weight(
            name="dbmask_att", shape=k_att.shape,
            initializer=tf.keras.initializers.Constant(k_att), trainable=False
        )
        self.k_rel  = self.add_weight(
            name="dbmask_rel", shape=k_rel.shape,
            initializer=tf.keras.initializers.Constant(k_rel), trainable=False
        )

        slope_val = float(np.log(9.0) / max(self.knee_db, 1e-3))
        self.slope = self.add_weight(
            name="dbmask_sigmoid_slope", shape=(1,),
            initializer=tf.keras.initializers.Constant([slope_val]), trainable=False
        )
        self.thr_db_w = self.add_weight(
            name="dbmask_thr_db", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.thr_db]), trainable=False
        )
        self.gain = self.add_weight(
            name="dbmask_out_gain", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.out_gain]), trainable=False
        )
        super().build(input_shape)

    def _stream_conv(self, x, k, s):
        x_cat = tf.concat([s, x], axis=1)  # (B, L-1+T, C)
        y = tf.nn.conv1d(x_cat, filters=k, stride=1, padding='VALID')
        T = tf.shape(x)[1]
        y = y[:, -T:, :]
        s_next = x_cat[:, tf.shape(x_cat)[1] - (tf.shape(k)[0] - 1):, :]
        return y, s_next

    def call(self, inputs):
        # Backward-compat: accept 6 or 7 inputs
        if len(inputs) == 6:
            x, s_hp, s_fast, s_slow, s_att, s_rel = inputs
            s_prev_m0 = None
        else:
            x, s_hp, s_fast, s_slow, s_att, s_rel, s_prev_m0 = inputs

        eps = tf.constant(1e-8, dtype=x.dtype)

        hp, s_hp_n = self._stream_conv(x, self.k_hp, s_hp)
        abs_hp = tf.abs(hp)

        env_hi, s_fast_n = self._stream_conv(abs_hp, self.k_fast, s_fast)
        if self.ref_is_hp:
            env_lo, s_slow_n = self._stream_conv(abs_hp, self.k_slow, s_slow)
        else:
            abs_x = tf.abs(x)
            env_lo, s_slow_n = self._stream_conv(abs_x, self.k_slow, s_slow)

        ln10 = tf.math.log(tf.constant(10.0, dtype=x.dtype))
        r_db = (20.0 / ln10) * (tf.math.log(env_hi + eps) - tf.math.log(env_lo + eps))
        z  = (r_db - self.thr_db_w[0]) * self.slope[0]
        m0 = tf.math.sigmoid(z)

        # Continuous prev across hops
        if s_prev_m0 is None:
            prev = tf.pad(m0, [[0,0],[1,0],[0,0]])[:, :-1, :]
        else:
            prev = tf.concat([s_prev_m0, m0[:, :-1, :]], axis=1)
        s_prev_m0_n = m0[:, -1:, :]

        rising = tf.cast(m0 > prev, x.dtype)
        m_att, s_att_n = self._stream_conv(m0, self.k_att, s_att)
        m_rel, s_rel_n = self._stream_conv(m0, self.k_rel, s_rel)
        m1 = tf.clip_by_value(rising * m_att + (1.0 - rising) * m_rel, 0.0, 1.0)

        if self.post_comp_p != 1.0:
            p = tf.constant(self.post_comp_p, dtype=x.dtype)
            y = 1.0 - tf.pow(1.0 - m1, p)
            if self.post_comp_gain != 1.0:
                y = tf.clip_by_value(tf.constant(self.post_comp_gain, dtype=x.dtype) * y, 0.0, 1.0)
        else:
            y = m1

        if self.floor > 0.0:
            fl = tf.constant(self.floor, dtype=x.dtype)
            y = fl + (1.0 - fl) * y

        y = tf.clip_by_value(y * self.gain[0], 0.0, 1.0)

        if s_prev_m0 is None:
            return [y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n]
        return [y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_m0_n]

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "hp_cut_hz": self.hp_cut_hz,
            "hp_order": self.hp_order,
            "env_fast_ms": self.env_fast_ms,
            "env_slow_ms": self.env_slow_ms,
            "thr_db": self.thr_db,
            "knee_db": self.knee_db,
            "smooth_attack_ms": self.smooth_attack_ms,
            "smooth_release_ms": self.smooth_release_ms,
            "ref_is_hp": self.ref_is_hp,
            "post_comp_p": self.post_comp_p,
            "post_comp_gain": self.post_comp_gain,
            "floor": self.floor,
            "out_gain": self.out_gain,
            **super().get_config()
        }

@register_keras_serializable()
class StreamingDiffPos(tf.keras.layers.Layer):
    """
    Streaming positive first difference with 1-sample continuity state:
      inputs: [x (B,T,1), s_prev (B,1,1)]
      outputs: [y (B,T,1), s_next (B,1,1)]
      y[n] = relu(x[n] - x[n-1]) with x[-1] coming from s_prev.
    """
    def call(self, inputs):
        x, s_prev = inputs
        prev = tf.concat([s_prev, x[:, :-1, :]], axis=1)
        y = tf.nn.relu(x - prev)
        s_next = x[:, -1:, :]
        return [y, s_next]

    def get_config(self):
        return super().get_config()
    
@register_keras_serializable()
class PreEmphasisStream(tf.keras.layers.Layer):
    """
    Streaming 1st-order pre-emphasis with explicit state:
      inputs: [x (B,T,1), s (B,1,1)] where s is previous x[-1]
      outputs: [y (B,T,1), s_next (B,1,1)]
      y[n] = x[n] - alpha * x[n-1]
    """
    def __init__(self, alpha=HP_PRE_EMPHASIS_HF, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        x, s = inputs
        prev = tf.concat([s, x[:, :-1, :]], axis=1)
        y = x - tf.constant(self.alpha, dtype=x.dtype) * prev
        s_next = x[:, -1:, :]
        return [y, s_next]

    def get_config(self):
        return {"alpha": self.alpha, **super().get_config()}


# ─────────────────────────────────────────────────────────────────────────────
        
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

        # Determine kind from sample rate
        if int(self.sample_rate) == int(OFFLINE_SAMPLE_RATE):
            kind = "offline"
        elif int(self.sample_rate) == int(REALTIME_SAMPLE_RATE):
            kind = "realtime"
        else:
            kind = "realtime"

        # Helper to filter a feed dict to the model’s actual input names
        model_input_names = [t.name.split(':')[0].split('/')[-1] for t in self.dynamic_model.inputs]
        def filter_feed(d):
            return {k: v for k, v in d.items() if k in model_input_names}

        # Compute slicing at file rate
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

        # Knobs
        knob_feeds = _prepare_knob_feeds(kind=kind, use_finetuned=False)

        # Predicted-style pass
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)
        base_feed = {
            f"{kind}_piezo_input": x_demo,
            f"{kind}_style_true_input": dummy_style,
            f"{kind}_style_force_input": force_zero,
        }
        base_feed.update(knob_feeds)
        feed = filter_feed(base_feed)

        y_inf_style_pred, style_pred_out = self.dynamic_model.predict(feed, batch_size=1, verbose=0)
        y_inf_style_pred = y_inf_style_pred[0, :, 0]
        style_pred = float(style_pred_out[0, 0])
        style_pred_name = 'pick' if style_pred > STYLE_FLAG_PROB_THRESHOLD else 'finger'

        # True-style pass if known
        if self.style_true_flag is not None:
            style_true = np.array([[self.style_true_flag]], dtype=np.float32)
            force_one  = np.ones((1, 1), dtype=np.float32)
            tf_feed = {
                f"{kind}_piezo_input": x_demo,
                f"{kind}_style_true_input": style_true,
                f"{kind}_style_force_input": force_one,
            }
            tf_feed.update(knob_feeds)
            feed_true = filter_feed(tf_feed)
            y_inf_style_true, _ = self.dynamic_model.predict(feed_true, batch_size=1, verbose=0)
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
class FinetuneCombinedLossLite(tf.keras.losses.Loss):
    """
    Lightweight loss for strict finetune:
      L = l1_w * L1 + hf_w * HF(preemph) + stft_w * 1-scale MR-STFT + env_w * RMS-env
          + ulb_w * L1( LPF_{<ulb_cut}(y_pred - y_true) )

    Single STFT scale by default to keep it very cheap.
    """
    def __init__(self,
                 sample_rate,
                 stft_scales=((1024, 256, 1024),),  # single scale by default
                 l1_w=1.0, hf_w=2.0, stft_w=1.0, env_w=0.0,
                 onset_gain=HF_ONSET_GAIN,
                 ulb_w=0.0,
                 ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                 ulb_order=None,
                 name="finetune_combined_loss_lite",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = int(sample_rate)
        self.stft_scales = tuple((int(n), int(h), int(w)) for (n, h, w) in stft_scales)
        self.l1_w  = float(l1_w)
        self.hf_w  = float(hf_w)
        self.stft_w= float(stft_w)
        self.env_w = float(env_w)
        self.onset_gain = float(onset_gain)

        # ULB term
        self.ulb_w     = float(ulb_w)
        self.ulb_cut_hz= float(ulb_cut_hz)
        if ulb_order is None:
            self.ulb_order = int(ULTRALOW_MAIN_ORDER)
        else:
            self.ulb_order = int(ulb_order)

        # Precompute Hann windows (shape (1,1,win)) as constants
        self._wins = []
        for _, _, win in self.stft_scales:
            w = tf.signal.hann_window(win, periodic=True)
            w = tf.reshape(w, (1, 1, win))
            self._wins.append(tf.constant(w, dtype=tf.float32))

        # LPF kernel for ultra-lowband error (as a constant)
        self._ulb_kernel = _make_lpf_kernel(self.sample_rate,
                                            cutoff_hz=self.ulb_cut_hz,
                                            order=self.ulb_order,
                                            window="hamming")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "stft_scales": self.stft_scales,
            "l1_w": self.l1_w,
            "hf_w": self.hf_w,
            "stft_w": self.stft_w,
            "env_w": self.env_w,
            "onset_gain": self.onset_gain,
            "ulb_w": self.ulb_w,
            "ulb_cut_hz": self.ulb_cut_hz,
            "ulb_order": self.ulb_order
        })
        return cfg

    def _mrstft(self, y_true, y_pred, gate_env):
        t = tf.squeeze(y_true, axis=-1)
        p = tf.squeeze(y_pred, axis=-1)

        losses = []
        for (n_fft, hop, win), w in zip(self.stft_scales, self._wins):
            ft = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)  # (B,frames,win)
            fp = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
            ftw = ft * w
            fpw = fp * w
            st_t = tf.signal.rfft(ftw, fft_length=[n_fft])
            st_p = tf.signal.rfft(fpw, fft_length=[n_fft])
            mag_t = tf.abs(st_t)
            mag_p = tf.abs(st_p)
            diff = tf.abs(mag_t - mag_p)  # (B,frames,F)

            if gate_env is not None:
                ge = tf.squeeze(gate_env, axis=-1)
                gef = tf.signal.frame(ge, frame_length=win, frame_step=hop, pad_end=True)
                ge_w = tf.reduce_mean(gef, axis=-1, keepdims=True)
                diff = diff * (1.0 + self.onset_gain * ge_w)

            losses.append(tf.reduce_mean(diff, axis=[1, 2]))  # (B,)

        loss = tf.add_n(losses) / float(len(losses))
        return loss[:, None, None]

    def _ulb(self, y_true, y_pred):
        if self.ulb_w == 0.0:
            # Shape-compatible zero so downstream weighting is clean
            return tf.zeros_like(y_true)
        err = y_pred - y_true
        k = self._ulb_kernel  # (L,1,1), already reversed for conv1d
        pad = tf.shape(k)[0] - 1
        err_pad = tf.pad(err, [[0,0],[pad,0],[0,0]])
        lp_err = tf.nn.conv1d(err_pad, filters=k, stride=1, padding='VALID')
        return tf.abs(lp_err)

    def call(self, y_true, y_pred):
        gate_env = gate_weight_dynamic(y_true, self.sample_rate)  # (B,T,1) in [0,1]
        td = waveform_l1_loss(y_true, y_pred) * (0.2 + 0.8 * gate_env)
        hf = hf_loss(y_true, y_pred) * (1.0 + self.onset_gain * gate_env)
        spec = self._mrstft(y_true, y_pred, gate_env) if self.stft_scales else 0.0
        env = rms_env_loss(y_true, y_pred) if self.env_w != 0.0 else 0.0
        ulb = self._ulb(y_true, y_pred)

        return (self.l1_w  * td
              + self.hf_w  * hf
              + self.stft_w* spec
              + self.env_w * env
              + self.ulb_w * ulb)

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
def band_limited_l1_loss_weighted(y_true, y_pred, sample_rate,
                                  low_hz, high_hz,
                                  order_off=OFFLINE_PICK_RESO_BP_ORDER, order_rt=REALTIME_PICK_RESO_BP_ORDER,
                                  onset_boost=4.0,
                                  thr_db=-60.0,
                                  ratio=6.0,
                                  attack_ms=0.3,
                                  hold_ms=20.0,
                                  release_ms=200.0):
    """
    L1 in 0.7–2 kHz, weighted by an envelope computed from the band-passed TARGET,
    so training emphasizes true resonance events even if the piezo has none.
    Returns (B,T,1).
    """
    order = order_off if int(sample_rate) == OFFLINE_SAMPLE_RATE else order_rt
    k = _get_minphase_bp_kernel(sample_rate, low_hz, high_hz, order)

    t_bp = _causal_conv1d(y_true, k)
    p_bp = _causal_conv1d(y_pred, k)

    # Build a resonance envelope from the TARGET band, not from the piezo HF gate.
    g = gate_gain_signal(
        t_bp,
        sample_rate=sample_rate,
        threshold_db=thr_db,
        ratio=ratio,
        attack_ms=attack_ms,
        hold_ms=hold_ms,
        release_ms=release_ms,
        cutoff_hz=max(200.0, 0.5 * low_hz),  # HP sidechain near band’s lower edge
        hp_order=129,
        env_ms=1.0,
        depth_smooth_ms=20.0,
        env_lmax=512,
        depth_lmax=512
    )

    return tf.abs(t_bp - p_bp) * (1.0 + onset_boost * g)

@register_keras_serializable()
def resonance_band_loss(y_true, y_pred, sample_rate):
    if int(sample_rate) == OFFLINE_SAMPLE_RATE:
        lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE)
        hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE)
        order = OFFLINE_PICK_RESO_BP_ORDER
    else:
        lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
        hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
        order = REALTIME_PICK_RESO_BP_ORDER
    return band_limited_l1_loss_weighted(
        y_true, y_pred, sample_rate,
        low_hz=lo, high_hz=hi,
        order_off=order, order_rt=order,
        onset_boost=5.0,  # you can lift this to emphasize pick onsets more
        thr_db=-60.0, ratio=6.0, attack_ms=0.3, hold_ms=20.0, release_ms=200.0
    )

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
                                   split_hz=OFFLINE_HP_CUTOFF_HZ,
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
                                          split_hz=hp_cutoff_for_sr(sample_rate),
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
    bl   = resonance_band_loss(y_true, y_pred, sample_rate)
    
    ulb = lowband_error_penalty(y_true, y_pred, sample_rate,
                                cutoff_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                order=ULTRALOW_MAIN_ORDER)

    return (L1_WEIGHT       * td
          + HF_WEIGHT       * hf
          + STFT_WEIGHT     * spec
          + MEL_WEIGHT      * mel
          + RMS_ENV_WEIGHT  * env
          + COMPLEX_STFT_WEIGHT * cplx
          + D2_WEIGHT       * d2
          + RESBP_BAND_LOSS_WEIGHT * bl
          + ULTRALOW_MAIN_WEIGHT   * ulb)

# Replace mean_hf_metric with this version
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    vals = hf_loss(y_true, y_pred)  # (B, T, 1)
    # Return per-sample metric with shape (B, 1, 1), compatible with (B, T, 1) sample weights
    return tf.reduce_mean(vals, axis=[1, 2], keepdims=True)

# ────────────────────────────── DATA PIPELINE HELPERS ───────────────────────

@register_keras_serializable()
class Abs1D(tf.keras.layers.Layer):
    def call(self, x):
        return tf.abs(x)

def _safe_get_attr(obj, attr, default=None):
    try:
        return getattr(obj, attr)
    except Exception:
        return default

def _get_layer_config_val(layer, key, default=None):
    try:
        return layer.get_config().get(key, default)
    except Exception:
        return default

def _find_layers(model, cls=None, name_prefix=None, name_regex=None):
    layers = []
    for l in model.layers:
        if cls is not None and not isinstance(l, cls):
            continue
        if name_prefix is not None and not l.name.startswith(name_prefix):
            continue
        if name_regex is not None:
            if re.match(name_regex, l.name) is None:
                continue
        layers.append(l)
    return layers

def _sum_sep_rf_for_prefix(model, name_regex):
    """
    Sum receptive field increments (k-1)*d across SeparableConv1D layers
    whose names match name_regex (e.g., r'^realtime_resbp_lr_b\\d+_sep$').
    Returns rf_samples = 1 + sum((k-1)*d)  (in 'the stack’s native rate').
    """
    rf_inc = 0
    for l in model.layers:
        if not isinstance(l, SeparableConv1D):
            continue
        if re.match(name_regex, l.name) is None:
            continue
        ks = _safe_get_attr(l, 'kernel_size', None)
        ds = _safe_get_attr(l, 'dilation_rate', None)
        k = int(ks[0]) if ks is not None else int(_get_layer_config_val(l, 'kernel_size', 1))
        d = int(ds[0]) if ds is not None else int(_get_layer_config_val(l, 'dilation_rate', 1))
        rf_inc += max(0, (k - 1) * d)
    return 1 + rf_inc

def precache_finetune_resampled_pair(x_full, y_full, file_sr, sr_rt=REALTIME_SAMPLE_RATE, sr_off=OFFLINE_SAMPLE_RATE):
    """
    Pre-resample the normalized pair once for finetune to avoid per-batch resample_poly.
    x_full,y_full: float32 arrays (N,C).
    Returns dict with mono arrays at 96k and 48k.
    """
    x_m = x_full[:, 0].astype(np.float32, copy=False)
    y_m = y_full[:, 0].astype(np.float32, copy=False)

    def to_sr(sig, src_sr, dst_sr):
        return sig if int(src_sr) == int(dst_sr) else resample_to_model_sr(sig, int(src_sr), int(dst_sr))

    x96 = to_sr(x_m, file_sr, sr_off).astype(np.float32, copy=False)
    y96 = to_sr(y_m, file_sr, sr_off).astype(np.float32, copy=False)
    x48 = to_sr(x_m, file_sr, sr_rt).astype(np.float32, copy=False)
    y48 = to_sr(y_m, file_sr, sr_rt).astype(np.float32, copy=False)
    return {"x96": x96, "y96": y96, "x48": x48, "y48": y48}

@register_keras_serializable()
def causal_sep_conv(x, filters, kernel, dilation, name, activation=None,
                    bias_constraint=None, bias_initializer=None):
    pad_left = (int(kernel) - 1) * int(dilation)
    x = ZeroPadding1D((pad_left, 0), name=f"{name}_pad")(x)
    x = SeparableConv1D(
        filters=filters,
        kernel_size=kernel,
        dilation_rate=dilation,
        padding="valid",
        bias_constraint=bias_constraint,
        bias_initializer=bias_initializer,
        name=name
    )(x)
    if activation:
        x = Activation(activation, name=f"{name}_{activation}")(x)
    return x

def segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks,
                                       factor=2, min_amp=SEGMENT_GENERATOR_MIN_AMP_THRESH):
    """
    Simple dual-rate generator backed by already-resampled arrays (48k and 96k).
    - seg_len_rt: segment length at realtime rate (48k).
    - x48, y48: mono float32 arrays at 48k.
    - x96: mono float32 array at 96k (teacher input).
    - is_pick_chunks: bool; unused for finetune labels (style disabled), but kept for symmetry.
    """
    T48 = len(x48)
    L96 = seg_len_rt * int(factor)
    while True:
        if T48 <= 1:
            raise RuntimeError("Input array too short for segment_generator_from_arrays_dual")
        s48 = random.randrange(max(T48 - seg_len_rt, 1))
        e48 = s48 + seg_len_rt

        xs48 = x48[s48:e48]
        ys48 = y48[s48:e48]

        if np.max(np.abs(xs48)) < float(min_amp):
            continue

        s96 = s48 * int(factor)
        e96 = s96 + L96
        xs96 = x96[s96:e96]

        # Pad to exact lengths if needed
        if len(xs48) < seg_len_rt:
            p = seg_len_rt - len(xs48)
            xs48 = np.pad(xs48, (0, p))
            ys48 = np.pad(ys48, (0, p))
        if len(xs96) < L96:
            p2 = L96 - len(xs96)
            xs96 = np.pad(xs96, (0, p2))

        style_flag = 1.0 if is_pick_chunks else 0.0  # ignored downstream (style loss disabled)
        yield (xs48[:, None].astype(np.float32),
               xs96[:, None].astype(np.float32)), \
              (ys48[:, None].astype(np.float32),
               np.float32(style_flag))

def segment_generator_from_arrays(seg_len, x, y, is_pick_chunks, min_amp=SEGMENT_GENERATOR_MIN_AMP_THRESH):
    """
    Simple, fast generator backed by already-resampled arrays.
    """
    T = len(x)
    while True:
        if T <= 1:
            raise RuntimeError("Input array too short for segment_generator_from_arrays")
        s = random.randrange(max(T - seg_len, 1))
        xs = x[s:s + seg_len]
        ys = y[s:s + seg_len]
        if np.max(np.abs(xs)) < float(min_amp):
            continue
        if len(xs) < seg_len:
            pad = seg_len - len(xs)
            xs = np.pad(xs, (0, pad))
            ys = np.pad(ys, (0, pad))
        style_flag = 1.0 if is_pick_chunks else 0.0
        yield xs[:, None].astype(np.float32), (ys[:, None].astype(np.float32), np.float32(style_flag))

def make_mixed_ds_from_arrays_realtime_dual(x48, x96, y48, batch_size, seg_len_rt=REALTIME_SEG_LEN):
    """
    Dual-rate dataset for finetuning realtime model with a 96k teacher.
    Returns a dataset of ((x48, x96), (y48, style_flag)).
    """
    factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE
    assert factor >= 2 and OFFLINE_SAMPLE_RATE == 96000 and REALTIME_SAMPLE_RATE == 48000, \
        "Expected 96k offline and 48k realtime rates with integer ratio."

    sig_x = (
        tf.TensorSpec((seg_len_rt, 1), tf.float32),                 # realtime 48k
        tf.TensorSpec((seg_len_rt * factor, 1), tf.float32)         # offline 96k
    )
    sig_y = (tf.TensorSpec((seg_len_rt, 1), tf.float32), tf.TensorSpec((), tf.float32))

    ds_pu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks=True, factor=factor),
        output_signature=(sig_x, sig_y)
    )
    ds_fu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks=False, factor=factor),
        output_signature=(sig_x, sig_y)
    )
    return tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO], seed=SEED)\
             .batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_mixed_ds_from_arrays_offline(x96, y96, batch_size, seg_len=OFFLINE_SEG_LEN):
    sig_x = tf.TensorSpec((seg_len, 1), tf.float32)
    sig_y_audio = tf.TensorSpec((seg_len, 1), tf.float32)
    sig_y_style = tf.TensorSpec((), tf.float32)
    sig_y = (sig_y_audio, sig_y_style)

    ds_pu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len, x96, y96, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    ds_fu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len, x96, y96, is_pick_chunks=False),
        output_signature=(sig_x, sig_y)
    )
    return tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO], seed=SEED)\
             .batch(batch_size).prefetch(tf.data.AUTOTUNE)

def approx_steps_from_length(total_samples, seg_len, batch_size, hop=None, cap=None):
    """
    Steps/epoch estimate from a single long stream length.
    Uses hop=seg_len//2 by default and an optional cap on steps.
    """
    hop = hop or max(1, seg_len // 2)
    windows = max((int(total_samples) - int(seg_len)) // int(hop) + 1, 1)
    steps = max(windows // int(batch_size), 1)
    if cap is not None:
        steps = min(steps, int(cap))
    return steps

def compute_ir_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    sr = int(sample_rate)
    ds8_layers = _find_layers(model, RealtimeIrDownsampleFIR, name_prefix="realtime_ir_downsample")
    L_ds8 = int(_safe_get_attr(ds8_layers[0], "kernel_size", REALTIME_IR_FIR_SIZE)) if ds8_layers else int(REALTIME_IR_FIR_SIZE)
    R_ir = _sum_sep_rf_for_prefix(model, r'^realtime_ir_sep_\d+$')
    F_ir = int(_safe_get_attr(ds8_layers[0], "factor", REALTIME_IR_DOWNSAMPLE_FACTOR)) if ds8_layers else int(REALTIME_IR_DOWNSAMPLE_FACTOR)
    # New: interp FIR
    interp_layers = _find_layers(model, FIRInterpByN, name_prefix="realtime_ir_interp")
    L_interp = int(getattr(interp_layers[0], "num_taps", REALTIME_LP_ORDER)) if interp_layers else int(REALTIME_LP_ORDER)
    LC_ir = (L_ds8 - 1) + F_ir * (int(R_ir) - 1) + (L_interp - 1)
    if verbose:
        ms = 1000.0 * LC_ir / float(sr)
        print(f"[ir] LC = {LC_ir} samples ({ms:.2f} ms)")
    return int(LC_ir)

def compute_main_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    R_main = _sum_sep_rf_for_prefix(model, r'^realtime_sep_\d+$')
    R_pre  = _sum_sep_rf_for_prefix(model, r'^realtime_pre_\d+_sep$')
    LC_main = int((R_main - 1) + (R_pre - 1))

    hp_layers = _find_layers(model, HFHighpass, name_prefix="realtime_with_picklearn")
    if hp_layers:
        L_hp = int(hp_layers[0].taps.shape[0])
        LC_main += (L_hp - 1)
        if verbose:
            ms = 1000.0 * (L_hp - 1) / float(sample_rate)
            print(f"[main] +HPF LC = {L_hp - 1} samples ({ms:.3f} ms)")

    if verbose:
        ms = 1000.0 * LC_main / float(sample_rate)
        print(f"[main] LC = {LC_main} samples ({ms:.2f} ms)")
    return LC_main













def compute_pickattack_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    """
    Left context for the realtime pickattack head:
      LC = LC(pre-emph) + RF(tiny CNN) + LC(post-HPF in head)

    - pre-emph layer 'realtime_hf_pre_emph' adds 1 sample (x[n-1]) if present.
    - tiny CNN = separable convs 'realtime_picklearn_c1' and 'realtime_picklearn_c2'
      RF increment = sum((k-1)*d).
    - post-HPF safety-net 'realtime_pickattack_post_hp' contributes (L-1).
    """
    sr = int(sample_rate)

    # Pre-emphasis: present → 1 sample, else 0
    lc_pre = 1 if any(l.name == "realtime_hf_pre_emph" for l in model.layers) else 0

    # Tiny CNN RF (two SeparableConv1D: c1, c2)
    R_pa = _sum_sep_rf_for_prefix(model, r'^realtime_picklearn_c[12]$')  # returns 1 + sum((k-1)*d)
    lc_cnn = max(0, int(R_pa) - 1)

    # Post-HPF inside the head
    hp_layers = _find_layers(model, HFHighpass, name_prefix="realtime_pickattack_post_hp")
    L_hp = int(hp_layers[0].taps.shape[0]) if hp_layers else 1
    lc_hp = L_hp - 1

    LC = lc_pre + lc_cnn + lc_hp
    if verbose:
        ms = 1000.0 * float(LC) / float(sr)
        print(f"[pickattack] LC = pre={lc_pre}, cnn={lc_cnn}, hpf={lc_hp} → {LC} samples ({ms:.3f} ms)")
    return int(LC)


def compute_pickreso_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    sr = int(sample_rate)

    # Onset pulse smoother
    env_layers = _find_layers(model, CausalExpSmoother, name_prefix="realtime_pickreso_onset_pulse")
    L_env = int(env_layers[0].kernel.shape[0]) if env_layers else 1
    lc_env = L_env - 1

    # Fast path pieces
    ds_layers = _find_layers(model, PickResoDownsampleFIR, name_prefix="realtime_pickreso_ds8")
    mlr_layers = _find_layers(model, LowrateModalBank,     name_prefix="realtime_pickreso_modal_lr")
    up_layers = _find_layers(model, FIRInterpByN,          name_prefix="realtime_pickreso_interp")

    if ds_layers and mlr_layers and up_layers:
        L_ds = int(getattr(ds_layers[0], "num_taps", REALTIME_PICK_RESO_DS_TAPS))
        L_mlr = int(getattr(mlr_layers[0], "length_lr",  _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME)))
        F = int(getattr(ds_layers[0], "factor", REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR))
        L_up = int(getattr(up_layers[0], "num_taps", REALTIME_PICK_RESO_INTERP_TAPS))
        LC = lc_env + (L_ds - 1) + F * (L_mlr - 1) + (L_up - 1)
        if verbose:
            ms = 1000.0 * float(LC) / float(sr)
            print(f"[pickreso-fast] env={lc_env}, ds={L_ds-1}, modal_lr={F*(L_mlr-1)}, interp={L_up-1} → {LC} samples ({ms:.3f} ms)")
        return int(LC)

    # Fallback: legacy path if encountered
    mb_layers = _find_layers(model, LearnedModalBank, name_prefix="realtime_pickreso_modal")
    if mb_layers:
        try:
            L_modal = int(getattr(mb_layers[0], "length"))
        except Exception:
            L_modal = int(mb_layers[0].kernel.shape[0])
        LC = lc_env + (L_modal - 1)
        if verbose:
            ms = 1000.0 * float(LC) / float(sr)
            print(f"[pickreso-legacy] env={lc_env}, modal={L_modal-1} → {LC} samples ({ms:.3f} ms)")
        return int(LC)

    if verbose:
        print("[pickreso] no recognizable branch; LC=0")
    return 0

def compute_realtime_streaming_context(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=True):
    """
    Max left context across realtime branches that feed the final sum:
      - main (pre + dilated stack [+ optional main HPF safety-net])
      - IR branch (ds FIR + lowrate RF + interp FIR)
      - pickattack head (pre-emph + tiny CNN + head HPF)
      - pickreso head (onset pulse + modal length)
      - coloreq bandpasses (if present)
    Returns (left_ctx_samples, right_ctx_samples). Right = 0 (purely causal).
    """
    lc_main = compute_main_left_context_realtime(model, sample_rate, verbose=False)
    lc_ir   = compute_ir_left_context_realtime(model, sample_rate, verbose=False)
    lc_reso = compute_pickreso_left_context_realtime(model, sample_rate, verbose=False)
    lc_patt = compute_pickattack_left_context_realtime(model, sample_rate, verbose=False)
    lc_col  = compute_coloreq_left_context_realtime(model, sample_rate, verbose=False)

    left_ctx = max(lc_main, lc_ir, lc_reso, lc_patt, lc_col)

    if verbose:
        def as_ms(samples):
            return 1000.0 * float(samples) / float(sample_rate)
        print("[streaming] realtime LC (samples): "
              f"main={lc_main}, ir={lc_ir}, pickattack={lc_patt}, pickreso={lc_reso}, coloreq={lc_col} "
              f"→ max={left_ctx} ({as_ms(left_ctx):.2f} ms)")

    return int(left_ctx), 0














@register_keras_serializable()
def add_coloreq(x, bands, sample_rate, name_prefix, order=65, trainable_scales=False):
    """
    Post-EQ inside the base model graph:
      y = x + sum_i g_i * BP_i(x)
    - g_i initialized to 0.0 (identity). When trainable_scales=False, they remain frozen.
    """
    terms = [x]
    for i, (lo, hi) in enumerate(bands):
        bp = BandpassFIR(low_hz=lo, high_hz=hi, order=order, sample_rate=sample_rate,
                         name=f"{name_prefix}_bp_{i}")(x)
        terms.append(IrScale(init=0.0, name=f"{name_prefix}_g_{i}", trainable=trainable_scales)(bp))
    return Add(name=f"{name_prefix}_sum")(terms)

def compute_coloreq_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    """
    Left context of the coloreq inside realtime models (if present).
    Looks for both new and legacy prefixes.
    """
    bp_layers = _find_layers(model, BandpassFIR, name_prefix="realtime_coloreq_bp_")
    if not bp_layers:
        # Backward-compat with old wrapper prefix
        bp_layers = _find_layers(model, BandpassFIR, name_prefix="realtime_ft_coloreq_bp_")
    if not bp_layers:
        return 0
    Ls = []
    for l in bp_layers:
        try:
            Ls.append(int(l.taps.shape[0]))
        except Exception:
            pass
    LC = max((L - 1) for L in Ls) if Ls else 0
    if verbose:
        ms = 1000.0 * LC / float(sample_rate)
        print(f"[coloreq] LC = {LC} samples ({ms:.3f} ms)")
    return int(LC)

def _select_stems_model_path(kind, use_finetuned=False):

    if kind == "offline":
        finetuned = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        baseline  = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    elif kind == "realtime":
        finetuned = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        baseline  = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

    if use_finetuned and os.path.exists(finetuned):
        return finetuned
    
    if os.path.exists(baseline):
        return baseline

    return None

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
    # Audio target
    if isinstance(y, (tuple, list)):
        y_true_audio = y[0]
    elif isinstance(y, dict):
        out_names = [t.name.split(':')[0].split('/')[-1] for t in model.outputs]
        y_true_audio = y[out_names[0]]
    else:
        y_true_audio = y

    # Feed inputs as a list in the model’s input order to avoid the KeyError
    if isinstance(x, dict):
        in_names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
        try:
            x = [x[k] for k in in_names]
        except KeyError:
            # Be tolerant to any prefixes (e.g., serving_default_) that might appear
            x_norm = {k.split(':')[0].split('/')[-1]: v for k, v in x.items()}
            missing = [k for k in in_names if k not in x_norm]
            if missing:
                raise KeyError(f"Batch is missing inputs required by the model: {missing}. "
                            f"Batch keys={list(x.keys())}, expected={in_names}")
            x = [x_norm[k] for k in in_names]

    y_pred_audio, _ = model(x, training=False)
    return y_true_audio, y_pred_audio

def analyze_loss_scales(model, batch, sample_rate):
    y_true, y_pred = _extract_audio_io(model, batch)

    gate_env = gate_weight_dynamic(y_true, sample_rate)

    l1   = tf.reduce_mean(waveform_l1_loss(y_true, y_pred)).numpy()
    mel  = tf.reduce_mean(log_mel_loss(y_true, y_pred, sample_rate)).numpy()
    env  = tf.reduce_mean(rms_env_loss(y_true, y_pred)).numpy()

    hf_on  = tf.reduce_mean(hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * gate_env)).numpy()
    stft_on = tf.reduce_mean(
        mrstft_mag_loss_weighted_onset(
            y_true, y_pred, sample_rate,
            split_hz=hp_cutoff_for_sr(sample_rate),
            high_boost=STFT_HIGH_BOOST,
            onset_env=gate_env,
            onset_gain=STFT_HF_ONSET_GAIN
        )
    ).numpy()

    bl = tf.reduce_mean(resonance_band_loss(y_true, y_pred, sample_rate)).numpy()

    return {'l1': l1, 'stft': stft_on, 'mel': mel, 'env': env, 'hf': hf_on, 'resbp_bl': bl}

def calibrate_loss_weights(model, dataset, sample_rate, n_batches=5,
                                      normalize_keys=('l1','stft','mel','env','hf','resbp_bl'),
                                      declared=None):
    if declared is None:
        declared = {
            'l1': L1_WEIGHT, 'stft': STFT_WEIGHT, 'mel': MEL_WEIGHT, 'env': RMS_ENV_WEIGHT, 'hf': HF_WEIGHT,
            'resbp_bl': RESBP_BAND_LOSS_WEIGHT  # NEW
        }
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


def _get_model_context_for_streaming(model_sample_rate: int, model=None):
    """
    Returns (left_ctx_model, right_ctx_model) in samples at the model's sample-rate.
    For realtime, if a model is provided, compute branch-aware left context automatically.
    """
    if model_sample_rate == OFFLINE_SAMPLE_RATE:
        # Left = receptive field - 1 (causal extent), Right = bounded look-ahead per layer
        return int(OFFLINE_RECEPTIVE_FIELD - 1), int(OFFLINE_LOOKAHEAD_SAMPLES)
    elif model_sample_rate == REALTIME_SAMPLE_RATE:
        # Prefer automatic per-branch analysis when model is available
        if model is not None:
            try:
                return compute_realtime_streaming_context(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=True)
            except Exception as e:
                print(f"[streaming] fallback to REALTIME_RECEPTIVE_FIELD due to: {e}")
        return int(REALTIME_RECEPTIVE_FIELD - 1), 0
    else:
        # Fallback: treat as causal with tiny RF (safe default)
        return 1024, 0  # adjust if you add other sample-rates

def generate_demo_from_files(model,
                             kind,
                             out_dir,
                             sample_rate,
                             path_src,
                             path_target,
                             prediction_name="demo_pred",
                             only_prediction=False,
                             offset_sec=LONG_DEMO_OFFSET_SEC,
                             sec=LONG_DEMO_SEC,
                             block_sec=12.0,
                             resample_margin_ms=50.0,
                             use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)
    knob_feeds = _prepare_knob_feeds(kind, use_finetuned=use_finetuned)

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    g_norm = 1.0

    if "pick" in path_src:
        style_true_flag = 1.0
    elif "finger" in path_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

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

    produced_model = 0
    prob_sum = 0.0
    prob_count = 0

    # Helper: filter a feed dict to the model’s actual input names
    model_input_names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
    def filter_feed(d):
        return {k: v for k, v in d.items() if k in model_input_names}

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
        read_b = min(END_FRAME, core_start_file + core_file_len + right_ctx_file + resample_margin_file)
        n_read = max(0, read_b - read_a)
        if n_read <= 0:
            break

        x_block_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        x_file = x_block_2d[:, 0] * g_norm
        if not only_prediction:
            y_block_2d, _ = sf.read(path_target, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0] * g_norm

        if file_sr != sample_rate:
            x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
            if not only_prediction:
                y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
        else:
            x_resamp = x_file
            if not only_prediction:
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

        x_demo = x_inp[None, :, None].astype(np.float32)

        # Predicted-style pass with knobs
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)

        base_feed = {
            f"{kind}_piezo_input": x_demo,
            f"{kind}_style_true_input": dummy_style,
            f"{kind}_style_force_input": force_zero,
        }
        base_feed.update(knob_feeds)
        feed = filter_feed(base_feed)

        y_inf_pred, style_pred_out = model.predict(feed, batch_size=1, verbose=0)
        y_inf_pred = y_inf_pred[0, :, 0]
        y_core_pred = y_inf_pred[core_start_in_inp:core_end_in_inp]
        w_pred.write(y_core_pred)

        prob_sum += float(style_pred_out[0, 0]) * float(core_len)
        prob_count += core_len

        # True-style pass (if known)
        if w_true is not None:
            style_true = np.array([[style_true_flag]], dtype=np.float32)
            force_one  = np.ones((1, 1), dtype=np.float32)
            true_feed = {
                f"{kind}_piezo_input": x_demo,
                f"{kind}_style_true_input": style_true,
                f"{kind}_style_force_input": force_one,
            }
            true_feed.update(knob_feeds)
            feed_true = filter_feed(true_feed)
            y_inf_true, _ = model.predict(feed_true, batch_size=1, verbose=0)
            y_inf_true = y_inf_true[0, :, 0]
            y_core_true = y_inf_true[core_start_in_inp:core_end_in_inp]
            w_true.write(y_core_true)

        if not only_prediction:
            core_start_global = inp_start + core_start_in_inp
            core_end_global = inp_start + core_end_in_inp
            x_core = x_resamp[core_start_global:core_end_global]
            y_core = y_resamp[core_start_global:core_end_global]
            w_src.write(x_core)
            w_tgt.write(y_core)

        produced_model += core_len

    w_pred.close()
    if w_true is not None:
        w_true.close()
    if w_src is not None:
        w_src.close()
    if w_tgt is not None:
        w_tgt.close()

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
    """
    Return a 1-D linear-phase prototype (no min-phase, no reshape).
    The downstream code will do minimum_phase + reverse + reshape.
    """
    N = int(REALTIME_IR_FIR_SIZE)
    if N < 2:
        raise ValueError("REALTIME_IR_FIR_SIZE must be >= 2")

    # Normalized cutoff for decimate-by-factor LPF: REALTIME_IR_CUTOFF is already 1/factor
    t = np.arange(N, dtype=np.float32) - (N - 1) / 2.0
    h = np.sinc(2.0 * REALTIME_IR_CUTOFF * t).astype(np.float32)

    # Sharper window than Hamming (optional)
    w = np.kaiser(N, beta=9.0).astype(np.float32)
    h *= w

    # Normalize sum to 1 to preserve DC
    s = np.sum(h, dtype=np.float64)
    h = (h / s).astype(np.float32)
    return h  # 1-D float32

_ir_fir     = make_sinc_filter()
h_lin = _ir_fir
h_min = minimum_phase(h_lin, method='homomorphic')
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)

# For tf.nn.conv1d (cross-correlation), use the time-reversed kernel to implement convolution
_ir_fir_kernel = h_min[::-1].reshape(-1, 1, 1).astype(np.float32)

_ir_fir_tf  = tf.constant(_ir_fir_kernel, dtype=tf.float32)  # downsampling (analysis) FIR

# ────────────────────────────── CUSTOM LAYERS ────────────────────────────────

@register_keras_serializable()
class ConstScale(tf.keras.layers.Layer):
    def __init__(self, scale, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
    def call(self, x):
        return x * self.scale
    def get_config(self):
        return {"scale": self.scale, **super().get_config()}

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
                        
@register_keras_serializable()
class FIRInterpByN(tf.keras.layers.Layer):
    """
    Upsample by integer factor N using zeros-insertion + fixed FIR (anti-imaging).
    - factor=N
    - num_taps: e.g. 257
    - cutoff is normalized to Nyquist of the upsampled rate as a fraction in (0,1).
    """
    def __init__(self, factor, num_taps=257, cutoff=None, window=('kaiser', 9.0), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)
        self.cutoff = float(0.45 / self.factor) if cutoff is None else float(cutoff)
        # Store as-is; we’ll normalize in build() to handle Keras TrackedList on load
        self.window = window

    def build(self, input_shape):
        # Normalize window to a SciPy-friendly type (str or tuple)
        win = self.window
        try:
            # Keras may deserialize tuples/lists as a TrackedList; convert to tuple
            if not isinstance(win, (str, tuple)):
                win = tuple(win)
        except TypeError:
            # Last-resort: fall back to a string window
            win = 'hann'

        # Design linear-phase LPF and reverse for conv1d
        try:
            taps = firwin(self.num_taps, cutoff=self.cutoff, window=win, pass_zero=True).astype(np.float32)
        except Exception:
            # Fallback if SciPy still complains about the window object
            taps = firwin(self.num_taps, cutoff=self.cutoff, window=('kaiser', 9.0), pass_zero=True).astype(np.float32)

        taps = taps / np.maximum(np.sum(taps), 1e-12)
        k = taps[::-1].reshape(-1, 1, 1).astype(np.float32)

        self.kernel = self.add_weight(
            name="fir_interp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, C_low)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        c = tf.shape(x)[2]

        # Insert zeros between samples along time axis
        x4 = tf.reshape(x, [b, t, 1, c])                         # (B,T,1,C)
        pad_zeros = tf.zeros([b, t, self.factor - 1, c], x.dtype)
        xz = tf.concat([x4, pad_zeros], axis=2)                   # (B,T,F,C)
        xu = tf.reshape(xz, [b, t * self.factor, c])              # (B,T*F,C)

        # Causal FIR
        L = self.num_taps
        xu_pad = tf.pad(xu, [[0,0],[L-1,0],[0,0]])
        y = tf.nn.conv1d(xu_pad, filters=self.kernel, stride=1, padding='VALID')  # (B, T*F, C)
        return y

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        return (b, None if t is None else t * self.factor, c)

    def get_config(self):
        # Return JSON-serializable fields; tuples will become lists in JSON, which is fine
        return {
            "factor": self.factor,
            "num_taps": self.num_taps,
            "cutoff": self.cutoff,
            "window": self.window,
            **super().get_config()
        }

@register_keras_serializable()
class BandpassFIR(tf.keras.layers.Layer):
    def __init__(self, low_hz, high_hz, order, sample_rate, window="hamming", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.order = int(order)
        self.sample_rate = int(sample_rate)
        self.window = str(window)

    def build(self, input_shape):
        k = _get_minphase_bp_kernel(self.sample_rate, self.low_hz, self.high_hz, self.order, window=self.window)
        self.taps = self.add_weight(
            name="bp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )

    def call(self, x):
        pad = tf.shape(self.taps)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "order": self.order,
            "sample_rate": self.sample_rate,
            "window": self.window
        })
        return cfg

@register_keras_serializable()
class HFHighpass(tf.keras.layers.Layer):
    def __init__(self, taps=None, sample_rate=None, cutoff_hz=OFFLINE_HP_CUTOFF_HZ, order=OFFLINE_HF_HP_ORDER, **kwargs):
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

# Add near your other custom layers
@register_keras_serializable()
class LossTether(tf.keras.layers.Layer):
    """
    Connects auxiliary tensors (e.g., outputs of KD layers that only add losses)
    to the model graph without altering the primary tensor.
    Inputs: [primary, aux1, aux2, ...]
    Output: primary (unchanged)
    This ensures all aux tensors are 'reachable' so their add_loss is kept.
    """
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return inputs[0]
        return inputs
    
    
@register_keras_serializable()
class CausalExpSmoother(tf.keras.layers.Layer):
    """
    Causal exponential moving average implemented as a fixed FIR (left-padded conv1d).
    time_const_ms is the EMA time constant in milliseconds.
    """
    def __init__(self, time_const_ms, sample_rate, scale=4.0, Lmax=4096, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_const_ms = float(time_const_ms)
        self.sample_rate   = int(sample_rate)
        self.scale         = float(scale)
        self.Lmax          = int(Lmax)

    def build(self, input_shape):
        tau_samples = max(1, int(round((self.time_const_ms / 1000.0) * self.sample_rate)))
        k = _exp_smoothing_kernel(tau_samples, scale=self.scale, Lmax=self.Lmax)
        # IMPORTANT: pass a NumPy value to Constant initializer to avoid cross-graph capture
        k_init = k.numpy() if tf.is_tensor(k) else np.array(k, dtype=np.float32, copy=False)
        self.kernel = self.add_weight(
            name="exp_smooth_taps",
            shape=k_init.shape,
            initializer=tf.keras.initializers.Constant(k_init),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'time_const_ms': self.time_const_ms,
            'sample_rate': self.sample_rate,
            'scale': self.scale,
            'Lmax': self.Lmax
        })
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
        if not training:
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
def lowband_error_penalty(y_true, y_pred, sample_rate,
                          cutoff_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                          order=129,
                          window="hamming"):
    """
    Training loss term: L1 of LPF(y_pred - y_true) below cutoff_hz.
    Returns (B,T,1) so it’s compatible with per-sample weighting.
    """
    err = y_pred - y_true
    k = _make_lpf_kernel(int(sample_rate), float(cutoff_hz), int(order), window=window)  # (L,1,1) reversed
    pad = tf.shape(k)[0] - 1
    err_pad = tf.pad(err, [[0,0],[pad,0],[0,0]])
    lp_err = tf.nn.conv1d(err_pad, filters=k, stride=1, padding='VALID')
    return tf.abs(lp_err)

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

# ─── Band-pass (700–2000 Hz) causal min-phase FIR ────────────────────────────
_bp_kernel_cache = {}
def _get_minphase_bp_kernel(sample_rate, f_lo, f_hi, order=257, window="hamming"):
    """
    Build a causal min-phase band-pass FIR kernel for tf.nn.conv1d (cross-correlation),
    i.e. time-reversed taps. We do not normalize DC since it's a band-pass.
    """
    key = (int(sample_rate), float(f_lo), float(f_hi), int(order), str(window))
    k_np = _bp_kernel_cache.get(key)
    if k_np is None:
        nyq = float(sample_rate) / 2.0
        lo  = float(f_lo) / nyq
        hi  = float(f_hi) / nyq
        taps = firwin(order, [lo, hi], pass_zero=False, window=window).astype(np.float32)
        # Convert to minimum-phase
        taps_min = minimum_phase(taps, method="homomorphic").astype(np.float32)
        # Reverse for conv1d (cross-correlation) to emulate convolution
        k_np = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        _bp_kernel_cache[key] = k_np
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

def _db_to_lin(db):
    return float(10.0 ** (float(db) / 20.0))

def _resolve_demo_knobs(kind, use_finetuned=False):
    # Return (pickattack_lin, pickreso_lin, ir_lin)
    if kind == "offline":
        if use_finetuned:
            pa_db = FINETUNE_OFFLINE_PICK_ATTACK_BOOST_DB
            pr_db = FINETUNE_OFFLINE_PICK_RESO_BOOST_DB
        else:
            pa_db = OFFLINE_PICK_ATTACK_BOOST_DB
            pr_db = OFFLINE_PICK_RESO_BOOST_DB
        ir_db = 0.0  # no IR branch in offline graph
    else:
        if use_finetuned:
            pa_db = FINETUNE_REALTIME_PICK_ATTACK_BOOST_DB
            pr_db = FINETUNE_REALTIME_PICK_RESO_BOOST_DB
            ir_db = FINETUNE_REALTIME_IR_BOOST_DB
        else:
            pa_db = REALTIME_PICK_ATTACK_BOOST_DB
            pr_db = REALTIME_PICK_RESO_BOOST_DB
            ir_db = REALTIME_IR_BOOST_DB

    return (_db_to_lin(pa_db), _db_to_lin(pr_db), _db_to_lin(ir_db))

def _prepare_knob_feeds(kind, use_finetuned=False):
    pa_lin, pr_lin, ir_lin = _resolve_demo_knobs(kind, use_finetuned=use_finetuned)
    dry_frac = _default_dryness_value()

    feeds = {}
    if kind == "realtime":
        feeds["realtime_knob_pickattack"] = np.array([[pa_lin]], dtype=np.float32)
        feeds["realtime_knob_pickreso"]   = np.array([[pr_lin]], dtype=np.float32)
        feeds["realtime_knob_ir"]         = np.array([[ir_lin]], dtype=np.float32)
        feeds["realtime_knob_drywet"]     = np.array([[dry_frac]], dtype=np.float32)
    elif kind == "offline":
        feeds["offline_knob_pickattack"]  = np.array([[pa_lin]], dtype=np.float32)
        feeds["offline_knob_pickreso"]    = np.array([[pr_lin]], dtype=np.float32)
        feeds["offline_knob_drywet"]      = np.array([[dry_frac]], dtype=np.float32)
    return feeds

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

@register_keras_serializable()
def clip_01x4(z):
    # Scale by 4 and clamp to [0, 1]
    return tf.clip_by_value(4.0 * z, 0.0, 1.0)

@register_keras_serializable()
def film_gamma_stable(raw, scale=1.0, mag=0.1):
    # γ = 1 + mag * tanh(scale * raw)
    return 1.0 + float(mag) * tf.tanh(scale * raw)

@register_keras_serializable()
def film_beta_stable(raw, scale=1.0, mag=0.1):
    # β = mag * tanh(scale * raw)
    return float(mag) * tf.tanh(scale * raw)

def compute_pads(kernel, dilation, lookahead_samples):
    P_tot = (kernel - 1) * dilation
    pad_right = min(int(lookahead_samples), P_tot)
    pad_left  = P_tot - pad_right
    return pad_left, pad_right

@register_keras_serializable()
def glu_sep_block(x, filters, kernel, dilation, prefix, lookahead=0, residual_scale=0.35, bias_constraint=None):
    """
    Gated Linear Unit version of the residual block:
      y = res + s * PW( (DW(x))_content * sigmoid((DW(x))_gate) )
    where content is linear (no tanh), gate is sigmoid, and s<1 is a fixed residual scale.
    This greatly reduces harmonic generation vs. tanh.
    """
    res = x
    pad_left, pad_right = compute_pads(kernel=kernel, dilation=dilation, lookahead_samples=lookahead)
    x   = ZeroPadding1D((pad_left, pad_right), name=f"{prefix}_pad")(x)
    g   = SeparableConv1D(2*filters, kernel, dilation_rate=dilation, padding="valid",
                          bias_constraint=bias_constraint, name=f"{prefix}_sep")(x)
    c   = Lambda(slice_content, arguments={"filters": filters}, name=f"{prefix}_slice_c")(g)   # linear content
    g2  = Lambda(slice_gate,    arguments={"filters": filters}, name=f"{prefix}_slice_g")(g)   # gate logits
    x   = Multiply(name=f"{prefix}_gate_mul")([c, Activation("sigmoid", name=f"{prefix}_sigmoid")(g2)])
    x   = Conv1D(filters, 1, padding="same", bias_constraint=bias_constraint, name=f"{prefix}_pw")(x)
    # Residual scaling to keep the stack linear-ish and avoid runaway gain
    x   = ConstScale(residual_scale, name=f"{prefix}_res_scale")(x)
    x   = Add(name=f"{prefix}_res_add")([x, res])
    return x

def gated_sep_block(x, filters, kernel, dilation, prefix, lookahead=0, bias_constraint=None):
    res = x
    pad_left, pad_right = compute_pads(kernel=kernel, dilation=dilation, lookahead_samples=lookahead)
    x   = ZeroPadding1D((pad_left, pad_right), name=f"{prefix}_pad")(x)
    g   = SeparableConv1D(2*filters, kernel, dilation_rate=dilation, padding="valid",
                          bias_constraint=bias_constraint, name=f"{prefix}_sep")(x)
    c   = Lambda(slice_content, arguments={"filters": filters}, name=f"{prefix}_slice_c")(g)
    g2  = Lambda(slice_gate,    arguments={"filters": filters}, name=f"{prefix}_slice_g")(g)
    x   = Multiply(name=f"{prefix}_gate_mul")([Activation("tanh", name=f"{prefix}_tanh")(c),
                                               Activation("sigmoid", name=f"{prefix}_sigmoid")(g2)])
    x   = Conv1D(filters, 1, padding="same", bias_constraint=bias_constraint, name=f"{prefix}_pw")(x)
    x   = Add(name=f"{prefix}_res_add")([x, res])
    return x

def film_apply(x, style_embed, filters, prefix,
               gamma_scale=1.0, beta_scale=1.0,
               gamma_mag=0.1, beta_mag=0.1):
    film_params = Dense(filters*2, name=f"{prefix}_film_dense")(style_embed)
    gamma_raw = Lambda(slice_film_gamma, arguments={'filters': filters},
                       name=f"{prefix}_film_gamma_raw")(film_params)
    beta_raw  = Lambda(slice_film_beta,  arguments={'filters': filters},
                       name=f"{prefix}_film_beta_raw")(film_params)

    gamma = Lambda(film_gamma_stable,
                   arguments={'scale': float(gamma_scale), 'mag': float(gamma_mag)},
                   name=f"{prefix}_film_gamma")(gamma_raw)
    beta  = Lambda(film_beta_stable,
                   arguments={'scale': float(beta_scale), 'mag': float(beta_mag)},
                   name=f"{prefix}_film_beta")(beta_raw)

    gamma3 = Reshape((1, filters), name=f"{prefix}_film_gamma_expand")(gamma)
    beta3  = Reshape((1, filters), name=f"{prefix}_film_beta_expand")(beta)
    return Add(name=f"{prefix}_film_apply")([
        Multiply(name=f"{prefix}_film_scale")([x, gamma3]),
        beta3
    ])
    
# ────────────────────────────── MODEL BUILDERS ───────────────────────────────

@register_keras_serializable()
def build_offline_model_causal(offline_src, dynamic_time=False):
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
def build_offline_model(dynamic_time=False,
                                  lookahead_samples=OFFLINE_LOOKAHEAD_SAMPLES,
                                  name="piezo2mic_offline",
                                  coloreq_trainable=False):
    T   = None if dynamic_time else OFFLINE_SEG_LEN
    inp = Input(shape=(T,1), name="offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="offline_style_true_input")
    style_force_in = Input(shape=(1,), name="offline_style_force_input")

    cls         = Conv1D(16,3, padding="same", activation="relu", name="offline_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="offline_style_gap")(cls)
    style_logit = Dense(1, name="offline_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="offline_style_prob")(style_logit)

    one_minus_force = Lambda(invert, name="offline_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="offline_style_used_scalar")([
        Multiply(name="offline_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="offline_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time = TileStyle(name="offline_style_time_used")([style_used_scalar, inp])
    style_embed= Dense(OFFLINE_EMBED_DIM, activation="linear", name="offline_style_embed")(style_used_scalar)

    x = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_init_pw")(inp)
    for i in range(OFFLINE_PRE_BLOCKS):
        x = gated_sep_block(x, OFFLINE_FILTERS, OFFLINE_PRE_KERNEL, 1, prefix=f"offline_pre_{i}", lookahead=lookahead_samples)
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_pre_{i}",
                       gamma_scale=OFFLINE_FILM_GAMMA_SCALE, beta_scale=OFFLINE_FILM_BETA_SCALE,
                       gamma_mag=OFFLINE_FILM_GAMMA_MAG, beta_mag=OFFLINE_FILM_BETA_MAG)

    for idx, d in enumerate(OFFLINE_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=OFFLINE_KERNEL_SIZE, dilation=d, lookahead_samples=lookahead_samples)
        x = ZeroPadding1D((pad_left, pad_right), name=f"pad_{idx}")(x)
        g = SeparableConv1D(2*OFFLINE_FILTERS, OFFLINE_KERNEL_SIZE, dilation_rate=d, padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':OFFLINE_FILTERS}, name=f"slice_content_{idx}")(g)
        g2= Lambda(slice_gate,    arguments={'filters':OFFLINE_FILTERS}, name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([Activation('tanh', name=f"tanh_{idx}")(c),
                                               Activation('sigmoid', name=f"sigmoid_{idx}")(g2)])
        x = Conv1D(OFFLINE_FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_dil_{idx}",
                       gamma_scale=OFFLINE_FILM_GAMMA_SCALE, beta_scale=OFFLINE_FILM_BETA_SCALE,
                       gamma_mag=OFFLINE_FILM_GAMMA_MAG, beta_mag=OFFLINE_FILM_BETA_MAG)

    main_pick   = Conv1D(1,1,padding="same",name="offline_final_pick")(x)
    main_finger = Conv1D(1,1,padding="same",name="offline_final_finger")(x)
    inv_style   = InvertStyle(name="offline_invert_style")(style_time)
    pick_c      = Multiply(name="offline_pick_contrib")([style_time, main_pick])
    finger_c    = Multiply(name="offline_finger_contrib")([inv_style,   main_finger])
    main_core   = Add(name="offline_main_core")([pick_c, finger_c])

    # Analytic pick mask
    mask = _pick_mask_engine(inp,
                             thr_db=OFFLINE_PICK_MASK_THRESHOLD_DB,
                             knee_db=OFFLINE_PICK_MASK_KNEE_DB,
                             ref_is_hp=True,
                             post_comp_p=OFFLINE_PICK_MASK_POST_COMP_P,
                             post_comp_gain=OFFLINE_PICK_MASK_POST_COMP_GAIN,
                             floor=OFFLINE_PICK_MASK_FLOOR,
                             out_gain=OFFLINE_PICK_MASK_OUT_GAIN,
                             env_fast_ms=OFFLINE_PICK_MASK_ENV_FAST_MS,
                             env_slow_ms=OFFLINE_PICK_MASK_ENV_SLOW_MS,
                             hp_order=OFFLINE_HF_HP_ORDER,
                             sample_rate=OFFLINE_SAMPLE_RATE,
                             smooth_attack_ms=OFFLINE_PICK_MASK_SMOOTH_ATTACK_MS,
                             smooth_release_ms=OFFLINE_PICK_MASK_SMOOTH_RELEASE_MS,
                             prefix="offline")

    # HF pre-emphasis for pickattack head
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(inp)

    # Style‑gated pick stems (now POST-HPF inside the heads)
    pickattack, pa_pen = _pick_attack(
        "offline", hf_in, mask, style_time,
        filters=16,
        sample_rate=OFFLINE_SAMPLE_RATE,
        softclip_drive=OFFLINE_PICK_ATTACK_SOFTCLIP_DRIVE,
        lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
        lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
        lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER
    )
    pickreso, pr_pen = _reson_learn_offline(inp, mask, style_time)
    
    knob_pa_in = Input(shape=(1,), name="offline_knob_pickattack")  # linear gain
    knob_pr_in = Input(shape=(1,), name="offline_knob_pickreso")    # linear gain
    knob_dw_in = Input(shape=(1,), name="offline_knob_drywet")

    # Multiply knobs post-limit/post-HPF so user gain doesn't re-open headroom or subsonics.
    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="offline_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="offline_pickreso_user_gain")

    # Mix (no bus HPF)
    out_prehp = Add(name="offline_with_picklearn_prehp")([main_core, pickattack, pickreso])

    # Tie penalty nodes so add_loss is kept (attach to post-no-op signal)
    out_prehp = LossTether(name="offline_pick_lowband_penalties")([out_prehp, pa_pen, pr_pen])

    # Final tone shaping (cannot re-introduce subsonics; first band starts at 80 Hz)
    out = add_coloreq(out_prehp, bands=COLOREQ_BANDS_OFFLINE_BODY, sample_rate=OFFLINE_SAMPLE_RATE,
                      name_prefix="offline_coloreq", order=97, trainable_scales=bool(coloreq_trainable))

    # dry/wet mixer (dry = piezo input)
    wet_gain   = Lambda(invert, name="offline_wet_from_drywet")(knob_dw_in)           # (B,1)
    wet_scaled = apply_runtime_gain(out,    wet_gain,       name="offline_wet_gain")  # (B,T,1)
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,     name="offline_dry_gain")  # (B,T,1)
    out = Add(name="offline_drywet_mix")([wet_scaled, dry_scaled])


    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in, knob_pa_in, knob_pr_in, knob_dw_in], outputs=[out, style_prob], name=name)

@register_keras_serializable()
def build_realtime_training_model(offline=None,
                        dynamic_time=False,
                        name="piezo2mic_realtime",
                        coloreq_trainable=False):
    """
    Realtime NON-streaming graph.
    - If 'offline' is provided: builds KD training graph (extra teacher input).
    - If 'offline' is None: builds teacherless (self-contained) NON-streaming graph.
    This function no longer supports a streaming flag.

    For streaming self-contained models (static T=32 or dynamic time),
    use build_realtime_streaming_model(...) below.
    """
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None, 1), name="realtime_offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")
    
    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")  # linear gain
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")    # linear gain
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")          # linear gain
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    style_pred_source = style_prob
    if offline is not None:
        teacher = offline; teacher.trainable = False
        audio_out, style_prob_out = teacher.outputs
        teacher_combo = tf.keras.Model(inputs=teacher.inputs, outputs=[audio_out, style_prob_out],
                                       name="offline_teacher_combo")
        combo = teacher_combo([offline_inp, style_true_in, style_force_in, knob_pa_in, knob_pr_in, knob_dw_in])
        offline_style_prob = combo[1]
        style_pred_source = BlendScalar(init_alpha=REALTIME_STYLE_BLEND_INIT,
                                        trainable=REALTIME_STYLE_BLEND_TRAINABLE,
                                        name="realtime_style_blend")([style_prob, offline_style_prob])

    one_minus_force = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_pred_source]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    style_scalar_b3 = Reshape((1, 1), name="realtime_style_scalar_broadcast")(style_used_scalar)
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_style_embed")(style_used_scalar)

    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(inp)
    for i in range(REALTIME_PRE_BLOCKS):
        x = glu_sep_block(x, REALTIME_FILTERS, REALTIME_PRE_KERNEL, 1,
                          prefix=f"realtime_pre_{i}", lookahead=0,
                          residual_scale=0.35, bias_constraint=BC)
        x = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}",
                       gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                       gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)

    for i, d in enumerate(REALTIME_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=REALTIME_KERNEL_SIZE, dilation=d, lookahead_samples=0)
        x = ZeroPadding1D((pad_left, pad_right), name=f"realtime_pad_{i}")(x)
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid",
                            bias_constraint=BC, name=f"realtime_sep_{i}")(x)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        xb = Multiply(name=f"realtime_gate_mul_{i}")([c, Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pw_{i}")(xb)
        xb = ConstScale(0.35, name=f"realtime_res_scale_{i}")(xb)
        x  = Add(name=f"realtime_res_add_{i}")([xb, res])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch (non-streaming)
    ir = RealtimeIrDownsampleFIR(name="realtime_ir_downsample")(inp)
    for i, d in enumerate(REALTIME_IR_DILATIONS):
        pad = (REALTIME_IR_KERNEL - 1) * d
        ir = ZeroPadding1D((pad, 0), name=f"realtime_ir_pad_{i}")(ir)
        ir = SeparableConv1D(filters=1, kernel_size=REALTIME_IR_KERNEL, dilation_rate=d, padding="valid",
                             depthwise_regularizer=regularizers.l2(1e-4),
                             pointwise_regularizer=regularizers.l2(1e-4),
                             depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                             pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                             bias_constraint=BC, name=f"realtime_ir_sep_{i}")(ir)
    ir = IrScale(init=0.6, name="realtime_ir_scale")(ir)
    ir = FIRInterpByN(factor=REALTIME_IR_DOWNSAMPLE_FACTOR, num_taps=max(193, REALTIME_LP_ORDER),
                      cutoff=0.45/REALTIME_IR_DOWNSAMPLE_FACTOR, window=('kaiser', 9.0),
                      name="realtime_ir_interp")(ir)
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir, inp])

    # Analytic pick mask
    mask = _pick_mask_engine(inp,
                             thr_db=REALTIME_PICK_MASK_THRESHOLD_DB,
                             knee_db=REALTIME_PICK_MASK_KNEE_DB,
                             ref_is_hp=True,
                             post_comp_p=REALTIME_PICK_MASK_POST_COMP_P,
                             post_comp_gain=REALTIME_PICK_MASK_POST_COMP_GAIN,
                             floor=REALTIME_PICK_MASK_FLOOR,
                             out_gain=REALTIME_PICK_MASK_OUT_GAIN,
                             env_fast_ms=REALTIME_PICK_MASK_ENV_FAST_MS,
                             env_slow_ms=REALTIME_PICK_MASK_ENV_SLOW_MS,
                             hp_order=REALTIME_HF_HP_ORDER,
                             sample_rate=REALTIME_SAMPLE_RATE,
                             smooth_attack_ms=REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
                             smooth_release_ms=REALTIME_PICK_MASK_SMOOTH_RELEASE_MS,
                             prefix="realtime")

    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(inp)

    # Pick-attack and pick-reso (POST-HPF inside heads)
    pickattack, pa_pen = _pick_attack(
        "realtime", hf_in, mask, style_scalar_b3,
        filters=16,
        sample_rate=REALTIME_SAMPLE_RATE,
        softclip_drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE,
        lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
        lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
        lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
    )
    pickreso, pr_pen = _reson_learn_realtime(inp, mask, style_scalar_b3)

    # Apply knobs post-limit/post-HPF
    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,         knob_ir_in, name="realtime_ir_user_gain")

    # Sum and color EQ
    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main, ir, pickattack, pickreso])
    y_prehp  = LossTether(name="realtime_pick_lowband_penalties")([y_prehp, pa_pen, pr_pen])

    y_wet = add_coloreq(y_prehp, bands=COLOREQ_BANDS_REALTIME_BODY, sample_rate=REALTIME_SAMPLE_RATE,
                        name_prefix="realtime_coloreq", order=65, trainable_scales=bool(coloreq_trainable))
    
    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_final    = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    inputs_list = [inp, style_true_in, style_force_in]
    if offline_inp is not None:
        inputs_list = [inp, offline_inp, style_true_in, style_force_in]
    inputs_list += [knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in]
    return tf.keras.models.Model(inputs=inputs_list, outputs=[y_final, style_prob], name=name)

@register_keras_serializable()
def build_realtime_streaming_model(dynamic_time=False,
                                   name="piezo2mic_realtime",
                                   coloreq_trainable=False,
                                   static_T=REALTIME_TFLITE_T):
    if dynamic_time:
        T = None
    else:
        T = int(static_T)

    inp = Input(shape=(T,1), name="realtime_piezo_input")
    state_inputs = []
    state_outputs = []

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")

    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    # Style head (unchanged)
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    # Streaming EMA for style (unchanged)
    s_style = Input(shape=(1,1), name="realtime_state_style_ema")
    state_inputs.append(s_style)
    s_style_flat = Reshape((1,), name="realtime_state_style_ema_flat")(s_style)
    style_prob_ema, s_style_next_flat = StreamingScalarEMA(alpha=0.98, name="realtime_style_prob_ema")([style_prob, s_style_flat])
    s_style_next = Reshape((1,1), name="realtime_state_style_ema_expand")(s_style_next_flat)
    state_outputs.append(Activation('linear', name="realtime_state_style_ema_next")(s_style_next))

    one_minus_force   = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_prob_ema]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    style_scalar_b3 = Reshape((1, 1), name="realtime_style_scalar_broadcast")(style_used_scalar)
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_style_embed")(style_used_scalar)

    # Main stack (unchanged)
    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(inp)
    for i in range(REALTIME_PRE_BLOCKS):
        pad_len = (REALTIME_PRE_KERNEL - 1) * 1
        s_in = Input(shape=(pad_len, REALTIME_FILTERS), name=f"realtime_state_pre_{i}")
        state_inputs.append(s_in)
        x_cat = ConcatTime(name=f"realtime_pre_{i}_concat")([s_in, x])
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_PRE_KERNEL, dilation_rate=1, padding="valid",
                            bias_constraint=BC, name=f"realtime_pre_{i}_sep")(x_cat)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_pre_{i}_slice_c")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_pre_{i}_slice_g")(g)
        xb = Multiply(name=f"realtime_pre_{i}_gate_mul")([c, Activation("sigmoid", name=f"realtime_pre_{i}_sigmoid")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pre_{i}_pw")(xb)
        xb = ConstScale(0.35, name=f"realtime_pre_{i}_res_scale")(xb)
        x  = Add(name=f"realtime_pre_{i}_res_add")([xb, x])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)
        s_out = TakeLastN(pad_len, name=f"realtime_state_pre_{i}_next")(x_cat)
        state_outputs.append(s_out)

    for i, d in enumerate(REALTIME_DILATIONS):
        pad_len = (REALTIME_KERNEL_SIZE - 1) * d
        s_in = Input(shape=(pad_len, REALTIME_FILTERS), name=f"realtime_state_sep_{i}")
        state_inputs.append(s_in)
        x_cat = ConcatTime(name=f"realtime_padcat_{i}")([s_in, x])
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid",
                            bias_constraint=BC, name=f"realtime_sep_{i}")(x_cat)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        xb = Multiply(name=f"realtime_gate_mul_{i}")([c, Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pw_{i}")(xb)
        xb = ConstScale(0.35, name=f"realtime_res_scale_{i}")(xb)
        x  = Add(name=f"realtime_res_add_{i}")([xb, x])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)
        s_out = TakeLastN(pad_len, name=f"realtime_state_sep_{i}_next")(x_cat)
        state_outputs.append(s_out)

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch: use exact prototype taps for DS to ensure equivalence
    s_ir_ds = Input(shape=(REALTIME_IR_FIR_SIZE - 1, 1), name="realtime_state_ir_ds")
    state_inputs.append(s_ir_ds)
    ir_ds, s_ir_ds_next = StreamingRealtimeIrDownsampleFIR(
        factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
        kernel_size=REALTIME_IR_FIR_SIZE,
        use_ir_prototype=True,
        name="realtime_ir_downsample"
    )([inp, s_ir_ds])
    state_outputs.append(Activation('linear', name="realtime_state_ir_ds_next")(s_ir_ds_next))

    ir_lr = ir_ds
    for i, d in enumerate(REALTIME_IR_DILATIONS):
        pad = (REALTIME_IR_KERNEL - 1) * d
        s_in = Input(shape=(pad, 1), name=f"realtime_state_ir_sep_{i}")
        state_inputs.append(s_in)
        ir_cat = ConcatTime(name=f"realtime_ir_padcat_{i}")([s_in, ir_lr])
        ir_lr = SeparableConv1D(filters=1, kernel_size=REALTIME_IR_KERNEL, dilation_rate=d, padding="valid",
                                depthwise_regularizer=regularizers.l2(1e-4),
                                pointwise_regularizer=regularizers.l2(1e-4),
                                depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                                pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                                bias_constraint=BC, name=f"realtime_ir_sep_{i}")(ir_cat)
        s_out = TakeLastN(pad, name=f"realtime_state_ir_sep_{i}_next")(ir_cat)
        state_outputs.append(s_out)
    ir_lr = IrScale(init=0.6, name="realtime_ir_scale")(ir_lr)

    s_ir_up = Input(shape=(max(193, REALTIME_LP_ORDER)-1, 1), name="realtime_state_ir_interp")
    state_inputs.append(s_ir_up)
    ir_up, s_ir_up_next = StreamingFIRInterpByN(factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
                                                num_taps=max(193, REALTIME_LP_ORDER),
                                                cutoff=0.45/REALTIME_IR_DOWNSAMPLE_FACTOR,
                                                window=('kaiser', 9.0),
                                                name="realtime_ir_interp")(inputs=[ir_lr, s_ir_up])
    state_outputs.append(Activation('linear', name="realtime_state_ir_interp_next")(s_ir_up_next))
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir_up, inp])

    # Pick mask (unchanged)
    def ema_len(ms, sr, scale, Lmax):
        s = max(1, int(round((float(ms)/1000.0) * float(sr))))
        L = int(np.clip(np.ceil(scale * s), 8, Lmax))
        return L
    L_hp   = REALTIME_HF_HP_ORDER
    L_fast = ema_len(REALTIME_PICK_MASK_ENV_FAST_MS, REALTIME_SAMPLE_RATE, 3.0, 1024)
    L_slow = ema_len(REALTIME_PICK_MASK_ENV_SLOW_MS, REALTIME_SAMPLE_RATE, 3.0, 4096)
    L_att  = ema_len(REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,  REALTIME_SAMPLE_RATE, 4.0, 2048)
    L_rel  = ema_len(REALTIME_PICK_MASK_SMOOTH_RELEASE_MS, REALTIME_SAMPLE_RATE, 4.0, 2048)

    s_m_hp   = Input(shape=(L_hp-1,   1), name="realtime_state_pickmask_hp")
    s_m_fast = Input(shape=(L_fast-1, 1), name="realtime_state_pickmask_fast")
    s_m_slow = Input(shape=(L_slow-1, 1), name="realtime_state_pickmask_slow")
    s_m_att  = Input(shape=(L_att-1,  1), name="realtime_state_pickmask_att")
    s_m_rel  = Input(shape=(L_rel-1,  1), name="realtime_state_pickmask_rel")
    s_m_prev = Input(shape=(1,1),         name="realtime_state_pickmask_prev")
    state_inputs += [s_m_hp, s_m_fast, s_m_slow, s_m_att, s_m_rel, s_m_prev]

    mask_outs = StreamingPickDBMask(
        sample_rate=REALTIME_SAMPLE_RATE,
        hp_cut_hz=hp_cutoff_for_sr(REALTIME_SAMPLE_RATE),
        hp_order=REALTIME_HF_HP_ORDER,
        env_fast_ms=REALTIME_PICK_MASK_ENV_FAST_MS,
        env_slow_ms=REALTIME_PICK_MASK_ENV_SLOW_MS,
        thr_db=REALTIME_PICK_MASK_THRESHOLD_DB,
        knee_db=REALTIME_PICK_MASK_KNEE_DB,
        smooth_attack_ms=REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
        smooth_release_ms=REALTIME_PICK_MASK_SMOOTH_RELEASE_MS,
        ref_is_hp=True,
        post_comp_p=REALTIME_PICK_MASK_POST_COMP_P,
        post_comp_gain=REALTIME_PICK_MASK_POST_COMP_GAIN,
        floor=REALTIME_PICK_MASK_FLOOR,
        out_gain=REALTIME_PICK_MASK_OUT_GAIN,
        name="realtime_pick_mask_db"
    )([inp, s_m_hp, s_m_fast, s_m_slow, s_m_att, s_m_rel, s_m_prev])
    mask, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_n = mask_outs
    state_outputs += [
        Activation('linear', name="realtime_state_pickmask_hp_next")(s_hp_n),
        Activation('linear', name="realtime_state_pickmask_fast_next")(s_fast_n),
        Activation('linear', name="realtime_state_pickmask_slow_next")(s_slow_n),
        Activation('linear', name="realtime_state_pickmask_att_next")(s_att_n),
        Activation('linear', name="realtime_state_pickmask_rel_next")(s_rel_n),
        Activation('linear', name="realtime_state_pickmask_prev_next")(s_prev_n),
    ]

    # Pre-emphasis state (unchanged)
    s_pre = Input(shape=(1,1), name="realtime_state_preemph")
    state_inputs.append(s_pre)
    hf_in, s_pre_next = PreEmphasisStream(alpha=HP_PRE_EMPHASIS_HF, name="realtime_hf_pre_emph_stream")([inp, s_pre])
    state_outputs.append(Activation('linear', name="realtime_state_preemph_next")(s_pre_next))

        # Pick-attack tiny CNN streaming (LeakyReLU to avoid dead branch)
    x_pa = Multiply(name="realtime_picklearn_gate")([hf_in, mask])

    pad_c1 = (7 - 1) * 1
    s_c1 = Input(shape=(pad_c1, 1), name="realtime_state_picklearn_c1")
    state_inputs.append(s_c1)
    x_cat1 = ConcatTime(name="realtime_picklearn_c1_concat")([s_c1, x_pa])
    x_pa1 = SeparableConv1D(16, 7, dilation_rate=1, padding="valid", name="realtime_picklearn_c1")(x_cat1)
    x_pa1 = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="realtime_picklearn_c1_relu")(x_pa1)
    s_c1n = TakeLastN(pad_c1, name="realtime_state_picklearn_c1_next")(x_cat1)
    state_outputs.append(s_c1n)

    pad_c2 = (7 - 1) * 2
    s_c2 = Input(shape=(pad_c2, 16), name="realtime_state_picklearn_c2")
    state_inputs.append(s_c2)
    x_cat2 = ConcatTime(name="realtime_picklearn_c2_concat")([s_c2, x_pa1])
    x_pa2 = SeparableConv1D(16, 7, dilation_rate=2, padding="valid", name="realtime_picklearn_c2")(x_cat2)
    x_pa2 = tf.keras.layers.LeakyReLU(negative_slope=0.1, name="realtime_picklearn_c2_relu")(x_pa2)
    s_c2n = TakeLastN(pad_c2, name="realtime_state_picklearn_c2_next")(x_cat2)
    state_outputs.append(s_c2n)

    x_pa_out = Conv1D(1, 1, padding="same", name="realtime_picklearn_out")(x_pa2)
    xg = Multiply(name="realtime_pickattack_gated")([x_pa_out, style_scalar_b3])
    xg = SoftClip1D(drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickattack_softclip")(xg)

    # NEW: streaming fade-in to suppress HPF startup pop (≈ 3 ms)
    s_pa_ramp = Input(shape=(1,1), name="realtime_state_pickattack_ramp")
    state_inputs.append(s_pa_ramp)
    ramp_pa, s_pa_ramp_n = StreamingLinearRamp(ramp_len=int(0.003 * REALTIME_SAMPLE_RATE),
                                               name="realtime_pickattack_ramp")([xg, s_pa_ramp])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_ramp_next")(s_pa_ramp_n))
    xg = Multiply(name="realtime_pickattack_ramp_apply")([xg, ramp_pa])

    # Pick-attack post HPF (exact-length taps, 2 kHz cutoff for realtime)
    s_hp_pa = Input(shape=(PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1, 1), name="realtime_state_pickattack_posthp")
    state_inputs.append(s_hp_pa)
    L_pa = int(PICK_ATTACK_LOWBAND_PENALTY_ORDER)
    hp_pa_k_full = _hp_minphase_rev_padded(REALTIME_SAMPLE_RATE, PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ, L_pa)
    xh_pa, s_hp_pa_n = StreamFixedFIR(kernel_len=L_pa, out_ch=1, weight_name="hp_taps",
                                      trainable_kernel=False, taps_init=hp_pa_k_full,
                                      name="realtime_pickattack_post_hp")([xg, s_hp_pa])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_posthp_next")(s_hp_pa_n))
    y_pa = IrScale(init=1.0, name="realtime_pickattack_scale")(xh_pa)
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y_pa = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=limit_lin, name="realtime_pickattack_limit")(y_pa)

    # Pick-reso streaming
    s_on_prev = Input(shape=(1,1), name="realtime_state_pickreso_onset_prev")
    state_inputs.append(s_on_prev)
    onset, s_on_prev_n = StreamingDiffPos(name="realtime_pickreso_onset_stream")([mask, s_on_prev])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_onset_prev_next")(s_on_prev_n))

    s_on = int(round((RESBP_STRIKE_ENV_MS_REALTIME / 1000.0) * REALTIME_SAMPLE_RATE))
    L_env = int(np.clip(np.ceil(4.0 * max(1, s_on)), 8, 512))
    s_env = Input(shape=(L_env - 1, 1), name="realtime_state_pickreso_onset_smooth")
    state_inputs.append(s_env)
    k_on = _exp_smoothing_kernel(max(1, s_on), scale=4.0, Lmax=L_env)
    strike, s_env_n = StreamFixedFIR(kernel_len=L_env, out_ch=1, weight_name="exp_smooth_taps",
                                     trainable_kernel=False, taps_init=k_on,
                                     name="realtime_pickreso_onset_pulse")([onset, s_env])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_onset_smooth_next")(s_env_n))

    s_bp = Input(shape=(REALTIME_PICK_RESO_BP_ORDER - 1, 1), name="realtime_state_pickreso_ctrl_bp")
    state_inputs.append(s_bp)
    bp_lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    bp_hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    bp_k_full = _bp_minphase_rev_padded(REALTIME_SAMPLE_RATE, bp_lo, bp_hi, REALTIME_PICK_RESO_BP_ORDER)
    bp_out, s_bp_n = StreamFixedFIR(kernel_len=REALTIME_PICK_RESO_BP_ORDER, out_ch=1, weight_name="bp_taps",
                                    trainable_kernel=False, taps_init=bp_k_full,
                                    name="realtime_pickreso_ctrl_bp")([inp, s_bp])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_bp_next")(s_bp_n))
    amp_raw = Abs1D()(bp_out)

    s_env2_len = int(np.clip(np.ceil(3.0 * max(1, int(round(3.0/1000.0 * REALTIME_SAMPLE_RATE)))), 8, 256))
    s_env2 = Input(shape=(s_env2_len - 1, 1), name="realtime_state_pickreso_ctrl_env")
    state_inputs.append(s_env2)
    tau_env2 = max(1, int(round((3.0/1000.0) * REALTIME_SAMPLE_RATE)))
    k_env2 = _exp_smoothing_kernel(tau_env2, scale=4.0, Lmax=s_env2_len)
    amp, s_env2_n = StreamFixedFIR(kernel_len=s_env2_len, out_ch=1, weight_name="exp_smooth_taps",
                                   trainable_kernel=False, taps_init=k_env2,
                                   name="realtime_pickreso_ctrl_env")([amp_raw, s_env2])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_env_next")(s_env2_n))
    amp = Lambda(clip_01x4, name="realtime_pickreso_ctrl_gain")(amp)

    exc = Multiply(name="realtime_pickreso_excite")([strike, amp])

    s_ds8 = Input(shape=(REALTIME_PICK_RESO_DS_TAPS - 1, 1), name="realtime_state_pickreso_ds8")
    state_inputs.append(s_ds8)
    ds8, s_ds8_n = StreamingRealtimeIrDownsampleFIR(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                                    kernel_size=REALTIME_PICK_RESO_DS_TAPS,
                                                    use_ir_prototype=False,
                                                    name="realtime_pickreso_ds8")([exc, s_ds8])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ds8_next")(s_ds8_n))

    num_modes = len(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    L_lr = _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    s_mlr = Input(shape=(L_lr - 1, 1), name="realtime_state_pickreso_modal_lr")
    state_inputs.append(s_mlr)
    modes_lr, s_mlr_n = StreamFixedFIR(kernel_len=L_lr, out_ch=num_modes, weight_name="modal_kernels_lr",
                                       trainable_kernel=True, name="realtime_pickreso_modal_lr")([ds8, s_mlr])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_modal_lr_next")(s_mlr_n))

    mixed_lr = Conv1D(1, 1, use_bias=True, padding="same", name="realtime_pickreso_mix")(modes_lr)

    s_interp = Input(shape=(REALTIME_PICK_RESO_INTERP_TAPS - 1, 1), name="realtime_state_pickreso_interp")
    state_inputs.append(s_interp)
    up, s_interp_n = StreamingFIRInterpByN(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                           num_taps=REALTIME_PICK_RESO_INTERP_TAPS,
                                           cutoff=0.45/REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                           window=('kaiser', 9.0),
                                           name="realtime_pickreso_interp")([mixed_lr, s_interp])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_interp_next")(s_interp_n))
    mixed = MatchTimeToRef(name="realtime_pickreso_matchlen")([up, inp])

    xg_reso = Multiply(name="realtime_pickreso_gated")([mixed, style_scalar_b3])

    s_hp_reso = Input(shape=(PICK_RESO_LOWBAND_PENALTY_ORDER - 1, 1), name="realtime_state_pickreso_posthp")
    state_inputs.append(s_hp_reso)
    L_pr = int(PICK_RESO_LOWBAND_PENALTY_ORDER)
    hp_pr_k_full = _hp_minphase_rev_padded(REALTIME_SAMPLE_RATE, PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ, L_pr)
    xh_reso, s_hp_reso_n = StreamFixedFIR(kernel_len=L_pr, out_ch=1, weight_name="hp_taps",
                                          trainable_kernel=False, taps_init=hp_pr_k_full,
                                          name="realtime_pickreso_post_hp")([xg_reso, s_hp_reso])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_posthp_next")(s_hp_reso_n))
    xh_reso = SoftClip1D(drive=REALTIME_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickreso_softclip")(xh_reso)
    y_reso = IrScale(init=1.0, name="realtime_pickreso_scale")(xh_reso)
    y_reso = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0), name="realtime_pickreso_limit")(y_reso)
    y_reso = IrScale(init=1.0, name="realtime_pickreso_calib")(y_reso)

    # Mix stems and coloreq (unchanged)
    pickattack = apply_runtime_gain(y_pa, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(y_reso, knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,   knob_ir_in,   name="realtime_ir_user_gain")
    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main, ir, pickattack, pickreso])

    # Color EQ streaming bandpasses
    coloreq_states_in  = []
    coloreq_states_out = []
    y_terms = [y_prehp]
    for bi, (lo, hi) in enumerate(COLOREQ_BANDS_REALTIME_BODY):
        s_b = Input(shape=(65 - 1, 1), name=f"realtime_state_coloreq_bp_{bi}")
        coloreq_states_in.append(s_b)
        bp_y, s_b_n = StreamFixedFIR(kernel_len=65, out_ch=1, weight_name="bp_taps",
                                     trainable_kernel=False, name=f"realtime_coloreq_bp_{bi}")([y_prehp, s_b])
        coloreq_states_out.append(Activation('linear', name=f"realtime_state_coloreq_bp_{bi}_next")(s_b_n))
        g = IrScale(init=0.0, name=f"realtime_coloreq_g_{bi}", trainable=bool(coloreq_trainable))(bp_y)
        y_terms.append(g)

    y_prehp = LossTether(name="realtime_pick_lowband_penalties_stream")([y_prehp])
    y_wet = Add(name="realtime_coloreq_sum")(y_terms)

    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_final    = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    inputs_list  = [inp, style_true_in, style_force_in, knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in]
    inputs_list += state_inputs + coloreq_states_in
    outputs_list = [y_final, style_prob] + state_outputs + coloreq_states_out

    return tf.keras.models.Model(inputs=inputs_list, outputs=outputs_list, name=name)

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
def add_shared_random_gain_offline(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to the offline student input and the mic target.
    Because the teacher is embedded in the finetune model and reuses this same input tensor,
    it automatically sees the same level, making mask KD invariant to absolute RMS.
    """
    def _map(inputs, outputs, sw):
        g_db = tf.random.uniform((), db_lo, db_hi, dtype=tf.float32)
        g = tf.pow(10.0, g_db / 20.0)

        # Student input (teacher reuses this)
        inputs['offline_piezo_input'] = inputs['offline_piezo_input'] * g

        # Scale only the audio target; style target is dummy and unused
        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_shared_random_gain_dual(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to:
      - realtime_piezo_input (student, 48k)
      - realtime_offline_piezo_input (teacher, 96k)
      - y_audio target (mic)
    This makes mask KD invariant to absolute input level.
    """
    def _map(inputs, outputs, sw):
        g_db = tf.random.uniform((), db_lo, db_hi, dtype=tf.float32)
        g = tf.pow(10.0, g_db / 20.0)

        xr  = inputs['realtime_piezo_input']
        xrt = inputs.get('realtime_offline_piezo_input', None)

        inputs['realtime_piezo_input'] = xr * g
        if xrt is not None:
            inputs['realtime_offline_piezo_input'] = xrt * g

        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_sw_offline(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    # Default knob values for training: PA/PR unity; dry = default dryness
    dry_default = tf.constant(_default_dryness_value(), dtype=tf.float32)
    inputs = {
        'offline_piezo_input': x,
        'offline_style_true_input': style_true,
        'offline_style_force_input': style_force,
        'offline_knob_pickattack': tf.ones((b,1), tf.float32),
        'offline_knob_pickreso':   tf.ones((b,1), tf.float32),
        'offline_knob_drywet':     tf.ones((b,1), tf.float32) * dry_default,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_realtime(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)

    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    dry_default = tf.constant(_default_dryness_value(), dtype=tf.float32)
    inputs = {
        'realtime_piezo_input': x,
        'realtime_style_true_input': style_true,
        'realtime_style_force_input': style_force,
        'realtime_knob_pickattack': tf.ones((b,1), tf.float32),
        'realtime_knob_pickreso':   tf.ones((b,1), tf.float32),
        'realtime_knob_ir':         tf.ones((b,1), tf.float32),
        'realtime_knob_drywet':     tf.ones((b,1), tf.float32) * dry_default,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_realtime_dual_nf(x, y):
    x_realtime, x_offline = x
    y_audio, _ = y
    b = tf.shape(x_realtime)[0]
    T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    style_force = tf.zeros((b, 1), dtype=tf.float32)

    dry_default = tf.constant(_default_dryness_value(), dtype=tf.float32)
    inputs = {
        'realtime_piezo_input':         x_realtime,
        'realtime_offline_piezo_input': x_offline,
        'realtime_style_true_input':    style_true,
        'realtime_style_force_input':   style_force,
        'realtime_knob_pickattack':     tf.ones((b,1), tf.float32),
        'realtime_knob_pickreso':       tf.ones((b,1), tf.float32),
        'realtime_knob_ir':             tf.ones((b,1), tf.float32),
        'realtime_knob_drywet':         tf.ones((b,1), tf.float32) * dry_default,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_realtime_dual(x, y):
    x_realtime, x_offline = x
    y_audio, style_flag = y
    b = tf.shape(x_realtime)[0]; T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)

    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    dry_default = tf.constant(_default_dryness_value(), dtype=tf.float32)
    inputs = {
        'realtime_piezo_input':         x_realtime,
        'realtime_offline_piezo_input': x_offline,
        'realtime_style_true_input':    style_true,
        'realtime_style_force_input':   style_force,
        'realtime_knob_pickattack':     tf.ones((b,1), tf.float32),
        'realtime_knob_pickreso':       tf.ones((b,1), tf.float32),
        'realtime_knob_ir':             tf.ones((b,1), tf.float32),
        'realtime_knob_drywet':         tf.ones((b,1), tf.float32) * dry_default,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_offline_finetune_nf(x, y):
    y_audio, _ = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]

    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    style_force = tf.zeros((b, 1), dtype=tf.float32)

    dry_default = tf.constant(_default_dryness_value(), dtype=tf.float32)
    inputs = {
        'offline_piezo_input':        x,
        'offline_style_true_input':   style_true,
        'offline_style_force_input':  style_force,
        'offline_knob_pickattack':    tf.ones((b,1), tf.float32),
        'offline_knob_pickreso':      tf.ones((b,1), tf.float32),
        'offline_knob_drywet':        tf.ones((b,1), tf.float32) * dry_default,
    }

    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

def fine_tune(offline=True, realtime=True):
    # IO
    src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
    tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    if not os.path.exists(src) or not os.path.exists(tgt):
        print("ERROR: finetune source/target not found in", FINETUNE_SRC_DIR); sys.exit(1)

    x_full, sr_x = sf.read(src, dtype='float32', always_2d=True)
    y_full, sr_y = sf.read(tgt, dtype='float32', always_2d=True)
    if sr_x != sr_y:
        print(f"Warning: finetune pair SR mismatch: {sr_x} vs {sr_y}")
    y_full = denoise_target_if_enabled(y_full, sr_y)

    cached = precache_finetune_resampled_pair(
        x_full, y_full, file_sr=sr_x,
        sr_rt=REALTIME_SAMPLE_RATE, sr_off=OFFLINE_SAMPLE_RATE
    )

    # Lightweight fine-tune losses
    loss_offlite = FinetuneCombinedLossLite(sample_rate=OFFLINE_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0, ulb_w=0.20,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER)
    loss_rt_lite = FinetuneCombinedLossLite(sample_rate=REALTIME_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER)

    off_dyn_export = None
    rt_dyn_export  = None

    if offline:
        print("\n🔧 Finetuning OFFLINE base (coloreq + film + scales only)…")
        base_off_dyn = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build the OFFLINE training graph (same as runtime graph)
        off_static_train = build_offline_model()
        copy_weights_by_name(off_static_train, base_off_dyn, verbose=True)

        # Unfreeze only FiLM + coloreq
        off_static_train = set_trainable_layers(
            off_static_train,
            enable_film=True,
            enable_coloreq=True
        )

        # Dataset
        dsO = make_mixed_ds_from_arrays_offline(
            cached["x96"], cached["y96"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len=OFFLINE_SEG_LEN
        ).map(add_sw_offline_finetune_nf
        ).map(add_shared_random_gain_offline(db_lo=-30.0, db_hi=+18.0)
        ).prefetch(tf.data.AUTOTUNE)

        steps_off = approx_steps_from_length(
            len(cached["x96"]), OFFLINE_SEG_LEN, FINETUNE_BATCH_SIZE, cap=FINETUNE_OFFLINE_STEPS_CAP
        )

        off_static_train.compile(optimizer=tf.keras.optimizers.Adam(FINETUNE_OFFLINE_LR),
                                 loss=[loss_offlite, "binary_crossentropy"],
                                 loss_weights=[1.0, 0.0],
                                 metrics=[["mae"], []])
        off_static_train.fit(dsO, epochs=OFFLINE_FINETUNE_EPOCHS, steps_per_epoch=steps_off, verbose=1)

        # Export OFFLINE dynamic
        off_dyn_export = build_offline_model(dynamic_time=True)
        copy_weights_by_name(off_dyn_export, off_static_train, verbose=True)

        ft_offline_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        off_dyn_export.save(ft_offline_dyn_path)
        print("✅ Saved finetuned offline model (dynamic):", ft_offline_dyn_path)

        ft_offline_static_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
        off_static_train.save(ft_offline_static_path)
        print("✅ Saved finetuned offline model (static):", ft_offline_static_path)

    if realtime:
        print("\n🔧 Finetuning REALTIME base (coloreq + film + scales only)…")

        offline_nc = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)
        offline_causal = build_offline_model_causal(offline_nc, dynamic_time=True)

        base_rt_dyn = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build KD training graph (teacher embedded) and copy weights from baseline
        rt_static_train = build_realtime_training_model(
            offline=offline_causal
        )
        copy_weights_by_name(rt_static_train, base_rt_dyn, verbose=True)

        # Unfreeze only FiLM + coloreq (and scales if desired)
        rt_static_train = set_trainable_layers(
            rt_static_train,
            enable_film=True,
            enable_coloreq=True
        )

        # KD dual-rate dataset
        dsRT_dual = make_mixed_ds_from_arrays_realtime_dual(
            cached["x48"], cached["x96"], cached["y48"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len_rt=REALTIME_SEG_LEN
        ).map(add_sw_realtime_dual_nf
        ).map(add_shared_random_gain_dual(db_lo=-30.0, db_hi=+18.0)
        ).prefetch(tf.data.AUTOTUNE)

        steps_rt = approx_steps_from_length(
            len(cached["x48"]), REALTIME_SEG_LEN, FINETUNE_BATCH_SIZE, cap=FINETUNE_REALTIME_STEPS_CAP
        )

        rt_static_train.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_REALTIME_LR, clipnorm=1.0),
                                loss=[loss_rt_lite, "binary_crossentropy"],
                                loss_weights=[1.0, 0.0],
                                metrics=[["mae"], []])
        rt_static_train.fit(dsRT_dual, epochs=REALTIME_FINETUNE_EPOCHS, steps_per_epoch=steps_rt, verbose=1)
        
        ft_rt_trainable_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME)
        rt_static_train.save(ft_rt_trainable_path)
        print("✅ Saved finetuned realtime training model:", ft_rt_trainable_path)

        # Export self-contained streaming runtime artifacts (dynamic + static)
        rt_dyn_stream = build_realtime_streaming_model(dynamic_time=True)
        copy_weights_by_name(rt_dyn_stream, rt_static_train, verbose=True)
        ft_rt_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        rt_dyn_stream.save(ft_rt_dyn_path)
        print("✅ Saved finetuned realtime streaming dynamic model:", ft_rt_dyn_path)

        rt_static_stream = build_realtime_streaming_model()
        copy_weights_by_name(rt_static_stream, rt_static_train, verbose=True)
        ft_rt_static_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        rt_static_stream.save(ft_rt_static_path)
        print("✅ Saved finetuned realtime streaming static model:", ft_rt_static_path)

        # Return dynamic stream (for optional demo)
        rt_dyn_export = rt_dyn_stream

    return off_dyn_export, rt_dyn_export

def set_trainable_layers(model,
                         enable_film=False,
                         enable_scales=False,
                         enable_coloreq=False,
                         enable_mask_head=False,
                         enable_heads=False,
                         enable_style_head=False,
                         enable_style_blend=False):
    """
    Freeze everything, then selectively unfreeze groups:
      - enable_film:    Dense layers generating FiLM params (name contains 'film_dense')
      - enable_scales:  All IrScale except coloreq (pickattack_scale, pickreso_scale, ir_scale, etc.)
      - enable_coloreq: Only coloreq IrScale bands (name starts with '*_coloreq_g_')
      - enable_mask_head: hf mask tiny conv stack (offline_*_hf_mask_* and realtime_* equivalents)
      - enable_heads:   final pick/finger heads (Conv1D layers named '*_final_pick'/'*_final_finger')
      - enable_style_head: style classifier tower (conv/gap/logit/prob)
      - enable_style_blend: realtime_style_blend scalar (if present)
    Works for both offline_* and realtime_* prefixes.
    """
    # 1) Freeze all
    for l in model.layers:
        l.trainable = False

    def is_coloreq_scale(layer):
        return isinstance(layer, IrScale) and ("_coloreq_g_" in layer.name)

    def is_regular_scale(layer):
        # Any IrScale that is not a coloreq scale
        return isinstance(layer, IrScale) and not is_coloreq_scale(layer)

    trainable_names = []

    # 2) Film generators (shared for offline/realtime by name convention)
    if enable_film:
        for l in model.layers:
            if "film_dense" in l.name:
                l.trainable = True
                trainable_names.append(l.name)

    # 3) Scales (all IrScale except coloreq)
    if enable_scales:
        for l in model.layers:
            if is_regular_scale(l):
                l.trainable = True
                trainable_names.append(l.name)

    # 4) COLOREQ band scales only
    if enable_coloreq:
        for l in model.layers:
            if is_coloreq_scale(l):
                l.trainable = True
                trainable_names.append(l.name)

    # 5) Mask tiny head (offline/realtime prefixes)
    if enable_mask_head:
        for lname in [
            "offline_hf_mask_c1", "offline_hf_mask_c2", "offline_hf_mask_logits",
            "realtime_hf_mask_c1","realtime_hf_mask_c2","realtime_hf_mask_logits"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

    # 6) Final pick/finger heads (Conv1D)
    if enable_heads:
        for lname in [
            "offline_final_pick", "offline_final_finger",
            "realtime_final_pick","realtime_final_finger"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

    # 7) Style classifier tower (disable by default to keep invariance)
    if enable_style_head:
        for lname in [
            "offline_style_conv", "offline_style_gap", "offline_style_logit", "offline_style_prob",
            "realtime_style_conv","realtime_style_gap","realtime_style_logit","realtime_style_prob"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

        # embedding Dense
        for lname in ["offline_style_embed","realtime_style_embed"]:
            try:
                layer = model.get_layer(lname)
                layer.trainable = True
                trainable_names.append(layer.name)
            except Exception:
                pass

    # 8) Realtime style_blend scalar (optional)
    if enable_style_blend:
        try:
            layer = model.get_layer("realtime_style_blend")
            layer.trainable = True
            trainable_names.append(layer.name)
        except Exception:
            pass

    print(f"[trainable] enabled: {sorted(set(trainable_names))}")
    return model

def build_offline_debug_model(model):
    used = set([l.name for l in model.layers])

    def unique_alias(base):
        root = f"dbg_{base}"
        name = root
        k = 1
        while name in used:
            name = f"{root}_{k}"
            k += 1
        used.add(name)
        return name

    outs = []
    def add(alias, lname):
        try:
            t = model.get_layer(lname).output
        except Exception:
            return
        outs.append(Activation('linear', name=unique_alias(alias))(t))

    # Full mix AFTER dry/wet so stems 'full' respects knob
    add("y_pred",       "offline_drywet_mix")
    # Fallback:
    if not outs:
        add("y_pred",   "offline_with_picklearn_prehp")

    add("main_core",    "offline_main_core")
    add("main_pick",    "offline_pick_contrib")
    add("main_finger",  "offline_finger_contrib")
    add("pickreso",     "offline_pickreso_user_gain_mul")
    add("pickattack",   "offline_pickattack_user_gain_mul")
    add("style_pred",   "offline_style_prob")
    add("mask",         "offline_pick_mask_db")
    add("style_time",   "offline_style_time_used")

    return tf.keras.Model(inputs=model.inputs, outputs=outs, name="offline_debug_model")

def build_realtime_debug_model(model):
    used = set([l.name for l in model.layers])

    def unique_alias(base):
        root = f"dbg_{base}"
        name = root
        k = 1
        while name in used:
            name = f"{root}_{k}"
            k += 1
        used.add(name)
        return name

    outs = []

    def add(alias, lname):
        try:
            t = model.get_layer(lname).output
        except Exception:
            return
        outs.append(Activation("linear", name=unique_alias(alias))(t))

    # Main/full mix AFTER dry/wet so stems 'full' respects knob
    add("y_pred", "realtime_drywet_mix")

    # Useful stems/features
    add("main_core",   "realtime_main")
    add("main_pick",   "realtime_pick_contrib")
    add("main_finger", "realtime_finger_contrib")
    add("ir_only",     "realtime_ir_user_gain_mul")
    add("pickreso",    "realtime_pickreso_user_gain_mul")
    add("pickattack",  "realtime_pickattack_user_gain_mul")
    add("style_pred",  "realtime_style_prob")
    add("mask",        "realtime_pick_mask_db")
    add("style_time",  "realtime_style_scalar_broadcast")

    # Also expose all streaming state outputs so the renderer can advance state.
    # These are already named uniquely inside the base model.
    for l in model.layers:
        nm = l.name
        if nm.startswith("realtime_state_") and nm.endswith("_next"):
            try:
                outs.append(model.get_layer(nm).output)
            except Exception:
                pass

    return tf.keras.Model(inputs=model.inputs, outputs=outs, name="realtime_debug_model")

def _rms(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(x*x) + 1e-12))

def _corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.size, b.size)
    if n < 8: return 0.0
    a = a[:n]; b = b[:n]
    den = np.sqrt((np.sum(a*a)+1e-12) * (np.sum(b*b)+1e-12))
    if den <= 0: return 0.0
    return float(np.sum(a*b) / den)

def render_isolated_stems_realtime_runtime(model,
                                           path_src,
                                           path_tgt,
                                           out_dir,
                                           sample_rate,
                                           sec=LONG_DEMO_SEC,
                                           offset_sec=LONG_DEMO_OFFSET_SEC,
                                           hop=1024,
                                           use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)

    def rms(x):
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0

    # Read/resample
    info = sf.info(path_src)
    file_sr = info.samplerate
    start = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x_blk_2d, _ = sf.read(path_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_blk_2d, _ = sf.read(path_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:, 0]; y_file = y_blk_2d[:, 0]
    x_mod = x_file if file_sr == sample_rate else resample_to_model_sr(x_file, file_sr, sample_rate)
    y_mod = y_file if file_sr == sample_rate else resample_to_model_sr(y_file, file_sr, sample_rate)

    # Build a tap-model that exposes post‑knob heads and main
    taps = []
    found = {}
    def tap(lname, tname):
        try:
            t = model.get_layer(lname).output
            taps.append(Activation('linear', name=tname)(t))
            found[tname] = True
            return True
        except Exception:
            found[tname] = False
            return False

    tap("realtime_pickattack_user_gain_mul", "tap_pickattack")
    tap("realtime_pickreso_user_gain_mul",   "tap_pickreso")
    tap("realtime_ir_user_gain_mul",         "tap_ir")
    tap("realtime_main",                     "tap_main")

    tap_model = tf.keras.Model(inputs=model.inputs, outputs=list(model.outputs) + taps, name=model.name + "_taps")

    # Names / mapping
    def _bn(t):
        nm = t.name.split(':')[0]
        if '/' in nm: nm = nm.split('/', 1)[0]
        if nm.startswith("serving_default_"): nm = nm[len("serving_default_"):]
        return nm
    in_tensors = list(tap_model.inputs)
    in_names   = [_bn(t) for t in in_tensors]
    out_names  = _output_names_from_model(tap_model)
    out_idx    = {n: i for i, n in enumerate(out_names)}

    # State mapping
    state_in_names = [n for n in in_names if n.startswith("realtime_state_")]
    state_out_idx = {}
    for n in state_in_names:
        nn = n + "_next"
        if nn in out_idx:
            state_out_idx[n] = out_idx[nn]
    state_bufs = {}
    for t in in_tensors:
        nm = _bn(t)
        if nm in state_in_names:
            L = int(t.shape[1]) if t.shape[1] is not None else 0
            C = int(t.shape[2]) if t.shape[2] is not None else 1
            state_bufs[nm] = np.zeros((1, L, C), dtype=np.float32)

    # Knobs
    pa_lin, pr_lin, ir_lin = _resolve_demo_knobs(kind="realtime", use_finetuned=use_finetuned)
    dry_default = _default_dryness_value()
    def knobs(pa, pr, ir, dry=dry_default):
        return {
            "realtime_knob_pickattack": np.array([[float(pa)]], dtype=np.float32),
            "realtime_knob_pickreso":   np.array([[float(pr)]], dtype=np.float32),
            "realtime_knob_ir":         np.array([[float(ir)]], dtype=np.float32),
            "realtime_knob_drywet":     np.array([[float(dry)]], dtype=np.float32),
        }
    def filter_feed(d): return {k: v for k, v in d.items() if k in in_names}

    # Single-pass streamer
    def run_stream(style_true=None, style_force=False, pa=None, pr=None, ir=None):
        for k in state_bufs: state_bufs[k].fill(0.0)
        pos, N = 0, len(x_mod)
        y_full = []; y_main = []; y_pa = []; y_pr = []; y_ir = []
        st_true  = np.zeros((1,1), np.float32) if style_true is None else np.array([[float(style_true)]], np.float32)
        st_force = np.zeros((1,1), np.float32) if not style_force else np.ones((1,1), np.float32)
        pa_val = pa if pa is not None else pa_lin
        pr_val = pr if pr is not None else pr_lin
        ir_val = ir if ir is not None else ir_lin

        while pos < N:
            core = min(hop, N - pos)
            xb = np.zeros((1, hop, 1), np.float32); xb[0, :core, 0] = x_mod[pos:pos+core]
            feed = {
                "realtime_piezo_input": xb,
                "realtime_style_true_input":  st_true,
                "realtime_style_force_input": st_force,
                **knobs(pa=pa_val, pr=pr_val, ir=ir_val, dry=dry_default)
            }
            for n in state_in_names: feed[n] = state_bufs[n]
            outs = tap_model.predict(filter_feed(feed), batch_size=1, verbose=0)

            # audio is first output
            y_full.append(np.asarray(outs[0], np.float32)[0, :core, 0])

            def take(name):
                idx = out_idx.get(name, None)
                if idx is None or idx >= len(outs): return None
                v = np.asarray(outs[idx], np.float32)
                if v.ndim == 3: return v[0, :core, 0]
                if v.ndim == 2: return v[0, :core]
                return None

            # taps
            tmn = take("tap_main") if found.get("tap_main", False) else None
            y_main.append(tmn if tmn is not None else y_full[-1])
            tpa = take("tap_pickattack") if found.get("tap_pickattack", False) else None
            tpr = take("tap_pickreso")   if found.get("tap_pickreso",   False) else None
            tir = take("tap_ir")         if found.get("tap_ir",         False) else None
            y_pa.append(tpa if tpa is not None else np.zeros((core,), np.float32))
            y_pr.append(tpr if tpr is not None else np.zeros((core,), np.float32))
            y_ir.append(tir if tir is not None else np.zeros((core,), np.float32))

            # advance states
            for n, i in state_out_idx.items():
                if i < len(outs): state_bufs[n] = outs[i]
            pos += core

        cat = lambda parts: np.concatenate(parts, axis=0) if parts else np.zeros(0, np.float32)
        return {"full":cat(y_full), "main":cat(y_main), "pa":cat(y_pa), "pr":cat(y_pr), "ir":cat(y_ir)}

    # DC removal + peak protection for stems
    def finalize(sig, headroom_db=1.0, dc_remove=True):
        s = np.asarray(sig, np.float32)
        if s.size == 0:
            return s
        if dc_remove:
            s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)) + 1e-12)
        ceiling = 10.0 ** (float(-headroom_db) / 20.0)  # ≈0.891 for 1 dB
        if peak > ceiling:
            s = s * (ceiling / peak)
        return s

    # Passes and writers
    passes = {
        "pred":        {"style_true": None, "style_force": False},
        "forced_pick": {"style_true": 1.0,  "style_force": True},
        "forced_fing": {"style_true": 0.0,  "style_force": True},
    }
    writers = {}
    def open_writers(key):
        if key == "pred":
            writers["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            writers["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            writers["full_pred"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        elif key == "forced_pick":
            writers["full_forced_pick"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            writers["full_forced_finger"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"main_core_{key}"] = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only_{key}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"pickattack_{key}"] = sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only_{key}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"pickreso_{key}"]   = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only_{key}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"ir_{key}"]         = sf.SoundFile(os.path.join(out_dir, f"stem_ir_lowpass_{key}.wav"),     mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"sumcheck_{key}"]   = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck_{key}.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

    print("[stems/realtime] taps found:", found)

    for key, cfg in passes.items():
        open_writers(key)
        res = run_stream(style_true=cfg["style_true"], style_force=cfg["style_force"])

        # Finalize stems (DC removal + peak protection)
        main_f = finalize(res["main"], headroom_db=1.0, dc_remove=True)
        pa_f   = finalize(res["pa"],   headroom_db=1.0, dc_remove=True)
        pr_f   = finalize(res["pr"],   headroom_db=1.0, dc_remove=True)
        ir_f   = finalize(res["ir"],   headroom_db=1.0, dc_remove=True)

        # Write full
        if key == "pred":
            writers["src"].write(x_mod); writers["tgt"].write(y_mod); writers["full_pred"].write(res["full"])
        elif key == "forced_pick":
            writers["full_forced_pick"].write(res["full"])
        else:
            writers["full_forced_finger"].write(res["full"])

        # Write stems (finalized)
        writers[f"main_core_{key}"].write(main_f)
        writers[f"pickattack_{key}"].write(pa_f)
        writers[f"pickreso_{key}"].write(pr_f)
        writers[f"ir_{key}"].write(ir_f)

        # Sumcheck (use finalized stems)
        sum_est = np.clip(main_f + pa_f + pr_f + ir_f, -1.0, 1.0)
        writers[f"sumcheck_{key}"].write(sum_est)

        print(f"[stems/realtime][{key}] wrote RMS main={rms(main_f):.6f} pa={rms(pa_f):.6f} pr={rms(pr_f):.6f} ir={rms(ir_f):.6f}")

    for w in list(writers.values()):
        try: w.close()
        except Exception: pass

    print(f"✅ Realtime stems (pred + forced_pick + forced_fing) written to {out_dir}")

def render_isolated_stems(debug_model,
                          kind,
                          path_src,
                          path_tgt,
                          out_dir,
                          sample_rate,
                          sec=LONG_DEMO_SEC,
                          offset_sec=LONG_DEMO_OFFSET_SEC,
                          block_sec=12.0,
                          resample_margin_ms=50.0,
                          use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)
    knob_feeds = _prepare_knob_feeds(kind=kind, use_finetuned=use_finetuned)

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))

    x_blk_2d, _ = sf.read(path_src, start=START_FRAME, frames=TOTAL_FRAMES, dtype='float32', always_2d=True)
    y_blk_2d, _ = sf.read(path_tgt, start=START_FRAME, frames=TOTAL_FRAMES, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:, 0]
    y_file = y_blk_2d[:, 0]

    if file_sr != sample_rate:
        x_mod = resample_to_model_sr(x_file, file_sr, sample_rate)
        y_mod = resample_to_model_sr(y_file,  file_sr, sample_rate)
    else:
        x_mod = x_file
        y_mod = y_file

    # Determine if model is streaming (has explicit realtime_state_* inputs)
    in_tensors = list(debug_model.inputs)
    in_names = [t.name.split(':')[0].split('/')[-1] for t in in_tensors]
    state_in_names = [n for n in in_names if n.startswith("realtime_state_")]
    streaming = (kind == "realtime") and (len(state_in_names) > 0)

    # Map outputs by name (producer layer names) for robust dict handling
    def predict_to_dict(model, feeds):
        out_any = model.predict(feeds, batch_size=1, verbose=0)
        out_names = [t.name.split(':')[0].split('/')[-1] for t in model.outputs]
        if isinstance(out_any, dict):
            return out_any
        if isinstance(out_any, (list, tuple)):
            return {out_names[i]: out_any[i] for i in range(min(len(out_names), len(out_any)))}
        return {out_names[0]: out_any}

    # Prepare writers
    def open_writers(pass_key):
        is_pred = (pass_key == "pred")
        suffix = "" if is_pred else f"_{pass_key.split('_', 1)[1]}"
        w = {}
        if is_pred:
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, f"pred_full_forced{suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

        w["main_core"] = sf.SoundFile(os.path.join(out_dir, "stem_main_core_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                      mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pick"] = sf.SoundFile(os.path.join(out_dir, "stem_pick_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["fing"] = sf.SoundFile(os.path.join(out_dir, "stem_finger_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickattack"] = None
        w["picklearn"]  = None
        w["ir"]         = None
        w["pickreso"]   = None
        w["sumcheck"] = sf.SoundFile(os.path.join(out_dir, "pred_full_sumcheck" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                     mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        return w

    forced_variants = [
        ("pred",         np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)),
        ("forced_pick",  np.ones((1, 1),  dtype=np.float32), np.ones((1, 1),  dtype=np.float32)),
        ("forced_finger",np.zeros((1, 1), dtype=np.float32), np.ones((1, 1),  dtype=np.float32)),
    ]
    pass_writers = {k: open_writers(k) for k, _, _ in forced_variants}

    # Helper to try dbg_ aliases
    out_name_set = set([t.name.split(':')[0].split('/')[-1] for t in debug_model.outputs])
    def aliases(k):
        arr = [k, f"dbg_{k}"]
        if kind == "realtime" and k == "y_pred":
            arr.append("realtime_drywet_mix")
        return arr

    # If streaming: prepare state buffers and a hop size
    if streaming:
        # Shapes for zero-initialized states
        state_in_tensors = [t for t in in_tensors if t.name.split(':')[0].split('/')[-1] in state_in_names]
        state_bufs = {}
        for t in state_in_tensors:
            nm = t.name.split(':')[0].split('/')[-1]
            s1 = int(t.shape[1]) if t.shape[1] is not None else 0
            s2 = int(t.shape[2]) if t.shape[2] is not None else 1
            state_bufs[nm] = np.zeros((1, s1, s2), dtype=np.float32)

        # Build map input_state_name -> output_state_name index (use name + "_next")
        out_names = [t.name.split(':')[0].split('/')[-1] for t in debug_model.outputs]
        state_out_map = {}
        for nm in state_in_names:
            cand = nm + "_next"
            if cand in out_names:
                state_out_map[nm] = cand

        # Choose hop; dynamic streaming graph accepts any T
        hop = 1024

        def run_stream_pass(style_true_scalar, style_force_scalar, writers):
            # reset states
            for nm in state_bufs.keys():
                state_bufs[nm].fill(0.0)

            pos = 0
            prob_sum = 0.0
            prob_count = 0
            N = len(x_mod)

            while pos < N:
                core = min(hop, N - pos)

                x_block = np.zeros((1, hop, 1), dtype=np.float32)
                x_block[0, :core, 0] = x_mod[pos:pos+core]

                feed = {
                    "realtime_piezo_input": x_block,
                    "realtime_style_true_input":  style_true_scalar,
                    "realtime_style_force_input": style_force_scalar,
                    "realtime_knob_pickattack": knob_feeds["realtime_knob_pickattack"],
                    "realtime_knob_pickreso":   knob_feeds["realtime_knob_pickreso"],
                    "realtime_knob_ir":         knob_feeds["realtime_knob_ir"],
                    "realtime_knob_drywet":     knob_feeds["realtime_knob_drywet"],
                }
                # Attach states
                for nm in state_in_names:
                    feed[nm] = state_bufs[nm]

                outs = predict_to_dict(debug_model, feed)

                def get_core(name_key):
                    for kname in aliases(name_key):
                        if kname in outs:
                            arr = np.asarray(outs[kname][0, :, 0], dtype=np.float32)
                            return arr[:core]
                    return np.zeros((core,), dtype=np.float32)

                y_full     = get_core("y_pred")
                main_c     = get_core("main_core")
                pick_only  = get_core("main_pick")
                fing_only  = get_core("main_finger")
                picklearn  = get_core("picklearn")
                pickattack = get_core("pickattack")
                pickreso   = get_core("pickreso")
                ir_only    = get_core("ir_only")
                mask_core  = get_core("mask")

                sum_est = main_c + pickattack + pickreso + ir_only

                # Metrics (optional target comparison on pred pass)
                core_outs = {
                    "y_pred": y_full,
                    "main_core": main_c,
                    "pickattack": pickattack,
                    "pickreso": pickreso,
                    "main_pick": pick_only,
                    "main_finger": fing_only,
                    "mask": mask_core
                }
                y_target_core = y_mod[pos:pos+core] if writers is pass_writers["pred"] else None
                m = _metrics_for_core(core_outs, y_target_core=y_target_core)
                metrics_fn = os.path.join(out_dir, f"metrics_{'pred' if writers is pass_writers['pred'] else 'forced'}.json")
                try:
                    with open(metrics_fn, "a") as f:
                        f.write(json.dumps(m) + "\n")
                except Exception:
                    pass

                # Writes
                writers["full"].write(y_full)
                writers["main_core"].write(main_c)
                if pickattack.size:
                    if writers.get("pickattack") is None:
                        suffix = "" if writers is pass_writers["pred"] else "_forced"
                        writers["pickattack"] = sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only{suffix}.wav"),
                                                             mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                    writers["pickattack"].write(pickattack)
                if picklearn.size:
                    if writers["picklearn"] is None:
                        suffix = "" if writers is pass_writers["pred"] else "_forced"
                        writers["picklearn"] = sf.SoundFile(os.path.join(out_dir, f"stem_picklearn_only{suffix}.wav"),
                                                            mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                    writers["picklearn"].write(picklearn)
                if ir_only.size:
                    if writers["ir"] is None:
                        suffix = "" if writers is pass_writers["pred"] else "_forced"
                        writers["ir"] = sf.SoundFile(os.path.join(out_dir, f"stem_ir_lowpass{suffix}.wav"),
                                                     mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                    writers["ir"].write(ir_only)
                if pickreso.size:
                    if writers["pickreso"] is None:
                        suffix = "" if writers is pass_writers["pred"] else "_forced"
                        writers["pickreso"] = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only{suffix}.wav"),
                                                           mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                    writers["pickreso"].write(pickreso)

                writers["pick"].write(pick_only)
                writers["fing"].write(fing_only)
                writers["sumcheck"].write(sum_est)

                if writers is pass_writers["pred"]:
                    for kname in aliases("style_pred"):
                        if kname in outs:
                            sp = float(np.asarray(outs[kname])[0, 0])
                            prob_sum += sp * float(core)
                            prob_count = prob_count + core
                            break

                # Update states
                for nm, onm in state_out_map.items():
                    if onm in outs:
                        state_bufs[nm] = outs[onm]

                # Source/target once per block
                if writers is pass_writers["pred"]:
                    writers["src"].write(x_mod[pos:pos+core])
                    writers["tgt"].write(y_mod[pos:pos+core])

                pos += core

            return prob_sum, prob_count

        # Run all passes streaming
        total_prob_sum = 0.0
        total_prob_cnt = 0

        for pass_key, st_true, st_force in forced_variants:
            writers = pass_writers[pass_key]
            ps, pc = run_stream_pass(st_true, st_force, writers)
            total_prob_sum += ps
            total_prob_cnt += pc

        for wset in pass_writers.values():
            for w in list(wset.values()):
                try:
                    if w is not None:
                        w.close()
                except Exception:
                    pass

        if total_prob_cnt > 0:
            p = total_prob_sum / float(total_prob_cnt)
            label = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
            print(f"✅ Stems saved to {out_dir}. Predicted style (avg): {label} (p={p:.4f})")
        else:
            print(f"✅ Stems saved to {out_dir}.")
        return

    # Non-streaming fallback (offline or legacy)
    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=debug_model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = len(x_mod)

    def open_writers(pass_key):
        is_pred = (pass_key == "pred")
        suffix = "" if is_pred else f"_{pass_key.split('_', 1)[1]}"
        w = {}
        if is_pred:
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, f"pred_full_forced{suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["main_core"] = sf.SoundFile(os.path.join(out_dir, "stem_main_core_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                      mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pick"] = sf.SoundFile(os.path.join(out_dir, "stem_pick_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["fing"] = sf.SoundFile(os.path.join(out_dir, "stem_finger_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickattack"] = None
        w["picklearn"]  = None
        w["ir"]         = None
        w["pickreso"]   = None
        w["sumcheck"] = sf.SoundFile(os.path.join(out_dir, "pred_full_sumcheck" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                     mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        return w

    pass_writers = {k: open_writers(k) for k, _, _ in forced_variants}

    produced_model = 0
    pred_prob_sum = 0.0
    pred_prob_count = 0

    while produced_model < want_total_model:
        core_model = min(block_core_model, want_total_model - produced_model)
        left_this = left_ctx_model
        right_this = right_ctx_model if (produced_model + core_model) < want_total_model else 0

        core_start = produced_model
        read_start = max(0, core_start - left_this)
        read_end   = min(want_total_model, core_start + core_model + right_this)

        x_inp = x_mod[read_start:read_end]
        if x_inp.size <= 0:
            break

        core_start_in_inp = core_start - read_start
        core_end_in_inp   = core_start_in_inp + core_model

        x_b = x_inp[None, :, None].astype(np.float32)

        for pass_key, style_in, force_in in forced_variants:
            feeds = {
                f"{kind}_piezo_input": x_b,
                f"{kind}_style_true_input": style_in,
                f"{kind}_style_force_input": force_in,
            }
            feeds.update(knob_feeds)

            outs = predict_to_dict(debug_model, feeds)

            def core_of(key):
                for kname in aliases(key):
                    if kname in outs:
                        arr = np.asarray(outs[kname][0, :, 0], dtype=np.float32)
                        return arr[core_start_in_inp:core_end_in_inp]
                return np.zeros((core_model,), dtype=np.float32)

            wset = pass_writers[pass_key]

            y_full     = core_of("y_pred")
            main_c     = core_of("main_core")
            pick_only  = core_of("main_pick")
            fing_only  = core_of("main_finger")
            picklearn  = core_of("picklearn")
            pickattack = core_of("pickattack")
            pickreso   = core_of("pickreso")
            ir_only    = core_of("ir_only")
            mask_core  = core_of("mask")

            sum_est = main_c + pickattack + pickreso + ir_only

            core_outs = {
                "y_pred": y_full,
                "main_core": main_c,
                "pickattack": pickattack,
                "pickreso": pickreso,
                "main_pick": pick_only,
                "main_finger": fing_only,
                "mask": mask_core
            }
            y_target_core = y_mod[core_start:core_start+core_model] if pass_key == "pred" else None
            m = _metrics_for_core(core_outs, y_target_core=y_target_core)
            metrics_fn = os.path.join(out_dir, f"metrics_{pass_key}.json")
            try:
                with open(metrics_fn, "a") as f:
                    f.write(json.dumps(m) + "\n")
            except Exception:
                pass

            wset["full"].write(y_full)
            wset["main_core"].write(main_c)
            if pickattack.size:
                if wset.get("pickattack") is None:
                    fn = "stem_pickattack_only" + ("" if pass_key == "pred" else f"_{pass_key.split('_', 1)[1]}") + ".wav"
                    wset["pickattack"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["pickattack"].write(pickattack)
            if picklearn.size:
                if wset["picklearn"] is None:
                    fn = "stem_picklearn_only" + ("" if pass_key == "pred" else f"_{pass_key.split('_', 1)[1]}") + ".wav"
                    wset["picklearn"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["picklearn"].write(picklearn)
            if ir_only.size:
                if wset["ir"] is None:
                    fn = "stem_ir_lowpass" + ("" if pass_key == "pred" else f"_{pass_key.split('_', 1)[1]}") + ".wav"
                    wset["ir"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["ir"].write(ir_only)
            if pickreso.size:
                if wset["pickreso"] is None:
                    fn = "stem_pickreso_only" + ("" if pass_key == "pred" else f"_{pass_key.split('_', 1)[1]}") + ".wav"
                    wset["pickreso"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["pickreso"].write(pickreso)

            wset["pick"].write(pick_only)
            wset["fing"].write(fing_only)
            wset["sumcheck"].write(sum_est)

            if pass_key == "pred":
                for kname in aliases("style_pred"):
                    if kname in outs:
                        sp = float(np.asarray(outs[kname])[0, 0])
                        pred_prob_sum += sp * float(core_model)
                        pred_prob_count += core_model
                        break

        pass_writers["pred"]["src"].write(x_mod[core_start:core_start+core_model])
        pass_writers["pred"]["tgt"].write(y_mod[core_start:core_start+core_model])
        produced_model += core_model

    for wset in pass_writers.values():
        for w in list(wset.values()):
            try:
                if w is not None:
                    w.close()
            except Exception:
                pass

    if pred_prob_count > 0:
        p = pred_prob_sum / float(pred_prob_count)
        label = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ Stems saved to {out_dir}. Predicted style (avg): {label} (p={p:.4f})")
    else:
        print(f"✅ Stems saved to {out_dir}.")

def run_render_stems_offline(use_finetuned=False):
    sample_rate = OFFLINE_SAMPLE_RATE
    
    src = None
    tgt = None
    
    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        if not all_pairs:
            print("ERROR: No baseline training pairs found."); sys.exit(1)
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="offline", use_finetuned=bool(use_finetuned))
    if dpath is None:
        print("[stems/offline] No offline dynamic models found."); sys.exit(1)
    print("[stems/offline] Using:", dpath)
    offline_dyn = tf.keras.models.load_model(dpath, compile=False)

    dbg = build_offline_debug_model(offline_dyn)
    out_dir = STEMS_FINETUNE_OFFLINE_OUT_DIR if use_finetuned else STEMS_OFFLINE_OUT_DIR
    render_isolated_stems(debug_model=dbg, kind="offline", path_src=src, path_tgt=tgt,
                          out_dir=out_dir,
                          sample_rate=sample_rate,
                          sec=LONG_DEMO_SEC, offset_sec=LONG_DEMO_OFFSET_SEC,
                          use_finetuned=use_finetuned)


def run_render_stems_realtime(use_finetuned=False):
    print("[stems/realtime] run_render_stems_realtime (runtime streaming path)")
    sample_rate = REALTIME_SAMPLE_RATE

    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="realtime", use_finetuned=bool(use_finetuned))
    if dpath is None:
        print("[stems/realtime] No realtime dynamic models found."); sys.exit(1)
    print("[stems/realtime] Using:", dpath)
    realtime_dyn = tf.keras.models.load_model(dpath, compile=False)

    out_dir = STEMS_FINETUNE_REALTIME_OUT_DIR if use_finetuned else STEMS_REALTIME_OUT_DIR

    # Use the runtime streaming renderer that taps post‑knob heads
    render_isolated_stems_realtime_runtime(model=realtime_dyn,
                                           path_src=src,
                                           path_tgt=tgt,
                                           out_dir=out_dir,
                                           sample_rate=sample_rate,
                                           sec=LONG_DEMO_SEC,
                                           offset_sec=LONG_DEMO_OFFSET_SEC,
                                           hop=1024,
                                           use_finetuned=use_finetuned)

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
    dual_offline=False,
    trainable_model_filepath=None
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
        ckpt_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        copy_weights_by_name(static_model, ckpt_model, verbose=True)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        initial_epoch = 0
  
    new_w, _ = calibrate_loss_weights(
        model=static_model, dataset=val_ds, sample_rate=sample_rate, n_batches=5
    )
    L1_WEIGHT              = new_w['l1']
    STFT_WEIGHT            = new_w['stft']
    MEL_WEIGHT             = new_w['mel']
    RMS_ENV_WEIGHT         = new_w['env']
    HF_WEIGHT              = new_w['hf']
    RESBP_BAND_LOSS_WEIGHT = new_w['resbp_bl']  # NEW
    print(f"→ normalized ({kind}) weights:"
        f" L1={L1_WEIGHT:.6f}"
        f" STFT={STFT_WEIGHT:.6f}"
        f" MEL={MEL_WEIGHT:.6f}"
        f" ENV={RMS_ENV_WEIGHT:.6f}"
        f" HF={HF_WEIGHT:.6f}"
        f" RESBP_BL={RESBP_BAND_LOSS_WEIGHT:.6f}")

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
    tb = TensorBoard(log_dir=logs_out_dir, write_graph=False)

    # Use streaming realtime for demo (teacherless), keep offline demo builder unchanged
    if kind == 'realtime':
        demo_build_fn = build_realtime_streaming_model
        demo_build_args = {}  # teacherless streaming
        demo_safe_copy = True
    else:
        demo_build_fn = build_model_fn
        demo_build_args = build_model_fn_args_dynamic
        demo_safe_copy = False

    demo_cb = DemoCallback(sample_rate=sample_rate,
                           build_model_fn=demo_build_fn,
                           build_model_fn_args_dynamic=demo_build_args,
                           demo_checkpoints_out_dir=demo_checkpoints_out_dir,
                           use_safe_weight_copy=demo_safe_copy)

    cbs = [es, cp, reduce_lr, terminate_on_nan, tb, demo_cb]

    if kind == 'realtime':
        kd_targets = {
            "audio_distill":       (REALTIME_DISTILL_ALPHA,              2.0,  0,  80),
            "audio_stft_distill":  (REALTIME_AUDIO_STFT_DISTILL_ALPHA,   0.2,  0,  80),
            "feat_distill_0":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
            "feat_distill_1":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
            "stem_pick_distill":   (REALTIME_STEM_PICK_DISTILL_ALPHA,    0.2,  0,  80),
            "stem_finger_distill": (REALTIME_STEM_FINGER_DISTILL_ALPHA,  0.2,  0,  80),
            "realtime_style_blend":(REALTIME_STYLE_BLEND_INIT,           0.3,  0, 100)
        }
        cbs.append(KDAlphaScheduler(kd_targets))

    static_model.fit(train_ds,
                     initial_epoch=initial_epoch,
                     epochs=1 if DEBUG else EPOCHS,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=val_ds,
                     validation_steps=validation_steps,
                     callbacks=cbs)

    # Export runtime artifacts
    if kind == 'offline':
        # Keep original offline exports (unchanged)
        static_model.save(static_model_filepath)
        print("✅ Saved FP32 static model:", static_model_filepath)
        dynamic_model = build_model_fn(**build_model_fn_args_dynamic, dynamic_time=True)
        dynamic_model.set_weights(static_model.get_weights())
        dynamic_model.save(dynamic_model_filepath)
        print("✅ Saved FP32 dynamic model:", dynamic_model_filepath)
        return static_model, dynamic_model
    
    # For realtime, first save the trainable version to disk (which should have the offline teacher baked into its graph)
    if trainable_model_filepath is not None:
        static_model.save(trainable_model_filepath)
        print("✅ Saved FP32 trainable static model (with teacher):", trainable_model_filepath)

    static_model.save(static_model_filepath)

    # kind == 'realtime': always export self-contained streaming runtime models
    rt_dyn_stream = build_realtime_streaming_model(dynamic_time=True)
    copy_weights_by_name(rt_dyn_stream, static_model, verbose=True)

    rt_dyn_stream.save(dynamic_model_filepath)
    print("✅ Saved realtime streaming dynamic model:", dynamic_model_filepath)

    rt_static_stream = build_realtime_streaming_model()
    copy_weights_by_name(rt_static_stream, static_model, verbose=True)

    rt_static_stream.save(static_model_filepath)
    print("✅ Saved realtime streaming static model:", static_model_filepath)

    return static_model, rt_dyn_stream

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

    # Load offline teacher (non-causal) and build a causalized teacher clone (KD)
    offline_nc = tf.keras.models.load_model(offline_dynamic_model_filepath, compile=False)
    print("Loaded offline (non-causal) from", offline_dynamic_model_filepath)
    offline_causal = build_offline_model_causal(offline_nc, dynamic_time=True)
    print("Built causal teacher clone for KD (right-lookahead=0).")

    # Train the baseline realtime KD model; export streaming runtime artifacts
    train_shared(
        kind='realtime',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=REALTIME_SEG_LEN,
        sample_rate=REALTIME_SAMPLE_RATE,
        batch_size=REALTIME_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
        trainable_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME),
        build_model_fn=build_realtime_training_model,                   # KD graph for training
        build_model_fn_args_static={ 'offline': offline_causal },       # teacher embedded
        build_model_fn_args_dynamic={ 'offline': offline_causal },      # teacher embedded
        checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
        logs_out_dir=LOGS_REALTIME_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
        dual_offline=True
    )

def generate_fine_tune_demo(model, kind, sample_rate):
    out_dir = os.path.join(FINETUNE_DEMOS_OUT_DIR, kind)
    os.makedirs(out_dir, exist_ok=True)
    return generate_demo_from_files(
        model           = model,
        kind            = kind,
        sample_rate     = sample_rate,
        offset_sec      = FINETUNE_DEMO_OFFSET_SEC,
        sec             = FINETUNE_DEMO_SEC,
        path_src        = os.path.join(FINETUNE_SRC_DIR, "source.wav"),
        path_target     = os.path.join(FINETUNE_SRC_DIR, "target.wav"),
        out_dir         = out_dir,
        prediction_name = "demo_pred",
        use_finetuned=True
    )

def _spectral_denoise_target(y, sr,
                             hop_ms=FINETUNE_TARGET_DENOISE_HOP_MS,
                             beta=FINETUNE_TARGET_DENOISE_BETA,
                             floor_db=FINETUNE_TARGET_DENOISE_FLOOR_DB,
                             noise_q=FINETUNE_TARGET_DENOISE_NOISE_Q,
                             min_noise_frames=FINETUNE_TARGET_DENOISE_MIN_FRAMES):
    """
    Light spectral denoiser that is COLA/NOLA-safe and compatible with older SciPy
    (no 'length' kw on istft). Falls back to manual trim/pad.
    """

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size < 8:
        return y

    # Enforce Hann + 50% overlap → COLA/NOLA satisfied
    hop  = max(16, int(round((hop_ms / 1000.0) * sr)))
    nper = max(64, 2 * hop)            # 50% overlap for Hann
    if (nper % 2) == 1:
        nper += 1
    nover = nper - hop
    nfft = nper                        # no need for extra zero-pad

    # STFT with zero boundary and pad to keep edges NOLA-safe
    f, t, Z = spsig.stft(
        y, fs=sr,
        window='hann',
        nperseg=nper,
        noverlap=nover,
        nfft=nfft,
        boundary='zeros',
        padded=True
    )

    mag = np.abs(Z)
    frame_energy = np.sqrt(np.mean(mag * mag, axis=0) + 1e-12)
    thr = np.quantile(frame_energy, float(noise_q))
    idx = np.where(frame_energy <= thr)[0]
    if idx.size >= int(min_noise_frames):
        noise_profile = np.median(mag[:, idx], axis=1)
    else:
        noise_profile = np.percentile(mag, 20.0, axis=1)

    eps = 1e-8
    mag2 = mag * mag
    n2   = (noise_profile[:, None] ** 2)
    G = mag2 / (mag2 + float(beta) * n2 + eps)  # Wiener-like mask
    floor = 10.0 ** (float(floor_db) / 20.0)
    G = np.clip(G, floor, 1.0)

    Z_d = G * Z

    # Older SciPy has no 'length' kw; try it, then fallback and trim manually
    try:
        _, y_rec = spsig.istft(
            Z_d, fs=sr,
            window='hann',
            nperseg=nper,
            noverlap=nover,
            nfft=nfft,
            input_onesided=True,
            length=y.size  # SciPy >= 1.10
        )
    except TypeError:
        _, y_rec = spsig.istft(
            Z_d, fs=sr,
            window='hann',
            nperseg=nper,
            noverlap=nover,
            nfft=nfft,
            input_onesided=True
        )
        # Trim/pad to original length
        if y_rec.size > y.size:
            y_rec = y_rec[:y.size]
        elif y_rec.size < y.size:
            y_rec = np.pad(y_rec, (0, y.size - y_rec.size))

    return y_rec.astype(np.float32, copy=False)

def denoise_target_if_enabled(y_full, sr):
    """
    Apply the light spectral denoiser to the mic target channel(s) if enabled.
    y_full: (N, C) float32
    """
    if not FINETUNE_TARGET_DENOISE_ENABLE:
        return y_full
    if y_full.ndim != 2 or y_full.shape[0] < 8:
        return y_full
    Y = y_full.copy()
    for c in range(Y.shape[1]):
        Y[:, c] = _spectral_denoise_target(Y[:, c], sr)
    print("[finetune] Applied light spectral denoise to mic target.")
    return Y

def cleanup_and_prepare_train (logs_out_dir, checkpoints_out_dir, demo_checkpoints_out_dir):
    for dir_path in [logs_out_dir, demo_checkpoints_out_dir]:
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
def generate_demo(model, kind, out_dir, sample_rate, prediction_name="demo_pred", only_prediction=False):
    """
    Offline: unchanged (windowed context).
    Realtime: requires a self-contained STREAMING model with fixed T=32.
              This function never builds any graph; it throws if the model is not streaming or not T=32.
    """
    assert_has_pairs()
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]

    # Style flag for display
    if "pick" in pth_src:
        style_true_flag = 1.0
    elif "finger" in pth_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    os.makedirs(out_dir, exist_ok=True)

    if kind == "offline":
        # Original offline behavior preserved
        return generate_demo_from_files(
            model           = model,
            kind            = "offline",
            out_dir         = out_dir,
            sample_rate     = sample_rate,
            path_src        = pth_src,
            path_target     = pth_tgt,
            prediction_name = prediction_name,
            only_prediction = only_prediction
        )

    # kind == "realtime": must be streaming + T=32
    in_tensors  = list(model.inputs)
    out_tensors = list(model.outputs)

    # Robust names
    in_names  = [_tensor_basename(t) for t in in_tensors]
    out_names = _output_names_from_model(model)

    # Find state I/O
    state_in_names  = [n for n in in_names if n.startswith("realtime_state_")]
    # Prefer named state outputs; if unavailable, fall back to ordered mapping
    named_state_outs = [n for n in out_names if n.startswith("realtime_state_") and n.endswith("_next")]
    n_states = len(state_in_names)
    n_out = len(out_tensors)

    use_ordered_state_mapping = False
    if n_states == 0:
        print("[generate_demo] inputs:", in_names)
        print("[generate_demo] outputs:", out_names)
        raise RuntimeError("generate_demo(realtime): provided model is not STREAMING (no realtime_state_* inputs).")
    if not named_state_outs:
        # Try ordered mapping: expect exactly [audio, style] + state outs
        if n_out == 2 + n_states:
            use_ordered_state_mapping = True
            print("[generate_demo] Note: output tensors are generically named; using ordered state mapping.")
        else:
            print("[generate_demo] inputs:", in_names)
            print("[generate_demo] outputs:", out_names)
            raise RuntimeError("generate_demo(realtime): provided model is not STREAMING. "
                               "Expected explicit realtime_state_* inputs and *_next outputs (or matching count for ordered mapping).")

    # Find the realtime piezo input and enforce fixed T=32
    def _find_piezo_input():
        for t in in_tensors:
            nm = _tensor_basename(t)
            if nm == "realtime_piezo_input":
                return t
        for t in in_tensors:
            nm = _tensor_basename(t)
            if "piezo_input" in nm:
                return t
        return None

    piezo_in = _find_piezo_input()
    if piezo_in is None:
        raise RuntimeError("generate_demo(realtime): could not find 'realtime_piezo_input' in model inputs.")

    T_dim = piezo_in.shape[1]
    if (T_dim is None) or (int(T_dim) != int(REALTIME_TFLITE_T)):
        raise RuntimeError(f"generate_demo(realtime): expected static T={REALTIME_TFLITE_T} streaming model; "
                           f"got T={T_dim}.")

    # Load audio
    info = sf.info(pth_src)
    file_sr = info.samplerate
    START_FRAME = int(round(LONG_DEMO_OFFSET_SEC * file_sr))
    TOTAL_FRAMES = int(round(LONG_DEMO_SEC * file_sr))

    x_long, _ = sf.read(pth_src, start=START_FRAME, frames=TOTAL_FRAMES, dtype='float32', always_2d=True)
    y_long, _ = sf.read(pth_tgt, start=START_FRAME, frames=TOTAL_FRAMES, dtype='float32', always_2d=True) if not only_prediction else (None, None)
    x_src = x_long[:, 0]
    y_src = y_long[:, 0] if y_long is not None else None

    x_mod = x_src if file_sr == sample_rate else resample_to_model_sr(x_src, file_sr, sample_rate)
    y_mod = y_src if (y_src is not None and file_sr == sample_rate) else (resample_to_model_sr(y_src, file_sr, sample_rate) if y_src is not None else None)

    # Writers
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

    # Streaming loop with fixed hop=T=REALTIME_TFLITE_T (pad last block with zeros)
    hop = REALTIME_TFLITE_T
    N = len(x_mod)

    # Prepare state buffers (zero-initialized by input spec order)
    state_in_tensors = [t for t in in_tensors if _tensor_basename(t).startswith("realtime_state_")]
    state_in_names   = [_tensor_basename(t) for t in state_in_tensors]
    state_bufs = {
        nm: np.zeros((1,
                      int(t.shape[1]) if t.shape[1] is not None else 0,
                      int(t.shape[2]) if t.shape[2] is not None else 1),
                     dtype=np.float32)
        for nm, t in zip(state_in_names, state_in_tensors)
    }

    # Build a position map for ordered mapping if needed
    if use_ordered_state_mapping:
        # outputs = [y, style] + states..., same order as state_inputs
        state_out_start = 2
        state_out_indices = {nm: (state_out_start + i) for i, nm in enumerate(state_in_names)}
    else:
        # Name-based mapping: out_names are producing layer names
        state_out_indices = {}
        out_names = _output_names_from_model(model)  # recompute in case
        out_index_map = {name: idx for idx, name in enumerate(out_names)}
        for nm in state_in_names:
            nm_next = nm + "_next"
            if nm_next in out_index_map:
                state_out_indices[nm] = out_index_map[nm_next]

    knob_feeds = _prepare_knob_feeds(kind="realtime", use_finetuned=False)

    def run_one_pass(force_style=None):
        pos = 0
        # Reset states
        for k in state_bufs.keys():
            st = state_bufs[k]
            st.fill(0.0)

        if force_style is None:
            st_true  = np.zeros((1, 1), dtype=np.float32)
            st_force = np.zeros((1, 1), dtype=np.float32)
        else:
            st_true  = np.array([[float(force_style)]], dtype=np.float32)
            st_force = np.ones((1, 1), dtype=np.float32)

        prob_sum = 0.0
        prob_count = 0

        while pos < N:
            core = min(hop, N - pos)
            x_block = np.zeros((1, hop, 1), dtype=np.float32)
            x_block[0, :core, 0] = x_mod[pos:pos + core].astype(np.float32, copy=False)

            feed = {
                "realtime_piezo_input": x_block,
                "realtime_style_true_input": st_true,
                "realtime_style_force_input": st_force,
                "realtime_knob_pickattack": knob_feeds["realtime_knob_pickattack"],
                "realtime_knob_pickreso":   knob_feeds["realtime_knob_pickreso"],
                "realtime_knob_ir":         knob_feeds["realtime_knob_ir"],
                "realtime_knob_drywet":     knob_feeds["realtime_knob_drywet"],
            }
            # Attach states by input name
            for nm in state_in_names:
                feed[nm] = state_bufs[nm]

            outs = model.predict(feed, batch_size=1, verbose=0)
            # Ordered: y=outs[0], style=outs[1]
            y_arr = outs[0]
            s_arr = outs[1]
            y_block = y_arr[0, :hop, 0]
            y_core  = y_block[:core]

            if force_style is None:
                w_pred.write(y_core)
            else:
                w_true.write(y_core)

            if not only_prediction:
                w_src.write(x_mod[pos:pos + core])
                if y_mod is not None:
                    w_tgt.write(y_mod[pos:pos + core])

            if force_style is None:
                s_prob = float(np.asarray(s_arr, dtype=np.float32)[0, 0])
                prob_sum   += s_prob * float(core)
                prob_count += core

            # Update states (ordered or name-based)
            for nm in state_in_names:
                idx = state_out_indices.get(nm, None)
                if idx is None:
                    continue
                state_bufs[nm] = outs[idx]

            pos += core

        return (prob_sum / float(prob_count)) if prob_count > 0 else None

    # Predicted-style pass and optional true-style pass
    p_avg = run_one_pass(force_style=None)
    if style_true_flag is not None:
        run_one_pass(force_style=style_true_flag)

    w_pred.close()
    if w_true is not None: w_true.close()
    if w_src is not None:  w_src.close()
    if w_tgt is not None:  w_tgt.close()

    if p_avg is not None:
        label = 'pick' if p_avg > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ Demo saved to {style_pred_path}. Predicted style (avg): {label} ({p_avg:.4f})")
        if style_true_flag is not None:
            tlabel = 'pick' if style_true_flag >= STYLE_FLAG_PROB_THRESHOLD else 'finger'
            print(f"✅ Demo saved to {style_true_path}. True style: {tlabel}")
    else:
        print(f"✅ Demo saved to {style_pred_path}.")
        
def prepare_and_generate_demo(kind, sample_rate, dynamic_model_filepath, demo_out_dir):
    """
    For demos, only load saved artifacts:
      - offline: dynamic offline model (non-streaming)
      - realtime: static T=32 streaming model (self-contained)
    This function never builds graphs.
    """
    # Prefer saved artifacts by kind
    prefer = None
    if kind == "offline":
        cand = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        if os.path.exists(cand):
            prefer = cand
    elif kind == "realtime":
        cand = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
        if os.path.exists(cand):
            prefer = cand
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

    dpath = prefer if prefer is not None else dynamic_model_filepath
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"[demo] Missing model for kind='{kind}'. Expected at: {dpath}. "
                                f"For realtime, train/export the static T={REALTIME_TFLITE_T} streaming model first.")
    model = tf.keras.models.load_model(dpath, compile=False)
    model.summary()

    generate_demo(model=model, kind=kind, out_dir=demo_out_dir, sample_rate=sample_rate)

def _crest(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0: return 0.0
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    peak = np.max(np.abs(x))
    return float(peak / (rms + 1e-12))

def _quantiles(x, qs=(0.5, 0.9, 0.99, 0.999)):
    x = np.abs(np.asarray(x, dtype=np.float32))
    return {f"q{int(q*1000):03d}": float(np.quantile(x, q)) for q in qs}

def _duty_cycle(m, thr=0.2):
    m = np.asarray(m, dtype=np.float32)
    if m.size == 0: return 0.0
    return float(np.mean((m > float(thr)).astype(np.float32)))

def _sum_rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32)**2) + 1e-12))

def _metrics_for_core(outs_dict, y_target_core=None):
    # outs must contain keys: y_pred, main_core, pickattack, pickreso, main_pick, main_finger
    y   = outs_dict.get("y_pred",    None)
    main= outs_dict.get("main_core", None)
    patt= outs_dict.get("pickattack",None)
    pres= outs_dict.get("pickreso",  None)
    mp  = outs_dict.get("main_pick", None)
    mf  = outs_dict.get("main_finger", None)
    msk = outs_dict.get("mask",      None)

    m = {}
    # RMS and crest
    for k, sig in [("full", y), ("main", main), ("pickattack", patt), ("pick_reso", pres),
                   ("pick", mp), ("fing", mf)]:
        if sig is not None:
            m[f"rms_{k}"]   = _sum_rms(sig)
            m[f"crest_{k}"] = _crest(sig)
            m.update({f"{k}_{q}":v for q,v in _quantiles(sig).items()})

    # Sum-check error
    if y is not None and main is not None:
        est = np.zeros_like(y)
        if patt is not None: est = est + patt
        if pres is not None: est = est + pres
        est = est + main
        m["rms_sum_err"] = _sum_rms(y - est)

    # Mask stats
    if msk is not None:
        m["mask_mean"]  = float(np.mean(msk))
        m["mask_duty20"] = _duty_cycle(msk, thr=0.2)
        m["mask_duty50"] = _duty_cycle(msk, thr=0.5)
        m.update({f"mask_{q}": v for q, v in _quantiles(msk).items()})

    # Pick/finger correlation
    if mp is not None and mf is not None:
        m["corr_pick_fing"] = _corr(mp, mf)

    # Optional: overshoot vs target (HF-ish proxy)
    if (y_target_core is not None) and (patt is not None):
        yt = np.asarray(y_target_core, dtype=np.float32)
        pa = np.asarray(patt,         dtype=np.float32)
        eps = 1e-6
        # Compare 99th percentile magnitudes normalized by RMS
        q_patt = np.quantile(np.abs(pa), 0.99)
        q_tgt  = np.quantile(np.abs(yt), 0.99)
        m["overshoot_q99_ratio"] = float(q_patt / (q_tgt + eps))

    return m

def _wrap_fixed_T_for_tflite(dyn_model, kind, T_fixed):
    if kind == "offline":
        inp  = Input(shape=(T_fixed,1), name="offline_piezo_input")
        st_t = Input(shape=(1,),       name="offline_style_true_input")
        st_f = Input(shape=(1,),       name="offline_style_force_input")
        k_pa = Input(shape=(1,),       name="offline_knob_pickattack")
        k_pr = Input(shape=(1,),       name="offline_knob_pickreso")
        k_dw = Input(shape=(1,),       name="offline_knob_drywet")
        y, s = dyn_model([inp, st_t, st_f, k_pa, k_pr, k_dw])

        # Give outputs stable, exported names without raw TF ops
        y = Activation('linear', name="offline_audio_out")(y)
        s = Activation('linear', name="offline_style_prob_out")(s)

        export = tf.keras.Model(
            inputs=[inp, st_t, st_f, k_pa, k_pr, k_dw],
            outputs=[y, s],
            name=dyn_model.name + "_tflite_export"
        )

    elif kind == "realtime":
        inp  = Input(shape=(T_fixed,1), name="realtime_piezo_input")
        st_t = Input(shape=(1,),       name="realtime_style_true_input")
        st_f = Input(shape=(1,),       name="realtime_style_force_input")
        k_pa = Input(shape=(1,),       name="realtime_knob_pickattack")
        k_pr = Input(shape=(1,),       name="realtime_knob_pickreso")
        k_ir = Input(shape=(1,),       name="realtime_knob_ir")
        k_dw = Input(shape=(1,),       name="realtime_knob_drywet")
        y, s = dyn_model([inp, st_t, st_f, k_pa, k_pr, k_ir, k_dw])

        # Give outputs stable, exported names without raw TF ops
        y = Activation('linear', name="realtime_audio_out")(y)
        s = Activation('linear', name="realtime_style_prob_out")(s)

        export = tf.keras.Model(
            inputs=[inp, st_t, st_f, k_pa, k_pr, k_ir, k_dw],
            outputs=[y, s],
            name=dyn_model.name + "_tflite_export"
        )
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")
    return export

def _convert_keras_to_tflite_fp32(model, out_path):
    """
    Convert the provided Keras model to FP32 TFLite. Tries pure builtin ops first,
    falls back to SELECT_TF_OPS if needed.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflm = converter.convert()
    except Exception as e:
        print(f"[TFLite] Builtin conversion failed ({e}); retrying with SELECT_TF_OPS…")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflm = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflm)
    print(f"✅ Saved FP32 TFLite: {out_path}")

def _build_dyn_model_for_export(kind):
    """
    Build the dynamic-time runtime graph (teacherless) for export.
    Realtime no longer uses this (we export the streaming/stateful graph directly).
    """
    if kind == "offline":
        dyn = build_offline_model(dynamic_time=True)
        for l in dyn.layers:
            l.trainable = False
        return dyn
    elif kind == "realtime":
        raise RuntimeError("_build_dyn_model_for_export is not used for realtime. "
                           "Use build_realtime_streaming_model with a fixed T instead.")
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

def _export_one_tflite(kind, static_src_path, out_tflite_path, T_fixed):
    """
    Export a single FP32 TFLite for a given 'kind' from a static Keras model.
    - offline: wrap teacherless runtime graph to fixed T
    - realtime: convert the self-contained STREAMING graph at fixed T (state I/O preserved)
    """
    if not os.path.exists(static_src_path):
        print(f"[TFLite/export] Missing source model: {static_src_path}")
        return False

    if kind == "offline":
        print(f"[TFLite/export] Loading offline static: {static_src_path}")
        static_m = tf.keras.models.load_model(static_src_path, compile=False)

        dyn = _build_dyn_model_for_export(kind="offline")
        copy_weights_by_name(dyn, static_m, verbose=True)

        export = _wrap_fixed_T_for_tflite(dyn, kind="offline", T_fixed=T_fixed)
        _convert_keras_to_tflite_fp32(export, out_tflite_path)
        return True

    # kind == "realtime": export streaming/stateful graph with fixed T
    print(f"[TFLite/export] Loading realtime streaming static: {static_src_path}")
    rt_static = tf.keras.models.load_model(static_src_path, compile=False)

    T_fixed = int(T_fixed)
    if T_fixed == int(REALTIME_TFLITE_T):
        # Already T=32 streaming: convert as-is
        _convert_keras_to_tflite_fp32(rt_static, out_tflite_path)
        return True

    # Rebuild streaming with desired fixed T and copy weights by name
    rt_fixedT = build_realtime_streaming_model(static_T=T_fixed)
    copy_weights_by_name(rt_fixedT, rt_static, verbose=True)
    _convert_keras_to_tflite_fp32(rt_fixedT, out_tflite_path)
    return True

def export_tflite_all():
    # OFFLINE (unchanged)
    offline_static_src = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
    offline_tflite_out = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_TFLITE_FP32_FILENAME)
    _export_one_tflite(kind="offline",
                       static_src_path=offline_static_src,
                       out_tflite_path=offline_tflite_out,
                       T_fixed=OFFLINE_TFLITE_T)

    offline_ft_src = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
    offline_ft_out = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_TFLITE_FP32_FILENAME)
    if os.path.exists(offline_ft_src):
        _export_one_tflite(kind="offline",
                           static_src_path=offline_ft_src,
                           out_tflite_path=offline_ft_out,
                           T_fixed=OFFLINE_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned OFFLINE (not found): {offline_ft_src}")

    # REALTIME (streaming/stateful)
    rt_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    rt_tflite_out      = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TFLITE_FP32_FILENAME)

    _export_one_tflite(kind="realtime",
                       static_src_path=rt_static_src,
                       out_tflite_path=rt_tflite_out,
                       T_fixed=REALTIME_TFLITE_T)

    rt_ft_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    rt_ft_out      = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TFLITE_FP32_FILENAME)
    if os.path.exists(rt_ft_static_src):
        _export_one_tflite(kind="realtime",
                           static_src_path=rt_ft_static_src,
                           out_tflite_path=rt_ft_out,
                           T_fixed=REALTIME_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned REALTIME (not found): {rt_ft_static_src}")

def _sum_rf(kernel, dilations):
    return (int(kernel) - 1) * sum(int(d) for d in dilations)

def _align_up(x, m):
    return int(((int(x) + int(m) - 1) // int(m)) * int(m))

def compute_streaming_ctx_constants(kind):
    """
    Streaming context purely from compile-time constants (no TF).
    Returns (left_ctx, right_ctx) in samples at the model's rate.
    """
    if kind == "realtime":
        # Main stack
        lc_pre  = REALTIME_PRE_BLOCKS * (REALTIME_PRE_KERNEL - 1)
        lc_main = lc_pre + _sum_rf(REALTIME_KERNEL_SIZE, REALTIME_DILATIONS)

        # IR branch (dominant)
        lc_ir = (REALTIME_IR_FIR_SIZE - 1) \
              + REALTIME_IR_DOWNSAMPLE_FACTOR * _sum_rf(REALTIME_IR_KERNEL, REALTIME_IR_DILATIONS) \
              + (max(193, REALTIME_LP_ORDER) - 1)

        # Pick-attack head: pre-emph(1) + tiny CNN + post-HPF
        lc_patt = 1 + (7 - 1) * 1 + (7 - 1) * 2 + (PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1)

        # Pick-reso head: onset smoother + DS + modal_lr + interp + post-HPF
        s = int(round((RESBP_STRIKE_ENV_MS_REALTIME / 1000.0) * REALTIME_SAMPLE_RATE))
        L_env = int(np.clip(np.ceil(4.0 * max(1, s)), 8, 512))
        L_lr  = _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
        lc_reso = (L_env - 1) \
                + (REALTIME_PICK_RESO_DS_TAPS - 1) \
                + REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR * (L_lr - 1) \
                + (REALTIME_PICK_RESO_INTERP_TAPS - 1) \
                + (PICK_RESO_LOWBAND_PENALTY_ORDER - 1)

        # Color EQ
        lc_col = 65 - 1

        left  = int(max(lc_main, lc_ir, lc_patt, lc_reso, lc_col))
        right = 0
        return left, right

    elif kind == "offline":
        # Main stack (dominant)
        lc_pre  = OFFLINE_PRE_BLOCKS * (OFFLINE_PRE_KERNEL - 1)
        lc_main = lc_pre + _sum_rf(OFFLINE_KERNEL_SIZE, OFFLINE_DILATIONS)

        # Pick-reso offline: onset smoother + BP + modal + post-HPF
        s = int(round((RESBP_STRIKE_ENV_MS_OFFLINE / 1000.0) * OFFLINE_SAMPLE_RATE))
        L_env = int(np.clip(np.ceil(4.0 * max(1, s)), 8, 4096))
        lc_reso = (L_env - 1) \
                + (OFFLINE_PICK_RESO_BP_ORDER - 1) \
                + (PICK_RESO_MODAL_LEN_OFFLINE - 1) \
                + (PICK_RESO_LOWBAND_PENALTY_ORDER - 1)

        lc_patt = 1 + (7 - 1) * 1 + (7 - 1) * 2 + (PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1)
        lc_col  = 97 - 1

        left  = int(max(lc_main, lc_reso, lc_patt, lc_col))
        right = int(OFFLINE_LOOKAHEAD_SAMPLES)
        return left, right

    else:
        raise ValueError("kind must be 'offline' or 'realtime'")


def _select_tflite_demo_paths(kind, use_finetuned=False):
    if kind == "offline":
        if use_finetuned:
            return os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_TFLITE_FP32_FILENAME)
        return os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_TFLITE_FP32_FILENAME)
    elif kind == "realtime":
        if use_finetuned:
            return os.path.join(
                MODELS_REALTIME_OUT_DIR,
                FINETUNE_REALTIME_TFLITE_FP32_FILENAME
            )
        return os.path.join(
            MODELS_REALTIME_OUT_DIR,
            REALTIME_TFLITE_FP32_FILENAME
        )
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

def _tflite_demo_blockwise(kind, tflite_runner, sample_rate, x_mod, y_mod=None,
                           style_true_flag=None, use_finetuned=False, out_dir=".",
                           fixed_hop=None):
    os.makedirs(out_dir, exist_ok=True)
    piezo_key = f"{kind}_piezo_input"
    T_fixed = tflite_runner.input_length_T(piezo_key)

    lc_m, rc_m = compute_streaming_ctx_constants(kind)

    # Choose left/core/right for each call
    if fixed_hop is not None:
        core_len = int(fixed_hop)
        if core_len < 1 or core_len > T_fixed:
            raise ValueError(f"fixed_hop must be in [1, {T_fixed}], got {fixed_hop}")
        # Exact pedal-style schedule: use all remaining as left context, no right look-ahead for realtime
        left_this  = T_fixed - core_len
        right_this = 0
        print(f"[TFLite/demo] fixed_hop={core_len} → left={left_this}, core={core_len}, right={right_this}")
    else:
        # Default policy (existing behavior): use model ctx but keep a reasonable core
        desired_core = max(1, min(T_fixed // 2, 8192 if kind == "offline" else 512))
        left_this  = min(int(lc_m), max(0, T_fixed - 1))
        right_this = min(int(rc_m), max(0, T_fixed - 1 - left_this))
        core_len   = T_fixed - left_this - right_this
        if core_len < desired_core:
            need = desired_core - core_len
            r_dec = min(right_this, need); right_this -= r_dec; need -= r_dec
            if need > 0:
                l_dec = min(left_this, need); left_this -= l_dec; need -= l_dec
            core_len = T_fixed - left_this - right_this
        print(f"[TFLite/demo] auto schedule: T={T_fixed} lc={lc_m} rc={rc_m} → left={left_this}, core={core_len}, right={right_this}")

    knob_feeds = _prepare_knob_feeds(kind=kind, use_finetuned=use_finetuned)
    style_pred_path = os.path.join(out_dir, "demo_pred_style_pred_tflite.wav")
    style_true_path = os.path.join(out_dir, "demo_pred_style_true_tflite.wav")

    w_src = sf.SoundFile(os.path.join(out_dir, "demo_source.wav"), mode='w',
                         samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_tgt = None
    if y_mod is not None:
        w_tgt = sf.SoundFile(os.path.join(out_dir, "demo_target.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_pred = sf.SoundFile(style_pred_path, mode='w',
                          samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_true = None
    if style_true_flag is not None:
        w_true = sf.SoundFile(style_true_path, mode='w',
                              samplerate=sample_rate, channels=1, subtype='FLOAT')

    pos = 0
    prob_sum = 0.0
    prob_count = 0
    N = len(x_mod)

    while pos < N:
        core = min(core_len, N - pos)
        ltake = min(left_this, pos)
        rtake = min(right_this, N - (pos + core))

        x_in = np.zeros((1, T_fixed, 1), dtype=np.float32)
        if ltake > 0:
            x_in[0, left_this - ltake:left_this, 0] = x_mod[pos - ltake:pos]
        x_in[0, left_this:left_this + core, 0] = x_mod[pos:pos + core]
        if rtake > 0:
            xs = pos + core; xe = xs + rtake
            x_in[0, left_this + core:left_this + core + rtake, 0] = x_mod[xs:xe]

        base_feed = {k: v.astype(np.float32) for k, v in knob_feeds.items()}
        base_feed[f"{kind}_piezo_input"]       = x_in
        base_feed[f"{kind}_style_true_input"]  = np.zeros((1,1), np.float32)
        base_feed[f"{kind}_style_force_input"] = np.zeros((1,1), np.float32)

        y_pred, s_pred = tflite_runner.invoke(base_feed)
        y_core = y_pred[0, left_this:left_this + core, 0]
        w_pred.write(y_core)

        prob_sum   += float(np.asarray(s_pred, dtype=np.float32)[0, 0]) * float(core)
        prob_count += core

        if w_true is not None:
            tf_feed = dict(base_feed)
            tf_feed[f"{kind}_style_true_input"]  = np.array([[float(style_true_flag)]], dtype=np.float32)
            tf_feed[f"{kind}_style_force_input"] = np.ones((1,1), dtype=np.float32)
            y_true_pred, _ = tflite_runner.invoke(tf_feed)
            y_true_core = y_true_pred[0, left_this:left_this + core, 0]
            w_true.write(y_true_core)

        w_src.write(x_mod[pos:pos + core])
        if w_tgt is not None:
            w_tgt.write(y_mod[pos:pos + core])

        pos += core

    w_src.close()
    if w_tgt is not None: w_tgt.close()
    w_pred.close()
    if w_true is not None: w_true.close()

    if prob_count > 0:
        p = prob_sum / float(prob_count)
        label = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ TFLite demo written to {style_pred_path}. Predicted style (avg): {label} ({p:.4f})")

class TFLiteRunner:
    """
    Robust TFLite runner for ai_edge_litert:
      - Prefer SignatureRunner (serving_default). Handles generic output names by shape.
      - Fall back to raw tensor API if signature is not available.
      - Always returns copies of outputs to avoid interpreter buffer aliasing.
      - Normalizes audio to [B,T,1] and style to [B,1].
    """
    def __init__(self, model_path):
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        # Try signature runner first
        self.use_sig = False
        self.sig_runner = None
        self.sig_inputs = set()
        try:
            sigs = self.interp.get_signature_list()
            if isinstance(sigs, dict) and sigs:
                sig_key = "serving_default" if "serving_default" in sigs else next(iter(sigs.keys()))
                self.sig_runner = self.interp.get_signature_runner(sig_key)
                # Note: keys here are just names; shapes are not provided in this dict
                self.sig_inputs = set(sigs[sig_key].get("inputs", []))
                self.use_sig = True
        except Exception:
            self.use_sig = False

        # Keep raw details for T and legacy fallback
        self.inp_details = self.interp.get_input_details()
        self.out_details = self.interp.get_output_details()
        self.in_map = { self._norm_name(d["name"]): d["index"] for d in self.inp_details }
        self.out_map = { self._norm_name(d["name"]): d["index"] for d in self.out_details }

    def _norm_name(self, n):
        n = n.split(":")[0]
        if n.startswith("serving_default_"):
            n = n[len("serving_default_"):]
        return n

    def input_length_T(self, piezo_key):
        key = self._norm_name(piezo_key)
        if key not in self.in_map and piezo_key in self.in_map:
            key = piezo_key
        if key not in self.in_map:
            # fallback to first input
            return int(self.inp_details[0]["shape"][1])
        idx = self.in_map[key]
        details_by_idx = {d["index"]: d for d in self.inp_details}
        return int(details_by_idx[idx]["shape"][1])

    def _normalize_audio_style(self, audio, style):
        # Ensure we detach from interpreter buffers (copy=True)
        a = np.array(audio, dtype=np.float32, copy=True)
        s = np.array(style, dtype=np.float32, copy=True)

        # Audio -> [B,T,1]
        if a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        elif a.ndim >= 4:
            a = a.reshape((a.shape[0], a.shape[1], -1))[:, :, :1]

        # Style -> [B,1]
        if s.ndim == 0:
            s = s.reshape((1,1))
        elif s.ndim == 1:
            s = s.reshape((1, -1))[:, :1]
        elif s.ndim >= 3:
            s = s.reshape((s.shape[0], -1))[:, :1]
        return a, s

    def invoke(self, feed_dict):
        # 1) Signature path (preferred)
        if self.use_sig and (self.sig_runner is not None):
            # Build feed by signature input names (accept exact or normalized names)
            sig_feed = {}
            for k, v in feed_dict.items():
                if k in self.sig_inputs:
                    sig_feed[k] = v.astype(np.float32, copy=False)
                else:
                    nk = self._norm_name(k)
                    if nk in self.sig_inputs:
                        sig_feed[nk] = v.astype(np.float32, copy=False)

            outs = self.sig_runner(**sig_feed)  # dict with keys like 'output_0','output_1'
            if not isinstance(outs, dict) or not outs:
                raise RuntimeError("Signature runner returned no outputs.")

            # Pick audio/style by shape/size (names may be generic)
            items = list(outs.items())
            audio_name, audio_arr = max(items, key=lambda t: t[1].size)
            rest = [(n, a) for (n, a) in items if n != audio_name]
            if not rest:
                raise RuntimeError("No secondary output found for style prob in signature outputs.")
            # Prefer a (1,1) tensor for style if present, else smallest
            style_arr = None
            for n, a in rest:
                aa = np.asarray(a)
                if aa.ndim == 2 and aa.shape == (1,1):
                    style_arr = a
                    break
            if style_arr is None:
                style_arr = min(rest, key=lambda t: t[1].size)[1]

            return self._normalize_audio_style(audio_arr, style_arr)

        # 2) Legacy tensor path
        return self._invoke_by_tensors(feed_dict)

    def _invoke_by_tensors(self, feed_dict):
        # Set tensors
        for k, v in feed_dict.items():
            nm = self._norm_name(k)
            if nm in self.in_map:
                self.interp.set_tensor(self.in_map[nm], v.astype(np.float32, copy=False))

        # Invoke
        self.interp.invoke()

        # Gather outputs (copy immediately to break buffer aliasing)
        outs = []
        for d in self.out_details:
            idx = d["index"]
            name = self._norm_name(d["name"])
            val = np.array(self.interp.get_tensor(idx), dtype=np.float32, copy=True)
            outs.append((name, val))

        if not outs:
            raise RuntimeError("TFLite model has no outputs.")

        audio_name, audio_arr = max(outs, key=lambda t: t[1].size)
        rest = [(n, a) for (n, a) in outs if n != audio_name]
        if not rest:
            raise RuntimeError("No secondary output found for style prob (model exposes only one graph output).")

        style_arr = None
        for n, a in rest:
            if a.ndim == 2 and a.shape == (1,1):
                style_arr = a
                break
        if style_arr is None:
            style_arr = min(rest, key=lambda t: t[1].size)[1]

        return self._normalize_audio_style(audio_arr, style_arr)

def generate_demo_tflite(kind, use_finetuned=False, out_dir=None, hop=None):
    """
    Run the TFLite model in a streaming fashion and write WAVs.
    kind: 'offline' or 'realtime'
    uses T=32.
    hop: if provided (int), use exact pedal-style schedule:
         left = T - hop, right = 0, core = hop.
    """
    assert_has_pairs()
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]

    if "pick" in pth_src:
        style_true_flag = 1.0
    elif "finger" in pth_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    if kind == "offline":
        sr = OFFLINE_SAMPLE_RATE
        base_out_dir = LONG_DEMOS_OFFLINE_OUT_DIR
        subdir = f"t{OFFLINE_TFLITE_T}"
    else:
        sr = REALTIME_SAMPLE_RATE
        base_out_dir = LONG_DEMOS_REALTIME_OUT_DIR
        subdir = f"t{REALTIME_TFLITE_T}"

    if out_dir is None:
        out_dir = os.path.join(base_out_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    info = sf.info(pth_src)
    file_sr = info.samplerate
    start = int(round(LONG_DEMO_OFFSET_SEC * file_sr))
    frames = int(round(LONG_DEMO_SEC * file_sr))

    x_long, _ = sf.read(pth_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_long, _ = sf.read(pth_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_src = x_long[:,0]
    y_src = y_long[:,0]
    x_mod = x_src if file_sr == sr else resample_to_model_sr(x_src, file_sr, sr)
    y_mod = y_src if file_sr == sr else resample_to_model_sr(y_src, file_sr, sr)

    tflite_path = _select_tflite_demo_paths(kind=kind, use_finetuned=use_finetuned)
    if not os.path.exists(tflite_path):
        print(f"[TFLite/demo] TFLite model not found at {tflite_path}. Run --export-tflite first.")
        return

    print(f"[TFLite/demo] Using: {tflite_path}")
    runner = TFLiteRunner(tflite_path)

    _tflite_demo_blockwise(kind, runner, sample_rate=sr,
                           x_mod=x_mod, y_mod=y_mod, style_true_flag=style_true_flag,
                           use_finetuned=use_finetuned, out_dir=out_dir,
                           fixed_hop=hop)

def _seed_pick_heads_from_legacy(rt_stream_model, legacy_rt_model, verbose=True):
    """
    Seed streaming pick heads from a legacy (non-streaming) realtime model:
      - realtime_pickreso_modal_lr: from legacy realtime_pickreso_modal (full-rate) by decimation,
        or from legacy realtime_pickreso_modal_lr if already present there.
      - realtime_picklearn_* layers explicitly copied if copy-by-name ever missed them.
    Leaves layers untouched if sources are not found.
    """
    # 1) Seed pick-reso modal bank
    k_lr = None

    # If legacy had the legacy full-rate modal bank, decimate it
    k_full = _extract_trained_modal_kernel_rt(legacy_rt_model, legacy_name="realtime_pickreso_modal")
    if k_full is not None:
        try:
            k_lr = _decimate_modal_kernels(k_full, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
            if verbose:
                print(f"[seed] Decimated legacy 'realtime_pickreso_modal' -> low-rate kernel with shape {k_lr.shape}")
        except Exception as e:
            if verbose:
                print(f"[seed] Could not decimate legacy full-rate modal kernels: {e}")

    # Otherwise try to get already-lowrate kernels from legacy
    if k_lr is None:
        try:
            k_lr = legacy_rt_model.get_layer("realtime_pickreso_modal_lr").get_weights()[0]
            if verbose:
                print(f"[seed] Reused legacy 'realtime_pickreso_modal_lr' weights with shape {k_lr.shape}")
        except Exception:
            k_lr = None

    # Set into streaming model if both present
    if k_lr is not None:
        try:
            rt_stream_model.get_layer("realtime_pickreso_modal_lr").set_weights([k_lr.astype(np.float32)])
            if verbose:
                w0 = rt_stream_model.get_layer("realtime_pickreso_modal_lr").get_weights()[0]
                print(f"[seed] Set streaming 'realtime_pickreso_modal_lr' weights; ||W||={float(np.linalg.norm(w0)):.6f}")
        except Exception as e:
            if verbose:
                print(f"[seed] Could not set 'realtime_pickreso_modal_lr' in streaming model: {e}")

    # 2) Ensure the tiny pick-attack CNN is copied (paranoia)
    for lname in ("realtime_picklearn_c1", "realtime_picklearn_c2", "realtime_picklearn_out"):
        try:
            w = legacy_rt_model.get_layer(lname).get_weights()
            rt_stream_model.get_layer(lname).set_weights(w)
            if verbose:
                s = sum(np.sum(np.abs(a)) for a in w)
                print(f"[seed] Copied weights for {lname}; L1 sum={float(s):.6f}")
        except Exception:
            pass

def migrate_realtime_models_inplace():
    # Destination paths
    destination_realtime_baseline_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_finetuned_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_static_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    destination_realtime_static_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
    
    # Load the two old models to migrate from. We can use those to generate the dynamic models and eventually flatten all weights
    realtime_static_baseline_training_model_old = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)
    realtime_static_finetuned_training_model_old = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)
    
    # Load offline teachers for KD distill
    offline_static_teacher = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME), compile=False)
    offline_dynamic_teacher = tf.keras.models.load_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)
    
    # Generate causal teacher clones
    offline_static_teacher_causal = build_offline_model_causal(offline_static_teacher)
    offline_dynamic_teacher_causal = build_offline_model_causal(offline_dynamic_teacher, dynamic_time=True)
    
    # Now, moving into the actual migration, first rebuild training graphs
    realtime_static_baseline_training_model = build_realtime_training_model(offline=offline_static_teacher_causal)
    realtime_dynamic_baseline_training_model = build_realtime_training_model(offline=offline_dynamic_teacher_causal, dynamic_time=True)
    
    realtime_static_finetuned_training_model = build_realtime_training_model(offline=offline_static_teacher_causal)
    realtime_dynamic_finetuned_training_model = build_realtime_training_model(offline=offline_dynamic_teacher_causal, dynamic_time=True)
    
    # Now copy weights by name from the training models to the new training graphs
    copy_weights_by_name(realtime_static_baseline_training_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_baseline_training_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_static_finetuned_training_model, realtime_static_finetuned_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_finetuned_training_model, realtime_static_finetuned_training_model_old, verbose=True)    
    
    # Now write the streaming-friendly, stateful flat models
    realtime_static_baseline_streaming_model = build_realtime_streaming_model()
    realtime_dynamic_baseline_streaming_model = build_realtime_streaming_model(dynamic_time=True)
    realtime_static_finetuned_streaming_model = build_realtime_streaming_model()
    realtime_dynamic_finetuned_streaming_model = build_realtime_streaming_model(dynamic_time=True)
    
    copy_weights_by_name(realtime_static_baseline_streaming_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_baseline_streaming_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_static_finetuned_streaming_model, realtime_static_finetuned_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_finetuned_streaming_model, realtime_static_finetuned_training_model_old, verbose=True)
    
    # Save the new training model to disk
    realtime_static_baseline_training_model.save(destination_realtime_baseline_training_model)
    realtime_static_finetuned_training_model.save(destination_realtime_finetuned_training_model)
    
    print(f"[migrate] Successfully wrote updated training models")
    
    realtime_static_baseline_streaming_model.save(destination_realtime_static_baseline_streaming_model)
    realtime_dynamic_baseline_streaming_model.save(destination_realtime_dynamic_baseline_streaming_model)
    realtime_static_finetuned_streaming_model.save(destination_realtime_static_finetuned_streaming_model)
    realtime_dynamic_finetuned_streaming_model.save(destination_realtime_dynamic_finetuned_streaming_model)

    print(f"[migrate] Successfully wrote updated streaming models (T={REALTIME_TFLITE_T})")
    print("✅ Migration complete.")

def _tensor_basename(x):
    nm = getattr(x, "name", str(x))
    nm = nm.split(":")[0]
    if "/" in nm:
        nm = nm.split("/", 1)[0]
    if nm.startswith("serving_default_"):
        nm = nm[len("serving_default_"):]
    return nm

def _producer_layer_name(t):
    # Keras 2.x: t._keras_history.layer; Keras 3: tuple(layer, node_index, tensor_index)
    kh = getattr(t, "_keras_history", None)
    try:
        if isinstance(kh, (tuple, list)) and len(kh) >= 1:
            layer = kh[0]
        else:
            layer = getattr(kh, "layer", None)
        return getattr(layer, "name", None)
    except Exception:
        return None

def _output_names_from_model(model):
    names = []
    for t in model.outputs:
        nm = _producer_layer_name(t)
        if not nm:
            nm = _tensor_basename(t)
        names.append(nm)
    return names

def _shape_tuple(x):
    s = getattr(x, "shape", None)
    if s is None and hasattr(x, "get_shape"):
        s = x.get_shape()
    try:
        return tuple(s.as_list())
    except Exception:
        try:
            return tuple(s)
        except Exception:
            return ()

def _print_model_io_summary(model):
    print("== Inputs ==")
    for t in model.inputs:
        print(f"  {_tensor_basename(t):<56} shape={_shape_tuple(t)}")
    print("== Outputs ==")
    for t in model.outputs:
        base = _tensor_basename(t)
        prod = _producer_layer_name(t)
        if prod:
            print(f"  {base:<32} <- {prod:<48} shape={_shape_tuple(t)}")
        else:
            print(f"  {base:<32} shape={_shape_tuple(t)}")

    in_names  = [_tensor_basename(t) for t in model.inputs]
    out_names = _output_names_from_model(model)
    state_in_names  = [n for n in in_names  if n.startswith("realtime_state_")]
    state_out_names = [n for n in out_names if n.startswith("realtime_state_") and n.endswith("_next")]
    print(f"Streaming states: inputs={len(state_in_names)} outputs={len(state_out_names)}")
    if state_in_names:
        print("  first inputs:", state_in_names[:5])
    if state_out_names:
        print("  first outputs:", state_out_names[:5])

def check_streaming_equivalence_realtime(use_finetuned=False,
                                         hop=None,
                                         sec=None,
                                         offset_sec=None,
                                         pair_index=None,
                                         style_mode="pred",
                                         only="full"):
    """
    Compare realtime DYNAMIC streaming model monolithic vs streaming.

    We FORCE THE SAME STYLE in both passes to remove EMA/hop effects:
      - style_mode='pick'   → style_true=1.0, style_force=1
      - style_mode='finger' → style_true=0.0, style_force=1
      - style_mode='pred'   → compute one global probability once on full clip
      - style_mode='prob=<float>' → force exact value

    Branch isolation:
      --only=full  → compare full mix (main + IR + PA + PR)
      --only=main  → compare main path only (PA=PR=IR=0)
      --only=ir    → compare [main+IR] - [main]
      --only=pa    → compare [main+PA] - [main]
      --only=pr    → compare [main+PR] - [main]

    Writes mono.wav, stream.wav, diff.wav under:
      out/demos/realtime_streaming_equivalence/only_<mode>
    """
    assert_has_pairs()
    if sec is None:
        sec = float(LONG_DEMO_SEC)
    if offset_sec is None:
        offset_sec = float(LONG_DEMO_OFFSET_SEC)
    if pair_index is None:
        pair_index = int(LONG_DEMO_PAIR_INDEX)
    if hop is None:
        hop = int(REALTIME_TFLITE_T)
    hop = int(hop)
    only = str(only).strip().lower()

    # Select pair
    try:
        path_src, path_tgt = all_pairs[pair_index]
    except Exception:
        raise RuntimeError(f"Invalid pair_index={pair_index}; there are {len(all_pairs)} pair(s).")

    # Load realtime DYNAMIC streaming model
    dpath = _select_stems_model_path(kind="realtime", use_finetuned=bool(use_finetuned))
    if dpath is None or (not os.path.exists(dpath)):
        raise FileNotFoundError(f"[check-streaming] Realtime dynamic model not found. Expected at: {dpath}")
    print(f"[check-streaming] Using realtime DYNAMIC model: {dpath}")
    rt_dyn = tf.keras.models.load_model(dpath, compile=False)

    # Read and resample audio to 48k
    info = sf.info(path_src)
    file_sr = info.samplerate
    start = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x_blk_2d, _ = sf.read(path_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_blk_2d, _ = sf.read(path_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:, 0]
    y_file = y_blk_2d[:, 0]
    x_mod = x_file if file_sr == REALTIME_SAMPLE_RATE else resample_to_model_sr(x_file, file_sr, REALTIME_SAMPLE_RATE)
    y_mod = y_file if file_sr == REALTIME_SAMPLE_RATE else resample_to_model_sr(y_file, file_sr, REALTIME_SAMPLE_RATE)
    N = len(x_mod)
    if N <= 0:
        raise RuntimeError("[check-streaming] Empty input after resampling.")

    # Helpers
    def _basename(t):
        nm = t.name.split(':')[0]
        if '/' in nm:
            nm = nm.split('/',1)[0]
        if nm.startswith("serving_default_"):
            nm = nm[len("serving_default_"):]
        return nm

    in_tensors = list(rt_dyn.inputs)
    in_names   = [_basename(t) for t in in_tensors]
    out_names  = _output_names_from_model(rt_dyn)

    # State input names and zero buffers
    state_in_names = [n for n in in_names if n.startswith("realtime_state_")]
    state_zero = {}
    for t in in_tensors:
        nm = _basename(t)
        if nm in state_in_names:
            L = int(t.shape[1]) if t.shape[1] is not None else 0
            C = int(t.shape[2]) if t.shape[2] is not None else 1
            state_zero[nm] = np.zeros((1, L, C), dtype=np.float32)

    # Map input-state -> output-state indices (…_next names)
    out_index_map = {name: idx for idx, name in enumerate(out_names)}
    state_out_idx = {}
    for nm in state_in_names:
        nm_next = nm + "_next"
        if nm_next in out_index_map:
            state_out_idx[nm] = out_index_map[nm_next]

    # Knobs fixed (fully wet by default)
    def _knobs(pa, pr, ir):
        return {
            "realtime_knob_pickattack": np.array([[float(pa)]], dtype=np.float32),
            "realtime_knob_pickreso":   np.array([[float(pr)]], dtype=np.float32),
            "realtime_knob_ir":         np.array([[float(ir)]], dtype=np.float32),
            "realtime_knob_drywet":     np.array([[_default_dryness_value()]], dtype=np.float32),
        }

    # Determine single style value to force for both passes
    style_mode = str(style_mode).strip().lower()
    if style_mode in ("pick", "p", "1", "true"):
        style_val = 1.0
        print("[check-streaming] Forcing style: pick (1.0)")
    elif style_mode in ("finger", "f", "0", "false"):
        style_val = 0.0
        print("[check-streaming] Forcing style: finger (0.0)")
    elif style_mode.startswith("prob="):
        try:
            style_val = float(style_mode.split("=",1)[1])
        except Exception:
            raise ValueError(f"Invalid style_mode '{style_mode}'. Use prob=<float 0..1>.")
        style_val = float(np.clip(style_val, 0.0, 1.0))
        print(f"[check-streaming] Forcing style: prob={style_val:.4f}")
    else:
        # style_mode == 'pred': compute one global probability on full clip
        x_b_probe = x_mod[None, :, None].astype(np.float32)
        probe = {
            "realtime_piezo_input":         x_b_probe,
            "realtime_style_true_input":    np.zeros((1,1), dtype=np.float32),
            "realtime_style_force_input":   np.zeros((1,1), dtype=np.float32),
            **_knobs(1.0, 1.0, 1.0),
        }
        for nm in state_in_names:
            probe[nm] = state_zero[nm]
        outs_probe = rt_dyn.predict(probe, batch_size=1, verbose=0)
        style_val = float(np.asarray(outs_probe[1], dtype=np.float32)[0, 0])
        style_val = float(np.clip(style_val, 0.0, 1.0))
        print(f"[check-streaming] Forcing global predicted style prob={style_val:.4f} (as a constant).")

    # Monolithic runner
    def forward_mono(pa, pr, ir):
        x_b = x_mod[None, :, None].astype(np.float32)
        feed = {
            "realtime_piezo_input": x_b,
            "realtime_style_true_input":  np.array([[style_val]], dtype=np.float32),
            "realtime_style_force_input": np.ones((1,1), dtype=np.float32),
            **_knobs(pa, pr, ir),
        }
        for nm in state_in_names:
            feed[nm] = state_zero[nm]
        outs = rt_dyn.predict(feed, batch_size=1, verbose=0)
        return outs[0][0, :, 0]

    # Streaming runner
    def forward_stream(pa, pr, ir):
        state_bufs = {k: v.copy() for k, v in state_zero.items()}
        y_accum = []
        pos = 0
        while pos < N:
            core = min(hop, N - pos)
            x_block = np.zeros((1, hop, 1), dtype=np.float32)
            x_block[0, :core, 0] = x_mod[pos:pos+core]
            feed = {
                "realtime_piezo_input": x_block,
                "realtime_style_true_input":  np.array([[style_val]], dtype=np.float32),
                "realtime_style_force_input": np.ones((1,1), dtype=np.float32),
                **_knobs(pa, pr, ir),
            }
            for nm in state_in_names:
                feed[nm] = state_bufs[nm]
            outs = rt_dyn.predict(feed, batch_size=1, verbose=0)
            y_blk = outs[0][0, :hop, 0]
            y_accum.append(y_blk[:core])
            for nm, idx in state_out_idx.items():
                state_bufs[nm] = outs[idx]
            pos += core
        return np.concatenate(y_accum, axis=0)

    # Decide which signals to compare based on --only
    if only == "full":
        y_mono = forward_mono(pa=1.0, pr=1.0, ir=1.0)
        y_stream = forward_stream(pa=1.0, pr=1.0, ir=1.0)
        label = "full"
    elif only == "main":
        y_mono = forward_mono(pa=0.0, pr=0.0, ir=0.0)
        y_stream = forward_stream(pa=0.0, pr=0.0, ir=0.0)
        label = "main"
    elif only == "ir":
        base_mono   = forward_mono(pa=0.0, pr=0.0, ir=0.0)
        ir_on_mono  = forward_mono(pa=0.0, pr=0.0, ir=1.0)
        base_stream = forward_stream(pa=0.0, pr=0.0, ir=0.0)
        ir_on_stream= forward_stream(pa=0.0, pr=0.0, ir=1.0)
        y_mono   = ir_on_mono   - base_mono
        y_stream = ir_on_stream - base_stream
        label = "ir"
    elif only == "pa":
        base_mono   = forward_mono(pa=0.0, pr=0.0, ir=0.0)
        pa_on_mono  = forward_mono(pa=1.0, pr=0.0, ir=0.0)
        base_stream = forward_stream(pa=0.0, pr=0.0, ir=0.0)
        pa_on_stream= forward_stream(pa=1.0, pr=0.0, ir=0.0)
        y_mono   = pa_on_mono   - base_mono
        y_stream = pa_on_stream - base_stream
        label = "pa"
    elif only == "pr":
        base_mono   = forward_mono(pa=0.0, pr=0.0, ir=0.0)
        pr_on_mono  = forward_mono(pa=0.0, pr=1.0, ir=0.0)
        base_stream = forward_stream(pa=0.0, pr=0.0, ir=0.0)
        pr_on_stream= forward_stream(pa=0.0, pr=1.0, ir=0.0)
        y_mono   = pr_on_mono   - base_mono
        y_stream = pr_on_stream - base_stream
        label = "pr"
    else:
        raise ValueError(f"Unknown --only='{only}'. Use one of: full, main, ir, pa, pr.")

    # Align lengths and compute metrics
    n = min(len(y_mono), len(y_stream))
    y_mono = y_mono[:n]
    y_stream = y_stream[:n]
    err = (y_mono - y_stream).astype(np.float32)
    rms_err = float(np.sqrt(np.mean(err*err) + 1e-12))
    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    rms_ref = float(np.sqrt(np.mean(y_mono*y_mono) + 1e-12))
    snr = float(20.0 * np.log10((rms_ref + 1e-12) / (rms_err + 1e-12)))
    corr = _corr(y_mono, y_stream)

    # Write audio
    out_dir = os.path.join(DEMOS_OUT_DIR, "realtime_streaming_equivalence", f"only_{label}")
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "mono.wav"),    y_mono,    REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "stream.wav"),  y_stream,  REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "diff.wav"),    err,       REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "source.wav"),  x_mod[:n], REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "target.wav"),  y_mod[:n], REALTIME_SAMPLE_RATE, subtype='FLOAT')

    # Report
    print("=== Streaming equivalence (realtime) ===")
    print(f"Pair: {os.path.basename(path_src)}")
    print(f"Length: {n} samples @ 48k, hop={hop}")
    print(f"Forced style value: {style_val:.6f} (mode='{style_mode}')")
    print(f"Branch: {label}")
    print(f"RMS error: {rms_err:.9f}")
    print(f"Max abs:   {max_abs:.9f}")
    print(f"SNR:       {snr:.2f} dB")
    print(f"Corr:      {corr:.6f}")
    print(f"WAVs written to: {out_dir}")

    # Tolerance (tight)
    thr_rms = 5e-6
    thr_max = 5e-6
    if rms_err <= thr_rms and max_abs <= thr_max:
        print("✅ PASS: Streaming equals monolithic within numerical tolerance.")
    else:
        print("⚠️  WARN: Differences exceed tolerance (inspect diff.wav and metrics).")

def inspect_realtime_heads(use_finetuned=False, sec=None, offset_sec=None, pair_index=None, style="pick", hop=None):
    assert_has_pairs()
    if sec is None: sec = float(LONG_DEMO_SEC)
    if offset_sec is None: offset_sec = float(LONG_DEMO_OFFSET_SEC)
    if pair_index is None: pair_index = int(LONG_DEMO_PAIR_INDEX)
    if hop is None: hop = int(REALTIME_TFLITE_T)

    # Pair
    try:
        path_src, _ = all_pairs[pair_index]
    except Exception:
        raise RuntimeError(f"Invalid pair_index={pair_index}; there are {len(all_pairs)} pair(s).")

    # Load streaming model
    dpath = _select_stems_model_path(kind="realtime", use_finetuned=bool(use_finetuned))
    if dpath is None or (not os.path.exists(dpath)):
        raise FileNotFoundError(f"[inspect-heads] Realtime dynamic model not found at: {dpath}")
    base = tf.keras.models.load_model(dpath, compile=False)

    # Build tap model (add many internal taps; handle multi-output layers by picking [0])
    taps = []
    def add_tap(layer_name, tap_name, child_idx=None):
        try:
            t = base.get_layer(layer_name).output
            if isinstance(t, (list, tuple)):
                idx = 0 if child_idx is None else int(child_idx)
                t = t[idx]
            taps.append(Activation('linear', name=tap_name)(t))
            return True
        except Exception:
            return False

    # Main + mask + user-scaled stems
    have_main   = add_tap("realtime_main",                    "tap_main")
    have_mask   = add_tap("realtime_pick_mask_db",            "tap_mask", 0)
    have_pauser = add_tap("realtime_pickattack_user_gain_mul","tap_pa_user")
    have_pruser = add_tap("realtime_pickreso_user_gain_mul",  "tap_pr_user")
    have_iruser = add_tap("realtime_ir_user_gain_mul",        "tap_ir_user")

    # Pick-attack internals
    have_hfpre  = add_tap("realtime_hf_pre_emph_stream",     "tap_hf_pre", 0)
    have_pag    = add_tap("realtime_picklearn_gate",         "tap_pa_gate")
    have_c1     = add_tap("realtime_picklearn_c1",           "tap_pa_c1")
    have_c2     = add_tap("realtime_picklearn_c2",           "tap_pa_c2")
    have_paout  = add_tap("realtime_picklearn_out",          "tap_pa_out")
    have_pas    = add_tap("realtime_pickattack_softclip",    "tap_pa_soft")
    have_pahp   = add_tap("realtime_pickattack_post_hp",     "tap_pa_posthp", 0)

    # Pick-reso internals
    have_onset  = add_tap("realtime_pickreso_onset_pulse",   "tap_pr_onset", 0)
    have_env    = add_tap("realtime_pickreso_ctrl_env",      "tap_pr_env", 0)
    have_exc    = add_tap("realtime_pickreso_excite",        "tap_pr_excite")
    have_ds8    = add_tap("realtime_pickreso_ds8",           "tap_pr_ds8", 0)
    have_modal  = add_tap("realtime_pickreso_modal_lr",      "tap_pr_modal", 0)
    have_mix    = add_tap("realtime_pickreso_mix",           "tap_pr_mix")
    have_prs    = add_tap("realtime_pickreso_softclip",      "tap_pr_soft")
    have_prhp   = add_tap("realtime_pickreso_post_hp",       "tap_pr_posthp", 0)

    tm = tf.keras.models.Model(inputs=base.inputs, outputs=list(base.outputs) + taps, name=base.name + "_debug_taps")
    # Print current HPF kernels inside the loaded streaming model
    dump_realtime_hpf_debug(base, sample_rate=REALTIME_SAMPLE_RATE)

    # Names and output map
    def _bn(t):
        nm = t.name.split(':')[0]
        if '/' in nm: nm = nm.split('/', 1)[0]
        if nm.startswith("serving_default_"): nm = nm[len("serving_default_"):]
        return nm

    in_tensors = list(tm.inputs)
    in_names   = [_bn(t) for t in in_tensors]
    out_names  = _output_names_from_model(tm)
    out_idx    = {n: i for i, n in enumerate(out_names)}

    # State mapping
    state_in_names = [n for n in in_names if n.startswith("realtime_state_")]
    state_out_idx = {}
    for n in state_in_names:
        nn = n + "_next"
        if nn in out_idx:
            state_out_idx[n] = out_idx[nn]

    state_bufs = {}
    for t in in_tensors:
        nm = _bn(t)
        if nm in state_in_names:
            L = int(t.shape[1]) if t.shape[1] is not None else 0
            C = int(t.shape[2]) if t.shape[2] is not None else 1
            state_bufs[nm] = np.zeros((1, L, C), dtype=np.float32)

    # Read and resample
    info = sf.info(path_src); file_sr = info.samplerate
    start  = int(round((offset_sec or 0.0) * file_sr))
    frames = int(round((sec or LONG_DEMO_SEC) * file_sr))
    x_blk_2d, _ = sf.read(path_src, start=start, frames=frames, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:,0]
    x_mod = x_file if file_sr == REALTIME_SAMPLE_RATE else resample_to_model_sr(x_file, file_sr, REALTIME_SAMPLE_RATE)
    N = len(x_mod)

    # Style force value
    style_val = 1.0 if str(style).lower().startswith("p") else 0.0

    # Head boosts (linear) and dryness
    pa_lin, pr_lin, ir_lin = _resolve_demo_knobs(kind="realtime", use_finetuned=use_finetuned)
    dry = _default_dryness_value()

    def filter_feed(d): return {k: v for k, v in d.items() if k in in_names}

    # Accumulators
    acc = {nm: [] for nm in ["main","mask","pa_user","pr_user","ir_user",
                             "hf_pre","pa_gate","pa_c1","pa_c2","pa_out","pa_soft","pa_posthp",
                             "pr_onset","pr_env","pr_excite","pr_ds8","pr_modal","pr_mix","pr_soft","pr_posthp"]}

    def take(outs, nm, core):
        i = out_idx.get(nm, None)
        if i is None or i >= len(outs): return None
        v = np.asarray(outs[i], dtype=np.float32)
        if v.ndim == 3: return v[0, :core, 0]
        if v.ndim == 2: return v[0, :core]
        return None

    pos = 0
    while pos < N:
        core = min(hop, N - pos)
        xb = np.zeros((1, hop, 1), dtype=np.float32)
        xb[0, :core, 0] = x_mod[pos:pos+core]

        feed = {
            "realtime_piezo_input": xb,
            "realtime_style_true_input":  np.array([[style_val]], np.float32),
            "realtime_style_force_input": np.ones((1,1), np.float32),
            "realtime_knob_pickattack":   np.array([[float(pa_lin)]], np.float32),
            "realtime_knob_pickreso":     np.array([[float(pr_lin)]], np.float32),
            "realtime_knob_ir":           np.array([[float(ir_lin)]], np.float32),
            "realtime_knob_drywet":       np.array([[dry]], np.float32),
        }
        for n in state_in_names: feed[n] = state_bufs[n]
        feed = filter_feed(feed)

        outs = tm.predict(feed, batch_size=1, verbose=0)

        # Fill
        if have_main:   acc["main"].append(take(outs,"tap_main", core))
        if have_mask:   acc["mask"].append(take(outs,"tap_mask", core))
        if have_pauser: acc["pa_user"].append(take(outs,"tap_pa_user", core))
        if have_pruser: acc["pr_user"].append(take(outs,"tap_pr_user", core))
        if have_iruser: acc["ir_user"].append(take(outs,"tap_ir_user", core))

        if have_hfpre:  acc["hf_pre"].append(take(outs,"tap_hf_pre", core))
        if have_pag:    acc["pa_gate"].append(take(outs,"tap_pa_gate", core))
        if have_c1:     acc["pa_c1"].append(take(outs,"tap_pa_c1", core))
        if have_c2:     acc["pa_c2"].append(take(outs,"tap_pa_c2", core))
        if have_paout:  acc["pa_out"].append(take(outs,"tap_pa_out", core))
        if have_pas:    acc["pa_soft"].append(take(outs,"tap_pa_soft", core))
        if have_pahp:   acc["pa_posthp"].append(take(outs,"tap_pa_posthp", core))

        if have_onset:  acc["pr_onset"].append(take(outs,"tap_pr_onset", core))
        if have_env:    acc["pr_env"].append(take(outs,"tap_pr_env", core))
        if have_exc:    acc["pr_excite"].append(take(outs,"tap_pr_excite", core))
        if have_ds8:    acc["pr_ds8"].append(take(outs,"tap_pr_ds8", core))
        if have_modal:  acc["pr_modal"].append(take(outs,"tap_pr_modal", core))
        if have_mix:    acc["pr_mix"].append(take(outs,"tap_pr_mix", core))
        if have_prs:    acc["pr_soft"].append(take(outs,"tap_pr_soft", core))
        if have_prhp:   acc["pr_posthp"].append(take(outs,"tap_pr_posthp", core))

        # Advance states present
        for n, i in state_out_idx.items():
            if i < len(outs):
                state_bufs[n] = outs[i]

        pos += core

    # Concatenate
    for k in list(acc.keys()):
        parts = [p for p in acc[k] if p is not None]
        acc[k] = np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)

    def rms(x): 
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0
    
        # Offline-apply the same HPF taps to pa_soft/pr_soft and compare RMS to model pa_posthp/pr_posthp
    try:
        # Grab canonical conv1d taps (time-reversed) and flip back to 'forward' FIR for lfilter
        k_pa_rev = _get_minphase_hp_kernel(REALTIME_SAMPLE_RATE,
                                           PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                                           PICK_ATTACK_LOWBAND_PENALTY_ORDER)
        k_pa_rev = k_pa_rev.numpy() if tf.is_tensor(k_pa_rev) else np.asarray(k_pa_rev, np.float32)
        b_pa = np.squeeze(k_pa_rev)[::-1].astype(np.float32)  # forward FIR taps

        k_pr_rev = _get_minphase_hp_kernel(REALTIME_SAMPLE_RATE,
                                           PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                                           PICK_RESO_LOWBAND_PENALTY_ORDER)
        k_pr_rev = k_pr_rev.numpy() if tf.is_tensor(k_pr_rev) else np.asarray(k_pr_rev, np.float32)
        b_pr = np.squeeze(k_pr_rev)[::-1].astype(np.float32)

        y_pa_np = spsig.lfilter(b_pa, [1.0], acc["pa_soft"]) if acc["pa_soft"].size else np.zeros((0,), np.float32)
        y_pr_np = spsig.lfilter(b_pr, [1.0], acc["pr_soft"]) if acc["pr_soft"].size else np.zeros((0,), np.float32)

        # Align to model’s VALID conv (drop the first L-1 samples)
        L_pa = int(len(b_pa)); L_pr = int(len(b_pr))
        y_pa_np_al = y_pa_np[L_pa-1:] if y_pa_np.size > (L_pa-1) else y_pa_np
        y_pr_np_al = y_pr_np[L_pr-1:] if y_pr_np.size > (L_pr-1) else y_pr_np

        # If lengths mismatch, crop to min for fair RMS
        def same_len(a, b):
            n = min(a.size, b.size)
            return a[:n], b[:n]

        pa_m, pa_np = same_len(acc["pa_posthp"], y_pa_np_al)
        pr_m, pr_np = same_len(acc["pr_posthp"], y_pr_np_al)

        def _rms(v): 
            v = np.asarray(v, np.float32)
            return float(np.sqrt(np.mean(v*v) + 1e-12)) if v.size else 0.0

        print("=== HPF post-debug compare (model vs offline) ===")
        print(f"pa_posthp: model_RMS={_rms(pa_m):.6f}  offline_RMS={_rms(pa_np):.6f}")
        print(f"pr_posthp: model_RMS={_rms(pr_m):.6f}  offline_RMS={_rms(pr_np):.6f}")

        # Write offline-filtered signals for listening
        out_dir = os.path.join(DEMOS_OUT_DIR, "realtime_heads_inspect")
        os.makedirs(out_dir, exist_ok=True)
        if pa_np.size:
            sf.write(os.path.join(out_dir, "pa_posthp_offline.wav"), pa_np, REALTIME_SAMPLE_RATE, subtype='FLOAT')
        if pr_np.size:
            sf.write(os.path.join(out_dir, "pr_posthp_offline.wav"), pr_np, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    except Exception as e:
        print(f"[hpf/debug] offline compare skipped due to: {e}")

    # Report
    print("=== Inspect realtime heads (forced style: {} ) ===".format("pick" if (style_val>=0.5) else "finger"))
    def _fmt(v): return "{:.6f}".format(v)
    print("Main/IR/Mask:")
    print("  main={}  ir_user={}".format(_fmt(rms(acc["main"])), _fmt(rms(acc["ir_user"]))))
    if acc["mask"].size:
        m = acc["mask"]
        duty20 = float(np.mean(m > 0.2))
        duty50 = float(np.mean(m > 0.5))
        q50 = float(np.quantile(m, 0.50)); q90 = float(np.quantile(m, 0.90)); q99 = float(np.quantile(m, 0.99))
        print("  mask: mean={} duty>0.2={:.3f} duty>0.5={:.3f} q50={:.3f} q90={:.3f} q99={:.3f}".format(
            _fmt(float(np.mean(m))), duty20, duty50, q50, q90, q99
        ))

    print("Pick-attack chain:")
    print("  hf_pre={}  pa_gate={}  c1={}  c2={}  pa_out={}  pa_soft={}  pa_posthp={}  pa_user={}".format(
        _fmt(rms(acc["hf_pre"])), _fmt(rms(acc["pa_gate"])), _fmt(rms(acc["pa_c1"])), _fmt(rms(acc["pa_c2"])),
        _fmt(rms(acc["pa_out"])), _fmt(rms(acc["pa_soft"])), _fmt(rms(acc["pa_posthp"])), _fmt(rms(acc["pa_user"]))
    ))

    print("Pick-reso chain:")
    print("  onset={}  env={}  excite={}  ds8={}  modal={}  mix={}  pr_soft={}  pr_posthp={}  pr_user={}".format(
        _fmt(rms(acc["pr_onset"])), _fmt(rms(acc["pr_env"])), _fmt(rms(acc["pr_excite"])),
        _fmt(rms(acc["pr_ds8"])), _fmt(rms(acc["pr_modal"])), _fmt(rms(acc["pr_mix"])),
        _fmt(rms(acc["pr_soft"])), _fmt(rms(acc["pr_posthp"])), _fmt(rms(acc["pr_user"]))
    ))

    # Write a few WAVs
    out_dir = os.path.join(DEMOS_OUT_DIR, "realtime_heads_inspect")
    os.makedirs(out_dir, exist_ok=True)
    def w(fn, y):
        if y.size: sf.write(os.path.join(out_dir, fn), y, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    w("main.wav", acc["main"]); w("mask.wav", acc["mask"])
    w("pa_gate.wav", acc["pa_gate"]); w("pa_out.wav", acc["pa_out"])
    w("pa_posthp.wav", acc["pa_posthp"]); w("pa_user.wav", acc["pa_user"])
    w("pr_onset.wav", acc["pr_onset"]); w("pr_excite.wav", acc["pr_excite"])
    w("pr_mix.wav", acc["pr_mix"]); w("pr_posthp.wav", acc["pr_posthp"]); w("pr_user.wav", acc["pr_user"])
    print(f"Written stems to {out_dir}")

@register_keras_serializable()
def _make_minphase_hp_kernel_exact(sr, cut_hz, numtaps):
    """
    Build an exact-length (numtaps) min-phase high-pass FIR, reversed for tf.nn.conv1d.
    Ensures odd numtaps (required by firwin HP).
    Returns shape (numtaps, 1, 1) float32.
    """
    nt = int(numtaps)
    if nt < 3:
        nt = 3
    if (nt % 2) == 0:
        nt += 1  # HP requires odd
    wn = float(cut_hz) / (float(sr) / 2.0)
    taps = firwin(nt, cutoff=wn, window="hamming", pass_zero=False).astype(np.float32)
    taps_min = minimum_phase(taps, method="homomorphic").astype(np.float32)
    k_rev = taps_min[::-1].reshape(nt, 1, 1).astype(np.float32)
    return k_rev


@register_keras_serializable()
def _hp_minphase_rev_padded(sr, cut_hz, target_len):
    """
    Build conv1d-ready reversed min-phase HPF, exact length target_len.
    If minimum_phase returns fewer taps, left-pad zeros so reversed taps are right-aligned.
    Returns (target_len,1,1) float32.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    # HPF firwin prefers odd numtaps; keep design odd but do not change target_len
    design_nt = nt if (nt % 2) == 1 else (nt - 1 if nt > 3 else 3)
    wn = float(cut_hz) / (float(sr) / 2.0)
    taps = firwin(design_nt, cutoff=wn, pass_zero=False, window="hamming").astype(np.float32)
    taps_min = minimum_phase(taps, method="homomorphic").astype(np.float32)
    k_rev_small = taps_min[::-1].reshape(-1)   # reversed for conv1d
    M = int(k_rev_small.shape[0])
    k_full = np.zeros((nt,), dtype=np.float32)
    if M >= nt:
        k_full[:] = k_rev_small[-nt:]
    else:
        pad = nt - M
        k_full[pad:] = k_rev_small
    return k_full.reshape(nt, 1, 1).astype(np.float32)

@register_keras_serializable()
def _bp_minphase_rev_padded(sr, low_hz, high_hz, target_len):
    """
    Build conv1d-ready reversed min-phase BP, exact length target_len.
    Left-pad zeros if minimum_phase returns fewer taps.
    Returns (target_len,1,1) float32.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    design_nt = nt if (nt % 2) == 1 else (nt - 1 if nt > 3 else 3)
    nyq = float(sr) / 2.0
    lo = float(low_hz) / nyq
    hi = float(high_hz) / nyq
    taps = firwin(design_nt, [lo, hi], pass_zero=False, window="hamming").astype(np.float32)
    taps_min = minimum_phase(taps, method="homomorphic").astype(np.float32)
    k_rev_small = taps_min[::-1].reshape(-1)
    M = int(k_rev_small.shape[0])
    k_full = np.zeros((nt,), dtype=np.float32)
    if M >= nt:
        k_full[:] = k_rev_small[-nt:]
    else:
        pad = nt - M
        k_full[pad:] = k_rev_small
    return k_full.reshape(nt, 1, 1).astype(np.float32)

@register_keras_serializable()
def dump_realtime_hpf_debug(model, sample_rate=REALTIME_SAMPLE_RATE):
    """
    Print kernel stats for the two streaming post-HPF layers and compare
    against freshly built taps for sanity.
    """
    def _get_kernel(m, lname):
        try:
            L = m.get_layer(lname)
            w = L.get_weights()
            if not w:
                return None
            k = np.asarray(w[0], dtype=np.float32).copy()
            return k.reshape(-1) if k.ndim == 3 else k
        except Exception:
            return None

    def _mk_hp(sr, cut, order):
        k = _get_minphase_hp_kernel(int(sr), float(cut), int(order))
        k = k.numpy() if tf.is_tensor(k) else np.asarray(k, dtype=np.float32)
        return k.reshape(-1)

    def _summ(name, k):
        if k is None:
            print(f"[hpf/debug] {name}: MISSING")
            return
        print(f"[hpf/debug] {name}: shape={k.shape}, L2={float(np.linalg.norm(k)):.6f}, "
              f"sum={float(np.sum(k)):.6e}, maxabs={float(np.max(np.abs(k))):.6f}")
        head = ", ".join(f"{v:.6f}" for v in k[:6])
        tail = ", ".join(f"{v:.6f}" for v in k[-6:])
        print(f"[hpf/debug] {name}: head=[{head}] ... tail=[{tail}]")

    # Pull current kernels from the loaded streaming model
    k_pa_m = _get_kernel(model, "realtime_pickattack_post_hp")
    k_pr_m = _get_kernel(model, "realtime_pickreso_post_hp")

    # Build canonical taps (time-reversed for conv1d)
    k_pa_ref = _mk_hp(sample_rate, PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ, PICK_ATTACK_LOWBAND_PENALTY_ORDER)
    k_pr_ref = _mk_hp(sample_rate, PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ, PICK_RESO_LOWBAND_PENALTY_ORDER)

    print("=== HPF kernel debug (streaming realtime) ===")
    _summ("model.pa_posthp", k_pa_m)
    _summ("ref.pa_posthp",   k_pa_ref)
    if k_pa_m is not None:
        try:
            cos = float(np.dot(k_pa_m, k_pa_ref) / (np.linalg.norm(k_pa_m) * np.linalg.norm(k_pa_ref) + 1e-12))
            print(f"[hpf/debug] pa cosine(model, ref) = {cos:.6f}")
        except Exception:
            pass

    _summ("model.pr_posthp", k_pr_m)
    _summ("ref.pr_posthp",   k_pr_ref)
    if k_pr_m is not None:
        try:
            cos = float(np.dot(k_pr_m, k_pr_ref) / (np.linalg.norm(k_pr_m) * np.linalg.norm(k_pr_ref) + 1e-12))
            print(f"[hpf/debug] pr cosine(model, ref) = {cos:.6f}")
        except Exception:
            pass
    print("============================================")

def main ():
    if len(sys.argv)>1 and sys.argv[1]=="--train-offline":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_OFFLINE_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR
        )
        if resume:
            train_offline(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_offline()
    elif len(sys.argv)>1 and sys.argv[1]=="--train-realtime":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_REALTIME_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR
        )
        if resume:
            train_realtime(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_realtime()
    elif (len(sys.argv)>1 and sys.argv[1]=="--demo"):
        print("Generating demo audio files...")
        args = set(sys.argv[2:])
        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)
        
        time_start = time.time()

        # Offline baseline: load saved dynamic offline model
        if do_offline:
            prepare_and_generate_demo(
                kind="offline",
                sample_rate=OFFLINE_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_OFFLINE_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
            )

        # Realtime baseline: load saved static T=32 streaming model
        if do_realtime:
            prepare_and_generate_demo(
                kind="realtime",
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_REALTIME_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
            )
        
        time_end = time.time()
        print(f"✅ Demo generation complete (took {time_end - time_start:.1f} sec).")
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        offline_dyn, realtime_dyn = fine_tune(offline=len(sys.argv) <= 2 or sys.argv[2] !="--realtime-only", realtime=len(sys.argv) <= 2 or sys.argv[2] !="--offline-only")
        
        if (offline_dyn is not None):
            generate_fine_tune_demo(model=offline_dyn, kind="offline", sample_rate=OFFLINE_SAMPLE_RATE)
        if (realtime_dyn is not None):
            generate_fine_tune_demo(model=realtime_dyn, kind="realtime", sample_rate=REALTIME_SAMPLE_RATE)

        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune-demo":
        # Load finetuned models if present and render both
        offline_dyn_path  = os.path.join(MODELS_OFFLINE_OUT_DIR,  FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        realtime_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)

        offline_dyn  = tf.keras.models.load_model(offline_dyn_path,  compile=False) if os.path.exists(offline_dyn_path)  else None
        realtime_dyn = tf.keras.models.load_model(realtime_dyn_path, compile=False) if os.path.exists(realtime_dyn_path) else None

        if offline_dyn is None and realtime_dyn is None:
            print("Fine-tuned dynamic models not found, run --fine-tune first")
            sys.exit(1)

        if offline_dyn is not None:
            generate_fine_tune_demo(model=offline_dyn, kind="offline", sample_rate=OFFLINE_SAMPLE_RATE)
        if realtime_dyn is not None:
            generate_fine_tune_demo(model=realtime_dyn, kind="realtime", sample_rate=REALTIME_SAMPLE_RATE)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--render-stems":        
        # Flags (can be combined):
        # --finetuned        → finetuned dynamic models (if present)
        # --both             → baseline + finetuned
        # --all              → baseline + finetuned + learned-pick
        # --offline-only / --realtime-only limits branches
        args = set(sys.argv[2:])

        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)
        do_finetuned = ("--finetuned"     in args)
        do_both      = ("--both"          in args)
        do_all       = ("--all"           in args)

        # OFFLINE: baseline or finetuned (existing behavior)
        if do_offline:
            if do_all:
                run_render_stems_offline(use_finetuned=False)
                run_render_stems_offline(use_finetuned=True)
            elif do_both:
                run_render_stems_offline(use_finetuned=False)
                run_render_stems_offline(use_finetuned=True)
            else:
                run_render_stems_offline(use_finetuned=do_finetuned)

        # REALTIME: baseline / finetuned / learned-pick
        if do_realtime:
            if do_all:
                run_render_stems_realtime(use_finetuned=False)
                run_render_stems_realtime(use_finetuned=True)
            elif do_both:
                run_render_stems_realtime(use_finetuned=False)
                run_render_stems_realtime(use_finetuned=True)
            else:
                # single selection
                run_render_stems_realtime(use_finetuned=do_finetuned)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--print-layers":
        
        paths = [
            os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        ]

        for p in paths:
            print("[offline/persist] loading:", p)
            m = tf.keras.models.load_model(p, compile=False)
            
            for l in m.layers:
                # Print layer name
                print(f"Layer: {l.name}, Type: {type(l)}")
            print("-----")
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-realtime":
        args = set(sys.argv[2:])
        use_finetuned = ("--finetuned" in args)
        path = os.path.join(
            MODELS_REALTIME_OUT_DIR,
            FINETUNE_REALTIME_STATIC_MODEL_FILENAME if use_finetuned else REALTIME_STATIC_MODEL_FILENAME
        )
        print("[inspect-realtime] Loading:", path)
        m = tf.keras.models.load_model(path, compile=False)
        _print_model_io_summary(m)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--export-tflite":
        print("Exporting FP32 TFLite models (offline T=40960, realtime T=32)…")
        export_tflite_all()
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--demo-tflite":
        args = set(sys.argv[2:])
        do_offline    = ("--realtime-only" not in args)
        do_realtime   = ("--offline-only"  not in args)
        use_finetuned = ("--finetuned"     in args)
        do_both       = ("--both" in args)
        
        # Measure the time it takes to do this:
        time_start = time.time()

        # Optional hop parsing: --hop=16
        hop = None
        for a in list(args):
            if a.startswith("--hop="):
                try:
                    hop = int(a.split("=",1)[1])
                except Exception:
                    hop = None

        if do_offline:
            generate_demo_tflite(kind="offline", use_finetuned=use_finetuned)

        if do_realtime:
            if do_both:
                # T=32 and large demo-T, both
                generate_demo_tflite(kind="realtime", use_finetuned=use_finetuned, hop=hop)
            else:
                generate_demo_tflite(kind="realtime", use_finetuned=use_finetuned, hop=hop)
        time_end = time.time()
        print(f"[TFLite/demo] All done in {time_end - time_start:.1f} seconds.")
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-realtime":
        migrate_realtime_models_inplace()
        
        m = tf.keras.models.load_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME), compile=False)
        print([t.name.split(':')[0] for t in m.inputs if 'realtime_knob_' in t.name])
        # Prints: realtime_knob_pickattack, realtime_knob_pickreso, realtime_knob_ir, realtime_knob_drywet
        
        w = m.get_layer("realtime_pickreso_modal_lr").get_weights()[0]
        print("modal_lr shape:", w.shape, "L2:", float(np.linalg.norm(w)))
        
        # Prints: modal_lr shape: (128, 1, 9) L2: 2.606722831726074
        
    elif len(sys.argv)>1 and sys.argv[1]=="--check-streaming":
        # Options:
        #   --finetuned
        #   --hop=NNN
        #   --sec=S
        #   --offset-sec=S
        #   --pair-index=K
        #   --style=pick|finger|pred|prob=<float>
        #   --only=full|main|ir|pa|pr
        args = sys.argv[2:]
        use_finetuned = any(a == "--finetuned" for a in args)
        hop = None
        sec = None
        offset = None
        pair_idx = None
        style = "pred"
        only = "full"
        for a in args:
            if a.startswith("--hop="):
                try: hop = int(a.split("=",1)[1])
                except: hop = None
            elif a.startswith("--sec="):
                try: sec = float(a.split("=",1)[1])
                except: sec = None
            elif a.startswith("--offset-sec=") or a.startswith("--offset="):
                try: offset = float(a.split("=",1)[1])
                except: offset = None
            elif a.startswith("--pair-index="):
                try: pair_idx = int(a.split("=",1)[1])
                except: pair_idx = None
            elif a.startswith("--style="):
                style = a.split("=",1)[1].strip()
            elif a.startswith("--only="):
                only = a.split("=",1)[1].strip().lower()

        check_streaming_equivalence_realtime(use_finetuned=use_finetuned,
                                             hop=hop,
                                             sec=sec,
                                             offset_sec=offset,
                                             pair_index=pair_idx,
                                             style_mode=style,
                                             only=only)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-heads":
        # Options: --finetuned --sec=S --offset-sec=S --pair-index=K --style=pick|finger --hop=N
        args = sys.argv[2:]
        use_finetuned = any(a=="--finetuned" for a in args)
        sec=None; off=None; idx=None; style="pick"; hop=None
        for a in args:
            if a.startswith("--sec="): sec=float(a.split("=",1)[1])
            elif a.startswith("--offset-sec=") or a.startswith("--offset="): off=float(a.split("=",1)[1])
            elif a.startswith("--pair-index="): idx=int(a.split("=",1)[1])
            elif a.startswith("--style="): style=a.split("=",1)[1]
            elif a.startswith("--hop="): hop=int(a.split("=",1)[1])
        inspect_realtime_heads(use_finetuned=use_finetuned, sec=sec, offset_sec=off, pair_index=idx, style=style, hop=hop)
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()