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

# On/off flags
DISABLE_SCALE_BOOSTS = False

EPOCHS      = 40
FINETUNE_EPOCHS = 3

FINETUNE_OFFLINE_STEPS_CAP       = 250
FINETUNE_REALTIME_STEPS_CAP      = FINETUNE_OFFLINE_STEPS_CAP * 10

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
OFFLINE_GATE_ENV_MS          = 0.35
OFFLINE_GATE_DEPTH_SMOOTH_MS = 20.0

# Finetune mic-target denoise (preamp hiss cleanup)
FINETUNE_TARGET_DENOISE_ENABLE      = True     # set False to disable
FINETUNE_TARGET_DENOISE_WIN_MS      = 20.0     # STFT window
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

# Offline (96k) mask distillation strengths
OFFLINE_HF_MASK_DISTILL_ALPHA   = 2.0   # KD weight for teacher gate → student learned mask
REALTIME_HF_MASK_DISTILL_ALPHA = 2.0  # KD strength for the learned HF mask (teacher gate → student mask)

OFFLINE_MASK_USED_DISTILL_ALPHA = 0.3   # KD weight for teacher gate → student mask_used (smaller)
REALTIME_MASK_USED_DISTILL_ALPHA    = 0.3   # KD of mask_used vs teacher gate (small; avoids cap conflict)

OFFLINE_HF_SUPPRESS_ALPHA       = 0.7   # suppress student HF when teacher gate is closed

REALTIME_HF_HP_ORDER          = OFFLINE_HF_HP_ORDER
REALTIME_STYLE_DISTILL_ALPHA = 1.0   # KD strength for the style probability (teacher→student)

REALTIME_AUDIO_STFT_DISTILL_ALPHA   = 0.6   # KD weight for STFT KD (teacher audio_ds vs student audio)
REALTIME_STEM_PICK_DISTILL_ALPHA    = 0.5   # KD weight for pick stem
REALTIME_STEM_FINGER_DISTILL_ALPHA  = 0.5   # KD weight for finger stem
REALTIME_STEM_HF_DISTILL_ALPHA      = 0.8   # KD weight for HF stem
REALTIME_FEAT_DISTILL_ALPHA         = 1.0
REALTIME_STEM_RESBP_DISTILL_ALPHA   = 0.4

# Training-only penalty to suppress <~3 kHz energy in the realtime model's HF branch
REALTIME_HF_LOWBAND_PENALTY_ALPHA   = 1.0   # L1 on LPF(hf_final) with ~2.8-3.0 kHz cutoff
REALTIME_HF_LOWBAND_PENALTY_CUTOFF_HZ = max(2900.0, 0.75 * REALTIME_HP_CUTOFF_HZ)
REALTIME_STYLE_BLEND_INIT           = 0.7
REALTIME_STYLE_BLEND_TRAINABLE      = True
REALTIME_OFFLINE_FEAT_LAYERS   = ["offline_dil_2_film_apply", "offline_dil_3_film_apply"]

REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ   = OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ
REALTIME_GATE_THRESHOLD_DB               = OFFLINE_GATE_THRESHOLD_DB
REALTIME_GATE_RATIO                      = OFFLINE_GATE_RATIO
REALTIME_GATE_ATTACK_MS                  = 0.02
REALTIME_GATE_HOLD_MS                    = 4.0
REALTIME_GATE_RELEASE_MS                 = 20.0
REALTIME_GATE_ENV_MS                     = 0.0
REALTIME_GATE_DEPTH_SMOOTH_MS            = 0.25
REALTIME_GATE_ENV_LMAX                  = 256     # keep kernels inside streaming left context
REALTIME_GATE_DEPTH_LMAX                = 256

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
OFFLINE_RESBP_BP_ORDER = 257
REALTIME_RESBP_BP_ORDER = 257

# Initial scalar (trainable) for the band-pass branch
OFFLINE_RESBP_SCALE_INIT  = 2.0
REALTIME_RESBP_SCALE_INIT = 2.0

OFFLINE_RESBP_BOOST_DB    = -1000.0 if DISABLE_SCALE_BOOSTS else 40.0
REALTIME_RESBP_BOOST_DB   = -1000.0 if DISABLE_SCALE_BOOSTS else 10.0

FINETUNE_OFFLINE_RESBP_BOOST_DB    = -1000.0 if DISABLE_SCALE_BOOSTS else 100.0
FINETUNE_REALTIME_RESBP_BOOST_DB   = -1000.0 if DISABLE_SCALE_BOOSTS else 60.0


# Frequencies chosen to cover the “weird” pick resonances roughly in 0.7–2.2 kHz.
# Realtime kernels are short to keep left context low; offline can be longer.

RESBP_MODAL_CENTERS_HZ_OFFLINE = [740.0, 880.0, 1040.0, 1200.0, 1380.0, 1580.0, 1800.0, 2050.0, 2200.0]
RESBP_MODAL_DECAYS_MS_OFFLINE  = [ 45.0,  40.0,   36.0,   32.0,   27.0,   24.0,   20.0,   18.0,   16.0]
RESBP_MODAL_LEN_OFFLINE        = 8192

RESBP_MODAL_CENTERS_HZ_REALTIME = [740.0, 880.0, 1040.0, 1200.0, 1380.0, 1580.0, 1800.0, 2050.0, 2200.0]
RESBP_MODAL_DECAYS_MS_REALTIME  = [ 45.0,  40.0,   36.0,   32.0,   27.0,   24.0,   20.0,   20.0,   28.0]
RESBP_MODAL_LEN_REALTIME        = 4096

# Strike envelope smoothing (controls amplitude vs pick intensity)
RESBP_STRIKE_ENV_MS_OFFLINE     = 15.0
RESBP_STRIKE_ENV_MS_REALTIME    = 4.0

REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_realtime.keras"
REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_realtime_dynamic.keras"

# Baseline realtime (learned pick) export filenames
REALTIME_LEARNED_PICK_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_realtime_learned_pick.keras"
REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_realtime_learned_pick_dynamic.keras"

REALTIME_LEARNED_PICK_MASK_EPOCHS    = 15
REALTIME_LEARNED_PICK_MASK_STEPS_CAP = 3000
REALTIME_LEARNED_PICK_MASK_LR        = 1e-4

# Learned-pick shaping defaults for realtime
REALTIME_MASK_RELEASE_MS = REALTIME_GATE_RELEASE_MS
REALTIME_MASK_SOFTKNEE_K = 10.0
REALTIME_MASK_SOFTKNEE_T = 0.35
REALTIME_HF_PEAK_CEILING = 2.0  # set to 1.5–2.0 to tame rare spikes; None disables

REALTIME_MASK_CLOSE_T   = 0.25   # dead-zone lower bound (close)
REALTIME_MASK_OPEN_T    = 0.60   # dead-zone upper bound (open)
REALTIME_MASK_SHARPEN_P = 60     # integer power (>=2) for mask sharpening
REALTIME_MASK_LOGIT_BIAS = -3.5  # more negative initializes mask closed

# Finetune params
FINETUNE_REALTIME_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_realtime.keras"
FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_realtime_dynamic.keras"
FINETUNE_OFFLINE_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_offline.keras"
FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_offline_dynamic.keras"

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

REALTIME_IR_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 1.5
FINETUNE_REALTIME_IR_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 1.5

REALTIME_LP_ORDER   = 257 
REALTIME_LP_CUTOFF  = 2800

LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

OFFLINE_HF_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 28
REALTIME_HF_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 8

FINETUNE_OFFLINE_HF_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 24
FINETUNE_REALTIME_HF_BOOST_DB = -1000.0 if DISABLE_SCALE_BOOSTS else 4

REALTIME_HF_TIME_SHIFT_SAMPLES = -1
OFFLINE_HF_TIME_SHIFT_SAMPLES  = 0

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

# Upper frequency limits for mel loss in finetune (body region only)
FINETUNE_BODY_FMAX_REALTIME = 12000.0
FINETUNE_BODY_FMAX_OFFLINE  = 16000.0

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

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_OFFLINE_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "offline")
DEMO_CHECKPOINTS_REALTIME_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "realtime")

# Stems demos
STEMS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "stems")
STEMS_OFFLINE_OUT_DIR = os.path.join(STEMS_OUT_DIR, "offline")
STEMS_REALTIME_OUT_DIR = os.path.join(STEMS_OUT_DIR, "realtime")
STEMS_REALTIME_LEARNED_OUT_DIR = os.path.join(STEMS_OUT_DIR, "realtime_learned_pick")

STEMS_FINETUNE_OUT_DIR = os.path.join(STEMS_OUT_DIR, "finetune")
STEMS_FINETUNE_OFFLINE_OUT_DIR = os.path.join(STEMS_FINETUNE_OUT_DIR, "offline")
STEMS_FINETUNE_REALTIME_OUT_DIR = os.path.join(STEMS_FINETUNE_OUT_DIR, "realtime")

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
    STEMS_FINETUNE_OFFLINE_OUT_DIR, STEMS_FINETUNE_REALTIME_OUT_DIR,
    STEMS_REALTIME_LEARNED_OUT_DIR
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
class MaskFloor(Layer):
    """
    y = eps + (1 - eps) * x; used to enforce a minimum mask floor without Lambda(lambda ...)
    """
    def __init__(self, eps=0.02, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = float(eps)
    def call(self, x):
        eps = tf.constant(self.eps, dtype=x.dtype)
        return eps + (1.0 - eps) * x
    def get_config(self):
        return {**super().get_config(), "eps": self.eps}

@register_keras_serializable()
class OnesLike(Layer):
    """
    Stateless replacement for Lambda(lambda z: tf.ones_like(z)).
    """
    def call(self, x):
        return tf.ones_like(x)

@register_keras_serializable()
class TimeShift1D(Layer):
    """
    Causal, Keras-serializable integer time shift with constant length.
    - shift > 0: delay by 'shift' samples (left-pad zeros, crop end)
    - shift < 0: advance by 'abs(shift)' samples (drop head, pad zeros at end)
    Output length is the same as input; no lookahead; no extra latency at I/O.
    """
    def __init__(self, shift=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shift = int(shift)

    def call(self, x):
        if self.shift == 0:
            return x
        T = tf.shape(x)[1]
        s = abs(self.shift)
        if self.shift > 0:
            # Delay: pad on the left, then crop to original length
            x_pad = tf.pad(x, [[0,0], [s,0], [0,0]])
            return x_pad[:, :T, :]
        else:
            # Advance: drop 's' from the head, pad zeros at the end
            x_crop = x[:, s:, :]
            x_pad  = tf.pad(x_crop, [[0,0], [0, s], [0,0]])
            return x_pad[:, :T, :]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "shift": self.shift}

@register_keras_serializable()
class ZeroBias(constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)
    def get_config(self):
        return {}
    
@register_keras_serializable()
class MaskRemapLinear(tf.keras.layers.Layer):
    """
    Static 2-threshold remap with dead zone:
      y = clip((x - close_t) / (open_t - close_t), 0, 1)
    - close_t < open_t in [0,1]
    """
    def __init__(self, close_t=0.25, open_t=0.60, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.close_t = float(close_t)
        self.open_t  = float(open_t)
        self._scale  = None

    def build(self, input_shape):
        denom = max(1e-6, (self.open_t - self.close_t))
        self._scale = np.float32(1.0 / denom)
        super().build(input_shape)

    def call(self, x):
        y = (x - self.close_t) * self._scale
        return tf.clip_by_value(y, 0.0, 1.0)

    def get_config(self):
        return {"close_t": self.close_t, "open_t": self.open_t, **super().get_config()}


@register_keras_serializable()
class MaskSharpen(tf.keras.layers.Layer):
    """
    Monotonic sharpening using repeated multiply: y = x^power (power integer >=2)
    Implemented with multiplies only (TFLite-friendly).
    """
    def __init__(self, power=2, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.power = int(power)

    def call(self, x):
        y = x
        for _ in range(max(1, self.power) - 1):
            y = y * x
        return y

    def get_config(self):
        return {"power": self.power, **super().get_config()}


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
class FinetuneCombinedLossLite(tf.keras.losses.Loss):
    """
    Lightweight loss for strict finetune:
      L = l1_w * L1 + hf_w * HF(preemph) + stft_w * 1-scale MR-STFT + env_w * RMS-env
    Single STFT scale by default to keep it very cheap.
    """
    def __init__(self,
                 sample_rate,
                 stft_scales=((1024, 256, 1024),),  # single scale by default
                 l1_w=1.0, hf_w=2.0, stft_w=1.0, env_w=0.0,
                 onset_gain=HF_ONSET_GAIN,
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

        # Precompute Hann windows (shape (1,1,win)) as constants
        self._wins = []
        for _, _, win in self.stft_scales:
            w = tf.signal.hann_window(win, periodic=True)
            w = tf.reshape(w, (1, 1, win))
            self._wins.append(tf.constant(w, dtype=tf.float32))

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

    def call(self, y_true, y_pred):
        gate_env = gate_weight_dynamic(y_true, self.sample_rate)  # (B,T,1) in [0,1]
        # Gate the L1 to de-emphasize background noise
        td = waveform_l1_loss(y_true, y_pred) * (0.2 + 0.8 * gate_env)
        hf = hf_loss(y_true, y_pred) * (1.0 + self.onset_gain * gate_env)
        spec = self._mrstft(y_true, y_pred, gate_env) if self.stft_scales else 0.0
        env = rms_env_loss(y_true, y_pred) if self.env_w != 0.0 else 0.0
        return (self.l1_w * td + self.hf_w * hf + self.stft_w * spec + self.env_w * env)

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
def log_mel_loss_band(y_true, y_pred, sample_rate, fmax_hz):
    n_fft   = 1024
    hop     = 256
    win     = 1024
    n_mels  = 64
    fmin    = 20.0
    fmax    = float(min(fmax_hz, 0.5 * sample_rate))
    t   = tf.squeeze(y_true, axis=-1)
    p   = tf.squeeze(y_pred, axis=-1)
    st_t = tf.signal.stft(t, frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
    st_p = tf.signal.stft(p, frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
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
def band_limited_l1_loss_weighted(y_true, y_pred, sample_rate,
                                  low_hz, high_hz,
                                  order_off=OFFLINE_RESBP_BP_ORDER, order_rt=REALTIME_RESBP_BP_ORDER,
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
        lo = 0.9 * min(RESBP_MODAL_CENTERS_HZ_OFFLINE)
        hi = 1.1 * max(RESBP_MODAL_CENTERS_HZ_OFFLINE)
        order = OFFLINE_RESBP_BP_ORDER
    else:
        lo = 0.9 * min(RESBP_MODAL_CENTERS_HZ_REALTIME)
        hi = 1.1 * max(RESBP_MODAL_CENTERS_HZ_REALTIME)
        order = REALTIME_RESBP_BP_ORDER
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
def coloreq_band_energy_loss(y_true, y_pred, sample_rate, bands, order=129):
    """
    For each (lo,hi) band: L += |RMS(BP(y_true)) - RMS(BP(y_pred))|.
    Returns (B,1,1).
    """
    losses = []
    for (lo, hi) in bands:
        k = _get_minphase_bp_kernel(sample_rate, lo, hi, order)
        t = _causal_conv1d(y_true, k)
        p = _causal_conv1d(y_pred, k)
        # Global RMS per sample
        rms_t = tf.sqrt(tf.reduce_mean(tf.square(t), axis=1, keepdims=True) + 1e-12)
        rms_p = tf.sqrt(tf.reduce_mean(tf.square(p), axis=1, keepdims=True) + 1e-12)
        losses.append(tf.abs(rms_t - rms_p))
    return tf.add_n(losses) / float(max(len(bands), 1.0))

@register_keras_serializable()
class FinetuneAdapterLoss(tf.keras.losses.Loss):
    """
    Drives the post‑EQ adapter:
      L = l1_w * L1 + mel_w * (band-limited LogMel) + band_w * BandEnergy(bands)
    """
    def __init__(self,
                 sample_rate,
                 bands,
                 band_order=129,
                 l1_w=0.5,
                 mel_w=1.5,
                 band_w=2.0,
                 mel_upper_hz=None,
                 name="finetune_adapter_loss",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = int(sample_rate)
        self.bands = [(float(a), float(b)) for (a, b) in bands]
        self.band_order = int(band_order)
        self.l1_w  = float(l1_w)
        self.mel_w = float(mel_w)
        self.band_w= float(band_w)
        self.mel_upper_hz = None if mel_upper_hz is None else float(mel_upper_hz)

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "bands": self.bands,
            "band_order": self.band_order,
            "l1_w": self.l1_w, "mel_w": self.mel_w, "band_w": self.band_w,
            "mel_upper_hz": self.mel_upper_hz,
            **super().get_config()
        }

    def call(self, y_true, y_pred):
        loss = 0.0
        if self.l1_w != 0.0:
            l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2], keepdims=True)
            loss = loss + self.l1_w * l1
        if self.mel_w != 0.0:
            if self.mel_upper_hz is None:
                mel = log_mel_loss(y_true, y_pred, self.sample_rate)
            else:
                mel = log_mel_loss_band(y_true, y_pred, self.sample_rate, fmax_hz=self.mel_upper_hz)
            loss = loss + self.mel_w * mel
        if self.band_w != 0.0:
            be = coloreq_band_energy_loss(y_true, y_pred, self.sample_rate, self.bands, order=self.band_order)
            loss = loss + self.band_w * be
        return loss

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

    return (L1_WEIGHT       * td
          + HF_WEIGHT       * hf
          + STFT_WEIGHT     * spec
          + MEL_WEIGHT      * mel
          + RMS_ENV_WEIGHT  * env
          + COMPLEX_STFT_WEIGHT * cplx
          + D2_WEIGHT       * d2
          + RESBP_BAND_LOSS_WEIGHT * bl)

# Replace mean_hf_metric with this version
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    vals = hf_loss(y_true, y_pred)  # (B, T, 1)
    # Return per-sample metric with shape (B, 1, 1), compatible with (B, T, 1) sample weights
    return tf.reduce_mean(vals, axis=[1, 2], keepdims=True)

# ────────────────────────────── DATA PIPELINE HELPERS ───────────────────────

@register_keras_serializable()
def stop_grad(x):
    return tf.stop_gradient(x)


@register_keras_serializable()
def _set_kd_alphas(model, alpha_map):
    for lname, a in alpha_map.items():
        try:
            layer = model.get_layer(lname)
            if hasattr(layer, "set_alpha"):
                layer.set_alpha(float(a))
                print(f"[KD] set {lname}.alpha = {a}")
        except Exception:
            pass

@register_keras_serializable()
def _disable_non_mask_kd(model):
    zero_names = [
        "audio_distill",
        "audio_stft_distill",
        "feat_distill_0", "feat_distill_1",
        "stem_pick_distill", "stem_finger_distill",
        "stem_hf_distill", "stem_resbp_distill",
    ]
    _set_kd_alphas(model, {n: 0.0 for n in zero_names})

@register_keras_serializable()
def _boost_mask_kd(model):
    strong = {
        "hf_mask_bce":          5.0,  # teacher gate → mask_pred (BCE)
        "mask_smooth_distill":  1.2,  # teacher gate → mask_smoothed (shift-aware)
        "hf_env_mask_distill":  1.2,  # teacher HF env → mask_pred
        "hf_suppress_closed":   1.0,  # penalize HF when teacher gate is closed
        "mask_suppress_closed": 0.8,  # penalize mask open when teacher gate is closed
        # if you add the two KD terms below in build_realtime_model (see patch B),
        # you can set their alphas here as well:
        "mask_audio_margin_kd": 2.0,  # margin on post-softknee mask_used_audio
        "mask_onset_kd":        0.8,  # onset distill
    }
    _set_kd_alphas(model, strong)

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

def _compute_gate_left_ctx_samples(sample_rate=REALTIME_SAMPLE_RATE,
                                   threshold_db=REALTIME_GATE_THRESHOLD_DB,
                                   ratio=REALTIME_GATE_RATIO,
                                   attack_ms=REALTIME_GATE_ATTACK_MS,
                                   hold_ms=REALTIME_GATE_HOLD_MS,
                                   release_ms=REALTIME_GATE_RELEASE_MS,
                                   hp_order=REALTIME_HF_HP_ORDER,
                                   env_ms=REALTIME_GATE_ENV_MS,
                                   depth_smooth_ms=REALTIME_GATE_DEPTH_SMOOTH_MS):
    """
    Conservative analytic left-context (samples) for GateInt8Linear pipeline:
      hp (Lhp-1) + env_smooth (Lenv-1) + hold (Lhold-1) + max(Latt-1, Lrel-1)
      + depth_smooth (Ldepth-1)
    where L = ceil(scale * tau) and clamps used inside GateInt8Linear are:
      - Attack/Release scale=4.0, Lmax=2048
      - Env scale=2.0, Lmax=env_lmax (we use small ms → short kernels)
      - Depth scale=2.0, Lmax=depth_lmax
    """
    def ema_len(samples, scale, Lmax):
        L = int(np.ceil(scale * max(1, int(samples))))
        if Lmax is not None:
            L = min(L, int(Lmax))
        return max(1, L)

    # HP
    L_hp = int(hp_order)  # taps count
    lc_hp = L_hp - 1

    # Env smoothing (scale=2, cap ~ small; conservative assume Lmax=512)
    s_env = int(round((float(env_ms) / 1000.0) * float(sample_rate))) if env_ms is not None else 0
    L_env = ema_len(s_env, scale=2.0, Lmax=512) if s_env > 0 else 1
    lc_env = L_env - 1

    # Hold (moving ones FIR)
    s_hold = int(round((float(hold_ms) / 1000.0) * float(sample_rate))) if hold_ms is not None else 0
    L_hold = max(1, s_hold) if s_hold > 0 else 1
    lc_hold = L_hold - 1

    # Attack/Release (scale=4, Lmax=2048)
    s_att = int(round((float(attack_ms) / 1000.0) * float(sample_rate)))
    s_rel = int(round((float(release_ms) / 1000.0) * float(sample_rate)))
    L_att = ema_len(s_att, scale=4.0, Lmax=2048)
    L_rel = ema_len(s_rel, scale=4.0, Lmax=2048)
    lc_open = max(L_att - 1, L_rel - 1)

    # Depth smoothing (scale=2, Lmax ~ 512)
    s_depth = int(round((float(depth_smooth_ms) / 1000.0) * float(sample_rate))) if depth_smooth_ms is not None else 0
    L_depth = ema_len(s_depth, scale=2.0, Lmax=512) if s_depth > 0 else 1
    lc_depth = L_depth - 1

    # Total gate left context
    return lc_hp + lc_env + lc_hold + lc_open + lc_depth

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

def make_mixed_ds_from_arrays_realtime(x48, y48, batch_size, seg_len_rt=REALTIME_SEG_LEN):
    sig_x = tf.TensorSpec((seg_len_rt, 1), tf.float32)
    sig_y = (tf.TensorSpec((seg_len_rt, 1), tf.float32), tf.TensorSpec((), tf.float32))

    ds_pu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len_rt, x48, y48, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    ds_fu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len_rt, x48, y48, is_pick_chunks=False),
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

def compute_resbp_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    """
    Realtime resbp LC (48 kHz) for the ModalBank stem:
      LC_total = LC_gate + (L_pulse - 1) + (L_modal - 1)
    where L_pulse is the length of the onset pulse smoother.
    """
    sr = int(sample_rate)

    LC_gate = _compute_gate_left_ctx_samples(
        sample_rate=sr,
        threshold_db=REALTIME_GATE_THRESHOLD_DB,
        ratio=REALTIME_GATE_RATIO,
        attack_ms=REALTIME_GATE_ATTACK_MS,
        hold_ms=REALTIME_GATE_HOLD_MS,
        release_ms=REALTIME_GATE_RELEASE_MS,
        hp_order=REALTIME_HF_HP_ORDER,
        env_ms=REALTIME_GATE_ENV_MS,
        depth_smooth_ms=REALTIME_GATE_DEPTH_SMOOTH_MS
    )

    # Find the pulse smoother first, or fall back to old onset_env name
    env_layers = _find_layers(model, CausalExpSmoother, name_prefix="realtime_resbp_onset_pulse")
    if not env_layers:
        env_layers = _find_layers(model, CausalExpSmoother, name_prefix="realtime_resbp_onset_env")
    L_env = int(env_layers[0].kernel.shape[0]) if env_layers else 1

    mb_layers = _find_layers(model, ModalBank, name_prefix="realtime_resbp_modalbank")
    if not mb_layers:
        raise ValueError("compute_resbp_left_context_realtime: missing realtime_resbp_modalbank layer.")
    L_modal = int(getattr(mb_layers[0], "length", int(mb_layers[0].kernel.shape[0])))

    LC_total = int(LC_gate) + (L_env - 1) + (L_modal - 1)

    if verbose:
        def as_ms(s): return 1000.0 * float(s) / float(sr)
        print(f"[resbp] gate LC   = {LC_gate} samples ({as_ms(LC_gate):.3f} ms)")
        print(f"[resbp] env  LC   = {L_env - 1} samples ({as_ms(L_env - 1):.3f} ms)")
        print(f"[resbp] modal LC  = {L_modal - 1} samples ({as_ms(L_modal - 1):.3f} ms)")
        print(f"[resbp] TOTAL  LC = {LC_total} samples ({as_ms(LC_total):.3f} ms)")
    return {
        "resbp_signal_left_ctx_samples": int(LC_total),
        "resbp_signal_left_ctx_ms":      1000.0 * float(LC_total) / float(sr),
        "resbp_gain_left_ctx_samples":   int(LC_total),
        "resbp_gain_left_ctx_ms":        1000.0 * float(LC_total) / float(sr),
        "resbp_left_ctx_samples":        int(LC_total),
        "resbp_left_ctx_ms":             1000.0 * float(LC_total) / float(sr),
        "downsample_taps":               [],
        "lowrate_rf_samples":            1,
        "bandpass_taps":                 0,
        "upsample_factor":               1,
    }

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
    if verbose:
        ms = 1000.0 * LC_main / float(sample_rate)
        print(f"[main] LC = {LC_main} samples ({ms:.2f} ms)")
    return LC_main

def compute_hf_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    sr = int(sample_rate)

    # Gate path LC (already conservative)
    LC_gate = _compute_gate_left_ctx_samples(
        sample_rate=sr,
        threshold_db=REALTIME_GATE_THRESHOLD_DB,
        ratio=REALTIME_GATE_RATIO,
        attack_ms=REALTIME_GATE_ATTACK_MS,
        hold_ms=REALTIME_GATE_HOLD_MS,
        release_ms=REALTIME_GATE_RELEASE_MS,
        hp_order=REALTIME_HF_HP_ORDER,
        env_ms=REALTIME_GATE_ENV_MS,
        depth_smooth_ms=REALTIME_GATE_DEPTH_SMOOTH_MS
    )

    # HF signal stack LC from GLU blocks
    RF_hf_stack = _sum_sep_rf_for_prefix(model, r'^realtime_hf_b\d+_sep$')  # 1 + sum((k-1)*d)
    LC_hf_stack = int(RF_hf_stack - 1)

    # HF mask convs LC (now causal)
    RF_mask = _sum_sep_rf_for_prefix(model, r'^realtime_hf_mask_(c1|c2|logits)$')
    LC_mask = int(RF_mask - 1)

    # Mask smoother LC
    L_smooth = 1
    try:
        sm = model.get_layer("realtime_hf_mask_smooth")
        L_smooth = int(sm.kernel.shape[0])
    except Exception:
        pass
    LC_mask_total = LC_mask + (L_smooth - 1)

    # Effective gating LC is max(gate, mask path)
    LC_gate_or_mask = max(int(LC_gate), int(LC_mask_total))

    # HF high-pass LC
    LC_hpf = int(REALTIME_HF_HP_ORDER) - 1

    # Final HF branch LC: HPF comes after gating; product needs max(signal, gate)
    LC_hf = LC_hpf + max(LC_hf_stack, LC_gate_or_mask)

    if verbose:
        def ms(x): return 1000.0 * float(x) / float(sr)
        print(f"[hf] stack={LC_hf_stack} ({ms(LC_hf_stack):.3f} ms), mask={LC_mask_total} ({ms(LC_mask_total):.3f} ms), gate={LC_gate} ({ms(LC_gate):.3f} ms), hpf={LC_hpf} → total={LC_hf} ({ms(LC_hf):.3f} ms)")
    return int(LC_hf)

def compute_realtime_streaming_context(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=True):
    lc_main = compute_main_left_context_realtime(model, sample_rate, verbose=False)
    lc_ir   = compute_ir_left_context_realtime(model, sample_rate, verbose=False)
    rbp     = compute_resbp_left_context_realtime(model, sample_rate, verbose=False)
    lc_rbp  = int(rbp["resbp_left_ctx_samples"])
    lc_hf   = compute_hf_left_context_realtime(model, sample_rate, verbose=False)
    lc_col  = compute_coloreq_left_context_realtime(model, sample_rate, verbose=False)  # NEW

    left_ctx = max(lc_main, lc_ir, lc_rbp, lc_hf, lc_col)
    if verbose:
        def as_ms(s): return 1000.0 * float(s) / float(sample_rate)
        print(f"[streaming] realtime left ctx (samples): main={lc_main}, ir={lc_ir}, resbp={lc_rbp}, hf={lc_hf}, coloreq_ft={lc_col} → max={left_ctx} ({as_ms(left_ctx):.2f} ms)")
    return int(left_ctx), 0

@register_keras_serializable()
def add_coloreq(x, bands, sample_rate, name_prefix, order=65):
    """
    Finetune-only post-EQ: y = x + sum_i g_i * BP_i(x)
    - BP_i are fixed causal min-phase FIR band-pass filters (BandpassFIR).
    - g_i are trainable IrScale scalars, init=0 (identity by default).
    """
    terms = [x]
    for i, (lo, hi) in enumerate(bands):
        bp = BandpassFIR(low_hz=lo, high_hz=hi, order=order, sample_rate=sample_rate,
                         name=f"{name_prefix}_bp_{i}")(x)
        terms.append(IrScale(init=0.0, name=f"{name_prefix}_g_{i}")(bp))
    return Add(name=f"{name_prefix}_sum")(terms)

def compute_coloreq_left_context_realtime(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=False):
    """
    Left context of the finetune adapter in realtime models (if present).
    """
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
        print(f"[coloreq-ft] LC = {LC} samples ({ms:.3f} ms)")
    return int(LC)

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
    """
    Streaming/chunked version: keeps memory flat by processing blocks with
    sufficient left/right context and a small resampling margin, and writes out
    results incrementally.

    - Handles sample-rate mismatch via resample_poly per block.
    - Uses overlap-and-trim so only the "valid" interior (free of edge effects) is written.
    - For offline (96 kHz), also applies HF/resbp demo boosts if requested.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    _boost_scales(model=model, sample_rate=sample_rate, use_finetuned=use_finetuned)

    # Determine FILE sample-rate and region to render in file-frame domain
    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    # Optional normalization gain computed once for the whole region at file SR
    g_norm = 1.0

    # Style flag from filename (same logic as your original function)
    if "pick" in path_src:
        style_true_flag = 1.0
    elif "finger" in path_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    # Model contexts and conversions
    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=model)
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
        x_file = x_block_2d[:, 0] * g_norm
        if not only_prediction:
            y_block_2d, _ = sf.read(path_target, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0] * g_norm

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

@register_keras_serializable()
class Sub(tf.keras.layers.Layer):
    def call(self, inputs):
        a, b = inputs
        return a - b
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

@register_keras_serializable()
class IdentityLayer(tf.keras.layers.Layer):
    """
    Serializable no-op layer to replace anonymous Lambda(lambda z: z).
    """
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def call(self, x):
        return tf.identity(x)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        return super().get_config()

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

# Proper band-limited decimate-by-2 layer for 1-D sequences (applies same FIR per channel)
@register_keras_serializable()
class FIRDecimateBy2(tf.keras.layers.Layer):
    """
    Band-limited decimator-by-2, causal and TFLite-friendly.
    - Uses a single 1-D FIR kernel shared across channels.
    - Implementation: left-pad by (L-1), reshape (B,T,C)->(B*C,T,1),
      tf.nn.conv1d with stride=2, then reshape back.
    - Parameter size: O(L), not O(L*C) or O(L*C*C).
    """
    def __init__(self, num_taps=127, cutoff=0.45, window="blackman", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_taps = int(num_taps)
        self.cutoff = float(cutoff)
        self.window = str(window)

    def build(self, input_shape):
        taps_lin = firwin(self.num_taps, cutoff=self.cutoff, window=self.window, pass_zero=True).astype(np.float32)
        # convert to minimum-phase to reduce group delay
        taps_min = minimum_phase(taps_lin, method="homomorphic").astype(np.float32)
        k = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)  # reversed for conv1d

        self.kernel = self.add_weight(
            name="fir_kernel_shared",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, C)
        L = self.num_taps
        # causal left pad by L-1
        x_pad = tf.pad(x, [[0, 0], [L - 1, 0], [0, 0]])
        b = tf.shape(x_pad)[0]
        t = tf.shape(x_pad)[1]
        c = tf.shape(x_pad)[2]

        # Flatten channels into batch and apply 1-D conv with stride=2
        x_flat = tf.reshape(x_pad, [b * c, t, 1])                  # (B*C, T+L-1, 1)
        y_flat = tf.nn.conv1d(x_flat, filters=self.kernel, stride=2, padding='VALID')  # (B*C, ceil(T/2), 1)

        # Restore channel dimension
        out_t = tf.shape(y_flat)[1]
        y = tf.reshape(y_flat, [b, out_t, c])                      # (B, ceil(T/2), C)
        return y

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        out_t = None if t is None else (t + 1) // 2  # ceil(T/2)
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
class ResizeNearest1D(tf.keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = int(factor)

    def call(self, x):
        # x: (B,T,C) -> (B, T*factor, C)
        # Deterministic nearest upsample via reshape + tile (no tf.image.resize)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        c = tf.shape(x)[2]

        # Insert a singleton axis then tile along it
        x4 = tf.reshape(x, [b, t, 1, c])
        y4 = tf.tile(x4, [1, 1, self.factor, 1])
        y  = tf.reshape(y4, [b, t * self.factor, c])
        return y

    def compute_output_shape(self, s):
        b, t, c = s
        return (b, None if t is None else t * self.factor, c)

    def get_config(self):
        return {'factor': self.factor, **super().get_config()}

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
        self.kernel = self.add_weight(
            name="exp_smooth_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
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
class CausalAsymExpSmoother(tf.keras.layers.Layer):
    """
    Causal asymmetric EMA: fast attack, slow release.
    y = up(x) when rising, y = down(x) when falling, chosen per‑sample by the sign of x[n]-x[n-1].
    """
    def __init__(self, attack_ms, release_ms, sample_rate, up_scale=4.0, down_scale=4.0, Lmax=1024, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.attack_ms   = float(attack_ms)
        self.release_ms  = float(release_ms)
        self.sample_rate = int(sample_rate)
        self.up_scale    = float(up_scale)
        self.down_scale  = float(down_scale)
        self.Lmax        = int(Lmax)

    def build(self, input_shape):
        up_smpl   = max(1, int(round((self.attack_ms  / 1000.0) * self.sample_rate)))
        down_smpl = max(1, int(round((self.release_ms / 1000.0) * self.sample_rate)))

        k_up   = _exp_smoothing_kernel(up_smpl,   scale=self.up_scale,   Lmax=self.Lmax)
        k_down = _exp_smoothing_kernel(down_smpl, scale=self.down_scale, Lmax=self.Lmax)

        self.k_up = self.add_weight(
            name="asym_up_taps", shape=k_up.shape,
            initializer=tf.keras.initializers.Constant(k_up),
            trainable=False
        )
        self.k_down = self.add_weight(
            name="asym_down_taps", shape=k_down.shape,
            initializer=tf.keras.initializers.Constant(k_down),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B,T,1) in [0,1]
        # Rising flag
        x_prev = tf.pad(x, [[0,0],[1,0],[0,0]])[:, :-1, :]
        rising = tf.cast(x > x_prev, x.dtype)

        y_up   = _causal_conv1d(x, self.k_up)
        y_down = _causal_conv1d(x, self.k_down)
        y = rising * y_up + (1.0 - rising) * y_down
        return tf.clip_by_value(y, 0.0, 1.0)

    def get_config(self):
        return {
            "attack_ms": self.attack_ms, "release_ms": self.release_ms,
            "sample_rate": self.sample_rate,
            "up_scale": self.up_scale, "down_scale": self.down_scale, "Lmax": self.Lmax,
            **super().get_config()
        }

@register_keras_serializable()
class GateSoftKnee(tf.keras.layers.Layer):
    """
    Soft‑knee shaping for a mask in [0,1]:
      y = sigmoid(k * (x - t))
    k: slope, t: knee center in [0,1]
    """
    def __init__(self, k=10.0, t=0.35, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = float(k)
        self.t = float(t)

    def call(self, m):
        m = tf.clip_by_value(m, 0.0, 1.0)
        y = tf.math.sigmoid(self.k * (m - self.t))
        return y

    def get_config(self):
        return {"k": self.k, "t": self.t, **super().get_config()}

@register_keras_serializable()
class SoftLimiterCeiling(tf.keras.layers.Layer):
    """
    Unity-slope soft peak limiter: y = C * tanh(x / C)
    - Small-signal region is untouched (slope=1 at 0) so it won't raise the floor.
    - Large signals smoothly saturate to ±C.
    """
    def __init__(self, ceiling=1.0, eps=1e-12, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ceiling = float(ceiling)
        self.eps = float(eps)

    def call(self, x):
        c = tf.constant(self.ceiling, dtype=x.dtype)
        e = tf.constant(self.eps, dtype=x.dtype)
        return c * tf.math.tanh(x / (c + e))

    def get_config(self):
        return {**super().get_config(), "ceiling": self.ceiling, "eps": self.eps}

@register_keras_serializable()
class SoftClip(tf.keras.layers.Layer):
    """
    Very light soft‑clipper: y = tanh(g*x)/tanh(g), with g in [1..4].
    Keeps the HF “spikes” tame when you boost.
    """
    def __init__(self, drive=2.0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.drive = float(drive)
        self._norm = None

    def build(self, input_shape):
        self._norm = np.tanh(self.drive).astype(np.float32)
        super().build(input_shape)

    def call(self, x):
        return tf.math.tanh(self.drive * x) / self._norm

    def get_config(self):
        return {"drive": self.drive, **super().get_config()}


def set_realtime_hf_peak_ceiling(model, ceiling):
    """
    Override the 'realtime_hf_peak_limiter' ceiling at inference.
    The layer stores 'ceiling' as a Python attribute (no weights).
    """
    try:
        layer = model.get_layer("realtime_hf_peak_limiter")
    except Exception:
        print("[hf-ceiling] Peak limiter layer not found; model may have been built without it.")
        return False
    try:
        layer.ceiling = float(ceiling)
        print(f"[hf-ceiling] Set peak ceiling to {float(ceiling):.6f}")
        return True
    except Exception as e:
        print(f"[hf-ceiling] Failed to set ceiling: {e}")
        return False

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
class HFSuppressByMask(tf.keras.layers.Layer, AdjustableAlphaMixin):
    """
    Training-only leak penalty using the STUDENT mask:
      loss += alpha * mean( |hf| * (1 - stop_grad(mask_audio)) )
    """
    def __init__(self, alpha=0.4, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(alpha)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs, training=None):
        hf, mask_audio = inputs
        if training is not False:
            w = tf.stop_gradient(1.0 - tf.clip_by_value(mask_audio, 0.0, 1.0))
            leak = tf.abs(hf) * w
            self.add_loss(self.alpha_var * tf.reduce_mean(leak))
        return hf

    def get_config(self):
        return {"alpha": float(self.init_alpha), **super().get_config()}
    
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
class DistillBCE(tf.keras.layers.Layer, AdjustableAlphaMixin):
    def __init__(self, alpha, eps=1e-6, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(alpha)
        self.eps = float(eps)
    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)
    def call(self, inputs, training=None):
        y_t, y_s = inputs  # (B,T,1) in [0,1]
        if training is False:
            return y_s
        t = tf.minimum(tf.shape(y_t)[1], tf.shape(y_s)[1])
        yt = tf.clip_by_value(y_t[:, :t, :], self.eps, 1.0 - self.eps)
        ys = tf.clip_by_value(y_s[:, :t, :], self.eps, 1.0 - self.eps)
        bce = -(yt * tf.math.log(ys) + (1.0 - yt) * tf.math.log(1.0 - ys))
        self.add_loss(self.alpha_var * tf.reduce_mean(bce))
        return y_s
    def get_config(self):
        return {"alpha": float(self.init_alpha), "eps": self.eps, **super().get_config()}

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

# Replace your current call(…, training=None) bodies

@register_keras_serializable()
class HFSuppressWhenClosed(tf.keras.layers.Layer, AdjustableAlphaMixin):
    def __init__(self, alpha=0.7, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(alpha)
    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)
    def call(self, inputs, training=None):
        hf, gate = inputs  # (B,T,1), (B,Tg,1)
        # IMPORTANT: be active when training is True OR None
        if training is not False:
            t = tf.minimum(tf.shape(hf)[1], tf.shape(gate)[1])
            leak = tf.abs(hf[:, :t, :] * tf.stop_gradient(1.0 - gate[:, :t, :]))
            self.add_loss(self.alpha_var * tf.reduce_mean(leak))
        return hf
    def get_config(self):
        return {"alpha": float(self.init_alpha), **super().get_config()}

# If you added the mask penalty, change it the same way:
@register_keras_serializable()
class MaskSuppressWhenClosed(tf.keras.layers.Layer, AdjustableAlphaMixin):
    def __init__(self, alpha=0.3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(alpha)
    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)
    def call(self, inputs, training=None):
        m, gate = inputs
        if training is not False:
            t = tf.minimum(tf.shape(m)[1], tf.shape(gate)[1])
            leak = tf.abs(m[:, :t, :] * tf.stop_gradient(1.0 - gate[:, :t, :]))
            self.add_loss(self.alpha_var * tf.reduce_mean(leak))
        return m
    def get_config(self):
        return {"alpha": float(self.init_alpha), **super().get_config()}
    
@register_keras_serializable()
class MaskMarginKD(tf.keras.layers.Layer, AdjustableAlphaMixin):
    """
    Margin penalty on the mask against a [0,1] teacher gate:
      - When gate is closed (weight ~ 1-g), push mask below m_close.
      - When gate is open   (weight ~ g), push mask above m_open.
    """
    def __init__(self, alpha=1.0, g_lo=0.15, g_hi=0.85, m_close=0.03, m_open=0.85, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(alpha)
        self.g_lo = float(g_lo); self.g_hi = float(g_hi)
        self.m_close = float(m_close); self.m_open = float(m_open)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs, training=None):
        m, g = inputs  # (B,T,1)
        if training is False:
            return m
        t = tf.minimum(tf.shape(m)[1], tf.shape(g)[1])
        m = m[:, :t, :]; g = g[:, :t, :]

        # Closed weight ~ 1 - g (smooth)
        w_close = tf.clip_by_value(1.0 - g, 0.0, 1.0)
        # Open weight ~ g (smooth)
        w_open  = tf.clip_by_value(g, 0.0, 1.0)

        # Hinge terms
        close_term = tf.nn.relu(m - self.m_close) * w_close
        open_term  = tf.nn.relu(self.m_open - m) * w_open

        loss = tf.reduce_mean(close_term + open_term)
        self.add_loss(self.alpha_var * loss)
        return m

    def get_config(self):
        return {
            "alpha": float(self.init_alpha),
            "g_lo": self.g_lo, "g_hi": self.g_hi,
            "m_close": self.m_close, "m_open": self.m_open,
            **super().get_config()
        }

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
        if training is False:
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

@register_keras_serializable()
class ModalBank(tf.keras.layers.Layer):
    """
    Fixed bank of causal decaying-sine FIRs (one output channel per mode).
    - Normal input: (B,T,1)
    - Output: (B,T,M)
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
        # Ensure rank >= 3 by collapsing all dims from axis=2 onward into the channel dim.
        # This tolerates legacy rank-4 inputs like (B,T,1,1).
        shp   = tf.shape(x)
        b     = shp[0]
        t     = shp[1]
        # new_c = product of all trailing dims (must be 1 for our fixed 1-in channel kernels)
        new_c = tf.reduce_prod(shp[2:])
        # Enforce single-channel at runtime (keeps conv1d filter compatibility: (L,1,M))
        assert_op = tf.debugging.assert_equal(
            new_c, 1,
            message="ModalBank expects a single input channel; got trailing dims product != 1."
        )
        with tf.control_dependencies([assert_op]):
            x3 = tf.reshape(x, [b, t, new_c])  # becomes (B,T,1)
        pad = self.length - 1
        x_pad = tf.pad(x3, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        # Works for (B,T,1) and legacy (B,T,1,1) since only B,T matter.
        b = input_shape[0]
        t = input_shape[1]
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
            "min_gain_db": 20.0 * np.log10(max(self.min_gain, 1e-12))
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

def _add_resbp_branch_offline(inp, style_embed, style_time_used, gate):
    """
    Offline (96 kHz) resbp with dynamic per-mode weights driven by the input band (≈0.7–2.2 kHz).
    Keeps modal kernels fixed and TFLite‑friendly; only the per‑mode gains vary over time.
    """
    sr = OFFLINE_SAMPLE_RATE
    M  = len(RESBP_MODAL_CENTERS_HZ_OFFLINE)

    # Pick gate (0..1) and onset detection
    g_pick   = Multiply(name="offline_resbp_pick_gate")([gate, style_time_used])
    onset    = Lambda(gate_onset, name="offline_resbp_onset")(g_pick)

    strike_env = CausalExpSmoother(
        time_const_ms=RESBP_STRIKE_ENV_MS_OFFLINE,
        sample_rate=sr,
        Lmax=4096,
        name="offline_resbp_onset_env"
    )(g_pick)
    strike = Multiply(name="offline_resbp_strike")([onset, strike_env])  # (B,T,1)

    # Fixed modal bank (LTI), as before
    modes = ModalBank(
        centers_hz=RESBP_MODAL_CENTERS_HZ_OFFLINE,
        decays_ms=RESBP_MODAL_DECAYS_MS_OFFLINE,
        length=RESBP_MODAL_LEN_OFFLINE,
        sample_rate=sr,
        name="offline_resbp_modalbank"
    )(strike)  # (B,T,M)

    # Analysis front‑end around the same band (causal, fixed), then envelope
    ana = BandpassFIR(
        low_hz=0.9 * min(RESBP_MODAL_CENTERS_HZ_OFFLINE),
        high_hz=1.1 * max(RESBP_MODAL_CENTERS_HZ_OFFLINE),
        order=OFFLINE_RESBP_BP_ORDER,
        sample_rate=sr,
        name="offline_resbp_ctrl_bp"
    )(inp)
    ana_env = CausalExpSmoother(
        time_const_ms=4.0,  # short envelope for fast per-onset adaptation
        sample_rate=sr,
        Lmax=1024,
        name="offline_resbp_ctrl_env"
    )(Abs1D()(ana))  # (B,T,1)

    # Tiny control net → per‑mode weights in [0,1], time‑varying
    # Keep this tiny to stay embedded‑friendly if you later mirror to realtime.
    ctrl = SeparableConv1D(16, 9, dilation_rate=1, padding="same", activation="relu", name="offline_resbp_ctrl_c1")(ana_env)
    ctrl = SeparableConv1D(16, 9, dilation_rate=2, padding="same", activation="relu", name="offline_resbp_ctrl_c2")(ctrl)
    ctrl_logits = SeparableConv1D(M, 3, dilation_rate=1, padding="same", name="offline_resbp_ctrl_logits")(ctrl)
    ctrl_w = Activation("sigmoid", name="offline_resbp_ctrl_w")(ctrl_logits)  # (B,T,M) ∈ (0,1)

    # Ensure strictly pick‑only contribution
    ctrl_w = Multiply(name="offline_resbp_ctrl_pick_only")([ctrl_w, style_time_used])

    # Apply per‑mode weights to modal outputs
    modes_mod = Multiply(name="offline_resbp_modes_mod")([modes, ctrl_w])  # (B,T,M)

    # Trainable 1x1 mix to mono
    mixed = Conv1D(1, 1, use_bias=False, padding="same", name="offline_resbp_mix")(modes_mod)

    # Safety HP
    mixed = HFHighpass(sample_rate=sr, cutoff_hz=0.9 * min(RESBP_MODAL_CENTERS_HZ_OFFLINE), order=65, name="offline_resbp_post_hp")(mixed)

    # Still pick‑only at the sum
    mixed = Multiply(name="offline_resbp_pick_only")([mixed, style_time_used])

    return IrScale(init=OFFLINE_RESBP_SCALE_INIT, name="offline_resbp_scale")(mixed)

def _add_resbp_branch_realtime(inp, gate, style_time_used, bias_constraint=None):
    """
    Realtime (48 kHz) resbp: synthetic resonances via ModalBank, excited by pick onsets.
    Changes:
      - strikes are formed by convolving the onset with a short causal smoother (a pulse),
        instead of multiplying an onset delta by a slow env (which injects only 1 sample).
      - optional causal band-pass analysis in the same band produces a gain that scales
        the strike amplitude with pick intensity.
    """
    sr = REALTIME_SAMPLE_RATE

    # Pick-only gate
    g_pick = Multiply(name="realtime_resbp_pick_gate")([gate, style_time_used])

    # Onset detection (rising edges of the pick gate)
    onset = Lambda(gate_onset, name="realtime_resbp_onset")(g_pick)

    # Form a short causal pulse from the onset (this injects energy over a few ms)
    strike_pulse = CausalExpSmoother(
        time_const_ms=RESBP_STRIKE_ENV_MS_REALTIME,
        sample_rate=sr,
        Lmax=512,
        name="realtime_resbp_onset_pulse"
    )(onset)  # (B,T,1)

    # Optional: scale pulse by a light causal BP envelope in the same band
    ana = BandpassFIR(
        low_hz=0.9 * min(RESBP_MODAL_CENTERS_HZ_REALTIME),
        high_hz=1.1 * max(RESBP_MODAL_CENTERS_HZ_REALTIME),
        order=REALTIME_RESBP_BP_ORDER,
        sample_rate=sr,
        name="realtime_resbp_ctrl_bp"
    )(inp)  # (B,T,1)

    ana_env = CausalExpSmoother(
        time_const_ms=3.0,   # very short env for 'strike loudness'
        sample_rate=sr,
        Lmax=256,
        name="realtime_resbp_ctrl_env"
    )(Abs1D()(ana))  # (B,T,1)

    # Simple, static compression to 0..1 (TFLite-friendly)
    ctrl_gain = Lambda(clip_01x4, name="realtime_resbp_ctrl_gain")(ana_env)

    # Final strike: pick-only pulse, scaled by the BP-derived gain
    strike = Multiply(name="realtime_resbp_strike")([strike_pulse, ctrl_gain])  # (B,T,1)

    # ModalBank: fixed modes; longer kernel to allow audible ringing
    modes = ModalBank(
        centers_hz=RESBP_MODAL_CENTERS_HZ_REALTIME,
        decays_ms=RESBP_MODAL_DECAYS_MS_REALTIME,
        length=RESBP_MODAL_LEN_REALTIME,
        sample_rate=sr,
        name="realtime_resbp_modalbank"
    )(strike)  # (B,T,M)

    # Trainable 1x1 mix to mono
    mixed = Conv1D(1, 1, use_bias=True, padding="same", bias_constraint=bias_constraint, name="realtime_resbp_mix")(modes)

    # Safety HP to remove LF residue
    mixed = HFHighpass(sample_rate=sr, cutoff_hz=0.9 * min(RESBP_MODAL_CENTERS_HZ_REALTIME), order=65, name="realtime_resbp_post_hp")(mixed)

    # Ensure strictly pick-only contribution
    mixed = Multiply(name="realtime_resbp_pick_only")([mixed, style_time_used])

    return IrScale(init=REALTIME_RESBP_SCALE_INIT, name="realtime_resbp_scale")(mixed)

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
def build_offline_student_for_finetune(teacher, dynamic_time=False, name="piezo2mic_offline_ft_train"):
    """
    Offline finetune student (96k) with learned HF mask, distilled from a teacher with a fixed gate.
    - Student is build_offline_model(cap_mask_with_gate=False) so the learned mask actually gates HF/resbp.
    - KD losses:
        * teacher fixed gate → student mask_pred (strong)
        * teacher fixed gate → student mask_used (small)
        * teacher HF envelope (0.08ms EMA) → student mask_pred (aux)
        * suppress student HF when teacher gate is closed
    Only the mask subnetwork and IrScale scalars are meant to be trainable.
    """
    # 1) Build the student (learned mask controls gating)
    student = build_offline_model(
        dynamic_time=dynamic_time,
        cap_mask_with_gate=False,
        mask_min_floor=0.05,
        name="piezo2mic_offline_ft_student"
    )
    y_audio_s, y_style_s = student.outputs

    # 2) Build a teacher combo to fetch the fixed gate and HF stem
    #    (we re-use the student's inputs so both see the same style gating and level)
    gate_t = teacher.get_layer("offline_gate").output
    hf_t   = teacher.get_layer("offline_hf_scale").output  # scaled HF stem (pre-pick-only)

    teacher_combo = tf.keras.Model(
        inputs=teacher.inputs,
        outputs=[gate_t, hf_t],
        name="offline_teacher_combo_ft"
    )
    gate_teacher, hf_teacher = teacher_combo(student.inputs)  # feed the same (x, style_true, style_force)

    # 3) Student internal taps for KD/penalties
    mask_pred_s = student.get_layer("offline_hf_mask").output
    mask_used_s = student.get_layer("offline_hf_mask_used").output
    hf_final_s  = student.get_layer("offline_hf_scale").output

    # 4) Teacher HF envelope → normalize to [0,1] via clip_01x4 (like in realtime)
    hf_env_96   = CausalExpSmoother(
        time_const_ms=0.08, sample_rate=OFFLINE_SAMPLE_RATE, Lmax=64,
        name="off_ft_hf_env96")(Abs1D()(hf_teacher))
    hf_env_96_n = Lambda(clip_01x4, name="off_ft_hf_env96_norm")(hf_env_96)

    # 5) KD + penalties
    kd_tensors = []

    # Gate → student mask (best-shift)
    kd_tensors.append(
        DistillLossBestShift(alpha=OFFLINE_HF_MASK_DISTILL_ALPHA, max_shift=12, name="off_hf_mask_distill")(
            [gate_teacher, mask_pred_s]
        )
    )
    kd_tensors.append(
        DistillLossBestShift(alpha=OFFLINE_MASK_USED_DISTILL_ALPHA, max_shift=12, name="off_mask_used_distill")(
            [gate_teacher, mask_used_s]
        )
    )
    # HF env → student mask (auxiliary; lighter)
    kd_tensors.append(
        DistillLoss(alpha=0.5 * OFFLINE_HF_MASK_DISTILL_ALPHA, name="off_hf_env_mask_distill")(
            [hf_env_96_n, mask_pred_s]
        )
    )
    # Suppress student HF when the teacher gate is closed
    kd_tensors.append(
        HFSuppressWhenClosed(alpha=OFFLINE_HF_SUPPRESS_ALPHA, name="off_hf_suppress_closed")(
            [hf_final_s, gate_teacher]
        )
    )

    # 6) Keep the same outputs, but anchor KD tensors into the graph
    y_audio_s = LossTether(name="offline_kd_tether")([y_audio_s] + kd_tensors)
    return tf.keras.Model(inputs=student.inputs, outputs=[y_audio_s, y_style_s], name=name)

@register_keras_serializable()
def build_offline_model(dynamic_time=False,
                        lookahead_samples=OFFLINE_LOOKAHEAD_SAMPLES,
                        name="piezo2mic_offline",
                        cap_mask_with_gate=True,
                        hf_time_shift_samples=OFFLINE_HF_TIME_SHIFT_SAMPLES,
                        mask_min_floor=0.0):
    """
    Offline (96 kHz) model.
    New: optional learned pick mask for HF/resonance branches.
      - cap_mask_with_gate=True  → behave like baseline (mask is safety-capped by the fixed gate)
      - cap_mask_with_gate=False → no fixed gate in signal path; mask fully controls HF/resbp
    This lets the finetuned offline model drop the fixed threshold dependence.
    """
    T   = None if dynamic_time else OFFLINE_SEG_LEN
    inp = Input(shape=(T,1), name="offline_piezo_input")
        
    # Known true style (may be unknown/forced off in training)
    style_true_in  = Input(shape=(1,), name="offline_style_true_input")
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
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_pre_{i}",
                        gamma_scale=OFFLINE_FILM_GAMMA_SCALE,
                        beta_scale=OFFLINE_FILM_BETA_SCALE,
                        gamma_mag=OFFLINE_FILM_GAMMA_MAG,
                        beta_mag=OFFLINE_FILM_BETA_MAG
                    )

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
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_dil_{idx}",
                        gamma_scale=OFFLINE_FILM_GAMMA_SCALE,
                        beta_scale=OFFLINE_FILM_BETA_SCALE,
                        gamma_mag=OFFLINE_FILM_GAMMA_MAG,
                        beta_mag=OFFLINE_FILM_BETA_MAG
                    )
        
    # Final pick/finger outputs
    main_pick   = Conv1D(1,1,padding="same",name="offline_final_pick")(x)
    main_finger = Conv1D(1,1,padding="same",name="offline_final_finger")(x)
 
    inv_style  = InvertStyle(name="offline_invert_style")(style_time_used)
    pick_c     = Multiply(name="offline_pick_contrib")([style_time_used, main_pick])
    finger_c   = Multiply(name="offline_finger_contrib")([inv_style,     main_finger])

    # Optional fixed gate (safety cap); present in baseline, removed for finetune if cap_mask_with_gate=False
    offline_gate = None
    if cap_mask_with_gate:
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

    # HF branch
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(inp)
    hf    = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_hf_init")(hf_in)
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b0", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b0")
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=5, dilation=1, prefix="offline_hf_b1", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b1")
    hf    = gated_sep_block(hf, OFFLINE_FILTERS, kernel=3, dilation=1, prefix="offline_hf_b2", lookahead=lookahead_samples)
    hf    = film_apply(hf, style_embed, OFFLINE_FILTERS, prefix="offline_hf_b2")
    hf    = Conv1D(1,1,padding="same", name="offline_hf_out")(hf)

    m = SeparableConv1D(8, 5, dilation_rate=1, padding="same", activation="relu", name="offline_hf_mask_c1")(hf_in)
    m = SeparableConv1D(8, 5, dilation_rate=2, padding="same", activation="relu", name="offline_hf_mask_c2")(m)
    m = SeparableConv1D(1, 3, dilation_rate=1, padding="same", name="offline_hf_mask_logits")(m)
    mask_pred = Activation("sigmoid", name="offline_hf_mask")(m)

    if float(mask_min_floor) > 0.0:
        mask_pred = MaskFloor(eps=float(mask_min_floor), name="offline_hf_mask_floor")(mask_pred)

    mask_smoothed = CausalExpSmoother(
        time_const_ms=0.08, sample_rate=OFFLINE_SAMPLE_RATE, Lmax=64, name="offline_hf_mask_smooth"
    )(mask_pred)
        
    if offline_gate is None:
        mask_used = IdentityLayer(name="offline_hf_mask_used")(mask_smoothed)
    else:
        mask_used = Minimum(name="offline_hf_mask_used")([mask_smoothed, offline_gate])

    # Apply mask BEFORE the HPF (same place the fixed gate used to be)
    hf_gated = Multiply(name="offline_hf_mult_gate")([hf, mask_used])
    hf_hp    = HFHighpass(sample_rate=OFFLINE_SAMPLE_RATE,
                          cutoff_hz=OFFLINE_HP_CUTOFF_HZ,
                          order=OFFLINE_HF_HP_ORDER,
                          name="offline_hf_highpass")(hf_gated)
    hf_final = IrScale(init=8.0, name="offline_hf_scale")(hf_hp)
    
    if int(hf_time_shift_samples) != 0:
        hf_final = TimeShift1D(shift=int(hf_time_shift_samples), name="offline_hf_timeshift")(hf_final)

    hf_final_pick = Multiply(name="offline_hf_pick_only")([hf_final, style_time_used])
    
    main_core = Add(name="offline_main_core")([pick_c, finger_c])
    
    # Resonance band‑pass stem (use the fixed gate if present, else the learned mask)
    gate_for_resbp = mask_used if (offline_gate is None) else offline_gate
    offline_resbp = _add_resbp_branch_offline(
        inp=inp,
        style_embed=style_embed,
        style_time_used=style_time_used,
        gate=gate_for_resbp
    )

    # Final sum (main + HF + band-pass)
    out = Add(name="offline_mix_main")([main_core, hf_final_pick, offline_resbp])

    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in], outputs=[out, style_prob], name=name)

@register_keras_serializable()
def build_realtime_model(offline=None,
                         dynamic_time=False,
                         cap_mask_with_gate=True,
                         use_learned_hf_mask=True,
                         enable_mask_kd=True,
                         detach_mask_from_audio=True,
                         mask_min_floor=0.0,
                         hf_time_shift_samples=REALTIME_HF_TIME_SHIFT_SAMPLES,
                         # learned-pick shaping
                         mask_release_ms=REALTIME_MASK_RELEASE_MS,
                         mask_softknee=(REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
                         hf_peak_ceiling=None):
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None, 1), name="realtime_offline_piezo_input")

    # Style IO
    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")

    # Style head (student)
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    kd_tensors = []
    style_pred_source = style_prob
    BC = ZeroBias()

    # Teacher bundle (optional)
    T_feats = []
    T_gate = None
    offline_audio = None
    offline_style_prob = None
    off_pick = None
    off_finger = None
    off_hf_scaled = None
    off_resbp_scaled = None

    if offline is not None:
        teacher = offline
        teacher.trainable = False

        feat_names = REALTIME_OFFLINE_FEAT_LAYERS or []
        feat_outs  = [teacher.get_layer(n).output for n in feat_names]

        gate_out = teacher.get_layer("offline_gate").output
        audio_out, style_prob_out = teacher.outputs
        pick_out   = teacher.get_layer("offline_final_pick").output
        finger_out = teacher.get_layer("offline_final_finger").output
        hf_scaled  = teacher.get_layer("offline_hf_scale").output
        resbp_scaled = teacher.get_layer("offline_resbp_scale").output

        teacher_combo = tf.keras.Model(
            inputs=teacher.inputs,
            outputs=feat_outs + [gate_out, audio_out, style_prob_out, pick_out, finger_out, hf_scaled, resbp_scaled],
            name="offline_teacher_combo"
        )
        combo = teacher_combo([offline_inp, style_true_in, style_force_in])

        idx = 0
        if feat_names:
            T_feats = combo[idx: idx + len(feat_names)]; idx += len(feat_names)
        T_gate             = combo[idx]; idx += 1
        offline_audio      = combo[idx]; idx += 1
        offline_style_prob = combo[idx]; idx += 1
        off_pick           = combo[idx]; idx += 1
        off_finger         = combo[idx]; idx += 1
        off_hf_scaled      = combo[idx]; idx += 1
        off_resbp_scaled   = combo[idx]

        # Style KD and blend
        style_prob_3d         = Reshape((1, 1), name="realtime_style_prob_3d")(style_prob)
        offline_style_prob_3d = Reshape((1, 1), name="offline_style_prob_3d")(offline_style_prob)
        kd_tensors.append(
            DistillLoss(alpha=REALTIME_STYLE_DISTILL_ALPHA, name="style_prob_distill")(
                [offline_style_prob_3d, style_prob_3d]
            )
        )
        style_pred_source = BlendScalar(
            init_alpha=REALTIME_STYLE_BLEND_INIT,
            trainable=REALTIME_STYLE_BLEND_TRAINABLE,
            name="realtime_style_blend"
        )([style_prob, offline_style_prob])

    # Style selection
    one_minus_force = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_pred_source]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    style_scalar_b3 = Reshape((1, 1), name="realtime_style_scalar_broadcast")(style_used_scalar)
    style_embed = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_style_embed")(style_used_scalar)

    # Main stack
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(inp)
    for i in range(REALTIME_PRE_BLOCKS):
        x = glu_sep_block(x, REALTIME_FILTERS, REALTIME_PRE_KERNEL, 1,
                          prefix=f"realtime_pre_{i}", lookahead=0,
                          residual_scale=0.35, bias_constraint=BC)
        x = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}",
                       gamma_scale=REALTIME_FILM_GAMMA_SCALE,
                       beta_scale=REALTIME_FILM_BETA_SCALE,
                       gamma_mag=REALTIME_FILM_GAMMA_MAG,
                       beta_mag=REALTIME_FILM_BETA_MAG)

    realtime_feats = []
    for i, d in enumerate(REALTIME_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=REALTIME_KERNEL_SIZE, dilation=d, lookahead_samples=0)
        x = ZeroPadding1D((pad_left, pad_right), name=f"realtime_pad_{i}")(x)
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid",
                            bias_constraint=BC, name=f"realtime_sep_{i}")(x)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        x_block = Multiply(name=f"realtime_gate_mul_{i}")([c, Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        x_block = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pw_{i}")(x_block)
        x_block = ConstScale(0.35, name=f"realtime_res_scale_{i}")(x_block)
        x = Add(name=f"realtime_res_add_{i}")([x_block, res])

        x = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}",
                       gamma_scale=REALTIME_FILM_GAMMA_SCALE,
                       beta_scale=REALTIME_FILM_BETA_SCALE,
                       gamma_mag=REALTIME_FILM_GAMMA_MAG,
                       beta_mag=REALTIME_FILM_BETA_MAG)

        offline_feat_name = f"offline_dil_{i}_film_apply"
        if REALTIME_OFFLINE_FEAT_LAYERS and offline_feat_name in REALTIME_OFFLINE_FEAT_LAYERS:
            realtime_feats.append(x)

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_realtime = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch
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
            bias_constraint=BC,
            name=f"realtime_ir_sep_{i}"
        )(ir)
    ir = IrScale(init=0.6, name="realtime_ir_scale")(ir)
    ir = FIRInterpByN(
        factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
        num_taps=max(193, REALTIME_LP_ORDER),
        cutoff=0.45/REALTIME_IR_DOWNSAMPLE_FACTOR,
        window=('kaiser', 9.0),
        name="realtime_ir_interp"
    )(ir)
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir, inp])

    # HF mask path
    hf_in  = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(inp)
    m = causal_sep_conv(hf_in, 8, 5, 1, name="realtime_hf_mask_c1", activation="relu", bias_constraint=BC)
    m = causal_sep_conv(m,    8, 5, 2, name="realtime_hf_mask_c2", activation="relu", bias_constraint=BC)
    
    m = causal_sep_conv(m, 1, 3, 1, name="realtime_hf_mask_logits",
                    bias_constraint=BC,
                    bias_initializer=tf.keras.initializers.Constant(REALTIME_MASK_LOGIT_BIAS))
    
    mask_pred = Activation("sigmoid", name="realtime_hf_mask")(m)
    if float(mask_min_floor) > 0.0:
        mask_pred = MaskFloor(eps=float(mask_min_floor), name="realtime_hf_mask_floor")(mask_pred)

    # Asymmetric smoothing: fast up (0.08 ms), slower down (mask_release_ms)
    down_tc = float(mask_release_ms) if (mask_release_ms is not None and mask_release_ms > 0.0) else 0.25
    mask_smooth_down = CausalExpSmoother(time_const_ms=down_tc, sample_rate=REALTIME_SAMPLE_RATE, Lmax=256, name="realtime_hf_mask_smooth")(mask_pred)
    mask_smooth_up   = CausalExpSmoother(time_const_ms=0.08,    sample_rate=REALTIME_SAMPLE_RATE, Lmax=128, name="realtime_hf_mask_smooth_up")(mask_pred)

    mask_prev = TimeShift1D(shift=1, name="realtime_hf_mask_prev")(mask_pred)
    mask_diff = Sub(name="realtime_hf_mask_diff")([mask_pred, mask_prev])
    rising_soft = Activation("relu", name="realtime_hf_mask_rising_relu")(mask_diff)
    rising = Lambda(clip_01x4, name="realtime_hf_mask_rising")(rising_soft)
    not_rising = Lambda(invert, name="realtime_hf_mask_not_rising")(rising)

    mask_asym = Add(name="realtime_hf_mask_asym_combine")([
        Multiply(name="realtime_hf_mask_asym_up")([rising,      mask_smooth_up]),
        Multiply(name="realtime_hf_mask_asym_down")([not_rising, mask_smooth_down])
    ])
    mask_smoothed = IdentityLayer(name="realtime_hf_mask_asym")(mask_asym)

    # Optional fixed gate cap during training/static builds
    realtime_gate = None
    if cap_mask_with_gate:
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
            min_gain_db=-60.0,
            name="realtime_gate"
        )(inp)

    if use_learned_hf_mask:
        mask_used_inner = Minimum(name="realtime_hf_mask_used_min")([mask_smoothed, realtime_gate]) if realtime_gate is not None else IdentityLayer(name="realtime_hf_mask_used_id")(mask_smoothed)
    else:
        if realtime_gate is not None:
            mask_used_inner = IdentityLayer(name="realtime_hf_mask_used_gate")(realtime_gate)
        else:
            ones = OnesLike(name="realtime_hf_mask_used_ones")(mask_smoothed)
            mask_used_inner = IdentityLayer(name="realtime_hf_mask_used_gate")(ones)

    mask_remap = MaskRemapLinear(
        close_t=REALTIME_MASK_CLOSE_T,
        open_t=REALTIME_MASK_OPEN_T,
        name="realtime_hf_mask_remap"
    )(mask_used_inner)

    # Keep the original debug name pointing to the remapped mask (pre-softknee)
    mask_used = IdentityLayer(name="realtime_hf_mask_used")(mask_remap)

    # Soft‑knee for audio
    if mask_softknee is not None:
        k_val, t_val = (float(mask_softknee[0]), float(mask_softknee[1])) if isinstance(mask_softknee, (tuple, list)) else (float(mask_softknee.get("k", 10.0)), float(mask_softknee.get("t", 0.55)))
        ones_knee = OnesLike(name="realtime_hf_mask_softknee_ones")(mask_used)
        neg_t = ConstScale(-t_val, name="realtime_hf_mask_softknee_neg_t")(ones_knee)
        shifted = Add(name="realtime_hf_mask_softknee_shift")([mask_used, neg_t])
        scaled  = ConstScale(k_val, name="realtime_hf_mask_softknee_scale")(shifted)
        mask_used_audio = Activation("sigmoid", name="realtime_hf_mask_softknee")(scaled)
    else:
        mask_used_audio = IdentityLayer(name="realtime_hf_mask_passthru")(mask_used)

    # Gentle sharpening (x^power) to make mid values close faster
    if REALTIME_MASK_SHARPEN_P and REALTIME_MASK_SHARPEN_P >= 2:
        mask_used_audio = MaskSharpen(power=int(REALTIME_MASK_SHARPEN_P), name="realtime_hf_mask_sharpen")(mask_used_audio)

    if float(mask_min_floor) > 0.0:
        mask_used_audio = MaskFloor(eps=float(mask_min_floor), name="realtime_hf_mask_audio_floor")(mask_used_audio)

    mask_for_mult = Lambda(stop_grad, name="realtime_hf_mask_stopgrad")(mask_used_audio) if detach_mask_from_audio else mask_used_audio

    # HF branch proper
    hf = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name="realtime_hf_init")(hf_in)
    hf = glu_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b0", lookahead=0, residual_scale=0.35, bias_constraint=BC)
    hf = glu_sep_block(hf, REALTIME_FILTERS, kernel=5, dilation=1, prefix="realtime_hf_b1", lookahead=0, residual_scale=0.35, bias_constraint=BC)
    hf = glu_sep_block(hf, REALTIME_FILTERS, kernel=3, dilation=1, prefix="realtime_hf_b2", lookahead=0, residual_scale=0.35, bias_constraint=BC)
    hf = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_hf_out")(hf)

    hf_gated = Multiply(name="realtime_hf_mult_gate")([hf, mask_for_mult])

    hf_hp    = HFHighpass(sample_rate=REALTIME_SAMPLE_RATE, cutoff_hz=REALTIME_HP_CUTOFF_HZ, order=REALTIME_HF_HP_ORDER, name="realtime_hf_highpass")(hf_gated)
    hf_final = IrScale(init=8.0, name="realtime_hf_scale")(hf_hp)

        # Optional HF peak limiter (preferred over softclip): unity slope at 0, ceiling=C
    if hf_peak_ceiling is not None:
        hf_final = SoftLimiterCeiling(ceiling=float(hf_peak_ceiling),
                                      name="realtime_hf_peak_limiter")(hf_final)

    hf_final = HFLowbandPenalty(alpha=REALTIME_HF_LOWBAND_PENALTY_ALPHA,
                                cutoff_hz=REALTIME_HF_LOWBAND_PENALTY_CUTOFF_HZ,
                                order=129,
                                sample_rate=REALTIME_SAMPLE_RATE,
                                name="realtime_hf_lowband_penalty")(hf_final)
    if int(hf_time_shift_samples) != 0:
        hf_final = TimeShift1D(shift=int(hf_time_shift_samples), name="realtime_hf_timeshift")(hf_final)
    hf_final_pick = Multiply(name="realtime_hf_pick_only")([hf_final, style_scalar_b3])

    # Mix
    y_realtime = Add(name="realtime_with_ir")([y_realtime, ir, hf_final_pick])

    # Resonance stem
    gate_for_resbp = mask_used if (realtime_gate is None) else realtime_gate
    realtime_resbp = _add_resbp_branch_realtime(inp=inp, gate=gate_for_resbp, style_time_used=style_scalar_b3, bias_constraint=BC)
    y_realtime = Add(name="realtime_with_ir_resbp")([y_realtime, realtime_resbp])

    # KD wiring
    if offline is not None:
        T_feats_ds = [FIRDecimateBy2(name=f"offline_feat_ds2_{i}")(t) for i, t in enumerate(T_feats)]
        T_gate_ds  = FIRDecimateBy2(name="offline_gate_ds2")(T_gate)

        kd_tensors.append(HFSuppressWhenClosed(alpha=0.7, name="hf_suppress_closed")([hf_final, T_gate_ds]))
        kd_tensors.append(MaskSuppressWhenClosed(alpha=0.5, name="mask_suppress_closed")([mask_smoothed, T_gate_ds]))
        kd_tensors.append(
            MaskMarginKD(alpha=1.5, m_close=0.01, m_open=0.90, name="mask_margin_kd")([mask_smoothed, T_gate_ds])
        )
        
        # Extra direct supervision on the mask that actually gates audio
        kd_tensors.append(
            MaskMarginKD(alpha=2.0, m_close=0.02, m_open=0.92, name="mask_audio_margin_kd")(
                [mask_used_audio, T_gate_ds]
            )
        )

        # Onset KD: align rising edges (what counts as a hit)
        g_on = Lambda(gate_onset, name="off_gate_onset_ds")(T_gate_ds)
        m_on = Lambda(gate_onset, name="rt_mask_onset")(mask_smoothed)
        kd_tensors.append(
            DistillLossBestShift(alpha=0.8, max_shift=6, name="mask_onset_kd")([g_on, m_on])
        )
        
        # Penalize HF when the student's mask says "closed"
        kd_tensors.append(
            HFSuppressByMask(alpha=0.4, name="hf_suppress_by_student_mask")([hf_final, mask_used_audio])
        )
        
        

        for i, (T_f_ds, S_f) in enumerate(zip(T_feats_ds, realtime_feats)):
            chan = int(T_f_ds.shape[-1])
            proj = Conv1D(chan, 1, padding="same", name=f"realtime_proj_feat_{i}")(S_f)
            kd_tensors.append(DistillLoss(alpha=REALTIME_FEAT_DISTILL_ALPHA, name=f"feat_distill_{i}")([T_f_ds, proj]))

        offline_audio_ds = FIRDecimateBy2(name="offline_audio_ds2")(offline_audio)
        kd_tensors.append(DistillLossBestShift(alpha=REALTIME_DISTILL_ALPHA, max_shift=8, name="audio_distill")([offline_audio_ds, y_realtime]))
        kd_tensors.append(MSTFTDistillLoss(alpha=REALTIME_AUDIO_STFT_DISTILL_ALPHA, name="audio_stft_distill")([offline_audio_ds, y_realtime]))

        off_pick_ds   = FIRDecimateBy2(name="offline_pick_ds2")(off_pick)
        off_finger_ds = FIRDecimateBy2(name="offline_finger_ds2")(off_finger)
        off_hf_ds     = FIRDecimateBy2(name="offline_hf_ds2")(off_hf_scaled)
        off_hf_ds_hp48 = HFHighpass(sample_rate=REALTIME_SAMPLE_RATE, cutoff_hz=REALTIME_HP_CUTOFF_HZ, order=REALTIME_HF_HP_ORDER, name="offline_hf_ds_hp48")(off_hf_ds)
        off_hf_ds_hp48_pick = Multiply(name="offline_hf_ds_hp48_pick_only")([off_hf_ds_hp48, style_scalar_b3])

        if enable_mask_kd:
            off_hf_env96 = CausalExpSmoother(time_const_ms=0.08, sample_rate=OFFLINE_SAMPLE_RATE, Lmax=64, name="off_hf_env96")(Abs1D()(off_hf_scaled))
            off_hf_env48 = FIRDecimateBy2(name="off_hf_env_ds2")(off_hf_env96)
            off_hf_env48_norm = Lambda(clip_01x4, name="off_hf_env48_norm")(off_hf_env48)

            kd_tensors.append(DistillBCE(alpha=REALTIME_HF_MASK_DISTILL_ALPHA, name="hf_mask_bce")([T_gate_ds, mask_pred]))
            kd_tensors.append(DistillLossBestShift(alpha=REALTIME_MASK_USED_DISTILL_ALPHA, max_shift=8, name="mask_smooth_distill")([T_gate_ds, mask_smoothed]))
            kd_tensors.append(DistillLoss(alpha=0.5 * REALTIME_HF_MASK_DISTILL_ALPHA, name="hf_env_mask_distill")([off_hf_env48_norm, mask_pred]))

        kd_tensors.append(DistillLossBestShift(alpha=REALTIME_STEM_PICK_DISTILL_ALPHA,   max_shift=8, name="stem_pick_distill")([off_pick_ds,   out_pick]))
        kd_tensors.append(DistillLossBestShift(alpha=REALTIME_STEM_FINGER_DISTILL_ALPHA, max_shift=8, name="stem_finger_distill")([off_finger_ds, out_finger]))
        kd_tensors.append(DistillLoss(alpha=REALTIME_STEM_HF_DISTILL_ALPHA, name="stem_hf_distill")([off_hf_ds_hp48_pick, hf_final_pick]))

        if realtime_resbp is not None and off_resbp_scaled is not None:
            off_resbp_ds = FIRDecimateBy2(name="offline_resbp_ds2")(off_resbp_scaled)
            kd_tensors.append(DistillLossBestShift(alpha=REALTIME_STEM_RESBP_DISTILL_ALPHA, max_shift=8, name="stem_resbp_distill")([off_resbp_ds, realtime_resbp]))

    if kd_tensors:
        y_realtime = LossTether(name="realtime_kd_tether")([y_realtime] + kd_tensors)

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
def add_sw_realtime_dual_nf(x, y):
    """
    Finetune (no style labels) for realtime model with an offline teacher.
    x = (x_realtime_48k, x_offline_96k)
    y = (y_audio_realtime_48k, style_flag_unused)

    - Forces the model to use its own classifier (style_force=0).
    - Disables the style BCE by providing zero sample-weights for the style output.
    """
    x_realtime, x_offline = x
    y_audio, _ = y
    b = tf.shape(x_realtime)[0]
    T_realtime = tf.shape(x_realtime)[1]

    # Dummy style label (ignored)
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    # Force the model to use its own classifier during finetune
    style_force = tf.zeros((b, 1), dtype=tf.float32)  # 0 = use prediction

    inputs = {
        'realtime_piezo_input':         x_realtime,
        'realtime_offline_piezo_input': x_offline,
        'realtime_style_true_input':    style_true,
        'realtime_style_force_input':   style_force,
    }
    outputs = (y_audio, style_true)

    # Sample weights: train audio only, disable style BCE
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)

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
        'realtime_piezo_input':         x_realtime,
        'realtime_offline_piezo_input': x_offline,
        'realtime_style_true_input':    style_true,
        'realtime_style_force_input':   style_force,
    }
    outputs = (y_audio, style_true)

    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)

    sample_weights = (sw_audio, sw_style)
    return inputs, outputs, sample_weights

@register_keras_serializable()
def add_sw_offline_finetune_nf(x, y):
    # y = (y_audio, style_flag) but finetune uses no true style labels
    y_audio, _ = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]

    # Dummy style label for the style head (B,1)
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    # Force the model to use its own classifier during finetune
    style_force = tf.zeros((b, 1), dtype=tf.float32)  # 0 = use prediction

    inputs = {
        'offline_piezo_input':        x,
        'offline_style_true_input':   style_true,
        'offline_style_force_input':  style_force,
    }

    outputs = (y_audio, style_true)

    # Sample weights: train audio only, disable style BCE
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)

    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime_dual_kd_only(x, y):
    """
    KD‑only mapping for converting the baseline realtime model to learned‑pick.
    x = (x_realtime_48k, x_offline_96k)
    y = (y_audio_realtime_48k, style_flag_unused)

    - Forces style_force=0 (use predicted) but disables both audio and style losses via sample weights.
    - Only the KD add_loss terms (teacher gate → student mask, etc.) will drive learning.
    """
    x_realtime, x_offline = x
    y_audio, _ = y
    b = tf.shape(x_realtime)[0]
    T = tf.shape(x_realtime)[1]

    # Dummy style label target (ignored)
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    # Force model to use its own style classifier
    style_force = tf.zeros((b, 1), dtype=tf.float32)

    inputs = {
        'realtime_piezo_input':         x_realtime,
        'realtime_offline_piezo_input': x_offline,
        'realtime_style_true_input':    style_true,
        'realtime_style_force_input':   style_force,
    }
    outputs = (y_audio, style_true)

    # Zero sample‑weights → disable the primary task losses; KD add_loss still applies
    sw_audio = tf.zeros((b, T, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

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

def _build_offline_dynamic_for_preview():
    """
    Prefer a trained baseline dynamic offline model if present; else try latest offline checkpoint;
    else build a fresh static+dynamic and copy weights (random if not trained).
    Returns an in-memory dynamic model (not saved to disk).
    """
    base_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    if os.path.exists(base_dyn_path):
        print("[stems/offline] Using baseline dynamic model for preview:", base_dyn_path)
        return tf.keras.models.load_model(base_dyn_path, compile=False)

    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_OFFLINE_OUT_DIR, "*.keras")))
    if ckpts:
        ckpt_path = ckpts[-1]
        print("[stems/offline] Building dynamic model from latest checkpoint:", ckpt_path)
        static = tf.keras.models.load_model(ckpt_path, compile=False)
        offline_dyn = build_offline_model(dynamic_time=True)
        offline_dyn.set_weights(static.get_weights())
        return offline_dyn

    print("[stems/offline] No baseline dynamic or checkpoint found; building fresh model (random weights).")
    static = build_offline_model(dynamic_time=False)
    offline_dyn = build_offline_model(dynamic_time=True)
    offline_dyn.set_weights(static.get_weights())
    return offline_dyn

def _build_realtime_dynamic_learned_pick_for_preview():
    dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME)
    if os.path.exists(dyn_path):
        return tf.keras.models.load_model(dyn_path, compile=False)

    stat_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_STATIC_MODEL_FILENAME)
    if os.path.exists(stat_path):
        static_src = tf.keras.models.load_model(stat_path, compile=False)
        realtime_dyn = build_realtime_model(
            offline=None, dynamic_time=True,
            cap_mask_with_gate=False, use_learned_hf_mask=True,
            enable_mask_kd=False, detach_mask_from_audio=True, mask_min_floor=0.0,
            mask_release_ms=REALTIME_MASK_RELEASE_MS,
            mask_softknee=(REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
            hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
        )
        copy_weights_by_name(realtime_dyn, static_src, verbose=True)
        return realtime_dyn

    base_dyn = _build_realtime_dynamic_for_preview()
    realtime_dyn = build_realtime_model(
        offline=None, dynamic_time=True,
        cap_mask_with_gate=False, use_learned_hf_mask=True,
        enable_mask_kd=False, detach_mask_from_audio=True, mask_min_floor=0.0,
        mask_release_ms=REALTIME_MASK_RELEASE_MS,
        mask_softknee=(REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
        hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
    )
    copy_weights_by_name(realtime_dyn, base_dyn, verbose=True)
    return realtime_dyn

def _build_realtime_dynamic_for_preview():
    """
    Prefer a trained baseline dynamic realtime model if present; else try latest realtime checkpoint;
    else build a fresh teacher‑free static+dynamic and copy weights (random if not trained).
    Returns an in-memory dynamic model (not saved to disk).
    """
    base_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    if os.path.exists(base_dyn_path):
        print("[stems/realtime] Using baseline dynamic model for preview:", base_dyn_path)
        return tf.keras.models.load_model(base_dyn_path, compile=False)

    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_REALTIME_OUT_DIR, "*.keras")))
    if ckpts:
        ckpt_path = ckpts[-1]
        print("[stems/realtime] Building dynamic model from latest checkpoint:", ckpt_path)
        static = tf.keras.models.load_model(ckpt_path, compile=False)
        realtime_dyn = build_realtime_model(dynamic_time=True)
        copy_weights_by_name(realtime_dyn, static)
        return realtime_dyn

    print("[stems/realtime] No baseline dynamic or checkpoint found; building fresh model (random weights).")
    static = build_realtime_model(dynamic_time=False)
    realtime_dyn = build_realtime_model(dynamic_time=True)
    copy_weights_by_name(realtime_dyn, static)
    return realtime_dyn

def _sum_bypass_terms(terms):
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return Add(name="ft_bypass_sum")(terms)

def _apply_body_only_adapter(x_body, bands, sample_rate, prefix, order):
    # y_body_eq = x_body + sum_i g_i * BP_i(x_body)
    return add_coloreq(x_body, bands=bands, sample_rate=sample_rate, name_prefix=prefix, order=order)

def _make_body_from_output(base_model, y_audio, sample_rate,
                           exclude_hf=True, exclude_ir=False, exclude_resbp=False):
    bypass_terms = []

    if exclude_hf:
        lname = "realtime_hf_pick_only" if sample_rate == REALTIME_SAMPLE_RATE else "offline_hf_pick_only"
        try:
            hf = base_model.get_layer(lname).output
            bypass_terms.append(hf)
        except Exception as e:
            raise RuntimeError(f"Missing HF pick-only layer '{lname}' in base model.") from e

    if exclude_ir and sample_rate == REALTIME_SAMPLE_RATE:
        # Prefer the post-aligned IR tensor so time length matches the main path
        try:
            ir = base_model.get_layer("realtime_ir_matchlen").output
        except Exception:
            # Older models: align lowpass output to y_audio explicitly
            ir_lp = base_model.get_layer("realtime_ir_lowpass").output
            ir = MatchTimeToRef(name="ft_ir_matchlen")([ir_lp, y_audio])
        bypass_terms.append(ir)

    if exclude_resbp:
        lname = "realtime_resbp_scale" if sample_rate == REALTIME_SAMPLE_RATE else "offline_resbp_scale"
        try:
            rbp = base_model.get_layer(lname).output
            bypass_terms.append(rbp)
        except Exception:
            pass

    bypass = _sum_bypass_terms(bypass_terms)
    if bypass is None:
        return y_audio, None

    body = Sub(name="ft_body_minus_bypass")([y_audio, bypass])
    return body, bypass

def _add_back_bypass(y_body_eq, bypass):
    if bypass is None:
        return y_body_eq
    return Add(name="ft_body_plus_bypass")([y_body_eq, bypass])

def _load_baseline_realtime_dyn():
    """
    Load the baseline realtime dynamic model if present; else fall back to preview.
    """
    dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    if os.path.exists(dyn_path):
        print("[hf-shift] Loading baseline realtime dynamic:", dyn_path)
        return tf.keras.models.load_model(dyn_path, compile=False)
    print("[hf-shift] Baseline dynamic not found; building preview.")
    return _build_realtime_dynamic_for_preview()

def make_realtime_dynamic_with_hf_shift(shift_samples):
    """
    Build a dynamic realtime model with a given HF shift and copy weights by name
    from the baseline dynamic model. The timeshift layer has no weights.
    """
    base_dyn = _load_baseline_realtime_dyn()
    # Mirror baseline dynamic gating behavior (teacher-free, masked by fixed gate)
    rt_dyn_shift = build_realtime_model(
        offline=None,
        dynamic_time=True,
        cap_mask_with_gate=True,
        use_learned_hf_mask=False,
        enable_mask_kd=False,
        detach_mask_from_audio=True,
        mask_min_floor=0.0,
        hf_time_shift_samples=int(shift_samples)
    )
    copy_weights_by_name(rt_dyn_shift, base_dyn, verbose=True)
    return rt_dyn_shift

def render_hf_shift_sweep_realtime(min_shift=-8, max_shift=8, sec=DEMO_SEC, offset=DEMO_OFFSET_SEC):
    """
    For each integer shift in [min_shift, max_shift], render a short demo so you can listen.
    Outputs go to ./out/demos/hf_shift_sweep/realtime/shift_{+/-d}/
    """
    assert_has_pairs()
    path_src, path_tgt = all_pairs[DEMO_PAIR_INDEX]
    for s in range(int(min_shift), int(max_shift) + 1):
        model = make_realtime_dynamic_with_hf_shift(s)
        out_dir = os.path.join(DEMOS_OUT_DIR, "hf_shift_sweep", "realtime", f"shift_{s:+d}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[hf-shift] Rendering shift={s} samples → {out_dir}")
        generate_demo_from_files(
            model=model,
            out_dir=out_dir,
            sample_rate=REALTIME_SAMPLE_RATE,
            path_src=path_src,
            path_target=path_tgt,
            prediction_name=f"demo_pred_shift_{s:+d}",
            only_prediction=False,
            offset_sec=offset,
            sec=sec
        )
        

        
def make_base_realtime_learned_pick(
    epochs=REALTIME_LEARNED_PICK_MASK_EPOCHS,
    steps_cap=REALTIME_LEARNED_PICK_MASK_STEPS_CAP,
    lr=REALTIME_LEARNED_PICK_MASK_LR
):
    """
    Create a learned‑pick variant of the baseline realtime model WITHOUT touching the baseline files.

    Flow:
      - Load baseline offline dynamic (teacher) → causal clone.
      - Load baseline realtime dynamic (student init).
      - Build a STATIC training student with: cap_mask_with_gate=True, use_learned_hf_mask=True, enable_mask_kd=True.
      - Freeze everything except the tiny HF mask subnetwork; compile with KD‑only sample weights.
      - Train briefly (few epochs) to mimic the teacher gate/envelope.
      - Export a TEACHER‑FREE DYNAMIC runtime model with cap_mask_with_gate=False (no fixed gate), use_learned_hf_mask=True.

    Saves:
      ./out/models/realtime/piezo2mic_baseline_realtime_learned_pick_dynamic.keras   (live)
      ./out/models/realtime/piezo2mic_baseline_realtime_learned_pick.keras           (static, with teacher; for inspection)
    """
    # 1) Locate teacher (offline baseline dynamic)
    base_off_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(base_off_dyn_path):
        print("ERROR: baseline offline dynamic model not found:", base_off_dyn_path)
        print("Run: python piezo2mic.py --train-offline")
        sys.exit(1)
    offline_nc = tf.keras.models.load_model(base_off_dyn_path, compile=False)
    offline_causal = build_offline_model_causal(offline_nc, dynamic_time=True)

    # 2) Load baseline realtime dynamic as student initializer (kept intact on disk)
    base_rt_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(base_rt_dyn_path):
        print("ERROR: baseline realtime dynamic model not found:", base_rt_dyn_path)
        print("Run: python piezo2mic.py --train-realtime")
        sys.exit(1)
    base_rt_dyn = tf.keras.models.load_model(base_rt_dyn_path, compile=False)

    # 3) Build STATIC training model (with teacher and mask KD enabled)
    rt_static_train = build_realtime_model(
        offline=offline_causal,
        dynamic_time=False,
        cap_mask_with_gate=True,
        use_learned_hf_mask=True,
        enable_mask_kd=True,
        detach_mask_from_audio=False,
        mask_min_floor=0.0,
        mask_release_ms=REALTIME_MASK_RELEASE_MS,
        mask_softknee=(REALTIME_MASK_SOFTKNEE_K, 0.55),
        hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
    )
    copy_weights_by_name(rt_static_train, base_rt_dyn, verbose=True)

    # 4) Only train the tiny HF mask layers
    set_trainable_for_finetune_realtime(rt_static_train, train_mask=True)

    # 5) Dataset: dual‑rate (48k student + 96k teacher input), KD‑only sample weights + shared random gain
    # Build chunks from the original training pairs (no re‑segmentation on disk)
    pick_chunks   = build_chunks(pairs_list=picked_pairs, seg_len=REALTIME_SEG_LEN, sample_rate=REALTIME_SAMPLE_RATE)
    finger_chunks = build_chunks(pairs_list=finger_pairs, seg_len=REALTIME_SEG_LEN, sample_rate=REALTIME_SAMPLE_RATE)

    def make_dual_ds(chunks, is_pick):
        sig_x = (
            tf.TensorSpec((REALTIME_SEG_LEN, 1), tf.float32),
            tf.TensorSpec((REALTIME_SEG_LEN * (OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE), 1), tf.float32)
        )
        sig_y = (tf.TensorSpec((REALTIME_SEG_LEN, 1), tf.float32), tf.TensorSpec((), tf.float32))
        return tf.data.Dataset.from_generator(
            lambda: segment_generator_dual(
                seg_len_realtime=REALTIME_SEG_LEN,
                chunks=chunks,
                is_pick_chunks=is_pick,
                realtime_sample_rate=REALTIME_SAMPLE_RATE,
                offline_sample_rate=OFFLINE_SAMPLE_RATE
            ),
            output_signature=(sig_x, sig_y)
        )

    ds_pick   = make_dual_ds(pick_chunks,   True)
    ds_finger = make_dual_ds(finger_chunks, False)

    train_ds = tf.data.Dataset.sample_from_datasets(
        [ds_pick, ds_finger],
        weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO],
        seed=SEED
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True)\
     .batch(FINETUNE_BATCH_SIZE)\
     .map(add_sw_realtime_dual_kd_only)\
     .map(add_shared_random_gain_dual(db_lo=-30.0, db_hi=+18.0))\
     .prefetch(tf.data.AUTOTUNE)

    # Conservative steps/epoch from both pools, then cap
    steps_pk = approx_steps_from_chunks(pick_chunks, REALTIME_SEG_LEN, REALTIME_SAMPLE_RATE, FINETUNE_BATCH_SIZE)
    steps_fg = approx_steps_from_chunks(finger_chunks, REALTIME_SEG_LEN, REALTIME_SAMPLE_RATE, FINETUNE_BATCH_SIZE)
    steps_per_epoch = min(max(1, steps_pk + steps_fg), int(steps_cap))

    # 6) Compile with any placeholder losses (weights irrelevant due to zero sample‑weights)
    loss_rt_lite = FinetuneCombinedLossLite(
        sample_rate=REALTIME_SAMPLE_RATE,
        stft_scales=((1024, 256, 1024),),
        l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
        onset_gain=HF_ONSET_GAIN
    )

    rt_static_train.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=[loss_rt_lite, "binary_crossentropy"],
        loss_weights=[1.0, 0.0],  # style BCE disabled; audio loss silenced by sample‑weights=0
        metrics=[["mae"], []]
    )
    
    # Make mask learning "pure": disable audio/feature/stem KD
    _disable_non_mask_kd(rt_static_train)
    # Make mask KD stronger (open/close and leak penalties)
    _boost_mask_kd(rt_static_train)

    # Optional: gently anneal mask KD strengths (short schedule)
    kd_targets = {
        "hf_mask_distill":   (REALTIME_HF_MASK_DISTILL_ALPHA,   0.4, 0, epochs),
        "mask_used_distill": (REALTIME_MASK_USED_DISTILL_ALPHA, 0.1, 0, epochs),
    }
    cbs = [
        KDAlphaScheduler(kd_targets),
        TerminateOnNaN(),
    ]

    print(f"[convert] Training mask‑only KD: epochs={epochs}, steps/epoch≈{steps_per_epoch}, lr={lr}")
    rt_static_train.fit(train_ds,
                        epochs=int(epochs),
                        steps_per_epoch=int(steps_per_epoch),
                        verbose=1,
                        callbacks=cbs)

    # 7) Save STATIC (with teacher; for inspection/retraining if needed)
    out_static = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_STATIC_MODEL_FILENAME)
    rt_static_train.save(out_static)
    print("✅ Saved base realtime learned‑pick (STATIC with teacher):", out_static)

    # 8) Export DYNAMIC teacher‑free runtime: cap OFF, learned mask gates HF/resbp.
    rt_dyn_export = build_realtime_model(
        offline=None,
        dynamic_time=True,
        cap_mask_with_gate=False,
        use_learned_hf_mask=True,
        enable_mask_kd=False,
        detach_mask_from_audio=True,
        mask_min_floor=0.0,
        mask_release_ms=REALTIME_MASK_RELEASE_MS,
        mask_softknee=(REALTIME_MASK_SOFTKNEE_K, 0.55),
        hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
    )
    copy_weights_by_name(rt_dyn_export, rt_static_train, verbose=True)

    out_dynamic = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME)
    rt_dyn_export.save(out_dynamic)
    print("✅ Saved base realtime learned‑pick (DYNAMIC, teacher‑free):", out_dynamic)

    return rt_dyn_export

def build_offline_finetune_wrapper(base_offline,
                                   bands=COLOREQ_BANDS_OFFLINE_BODY,
                                   order=97,
                                   name="piezo2mic_offline_ft_adapter",
                                   exclude_hf=True,
                                   exclude_ir=True,
                                   exclude_resbp=True):
    """
    Finetune adapter that only shapes the 'body' (default: excludes HF).
    """
    base_offline.trainable = False
    y_audio, y_style = base_offline.outputs

    body, bypass = _make_body_from_output(
        base_offline, y_audio,
        sample_rate=OFFLINE_SAMPLE_RATE,
        exclude_hf=exclude_hf,
        exclude_ir=exclude_ir,
        exclude_resbp=exclude_resbp
    )

    y_body_eq = _apply_body_only_adapter(body,
                                         bands=bands,
                                         sample_rate=OFFLINE_SAMPLE_RATE,
                                         prefix="offline_ft_coloreq",
                                         order=order)
    y_tuned = _add_back_bypass(y_body_eq, bypass)
    return tf.keras.Model(inputs=base_offline.inputs, outputs=[y_tuned, y_style], name=name)

def build_realtime_finetune_wrapper(base_realtime,
                                    bands=COLOREQ_BANDS_REALTIME_BODY,
                                    order=65,
                                    name="piezo2mic_realtime_ft_adapter",
                                    exclude_hf=True,
                                    exclude_ir=True,
                                    exclude_resbp=True):
    """
    Finetune adapter that only shapes the 'body' (default: excludes HF).
    """
    base_realtime.trainable = False
    y_audio, y_style = base_realtime.outputs

    body, bypass = _make_body_from_output(
        base_realtime, y_audio,
        sample_rate=REALTIME_SAMPLE_RATE,
        exclude_hf=exclude_hf,
        exclude_ir=exclude_ir,
        exclude_resbp=exclude_resbp
    )

    y_body_eq = _apply_body_only_adapter(body,
                                         bands=bands,
                                         sample_rate=REALTIME_SAMPLE_RATE,
                                         prefix="realtime_ft_coloreq",
                                         order=order)
    y_tuned = _add_back_bypass(y_body_eq, bypass)
    return tf.keras.Model(inputs=base_realtime.inputs, outputs=[y_tuned, y_style], name=name)

def set_trainable_for_finetune_adapter(model, prefix):
    """
    Freeze everything except IrScale layers named '{prefix}_g_*'.
    """
    for l in model.layers:
        l.trainable = False
    tr = []
    for l in model.layers:
        if isinstance(l, IrScale) and l.name.startswith(f"{prefix}_g_"):
            l.trainable = True
            tr.append(l.name)
    print(f"[finetune/adapter] trainable: {sorted(tr)}")
    return model

def fine_tune(offline=True, realtime=True):
    """
    Finetune stage.

    - Offline: body-only adapter (post-EQ) on top of a frozen baseline dynamic offline model.
    - Realtime: train a static student that distills from the baseline offline teacher;
                export a separate dynamic student that is teacher-free.
    """
    # Load finetune pair and pre-resample
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
    
    # Losses (offline adapter and realtime full)
    loss_off = FinetuneAdapterLoss(
        sample_rate=OFFLINE_SAMPLE_RATE,
        bands=COLOREQ_BANDS_OFFLINE_BODY,
        band_order=129,
        l1_w=0.4, mel_w=1.2, band_w=2.5,
        mel_upper_hz=FINETUNE_BODY_FMAX_OFFLINE
    )

    loss_rt_lite = FinetuneCombinedLossLite(
        sample_rate=REALTIME_SAMPLE_RATE,
        stft_scales=((1024, 256, 1024),),
        l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
        onset_gain=HF_ONSET_GAIN
    )

    off_dyn_wrap = None
    rt_dyn_export = None

    if offline:
        print("\n🔧 Finetuning OFFLINE student (learn pick via mask distillation from fixed‑gate teacher)…")

        # Teacher: baseline offline dynamic (96k) with fixed gate
        base_off_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(base_off_dyn_path):
            print("ERROR: baseline offline dynamic model not found:", base_off_dyn_path); sys.exit(1)
        teacher_off_dyn = tf.keras.models.load_model(base_off_dyn_path, compile=False)
        teacher_off_dyn.trainable = False

        # Student TRAIN model (static) with KD heads attached
        off_static_train = build_offline_student_for_finetune(
            teacher=teacher_off_dyn,
            dynamic_time=False
        )

        # Init student weights from baseline offline (copy by name)
        # Prefer baseline offline STATIC if available; else copy from dynamic
        base_off_static_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
        if os.path.exists(base_off_static_path):
            base_off_static = tf.keras.models.load_model(base_off_static_path, compile=False)
            copy_weights_by_name(off_static_train, base_off_static, verbose=True)
        else:
            copy_weights_by_name(off_static_train, teacher_off_dyn, verbose=True)

        # Train only the small parts: FiLM generators (optional) and mask convs
        set_trainable_for_finetune_offline(off_static_train, train_scales=True, train_film=True)

        # Lightweight finetune loss at 96 kHz; gate-aware HF/stft weighting is inside the loss
        loss_offlite = FinetuneCombinedLossLite(
            sample_rate=OFFLINE_SAMPLE_RATE,
            stft_scales=((1024, 256, 1024),),
            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
            onset_gain=HF_ONSET_GAIN
        )

        off_static_train.compile(
            optimizer=tf.keras.optimizers.Adam(FINETUNE_OFFLINE_LR),
            loss=[loss_offlite, "binary_crossentropy"],
            loss_weights=[1.0, 0.0],       # disable style BCE via sample-weights, too
            metrics=[["mae"], []]
        )

        # 96k dataset (student and teacher read the same student inputs)
        dsO = make_mixed_ds_from_arrays_offline(
            cached["x96"], cached["y96"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len=OFFLINE_SEG_LEN
        ).map(add_sw_offline_finetune_nf            # style_force=0, style weights = 0
        ).map(add_shared_random_gain_offline(       # level-invariance for KD
            db_lo=-30.0, db_hi=+18.0)
        ).prefetch(tf.data.AUTOTUNE)

        steps_off = approx_steps_from_length(
            total_samples=len(cached["x96"]),
            seg_len=OFFLINE_SEG_LEN,
            batch_size=FINETUNE_BATCH_SIZE,
            cap=FINETUNE_OFFLINE_STEPS_CAP
        )

        off_static_train.fit(dsO, epochs=FINETUNE_EPOCHS, steps_per_epoch=steps_off, verbose=1)

        # Export DYNAMIC teacher‑free (cap OFF; learned mask gates)
        off_dyn_export = build_offline_model(
            dynamic_time=True,
            cap_mask_with_gate=False,
            mask_min_floor=0.05,
            name="piezo2mic_offline_ft_dynamic"
        )
        copy_weights_by_name(off_dyn_export, off_static_train, verbose=True)

        ft_offline_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        off_dyn_export.save(ft_offline_dyn_path)
        print("✅ Saved finetuned offline model (dynamic, learned pick mask):", ft_offline_dyn_path)

        # Export STATIC teacher‑free
        off_static_export = build_offline_model(
            dynamic_time=False,
            cap_mask_with_gate=False,
            mask_min_floor=0.05,
            name="piezo2mic_offline_ft_static"
        )
        copy_weights_by_name(off_static_export, off_static_train, verbose=True)

        ft_offline_static_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
        off_static_export.save(ft_offline_static_path)
        print("✅ Saved finetuned offline model (static, learned pick mask):", ft_offline_static_path)

        # Use this for demo rendering
        off_dyn_wrap = off_dyn_export

    if realtime:
        print("\n🔧 Finetuning REALTIME model (static student distills from teacher; dynamic export is teacher-free)…")

        # Teacher: baseline offline dynamic → causal clone (no right-lookahead)
        base_off_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(base_off_dyn_path):
            print("ERROR: baseline offline dynamic model not found for teacher:", base_off_dyn_path); sys.exit(1)
        offline_nc = tf.keras.models.load_model(base_off_dyn_path, compile=False)
        offline_causal = build_offline_model_causal(offline_nc, dynamic_time=True)

        # Student init from baseline realtime dynamic
        base_rt_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(base_rt_dyn_path):
            print("ERROR: baseline realtime dynamic model not found:", base_rt_dyn_path); sys.exit(1)
        base_rt_dyn = tf.keras.models.load_model(base_rt_dyn_path, compile=False)

        # Build the TRAINING MODEL as STATIC with teacher attached (KD heads live here)
        rt_static_train = build_realtime_model(
            offline=offline_causal,
            dynamic_time=False,
            cap_mask_with_gate=True,
            use_learned_hf_mask=True,
            enable_mask_kd=True,
            detach_mask_from_audio=True,
            mask_min_floor=0.05,
            # NEW
            mask_release_ms=REALTIME_MASK_RELEASE_MS,
            mask_softknee=(REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
            hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
        )
        # Initialize from baseline realtime dynamic (safe by-name copy)
        copy_weights_by_name(rt_static_train, base_rt_dyn, verbose=True)

        # Train only FiLM generators and mask convs (keep small footprint)
        set_trainable_for_finetune_realtime(rt_static_train, train_scales=True, train_film=False)

        # Compile: audio-only finetune loss; style BCE disabled by sample weights
        rt_static_train.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_REALTIME_LR, clipnorm=1.0),
            loss=[loss_rt_lite, "binary_crossentropy"],
            loss_weights=[1.0, 0.0],
            metrics=[["mae"], []]
        )

        # Dual-rate finetune dataset (48k student + 96k teacher input)
        dsRT_dual = make_mixed_ds_from_arrays_realtime_dual(
            cached["x48"], cached["x96"], cached["y48"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len_rt=REALTIME_SEG_LEN
        ).map(add_sw_realtime_dual_nf
        ).map(add_shared_random_gain_dual(db_lo=-30.0, db_hi=+18.0)  # ← new
        ).prefetch(tf.data.AUTOTUNE)

        steps_rt = approx_steps_from_length(len(cached["x48"]), REALTIME_SEG_LEN,
                                            FINETUNE_BATCH_SIZE, cap=FINETUNE_REALTIME_STEPS_CAP)

        rt_static_train.fit(dsRT_dual, epochs=FINETUNE_EPOCHS, steps_per_epoch=steps_rt, verbose=1)

        # 1) Save STATIC finetuned realtime model WITH teacher reference (for inspection/retraining)
        ft_rt_static_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        rt_static_train.save(ft_rt_static_path)
        print("✅ Saved finetuned realtime model (static WITH teacher):", ft_rt_static_path)

        # 2) Build a teacher‑FREE DYNAMIC export and copy weights by name
        rt_dyn_export = build_realtime_model(
            offline=None,
            dynamic_time=True,
            cap_mask_with_gate=False,
            use_learned_hf_mask=True,
            enable_mask_kd=False,
            detach_mask_from_audio=True,
            mask_min_floor=0.0,
            # NEW
            mask_release_ms=REALTIME_MASK_RELEASE_MS,
            mask_softknee=(REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
            hf_time_shift_samples=REALTIME_HF_TIME_SHIFT_SAMPLES,
            hf_peak_ceiling=REALTIME_HF_PEAK_CEILING
        )
        copy_weights_by_name(rt_dyn_export, rt_static_train, verbose=True)

        ft_rt_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        rt_dyn_export.save(ft_rt_dyn_path)
        print("✅ Saved finetuned realtime model (dynamic, teacher‑free):", ft_rt_dyn_path)

    return off_dyn_wrap, rt_dyn_export

def set_trainable_for_finetune_offline(model, train_scales=False, train_mask=False, train_film=False):
    """
    Fast path for embedded:
      - Train only FiLM generators (Dense '*film_dense') + optional IrScale scalars
      - If train_mask=True, also train tiny mask convs: offline_hf_mask_c1/c2/logits
    """
    trainable = []

    # Freeze everything by default
    for l in model.layers:
        l.trainable = False

    # Enable only FiLM parameter generators
    if train_film:
        for l in model.layers:
            if "film_dense" in l.name:
                l.trainable = True
                trainable.append(l.name)

    # Optional: also allow the small IrScale scalars to adapt
    if train_scales:
        for l in model.layers:
            if isinstance(l, IrScale):
                l.trainable = True
                trainable.append(l.name)

    # Optional: train the mask subnetwork if present
    if train_mask:
        for lname in ["offline_hf_mask_c1", "offline_hf_mask_c2", "offline_hf_mask_logits"]:
            try:
                layer = model.get_layer(lname)
                layer.trainable = True
                trainable.append(layer.name)
            except Exception:
                pass  # layer may not exist in older/baseline models

    # Explicitly freeze style head and heads
    freeze_names = {
        "offline_style_embed",
        "offline_style_conv", "offline_style_gap", "offline_style_logit", "offline_style_prob",
        "offline_final_pick", "offline_final_finger",
    }
    for name in freeze_names:
        try:
            model.get_layer(name).trainable = False
        except Exception:
            pass

    tot_trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    print(f"[finetune/offline] trainable layers: {sorted(trainable)}")
    print(f"[finetune/offline] trainable params: {tot_trainable:,d}")
    return model

def with_random_gain(input_key, out_scale_audio=True, db_lo=-24.0, db_hi=+12.0):
    """
    Apply one random dB gain per-batch to inputs[input_key] and scale the audio target.
    out_scale_audio=True scales y_audio; style target is untouched.
    """
    def _map(inputs, outputs, sw):
        x = inputs[input_key]
        y_audio, y_style = outputs
        g_db = tf.random.uniform((), db_lo, db_hi)
        g = tf.pow(10.0, g_db / 20.0)
        inputs[input_key] = x * g
        outputs = (y_audio * g if out_scale_audio else y_audio, y_style)
        return inputs, outputs, sw
    return _map

def set_trainable_for_finetune_realtime(model, train_scales=False, train_film=False, train_mask=False):
    trainable = []
    for l in model.layers:
        l.trainable = False

    # FiLM generators
    if train_film:
        for l in model.layers:
            if "film_dense" in l.name:
                l.trainable = True
                trainable.append(l.name)

    # Optional: IrScale scalars
    if train_scales:
        for l in model.layers:
            if isinstance(l, IrScale):
                l.trainable = True
                trainable.append(l.name)

    if train_mask:
        for lname in ["realtime_hf_mask_c1", "realtime_hf_mask_c2", "realtime_hf_mask_logits"]:
            try:
                layer = model.get_layer(lname)
                layer.trainable = True
                trainable.append(layer.name)
            except:
                pass

    # Keep style head and heads frozen
    for name in {
        "realtime_style_embed",
        "realtime_style_conv", "realtime_style_gap", "realtime_style_logit", "realtime_style_prob",
        "realtime_style_blend",
        "realtime_final_pick", "realtime_final_finger",
    }:
        try: model.get_layer(name).trainable = False
        except: pass

    print(f"[finetune/realtime] trainable layers: {sorted(trainable)}")
    return model

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

def _boost_scales(model, sample_rate, use_finetuned=False):
    """
    Apply optional demo boosts to the debug model's IrScale layers.
    Only applies when the *_BOOST_DB values exceed 0.0.
    """
    
    hf_scale_layer_name = "offline_hf_scale" if sample_rate == OFFLINE_SAMPLE_RATE else "realtime_hf_scale"
    ir_scale_layer_name = "realtime_ir_scale"
    resbp_boost_layer_name = "offline_resbp_scale" if sample_rate == OFFLINE_SAMPLE_RATE else "realtime_resbp_scale"

    hf_boost_db = 0.0
    ir_boost_db = 0.0
    resbp_boost_db = 0.0

    if use_finetuned:
        if sample_rate == OFFLINE_SAMPLE_RATE:
            hf_boost_db = FINETUNE_OFFLINE_HF_BOOST_DB
            resbp_boost_db = FINETUNE_OFFLINE_RESBP_BOOST_DB
        elif sample_rate == REALTIME_SAMPLE_RATE:
            hf_boost_db = FINETUNE_REALTIME_HF_BOOST_DB
            ir_boost_db = FINETUNE_REALTIME_IR_BOOST_DB
            resbp_boost_db = FINETUNE_REALTIME_RESBP_BOOST_DB
    else:
        if (sample_rate == OFFLINE_SAMPLE_RATE):
            hf_boost_db = OFFLINE_HF_BOOST_DB
            resbp_boost_db = OFFLINE_RESBP_BOOST_DB
        elif (sample_rate == REALTIME_SAMPLE_RATE):
            hf_boost_db = REALTIME_HF_BOOST_DB
            ir_boost_db = REALTIME_IR_BOOST_DB
            resbp_boost_db = REALTIME_RESBP_BOOST_DB
    
    if hf_boost_db != 0.0:
        model = boost_hf_scale_in_model(model, hf_scale_layer_name, hf_boost_db)
        
    if (ir_boost_db != 0.0):
        model = boost_hf_scale_in_model(model, ir_scale_layer_name, gain_db=ir_boost_db)
    
    if (resbp_boost_db != 0.0):
        model = boost_hf_scale_in_model(model, resbp_boost_layer_name, gain_db=resbp_boost_db)

def _find_last_layer_with_suffix(model, suffix):
    cand = [l for l in model.layers if l.name.endswith(suffix)]
    return cand[-1] if cand else None

@register_keras_serializable()
def build_offline_debug_model(offline):
    outs = {
        "hf_pre":     offline.get_layer("offline_hf_out").output,
        "hf_gated":   offline.get_layer("offline_hf_mult_gate").output,
        "hf_hp":      offline.get_layer("offline_hf_highpass").output,
        "hf_scaled":  offline.get_layer("offline_hf_scale").output,
        "hf_pick":    offline.get_layer("offline_hf_pick_only").output,  # pick-only HF
        "pick_c":     offline.get_layer("offline_pick_contrib").output,
        "finger_c":   offline.get_layer("offline_finger_contrib").output,
        "y_pred":     offline.get_layer("offline_mix_main").output,
        "style_pred": offline.get_layer("offline_style_prob").output,
    }
    # Optional: the fixed gate may not exist (cap_mask_with_gate=False)
    try: outs["gate"] = offline.get_layer("offline_gate").output
    except Exception: pass
    # Optional stems
    try: outs["resbp"] = offline.get_layer("offline_resbp_scale").output
    except Exception: pass
    # Optional mask tensors
    for name in ["offline_hf_mask", "offline_hf_mask_smooth", "offline_hf_mask_used"]:
        try:
            outs[name.replace("offline_", "")] = offline.get_layer(name).output
        except Exception:
            pass
    # Optional last FiLM params
    g_last = _find_last_layer_with_suffix(offline, "_film_gamma_expand")
    b_last = _find_last_layer_with_suffix(offline, "_film_beta_expand")
    if g_last is not None: outs["film_gamma_last"] = g_last.output
    if b_last is not None: outs["film_beta_last"]  = b_last.output

    return tf.keras.Model(inputs=offline.inputs, outputs=outs, name="offline_debug_model")


def build_realtime_debug_model(realtime):
    # Robustly pick the final audio tensor (depends on model version)
    def _get_first(names):
        for n in names:
            try:
                return realtime.get_layer(n).output
            except Exception:
                pass
        return None

    y_pred = _get_first(["realtime_with_ir_resbp", "realtime_with_ir", "realtime_main"])
    if y_pred is None:
        raise RuntimeError("Could not find a final audio tensor in the realtime model.")

    outs = {"y_pred": y_pred}

    # Helper to add an output only if the layer exists
    def add(key, lname):
        try:
            outs[key] = realtime.get_layer(lname).output
        except Exception:
            pass

    # HF branch probes (add what exists)
    add("hf_pre",   "realtime_hf_out")
    add("hf_gated", "realtime_hf_mult_gate")
    add("hf_hp",    "realtime_hf_highpass")
    add("hf_scaled","realtime_hf_scale")
    add("hf_pick",  "realtime_hf_pick_only")

    # Optional learned mask probes (older baselines may not have these)
    add("mask_pred",      "realtime_hf_mask")
    add("hf_mask_smooth", "realtime_hf_mask_smooth")
    add("mask_used",      "realtime_hf_mask_used")
    add("mask_audio",     "realtime_hf_mask_softknee")      # post-softknee
    add("mask_sharpen",   "realtime_hf_mask_sharpen")       # if present
    

    # Stems and style
    add("pick_c",     "realtime_pick_contrib")
    add("finger_c",   "realtime_finger_contrib")
    add("resbp",      "realtime_resbp_scale")
    add("style_pred", "realtime_style_prob")

    # IR-only probe (name depends on version)
    ir = _get_first(["realtime_ir_matchlen", "realtime_ir_interp", "realtime_ir_lowpass", "realtime_ir_upsample"])
    if ir is not None:
        outs["ir_only"] = ir

    # Optional fixed gate
    add("gate", "realtime_gate")

    return tf.keras.Model(inputs=realtime.inputs, outputs=outs, name="realtime_debug_model")

def render_isolated_stems(
    debug_model,
    path_src,
    path_tgt,
    out_dir,
    sample_rate,
    sec=LONG_DEMO_SEC,
    offset_sec=LONG_DEMO_OFFSET_SEC,
    block_sec=12.0,
    resample_margin_ms=50.0,
    use_finetuned=False
):
    os.makedirs(out_dir, exist_ok=True)
    _boost_scales(model=debug_model, sample_rate=sample_rate, use_finetuned=use_finetuned)

    # Decide style_true_flag from filename (unchanged)...
    if "pick" in path_src:
        style_true_flag = 1.0
    elif "finger" in path_src:
        style_true_flag = 0.0
    else:
        style_true_flag = None

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    g_norm = 1.0

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=debug_model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))

    forced_variants = [
        ("pred",   np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)),  # predicted style
        ("forced_pick",   np.array([[1.0]], dtype=np.float32), np.ones((1, 1), dtype=np.float32)),
        ("forced_finger", np.array([[0.0]], dtype=np.float32), np.ones((1, 1), dtype=np.float32)),
    ]

    def open_writers_for_pass(pass_key):
        is_pred = (pass_key == "pred")
        suffix = "" if is_pred else f"_{pass_key.split('_', 1)[1]}"
        w = {}

        if is_pred:
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, f"pred_full_forced{suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

        name_hf = "stem_hf_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"
        w["hf"] = sf.SoundFile(os.path.join(out_dir, name_hf), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

        # Optional branches
        w["ir"] = None
        w["resbp"] = None
        # Optional: write mask_used if present (debug)
        w["mask"] = None
        # Pick/finger stems
        w["pick"] = sf.SoundFile(os.path.join(out_dir, "stem_pick_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["fing"] = sf.SoundFile(os.path.join(out_dir, "stem_finger_only" + ("" if is_pred else f"_forced{suffix}") + ".wav"),
                                 mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        return w

    pass_writers = {k: open_writers_for_pass(k) for k, _, _ in forced_variants}

    produced_model = 0
    pred_prob_sum = 0.0
    pred_prob_count = 0

    def _core_crop(arr, a, b):
        return arr[a:b]

    while produced_model < want_total_model:
        core_model = min(block_core_model, want_total_model - produced_model)
        left_this  = left_ctx_model if produced_model > 0 else 0
        right_this = right_ctx_model if (produced_model + core_model) < want_total_model else 0

        core_start_file_off = int(round(produced_model * (file_sr / float(sample_rate))))
        core_start_file = START_FRAME + core_start_file_off

        left_ctx_file = int(math.ceil(left_this * (file_sr / float(sample_rate))))
        right_ctx_file= int(math.ceil(right_this * (file_sr / float(sample_rate))))
        core_file_len = int(math.ceil(core_model * (file_sr / float(sample_rate))))

        read_a = max(START_FRAME, core_start_file - left_ctx_file - resample_margin_file)
        read_b = min(END_FRAME, core_start_file + core_file_len + right_ctx_file + resample_margin_file)
        n_read = max(0, read_b - read_a)
        if n_read <= 0:
            break

        x_blk_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        y_blk_2d, _ = sf.read(path_tgt, start=read_a, frames=n_read, dtype='float32', always_2d=True)
        x_file = x_blk_2d[:, 0] * g_norm
        y_file = y_blk_2d[:, 0] * g_norm

        if file_sr != sample_rate:
            x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
            y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
        else:
            x_resamp = x_file
            y_resamp = y_file

        left_off_file  = (core_start_file - left_ctx_file) - read_a
        left_off_model = int(round(left_off_file * (sample_rate / float(file_sr))))
        inp_start = max(0, min(left_off_model, len(x_resamp)))
        inp_end   = max(inp_start, min(inp_start + left_this + core_model + right_this, len(x_resamp)))
        x_inp     = x_resamp[inp_start:inp_end]
        if x_inp.size <= 0:
            break

        core_start_in_inp = min(left_this, len(x_inp))
        core_end_in_inp   = min(core_start_in_inp + core_model, len(x_inp))
        core_len          = max(0, core_end_in_inp - core_start_in_inp)
        if core_len <= 0:
            break

        x_b = x_inp[None, :, None].astype(np.float32)

        for pass_key, style_in, force_in in forced_variants:
            outs = debug_model.predict([x_b, style_in, force_in], verbose=0)
            if isinstance(outs, (list, tuple)):
                raise RuntimeError("Expected dict outputs from debug_model; got list/tuple.")

            # Prefer pick-only HF if available
            hf_key = "hf_pick" if "hf_pick" in outs else ("hf_scaled" if "hf_scaled" in outs else ("hf_gated" if "hf_gated" in outs else "hf_pre"))

            def core_of(key):
                arr = np.asarray(outs[key][0, :, 0], dtype=np.float32)
                return _core_crop(arr, core_start_in_inp, core_end_in_inp)

            wset = pass_writers[pass_key]

            if pass_key == "pred" and "style_pred" in outs:
                sp = float(np.asarray(outs["style_pred"])[0, 0])
                pred_prob_sum += sp * float(core_len)
                pred_prob_count += core_len

            wset["full"].write(core_of("y_pred"))
            
            
            hf_core = core_of(hf_key)

            # Optional: sharpen hf_only using mask^(P)/mask (P>1 ⇒ stronger close between hits)
            if REALTIME_MASK_SHARPEN_P is not None and float(REALTIME_MASK_SHARPEN_P) > 1.0:
                # Prefer post-softknee mask if available; else fall back to mask_used
                mask_key = "mask_audio" if "mask_audio" in outs else ("mask_used" if "mask_used" in outs else None)
                if mask_key is not None:
                    m_full = np.asarray(outs[mask_key][0, :, 0], dtype=np.float32)
                    m_core = _core_crop(m_full, core_start_in_inp, core_end_in_inp)
                    m_core = np.clip(m_core, 1e-6, 1.0)

                    # scale factor = mask^P / mask = mask^(P-1); clamps ensure ≤1 for P>=1
                    P = float(REALTIME_MASK_SHARPEN_P)
                    factor = np.power(m_core, max(0.0, P - 1.0))

                    # If we have pick-only HF, prefer that as base so the style gate stays intact
                    if "hf_pick" in outs:
                        hf_pick_full = np.asarray(outs["hf_pick"][0, :, 0], dtype=np.float32)
                        hf_core = _core_crop(hf_pick_full, core_start_in_inp, core_end_in_inp)
                    # Apply extra suppression
                    hf_core = hf_core * factor

            wset["hf"].write(hf_core)
            

            # Optional branches
            if "ir_only" in outs:
                if wset["ir"] is None:
                    fn = "stem_ir_lowpass.wav" if pass_key == "pred" else f"stem_ir_lowpass_{pass_key.replace('forced_', 'forced_')}.wav"
                    wset["ir"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["ir"].write(core_of("ir_only"))

            if "resbp" in outs:
                if wset["resbp"] is None:
                    fn = "stem_resbp_only.wav" if pass_key == "pred" else f"stem_resbp_only_{pass_key.replace('forced_', 'forced_')}.wav"
                    wset["resbp"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                wset["resbp"].write(core_of("resbp"))

            # Optional: write mask_used for debugging if present
            for key_mask in ("hf_mask_used", "mask_used"):
                if key_mask in outs:
                    if wset["mask"] is None:
                        fn = "stem_hf_mask_used.wav" if pass_key == "pred" else f"stem_hf_mask_used_{pass_key.replace('forced_', 'forced_')}.wav"
                        wset["mask"] = sf.SoundFile(os.path.join(out_dir, fn), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
                    wset["mask"].write(core_of(key_mask))

            wset["pick"].write(core_of("pick_c"))
            wset["fing"].write(core_of("finger_c"))

        core_start_global = inp_start + core_start_in_inp
        core_end_global   = inp_start + core_end_in_inp
        if "src" in pass_writers["pred"]:
            pass_writers["pred"]["src"].write(x_resamp[core_start_global:core_end_global])
            pass_writers["pred"]["tgt"].write(y_resamp[core_start_global:core_end_global])

        produced_model += core_len

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
    """
    Render offline stems from either baseline or fine-tuned dynamic model.
    """
    src = None
    tgt = None
    sample_rate = OFFLINE_SAMPLE_RATE

    # Always prefer the finetune pair when --finetuned is requested, even if model is missing.
    out_dir = STEMS_FINETUNE_OFFLINE_OUT_DIR if use_finetuned else STEMS_OFFLINE_OUT_DIR

    if use_finetuned:
        candidate = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        # Choose finetune src/tgt if they exist; else fall back to baseline pair
        src_ft = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt_ft = os.path.join(FINETUNE_SRC_DIR, "target.wav")
        if os.path.exists(src_ft) and os.path.exists(tgt_ft):
            src, tgt = src_ft, tgt_ft
        else:
            if not all_pairs:
                print("ERROR: No baseline training pairs found and finetune source/target missing.")
                sys.exit(1)
            src, tgt = all_pairs[DEMO_PAIR_INDEX]

        if os.path.exists(candidate):
            print("[stems/offline] Loading finetuned dynamic model:", candidate)
            offline_dyn = tf.keras.models.load_model(candidate, compile=False)
        else:
            print("[stems/offline] Finetuned dynamic model not found; building a preview model for tuning.")
            offline_dyn = _build_offline_dynamic_for_preview()
    else:
        # Baseline flow (as before)
        dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        if os.path.exists(dyn_path):
            print("[stems/offline] Loading baseline dynamic model:", dyn_path)
            offline_dyn = tf.keras.models.load_model(dyn_path, compile=False)
        else:
            ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_OFFLINE_OUT_DIR, "*.keras")))
            if ckpts:
                print("Loading latest checkpoint:", ckpts[-1])
                static = tf.keras.models.load_model(ckpts[-1], compile=False)
                offline_dyn = build_offline_model(dynamic_time=True)
                offline_dyn.set_weights(static.get_weights())
            else:
                static = build_offline_model(dynamic_time=False)
                offline_dyn = build_offline_model(dynamic_time=True)
                offline_dyn.set_weights(static.get_weights())
        # Baseline demo pair if not set already
        if src is None or tgt is None:
            assert_has_pairs()
            src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dbg = build_offline_debug_model(offline_dyn)
    render_isolated_stems(
        debug_model=dbg,
        path_src=src,
        path_tgt=tgt,
        out_dir=out_dir,
        sample_rate=sample_rate,
        sec=LONG_DEMO_SEC,
        offset_sec=LONG_DEMO_OFFSET_SEC,
        use_finetuned=use_finetuned
    )

def run_render_stems_realtime_learned_pick():
    """
    Render stems from the base realtime learned-pick dynamic model (teacher-free, no fixed gate).
    """
    src = None
    tgt = None
    sample_rate = REALTIME_SAMPLE_RATE
    out_dir = STEMS_REALTIME_LEARNED_OUT_DIR

    # Choose a pair (prefer baseline training pair for consistency)
    if not all_pairs:
        print("ERROR: No baseline training pairs found.")
        sys.exit(1)
    src, tgt = all_pairs[DEMO_PAIR_INDEX]

    # Load learned-pick dynamic or build a preview
    cand = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME)
    if os.path.exists(cand):
        print("[stems/realtime/learned] Loading learned-pick dynamic model:", cand)
        realtime_dyn = tf.keras.models.load_model(cand, compile=False)
    else:
        print("[stems/realtime/learned] Learned-pick dynamic not found; building preview.")
        realtime_dyn = _build_realtime_dynamic_learned_pick_for_preview()

    dbg = build_realtime_debug_model(realtime_dyn)
    render_isolated_stems(
        debug_model=dbg,
        path_src=src,
        path_tgt=tgt,
        out_dir=out_dir,
        sample_rate=sample_rate,
        sec=LONG_DEMO_SEC,
        offset_sec=LONG_DEMO_OFFSET_SEC,
        use_finetuned=False
    )

def run_render_stems_realtime(use_finetuned=False):
    src = None
    tgt = None
    sample_rate = REALTIME_SAMPLE_RATE

    # Always prefer the finetune pair when --finetuned is requested, even if model is missing.
    out_dir = STEMS_FINETUNE_REALTIME_OUT_DIR if use_finetuned else STEMS_REALTIME_OUT_DIR

    if use_finetuned:
        candidate = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        # Choose finetune src/tgt if they exist; else fall back to baseline pair
        src_ft = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt_ft = os.path.join(FINETUNE_SRC_DIR, "target.wav")
        if os.path.exists(src_ft) and os.path.exists(tgt_ft):
            src, tgt = src_ft, tgt_ft
        else:
            if not all_pairs:
                print("ERROR: No baseline training pairs found and finetune source/target missing.")
                sys.exit(1)
            src, tgt = all_pairs[DEMO_PAIR_INDEX]

        if os.path.exists(candidate):
            print("[stems/realtime] Loading finetuned dynamic model:", candidate)
            realtime_dyn = tf.keras.models.load_model(candidate, compile=False)
        else:
            print("[stems/realtime] Finetuned dynamic model not found; building a preview model for tuning.")
            realtime_dyn = _build_realtime_dynamic_for_preview()
    else:
        # Baseline flow (as before)
        dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        if os.path.exists(dyn_path):
            print("[stems/realtime] Loading baseline dynamic model:", dyn_path)
            realtime_dyn = tf.keras.models.load_model(dyn_path, compile=False)
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
        # Baseline demo pair if not set already
        if src is None or tgt is None:
            assert_has_pairs()
            src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dbg = build_realtime_debug_model(realtime_dyn)
    
    render_isolated_stems(
        debug_model=dbg,
        path_src=src,
        path_tgt=tgt,
        out_dir=out_dir,
        sample_rate=sample_rate,
        sec=LONG_DEMO_SEC,
        offset_sec=LONG_DEMO_OFFSET_SEC,
        use_finetuned=use_finetuned
    )

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
        # Load weights from the checkpoint, but rebuild the full graph (with teacher) first
        ckpt_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)

        # Critical: rebuild the realtime model with the teacher again
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)

        # Copy weights by layer name to keep shapes/graph, including KD heads
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
            "audio_distill":       (REALTIME_DISTILL_ALPHA,              2.0,  0,  80),
            "audio_stft_distill":  (REALTIME_AUDIO_STFT_DISTILL_ALPHA,   0.2,  0,  80),
            # feature/stem KD
            "feat_distill_0":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
            "feat_distill_1":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
            "stem_pick_distill":   (REALTIME_STEM_PICK_DISTILL_ALPHA,    0.2,  0,  80),
            "stem_finger_distill": (REALTIME_STEM_FINGER_DISTILL_ALPHA,  0.2,  0,  80),
            "stem_hf_distill":     (REALTIME_STEM_HF_DISTILL_ALPHA,      0.2,  0,  80),
            "stem_resbp_distill":  (REALTIME_STEM_RESBP_DISTILL_ALPHA,   0.2,  0,  80),  # new
            # mask/gate KD
            "hf_mask_distill":     (REALTIME_HF_MASK_DISTILL_ALPHA,      1.0,  0,  80),
            "mask_used_distill":   (REALTIME_MASK_USED_DISTILL_ALPHA,    0.1,  0,  80),
            # style blend schedule
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

    # 1) Train the baseline realtime model (fixed gate)
    static_model, dynamic_model = train_shared(
        kind='realtime',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=REALTIME_SEG_LEN,
        sample_rate=REALTIME_SAMPLE_RATE,
        batch_size=REALTIME_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_realtime_model,
        build_model_fn_args_static={
            'offline': offline_causal,
            'cap_mask_with_gate': True,       # rely on fixed gate in base training
            'use_learned_hf_mask': False,     # learned mask not used to gate audio in base model
            'enable_mask_kd': False,          # do not KD the mask in base training
            'detach_mask_from_audio': True,   # if/when enabled later, keep mask grads clean
            'mask_min_floor': 0.0
        },
        build_model_fn_args_dynamic={
            # dynamic preview model used by DemoCallback; mirror the static gating behavior
            'offline': None,
            'cap_mask_with_gate': True,
            'use_learned_hf_mask': False,
            'enable_mask_kd': False,
            'detach_mask_from_audio': True,
            'mask_min_floor': 0.0
        },
        checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
        logs_out_dir=LOGS_REALTIME_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
        dual_offline=True  # keep dual-rate dataset (48k student + 96k teacher input)
    )

    # 2) Immediately derive and save the learned-pick variant (teacher-free, no fixed gate)
    print("\n🔁 Converting baseline realtime → learned‑pick (mask‑only KD)…")
    try:
        make_base_realtime_learned_pick(
            epochs=REALTIME_LEARNED_PICK_MASK_EPOCHS,
            steps_cap=REALTIME_LEARNED_PICK_MASK_STEPS_CAP,
            lr=REALTIME_LEARNED_PICK_MASK_LR
        )
    except NameError as e:
        print("ERROR: make_base_realtime_learned_pick is not defined. "
              "Please add the conversion function and constants as provided earlier.")
        raise e

def generate_fine_tune_demo(model, sample_rate, subdir_name):
    out_dir = os.path.join(FINETUNE_DEMOS_OUT_DIR, subdir_name)
    os.makedirs(out_dir, exist_ok=True)
    return generate_demo_from_files(
        model=model,
        sample_rate     = sample_rate,
        offset_sec      = FINETUNE_DEMO_OFFSET_SEC,
        sec             = FINETUNE_DEMO_SEC,
        path_src        = os.path.join(FINETUNE_SRC_DIR, "source.wav"),
        path_target     = os.path.join(FINETUNE_SRC_DIR, "target.wav"),
        out_dir         = out_dir,
        prediction_name = "demo_pred",
        use_finetuned=True
    )



def compute_rms(sig):
    sig64 = sig.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(sig64 * sig64) + 1e-12))

def _next_pow2(n):
    n = int(max(8, n))
    return int(1 << (int(np.ceil(np.log2(n)))))

def _spectral_denoise_target(y, sr,
                             win_ms=FINETUNE_TARGET_DENOISE_WIN_MS,
                             hop_ms=FINETUNE_TARGET_DENOISE_HOP_MS,
                             beta=FINETUNE_TARGET_DENOISE_BETA,
                             floor_db=FINETUNE_TARGET_DENOISE_FLOOR_DB,
                             noise_q=FINETUNE_TARGET_DENOISE_NOISE_Q,
                             min_noise_frames=FINETUNE_TARGET_DENOISE_MIN_FRAMES):
    """
    Light spectral denoiser that is COLA/NOLA-safe and compatible with older SciPy
    (no 'length' kw on istft). Falls back to manual trim/pad.
    """
    import numpy as np
    import scipy.signal as spsig

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

def print_coloreq_gains(model, prefix):
    """
    Print finetune-adapter band scales (IrScale) for debugging.
    prefix: "realtime_ft_coloreq" or "offline_ft_coloreq"
    """
    import numpy as np
    i = 0
    while True:
        lname = f"{prefix}_g_{i}"
        try:
            layer = model.get_layer(lname)
        except Exception:
            break
        w = layer.get_weights()
        g = float(w[0][0]) if (w and len(w[0]) == 1) else 0.0
        print(f"{lname}: scale={g:+.4f}")
        i += 1

def set_coloreq_gain(model, prefix, band_idx, scale):
    """
    Manually set one adapter band gain to test audibility.
    scale is linear (0.3 is a strong boost, -0.3 is a deep cut).
    """
    lname = f"{prefix}_g_{band_idx}"
    layer = model.get_layer(lname)
    w = layer.get_weights()
    if not w:
        w = [np.array([0.0], dtype=np.float32)]
    w[0][0] = float(scale)
    layer.set_weights(w)
    print(f"Set {lname} to {scale:+.4f}")

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
        do_rt_learned = ("--learned-pick" in args)

        # Offline baseline (unchanged)
        if do_offline:
            prepare_and_generate_demo(
                sample_rate=OFFLINE_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_OFFLINE_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
                build_model_fn=build_offline_model,
                use_safe_weight_copy=False
            )

        # Realtime baseline (fixed gate)
        if do_realtime:
            prepare_and_generate_demo(
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_REALTIME_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
                build_model_fn=build_realtime_model,
                build_model_fn_args_dynamic={
                    'offline': None,
                    'cap_mask_with_gate': False,
                    'use_learned_hf_mask': True,
                    'enable_mask_kd': False,
                    'detach_mask_from_audio': True,
                    'mask_min_floor': 0.0,
                    'mask_release_ms': REALTIME_MASK_RELEASE_MS,
                    'mask_softknee': (REALTIME_MASK_SOFTKNEE_K, REALTIME_MASK_SOFTKNEE_T),
                    'hf_peak_ceiling': REALTIME_HF_PEAK_CEILING
                },
                use_safe_weight_copy=True
            )

        # Realtime learned-pick (teacher-free, no fixed gate)
        if do_realtime and do_rt_learned:
            prepare_and_generate_demo(
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=os.path.join(DEMOS_OUT_DIR, "realtime_learned_pick"),
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
                build_model_fn=build_realtime_model,
                build_model_fn_args_dynamic={
                    'offline': None,
                    'cap_mask_with_gate': False,   # learned pick active (no fixed gate)
                    'use_learned_hf_mask': True,
                    'enable_mask_kd': False,
                    'detach_mask_from_audio': True,
                    'mask_min_floor': 0.0
                },
                use_safe_weight_copy=True
            )
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        offline_dyn, realtime_dyn = fine_tune(offline=len(sys.argv) <= 2 or sys.argv[2] !="--realtime-only", realtime=len(sys.argv) <= 2 or sys.argv[2] !="--offline-only")
        
        if (offline_dyn is not None):
            generate_fine_tune_demo(offline_dyn,  OFFLINE_SAMPLE_RATE,  subdir_name="offline")
        if (realtime_dyn is not None):
            generate_fine_tune_demo(realtime_dyn, REALTIME_SAMPLE_RATE, subdir_name="realtime")

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
            generate_fine_tune_demo(offline_dyn,  OFFLINE_SAMPLE_RATE,  subdir_name="offline")
        if realtime_dyn is not None:
            generate_fine_tune_demo(realtime_dyn, REALTIME_SAMPLE_RATE, subdir_name="realtime")
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--render-stems":
        # Flags (can be combined):
        # --finetuned        → finetuned dynamic models (if present)
        # --learned-pick     → base realtime learned-pick dynamic (if present)
        # --both             → baseline + finetuned
        # --all              → baseline + finetuned + learned-pick
        # --offline-only / --realtime-only limits branches
        args = set(sys.argv[2:])

        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)
        do_finetuned = ("--finetuned"     in args)
        do_both      = ("--both"          in args)
        do_learned   = ("--learned-pick"  in args)
        do_all       = ("--all"           in args)

        # OFFLINE: baseline or finetuned (existing behavior)
        if do_offline:
            if do_both or do_all:
                run_render_stems_offline(use_finetuned=False)
                run_render_stems_offline(use_finetuned=True)
            else:
                run_render_stems_offline(use_finetuned=do_finetuned)

        # REALTIME: baseline / finetuned / learned-pick
        if do_realtime:
            if do_all:
                run_render_stems_realtime(use_finetuned=False)
                run_render_stems_realtime(use_finetuned=True)
                run_render_stems_realtime_learned_pick()
            elif do_both:
                run_render_stems_realtime(use_finetuned=False)
                run_render_stems_realtime(use_finetuned=True)
                if do_learned:
                    run_render_stems_realtime_learned_pick()
            else:
                # single selection
                if do_learned and not do_finetuned:
                    run_render_stems_realtime_learned_pick()
                else:
                    run_render_stems_realtime(use_finetuned=do_finetuned)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--hf-shift-sweep":
        # Optional range: --hf-shift-sweep [-6:6]
        rng = (-8, 8)
        if len(sys.argv) > 2 and ":" in sys.argv[2]:
            a, b = sys.argv[2].split(":")
            rng = (int(a), int(b))
        render_hf_shift_sweep_realtime(min_shift=rng[0], max_shift=rng[1])
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--make-base-realtime-learned-pick":
        # Optional overrides: --epochs N --steps-cap M --lr X
        args = sys.argv[2:]
        def _get_arg(flag, cast, default):
            if flag in args:
                i = args.index(flag)
                if i+1 < len(args):
                    try: return cast(args[i+1])
                    except: pass
            return default
        ep   = _get_arg("--epochs",   int, REALTIME_LEARNED_PICK_MASK_EPOCHS)
        sc   = _get_arg("--steps-cap",int, REALTIME_LEARNED_PICK_MASK_STEPS_CAP)
        lr   = _get_arg("--lr",     float, REALTIME_LEARNED_PICK_MASK_LR)

        print("Converting baseline realtime → learned‑pick (mask‑only KD)…")
        model = make_base_realtime_learned_pick(epochs=ep, steps_cap=sc, lr=lr)

        # Optional quick demo using the newly saved dynamic model
        prepare_and_generate_demo(
            sample_rate=REALTIME_SAMPLE_RATE,
            demo_out_dir=os.path.join(DEMOS_OUT_DIR, "realtime_learned_pick"),
            dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_LEARNED_PICK_DYNAMIC_MODEL_FILENAME),
            checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
            build_model_fn=build_realtime_model,
            build_model_fn_args_dynamic={
                'offline': None,
                'cap_mask_with_gate': False,
                'use_learned_hf_mask': True,
                'enable_mask_kd': False,
                'detach_mask_from_audio': True,
                'mask_min_floor': 0.0
            },
            use_safe_weight_copy=True
        )
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()