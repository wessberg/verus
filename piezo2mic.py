#!/usr/bin/env python3
"""
===================================
INTRODUCTION
===================================
This architecture is designed to train a causal ResNet to map an acoustic‐guitar pickup signal
(piezo or magnetic) to the equivalent mic’d signal.

==================================
ARCHITECTURE
==================================

The fully Keras Serializable TensorFlow network is trained on pre-segmented audio chunks, split into pick and finger playing styles, and different
FILM (Feature-wise Linear Modulation) parameters are used to control the style of the output.

The network predicts not just the mic'd audio, but also the style of playing (pick vs finger) as a binary classification task, and
is aided by the true style labels during training.

A second conditioning axis is the pickup type (piezo vs magnetic). Unlike playing style, the pickup type is
fixed per rig, so it is a user-provided input (0.0 = piezo, 1.0 = magnetic) rather than a classified quantity.
A small pickup-conditional input adapter (a gated delta network that is an exact pass-through at pickup=0)
normalizes magnetic input into the piezo-like domain before the shared core, and the FiLM conditioning
receives both the style scalar and the pickup scalar. This also enables transforming electric guitars
(magnetic pickups) toward the acoustic mic sound.

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

- Inputs are pre-aligned offline (source and target must be time-aligned before training).
- Audio is split into contiguous non-overlapping 5-second chunks, plus an optional remainder.
- Each chunk is split into pick and finger segments, with a 50% ratio of each style.
- Train/val split is by contiguous time regions per file (the last ~20% of each file's chunks are
  held out) to avoid leakage between near-identical neighboring chunks.

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
   A shared random gain augmentation (applied identically to source, teacher input, and target) makes both
   base training and fine-tuning invariant to absolute input level.

Knowledge distillation (realtime training only): the causalized offline teacher runs inside the training
graph and the student is pulled toward it with several annealed KD losses (see KDAlphaScheduler):
- audio_distill:        waveform L1 between student audio (48k) and teacher audio (96k, avg-pooled 2×)
- audio_stft_distill:   multi-resolution STFT magnitude L1 between the same pair
- stem_pick_distill /
  stem_finger_distill:  L1 between the student's pick/finger head outputs and the teacher's
- feat_distill_0/1:     L1 between projected student features and teacher FiLM-block features
   
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

# Determinism env vars must be set BEFORE TensorFlow is imported, otherwise they are ignored.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import soundfile as sf
import tensorflow as tf
import scipy.signal as spsig
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import minimum_phase, firwin, resample_poly
from tensorflow.keras.utils import register_keras_serializable # pyright: ignore[reportMissingImports]
from tensorflow.keras.callbacks import ( # pyright: ignore[reportMissingImports]
    TensorBoard, EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, TerminateOnNaN
)
from tensorflow.keras import regularizers, constraints  # pyright: ignore[reportMissingImports]
from tensorflow.keras.losses import Loss  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (  # pyright: ignore[reportMissingImports]
    Input, Conv1D, SeparableConv1D, ZeroPadding1D, GlobalAveragePooling1D,
    Activation, Multiply, Add, Lambda, Dense, Layer, Reshape, Concatenate
)

# Determinism (env vars are set above, before the TF import)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set (only) by legacy_minphase_filters() while loading .keras files saved before the Sept-2026
# min-phase fix: their fixed-FIR weights have the old half lengths, so the layers must rebuild
# their kernels the old way for the shapes to match.
_LEGACY_MINPHASE_HALF = False

class legacy_minphase_filters:
    """Context manager: design min-phase FIRs the pre-Sept-2026 way (half taps, sqrt magnitude)."""
    def __enter__(self):
        global _LEGACY_MINPHASE_HALF
        self._prev = _LEGACY_MINPHASE_HALF
        _LEGACY_MINPHASE_HALF = True
        return self
    def __exit__(self, *exc):
        global _LEGACY_MINPHASE_HALF
        _LEGACY_MINPHASE_HALF = self._prev
        return False

def _to_minimum_phase(h_linear):
    """
    Convert a linear-phase FIR prototype to a minimum-phase FIR with the SAME magnitude response
    and (about) the same number of taps.

    scipy's minimum_phase(method='homomorphic') defaults to half=True, which returns HALF the
    taps AND the SQUARE ROOT of the magnitude response (i.e. half the stopband attenuation in dB).
    Every min-phase filter in this file used to go through that path, so a "60 dB" design shipped
    as a ~30 dB filter. Prefer half=False (scipy >= 1.14); on older scipy fall back to the
    autocorrelation trick (|H|^2 prototype -> sqrt -> |H|), which is equivalent.
    """
    h = np.asarray(h_linear, dtype=np.float64)
    if _LEGACY_MINPHASE_HALF:
        return np.asarray(minimum_phase(h, method='homomorphic'), dtype=np.float32)
    try:
        hm = minimum_phase(h, method='homomorphic', half=False)
    except TypeError:  # scipy < 1.14: no 'half' kwarg
        hm = minimum_phase(np.convolve(h, h[::-1]), method='homomorphic')
    return np.asarray(hm, dtype=np.float32)

def load_keras_model(path, compile=False):
    """
    tf.keras.models.load_model with a fallback for .keras files saved before the Sept-2026
    min-phase fix (their fixed-FIR weights are half the current length, so a plain load fails on a
    shape mismatch). Such models are loaded with the legacy filter designs — their own trained
    behaviour is preserved exactly; retrain or --migrate-* to get the corrected filters.
    """
    try:
        return tf.keras.models.load_model(path, compile=compile)
    except ValueError as e:
        if "could not be loaded" not in str(e):
            raise
    print(f"[load] {os.path.basename(path)} predates the Sept-2026 min-phase fix (half-length FIR "
          "weights); loading it with the legacy filter designs. Retrain (or --migrate-*) to get "
          "the corrected filters.")
    with legacy_minphase_filters():
        return tf.keras.models.load_model(path, compile=compile)

# On/off flags
DISABLE_DEMO_SCALE_BOOSTS = False

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
REALTIME_BATCH_SIZE         = 8    # segments are ~4x longer than before (full left context + 8192 core)
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
OFFLINE_TRAIN_BLOCK_SIZE       = 8192   # loss core per training segment; OFFLINE_SEG_LEN adds the context
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
REALTIME_TRAIN_BLOCK_SIZE = 8192   # loss core per training segment; REALTIME_SEG_LEN adds the context
REALTIME_EMBED_DIM        = 16
REALTIME_DISTILL_ALPHA    = 5.0

REALTIME_HF_HP_ORDER          = OFFLINE_HF_HP_ORDER

REALTIME_AUDIO_STFT_DISTILL_ALPHA   = 0.6   # KD weight for STFT KD (teacher audio_ds vs student audio)
REALTIME_STEM_PICK_DISTILL_ALPHA    = 0.5   # KD weight for pick stem
REALTIME_STEM_FINGER_DISTILL_ALPHA  = 0.5   # KD weight for finger stem
REALTIME_FEAT_DISTILL_ALPHA         = 1.0

REALTIME_STYLE_BLEND_INIT           = 0.7
REALTIME_STYLE_BLEND_TRAINABLE      = True

# KD annealing schedule: layer name -> (start_alpha, end_alpha, start_epoch, end_epoch).
# Used by KDAlphaScheduler during base training; fine-tuning distills at the end values.
KD_ALPHA_SCHEDULE = {
    "audio_distill":       (REALTIME_DISTILL_ALPHA,              2.0,  0,  80),
    "audio_stft_distill":  (REALTIME_AUDIO_STFT_DISTILL_ALPHA,   0.2,  0,  80),
    "feat_distill_0":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
    "feat_distill_1":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  80),
    "stem_pick_distill":   (REALTIME_STEM_PICK_DISTILL_ALPHA,    0.2,  0,  80),
    "stem_finger_distill": (REALTIME_STEM_FINGER_DISTILL_ALPHA,  0.2,  0,  80),
}

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

# Single source of truth for the pick-branch make-up gain applied by every runtime at "0 dB" knob
# position. The C++ pedal mirrors these as kPickAttackBaselineBoostDB / kPickResoBaselineBoostDB in
# cpp/src/main.cpp — keep them in sync. The demo boosts and --process defaults derive from these so
# demos, --process output, and the pedal all sit at the same level.
REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB = 22.0
REALTIME_PICK_RESO_RUNTIME_BASELINE_DB   = 32.0
OFFLINE_PICK_ATTACK_RUNTIME_BASELINE_DB  = 38.0
OFFLINE_PICK_RESO_RUNTIME_BASELINE_DB    = 8.0

OFFLINE_PICK_RESO_DEMO_BOOST_DB    = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else OFFLINE_PICK_RESO_RUNTIME_BASELINE_DB
REALTIME_PICK_RESO_DEMO_BOOST_DB   = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else REALTIME_PICK_RESO_RUNTIME_BASELINE_DB

FINETUNE_OFFLINE_PICK_RESO_DEMO_BOOST_DB    = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 8.0
FINETUNE_REALTIME_PICK_RESO_DEMO_BOOST_DB   = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 8.0


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
REALTIME_PICK_MASK_SMOOTH_RELEASE_MS = 36.0

OFFLINE_PICK_MASK_OUT_GAIN = 38.0
REALTIME_PICK_MASK_OUT_GAIN = 38.0

OFFLINE_PICK_MASK_KNEE_DB = 6.0
REALTIME_PICK_MASK_KNEE_DB = 4.0

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

OFFLINE_PICK_ATTACK_CUTOFF_HZ = 3000.0
REALTIME_PICK_ATTACK_CUTOFF_HZ = 3000.0

# Smoothing time constant (ms) for pickattack style envelope in streaming graph
PICK_ATTACK_STYLE_SMOOTH_MS = 2.0

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
REALTIME_TFLITE_T = 32
OFFLINE_TFLITE_FP32_FILENAME = f"piezo2mic_baseline_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
REALTIME_TFLITE_FP32_FILENAME = f"piezo2mic_baseline_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"
FINETUNE_OFFLINE_TFLITE_FP32_FILENAME = f"piezo2mic_finetuned_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
FINETUNE_REALTIME_TFLITE_FP32_FILENAME = f"piezo2mic_finetuned_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"

FINETUNE_OFFLINE_LR     = 1e-4
FINETUNE_REALTIME_LR     = 3e-5

FINETUNE_DEMO_SEC = 20
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

REALTIME_IR_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 0.0
FINETUNE_REALTIME_IR_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 0.0

REALTIME_LP_ORDER   = 257 

LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

# Loss-weight calibration: brief warmup before measuring loss scales, so the calibration reflects a
# non-random model instead of the initialization (random-init magnitudes are not representative).
CALIB_WARMUP_STEPS = 60

# Shared random gain augmentation for BASE training (finetune uses its own wider range).
BASE_TRAIN_GAIN_DB_LO = -24.0
BASE_TRAIN_GAIN_DB_HI = +12.0

# ─── Pickup-type conditioning ────────────────────────────────────────────────
# 0.0 = piezo, 1.0 = magnetic. Fixed per rig → user-provided input, never classified.
PICKUP_PIEZO    = 0.0
PICKUP_MAGNETIC = 1.0

# Pickup input adapter (gated delta network; exact pass-through at pickup=0)
PICKUP_ADAPTER_FILTERS  = 16
PICKUP_ADAPTER_KERNEL   = 9
PICKUP_ADAPTER_DILATIONS = (1, 2)

def pickup_adapter_left_context():
    return sum((PICKUP_ADAPTER_KERNEL - 1) * d for d in PICKUP_ADAPTER_DILATIONS)

# ─── Training-segment layout ───────────────────────────────────────────────────
# A training segment = [left context | loss core | right context]. Only the core is fully
# contexted (every branch sees real history, the offline model sees its full look-ahead), and
# ONLY the core enters the losses. Previously the loss covered whole segments in which most
# samples had a zero-padded history (offline: 86% of the segment; realtime: the IR branch's
# context alone exceeded the entire segment), i.e. the models were trained to behave under
# boundary conditions that never occur in inference.

def _offline_sum_right_pads():
    total = 0
    for _ in range(OFFLINE_PRE_BLOCKS):
        total += min(OFFLINE_LOOKAHEAD_SAMPLES, (OFFLINE_PRE_KERNEL - 1) * 1)
    for d in OFFLINE_DILATIONS:
        total += min(OFFLINE_LOOKAHEAD_SAMPLES, (OFFLINE_KERNEL_SIZE - 1) * d)
    return int(total)

OFFLINE_TRAIN_RIGHT_CTX = _offline_sum_right_pads()                                   # sum of per-layer look-aheads
OFFLINE_TRAIN_LEFT_CTX  = ((OFFLINE_RECEPTIVE_FIELD - 1) + OFFLINE_PRE_BLOCKS * (OFFLINE_PRE_KERNEL - 1)
                           - OFFLINE_TRAIN_RIGHT_CTX + pickup_adapter_left_context())  # sum of per-layer left pads
OFFLINE_SEG_LEN         = OFFLINE_TRAIN_LEFT_CTX + OFFLINE_TRAIN_BLOCK_SIZE + OFFLINE_TRAIN_RIGHT_CTX
OFFLINE_LOSS_CORE       = (OFFLINE_TRAIN_LEFT_CTX, OFFLINE_TRAIN_LEFT_CTX + OFFLINE_TRAIN_BLOCK_SIZE)

def _ema_fir_len(ms, sr, scale, Lmax):
    s = max(1, int(round((float(ms) / 1000.0) * float(sr))))
    return int(np.clip(np.ceil(scale * s), 8, Lmax))

def _realtime_train_left_context():
    """
    Conservative (design-order, not min-phase-probed) max left context over every realtime branch
    that feeds the final sum, mirroring build_realtime_streaming_model. Over-estimating by a few
    samples only makes the segment slightly longer.
    """
    sr = REALTIME_SAMPLE_RATE
    lc_mask = ((REALTIME_HF_HP_ORDER - 1)
               + max(_ema_fir_len(REALTIME_PICK_MASK_ENV_FAST_MS, sr, 3.0, 1024),
                     _ema_fir_len(REALTIME_PICK_MASK_ENV_SLOW_MS, sr, 3.0, 4096)) - 1
               + max(_ema_fir_len(REALTIME_PICK_MASK_SMOOTH_ATTACK_MS, sr, 4.0, 2048),
                     _ema_fir_len(REALTIME_PICK_MASK_SMOOTH_RELEASE_MS, sr, 4.0, 2048)) - 1)
    lc_pa = 1 + lc_mask + (7 - 1) * 1 + (7 - 1) * 2 + (PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1)
    L_strike = _ema_fir_len(RESBP_STRIKE_ENV_MS_REALTIME, sr, 4.0, 512)
    L_lr = (PICK_RESO_MODAL_LEN_REALTIME - 1) // REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR + 1
    lc_pr = (lc_mask + 1 + (L_strike - 1) + (REALTIME_PICK_RESO_DS_TAPS - 1)
             + REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR * (L_lr - 1)
             + (REALTIME_PICK_RESO_INTERP_TAPS - 1) + (PICK_RESO_LOWBAND_PENALTY_ORDER - 1))
    R_ir = 1 + (REALTIME_IR_KERNEL - 1) * sum(REALTIME_IR_DILATIONS)
    lc_ir = (REALTIME_IR_FIR_SIZE - 1) + REALTIME_IR_DOWNSAMPLE_FACTOR * (R_ir - 1) + (max(193, REALTIME_LP_ORDER) - 1)
    lc_main = REALTIME_PRE_BLOCKS * (REALTIME_PRE_KERNEL - 1) + (REALTIME_KERNEL_SIZE - 1) * sum(REALTIME_DILATIONS)
    lc_col = 65 - 1
    return int(max(lc_main, lc_ir, lc_pa, lc_pr, lc_col) + pickup_adapter_left_context())

REALTIME_TRAIN_LEFT_CTX = _realtime_train_left_context()
REALTIME_SEG_LEN        = REALTIME_TRAIN_LEFT_CTX + REALTIME_TRAIN_BLOCK_SIZE
REALTIME_LOSS_CORE      = (REALTIME_TRAIN_LEFT_CTX, REALTIME_SEG_LEN)

# KD teacher context (96 kHz samples) prepended/appended to the teacher's copy of the segment so
# the (non-causal) offline teacher produces fully-contexted outputs over the student's segment.
KD_TEACHER_LEFT_CTX_SAMPLES  = OFFLINE_TRAIN_LEFT_CTX
KD_TEACHER_RIGHT_CTX_SAMPLES = OFFLINE_TRAIN_RIGHT_CTX

# Hop used when running a dynamic-T streaming Keras model offline (demos, --process). Must be a
# multiple of the decimation factors (8). The style EMA is hop-invariant (see StreamingScalarEMA).
KERAS_STREAM_HOP = 1024

# Style-logit EMA: alpha per REALTIME_TFLITE_T-sample hop (time constant ~33 ms at 48 kHz).
STYLE_EMA_ALPHA_PER_HOP = 0.98
STYLE_EMA_REF_HOP       = REALTIME_TFLITE_T

OFFLINE_PICK_ATTACK_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else OFFLINE_PICK_ATTACK_RUNTIME_BASELINE_DB
REALTIME_PICK_ATTACK_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB

FINETUNE_OFFLINE_PICK_ATTACK_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 32.0
FINETUNE_REALTIME_PICK_ATTACK_DEMO_BOOST_DB = 0.0 if DISABLE_DEMO_SCALE_BOOSTS else 32.0

KNOB_DEFAULT_MAIN_SCALE = 1.0
KNOB_DEFAULT_PICK_ATTACK_SCALE = 1.0
KNOB_DEFAULT_PICK_RESO_SCALE = 1.0
KNOB_DEFAULT_IR_SCALE = 1.0
KNOB_DEFAULT_DRY_PCT = 0.0
KNOB_DEFAULT_OUT_GAIN = 1.0
KNOB_DEFAULT_PHASE_INVERT = 0.0
KNOB_DEFAULT_LOW_CUT_HZ = 20.0
KNOB_DEFAULT_LOW_CUT_SLOPE_DB_PER_OCT = 6.0
KNOB_DEFAULT_LOW_SHELF_GAIN_DB = 0.0
KNOB_DEFAULT_LOW_SHELF_HZ = 200.0
KNOB_DEFAULT_LOW_SHELF_SLOPE_DB_PER_OCT = 12.0
KNOB_DEFAULT_MID_GAIN_DB = 0.0
KNOB_DEFAULT_MID_HZ = 1000.0
KNOB_DEFAULT_MID_Q = 1.0
KNOB_DEFAULT_HIGH_SHELF_GAIN_DB = 0.0
KNOB_DEFAULT_HIGH_SHELF_HZ = 4000.0
KNOB_DEFAULT_HIGH_SHELF_SLOPE_DB_PER_OCT = 12.0

def _common_knob_defaults():
    # In-graph knobs only. The master EQ is a runtime effect (see master_eq_defaults()).
    return {
        "dry_pct": KNOB_DEFAULT_DRY_PCT,
        "outgain": KNOB_DEFAULT_OUT_GAIN,
        "phaseinv": KNOB_DEFAULT_PHASE_INVERT,
        "main": KNOB_DEFAULT_MAIN_SCALE,
        "pickattack": KNOB_DEFAULT_PICK_ATTACK_SCALE,
        "pickreso": KNOB_DEFAULT_PICK_RESO_SCALE
    }

def realtime_knob_defaults():
    d = _common_knob_defaults()
    return {
        **d,
        "ir": KNOB_DEFAULT_IR_SCALE
    }

def offline_knob_defaults():
    d = _common_knob_defaults()
    return {
        **d
    }

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
VAL_MAX_BATCHES     = 64   # fixed validation set size (batches); cached in RAM for determinism

DEMO_SEC        = 2.0
DEMO_OFFSET_SEC = 0.2
DEMO_PAIR_INDEX = 0

LONG_DEMO_SEC        = 2.0
LONG_DEMO_OFFSET_SEC = 0.2
LONG_DEMO_PAIR_INDEX = 0

INPUT_DIR = "./input"
PICKUP_DIR_NAMES = ("piezo", "magnetic")
STYLE_DIR_NAMES  = ("pick", "finger")

# Legacy layout (treated as piezo)
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

# ───────── Master EQ (runtime IIR, NOT part of the model graph) ─────────
# The master EQ used to be four in-graph FIR banks (LC/LS/mid/HS blended over a knob grid).
# 33–193-tap FIRs cannot realise 20–300 Hz cuts (the bank filters were near-transparent) and the
# banks were ~35% of the realtime model's MACs, so the EQ now lives in the runtimes as RBJ
# biquads: apply_master_eq_iir() here (used by --process --eq ...) and the identical
# MasterEqIIR in cpp/src/main.cpp. Training and export are EQ-free. Defaults below are the
# neutral position (LC 20 Hz = bypass, all gains 0 dB) and are mirrored by the C++ Knobs.

def master_eq_defaults():
    return {
        "lc_hz": KNOB_DEFAULT_LOW_CUT_HZ, "lc_slope": KNOB_DEFAULT_LOW_CUT_SLOPE_DB_PER_OCT,
        "ls_gain_db": KNOB_DEFAULT_LOW_SHELF_GAIN_DB, "ls_hz": KNOB_DEFAULT_LOW_SHELF_HZ,
        "ls_slope": KNOB_DEFAULT_LOW_SHELF_SLOPE_DB_PER_OCT,
        "mid_gain_db": KNOB_DEFAULT_MID_GAIN_DB, "mid_hz": KNOB_DEFAULT_MID_HZ, "mid_q": KNOB_DEFAULT_MID_Q,
        "hs_gain_db": KNOB_DEFAULT_HIGH_SHELF_GAIN_DB, "hs_hz": KNOB_DEFAULT_HIGH_SHELF_HZ,
        "hs_slope": KNOB_DEFAULT_HIGH_SHELF_SLOPE_DB_PER_OCT,
    }

def _rbj_shelf_q_from_slope(gain_db, slope_dbpoct):
    # RBJ shelf slope parameter S in (0, 1]; 12 dB/oct ~ S=1 (steepest without overshoot)
    A = 10.0 ** (float(gain_db) / 40.0)
    S = min(1.0, max(0.25, float(slope_dbpoct) / 12.0))
    return 1.0 / math.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

def _rbj_biquad(kind, sr, f0, q, gain_db=0.0):
    """RBJ cookbook biquad -> normalized (b0,b1,b2,a1,a2). kind: hp1|hp2|ls|hs|peak"""
    sr = float(sr); f0 = min(max(float(f0), 1.0), 0.49 * sr)
    w0 = 2.0 * math.pi * f0 / sr
    if kind == "hp1":
        # first-order high-pass (6 dB/oct), bilinear
        k = math.tan(w0 / 2.0)
        n = 1.0 / (1.0 + k)
        return (n, -n, 0.0, (k - 1.0) * n, 0.0)
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * max(float(q), 1e-3))
    A = 10.0 ** (float(gain_db) / 40.0)
    if kind == "hp2":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "peak":
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    elif kind == "ls":
        sa = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cw + sa); b1 = 2 * A * ((A - 1) - (A + 1) * cw); b2 = A * ((A + 1) - (A - 1) * cw - sa)
        a0 = (A + 1) + (A - 1) * cw + sa;       a1 = -2 * ((A - 1) + (A + 1) * cw);    a2 = (A + 1) + (A - 1) * cw - sa
    elif kind == "hs":
        sa = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + sa); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - sa)
        a0 = (A + 1) - (A - 1) * cw + sa;       a1 = 2 * ((A - 1) - (A + 1) * cw);     a2 = (A + 1) - (A - 1) * cw - sa
    else:
        raise ValueError(kind)
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

# Butterworth Q values for cascaded 2nd-order sections (12/24/36 dB/oct)
_BUTTERWORTH_Q = {12.0: [0.70710678], 24.0: [0.54119610, 1.30656296], 36.0: [0.51763809, 0.70710678, 1.93185165]}

def master_eq_sos(sr, lc_hz=None, lc_slope=None, ls_gain_db=None, ls_hz=None, ls_slope=None,
                  mid_gain_db=None, mid_hz=None, mid_q=None, hs_gain_db=None, hs_hz=None, hs_slope=None):
    """
    Build the master EQ as a list of biquad sections (scipy 'sos' rows [b0,b1,b2,1,a1,a2]).
    Semantics mirror cpp/src/main.cpp MasterEqIIR exactly:
      - low cut: bypass at <= KNOB_DEFAULT_LOW_CUT_HZ; slope 6 = 1st-order, 12/24/36 = Butterworth cascade
      - low/high shelf and mid bell: RBJ; 0 dB gain = bypass
    Returns an empty list when everything is at the neutral position.
    """
    d = master_eq_defaults()
    v = lambda x, k: d[k] if x is None else float(x)
    lc_hz, lc_slope = v(lc_hz, "lc_hz"), v(lc_slope, "lc_slope")
    ls_gain_db, ls_hz, ls_slope = v(ls_gain_db, "ls_gain_db"), v(ls_hz, "ls_hz"), v(ls_slope, "ls_slope")
    mid_gain_db, mid_hz, mid_q = v(mid_gain_db, "mid_gain_db"), v(mid_hz, "mid_hz"), v(mid_q, "mid_q")
    hs_gain_db, hs_hz, hs_slope = v(hs_gain_db, "hs_gain_db"), v(hs_hz, "hs_hz"), v(hs_slope, "hs_slope")
    sos = []
    def add(c):
        b0, b1, b2, a1, a2 = c
        sos.append([b0, b1, b2, 1.0, a1, a2])
    if lc_hz > KNOB_DEFAULT_LOW_CUT_HZ + 1e-6:
        slope = min(_BUTTERWORTH_Q.keys() | {6.0}, key=lambda s: abs(s - lc_slope))
        if slope == 6.0:
            add(_rbj_biquad("hp1", sr, lc_hz, 0.0))
        else:
            for q in _BUTTERWORTH_Q[slope]:
                add(_rbj_biquad("hp2", sr, lc_hz, q))
    if abs(ls_gain_db) > 1e-6:
        add(_rbj_biquad("ls", sr, ls_hz, _rbj_shelf_q_from_slope(ls_gain_db, ls_slope), ls_gain_db))
    if abs(mid_gain_db) > 1e-6:
        add(_rbj_biquad("peak", sr, mid_hz, mid_q, mid_gain_db))
    if abs(hs_gain_db) > 1e-6:
        add(_rbj_biquad("hs", sr, hs_hz, _rbj_shelf_q_from_slope(hs_gain_db, hs_slope), hs_gain_db))
    return sos

def apply_master_eq_iir(x, sr, **knobs):
    """Apply master_eq_sos(sr, **knobs) to a mono float array (no-op at the neutral position)."""
    sos = master_eq_sos(sr, **knobs)
    if not sos:
        return np.asarray(x, dtype=np.float32)
    return spsig.sosfilt(np.asarray(sos, dtype=np.float64), np.asarray(x, dtype=np.float64)).astype(np.float32)

def parse_master_eq_arg(spec):
    """'--eq lc_hz=80,lc_slope=12,hs_gain_db=2' -> dict of master_eq_sos kwargs (validated)."""
    out = {}
    if not spec:
        return out
    valid = set(master_eq_defaults().keys())
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--eq entries must be key=value, got '{item}'")
        k, val = item.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise ValueError(f"--eq: unknown key '{k}'. Valid: {sorted(valid)}")
        out[k] = float(val)
    return out

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
# Loaded lazily (see load_training_data) so that non-training commands like --process or
# --demo-tflite do not pull the entire training corpus into memory at import time.
pairs_by_group = {}   # (pickup_name, style_name) -> [(src, tgt), ...]
picked_pairs = []     # every pick-style pair, across pickups (legacy consumers)
finger_pairs = []     # every finger-style pair, across pickups
all_pairs = []
audio_cache = {}
_file_sr_cache = {}
_resampled_mono_cache = {}
_training_data_loaded = False

def _scan_pair_dir(src_dir, tgt_dir):
    pairs = []
    for src in sorted(glob.glob(os.path.join(src_dir, "*_source.wav"))):
        tgt = os.path.join(tgt_dir, os.path.basename(src).replace("_source.wav", "_target.wav"))
        if os.path.exists(tgt):
            pairs.append((src, tgt))
        else:
            print("Warning: no target for", src)
    return pairs

def load_training_data():
    """
    Lazily scan ./input and load all training pairs into memory.

    Canonical layout: input/<pickup>/<style>/{source,target} with pickup in {piezo, magnetic}
    and style in {pick, finger}. The legacy layout input/<style>/{source,target} is treated
    as piezo.

    TODO(data): once simultaneous piezo+magnetic+mic triplet recordings exist, pair the two
    sources of the same take here and add a cross-pickup consistency loss during training
    (model(piezo_src) ≈ model(magnetic_src) for the same performance).
    """
    global _training_data_loaded
    if _training_data_loaded:
        return
    _training_data_loaded = True

    for pickup in PICKUP_DIR_NAMES:
        for style in STYLE_DIR_NAMES:
            src_dir = os.path.join(INPUT_DIR, pickup, style, "source")
            tgt_dir = os.path.join(INPUT_DIR, pickup, style, "target")
            pairs_by_group[(pickup, style)] = _scan_pair_dir(src_dir, tgt_dir)

    # Legacy piezo layout
    for style, (sdir, tdir) in (("pick", (PICK_SRC_DIR, PICK_TGT_DIR)),
                                ("finger", (FINGER_SRC_DIR, FINGER_TGT_DIR))):
        for pair in _scan_pair_dir(sdir, tdir):
            if pair not in pairs_by_group[("piezo", style)]:
                pairs_by_group[("piezo", style)].append(pair)

    picked_pairs[:] = [p for pu in PICKUP_DIR_NAMES for p in pairs_by_group[(pu, "pick")]]
    finger_pairs[:] = [p for pu in PICKUP_DIR_NAMES for p in pairs_by_group[(pu, "finger")]]
    all_pairs[:] = picked_pairs + finger_pairs

    for src, tgt in all_pairs:
        data_src, _ = sf.read(src, dtype='float32', always_2d=True)
        data_tgt, _ = sf.read(tgt, dtype='float32', always_2d=True)
        info_src = sf.info(src)
        info_tgt = sf.info(tgt)
        if info_src.samplerate != info_tgt.samplerate:
            print(f"Warning: sample-rate mismatch for pair {src} vs {tgt}: {info_src.samplerate} vs {info_tgt.samplerate}")
        audio_cache[src] = data_src
        audio_cache[tgt] = data_tgt
        _file_sr_cache[src] = info_src.samplerate
        _file_sr_cache[tgt] = info_tgt.samplerate

def _file_sr(path):
    sr = _file_sr_cache.get(path)
    if sr is None:
        sr = sf.info(path).samplerate
        _file_sr_cache[path] = sr
    return int(sr)

def _mono_at_sr(path, model_sr):
    """
    Whole-file mono view of a cached file at model_sr. Files are resampled ONCE and cached,
    so training segments avoid per-segment resample_poly edge transients.
    """
    file_sr = _file_sr(path)
    mono = audio_cache[path][:, 0]
    if int(file_sr) == int(model_sr):
        return mono
    key = (path, int(model_sr))
    arr = _resampled_mono_cache.get(key)
    if arr is None:
        arr = resample_to_model_sr(mono, int(file_sr), int(model_sr))
        _resampled_mono_cache[key] = arr
    return arr

def _path_components(path):
    return [p.lower() for p in os.path.normpath(str(path)).split(os.sep)]

def pickup_flag_for_path(path):
    # Directory-component match (a substring match would also hit 'magnetic' inside a file name)
    return PICKUP_MAGNETIC if "magnetic" in _path_components(path) else PICKUP_PIEZO

def style_flag_for_path(path):
    """1.0 for pick takes, 0.0 for finger takes, None if the path names neither style directory."""
    comps = _path_components(path)
    if "pick" in comps:
        return 1.0
    if "finger" in comps:
        return 0.0
    return None

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
    load_training_data()
    if not all_pairs:
        print("ERROR: No training pairs found. Expected files named *_source.wav and *_target.wav "
            "under input/<pickup>/<style>/{source,target} (pickup: piezo|magnetic, style: pick|finger) "
            "or the legacy input/<style>/{source,target} layout.")
        sys.exit(1)
        
def _seed_cond_embed_from_legacy_style_embed(dst_model, src_model, verbose=False):
    """
    Legacy-compat shim: old graphs conditioned FiLM on a Dense named '<prefix>_style_embed'
    (input dim 1: style only); new graphs use '<prefix>_cond_embed' (input dim 2: style +
    pickup). Seed the new kernel's style row from the legacy weights and zero the pickup row,
    so a migrated model behaves IDENTICALLY to the old one at pickup=0 instead of getting a
    randomly initialized conditioning path.
    """
    seeded = 0
    for prefix in ("offline", "realtime"):
        try:
            dst = dst_model.get_layer(f"{prefix}_cond_embed")
            src = src_model.get_layer(f"{prefix}_style_embed")
        except Exception:
            continue
        sw = src.get_weights()  # [(1,E), (E,)]
        dw = dst.get_weights()  # [(2,E), (E,)]
        if len(sw) != 2 or len(dw) != 2:
            continue
        if sw[0].ndim != 2 or dw[0].ndim != 2:
            continue
        if sw[0].shape[1] != dw[0].shape[1] or dw[0].shape[0] != sw[0].shape[0] + 1:
            continue
        if np.shape(sw[1]) != np.shape(dw[1]):
            continue
        k = np.zeros_like(dw[0])
        k[:sw[0].shape[0], :] = sw[0]
        dst.set_weights([k, sw[1]])
        seeded += 1
    if verbose and seeded:
        print(f"[copy_weights_by_name] seeded {seeded} cond_embed layer(s) from legacy style_embed")
    return seeded

def copy_weights_by_name(dst_model, src_model, verbose=False):
    """
    Copy weights layer-by-layer by matching layer names and shapes.
    Returns the number of layers whose weights were copied.
    Leaves non-matching layers untouched.
    Also seeds new-style '<prefix>_cond_embed' layers from legacy '<prefix>_style_embed'
    weights when copying from an old-format model (no-op otherwise).
    """
    src_layers = {l.name: l for l in src_model.layers}
    matched = 0
    for l in dst_model.layers:
        if isinstance(l, FrozenTeacher):
            # The embedded KD teacher is always loaded fresh from the offline model on disk;
            # never overwrite it with the (possibly stale) teacher baked into an old checkpoint.
            continue
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
    matched += _seed_cond_embed_from_legacy_style_embed(dst_model, src_model, verbose=verbose)
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

_minphase_decimator_len_cache = {}

def _minphase_decimator_len(num_taps, factor):
    """
    Actual tap count of the min-phase decimator kernel built by
    StreamingRealtimeIrDownsampleFIR / PickResoDownsampleFIR (scipy's minimum_phase roughly
    halves the prototype length). Probing the real design keeps state declarations exact
    across scipy versions.
    """
    key = (int(num_taps), int(factor), bool(_LEGACY_MINPHASE_HALF))
    L = _minphase_decimator_len_cache.get(key)
    if L is None:
        cutoff = 0.45 / float(factor)
        taps = firwin(int(num_taps), cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
        L = int(len(_to_minimum_phase(taps)))
        _minphase_decimator_len_cache[key] = L
    return L

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
    inputs: [x (B,1), s_prev (B,1)] or [x, s_prev, ref (B,T,C)]
    outputs: [y (B,1), s_next (B,1)],   y = a * s_prev + (1 - a) * x

    'alpha' is the coefficient per 'ref_hop'-sample hop. When a reference tensor is supplied the
    coefficient is rescaled to the ACTUAL hop length T: a = alpha ** (T / ref_hop), so the EMA has
    the same time constant whether the graph is stepped 32 samples at a time (pedal/TFLite) or
    1024 samples at a time (dynamic-T Keras inference). Without 'ref' (legacy graphs) a = alpha.
    """
    def __init__(self, alpha=0.98, ref_hop=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.ref_hop = None if ref_hop is None else int(ref_hop)

    def call(self, inputs):
        if len(inputs) == 3:
            x, s_prev, ref = inputs
        else:
            x, s_prev = inputs
            ref = None
        if ref is not None and self.ref_hop:
            T = tf.cast(tf.shape(ref)[1], x.dtype)
            a = tf.exp(tf.math.log(tf.constant(self.alpha, dtype=x.dtype)) * T / float(self.ref_hop))
        else:
            a = tf.constant(self.alpha, dtype=x.dtype)
        y = a * s_prev + (1.0 - a) * x
        return [y, y]  # y is also the next state

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "ref_hop": self.ref_hop})
        return cfg

@register_keras_serializable()
class StreamingStyleLerp(tf.keras.layers.Layer):
    """
    Expand a per-hop style scalar into a per-sample envelope by linearly
    interpolating from the previous-hop scalar to the current-hop scalar.

    Inputs:
      [ref (B,T,1), style_scalar (B,1), s_prev (B,1,1)]
    Outputs:
      [style_time (B,T,1), s_next (B,1,1)]
    """
    def call(self, inputs):
        ref, style_scalar, s_prev = inputs
        T = tf.shape(ref)[1]

        # prev/current scalars shaped (B,)
        prev = tf.squeeze(s_prev, axis=[1, 2])
        curr = tf.squeeze(style_scalar, axis=-1)

        # r: (1,T) in [0,1], use (T-1) so last sample == curr
        denom = tf.maximum(tf.cast(T - 1, ref.dtype), tf.constant(1.0, dtype=ref.dtype))
        r = tf.cast(tf.range(T), ref.dtype)[None, :] / denom

        # y[n] = prev + r[n]*(curr - prev)
        delta = curr - prev
        y = prev[:, None] + r * delta[:, None]     # (B,T)
        y3 = y[:, :, None]                         # (B,T,1)

        s_next = tf.reshape(curr, (-1, 1, 1))      # (B,1,1)
        return [y3, s_next]

    def get_config(self):
        return super().get_config()
    
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
        taps_min = _to_minimum_phase(taps)
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
                 floor=0.00,
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
                         floor=0.00,
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
                 lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                 cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                 core=(None, None)):
    """
    Tiny causal HF synthesis gated by the analytic pick mask, then style-gated,
    then a runtime HPF safety-net, then final scale. 'core' = loss core for the training-only
    lowband penalty (see *_LOSS_CORE).

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
                    cutoff_hz=cutoff_hz,
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
                            core_start=core[0], core_end=core[1],
                            name=f"{prefix}_pickattack_lowband_penalty")(y)
    return y, patt

def _reson_learn_offline(inp, mask, style_time, core=(None, None)):
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
                            core_start=core[0], core_end=core[1],
                            name="offline_pickreso_lowband_penalty")(y)
    return y, patt

def _reson_learn_realtime(inp, mask, style_time, core=(None, None)):
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
                            core_start=core[0], core_end=core[1],
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

def _crop_time(x, core_start, core_end):
    """
    x[:, core_start:core_end, :] with bounds clamped to the (possibly dynamic) time length, so the
    same layer config is safe on shorter inputs (never yields an empty slice on non-empty input).
    (None, None) → x unchanged.
    """
    if core_start is None and core_end is None:
        return x
    T = tf.shape(x)[1]
    ce = T if core_end is None else tf.minimum(int(core_end), T)
    cs = 0 if core_start is None else tf.minimum(int(core_start), tf.maximum(ce - 1, 0))
    return x[:, cs:ce, :]

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
      outputs: [y_ds (B, ceil(T/N), 1), state_next (B,L-1,1)]
    """
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, kernel_size=REALTIME_IR_FIR_SIZE, use_ir_prototype=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.kernel_size = int(kernel_size)
        self.use_ir_prototype = bool(use_ir_prototype)

    def build(self, input_shape):
        if self.use_ir_prototype:
            k = _ir_prototype_kernel().astype(np.float32)  # (L,1,1), already reversed for conv1d
        else:
            cutoff = 0.45 / float(self.factor)
            taps = firwin(self.kernel_size, cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
            taps_min = _to_minimum_phase(taps)
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
        # Inputs are [x_fullrate, state] where state has (L-1) previous full-rate samples
        x, state = inputs  # x: (B,T,1), state: (B,L-1,1)
        # Concatenate state + new hop
        x_cat = tf.concat([state, x], axis=1)  # (B, L-1+T, 1)
        # Strided causal conv (decimator)
        y_full = tf.nn.conv1d(x_cat, filters=self.kernel, stride=self.factor, padding='VALID')  # (B, ?, 1)

        # Keep only the "new" low-rate samples for this hop:
        # n_new = ceil(T/factor)
        T = tf.shape(x)[1]
        n_new = tf.cast((T + self.factor - 1) // self.factor, tf.int32)
        y = y_full[:, -n_new:, :]  # (B, n_new, 1)

        # Next state is last (L-1) full-rate samples
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
                 floor=0.00,
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

class KerasStreamingRunner:
    """
    Hop-wise driver for a self-contained STREAMING realtime Keras model (static T or dynamic T).

    - discovers the realtime_state_* inputs and their *_next outputs (by producer-layer name,
      with an ordered fallback), zero-initialises them and carries them across hops;
    - steps the model through a tf.function (model.predict() per 32-sample hop costs ~10 ms of
      overhead each, i.e. ~45 min for a 3-minute file);
    - hop == T for static-T models (anything else corrupts the states); dynamic-T models are
      stepped KERAS_STREAM_HOP samples at a time. Style tracking is hop-invariant (see
      StreamingScalarEMA), and every decimator needs hop % 8 == 0;
    - optional output taps (by output index) are collected per hop, e.g. for stems;
    - the step runs on `device` (default CPU): these per-hop graphs are ~1500 tiny ops, and on
      tensorflow-metal the per-op launch latency made a 32-sample hop cost ~60 ms vs ~25 ms on
      the CPU (measured Sept 2026). Pass device=None to use TF's default placement.
    """
    def __init__(self, model, hop=None, device="/CPU:0"):
        self.model = model
        self.device = device
        self.in_names = [_tensor_basename(t) for t in model.inputs]
        piezo = next((t for t in model.inputs if _tensor_basename(t) == "realtime_piezo_input"), None)
        if piezo is None:
            raise RuntimeError("KerasStreamingRunner: model has no 'realtime_piezo_input'.")
        self.T = int(piezo.shape[1]) if piezo.shape[1] is not None else None
        if self.T is None:
            self.hop = int(hop or KERAS_STREAM_HOP)
        else:
            if hop is not None and int(hop) != self.T:
                raise ValueError(f"Streaming model with static T={self.T} requires hop == T (got {hop}).")
            self.hop = self.T
        f = max(REALTIME_IR_DOWNSAMPLE_FACTOR, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
        if self.hop % f != 0:
            raise ValueError(f"hop ({self.hop}) must be a multiple of the decimation factor ({f}).")

        self.state_names = [n for n in self.in_names if n.startswith("realtime_state_")]
        if not self.state_names:
            raise RuntimeError("KerasStreamingRunner: model is not STREAMING (no realtime_state_* inputs).")
        self.state_shapes = {}
        for t in model.inputs:
            nm = _tensor_basename(t)
            if nm in self.state_names:
                self.state_shapes[nm] = (1,
                                         int(t.shape[1]) if t.shape[1] is not None else 0,
                                         int(t.shape[2]) if t.shape[2] is not None else 1)
        self.out_names = _output_names_from_model(model)
        self.out_index = {n: i for i, n in enumerate(self.out_names)}
        self.state_out_idx = {n: self.out_index[n + "_next"] for n in self.state_names
                              if (n + "_next") in self.out_index}
        if len(self.state_out_idx) != len(self.state_names):
            # Ordered fallback: state outputs follow the non-state outputs, in state-input order
            n_non = len(model.outputs) - len(self.state_names)
            if n_non < 2:
                raise RuntimeError("KerasStreamingRunner: could not map state outputs.")
            self.state_out_idx = {n: n_non + i for i, n in enumerate(self.state_names)}
        self.states = None
        self.reset()

        def _step_fn(feed):
            if self.device:
                with tf.device(self.device):
                    return model(feed, training=False)
            return model(feed, training=False)
        self._step = tf.function(_step_fn, reduce_retracing=True)

    def reset(self):
        self.states = {n: np.zeros(s, dtype=np.float32) for n, s in self.state_shapes.items()}

    def run(self, x_mod, knob_feeds, style_true=None, taps=None):
        """
        Stream x_mod (1-D float array) through the model.
          style_true: None → classifier ('auto'); 0.0/1.0 → forced style.
          taps: {tag: output_index} of extra outputs to collect ((T,) or (T,C) per hop).
        Returns {'audio': (N,), 'avg_prob': float|None, 'taps': {tag: (N,...)}}.
        """
        self.reset()
        x_mod = np.asarray(x_mod, dtype=np.float32)
        N = len(x_mod)
        hop = self.hop
        if style_true is None:
            st_true = np.zeros((1, 1), np.float32); st_force = np.zeros((1, 1), np.float32)
        else:
            st_true = np.array([[float(style_true)]], np.float32); st_force = np.ones((1, 1), np.float32)
        const = {k: np.asarray(v, np.float32) for k, v in knob_feeds.items() if k in self.in_names}
        const["realtime_style_true_input"] = st_true
        const["realtime_style_force_input"] = st_force
        taps = taps or {}
        audio, tap_parts = [], {t: [] for t in taps}
        prob_sum, prob_count = 0.0, 0
        pos = 0
        while pos < N:
            core = min(hop, N - pos)
            xb = np.zeros((1, hop, 1), np.float32)
            xb[0, :core, 0] = x_mod[pos:pos + core]
            feed = dict(const)
            feed["realtime_piezo_input"] = xb
            feed.update(self.states)
            outs = self._step(feed)
            outs = [np.asarray(o) for o in outs]
            audio.append(outs[0][0, :core, 0])
            if style_true is None:
                prob_sum += float(outs[1][0, 0]) * core
                prob_count += core
            for tag, idx in taps.items():
                v = outs[idx]
                tap_parts[tag].append(v[0, :core, 0] if (v.ndim == 3 and v.shape[2] == 1) else v[0, :core])
            for n, idx in self.state_out_idx.items():
                self.states[n] = outs[idx]
            pos += core
        cat = lambda parts: np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)
        return {
            "audio": cat(audio),
            "avg_prob": (prob_sum / float(prob_count)) if prob_count > 0 else None,
            "taps": {t: cat(p) for t, p in tap_parts.items()},
        }

def _read_pair_segment(src_path, tgt_path, sample_rate, offset_sec, sec):
    """Read [offset, offset+sec) of a source/target pair as mono arrays at sample_rate."""
    info = sf.info(src_path)
    file_sr = info.samplerate
    start = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0]
    x_mod = x if file_sr == sample_rate else resample_to_model_sr(x, file_sr, sample_rate)
    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0]
        y_mod = y if file_sr == sample_rate else resample_to_model_sr(y, file_sr, sample_rate)
    return x_mod, y_mod

class DemoCallback(Callback):
    """
    Per-epoch listening demo. Offline: dynamic-time graph, single shot. Realtime: the actual
    streaming graph, stepped hop-wise through KerasStreamingRunner (a single-shot pass through
    the streaming graph would apply exactly one EMA update to the style logit and mix pick/finger
    ~50/50 regardless of what the classifier says).
    """
    def __init__(self, sample_rate, demo_checkpoints_out_dir, build_model_fn, build_model_fn_args_dynamic={}, use_safe_weight_copy=False):
        super().__init__()
        assert_has_pairs()
        demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
        self.style_true_flag = style_flag_for_path(demo_src)
        self.pickup_flag = pickup_flag_for_path(demo_src)
        self.src_path = demo_src
        self.tgt_path = demo_tgt
        self.demo_sec = DEMO_SEC
        self.offset_sec = DEMO_OFFSET_SEC
        self.build_model_fn = build_model_fn
        self.build_model_fn_args_dynamic = build_model_fn_args_dynamic
        self.sample_rate = sample_rate
        self.kind = "offline" if int(sample_rate) == int(OFFLINE_SAMPLE_RATE) else "realtime"
        self.every = 1
        self.demos_out_dir = demo_checkpoints_out_dir
        self.use_safe_weight_copy = use_safe_weight_copy
        os.makedirs(self.demos_out_dir, exist_ok=True)
        self.static_model = None
        self.dynamic_model = None
        self.runner = None

    def set_model(self, model):
        super().set_model(model)
        self.static_model = model
        self.dynamic_model = self.build_model_fn(**self.build_model_fn_args_dynamic, dynamic_time=True)
        self._sync_weights()
        if self.kind == "realtime":
            self.runner = KerasStreamingRunner(self.dynamic_model, hop=KERAS_STREAM_HOP)

    def _sync_weights(self):
        if self.use_safe_weight_copy:
            copy_weights_by_name(self.dynamic_model, self.static_model)
        else:
            self.dynamic_model.set_weights(self.static_model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every != 0:
            return
        self._sync_weights()
        kind = self.kind
        x_mod, y_mod = _read_pair_segment(self.src_path, self.tgt_path, self.sample_rate,
                                          self.offset_sec, self.demo_sec)
        knob_feeds = prepare_demo_knob_feeds(kind=kind, use_finetuned=False, pickup=self.pickup_flag)

        if kind == "realtime":
            res = self.runner.run(x_mod, knob_feeds, style_true=None)
            y_pred_pass = res["audio"]; style_pred = res["avg_prob"]
            y_true_pass = self.runner.run(x_mod, knob_feeds, style_true=self.style_true_flag)["audio"] \
                          if self.style_true_flag is not None else None
        else:
            in_names = [_tensor_basename(t) for t in self.dynamic_model.inputs]
            def predict(style_flag):
                feed = {f"{kind}_piezo_input": x_mod[None, :, None].astype(np.float32),
                        f"{kind}_style_true_input": np.array([[0.0 if style_flag is None else style_flag]], np.float32),
                        f"{kind}_style_force_input": np.array([[0.0 if style_flag is None else 1.0]], np.float32),
                        **knob_feeds}
                outs = self.dynamic_model.predict({k: v for k, v in feed.items() if k in in_names},
                                                  batch_size=1, verbose=0)
                return np.asarray(outs[0], np.float32)[0, :, 0], float(np.asarray(outs[1], np.float32)[0, 0])
            y_pred_pass, style_pred = predict(None)
            y_true_pass = predict(self.style_true_flag)[0] if self.style_true_flag is not None else None

        style_pred_name = _style_label(style_pred)
        style_true_name = _style_label(self.style_true_flag) if self.style_true_flag is not None else None

        epoch_dir = os.path.join(self.demos_out_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        sf.write(os.path.join(epoch_dir, "demo_source.wav"), x_mod, self.sample_rate, subtype='FLOAT')
        if y_mod is not None:
            sf.write(os.path.join(epoch_dir, "demo_target.wav"), y_mod, self.sample_rate, subtype='FLOAT')
        sf.write(os.path.join(epoch_dir, "demo_pred_style_pred.wav"), y_pred_pass, self.sample_rate, subtype='FLOAT')
        if y_true_pass is not None:
            sf.write(os.path.join(epoch_dir, "demo_pred_style_true.wav"), y_true_pass, self.sample_rate, subtype='FLOAT')

        p_txt = f"{style_pred:.3f}" if style_pred is not None else "n/a"
        print(f"[DemoCallback] epoch {epoch+1}: predicted style={style_pred_name} ({p_txt}), "
              f"true style={style_true_name}, written to {epoch_dir}")

# ────────────────────────────── LOSSES ───────────────────────────────────────


def _loss_core_from_kind_or_sr(sample_rate):
    """Default loss core (segment sample range that enters the losses) for a model rate."""
    return OFFLINE_LOSS_CORE if int(sample_rate) == OFFLINE_SAMPLE_RATE else REALTIME_LOSS_CORE

@register_keras_serializable()
class CombinedLoss(Loss):
    """
    Full training loss. 'core' = (start, end) sample range of the segment that enters the loss
    (defaults to the kind's *_LOSS_CORE); pass core=(None, None) to use the whole segment.
    """
    def __init__(self, sample_rate, extra_small_cplx=False, core=None, name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.extra_small_cplx = bool(extra_small_cplx)
        core = _loss_core_from_kind_or_sr(sample_rate) if core is None else tuple(core)
        self.core = (None if core[0] is None else int(core[0]), None if core[1] is None else int(core[1]))
    def call(self, y_true, y_pred):
        return combined_loss(y_true, y_pred, self.sample_rate,
                             extra_small_cplx=self.extra_small_cplx, core=self.core)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "extra_small_cplx": self.extra_small_cplx,
            "core": list(self.core)
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
                 core=None,
                 name="finetune_combined_loss_lite",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = int(sample_rate)
        core = _loss_core_from_kind_or_sr(sample_rate) if core is None else tuple(core)
        self.core = (None if core[0] is None else int(core[0]), None if core[1] is None else int(core[1]))
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
            "ulb_order": self.ulb_order,
            "core": list(self.core)
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
        cs, ce = self.core
        gate_env = gate_weight_dynamic(y_true, self.sample_rate)  # (B,T,1) in [0,1], warmed up on the full segment
        def core_mean(m):
            return tf.reduce_mean(_crop_time(m, cs, ce), axis=1, keepdims=True)  # (B,1,1)
        td = core_mean(waveform_l1_loss(y_true, y_pred) * (0.2 + 0.8 * gate_env))
        hf = core_mean(hf_loss(y_true, y_pred) * (1.0 + self.onset_gain * gate_env))
        ulb = core_mean(self._ulb(y_true, y_pred))
        yt_c, yp_c, g_c = _crop_time(y_true, cs, ce), _crop_time(y_pred, cs, ce), _crop_time(gate_env, cs, ce)
        spec = self._mrstft(yt_c, yp_c, g_c) if self.stft_scales else 0.0
        env = rms_env_loss(yt_c, yp_c) if self.env_w != 0.0 else 0.0

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

def _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=False, core=(None, None)):
    """
    All CombinedLoss terms as (B,1,1) tensors. Per-sample terms are computed on the FULL segment
    (their internal filters / onset gate are then warmed up by the context) and averaged over the
    loss core only; spectral terms are computed on the cropped core. Shared by combined_loss and
    the loss-weight calibration so both see exactly the same quantities.
    """
    cs, ce = core
    gate_env = gate_weight_dynamic(y_true, sample_rate)

    def core_mean(m):
        return tf.reduce_mean(_crop_time(m, cs, ce), axis=1, keepdims=True)  # (B,1,1)

    td = core_mean(waveform_l1_loss(y_true, y_pred))
    hf = core_mean(hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * gate_env))
    d2 = core_mean(second_diff_l1_onset(y_true, y_pred, onset_env=gate_env))
    bl = core_mean(resonance_band_loss(y_true, y_pred, sample_rate))
    ulb = core_mean(lowband_error_penalty(y_true, y_pred, sample_rate,
                                          cutoff_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                          order=ULTRALOW_MAIN_ORDER))

    yt_c = _crop_time(y_true, cs, ce)
    yp_c = _crop_time(y_pred, cs, ce)
    g_c  = _crop_time(gate_env, cs, ce)
    spec = mrstft_mag_loss_weighted_onset(yt_c, yp_c, sample_rate,
                                          split_hz=hp_cutoff_for_sr(sample_rate),
                                          high_boost=STFT_HIGH_BOOST,
                                          onset_env=g_c, onset_gain=STFT_HF_ONSET_GAIN)
    mel = log_mel_loss(yt_c, yp_c, sample_rate)
    env = rms_env_loss(yt_c, yp_c)
    cplx = complex_stft_loss_onset(yt_c, yp_c, sample_rate, n_fft=256, hop=64, win=256, onset_env=g_c)
    if extra_small_cplx:
        cplx_64 = complex_stft_loss_onset(yt_c, yp_c, sample_rate, n_fft=64, hop=16, win=64, onset_env=g_c)
        cplx = 0.5 * (cplx + cplx_64)

    return {'l1': td, 'hf': hf, 'stft': spec, 'mel': mel, 'env': env,
            'cplx': cplx, 'd2': d2, 'resbp_bl': bl, 'ulb': ulb}

@register_keras_serializable()
def combined_loss(y_true, y_pred, sample_rate, extra_small_cplx=False, core=(None, None)):
    t = _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=extra_small_cplx, core=core)
    return (L1_WEIGHT       * t['l1']
          + HF_WEIGHT       * t['hf']
          + STFT_WEIGHT     * t['stft']
          + MEL_WEIGHT      * t['mel']
          + RMS_ENV_WEIGHT  * t['env']
          + COMPLEX_STFT_WEIGHT * t['cplx']
          + D2_WEIGHT       * t['d2']
          + RESBP_BAND_LOSS_WEIGHT * t['resbp_bl']
          + ULTRALOW_MAIN_WEIGHT   * t['ulb'])

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

def _slice_zero_padded(arr, start, end):
    """arr[start:end] with zeros wherever the range falls outside the array."""
    n = int(end) - int(start)
    out = np.zeros((n,), dtype=np.float32)
    a = max(0, int(start)); b = min(len(arr), int(end))
    if b > a:
        out[a - int(start): b - int(start)] = arr[a:b]
    return out

def kd_teacher_segment_len(seg_len_rt, factor=2):
    """Length of the teacher (96k) copy of a realtime training segment incl. teacher context."""
    return int(KD_TEACHER_LEFT_CTX_SAMPLES + int(factor) * int(seg_len_rt) + KD_TEACHER_RIGHT_CTX_SAMPLES)

def segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks,
                                       factor=2, min_amp=SEGMENT_GENERATOR_MIN_AMP_THRESH):
    """
    Simple dual-rate generator backed by already-resampled arrays (48k and 96k).
    - seg_len_rt: segment length at realtime rate (48k).
    - x48, y48: mono float32 arrays at 48k.
    - x96: mono float32 array at 96k (teacher input, extended by the KD teacher context so the
      non-causal teacher is fully contexted over the student's segment).
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
        xs96 = _slice_zero_padded(x96, s96 - KD_TEACHER_LEFT_CTX_SAMPLES,
                                  s96 + L96 + KD_TEACHER_RIGHT_CTX_SAMPLES)

        # Pad to exact lengths if needed
        if len(xs48) < seg_len_rt:
            p = seg_len_rt - len(xs48)
            xs48 = np.pad(xs48, (0, p))
            ys48 = np.pad(ys48, (0, p))

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
        tf.TensorSpec((seg_len_rt, 1), tf.float32),                            # realtime 48k
        tf.TensorSpec((kd_teacher_segment_len(seg_len_rt, factor), 1), tf.float32)  # offline 96k (+ctx)
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

    # The pickup input adapter feeds every branch, so its context adds to all of them.
    lc_adapt = pickup_adapter_left_context() if any(
        l.name.startswith("realtime_pickup_adapt") for l in model.layers) else 0

    left_ctx = max(lc_main, lc_ir, lc_reso, lc_patt, lc_col) + lc_adapt

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

def _select_stems_model_path(kind, use_finetuned=False, prefer_static=False):
    """
    Select a saved Keras model filepath for stems/demos.
    - kind: 'offline' or 'realtime'
    - use_finetuned: prefer finetuned artifacts if present
    - prefer_static: if True, prefer 'static' artifacts; else prefer 'dynamic' artifacts
      Note: realtime static is the streaming T={REALTIME_TFLITE_T} graph; offline static uses fixed T train graph.
    """
    if kind == "offline":
        # Offline artifacts
        finetuned_dyn  = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        finetuned_stat = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
        baseline_dyn   = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        baseline_stat  = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)

        if use_finetuned:
            if prefer_static and os.path.exists(finetuned_stat):
                return finetuned_stat
            if os.path.exists(finetuned_dyn):
                return finetuned_dyn
            # fallback to baseline if finetuned missing
            if prefer_static and os.path.exists(baseline_stat):
                return baseline_stat
            if os.path.exists(baseline_dyn):
                return baseline_dyn
            return None

        # baseline
        if prefer_static and os.path.exists(baseline_stat):
            return baseline_stat
        if os.path.exists(baseline_dyn):
            return baseline_dyn
        return None

    elif kind == "realtime":
        # Realtime artifacts (streaming)
        finetuned_dyn  = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        finetuned_stat = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        baseline_dyn   = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        baseline_stat  = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)

        if use_finetuned:
            if prefer_static and os.path.exists(finetuned_stat):
                return finetuned_stat
            if os.path.exists(finetuned_dyn):
                return finetuned_dyn
            # fallback to baseline if finetuned missing
            if prefer_static and os.path.exists(baseline_stat):
                return baseline_stat
            if os.path.exists(baseline_dyn):
                return baseline_dyn
            return None

        # baseline
        if prefer_static and os.path.exists(baseline_stat):
            return baseline_stat
        if os.path.exists(baseline_dyn):
            return baseline_dyn
        return None

    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

@register_keras_serializable()
def pre_emphasis(x, alpha=HP_PRE_EMPHASIS_HF):
    """
    Simple 1st-order high-pass (pre-emphasis): y[n] = x[n] - alpha * x[n-1]
    x shape: (B, T, C)
    """
    x_shift = tf.pad(x, [[0,0],[1,0],[0,0]])[:, :-1, :]
    return x - alpha * x_shift

def _fit_len(seg, seg_len):
    if len(seg) < seg_len:
        return np.pad(seg, (0, seg_len - len(seg)))
    return seg[:seg_len]

def segment_generator(seg_len, chunks, is_pick_chunks, model_sample_rate, pickup_flag=PICKUP_PIEZO):
    """
    Yields (x, (y_audio, style_flag, pickup_flag)) where x and y_audio are (seg_len, 1) at
    model_sample_rate. Slices whole-file pre-resampled arrays (see _mono_at_sr), which avoids
    per-segment resample edge transients and repeated resample_poly work.
    """
    style_flag = 1.0 if is_pick_chunks else 0.0
    pickup_flag = float(pickup_flag)
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = _file_sr(src)

        x_all = _mono_at_sr(src, model_sample_rate)
        y_all = _mono_at_sr(tgt, model_sample_rate)

        # Chunk bounds converted from file-rate to model-rate samples
        ms = (cstart * int(model_sample_rate)) // file_sr
        md = (cdur   * int(model_sample_rate)) // file_sr
        if md <= 0:
            continue

        off = random.randrange(max(md - seg_len, 1))
        s = ms + off
        xs = x_all[s:s + seg_len]
        ys = y_all[s:s + seg_len]

        if xs.size == 0 or np.max(np.abs(xs)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        xs = _fit_len(xs, seg_len)
        ys = _fit_len(ys, seg_len)

        yield xs.astype('float32')[:, None], \
              (ys.astype('float32')[:, None], np.float32(style_flag), np.float32(pickup_flag))

def segment_generator_dual(seg_len_realtime,
                           chunks,
                           is_pick_chunks,
                           realtime_sample_rate=REALTIME_SAMPLE_RATE,
                           offline_sample_rate=OFFLINE_SAMPLE_RATE,
                           pickup_flag=PICKUP_PIEZO):
    """
    Yields ((x_realtime, x_offline), (y_audio_realtime, style_flag, pickup_flag)).

    Supports any integer ratio offline_sample_rate : realtime_sample_rate. The realtime and
    offline segments are sliced from whole-file pre-resampled arrays with exactly aligned
    starts (s_offline = factor * s_realtime).
    """
    if offline_sample_rate % realtime_sample_rate != 0:
        raise ValueError(f"offline_sample_rate ({offline_sample_rate}) must be an integer multiple of realtime_sample_rate ({realtime_sample_rate}).")

    factor = offline_sample_rate // realtime_sample_rate
    seg_len_offline = factor * seg_len_realtime
    style_flag = 1.0 if is_pick_chunks else 0.0
    pickup_flag = float(pickup_flag)

    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = _file_sr(src)

        x48 = _mono_at_sr(src, realtime_sample_rate)
        y48 = _mono_at_sr(tgt, realtime_sample_rate)
        x96 = _mono_at_sr(src, offline_sample_rate)

        ms48 = (cstart * int(realtime_sample_rate)) // file_sr
        md48 = (cdur   * int(realtime_sample_rate)) // file_sr
        if md48 <= 0:
            continue

        off48 = random.randrange(max(md48 - seg_len_realtime, 1))
        s48 = ms48 + off48
        s96 = s48 * factor

        xs48 = x48[s48:s48 + seg_len_realtime]
        ys48 = y48[s48:s48 + seg_len_realtime]
        # Teacher copy: the same span at 96k plus the (non-causal) teacher's own left/right context
        xs96 = _slice_zero_padded(x96, s96 - KD_TEACHER_LEFT_CTX_SAMPLES,
                                  s96 + seg_len_offline + KD_TEACHER_RIGHT_CTX_SAMPLES)

        if xs48.size == 0 or np.max(np.abs(xs48)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        xs48 = _fit_len(xs48, seg_len_realtime)
        ys48 = _fit_len(ys48, seg_len_realtime)

        yield (xs48.astype('float32')[:, None], xs96.astype('float32')[:, None]), \
              (ys48.astype('float32')[:, None], np.float32(style_flag), np.float32(pickup_flag))

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

def analyze_loss_scales(model, batch, sample_rate, extra_small_cplx=False, core=None):
    """Mean of every CombinedLoss term on one batch, computed exactly as the loss computes them."""
    y_true, y_pred = _extract_audio_io(model, batch)
    core = _loss_core_from_kind_or_sr(sample_rate) if core is None else core
    terms = _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=extra_small_cplx, core=core)
    return {k: float(tf.reduce_mean(v).numpy()) for k, v in terms.items()}

def calibrate_loss_weights(model, dataset, sample_rate, n_batches=5,
                                      normalize_keys=('l1','stft','mel','env','hf','resbp_bl'),
                                      declared=None, extra_small_cplx=False, core=None):
    if declared is None:
        declared = {
            'l1': L1_WEIGHT, 'stft': STFT_WEIGHT, 'mel': MEL_WEIGHT, 'env': RMS_ENV_WEIGHT, 'hf': HF_WEIGHT,
            'resbp_bl': RESBP_BAND_LOSS_WEIGHT  # NEW
        }
    sums = {k:0.0 for k in normalize_keys}
    count = 0
    for batch in dataset.take(n_batches):
        sc = analyze_loss_scales(model, batch, sample_rate, extra_small_cplx=extra_small_cplx, core=core)
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


def compute_offline_total_lookahead_samples():
    """
    Total right context of the offline model. OFFLINE_LOOKAHEAD_SAMPLES bounds the look-ahead
    of EACH layer (see compute_pads), and right context is additive across layers, so the
    total is the sum of per-layer right pads — NOT the per-layer bound. Under-reserving this
    at block boundaries truncates real look-ahead and causes audible seams.
    """
    total = 0
    for _ in range(OFFLINE_PRE_BLOCKS):
        total += min(OFFLINE_LOOKAHEAD_SAMPLES, (OFFLINE_PRE_KERNEL - 1) * 1)
    for d in OFFLINE_DILATIONS:
        total += min(OFFLINE_LOOKAHEAD_SAMPLES, (OFFLINE_KERNEL_SIZE - 1) * d)
    return int(total)

def _get_model_context_for_streaming(model_sample_rate: int, model=None):
    """
    Returns (left_ctx_model, right_ctx_model) in samples at the model's sample-rate.
    For realtime, if a model is provided, compute branch-aware left context automatically.
    """
    if model_sample_rate == OFFLINE_SAMPLE_RATE:
        # Left = receptive field - 1 (causal extent) + input adapter,
        # Right = SUM of per-layer bounded look-aheads (additive across layers)
        return (int(OFFLINE_RECEPTIVE_FIELD - 1) + pickup_adapter_left_context(),
                compute_offline_total_lookahead_samples())
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
    knob_feeds = prepare_demo_knob_feeds(kind, use_finetuned=use_finetuned,
                                         pickup=pickup_flag_for_path(path_src))
    style_true_flag = style_flag_for_path(path_src)

    if kind == "realtime":
        # Streaming graphs need their state inputs fed and carried hop-to-hop; a windowed
        # predict() call is not possible (Keras refuses a feed with missing inputs).
        runner = KerasStreamingRunner(model)
        x_mod, y_mod = _read_pair_segment(path_src, path_target if not only_prediction else None,
                                          sample_rate, offset_sec, sec)
        res = runner.run(x_mod, knob_feeds, style_true=None)
        style_pred_path = os.path.join(out_dir, prediction_name + "_style_pred.wav")
        sf.write(style_pred_path, res["audio"], sample_rate, subtype='FLOAT')
        if style_true_flag is not None:
            y_true = runner.run(x_mod, knob_feeds, style_true=style_true_flag)["audio"]
            sf.write(os.path.join(out_dir, prediction_name + "_style_true.wav"), y_true, sample_rate, subtype='FLOAT')
        if not only_prediction:
            sf.write(os.path.join(out_dir, "demo_source.wav"), x_mod, sample_rate, subtype='FLOAT')
            if y_mod is not None:
                sf.write(os.path.join(out_dir, "demo_target.wav"), y_mod, sample_rate, subtype='FLOAT')
        p = res["avg_prob"]
        print(f"✅ Demo saved to {style_pred_path}. Predicted style (avg): {_style_label(p)} "
              f"({p:.4f})" if p is not None else f"✅ Demo saved to {style_pred_path}.")
        return

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    g_norm = 1.0

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

        outs_pred = model.predict(feed, batch_size=1, verbose=0)
        if not isinstance(outs_pred, (list, tuple)):
            outs_pred = [outs_pred]
        y_inf_pred = np.asarray(outs_pred[0], np.float32)[0, :, 0]
        style_pred_out = np.asarray(outs_pred[1], np.float32)
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
            outs_true = model.predict(feed_true, batch_size=1, verbose=0)
            if not isinstance(outs_true, (list, tuple)):
                outs_true = [outs_true]
            y_inf_true = np.asarray(outs_true[0], np.float32)[0, :, 0]
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
h_min = _to_minimum_phase(h_lin)
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)

# For tf.nn.conv1d (cross-correlation), use the time-reversed kernel to implement convolution
_ir_fir_kernel = h_min[::-1].reshape(-1, 1, 1).astype(np.float32)

# Legacy (pre-Sept-2026) variant: half taps, sqrt magnitude — only for loading old .keras files
with legacy_minphase_filters():
    _h_min_legacy = _to_minimum_phase(h_lin)
_h_min_legacy /= np.sum(_h_min_legacy)
_ir_fir_kernel_legacy = _h_min_legacy.astype(np.float32)[::-1].reshape(-1, 1, 1)

def _ir_prototype_kernel():
    return _ir_fir_kernel_legacy if _LEGACY_MINPHASE_HALF else _ir_fir_kernel

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
            except Exception:
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
        # Trainability is a LAYER-level flag: a variable created with add_weight(trainable=False)
        # can never be re-enabled by `layer.trainable = True` in Keras 3 (it lives in the
        # non-trainable list), which silently defeated set_trainable_layers(enable_style_blend).
        super().__init__(trainable=bool(trainable), **kwargs)
        self.init_alpha = float(init_alpha)
        self._trainable_alpha = bool(trainable)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_alpha),
            trainable=True
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
    def __init__(self, alpha, cutoff_hz=2800.0, order=129, sample_rate=REALTIME_SAMPLE_RATE, window="hamming",
                 core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self.sample_rate = int(sample_rate)
        self.window = str(window)
        # Only the fully-contexted core of a training segment is penalised (see *_LOSS_CORE)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

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
        loss = tf.reduce_mean(tf.abs(_crop_time(lp, self.core_start, self.core_end)))
        self.add_loss(self.alpha_var * loss)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'alpha': float(self.init_alpha),
            'cutoff_hz': self.cutoff_hz,
            'order': self.order,
            'sample_rate': self.sample_rate,
            'window': self.window,
            'core_start': self.core_start,
            'core_end': self.core_end
        })
        return cfg
    
# ─── Knowledge-distillation layers (realtime training graph only) ────────────
# These attach annealed KD losses (see KDAlphaScheduler) and return the student tensor
# unchanged, so they can be wired in-line on the student's output path. The teacher runs at
# `factor`× the student rate and is avg-pooled down — adequate for a soft target.

def _kd_align(student, teacher, factor, core_start, core_end):
    """
    Avg-pool the teacher tensor (B, factor*T, C) to the student rate, trim both to a common length
    and crop both to the loss core [core_start, core_end) (student-rate samples).
    """
    t = tf.nn.avg_pool1d(teacher, ksize=factor, strides=factor, padding="SAME")
    n = tf.minimum(tf.shape(student)[1], tf.shape(t)[1])
    s = _crop_time(student[:, :n, :], core_start, core_end)
    t = _crop_time(t[:, :n, :], core_start, core_end)
    return s, t


@register_keras_serializable()
class KDWaveformDistill(Layer, AdjustableAlphaMixin):
    """
    L1 KD between a student signal (B,T,1) and a teacher signal (B,factor*T,1), evaluated on the
    loss core only. Returns the student unchanged.
    """
    def __init__(self, alpha, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs):
        student, teacher = inputs
        s, t = _kd_align(student, teacher, self.factor, self.core_start, self.core_end)
        loss = tf.reduce_mean(tf.abs(s - t))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "factor": self.factor,
                "core_start": self.core_start, "core_end": self.core_end, **super().get_config()}


@register_keras_serializable()
class KDSTFTDistill(Layer, AdjustableAlphaMixin):
    """
    Multi-resolution STFT-magnitude L1 KD between student audio (B,T,1) and teacher audio
    (B,factor*T,1), evaluated on the loss core only. Returns the student unchanged.
    """
    SCALES = ((256, 64), (1024, 256))  # (win/n_fft, hop)

    def __init__(self, alpha, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        self._wins = [tf.constant(tf.reshape(tf.signal.hann_window(w, periodic=True), (1, 1, w)))
                      for (w, _) in self.SCALES]
        super().build(input_shape)

    def call(self, inputs):
        student, teacher = inputs
        s3, t3 = _kd_align(student, teacher, self.factor, self.core_start, self.core_end)
        s = tf.squeeze(s3, axis=-1)
        t = tf.squeeze(t3, axis=-1)

        losses = []
        for (win, hop), w in zip(self.SCALES, self._wins):
            fs = tf.signal.frame(s, frame_length=win, frame_step=hop, pad_end=True) * w
            ft = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True) * w
            mag_s = tf.abs(tf.signal.rfft(fs, fft_length=[win]))
            mag_t = tf.abs(tf.signal.rfft(ft, fft_length=[win]))
            losses.append(tf.reduce_mean(tf.abs(mag_s - mag_t)))
        loss = tf.add_n(losses) / float(len(losses))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "factor": self.factor,
                "core_start": self.core_start, "core_end": self.core_end, **super().get_config()}


@register_keras_serializable()
class KDFeatureDistill(Layer, AdjustableAlphaMixin):
    """
    Feature-space KD: projects student features (B,T,Cs) to the teacher width with a trainable
    1×1 conv, avg-pools teacher features (B,factor*T,Ct) down to the student rate, and adds an
    L1 loss on the loss core. Returns the student features unchanged.
    """
    def __init__(self, alpha, teacher_channels, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.teacher_channels = int(teacher_channels)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)
        self.proj = Conv1D(self.teacher_channels, 1, padding="same",
                           name=f"{self.name}_proj")

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs):
        student, teacher = inputs
        p = self.proj(student)
        s, t = _kd_align(p, teacher, self.factor, self.core_start, self.core_end)
        loss = tf.reduce_mean(tf.abs(s - t))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "teacher_channels": self.teacher_channels,
                "factor": self.factor, "core_start": self.core_start, "core_end": self.core_end,
                **super().get_config()}


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
class GaussianWeights1DLayer(tf.keras.layers.Layer):
    def __init__(self, centers, alpha=0.002, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._centers_np = np.asarray(centers, dtype=np.float32).reshape(1, -1)
        self.alpha = float(alpha)

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=self._centers_np.shape,
            initializer=tf.keras.initializers.Constant(self._centers_np),
            trainable=False
        )
        super().build(input_shape)

    def call(self, knob_scalar):
        # knob_scalar: (B,1), centers: (1,N)
        x = tf.abs(knob_scalar - self.centers)            # (B,N)
        w = tf.exp(-self.alpha * x)                       # (B,N)
        w = w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-8)
        return w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "centers": self._centers_np.tolist(),
            "alpha": self.alpha
        })
        return cfg


@register_keras_serializable()
class OuterWeightsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        w_a, w_b = inputs  # w_a: (B,Na), w_b: (B,Nb)
        a = tf.expand_dims(w_a, axis=2)  # (B,Na,1)
        b = tf.expand_dims(w_b, axis=1)  # (B,1,Nb)
        w = a * b                        # (B,Na,Nb)
        shape = tf.shape(w)
        return tf.reshape(w, [shape[0], shape[1] * shape[2]])  # (B, Na*Nb)


@register_keras_serializable()
class ClampScalarLayer(tf.keras.layers.Layer):
    def __init__(self, lo, hi, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lo = float(lo)
        self.hi = float(hi)

    def call(self, x):
        return tf.clip_by_value(x, self.lo, self.hi)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lo": self.lo, "hi": self.hi})
        return cfg


@register_keras_serializable()
class DbToLinScalarLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.pow(10.0, x / 20.0)


@register_keras_serializable()
class AddScalarLayer(tf.keras.layers.Layer):
    def __init__(self, value, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = float(value)

    def call(self, x):
        return x + self.value

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg


@register_keras_serializable()
class ReduceSumAxis2Layer(tf.keras.layers.Layer):
    def __init__(self, keepdims=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.keepdims = bool(keepdims)

    def call(self, x):
        return tf.reduce_sum(x, axis=2, keepdims=self.keepdims)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"keepdims": self.keepdims})
        return cfg


@register_keras_serializable()
def gaussian_weights_1d(knob_scalar, centers, alpha=0.002):
    return GaussianWeights1DLayer(centers=centers, alpha=alpha)(knob_scalar)

@register_keras_serializable()
def outer_weights(w_a, w_b):
    return OuterWeightsLayer()([w_a, w_b])

@register_keras_serializable()
def clamp_scalar(x, lo, hi):
    return ClampScalarLayer(lo=lo, hi=hi)(x)

@register_keras_serializable()
def db_to_lin_scalar(x):
    return DbToLinScalarLayer()(x)

@register_keras_serializable()
def phase_polarity_gain(switch01):
    """
    switch01: (B,1), 0 → +1, 1 → -1. 
    """
    s = tf.clip_by_value(switch01, 0.0, 1.0)
    return 1.0 - 2.0 * s

@register_keras_serializable()
def _lp_rev_padded(sr, cut_hz, target_len, window='hamming'):
    """
    Build a linear-phase LPF (reversed taps for conv1d), exact length target_len.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    wn = float(cut_hz) / (float(sr) / 2.0)
    taps = firwin(nt, cutoff=wn, window=window, pass_zero=True).astype(np.float32)
    taps = taps / np.maximum(np.sum(taps), 1e-12)
    k_rev = taps[::-1].reshape(nt, 1, 1).astype(np.float32)
    return k_rev

@register_keras_serializable()
class FixedFIRBank(tf.keras.layers.Layer):
    """
    Non-streaming, fixed FIR bank: taps shape (L,1,K) reversed for conv1d (cross-correlation).
    Outputs (B,T,K). Left-padded by (L-1).

    Keras-serialization-friendly:
      - taps_3d is optional. If absent, we create a zero-initialized (L,1,K) kernel and rely on
        weight restoration from the saved model. This allows load_model to succeed even when only
        L and K are present in config (the usual case).
      - If taps_3d is provided (during graph construction), we use it as the initializer.
    """
    def __init__(self, taps_3d=None, L=None, K=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._taps_init = None
        # L and K may be None when building brand-new layers; they are present when deserializing
        self.L = None if L is None else int(L)
        self.K = None if K is None else int(K)

        if taps_3d is not None:
            k = np.asarray(taps_3d, dtype=np.float32)
            if k.ndim != 3 or k.shape[1] != 1:
                raise ValueError(f"taps_3d must have shape (L,1,K); got {k.shape}")
            self._taps_init = k
            self.L = int(k.shape[0])
            self.K = int(k.shape[2])

    def build(self, input_shape):
        if self._taps_init is not None:
            shape = self._taps_init.shape
            initializer = tf.keras.initializers.Constant(self._taps_init)
        else:
            # When loading from a saved model, weights will be restored after construction.
            # Create a kernel with the correct shape so weight loading can succeed.
            if self.L is None or self.K is None:
                # As a last resort, default to (3,1,1) (will be overwritten by weight loading).
                L = 3 if self.L is None else int(self.L)
                K = 1 if self.K is None else int(self.K)
            else:
                L = int(self.L)
                K = int(self.K)
            shape = (L, 1, K)
            initializer = tf.keras.initializers.Zeros()

        self.kernel = self.add_weight(
            name="firbank_taps",
            shape=shape,
            initializer=initializer,
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B,T,1) → pad left by (L-1)
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        k = self.K if self.K is not None else (int(self.kernel.shape[2]) if hasattr(self, "kernel") else None)
        return (b, t, k)

    def get_config(self):
        # We only need to serialize L and K. taps are stored as weights.
        cfg = super().get_config()
        cfg.update({
            "L": self.L,
            "K": self.K
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
    key = (int(sample_rate), float(cutoff_hz), int(order), bool(_LEGACY_MINPHASE_HALF))
    k_np = _gate_hp_kernel_cache.get(key)
    if k_np is None:
        taps = firwin(
        order,
        cutoff=cutoff_hz / (sample_rate / 2.0),
        window="hamming",
        pass_zero=False
        ).astype(np.float32)
        taps_min = _to_minimum_phase(taps)
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
    key = (int(sample_rate), float(f_lo), float(f_hi), int(order), str(window), bool(_LEGACY_MINPHASE_HALF))
    k_np = _bp_kernel_cache.get(key)
    if k_np is None:
        nyq = float(sample_rate) / 2.0
        lo  = float(f_lo) / nyq
        hi  = float(f_hi) / nyq
        taps = firwin(order, [lo, hi], pass_zero=False, window=window).astype(np.float32)
        # Convert to minimum-phase (same magnitude response as the linear-phase design)
        taps_min = _to_minimum_phase(taps)
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
        defaults = offline_knob_defaults()
        if use_finetuned:
            return {
                **defaults,
                "pickattack": _db_to_lin(FINETUNE_OFFLINE_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(FINETUNE_OFFLINE_PICK_RESO_DEMO_BOOST_DB)
            }
        else:
            return {
                **defaults,
                "pickattack": _db_to_lin(OFFLINE_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(OFFLINE_PICK_RESO_DEMO_BOOST_DB)
            }
    else:
        defaults = realtime_knob_defaults()
        if use_finetuned:
            return {
                **defaults,
                "pickattack": _db_to_lin(FINETUNE_REALTIME_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(FINETUNE_REALTIME_PICK_RESO_DEMO_BOOST_DB),
                "ir": _db_to_lin(FINETUNE_REALTIME_IR_DEMO_BOOST_DB)
            }
        else:
            return {
                **defaults,
                "pickattack": _db_to_lin(REALTIME_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(REALTIME_PICK_RESO_DEMO_BOOST_DB),
                "ir": _db_to_lin(REALTIME_IR_DEMO_BOOST_DB)
            }

def prepare_training_knob_feeds(kind, piezo_input, style_true, style_force, pickup=None):
    b = tf.shape(piezo_input)[0]
    if pickup is None:
        pickup = tf.zeros((b, 1), tf.float32)

    if kind == "offline":
        knobs = offline_knob_defaults()
        return {
            'offline_piezo_input': piezo_input,
            'offline_style_true_input': style_true,
            'offline_style_force_input': style_force,
            'offline_pickup_input': pickup,
            'offline_knob_main':       tf.ones((b,1), tf.float32) * knobs["main"],
            'offline_knob_pickattack': tf.ones((b,1), tf.float32) * knobs["pickattack"],
            'offline_knob_pickreso':   tf.ones((b,1), tf.float32) * knobs["pickreso"],
            'offline_knob_drywet':     tf.ones((b,1), tf.float32) * knobs["dry_pct"],
            'offline_knob_outgain':    tf.ones((b,1), tf.float32) * knobs["outgain"],
            'offline_knob_phaseinv':   tf.ones((b,1), tf.float32) * knobs["phaseinv"],
        }
    else:
        knobs = realtime_knob_defaults()
        return {
            'realtime_piezo_input': piezo_input,
            'realtime_style_true_input': style_true,
            'realtime_style_force_input': style_force,
            'realtime_pickup_input': pickup,
            'realtime_knob_main':       tf.ones((b,1), tf.float32) * knobs["main"],
            'realtime_knob_pickattack': tf.ones((b,1), tf.float32) * knobs["pickattack"],
            'realtime_knob_pickreso':   tf.ones((b,1), tf.float32) * knobs["pickreso"],
            'realtime_knob_ir':         tf.ones((b,1), tf.float32) * knobs["ir"],
            'realtime_knob_drywet':     tf.ones((b,1), tf.float32) * knobs["dry_pct"],
            'realtime_knob_outgain':    tf.ones((b,1), tf.float32) * knobs["outgain"],
            'realtime_knob_phaseinv':   tf.ones((b,1), tf.float32) * knobs["phaseinv"],
        }

def _legacy_master_eq_knob_feeds(kind):
    """
    Neutral values for the in-graph master-EQ knob inputs that models saved before Sept 2026 still
    have. Every consumer filters feeds down to the model's actual inputs, so current (EQ-free)
    models simply ignore these keys, while legacy models get a transparent EQ.
    """
    d = master_eq_defaults()
    m = {"lc_hz": "lc_hz", "lc_slope": "lc_slope_dbpoct", "ls_gain_db": "ls_gain_db", "ls_hz": "ls_hz",
         "ls_slope": "ls_slope_dbpoct", "mid_gain_db": "mid_gain_db", "mid_hz": "mid_hz", "mid_q": "mid_q",
         "hs_gain_db": "hs_gain_db", "hs_hz": "hs_hz", "hs_slope": "hs_slope_dbpoct"}
    return {f"{kind}_knob_{suffix}": np.array([[float(d[k])]], dtype=np.float32) for k, suffix in m.items()}

def prepare_demo_knob_feeds(kind, use_finetuned=False, pickattack=None, pickreso=None, ir=None, main=None, pickup=None):
    knobs = _resolve_demo_knobs(kind, use_finetuned=use_finetuned)
    pickup_val = PICKUP_PIEZO if pickup is None else float(pickup)

    feeds = dict(_legacy_master_eq_knob_feeds(kind))
    if kind == "realtime":
        feeds["realtime_pickup_input"]    = np.array([[pickup_val]], dtype=np.float32)
        feeds["realtime_knob_main"]       = np.array([[main if main is not None else knobs["main"]]], dtype=np.float32)
        feeds["realtime_knob_pickattack"] = np.array([[pickattack if pickattack is not None else knobs["pickattack"]]], dtype=np.float32)
        feeds["realtime_knob_pickreso"]   = np.array([[pickreso if pickreso is not None else knobs["pickreso"]]], dtype=np.float32)
        feeds["realtime_knob_ir"]         = np.array([[ir if ir is not None else knobs["ir"]]], dtype=np.float32)
        feeds["realtime_knob_drywet"]     = np.array([[knobs["dry_pct"]]], dtype=np.float32)
        feeds["realtime_knob_outgain"]    = np.array([[knobs["outgain"]]], dtype=np.float32)
        feeds["realtime_knob_phaseinv"]   = np.array([[knobs["phaseinv"]]], dtype=np.float32)
    elif kind == "offline":
        feeds["offline_pickup_input"]     = np.array([[pickup_val]], dtype=np.float32)
        feeds["offline_knob_main"]        = np.array([[main if main is not None else knobs["main"]]], dtype=np.float32)
        feeds["offline_knob_pickattack"]  = np.array([[pickattack if pickattack is not None else knobs["pickattack"]]], dtype=np.float32)
        feeds["offline_knob_pickreso"]    = np.array([[pickreso if pickreso is not None else knobs["pickreso"]]], dtype=np.float32)
        feeds["offline_knob_drywet"]      = np.array([[knobs["dry_pct"]]], dtype=np.float32)
        feeds["offline_knob_outgain"]     = np.array([[knobs["outgain"]]], dtype=np.float32)
        feeds["offline_knob_phaseinv"]    = np.array([[knobs["phaseinv"]]], dtype=np.float32)
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
    
def _pickup_adapter(inp, pickup_scalar, prefix):
    """
    Pickup-conditional input adapter: y = x + pickup * Delta(x).
    - Exact pass-through for piezo (pickup = 0).
    - The delta network's output conv is zero-initialized, so magnetic input also starts as a
      pass-through and learns the magnetic→piezo-domain correction during training.
    Fully causal; adds pickup_adapter_left_context() samples of left context to every branch fed
    from its output.
    """
    a = inp
    for i, d in enumerate(PICKUP_ADAPTER_DILATIONS):
        a = causal_sep_conv(a, PICKUP_ADAPTER_FILTERS, PICKUP_ADAPTER_KERNEL, d,
                            name=f"{prefix}_pickup_adapt_c{i+1}", activation="relu")
    a = Conv1D(1, 1, padding="same", kernel_initializer="zeros",
               name=f"{prefix}_pickup_adapt_out")(a)
    p3 = Reshape((1, 1), name=f"{prefix}_pickup_scalar_b3")(pickup_scalar)
    delta = Multiply(name=f"{prefix}_pickup_adapt_gate")([a, p3])
    return Add(name=f"{prefix}_pickup_adapt_add")([inp, delta])

# ────────────────────────────── MODEL BUILDERS ───────────────────────────────

@register_keras_serializable()
class FrozenTeacher(Layer):
    """
    Runs a frozen Keras model in INFERENCE mode regardless of the outer training flag.

    Keras 3's functional graph replays a nested model call with the OUTER `training` value
    (it overrides the `training=False` recorded at graph-construction time), so a teacher that
    is simply called inside the student's training graph runs with training=True and its
    training-only layers (HFLowbandPenalty.add_loss) pollute the student's objective. This
    wrapper pins training=False for the wrapped model.
    """
    def __init__(self, model, **kwargs):
        kwargs.setdefault("trainable", False)
        super().__init__(**kwargs)
        self.model = model
        self.model.trainable = False

    def call(self, inputs, training=None):
        return self.model(inputs, training=False)

    def get_config(self):
        cfg = super().get_config()
        cfg["model"] = tf.keras.utils.serialize_keras_object(self.model)
        return cfg

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model, **config)


def load_offline_teacher(path=None):
    """
    Load the (non-causal, dynamic-time) offline model as the frozen KD teacher.

    NOTE: earlier versions distilled from a "causalized" clone (all look-ahead moved to the left
    pad). That is NOT the same network delayed: each residual block's conv path gets delayed by
    its right pad while the skip path does not, so the clone's residual sums are misaligned by
    0.5–12 ms per block. The student was being pulled toward a broken teacher. The real teacher
    is used now; it receives extra left/right context (KD_TEACHER_*_CTX_SAMPLES) and its outputs
    are cropped back onto the student's segment inside build_realtime_training_model.
    """
    if path is None:
        path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    teacher = load_keras_model(path, compile=False)
    in_names = [_tensor_basename(t) for t in teacher.inputs]
    if any("knob_lc_hz" in n for n in in_names) or "offline_pickup_input" not in in_names:
        raise RuntimeError(
            f"{path} predates the current offline graph (it still has in-graph master-EQ knob "
            "inputs / lacks the pickup input). Retrain it with --train-offline, or rebuild it from "
            "the old weights with --migrate-offline, before running --train-realtime / --fine-tune.")
    teacher.trainable = False
    for l in teacher.layers:
        l.trainable = False
    return teacher

@register_keras_serializable()
def build_offline_model(dynamic_time=False,
                                  lookahead_samples=OFFLINE_LOOKAHEAD_SAMPLES,
                                  name="piezo2mic_offline",
                                  coloreq_trainable=False,
                                  include_dry_output=True):
    """
    Offline model. During training, set include_dry_output=False to keep 2 outputs:
      [final mix, style]
    For runtime/export, include_dry_output=True (default): expose 3 outputs:
      [final mix, style, dry]
    """
    T   = None if dynamic_time else OFFLINE_SEG_LEN
    inp = Input(shape=(T,1), name="offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="offline_style_true_input")
    style_force_in = Input(shape=(1,), name="offline_style_force_input")
    pickup_in      = Input(shape=(1,), name="offline_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="offline_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="offline_knob_phaseinv")

    # Loss core for the training-only lowband penalties (static training graph only)
    core = OFFLINE_LOSS_CORE if not dynamic_time else (None, None)

    # Pickup-conditional input adapter (pass-through at pickup=0). All analysis/synthesis
    # branches read the adapted signal; the dry paths keep the raw input.
    x_core = _pickup_adapter(inp, pickup_in, prefix="offline")

    # Style
    cls         = Conv1D(16,3, padding="same", activation="relu", name="offline_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="offline_style_gap")(cls)
    style_logit = Dense(1, name="offline_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="offline_style_prob")(style_logit)

    one_minus_force = Lambda(invert, name="offline_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="offline_style_used_scalar")([
        Multiply(name="offline_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="offline_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time = TileStyle(name="offline_style_time_used")([style_used_scalar, inp])
    # FiLM conditioning on both axes (style + pickup)
    cond = Concatenate(name="offline_cond_concat")([style_used_scalar, pickup_in])
    style_embed= Dense(OFFLINE_EMBED_DIM, activation="linear", name="offline_cond_embed")(cond)

    x = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_init_pw")(x_core)
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

    # Analytic pick mask and heads (fed from the adapted signal so the fixed dB-domain
    # thresholds see piezo-like HF content regardless of pickup type)
    mask = _pick_mask_engine(x_core,
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

    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(x_core)
    pickattack, pa_pen = _pick_attack("offline", hf_in, mask, style_time, filters=16,
                                      sample_rate=OFFLINE_SAMPLE_RATE,
                                      softclip_drive=OFFLINE_PICK_ATTACK_SOFTCLIP_DRIVE,
                                      lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                                      lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                                      lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                                      cutoff_hz=OFFLINE_PICK_ATTACK_CUTOFF_HZ,
                                      core=core)
    pickreso, pr_pen = _reson_learn_offline(x_core, mask, style_time, core=core)

    knob_pa_in  = Input(shape=(1,), name="offline_knob_pickattack")
    knob_pr_in  = Input(shape=(1,), name="offline_knob_pickreso")
    knob_main_in= Input(shape=(1,), name="offline_knob_main")
    knob_dw_in  = Input(shape=(1,), name="offline_knob_drywet")

    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="offline_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="offline_pickreso_user_gain")
    main_scaled = apply_runtime_gain(main_core, knob_main_in, name="offline_main_user_gain")

    out_prehp = Add(name="offline_with_picklearn_prehp")([main_scaled, pickattack, pickreso])
    out_prehp = LossTether(name="offline_pick_lowband_penalties")([out_prehp, pa_pen, pr_pen])

    # ColorEQ (pre dry/wet)
    out_col = add_coloreq(out_prehp, bands=COLOREQ_BANDS_OFFLINE_BODY, sample_rate=OFFLINE_SAMPLE_RATE,
                          name_prefix="offline_coloreq", order=97, trainable_scales=bool(coloreq_trainable))

    # Dry/wet mix
    wet_gain   = Lambda(invert, name="offline_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(out_col, wet_gain,   name="offline_wet_gain")
    dry_scaled = apply_runtime_gain(inp,     knob_dw_in, name="offline_dry_gain")
    y_mix      = Add(name="offline_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + output gain
    phase_gain = Lambda(phase_polarity_gain, name="offline_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="offline_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="offline_output_gain")

    # Dry passthrough (raw piezo, time-aligned)
    dry_out = Activation('linear', name="offline_dry_out")(inp)

    inputs_list = [inp, style_true_in, style_force_in, pickup_in,
                   knob_main_in, knob_pa_in, knob_pr_in, knob_dw_in,
                   knob_outgain, knob_phaseinv]

    if include_dry_output:
        return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob, dry_out], name=name)
    else:
        return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob], name=name)

@register_keras_serializable()
def build_realtime_training_model(offline=None,
                        dynamic_time=False,
                        name="piezo2mic_realtime",
                        coloreq_trainable=False):
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None, 1), name="realtime_offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")
    pickup_in      = Input(shape=(1,), name="realtime_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="realtime_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="realtime_knob_phaseinv")

    knob_main_in = Input(shape=(1,), name="realtime_knob_main")
    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    # Loss core (student-rate samples) for KD and the lowband penalties: static training graph only
    core = REALTIME_LOSS_CORE if not dynamic_time else (None, None)

    # Pickup-conditional input adapter (pass-through at pickup=0)
    x_core = _pickup_adapter(inp, pickup_in, prefix="realtime")

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    kd_factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE

    style_pred_source = style_prob
    teacher_audio = None
    teacher_pick = None
    teacher_finger = None
    teacher_feats = None
    if offline is not None:
        teacher = offline
        teacher.trainable = False

        # Teacher wrapper exposing audio, style, the pick/finger heads, and two FiLM-block
        # feature maps for knowledge distillation. The teacher is the real (non-causal) offline
        # model; it gets KD_TEACHER_*_CTX_SAMPLES of extra context around the student's segment
        # (see segment_generator_dual) and its time-domain outputs are cropped back below.
        audio_out, style_prob_out = teacher.outputs[0], teacher.outputs[1]
        t_feat_mid_idx = len(OFFLINE_DILATIONS) // 2
        t_feat_last_idx = len(OFFLINE_DILATIONS) - 1
        teacher_combo = FrozenTeacher(tf.keras.Model(
            inputs=teacher.inputs,
            outputs=[audio_out, style_prob_out,
                     teacher.get_layer("offline_final_pick").output,
                     teacher.get_layer("offline_final_finger").output,
                     teacher.get_layer(f"offline_dil_{t_feat_mid_idx}_film_apply").output,
                     teacher.get_layer(f"offline_dil_{t_feat_last_idx}_film_apply").output],
            name="offline_teacher_combo_model"
        ), name="offline_teacher_combo")

        # IMPORTANT: feed inputs in the exact order of build_offline_model(inputs_list)
        teacher_inputs = [
            offline_inp,                # offline_piezo_input (student segment ×2 + teacher context)
            style_true_in,              # offline_style_true_input
            style_force_in,             # offline_style_force_input
            pickup_in,                  # offline_pickup_input

            knob_main_in,               # offline_knob_main
            knob_pa_in,                 # offline_knob_pickattack
            knob_pr_in,                 # offline_knob_pickreso
            knob_dw_in,                 # offline_knob_drywet

            knob_outgain,               # offline_knob_outgain
            knob_phaseinv,              # offline_knob_phaseinv
        ]

        # FrozenTeacher pins training=False: the teacher's training-only penalty layers must not
        # add (constant) losses to the student's objective.
        (teacher_audio, offline_style_prob,
         teacher_pick, teacher_finger,
         t_feat_mid, t_feat_last) = teacher_combo(teacher_inputs)

        # Crop the teacher's context so every teacher tensor is time-aligned with the student's
        # segment (96k: 2× the student length).
        t_crop = (int(KD_TEACHER_LEFT_CTX_SAMPLES), int(KD_TEACHER_RIGHT_CTX_SAMPLES))
        teacher_audio  = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_audio_crop")(teacher_audio)
        teacher_pick   = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_pick_crop")(teacher_pick)
        teacher_finger = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_finger_crop")(teacher_finger)
        t_feat_mid     = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_feat0_crop")(t_feat_mid)
        t_feat_last    = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_feat1_crop")(t_feat_last)
        teacher_feats = (t_feat_mid, t_feat_last)

        # Blend student style prob with teacher’s style prob
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
    # FiLM conditioning on both axes (style + pickup)
    cond            = Concatenate(name="realtime_cond_concat")([style_used_scalar, pickup_in])
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_cond_embed")(cond)

    kd_feat_mid_idx  = len(REALTIME_DILATIONS) // 2
    kd_feat_last_idx = len(REALTIME_DILATIONS) - 1

    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(x_core)
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
        # Feature-space knowledge distillation (returns x unchanged)
        if teacher_feats is not None and i == kd_feat_mid_idx:
            x = KDFeatureDistill(alpha=REALTIME_FEAT_DISTILL_ALPHA, teacher_channels=OFFLINE_FILTERS,
                                 factor=kd_factor, core_start=core[0], core_end=core[1],
                                 name="feat_distill_0")([x, teacher_feats[0]])
        if teacher_feats is not None and i == kd_feat_last_idx:
            x = KDFeatureDistill(alpha=REALTIME_FEAT_DISTILL_ALPHA, teacher_channels=OFFLINE_FILTERS,
                                 factor=kd_factor, core_start=core[0], core_end=core[1],
                                 name="feat_distill_1")([x, teacher_feats[1]])

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)
    # Stem-level knowledge distillation against the teacher's pick/finger heads
    if teacher_pick is not None:
        out_pick = KDWaveformDistill(alpha=REALTIME_STEM_PICK_DISTILL_ALPHA, factor=kd_factor,
                                     core_start=core[0], core_end=core[1],
                                     name="stem_pick_distill")([out_pick, teacher_pick])
    if teacher_finger is not None:
        out_finger = KDWaveformDistill(alpha=REALTIME_STEM_FINGER_DISTILL_ALPHA, factor=kd_factor,
                                       core_start=core[0], core_end=core[1],
                                       name="stem_finger_distill")([out_finger, teacher_finger])
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch (non-streaming)
    ir = RealtimeIrDownsampleFIR(name="realtime_ir_downsample")(x_core)
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

    # Pick mask and heads (fed from the adapted signal)
    mask = _pick_mask_engine(x_core,
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
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(x_core)
    pickattack, pa_pen = _pick_attack("realtime", hf_in, mask, style_scalar_b3,
                                      filters=16, sample_rate=REALTIME_SAMPLE_RATE,
                                      softclip_drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE,
                                      lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                                      lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                                      lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                                      cutoff_hz=REALTIME_PICK_ATTACK_CUTOFF_HZ,
                                      core=core)
    pickreso, pr_pen = _reson_learn_realtime(x_core, mask, style_scalar_b3, core=core)

    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,         knob_ir_in, name="realtime_ir_user_gain")
    
    y_main_scaled = apply_runtime_gain(y_main, knob_main_in, name="realtime_main_user_gain")

    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main_scaled, ir, pickattack, pickreso])
    y_prehp  = LossTether(name="realtime_pick_lowband_penalties")([y_prehp, pa_pen, pr_pen])

    y_wet = add_coloreq(y_prehp, bands=COLOREQ_BANDS_REALTIME_BODY, sample_rate=REALTIME_SAMPLE_RATE,
                        name_prefix="realtime_coloreq", order=65, trainable_scales=bool(coloreq_trainable))
    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_mix      = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + Output gain
    phase_gain = Lambda(phase_polarity_gain, name="realtime_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="realtime_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="realtime_output_gain")

    # Audio-level knowledge distillation against the teacher's output (loss core only).
    # Both sides see identical knob values during training, so the comparison is level-matched.
    if teacher_audio is not None:
        y_out = KDWaveformDistill(alpha=REALTIME_DISTILL_ALPHA, factor=kd_factor,
                                  core_start=core[0], core_end=core[1],
                                  name="audio_distill")([y_out, teacher_audio])
        y_out = KDSTFTDistill(alpha=REALTIME_AUDIO_STFT_DISTILL_ALPHA, factor=kd_factor,
                              core_start=core[0], core_end=core[1],
                              name="audio_stft_distill")([y_out, teacher_audio])

    inputs_list = [inp, style_true_in, style_force_in, pickup_in]
    if offline_inp is not None:
        inputs_list = [inp, offline_inp, style_true_in, style_force_in, pickup_in]
    inputs_list += [knob_main_in, knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in,
                    knob_outgain, knob_phaseinv]

    return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob], name=name)

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
    pickup_in      = Input(shape=(1,), name="realtime_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    knob_main_in = Input(shape=(1,), name="realtime_knob_main")
    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    # Pickup-conditional input adapter (streaming; pass-through at pickup=0).
    # Layer names mirror the training graph so copy_weights_by_name transfers the weights.
    a = inp
    for i, d in enumerate(PICKUP_ADAPTER_DILATIONS):
        pad = (PICKUP_ADAPTER_KERNEL - 1) * d
        ch_in = 1 if i == 0 else PICKUP_ADAPTER_FILTERS
        s_ad = Input(shape=(pad, ch_in), name=f"realtime_state_pickup_adapt_c{i+1}")
        state_inputs.append(s_ad)
        a_cat = ConcatTime(name=f"realtime_pickup_adapt_c{i+1}_concat")([s_ad, a])
        a = SeparableConv1D(PICKUP_ADAPTER_FILTERS, PICKUP_ADAPTER_KERNEL, dilation_rate=d,
                            padding="valid", name=f"realtime_pickup_adapt_c{i+1}")(a_cat)
        a = Activation("relu", name=f"realtime_pickup_adapt_c{i+1}_relu")(a)
        s_ad_n = TakeLastN(pad, name=f"realtime_state_pickup_adapt_c{i+1}_next")(a_cat)
        state_outputs.append(s_ad_n)
    a = Conv1D(1, 1, padding="same", kernel_initializer="zeros", name="realtime_pickup_adapt_out")(a)
    p3 = Reshape((1, 1), name="realtime_pickup_scalar_b3")(pickup_in)
    delta = Multiply(name="realtime_pickup_adapt_gate")([a, p3])
    x_core = Add(name="realtime_pickup_adapt_add")([inp, delta])

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)

    # EMA of per-hop style LOGIT. The state 'realtime_state_style_ema' stores the logit, so a
    # zero state corresponds to probability 0.5 (unbiased at reset). Applying sigmoid AFTER the
    # EMA matches the training-time pooling (sigmoid of a linear pool over ~90 ms) far better
    # than an EMA of per-32-sample sigmoids, which is both noisy and biased toward 0.5.
    s_style = Input(shape=(1,1), name="realtime_state_style_ema")
    state_inputs.append(s_style)
    s_style_flat = Reshape((1,), name="realtime_state_style_ema_flat")(s_style)
    # Hop-invariant EMA: alpha is defined per STYLE_EMA_REF_HOP samples and rescaled to the actual
    # hop length (T), so a dynamic-T graph stepped 1024 samples at a time tracks style exactly like
    # the T=32 pedal graph.
    style_logit_ema, s_style_next_flat = StreamingScalarEMA(
        alpha=STYLE_EMA_ALPHA_PER_HOP, ref_hop=STYLE_EMA_REF_HOP,
        name="realtime_style_logit_ema")([style_logit, s_style_flat, inp])
    s_style_next = Reshape((1,1), name="realtime_state_style_ema_expand")(s_style_next_flat)
    state_outputs.append(Activation('linear', name="realtime_state_style_ema_next")(s_style_next))
    style_prob_ema = Activation("sigmoid", name="realtime_style_prob")(style_logit_ema)

    # Select predicted vs true style
    one_minus_force   = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_prob_ema]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])

    # FiLM conditioning on both axes (style + pickup)
    cond            = Concatenate(name="realtime_cond_concat")([style_used_scalar, pickup_in])
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_cond_embed")(cond)

    # NEW: per-sample style_time via streaming linear interpolation between hops
    s_style_prev = Input(shape=(1,1), name="realtime_state_style_prev")
    state_inputs.append(s_style_prev)
    style_time, s_style_prev_next = StreamingStyleLerp(name="realtime_style_time_stream")([inp, style_used_scalar, s_style_prev])
    
        # NEW: per-sample smoothing of the style envelope for the pickattack head only.
    # This removes the subtle “zipper/bitcrush” artifact caused by per-hop envelope slope resets.
    tau_pa = max(1, int(round((PICK_ATTACK_STYLE_SMOOTH_MS / 1000.0) * REALTIME_SAMPLE_RATE)))
    L_pa   = int(np.clip(np.ceil(3.0 * tau_pa), 8, 1024))  # keep the state small
    s_style_pa = Input(shape=(L_pa - 1, 1), name="realtime_state_pickattack_style_smooth")
    state_inputs.append(s_style_pa)
    k_pa_smooth = _exp_smoothing_kernel(tau_pa, scale=3.0, Lmax=L_pa)
    style_time_pa, s_style_pa_n = StreamFixedFIR(
        kernel_len=L_pa,
        out_ch=1,
        weight_name="exp_smooth_taps",
        trainable_kernel=False,
        taps_init=k_pa_smooth,
        name="realtime_pickattack_style_smooth"
    )([style_time, s_style_pa])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_style_smooth_next")(s_style_pa_n))
    
    state_outputs.append(Activation('linear', name="realtime_state_style_prev_next")(s_style_prev_next))
    inv_style_time = Lambda(invert, name="realtime_invert_style_time")(style_time)

    # Main stack
    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(x_core)
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

    pick_c     = Multiply(name="realtime_pick_contrib")([style_time, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style_time, out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch (streaming). The decimator uses the min-phase prototype kernel, whose actual
    # length is roughly half of REALTIME_IR_FIR_SIZE (scipy minimum_phase halves the taps) —
    # declare the state from the real kernel so it matches the next-state the layer emits.
    L_ir_ds = int(_ir_prototype_kernel().shape[0])
    s_ir_ds = Input(shape=(L_ir_ds - 1, 1), name="realtime_state_ir_ds")
    state_inputs.append(s_ir_ds)
    ir_ds, s_ir_ds_next = StreamingRealtimeIrDownsampleFIR(
        factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
        kernel_size=REALTIME_IR_FIR_SIZE,
        use_ir_prototype=True,
        name="realtime_ir_downsample"
    )([x_core, s_ir_ds])
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

    # Pick mask (fed from the adapted signal)
    def ema_len(ms, sr, scale, Lmax):
        s = max(1, int(round((float(ms)/1000.0) * float(sr))))
        L = int(np.clip(np.ceil(scale * s), 8, Lmax))
        return L
    # scipy's minimum_phase halves the tap count, so derive the true HPF kernel length from the
    # kernel itself; the declared state must match the next-state StreamingPickDBMask emits.
    _k_hp_probe = _get_minphase_hp_kernel(REALTIME_SAMPLE_RATE,
                                          hp_cutoff_for_sr(REALTIME_SAMPLE_RATE),
                                          REALTIME_HF_HP_ORDER)
    L_hp   = int(_k_hp_probe.shape[0])
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
    )([x_core, s_m_hp, s_m_fast, s_m_slow, s_m_att, s_m_rel, s_m_prev])
    mask, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_n = mask_outs
    state_outputs += [
        Activation('linear', name="realtime_state_pickmask_hp_next")(s_hp_n),
        Activation('linear', name="realtime_state_pickmask_fast_next")(s_fast_n),
        Activation('linear', name="realtime_state_pickmask_slow_next")(s_slow_n),
        Activation('linear', name="realtime_state_pickmask_att_next")(s_att_n),
        Activation('linear', name="realtime_state_pickmask_rel_next")(s_rel_n),
        Activation('linear', name="realtime_state_pickmask_prev_next")(s_prev_n),
    ]

    # Pre-emphasis
    s_pre = Input(shape=(1,1), name="realtime_state_preemph")
    state_inputs.append(s_pre)
    hf_in, s_pre_next = PreEmphasisStream(alpha=HP_PRE_EMPHASIS_HF, name="realtime_hf_pre_emph_stream")([x_core, s_pre])
    state_outputs.append(Activation('linear', name="realtime_state_preemph_next")(s_pre_next))

    # Pick-attack tiny CNN (smoothed style gating)
    x_pa = Multiply(name="realtime_picklearn_gate")([hf_in, mask])

    pad_c1 = (7 - 1) * 1
    s_c1 = Input(shape=(pad_c1, 1), name="realtime_state_picklearn_c1")
    state_inputs.append(s_c1)
    x_cat1 = ConcatTime(name="realtime_picklearn_c1_concat")([s_c1, x_pa])
    x_pa1 = SeparableConv1D(16, 7, dilation_rate=1, padding="valid", name="realtime_picklearn_c1")(x_cat1)
    x_pa1 = tf.keras.layers.LeakyReLU(negative_slope=0.0, name="realtime_picklearn_c1_relu")(x_pa1)
    s_c1n = TakeLastN(pad_c1, name="realtime_state_picklearn_c1_next")(x_cat1)
    state_outputs.append(s_c1n)

    pad_c2 = (7 - 1) * 2
    s_c2 = Input(shape=(pad_c2, 16), name="realtime_state_picklearn_c2")
    state_inputs.append(s_c2)
    x_cat2 = ConcatTime(name="realtime_picklearn_c2_concat")([s_c2, x_pa1])
    x_pa2 = SeparableConv1D(16, 7, dilation_rate=2, padding="valid", name="realtime_picklearn_c2")(x_cat2)
    x_pa2 = tf.keras.layers.LeakyReLU(negative_slope=0.0, name="realtime_picklearn_c2_relu")(x_pa2)
    s_c2n = TakeLastN(pad_c2, name="realtime_state_picklearn_c2_next")(x_cat2)
    state_outputs.append(s_c2n)

    x_pa_out = Conv1D(1, 1, padding="same", name="realtime_picklearn_out")(x_pa2)
    # NEW: gate with smoothed time envelope
    xg = Multiply(name="realtime_pickattack_gated")([x_pa_out, style_time_pa])
    xg = SoftClip1D(drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickattack_softclip")(xg)

    # Fade-in ramp to suppress startup pop
    s_pa_ramp = Input(shape=(1,1), name="realtime_state_pickattack_ramp")
    state_inputs.append(s_pa_ramp)
    ramp_pa, s_pa_ramp_n = StreamingLinearRamp(ramp_len=int(0.003 * REALTIME_SAMPLE_RATE),
                                               name="realtime_pickattack_ramp")([xg, s_pa_ramp])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_ramp_next")(s_pa_ramp_n))
    xg = Multiply(name="realtime_pickattack_ramp_apply")([xg, ramp_pa])

    # Post-HPF + scale + limit
    s_hp_pa = Input(shape=(PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1, 1), name="realtime_state_pickattack_posthp")
    state_inputs.append(s_hp_pa)
    L_pa = int(PICK_ATTACK_LOWBAND_PENALTY_ORDER)
    hp_pa_k_full = _hp_minphase_rev_padded(REALTIME_SAMPLE_RATE, REALTIME_PICK_ATTACK_CUTOFF_HZ, L_pa)
    xh_pa, s_hp_pa_n = StreamFixedFIR(kernel_len=L_pa, out_ch=1, weight_name="hp_taps",
                                      trainable_kernel=False, taps_init=hp_pa_k_full,
                                      name="realtime_pickattack_post_hp")([xg, s_hp_pa])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_posthp_next")(s_hp_pa_n))
    y_pa = IrScale(init=1.0, name="realtime_pickattack_scale")(xh_pa)
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y_pa = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=limit_lin, name="realtime_pickattack_limit")(y_pa)

    # Pick-reso streaming (smoothed style gating)
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
                                    name="realtime_pickreso_ctrl_bp")([x_core, s_bp])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_bp_next")(s_bp_n))
    amp_raw = Abs1D()(bp_out)

    # 3 ms time constant in samples
    tau_env2 = max(1, int(round((3.0 / 1000.0) * REALTIME_SAMPLE_RATE)))
    # FIR length ≈ 3 × tau, clipped to [8, 256]
    s_env2_len = int(np.clip(np.ceil(3.0 * tau_env2), 8, 256))

    s_env2 = Input(shape=(s_env2_len - 1, 1), name="realtime_state_pickreso_ctrl_env")
    state_inputs.append(s_env2)
    k_env2 = _exp_smoothing_kernel(tau_env2, scale=4.0, Lmax=s_env2_len)
    amp, s_env2_n = StreamFixedFIR(kernel_len=s_env2_len, out_ch=1, weight_name="exp_smooth_taps",
                                trainable_kernel=False, taps_init=k_env2,
                                name="realtime_pickreso_ctrl_env")([amp_raw, s_env2])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_env_next")(s_env2_n))
    amp = Lambda(clip_01x4, name="realtime_pickreso_ctrl_gain")(amp)

    exc = Multiply(name="realtime_pickreso_excite")([strike, amp])

    # Same min-phase halving applies to the pickreso decimator kernel
    L_reso_ds = _minphase_decimator_len(REALTIME_PICK_RESO_DS_TAPS, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    s_ds8 = Input(shape=(L_reso_ds - 1, 1), name="realtime_state_pickreso_ds8")
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

    # NEW: gate with smoothed style envelope
    xg_reso = Multiply(name="realtime_pickreso_gated")([mixed, style_time])

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

    # Mix stems and coloreq
    y_main_scaled = apply_runtime_gain(y_main, knob_main_in, name="realtime_main_user_gain")
    pickattack = apply_runtime_gain(y_pa, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(y_reso, knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,   knob_ir_in,   name="realtime_ir_user_gain")
    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main_scaled, ir, pickattack, pickreso])

    # Color EQ streaming bandpasses
    coloreq_states_in  = []
    coloreq_states_out = []
    y_terms = [y_prehp]
    for bi, (lo, hi) in enumerate(COLOREQ_BANDS_REALTIME_BODY):
        s_b = Input(shape=(65 - 1, 1), name=f"realtime_state_coloreq_bp_{bi}")
        coloreq_states_in.append(s_b)
        # taps_init MUST be given: the training graph's BandpassFIR weight has the min-phase
        # kernel's own length, so copy_weights_by_name never transfers it into this (65-tap)
        # streaming FIR. Without an initializer the streaming coloreq bands were all-zero and any
        # fine-tuned coloreq gain was silently lost in the exported models.
        bp_y, s_b_n = StreamFixedFIR(kernel_len=65, out_ch=1, weight_name="bp_taps",
                                     trainable_kernel=False,
                                     taps_init=_bp_minphase_rev_padded(REALTIME_SAMPLE_RATE, lo, hi, 65),
                                     name=f"realtime_coloreq_bp_{bi}")([y_prehp, s_b])
        coloreq_states_out.append(Activation('linear', name=f"realtime_state_coloreq_bp_{bi}_next")(s_b_n))
        g = IrScale(init=0.0, name=f"realtime_coloreq_g_{bi}", trainable=bool(coloreq_trainable))(bp_y)
        y_terms.append(g)

    y_prehp = LossTether(name="realtime_pick_lowband_penalties_stream")([y_prehp])
    y_wet = Add(name="realtime_coloreq_sum")(y_terms)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="realtime_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="realtime_knob_phaseinv")

    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_mix      = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + Output gain
    phase_gain = Lambda(phase_polarity_gain, name="realtime_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="realtime_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="realtime_output_gain")

    # Dry passthrough (raw piezo, time-aligned)
    dry_out = Activation('linear', name="realtime_dry_out")(inp)

    inputs_list  = [inp, style_true_in, style_force_in, pickup_in,
                    knob_main_in, knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in,
                    knob_outgain, knob_phaseinv]
    inputs_list += state_inputs + coloreq_states_in

    # Style output is the EMA-smoothed probability (matches what drives FiLM), not the noisy
    # per-hop estimate — runtimes can consume it directly.
    outputs_list = [y_out, style_prob_ema, dry_out] + state_outputs + coloreq_states_out

    return tf.keras.models.Model(inputs=inputs_list, outputs=outputs_list, name=name)

def approx_steps_from_chunks(chunks, seg_len, model_sample_rate, batch_size):
    """
    Estimate steps/epoch given chunks defined in file-rate samples while model uses model_sample_rate.
    Uses 50% overlap.
    """
    total_windows = 0
    for src, _, _, cdur in chunks:
        file_sr = _file_sr(src)
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
        b = tf.shape(inputs['offline_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
        g = tf.pow(10.0, g_db / 20.0)

        # Student input (teacher reuses this)
        inputs['offline_piezo_input'] = inputs['offline_piezo_input'] * g

        # Scale only the audio target; style target is dummy and unused
        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_shared_random_gain_realtime(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to the realtime student input and the mic target
    (single-rate variant, used when no teacher input is present).
    """
    def _map(inputs, outputs, sw):
        b = tf.shape(inputs['realtime_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
        g = tf.pow(10.0, g_db / 20.0)
        inputs['realtime_piezo_input'] = inputs['realtime_piezo_input'] * g
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
        b = tf.shape(inputs['realtime_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
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
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x)[0];
    T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    inputs = prepare_training_knob_feeds(
        kind="offline",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )

    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime(x, y):
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x)[0];
    T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )

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
    
    base_inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x_realtime,
        style_true=style_true,
        style_force=style_force
    )
    
    inputs = {
        **base_inputs,
        'realtime_offline_piezo_input': x_offline
    }
    
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_realtime_dual(x, y):
    x_realtime, x_offline = x
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x_realtime)[0];
    T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    base_inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x_realtime,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )
    
    inputs = {
        **base_inputs,
        'realtime_offline_piezo_input': x_offline
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
    
    inputs = prepare_training_knob_feeds(
        kind="offline",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force
    )

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

    # Lightweight fine-tune losses (restricted to the contexted core of each segment)
    loss_offlite = FinetuneCombinedLossLite(sample_rate=OFFLINE_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0, ulb_w=0.20,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER,
                                            core=OFFLINE_LOSS_CORE)
    loss_rt_lite = FinetuneCombinedLossLite(sample_rate=REALTIME_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER,
                                            core=REALTIME_LOSS_CORE)

    off_dyn_export = None
    rt_dyn_export  = None

    if offline:
        print("\n🔧 Finetuning OFFLINE base (coloreq + film + scales only)…")
        base_off_dyn = load_keras_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build the OFFLINE training graph. include_dry_output=False keeps it at 2 outputs so
        # the 2-entry loss list below matches (the dry output is a runtime/export concern).
        off_static_train = build_offline_model(include_dry_output=False)
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

        offline_teacher = load_offline_teacher()

        base_rt_dyn = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build KD training graph (teacher embedded) and copy weights from baseline
        rt_static_train = build_realtime_training_model(
            offline=offline_teacher
        )
        copy_weights_by_name(rt_static_train, base_rt_dyn, verbose=True)

        # The streaming model does not contain the KD feature projections (feat_distill_*_proj);
        # without them the feature KD would compare the teacher to a RANDOM frozen projection.
        # Seed them from the base trainable model when it exists, else disable feature KD.
        base_rt_trainable_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME)
        feat_kd_seeded = 0
        if os.path.exists(base_rt_trainable_path):
            base_rt_trainable = load_keras_model(base_rt_trainable_path, compile=False)
            src_layers = {l.name: l for l in base_rt_trainable.layers}
            for lname in ("feat_distill_0", "feat_distill_1"):
                try:
                    dst, src_l = rt_static_train.get_layer(lname), src_layers.get(lname)
                    if src_l is not None and [w.shape for w in src_l.get_weights()] == [w.shape for w in dst.get_weights()]:
                        dst.set_weights(src_l.get_weights()); feat_kd_seeded += 1
                except Exception:
                    pass
            del base_rt_trainable
        if feat_kd_seeded < 2:
            print("[finetune] KD feature projections unavailable → feature KD disabled for this fine-tune.")
            for lname in ("feat_distill_0", "feat_distill_1"):
                try:
                    rt_static_train.get_layer(lname).set_alpha(0.0)
                except Exception:
                    pass

        # Unfreeze only FiLM + coloreq (and scales if desired)
        rt_static_train = set_trainable_layers(
            rt_static_train,
            enable_film=True,
            enable_coloreq=True
        )

        # Fine-tune distills from the base teacher at the end-of-anneal KD strengths
        for lname, (_, a_end, _, _) in KD_ALPHA_SCHEDULE.items():
            if lname.startswith("feat_distill") and feat_kd_seeded < 2:
                continue
            try:
                rt_static_train.get_layer(lname).set_alpha(a_end)
            except Exception:
                pass

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

    # 5) Pick-attack tiny CNN (offline/realtime prefixes)
    if enable_mask_head:
        for lname in [
            "offline_picklearn_c1", "offline_picklearn_c2", "offline_picklearn_out",
            "realtime_picklearn_c1","realtime_picklearn_c2","realtime_picklearn_out"
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

        # conditioning embedding Dense (style + pickup); legacy name kept for old graphs
        for lname in ["offline_cond_embed","realtime_cond_embed",
                      "offline_style_embed","realtime_style_embed"]:
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
                                           hop=REALTIME_TFLITE_T,
                                           use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)

    def rms(x):
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0

    # Read/resample
    x_mod, y_mod = _read_pair_segment(path_src, path_tgt, sample_rate, offset_sec, sec)

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
    # Prefer post-user-gain main; fall back to pre-user main if needed
    tap("realtime_main_user_gain_mul",       "tap_main_user")
    tap("realtime_main",                     "tap_main")

    tap_model = tf.keras.Model(inputs=model.inputs, outputs=list(model.outputs) + taps, name=model.name + "_taps")
    runner = KerasStreamingRunner(tap_model, hop=(hop if tap_model.inputs[0].shape[1] is not None else None))
    tap_idx = {n: runner.out_index[n] for n in found if found[n] and n in runner.out_index}

    knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned,
                                    pickup=pickup_flag_for_path(path_src))

    # Single-pass streamer
    def run_stream(style_true=None, style_force=False):
        res = runner.run(x_mod, knobs, style_true=(style_true if style_force else None), taps=tap_idx)
        t = res["taps"]
        y_full = res["audio"]
        y_main = t.get("tap_main_user", t.get("tap_main", y_full))
        z = np.zeros_like(y_full)
        return {"full": y_full, "main": y_main,
                "pa": t.get("tap_pickattack", z), "pr": t.get("tap_pickreso", z), "ir": t.get("tap_ir", z)}

    # DC removal + peak protection for stems
    def finalize(sig, headroom_db=1.0, dc_remove=True):
        s = np.asarray(sig, np.float32)
        if s.size == 0:
            return s
        if dc_remove:
            s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)) + 1e-12)
        ceiling = 10.0 ** (float(-headroom_db) / 20.0)
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

def render_isolated_stems_offline_runtime(model,
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
    knob_base = prepare_demo_knob_feeds(kind="offline", use_finetuned=use_finetuned,
                                        pickup=pickup_flag_for_path(path_src))

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    # Style inference for file naming; forced passes are always emitted
    style_true_flag = style_flag_for_path(path_src)

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    def filter_feed(d):
        names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
        return {k: v for k, v in d.items() if k in names}

    # Writers
    def open_writers(pass_key):
        w = {}
        if pass_key == "pred":
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        elif pass_key == "forced_pick":
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

        # Stems (direct passes)
        suffix = "" if pass_key == "pred" else ("_forced_pick" if pass_key == "forced_pick" else "_forced_finger")
        w["main"]      = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only{'' if pass_key=='pred' else suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickattack"]= sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only{'' if pass_key=='pred' else suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickreso"]  = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only{'' if pass_key=='pred' else suffix}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["sum"]       = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck{'' if pass_key=='pred' else suffix}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        return w

    def close_writers(w):
        for v in w.values():
            try: v.close()
            except: pass

    # Make a copy with branch knobs zeroed helper
    def iso_knobs(base, which):
        k = {kk: base[kk].astype(np.float32, copy=True) for kk in base.keys()}
        # zeros
        k["offline_knob_main"]       = np.array([[0.0]], np.float32)
        k["offline_knob_pickattack"] = np.array([[0.0]], np.float32)
        k["offline_knob_pickreso"]   = np.array([[0.0]], np.float32)
        # select
        if which == "main":
            k["offline_knob_main"] = knob_base["offline_knob_main"]
        elif which == "pickattack":
            k["offline_knob_pickattack"] = knob_base["offline_knob_pickattack"]
        elif which == "pickreso":
            k["offline_knob_pickreso"] = knob_base["offline_knob_pickreso"]
        return k

    passes = [
        ("pred",         np.zeros((1,1), np.float32), np.zeros((1,1), np.float32)),
        ("forced_pick",  np.ones((1,1),  np.float32), np.ones((1,1),  np.float32)),
        ("forced_finger",np.zeros((1,1), np.float32), np.ones((1,1),  np.float32)),
    ]

    produced_model = 0
    for pass_key, st_true, st_force in passes:
        w = open_writers(pass_key)
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
            read_b = min(END_FRAME, core_start_file + core_file_len + right_ctx_file + resample_margin_file)
            n_read = max(0, read_b - read_a)
            if n_read <= 0:
                break

            x_block_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            x_file = x_block_2d[:, 0]
            y_block_2d, _ = sf.read(path_tgt, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0]

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
            if inp_end <= inp_start:
                break

            x_inp = x_resamp[inp_start:inp_end]
            core_start_in_inp = min(left_this, len(x_inp))
            core_end_in_inp = min(core_start_in_inp + core_model, len(x_inp))
            core_len = max(0, core_end_in_inp - core_start_in_inp)
            if core_len <= 0:
                break

            x_demo = x_inp[None, :, None].astype(np.float32)

            # Build knob sets
            k_full = {kk: vv for kk, vv in knob_base.items()}
            k_main = iso_knobs(knob_base, "main")
            k_pa   = iso_knobs(knob_base, "pickattack")
            k_pr   = iso_knobs(knob_base, "pickreso")

            base_feed = {
                "offline_piezo_input": x_demo,
                "offline_style_true_input":  st_true,
                "offline_style_force_input": st_force,
            }
            # Run passes (index the outputs; the runtime model may expose 2 or 3 of them)
            def run_with(knobs):
                fd = filter_feed({**base_feed, **knobs})
                outs = model.predict(fd, batch_size=1, verbose=0)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]
                y = np.asarray(outs[0], np.float32)[0, :, 0]
                return y[core_start_in_inp:core_end_in_inp]

            y_full = run_with(k_full)
            y_main = run_with(k_main)
            y_pa   = run_with(k_pa)
            y_pr   = run_with(k_pr)

            # write
            w["full"].write(y_full)
            w["main"].write(y_main)
            w["pickattack"].write(y_pa)
            w["pickreso"].write(y_pr)
            # sumcheck:
            y_sum = np.clip(y_main + y_pa + y_pr, -1.0, 1.0)
            w["sum"].write(y_sum)

            # src/tgt for pred
            if pass_key == "pred":
                w["src"].write(x_resamp[inp_start + core_start_in_inp:inp_start + core_end_in_inp])
                w["tgt"].write(y_resamp[inp_start + core_start_in_inp:inp_start + core_end_in_inp])

            produced_model += core_len

        close_writers(w)

    print(f"✅ Offline stems (pred + forced_pick + forced_finger) written to {out_dir}")

def run_render_stems_offline(use_finetuned=False):
    sample_rate = OFFLINE_SAMPLE_RATE
    
    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="offline", use_finetuned=bool(use_finetuned))
    if dpath is None:
        print("[stems/offline] No offline dynamic models found."); sys.exit(1)
    print("[stems/offline] Using:", dpath)
    offline_dyn = load_keras_model(dpath, compile=False)

    out_dir = STEMS_FINETUNE_OFFLINE_OUT_DIR if use_finetuned else STEMS_OFFLINE_OUT_DIR

    # Use the runtime renderer (no diffing, uses knob_main/pickattack/pickreso)
    render_isolated_stems_offline_runtime(model=offline_dyn,
                                          path_src=src,
                                          path_tgt=tgt,
                                          out_dir=out_dir,
                                          sample_rate=sample_rate,
                                          sec=LONG_DEMO_SEC,
                                          offset_sec=LONG_DEMO_OFFSET_SEC,
                                          block_sec=12.0,
                                          resample_margin_ms=50.0,
                                          use_finetuned=use_finetuned)


def run_render_stems_realtime(use_finetuned=False, prefer_static=False):
    print("[stems/realtime] run_render_stems_realtime (runtime streaming path)")
    sample_rate = REALTIME_SAMPLE_RATE

    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="realtime",
                                     use_finetuned=bool(use_finetuned),
                                     prefer_static=bool(prefer_static))
    if dpath is None:
        print("[stems/realtime] No realtime models found."); sys.exit(1)
    print("[stems/realtime] Using:", dpath)
    realtime_m = load_keras_model(dpath, compile=False)

    out_dir = STEMS_FINETUNE_REALTIME_OUT_DIR if use_finetuned else STEMS_REALTIME_OUT_DIR

    # Use the runtime streaming renderer that taps post‑knob heads
    render_isolated_stems_realtime_runtime(model=realtime_m,
                                           path_src=src,
                                           path_tgt=tgt,
                                           out_dir=out_dir,
                                           sample_rate=sample_rate,
                                           sec=LONG_DEMO_SEC,
                                           offset_sec=LONG_DEMO_OFFSET_SEC,
                                           hop=REALTIME_TFLITE_T,
                                           use_finetuned=use_finetuned)


def _stems_tflite_outdir(kind, use_finetuned=False):
    base = STEMS_FINETUNE_REALTIME_OUT_DIR if (use_finetuned and kind == "realtime") else \
           STEMS_FINETUNE_OFFLINE_OUT_DIR  if (use_finetuned and kind == "offline")  else \
           STEMS_REALTIME_OUT_DIR          if (kind == "realtime")                   else \
           STEMS_OFFLINE_OUT_DIR
    out_dir = os.path.join(base, "tflite")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _tflite_schedule(kind, runner, T_fixed, core_len=None):
    """
    Decide left/core/right schedule for a single pass.
    Returns dict with keys: is_stream, T_fixed, left, core, right.
    """
    is_stream = runner.is_streaming()
    if is_stream:
        if core_len is None:
            core_len = T_fixed
        # A streaming hop < T would feed the zero-padded tail of every block into the model
        # states and then re-feed the same samples next hop — state corruption, not just
        # "reduced style stability". Only hop == T is valid mid-stream (the final partial
        # block is fine because nothing follows it).
        if int(core_len) != int(T_fixed):
            raise ValueError(
                f"Streaming models require hop == T ({T_fixed}); got hop={core_len}. "
                f"A smaller hop corrupts the streaming states.")
        return {"is_stream": True, "T_fixed": T_fixed, "left": 0, "core": int(core_len), "right": 0}
    # Windowed (offline TFLite, or any non-streaming)
    if core_len is None:
        core_len = max(1, min(T_fixed // 2, 8192 if kind == "offline" else 512))
    lc_m, rc_m = compute_streaming_ctx_constants(kind)
    left_this  = min(int(lc_m), max(0, T_fixed - 1))
    right_this = min(int(rc_m), max(0, T_fixed - 1 - left_this))
    if (T_fixed - left_this - right_this) < core_len:
        need = core_len - (T_fixed - left_this - right_this)
        r_dec = min(right_this, need); right_this -= r_dec; need -= r_dec
        if need > 0:
            l_dec = min(left_this, need); left_this -= l_dec
    return {"is_stream": False, "T_fixed": T_fixed, "left": int(left_this), "core": int(core_len), "right": int(right_this)}

def _tflite_run_full_audio(runner, kind, x_mod, sample_rate, style_mode, knobs, fixed_hop=None):
    """
    Run one full pass and return mono audio vector.
    style_mode: dict with keys:
      - mode = 'pred' or 'forced'
      - value = None (pred) or float scalar in [0,1] (forced)
    knobs: dict of knob tensors keyed by model input name
    """
    # Determine T/hop
    piezo_key = f"{kind}_piezo_input"
    T_fixed = runner.input_length_T(piezo_key)
    sched = _tflite_schedule(kind, runner, T_fixed, core_len=(fixed_hop if runner.is_streaming() else None))
    is_stream = sched["is_stream"]
    core_len  = sched["core"]
    left      = sched["left"]
    right     = sched["right"]

    # Ensure we start with zero states for streaming
    if runner.is_streaming():
        runner.reset_states()

    # Build per-pass constant style inputs
    if style_mode["mode"] == "pred":
        st_true  = np.zeros((1,1), dtype=np.float32)
        st_force = np.zeros((1,1), dtype=np.float32)
    else:
        v = float(np.clip(style_mode.get("value", 0.0), 0.0, 1.0))
        st_true  = np.array([[v]], dtype=np.float32)
        st_force = np.ones((1,1), dtype=np.float32)

    # Output buffers, also return style probability average on 'pred'
    y_full = []
    prob_sum = 0.0
    prob_count = 0

    pos = 0
    N = len(x_mod)
    while pos < N:
        core = min(core_len, N - pos)
        x_in = np.zeros((1, T_fixed, 1), dtype=np.float32)
        if is_stream:
            x_in[0, :core, 0] = x_mod[pos:pos+core]
        else:
            # left / core / right
            ltake = min(left, pos)
            rtake = min(right, N - (pos + core))
            if ltake > 0:
                x_in[0, left - ltake:left, 0] = x_mod[pos - ltake:pos]
            x_in[0, left:left + core, 0] = x_mod[pos:pos + core]
            if rtake > 0:
                xs = pos + core; xe = xs + rtake
                x_in[0, left + core:left + core + rtake, 0] = x_mod[xs:xe]

        # Make feed
        feed = {piezo_key: x_in}
        if kind == "realtime":
            feed["realtime_style_true_input"]  = st_true
            feed["realtime_style_force_input"] = st_force
        else:
            feed["offline_style_true_input"]  = st_true
            feed["offline_style_force_input"] = st_force
        for k, v in knobs.items():
            feed[k] = v.astype(np.float32, copy=False)

        y_pred, s_pred = runner.invoke(feed)

        if is_stream:
            y_core = y_pred[0, :core, 0]
        else:
            y_core = y_pred[0, left:left + core, 0]

        y_full.append(np.asarray(y_core, dtype=np.float32))
        if style_mode["mode"] == "pred":
            prob_sum   += float(np.asarray(s_pred, dtype=np.float32)[0, 0]) * float(core)
            prob_count += core

        pos += core

    y = np.concatenate(y_full, axis=0) if y_full else np.zeros((0,), np.float32)
    avg_prob = (prob_sum / float(prob_count)) if prob_count > 0 else None
    return y, avg_prob

def render_isolated_stems_tflite(kind, use_finetuned=False, hop=None):
    """
    Render stems from the TFLite model using direct isolation:
      - main only:      main=base, pa=0, pr=0, ir=0
      - pickattack only:main=0,   pa=base, pr=0, ir=0
      - pickreso only:  main=0,   pa=0,    pr=base, ir=0
      - IR only (rt):   main=0,   pa=0,    pr=0, ir=base

    Writes WAVs under out/demos/stems/<kind>/tflite/.
    """
    assert kind in ("offline", "realtime")
    # Choose pair
    if use_finetuned:
        path_src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        path_tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        path_src, path_tgt = all_pairs[DEMO_PAIR_INDEX]

    # Style flag for display (only used to decide if forced pass is written)
    style_true_flag = style_flag_for_path(path_src)

    # Sample rate and TFLite path
    sr = OFFLINE_SAMPLE_RATE if (kind == "offline") else REALTIME_SAMPLE_RATE
    tflite_path = _select_tflite_demo_paths(kind=kind, use_finetuned=use_finetuned)
    if not os.path.exists(tflite_path):
        print(f"[stems/tflite] TFLite not found at {tflite_path}. Run --export-tflite first.")
        return
    print(f"[stems/tflite] Using: {tflite_path}")

    # Read/resample audio
    info = sf.info(path_src)
    file_sr = info.samplerate
    start = int(round(LONG_DEMO_OFFSET_SEC * file_sr))
    frames = int(round(LONG_DEMO_SEC * file_sr))
    x_blk_2d, _ = sf.read(path_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_blk_2d, _ = sf.read(path_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:, 0]; y_file = y_blk_2d[:, 0]
    x_mod = x_file if file_sr == sr else resample_to_model_sr(x_file, file_sr, sr)
    y_mod = y_file if file_sr == sr else resample_to_model_sr(y_file, file_sr, sr)

    # Runner and default knobs
    runner = TFLiteRunner(tflite_path)
    base_knobs = prepare_demo_knob_feeds(kind=kind, use_finetuned=use_finetuned,
                                         pickup=pickup_flag_for_path(path_src))

    # Helpers to build per-stem knob sets
    def clone_knobs(k):
        return {kk: k[kk].astype(np.float32, copy=True) for kk in k.keys()}

    def knobs_main_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
            k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
            # keep realtime_knob_main as base
        else:
            k["offline_knob_pickattack"]  = np.array([[0.0]], np.float32)
            k["offline_knob_pickreso"]    = np.array([[0.0]], np.float32)
            # keep offline_knob_main as base
        return k

    def knobs_pa_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
            k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
            # keep pa
        else:
            k["offline_knob_main"]        = np.array([[0.0]], np.float32)
            k["offline_knob_pickreso"]    = np.array([[0.0]], np.float32)
            # keep pa
        return k

    def knobs_pr_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
            k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
        else:
            k["offline_knob_main"]        = np.array([[0.0]], np.float32)
            k["offline_knob_pickattack"]  = np.array([[0.0]], np.float32)
        return k

    def knobs_ir_only():
        k = clone_knobs(base_knobs)
        # realtime only
        k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
        k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
        k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
        return k

    def finalize(sig, headroom_db=1.0, dc_remove=True):
        s = np.asarray(sig, np.float32)
        if s.size == 0: return s
        if dc_remove: s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)) + 1e-12)
        ceiling = 10.0 ** (-float(headroom_db) / 20.0)
        if peak > ceiling: s = s * (ceiling / peak)
        return s

    def rms(x):
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0

    out_dir = _stems_tflite_outdir(kind, use_finetuned=use_finetuned)

    def open_writers(pass_key):
        w = {}
        if pass_key == "pred":
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        elif pass_key == "forced_pick":
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        # Stems
        suffix = "" if pass_key == "pred" else ("_forced_pick" if pass_key == "forced_pick" else "_forced_finger")
        w["main"]      = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only{'' if pass_key=='pred' else suffix}_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["pickattack"]= sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only{'' if pass_key=='pred' else suffix}_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["pickreso"]  = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only{'' if pass_key=='pred' else suffix}_tflite.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        if kind == "realtime":
            w["ir"]    = sf.SoundFile(os.path.join(out_dir, f"stem_ir_lowpass{'' if pass_key=='pred' else suffix}_tflite.wav"),      mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["sum"]       = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck{'' if pass_key=='pred' else suffix}_tflite.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        return w

    def close_all(w):
        for v in w.values():
            try: v.close()
            except: pass

    passes = [
        ("pred",         {"mode":"pred",   "value": None}),
        ("forced_pick",  {"mode":"forced", "value": 1.0}),
        ("forced_finger",{"mode":"forced", "value": 0.0})
    ]

    for key, sty in passes:
        w = open_writers(key)

        # Full mix (base knobs)
        y_full, p_avg = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=base_knobs, fixed_hop=hop)
        w["full"].write(y_full)

        # Direct stems by one-pass isolation
        y_main, _ = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_main_only(), fixed_hop=hop)
        y_pa, _   = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_pa_only(),   fixed_hop=hop)
        y_pr, _   = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_pr_only(),   fixed_hop=hop)
        if kind == "realtime":
            y_ir, _ = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_ir_only(), fixed_hop=hop)
        else:
            y_ir = np.zeros_like(y_main)

        # Finalize and write
        main_f = finalize(y_main)
        pa_f   = finalize(y_pa)
        pr_f   = finalize(y_pr)
        ir_f   = finalize(y_ir)

        if key == "pred":
            w["src"].write(x_mod); w["tgt"].write(y_mod)

        w["main"].write(main_f)
        w["pickattack"].write(pa_f)
        w["pickreso"].write(pr_f)
        if kind == "realtime":
            w["ir"].write(ir_f)

        # Sumcheck
        y_sum = np.clip(main_f + pa_f + pr_f + (ir_f if kind == "realtime" else 0.0), -1.0, 1.0)
        w["sum"].write(y_sum)

        if key == "pred" and p_avg is not None:
            lbl = "pick" if p_avg > STYLE_FLAG_PROB_THRESHOLD else "finger"
            print(f"[stems/tflite][pred] avg style={lbl} ({p_avg:.4f})")
        print(f"[stems/tflite][{key}] RMS main={rms(main_f):.6f} pa={rms(pa_f):.6f} pr={rms(pr_f):.6f}" + (f" ir={rms(ir_f):.6f}" if kind=="realtime" else ""))

        close_all(w)

    print(f"✅ TFLite stems ({kind}) written to {out_dir}")

def run_render_stems_tflite(args):
    """
    CLI front-end: --render-stems-tflite [--offline-only|--realtime-only] [--finetuned] [--both|--all] [--hop=N]
    """
    # Parse flags from args list (strings)
    argset = set(args)
    do_offline   = ("--realtime-only" not in argset)
    do_realtime  = ("--offline-only"  not in argset)
    use_finetuned = ("--finetuned"    in argset)
    do_both       = ("--both"         in argset) or ("--all" in argset)

    # Optional hop
    hop = None
    for a in list(argset):
        if a.startswith("--hop="):
            try:
                hop = int(a.split("=",1)[1])
            except Exception:
                hop = None

    # Render
    if do_offline:
        if do_both:
            render_isolated_stems_tflite("offline", use_finetuned=False, hop=None)
            render_isolated_stems_tflite("offline", use_finetuned=True,  hop=None)
        else:
            render_isolated_stems_tflite("offline", use_finetuned=use_finetuned, hop=None)

    if do_realtime:
        if do_both:
            render_isolated_stems_tflite("realtime", use_finetuned=False, hop=hop)
            render_isolated_stems_tflite("realtime", use_finetuned=True,  hop=hop)
        else:
            render_isolated_stems_tflite("realtime", use_finetuned=use_finetuned, hop=hop)

def _split_chunks_train_val(chunks, val_fraction=0.2):
    """
    Split chunks into train/val by CONTIGUOUS time regions per source file: the tail of each
    file is held out. A random chunk shuffle would put val chunks seconds away from train
    chunks of the same take, which leaks and makes val_loss (EarlyStopping/checkpoints)
    optimistic.
    """
    by_src = {}
    for c in chunks:
        by_src.setdefault(c[0], []).append(c)
    train, val = [], []
    for src in sorted(by_src):
        cs = sorted(by_src[src], key=lambda c: c[2])
        n_val = int(round(len(cs) * val_fraction))
        if len(cs) > 1 and n_val == 0:
            n_val = 1
        cut = len(cs) - n_val
        train += cs[:cut]
        val   += cs[cut:]
    return train, val

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
    global L1_WEIGHT, HF_WEIGHT, STFT_WEIGHT, MEL_WEIGHT, RMS_ENV_WEIGHT, RESBP_BAND_LOSS_WEIGHT

    assert_has_pairs()

    # Chunks per (pickup, style) group; groups without data are simply absent, so a
    # piezo-only corpus behaves exactly like before the pickup axis existed.
    group_specs = []   # (style_name, pickup_flag, train_chunks, val_chunks)
    train_chunks_all, val_chunks_all = [], []
    for pickup_name in PICKUP_DIR_NAMES:
        for style_name in STYLE_DIR_NAMES:
            pairs = pairs_by_group.get((pickup_name, style_name), [])
            if not pairs:
                continue
            chunks = build_chunks(pairs_list=pairs, seg_len=seg_len, sample_rate=sample_rate)
            if not chunks:
                continue
            tr, va = _split_chunks_train_val(chunks, val_fraction=0.2)
            pickup_flag = PICKUP_MAGNETIC if pickup_name == "magnetic" else PICKUP_PIEZO
            group_specs.append((style_name, pickup_flag, tr, va))
            train_chunks_all += tr
            val_chunks_all   += va

    if not group_specs or not val_chunks_all or not train_chunks_all:
        raise RuntimeError("Not enough data to split into train/val sets.")

    steps_per_epoch  = approx_steps_from_chunks(chunks=train_chunks_all, seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    validation_steps = approx_steps_from_chunks(chunks=val_chunks_all,   seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps "
          f"({len(group_specs)} pickup×style group(s))")

    sig_y = (
        tf.TensorSpec((seg_len, 1), tf.float32),
        tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.float32)
    )

    # Per-group sampling weights: each style keeps its configured ratio, split evenly across
    # the pickups that actually have data for that style.
    style_ratio = {"pick": PICK_CHUNK_RATIO, "finger": FINGER_CHUNK_RATIO}
    n_by_style = {s: sum(1 for (sn, _, _, _) in group_specs if sn == s) for s in STYLE_DIR_NAMES}
    weights = [style_ratio[sn] / max(1, n_by_style[sn]) for (sn, _, _, _) in group_specs]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    if kind == 'realtime' and dual_offline:
        # offline:realtime SR = 96k:48k = 2:1
        offline_factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE
        assert offline_factor == 2, "This code assumes 96k offline and 48k realtime."

        sig_x = (
            tf.TensorSpec((seg_len, 1), tf.float32),                                      # realtime 48k
            tf.TensorSpec((kd_teacher_segment_len(seg_len, offline_factor), 1), tf.float32)  # offline 96k (+ctx)
        )

        def make_gen(chunks, style_name, pickup_flag):
            return lambda: segment_generator_dual(
                seg_len_realtime=seg_len, chunks=chunks,
                is_pick_chunks=(style_name == "pick"),
                realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE,
                pickup_flag=pickup_flag)

        sw_map_fn = add_sw_realtime_dual
        gain_map  = add_shared_random_gain_dual(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI)
    else:
        sig_x = tf.TensorSpec((seg_len, 1), tf.float32)

        def make_gen(chunks, style_name, pickup_flag):
            return lambda: segment_generator(
                seg_len=seg_len, chunks=chunks,
                is_pick_chunks=(style_name == "pick"),
                model_sample_rate=sample_rate,
                pickup_flag=pickup_flag)

        sw_map_fn = add_sw_offline if kind == 'offline' else add_sw_realtime
        gain_map  = (add_shared_random_gain_offline(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI)
                     if kind == 'offline'
                     else add_shared_random_gain_realtime(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI))

    train_gens = [tf.data.Dataset.from_generator(make_gen(tr, sn, pf), output_signature=(sig_x, sig_y))
                  for (sn, pf, tr, _) in group_specs]
    val_gens   = [tf.data.Dataset.from_generator(make_gen(va, sn, pf), output_signature=(sig_x, sig_y))
                  for (sn, pf, _, va) in group_specs]

    train_ds = tf.data.Dataset.sample_from_datasets(
        train_gens, weights=weights
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(batch_size)

    # Shared random gain augmentation (train only) → level-invariant model + analytic branches
    train_ds = train_ds.map(sw_map_fn).map(gain_map).repeat().prefetch(tf.data.AUTOTUNE)

    # Validation set: a FIXED, cached set of batches. The generators draw random offsets, so an
    # uncached val pipeline would present different audio every epoch and EarlyStopping /
    # ModelCheckpoint / ReduceLROnPlateau would react to sampling noise instead of progress.
    validation_steps = max(1, min(validation_steps, VAL_MAX_BATCHES))
    val_raw = tf.data.Dataset.sample_from_datasets(
        val_gens, weights=weights, seed=SEED
    ).batch(batch_size=batch_size).take(validation_steps).cache()
    t_val = time.time()
    n_val = sum(1 for _ in val_raw)  # fill the cache once so every later pass is identical
    print(f"[val] cached {n_val} fixed validation batch(es) in {time.time() - t_val:.1f}s")
    val_ds = val_raw.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    if resume and resume_ckpt_path:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        ckpt_model = load_keras_model(resume_ckpt_path, compile=False)
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        copy_weights_by_name(static_model, ckpt_model, verbose=True)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        initial_epoch = 0

    # Loss-scale calibration. Random-init loss magnitudes are not representative, so run a brief
    # warmup with the declared weights first and calibrate against the warmed-up model. The
    # calibrated weights are persisted next to the checkpoints and reloaded on resume: re-running
    # the calibration against a partially trained model would silently change the objective
    # mid-run (the ratios between terms drift as the model improves).
    loss_core = OFFLINE_LOSS_CORE if kind == 'offline' else REALTIME_LOSS_CORE
    weights_json = os.path.join(checkpoint_out_dir, "loss_weights.json")
    new_w = None
    if resume and resume_ckpt_path and os.path.exists(weights_json):
        with open(weights_json, "r") as f:
            new_w = json.load(f)
        print(f"[calib] Reusing persisted loss weights from {weights_json}")
    if new_w is None:
        if not (resume and resume_ckpt_path):
            warmup_steps = max(1, min(CALIB_WARMUP_STEPS, steps_per_epoch))
            print(f"[calib] Warmup for {warmup_steps} step(s) before loss-weight calibration…")
            static_model.compile(
                optimizer=tf.keras.optimizers.Adam(LR),
                loss=[CombinedLoss(sample_rate, extra_small_cplx=(kind == 'offline'), core=loss_core),
                      "binary_crossentropy"],
                loss_weights=[1.0, LAMBDA_STYLE_RATIO]
            )
            static_model.fit(train_ds, epochs=1, steps_per_epoch=warmup_steps, verbose=1)
        new_w, _ = calibrate_loss_weights(
            model=static_model, dataset=val_ds, sample_rate=sample_rate, n_batches=5,
            extra_small_cplx=(kind == 'offline'), core=loss_core
        )
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        with open(weights_json, "w") as f:
            json.dump({k: float(v) for k, v in new_w.items()}, f, indent=2)
    L1_WEIGHT              = new_w['l1']
    STFT_WEIGHT            = new_w['stft']
    MEL_WEIGHT             = new_w['mel']
    RMS_ENV_WEIGHT         = new_w['env']
    HF_WEIGHT              = new_w['hf']
    RESBP_BAND_LOSS_WEIGHT = new_w['resbp_bl']  # requires the matching `global` declaration above
    print(f"→ normalized ({kind}) weights:"
        f" L1={L1_WEIGHT:.6f}"
        f" STFT={STFT_WEIGHT:.6f}"
        f" MEL={MEL_WEIGHT:.6f}"
        f" ENV={RMS_ENV_WEIGHT:.6f}"
        f" HF={HF_WEIGHT:.6f}"
        f" RESBP_BL={RESBP_BAND_LOSS_WEIGHT:.6f}")

    # Instantiate the loss with the per-model smoothing (loss restricted to the contexted core)
    _combined_loss = CombinedLoss(sample_rate, extra_small_cplx=(kind == 'offline'), core=loss_core)
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
            **KD_ALPHA_SCHEDULE,
            "realtime_style_blend": (REALTIME_STYLE_BLEND_INIT, 0.3, 0, 100)
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
        build_model_fn_args_static={ "include_dry_output": False },   # training: 2 outputs
        build_model_fn_args_dynamic={ "include_dry_output": True },    # runtime/export: 3 outputs
        checkpoint_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
        logs_out_dir=LOGS_OFFLINE_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR
    )

def train_realtime(resume=False, resume_ckpt_path=None):
    offline_dynamic_model_filepath = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(offline_dynamic_model_filepath):
        print("Offline model not found, run --train-offline first")
        sys.exit(1)

    # Load the offline teacher (the real non-causal model; it gets extra context per segment)
    offline_teacher = load_offline_teacher(offline_dynamic_model_filepath)
    print("Loaded offline teacher (non-causal, frozen) from", offline_dynamic_model_filepath)

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
        build_model_fn_args_static={ 'offline': offline_teacher },      # teacher embedded
        build_model_fn_args_dynamic={ 'offline': offline_teacher },     # teacher embedded
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
    # Sort numerically by epoch (a plain lexicographic sort breaks past epoch 99)
    ckpts = sorted(glob.glob(os.path.join(checkpoints_out_dir, "*.keras")),
                   key=_extract_epoch_from_ckpt)
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
    os.makedirs(out_dir, exist_ok=True)

    # Offline: windowed dynamic graph. Realtime: streaming graph (static T=32 → hop 32, exactly
    # like the pedal; dynamic T → KERAS_STREAM_HOP), both via KerasStreamingRunner.
    return generate_demo_from_files(
        model           = model,
        kind            = kind,
        out_dir         = out_dir,
        sample_rate     = sample_rate,
        path_src        = pth_src,
        path_target     = pth_tgt,
        prediction_name = prediction_name,
        only_prediction = only_prediction,
        offset_sec      = LONG_DEMO_OFFSET_SEC,
        sec             = LONG_DEMO_SEC
    )
        
def prepare_and_generate_demo(kind, sample_rate, dynamic_model_filepath, demo_out_dir, prefer_static=False):
    """
    For demos, only load saved artifacts:
      - offline: dynamic offline model (non-streaming) by default; if prefer_static=True and static exists, use it
      - realtime: static T=32 streaming model (self-contained) is preferred; prefer_static=True is recommended
    This function never builds graphs; it loads saved .keras models and runs generate_demo.
    """
    # Prefer saved artifacts by kind
    prefer = None
    if kind == "offline":
        if prefer_static:
            cand = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
            if os.path.exists(cand):
                prefer = cand
        if prefer is None:
            cand_dyn = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
            if os.path.exists(cand_dyn):
                prefer = cand_dyn

    elif kind == "realtime":
        # Realtime demo: by default the DYNAMIC streaming model, stepped KERAS_STREAM_HOP samples
        # at a time (~30x faster; hop-invariant style EMA, verified identical to 5e-5). With
        # --static (prefer_static) the static T=32 model is stepped exactly like the pedal.
        cand_static = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
        cand_dyn    = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        if prefer_static and os.path.exists(cand_static):
            prefer = cand_static
        elif os.path.exists(cand_dyn):
            prefer = cand_dyn
        elif os.path.exists(cand_static):
            prefer = cand_static
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

    dpath = prefer if prefer is not None else dynamic_model_filepath
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"[demo] Missing model for kind='{kind}'. Expected at: {dpath}. "
                                f"For realtime, train/export the static T={REALTIME_TFLITE_T} streaming model first.")
    model = load_keras_model(dpath, compile=False)
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
    """
    Wrap a dynamic-time runtime graph for fixed-T TFLite export.
    offline → outputs: [audio, style, dry]
    realtime → done by build_realtime_streaming_export_for_tflite
    """
    if kind == "offline":
        inp  = Input(shape=(T_fixed,1), name="offline_piezo_input")
        st_t = Input(shape=(1,),       name="offline_style_true_input")
        st_f = Input(shape=(1,),       name="offline_style_force_input")
        pkup = Input(shape=(1,),       name="offline_pickup_input")
        k_main = Input(shape=(1,),     name="offline_knob_main")
        k_pa = Input(shape=(1,),       name="offline_knob_pickattack")
        k_pr = Input(shape=(1,),       name="offline_knob_pickreso")
        k_dw = Input(shape=(1,),       name="offline_knob_drywet")
        k_og  = Input(shape=(1,), name="offline_knob_outgain")
        k_pi  = Input(shape=(1,), name="offline_knob_phaseinv")

        outs = dyn_model([inp, st_t, st_f, pkup, k_main, k_pa, k_pr, k_dw, k_og, k_pi])
        if not isinstance(outs, (list, tuple)) or len(outs) < 2:
            raise RuntimeError("Unexpected outputs from offline dynamic model during TFLite wrapping.")
        # Export [audio, style] only. The old "dry" output was a copy of the input (the models
        # are zero-latency) and, being a second [1,T,1] float tensor, made shape-based output
        # discovery in the runtimes ambiguous.
        y = Activation('linear', name="offline_audio_out")(outs[0])
        s = Activation('linear', name="offline_style_prob_out")(outs[1])

        export = tf.keras.Model(
            inputs=[inp, st_t, st_f, pkup, k_main, k_pa, k_pr, k_dw, k_og, k_pi],
            outputs=[y, s],
            name=dyn_model.name + "_tflite_export"
        )
        return export

    elif kind == "realtime":
        raise RuntimeError("_wrap_fixed_T_for_tflite(kind='realtime') is not used; "
                           "use build_realtime_streaming_export_for_tflite instead.")
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

@register_keras_serializable()
def build_realtime_streaming_export_for_tflite(rt_stream_model, strict=True):
    import re
    import numpy as np
    import tensorflow as tf

    def norm_tensor_name(t):
        nm = t.name.split(':')[0]
        if '/' in nm:
            nm = nm.split('/', 1)[0]
        if nm.startswith("serving_default_"):
            nm = nm[len("serving_default_"):]
        return nm

    def producer_layer_name(t):
        kh = getattr(t, "_keras_history", None)
        try:
            layer = kh[0] if isinstance(kh, (tuple, list)) else getattr(kh, "layer", None)
            return getattr(layer, "name", None)
        except Exception:
            return None

    # audio/style from model outputs. The old "dry" export output is intentionally dropped: it
    # was a copy of the input and, as a second [1,T,1] float tensor, made the shape-based output
    # discovery in TFLiteRunner / the C++ pedal ambiguous (audio vs dry only differed by name).
    if len(rt_stream_model.outputs) < 1:
        raise ValueError("[export/tflite] realtime streaming model has no outputs.")
    audio = rt_stream_model.outputs[0]
    style = None
    if len(rt_stream_model.outputs) >= 2:
        style = rt_stream_model.outputs[1]

    # If style missing, synthesize a zero scalar
    if style is None:
        style = tf.keras.layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], 1), tf.float32),
            name="realtime_style_prob")(audio)

    inputs = list(rt_stream_model.inputs)

    # Discover streaming state inputs in lex order, and lengths
    state_in_meta = []
    for t in inputs:
        nm = norm_tensor_name(t)
        if nm.startswith("realtime_state_"):
            L = int(t.shape[1]) if t.shape[1] is not None else 0
            C = int(t.shape[2]) if t.shape[2] is not None else 1
            state_in_meta.append((nm, L * C))
    state_in_meta.sort(key=lambda p: p[0])

    outs = list(rt_stream_model.outputs)
    out_by_prod = {}
    for t in outs:
        pn = producer_layer_name(t)
        if pn:
            out_by_prod[pn] = t
    out_by_name = {norm_tensor_name(t): t for t in outs}

    flat_list = []
    declared_lengths = []
    missing = []

    for (nm, expected_len) in state_in_meta:
        nm_next = nm + "_next"

        t = out_by_prod.get(nm_next, None)
        if t is None:
            t = out_by_name.get(nm_next, None)
        if t is None:
            # Try removing numeric suffixes
            base_map_prod = {re.sub(r"_\d+$", "", k): v for (k, v) in out_by_prod.items()}
            t = base_map_prod.get(nm_next, None)
            if t is None:
                base_map_name = {re.sub(r"_\d+$", "", k): v for (k, v) in out_by_name.items()}
                t = base_map_name.get(nm_next, None)

        exp_len = int(expected_len)

        if t is None:
            missing.append(nm_next)
            if strict:
                continue
            # Synthesize zeros (B,exp_len) tied to batch of 'audio'
            z = tf.keras.layers.Lambda(
                lambda a, e=exp_len: tf.tile(tf.zeros((1, e), dtype=tf.float32),
                                             [tf.shape(a)[0], 1]),
                name=f"{nm}_zeros")(audio)
            flat_list.append(z)
            declared_lengths.append(exp_len)
            continue

        flat = tf.keras.layers.Flatten(name=f"{nm}_flatten")(t)

        got_len = flat.shape[1]
        if got_len is None:
            Ls = t.shape[1] or 0
            Cs = t.shape[2] or 1
            got_len = int(Ls) * int(Cs)
        got_len = int(got_len)

        delta = exp_len - got_len
        if delta > 0:
            flat_fixed = tf.keras.layers.Lambda(
                lambda f, d=delta: tf.concat(
                    [tf.zeros((tf.shape(f)[0], d), dtype=f.dtype), f], axis=1),
                name=f"{nm}_padleft")(flat)
        elif delta < 0:
            flat_fixed = tf.keras.layers.Lambda(
                lambda f, e=exp_len: f[:, -e:],
                name=f"{nm}_tailcrop")(flat)
        else:
            flat_fixed = flat

        flat_list.append(flat_fixed)
        declared_lengths.append(exp_len)

    if strict and missing:
        miss_str = "\n  - ".join(missing)
        raise ValueError(f"[export/tflite] Missing '*_next' outputs for {len(missing)} state(s):\n  - {miss_str}\n"
                         "Export cannot proceed safely.")

    # Pack bundle/splits
    if flat_list:
        bundle = tf.keras.layers.Concatenate(axis=1, name="realtime_states_bundle_out")(flat_list)
    else:
        bundle = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 1), tf.float32),
                                        name="realtime_states_bundle_out")(audio)

    lens_np = np.array(declared_lengths, dtype=np.int32).reshape(1, -1)
    splits = tf.keras.layers.Lambda(lambda _: tf.constant(lens_np, dtype=tf.int32),
                                    name="realtime_states_splits_out")(audio)

    # Final named outputs
    y_out = tf.keras.layers.Activation('linear', name="realtime_audio_out")(audio)
    s_out = tf.keras.layers.Activation('linear', name="realtime_style_prob_out")(style)

    export = tf.keras.models.Model(
        inputs=inputs,
        outputs=[y_out, s_out, bundle, splits],
        name=rt_stream_model.name + "_tflite_export"
    )

    try:
        total = int(np.sum(lens_np))
        print("[export/tflite] Streaming realtime export wrapper")
        print(f"  states discovered: {len(state_in_meta)} (lex order by input name)")
        for (nm, exp_len) in state_in_meta[:8]:
            flag = "" if (nm + "_next") not in missing else " (MISSING_NEXT)"
            print(f"    - {nm:<40} len={exp_len:<6d}{flag}")
        if len(state_in_meta) > 8:
            print(f"    ... {len(state_in_meta)-8} more")
        print("[export/tflite] outputs:")
        for t_ in export.outputs:
            print("  -", t_.name.split(':')[0], "shape=", t_.shape)
        print(f"[export/tflite] sum(splits)={total}")
        if missing and not strict:
            print(f"[export/tflite] WARNING: {len(missing)} state(s) had MISSING_NEXT and were zero-padded in bundle.")
    except Exception:
        pass

    return export

def _inspect_tflite_file(path):
    try:
        intr = Interpreter(model_path=path)
        intr.allocate_tensors()
        outs = intr.get_output_details()
        ins  = intr.get_input_details()
        out_list = [(str(d["name"]), tuple(int(x) for x in d["shape"])) for d in outs]
        in_list  = [(str(d["name"]), tuple(int(x) for x in d["shape"])) for d in ins]
        print("[TFLite/inspect] Inputs:")
        for n, s in in_list:
            print("  ", n, s)
        print("[TFLite/inspect] Outputs:")
        for n, s in out_list:
            print("  ", n, s)
        return in_list, out_list
    except Exception as e:
        print(f"[TFLite/inspect] Could not inspect TFLite file: {e}")
        return [], []


def _validate_tflite_outputs_by_shape(kind, outs, T_fixed=None):
    """
    offline : exactly one 3D float output (audio) + a (1,1) style output
    realtime: exactly one 3D float output (audio), (1,1) style, 2D bundle, int splits
    Exactly ONE 3D output is a hard requirement: the runtimes pick the audio tensor by shape.
    """
    shapes = [s for (_, s) in outs]
    three_d = [s for s in shapes if len(s) == 3]
    two_d   = [s for s in shapes if len(s) == 2]
    if len(three_d) != 1:
        raise RuntimeError(f"[TFLite/export] {kind.upper()} expected exactly ONE 3D (audio) output, got {len(three_d)}: {outs}")
    if (1, 1) not in two_d:
        raise RuntimeError(f"[TFLite/export] {kind.upper()} expected a (1,1) style output; got: {outs}")
    if kind == "realtime":
        if len(shapes) < 4:
            raise RuntimeError(f"[TFLite/export] REALTIME expected ≥4 outputs (audio, style, bundle, splits) but found {len(shapes)}: {outs}")
        if T_fixed is not None and three_d[0][1] != int(T_fixed):
            print(f"[TFLite/export] WARNING: REALTIME audio T={three_d[0][1]} != fixed T={T_fixed}. Continuing.")
    return True

def verify_tflite_bundle_file(path, strict=True):
    """
    Inspect a realtime TFLite and verify:
      - 4 outputs: audio (3D), style (1,1), bundle (2D float), splits (2D int)
      - sum(splits) == bundle length
      - sum(L*C of 'realtime_state_*' inputs) == bundle length (optional; exporter may omit states with len=0)
    Chooses bundle/splits by SHAPE and DTYPE (robust to generic names and duplicate 'StatefulPartitionedCall' bases).
    """
    try:
        intr = Interpreter(model_path=path)
        intr.allocate_tensors()
    except Exception as e:
        raise RuntimeError(f"[verify] Failed to open {path}: {e}")

    ins  = intr.get_input_details()
    outs = intr.get_output_details()

    # Streaming state inputs
    state_inputs = []
    for d in ins:
        nm = str(d["name"])
        if b"realtime_state_" in d["name"] if isinstance(d["name"], bytes) else "realtime_state_" in nm:
            state_inputs.append(d)
    exp_sum = 0
    for d in state_inputs:
        s = tuple(int(x) for x in d["shape"])
        L = s[1] if len(s) > 1 else 0
        C = s[2] if len(s) > 2 else 1
        exp_sum += (L * C)

    # Pick outputs by shape/dtype
    def is_int(arr_dtype):
        return np.issubdtype(arr_dtype, np.integer)

    # Run once so get_tensor works for all consts
    intr.invoke()

    audio_detail = None
    bundle_detail = None
    splits_detail = None

    # Audio = the only 3D output (an export with two 3D outputs is ambiguous for the runtimes)
    three_d = [d for d in outs if len(tuple(int(x) for x in d["shape"])) == 3]
    if len(three_d) != 1:
        msg = f"[verify] expected exactly one 3D (audio) output, found {len(three_d)} — re-run --export-tflite"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
    audio_detail = three_d[0] if three_d else None

    # Bundle = largest 2D float output (excluding (1,1))
    best_total = -1
    for d in outs:
        shp = tuple(int(x) for x in d["shape"])
        if len(shp) == 2 and shp != (1,1):
            arr = intr.get_tensor(d["index"])
            if arr.dtype == np.float32:
                total = int(np.prod(shp))
                if total > best_total:
                    best_total = total
                    bundle_detail = d

    # Splits = (1,N) or (N,) integer vector where N == #states
    n_states = len(state_inputs)
    for d in outs:
        shp = tuple(int(x) for x in d["shape"])
        if len(shp) == 2 and shp[0] == 1 and shp[1] == n_states:
            arr = intr.get_tensor(d["index"])
            if is_int(arr.dtype):
                splits_detail = d
                break
    if splits_detail is None:
        for d in outs:
            shp = tuple(int(x) for x in d["shape"])
            if (len(shp) == 1 and shp[0] == n_states) or (len(shp) == 2 and shp == (n_states,1)):
                arr = intr.get_tensor(d["index"])
                if is_int(arr.dtype):
                    splits_detail = d
                    break

    # Print a summary
    print("[verify] TFLite:", path)
    print(f"[verify] inputs(states): {n_states} expected bundle len: {exp_sum}")
    for d in state_inputs[:8]:
        print(f"  in: {d['name']} {tuple(int(x) for x in d['shape'])}")
    if len(state_inputs) > 8:
        print(f"  ... {len(state_inputs)-8} more")

    print("[verify] outputs:")
    for d in outs:
        print(f"  out: {d['name']} {tuple(int(x) for x in d['shape'])}")

    if bundle_detail is None or splits_detail is None:
        msg = "[verify] Missing bundle or splits (by shape). Exporter likely failed to pack states."
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
        return

    b = intr.get_tensor(bundle_detail["index"])
    if b.ndim == 2:
        bundle_len = int(np.prod(b.shape[1:]))
    elif b.ndim == 1:
        bundle_len = int(b.shape[0])
    else:
        bundle_len = int(b.size)
    print(f"[verify] bundle len (tensor): {bundle_len}")

    s = intr.get_tensor(splits_detail["index"])
    s = s.reshape(1, -1).astype(np.int32, copy=False)
    s_sum = int(np.sum(s))
    s_cnt = int(s.shape[1])
    print(f"[verify] splits count: {s_cnt}, sum(splits): {s_sum}")

    if s_cnt != n_states:
        msg = f"[verify] splits count ({s_cnt}) != #state inputs ({n_states})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)

    if s_sum != bundle_len:
        msg = f"[verify] sum(splits) ({s_sum}) != bundle len ({bundle_len})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)

    if exp_sum != bundle_len:
        msg = f"[verify] sum(L*C of inputs) ({exp_sum}) != bundle len ({bundle_len})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
    else:
        print("[verify] OK: bundle matches streaming state sizes.")

def _convert_keras_to_tflite_fp32_checked(model, out_path, validate_kind=None, T_fixed=None):
    """
    Convert Keras model to FP32 TFLite, write to out_path, then inspect and validate by SHAPES.
    validate_kind: 'offline' or 'realtime' or None (skip validation).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflm = converter.convert()
    except Exception as e:
        if validate_kind == "realtime":
            # The pedal links the builtin op resolver only (no Flex delegate): a SELECT_TF_OPS
            # model would load-fail on the Pi, so do not silently produce one.
            raise RuntimeError(f"[TFLite] Realtime export needs non-builtin ops ({e}). "
                               "The C++ pedal cannot run SELECT_TF_OPS models; fix the graph instead.") from e
        print(f"[TFLite] WARNING: builtin conversion failed ({e}); retrying with SELECT_TF_OPS. "
              "This file needs the Flex delegate (Python/ai_edge_litert only).")
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

    if validate_kind:
        _, outs = _inspect_tflite_file(out_path)
        _validate_tflite_outputs_by_shape(validate_kind, outs, T_fixed=T_fixed)

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
    - offline: wrap fixed‑T and validate [3D audio, 2D style]
    - realtime: wrap streaming and validate [3D audio, 2D style, 2D bundle]
    """
    if not os.path.exists(static_src_path):
        print(f"[TFLite/export] Missing source model: {static_src_path}")
        return False

    if kind == "offline":
        print(f"[TFLite/export] Loading offline static: {static_src_path}")
        static_m = load_keras_model(static_src_path, compile=False)

        dyn = _build_dyn_model_for_export(kind="offline")
        copy_weights_by_name(dyn, static_m, verbose=True)

        export = _wrap_fixed_T_for_tflite(dyn, kind="offline", T_fixed=T_fixed)
        _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="offline", T_fixed=T_fixed)
        return True

    # kind == "realtime"
    print(f"[TFLite/export] Loading realtime streaming static: {static_src_path}")
    rt_static = load_keras_model(static_src_path, compile=False)

    T_fixed = int(T_fixed)
    t_in = None
    try:
        t_in = int(rt_static.inputs[0].shape[1]) if rt_static.inputs else None
    except Exception:
        t_in = None
    if (t_in is not None) and (t_in not in (T_fixed,)):
        rt_fixedT = build_realtime_streaming_model(static_T=T_fixed)
        copy_weights_by_name(rt_fixedT, rt_static, verbose=True)
        rt_static = rt_fixedT

    export = build_realtime_streaming_export_for_tflite(rt_static)

    # Write to a distinct filename to avoid confusion with older files
    base, ext = os.path.splitext(out_tflite_path)
    _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="realtime", T_fixed=T_fixed)
    print(f"✅ Saved realtime TFLite: {out_tflite_path}")
    return True

def export_tflite_all():
    # OFFLINE
    offline_static_src = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
    offline_tflite_out = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_TFLITE_FP32_FILENAME)
    ok_off = _export_one_tflite(kind="offline",
                       static_src_path=offline_static_src,
                       out_tflite_path=offline_tflite_out,
                       T_fixed=OFFLINE_TFLITE_T)

    offline_ft_src = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
    offline_ft_out = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_TFLITE_FP32_FILENAME)
    ok_off_ft = False
    if os.path.exists(offline_ft_src):
        ok_off_ft = _export_one_tflite(kind="offline",
                           static_src_path=offline_ft_src,
                           out_tflite_path=offline_ft_out,
                           T_fixed=OFFLINE_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned OFFLINE (not found): {offline_ft_src}")

    # REALTIME baseline
    rt_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    rt_tflite_out = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TFLITE_FP32_FILENAME)
    ok_rt = _export_one_tflite(kind="realtime",
                       static_src_path=rt_static_src,
                       out_tflite_path=rt_tflite_out,
                       T_fixed=REALTIME_TFLITE_T)

    # REALTIME finetuned
    rt_ft_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    rt_ft_out = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TFLITE_FP32_FILENAME)
    ok_rt_ft = False
    if os.path.exists(rt_ft_static_src):
        ok_rt_ft = _export_one_tflite(kind="realtime",
                           static_src_path=rt_ft_static_src,
                           out_tflite_path=rt_ft_out,
                           T_fixed=REALTIME_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned REALTIME (not found): {rt_ft_static_src}")

    print("\n[TFLite/export] Summary:")
    print(f"  offline baseline exported: {ok_off}")
    print(f"  offline finetuned exported: {ok_off_ft}")
    print(f"  realtime baseline exported: {ok_rt} -> {rt_tflite_out if ok_rt else '(old file may still exist!)'}")
    print(f"  realtime finetuned exported: {ok_rt_ft} -> {rt_ft_out if ok_rt_ft else '(old file may still exist!)'}")

def _sum_rf(kernel, dilations):
    return (int(kernel) - 1) * sum(int(d) for d in dilations)

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

        # Pickup input adapter feeds every branch
        lc_adapt = pickup_adapter_left_context()

        left  = int(max(lc_main, lc_ir, lc_patt, lc_reso, lc_col)) + lc_adapt
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

        left  = int(max(lc_main, lc_reso, lc_patt, lc_col)) + pickup_adapter_left_context()
        # Right context is the SUM of per-layer bounded look-aheads, not the per-layer bound.
        right = compute_offline_total_lookahead_samples()
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
    """
    TFLite demo runner.
    - Streaming: T=hop, left=0, right=0; TFLiteRunner keeps states.
    - Otherwise: windowed schedule.
    """
    os.makedirs(out_dir, exist_ok=True)

    if kind == "realtime" and tflite_runner.is_streaming() and not tflite_runner.has_bundle():
        print("⚠️  [TFLite/demo] This realtime TFLite does not expose 'realtime_states_bundle_out' by name; "
              "the runner will try to find the bundle by shape. If you still hear artifacts, re-run --export-tflite.")

    piezo_key = f"{kind}_piezo_input"
    T_fixed = tflite_runner.input_length_T(piezo_key)

    is_stream = tflite_runner.is_streaming()

    if fixed_hop is not None:
        core_len = int(fixed_hop)
        if core_len < 1 or core_len > T_fixed:
            raise ValueError(f"fixed_hop must be in [1, {T_fixed}], got {fixed_hop}")
        if is_stream and core_len != T_fixed:
            # See _tflite_schedule: hop < T pushes zero-padding into the streaming states.
            raise ValueError(
                f"Streaming models require hop == T ({T_fixed}); got hop={core_len}. "
                f"A smaller hop corrupts the streaming states.")
    else:
        core_len = T_fixed if is_stream else max(1, min(T_fixed // 2, 8192 if kind == "offline" else 512))

    if is_stream:
        print(f"[TFLite/demo] STREAMING: T={T_fixed}, hop={core_len}, left=0, right=0")
        left_this, right_this = 0, 0
    else:
        lc_m, rc_m = compute_streaming_ctx_constants(kind)
        left_this  = min(int(lc_m), max(0, T_fixed - 1))
        right_this = min(int(rc_m), max(0, T_fixed - 1 - left_this))
        if (T_fixed - left_this - right_this) < core_len:
            need = core_len - (T_fixed - left_this - right_this)
            r_dec = min(right_this, need); right_this -= r_dec; need -= r_dec
            if need > 0:
                l_dec = min(left_this, need); left_this -= l_dec
        print(f"[TFLite/demo] WINDOWED: T={T_fixed}, left={left_this}, core={core_len}, right={right_this}")

    knob_feeds = prepare_demo_knob_feeds(kind=kind, use_finetuned=use_finetuned)
    
    # DEBUG: isolate main path only (no IR, no pick branches; fully wet)
    #if kind == "realtime":
        #knob_feeds["realtime_knob_ir"]         = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_pickattack"] = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_pickreso"]   = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_drywet"]     = np.array([[0.0]], dtype=np.float32)
    
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

    # Helper to run one pass (predicted style or forced style)
    def run_pass(force_style=None):
        nonlocal pos, prob_sum, prob_count

        # Reset globals
        pos = 0
        if tflite_runner.is_streaming():
            tflite_runner.reset_states()

        while pos < N:
            core = min(core_len, N - pos)

            x_in = np.zeros((1, T_fixed, 1), dtype=np.float32)
            if is_stream:
                # stream: fill the first 'core' samples, leave the rest zeros
                x_in[0, :core, 0] = x_mod[pos:pos + core]
            else:
                # windowed: left/core/right fill
                ltake = min(left_this, pos)
                rtake = min(right_this, N - (pos + core))
                if ltake > 0:
                    x_in[0, left_this - ltake:left_this, 0] = x_mod[pos - ltake:pos]
                x_in[0, left_this:left_this + core, 0] = x_mod[pos:pos + core]
                if rtake > 0:
                    xs = pos + core; xe = xs + rtake
                    x_in[0, left_this + core:left_this + core + rtake, 0] = x_mod[xs:xe]

            base_feed = {k: v.astype(np.float32) for k, v in knob_feeds.items()}
            base_feed[f"{kind}_piezo_input"] = x_in

            if force_style is None:
                base_feed[f"{kind}_style_true_input"]  = np.zeros((1,1), np.float32)
                base_feed[f"{kind}_style_force_input"] = np.zeros((1,1), np.float32)
            else:
                base_feed[f"{kind}_style_true_input"]  = np.array([[float(force_style)]], dtype=np.float32)
                base_feed[f"{kind}_style_force_input"] = np.ones((1,1), dtype=np.float32)

            y_pred, s_pred = tflite_runner.invoke(base_feed)

            if is_stream:
                y_core = y_pred[0, :core, 0]
            else:
                y_core = y_pred[0, left_this:left_this + core, 0]

            if force_style is None:
                w_pred.write(y_core)
                prob_sum   += float(np.asarray(s_pred, dtype=np.float32)[0, 0]) * float(core)
                prob_count += core
                # Write src/tgt only on the predicted-style pass; writing on every pass would
                # duplicate the reference files' content.
                w_src.write(x_mod[pos:pos + core])
                if w_tgt is not None:
                    w_tgt.write(y_mod[pos:pos + core])
            else:
                w_true.write(y_core)

            pos += core

    # Predicted-style pass
    run_pass(force_style=None)
    # Forced-style pass if we have a ground truth style
    if style_true_flag is not None:
        run_pass(force_style=style_true_flag)

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
      - Uses a single packed bundle + splits (lengths) to update states.
      - Finds bundle and splits by SHAPE (not by name), so it survives TFLite's generic output names.
      - Splits order: lexicographic state-input name order, independent of TFLite input reordering.
      - Falls back to declared vs lex-sorted order if splits are absent.
      - Ensures the 3D piezo block is fed even if input names are generic.
      - Throttles noisy warnings.
      - Deep-copies outputs to avoid internal buffer aliasing.
    """
    def __init__(self, model_path, debug=True):
        self.debug = bool(debug)
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        # Cache IO
        self.inp_details = self.interp.get_input_details()
        self.out_details = self.interp.get_output_details()
        self._in_details_by_idx  = {d["index"]: d for d in self.inp_details}
        self._out_details_by_idx = {d["index"]: d for d in self.out_details}

        # Name maps (normalized) for inputs only; outputs have generic names, so we avoid name maps for them.
        self.in_map = { self._norm_name_input(d["name"]): d["index"] for d in self.inp_details }
        self._in_details_by_name  = { self._norm_name_input(d["name"]): d for d in self.inp_details }

        # Kind
        self._kind = "realtime" if any("realtime_piezo_input" in k for k in self.in_map.keys()) else (
                     "offline"  if any("offline_piezo_input"  in k for k in self.in_map.keys()) else "unknown")

        # Streaming discovery
        self._discover_stream_states()

        # Expected bundle length = sum(L*C) over state inputs (from TFLite input shapes)
        self._expected_bundle_len = 0
        for nm in self.state_in_names:
            dinfo = self._in_details_by_name.get(nm, None)
            if dinfo is None:
                continue
            shp = tuple(int(x) for x in dinfo["shape"])
            L = shp[1] if len(shp) > 1 else 0
            C = shp[2] if len(shp) > 2 else 1
            self._expected_bundle_len += (L * C)

        # Shape-based discovery for bundle and splits
        # Bundle: 2D output with largest total length (excluding (1,1))
        # Splits: 2D output (1, N_states) or (N_states,) if TFLite flattens, prefer exact row-major (1,N)
        self._shape_bundle_candidate_name = None
        self._shape_splits_candidate_name = None
        self._shape_bundle_len = 0
        self._shape_splits_len = 0
        self._shape_splits_count = 0

        # Evaluate outputs by shape without relying on names
        for d in self.out_details:
            nm = str(d["name"])  # keep raw (unique)
            shp = tuple(int(x) for x in d["shape"])
            if len(shp) == 2:
                if shp != (1, 1):
                    total = int(shp[0] * shp[1])
                    # Candidate for bundle: pick the largest 2D (excluding style)
                    if total > self._shape_bundle_len:
                        self._shape_bundle_len = total
                        self._shape_bundle_candidate_name = nm
                # Candidate for splits: (1, N_states)
                if shp[0] == 1 and shp[1] == len(self.state_in_names):
                    self._shape_splits_candidate_name = nm
                    self._shape_splits_len = int(shp[1])
                    self._shape_splits_count = int(shp[1])
            elif len(shp) == 1:
                # Sometimes TFLite can collapse (1,N) to (N,), accept that too for splits
                if shp[0] == len(self.state_in_names) and self._shape_splits_candidate_name is None:
                    self._shape_splits_candidate_name = nm
                    self._shape_splits_len = int(shp[0])
                    self._shape_splits_count = int(shp[0])

        # Fallback ordered indices: [audio, style] + states... (very rare)
        self._ordered_state_out_idx = {}
        if self._is_streaming:
            for i, nm in enumerate(self.state_in_names):
                self._ordered_state_out_idx[nm] = 2 + i

        # Precompute lex-sorted name list for robust splitting
        self.state_in_names_sorted = sorted(list(self.state_in_names))

        # Dynamic decision for bundle splitting order ('declared' or 'sorted')
        self._bundle_order_mode = None

        # Throttled warnings
        self._warn_counts = {}
        self._piezo_fallback_logged = False

        self._print_io_summary()

        # Quick self test (zero-input)
        try:
            self._quick_self_test()
        except Exception as e:
            print(f"[TFLiteRunner] Self-test skipped: {e}")

    # ------------- helpers -------------
    def _norm_name_input(self, n):
        """
        Normalize input tensor names:
          - Strip ':0' suffix
          - Drop 'serving_default_' prefix
          - Remove TF scope prefixes ending with '/'
        """
        n = str(n)
        # Strip ':0' or ':<num>'
        if ":" in n:
            n = n.split(":", 1)[0]
        if n.startswith("serving_default_"):
            n = n[len("serving_default_"):]
        # Remove any leading scopes like 'StatefulPartitionedCall/...'
        if "/" in n:
            n = n.split("/")[-1]
        return n

    def _discover_stream_states(self):
        # Input names in interpreter-declared order (normalized)
        in_order_names = [ self._norm_name_input(d["name"]) for d in self.inp_details ]
        self.state_in_names = [n for n in in_order_names if n.startswith("realtime_state_")]

        # Name-based output map (unreliable with generic TFLite names) → leave empty
        self.state_out_map = {}
        self._is_streaming = len(self.state_in_names) > 0

        # Allocate zero states by input shapes
        self._state = {}
        self._state_shapes = {}
        for nm in self.state_in_names:
            dinfo = self._in_details_by_name.get(nm, None)
            if dinfo is None:
                continue
            shp = tuple(int(x) for x in dinfo["shape"])
            if len(shp) != 3:
                shp = (1, shp[1] if len(shp) > 1 else 0, 1)
            self._state[nm] = np.zeros(shp, dtype=np.float32)
            self._state_shapes[nm] = shp

    def _print_io_summary(self, max_layer_names=48):
        print("== TFLite IO Summary ==")
        print(f"  kind: {self._kind}")
        print(f"  inputs ({len(self.inp_details)}):")
        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            shp = tuple(int(x) for x in d["shape"])
            print(f"    {nm:<48} shape={shp}")
        print(f"  outputs ({len(self.out_details)}):")
        for d in self.out_details:
            nm = str(d["name"])
            shp = tuple(int(x) for x in d["shape"])
            print(f"    {nm:<48} shape={shp}")
        print(f"  streaming: {self._is_streaming} (states={len(self.state_in_names)})")
        if self._is_streaming:
            bundle_note = "present(by shape)" if self._shape_bundle_candidate_name is not None else "absent"
            splits_note = "present(by shape)" if self._shape_splits_candidate_name is not None else "absent"
            print(f"  expected bundle length: {self._expected_bundle_len}")
            print(f"  packed bundle: {bundle_note}")
            print(f"  splits vector: {splits_note}")
            print("  state inputs (in declared order):")
            for nm in self.state_in_names[:max_layer_names]:
                shp = tuple(int(x) for x in self._in_details_by_name[nm]["shape"])
                print(f"    {nm:<48} shape={shp}")
            if len(self.state_in_names) > max_layer_names:
                print(f"    ... {len(self.state_in_names)-max_layer_names} more")
        print("========================")

    def is_streaming(self):
        return bool(self._is_streaming)

    def has_bundle(self):
        # A bundle is considered present if shape-based candidate exists
        return bool(self._shape_bundle_candidate_name)

    def reset_states(self):
        for k in self._state.keys():
            self._state[k].fill(0.0)

    def input_length_T(self, piezo_key):
        # Prefer named piezo input if present
        if piezo_key in self.in_map:
            shp = self._in_details_by_name[piezo_key]["shape"]
            if len(shp) >= 2:
                return int(shp[1])
        # Fallback: any 3D input
        for d in self.inp_details:
            shp = d["shape"]
            if len(shp) == 3 and shp[1] > 0:
                return int(shp[1])
        # Last resort: first input second dim
        return int(self.inp_details[0]["shape"][1])

    def _piezo_T_from_feed(self, feed):
        for k, v in feed.items():
            # feed keys already normalized by caller (we normalize below too)
            nk = self._norm_name_input(k)
            if "piezo_input" in nk:
                arr = np.asarray(v)
                if arr.ndim >= 2:
                    return int(arr.shape[1])
        for v in feed.values():
            arr = np.asarray(v)
            if arr.ndim == 3:
                return int(arr.shape[1])
        return None

    def _normalize_audio_style(self, audio, style):
        a = np.array(audio, dtype=np.float32, copy=True)
        s = np.array(style, dtype=np.float32, copy=True)
        if a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        elif a.ndim >= 4:
            a = a.reshape((a.shape[0], a.shape[1], -1))[:, :, :1]
        if s.ndim == 0:
            s = s.reshape((1, 1))
        elif s.ndim == 1:
            s = s.reshape((1, -1))[:, :1]
        elif s.ndim >= 3:
            s = s.reshape((s.shape[0], -1))[:, :1]
        return a, s

    def _select_audio_style(self, outs_list, T_cur):
        # Prefer 3D as audio (by name if the export kept names, else match T), 2D(1,1) as style
        three_d = [(n,a) for (n,a) in outs_list if np.asarray(a).ndim == 3]
        two_d   = [(n,a) for (n,a) in outs_list if np.asarray(a).ndim == 2]

        audio = None
        if three_d:
            named = [t for t in three_d if "audio_out" in str(t[0])]
            cand = named or [t for t in three_d if (T_cur is None or int(np.asarray(t[1]).shape[1]) == int(T_cur))]
            if len(three_d) > 1 and not named:
                self._warn_throttled('ambiguous_3d',
                    f"⚠️  [TFLiteRunner] {len(three_d)} 3D outputs and none named '*audio_out' — "
                    "audio selection is ambiguous; re-run --export-tflite.")
            audio = (cand or three_d)[0][1]
        else:
            # Fallback to largest tensor
            audio = max(outs_list, key=lambda t: np.asarray(t[1]).size)[1]

        style = None
        for (n,a) in two_d:
            arr = np.asarray(a)
            if arr.shape == (1,1):
                style = arr
                break
        if style is None:
            style = np.zeros((1,1), dtype=np.float32)
        return audio, style

    def _pick_bundle_from_outs(self, outs_list):
        # Choose 2D float32 output with largest length, excluding (1,1)
        best = None
        best_len = 0
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if arr.ndim == 2 and arr.dtype == np.float32 and tuple(arr.shape) != (1, 1):
                total = int(np.prod(arr.shape))
                if total > best_len:
                    best_len = total
                    best = arr
        # Fallback: any 2D (non (1,1))
        if best is None:
            for (n, a) in outs_list:
                arr = np.asarray(a)
                if arr.ndim == 2 and tuple(arr.shape) != (1, 1):
                    total = int(np.prod(arr.shape))
                    if total > best_len:
                        best_len = total
                        best = arr
        return best

    def _pick_splits_from_outs(self, outs_list):
        # Choose a (1,N_states) or (N_states,) integer vector; cast to int32
        target = len(self.state_in_names)
        # Prefer (1, target) int
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if arr.ndim == 2 and arr.shape == (1, target) and np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int32, copy=False)
        # Fallback: (target,) or (target,1) int
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if ((arr.ndim == 1 and arr.shape[0] == target) or
                (arr.ndim == 2 and arr.shape == (target, 1))) and np.issubdtype(arr.dtype, np.integer):
                return arr.reshape(1, -1).astype(np.int32, copy=False)
        return None

    def _warn_throttled(self, key, msg, first_n=3, every=200):
        c = self._warn_counts.get(key, 0)
        self._warn_counts[key] = c + 1
        if c < first_n or (every > 0 and (c % every) == 0):
            print(msg)

    def _try_update_bundle_with_order(self, names_order, bundle_vec, lengths=None):
        """
        Attempt to update states using a given names_order (list of names),
        bundle_vec flattened (1,N) or (N,), and optional 'lengths' (1,M) int32.
        If lengths is None, it will slice by TFLite input shapes.
        Returns (updated_count, consumed_len).
        """
        vec = np.asarray(bundle_vec, dtype=np.float32)
        if vec.ndim == 2:
            vec = vec[0]
        elif vec.ndim >= 3:
            vec = vec.reshape(vec.shape[0], -1)[0]

        offset = 0
        updated = 0
        for idx, nm in enumerate(names_order):
            shp = self._state_shapes.get(nm, (1,0,1))
            if lengths is not None:
                lens = np.asarray(lengths).astype(np.int64)
                if lens.ndim == 2:
                    lens = lens[0]
                need = int(lens[idx]) if idx < lens.size else 0
            else:
                need = int(shp[1] * shp[2])

            if need <= 0:
                # Missing '_next' on exporter side or zero-size state: skip
                continue

            if offset + need > vec.size:
                self._warn_throttled('bundle_short',
                    f"⚠️  [TFLiteRunner] bundle too short for state '{nm}': need {need}, have {max(0, vec.size - offset)}")
                break

            seg = vec[offset:offset+need].reshape((1, int(shp[1]), int(shp[2]))).astype(np.float32, copy=False)
            self._state[nm] = seg
            offset += need
            updated += 1

        return updated, offset

    def _update_states_from_bundle(self, bundle, splits=None):
        """
        Update states from bundle. If 'splits' (lengths) is present, prefer lexicographic ORDER with those lengths.
        Else auto-try declared order then sorted; cache whichever works best.
        """
        # With splits: use lexicographic order deterministically
        if splits is not None:
            try:
                ss = np.asarray(splits)
                total = int(np.sum(ss))
                # Commented out because it completely spams the logs:
                # if self.debug:
                #     print(f"[TFLiteRunner] Using bundle with splits: sum={total}, expected≈{self._expected_bundle_len}")
                # If total != expected, we still proceed; zeros in splits indicate missing '_next' states.
            except Exception:
                pass
            updated, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=splits)
            self._last_state_updates_count = updated
            self._bundle_order_mode = 'sorted'
            if updated != len(self.state_in_names):
                self._warn_throttled('bundle_partial_sorted',
                    f"⚠️  [TFLiteRunner] bundle (with splits) updated {updated}/{len(self.state_in_names)} states.")
            # Commented out because it completely spams the logs:
            # else:
            #     if self.debug:
            #         print(f"[TFLiteRunner] bundle (with splits) updated all {updated} states.")
            return updated

        # No splits: reuse mode if already decided
        if self._bundle_order_mode == 'declared':
            upd, _ = self._try_update_bundle_with_order(self.state_in_names, bundle, lengths=None)
            self._last_state_updates_count = upd
            return upd
        if self._bundle_order_mode == 'sorted':
            upd, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=None)
            self._last_state_updates_count = upd
            return upd

        # First-time: try declared then sorted
        upd_decl, _ = self._try_update_bundle_with_order(self.state_in_names, bundle, lengths=None)
        upd_sort, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=None)

        if upd_sort > upd_decl:
            self._bundle_order_mode = 'sorted'
            self._last_state_updates_count = upd_sort
            if upd_sort != len(self.state_in_names):
                self._warn_throttled('bundle_partial_sorted',
                    f"⚠️  [TFLiteRunner] bundle updated {upd_sort}/{len(self.state_in_names)} states with sorted order.")
            else:
                if self.debug:
                    print(f"[TFLiteRunner] bundle updated all {upd_sort} states (sorted order).")
            return upd_sort
        else:
            self._bundle_order_mode = 'declared'
            self._last_state_updates_count = upd_decl
            if upd_decl != len(self.state_in_names):
                self._warn_throttled('bundle_partial_decl',
                    f"⚠️  [TFLiteRunner] bundle updated {upd_decl}/{len(self.state_in_names)} states with declared order.")
            else:
                if self.debug:
                    print(f"[TFLiteRunner] bundle updated all {upd_decl} states (declared order).")
            return upd_decl

    # ------------- inference -------------
    def invoke(self, feed_dict):
        # Normalize feed and inject states if missing
        feed = { self._norm_name_input(k): np.asarray(v, np.float32, order="C") for k, v in feed_dict.items() }
        if self._is_streaming:
            for nm in self.state_in_names:
                if nm not in feed:
                    feed[nm] = self._state[nm]

        T_cur = self._piezo_T_from_feed(feed)

        # Set inputs by name; remember whether the 3D piezo was actually set
        set_piezo = False
        piezo_block = None

        # Try to remember any 3D array from the feed (expected to be the piezo block)
        for v in feed.values():
            arr = np.asarray(v)
            if arr.ndim == 3 and (T_cur is None or int(arr.shape[1]) == int(T_cur)):
                piezo_block = arr
                break

        for k, v in feed.items():
            nm = self._norm_name_input(k)
            if nm in self.in_map:
                idx = self.in_map[nm]
                # Track if this is the 3D piezo feed
                shp = tuple(int(x) for x in self._in_details_by_idx[idx]["shape"])
                if len(shp) == 3 and np.asarray(v).ndim == 3 and (T_cur is None or int(v.shape[1]) == int(T_cur)):
                    set_piezo = True
                self.interp.set_tensor(idx, v)

        # Fallback mapping: if we didn't set any 3D piezo by name, map the 3D block by shape
        if (not set_piezo) and (piezo_block is not None):
            for d in self.inp_details:
                nm_raw = str(d["name"])
                shp = tuple(int(x) for x in d["shape"])
                if len(shp) == 3 and (T_cur is None or shp[1] == T_cur or shp[1] == -1) and shp[2] == 1:
                    try:
                        self.interp.set_tensor(d["index"], piezo_block)
                        if not self._piezo_fallback_logged:
                            print(f"[TFLiteRunner] Mapped 3D piezo block to unnamed input '{nm_raw}' shape={shp} (fallback by shape).")
                            self._piezo_fallback_logged = True
                        break
                    except Exception:
                        continue

        self.interp.invoke()

        # Deep-copy outputs
        outs_list = []
        for d in self.out_details:
            idx = d["index"]
            name = str(d["name"])
            val = np.copy(self.interp.get_tensor(idx))
            outs_list.append((name, val))
        self._last_state_updates_count = 0

        # Try packed bundle path first (by shape)
        if self._is_streaming:
            bundle = self._pick_bundle_from_outs(outs_list)
            splits = self._pick_splits_from_outs(outs_list)
            if bundle is not None:
                self._update_states_from_bundle(bundle, splits=splits)

                if self._is_streaming:
                    total_states = len(self.state_in_names)
                    updated = int(getattr(self, "_last_state_updates_count", 0))
                    if updated != total_states:
                        self._warn_throttled('bundle_partial_update',
                            f"[TFLiteRunner] bundle updated {updated}/{total_states} states (check export wiring).")

        # Name-based per-state mapping is skipped (generic TFLite names).
        # Ordered fallback [audio, style, states...]
        if self._is_streaming and self._last_state_updates_count < len(self.state_in_names):
            for in_nm in self.state_in_names:
                idx = self._ordered_state_out_idx.get(in_nm, None)
                if idx is None or idx >= len(outs_list):
                    continue
                cand = outs_list[idx][1]
                want = self._state_shapes.get(in_nm, None)
                if (want is None) or (tuple(cand.shape) == tuple(want)):
                    self._state[in_nm] = np.array(cand, dtype=np.float32, copy=True)
            self._last_state_updates_count = len(self.state_in_names)

        if self._is_streaming and self._last_state_updates_count == 0:
            self._warn_throttled('no_state_update',
                "⚠️  [TFLiteRunner] WARNING: states did NOT update; expect tonal/bitcrushed noise. "
                "Ensure your realtime TFLite includes a states bundle and (optionally) a splits vector.")

        # Select audio/style from outs_list
        audio_arr, style_arr = self._select_audio_style(outs_list, T_cur)
        return self._normalize_audio_style(audio_arr, style_arr)

    def _prepare_default_feed(self, T):
        feed = {}
        piezo_key = "realtime_piezo_input" if self._kind == "realtime" else "offline_piezo_input"
        if piezo_key in self.in_map:
            feed[piezo_key] = np.zeros((1, int(T), 1), dtype=np.float32)

        st_true = "realtime_style_true_input" if self._kind == "realtime" else "offline_style_true_input"
        st_force= "realtime_style_force_input" if self._kind == "realtime" else "offline_style_force_input"
        pkup    = "realtime_pickup_input" if self._kind == "realtime" else "offline_pickup_input"
        if st_true in self.in_map:  feed[st_true]  = np.zeros((1,1), np.float32)
        if st_force in self.in_map: feed[st_force] = np.zeros((1,1), np.float32)
        if pkup in self.in_map:     feed[pkup]     = np.zeros((1,1), np.float32)  # default: piezo

        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            if "knob_" in nm:
                val = 1.0
                feed[nm] = np.array([[val]], dtype=np.float32)

        if self._is_streaming:
            for nm in self.state_in_names:
                feed[nm] = self._state[nm]
        return feed

    def _quick_self_test(self):
        if not self._is_streaming:
            return
        T = None
        # Try to get T from piezo input
        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            if "piezo_input" in nm:
                shp = d["shape"]
                T = int(shp[1]) if len(shp) > 1 else None
                break
        if T is None:
            # Try any 3D input
            for d in self.inp_details:
                shp = tuple(int(x) for x in d["shape"])
                if len(shp) == 3 and shp[1] > 0:
                    T = shp[1]
                    break
        if T is None or T <= 0:
            return
        feed = self._prepare_default_feed(T)
        _ = self.invoke(feed)

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
    style_true_flag = style_flag_for_path(pth_src)

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
    
def migrate_offline_models_inplace():
    """
    Add master EQ/phase/output knobs to offline models by rebuilding graphs and copying weights by name.
    Saves to the same baseline/finetuned filenames.
    """
    dst_offline_static_baseline  = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
    dst_offline_dynamic_baseline = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    dst_offline_static_ft        = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
    dst_offline_dynamic_ft       = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)

    # Load old
    if not os.path.exists(dst_offline_static_baseline) or not os.path.exists(dst_offline_dynamic_baseline):
        print("[migrate/offline] Missing baseline offline models; run --train-offline first.")
        return

    off_static_old  = load_keras_model(dst_offline_static_baseline,  compile=False)
    off_dynamic_old = load_keras_model(dst_offline_dynamic_baseline, compile=False)

    # Rebuild new graphs with new knobs
    off_static_new  = build_offline_model(dynamic_time=False)
    off_dynamic_new = build_offline_model(dynamic_time=True)

    copy_weights_by_name(off_static_new,  off_static_old,  verbose=True)
    copy_weights_by_name(off_dynamic_new, off_dynamic_old, verbose=True)

    off_static_new.save(dst_offline_static_baseline)
    off_dynamic_new.save(dst_offline_dynamic_baseline)
    print("✅ [migrate/offline] Saved updated baseline offline static/dynamic.")

    # Finetuned (if present)
    if os.path.exists(dst_offline_static_ft):
        off_static_ft_old  = load_keras_model(dst_offline_static_ft,  compile=False)
        off_dynamic_ft_old = load_keras_model(dst_offline_dynamic_ft, compile=False) if os.path.exists(dst_offline_dynamic_ft) else None

        off_static_ft_new  = build_offline_model(dynamic_time=False)
        copy_weights_by_name(off_static_ft_new, off_static_ft_old, verbose=True)
        off_static_ft_new.save(dst_offline_static_ft)
        print("✅ [migrate/offline] Saved updated finetuned offline static.")

        if off_dynamic_ft_old is not None:
            off_dynamic_ft_new = build_offline_model(dynamic_time=True)
            copy_weights_by_name(off_dynamic_ft_new, off_dynamic_ft_old, verbose=True)
            off_dynamic_ft_new.save(dst_offline_dynamic_ft)
            print("✅ [migrate/offline] Saved updated finetuned offline dynamic.")

def migrate_realtime_models_inplace():
    # Destination paths
    destination_realtime_baseline_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_finetuned_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_static_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    destination_realtime_static_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
    
    # Guard against missing prerequisites instead of crashing mid-migration
    required = [
        os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME),
        os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME),
        os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
        os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[migrate/realtime] Missing prerequisite model(s):")
        for p in missing:
            print("  -", p)
        print("[migrate/realtime] Aborting migration.")
        return

    # Load the two old models to migrate from. We can use those to generate the dynamic models and eventually flatten all weights
    realtime_static_baseline_training_model_old = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)
    realtime_static_finetuned_training_model_old = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)

    # Load the offline teacher (non-causal, dynamic-time; used by every training graph)
    offline_teacher = load_offline_teacher()

    # Now, moving into the actual migration, first rebuild training graphs
    realtime_static_baseline_training_model = build_realtime_training_model(offline=offline_teacher)
    realtime_dynamic_baseline_training_model = build_realtime_training_model(offline=offline_teacher, dynamic_time=True)
    
    realtime_static_finetuned_training_model = build_realtime_training_model(offline=offline_teacher)
    realtime_dynamic_finetuned_training_model = build_realtime_training_model(offline=offline_teacher, dynamic_time=True)
    
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
    taps_min = _to_minimum_phase(taps)
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
    taps_min = _to_minimum_phase(taps)
    k_rev_small = taps_min[::-1].reshape(-1)
    M = int(k_rev_small.shape[0])
    k_full = np.zeros((nt,), dtype=np.float32)
    if M >= nt:
        k_full[:] = k_rev_small[-nt:]
    else:
        pad = nt - M
        k_full[pad:] = k_rev_small
    return k_full.reshape(nt, 1, 1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FAST PICKATTACK AUDITION: non-streaming realtime graph + in-place param tweak
# ─────────────────────────────────────────────────────────────────────────────

def _fit_rev_kernel_length(k_rev_np, target_len):
    """
    Ensure reversed FIR k_rev_np (L',1,1) matches target_len (L,1,1) by left-padding zeros
    or tail-cropping (keep most recent taps).
    """
    k = np.asarray(k_rev_np, dtype=np.float32).reshape(-1)
    Lwant = int(target_len)
    Lhave = int(k.shape[0])
    if Lhave == Lwant:
        return k.reshape(Lwant, 1, 1).astype(np.float32)
    out = np.zeros((Lwant,), dtype=np.float32)
    if Lhave > Lwant:
        out[:] = k[-Lwant:]
    else:
        out[Lwant - Lhave:] = k
    return out.reshape(Lwant, 1, 1).astype(np.float32)

def _exp_kernel_rev_ms(ms, sr, scale, Lwant):
    """
    Build reversed EMA kernel for ms, then pad/crop to Lwant.
    """
    s = max(1, int(round((float(ms)/1000.0) * float(sr))))
    k = _exp_smoothing_kernel(s, scale=scale, Lmax=max(Lwant, 8))
    k_np = (k.numpy() if tf.is_tensor(k) else np.asarray(k, np.float32)).reshape(-1)
    return _fit_rev_kernel_length(k_np, Lwant)

def _patch_pickmask_layer_streaming(layer,
                                    sample_rate,
                                    env_fast_ms=None,
                                    env_slow_ms=None,
                                    smooth_attack_ms=None,
                                    smooth_release_ms=None,
                                    thr_db=None,
                                    knee_db=None,
                                    post_comp_p=None,
                                    post_comp_gain=None,
                                    floor=None,
                                    out_gain=None):
    """
    Patch StreamingPickDBMask in-place:
      - set attributes: post_comp_p, post_comp_gain, floor
      - update scalar weights: thr_db, sigmoid slope (from knee), out_gain
      - rebuild envelope taps (keep lengths)
    Call BEFORE converting to TFLite.
    """
    # Attributes (captured as constants at conversion)
    if post_comp_p is not None:    layer.post_comp_p   = float(post_comp_p)
    if post_comp_gain is not None: layer.post_comp_gain= float(post_comp_gain)
    if floor is not None:          layer.floor         = float(floor)

    ws = layer.get_weights()
    if not ws:
        return
    # name->index map
    wmap = {w.name.split('/')[-1].split(':')[0]: i for i, w in enumerate(layer.weights)}

    def _set_scalar_weight(key, valf):
        if key in wmap and valf is not None:
            i = wmap[key]
            ws[i] = np.array([float(valf)], dtype=np.float32).reshape(ws[i].shape)

    _set_scalar_weight("dbmask_thr_db", thr_db)
    if knee_db is not None:
        slope_val = np.log(9.0) / max(float(knee_db), 1e-3)
        _set_scalar_weight("dbmask_sigmoid_slope", slope_val)
    _set_scalar_weight("dbmask_out_gain", out_gain)

    def _maybe_set_kernel(key, new_ms, scale):
        if key not in wmap or new_ms is None:
            return
        i = wmap[key]
        L = int(ws[i].shape[0])
        ws[i] = _exp_kernel_rev_ms(new_ms, sample_rate, scale, L)

    _maybe_set_kernel("dbmask_env_fast", env_fast_ms, 3.0)
    _maybe_set_kernel("dbmask_env_slow", env_slow_ms, 3.0)
    _maybe_set_kernel("dbmask_att",      smooth_attack_ms, 4.0)
    _maybe_set_kernel("dbmask_rel",      smooth_release_ms,4.0)

    layer.set_weights(ws)

def _patch_streaming_posthp_taps(layer, sample_rate, cutoff_hz):
    """
    Patch StreamFixedFIR 'realtime_pickattack_post_hp' taps to a new cutoff, keeping length.
    """
    if cutoff_hz is None:
        return
    ws = layer.get_weights()
    if not ws:
        return
    k0 = ws[0]
    L = int(k0.shape[0]) if k0.ndim == 3 else int(k0.shape[0])
    k_new = _hp_minphase_rev_padded(int(sample_rate), float(cutoff_hz), int(L))
    k_new = k_new.numpy() if tf.is_tensor(k_new) else np.asarray(k_new, np.float32)
    if k_new.shape != k0.shape:
        k_new = k_new.reshape(k0.shape)
    ws[0] = k_new.astype(np.float32)
    layer.set_weights(ws)

def _patch_softclip_attrs(layer, drive=None, limit=None):
    if drive is not None:
        layer.drive = float(drive)
    if limit is not None:
        layer.limit = float(limit)

def build_streaming_rt_with_params(load_from_path=None,
                                   softclip_drive=None, softclip_limit=None,
                                   env_fast_ms=None, env_slow_ms=None,
                                   smooth_attack_ms=None, smooth_release_ms=None,
                                   thr_db=None, knee_db=None,
                                   post_comp_p=None, post_comp_gain=None,
                                   mask_floor=None, mask_out_gain=None,
                                   posthp_cutoff_hz=None,
                                   static_T=REALTIME_TFLITE_T):
    """
    Build a streaming realtime model (static_T), copy weights from saved streaming model,
    then patch pickattack parameters in-place so conversion will capture them.
    """
    if load_from_path is None:
        load_from_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    base_stream = load_keras_model(load_from_path, compile=False)

    m = build_realtime_streaming_model(dynamic_time=False, static_T=int(static_T))
    copy_weights_by_name(m, base_stream, verbose=False)

    # Mask (StreamingPickDBMask) and post-HPF (StreamFixedFIR) and softclip attrs
    try:
        pm = m.get_layer("realtime_pick_mask_db")
        _patch_pickmask_layer_streaming(
            pm, sample_rate=REALTIME_SAMPLE_RATE,
            env_fast_ms=env_fast_ms, env_slow_ms=env_slow_ms,
            smooth_attack_ms=smooth_attack_ms, smooth_release_ms=smooth_release_ms,
            thr_db=thr_db, knee_db=knee_db, post_comp_p=post_comp_p, post_comp_gain=post_comp_gain,
            floor=mask_floor, out_gain=mask_out_gain
        )
    except Exception as e:
        print(f"[sweep/tflite] mask patch skipped: {e}")

    try:
        hp = m.get_layer("realtime_pickattack_post_hp")
        _patch_streaming_posthp_taps(hp, sample_rate=REALTIME_SAMPLE_RATE, cutoff_hz=posthp_cutoff_hz)
    except Exception as e:
        print(f"[sweep/tflite] post-HPF patch skipped: {e}")

    try:
        sc = m.get_layer("realtime_pickattack_softclip")
        _patch_softclip_attrs(sc, drive=softclip_drive, limit=softclip_limit)
    except Exception as e:
        print(f"[sweep/tflite] softclip patch skipped: {e}")

    return m

def export_streaming_rt_to_tflite(m_stream, out_tflite_path, T_fixed=REALTIME_TFLITE_T):
    """
    Wrap the streaming model for TFLite (bundle states), convert to FP32, save.
    """
    export = build_realtime_streaming_export_for_tflite(m_stream)
    _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="realtime", T_fixed=int(T_fixed))
    return out_tflite_path

def _render_pickattack_diff_keras_streaming(
    m_stream,
    x_mod,
    y_mod,
    use_finetuned,
    out_dir,
    apply_demo_boosts=True
):
    """
    STREAMING Keras large-T: isolate pickattack by knob-gating (no subtraction).
      - full:    demo knobs as-is
      - pa_only: main=0, pickattack=demo (or 1.0), pickreso=0, ir=0
    """
    os.makedirs(out_dir, exist_ok=True)

    # Zero states for streaming graph
    zero_states = _make_zero_state_feeds_for_streaming_model(m_stream)

    # Find fixed T and pad one shot
    in_piezo = None
    for t in m_stream.inputs:
        nm = t.name.split(':')[0].split('/')[-1]
        if nm == "realtime_piezo_input":
            in_piezo = t
            break
    if in_piezo is None:
        raise RuntimeError("Streaming model is missing 'realtime_piezo_input'.")

    T_fixed = int(in_piezo.shape[1]) if in_piezo.shape[1] is not None else len(x_mod)
    core = min(T_fixed, len(x_mod))

    x_pad = np.zeros((1, T_fixed, 1), dtype=np.float32)
    x_pad[0, :core, 0] = x_mod[:core]

    # Predicted style
    st_true  = np.zeros((1, 1), dtype=np.float32)
    st_force = np.zeros((1, 1), dtype=np.float32)

    # Base/demo knobs for full mix
    base_knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned)

    # Demo pickattack linear gain (or unity if you prefer)
    demo_pa_lin = _resolve_demo_knobs("realtime", use_finetuned=use_finetuned)["pickattack"]
    pa_on_val   = float(demo_pa_lin) if apply_demo_boosts else 1.0

    # Pickattack-only knobs: main=0, PR=0, IR=0, PA=pa_on_val
    pa_knobs = prepare_demo_knob_feeds(
        kind="realtime",
        use_finetuned=use_finetuned,
        main=0.0,
        pickattack=pa_on_val,
        pickreso=0.0,
        ir=0.0
    )

    def _predict_with_knobs(knobs_dict):
        feed = {
            "realtime_piezo_input": x_pad,
            "realtime_style_true_input":  st_true,
            "realtime_style_force_input": st_force,
            **zero_states,
            **knobs_dict
        }
        feed = _filter_feed_to_model_inputs(m_stream, feed)
        outs = m_stream.predict(feed, batch_size=1, verbose=0)
        y = np.asarray(outs[0], dtype=np.float32)[0, :, 0]
        return y[:core]

    # Full mix (demo knobs)
    y_full = _predict_with_knobs(base_knobs)

    # Pickattack-only mix (direct isolation)
    y_pa_only = _predict_with_knobs(pa_knobs)

    # Writes
    sf.write(os.path.join(out_dir, "pred_full_keras.wav"), y_full, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "stem_pickattack_only_keras.wav"), y_pa_only, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    if y_mod is not None:
        sf.write(os.path.join(out_dir, "mic_target.wav"), y_mod[:core], REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "piezo_source.wav"), x_mod[:core], REALTIME_SAMPLE_RATE, subtype='FLOAT')

    pa_db = 20.0 * np.log10(pa_on_val + 1e-12)
    print(f"[keras/stream] wrote full and pickattack stem | PA linear={pa_on_val:.4f} ({pa_db:.2f} dB) | "
          f"rms full={_rms(y_full):.6f} pa_only={_rms(y_pa_only):.6f}")
    
def _render_pickattack_diff_tflite(
    tflite_path,
    src_path,
    tgt_path=None,
    sec=10.0,
    offset_sec=0.0,
    out_dir="./out/ab_tflite",
    hop=None,
    use_finetuned=False,
    apply_demo_boosts=True
):
    """
    TFLite: isolate pickattack by knob-gating (no subtraction).
      - full:    demo knobs as-is
      - pa_only: main=0, pickattack=demo (or 1.0), pickreso=0, ir=0
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load/resample
    info = sf.info(src_path)
    sr_in = info.samplerate
    start = int(round(offset_sec * sr_in))
    frames = int(round(sec * sr_in))

    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0].astype(np.float32, copy=False)
    x_mod = resample_to_model_sr(x, sr_in, REALTIME_SAMPLE_RATE) if sr_in != REALTIME_SAMPLE_RATE else x

    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0].astype(np.float32, copy=False)
        y_mod = resample_to_model_sr(y, sr_in, REALTIME_SAMPLE_RATE) if sr_in != REALTIME_SAMPLE_RATE else y

    runner = TFLiteRunner(tflite_path)

    # Demo PA lin or unity
    demo_pa_lin = _resolve_demo_knobs("realtime", use_finetuned=use_finetuned)["pickattack"]
    pa_on_val   = float(demo_pa_lin) if apply_demo_boosts else 1.0
    core_hop    = hop if hop is not None else REALTIME_TFLITE_T

    # Base knobs → full mix
    base_knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned)

    # PA-only knobs: main=0, PR=0, IR=0, PA=pa_on_val
    pa_knobs = prepare_demo_knob_feeds(
        kind="realtime",
        use_finetuned=use_finetuned,
        main=0.0,
        pickattack=pa_on_val,
        pickreso=0.0,
        ir=0.0
    )

    # Full (demo knobs)
    y_full, _ = _tflite_run_full_audio(
        runner, kind="realtime", x_mod=x_mod, sample_rate=REALTIME_SAMPLE_RATE,
        style_mode={"mode": "pred", "value": None},
        knobs=base_knobs,
        fixed_hop=core_hop
    )

    # Pickattack-only (direct isolation)
    y_pa_only, _ = _tflite_run_full_audio(
        runner, kind="realtime", x_mod=x_mod, sample_rate=REALTIME_SAMPLE_RATE,
        style_mode={"mode": "pred", "value": None},
        knobs=pa_knobs,
        fixed_hop=core_hop
    )

    # Writes
    sf.write(os.path.join(out_dir, "pred_full_tflite.wav"), y_full, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "stem_pickattack_only_tflite.wav"), y_pa_only, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    if y_mod is not None:
        sf.write(os.path.join(out_dir, "mic_target.wav"), y_mod, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "piezo_source.wav"), x_mod, REALTIME_SAMPLE_RATE, subtype='FLOAT')

    pa_db = 20.0 * np.log10(pa_on_val + 1e-12)
    print(f"[tflite] wrote full and pickattack stem | PA linear={pa_on_val:.4f} ({pa_db:.2f} dB) | "
          f"rms full={_rms(y_full):.6f} pa_only={_rms(y_pa_only):.6f}")

def _load_and_resample_pair_for_realtime(src_path, tgt_path, sec, offset_sec, sample_rate=REALTIME_SAMPLE_RATE):
    info = sf.info(src_path)
    file_sr = info.samplerate
    start  = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0].astype(np.float32, copy=False)
    if file_sr != sample_rate:
        x_mod = resample_to_model_sr(x, file_sr, sample_rate)
    else:
        x_mod = x

    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0].astype(np.float32, copy=False)
        y_mod = resample_to_model_sr(y, file_sr, sample_rate) if (file_sr != sample_rate) else y
    return x_mod, y_mod

def _make_zero_state_feeds_for_streaming_model(model):
    feed = {}
    for t in model.inputs:
        nm = t.name.split(':')[0].split('/')[-1]
        if nm.startswith("realtime_state_"):
            s1 = int(t.shape[1]) if t.shape[1] is not None else 0
            s2 = int(t.shape[2]) if t.shape[2] is not None else 1
            feed[nm] = np.zeros((1, s1, s2), dtype=np.float32)
    return feed

def _rms(x):
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0

def sweep_pickattack_params_with_tflite(
    src_path,
    tgt_path=None,
    param_grid=None,
    sec=10.0,
    offset_sec=0.0,
    base_out_dir="./out/fast_sweeps_ab",
    load_streaming_from=None,
    hop=REALTIME_TFLITE_T,
    use_finetuned=False,
    apply_demo_boosts=True,
    keras_stream_T=60000
):
    """
    For each param combo:
      1) Build a streaming Keras model with static_T = max(keras_stream_T, len(x_mod)) and render A/B in one predict().
      2) Build a streaming model with static_T=32 and export TFLite; render A/B (difference-of-passes) via the runner.

    keras_stream_T: lower bound for the Keras streaming static T (int). The actual T will be max(keras_stream_T, len(x_mod)).
    """
    # Load/resample once
    x_mod, y_mod = _load_and_resample_pair_for_realtime(
        src_path=src_path,
        tgt_path=tgt_path,
        sec=sec,
        offset_sec=offset_sec,
        sample_rate=REALTIME_SAMPLE_RATE
    )
    core_len = int(x_mod.shape[0])
    T_build = int(max(int(keras_stream_T), core_len))

    # Default param grid
    if param_grid is None:
        param_grid = {
            "softclip_drive": [1.6, 1.8, 2.0],
            "knee_db": [6.0, 10.0, 14.0],
            "smooth_release_ms": [24.0, 36.0],
            "posthp_cutoff_hz": [2800.0, 3200.0, 3600.0]
        }

    # Build all combos
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    def _recurse(i, cur):
        if i == len(keys):
            combos.append(dict(cur))
            return
        k = keys[i]
        for v in values[i]:
            cur[k] = v
            _recurse(i+1, cur)
    _recurse(0, {})

    # Ensure base directory
    os.makedirs(base_out_dir, exist_ok=True)

    print(f"[sweep/ab] {len(combos)} variants… (Keras streaming T={T_build}, TFLite T={REALTIME_TFLITE_T})")

    # Iterate variants
    for idx, cfg in enumerate(combos):
        # Folder name with params
        tag_parts = [f"{k}={str(cfg[k]).replace('.','p')}" for k in keys if k in cfg]
        tag = "_".join(tag_parts) if tag_parts else f"combo_{idx:03d}"
        out_dir = os.path.join(base_out_dir, f"{idx:03d}_{tag}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Keras STREAMING large-T model (build JIT and patch params)
        m_stream_large = build_streaming_rt_with_params(
            load_from_path=load_streaming_from,
            softclip_drive=cfg.get("softclip_drive", None),
            softclip_limit=None,
            env_fast_ms=cfg.get("env_fast_ms", None),
            env_slow_ms=cfg.get("env_slow_ms", None),
            smooth_attack_ms=cfg.get("smooth_attack_ms", None),
            smooth_release_ms=cfg.get("smooth_release_ms", None),
            thr_db=cfg.get("thr_db", None),
            knee_db=cfg.get("knee_db", None),
            post_comp_p=cfg.get("post_comp_p", None),
            post_comp_gain=cfg.get("post_comp_gain", None),
            mask_floor=cfg.get("mask_floor", None),
            mask_out_gain=cfg.get("mask_out_gain", None),
            posthp_cutoff_hz=cfg.get("posthp_cutoff_hz", None),
            static_T=T_build
        )
        out_keras = os.path.join(out_dir, "keras")
        os.makedirs(out_keras, exist_ok=True)
        _render_pickattack_diff_keras_streaming(
            m_stream=m_stream_large,
            x_mod=x_mod,
            y_mod=y_mod,
            use_finetuned=use_finetuned,
            out_dir=out_keras,
            apply_demo_boosts=apply_demo_boosts
        )

        # 2) Streaming T=32 → TFLite export and render via runner (unchanged)
        m_stream_t32 = build_streaming_rt_with_params(
            load_from_path=load_streaming_from,
            softclip_drive=cfg.get("softclip_drive", None),
            softclip_limit=None,
            env_fast_ms=cfg.get("env_fast_ms", None),
            env_slow_ms=cfg.get("env_slow_ms", None),
            smooth_attack_ms=cfg.get("smooth_attack_ms", None),
            smooth_release_ms=cfg.get("smooth_release_ms", None),
            thr_db=cfg.get("thr_db", None),
            knee_db=cfg.get("knee_db", None),
            post_comp_p=cfg.get("post_comp_p", None),
            post_comp_gain=cfg.get("post_comp_gain", None),
            mask_floor=cfg.get("mask_floor", None),
            mask_out_gain=cfg.get("mask_out_gain", None),
            posthp_cutoff_hz=cfg.get("posthp_cutoff_hz", None),
            static_T=REALTIME_TFLITE_T
        )
        tflite_dir = os.path.join(out_dir, "tflite")
        os.makedirs(tflite_dir, exist_ok=True)
        tfl_path = os.path.join(tflite_dir, "model.tflite")
        export_streaming_rt_to_tflite(m_stream_t32, tfl_path, T_fixed=REALTIME_TFLITE_T)

        _render_pickattack_diff_tflite(
            tflite_path=tfl_path,
            src_path=src_path,
            tgt_path=tgt_path,
            sec=sec,
            offset_sec=offset_sec,
            out_dir=tflite_dir,
            hop=hop,
            use_finetuned=use_finetuned,
            apply_demo_boosts=apply_demo_boosts
        )

        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[sweep/ab] Variant {idx+1}/{len(combos)} → {out_dir}")









    
    
    
def _cli_arg_value(args, flag, default=None):
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else default
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _open_output_writer(path, sample_rate):
    _ensure_parent_dir(path)
    ext = os.path.splitext(path)[1].lower()
    kwargs = dict(mode="w", samplerate=int(sample_rate), channels=1)
    if ext == ".wav":
        kwargs["subtype"] = "FLOAT"
    return sf.SoundFile(path, **kwargs)

def _load_input_mono(input_path):
    x2d, sr = sf.read(input_path, dtype="float32", always_2d=True)
    if x2d.size == 0:
        raise RuntimeError(f"Input file is empty: {input_path}")
    if x2d.shape[1] > 1:
        print(f"[process] '{input_path}' has {x2d.shape[1]} channels; using channel 0.")
    return x2d[:, 0].astype(np.float32, copy=False), int(sr)

def _filter_feed_to_model_inputs(model, feed):
    in_names = [_tensor_basename(t) for t in model.inputs]
    return {k: v for (k, v) in feed.items() if k in in_names}

def prepare_process_knob_feeds(kind, use_finetuned=False, pickup=None):
    # --process uses the same runtime baseline make-up gains as the demos and the C++ pedal
    # (see *_RUNTIME_BASELINE_DB), so a file rendered here matches the pedal at 0 dB knobs.
    if kind == "realtime":
        return prepare_demo_knob_feeds(
            kind="realtime",
            use_finetuned=use_finetuned,
            main=KNOB_DEFAULT_MAIN_SCALE,
            pickattack=_db_to_lin(REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB),
            pickreso=_db_to_lin(REALTIME_PICK_RESO_RUNTIME_BASELINE_DB),
            ir=KNOB_DEFAULT_IR_SCALE,
            pickup=pickup
        )
    return prepare_demo_knob_feeds(
        kind="offline",
        use_finetuned=use_finetuned,
        main=KNOB_DEFAULT_MAIN_SCALE,
        pickattack=_db_to_lin(OFFLINE_PICK_ATTACK_RUNTIME_BASELINE_DB),
        pickreso=_db_to_lin(OFFLINE_PICK_RESO_RUNTIME_BASELINE_DB),
        pickup=pickup
    )


def _normalize_process_style(forced_style):
    """
    Normalize a requested process style to:
      - None  -> auto
      - 1.0   -> pick / picked
      - 0.0   -> finger
    Accepts strings or numeric values.
    """
    if forced_style is None:
        return None

    if isinstance(forced_style, (int, float, np.integer, np.floating)):
        v = float(forced_style)
        if abs(v - 1.0) <= 1e-7:
            return 1.0
        if abs(v - 0.0) <= 1e-7:
            return 0.0
        raise ValueError("forced_style must be None, 0/1, or one of: auto, pick, picked, finger")

    s = str(forced_style).strip().lower()
    if s in ("", "auto", "autodetect", "none", "pred", "predicted", "detect", "classifier"):
        return None
    if s in ("pick", "picked", "p", "1", "true", "yes"):
        return 1.0
    if s in ("finger", "fingerstyle", "f", "0", "false", "no"):
        return 0.0

    raise ValueError(
        f"Invalid style '{forced_style}'. Valid values are: auto, pick, picked, finger"
    )


def _parse_process_style_arg(args_list):
    """
    CLI parser for process style.
    Supports:
      --style <auto|pick|picked|finger>
      --style=<...>
      --force-style <...>
      --force-style=<...>
      --pick / --picked
      --finger
    """
    args = set(args_list)

    raw_style = _cli_arg_value(args_list, "--style")
    raw_force_style = _cli_arg_value(args_list, "--force-style")

    if raw_style is not None and str(raw_style).startswith("--"):
        print("ERROR: --style requires a value: auto, pick, picked, or finger")
        sys.exit(1)

    if raw_force_style is not None and str(raw_force_style).startswith("--"):
        print("ERROR: --force-style requires a value: auto, pick, picked, or finger")
        sys.exit(1)

    short_values = []
    if "--pick" in args or "--picked" in args:
        short_values.append("pick")
    if "--finger" in args:
        short_values.append("finger")

    raw_values = [v for v in (raw_style, raw_force_style) if v is not None] + short_values
    if not raw_values:
        return None

    try:
        normalized = [_normalize_process_style(v) for v in raw_values]
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    uniq = {("auto" if v is None else _style_label(v)) for v in normalized}
    if len(uniq) > 1:
        print("ERROR: conflicting style flags. Use only one of --style/--force-style/--pick/--finger.")
        sys.exit(1)

    return normalized[0]


def _normalize_process_pickup(pickup):
    """
    Normalize a requested pickup type to 0.0 (piezo) or 1.0 (magnetic).
    'electric' is an alias for magnetic (electric guitars use magnetic pickups).
    """
    if pickup is None:
        return PICKUP_PIEZO
    if isinstance(pickup, (int, float, np.integer, np.floating)):
        v = float(pickup)
        if abs(v - PICKUP_PIEZO) <= 1e-7:
            return PICKUP_PIEZO
        if abs(v - PICKUP_MAGNETIC) <= 1e-7:
            return PICKUP_MAGNETIC
        raise ValueError("pickup must be one of: piezo, magnetic, electric (or 0/1)")
    s = str(pickup).strip().lower()
    if s in ("", "piezo", "p", "0"):
        return PICKUP_PIEZO
    if s in ("magnetic", "mag", "m", "electric", "e", "1"):
        return PICKUP_MAGNETIC
    raise ValueError(f"Invalid pickup '{pickup}'. Valid values are: piezo, magnetic, electric")


def _parse_process_pickup_arg(args_list):
    """
    CLI parser for pickup type: --pickup <piezo|magnetic|electric> (or --pickup=<...>).
    """
    raw = _cli_arg_value(args_list, "--pickup")
    if raw is not None and str(raw).startswith("--"):
        print("ERROR: --pickup requires a value: piezo, magnetic, or electric")
        sys.exit(1)
    if raw is None:
        return PICKUP_PIEZO
    try:
        return _normalize_process_pickup(raw)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _make_process_style_arrays(forced_style):
    """
    Returns:
      style_true_input  shape (1,1)
      style_force_input shape (1,1)
      normalized_style  None|0.0|1.0
    """
    forced_style = _normalize_process_style(forced_style)

    if forced_style is None:
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            None
        )

    return (
        np.array([[forced_style]], dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        forced_style
    )


def _format_process_write_message(label, output_path, sample_rate, avg_prob=None, forced_style=None):
    msg = f"✅ {label} written: {output_path} ({int(sample_rate)} Hz"

    if forced_style is None:
        if avg_prob is not None:
            msg += f", avg style={_style_label(avg_prob)} ({avg_prob:.4f})"
    else:
        msg += f", forced style={_style_label(forced_style)}"
        if avg_prob is not None:
            msg += f", classifier avg={_style_label(avg_prob)} ({avg_prob:.4f})"

    msg += ")"
    return msg

def _resolve_process_output_paths(input_path, output_arg, do_offline, do_realtime):
    if not do_offline and not do_realtime:
        raise ValueError("Nothing selected; choose offline, realtime, or both.")

    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    looks_like_dir = output_arg.endswith(os.sep) or output_arg.endswith("/") or os.path.isdir(output_arg)
    root, ext = os.path.splitext(output_arg)
    out = {}

    if looks_like_dir:
        os.makedirs(output_arg, exist_ok=True)
        if do_offline:
            out["offline"] = os.path.join(output_arg, f"{input_stem}_offline.wav")
        if do_realtime:
            out["realtime"] = os.path.join(output_arg, f"{input_stem}_realtime.wav")
        return out

    if ext:
        if do_offline and do_realtime:
            out["offline"] = root + "_offline" + ext
            out["realtime"] = root + "_realtime" + ext
        elif do_offline:
            out["offline"] = output_arg
        else:
            out["realtime"] = output_arg
    else:
        if do_offline and do_realtime:
            out["offline"] = output_arg + "_offline.wav"
            out["realtime"] = output_arg + "_realtime.wav"
        elif do_offline:
            out["offline"] = output_arg + ".wav"
        else:
            out["realtime"] = output_arg + ".wav"

    for p in out.values():
        _ensure_parent_dir(p)
    return out

def _style_label(p):
    if p is None:
        return "unknown"
    return "pick" if p >= STYLE_FLAG_PROB_THRESHOLD else "finger"

def process_audio_array_offline(model, x_mod, output_path, use_finetuned=False, block_sec=12.0, forced_style=None, pickup=None,
                                eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)
    style_true_arr, style_force_arr, _ = _make_process_style_arrays(forced_style)

    knobs = prepare_process_knob_feeds("offline", use_finetuned=use_finetuned,
                                       pickup=_normalize_process_pickup(pickup))
    writer = _open_output_writer(output_path, OFFLINE_SAMPLE_RATE)

    left_ctx, right_ctx = _get_model_context_for_streaming(OFFLINE_SAMPLE_RATE, model=model)
    block_core = max(1, int(round(block_sec * OFFLINE_SAMPLE_RATE)))
    eq_state, eq_sos = None, None

    prob_sum = 0.0
    prob_count = 0
    pos = 0
    N = len(x_mod)

    while pos < N:
        core = min(block_core, N - pos)
        lctx = min(left_ctx, pos)
        rctx = min(right_ctx, N - (pos + core))

        x_block = x_mod[pos - lctx: pos + core + rctx]
        x_in = x_block[None, :, None].astype(np.float32, copy=False)

        feed = {
            "offline_piezo_input": x_in,
            "offline_style_true_input": style_true_arr,
            "offline_style_force_input": style_force_arr,
            **knobs
        }
        outs = model.predict(_filter_feed_to_model_inputs(model, feed), batch_size=1, verbose=0)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]

        y_pred = np.asarray(outs[0], dtype=np.float32)[0, :, 0]
        style_prob = float(np.asarray(outs[1], dtype=np.float32)[0, 0]) if len(outs) > 1 else 0.0
        y_core = y_pred[lctx:lctx + core]

        if eq_knobs:
            # Stateful IIR across blocks so block seams stay continuous
            if eq_state is None:
                eq_sos = np.asarray(master_eq_sos(OFFLINE_SAMPLE_RATE, **eq_knobs), dtype=np.float64)
                eq_state = spsig.sosfilt_zi(eq_sos) * 0.0 if len(eq_sos) else None
            if eq_state is not None:
                y_core, eq_state = spsig.sosfilt(eq_sos, y_core.astype(np.float64), zi=eq_state)
                y_core = y_core.astype(np.float32)

        writer.write(y_core)
        prob_sum += style_prob * float(core)
        prob_count += core
        pos += core

    writer.close()
    return (prob_sum / float(prob_count)) if prob_count > 0 else None

def process_audio_array_realtime(model, x_mod, output_path, use_finetuned=False, forced_style=None, pickup=None,
                                 eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)

    knobs = prepare_process_knob_feeds("realtime", use_finetuned=use_finetuned,
                                       pickup=_normalize_process_pickup(pickup))
    runner = KerasStreamingRunner(model)  # hop = T for static graphs, KERAS_STREAM_HOP for dynamic
    res = runner.run(x_mod, knobs, style_true=forced_style)
    y = res["audio"]
    if eq_knobs:
        y = apply_master_eq_iir(y, REALTIME_SAMPLE_RATE, **eq_knobs)
    writer = _open_output_writer(output_path, REALTIME_SAMPLE_RATE)
    writer.write(y)
    writer.close()
    if forced_style is not None:
        # Classifier average for reference (a cheap second pass is not worth it; report n/a)
        return None
    return res["avg_prob"]

def run_process_cli(input_path, output_arg, do_offline=True, do_realtime=True, use_finetuned=False, forced_style=None, pickup=None,
                    eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)
    pickup = _normalize_process_pickup(pickup)
    eq_knobs = dict(eq_knobs or {})

    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    x_raw, input_sr = _load_input_mono(input_path)
    outputs = _resolve_process_output_paths(input_path, output_arg, do_offline, do_realtime)

    if forced_style is None:
        print("[process] style mode: auto")
    else:
        print(f"[process] style mode: forced {_style_label(forced_style)}")
    print(f"[process] pickup: {'magnetic' if pickup >= 0.5 else 'piezo'}")
    if eq_knobs:
        print(f"[process] master EQ: {eq_knobs}")

    if do_offline:
        offline_model_path = _select_stems_model_path("offline", use_finetuned=use_finetuned, prefer_static=False)
        if offline_model_path is None or not os.path.exists(offline_model_path):
            print("ERROR: offline model not found. Run --train-offline first.")
            sys.exit(1)

        print("[process/offline] Loading:", offline_model_path)
        offline_model = load_keras_model(offline_model_path, compile=False)
        x_off = x_raw if input_sr == OFFLINE_SAMPLE_RATE else resample_to_model_sr(x_raw, input_sr, OFFLINE_SAMPLE_RATE)
        p_off = process_audio_array_offline(
            model=offline_model,
            x_mod=x_off,
            output_path=outputs["offline"],
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        print(_format_process_write_message(
            label="Offline",
            output_path=outputs["offline"],
            sample_rate=OFFLINE_SAMPLE_RATE,
            avg_prob=p_off,
            forced_style=forced_style
        ))

    if do_realtime:
        # The dynamic streaming graph is stepped KERAS_STREAM_HOP samples at a time (the style EMA
        # is hop-invariant); the T=32 static graph would need ~90k model calls for a 1-minute file.
        realtime_model_path = _select_stems_model_path("realtime", use_finetuned=use_finetuned, prefer_static=False)
        if realtime_model_path is None or not os.path.exists(realtime_model_path):
            print("ERROR: realtime model not found. Run --train-realtime first.")
            sys.exit(1)

        print("[process/realtime] Loading:", realtime_model_path)
        realtime_model = load_keras_model(realtime_model_path, compile=False)
        x_rt = x_raw if input_sr == REALTIME_SAMPLE_RATE else resample_to_model_sr(x_raw, input_sr, REALTIME_SAMPLE_RATE)
        p_rt = process_audio_array_realtime(
            model=realtime_model,
            x_mod=x_rt,
            output_path=outputs["realtime"],
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        print(_format_process_write_message(
            label="Realtime",
            output_path=outputs["realtime"],
            sample_rate=REALTIME_SAMPLE_RATE,
            avg_prob=p_rt,
            forced_style=forced_style
        ))

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

        # Flags:
        # --static-realtime or --static: step the static T=32 streaming model exactly like the pedal
        #                                (default: dynamic streaming model at KERAS_STREAM_HOP, much faster)
        # --static-offline: force offline static artifact (if present)
        prefer_static_rt  = ("--static-realtime" in args) or ("--static" in args)
        prefer_static_off = ("--static-offline" in args)

        time_start = time.time()

        # Offline baseline: load saved offline model (dynamic by default unless --static-offline)
        if do_offline:
            prepare_and_generate_demo(
                kind="offline",
                sample_rate=OFFLINE_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_OFFLINE_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
                prefer_static=prefer_static_off
            )

        # Realtime baseline: dynamic streaming model by default; --static/--static-realtime → T=32 model
        if do_realtime:
            prepare_and_generate_demo(
                kind="realtime",
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_REALTIME_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
                prefer_static=prefer_static_rt
            )

        time_end = time.time()
        print(f"✅ Demo generation complete (took {time_end - time_start:.1f} sec).")
        sys.exit(0)
    
    elif len(sys.argv)>1 and sys.argv[1] in ("--sweep-pickattack", "--foo"):
        # Developer tool: A/B the pick-attack detector parameters through Keras streaming + TFLite
        assert_has_pairs()
        sweep_pickattack_params_with_tflite(
            src_path=all_pairs[DEMO_PAIR_INDEX][0],
            tgt_path=all_pairs[DEMO_PAIR_INDEX][1],
            param_grid={
                "knee_db":       [REALTIME_PICK_MASK_KNEE_DB], # [4.0],
                "posthp_cutoff_hz": [REALTIME_PICK_ATTACK_CUTOFF_HZ] # [1000.0, 3000.0, 4000.0, 5000.0, 6000.0]
            },
            sec=5.0,
            offset_sec=480.0,
            base_out_dir="./out/fast_sweeps_ab",
            load_streaming_from=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),  # or None for default
            hop=REALTIME_TFLITE_T,
            keras_stream_T=120000  # large T for Keras streaming
        )
    
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

        offline_dyn  = load_keras_model(offline_dyn_path,  compile=False) if os.path.exists(offline_dyn_path)  else None
        realtime_dyn = load_keras_model(realtime_dyn_path, compile=False) if os.path.exists(realtime_dyn_path) else None

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
        # --finetuned        → finetuned dynamic/static models (if present)
        # --both             → baseline + finetuned
        # --all              → baseline + finetuned
        # --offline-only / --realtime-only limits branches
        # New:
        # --static or --static-realtime → force realtime STATIC T=32 model for stems
        args = set(sys.argv[2:])

        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)
        do_finetuned = ("--finetuned"     in args)
        do_both      = ("--both"          in args)
        do_all       = ("--all"           in args)

        # New static realtime selector
        prefer_static_rt = ("--static-realtime" in args) or ("--static" in args)

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

        # REALTIME: baseline / finetuned; prefer static T=32 if requested
        if do_realtime:
            if do_all:
                run_render_stems_realtime(use_finetuned=False, prefer_static=prefer_static_rt)
                run_render_stems_realtime(use_finetuned=True,  prefer_static=prefer_static_rt)
            elif do_both:
                run_render_stems_realtime(use_finetuned=False, prefer_static=prefer_static_rt)
                run_render_stems_realtime(use_finetuned=True,  prefer_static=prefer_static_rt)
            else:
                run_render_stems_realtime(use_finetuned=do_finetuned, prefer_static=prefer_static_rt)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--render-stems-tflite":
        # Same options as --render-stems:
        #   --offline-only / --realtime-only
        #   --finetuned
        #   --both | --all
        #   --hop=N  (for realtime streaming hop)
        run_render_stems_tflite(sys.argv[2:])
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
            m = load_keras_model(p, compile=False)
            
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
        m = load_keras_model(path, compile=False)
        _print_model_io_summary(m)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-tflite-file":
        if len(sys.argv) < 3:
            print("Usage: --inspect-tflite-file <path.tflite>"); sys.exit(1)
        path = sys.argv[2]
        _inspect_tflite_file(path)
        try:
            verify_tflite_bundle_file(path, strict=False)
        except Exception as e:
            print("verify_tflite_bundle_file (non-strict) raised:", e)
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-tflite-all":
        # Convenience: check both baseline and finetuned realtime TFLite files if present
        paths = [
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TFLITE_FP32_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TFLITE_FP32_FILENAME),
        ]
        for p in paths:
            if os.path.exists(p):
                print("\n=== Inspecting:", p, "===")
                _inspect_tflite_file(p)
                try:
                    verify_tflite_bundle_file(p, strict=False)
                except Exception as e:
                    print("verify_tflite_bundle_file (non-strict) raised:", e)
            else:
                print("\n=== Missing:", p, "===")
        sys.exit(0)
    
    elif len(sys.argv)>1 and sys.argv[1]=="--export-tflite":
        
        # load the streaming model to use for conversion
        #m = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME), compile=False)
        #debug_streaming_state_lengths(m)
        
        
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
    elif len(sys.argv) > 1 and sys.argv[1] == "--process":
        args_list = sys.argv[2:]
        args = set(args_list)

        input_path = _cli_arg_value(args_list, "--input")
        output_arg = _cli_arg_value(args_list, "--output")

        if input_path is not None and str(input_path).startswith("--"):
            input_path = None
        if output_arg is not None and str(output_arg).startswith("--"):
            output_arg = None

        forced_style = _parse_process_style_arg(args_list)
        pickup = _parse_process_pickup_arg(args_list)
        try:
            eq_knobs = parse_master_eq_arg(_cli_arg_value(args_list, "--eq"))
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if input_path is None or output_arg is None:
            print(
                "Usage:\n"
                "  python piezo2mic.py --process --input <input_audio> --output <output_path> "
                "[--offline-only|--realtime-only|--both] [--finetuned] "
                "[--style <auto|pick|picked|finger>] [--pickup <piezo|magnetic|electric>]\n"
                "  [--eq lc_hz=80,lc_slope=12,ls_gain_db=0,ls_hz=200,ls_slope=12,mid_gain_db=0,mid_hz=1000,mid_q=1,"
                "hs_gain_db=0,hs_hz=4000,hs_slope=12]\n\n"
                "Style aliases:\n"
                "  --force-style <auto|pick|picked|finger>\n"
                "  --pick / --picked\n"
                "  --finger\n\n"
                "Notes:\n"
                "  - If no style is provided, style is auto-detected (same behavior as today).\n"
                "  - If a style is forced, the render uses that style.\n"
                "  - --pickup declares the source pickup type (default piezo). Use magnetic (or\n"
                "    its alias electric) for magnetic-pickup sources, e.g. electric guitars.\n"
                "  - The printed style average is still the classifier prediction, for reference.\n"
                "  - If neither --offline-only nor --realtime-only is passed, both models are run.\n"
                "  - If both models are run, --output is treated as a prefix or directory and\n"
                "    '_offline' / '_realtime' are appended automatically.\n"
                "  - Output is written at the model sample rate: offline=96k, realtime=48k.\n"
                "  - --eq applies the same post-model master EQ the pedal runs (RBJ biquads); default = off.\n"
            )
            sys.exit(1)

        do_offline = ("--realtime-only" not in args)
        do_realtime = ("--offline-only" not in args)

        if not do_offline and not do_realtime:
            print("ERROR: choose --offline-only, --realtime-only, or neither/--both.")
            sys.exit(1)

        use_finetuned = ("--finetuned" in args)

        run_process_cli(
            input_path=input_path,
            output_arg=output_arg,
            do_offline=do_offline,
            do_realtime=do_realtime,
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-realtime":
        migrate_realtime_models_inplace()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-offline":
        migrate_offline_models_inplace()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-all":
        migrate_offline_models_inplace()
        migrate_realtime_models_inplace()
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()