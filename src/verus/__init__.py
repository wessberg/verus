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
  The realtime TRAINING graph embeds the offline teacher for distillation; the realtime STREAMING graphs (static T=32 for the pedal/TFLite, dynamic T for Python inference) are self-contained and receive the trained weights by layer name (copy_weights_by_name).
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

Knowledge distillation (realtime training only): the real (non-causal) offline teacher runs frozen, in
inference mode, inside the training graph with extra left/right context, and the student is pulled toward
it with several annealed KD losses (see KDAlphaScheduler; training-time only, never part of val_loss):
- audio_distill:        waveform L1 between student audio (48k) and teacher audio (96k, decimated with the
                        same zero-phase anti-alias filter resample_poly used for the 48k targets)
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

import os
import random
import sys

from .config import SEED

# Determinism env vars must be set BEFORE TensorFlow is imported, otherwise they are ignored.
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import tensorflow as tf

if "--cpu" in sys.argv:
    # Run on the CPU even with tensorflow-metal installed (TRAINING.md §1: on a 16 GB Mac the GPU is
    # no faster for these models and takes more of the shared memory). Must happen before any TF op.
    tf.config.set_visible_devices([], "GPU")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

__version__ = "0.1.0"
