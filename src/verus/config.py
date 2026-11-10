import numpy as np
import os

SEED = 42   # global RNG seed; the determinism env vars are set in verus/__init__.py before TensorFlow loads


# On/off flags
DISABLE_DEMO_SCALE_BOOSTS = False

EPOCHS      = 40      # realtime (the KD schedules span it)
# Offline epochs are OFFLINE_MAX_STEPS_PER_EPOCH steps: a full pass over the chunks at batch 1 is ~31k
# steps (~17 h on the dev Mac's CPU), far too coarse for checkpoints / LR plateau / early stopping.
# 12 × 2000 steps (~14 h) is the warm-start budget (`--train-offline --init-from`, TRAINING.md §5).
OFFLINE_EPOCHS               = 12
OFFLINE_MAX_STEPS_PER_EPOCH  = 2000   # None = one pass over the training chunks (50 % overlap)
REALTIME_MAX_STEPS_PER_EPOCH = None
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

# Batch params. Offline, measured on the 16 GB Apple Silicon dev Mac (CPU, adversarial objective):
# batch 1 ≈ 5.3 GB peak RSS at 2.0 s/step; batch 2 ≈ 7.4 GB (starts swapping); batch 4 > 8 GB (swapped
# the machine until the disk filled). The Metal GPU was no faster and took more of the shared memory.
# Use 4 on a machine with the headroom.
OFFLINE_BATCH_SIZE          = 1
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

OFFLINE_STATIC_MODEL_FILENAME  = "verus_baseline_offline.keras"
OFFLINE_DYNAMIC_MODEL_FILENAME = "verus_baseline_offline_dynamic.keras"

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

# Style blend (training only): the style that drives FiLM/heads for NON-forced samples is
# (1-a)*student_prob + a*teacher_prob. It is SCHEDULED (see KD_STYLE_BLEND_SCHEDULE), not trained:
# a trainable alpha was overwritten by the scheduler every epoch anyway, and gradient descent pulls
# it toward the (better) teacher — away from what inference has (the student's own style only).
REALTIME_STYLE_BLEND_INIT           = 0.7
REALTIME_STYLE_BLEND_TRAINABLE      = False

# KD annealing schedule: layer name -> (start_alpha, end_alpha, start_epoch, end_epoch).
# Used by KDAlphaScheduler during base training; fine-tuning distills at the end values. The
# anneal spans the actual training length (it used to end at epoch 80 while EPOCHS=40, so neither
# the base run nor the "end-of-anneal" fine-tune strengths were ever reached consistently).
KD_ANNEAL_END_EPOCH = EPOCHS
KD_ALPHA_SCHEDULE = {
    "audio_distill":       (REALTIME_DISTILL_ALPHA,              2.0,  0,  KD_ANNEAL_END_EPOCH),
    "audio_stft_distill":  (REALTIME_AUDIO_STFT_DISTILL_ALPHA,   0.2,  0,  KD_ANNEAL_END_EPOCH),
    "feat_distill_0":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  KD_ANNEAL_END_EPOCH),
    "feat_distill_1":      (REALTIME_FEAT_DISTILL_ALPHA,         0.5,  0,  KD_ANNEAL_END_EPOCH),
    "stem_pick_distill":   (REALTIME_STEM_PICK_DISTILL_ALPHA,    0.2,  0,  KD_ANNEAL_END_EPOCH),
    "stem_finger_distill": (REALTIME_STEM_FINGER_DISTILL_ALPHA,  0.2,  0,  KD_ANNEAL_END_EPOCH),
}
# The blend reaches 0 (student style only, exactly like inference) for the last quarter of training
KD_STYLE_BLEND_SCHEDULE = {"realtime_style_blend": (REALTIME_STYLE_BLEND_INIT, 0.0, 0, (3 * EPOCHS) // 4)}

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
# runtime/src/state.hpp — keep them in sync. The demo boosts and --process defaults derive from these so
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
REALTIME_STATIC_MODEL_FILENAME  = "verus_baseline_realtime.keras"
REALTIME_DYNAMIC_MODEL_FILENAME = "verus_baseline_realtime_dynamic.keras"

# This represents a trainable version of the realtime model, where the entire model is trainable (e.g. non-streamable), and have the offline teacher baked into their graphs
REALTIME_TRAINABLE_MODEL_FILENAME  = "verus_baseline_trainable_realtime.keras"

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
FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME  = "verus_finetuned_trainable_realtime.keras"
FINETUNE_REALTIME_STATIC_MODEL_FILENAME  = "verus_finetuned_realtime.keras"
FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME = "verus_finetuned_realtime_dynamic.keras"
FINETUNE_OFFLINE_STATIC_MODEL_FILENAME  = "verus_finetuned_offline.keras"
FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME = "verus_finetuned_offline_dynamic.keras"

OFFLINE_TFLITE_T = 40960
REALTIME_TFLITE_T = 32
OFFLINE_TFLITE_FP32_FILENAME = f"verus_baseline_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
REALTIME_TFLITE_FP32_FILENAME = f"verus_baseline_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"
FINETUNE_OFFLINE_TFLITE_FP32_FILENAME = f"verus_finetuned_offline_t{OFFLINE_TFLITE_T}_fp32.tflite"
FINETUNE_REALTIME_TFLITE_FP32_FILENAME = f"verus_finetuned_realtime_t{REALTIME_TFLITE_T}_fp32.tflite"

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
LMAG_WEIGHT               = 2.0    # log-magnitude HF STFT term (only when a phase cutoff is set, see below)

# ─── Phase-sensitive vs magnitude-only supervision (pick-attack fidelity) ───────
# Above a few kHz the mic's content at pick attacks is only ~1-10 % phase-predictable from the
# piezo (verus --analyze-attacks; the pick's contact click even precedes the string
# release the piezo registers by ~7 ms). Phase-sensitive losses (waveform L1, pre-emphasis HF L1,
# second difference, complex STFT) are minimised there by predicting ~nothing, which is exactly the
# 12-13 dB attack-HF deficit the old offline model had on picked playing. With a cutoff set, those
# terms only see the band below it; above it the model is supervised on (log) magnitude only
# ('lmag' + the existing linear-magnitude MR-STFT and log-mel) and, for the offline model, by an
# adversarial spectrogram discriminator (OFFLINE_ADVERSARIAL). None = legacy full-band behaviour.
OFFLINE_PHASE_LOSS_CUTOFF_HZ  = 3000.0
REALTIME_PHASE_LOSS_CUTOFF_HZ = None     # realtime unchanged for now (it cannot place the click causally)
PHASE_LOSS_LPF_TAPS           = 255      # zero-phase FIR (loss only) at 96k; halved at 48k
LMAG_SCALES_OFFLINE  = ((128, 32), (256, 64), (512, 128), (1024, 256))   # (win = n_fft, hop) @96k
LMAG_SCALES_REALTIME = ((64, 16), (128, 32), (256, 64), (512, 128))      # same durations @48k
LMAG_FLOOR_DB        = -60.0   # per-example floor re the target's loudest bin (ignores hiss / silence)

# ─── Adversarial training of the offline model ──────────────────────────────────
# Multi-resolution log-spectrogram discriminator (LSGAN) + feature matching, trained alongside the
# offline model (AdversarialTrainer). Weights are relative to the calibrated reconstruction loss.
OFFLINE_ADVERSARIAL       = True
OFFLINE_ADV_WEIGHT        = 0.1
OFFLINE_FM_WEIGHT         = 0.2
OFFLINE_ADV_START_EPOCH   = 1      # the generator gets adversarial gradients from this epoch (D trains from epoch 0)
OFFLINE_ADV_RAMP_EPOCHS   = 2      # ... ramped in linearly over this many epochs
DISC_LR                   = 1e-4
DISC_RESOLUTIONS_OFFLINE  = ((256, 64), (512, 128), (1024, 256))
DISC_MIN_HZ               = 1000.0  # the discriminator judges the band the magnitude losses own
DISC_CHANNELS             = 32

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
# Segments the loss-weight calibration averages over (from the fixed val set; capped by its size). The
# weights are ratios of term averages and the per-segment terms are heavy-tailed, so a small sample makes
# the objective depend on which segments were drawn: on the warm-start offline model, 5 segments (the
# old 5 batches at batch 1) put STFT ~10x and HF ~3x off; even disjoint 64-segment slices spread 2x.
CALIB_SEGMENTS = 256

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

def _ema_fir_len(ms, sr, scale, Lmax):
    s = max(1, int(round((float(ms) / 1000.0) * float(sr))))
    return int(np.clip(np.ceil(scale * s), 8, Lmax))

def _pick_mask_left_context(sr, hp_order, fast_ms, slow_ms, att_ms, rel_ms):
    """Left context of CausalPickDBMask / StreamingPickDBMask: HPF → max(env FIRs) → max(smoothers)."""
    return ((hp_order - 1)
            + max(_ema_fir_len(fast_ms, sr, 3.0, 1024), _ema_fir_len(slow_ms, sr, 3.0, 4096)) - 1
            + max(_ema_fir_len(att_ms, sr, 4.0, 2048), _ema_fir_len(rel_ms, sr, 4.0, 2048)) - 1)

# Tap counts of the in-graph colour EQ bandpasses (add_coloreq / the streaming coloreq)
OFFLINE_COLOREQ_ORDER  = 97
REALTIME_COLOREQ_ORDER = 65

def _offline_train_left_context():
    """
    Max left context over the offline branches that feed the final sum. The colour EQ runs on the
    SUM of the branches and the pickup adapter feeds all of them, so both ADD to the branch maximum
    (the pre-2026-09-22 formula took the main stack only and missed the coloreq's 96 samples).
    """
    sr = OFFLINE_SAMPLE_RATE
    lc_main = ((OFFLINE_RECEPTIVE_FIELD - 1) + OFFLINE_PRE_BLOCKS * (OFFLINE_PRE_KERNEL - 1)
               - _offline_sum_right_pads())                                         # sum of per-layer left pads
    lc_mask = _pick_mask_left_context(sr, OFFLINE_HF_HP_ORDER, OFFLINE_PICK_MASK_ENV_FAST_MS,
                                      OFFLINE_PICK_MASK_ENV_SLOW_MS, OFFLINE_PICK_MASK_SMOOTH_ATTACK_MS,
                                      OFFLINE_PICK_MASK_SMOOTH_RELEASE_MS)
    lc_pa = 1 + lc_mask + (7 - 1) * 1 + (7 - 1) * 2 + (PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1)
    lc_strike = lc_mask + 1 + (_ema_fir_len(RESBP_STRIKE_ENV_MS_OFFLINE, sr, 4.0, 4096) - 1)
    lc_ctrl = (OFFLINE_PICK_RESO_BP_ORDER - 1) + (_ema_fir_len(4.0, sr, 4.0, 1024) - 1)
    lc_pr = max(lc_strike, lc_ctrl) + (PICK_RESO_MODAL_LEN_OFFLINE - 1) + (PICK_RESO_LOWBAND_PENALTY_ORDER - 1)
    return int(max(lc_main, lc_pa, lc_pr) + (OFFLINE_COLOREQ_ORDER - 1) + pickup_adapter_left_context())

OFFLINE_TRAIN_RIGHT_CTX = _offline_sum_right_pads()          # sum of per-layer look-aheads (all other branches are causal)
OFFLINE_TRAIN_LEFT_CTX  = _offline_train_left_context()
OFFLINE_SEG_LEN         = OFFLINE_TRAIN_LEFT_CTX + OFFLINE_TRAIN_BLOCK_SIZE + OFFLINE_TRAIN_RIGHT_CTX
OFFLINE_LOSS_CORE       = (OFFLINE_TRAIN_LEFT_CTX, OFFLINE_TRAIN_LEFT_CTX + OFFLINE_TRAIN_BLOCK_SIZE)

def _realtime_branch_left_contexts():
    """
    Design-order left context of every realtime branch that feeds the final sum (mirrors
    build_realtime_streaming_model; the colour EQ and the pickup adapter are NOT included — they
    add to all branches, see _realtime_train_left_context).
    """
    sr = REALTIME_SAMPLE_RATE
    lc_mask = _pick_mask_left_context(sr, REALTIME_HF_HP_ORDER, REALTIME_PICK_MASK_ENV_FAST_MS,
                                      REALTIME_PICK_MASK_ENV_SLOW_MS, REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
                                      REALTIME_PICK_MASK_SMOOTH_RELEASE_MS)
    lc_pa = 1 + lc_mask + (7 - 1) * 1 + (7 - 1) * 2 + (PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1)
    L_strike = _ema_fir_len(RESBP_STRIKE_ENV_MS_REALTIME, sr, 4.0, 512)
    lc_ctrl = (REALTIME_PICK_RESO_BP_ORDER - 1) + (_ema_fir_len(3.0, sr, 4.0, 256) - 1)
    L_lr = (PICK_RESO_MODAL_LEN_REALTIME - 1) // REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR + 1
    lc_pr = (max(lc_mask + 1 + (L_strike - 1), lc_ctrl) + (REALTIME_PICK_RESO_DS_TAPS - 1)
             + REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR * (L_lr - 1)
             + (REALTIME_PICK_RESO_INTERP_TAPS - 1) + (PICK_RESO_LOWBAND_PENALTY_ORDER - 1))
    R_ir = 1 + (REALTIME_IR_KERNEL - 1) * sum(REALTIME_IR_DILATIONS)
    lc_ir = (REALTIME_IR_FIR_SIZE - 1) + REALTIME_IR_DOWNSAMPLE_FACTOR * (R_ir - 1) + (max(193, REALTIME_LP_ORDER) - 1)
    lc_main = REALTIME_PRE_BLOCKS * (REALTIME_PRE_KERNEL - 1) + (REALTIME_KERNEL_SIZE - 1) * sum(REALTIME_DILATIONS)
    return {"main": int(lc_main), "ir": int(lc_ir), "pickattack": int(lc_pa), "pickreso": int(lc_pr)}

def _realtime_train_left_context():
    """
    Conservative (design-order, not min-phase-probed) left context of the realtime model: the max
    over the branches, PLUS the colour EQ (it filters the branch sum) and the pickup adapter (it
    feeds every branch). Over-estimating by a few samples only makes the segment slightly longer.
    """
    return int(max(_realtime_branch_left_contexts().values())
               + (REALTIME_COLOREQ_ORDER - 1) + pickup_adapter_left_context())

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
VAL_MAX_SEGMENTS    = {"offline": 256, "realtime": 512}   # fixed validation set size (segments, whatever the batch size); cached in RAM for determinism

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









    
    
    
# ─────────────────────────────────────────────────────────────────────────────
# ATTACK ANALYSIS (--analyze-attacks): how well does a model reproduce the mic's attacks?
# ─────────────────────────────────────────────────────────────────────────────
# Validation loss says little about the pick attack (17 % of the time, ~1/3 of the mic's HF energy
# on picked takes). This renders the held-out tail of each training group and reports, per band:
#   attack  = energy in the attack windows (-1..+15 ms around each mic onset), model vs mic (dB)
#   between = energy outside them, model vs mic (dB) — boosted pick branches leak here
#   coh     = phase coherence model↔mic at the attacks (piezo↔mic shown as the linear reference)
# plus the onset-triggered 6-12 kHz time course (mic / piezo / model). Results go to
# out/analysis/attacks_<kind>[_finetuned][_astrained].json and are diffed against the previous run.

ATTACK_BANDS = ((1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
ATTACK_PRE_MS, ATTACK_POST_MS = 1.0, 15.0
ATTACK_TIMING_GRID_MS = ((-2, 0), (0, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 60))
ANALYSIS_OUT_DIR = os.path.join(OUT_DIR, "analysis")

ANALYSIS_EXCERPT_POSITIONS = (0.82, 0.88, 0.94)   # fractions of the take: all inside the held-out last 20 %
ANALYSIS_PREROLL_SEC = 0.3                       # rendered but not scored (model context / style EMA warm-up)

def ensure_output_dirs():
    """Create the output directories (done once at CLI start-up, not on import)."""
    for dir_path in [
        CHECKPOINT_OFFLINE_OUT_DIR, CHECKPOINT_REALTIME_OUT_DIR,
        LOGS_OFFLINE_OUT_DIR, LOGS_REALTIME_OUT_DIR, DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
        DEMO_CHECKPOINTS_REALTIME_OUT_DIR, LONG_DEMOS_OFFLINE_OUT_DIR, LONG_DEMOS_REALTIME_OUT_DIR,
        FINETUNE_DEMOS_OUT_DIR, MODELS_OFFLINE_OUT_DIR, MODELS_REALTIME_OUT_DIR,
        STEMS_OFFLINE_OUT_DIR, STEMS_REALTIME_OUT_DIR,
        STEMS_FINETUNE_OFFLINE_OUT_DIR, STEMS_FINETUNE_REALTIME_OUT_DIR
    ]:
        os.makedirs(dir_path, exist_ok=True)

