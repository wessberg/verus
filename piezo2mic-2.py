#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal.

- Automatic mic→piezo alignment using GCC-PHAT and fractional (sub-sample) shift (disabled by default because inputs are pre-aligned offline)
- Look-ahead (bounded, acausal) teacher to better model pick onsets
- Onset-gated HF branch in both teacher and student
- Stabilized FiLM (gamma≈1, beta≈0)
- Pre-dilation micro-kernel blocks for local HF structure
- Onset/HF-focused spectral and time-domain loss shaping
- Per-output labels and sample weights provided as dicts keyed by output names

Everything is fully Keras serializable.
- The teacher is a static offline model, and therefore non-causal and has look-ahead.
- The student is a dynamic model, and therefore causal and has no look-ahead.

=============================================================
Style forcing blend (base training) and clean finetuning:
=============================================================

During base training, both teacher and student expose two scalar style inputs (style_true_input and style_force_input)
and use style_used = (1 - force) * style_pred + force * style_true for the pick/finger gate.
We feed force=1 (teacher forcing) to stabilize learning of the heads while the classifier is still weak,
and train the classifier with BCE.

At inference we can feed force=0 so the model relies on its own classifier.

For finetuning, we do not have any true style labels, as we accept general audio that may combine playing styles.
We build the student without forcing the style and apply dummy-style inputs (force=0) to the model, so it always uses its own classifier.
It inherits the style head by name from the base model, and disable the style BCE loss.
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
    Input, Conv1D, SeparableConv1D, ZeroPadding1D, GlobalAveragePooling1D,
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
TEACHER_SAMPLE_RATE = 96000
STUDENT_SAMPLE_RATE = 48000
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
DEBUG               = False

# Batch params
BATCH_SIZE          = 8

# Chunk ratios
PICK_CHUNK_RATIO = 0.5
FINGER_CHUNK_RATIO = 0.5

# Teacher params
TEACHER_FILTERS                = 128
TEACHER_KERNEL_SIZE            = 51
TEACHER_DILATIONS              = (1,2,4,8,16,32,64,128)
TEACHER_PRE_BLOCKS             = 2
TEACHER_PRE_KERNEL             = 7
# Bounded look-ahead per layer (in ms):
TEACHER_LOOKAHEAD_MS           = 4.0
TEACHER_HF_ONSET_SMOOTH_MS     = 3.0
TEACHER_LOOKAHEAD_SAMPLES      = int(round(TEACHER_SAMPLE_RATE * TEACHER_LOOKAHEAD_MS / 1000.0))
TEACHER_RECEPTIVE_FIELD        = 1 + (TEACHER_KERNEL_SIZE - 1) * sum(TEACHER_DILATIONS)
TEACHER_TRAIN_BLOCK_SIZE       = 4096
TEACHER_SEG_LEN                = TEACHER_RECEPTIVE_FIELD + TEACHER_TRAIN_BLOCK_SIZE
TEACHER_EMBED_DIM              = 32
TEACHER_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_teacher.keras"
TEACHER_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_teacher_dynamic.keras"

# Student params
STUDENT_FILTERS          = 48
STUDENT_KERNEL_SIZE      = 25
STUDENT_DILATIONS        = (1,2,4,8)
STUDENT_PRE_BLOCKS       = 1
STUDENT_PRE_KERNEL       = 5
STUDENT_RECEPTIVE_FIELD  = 1 + (STUDENT_KERNEL_SIZE - 1) * sum(STUDENT_DILATIONS)
STUDENT_TRAIN_BLOCK_SIZE = 4096
STUDENT_SEG_LEN          = STUDENT_RECEPTIVE_FIELD + STUDENT_TRAIN_BLOCK_SIZE
STUDENT_EMBED_DIM        = 16
STUDENT_DISTILL_ALPHA    = 5.0
STUDENT_TEACHER_FEAT_LAYERS   = ["res_add_2", "res_add_3"]
STUDENT_FEAT_DISTILL_ALPHA    = 1.0
STUDENT_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_student.keras"
STUDENT_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_student_dynamic.keras"

# Finetune params
FINETUNE_TEACHER_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_teacher.keras"
FINETUNE_TEACHER_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_teacher_dynamic.keras"
FINETUNE_STUDENT_STATIC_MODEL_FILENAME  = "piezo2mic_finetuned_student.keras"
FINETUNE_STUDENT_DYNAMIC_MODEL_FILENAME = "piezo2mic_finetuned_student_dynamic.keras"

FINETUNE_TEACHER_EPOCHS      = 10
FINETUNE_STUDENT_EPOCHS      = 10
FINETUNE_TEACHER_LR          = 1e-5
FINETUNE_STUDENT_LR          = 1e-4
FINETUNE_L1_WEIGHT           = 15.0
FINETUNE_MEL_WEIGHT          = 15.0

FINETUNE_DEMO_SEC = 60
FINETUNE_DEMO_OFFSET_SEC = 0

# Onset/HF loss shaping
HP_PRE_EMPHASIS           = 0.8
HP_PRE_EMPHASIS_HF        = 0.98
ONSET_SMOOTH_MS           = 12.0
HF_ONSET_GAIN             = 6.0
STFT_HF_ONSET_GAIN        = 4.0
STFT_HIGH_SPLIT_HZ        = 6000.0
STFT_HIGH_BOOST           = 5.0

L1_WEIGHT                 = 1.0
HF_WEIGHT                 = 1.0
STFT_WEIGHT               = 1.0
MEL_WEIGHT                = 1.0
RMS_ENV_WEIGHT            = 1.0
COMPLEX_STFT_WEIGHT       = 0.5
D2_WEIGHT                 = 0.2
LAMBDA_STYLE_RATIO        = 0.5

# ──────── TUNING KNOBS ─────────────────────────────────────────────────────────
ATTACK_SEGMENT_GENERATOR_MIN_DIFF_THRESH = 1e-6
ATTACK_SEGMENT_GENERATOR_MIN_ENV_THRESH  = 1e-5
ATTACK_SEGMENT_GENERATOR_MIN_AMP_THRESH  = 1e-6
SEGMENT_GENERATOR_MIN_AMP_THRESH         = 1e-5
#───────────────────────────────────────────────────────────────────────────────

# ─── IR branch FIR FILTER DESIGN ────────────────────────────────────────────
STUDENT_IR_DOWNSAMPLE_FACTOR = 8
STUDENT_IR_FIR_SIZE          = 129
STUDENT_IR_CUTOFF            = 1.0 / STUDENT_IR_DOWNSAMPLE_FACTOR
STUDENT_IR_KERNEL    = 33
STUDENT_IR_DILATIONS = (1,2,4,8)
STUDENT_IR_ALIGN_OFFSET_CORRECTION = -64
STUDENT_TOTAL_EXPECTED_IR_DELAY = (STUDENT_IR_FIR_SIZE - 1) + ((STUDENT_IR_FIR_SIZE - 1)//2)

coarse_steps = math.ceil((STUDENT_IR_FIR_SIZE - 1) / STUDENT_IR_DOWNSAMPLE_FACTOR)
IR_PREDELAY = coarse_steps * STUDENT_IR_DOWNSAMPLE_FACTOR

STUDENT_LP_ORDER   = 65
STUDENT_LP_CUTOFF  = 4000

student_lp_taps_np = firwin(STUDENT_LP_ORDER,
                     cutoff=STUDENT_LP_CUTOFF/(STUDENT_SAMPLE_RATE/2),
                     window="hamming",
                     pass_zero=True).astype(np.float32)
student_lp_taps_tf = tf.constant(student_lp_taps_np.reshape(-1,1,1), dtype=tf.float32)

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

OUT_DIR        = "./out2"

CHECKPOINT_OUT_DIR = os.path.join(OUT_DIR, "checkpoints")
CHECKPOINT_TEACHER_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "teacher")
CHECKPOINT_STUDENT_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "student")

MODELS_OUT_DIR = os.path.join(OUT_DIR, "models")
MODELS_TEACHER_OUT_DIR = os.path.join(MODELS_OUT_DIR, "teacher")
MODELS_STUDENT_OUT_DIR = os.path.join(MODELS_OUT_DIR, "student")

LOGS_OUT_DIR = os.path.join(OUT_DIR, "logs")
LOGS_TEACHER_OUT_DIR = os.path.join(LOGS_OUT_DIR, "teacher")
LOGS_STUDENT_OUT_DIR = os.path.join(LOGS_OUT_DIR, "student")

DEMOS_OUT_DIR = os.path.join(OUT_DIR, "demos")

LONG_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "long")
LONG_DEMOS_TEACHER_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "teacher")
LONG_DEMOS_STUDENT_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "student")

FINETUNE_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "finetune")

DEMO_GENERATOR_OUTPUTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "generator-outputs")

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_TEACHER_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "teacher")
DEMO_CHECKPOINTS_STUDENT_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "student")

SAVE_WEIGHTS           = False

for dir_path in [DEMO_GENERATOR_OUTPUTS_OUT_DIR, CHECKPOINT_TEACHER_OUT_DIR, CHECKPOINT_STUDENT_OUT_DIR, LOGS_TEACHER_OUT_DIR, LOGS_STUDENT_OUT_DIR, DEMO_CHECKPOINTS_TEACHER_OUT_DIR, DEMO_CHECKPOINTS_STUDENT_OUT_DIR, LONG_DEMOS_TEACHER_OUT_DIR, LONG_DEMOS_STUDENT_OUT_DIR, FINETUNE_DEMOS_OUT_DIR, MODELS_TEACHER_OUT_DIR, MODELS_STUDENT_OUT_DIR]:
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
    down = audio_sample_rate           // g
    return resample_poly(audio, up, down)

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
    chunks = []
    chunk_len = CHUNK_SEC * sample_rate
    for src,tgt in pairs_list:
        info  = sf.info(src)
        total = info.frames
        n_full = total // chunk_len
        for i in range(n_full):
            chunks.append((src,tgt,i*chunk_len,chunk_len))
        rem = total - n_full*chunk_len
        if rem >= seg_len:
            chunks.append((src,tgt,n_full*chunk_len,rem))
    return chunks

def validate_ir_delay(dynamic_student_model):
    fir_down = dynamic_student_model.get_layer("stu_ir_downsample")
    fir_post = dynamic_student_model.get_layer("stu_ir_post_lp")
    align   = dynamic_student_model.get_layer("stu_ir_align")
    expected = int(align.offset)
    factor = STUDENT_IR_DOWNSAMPLE_FACTOR
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
        print("⚠️  MISMATCH!  adjust StuIrAlign.offset to", measured)
    else:
        print("✅ FIR-delay lines up.")

class DemoCallback(Callback):
    def __init__(self, sample_rate, demo_checkpoints_out_dir, logs_out_dir, build_model_fn, build_model_fn_args={}):
        super().__init__()
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
        self.build_model_fn_args = build_model_fn_args
        self.sample_rate = sample_rate
        self.every = 1
        self.demos_out_dir = demo_checkpoints_out_dir
        os.makedirs(self.demos_out_dir, exist_ok=True)
        self.writer = tf.summary.create_file_writer(os.path.join(logs_out_dir,"demo"))
        self.demo_frames  = DEMO_SEC * sample_rate
        self.offset_frames = DEMO_OFFSET_SEC * sample_rate

    def set_model(self, model):
        self.static_model = model
        self.dynamic_model = self.build_model_fn(
                            **self.build_model_fn_args,
                            dynamic_time=True)
        self.dynamic_model.set_weights(self.static_model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every != 0:
            return
        self.dynamic_model.set_weights(self.static_model.get_weights())
        x_long, x_src_sample_rate = sf.read(self.src_path,
                            start=self.offset_frames,
                            frames=self.demo_frames,
                            dtype='float32',
                            always_2d=True)
        y_long, _ = sf.read(self.tgt_path,
                            start=self.offset_frames,
                            frames=self.demo_frames,
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
        # This helps inform whether or not the model accurately predicts the correct style.
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)
        y_inf_style_pred, style_pred_out = self.dynamic_model.predict([x_demo, dummy_style, force_zero], batch_size=1, verbose=0)
        y_inf_style_pred = y_inf_style_pred[0, :, 0]
        style_pred = float(style_pred_out[0, 0])
        style_pred_name = 'pick' if style_pred > STYLE_FLAG_PROB_THRESHOLD else 'finger'

        # Now, if the true style is known (such as is the case for the base models),
        # also infer the audio, but guided by the true style flag
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
        return combined_loss(y_true, y_pred, self.sample_rate, extra_small_cplx=self.extra_small_cplx)
    def get_config(self):
        config = super().get_config()
        config.update({"sample_rate": self.sample_rate, "extra_small_cplx": self.extra_small_cplx})
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

@register_keras_serializable()
def onset_envelope(y, sample_rate, pre_emph_alpha=0.98, smooth_ms=ONSET_SMOOTH_MS):
    pe = y[:,1:,:] - pre_emph_alpha*y[:,:-1,:]
    pe = tf.pad(pe, [[0,0],[1,0],[0,0]])
    env = tf.abs(pe)
    win = tf.cast(tf.maximum(1, tf.cast(smooth_ms * sample_rate / 1000.0, tf.int32)), tf.int32)
    ker = tf.ones((win,1,1), dtype=tf.float32) / tf.cast(win, tf.float32)
    env_sm = tf.nn.conv1d(env, ker, stride=1, padding='SAME')
    m = tf.reduce_max(env_sm, axis=1, keepdims=True) + 1e-6
    env_n = tf.clip_by_value(env_sm / m, 0.0, 1.0)
    return env_n

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
def combined_loss(y_true, y_pred, sample_rate, extra_small_cplx=False):
    onset_env = onset_envelope(y_true, sample_rate=sample_rate, pre_emph_alpha=HP_PRE_EMPHASIS_HF, smooth_ms=ONSET_SMOOTH_MS)
    td   = waveform_l1_loss(y_true, y_pred)
    hf   = hf_loss(y_true, y_pred) * (1.0 + HF_ONSET_GAIN * onset_env)
    spec = mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                          split_hz=STFT_HIGH_SPLIT_HZ,
                                          high_boost=STFT_HIGH_BOOST,
                                          onset_env=onset_env, onset_gain=STFT_HF_ONSET_GAIN)
    mel  = log_mel_loss(y_true, y_pred, sample_rate)
    env  = rms_env_loss(y_true, y_pred)

    # Base complex STFT (existing)
    cplx_base = complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=256, hop=64, win=256, onset_env=onset_env)

    # Extra tiny-window complex STFT to lock the first ms (teacher only)
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

@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    return tf.reduce_mean(hf_loss(y_true, y_pred))

# ────────────────────────────── DATA PIPELINE HELPERS ───────────────────────

def hp_diff_env(window, α=0.8):
    hp = np.empty_like(window)
    hp[0] = window[0]
    hp[1:] = window[1:] - α * window[:-1]
    return np.abs(np.diff(hp))

def segment_generator(seg_len, chunks, is_pick_chunks):
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        mono = audio_cache[src][:,0]
        off = random.randrange(max(cdur - seg_len,1))
        rs = cstart + off
        piezo_seg = mono[rs:rs+seg_len]
        mic_seg   = audio_cache[tgt][rs:rs+seg_len,0]
        if np.max(np.abs(piezo_seg)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue
        if piezo_seg.shape[0] < seg_len:
            p = seg_len - piezo_seg.shape[0]
            piezo_seg = np.pad(piezo_seg,  (0,p))
            mic_seg   = np.pad(mic_seg,    (0,p))
        x = piezo_seg.astype('float32')[:,None]
        y_audio = mic_seg.astype('float32')[:,None]
        style_flag = 1.0 if is_pick_chunks else 0.0
        yield x, (y_audio, style_flag)

def analyze_loss_scales(model, dataset, sample_rate):
    # Take one batch from your validation dataset
    batch = next(iter(dataset))
    # Unpack batch: it can be (x, y_dict, sw_dict) OR (x, (y_audio, style), sw)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x = batch[0]
        y = batch[1]
    else:
        x, y = batch
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

def calibrate_loss_weights(model, dataset, sample_rate, n_batches=5):
    sums = {'l1':0., 'stft':0., 'mel':0., 'env':0., 'hf':0.}
    for i, (x,y,sw) in enumerate(dataset.take(n_batches)):
        scales = analyze_loss_scales(model=model, sample_rate=sample_rate, dataset=tf.data.Dataset.from_tensors((x,y,sw)))
        for k,v in scales.items():
            sums[k] += v
    print("Average loss scales over", n_batches, "batches:")
    for k,v in sums.items():
        print(f"Initial {k} loss: {v:.4f}")
    return {k: sums[k]/n_batches for k in sums}

def generate_demo (model, out_dir, sample_rate, prediction_name="demo_pred", only_prediction=False):
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
    FRAMES      = sec * sample_rate
    START_FRAME = offset_sec * sample_rate
    
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
    # This helps inform whether or not the model accurately predicts the correct style.
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
    t = np.arange(STUDENT_IR_FIR_SIZE) - (STUDENT_IR_FIR_SIZE - 1)/2.0
    h = np.sinc(2 * STUDENT_IR_CUTOFF * t)
    w = np.hamming(STUDENT_IR_FIR_SIZE)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float32)

_ir_fir     = make_sinc_filter()
h_lin = _ir_fir
h_min = minimum_phase(h_lin, method='homomorphic')
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)
_ir_fir_rev = h_min[::-1].reshape(-1,1,1)
_ir_fir_fwd = h_min.    reshape(-1,1,1)
_ir_fir_tf  = tf.constant(_ir_fir_rev, dtype=tf.float32)
_ir_fir_up  = tf.constant(_ir_fir_fwd, dtype=tf.float32)

# ────────────────────────────── CUSTOM LAYERS ────────────────────────────────

@register_keras_serializable()
class StuIrAlign(Layer):
    def __init__(self, offset, **kwargs):
        super().__init__(**kwargs)
        self.offset = int(offset)
    def call(self, inputs):
        ir, main = inputs
        T = tf.shape(main)[1]
        T_ir = tf.shape(ir)[1]
        needed = self.offset + T
        pad_end = needed - T_ir
        ir = tf.cond(pad_end>0, lambda: tf.pad(ir, [[0,0],[0,pad_end],[0,0]]), lambda: ir)
        return ir[:, self.offset : self.offset + T, :]
    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape[1]

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        out_T = T // self.factor
        return (b, out_T, 1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"offset": self.offset})
        return cfg

@register_keras_serializable()
class StuIrDownsampleFIR(Layer):
    def __init__(self, factor=STUDENT_IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.kernel_size = STUDENT_IR_FIR_SIZE
    def call(self, x):
        p = x[...,0:1]
        pad = self.kernel_size - 1
        p_pad = tf.pad(p, [[0,0], [pad,0], [0,0]])
        return tf.nn.conv1d(p_pad, filters=_ir_fir_tf, stride=self.factor, padding='VALID')
    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        out_T = (T + self.factor - 1) // self.factor
        return (b, out_T, 1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'factor': self.factor})
        return cfg

@register_keras_serializable()
class StuIrPostLP(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = STUDENT_IR_FIR_SIZE
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
    def __init__(self, taps=student_lp_taps_tf, N=None, **kwargs):
        kwargs.pop('N', None)
        super().__init__(**kwargs)
        self.taps = tf.Variable(taps, trainable=False, dtype=tf.float32)
        self.N = int(self.taps.shape[0])
    def call(self, x):
        pad_amt = self.N - 1
        x_pad = tf.pad(x, [[0,0],[pad_amt,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"N": self.N})
        return cfg

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
        b = tf.shape(style_scalar)[0]
        T = tf.shape(ref)[1]
        s = tf.reshape(style_scalar, (b,1,1))
        return tf.tile(s, (1, T, 1))
    def compute_output_shape(self, input_shapes):
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

@register_keras_serializable()
class OnsetGate(Layer):
    def __init__(self, sr, alpha=0.98, smooth_ms=ONSET_SMOOTH_MS, **kwargs):
        super().__init__(**kwargs)
        self.sr = float(sr)
        self.alpha = float(alpha)
        self.smooth_ms = float(smooth_ms)
    def call(self, x):
        pe = x[:,1:,:] - self.alpha * x[:,:-1,:]
        pe = tf.pad(pe, [[0,0],[1,0],[0,0]])
        env = tf.abs(pe)
        win = tf.cast(max(1, int(self.smooth_ms * self.sr / 1000.0)), tf.int32)
        ker = tf.ones((win,1,1), dtype=tf.float32) / tf.cast(win, tf.float32)
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

def build_teacher_model(dynamic_time=False):
    T   = None if dynamic_time else TEACHER_SEG_LEN
    inp = Input(shape=(T,1), name="tch_piezo_input")
        
    # The known, true style of the input (may or may not be known)
    style_true_in  = Input(shape=(1,), name="tch_style_true_input")
    # Whether to use the style prediction or the true style for training
    style_force_in = Input(shape=(1,), name="tch_style_force_input")  # 0=predict, 1=ground-truth
    
    # Style head
    cls         = Conv1D(16,3, padding="same", activation="relu", name="tch_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="tch_style_gap")(cls)
    style_logit = Dense(1, name="tch_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="tch_style_prob")(style_logit)

    x = Conv1D(TEACHER_FILTERS, 1, padding="same", name="tch_init_pw")(inp)

    # Pre-blocks with look-ahead
    for i in range(TEACHER_PRE_BLOCKS):
        x = gated_sep_block(x, TEACHER_FILTERS, TEACHER_PRE_KERNEL, 1, prefix=f"tch_pre_{i}", lookahead=TEACHER_LOOKAHEAD_SAMPLES)
        x = film_apply(x, style_embed, TEACHER_FILTERS, prefix=f"tch_pre_{i}")

    # Dilated stack with bounded look-ahead
    for idx, d in enumerate(TEACHER_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(TEACHER_KERNEL_SIZE, d, TEACHER_LOOKAHEAD_SAMPLES)
        x = ZeroPadding1D((pad_left, pad_right), name=f"pad_{idx}")(x)
        g = SeparableConv1D(TEACHER_FILTERS*2, TEACHER_KERNEL_SIZE, dilation_rate=d, padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':TEACHER_FILTERS}, name=f"slice_content_{idx}")(g)
        g2 = Lambda(slice_gate, arguments={'filters':TEACHER_FILTERS}, name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([Activation('tanh', name=f"tanh_{idx}")(c),
                                               Activation('sigmoid', name=f"sigmoid_{idx}")(g2)])
        x = Conv1D(TEACHER_FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        
        film_params = Dense(TEACHER_FILTERS*2, name=f"film_dense_{idx}")(style_embed)
        gamma_raw = Lambda(slice_film_gamma, arguments={'filters': TEACHER_FILTERS}, name=f"film_gamma_raw_{idx}")(film_params)
        beta_raw  = Lambda(slice_film_beta,  arguments={'filters': TEACHER_FILTERS}, name=f"film_beta_raw_{idx}")(film_params)
        gamma = Lambda(film_gamma_stable, name=f"film_gamma_{idx}")(gamma_raw)
        beta  = Lambda(film_beta_stable,  name=f"film_beta_{idx}")(beta_raw)
        # Broadcast FiLM parameters across time: [B, F] -> [B, 1, F]
        gamma3 = Reshape((1, TEACHER_FILTERS), name=f"film_gamma_expand_{idx}")(gamma)
        beta3  = Reshape((1, TEACHER_FILTERS), name=f"film_beta_expand_{idx}")(beta)
        x = Add(name=f"film_apply_{idx}")([
            Multiply(name=f"film_scale_{idx}")([x, gamma3]),
            beta3
        ])
        
    # Final pick/finger outputs
    main_pick   = Conv1D(1,1,padding="same",name="tch_final_pick")(x)
    main_finger = Conv1D(1,1,padding="same",name="tch_final_finger")(x)
    
    one_minus_force = Lambda(invert, name="tch_one_minus_force")(style_force_in)
    # style_used_scalar = (1 - force)pred + forcetrue  → shape (B,1)
    style_used_scalar = Add(name="tch_style_used_scalar")([
        Multiply(name="tch_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="tch_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time_used = TileStyle(name="tch_style_time_used")([style_used_scalar, inp])
    style_embed = Dense(TEACHER_EMBED_DIM, activation="linear", name="tch_style_embed")(style_used_scalar)
 
    inv_style  = InvertStyle(name="tch_invert_style")(style_time_used)
    pick_c     = Multiply(name="tch_pick_contrib")([style_time_used, main_pick])
    finger_c   = Multiply(name="tch_finger_contrib")([inv_style,     main_finger])

    # Onset-gated HF branch with look-ahead in its convs    
    gate = OnsetGate(sr=TEACHER_SAMPLE_RATE, alpha=HP_PRE_EMPHASIS_HF, smooth_ms=TEACHER_HF_ONSET_SMOOTH_MS, name="tch_onset_gate")(inp)
    gate = Lambda(onset_gate_shape, name="tch_onset_gate_shaped")(gate)
    
    g_feat = Conv1D(32, 5, padding="same", activation="relu", name="tch_gate_c1")(x)
    g_feat = Conv1D(1,  3, padding="same", activation="sigmoid", name="tch_gate_c2")(g_feat)
    
    gate = Multiply(name="tch_onset_gate_total")([gate, g_feat])
    
    hf = Conv1D(TEACHER_FILTERS, 1, padding="same", name="tch_hf_init")(inp)
    hf = gated_sep_block(hf, TEACHER_FILTERS, kernel=5, dilation=1, prefix="tch_hf_b0", lookahead=TEACHER_LOOKAHEAD_SAMPLES)
    hf = gated_sep_block(hf, TEACHER_FILTERS, kernel=5, dilation=1, prefix="tch_hf_b1", lookahead=TEACHER_LOOKAHEAD_SAMPLES)
    hf = gated_sep_block(hf, TEACHER_FILTERS, kernel=3, dilation=1, prefix="tch_hf_b2", lookahead=TEACHER_LOOKAHEAD_SAMPLES)
    hf = Conv1D(1,1,padding="same", name="tch_hf_out")(hf)
    hf = Multiply(name="tch_hf_gate")([hf, gate])
    hf = IrScale(init=0.25, name="tch_hf_scale")(hf)

    out = Add(name="tch_mix_main")([pick_c, finger_c, hf])

    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in], outputs=[out, style_prob], name="piezo2mic_teacher")

@register_keras_serializable()
def build_student_model(teacher=None, dynamic_time=False):
    T   = None if dynamic_time else STUDENT_SEG_LEN
    inp = Input(shape=(T,1), name="stu_piezo_input")
    
    
    # The known, true style of the input (may or may not be known)
    style_true_in  = Input(shape=(1,), name="stu_style_true_input")
    # Whether to use the style prediction or the true style for training
    style_force_in = Input(shape=(1,), name="stu_style_force_input")  # 0=predict, 1=ground-truth

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="stu_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="stu_style_gap")(cls)
    style_logit = Dense(1, name="stu_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="stu_style_prob")(style_logit)
    
    x = Conv1D(STUDENT_FILTERS, kernel_size=1, padding="same", name="stu_init_pw")(inp)

    # Pre-blocks (causal)
    for i in range(STUDENT_PRE_BLOCKS):
        x = gated_sep_block(x, STUDENT_FILTERS, STUDENT_PRE_KERNEL, 1, prefix=f"stu_pre_{i}", lookahead=0)
        x = film_apply(x, style_embed, STUDENT_FILTERS, prefix=f"stu_pre_{i}")

    # Dilated TCN Blocks (causal)
    student_feats = []
    for i, d in enumerate(STUDENT_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(STUDENT_KERNEL_SIZE, d, lookahead=0)
        x   = ZeroPadding1D((pad_left, pad_right), name=f"stu_pad_{i}")(x)
        g   = SeparableConv1D(2*STUDENT_FILTERS, STUDENT_KERNEL_SIZE, dilation_rate=d, padding="valid", name=f"stu_sep_{i}")(x)
        c   = Lambda(slice_content, arguments={"filters": STUDENT_FILTERS}, name=f"stu_slice_c_{i}")(g)
        g2  = Lambda(slice_gate,    arguments={"filters": STUDENT_FILTERS}, name=f"stu_slice_g_{i}")(g)
        x   = Multiply(name=f"stu_gate_mul_{i}")([Activation("tanh",    name=f"stu_tanh_{i}")(c),
                                                  Activation("sigmoid", name=f"stu_sigmoid_{i}")(g2)])
        x   = Conv1D(STUDENT_FILTERS, kernel_size=1, padding="same", name=f"stu_pw_{i}")(x)
        x   = Add(name=f"stu_res_add_{i}")([x, res])
    
        film  = Dense(2*STUDENT_FILTERS, name=f"stu_film_dense_{i}")(style_embed)
        gamma_raw = Lambda(slice_film_gamma, arguments={"filters": STUDENT_FILTERS}, name=f"stu_film_gamma_raw_{i}")(film)
        beta_raw  = Lambda(slice_film_beta,  arguments={"filters": STUDENT_FILTERS}, name=f"stu_film_beta_raw_{i}")(film)
        gamma = Lambda(film_gamma_stable, name=f"stu_film_gamma_{i}")(gamma_raw)
        beta  = Lambda(film_beta_stable,  name=f"stu_film_beta_{i}")(beta_raw)
        # Broadcast FiLM parameters across time: [B, F] -> [B, 1, F]
        gamma3 = Reshape((1, STUDENT_FILTERS), name=f"stu_film_gamma_expand_{i}")(gamma)
        beta3  = Reshape((1, STUDENT_FILTERS), name=f"stu_film_beta_expand_{i}")(beta)
        x     = Add(name=f"stu_film_apply_{i}")([
            Multiply(name=f"stu_film_scale_{i}")([x, gamma3]),
            beta3
        ])

        layer_name = f"res_add_{i}"
        if STUDENT_TEACHER_FEAT_LAYERS and layer_name in STUDENT_TEACHER_FEAT_LAYERS:
            student_feats.append(x)
    
    # Style-gated pick/finger blend
    out_pick   = Conv1D(1,1,padding="same", name="stu_final_pick")(x)
    out_finger = Conv1D(1,1,padding="same", name="stu_final_finger")(x)
    
    one_minus_force = Lambda(invert, name="stu_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="stu_style_used_scalar")([
        Multiply(name="stu_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="stu_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time_used = TileStyle(name="stu_style_time_used")([style_used_scalar, inp])
    style_embed = Dense(TEACHER_EMBED_DIM, activation="linear", name="stu_style_embed")(style_used_scalar)

    inv_style  = InvertStyle(name="stu_invert_style")(style_time_used)
    pick_c     = Multiply(name="stu_pick_contrib")([style_time_used, out_pick])
    finger_c   = Multiply(name="stu_finger_contrib")([inv_style,     out_finger])
    y_student  = Add(name="stu_main")([pick_c, finger_c])

    # Low-rate IR Branch (causal)
    ir = StuIrDownsampleFIR(name="stu_ir_downsample")(inp)
    for i, d in enumerate(STUDENT_IR_DILATIONS):
        pad = (STUDENT_IR_KERNEL - 1) * d
        ir = ZeroPadding1D((pad, 0), name=f"stu_ir_pad_{i}")(ir)
        ir = SeparableConv1D(
            filters=1,
            kernel_size=STUDENT_IR_KERNEL,
            dilation_rate=d,
            padding="valid",
            depthwise_regularizer=regularizers.l2(1e-4),
            pointwise_regularizer=regularizers.l2(1e-4),
            depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
            pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
            name=f"stu_ir_sep_{i}"
        )(ir)
    ir = IrScale(init=0.6, name="stu_ir_scale")(ir)
    ir = UpSampling1D(size=STUDENT_IR_DOWNSAMPLE_FACTOR, name="stu_ir_upsample")(ir)
    ir = StuIrPostLP(name="stu_ir_post_lp")(ir)
    total_delay = STUDENT_TOTAL_EXPECTED_IR_DELAY
    ir = StuIrAlign(offset=total_delay + STUDENT_IR_ALIGN_OFFSET_CORRECTION, name='stu_ir_align')([ir, inp])
    ir = IRLowpass(name="stu_ir_lowpass")(ir)

    # Onset-gated HF branch (causal)
    gate = OnsetGate(sr=STUDENT_SAMPLE_RATE, alpha=HP_PRE_EMPHASIS_HF, smooth_ms=ONSET_SMOOTH_MS, name="stu_onset_gate")(inp)
    g_feat = Conv1D(32, 5, padding="same", activation="relu", name="stu_gate_c1")(x)
    g_feat = Conv1D(1,  3, padding="same", activation="sigmoid", name="stu_gate_c2")(g_feat)
    
    
    
    gate = Multiply(name="stu_onset_gate_total")([gate, g_feat])

    hf = Conv1D(STUDENT_FILTERS, 1, padding="same", name="stu_hf_init")(inp)
    hf = gated_sep_block(hf, STUDENT_FILTERS, kernel=5, dilation=1, prefix="stu_hf_b0", lookahead=0)
    hf = gated_sep_block(hf, STUDENT_FILTERS, kernel=5, dilation=1, prefix="stu_hf_b1", lookahead=0)
    hf = gated_sep_block(hf, STUDENT_FILTERS, kernel=3, dilation=1, prefix="stu_hf_b2", lookahead=0)
    hf = Conv1D(1,1,padding="same", name="stu_hf_out")(hf)
    hf = Multiply(name="stu_hf_gate")([hf, gate])
    hf = IrScale(init=0.25, name="stu_hf_scale")(hf)

    y_student = Add(name="stu_with_ir")([y_student, ir, hf])

    # Distillation
    if teacher is not None:
        teacher.trainable = False
        if STUDENT_TEACHER_FEAT_LAYERS:
            teacher_feat_model = tf.keras.models.Model(
                inputs = teacher.input,
                outputs= [teacher.get_layer(n).output for n in STUDENT_TEACHER_FEAT_LAYERS]
            )
            T_feats = teacher_feat_model(([inp, style_true_in, style_force_in]))
            if not isinstance(T_feats, list):
                T_feats = [T_feats]
            for idx, (T_f, S_f) in enumerate(zip(T_feats, student_feats)):
                chan = int(T_f.shape[-1])
                proj = Conv1D(chan, 1, padding="same", name=f"stu_proj_feat_{idx}")(S_f)
                _ = DistillLoss(alpha=STUDENT_FEAT_DISTILL_ALPHA, name=f"feat_distill_{idx}")([T_f, proj])
        
        teacher_audio, teacher_style = teacher([inp, style_true_in, style_force_in])
        
        y_student = DistillLoss(alpha=STUDENT_DISTILL_ALPHA, name="distillation")([teacher_audio, y_student])

    return tf.keras.models.Model(inputs=[inp, style_true_in, style_force_in], outputs=[y_student, style_prob], name="piezo2mic_student")

def approx_steps(durs, seg_len):
    return 10 if DEBUG else sum(max((d-seg_len)//(seg_len//2)+1,0) for d in durs)//BATCH_SIZE

# ────────────────────────────── TRAINING ─────────────────────────────────────

@register_keras_serializable()
def add_sw_teacher(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]
    style_true = tf.expand_dims(style_flag, axis=-1)  # (B,1)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)   # (B,T,1)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)  # (B,1)
    # Return tuples, not lists
    return x, (y_audio, style_true), (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_student(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]
    style_true = tf.expand_dims(style_flag, axis=-1)  # (B,1)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)   # (B,T,1)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)  # (B,1)
    # Return tuples, not lists
    return x, (y_audio, style_true), (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_teacher_tf(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    style_force = tf.ones((b, 1), dtype=tf.float32)            # 1.0 = use ground-truth
    inputs = {
    'tch_piezo_input': x,
    'tch_style_true_input': style_true,
    'tch_style_force_input': style_force,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_student_tf(x, y):
    y_audio, style_flag = y
    b = tf.shape(x)[0]; T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)          # (B,1)
    style_force = tf.ones((b, 1), dtype=tf.float32)            # 1.0 = use ground-truth
    inputs = {
    'stu_piezo_input': x,
    'stu_style_true_input': style_true,
    'stu_style_force_input': style_force,
    }
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_student_finetune_nf(x, y):
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
        'stu_piezo_input':        x,
        'stu_style_true_input':   style_true,
        'stu_style_force_input':  style_force,
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
    build_model_fn_args={}
):
    global L1_WEIGHT, HF_WEIGHT, STFT_WEIGHT, MEL_WEIGHT, RMS_ENV_WEIGHT
    
    all_chunks = build_chunks(pairs_list=all_pairs, seg_len=seg_len, sample_rate=sample_rate)
    pick_chunks = build_chunks(pairs_list=picked_pairs, seg_len=seg_len, sample_rate=sample_rate)
    finger_chunks = build_chunks(pairs_list=finger_pairs, seg_len=seg_len, sample_rate=sample_rate)

    idx = list(range(len(all_chunks)))
    if len(idx)>1:
        tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=SEED)
    else:
        raise RuntimeError("Not enough data to split into train/val sets.")

    train_chunks  = [all_chunks[i] for i in tr_i]
    val_chunks    = [all_chunks[i] for i in va_i]
    train_pick_chunks   = [c for c in train_chunks if c in pick_chunks]
    train_finger_chunks = [c for c in train_chunks if c in finger_chunks]
    val_pick_chunks     = [c for c in val_chunks   if c in pick_chunks]
    val_finger_chunks   = [c for c in val_chunks   if c in finger_chunks]
    
    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur for (_,_,_,dur) in val_chunks]
    steps_per_epoch  = max(approx_steps(durs=train_durs, seg_len=seg_len),1)
    validation_steps = max(approx_steps(durs=val_durs, seg_len=seg_len),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    sig_x = tf.TensorSpec((seg_len,1), tf.float32)
    sig_y = (
        tf.TensorSpec((seg_len,1), tf.float32),
        tf.TensorSpec((),       tf.float32)
    )

    pick_ds    = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_pick_chunks, is_pick_chunks=True), output_signature=(sig_x,sig_y)
    )
    finger_ds  = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_finger_chunks, is_pick_chunks=False), output_signature=(sig_x,sig_y)
    )
    train_ds = tf.data.Dataset.sample_from_datasets(
        [pick_ds, finger_ds], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO]
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    # Choose the correct mapper based on which model is being built
    
    if kind == 'teacher':
        sw_map_fn = add_sw_teacher_tf
    else:
        sw_map_fn = add_sw_student_tf
    
    train_ds = train_ds.map(sw_map_fn).repeat().prefetch(tf.data.AUTOTUNE)

    val_pick_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_pick_chunks, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    val_finger_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_finger_chunks, is_pick_chunks=False),
        output_signature=(sig_x, sig_y)
    )
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
        static_model = build_model_fn(**build_model_fn_args, dynamic_time=False)
        initial_epoch = 0
    
    baseline = calibrate_loss_weights(model=static_model, dataset=val_ds, sample_rate=sample_rate, n_batches=5)
    ref_scale = float(baseline['l1'])
    user_w    = {'l1':L1_WEIGHT, 'stft':STFT_WEIGHT, 'mel':MEL_WEIGHT, 'env':RMS_ENV_WEIGHT, 'hf':HF_WEIGHT}
    new_w = {}
    for key, uw in user_w.items():
        bs = float(baseline[key])
        new_w[key] = uw / bs * ref_scale if bs > 0.0 else 0.0
    L1_WEIGHT       = new_w['l1']
    STFT_WEIGHT     = new_w['stft']
    MEL_WEIGHT      = new_w['mel']
    RMS_ENV_WEIGHT  = new_w['env']
    HF_WEIGHT       = new_w['hf']
    print(f"→ normalized loss weights:"
          f"  L1={L1_WEIGHT:.6f}"
          f"  STFT={STFT_WEIGHT:.6f}"
          f"  MEL={MEL_WEIGHT:.6f}"
          f"  ENV={RMS_ENV_WEIGHT:.6f}"
          f"  HF={HF_WEIGHT:.6f}")

    _combined_loss = CombinedLoss(sample_rate, extra_small_cplx=(kind == 'teacher'))
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
                           build_model_fn_args=build_model_fn_args,
                           logs_out_dir=logs_out_dir,
                           demo_checkpoints_out_dir=demo_checkpoints_out_dir)
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

    dynamic_model = build_model_fn(**build_model_fn_args, dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    dynamic_model.save(dynamic_model_filepath)
    print("✅ Saved FP32 dynamic model:", dynamic_model_filepath)

    return static_model, dynamic_model

def train_teacher(resume=False, resume_ckpt_path=None):
    train_shared(
        kind='teacher',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=TEACHER_SEG_LEN,
        sample_rate=TEACHER_SAMPLE_RATE,
        static_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_teacher_model,
        checkpoint_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
        logs_out_dir=LOGS_TEACHER_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_TEACHER_OUT_DIR
    )

def train_student(resume=False, resume_ckpt_path=None):
    teacher_dynamic_model_filepath = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(teacher_dynamic_model_filepath):
        print("Teacher model not found, run --train-teacher first")
        sys.exit(1)
    teacher = tf.keras.models.load_model(teacher_dynamic_model_filepath, compile=False)
    print("Loaded teacher from", teacher_dynamic_model_filepath)

    tmp_dynamic_model = build_student_model(teacher=teacher, dynamic_time=True)
    
    validate_ir_delay(tmp_dynamic_model)
    
    train_shared(
        kind='student',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=STUDENT_SEG_LEN,
        sample_rate=STUDENT_SAMPLE_RATE,
        static_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_student_model,
        build_model_fn_args={ 'teacher': teacher },
        checkpoint_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
        logs_out_dir=LOGS_STUDENT_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_STUDENT_OUT_DIR
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
            for x,(y_audio, style_flag) in segment_generator(seg_len=seg_len, chunks=chunks, is_pick_chunks=True):
                yield x, (y_audio, style_flag)
        def fu_gen():
            for x,(y_audio,style_flag) in segment_generator(seg_len=seg_len, chunks=chunks, is_pick_chunks=False):
                yield x, (y_audio, style_flag)
        ds_pu = tf.data.Dataset.from_generator(pu_gen, output_signature=(spec_x, spec_y))
        ds_fu = tf.data.Dataset.from_generator(fu_gen, output_signature=(spec_x, spec_y))
        mixed = tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO,FINGER_CHUNK_RATIO], seed=SEED)
        return mixed.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print("\n🔧 Fine-tuning STUDENT on new pair…")
    base_student_full = tf.keras.models.load_model(os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_STATIC_MODEL_FILENAME), compile=False)
    finetuned_student_small = build_student_model(teacher=None, dynamic_time=False)

    for layer in finetuned_student_small.layers:
        if layer.name.startswith("stu_"):
            try:
                layer.set_weights(base_student_full.get_layer(layer.name).get_weights())
            except ValueError:
                pass

    LAST_STUDENT_DILATION = len(STUDENT_DILATIONS) - 1
    for layer in finetuned_student_small.layers:
        name = layer.name
        layer.trainable = \
            name.startswith("stu_film_dense_") \
            or name.startswith("stu_final_") \
            or name == "stu_ir_scale" \
            or name == f"stu_sep_{LAST_STUDENT_DILATION}" \
            or name == f"stu_ir_sep_{LAST_STUDENT_DILATION}" \
            or name == f"stu_pw_{LAST_STUDENT_DILATION}" \
            or name.startswith("stu_hf_")

    dsS_val = make_mixed_ds(STUDENT_SEG_LEN)
    dsS_val = dsS_val.map(add_sw_student_finetune_nf)
    dsS_val = dsS_val.take(5)

    baseline_S = calibrate_loss_weights(model=finetuned_student_small, dataset=dsS_val, sample_rate=FINETUNE_SAMPLE_RATE, n_batches=5)
    ref = float(baseline_S['l1'])
    L1_WEIGHT      = corrected_weights['l1']   / baseline_S['l1']   * ref
    HF_WEIGHT      = corrected_weights['hf']   / baseline_S['hf']   * ref
    STFT_WEIGHT    = corrected_weights['stft'] / baseline_S['stft'] * ref
    MEL_WEIGHT     = corrected_weights['mel']  / baseline_S['mel']  * ref
    RMS_ENV_WEIGHT = corrected_weights['env']  / baseline_S['env']  * ref
    print("→ finetune-student loss weights recalibrated:"
        f"  L1={L1_WEIGHT:.4f}"
        f"  HF={HF_WEIGHT:.4f}"
        f"  STFT={STFT_WEIGHT:.4f}"
        f"  MEL={MEL_WEIGHT:.4f}"
        f"  ENV={RMS_ENV_WEIGHT:.4f}")
    
    _combined_loss = CombinedLoss(FINETUNE_SAMPLE_RATE)
    
    finetuned_student_small.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_STUDENT_LR),
        loss=[_combined_loss, "binary_crossentropy"],
        loss_weights=[1.0, 0.0],   # disable BCE during finetune - we fully inherit style classification weights from the base models
        metrics=[["mae"], []]
    )
    
    dsS = make_mixed_ds(STUDENT_SEG_LEN)
    dsS = dsS.map(add_sw_student_finetune_nf)
    ft_chunks_T = build_chunks(pairs_list=[(src,tgt)], seg_len=STUDENT_SEG_LEN, sample_rate=FINETUNE_SAMPLE_RATE)
    stepsS = max(approx_steps( durs=[dur for (_,_,_,dur) in ft_chunks_T], seg_len=STUDENT_SEG_LEN), 1)
    print("Finetune student model summary:")
    finetuned_student_small.summary()
    finetuned_student_small.fit(dsS, epochs=FINETUNE_STUDENT_EPOCHS, steps_per_epoch=stepsS, verbose=1)

    ft_s_static = os.path.join(MODELS_STUDENT_OUT_DIR, FINETUNE_STUDENT_STATIC_MODEL_FILENAME)
    finetuned_student_small.save(ft_s_static)
    finetuned_student_dyn = build_student_model(teacher=None, dynamic_time=True)
    finetuned_student_dyn.set_weights(finetuned_student_small.get_weights())
    ft_s_dyn = os.path.join(MODELS_STUDENT_OUT_DIR, FINETUNE_STUDENT_DYNAMIC_MODEL_FILENAME)
    finetuned_student_dyn.save(ft_s_dyn)
    return finetuned_student_dyn

def compute_rms(sig):
    return np.sqrt(np.mean(sig**2))

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

def prepare_and_generate_demo (sample_rate, dynamic_model_filepath, checkpoint_out_dir, demo_out_dir, build_model_fn, build_model_fn_args={}):
    dpath = dynamic_model_filepath
    if not os.path.exists(dpath):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_out_dir, "*.keras")))
        if checkpoints:
            dpath = checkpoints[-1]
            print("Loading model from checkpoint:", dpath)
            static_model = tf.keras.models.load_model(dpath, compile=False)
            dynamic_model = build_model_fn(**build_model_fn_args, dynamic_time=True)
            dynamic_model.set_weights(static_model.get_weights())
        else:
            dpath = None
        if not dpath:
            print("Dynamic model not found. run: python piezo2mic.py --train-teacher and/or python piezo2mic.py --train-student")
            sys.exit(1)
    else:
        print("Loading dynamic model from:", dpath)
        dynamic_model = tf.keras.models.load_model(dpath, compile=False)
    generate_demo(model=dynamic_model, out_dir=demo_out_dir, sample_rate=sample_rate)

def main ():
    if len(sys.argv)>1 and sys.argv[1]=="--train-teacher":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_TEACHER_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_TEACHER_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )
        if resume:
            train_teacher(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_teacher()
    elif len(sys.argv)>1 and sys.argv[1]=="--train-student":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_STUDENT_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_STUDENT_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )
        if resume:
            train_student(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_student()
    elif (len(sys.argv)>1 and sys.argv[1]=="--demo"):
        teacher_dynamic_model_filepath = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(teacher_dynamic_model_filepath):
            print("Teacher model not found, run --train-teacher first")
            sys.exit(1)
        teacher = tf.keras.models.load_model(teacher_dynamic_model_filepath, compile=False)
        print("Generating demo audio files...")
        if (len(sys.argv) <= 2 or sys.argv[2] !="--student-only"):
            prepare_and_generate_demo(
                sample_rate=TEACHER_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_TEACHER_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
                build_model_fn=build_teacher_model
            )
        if (len(sys.argv) <= 2 or sys.argv[2] !="--teacher-only"):
            prepare_and_generate_demo(
                sample_rate=STUDENT_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_STUDENT_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
                build_model_fn=build_student_model,
                build_model_fn_args={ 'teacher': teacher }
            )
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        student_dyn = fine_tune()
        generate_fine_tune_demo(model=student_dyn)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune-demo":
        student_dyn_path = os.path.join(MODELS_STUDENT_OUT_DIR, FINETUNE_STUDENT_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(student_dyn_path):
            print("Fine-tuned dynamic model not found, run --fine-tune first")
            sys.exit(1)
        student_dyn = None if not os.path.exists(student_dyn_path) else tf.keras.models.load_model(student_dyn_path, compile=False)
        generate_fine_tune_demo(model=student_dyn)
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()