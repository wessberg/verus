#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal.

The model has a tone-head and a pick-head, so they can have separate losses,
and can be trained separately.

Everything is fully Keras serializable.

It is designed to be ultra-low latency for real-time inference for live-use.
This is why dilations and kernel_size are small, and the model is causal.
"""
import os, glob, random, shutil, sys, re, math
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import butter, sosfilt, find_peaks, hilbert, minimum_phase, firwin
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau
from tensorflow.keras import regularizers, constraints

from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, ZeroPadding1D, GlobalAveragePooling1D,
    Activation, Multiply, Add, Lambda, Dense, Layer, UpSampling1D
)

# Turn off GPU acceleration for inference, as it is more reliable.
# We need to do this before doing any TensorFlow stuff at root level
if len(sys.argv)>1 and sys.argv[1]=="--demo":
    tf.config.set_visible_devices([], 'GPU')

# Try to enforce locked seeds so we can more reliably change config options and trust that the change in results was an effect of the change.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'      # try to force TF→deterministic kernels
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'     # on some TF versions also needed

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Audio channel configuration
SR                        = 48000
INPUT_AVG_RMS_DBFS_TARGET = -23.0

# Debugging params
DEBUG               = False

# Batch params
BATCH_SIZE          = 32

# Teacher params
TEACHER_FILTERS                = 128
TEACHER_KERNEL_SIZE            = 51
TEACHER_DILATIONS              = (1,2,4,8,16,32,64)
TEACHER_RECEPTIVE_FIELD        = 1 + (TEACHER_KERNEL_SIZE - 1) * sum(TEACHER_DILATIONS)
TEACHER_TRAIN_BLOCK_SIZE       = 4096
TEACHER_SEG_LEN                = TEACHER_RECEPTIVE_FIELD + TEACHER_TRAIN_BLOCK_SIZE
TEACHER_EMBED_DIM              = 32
TEACHER_STATIC_MODEL_FILENAME  = "piezo2mic_baseline_teacher.keras"
TEACHER_DYNAMIC_MODEL_FILENAME = "piezo2mic_baseline_teacher_dynamic.keras"

# Student params
STUDENT_FILTERS          = 48       # number of channels in the student
STUDENT_KERNEL_SIZE      = 25       # (25−1)*(1+2+4+8)=24*15=360 +1 ≃361 samples ≃7.5 ms RF
STUDENT_DILATIONS        = (1,2,4,8)
STUDENT_RECEPTIVE_FIELD  = 1 + (STUDENT_KERNEL_SIZE - 1) * sum(STUDENT_DILATIONS)
STUDENT_TRAIN_BLOCK_SIZE = 4096
STUDENT_SEG_LEN          = STUDENT_RECEPTIVE_FIELD + STUDENT_TRAIN_BLOCK_SIZE
STUDENT_EMBED_DIM        = 16       # style‐embedding dim for the student
STUDENT_DISTILL_ALPHA    = 5.0      # weight on the distillation loss

# which teacher‐layers to use for intermediate‐feature hints:
STUDENT_TEACHER_FEAT_LAYERS   = ["res_add_2", "res_add_3"]
# how strongly to weight each of those feature‐distill losses
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

FINE_TUNE_DEMO_SEC = 60
FINE_TUNE_DEMO_OFFSET_SEC = 0

CLICK_THRESH       = 1e-4     # same threshold you use for attack detection
CLICK_KERNEL_SIZE  = 31       # ~0.65 ms @48 kHz for spectral shaping
CLICK_TAU_SEC      = 0.003       # a 3 ms time constant

# build a causal exponential‐decay kernel of length DECAY_LENGTH
t = tf.range(CLICK_KERNEL_SIZE, dtype=tf.float32) / SR
decay = tf.exp(-t / CLICK_TAU_SEC)
decay = decay / tf.reduce_sum(decay)      # normalize area = 1
decay = tf.reshape(decay, (CLICK_KERNEL_SIZE, 1, 1))  # shape = [L,1,1]

HP_PRE_EMPHASIS           = 0.8 # coefficient for 1-sample 1st-order high-pass

L1_WEIGHT                 = 1.0   # waveform l1 loss
HF_WEIGHT                 = 1.0   # scale of the high-pass loss
STFT_WEIGHT               = 1.0   # weight of the spectral loss
MEL_WEIGHT                = 1.0   # weight of the log‐mel‐spectrogram L1 loss
RMS_ENV_WEIGHT            = 1.0   # weight of the RMS‐envelope L1 loss

PICK_ATTACK_CHUNK_RATIO = 0.4 # How many pick attack segments the model will see
PICK_UNIFORM_CHUNK_RATIO = 0.1 # How many general pick segments the model will see

FINGER_ATTACK_CHUNK_RATIO = 0.15 # How many finger attack segments the model will see
FINGER_UNIFORM_CHUNK_RATIO = 0.35 # How many general finger segments the model will see

# ──────── TUNING KNOBS ─────────────────────────────────────────────────────────
ATTACK_SEGMENT_GENERATOR_MIN_DIFF_THRESH = 1e-6    # absolute floor on |Δ| spikes
ATTACK_SEGMENT_GENERATOR_MIN_ENV_THRESH  = 1e-5    # absolute floor on HF envelope
ATTACK_SEGMENT_GENERATOR_MIN_AMP_THRESH  = 1e-6    # absolute floor on the piezo signal itself, for the attack segment generator
SEGMENT_GENERATOR_MIN_AMP_THRESH         = 1e-5    # absolute floor on the piezo signal itself, for the standard segment generator

# Pick-attack band pass filters
SOS_HF = butter(4, [6000, 20000], fs=SR, btype='bandpass', output='sos')
SOS_LF = butter(4, [300, 2000],  fs=SR, btype='bandpass', output='sos')
#───────────────────────────────────────────────────────────────────────────────

# ─── IR branch FIR FILTER DESIGN ────────────────────────────────────────────
STUDENT_IR_DOWNSAMPLE_FACTOR = 8
STUDENT_IR_FIR_SIZE          = 129    # odd for a symmetric window
STUDENT_IR_CUTOFF            = 1.0 / STUDENT_IR_DOWNSAMPLE_FACTOR  # normalized cutoff (Nyquist=0.5→fc at 0.5/D)
STUDENT_IR_KERNEL    = 33
STUDENT_IR_DILATIONS = (1,2,4,8)
STUDENT_IR_ALIGN_OFFSET_CORRECTION = -64
STUDENT_TOTAL_EXPECTED_IR_DELAY = (STUDENT_IR_FIR_SIZE - 1) + ((STUDENT_IR_FIR_SIZE - 1)//2)

# how many *coarse* steps until the FIR is “fully” onto the input?
coarse_steps = math.ceil((STUDENT_IR_FIR_SIZE - 1) / STUDENT_IR_DOWNSAMPLE_FACTOR)
IR_PREDELAY = coarse_steps * STUDENT_IR_DOWNSAMPLE_FACTOR   # = 8 * 8 = 64

STUDENT_LP_ORDER   = 65
STUDENT_LP_CUTOFF  = 4000    # Hz

student_lp_taps_np = firwin(STUDENT_LP_ORDER,
                     cutoff=STUDENT_LP_CUTOFF/(SR/2),
                     window="hamming",
                     pass_zero=True).astype(np.float32)
student_lp_taps_tf = tf.constant(student_lp_taps_np.reshape(-1,1,1), dtype=tf.float32)

EPOCHS      = 1000
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

PICK_SRC_DIR   = "./input/picked/source"
PICK_TGT_DIR   = "./input/picked/target"
FINGER_SRC_DIR = "./input/fingerpicked/source"
FINGER_TGT_DIR = "./input/fingerpicked/target"

FINETUNE_SRC_DIR   = "./input/finetune"

OUT_DIR        = "./out"

CHECKPOINT_OUT_DIR = os.path.join(OUT_DIR, "checkpoints")
CHECKPOINT_TEACHER_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "teacher")
CHECKPOINT_STUDENT_OUT_DIR = os.path.join(CHECKPOINT_OUT_DIR, "student")

MODELS_OUT_DIR = os.path.join(OUT_DIR, "models")
MODELS_TEACHER_OUT_DIR = os.path.join(MODELS_OUT_DIR, "teacher")
MODELS_STUDENT_OUT_DIR = os.path.join(MODELS_OUT_DIR, "student")

LOGS_OUT_DIR = os.path.join(OUT_DIR, "logs")
LOGS_TEACHER_OUT_DIR = os.path.join(LOGS_OUT_DIR, "teacher")
LOGS_STUDENT_OUT_DIR = os.path.join(LOGS_OUT_DIR, "student")

CSV_LOGS_OUT_DIR = os.path.join(OUT_DIR, "csv-logs")
CSV_LOGS_TEACHER_OUT_DIR = os.path.join(CSV_LOGS_OUT_DIR, "teacher")
CSV_LOGS_STUDENT_OUT_DIR = os.path.join(CSV_LOGS_OUT_DIR, "student")

DEMOS_OUT_DIR = os.path.join(OUT_DIR, "demos")

LONG_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "long")
LONG_DEMOS_TEACHER_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "teacher")
LONG_DEMOS_STUDENT_OUT_DIR = os.path.join(LONG_DEMOS_OUT_DIR, "student")

FINETUNE_DEMOS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "finetune")

DEMO_GENERATOR_OUTPUTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "generator-outputs")

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_TEACHER_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "teacher")
DEMO_CHECKPOINTS_STUDENT_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "student")


SAVE_WEIGHTS           = False  # set to True to dump .npy weights

for dir_path in [DEMO_GENERATOR_OUTPUTS_OUT_DIR, CHECKPOINT_TEACHER_OUT_DIR, CHECKPOINT_STUDENT_OUT_DIR, LOGS_TEACHER_OUT_DIR, LOGS_STUDENT_OUT_DIR, DEMO_CHECKPOINTS_TEACHER_OUT_DIR, DEMO_CHECKPOINTS_STUDENT_OUT_DIR, LONG_DEMOS_TEACHER_OUT_DIR, LONG_DEMOS_STUDENT_OUT_DIR, FINETUNE_DEMOS_OUT_DIR, MODELS_TEACHER_OUT_DIR, MODELS_STUDENT_OUT_DIR, CSV_LOGS_TEACHER_OUT_DIR, CSV_LOGS_STUDENT_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

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

# normalize all files to target RMS once
target_rms = 10 ** (INPUT_AVG_RMS_DBFS_TARGET / 20.0)

# normalize all files to target RMS once
target_rms = 10 ** (INPUT_AVG_RMS_DBFS_TARGET / 20.0)

audio_cache = {}
for src, tgt in all_pairs:
    for path in (src, tgt):
        # 1) load
        data, _ = sf.read(path,
                          dtype='float32',
                          always_2d=True)          # shape = (frames, channels)
        # 2) measure RMS on channel-0
        mono = data[:,0]
        rms = np.sqrt(np.mean(mono**2) + 1e-12)
        # 3) compute and apply gain
        gain = target_rms / rms
        data_norm = data * gain
        # 4) cache normalized audio
        audio_cache[path] = data_norm

def make_click_decay_kernel(tau_secs=0.002, sr=48000, length=None):
    # build a 1–3 ms exponential decay kernel for the click envelope
    if length is None:
        length = int(sr * 0.003)   # ~3 ms
    t = np.arange(length, dtype=np.float32) / sr
    dec = np.exp(-t / tau_secs)
    dec = dec / np.sum(dec)       # normalize
    return dec.astype(np.float32)

# one TensorFlow constant filter (shape = [L,1,1])
_click_decay_np = make_click_decay_kernel()
_click_decay_tf = tf.constant(_click_decay_np.reshape(-1,1,1),
                              dtype=tf.float32)

def build_chunks(pairs_list, seg_len):
    """
    Turn a list of (src,tgt) into
    [(src,tgt,chunk_start,chunk_len), …].
    """
    chunks = []
    chunk_len = CHUNK_SEC * SR
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
    """
    Measure the group-delay of the two FIRs in your IR branch
    (downsampler + post-LP) and check it against the offset
    in StuIrAlign.
    """

    # grab the two FIR layers out of your model
    fir_down = dynamic_student_model.get_layer("stu_ir_downsample")
    fir_post = dynamic_student_model.get_layer("stu_ir_post_lp")
    # grab the align layer so we can read back its offset
    align   = dynamic_student_model.get_layer("stu_ir_align")
    expected = int(align.offset)  # what you told it to crop

    # how many samples to test? make sure > expected and a multiple of factor
    factor = STUDENT_IR_DOWNSAMPLE_FACTOR
    N = expected + factor
    N = int(math.ceil(N / factor) * factor)
    
    # build a single‐impulse test signal on piezo only:
    # shape = (batch=1, time=N, channels=1)
    x = np.zeros((1, N, 1), dtype=np.float32)
    x[0, 0, 0] = 1.0

    # run it through down-sample FIR
    y_down = fir_down(x)          # -> (1, ceil(N/factor), 1)
    # then up-sample back
    y_up   = UpSampling1D(size=factor)(y_down)  # -> (1, ~N, 1)
    # then the post-LP FIR
    y_fir  = fir_post(y_up)       # -> (1, T_fir, 1)
    ir     = y_fir.numpy()[0, :, 0]

    # find first non-zero (above a tiny threshold)
    thresh = 1e-6
    nz     = np.where(np.abs(ir) > thresh)[0]
    measured = int(nz[0]) if len(nz) else None

    print(f"FIR-only → expected delay {expected}, measured {measured}")
    if measured != expected:
        print("⚠️  MISMATCH!  adjust StuIrAlign.offset to", measured)
    else:
        print("✅ FIR-delay lines up.")

class DemoCallback(Callback):
    def __init__(self, demo_checkpoints_out_dir, logs_out_dir, build_model_fn, build_model_fn_args={}):
        super().__init__()
        demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
        self.src_path = demo_src
        self.tgt_path = demo_tgt
        self.demo_sec = DEMO_SEC
        self.offset_sec = DEMO_OFFSET_SEC
        self.build_model_fn = build_model_fn
        self.build_model_fn_args = build_model_fn_args
        self.sr = SR
        self.every = 1
        self.demos_out_dir = demo_checkpoints_out_dir
        os.makedirs(self.demos_out_dir, exist_ok=True)
        # TensorBoard writer for audio
        self.writer = tf.summary.create_file_writer(os.path.join(logs_out_dir,"demo"))

        # how many frames to read
        self.demo_frames  = DEMO_SEC * SR
        self.offset_frames = DEMO_OFFSET_SEC * SR

    def set_model(self, model):
        self.static_model = model
        
        # build a *stateful* dynamic model
        self.dynamic_model = self.build_model_fn(
                            **self.build_model_fn_args,
                            dynamic_time=True)
        
        # copy weights
        self.dynamic_model.set_weights(self.static_model.get_weights())


    def on_epoch_end(self, epoch, logs=None):
        # only every Nth epoch
        if epoch % self.every != 0:
            return
        
        friendly_epoch = epoch + 1

        # re‐sync weights before each inference
        self.dynamic_model.set_weights(self.static_model.get_weights())

        # 1) read the long snippets from disk
        x_long, _ = sf.read(self.src_path,
                            start=self.offset_frames,
                            frames=self.demo_frames,
                            dtype='float32',
                            always_2d=True)  # shape (demo_frames, channels)
        y_long, _ = sf.read(self.tgt_path,
                            start=self.offset_frames,
                            frames=self.demo_frames,
                            dtype='float32',
                            always_2d=True)
    
        # 2) only piezo
        x_piezo = x_long[:,0:1]             # (T,1)
        x_demo  = x_piezo[None,...]         # (1, T,1)
        
        preds = self.dynamic_model.predict(x_demo, batch_size=1)
        if isinstance(preds, (list, tuple)):
            tone_pred, pick_pred = preds
            demo_pred = tone_pred + pick_pred     # sum the two heads
        else:
            demo_pred = preds
        y_pred = demo_pred[0, :, 0]
        

        # 4) write to disk
        epoch_dir = os.path.join(self.demos_out_dir, f"epoch_{friendly_epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        sf.write(os.path.join(epoch_dir, "demo_source.wav"),
                 x_long[:,0], self.sr, subtype='FLOAT')
        sf.write(os.path.join(epoch_dir, "demo_target.wav"),
                 y_long[:,0], self.sr, subtype='FLOAT')
        sf.write(os.path.join(epoch_dir, "demo_pred.wav"),
                 y_pred,         self.sr, subtype='FLOAT')

        # 5) log into TensorBoard
        #    tf.summary.audio expects shape [batch, frames, channels]
        with self.writer.as_default():
            tf.summary.audio("demo/source",
                             x_demo[...,0:1],  # (1, T, 1)
                             sample_rate=self.sr,
                             step=friendly_epoch)
            tf.summary.audio("demo/target",
                             y_long[np.newaxis, :, 0:1],  # (1, T, 1)
                             sample_rate=self.sr,
                             step=friendly_epoch)
            tf.summary.audio("demo/pred",
                             y_pred[np.newaxis, :, np.newaxis],
                             sample_rate=self.sr,
                             step=friendly_epoch)
            self.writer.flush()
        print(f"[DemoCallback] epoch {friendly_epoch}: demo written to {epoch_dir}")

    def get_config(self):
        config = super().get_config()
        config.update({
            "demos_out_dir": self.demos_out_dir,
            "logs_out_dir":  self.logs_out_dir
        })
        return config
    
@register_keras_serializable()
class ClickScale(Layer):
    """
    A tiny trainable scalar to scale the click residual.
    init: starting value (e.g. 3.0 to be 3× louder at graph‐init).
    """
    def __init__(self, init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init = float(init)

    def build(self, input_shape):
        # one scalar weight
        self.scale = self.add_weight(
            name="click_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init": self.init})
        return cfg

@register_keras_serializable()
class IRDecayTriggered(Layer):
    """
    y[t] = ir[t] * env[t]
    where env is the convolution of
       trigger[t] = ( |piezo[t]-piezo[t-1]| > threshold )
    with the exponential kernel exp(-t/tau).
    Tau is trainable.
    """
    def __init__(self, sr=48000, init_tau=0.08, threshold=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.sr        = float(sr)
        self.init_tau  = float(init_tau)
        self.threshold = float(threshold)

    def build(self, input_shape):
        self.tau = self.add_weight(
            name="tau_secs",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_tau),
            constraint=constraints.MinMaxNorm(
                min_value=1e-4, max_value=1.0,
                axis=0      # now valid because tau has dim-0
            ),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs = [ir_branch, piezo_raw]  each shape (B, T, 1)
        ir, piezo = inputs

        # 1) compute abs‐diff envelope
        d = tf.abs(piezo[:,1:,:] - piezo[:,:-1,:])
        d = tf.pad(d, [[0,0],[1,0],[0,0]])  # back to (B,T,1)

        # 2) simple hard threshold → impulses
        triggers = tf.cast(d > self.threshold, tf.float32)  # (B,T,1)

        # 3) build exponential kernel of length T
        T = tf.shape(ir)[1]
        t = tf.cast(tf.range(T), tf.float32) / self.sr        # (T,)
        dec = tf.exp(-t / tf.reshape(self.tau, ()))           # (T,)
        dec = tf.reshape(dec, (T,1,1))                        # (T,1,1)

        # 4) convolve triggers with reversed exp-kernel (causal conv)
        #    reverse the kernel so that dec[0] multiplies triggers[t],
        #    dec[1] multiplies triggers[t-1], etc.
        env = tf.nn.conv1d(triggers,
                           filters=tf.reverse(dec, axis=[0]),
                           stride=1, padding="SAME")
        # 5) clamp to [0,1]
        env = tf.clip_by_value(env, 0.0, 1.0)

        # 6) apply envelope to IR branch
        return ir * env

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sr": self.sr,
            "init_tau": self.init_tau,
            "threshold": self.threshold
        })
        return cfg

@register_keras_serializable()
class IRLowpass(Layer):
    """
    Simple causal FIR‐LPF: pad on the left, VALID conv.
    """
    def __init__(self,
                 taps=student_lp_taps_tf,   # we keep using the global taps
                 N=None,
                 **kwargs):
        # pop off the auto-inserted N so that base Layer doesn't choke
        kwargs.pop('N', None)
        super().__init__(**kwargs)

        # store the FIR taps as a non-trainable variable
        # (always the same global student_lp_taps_tf)
        self.taps = tf.Variable(taps,
                                trainable=False,
                                dtype=tf.float32)
        # keep track of its length
        self.N = int(self.taps.shape[0])

    def call(self, x):
        # x: (batch, T, 1)
        pad_amt = self.N - 1
        x_pad = tf.pad(x, [[0,0],[pad_amt,0],[0,0]])
        return tf.nn.conv1d(x_pad,
                            filters=self.taps,
                            stride=1,
                            padding='VALID')

    def get_config(self):
        cfg = super().get_config()
        # put N back into the config so future saves still carry the right length
        cfg.update({"N": self.N})
        return cfg

@register_keras_serializable()
class IrScale(Layer):
    """A tiny trainable scalar (<1) to keep the IR branch subtle."""
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
        # inputs = [y_teacher, y_student], both shape (batch, T, 1)
        y_t, y_s = inputs
        # squared error → (batch, T, 1)
        sq = tf.math.squared_difference(y_t, y_s)
        # mean over everything → scalar
        loss = tf.reduce_mean(sq)
        # register it (automatically gets added to model.losses)
        self.add_loss(self.alpha * loss)
        # pass the student output on unchanged
        return y_s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': self.alpha})
        return cfg
        

# right after you define CHECKPOINT_OUT_DIR, etc:
def _extract_epoch_from_ckpt(path):
    """
    Given a filename like 'best_model_epoch_05_val_loss_0.1234.keras',
    return 5 as an int.  If no match, returns 0.
    """
    fn = os.path.basename(path)
    m = re.search(r'epoch_(\d+)', fn)
    return int(m.group(1)) if m else 0

def dump_pick_attacks(gap_sec=0.1, n_segments=None):
    """
    Use your attack_segment_generator() to pull out N pick‐attack windows,
    and write both piezo and mic signals (mono or stereo).
    """
    
    # It is fine to always use TEACHER_SEG_LEN here, as we're dumping them based on the teacher model's receptive field
    seg_len=TEACHER_SEG_LEN
    
    pick_chunks   = build_chunks(pairs_list=picked_pairs, seg_len=seg_len)
    
    gap      = int(gap_sec * SR)
    gap_arr  = np.zeros((gap,), dtype='float32')

    gen      = attack_segment_generator(seg_len=seg_len, chunks=pick_chunks)
    collected_piezo = []
    collected_mic   = []

    total = n_segments or len(pick_chunks)

    for i in range(total):
        x, (y, style_flag) = next(gen)
        # x.shape = (SEG_LEN,2),  x[:,0]=piezo,  y.shape=(SEG_LEN,1)
        piezo = x[:,0]       # (SEG_LEN,)
        mic   = y[:,0]       # (SEG_LEN,)

        collected_piezo.append(piezo)
        collected_piezo.append(gap_arr)

        collected_mic.append(mic)
        collected_mic.append(gap_arr)

    if not collected_piezo:
        print("⚠️  No pick‐attack segments were generated.")
        return

    out_p = np.concatenate(collected_piezo)
    out_m = np.concatenate(collected_mic)

    # 1) write two monos
    p_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "pick-attack-piezo.wav")
    m_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "pick-attack-mic.wav")
    sf.write(p_path, out_p, SR, subtype='FLOAT')
    sf.write(m_path, out_m, SR, subtype='FLOAT')
    print(f"✅  Wrote piezo → {p_path}")
    print(f"✅  Wrote mic   → {m_path}")

    # 2) (optional) write a stereo file: L=piezo, R=mic
    stereo = np.stack([out_p, out_m], axis=1)  # shape (frames,2)
    st_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "pick-attack-stereo.wav")
    sf.write(st_path, stereo, SR, subtype='FLOAT')
    print(f"✅  Wrote stereo (L=piezo,R=mic) → {st_path}")

def dump_finger_attacks(gap_sec=0.1, n_segments=None):
    """
    Same as above but for finger-attack windows.
    """
    gap      = int(gap_sec * SR)
    gap_arr  = np.zeros((gap,), dtype='float32')
    
    # It is fine to always use TEACHER_SEG_LEN here, as we're dumping them based on the teacher model's receptive field
    seg_len = TEACHER_SEG_LEN
    
    finger_chunks   = build_chunks(pairs_list=finger_pairs, seg_len=seg_len)
    gen      = attack_segment_generator(seg_len=seg_len, chunks=finger_chunks)
    collected_piezo = []
    collected_mic   = []

    total = n_segments or len(finger_chunks)

    for i in range(total):
        x, y = next(gen)
        piezo = x[:,0]
        mic   = y[:,0]

        collected_piezo.append(piezo)
        collected_piezo.append(gap_arr)

        collected_mic.append(mic)
        collected_mic.append(gap_arr)

    if not collected_piezo:
        print("⚠️  No finger‐attack segments were generated.")
        return

    out_p = np.concatenate(collected_piezo)
    out_m = np.concatenate(collected_mic)

    p_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "finger-attack-piezo.wav")
    m_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "finger-attack-mic.wav")
    sf.write(p_path, out_p, SR, subtype='FLOAT')
    sf.write(m_path, out_m, SR, subtype='FLOAT')
    print(f"✅  Wrote piezo → {p_path}")
    print(f"✅  Wrote mic   → {m_path}")

    stereo = np.stack([out_p, out_m], axis=1)
    st_path = os.path.join(DEMO_GENERATOR_OUTPUTS_OUT_DIR, "finger-attack-stereo.wav")
    sf.write(st_path, stereo, SR, subtype='FLOAT')
    print(f"✅  Wrote stereo (L=piezo,R=mic) → {st_path}")
    
@register_keras_serializable()
def pack_for_rnn(x):
    # no-op but forces TensorFlow to materialize a contiguous buffer
    return tf.identity(x)


def find_attack_spikes(window,
                     pct_diff=99.0,    # diff-percentile
                     pct_hf=95.0,      # HF envelope percentile
                     pct_ratio=90.0,   # HF/LF envelope ratio percentile
                     min_dist=SR//100):# 480 samples = 10ms spacing
    """
    Detect candidate pick‐clicks by:
     - looking for large diffs,
     - requiring a strong HF envelope,
     - requiring HF / LF envelope > some ratio.
    """
    # 1) fast-diff
    denv = hp_diff_env(window)   # abs diff after pre-emphasis

    # 2) band-pass → analytic envelopes
    hf  = sosfilt(SOS_HF, window)
    env_hf = np.abs(hilbert(hf))

    lf  = sosfilt(SOS_LF, window)
    env_lf = np.abs(hilbert(lf))

    # 3) ratio of HF to LF
    ratio = env_hf / (env_lf + 1e-8)

    # 4) set thresholds from percentiles + absolute floors
    thr_d = max(np.percentile(denv, pct_diff),
                ATTACK_SEGMENT_GENERATOR_MIN_DIFF_THRESH)
    thr_h = max(np.percentile(env_hf, pct_hf),
                ATTACK_SEGMENT_GENERATOR_MIN_ENV_THRESH)
    
    thr_r = max(np.percentile(ratio, pct_ratio), 1.0)  # at least 1×

    # 5) find all diff‐peaks
    peaks, _ = find_peaks(denv, height=thr_d, distance=min_dist)

    # 6) only keep those with both HF energy *and* HF/LF ratio
    good = [i for i in peaks
            if env_hf[i] >= thr_h and ratio[i] >= thr_r]

    return np.array(good, dtype=int)

def hp_diff_env(window, α=0.8):
    """1-tap pre-emphasis → abs(diff)"""
    hp = np.empty_like(window)
    hp[0] = window[0]
    hp[1:] = window[1:] - α * window[:-1]
    return np.abs(np.diff(hp))


def attack_segment_generator(seg_len, chunks, context_pre=1024):
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        mono = audio_cache[src][:,0]
        window = mono[cstart:cstart+cdur]
        spikes = find_attack_spikes(window)
        if spikes.size == 0:
            continue
        idx = int(random.choice(spikes))
        start = np.clip(idx - context_pre, 0, cdur-seg_len)
        rs = cstart + start

        piezo_seg = mono[rs: rs+seg_len]
        mic_seg   = audio_cache[tgt][rs: rs+seg_len,0]

        if np.max(np.abs(piezo_seg)) < ATTACK_SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        x = piezo_seg.astype('float32')[:,None]  # (seg_len,1)
        y = mic_seg.astype('float32')[:,None]

        yield x, y

def segment_generator(seg_len, chunks):
    """Uniform random windows, style=1.0 if pick-path else 0.0."""
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

        x = piezo_seg.astype('float32')[:,None]  # (seg_len,1)
        y = mic_seg.astype('float32')[:,None]

        yield x, y

def analyze_loss_scales(model, dataset):
    # pull out one batch
    (x, y) = next(iter(dataset))

    # run inference; may get a list of two outputs
    y_pred = model(x, training=False)

    # if it’s a list/tuple, sum the two heads
    if isinstance(y_pred, (list, tuple)):
        y_pred_audio = tf.add_n(y_pred)   # tone + pick
    else:
        y_pred_audio = y_pred

    # now y_pred_audio is shape (batch, T, 1)
    # and we can safely call each of your single‐headed losses:
    l1_val   = float(tf.reduce_mean(waveform_l1_loss(   y, y_pred_audio)))
    stft_val = float(tf.reduce_mean(multi_scale_stft_loss(y, y_pred_audio)))
    mel_val  = float(tf.reduce_mean(log_mel_loss(       y, y_pred_audio)))
    env_val  = float(tf.reduce_mean(rms_env_loss(       y, y_pred_audio)))
    hf_val   = float(tf.reduce_mean(hf_loss(            y, y_pred_audio)))

    return {
      'l1':   l1_val,
      'stft': stft_val,
      'mel':  mel_val,
      'env':  env_val,
      'hf':   hf_val
    }

def calibrate_loss_weights(model, dataset, n_batches=5):
    sums = {'l1':0., 'stft':0., 'mel':0., 'env':0., 'hf':0.}
    for i, (x,y) in enumerate(dataset.take(n_batches)):
        scales = analyze_loss_scales(model, tf.data.Dataset.from_tensors((x,y)))
        for k,v in scales.items():
            sums[k] += v
            
    # Print the average loss scales
    print("Average loss scales over", n_batches, "batches:")
    for k,v in sums.items():
        print(f"Initial {k} loss: {v:.4f}")
        
    return {k: sums[k]/n_batches for k in sums}

def generate_demo (model, out_dir, prediction_name="demo_pred", only_prediction=False):
    print("\nGenerating demo…")
    FRAMES      = LONG_DEMO_SEC * SR
    START_FRAME = LONG_DEMO_OFFSET_SEC * SR
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]

    x_long,_ = sf.read(pth_src,
                      start=START_FRAME,
                      frames=FRAMES,
                      dtype='float32',
                      always_2d=True)
    y_ref,_  = sf.read(pth_tgt,
                      start=START_FRAME,
                      frames=FRAMES,
                      dtype='float32',
                      always_2d=True)
    
    # only piezo
    x_piezo = x_long[:,0:1]             # (FRAMES,1)
    x_in    = x_piezo[None,...]         # (1,FRAMES,1)
    print("Running inference…")
    
    preds = model.predict(x_in, batch_size=1)
    if isinstance(preds, (list, tuple)):
        tone_pred, pick_pred = preds
        demo_pred = tone_pred + pick_pred
        
        tone_prediction_filepath = os.path.join(out_dir, "tone_" + prediction_name + ".wav")
        pick_prediction_filepath = os.path.join(out_dir, "pick_" + prediction_name + ".wav")
        sf.write(tone_prediction_filepath, tone_pred[0, :, 0], SR, subtype='FLOAT')
        sf.write(pick_prediction_filepath, pick_pred[0, :, 0], SR, subtype='FLOAT')

        print(f"✅ Tone-Demo saved to {tone_prediction_filepath}")
        print(f"✅ Pick-Demo saved to {pick_prediction_filepath}")

    else:
        demo_pred = preds
    y_pred_dyn = demo_pred[0, :, 0]
    
    prediction_filepath = os.path.join(out_dir, prediction_name + ".wav")
    sf.write(prediction_filepath, y_pred_dyn, SR, subtype='FLOAT')
    
    if (not only_prediction):
        source_filepath = os.path.join(out_dir, "demo_source" +".wav")
        target_filepath = os.path.join(out_dir, "demo_target" + ".wav")
        sf.write(source_filepath,
             x_long, SR, subtype='FLOAT')
        sf.write(target_filepath,
                y_ref, SR, subtype='FLOAT')
    
    print(f"✅ Demo saved to {prediction_filepath}")
    
def make_sinc_filter():
    t = np.arange(STUDENT_IR_FIR_SIZE) - (STUDENT_IR_FIR_SIZE - 1)/2.0
    h = np.sinc(2 * STUDENT_IR_CUTOFF * t)
    w = np.hamming(STUDENT_IR_FIR_SIZE)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float32)

_ir_fir     = make_sinc_filter()

h_lin = _ir_fir

# convert to minimum‐phase
h_min = minimum_phase(h_lin, method='homomorphic')
# re-normalize just in case
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)

_ir_fir_rev = h_min[::-1].reshape(-1,1,1)
_ir_fir_fwd = h_min.    reshape(-1,1,1)
_ir_fir_tf  = tf.constant(_ir_fir_rev, dtype=tf.float32)
_ir_fir_up  = tf.constant(_ir_fir_fwd, dtype=tf.float32)

@register_keras_serializable()
class StuIrAlign(Layer):
    """
    1) Pads the end of the IR branch if it's too short
    2) Crops off exactly `offset` samples from the front
    3) Then takes exactly as many samples as the main-branch (so Add(…,ir) always lines up)
    """
    def __init__(self,
                 offset,        # total group-delay in *input* samples
                 **kwargs):
        super().__init__(**kwargs)
        self.offset = int(offset)

    def call(self, inputs):
        ir, main = inputs
        # main shape = (batch, T, 2)  →  we only care about T
        T = tf.shape(main)[1]

        # after all your ↓→dilate→↑→postLP,  ir.shape = (batch, T′, 1)
        T_ir = tf.shape(ir)[1]

        # we need at least offset + T samples; pad if we ran short
        needed = self.offset + T
        pad_end = needed - T_ir
        # if pad_end>0, pad zeros on the right; else do nothing
        ir = tf.cond(
            pad_end>0,
            lambda: tf.pad(ir, [[0,0],[0,pad_end],[0,0]]),
            lambda: ir
        )

        # now crop out the first `offset` samples and take exactly T
        return ir[:, self.offset : self.offset + T, :]

    def compute_output_shape(self, input_shape):
        # input_shape = [ (batch, T′,1), (batch, T,2) ]
        b, T, _ = input_shape[1]
        return (b, T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"offset": self.offset})
        return cfg

@register_keras_serializable()
class StuIrDownsampleFIR(Layer):
    """
    Causal FIR+decimate on the piezo channel.
    Input:  (batch, T, 2)
    Output: (batch, ceil(T/ factor), 1)
    """
    def __init__(self, factor=STUDENT_IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.kernel_size = STUDENT_IR_FIR_SIZE

    def call(self, x):
        # x[...,0:1] is (batch, T, 1)
        p = x[...,0:1]
        # pad on left by kernel_size-1
        pad = self.kernel_size - 1
        p_pad = tf.pad(p, [[0,0], [pad,0], [0,0]])
        # VALID conv1d + stride = decimate
        return tf.nn.conv1d(
            p_pad,
            filters=_ir_fir_tf,
            stride=self.factor,
            padding='VALID'
        )

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        # after pad left by K-1, length = T + (K-1)
        padded = T + (self.kernel_size - 1)
        # VALID conv length = padded - (K-1) = T
        # then stride -> ceil(T/factor)
        out_T = (T + self.factor - 1) // self.factor
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'factor': self.factor})
        return cfg


@register_keras_serializable()
class StuIrPostLP(Layer):
    """
    Causal FIR smooth after upsampling.
    Input:  (batch, T, 1)
    Output: (batch, T, 1)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = STUDENT_IR_FIR_SIZE

    def call(self, x):
        # pad left by K-1
        pad = self.kernel_size - 1
        x_pad = tf.pad(x, [[0,0], [pad,0], [0,0]])
        return tf.nn.conv1d(
            x_pad,
            filters=_ir_fir_up,
            stride=1,
            padding='VALID'
        )

    def compute_output_shape(self, input_shape):
        # stays exactly (batch, T, 1)
        return input_shape

    def get_config(self):
        return super().get_config()

@register_keras_serializable()
def waveform_l1_loss(y_true, y_pred):
    return tf.abs(y_true - y_pred)

@register_keras_serializable()
def hf_loss(y_true, y_pred):
    α = HP_PRE_EMPHASIS
    # 1st‐order high-pass (pre-emphasis):
    true_hp = y_true[:,1:,:] - α*y_true[:,:-1,:]
    pred_hp = y_pred[:,1:,:] - α*y_pred[:,:-1,:]
    true_hp = tf.pad(true_hp, [[0,0],[1,0],[0,0]])
    pred_hp = tf.pad(pred_hp, [[0,0],[1,0],[0,0]])
    return tf.abs(true_hp - pred_hp)

@register_keras_serializable()
def multi_scale_stft_loss(y_true, y_pred):
    """
    Multi‐scale STFT L1 loss, averaged across time, frequency, and scale.
    Returns a tensor of shape (batch, 1, 1).
    """
    # 1) remove channel dim → (batch, T)
    y_true_s = tf.squeeze(y_true, axis=-1)
    y_pred_s = tf.squeeze(y_pred, axis=-1)

    # 2) STFT parameters for each scale
    scales = [
        (256,   64,   256),
        (512,  128,   512),
        (1024, 256,  1024),
        (2048, 512,  2048),
        (8192, 2048, 8192),
    ]

    losses = []
    for n_fft, hop, win in scales:
        # 2a) framing & windowing
        frames_true = tf.signal.frame(
            y_true_s, frame_length=win, frame_step=hop, pad_end=True
        )  # → (batch, frames, win)
        frames_pred = tf.signal.frame(
            y_pred_s, frame_length=win, frame_step=hop, pad_end=True
        )
        hann   = tf.signal.hann_window(win, periodic=True)          # (win,)
        window = tf.reshape(hann, (1, 1, win))                     # (1,1,win)
        win_t  = frames_true * window
        win_p  = frames_pred * window

        # 2b) FFT & magnitude
        fft_len = tf.constant([n_fft], dtype=tf.int32)
        st_t    = tf.signal.rfft(win_t,    fft_length=fft_len)     # (batch,frames,nbins)
        st_p    = tf.signal.rfft(win_p,    fft_length=fft_len)
        mag_t   = tf.abs(st_t)
        mag_p   = tf.abs(st_p)

        # 2c) L1 difference & mean over time+freq → (batch,)
        diff    = tf.abs(mag_t - mag_p)
        loss_sf = tf.reduce_mean(diff, axis=[1,2])  # now a 1‐D tensor
        losses.append(loss_sf)

    # 3) sum across scales and average → (batch,)
    total = tf.add_n(losses)
    avg   = total / float(len(scales))

    # 4) reshape → (batch,1,1)
    return tf.reshape(avg, (-1, 1, 1))

@register_keras_serializable()
def log_mel_loss(y_true, y_pred):
    """
    Compute L1 between log‐mel‐spectra of y_true and y_pred.
    Returns shape (batch,1,1).
    """
    # 1) STFT / mel‐spectrogram parameters
    n_fft   = 1024
    hop     = 256
    win     = 1024
    n_mels  = 64
    fmin    = 20.0
    fmax    = float(SR // 2)

    # 2) remove channel dim → (batch, T)
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)

    # 3) STFT magnitudes → (batch, frames, freq_bins)
    st_t  = tf.signal.stft(t,
                           frame_length=win,
                           frame_step=hop,
                           fft_length=n_fft,
                           pad_end=True)
    st_p  = tf.signal.stft(p,
                           frame_length=win,
                           frame_step=hop,
                           fft_length=n_fft,
                           pad_end=True)
    mag_t = tf.abs(st_t)   # (batch, frames, n_fft//2+1)
    mag_p = tf.abs(st_p)

    # 4) build mel‐filterbank → (freq_bins, n_mels)
    fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft//2+1,
        sample_rate=SR,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )

    # 5) apply mel‐filterbank: contract freq_bins
    #    mag_*: (batch, frames, freq_bins)
    #    fb:     (freq_bins, n_mels)
    # → mel_*: (batch, frames, n_mels)
    mel_t = tf.matmul(mag_t, fb)
    mel_p = tf.matmul(mag_p, fb)

    # 6) log compress
    log_t = tf.math.log(mel_t + 1e-6)
    log_p = tf.math.log(mel_p + 1e-6)

    # 7) L1 over time+mel → (batch,)
    mel_diff = tf.reduce_mean(tf.abs(log_t - log_p), axis=[1,2])

    # 8) reshape → (batch,1,1)
    return mel_diff[:, None, None]

@register_keras_serializable()
def rms_env_loss(y_true, y_pred):
    """
    Compute L1 between slow RMS envelopes of y_true and y_pred.
    Returns a tensor of shape (batch, 1, 1).
    """
    # window & hop for the slow‐RMS envelope
    env_win = 1024
    env_hop = 512

    # 1) remove the channel dim → (batch, T)
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)

    # 2) frame into (batch, num_frames, env_win)
    ft = tf.signal.frame(t,
                         frame_length=env_win,
                         frame_step=env_hop,
                         pad_end=True)
    fp = tf.signal.frame(p,
                         frame_length=env_win,
                         frame_step=env_hop,
                         pad_end=True)

    # 3) compute per‐frame RMS → (batch, num_frames)
    #    mean over the last axis (the env_win dim)
    rms_t = tf.sqrt(tf.reduce_mean(tf.square(ft), axis=-1) + 1e-12)
    rms_p = tf.sqrt(tf.reduce_mean(tf.square(fp), axis=-1) + 1e-12)

    # 4) L1 difference frame‐by‐frame → (batch, num_frames)
    diff = tf.abs(rms_t - rms_p)

    # 5) mean across frames → (batch,)
    env_diff = tf.reduce_mean(diff, axis=-1)

    # 6) reshape to (batch,1,1) to match your other losses
    return tf.reshape(env_diff, (-1, 1, 1))

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    td   = waveform_l1_loss(y_true, y_pred)      # (batch,T,1)
    hf   = hf_loss(       y_true, y_pred)        # (batch,T,1)
    spec = multi_scale_stft_loss(y_true, y_pred) # (batch,1,1)
    mel  = log_mel_loss(  y_true, y_pred)        # (batch,1,1)
    env  = rms_env_loss(  y_true, y_pred)        # (batch,1,1)

    return (L1_WEIGHT       * td
          + HF_WEIGHT       * hf
          + STFT_WEIGHT     * spec
          + MEL_WEIGHT      * mel
          + RMS_ENV_WEIGHT  * env)

# 3) A proper named metric instead of a lambda:
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    # This just reduces hf_loss to a scalar
    return tf.reduce_mean(hf_loss(y_true, y_pred))

@register_keras_serializable()
class TileStyle(Layer):
    def call(self, inputs):
        style_scalar, ref = inputs    # (batch,1), (batch,T,C)
        b = tf.shape(style_scalar)[0]
        T = tf.shape(ref)[1]
        s = tf.reshape(style_scalar, (b,1,1))
        return tf.tile(s, (1, T, 1))  # (batch, T, 1)

    def compute_output_shape(self, input_shapes):
        # input_shapes = [(b,1), (b,T,C)]
        return (input_shapes[1][0], input_shapes[1][1], 1)

@register_keras_serializable()
class InvertStyle(Layer):
    def call(self, style_time):
        return 1.0 - style_time
    def compute_output_shape(self, input_shape):
        return input_shape

@register_keras_serializable()
class ClickDelta(Layer):
    def call(self, x):
        # |piezo[t] - piezo[t-1]| with a leading zero
        d = tf.abs(x[:,1:,:] - x[:,:-1,:])
        return tf.pad(d, [[0,0],[1,0],[0,0]])
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class ClickTrigger(Layer):
    def __init__(self, threshold=CLICK_THRESH, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)
    def call(self, delta):
        return tf.cast(delta > self.threshold, tf.float32)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'threshold': self.threshold})
        return cfg
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class ClickEnv(Layer):
    def __init__(self, click_decay, **kwargs):
        super().__init__(**kwargs)
        # store decay kernel as numpy array
        self.click_decay = np.array(click_decay, dtype=np.float32)
    def build(self, input_shape):
        L = self.click_decay.shape[0]
        # one non-trainable kernel weight
        self.kernel = self.add_weight(
            name='kernel',
            shape=(L,1,1),
            initializer=tf.keras.initializers.Constant(
                self.click_decay.reshape(-1,1,1)
            ),
            trainable=False
        )
        super().build(input_shape)
    def call(self, inputs):
        L = self.click_decay.shape[0]
        # pad left by L−1 → VALID conv gives causal convolution
        x_pad = tf.pad(inputs, [[0,0],[L-1,0],[0,0]])
        return tf.nn.conv1d(x_pad, self.kernel, stride=1, padding='VALID')
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'click_decay': self.click_decay.tolist()})
        return cfg
    @classmethod
    def from_config(cls, config):
        return cls(**config)



@register_keras_serializable()
def get_piezo(x):
    # take channel 0 (piezo)
    return x[:, :, 0:1]

@register_keras_serializable()
def get_style(x):
    # take channel 1 (0.0 or 1.0)
    return x[:, :, 1:2]

@register_keras_serializable()
def invert_style(x):
    return 1.0 - x

# you already have these, but make sure they are @register_keras_serializable
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def extract_style_scalar(x):
    # x has shape (batch, T, 1) → we grab the first time‐step value
    # but keep the last dim so output is (batch, 1)
    return x[:, 0:1, 0:1]       # now shape (batch, 1)

@register_keras_serializable()
def expand_time_axis(x):
    # x has shape (batch, features) → we want (batch, 1, features)
    return tf.expand_dims(x, axis=1)

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

@register_keras_serializable()
def slice_piezo(x):
    # Takes channel 0 of a (batch, time, 2) tensor
    return x[:, :, 0:1]

@register_keras_serializable()
def slice_film_gamma(x, filters):
    # will work if x is (batch, 2*filters) → returns (batch, filters)
    # or if x is (batch, time, 2*filters) → returns (batch, time, filters)
    return x[..., :filters]

@register_keras_serializable()
def slice_film_beta(x, filters):
    return x[..., filters:]

@register_keras_serializable()
def slice_style(x):
    # Takes channel 1 of a (batch, time, 2) tensor
    return x[:, :, 1:2]

@register_keras_serializable()
def invert(x):
    return 1.0 - x


def build_teacher_model(dynamic_time=False):

    T   = None if dynamic_time else TEACHER_SEG_LEN
    inp = Input(shape=(T,1), name="tch_piezo_input")
    
    cls         = Conv1D(16,3,padding="same", activation="relu", name="tch_style_conv")(inp)
    cls         = GlobalAveragePooling1D(name="tch_style_gap")(cls)
    style_embed = Dense(TEACHER_EMBED_DIM,
                        activation="linear",
                        name="tch_style_embed")(cls)
    
    x = Conv1D(TEACHER_FILTERS, 1,
               padding="same",
               name="tch_init_pw")(inp)

    for idx, d in enumerate(TEACHER_DILATIONS):
        res = x
        pad = (TEACHER_KERNEL_SIZE - 1) * d
        x = ZeroPadding1D((pad,0), name=f"pad_{idx}")(x)
        g = SeparableConv1D(TEACHER_FILTERS*2, TEACHER_KERNEL_SIZE, dilation_rate=d,
                            padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':TEACHER_FILTERS},
                   name=f"slice_content_{idx}")(g)
        g2 = Lambda(slice_gate, arguments={'filters':TEACHER_FILTERS},
                    name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([
                Activation('tanh', name=f"tanh_{idx}")(c),
                Activation('sigmoid', name=f"sigmoid_{idx}")(g2)
            ])
        x = Conv1D(TEACHER_FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        # FiLM modulation

        film_params = Dense(TEACHER_FILTERS*2, name=f"film_dense_{idx}")(style_embed)
        gamma = Lambda(slice_film_gamma,
                    arguments={'filters': TEACHER_FILTERS},
                    output_shape=(TEACHER_FILTERS,),
                    name=f"film_gamma_{idx}")(film_params)
        beta  = Lambda(slice_film_beta,
                    arguments={'filters': TEACHER_FILTERS},
                    output_shape=(TEACHER_FILTERS,),
                    name=f"film_beta_{idx}")(film_params)

        x = Add(name=f"film_apply_{idx}")([
            Multiply(name=f"film_scale_{idx}")([x, gamma]),
            beta
        ])
    
    # 1) tone head (everything except pick clicks)
    tone_out = Conv1D(1,1, padding="same", name="tch_tone_out")(x)

    # 2) pick head (the pluck‐transient only)
    piezo_raw = inp
    attack_trig = ClickTrigger(name="tch_click_trig")(
                    ClickDelta(name="tch_click_delta")(piezo_raw)
                )
    
    # now learn, say, 8 different 31-tap impulse responses
    click_bank = Conv1D(8,
                        kernel_size=CLICK_KERNEL_SIZE,
                        padding="same",
                        name="tch_click_bank")(attack_trig)
    click_bank = Activation("linear")(click_bank)       # optional non-linearity
    # collapse them back to 1 channel with a zero-latency 1×1
    click_ir   = Conv1D(1,
                        kernel_size=1,
                        padding="same",
                        name="tch_click_ir")(click_bank)
    
    pick_out   = ClickScale(init=1.0, name="tch_pick_out")(click_ir)

    return tf.keras.models.Model(
        inputs  = inp,
        outputs = [tone_out, pick_out],
        name    = "piezo2mic_teacher"
    )

@register_keras_serializable()
def build_student_model(
    teacher=None,
    dynamic_time=False
):
    # 1) input length: fixed for training, variable for inference
    T = None if dynamic_time else STUDENT_SEG_LEN
    inp = Input(shape=(T, 1), name="stu_piezo_input")

    # 2) style‐head (same as teacher)
    cls = Conv1D(16, 3, padding="same", activation="relu", name="stu_style_conv")(inp)
    cls = GlobalAveragePooling1D(name="stu_style_gap")(cls)
    style_embed = Dense(STUDENT_EMBED_DIM,
                        activation="linear",
                        name="stu_style_embed")(cls)

    # 3) initial pointwise projection
    x = Conv1D(STUDENT_FILTERS,
               kernel_size=1,
               padding="same",
               name="stu_init_pw")(inp)

    # 4) Dilated TCN Blocks + FiLM
    student_feats = []
    for i, d in enumerate(STUDENT_DILATIONS):
        res = x
        pad = (STUDENT_KERNEL_SIZE - 1) * d
        x = ZeroPadding1D((pad, 0), name=f"stu_pad_{i}")(x)
        g = SeparableConv1D(2 * STUDENT_FILTERS,
                            STUDENT_KERNEL_SIZE,
                            dilation_rate=d,
                            padding="valid",
                            name=f"stu_sep_{i}")(x)

        c  = Lambda(slice_content,
                    arguments={"filters": STUDENT_FILTERS},
                    name=f"stu_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,
                    arguments={"filters": STUDENT_FILTERS},
                    name=f"stu_slice_g_{i}")(g)

        x = Multiply(name=f"stu_gate_mul_{i}")([
            Activation("tanh",    name=f"stu_tanh_{i}")(c),
            Activation("sigmoid", name=f"stu_sigmoid_{i}")(g2)
        ])
        x = Conv1D(STUDENT_FILTERS,
                   kernel_size=1,
                   padding="same",
                   name=f"stu_pw_{i}")(x)
        x = Add(name=f"stu_res_add_{i}")([x, res])

        # FiLM modulation
        film = Dense(2 * STUDENT_FILTERS,
                     name=f"stu_film_dense_{i}")(style_embed)
        gamma = Lambda(slice_film_gamma,
                       arguments={"filters": STUDENT_FILTERS},
                       name=f"stu_film_gamma_{i}")(film)
        beta  = Lambda(slice_film_beta,
                       arguments={"filters": STUDENT_FILTERS},
                       name=f"stu_film_beta_{i}")(film)
        x     = Add(name=f"stu_film_apply_{i}")([
            Multiply(name=f"stu_film_scale_{i}")([x, gamma]),
            beta
        ])

        # collect for feature-distill if needed
        layer_name = f"res_add_{i}"
        if STUDENT_TEACHER_FEAT_LAYERS and layer_name in STUDENT_TEACHER_FEAT_LAYERS:
            student_feats.append(x)

    # 5) tone head (before click)
    tone_out = Conv1D(1, 1, padding="same", name="stu_tone_out")(x)

    # 6) low-rate IR branch
    ir = StuIrDownsampleFIR(name="stu_ir_downsample")(inp)
    for i, d in enumerate(STUDENT_IR_DILATIONS):
        pad = (STUDENT_IR_KERNEL - 1) * d
        ir = ZeroPadding1D((pad, 0), name=f"stu_ir_pad_{i}")(ir)
        ir = SeparableConv1D(
            filters=1,
            kernel_size=STUDENT_IR_KERNEL,
            dilation_rate=d,
            padding="valid",
            name=f"stu_ir_sep_{i}"
        )(ir)
    ir = IrScale(init=0.6, name="stu_ir_scale")(ir)
    ir = UpSampling1D(size=STUDENT_IR_DOWNSAMPLE_FACTOR, name="stu_ir_upsample")(ir)
    ir = StuIrPostLP(name="stu_ir_post_lp")(ir)
    total_delay = STUDENT_TOTAL_EXPECTED_IR_DELAY
    ir = StuIrAlign(offset=total_delay + STUDENT_IR_ALIGN_OFFSET_CORRECTION,
                    name="stu_ir_align")([ir, inp])
    ir = IRLowpass(name="stu_ir_lowpass")(ir)

    # fuse IR into the tone head
    tone_out = Add(name="stu_tone_with_ir")([tone_out, ir])

    # 7) pick head (click-transient)
    raw = inp
    trig = ClickTrigger(name="stu_click_trig")(
             ClickDelta(name="stu_click_delta")(raw)
           )
    # now learn, say, 8 different 31-tap impulse responses
    click_bank = Conv1D(8,
                        kernel_size=CLICK_KERNEL_SIZE,
                        padding="same",
                        name="stu_click_bank")(trig)
    click_bank = Activation("linear")(click_bank)       # optional non-linearity
    # collapse them back to 1 channel with a zero-latency 1×1
    click_ir   = Conv1D(1,
                        kernel_size=1,
                        padding="same",
                        name="stu_click_ir")(click_bank)
    pick_out = ClickScale(init=1.0, name="stu_pick_out")(click_ir)

    # 8) distillation losses (unchanged)
    if teacher is not None:
        teacher.trainable = False

        # a) intermediate-feature distillation
        if STUDENT_TEACHER_FEAT_LAYERS:
            teacher_feat_model = tf.keras.models.Model(
                inputs  = teacher.input,
                outputs = [teacher.get_layer(n).output
                           for n in STUDENT_TEACHER_FEAT_LAYERS]
            )
            T_feats = teacher_feat_model(inp)
            if not isinstance(T_feats, list):
                T_feats = [T_feats]
            for idx, (T_f, S_f) in enumerate(zip(T_feats, student_feats)):
                chan = int(T_f.shape[-1])
                proj = Conv1D(chan, 1,
                              padding="same",
                              name=f"stu_proj_feat_{idx}")(S_f)
                _ = DistillLoss(alpha=STUDENT_FEAT_DISTILL_ALPHA,
                                name=f"feat_distill_{idx}")([T_f, proj])

        # b) final-waveform distillation on the _sum_ of our two heads
        teacher_audio, _ = teacher(inp)
        combined = Add()([tone_out, pick_out])
        _ = DistillLoss(alpha=STUDENT_DISTILL_ALPHA,
                        name="distillation")([teacher_audio, combined])

    return tf.keras.models.Model(
        inputs  = inp,
        outputs = [tone_out, pick_out],
        name    = "piezo2mic_student"
    )

def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations, filters):
    # TODO
    print("TODO")


def approx_steps(durs, seg_len):
        return 10 if DEBUG else sum(max((d-seg_len)//(seg_len//2)+1,0) for d in durs)//BATCH_SIZE

def train_shared(
    seg_len,
    build_model_fn,
    checkpoint_out_dir,
    logs_out_dir,
    csv_logs_out_dir,
    demo_checkpoints_out_dir,
    static_model_filepath,
    dynamic_model_filepath,
    resume=False,
    resume_ckpt_path=None,
    build_model_fn_args={}):
    global L1_WEIGHT, HF_WEIGHT, STFT_WEIGHT, MEL_WEIGHT, RMS_ENV_WEIGHT
    
    all_chunks = build_chunks(pairs_list=all_pairs, seg_len=seg_len)
    pick_chunks = build_chunks(pairs_list=picked_pairs, seg_len=seg_len)
    finger_chunks = build_chunks(pairs_list=finger_pairs, seg_len=seg_len)

    # train/val split
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

    # datasets
    sig_x = tf.TensorSpec((seg_len,1), tf.float32)    
    sig_y = tf.TensorSpec((seg_len,1), tf.float32)

    pick_attack_ds    = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=train_pick_chunks), output_signature=(sig_x,sig_y)
    )
    
    pick_uniform_ds    = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_pick_chunks), output_signature=(sig_x,sig_y)
    )
    
    finger_attack_ds  = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=train_finger_chunks), output_signature=(sig_x,sig_y)
    )
    
    finger_uniform_ds  = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_finger_chunks), output_signature=(sig_x,sig_y)
    )

    train_ds = tf.data.Dataset.sample_from_datasets(
        [pick_attack_ds, pick_uniform_ds, finger_attack_ds, finger_uniform_ds], weights=[PICK_ATTACK_CHUNK_RATIO, PICK_UNIFORM_CHUNK_RATIO, FINGER_ATTACK_CHUNK_RATIO, FINGER_UNIFORM_CHUNK_RATIO]
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    
    train_ds = train_ds.repeat().prefetch(tf.data.AUTOTUNE)

    # validation streams
    val_pick_attack_ds = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=val_pick_chunks),
        output_signature=(sig_x, sig_y)
    )
    
    val_pick_uniform_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_pick_chunks),
        output_signature=(sig_x, sig_y)
    )
    
    val_finger_attack_ds = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=val_finger_chunks),
        output_signature=(sig_x, sig_y)
    )
    
    val_finger_uniform_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_finger_chunks),
        output_signature=(sig_x, sig_y)
    )
    
    val_ds = tf.data.Dataset.sample_from_datasets(
        [val_pick_attack_ds, val_pick_uniform_ds, val_finger_attack_ds, val_finger_uniform_ds],
        weights=[PICK_ATTACK_CHUNK_RATIO, PICK_UNIFORM_CHUNK_RATIO, FINGER_ATTACK_CHUNK_RATIO, FINGER_UNIFORM_CHUNK_RATIO],
        seed=SEED
    ).batch(BATCH_SIZE)
    
    val_ds   = val_ds.repeat().prefetch(tf.data.AUTOTUNE)

    # build or load model
    if resume and resume_ckpt_path:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        # load weights (we do compile=False so we can recompile with fresh optimizer)
        static_model = tf.keras.models.load_model(resume_ckpt_path, compile=False)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args, dynamic_time=False)
        initial_epoch = 0
    
    # Measure raw scales of each loss term on one batch of val_ds
    # baseline = calibrate_loss_weights(static_model, val_ds, n_batches=5)
    
    # ref_scale = float(baseline['l1'])
    # user_w    = {
    #     'l1':  L1_WEIGHT,
    #     'stft': STFT_WEIGHT,
    #     'mel':  MEL_WEIGHT,
    #     'env':  RMS_ENV_WEIGHT,
    #     'hf':   HF_WEIGHT
    # }

    # new_w = {}
    # for key, uw in user_w.items():
    #     bs = float(baseline[key])
    #     if bs > 0.0:
    #         new_w[key] = uw / bs * ref_scale
    #     else:
    #         new_w[key] = 0.0

    # L1_WEIGHT       = new_w['l1']
    # STFT_WEIGHT     = new_w['stft']
    # MEL_WEIGHT      = new_w['mel']
    # RMS_ENV_WEIGHT  = new_w['env']
    # HF_WEIGHT       = new_w['hf']
    
    # print(f"→ normalized loss weights:"
    #       f"  L1={L1_WEIGHT:.6f}"
    #       f"  STFT={STFT_WEIGHT:.6f}"
    #       f"  MEL={MEL_WEIGHT:.6f}"
    #       f"  ENV={RMS_ENV_WEIGHT:.6f}"
    #       f"  HF={HF_WEIGHT:.6f}")

    static_model.compile(
        optimizer   = tf.keras.optimizers.Adam(LR),
        loss        = [combined_loss, combined_loss],
        loss_weights= [1.0, 1.0],
        metrics     = [
                [
                    "mae"
                ],
                [
                    "mae"
                ]
            
            ]
    )

    static_model.summary()
    
    # Early stopping callback
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=LR_PATIENCE, restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=MIN_LR,
        mode='min',
        verbose=1
    )
    
    # ModelCheckpoint: save whenever val_loss improves
    cp = ModelCheckpoint(
        filepath=os.path.join(checkpoint_out_dir,
                            "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
        
    demo_cb = DemoCallback(
        build_model_fn=build_model_fn,
        build_model_fn_args=build_model_fn_args,
        logs_out_dir=logs_out_dir,
        demo_checkpoints_out_dir=demo_checkpoints_out_dir
    )
    tb = TensorBoard(log_dir=logs_out_dir, write_graph=False)
    csv_logger = CSVLogger(os.path.join(csv_logs_out_dir, "training.csv"), append=False)

    # ── split each batch into tone/pick targets and sample‐weights ───────────
    def split_targets(x, y):
        # x: (B,T,1) piezo,   y: (B,T,1) mic
        # 1) detect raw click‐triggers just like you had
        delta = tf.abs(x[:,1:,:] - x[:,:-1,:])
        delta = tf.pad(delta, [[0,0],[1,0],[0,0]])
        triggers = tf.cast(delta > CLICK_THRESH, tf.float32)  # (B,T,1)

        # 2) convolve with our small exp-decay kernel → a soft “pick” envelope
        pick_env = tf.nn.conv1d(triggers,
                                filters=decay,
                                stride=1,
                                padding='SAME')
        pick_env = tf.clip_by_value(pick_env, 0.0, 1.0)        # still (B,T,1)

        # 3) carve teacher-targets for tone vs pick
        tone_env = 1.0 - pick_env                          # (B,T,1)
        y_tone = y * tone_env
        y_pick = y * pick_env

        # 4) sample-weights for each head
        #    Keras wants shape (B,T), so we squeeze:
        sw_tone = tf.squeeze(tone_env, axis=-1)   # (B,T)
        sw_pick = tf.squeeze(pick_env,  axis=-1)  # (B,T)

        # 5) return exactly the same signature as before:
        #    x_in, (y_tone,y_pick), (sw_tone,sw_pick)
        return x, (y_tone, y_pick), (sw_tone, sw_pick)

    # map your train/val datasets:
    train_multi = train_ds.map(split_targets)
    val_multi   = val_ds.map(split_targets)

    static_model.fit(
        train_multi,
        initial_epoch   = initial_epoch,
        epochs          = 1 if DEBUG else EPOCHS,
        steps_per_epoch = steps_per_epoch,
        validation_data   = val_multi,
        validation_steps  = validation_steps,
        callbacks       = [es, cp, reduce_lr, tb, demo_cb, csv_logger]
    )

    # trained_epochs = len(history.history["loss"])
    
    # 1) save & inject static
    static_model.save(static_model_filepath)
    print("✅ Saved FP32 static model:", static_model_filepath)

    # 2) build, save & inject dynamic
    dynamic_model = build_model_fn(**build_model_fn_args, dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    dynamic_model.save(dynamic_model_filepath)
    print("✅ Saved FP32 dynamic model:", dynamic_model_filepath)

    # ── Optional: dump .npy FP32 weights & extract INT8 ───────────────────
    if SAVE_WEIGHTS:
        pass

    return static_model, dynamic_model

def train_teacher(resume=False, resume_ckpt_path=None):
    train_shared(
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=TEACHER_SEG_LEN,
        static_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_teacher_model,
        checkpoint_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
        logs_out_dir=LOGS_TEACHER_OUT_DIR,
        csv_logs_out_dir=CSV_LOGS_TEACHER_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_TEACHER_OUT_DIR
    )

def train_student(resume=False, resume_ckpt_path=None):
    teacher_dynamic_model_filepath = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(teacher_dynamic_model_filepath):
        print("Teacher model not found, run --train-teacher first")
        sys.exit(1)
    teacher = tf.keras.models.load_model(teacher_dynamic_model_filepath, compile=False)
    
    print("Loaded teacher from", teacher_dynamic_model_filepath)
    
    tmp_dynamic_model = build_student_model(
        teacher=teacher,
        dynamic_time=True
    )
    
    # Validate the final group delay
    validate_ir_delay(tmp_dynamic_model)
    
    train_shared(
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=STUDENT_SEG_LEN,
        static_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_student_model,
        build_model_fn_args={
            'teacher': teacher
        },
        checkpoint_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
        logs_out_dir=LOGS_STUDENT_OUT_DIR,
        csv_logs_out_dir=CSV_LOGS_STUDENT_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_STUDENT_OUT_DIR
    )

def generate_fine_tune_demo(
    model,                 # your dynamic model that returns [waveform, style_prob]
    src_path,              # "./finetune/source.wav"
    tgt_path,              # "./finetune/target.wav"
    out_dir,               # where to write .wav files
    prediction_name="pred" # e.g. "teacher_pred" or "student_pred"
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load full signals (always_2d=True just in case your files
    #    are still two-channel on disk)
    x_full, _ = sf.read(src_path,
                        dtype='float32',
                        always_2d=True)  # shape=(N,2) or (N,1)
    y_ref,  _ = sf.read(tgt_path,
                        dtype='float32',
                        always_2d=True)

    # 2) Truncate to your demo length
    max_frames = int(FINE_TUNE_DEMO_SEC * SR)
    if x_full.shape[0] > max_frames:
        x_full = x_full[:max_frames]
        y_ref   = y_ref[:max_frames]

    # 3) Take *only* the piezo channel
    #    (assumes piezo was channel 0 on disk)
    x_piezo = x_full[:,0:1]         # shape (N,1)
    x_in    = x_piezo[None, ...]    # shape (1, N, 1)

    # 4) Run inference → y_out=(1,N,1), style_out=(1,1)
    preds = model.predict(x_in, batch_size=1)
    if isinstance(preds, (list, tuple)):
        tone_pred, pick_pred = preds
        demo_pred = tone_pred + pick_pred
    else:
        demo_pred = preds
    y_pred = demo_pred[0, :, 0]
    
    # 5) Write to disk
    sf.write(os.path.join(out_dir, "demo_source.wav"),
             x_full[:,0], SR, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "demo_target.wav"),
             y_ref[:,0],  SR, subtype='FLOAT')
    sf.write(os.path.join(out_dir, f"{prediction_name}.wav"),
             y_pred,      SR, subtype='FLOAT')

    print(f"   → Wrote: {out_dir}/{prediction_name}.wav\n")


def generate_fine_tune_demo_from_models(teacher_dyn=None, student_dyn=None):
    print("\n▶ Generating finetune demos…")

    # for tag, model_dyn in [("teacher", teacher_dyn),
    #                        ("student", student_dyn)]:
    for tag, model_dyn in [("student", student_dyn)]:
        # Skip if model is None
        if model_dyn is None:
            continue

        demo_dir = os.path.join(FINETUNE_DEMOS_OUT_DIR, f"demo_{tag}")
        # only one pass per model now, no hard-coded style flags
        generate_fine_tune_demo(
            model           = model_dyn,
            src_path        = os.path.join(FINETUNE_SRC_DIR, "source.wav"),
            tgt_path        = os.path.join(FINETUNE_SRC_DIR, "target.wav"),
            out_dir         = demo_dir,
            prediction_name = f"{tag}_pred"
        )

    print("✅ Fine-tune complete, demos in:\n"
          "   " + "\n   ".join([
             os.path.join(FINETUNE_DEMOS_OUT_DIR, f"demo_teacher"),
             os.path.join(FINETUNE_DEMOS_OUT_DIR, f"demo_student")
          ]))

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

    # 1) load raw waveforms
    x_full, sr_x = sf.read(src, dtype='float32', always_2d=True)
    y_full, sr_y = sf.read(tgt, dtype='float32', always_2d=True)
    assert sr_x == sr_y == SR, f"Sample rate mismatch: {sr_x} vs {sr_y} vs {SR}"

    # 2) measure RMS on piezo channel (assumed channel 0)
    rms_x = compute_rms(x_full[:,0])
    rms_y = compute_rms(y_full[:,0])
    print(f"[fine_tune] input RMS: source={rms_x:.6f}, target={rms_y:.6f}")

    # 3) compute target RMS (–23 dBFS) and gains
    target_rms = 10 ** ( INPUT_AVG_RMS_DBFS_TARGET / 20.0 )   # ≃0.0708
    gain_x     = target_rms / (rms_x + 1e-12)
    gain_y     = target_rms / (rms_y + 1e-12)
    print(f"[fine_tune] applying gains: source×{gain_x:.3f}, target×{gain_y:.3f}")

    # 4) normalize
    x_norm = x_full * gain_x
    y_norm = y_full * gain_y

    # re‐measure RMS to double‐check
    rms_x2 = compute_rms(x_norm[:,0])
    rms_y2 = compute_rms(y_norm[:,0])
    print(f"[fine_tune] post‐norm RMS: source={rms_x2:.6f} ({20*np.log10(rms_x2):.2f} dB),"
          f" target={rms_y2:.6f} ({20*np.log10(rms_y2):.2f} dB)")

    # 6) repopulate your caches so the rest of fine-tune uses the normalized versions
    picked_pairs = [(src, tgt)]
    finger_pairs = [(src, tgt)]
    all_pairs    = picked_pairs + finger_pairs

    audio_cache.clear()
    audio_cache[src] = x_norm      # shape=(frames,2) or (frames,1)
    audio_cache[tgt] = y_norm

    def make_mixed_ds(seg_len):
        chunks = build_chunks(all_pairs, seg_len)
        
        spec_x       = tf.TensorSpec((seg_len,1), tf.float32)
        spec_y       = tf.TensorSpec((seg_len,1), tf.float32)
        
        def pa_gen():
            for x,y in attack_segment_generator(seg_len, chunks):
                yield x, y

        def pu_gen():
            for x,y in segment_generator(seg_len, chunks):
                yield x, y
        
        def fa_gen():
            for x,y in attack_segment_generator(seg_len, chunks):
                yield x,y
        def fu_gen():
            for x,y in segment_generator(seg_len, chunks):
                yield x,y

        ds_pa = tf.data.Dataset.from_generator(pa_gen, output_signature=(spec_x, spec_y))
        
        ds_pu = tf.data.Dataset.from_generator(pu_gen, output_signature=(spec_x, spec_y))
        ds_fa = tf.data.Dataset.from_generator(fa_gen, output_signature=(spec_x, spec_y))
        ds_fu = tf.data.Dataset.from_generator(fu_gen, output_signature=(spec_x, spec_y))

        mixed = tf.data.Dataset.sample_from_datasets(
            [ds_pa, ds_pu, ds_fa, ds_fu],
            weights=[PICK_ATTACK_CHUNK_RATIO,
                     PICK_UNIFORM_CHUNK_RATIO,
                     FINGER_ATTACK_CHUNK_RATIO,
                     FINGER_UNIFORM_CHUNK_RATIO],
            seed=SEED
        )

        return mixed.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    finetuned_teacher = None
    finetuned_teacher_dyn = None
    finetuned_student = None
    finetuned_student_dyn = None
    
    # if len(sys.argv) <= 2 or sys.argv[2] !="--student-only":

    #     # ─── Fine-tune TEACHER ────────────────────────────────────────────────
    #     print("\n🔧 Fine-tuning TEACHER on new pair…")
    #     base_teacher_path = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_STATIC_MODEL_FILENAME)
    #     teacher = tf.keras.models.load_model(base_teacher_path, compile=False)
        
    #     # We'll base the finetuned teacher on the original teacher with some frozen layers
    #     finetuned_teacher = teacher
        
    #     LAST_TEACHER_DILATION = len(TEACHER_DILATIONS) - 1
    #     NEXT_LAST_TEACHER_DILATION = len(TEACHER_DILATIONS) - 2

    #     for layer in finetuned_teacher.layers:
    #         name = layer.name
    #         layer.trainable = \
    #             name.startswith("tch_film_dense_") \
    #             or name.startswith("tch_final_") \
    #             or name == f"sep_{LAST_TEACHER_DILATION}" \
    #             or name == f"sep_{NEXT_LAST_TEACHER_DILATION}" \
    #             or name == f"pw_{LAST_TEACHER_DILATION}" \
    #             or name == f"pw_{NEXT_LAST_TEACHER_DILATION}" \
        
    #     # build a tiny "val" dataset of mixed segments (already batched)
    #     dsT_val = make_mixed_ds(TEACHER_SEG_LEN)
    #     # now attach your sample‐weight map exactly as in train_shared()
    #     # only take 2 batches (you just need a few to get stable scales)
    #     dsT_val = dsT_val.take(5)

    #     # measure each loss term on that small ds
    #     baseline_T = calibrate_loss_weights(finetuned_teacher,
    #                                         dsT_val,
    #                                         n_batches=5)

    #     # pick L1 as our reference scale
    #     ref = float(baseline_T['l1'])

    #     # globally re-scale your HF/STFT/MEL/ENV weights        
    #     L1_WEIGHT      = corrected_weights['l1']   / baseline_T['l1']   * ref
    #     HF_WEIGHT      = corrected_weights['hf']   / baseline_T['hf']   * ref
    #     STFT_WEIGHT    = corrected_weights['stft'] / baseline_T['stft'] * ref
    #     MEL_WEIGHT     = corrected_weights['mel']  / baseline_T['mel']  * ref
    #     RMS_ENV_WEIGHT = corrected_weights['env']  / baseline_T['env']  * ref

    #     print("→ finetune-teacher loss weights recalibrated:"
    #         f"  HF={HF_WEIGHT:.4f}"
    #         f"  STFT={STFT_WEIGHT:.4f}"
    #         f"  MEL={MEL_WEIGHT:.4f}"
    #         f"  ENV={RMS_ENV_WEIGHT:.4f}")
        
    #     finetuned_teacher.compile(
    #         optimizer = tf.keras.optimizers.Adam(FINETUNE_TEACHER_LR),
    #         loss        = [ combined_loss, "binary_crossentropy" ],
    #         loss_weights= [ 1.0,             λ_STYLE_RATIO      ],
    #         metrics     = [
    #         # metrics for the audio head - we keep them at a minimum to for performance
    #         [ "mae"
    #         ],
    #         # metrics for the style head (we don't need any, as the style layer is frozen)
    #         []
    #         ]
    #     )
        
    #     dsT = make_mixed_ds(TEACHER_SEG_LEN)
        
    #     ft_chunks_T = build_chunks([(src,tgt)], TEACHER_SEG_LEN)
    #     stepsT = max(approx_steps( durs=[dur for (_,_,_,dur) in ft_chunks_T], seg_len=TEACHER_SEG_LEN), 1)
        
    #     # Print the model summary
    #     print("Finetune teacher model summary:")
    #     finetuned_teacher.summary()
        
    #     finetuned_teacher.fit(dsT, epochs=FINETUNE_TEACHER_EPOCHS, steps_per_epoch=stepsT, verbose=1)

    #     # Save new teacher
    #     ft_t_static = os.path.join(MODELS_TEACHER_OUT_DIR, FINETUNE_TEACHER_STATIC_MODEL_FILENAME)
    #     finetuned_teacher.save(ft_t_static)
    #     print("✅ Saved finetuned teacher:", ft_t_static)

    #     finetuned_teacher_dyn = build_teacher_model(dynamic_time=True)
    #     finetuned_teacher_dyn.set_weights(finetuned_teacher.get_weights())
    #     ft_t_dyn = os.path.join(MODELS_TEACHER_OUT_DIR, FINETUNE_TEACHER_DYNAMIC_MODEL_FILENAME)
    #     finetuned_teacher_dyn.save(ft_t_dyn)
    #     print("✅ Saved finetuned teacher dynamic:", ft_t_dyn)

    if len(sys.argv) <= 2 or sys.argv[2] !="--teacher-only":
        # ─── Fine-tune STUDENT ────────────────────────────────────────────────
        print("\n🔧 Fine-tuning STUDENT on new pair…")
        
        # load the PRE-TRAINED (base) teacher dynamic model
        # teacher_dyn = tf.keras.models.load_model(
        #     os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
        #     compile=False
        # )
        # teacher_dyn.trainable = False

        # This is the full original base student model with the teacher weights baked in + distillation layers
        base_student_full = tf.keras.models.load_model(
            os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_STATIC_MODEL_FILENAME),
            compile=False
        )

        finetuned_student_small = build_student_model(teacher=None, dynamic_time=False)

        # Copy over all of the 'stu_' weights from the full original base student model,
        # while making the FILM and final layers trainable
        for layer in finetuned_student_small.layers:
            if layer.name.startswith("stu_"):
                try:
                    layer.set_weights(
                        base_student_full.get_layer(layer.name).get_weights()
                    )
                except ValueError:
                    # some layers have no weights
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
                or name == f"stu_pw_{LAST_STUDENT_DILATION}"

        dsS_val = make_mixed_ds(STUDENT_SEG_LEN)
        dsS_val = dsS_val.take(5)

        baseline_S = calibrate_loss_weights(finetuned_student_small,
                                            dsS_val,
                                            n_batches=5)
        
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

        
        # 5) compile & fit   
        finetuned_student_small.compile(
            optimizer   = tf.keras.optimizers.Adam(FINETUNE_STUDENT_LR),
            loss        = [combined_loss, waveform_l1_loss],
            loss_weights= [1.0, 1.0],
            metrics     = [
                    [
                        "mae"
                    ],
                    [
                        "mae"
                    ]
                
                ]
        )
        
        
        
        dsS = make_mixed_ds(STUDENT_SEG_LEN)
        
        ft_chunks_T = build_chunks([(src,tgt)], STUDENT_SEG_LEN)
        stepsS = max(approx_steps( durs=[dur for (_,_,_,dur) in ft_chunks_T], seg_len=STUDENT_SEG_LEN), 1)
        
        # Print the model summary
        print("Finetune student model summary:")
        finetuned_student_small.summary()
        
        finetuned_student_small.fit(dsS,
                        epochs=FINETUNE_STUDENT_EPOCHS,
                        steps_per_epoch=stepsS,
                        verbose=1)

        # 6) save the new static student
        ft_s_static = os.path.join(MODELS_STUDENT_OUT_DIR,
                                FINETUNE_STUDENT_STATIC_MODEL_FILENAME)
        finetuned_student_small.save(ft_s_static)

        # 7) build a dynamic‐time‐friendly student (without the teacher!)
        finetuned_student_dyn = build_student_model(teacher=None, dynamic_time=True)
        finetuned_student_dyn.set_weights(finetuned_student_small.get_weights())
        ft_s_dyn = os.path.join(MODELS_STUDENT_OUT_DIR,
                                FINETUNE_STUDENT_DYNAMIC_MODEL_FILENAME)
        finetuned_student_dyn.save(ft_s_dyn)
    
    return (finetuned_teacher_dyn, finetuned_student_dyn)

# 2) measure RMS on piezo channel (assumed channel 0)
def compute_rms(sig):
    return np.sqrt(np.mean(sig**2))

def cleanup_and_prepare_train (logs_out_dir, csv_logs_out_dir, checkpoints_out_dir, demo_checkpoints_out_dir, demo_generator_outputs_out_dir):
    # Clean logs and checkpoints
        for dir_path in [logs_out_dir, csv_logs_out_dir, demo_checkpoints_out_dir, demo_generator_outputs_out_dir]:
            for sub in os.listdir(dir_path):
                sub_path = os.path.join(dir_path, sub)
                if os.path.isfile(sub_path):
                    os.remove(sub_path)
                elif os.path.isdir(sub_path):
                    shutil.rmtree(sub_path)
        
         # look for existing checkpoints
        ckpts = sorted(glob.glob(os.path.join(checkpoints_out_dir, "*.keras")))
        resume = len(ckpts) > 0
        resume_ckpt = ckpts[-1] if resume else None

        if resume:
            ep = _extract_epoch_from_ckpt(resume_ckpt)
            print(f"→ Found checkpoint, will resume from epoch {ep}: {resume_ckpt}")
            return (resume, resume_ckpt)
        else:
            return (False, None)

def prepare_and_generate_demo (dynamic_model_filepath, checkpoint_out_dir, demo_out_dir, build_model_fn, build_model_fn_args={}):

        dpath = dynamic_model_filepath
        if not os.path.exists(dpath):
            # Read the latest checkpoint by finding the most recent file
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_out_dir, "*.keras")))

            if checkpoints:
                dpath = checkpoints[-1]
                 
                print("Loading model from checkpoint:", dpath)
                static_model = tf.keras.models.load_model(
                    dpath,
                    compile=False
                )
                
                dynamic_model = build_model_fn(**build_model_fn_args, dynamic_time=True)
                dynamic_model.set_weights(static_model.get_weights())
                
            else:
                dpath = None

            if not dpath:
                print("Dynamic model not found. run: python piezo2mic.py --train-teacher and/or python piezo2mic.py --train-student")
                sys.exit(1)
        else:
             print("Loading dynamic model from:", dpath)
             dynamic_model = tf.keras.models.load_model(
                dpath,
                compile=False
            )

        generate_demo(model=dynamic_model, out_dir=demo_out_dir)

def main ():
    if len(sys.argv)>1 and sys.argv[1]=="--train-teacher":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_TEACHER_OUT_DIR,
            csv_logs_out_dir=CSV_LOGS_TEACHER_OUT_DIR,
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
            csv_logs_out_dir=CSV_LOGS_STUDENT_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_STUDENT_OUT_DIR,
            demo_generator_outputs_out_dir=DEMO_GENERATOR_OUTPUTS_OUT_DIR
        )

        if resume:
            train_student(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_student()
    elif (len(sys.argv)>1 and sys.argv[1]=="--demo"):
        
        print("Generating demo audio files...")
        
        if (len(sys.argv) <= 2 or sys.argv[2] !="--student-only"):
            prepare_and_generate_demo(
                demo_out_dir=LONG_DEMOS_TEACHER_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
                build_model_fn=build_teacher_model,
            )
            
        if (len(sys.argv) <= 2 or sys.argv[2] !="--teacher-only"):
            teacher_dynamic_model_filepath = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME)
            if not os.path.exists(teacher_dynamic_model_filepath):
                print("Teacher model not found, run --train-teacher first")
                sys.exit(1)
            teacher = tf.keras.models.load_model(teacher_dynamic_model_filepath, compile=False)

            prepare_and_generate_demo(
                demo_out_dir=LONG_DEMOS_STUDENT_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_DYNAMIC_MODEL_FILENAME),
                checkpoint_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
                build_model_fn=build_student_model,
                build_model_fn_args={
                    'teacher': teacher
                }
            )
    elif len(sys.argv) > 1 and sys.argv[1]=="--dump-pick-attacks":
        dump_pick_attacks()
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1]=="--dump-finger-attacks":
        dump_finger_attacks()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        teacher_dyn, student_dyn = fine_tune()
        generate_fine_tune_demo_from_models(teacher_dyn=teacher_dyn, student_dyn=student_dyn)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune-demo":
        
        teacher_dyn_path = os.path.join(MODELS_TEACHER_OUT_DIR, FINETUNE_TEACHER_DYNAMIC_MODEL_FILENAME)
        student_dyn_path = os.path.join(MODELS_STUDENT_OUT_DIR, FINETUNE_STUDENT_DYNAMIC_MODEL_FILENAME)

        if not os.path.exists(teacher_dyn_path) and not os.path.exists(student_dyn_path):
            print("Fine-tuned teacher/student dynamic models not found, run --fine-tune first")
            sys.exit(1)
        
        teacher_dyn = None if not os.path.exists(teacher_dyn_path) else tf.keras.models.load_model(teacher_dyn_path, compile=False)
        student_dyn = None if not os.path.exists(student_dyn_path) else tf.keras.models.load_model(student_dyn_path, compile=False)
        
        generate_fine_tune_demo_from_models(teacher_dyn=teacher_dyn, student_dyn=student_dyn)
        sys.exit(0)
    
    else:
        print("No command given.")
        sys.exit(1)
    

# Main
if __name__=="__main__":
    main()