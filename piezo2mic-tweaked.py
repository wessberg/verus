#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal, with pick‐attack oversampling ramp,
and a 2nd “style” channel so one model can handle finger vs pick.
It is designed to be ultra-low latency for real-time inference for live-use.
This is why dilations and kernel_size are small, and the model is causal.
"""
import os, glob, random, shutil, sys, re
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import butter, sosfilt, find_peaks, hilbert
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, Callback, ReduceLROnPlateau

from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, ZeroPadding1D,
    Activation, Multiply, Add, Lambda, Dense, Layer, UpSampling1D
)

# Try to enforce locked seeds so we can more reliably change config options and trust that the change in results was an effect of the change.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'      # try to force TF→deterministic kernels
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'     # on some TF versions also needed

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Audio channel configuration
SR                  = 48000 

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

TRANSIENT_UPWEIGHT_RATIO  = 0.05   # how strongly to up-weight the ∣Δx∣ spikes
HP_PRE_EMPHASIS           = 0.8 # coefficient for 1-sample 1st-order high-pass

L1_WEIGHT                 = 1.0 # waveform l1 loss
HF_WEIGHT                 = 1.0  # scale of the high-pass loss
STFT_WEIGHT               = 1.0    # weight of the spectral loss
MEL_WEIGHT                = 1.0   # weight of the log‐mel‐spectrogram L1 loss
RMS_ENV_WEIGHT            = 1.0   # weight of the RMS‐envelope L1 loss

PICK_ATTACK_CHUNK_RATIO = 0.15 # How many pick attack segments the model will see
PICK_UNIFORM_CHUNK_RATIO = 0.35 # How many general pick segments the model will see

FINGER_ATTACK_CHUNK_RATIO = 0.15 # How many finger attack segments the model will see
FINGER_UNIFORM_CHUNK_RATIO = 0.35 # How many general finger segments the model will see

# ──────── TUNING KNOBS ─────────────────────────────────────────────────────────
ATTACK_SEGMENT_GENERATOR_MIN_DIFF_THRESH = 1e-3    # absolute floor on |Δ| spikes
ATTACK_SEGMENT_GENERATOR_MIN_ENV_THRESH  = 1e-3    # absolute floor on HF envelope
ATTACK_SEGMENT_GENERATOR_MIN_AMP_THRESH  = 1e-3    # absolute floor on the piezo signal itself

# Pick-attack band pass filters
SOS_HF = butter(4, [6000, 20000], fs=SR, btype='bandpass', output='sos')
SOS_LF = butter(4, [300, 2000],  fs=SR, btype='bandpass', output='sos')
#───────────────────────────────────────────────────────────────────────────────

# ─── IR branch FIR FILTER DESIGN ────────────────────────────────────────────
IR_DOWNSAMPLE_FACTOR = 8
IR_FIR_SIZE          = 65    # odd for a symmetric window
IR_CUTOFF            = 1.0 / IR_DOWNSAMPLE_FACTOR  # normalized cutoff (Nyquist=0.5→fc at 0.5/D)

EPOCHS      = 1000
LR          = 1e-4
LR_FACTOR  = 0.5
MIN_LR     = 1e-6
LR_PATIENCE = 8
LR_PLATEAU_PATIENCE = 5

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

PICK_SRC_DIR   = "./input/picked/source"
PICK_TGT_DIR   = "./input/picked/target"
FINGER_SRC_DIR = "./input/fingerpicked/source"
FINGER_TGT_DIR = "./input/fingerpicked/target"

OUT_DIR        = "./out-tweaked"

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

DEMO_GENERATOR_OUTPUTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "generator-outputs")

DEMO_CHECKPOINTS_OUT_DIR = os.path.join(DEMOS_OUT_DIR, "checkpoints")
DEMO_CHECKPOINTS_TEACHER_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "teacher")
DEMO_CHECKPOINTS_STUDENT_OUT_DIR = os.path.join(DEMO_CHECKPOINTS_OUT_DIR, "student")

DEMO_SEC        = 10
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

LONG_DEMO_SEC        = 60
LONG_DEMO_OFFSET_SEC = 60 * 5
LONG_DEMO_PAIR_INDEX = 6


SAVE_WEIGHTS           = False  # set to True to dump .npy weights

for dir_path in [DEMO_GENERATOR_OUTPUTS_OUT_DIR, CHECKPOINT_TEACHER_OUT_DIR, CHECKPOINT_STUDENT_OUT_DIR, LOGS_TEACHER_OUT_DIR, LOGS_STUDENT_OUT_DIR, DEMO_CHECKPOINTS_TEACHER_OUT_DIR, DEMO_CHECKPOINTS_STUDENT_OUT_DIR, LONG_DEMOS_TEACHER_OUT_DIR, LONG_DEMOS_STUDENT_OUT_DIR, MODELS_TEACHER_OUT_DIR, MODELS_STUDENT_OUT_DIR, CSV_LOGS_TEACHER_OUT_DIR, CSV_LOGS_STUDENT_OUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ────────────────────────────── DATA LOADING ───────────────────────────────────
print("Looking for audio pairs…")
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
print(f"  → {len(picked_pairs)} picked‐attack pairs, {len(finger_pairs)} finger‐picked pairs")
all_pairs = picked_pairs + finger_pairs

audio_cache = {}
for (src,tgt) in all_pairs:
    audio_cache[src] = sf.read(src, dtype='float32', always_2d=True)[0]
    audio_cache[tgt] = sf.read(tgt, dtype='float32', always_2d=True)[0]

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

        # 2) tack on the style channel (0 or 1) exactly as in your generators
        style_flag = 1.0 if self.src_path.startswith(PICK_SRC_DIR) else 0.0
        style_chan = np.full((x_long.shape[0],1), style_flag, dtype='float32')
        x_demo = np.concatenate([x_long, style_chan], axis=-1)[np.newaxis, ...]  # (1, T, 2)

        # 3) predict
        y_pred = self.dynamic_model.predict(x_demo, batch_size=1)[0,:,0]  # (T,)

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

    gen      = attack_segment_generator(seg_len=seg_len, chunks=pick_chunks, is_pick_chunks=True)
    collected_piezo = []
    collected_mic   = []

    total = n_segments or len(pick_chunks)

    for i in range(total):
        x, y = next(gen)
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
    gen      = attack_segment_generator(seg_len=seg_len, chunks=finger_chunks, is_pick_chunks=False)
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


def attack_segment_generator(seg_len, chunks, is_pick_chunks, context_pre=1024):
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        mono = audio_cache[src][:,0]
        window = mono[cstart : cstart + cdur]

        spikes = find_attack_spikes(window)
        if spikes.size == 0:
            continue

        idx   = int(random.choice(spikes))
        start = idx - context_pre
        start = np.clip(start, 0, cdur - seg_len)
        rs    = cstart + start

        # 3) slice out exactly SEG_LEN samples
        x = mono[rs: rs + seg_len]
        y = audio_cache[tgt][rs: rs + seg_len, 0]

        # 4) enforce a minimum waveform amplitude
        if np.max(np.abs(x)) < ATTACK_SEGMENT_GENERATOR_MIN_AMP_THRESH:
            # still too quiet → skip
            continue

        x = x.astype('float32')[:,None]
        style_flag = 1.0 if is_pick_chunks else 0.0
        style = np.full((seg_len,1), style_flag, dtype='float32')
        xx = np.concatenate([x, style], axis=-1)
        yy = y.astype('float32')[:,None]
        
        yield xx, yy

def segment_generator(seg_len, chunks, is_pick_chunks):
    """Uniform random windows, style=1.0 if pick-path else 0.0."""
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        src_wav = audio_cache[src]
        tgt_wav = audio_cache[tgt]

        off = random.randrange(max(cdur - seg_len, 1))
        read_start = cstart + off
        x = src_wav[read_start : read_start + seg_len]
        y = tgt_wav[read_start : read_start + seg_len]

        if x.shape[0] < seg_len:
            p = seg_len - x.shape[0]
            x = np.pad(x, ((0,p),(0,0)))
            y = np.pad(y, ((0,p),(0,0)))

        style_flag = 1.0 if is_pick_chunks else 0.0
        style = np.full((seg_len,1), style_flag, dtype='float32')
        x = np.concatenate([x, style], axis=-1)

        yield x, y
        
def analyze_loss_scales(model, dataset ):
    # Take one batch from your validation dataset
    (x, y, sw) = next(iter(dataset))
    
    y_pred = model(x, training=False)
    y_true = y

    l1_val = tf.reduce_mean(waveform_l1_loss(y_true, y_pred)).numpy()
    stft_val = tf.reduce_mean(multi_scale_stft_loss(y_true, y_pred)).numpy()
    mel_val = tf.reduce_mean(log_mel_loss(y_true, y_pred)).numpy()
    env_val = tf.reduce_mean(rms_env_loss(y_true, y_pred)).numpy()
    hf_val = tf.reduce_mean(hf_loss(y_true, y_pred)).numpy()

    print(f"Initial L1 Loss: {l1_val}")
    print(f"Initial STFT Loss: {stft_val}")
    print(f"Initial Mel Loss: {mel_val}")
    print(f"Initial RMS Env Loss: {env_val}")
    print(f"Initial HF Loss: {hf_val}")

    return {'l1': l1_val, 'stft': stft_val, 'mel': mel_val, 'env': env_val, 'hf': hf_val}
        
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


    # --- ADD STYLE CHANNEL TO LONG SIGNAL ---
    if pth_src.startswith(PICK_SRC_DIR):
        style = 1.0
    else:
        style = 0.0
    style_chan = np.full((FRAMES,1), style, dtype='float32')
    x_long2   = np.concatenate([x_long, style_chan], axis=-1)  # (FRAMES,2)

    x_in = x_long2[np.newaxis,...]  # (1,FRAMES,2)
    print("Running inference…")

    y_pred_dyn = model.predict(x_in, batch_size=1)[0,:,0]
    
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
    
def make_sinc_filter(cutoff, num_taps):
    t = np.arange(num_taps) - (num_taps - 1)/2.0
    h = np.sinc(2 * cutoff * t)
    w = np.hamming(num_taps)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float32)

_ir_fir     = make_sinc_filter(IR_CUTOFF, IR_FIR_SIZE)           # (65,)
_ir_fir_rev = _ir_fir[::-1].reshape(-1,1,1)                      # (65,1,1)
_ir_fir_fwd = _ir_fir.reshape(-1,1,1)                            # (65,1,1)
_ir_fir_tf  = tf.constant(_ir_fir_rev, dtype=tf.float32)
_ir_fir_up  = tf.constant(_ir_fir_fwd, dtype=tf.float32)


@register_keras_serializable()
class StuIrDownsampleFIR(Layer):
    """
    Causal FIR+decimate on the piezo channel.
    Input:  (batch, T, 2)
    Output: (batch, ceil(T/ factor), 1)
    """
    def __init__(self, factor=IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.kernel_size = IR_FIR_SIZE

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
        self.kernel_size = IR_FIR_SIZE

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
class StuIrTrimToInput(Layer):
    """
    Crop the IR branch to match main-branch length.
    inputs = [ir:(batch,≥T,1), main_in:(batch,T,2)]
    output = ir[:, :T, :]
    """
    def call(self, inputs):
        ir, main = inputs
        T = tf.shape(main)[1]
        return ir[:, :T, :]

    def compute_output_shape(self, input_shape):
        # input_shape = [(batch, Tir, 1),(batch, Tmain, 2)]
        b, Tmain, _ = input_shape[1]
        return (b, Tmain, input_shape[0][2])

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
    Per-sample, multi-scale L1 loss between |STFT(y_true)| and |STFT(y_pred)|.
    Returns shape (batch,1,1) so it broadcasts in your combined_loss.
    """
    scales = [
        (256,   64,  256),
        (512,  128,  512),
        (1024, 256, 1024),
        (2048, 512, 2048),
        (8192, 2048, 8192)
    ]

    # squeeze off the channel dim → (batch, T)
    y_true_s = tf.squeeze(y_true, axis=-1)
    y_pred_s = tf.squeeze(y_pred, axis=-1)

    per_scale_losses = []
    for n_fft, hop, win in scales:
        # 1) frame & pad
        frames_true = tf.signal.frame(y_true_s, frame_length=win,
                                      frame_step=hop, pad_end=True)
        frames_pred = tf.signal.frame(y_pred_s, frame_length=win,
                                      frame_step=hop, pad_end=True)

        # 2) window
        window = tf.signal.hann_window(win, periodic=True)        # (win,)
        window = tf.reshape(window, (1,1,win))                    # → (1,1,win)
        win_true = frames_true * window
        win_pred = frames_pred * window

        # 3) rFFT  ← here is the change:
        fft_len = tf.constant([n_fft], dtype=tf.int32)            # MUST be 1-D
        st_true = tf.signal.rfft(win_true,    fft_length=fft_len)
        st_pred = tf.signal.rfft(win_pred,    fft_length=fft_len)

        # 4) mag-L1 per frame, then avg over time & freq → (batch,)
        mag_true = tf.abs(st_true)
        mag_pred = tf.abs(st_pred)
        loss_sf  = tf.reduce_mean(
                       tf.abs(mag_true - mag_pred),
                       axis=[1,2]
                   )  # → (batch,)
        per_scale_losses.append(loss_sf)

    # stack over scales → mean → (batch,)
    loss_stack = tf.stack(per_scale_losses, axis=0)  # (n_scales, batch)
    loss_batch = tf.reduce_mean(loss_stack, axis=0)

    return loss_batch[:, None, None]

@register_keras_serializable()
def log_mel_loss(y_true, y_pred):
    """
    Compute L1 between log‐mel‐spectra of y_true and y_pred.
    Returns shape (batch,1,1).
    """
    # parameters
    n_fft   = 1024
    hop     = 256
    win     = 1024
    n_mels  = 64
    fmin    = 20.0
    fmax    = float(SR//2)

    # squeeze off the channel → (batch, T)
    t   = tf.squeeze(y_true, axis=-1)
    p   = tf.squeeze(y_pred, axis=-1)

    # STFT magnitudes
    st_t = tf.signal.stft(t,   frame_length=win, frame_step=hop,
                          fft_length=n_fft, pad_end=True)
    st_p = tf.signal.stft(p,   frame_length=win, frame_step=hop,
                          fft_length=n_fft, pad_end=True)
    mag_t = tf.abs(st_t)    # (batch, frames, freq_bins)
    mag_p = tf.abs(st_p)

    # build a (freq_bins × n_mels) mel‐weight matrix
    fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft//2+1,
        sample_rate=SR,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )  # shape (freq_bins, n_mels)

    # mel spectrograms → (batch, frames, n_mels)
    mel_t = tf.tensordot(mag_t, fb,   axes=[[2],[0]])
    mel_p = tf.tensordot(mag_p, fb,   axes=[[2],[0]])

    # log compress
    log_t = tf.math.log(mel_t + 1e-6)
    log_p = tf.math.log(mel_p + 1e-6)

    # L1 & mean over time+mel‐bins → (batch,)
    mel_diff = tf.reduce_mean(tf.abs(log_t - log_p), axis=[1,2])

    # shape to (batch,1,1)
    return mel_diff[:,None,None]


@register_keras_serializable()
def rms_env_loss(y_true, y_pred):
    """
    Compute L1 between slow RMS envelopes.
    Returns shape (batch,1,1).
    """
    # envelope params
    env_win  = 1024
    env_hop  = 512

    # squeeze → (batch, T)
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)

    # frame → (batch, frames, env_win)
    ft   = tf.signal.frame(t, frame_length=env_win,
                           frame_step=env_hop, pad_end=True)
    fp   = tf.signal.frame(p, frame_length=env_win,
                           frame_step=env_hop, pad_end=True)

    # RMS per frame → (batch, frames)
    rms_t = tf.sqrt(tf.reduce_mean(ft**2, axis=2) + 1e-12)
    rms_p = tf.sqrt(tf.reduce_mean(fp**2, axis=2) + 1e-12)

    # L1 & mean over frames → (batch,)
    env_diff = tf.reduce_mean(tf.abs(rms_t - rms_p), axis=1)

    return env_diff[:,None,None]

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

# ──────────────────────────────────────────────────────────────────────────────
@register_keras_serializable()
def add_sw(x, y):
    # x: (batch, T, 2) → x[...,0]=piezo, x[...,1]=style flag (0=finger,1=pick)
    piezo = x[...,0]            # (batch, T)
    style = x[...,1]            # (batch, T)

    # 1) extract one scalar per example in the batch
    #    (all time‐samples are identical so max or mean both work)
    style_flag = tf.reduce_max(style, axis=1)   # (batch,)

    # 2) define your base weights for each style
    #    here we set both to 1.0 (i.e. no style-bias), but you can
    #    make pick_weight != finger_weight if you like.
    pick_weight   = 1.0
    finger_weight = 1.0

    #    this yields a shape (batch,) where finger examples get
    #    finger_weight, pick examples get pick_weight
    base_scalar = pick_weight*style_flag + finger_weight*(1.0 - style_flag)
    #––– if you truly want them *exactly* equal, you can just do:
    #    base_scalar = tf.ones_like(style_flag)

    # 3) build your transient re-weighting term exactly as before
    Δ = tf.abs(piezo[:,1:] - piezo[:,:-1])        # (batch, T-1)
    Δ = tf.pad(Δ, [[0,0],[1,0]], constant_values=0)  # back to (batch, T)
    m = tf.reduce_max(Δ, axis=1, keepdims=True)   # (batch,1)
    m = tf.maximum(m, 1e-6)                       # avoid div0
    wn = 1.0 + TRANSIENT_UPWEIGHT_RATIO * (Δ / m)          # (batch, T)

    # 4) combine them
    sw = base_scalar[:, None] * wn                # (batch, T)
    sw = tf.expand_dims(sw, axis=-1)              # (batch, T, 1)

    return x, y, sw

# ─── helper functions ─────────────────────────────────────────────────────────

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
    # x shape: (batch, T, 2*filters) → return (batch, T, filters)
    return x[:, :, :filters]

@register_keras_serializable()
def slice_film_beta(x, filters):
    # x shape: (batch, T, 2*filters) → return (batch, T, filters)
    return x[:, :, filters:]

@register_keras_serializable()
def slice_style(x):
    # Takes channel 1 of a (batch, time, 2) tensor
    return x[:, :, 1:2]

@register_keras_serializable()
def invert(x):
    return 1.0 - x


def build_teacher_model(dynamic_time=False):
    T = None if dynamic_time else TEACHER_SEG_LEN
    inp = Input(shape=(T,2), name="piezo_plus_style")
    style = Lambda(slice_style, name="take_style")(inp)                   # (batch, T, 1)
    piezo = Lambda(slice_piezo, name="take_piezo")(inp)
    # derive scalar style flag
    style_scalar = Lambda(extract_style_scalar, name="style_scalar")(style)  # (batch,)
    # style embedding
    style_embed = Dense(TEACHER_EMBED_DIM, activation='linear', name="style_embed")(style_scalar)
    # initial 1x1 projection on both channels
    x = Conv1D(TEACHER_FILTERS, 1, padding='same', name="init_pw")(inp)

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
                    name=f"film_gamma_{idx}")(film_params)
        beta  = Lambda(slice_film_beta,
                    arguments={'filters': TEACHER_FILTERS},
                    name=f"film_beta_{idx}")(film_params)

        x = Add(name=f"film_apply_{idx}")([
            Multiply(name=f"film_scale_{idx}")([x, gamma]),
            beta
        ])

    main_pick   = Conv1D(1,1,padding='same',name="final_pick")(x)
    main_finger = Conv1D(1,1,padding='same',name="final_finger")(x)
    pick_c   = Multiply(name="pick_contrib")([style, main_pick])
    finger_c = Multiply(name="finger_contrib")([
        Lambda(invert, name="invert_style")(style), main_finger])
    mixed    = Add(name="mix_main")([pick_c, finger_c])
    out = mixed

    return tf.keras.models.Model(inp, out, name="piezo2mic_teacher")

@register_keras_serializable()
def build_student_model(
    teacher=None,
    dynamic_time=False
):
    # ── 1) Input ────────────────────────────────────────────────────────────
    if dynamic_time:
        inp = Input(batch_shape=(1, None, 2),
                    name="stu_piezo_plus_style")
    else:
        inp = Input(shape=(STUDENT_SEG_LEN, 2),
                    name="stu_piezo_plus_style")

    # ── 2) Style + Piezo and Style‐Embedding ───────────────────────────────
    style        = Lambda(slice_style,   name="stu_slice_style")(inp)       # (B,T,1)
    piezo        = Lambda(slice_piezo,   name="stu_slice_piezo")(inp)       # (B,T,1)
    style_scalar = Lambda(extract_style_scalar, name="stu_style_scalar")(style)  # (B,1,1)
    style_embed  = Dense(STUDENT_EMBED_DIM,
                         activation="linear",
                         name="stu_style_embed")(style_scalar)               # (B,EMB)

    # ── 3) Initial 1×1 projection ───────────────────────────────────────────
    x = Conv1D(STUDENT_FILTERS,
               kernel_size=1,
               padding="same",
               name="stu_init_pw")(inp)  # (B,T,F)

    # ── 4) Dilated TCN Blocks (gated SeparableConv + FiLM) ─────────────────
    student_feats = []
    for i, d in enumerate(STUDENT_DILATIONS):
        res = x
        pad = (STUDENT_KERNEL_SIZE - 1) * d
        x   = ZeroPadding1D((pad, 0), name=f"stu_pad_{i}")(x)
        g   = SeparableConv1D(2*STUDENT_FILTERS,
                              STUDENT_KERNEL_SIZE,
                              dilation_rate=d,
                              padding="valid",
                              name=f"stu_sep_{i}")(x)
        c   = Lambda(slice_content,
                     arguments={"filters": STUDENT_FILTERS},
                     name=f"stu_slice_c_{i}")(g)
        g2  = Lambda(slice_gate,
                     arguments={"filters": STUDENT_FILTERS},
                     name=f"stu_slice_g_{i}")(g)
        x   = Multiply(name=f"stu_gate_mul_{i}")([
                  Activation("tanh",    name=f"stu_tanh_{i}")(c),
                  Activation("sigmoid", name=f"stu_sigmoid_{i}")(g2)
              ])
        x   = Conv1D(STUDENT_FILTERS,
                     kernel_size=1,
                     padding="same",
                     name=f"stu_pw_{i}")(x)
        x   = Add(name=f"stu_res_add_{i}")([x, res])

        # FiLM
        film  = Dense(2*STUDENT_FILTERS,
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

        # collect this feature if we're hinting at it
        layer_name = f"res_add_{i}"
        if STUDENT_TEACHER_FEAT_LAYERS and layer_name in STUDENT_TEACHER_FEAT_LAYERS:
            student_feats.append(x)   # (B,T,STUDENT_FILTERS)

    # ── 6) Final Style-conditioned 1×1 pick/finger outputs ──────────────────
    out_pick   = Conv1D(1,1,padding="same", name="stu_final_pick")(x)
    out_finger = Conv1D(1,1,padding="same", name="stu_final_finger")(x)
    pick_c   = Multiply(name="stu_pick_contrib")([style, out_pick])
    finger_c = Multiply(name="stu_finger_contrib")([
                    Lambda(invert, name="stu_invert_style")(style),
                    out_finger
                ])
    y_student = Add(name="stu_out")([pick_c, finger_c])  # (B,T,1)

    # ── 7) Low-rate IR Branch ───────────────────────────────────────────────
    IR_KERNEL = 33
    IR_DILATIONS = (1,2,4,8,16,32)

    # ─── 7) PROPER FIR‐BASED IR BRANCH ──────────────────────────────────────

    # 7.1) anti‐alias & decimate
    ir = StuIrDownsampleFIR(name='stu_ir_downsample')(inp)    # → (batch,≈T/8,1)

    # 7.2) linear dilated conv stack (no ReLU)
    for i, d in enumerate(IR_DILATIONS):
        ir = ZeroPadding1D(((IR_KERNEL-1)*d,0),
                           name=f'stu_ir_pad_{i}')(ir)
        ir = SeparableConv1D(1, IR_KERNEL,
                             dilation_rate=d,
                             padding='valid',
                             name=f'stu_ir_sep_{i}')(ir)
        # <-- deliberately no Activation()

    # 7.3) zero‐order‐hold upsample
    ir = UpSampling1D(size=IR_DOWNSAMPLE_FACTOR,
                      name='stu_ir_upsample')(ir)

    # 7.4) FIR‐smooth the steps
    ir = StuIrPostLP(name='stu_ir_post_lp')(ir)

    # 7.5) trim to exactly the main‐branch length
    ir = StuIrTrimToInput(name='stu_ir_trim')([ir, inp])

    # 7.6) fuse back in
    y_student = Add(name='stu_with_ir')([y_student, ir])

    # ── 8) Distillation Losses (if teacher is provided) ──────────────────────
    if teacher is not None:
        teacher.trainable = False

        # a) intermediate-feature distillation
        if STUDENT_TEACHER_FEAT_LAYERS:
            teacher_feat_model = tf.keras.models.Model(
                inputs = teacher.input,
                outputs= [teacher.get_layer(n).output
                          for n in STUDENT_TEACHER_FEAT_LAYERS]
            )
            T_feats = teacher_feat_model(inp)
            if not isinstance(T_feats, list):
                T_feats = [T_feats]

            for idx, (T_f, S_f) in enumerate(zip(T_feats, student_feats)):
                chan = int(T_f.shape[-1])
                proj = Conv1D(chan, 1, padding="same",
                              name=f"stu_proj_feat_{idx}")(S_f)
                _ = DistillLoss(alpha=STUDENT_FEAT_DISTILL_ALPHA,
                                name=f"feat_distill_{idx}")([T_f, proj])

        # b) final‐waveform distillation
        Y_t = teacher(inp)  # (B,T,1)
        y_student = DistillLoss(alpha=STUDENT_DISTILL_ALPHA,
                                name="distillation")([Y_t, y_student])

    # ── 9) Final model ───────────────────────────────────────────────────────
    return tf.keras.models.Model(
        inputs = inp,
        outputs= y_student,
        name   ="piezo2mic_student"
    )

def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations, filters):
    # TODO
    print("TODO")




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
    

    def approx_steps(durs):
        return sum(max((d-seg_len)//(seg_len//2)+1,0) for d in durs)//BATCH_SIZE
    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur for (_,_,_,dur) in val_chunks]
    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # datasets
    sig_x = tf.TensorSpec((seg_len,2), tf.float32)
    sig_y = tf.TensorSpec((seg_len,1), tf.float32)

    pick_attack_ds    = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=train_pick_chunks, is_pick_chunks=True), output_signature=(sig_x,sig_y)
    )
    
    pick_uniform_ds    = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_pick_chunks, is_pick_chunks=True), output_signature=(sig_x,sig_y)
    )
    
    finger_attack_ds  = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=train_finger_chunks, is_pick_chunks=False), output_signature=(sig_x,sig_y)
    )
    
    finger_uniform_ds  = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=train_finger_chunks, is_pick_chunks=False), output_signature=(sig_x,sig_y)
    )

    train_ds = tf.data.Dataset.sample_from_datasets(
        [pick_attack_ds, pick_uniform_ds, finger_attack_ds, finger_uniform_ds], weights=[PICK_ATTACK_CHUNK_RATIO, PICK_UNIFORM_CHUNK_RATIO, FINGER_ATTACK_CHUNK_RATIO, FINGER_UNIFORM_CHUNK_RATIO]
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    train_ds = train_ds.map(add_sw).repeat().prefetch(tf.data.AUTOTUNE)

    # validation streams
    val_pick_attack_ds = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=val_pick_chunks, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    
    val_pick_uniform_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_pick_chunks, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    
    val_finger_attack_ds = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(seg_len=seg_len, chunks=val_finger_chunks, is_pick_chunks=False),
        output_signature=(sig_x, sig_y)
    )
    
    val_finger_uniform_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(seg_len=seg_len, chunks=val_finger_chunks, is_pick_chunks=False),
        output_signature=(sig_x, sig_y)
    )
    
    val_ds = tf.data.Dataset.sample_from_datasets(
        [val_pick_attack_ds, val_pick_uniform_ds, val_finger_attack_ds, val_finger_uniform_ds],
        weights=[PICK_ATTACK_CHUNK_RATIO, PICK_UNIFORM_CHUNK_RATIO, FINGER_ATTACK_CHUNK_RATIO, FINGER_UNIFORM_CHUNK_RATIO],
        seed=SEED
    ).batch(BATCH_SIZE)
    
    val_ds = val_ds.map(add_sw).repeat().prefetch(tf.data.AUTOTUNE)

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
    baseline = analyze_loss_scales(static_model, val_ds)
    
    ref_scale = float(baseline['l1'])
    user_w    = {
        'l1':  L1_WEIGHT,
        'stft': STFT_WEIGHT,
        'mel':  MEL_WEIGHT,
        'env':  RMS_ENV_WEIGHT,
        'hf':   HF_WEIGHT
    }

    new_w = {}
    for key, uw in user_w.items():
        bs = float(baseline[key])
        if bs > 0.0:
            new_w[key] = uw / bs * ref_scale
        else:
            new_w[key] = 0.0

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
    
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=combined_loss,
        metrics=[
                'mae',
                log_mel_loss,
                mean_hf_metric,
                multi_scale_stft_loss,
                rms_env_loss
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

    static_model.fit(
        train_ds,
        initial_epoch=initial_epoch,
        epochs=1 if DEBUG else EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[es, cp, reduce_lr, tb, demo_cb, csv_logger]
    )
    
    # trained_epochs = len(history.history["loss"])

    # Save FP32 static and dynamic models
    static_model.save(static_model_filepath)
    print("✅ Saved FP32 static model:", static_model_filepath)

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
    
    
    teacher_feature_model = tf.keras.Model(
        inputs=teacher.input,
        outputs=[
            teacher.get_layer("slice_content_2").output,   # e.g. after dilation=4
            teacher.get_layer("slice_content_4").output,   # e.g. after dilation=16
        ]
    )
    
    print("Loaded teacher from", teacher_dynamic_model_filepath)
    
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
        
        # Turn off GPU acceleration for inference, as it is more reliable
        tf.config.set_visible_devices([], 'GPU')
        
        teacher_dynamic_model_filepath = os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(teacher_dynamic_model_filepath):
            print("Teacher model not found, run --train-teacher first")
            sys.exit(1)
        teacher = tf.keras.models.load_model(teacher_dynamic_model_filepath, compile=False)
        
        print("Generating demo audio files...")

        prepare_and_generate_demo(
            demo_out_dir=LONG_DEMOS_TEACHER_OUT_DIR,
            dynamic_model_filepath=os.path.join(MODELS_TEACHER_OUT_DIR, TEACHER_DYNAMIC_MODEL_FILENAME),
            checkpoint_out_dir=CHECKPOINT_TEACHER_OUT_DIR,
            build_model_fn=build_teacher_model,
        )
        
        prepare_and_generate_demo(
            demo_out_dir=LONG_DEMOS_STUDENT_OUT_DIR,
            dynamic_model_filepath=os.path.join(MODELS_STUDENT_OUT_DIR, STUDENT_DYNAMIC_MODEL_FILENAME),
            checkpoint_out_dir=CHECKPOINT_STUDENT_OUT_DIR,
            build_model_fn=build_student_model,
            build_model_fn_args={
                'teacher': teacher
            }
        )
    elif len(sys.argv)>1 and sys.argv[1]=="--dump-pick-attacks":
        dump_pick_attacks()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--dump-finger-attacks":
        dump_finger_attacks()
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    

# Main
if __name__=="__main__":
    main()