#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal.
"""
import os
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = "15192"

import glob
import random
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, Add, UpSampling1D, Lambda, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import Constraint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Cropping1D, SeparableConv1D

# ——— compute receptive fields from (kernel, dilations) ——————————————————
def compute_rf(kernel_size, dilations):
    # standard 1D conv‐stack RF = 1 + (kernel_size–1) * sum(dilations)
    return 1 + (kernel_size - 1) * sum(dilations)


# ───────────────────────────── PARAMETERS ─────────────────────────────────────
DECIM_FACTOR     = 8     # downsample 48 k → 6 k

# ————————————————————————————————————————————————————————————
# DESIGN A LINEAR-PHASE LOW-PASS FOR DECIMATION/INTERP
# ————————————————————————————————————————————————————————————
# cutoff at 1/(2*DECIM_FACTOR) of original Nyquist
f_c = 1.0 / (2 * DECIM_FACTOR)
# number of FIR taps (odd → exact linear phase)
DECIM_TAPS = 63
n = np.arange(DECIM_TAPS, dtype=np.float32)
n0 = (DECIM_TAPS - 1) / 2.0
# ideal sinc
h_lp = np.sinc(2 * f_c * (n - n0))
# scale by cutoff
h_lp *= 2 * f_c
# window to control sidelobes
h_lp *= np.hamming(DECIM_TAPS)
# normalize DC gain = 1
h_lp /= np.sum(h_lp)
# reshape for Conv1D kernel: (kernel_size, in_channels, out_channels)
h_lp = h_lp.reshape((DECIM_TAPS, 1, 1)).astype(np.float32)
# ————————————————————————————————————————————————————————————

# — fine branch @48 kHz (∼7.5 ms RF)
FINE_FILTERS     = 64
FINE_KERNEL      = 25
FINE_DILATIONS   = (1, 2, 4, 8)

# — coarse branch @6 kHz (∼85 ms RF in decimated domain)
COARSE_FILTERS   = 16
COARSE_KERNEL    = 3
COARSE_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)

# — tiny HF‐bypass FIR
HF_KERNEL        = 31

def compute_rf(kernel, dilations):
    return 1 + (kernel - 1) * sum(dilations)

RF_FINE       = compute_rf(FINE_KERNEL,   FINE_DILATIONS)   # 361 samples
RF_COARSE_DEC = compute_rf(COARSE_KERNEL, COARSE_DILATIONS) # 511 decimated
RF_COARSE     = DECIM_FACTOR * RF_COARSE_DEC + (3 - 1)      # + (smoothing_kernel−1)
RECEPTIVE_FIELD = max(RF_FINE, RF_COARSE, HF_KERNEL)        # = 4090 samples

TRAIN_BLOCK_SIZE = 4096
SR               = 48000
SEG_LEN          = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE     # = 4090 + 4096 = 8186
BATCH_SIZE       = 32

# how many “extra” samples the upsampled coarse branch emits?
EXTRA_COARSE = ((SEG_LEN + DECIM_FACTOR - 1)//DECIM_FACTOR)*DECIM_FACTOR - SEG_LEN
# for SEG_LEN=8186, DECIM_FACTOR=8 → EXTRA_COARSE = 6

PHASE_1_EPOCHS   = 3
PHASE_2_EPOCHS   = 50
PATIENCE         = 5
SHUFFLE_BUF      = 1000
CHUNK_SEC        = 15

SRC_DIR             = "./input/source"
TGT_DIR             = "./input/target"
OUT_DIR             = "./out-alt"

DEMO_SEC            = 60 * 2
DEMO_OFFSET_SEC     = 0
DEMO_PAIR_INDEX     = 0

STATIC_MODEL_FILENAME  = "piezo2mic_multirate_small_static.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_multirate_small_dynamic.keras"
SAVE_WEIGHTS = False

os.makedirs(OUT_DIR, exist_ok=True)

phase1_weights = dict(
    w_l1   = 0.07,   # absolute waveform
    w_stft = 0.10,   # multi-res STFT
    w_fw   = 0.07,   # freq-weighted STFT
    w_grad = 0.10,   # slope loss
    w_pre  = 0.10,   # pre-emphasis
    w_mel  = 0.06,   # mel spectrogram
    w_hf   = 0.25,   # 3–10 kHz band
    w_tr   = 0.25    # NEW: transient (fast-attack) loss
)
phase2_weights = dict(
    w_l1   = 0.15,
    w_stft = 0.15,
    w_fw   = 0.10,
    w_grad = 0.15,
    w_pre  = 0.15,
    w_mel  = 0.15,
    w_hf   = 0.10,   # still some HF, but lower
    w_tr   = 0.05    # small transient term
)

# ────────────────────────────── DATA LOADING ───────────────────────────────────
print("Looking for audio pairs…")
srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*_source.wav")))
pairs = []
for s in srcs:
    t = s.replace("_source.wav","_target.wav").replace(SRC_DIR, TGT_DIR)
    if os.path.exists(t):
        pairs.append((s,t))
if not pairs:
    print("Error: no _source/_target pairs found"); sys.exit(1)
print(f"Found {len(pairs)} pairs.")

@register_keras_serializable()
class ResamplePoly(Layer):
    """A small wrapper around tf.signal.resample_poly that is properly serializable."""
    def __init__(self, up=1, down=1, **kwargs):
        super().__init__(**kwargs)
        self.up = up
        self.down = down

    def call(self, inputs):
        # inputs: (batch, time, channels)
        return tf.signal.resample_poly(inputs,
                                       up=self.up,
                                       down=self.down,
                                       axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"up": self.up, "down": self.down})
        return cfg

@register_keras_serializable()
class DynamicCrop(tf.keras.layers.Layer):
    def call(self, inputs):
        coarse, piezo = inputs
        # chop off exactly the extra samples
        return coarse[:, : tf.shape(piezo)[1], :]
    def get_config(self):
        # no extra config needed
        return super().get_config()

@register_keras_serializable()
def transient_loss(y_true, y_pred,
                  threshold=1e-3, slope=2000.0):
    # 1D first difference
    dy_t = y_true[:,1:,:] - y_true[:,:-1,:]
    dy_p = y_pred[:,1:,:] - y_pred[:,:-1,:]
    # soft mask where |dy_t| > threshold
    mask = tf.sigmoid((tf.abs(dy_t) - threshold) * slope)
    err  = tf.abs(dy_t - dy_p) * mask
    # pad back to full length
    err = tf.pad(err, [[0,0],[1,0],[0,0]])
    return tf.reduce_mean(err)

@register_keras_serializable()
class AbsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)
    def get_config(self):
        return super().get_config()

# ——— linear ramp & scale for your gate — no lambda ———
@register_keras_serializable()
class EnvScaleLayer(tf.keras.layers.Layer):
    def __init__(self, shift=1e-3, slope=200.0, **kw):
        super().__init__(**kw)
        self.shift = shift
        self.slope = slope
    def call(self, x):
        return (x - self.shift) * self.slope
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'shift':self.shift, 'slope':self.slope})
        return cfg

@register_keras_serializable()
class Diff1D(tf.keras.layers.Layer):
    def call(self, x):
        d = x[:,1:,:] - x[:,:-1,:]
        # pad a zero at t=0 so we keep the same length
        return tf.pad(d, [[0,0],[1,0],[0,0]])
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@register_keras_serializable()
def stft_metric(y_true, y_pred):
    return multi_res_stft_loss(y_true, y_pred)

@register_keras_serializable()
def fw_metric(y_true, y_pred):
    return freq_weighted_stft_loss(y_true, y_pred)

@register_keras_serializable()
def grad_metric(y_true, y_pred):
    return gradient_loss(y_true, y_pred)

@register_keras_serializable()
def pre_metric(y_true, y_pred):
    return pre_emphasis_loss(y_true, y_pred)

@register_keras_serializable()
def mel_metric(y_true, y_pred):
    return mel_loss(y_true, y_pred)

@register_keras_serializable()
def hf_metric(y_true, y_pred):
    # exactly your 3–10 kHz band penaliser
    return band_limited_stft_loss(y_true, y_pred,
                                  fmin=3000.0, fmax=10000.0,
                                  n_fft=512, hop=128)

@register_keras_serializable()
def slice_content(x, filters):
    # returns the first `filters` channels
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    # returns the remaining channels
    return x[:, :, filters:]

# 1) Pre-emphasis and its L1 loss
def pre_emphasis(x, μ=0.99):
    # x: (batch, time, 1)
    x0 = x[:, :1, :]                             # first sample
    x1 = x[:, 1:, :] - μ * x[:, :-1, :]          # high-pass residual
    return tf.concat([x0, x1], axis=1)

def pre_emphasis_loss(y_true, y_pred, μ=0.99):
    yt = pre_emphasis(y_true, μ)
    yp = pre_emphasis(y_pred, μ)
    return tf.reduce_mean(tf.abs(yt - yp))

def band_limited_stft_loss(y_true, y_pred,
                           fmin=3000.0, fmax=10000.0,
                           n_fft=512, hop=128):
    # y_true, y_pred: (B, T, 1)
    yt = tf.squeeze(y_true, -1)   # (B, T)
    yp = tf.squeeze(y_pred, -1)
    # compute a single-resolution STFT
    S_t = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    S_p = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    mag_t = tf.abs(S_t)
    mag_p = tf.abs(S_p)

    # build a binary mask over freq‐bins
    freq_bins = tf.shape(mag_t)[-1]
    freqs = tf.linspace(0.0, SR/2, freq_bins)   # SR is your global sample­rate
    band_mask = tf.logical_and(freqs >= fmin, freqs <= fmax)
    band_mask = tf.cast(band_mask, mag_t.dtype)  # shape (freq_bins,)
    band_mask = band_mask[tf.newaxis, tf.newaxis, :]  # (1,1,F)

    # L1 on magnitudes *only in that band*
    diff = tf.abs(mag_t - mag_p) * band_mask
    return tf.reduce_mean(diff)

# 2) Frequency-weighted magnitude STFT loss
def freq_weighted_stft_loss(y_true, y_pred, α=1.5):
    yt = tf.squeeze(y_true, -1)   # (B, T)
    yp = tf.squeeze(y_pred, -1)
    total = 0.0
    specs = [(512,128), (1024,256), (2048,512)]
    for n_fft, hop in specs:
        S_t = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_p = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_t)
        mag_p = tf.abs(S_p)
        diff = (tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2

        # build frequency-bin weights (shape 1×1×freq_bins)
        freq_bins = tf.shape(mag_t)[-1]
        freqs = tf.linspace(0.0, SR/2, freq_bins)
        w = tf.reshape((freqs/(SR/2))**α, [1,1,-1])

        total += tf.reduce_mean(diff * w)

    return total / float(len(specs))


# 3) Multi-resolution STFT (log-mag + complex) loss
def multi_res_stft_loss(y_true, y_pred):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    
    specs = [
      (  64,  16),   # 1.3 ms frames
      ( 128,  32),   # 2.7 ms frames
      ( 256,  64),   # 5.3 ms frames
      ( 512, 128),   # 10.7 ms frames
      (1024, 256),   # 21.3 ms frames
      (2048, 512),   # 42.7 ms frames
    ]
    
    mag_loss = 0.0
    comp_loss = 0.0
    for n_fft, hop in specs:
        S_t = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_p = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_t)
        mag_p = tf.abs(S_p)
        mag_loss  += tf.reduce_mean((tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2)
        comp_loss += tf.reduce_mean(
            (tf.math.real(S_t) - tf.math.real(S_p))**2 +
            (tf.math.imag(S_t) - tf.math.imag(S_p))**2
        )
    N = float(len(specs))
    return (mag_loss + comp_loss) / N


# 4) Gradient (slope) loss
def gradient_loss(y_true, y_pred):
    dy_true = y_true[:,1:,:] - y_true[:,:-1,:]
    dy_pred = y_pred[:,1:,:] - y_pred[:,:-1,:]
    return tf.reduce_mean(tf.abs(dy_true - dy_pred))


# 5) Mel-spectrogram L1 loss
def mel_loss(y_true, y_pred, n_mels=64, n_fft=1024, hop=256):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    # compute linear-freq spectrograms
    S_t = tf.abs(tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft))
    S_p = tf.abs(tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft))

    # build mel weight matrix
    num_spec_bins = n_fft//2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=SR,
        lower_edge_hertz=50.0,
        upper_edge_hertz=SR/2
    )

    # project to mel
    M_t = tf.tensordot(S_t, mel_w, axes=[[2],[0]])  # shape (B, frames, n_mels)
    M_p = tf.tensordot(S_p, mel_w, axes=[[2],[0]])
    return tf.reduce_mean(tf.abs(tf.math.log1p(M_t) - tf.math.log1p(M_p)))

# ——— factory that captures your weight‐set in a closure ———
def make_combined_loss(w_l1, w_stft, w_fw, w_grad, w_pre, w_mel, w_hf, w_tr):
    @register_keras_serializable()
    def _loss(y_true, y_pred):
        l1   = tf.reduce_mean(tf.abs(y_true - y_pred))
        stft = multi_res_stft_loss(y_true, y_pred)
        fw   = freq_weighted_stft_loss(y_true, y_pred)
        grad = gradient_loss(y_true, y_pred)
        pre  = pre_emphasis_loss(y_true, y_pred)
        mel  = mel_loss(y_true, y_pred)
        hf   = band_limited_stft_loss(y_true, y_pred,
                                      fmin=3000.0, fmax=10000.0,
                                      n_fft=512, hop=128)
        tr   = transient_loss(y_true, y_pred)

        return (
          w_l1   * l1   +
          w_stft * stft +
          w_fw   * fw   +
          w_grad * grad +
          w_pre  * pre  +
          w_mel  * mel  +
          w_hf   * hf   +
          w_tr   * tr
        )
    return _loss

# ────────────────────────── MODEL ARCHITECTURE ─────────────────────────────────

def build_multirate_resnet(dynamic_time: bool=False):
    T = None if dynamic_time else SEG_LEN
    piezo_in = Input(shape=(T,1), name="piezo_in")

    # ─── fine branch @48 kHz ─────────────────────────────────────
    x = Conv1D(FINE_FILTERS, 1, padding='same', name="fine_init")(piezo_in)
    for i, d in enumerate(FINE_DILATIONS):
        res = x
        pad = (FINE_KERNEL - 1) * d
        x   = ZeroPadding1D((pad,0), name=f"fine_pad_{i}")(x)
        x   = SeparableConv1D(
                 FINE_FILTERS,
                 FINE_KERNEL,
                 dilation_rate=d,
                 padding='valid',
                 activation='relu',
                 name=f"fine_sep_{i}"
             )(x)
        x   = Add(name=f"fine_res_{i}")([x, res])
    # final 1×1 to get single‐channel
    fine_out = Conv1D(1, 1, padding='same', name="fine_out")(x)

    # ─── coarse branch @6 kHz with true FIR polyphase ─────────────────
    # 1) DECIMATE by 8 using our FIR
    decim = Conv1D(
        filters=1,
        kernel_size=DECIM_TAPS,
        strides=DECIM_FACTOR,
        padding='same',         # zero‐phase align
        use_bias=False,
        trainable=False,
        kernel_initializer=Constant(h_lp),
        name="coarse_decim_fir"
    )(piezo_in)

    # 2) small TCN @6 kHz (exactly as your existing code)
    y = Conv1D(COARSE_FILTERS, 1, padding='same', name="coarse_init")(decim)
    for i, d in enumerate(COARSE_DILATIONS):
        res = y
        pad = (COARSE_KERNEL - 1) * d
        y   = ZeroPadding1D((pad,0), name=f"coarse_pad_{i}")(y)
        y   = SeparableConv1D(
                 COARSE_FILTERS,
                 COARSE_KERNEL,
                 dilation_rate=d,
                 padding='valid',
                 activation='relu',
                 name=f"coarse_sep_{i}"
             )(y)
        y   = Add(name=f"coarse_res_{i}")([y, res])
    y = Conv1D(1,1,padding='same',name="coarse_out")(y)

    # 3) UPSAMPLE + RECONSTRUCT with same FIR
    y = UpSampling1D(DECIM_FACTOR, name="coarse_upsample_nn")(y)
    y = Conv1D(
        filters=1,
        kernel_size=DECIM_TAPS,
        padding='same',
        use_bias=False,
        trainable=False,
        kernel_initializer=Constant(h_lp),
        name="coarse_interp_fir"
    )(y)

    # 4) crop the tiny extra tail so lengths match
    if not dynamic_time:
        if EXTRA_COARSE > 0:
            y = Cropping1D((0, EXTRA_COARSE), name="coarse_crop")(y)
    else:
        y = DynamicCrop(name="coarse_crop_dynamic")([y, piezo_in])

    # ─── MERGE + tiny HF bypass ──────────────────────────────────────
    merged = Add(name="merge")([fine_out, y])
    hf = Conv1D(
        filters=1,
        kernel_size=HF_KERNEL,
        padding='causal',
        use_bias=False,
        kernel_initializer='zeros',
        name='hf_transient'
    )(piezo_in)
    out = Add(name="output")([merged, hf])

    return Model(piezo_in, out, name="piezo2mic_multirate_small")

# ─────────────────────────────── TRAINING ──────────────────────────────────────
def train():
        # ────────────────────────── CHUNK‐BASED SPLIT ──────────────────────────────
    chunk_len = CHUNK_SEC * SR
    chunks = []
    for src_path, tgt_path in pairs:
        info     = sf.info(src_path)
        total    = info.frames
        n_full   = total // chunk_len

        # full 15s chunks
        for i in range(n_full):
            start = i * chunk_len
            chunks.append((src_path, tgt_path, start, chunk_len))

        # optional last partial chunk (only if it's at least one training window long)
        rem = total - n_full * chunk_len
        if rem >= SEG_LEN:
            start = n_full * chunk_len
            chunks.append((src_path, tgt_path, start, rem))

    print(f"Generated {len(chunks)} chunks from {len(pairs)} files")

    idx_chunks = list(range(len(chunks)))
    if len(chunks) >= 2:
        tr_i, va_i = train_test_split(idx_chunks,
                                      test_size=0.1,
                                      random_state=42)
    else:
        tr_i = va_i = idx_chunks

    train_chunks = [chunks[i] for i in tr_i]
    val_chunks   = [chunks[i] for i in va_i]

    # durations in frames of each chunk, used to compute approx steps
    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur for (_,_,_,dur) in val_chunks]

    def approx_steps(durs):
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # data generator (no per-segment mean removal)
    def segment_generator(chunk_list):
        while True:
            # pick one 30 s chunk
            src_path, tgt_path, chunk_start, chunk_dur = random.choice(chunk_list)

            # pick a random SEG_LEN‐long window inside that chunk
            max_offset = max(chunk_dur - SEG_LEN, 1)
            offset     = random.randrange(max_offset)
            read_start = chunk_start + offset

            x, _ = sf.read(src_path,
                           start=read_start,
                           frames=SEG_LEN,
                           dtype='float32',
                           always_2d=True)
            y, _ = sf.read(tgt_path,
                           start=read_start,
                           frames=SEG_LEN,
                           dtype='float32',
                           always_2d=True)

            # pad if we hit EOF
            if x.shape[0] < SEG_LEN:
                pad = SEG_LEN - x.shape[0]
                x = np.pad(x, ((0,pad),(0,0)))
                y = np.pad(y, ((0,pad),(0,0)))

            yield x, y

    sig = tf.TensorSpec((SEG_LEN,1), tf.float32)

    train_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_chunks),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_chunks),
        output_signature=(sig, sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=PATIENCE,
                                          restore_best_weights=True)

    # 1) train FP32 static
    print("\n≈ Training FP32 model")
    static_model = build_multirate_resnet(dynamic_time=False)

    # ——— PHASE 1: high‐hf for first 3 epochs ———
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=make_combined_loss(**phase1_weights),
        metrics=[ 'mae',
                l1_metric,
                stft_metric,
                fw_metric,
                grad_metric,
                pre_metric,
                mel_metric,
                hf_metric ]
    )
    
    static_model.summary()
    
    static_model.fit(
        train_ds,
        epochs=PHASE_1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[ es, TensorBoard(log_dir="./logs") ]
    )

    # ——— PHASE 2: flatter weighting for the rest ———
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),   # you might also drop LR
        loss=make_combined_loss(**phase2_weights),
        metrics=[ 'mae',
                l1_metric,
                stft_metric,
                fw_metric,
                grad_metric,
                pre_metric,
                mel_metric,
                hf_metric ]
    )
    static_model.fit(
        train_ds,
        initial_epoch=PHASE_1_EPOCHS,
        epochs=PHASE_2_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[ es, TensorBoard(log_dir="./tb") ]
    )

    path_static = os.path.join(OUT_DIR, STATIC_MODEL_FILENAME)
    static_model.save(path_static)
    print("✅ Saved FP32 static model:", path_static)

    # 2) build + save FP32 dynamic
    print("\n≈ Building + saving FP32 dynamic-time model")
    dynamic_model = build_multirate_resnet(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)
    
    if (SAVE_WEIGHTS):
        # Todo
        print("todo")

    return static_model, dynamic_model

# ─────────────────────────────── MAIN / DEMO ──────────────────────────────────
if __name__=="__main__":   
    if len(sys.argv)>1 and sys.argv[1]=="--train":
        static_model, dynamic_model = train()
    else:
        dpath = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(dpath):
            print("Dynamic model not found. run: python piezo2mic.py --train")
            sys.exit(1)
        dynamic_model = tf.keras.models.load_model(
                dpath,
                compile=False,
                custom_objects={
                     'AbsLayer': AbsLayer,
                     'Diff1D': Diff1D,
                     'EnvScaleLayer': EnvScaleLayer,
                     'slice_content': slice_content,
                     'slice_gate':    slice_gate
        })

    # DEMO
    print("\nGenerating demo…")
    FRAMES      = DEMO_SEC * SR
    START_FRAME = DEMO_OFFSET_SEC * SR
    pth_src, pth_tgt = pairs[DEMO_PAIR_INDEX]

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

    x_in = x_long[np.newaxis,...]
    print("Running inference…")
    y_pred_dyn = dynamic_model.predict(x_in, batch_size=1)[0,:,0]

    sf.write(os.path.join(OUT_DIR,"demo_input_piezo.wav"),
             x_long, SR, subtype='FLOAT')
    sf.write(os.path.join(OUT_DIR,"demo_reference_mic.wav"),
             y_ref, SR, subtype='FLOAT')
    sf.write(os.path.join(OUT_DIR,"demo_output_pred_fp32.wav"),
             y_pred_dyn, SR, subtype='FLOAT')
    print("✅ Demo files in", OUT_DIR)