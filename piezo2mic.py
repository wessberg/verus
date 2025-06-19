#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal.
"""
import os
import glob
import random
import sys
import re
import numpy as np
import soundfile as sf
import tensorflow as tf
from tflite import Model, TensorType, BuiltinOperator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

# ───────────────────────────── PARAMETERS ─────────────────────────────────────
FILTERS             = 64
KERNEL_SIZE         = 51
DILATIONS           = (1,2,4,8,16,32)
RECEPTIVE_FIELD     = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
TRAIN_BLOCK_SIZE    = 4096
SR                  = 48000
SEG_LEN             = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE
BATCH_SIZE          = 32
EPOCHS              = 1
PATIENCE            = 5
SHUFFLE_BUF         = 1000

SRC_DIR             = "./input/source"
TGT_DIR             = "./input/target"
OUT_DIR             = "./out"
DEMO_SEC            = 60
DEMO_OFFSET_SEC     = 0
DEMO_PAIR_INDEX     = 0

STATIC_MODEL_FILENAME  = "piezo2mic_resnet.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_resnet_dynamic.keras"
SAVE_WEIGHTS = True  # set to True to save layer weights as .npy files

os.makedirs(OUT_DIR, exist_ok=True)

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
def freq_weighted_stft_loss(y_true, y_pred, α=2.0):
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
    specs = [(256,64), (512,128), (1024,256), (2048,512)]
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


# 6) Combined loss with tuned weights
def combined_loss(y_true, y_pred):
    l1    = tf.reduce_mean(tf.abs(y_true - y_pred))
    stft  = multi_res_stft_loss(y_true, y_pred)
    fw    = freq_weighted_stft_loss(y_true, y_pred)
    grad  = gradient_loss(y_true, y_pred)
    pre   = pre_emphasis_loss(y_true, y_pred)
    mel = mel_loss(y_true, y_pred)
    hf    = band_limited_stft_loss(y_true, y_pred,
                                   fmin=3000.0, fmax=10000.0,
                                   n_fft=512, hop=128)

    # weights must sum to 1 (or you can renormalize)
    w_l1   = 0.10
    w_stft = 0.15
    w_fw   = 0.10
    w_grad = 0.15
    w_pre  = 0.15
    w_mel  = 0.10
    w_hf   = 0.25   # give the pick‐attack band a big say

    return (
        w_l1   * l1   +
        w_stft * stft +
        w_fw   * fw   +
        w_grad * grad +
        w_pre  * pre  +
        w_mel  * mel  +
        w_hf   * hf
    )

# ────────────────────────── MODEL ARCHITECTURE ─────────────────────────────────
import tensorflow as tf

def build_resnet_gated(filters=FILTERS,
                       kernel_size=KERNEL_SIZE,
                       dilations=DILATIONS,
                       dynamic_time=False):
    # time-dim None for dynamic, fixed = SEG_LEN for static
    T = None if dynamic_time else SEG_LEN

    inp = tf.keras.Input(shape=(T,1), name="piezo_in")

    # initial 1×1 projection
    x = tf.keras.layers.Conv1D(filters,
                               1,
                               padding='same',
                               name="init_pw")(inp)

    # ── Gated ResNet blocks ────────────────────────────────────
    for idx, d in enumerate(dilations):
        res = x
        pad_amt = (kernel_size - 1) * d

        # causal left-pad
        x = tf.keras.layers.ZeroPadding1D((pad_amt,0),
                                          name=f"pad_{idx}")(x)

        # separable conv → 2*channels so we can split for gating
        g = tf.keras.layers.SeparableConv1D(
                filters * 2,
                kernel_size,
                dilation_rate=d,
                padding='valid',
                use_bias=True,
                activation=None,
                name=f"sep_{idx}"
            )(x)

        # slice out “content” and “gate” parts        
        content = tf.keras.layers.Lambda(
                  slice_content,
                  arguments={'filters': filters},
                  name=f"content_{idx}"
              )(g)
        gate    = tf.keras.layers.Lambda(
                    slice_gate,
                    arguments={'filters': filters},
                    name=f"gate_{idx}"
                )(g)

        # gated non-linearity: tanh(content) * sigmoid(gate)
        c_act = tf.keras.layers.Activation('tanh',
                                           name=f"tanh_{idx}")(content)
        g_act = tf.keras.layers.Activation('sigmoid',
                                           name=f"sigmoid_{idx}")(gate)
        x = tf.keras.layers.Multiply(name=f"gated_mul_{idx}")([c_act, g_act])

        # 1×1 projection & residual add
        x = tf.keras.layers.Conv1D(filters,
                                   1,
                                   padding='same',
                                   name=f"pw_{idx}")(x)
        x = tf.keras.layers.Add(name=f"res_add_{idx}")([x, res])

    # ── final 1×1 conv + identity skip ────────────────────────
    final = tf.keras.layers.Conv1D(1,
                                   1,
                                   padding='same',
                                   name="final_pw")(x)

    final = tf.keras.layers.Conv1D(1,1,padding='same', name="final_pw")(x)
    skip  = tf.keras.layers.Conv1D(
                1,1,use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(0.5),
                name="skip_conv")(inp)
    main_out = tf.keras.layers.Add(name="res_out")([final, skip])

    # ── (4) tiny HF bypass ─────────────────────────────────
    # a little 3‐tap conv straight from input -> pick spikes
    hf = tf.keras.layers.Conv1D(
            1, 3,
            padding='same',
            use_bias=False,
            kernel_initializer='zeros',   # start with zero so no effect at init
            name="hf_transient")(inp)

    merged = tf.keras.layers.Add(name="add_hf")([main_out, hf])

    # ── smooth limiter ────────────────────────────────────
    out = tf.keras.layers.Activation('tanh', name="tanh_out")(merged)

    return tf.keras.Model(inp, out, name="piezo2mic_resnet_gated")
    

def build_resnet(filters=FILTERS,
                 kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS,
                 dynamic_time=False):
    T   = None if dynamic_time else SEG_LEN
    inp = tf.keras.Input(shape=(T,1), name="piezo_in")
    x   = tf.keras.layers.Conv1D(filters,
                                 1,
                                 padding='same',
                                 name="init_pw")(inp)

    # causal ResNet blocks (unchanged)
    for idx,d in enumerate(dilations):
        res     = x
        pad_amt = (kernel_size-1)*d
        x       = tf.keras.layers.ZeroPadding1D((pad_amt,0),
                                                name=f"pad_{idx}")(x)
        x       = tf.keras.layers.SeparableConv1D(
                      filters=filters,
                      kernel_size=kernel_size,
                      dilation_rate=d,
                      padding='valid',
                      activation='relu',
                      name=f"sep_{idx}"
                  )(x)
        x       = tf.keras.layers.Conv1D(filters,
                                         1,
                                         padding='same',
                                         name=f"pw_{idx}")(x)
        x       = tf.keras.layers.Add(name=f"res_add_{idx}")([x,res])

    # final 1×1 conv
    final = tf.keras.layers.Conv1D(1,
                                   1,
                                   padding='same',
                                   name="final_pw")(x)

    # ---- identity-initialised skip path ----
    # we keep the Conv1D but init it to 1.0 so at training start
    # it is exactly identity, and only learns a small gain correction.
    skip  = tf.keras.layers.Conv1D(1,
                                   1,
                                   use_bias=False,
                                   kernel_initializer=tf.keras.initializers.Ones(),
                                   name="skip_conv")(inp)

    # add the two paths
    res   = tf.keras.layers.Add(name="res_out")([final, skip])

    # ---- smooth limiter instead of hard clip ----
    # this guarantees your network output is always in [-1,1]
    out   = tf.keras.layers.Activation('tanh', name="tanh_out")(res)

    return tf.keras.Model(inputs=inp, outputs=out,
                          name="piezo2mic_resnet_causal")

def extract_and_save_int8_weights(tflite_model_buf,
                                  outdir,
                                  dilations,
                                  filters,
                                  hf_kernel_size=3):
    """
    tflite_model_buf  = the bytes from converter.convert()
    outdir            = path to save files into
    dilations         = your DILATIONS tuple, e.g. (1,2,4)
    filters           = your F, e.g. 8
    hf_kernel_size    = 3 (your tiny HF‐bypass window)
    """
    os.makedirs(outdir, exist_ok=True)
    model = Model.GetRootAsModel(tflite_model_buf, 0)
    subg  = model.Subgraphs(0)
    op_map = {i: model.OperatorCodes(i).BuiltinCode()
              for i in range(model.OperatorCodesLength())}

    def load_tensor(tidx):
        t = subg.Tensors(tidx)
        b = model.Buffers(t.Buffer())
        raw = b.DataAsNumpy()
        dtype = np.int8 if t.Type()==TensorType.INT8 else np.int32
        arr = np.frombuffer(raw, dtype=dtype).reshape(t.ShapeAsNumpy())
        qp = t.Quantization()
        scales      = [qp.Scale(i)     for i in range(qp.ScaleLength())]
        zero_points = [qp.ZeroPoint(i) for i in range(qp.ZeroPointLength())]
        quant_txt   = np.stack([scales, zero_points], axis=1)
        return arr, quant_txt

    sep_ctr    = 0
    pwsep_ctr  = 0
    pw_ctr     = 0

    for oi in range(subg.OperatorsLength()):
        op   = subg.Operators(oi)
        code = op_map[op.OpcodeIndex()]

        # ─────────────────── depthwise (SeparableConv1D) ──────────────────
        if code == BuiltinOperator.DEPTHWISE_CONV_2D:
            w_i = op.Inputs(1)
            b_i = op.InputsLength()>2 and op.Inputs(2)
            tag = f"dw{sep_ctr}"
            sep_ctr += 1

            w_arr, w_q = load_tensor(w_i)
            np.save(f"{outdir}/{tag}_w_int8.npy",        w_arr)
            np.savetxt(f"{outdir}/{tag}_w_int8_quant.txt",
                       w_q, fmt="%g", header="scale,zero_point", comments="")

            if b_i:
                b_arr, b_q = load_tensor(b_i)
                np.save(f"{outdir}/{tag}_b_int32.npy",    b_arr)
                np.savetxt(f"{outdir}/{tag}_b_int32_quant.txt",
                           b_q, fmt="%g", header="scale,zero_point", comments="")
            continue

        # ─────────────────── all 1×1 convs + HF bypass ──────────────────
        if code == BuiltinOperator.CONV_2D:
            w_i, b_i     = op.Inputs(1), op.Inputs(2)
            w_arr, w_q   = load_tensor(w_i)
            b_arr, b_q   = load_tensor(b_i)
            W              = w_arr.size
            B              = b_arr.size

            # 1) tiny HF‐transient: ONLY 3 weights
            if W == hf_kernel_size:
                tag = "hf_transient"

            # 2) skip_conv     : ONLY 1 weight, 1 bias
            elif W == 1 and B == 1:
                tag = "skip_conv"

            # 3) init_pw / final_pw : EXACTLY F weights
            elif W == filters:
                # outC = B
                if B == filters:
                    tag = "init_pw"
                elif B == 1:
                    tag = "final_pw"
                else:
                    raise RuntimeError(
                      f"Found {W} weights but bias‐length={B} is neither 1 nor {filters}")

            # 4) pwsep blocks : F×2F weights, 2F biases
            elif W == filters*(2*filters) and B == (2*filters):
                tag = f"pwsep{pwsep_ctr}"
                pwsep_ctr += 1

            # 5) pw blocks    : F×F weights, F biases
            elif W == filters*filters and B == filters:
                tag = f"pw{pw_ctr}"
                pw_ctr += 1

            else:
                raise RuntimeError(f"Unexpected 1×1‐Conv shape: {w_arr.shape}, bias‐len={B}")

            # write‐out weight + quant
            fnw = os.path.join(outdir, f"{tag}_w_int8.npy")
            np.save(fnw, w_arr)
            np.savetxt(fnw.replace(".npy","_quant.txt"),
                       w_q, fmt="%g", header="scale,zero_point", comments="")

            # write‐out bias + quant
            fnb = os.path.join(outdir, f"{tag}_b_int32.npy")
            np.save(fnb, b_arr)
            np.savetxt(fnb.replace(".npy","_quant.txt"),
                       b_q, fmt="%g", header="scale,zero_point", comments="")

            continue

    # final sanity checks
    assert sep_ctr   == len(dilations), \
           f"got {sep_ctr} depthwise ops, expected {len(dilations)}"
    assert pwsep_ctr == len(dilations), \
           f"got {pwsep_ctr} pwsep ops, expected {len(dilations)}"
    assert pw_ctr    == len(dilations), \
           f"got {pw_ctr} pw ops, expected {len(dilations)}"

    print("✅ saved all int8 kernels + int32 biases in", outdir)

# ─────────────────────────────── TRAINING ──────────────────────────────────────
def train():
    # split train/val
    n = len(pairs)
    idx = list(range(n))
    if n>=2:
        tr_i,va_i = train_test_split(idx, test_size=0.1, random_state=42)
    else:
        tr_i = va_i = idx
    train_pairs = [pairs[i] for i in tr_i]
    val_pairs   = [pairs[i] for i in va_i]
    train_durs  = [sf.info(s).frames for s,_ in train_pairs]
    val_durs    = [sf.info(s).frames for _,t in val_pairs]

    def approx_steps(durs):
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # data generator (no per-segment mean removal)
    def segment_generator(pairs_list, durs_list):
        while True:
            i      = random.randrange(len(pairs_list))
            sp,tp  = pairs_list[i]
            dur    = durs_list[i]
            offset = random.randrange(max(dur-SEG_LEN,1))
            x,_    = sf.read(sp, start=offset, frames=SEG_LEN,
                             dtype='float32', always_2d=True)
            y,_    = sf.read(tp, start=offset, frames=SEG_LEN,
                             dtype='float32', always_2d=True)
            if x.shape[0]<SEG_LEN:
                pad = SEG_LEN - x.shape[0]
                x = np.pad(x,((0,pad),(0,0)))
                y = np.pad(y,((0,pad),(0,0)))
            yield x, y

    # representative dataset for full-integer quant
    def representative_data_gen():
        for _ in range(100):
            i        = random.randrange(len(train_pairs))
            src_path = train_pairs[i][0]
            dur      = train_durs[i]
            offset   = random.randrange(max(dur-SEG_LEN,1))
            x,_      = sf.read(src_path, start=offset, frames=SEG_LEN,
                               dtype="float32", always_2d=True)
            if x.shape[0]<SEG_LEN:
                x = np.pad(x,((0,SEG_LEN-x.shape[0]),(0,0)))
            yield [x.reshape(1,SEG_LEN,1)]

    sig = tf.TensorSpec((SEG_LEN,1), tf.float32)
    train_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_pairs, train_durs),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_pairs, val_durs),
        output_signature=(sig, sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # 1) train FP32 static
    print("\n≈ Training FP32 model")
    static_model = build_resnet_gated(dynamic_time=False)
    static_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                         loss=combined_loss, metrics=['mae'])
    static_model.summary()
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=PATIENCE,
                                          restore_best_weights=True)
    static_model.fit(train_ds,
                     epochs=EPOCHS,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=val_ds,
                     validation_steps=validation_steps,
                     callbacks=[es])
    path_static = os.path.join(OUT_DIR, STATIC_MODEL_FILENAME)
    static_model.save(path_static)
    print("✅ Saved FP32 static model:", path_static)

    # 2) build + save FP32 dynamic
    print("\n≈ Building + saving FP32 dynamic-time model")
    dynamic_model = build_resnet_gated(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)
    
    if (SAVE_WEIGHTS):
        # 4) export layer weights (.npy)
        weights_dir = os.path.join(OUT_DIR,"weights")
        os.makedirs(weights_dir, exist_ok=True)


        # init_pw
        w0,b0 = dynamic_model.get_layer("init_pw").get_weights()
        np.save(os.path.join(weights_dir,"init_pw_w.npy"), w0.astype(np.float32))
        np.save(os.path.join(weights_dir,"init_pw_b.npy"), b0.astype(np.float32))

        # depthwise + pwsep + pw per block
        for idx,_ in enumerate(DILATIONS):
            sep = dynamic_model.get_layer(f"sep_{idx}")
            dw, pw_w1, pw_b1 = sep.get_weights()
            np.save(os.path.join(weights_dir,f"dw{idx}_w.npy"),   dw.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pwsep{idx}_w.npy"),pw_w1.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pwsep{idx}_b.npy"),pw_b1.astype(np.float32))

            pw = dynamic_model.get_layer(f"pw_{idx}")
            w2,b2 = pw.get_weights()
            np.save(os.path.join(weights_dir,f"pw{idx}_w.npy"),   w2.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pw{idx}_b.npy"),   b2.astype(np.float32))

        # final_pw
        wf,bf = dynamic_model.get_layer("final_pw").get_weights()
        np.save(os.path.join(weights_dir,"final_pw_w.npy"), wf.astype(np.float32))
        np.save(os.path.join(weights_dir,"final_pw_b.npy"), bf.astype(np.float32))

        # skip_conv
        sw = dynamic_model.get_layer("skip_conv").get_weights()[0]
        np.save(os.path.join(weights_dir,"skip_conv_w.npy"), sw.astype(np.float32))

        # ── hf_transient (3-tap bypass) ──────────────────────────────
        hf_kernel = dynamic_model.get_layer("hf_transient").get_weights()[0]
        # hf_kernel.shape == (3,1,1)  i.e. (kernel_size, in_channels, out_channels)
        # we only need it as a (K, C) matrix, so drop the last dimension:
        hf_kernel = hf_kernel[:,:,0]       # now shape is (3,1)
        np.save(os.path.join(weights_dir,"hf_transient_w.npy"),
                hf_kernel.astype(np.float32))
        

        # 5) convert + extract int8 TFLite
        print("\n≈ Converting dynamic model to TFLite INT8 model for weight extraction")
        converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        extract_and_save_int8_weights(
            tflite_model,
            os.path.join(OUT_DIR,"int8_weights"),
            dilations=DILATIONS,
            filters=FILTERS
        )

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