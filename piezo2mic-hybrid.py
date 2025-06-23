#!/usr/bin/env python3
"""
Train a causal ResNet+GRU to map an acoustic‐guitar piezo pickup signal
to the equivalent mic’d signal, with end‐to‐end latency under 10 ms.
"""

import os
import glob
import random
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable

# ───────────────────────────── PARAMETERS ─────────────────────────────────────
FILTERS             = 64
KERNEL_SIZE         = 31               # ↓ from 41
DILATIONS           = (1, 2, 4)        # drop the “8×” block
RECEPTIVE_FIELD     = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)  # = 1 + 30·7 = 211
TRAIN_BLOCK_SIZE    = 4096
SR                  = 48000
SEG_LEN             = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE
BATCH_SIZE          = 32
EPOCHS              = 20
PATIENCE            = 5
SHUFFLE_BUF         = 1000

CHUNK_SEC           = 30

SRC_DIR             = "./input/source"
TGT_DIR             = "./input/target"
OUT_DIR             = "./out-hybrid"

DEMO_SEC            = 60 * 2
DEMO_OFFSET_SEC     = 0
DEMO_PAIR_INDEX     = 0

STATIC_MODEL_FILENAME  = "piezo2mic_resnet.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_resnet_dynamic.keras"
SAVE_WEIGHTS           = True   # dump .npy weights for C++ inference

os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────── DATA LOADING ───────────────────────────────────
print("Looking for audio pairs…")
srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*_source.wav")))
pairs = []
for s in srcs:
    t = s.replace("_source.wav", "_target.wav").replace(SRC_DIR, TGT_DIR)
    if os.path.exists(t):
        pairs.append((s, t))
if not pairs:
    print("Error: no _source/_target pairs found")
    sys.exit(1)
print(f"Found {len(pairs)} pairs.")

@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

# ───────────────────────────── LOSS FUNCTIONS ─────────────────────────────────
def pre_emphasis(x, μ=0.99):
    x0 = x[:, :1, :]
    x1 = x[:, 1:, :] - μ * x[:, :-1, :]
    return tf.concat([x0, x1], axis=1)

def pre_emphasis_loss(y_true, y_pred, μ=0.99):
    yt = pre_emphasis(y_true, μ)
    yp = pre_emphasis(y_pred, μ)
    return tf.reduce_mean(tf.abs(yt - yp))

def band_limited_stft_loss(y_true, y_pred,
                           fmin=3000.0, fmax=10000.0,
                           n_fft=512, hop=128):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    S_t = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    S_p = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
    mag_t = tf.abs(S_t)
    mag_p = tf.abs(S_p)
    freq_bins = tf.shape(mag_t)[-1]
    freqs = tf.linspace(0.0, SR/2, freq_bins)
    mask = tf.cast(tf.logical_and(freqs>=fmin, freqs<=fmax), mag_t.dtype)
    mask = mask[tf.newaxis, tf.newaxis, :]
    diff = tf.abs(mag_t - mag_p) * mask
    return tf.reduce_mean(diff)

def freq_weighted_stft_loss(y_true, y_pred, α=2.0):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    total = 0.0
    specs = [(512,128), (1024,256), (2048,512)]
    for n_fft, hop in specs:
        S_t = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_p = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_t)
        mag_p = tf.abs(S_p)
        diff = (tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2
        freq_bins = tf.shape(mag_t)[-1]
        freqs = tf.linspace(0.0, SR/2, freq_bins)
        w = tf.reshape((freqs/(SR/2))**α, [1,1,-1])
        total += tf.reduce_mean(diff * w)
    return total / float(len(specs))

def multi_res_stft_loss(y_true, y_pred):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    specs = [(256,64), (512,128), (1024,256), (2048,512)]
    mag_loss = comp_loss = 0.0
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
    return (mag_loss + comp_loss) / float(len(specs))

def gradient_loss(y_true, y_pred):
    dy_true = y_true[:,1:,:] - y_true[:,:-1,:]
    dy_pred = y_pred[:,1:,:] - y_pred[:,:-1,:]
    return tf.reduce_mean(tf.abs(dy_true - dy_pred))

def mel_loss(y_true, y_pred, n_mels=64, n_fft=1024, hop=256):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    S_t = tf.abs(tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft))
    S_p = tf.abs(tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft))
    num_bins = n_fft//2 + 1
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_bins,
        sample_rate=SR,
        lower_edge_hertz=50.0,
        upper_edge_hertz=SR/2
    )
    M_t = tf.tensordot(S_t, mel_w, axes=[[2],[0]])
    M_p = tf.tensordot(S_p, mel_w, axes=[[2],[0]])
    return tf.reduce_mean(tf.abs(tf.math.log1p(M_t) - tf.math.log1p(M_p)))

def combined_loss(y_true, y_pred):
    l1    = tf.reduce_mean(tf.abs(y_true - y_pred))
    stft  = multi_res_stft_loss(y_true, y_pred)
    fw    = freq_weighted_stft_loss(y_true, y_pred)
    grad  = gradient_loss(y_true, y_pred)
    pre   = pre_emphasis_loss(y_true, y_pred)
    mel   = mel_loss(y_true, y_pred)
    hf    = band_limited_stft_loss(y_true, y_pred,
                                   fmin=3000.0, fmax=10000.0,
                                   n_fft=512, hop=128)
    w = dict(l1=0.10, stft=0.15, fw=0.10, grad=0.15, pre=0.15, mel=0.10, hf=0.25)
    return (
        w['l1'] * l1    +
        w['stft'] * stft+
        w['fw'] * fw    +
        w['grad'] * grad+
        w['pre'] * pre  +
        w['mel'] * mel  +
        w['hf'] * hf
    )

# ────────────────────────── MODEL ARCHITECTURE ─────────────────────────────────
def build_resnet_gated(filters=FILTERS,
                       kernel_size=KERNEL_SIZE,
                       dilations=DILATIONS,
                       dynamic_time=False):
    T = None if dynamic_time else SEG_LEN
    inp = tf.keras.Input(shape=(T,1), name="piezo_in")

    # initial 1×1
    x = tf.keras.layers.Conv1D(filters, 1, padding='same', name="init_pw")(inp)

    # gated ResNet blocks
    for idx, d in enumerate(dilations):
        res = x
        pad = (kernel_size - 1) * d
        x = tf.keras.layers.ZeroPadding1D((pad,0), name=f"pad_{idx}")(x)

        g = tf.keras.layers.SeparableConv1D(
                filters*2, kernel_size,
                dilation_rate=d,
                padding='valid',
                use_bias=True,
                name=f"sep_{idx}"
            )(x)

        content = tf.keras.layers.Lambda(slice_content,
                                         arguments={'filters':filters},
                                         name=f"content_{idx}")(g)
        gate    = tf.keras.layers.Lambda(slice_gate,
                                         arguments={'filters':filters},
                                         name=f"gate_{idx}")(g)

        c_act = tf.keras.layers.Activation('tanh',    name=f"tanh_{idx}")(content)
        g_act = tf.keras.layers.Activation('sigmoid', name=f"sigmoid_{idx}")(gate)
        x = tf.keras.layers.Multiply(name=f"gated_mul_{idx}")([c_act, g_act])

        x = tf.keras.layers.Conv1D(filters,1,padding='same',name=f"pw_{idx}")(x)
        x = tf.keras.layers.Add(name=f"res_add_{idx}")([x, res])

    # ── new: GRU for infinite look‐back (1-sample delay)
    x = tf.keras.layers.GRU(filters,
                             return_sequences=True,
                             name="context_gru")(x)

    # final mix
    final = tf.keras.layers.Conv1D(1,1,padding='same',name="final_pw")(x)
    skip  = tf.keras.layers.Conv1D(
                1,1,use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(0.5),
                name="skip_conv")(inp)
    main  = tf.keras.layers.Add(name="res_out")([final, skip])

    hf = tf.keras.layers.Conv1D(
             1,3,
             padding='same',
             use_bias=False,
             kernel_initializer='zeros',
             name="hf_transient")(inp)

    out = tf.keras.layers.Add(name="piezo2mic_out")([main, hf])
    return tf.keras.Model(inp, out, name="piezo2mic_resnet_hybrid")

# ─────────────────────────────── TRAINING ──────────────────────────────────────
def train():
        # ────────────────────────── CHUNK‐BASED SPLIT ──────────────────────────────
    chunk_len = CHUNK_SEC * SR
    chunks = []
    for src_path, tgt_path in pairs:
        info     = sf.info(src_path)
        total    = info.frames
        n_full   = total // chunk_len

        # full 30 s chunks
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

    # — train FP32 static
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

    # — build + save FP32 dynamic
    print("\n≈ Building + saving FP32 dynamic‐time model")
    dynamic_model = build_resnet_gated(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)

    # Dump .npy weights for C++ inference
    if SAVE_WEIGHTS:
        weights_dir = os.path.join(OUT_DIR, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # init_pw
        w0, b0 = dynamic_model.get_layer("init_pw").get_weights()
        np.save(os.path.join(weights_dir, "init_pw_w.npy"), w0.astype(np.float32))
        np.save(os.path.join(weights_dir, "init_pw_b.npy"), b0.astype(np.float32))

        # gated blocks
        for idx, _ in enumerate(DILATIONS):
            sep = dynamic_model.get_layer(f"sep_{idx}")
            dw, pw_w1, pw_b1 = sep.get_weights()
            np.save(os.path.join(weights_dir, f"dw{idx}_w.npy"),
                    dw.astype(np.float32))
            np.save(os.path.join(weights_dir, f"pwsep{idx}_w.npy"),
                    pw_w1.astype(np.float32))
            np.save(os.path.join(weights_dir, f"pwsep{idx}_b.npy"),
                    pw_b1.astype(np.float32))

            pw = dynamic_model.get_layer(f"pw_{idx}")
            w2, b2 = pw.get_weights()
            np.save(os.path.join(weights_dir, f"pw{idx}_w.npy"),
                    w2.astype(np.float32))
            np.save(os.path.join(weights_dir, f"pw{idx}_b.npy"),
                    b2.astype(np.float32))

        # final_pw
        wf, bf = dynamic_model.get_layer("final_pw").get_weights()
        np.save(os.path.join(weights_dir, "final_pw_w.npy"),
                wf.astype(np.float32))
        np.save(os.path.join(weights_dir, "final_pw_b.npy"),
                bf.astype(np.float32))

        # skip_conv
        sw = dynamic_model.get_layer("skip_conv").get_weights()[0]
        np.save(os.path.join(weights_dir, "skip_conv_w.npy"),
                sw.astype(np.float32))

        # hf_transient
        hf_k = dynamic_model.get_layer("hf_transient").get_weights()[0]  # (3,1,1)
        hf_k = hf_k[:, :, 0]  # shape → (3,1)
        np.save(os.path.join(weights_dir, "hf_transient_w.npy"),
                hf_k.astype(np.float32))

        # context_gru
        gru = dynamic_model.get_layer("context_gru")
        gru_kernel, gru_recurrent, gru_bias = gru.get_weights()
        np.save(os.path.join(weights_dir, "context_gru_kernel.npy"),
                gru_kernel.astype(np.float32))
        np.save(os.path.join(weights_dir, "context_gru_recurrent_kernel.npy"),
                gru_recurrent.astype(np.float32))
        np.save(os.path.join(weights_dir, "context_gru_bias.npy"),
                gru_bias.astype(np.float32))

        print("✅ Saved float32 weights in", weights_dir)

    return static_model, dynamic_model

# ─────────────────────────────────── MAIN ──────────────────────────────────────
if __name__ == "__main__":
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
            custom_objects={'slice_content': slice_content,
                            'slice_gate':    slice_gate}
        )

        # DEMO
        print("\nGenerating demo…")
        FRAMES      = DEMO_SEC * SR
        START_FRAME = DEMO_OFFSET_SEC * SR
        pth_src, pth_tgt = pairs[DEMO_PAIR_INDEX]

        x_long,_ = sf.read(pth_src,
                           start=START_FRAME,
                           frames=FRAMES,
                           dtype='float32', always_2d=True)
        y_ref,_  = sf.read(pth_tgt,
                           start=START_FRAME,
                           frames=FRAMES,
                           dtype='float32', always_2d=True)

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