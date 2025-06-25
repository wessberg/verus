#!/usr/bin/env python3
# ------------------------------------------------------------
#  piezo2mic_train.py
#
#  Train a very small, causal CNN that converts an acoustic
#  guitar piezo signal to the corresponding mic’ed signal.
#  – Real-time (< 1 ms algorithmic latency, <10 ms CPU time)
#  – <20 k trainable parameters → pedal-friendly
#  – Produces FP32 (Keras .keras) and fully–INT8 TFLite files
#  – Creates a 2-minute demo render after training
# ------------------------------------------------------------
import os, glob, random, shutil
from functools import partial

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Conv1D, SeparableConv1D, ZeroPadding1D, Activation
from tensorflow.keras.models import Model


# ----------------------------------------------------------------
# Fixed parameters (edit here if you like)
# ----------------------------------------------------------------
SAMPLE_RATE          = 48_000           # 48 kHz
CHUNK_SEC            = 5                # length of the “big” chunks we cut first
FRAME_SAMPLES        = 2_048           # 42.7 ms frames that the net actually sees
FRAME_HOP            = FRAME_SAMPLES//2 # 50 % overlap for training
BATCH_SIZE           = 32
EPOCHS               = 200

# Early‐stop patience, LR‐reduce on plateau
PATIENCE_EARLYSTOP   = 10
PATIENCE_LR          = 5
REDUCE_FACTOR        = 0.5
MIN_LR               = 1e-6

SRC_DIR              = './input/source'
TGT_DIR              = './input/target'
OUT_DIR              = './out-alt'

DEMO_SEC             = 60*2             # two minute demo render
DEMO_OFFSET_SEC      = 0                # start position in file
DEMO_PAIR_INDEX      = 0                # which file-pair to render

SAVE_WEIGHTS         = True             # emit .keras and .tflite files

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------
# Estimate how many batches (steps) one pass through the files will give us
# ----------------------------------------------------------------

def estimate_steps(pairs):
    chunk_samp       = CHUNK_SEC * SAMPLE_RATE
    frames_per_chunk = (chunk_samp - FRAME_SAMPLES) // FRAME_HOP + 1
    total_frames     = 0

    for src_path, tgt_path in pairs:
        info_s = sf.info(src_path)
        info_t = sf.info(tgt_path)
        if info_s.samplerate != SAMPLE_RATE or info_t.samplerate != SAMPLE_RATE:
            raise RuntimeError("All files must be at %d Hz" % SAMPLE_RATE)

        N = min(info_s.frames, info_t.frames)
        if N < chunk_samp:
            continue
        n_chunks = (N - chunk_samp) // chunk_samp + 1
        total_frames += n_chunks * frames_per_chunk

    return max(1, total_frames // BATCH_SIZE)


# ----------------------------------------------------------------
# Revised train() that repeats the train‐Dataset and feeds steps_per_epoch
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Revised train() with EarlyStopping & ReduceLROnPlateau
# ----------------------------------------------------------------
def train():
    pairs = list_pairs(SRC_DIR, TGT_DIR)
    random.shuffle(pairs)
    split          = int(len(pairs)*0.8)
    train_pairs    = pairs[:split]
    val_pairs      = pairs[split:]

    # –––––––  make the Datasets  –––––––
    # training: shuffle, batch, prefetch, then repeat indefinitely
    ds_train = make_tf_dataset(train_pairs, shuffle=True).repeat()
    # validation: batch, prefetch, then repeat indefinitely
    ds_val   = make_tf_dataset(val_pairs,   shuffle=False).repeat()

    # –––––––  compute how many batches per epoch  –––––––
    steps_per_epoch  = estimate_steps(train_pairs)
    validation_steps = estimate_steps(val_pairs)
    print(f"steps_per_epoch = {steps_per_epoch}, validation_steps = {validation_steps}")

    # –––––––  build & compile  –––––––
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=hybrid_loss)

    # –––––––  callbacks  –––––––
    ckpt = tf.keras.callbacks.ModelCheckpoint(
               os.path.join(OUT_DIR,'best.h5'),
               monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(
                     monitor='val_loss', patience=PATIENCE_EARLYSTOP,
                     restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=REDUCE_FACTOR,
                    patience=PATIENCE_LR, min_lr=MIN_LR, verbose=1)

    # –––––––  fit  –––––––
    model.fit(
        ds_train,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=validation_steps,
        callbacks=[ckpt, early_stop, reduce_lr]
    )

    model = tf.keras.models.load_model(
                os.path.join(OUT_DIR,'best.h5'),
                custom_objects={'hybrid_loss':hybrid_loss})
    return model, pairs

# ----------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------
def list_pairs(src_dir, tgt_dir):
    """Return [(src_path, tgt_path), …] where the basename (no ext) matches."""
    pairs = []
    src_files = sorted(glob.glob(os.path.join(src_dir, '*')))
    for s in src_files:
        base = os.path.splitext(os.path.basename(s))[0]
        for ext in ('.wav', '.flac', '.aif', '.aiff'):
            t = os.path.join(tgt_dir, base.replace("_source", "_target") +ext)
            if os.path.exists(t):
                pairs.append((s, t))
                break
    if not pairs:
        raise RuntimeError('No matching file pairs found!')
    return pairs

def read_mono(path):
    """Read 48 kHz file, force mono float32 in [-1..1]."""
    audio, sr = sf.read(path, dtype='float32')
    if sr != SAMPLE_RATE:
        raise RuntimeError(f'{path} is {sr} Hz, expected {SAMPLE_RATE}')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / max(1e-9, np.abs(audio).max())  # normalise
    return audio.astype(np.float32)

def frame_iter(src, tgt, frame, hop):
    """Yield many (frame, frame) pairs from one long chunk."""
    assert len(src) == len(tgt)
    for s in range(0, len(src)-frame, hop):
        e = s + frame
        yield (src[s:e       ,None],   # shape (frame, 1)
               tgt[s:e       ,None])

# ----------------------------------------------------------------
# tf.data pipeline
# ----------------------------------------------------------------
def make_tf_dataset(pairs, shuffle=True):
    chunk_samples = CHUNK_SEC * SAMPLE_RATE

    def generator():
        for src_path, tgt_path in pairs:
            src = read_mono(src_path)
            tgt = read_mono(tgt_path)
            L   = min(len(src), len(tgt))
            src = src[:L]; tgt = tgt[:L]

            # cut into big 5-second chunks first (keeps memory down)
            for c in range(0, L-chunk_samples, chunk_samples):
                s  = c
                e  = s + chunk_samples
                src_c = src[s:e]
                tgt_c = tgt[s:e]
                for xs, ys in frame_iter(src_c, tgt_c,
                                         FRAME_SAMPLES, FRAME_HOP):
                    yield xs, ys

    output_sig = (tf.TensorSpec(shape=(FRAME_SAMPLES,1), dtype=tf.float32),
                  tf.TensorSpec(shape=(FRAME_SAMPLES,1), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    if shuffle:
        ds = ds.shuffle(1000, seed=0, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------------------------------------------
# Model – a miniature causal TCN (~17 k params, <1 ms look-ahead)
# ----------------------------------------------------------------
def build_model():
    """
    Causal TCN with depthwise‐separable convolutions.
    ~49–50 K trainable parameters, receptive field ≈1.6 ms @48 kHz.
    """
    inp = Input(shape=(None, 1), name='audio_in')  # [B,T,1]

    # ─── Stem: sep‐conv k=15 ───────────────────────────
    pad0 = 14  # (kernel_size-1)*dilation = 14*1
    x = ZeroPadding1D((pad0, 0), name='stem_pad')(inp)
    x = SeparableConv1D(
        filters=96,
        kernel_size=15,
        padding='valid',      # we've already padded
        activation='relu',
        name='stem_sep'
    )(x)

    # ─── Five dilated residual blocks ─────────────────
    for i, d in enumerate([1, 2, 4, 8, 16], start=1):
        # first sep‐conv (k=3)
        pad1 = d * (3 - 1)    # causal pad
        y = ZeroPadding1D((pad1, 0), name=f'res_pad1_d{i}')(x)
        y = SeparableConv1D(
            filters=96,
            kernel_size=3,
            dilation_rate=d,
            padding='valid',
            activation='relu',
            name=f'res_sep1_d{i}'
        )(y)

        # pointwise sep‐conv (k=1), no extra pad needed
        y = SeparableConv1D(
            filters=96,
            kernel_size=1,
            padding='same',
            activation=None,
            name=f'res_sep2_d{i}'
        )(y)

        x = Add(name=f'res_add_d{i}')([x, y])

    # ─── Refine: sep‐conv k=3 @32 ch ──────────────────
    pad2 = 2  # (3-1)*1
    x = ZeroPadding1D((pad2, 0), name='refine_pad')(x)
    x = SeparableConv1D(
        filters=32,
        kernel_size=3,
        padding='valid',
        activation='relu',
        name='refine_sep'
    )(x)

    # ─── Residual output conv & global skip ───────────
    res = Conv1D(
        filters=1,
        kernel_size=1,
        padding='same',
        activation=None,
        name='res_conv'
    )(x)

    out = Add(name='global_residual')([inp, res])
    out = Activation('tanh', name='tanh_out')(out)

    model = Model(inputs=inp, outputs=out, name='piezo2mic_50k')
    return model

# ----------------------------------------------------------------
# Loss:  time-domain L1  +  0.1 × mag-STFT L1   (captures pick/body)
# ----------------------------------------------------------------
def stft_mag(x):
    x = tf.squeeze(x, -1)                       # (B,T)
    S = tf.signal.stft(x,
                       frame_length=1024,
                       frame_step =256,
                       fft_length =1024,
                       pad_end=True)
    return tf.abs(S)

def hybrid_loss(y_true, y_pred):
    l_time = tf.reduce_mean(tf.abs(y_true - y_pred))
    l_spec = tf.reduce_mean(tf.abs(stft_mag(y_true) -
                                   stft_mag(y_pred)))
    return l_time + 0.1*l_spec

# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------
def train():
    # 1) find & split your file‐pairs
    pairs = list_pairs(SRC_DIR, TGT_DIR)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    # 2) build datasets
    ds_train = make_tf_dataset(train_pairs, shuffle=True).repeat()   # <-- infinite
    ds_val   = make_tf_dataset(val_pairs, shuffle=False)            # <-- finite

    # 3) estimate how many batches are in each
    steps_per_epoch = estimate_steps(train_pairs)
    validation_steps = estimate_steps(val_pairs)

    print(f"--> steps_per_epoch = {steps_per_epoch}, validation_steps = {validation_steps}")

    # 4) build & compile
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss=hybrid_loss)

    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUT_DIR, 'best.keras'),
        monitor='val_loss', save_best_only=True, verbose=1)

    # 5) fit, now with steps_per_epoch and validation_steps
    model.fit(
        ds_train,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=validation_steps,
        callbacks=[ckpt]
    )

    # 6) load best‐so‐far and return
    model = tf.keras.models.load_model(
        os.path.join(OUT_DIR, 'best.keras'),
        custom_objects={'hybrid_loss': hybrid_loss}
    )
    return model, pairs

# ----------------------------------------------------------------
# Demo render (offline, not real-time)
# ----------------------------------------------------------------
def render_demo(model, pairs):
    src_path, tgt_path = pairs[DEMO_PAIR_INDEX]
    src = read_mono(src_path)
    tgt = read_mono(tgt_path)

    start = int(DEMO_OFFSET_SEC*SAMPLE_RATE)
    end   = start+int(DEMO_SEC*SAMPLE_RATE)
    src   = src[start:end, None]        # (T,1)

    print('Rendering demo …')
    pred = model.predict(src[None,...], batch_size=1)[0,:,0]     # (T,)

    # save everything
    def wav(name, audio):
        sf.write(os.path.join(OUT_DIR, name),
                 audio, SAMPLE_RATE, subtype='PCM_24')
    wav('demo_piezo.wav',   src[:,0])
    wav('demo_target.wav',  tgt[start:end])
    wav('demo_pred.wav',    pred)

# ----------------------------------------------------------------
#  Post-training weight export
# ----------------------------------------------------------------
def export_weights(model, ds_train):
    if not SAVE_WEIGHTS: return
    print('Exporting FP32 & INT8 weights …')

    # FP32 --------------------------------------------------------
    model.save(os.path.join(OUT_DIR,'model_fp32.keras'))

    # representative samples for quantisation
    def rep_gen():
        for xs, _ in ds_train.take(100):
            yield [xs]

    # INT8 --------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8
    tflite_model = converter.convert()
    with open(os.path.join(OUT_DIR,'model_int8.tflite'), 'wb') as f:
        f.write(tflite_model)

# ----------------------------------------------------------------
#  Main
# ----------------------------------------------------------------
if __name__ == '__main__':
    model, all_pairs = train()
    render_demo(model, all_pairs)
    repr_ds = make_tf_dataset(all_pairs, shuffle=False)
    export_weights(model, repr_ds)
    print('Done – everything is in', OUT_DIR)