import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import glob

def segment_all_audio_pairs(
    source_dir="./input/source",
    target_dir="./input/target",
    segment_length_s=7,
    hop_factor=0.5,
    validation_split=0.2,
    random_seed=42
):
    all_segments = []

    # find matching filenames
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.wav")))
    for source_path in source_files:
        basename = os.path.basename(source_path).replace("_source.wav", "")
        target_path  = os.path.join(target_dir, basename + "_target.wav")
        if not os.path.exists(target_path):
            continue  # skip unmatched

        # load each pair
        sr = sf.info(source_path).samplerate
        src, _ = librosa.load(source_path, sr=sr, mono=True)
        tgt, _ = librosa.load(target_path, sr=sr, mono=True)

        seg_len = int(segment_length_s * sr)
        hop     = int(seg_len * hop_factor)
        # pad if needed
        if len(src) < seg_len:
            pad = seg_len - len(src)
            src = np.pad(src, (0, pad))
            tgt = np.pad(tgt, (0, pad))

        # slice into segments
        for start in range(0, len(src) - seg_len + 1, hop):
            x = src[start:start + seg_len]
            y = tgt[start:start + seg_len]
            all_segments.append((x, y))

    # then stack, add channel dim, split/train as before
    X = np.stack([s[0] for s in all_segments])[..., np.newaxis]
    Y = np.stack([s[1] for s in all_segments])[..., np.newaxis]
    
    X = X[..., np.newaxis] # (N, L, 1)
    Y = Y[..., np.newaxis]
    
    # train/val split
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X, Y, test_size=validation_split, random_state=random_seed
    )
    
    # build tf.data pipelines
    batch = 4
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
        .shuffle(len(X_tr))
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_va, Y_va))
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return train_ds, val_ds, sr, X_va, Y_va

def spectral_loss(y_true, y_pred,
                  fft_size=512, hop=128,
                  power=2.0, use_log=True, eps=1e-6):
    # y: (batch, T, 1) → (batch, T)
    y_t = tf.squeeze(y_true, -1)
    y_p = tf.squeeze(y_pred, -1)
    S_t = tf.signal.stft(y_t, frame_length=fft_size, frame_step=hop, fft_length=fft_size)
    S_p = tf.signal.stft(y_p, frame_length=fft_size, frame_step=hop, fft_length=fft_size)
    M_t = tf.abs(S_t)**power
    M_p = tf.abs(S_p)**power
    if use_log:
        M_t = tf.math.log(M_t + eps)
        M_p = tf.math.log(M_p + eps)
    return tf.reduce_mean((M_t - M_p)**2)


def composite_loss(y_true, y_pred):
    mae  = tf.reduce_mean(tf.abs(y_true - y_pred))
    spec = spectral_loss(y_true, y_pred, power=2.0) * 0.1
    # deep = tf.reduce_mean((feat_model(y_true) - feat_model(y_pred))**2) * 0.05
    return mae + spec  # + deep

def build_tiny_unet(input_len=None):
    inp = layers.Input(shape=(input_len, 1))
    # Encoder
    x1 = layers.Conv1D(16, 15, padding='same', activation='relu')(inp)
    d1 = layers.Conv1D(32, 15, strides=2, padding='same', activation='relu')(x1)
    # Bottleneck (dilated)
    b = layers.Conv1D(32, 15, padding='same', dilation_rate=2, activation='relu')(d1)
    # Decoder
    u = layers.Conv1DTranspose(16, 15, strides=2, padding='same', activation='relu')(b)
    c = layers.Concatenate()([u, x1])
    out = layers.Conv1D(1, 1, padding='same')(c)
    return Model(inp, out, name="tiny_unet")

def train_and_save():
    # prepare data
    train_ds, val_ds, sr, X_val, Y_val = segment_all_audio_pairs()
    
    # build & compile
    model = build_tiny_unet(input_len=X_val.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=composite_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    
    # callback: early stop at first plateau
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=0,
        restore_best_weights=True,
        verbose=1
    )
    
    # train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[cb]
    )
    
    # save in Keras v3 format
    model.save("source_to_target_model.keras")
    
    # generate & save a few examples
    preds = model.predict(X_val[:5])
    os.makedirs("preds", exist_ok=True)
    for i, (p, t) in enumerate(zip(preds, Y_val[:5])):
        sf.write(f"preds/pred_{i}.wav", p.squeeze(), sr)
        sf.write(f"preds/target_{i}.wav", t.squeeze(), sr)
    
    print("Done. Samples in ./preds/")

if __name__ == "__main__":
    train_and_save()
