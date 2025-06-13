# piezo2mic_improved.py
import os, glob, random
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys

# ───────────────────────── PARAMETERS ─────────────────────────────────────────
SR           = 48000          # sample rate
SEG_LEN      = 2048           # training segment length
BATCH_SIZE   = 32
EPOCHS       = 50
PATIENCE     = 3
SHUFFLE_BUF  = 1000
SRC_DIR      = "./input/source"
TGT_DIR      = "./input/target"
OUT_DIR      = "./out"
DEMO_SEC     = 60 * 5
os.makedirs(OUT_DIR, exist_ok=True)

srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*_source.wav")))
pairs = []
for s in srcs:
    t = s.replace("_source.wav","_target.wav").replace(SRC_DIR, TGT_DIR)
    if os.path.exists(t):
        pairs.append((s,t))
        
def multi_stft_log_mse(y_true, y_pred):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    loss = 0.0
    for n_fft, hop in ((512,128),(1024,256),(2048,512)):
        S_true = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_pred = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_true)
        mag_p = tf.abs(S_pred)
        loss += tf.reduce_mean((tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2)
    return loss
        
def combined_loss(y_true, y_pred):
    # mse_time = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    # mse_spec = multi_stft_log_mse(y_true, y_pred)
    # return 0.5*mse_time + 0.5*mse_spec
    l1 = tf.reduce_mean(tf.abs(y_true-y_pred))
    spec = multi_stft_log_mse(y_true,y_pred)
    return 0.8*l1 + 0.2*spec 

def build_resnet(filters=32, kernel_size=63, dilations=(1,2,4,8,16,32,64), dynamic_time=False):
        T = None if dynamic_time else SEG_LEN
        inp = tf.keras.Input(shape=(T,1), name="piezo_in")
        x = inp
        # initial 1×1 to mix channels
        x = tf.keras.layers.Conv1D(filters,1,padding='same', name="init_pw")(x)
        # dilated residual blocks
        for idx, d in enumerate(dilations):
            res = x
            x = tf.keras.layers.SeparableConv1D(
                    filters, kernel_size,
                    dilation_rate=d,
                    padding='same',
                    activation='relu',
                    name=f"sep_{idx}"
                )(x)
            x = tf.keras.layers.Conv1D(filters,1,padding='same', name=f"pw_{idx}")(x)
            x = tf.keras.layers.Add(name=f"res_add_{idx}")([x, res])
        # final 1×1 to single channel
        x = tf.keras.layers.Conv1D(1,1,padding='same', name="final_pw")(x)
        # residual to input
        out = tf.keras.layers.Add(name="res_out")([x, inp])
        return tf.keras.Model(inp, out, name="piezo2mic_resnet")

def train ():
    # ───────────────────────── PAIR FILES & SPLIT ─────────────────────────────────
    if not pairs:
        raise RuntimeError("No source/target pairs found!")
    n = len(pairs)
    idx = list(range(n))
    if n>=2:
        tr_i, va_i = train_test_split(idx, test_size=0.1, random_state=42)
    else:
        tr_i = va_i = idx
    train_pairs = [pairs[i] for i in tr_i]
    val_pairs   = [pairs[i] for i in va_i]
    train_durs  = [sf.info(s).frames for s,_ in train_pairs]
    val_durs    = [sf.info(s).frames for s,_ in val_pairs]

    def approx_steps(durs):
        # how many segments per epoch?
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈{steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # ───────────────────── ON-THE-FLY DATA GENERATOR ────────────────────────────
    def segment_generator(pairs_list, durs_list):
        """Yields (x_seg, y_seg) of shape (SEG_LEN,1), zero-mean float32."""
        while True:
            i = random.randrange(len(pairs_list))
            sp, tp = pairs_list[i]
            dur    = durs_list[i]
            offset = random.randrange(max(dur-SEG_LEN, 1))
            x,_ = sf.read(sp,  start=offset, frames=SEG_LEN, dtype='float32')
            y,_ = sf.read(tp,  start=offset, frames=SEG_LEN, dtype='float32')
            if x.shape[0]<SEG_LEN:
                pad = SEG_LEN - x.shape[0]
                x = np.pad(x, (0,pad))
                y = np.pad(y, (0,pad))
            x = x - x.mean()
            y = y - y.mean()
            yield x.reshape(SEG_LEN,1), y.reshape(SEG_LEN,1)

    sig = tf.TensorSpec((SEG_LEN,1), tf.float32)
    train_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_pairs, train_durs),
        output_signature=(sig,sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_pairs, val_durs),
        output_signature=(sig,sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # ───────────────────── COMBINED LOSS FUNCTION ──────────────────────────────
    

 

    # ───────────────────── BUILD THE MODEL ─────────────────────────────────────

    model = build_resnet(dynamic_time=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=combined_loss,
                metrics=['mae'])

    # ─────────────────────────── TRAIN ──────────────────────────────────────────
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[es]
    )

    # ──────────────────────── SAVE KERAS MODEL ─────────────────────────────────
    model.save(os.path.join(OUT_DIR, "piezo2mic_resnet.keras"))
    print("✅ Keras model saved to:", OUT_DIR)
    return model

# Check if ---train arg was given
if len(sys.argv) > 1 and sys.argv[1] == "--train":
    print("Training model...")
    model = train()
else:
    print("Loading existing model...")
    # If not, load the existing model
    model = tf.keras.models.load_model(
        os.path.join(OUT_DIR, "piezo2mic_resnet.keras"),
        compile=False)

model.summary()

# ───────────────── BUILD INFERENCE MODEL (any length) ────────────────────────
inf_model = build_resnet(dynamic_time=True)
inf_model.set_weights(model.get_weights())
inf_model.save(os.path.join(OUT_DIR, "piezo2mic_resnet_lite_dynamic_time.keras"))

# ───────────────────────── DEMO INFERENCE ──────────────────────────────────
FRAMES   = DEMO_SEC * SR
piezo_path, mic_path = pairs[0]
x_long, _ = sf.read(piezo_path, frames=FRAMES, dtype='float32')
y_ref,   _ = sf.read(mic_path,   frames=FRAMES, dtype='float32')
x_long = x_long - x_long.mean()
x_in = x_long[np.newaxis,:,np.newaxis]
y_pred = inf_model.predict(x_in, batch_size=1)[0,:,0]

sf.write(os.path.join(OUT_DIR,"demo_piezo_piezo2mic.wav"), x_long, SR)
sf.write(os.path.join(OUT_DIR,"demo_pred_piezo2mic.wav"), y_pred,   SR)
sf.write(os.path.join(OUT_DIR,"demo_mic_piezo2mic.wav"),  y_ref,    SR)
print("✅ Wrote demo files to", OUT_DIR)