# piezo2mic_improved.py
import os, glob, random, sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ───────────────────────── PARAMETERS ─────────────────────────────────────────
SR           = 48000          # Sample Rate
SEG_LEN      = 8192*2           # Training segment length (model's fixed input size for training)
BATCH_SIZE   = 32
EPOCHS       = 1
PATIENCE     = 5              # Increased patience for more stable convergence
SHUFFLE_BUF  = 1000
SRC_DIR      = "./input/source"
TGT_DIR      = "./input/target"
OUT_DIR      = "./out"
DEMO_SEC     = 60 * 5         # Length of the demo audio file to generate after training
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────── DATA LOADING ─────────────────────────────────────
print("Looking for audio pairs...")
srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*_source.wav")))
pairs = []
for s in srcs:
    # Find corresponding target file
    t = s.replace("_source.wav", "_target.wav").replace(SRC_DIR, TGT_DIR)
    if os.path.exists(t):
        pairs.append((s, t))
        
if not pairs:
    print("Error: No source/target audio pairs found in ./input/ !")
    print("Please ensure your files are named '..._source.wav' and '..._target.wav'")
    sys.exit(1)

print(f"Found {len(pairs)} pairs.")

# ───────────────────── LOSS FUNCTIONS ───────────────────────────────────────
def multi_stft_log_mse(y_true, y_pred):
    """
    Computes loss based on the log magnitude of the STFT across multiple resolutions.
    This helps the model learn perceptually important frequency details.
    """
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    loss = 0.0
    # Using multiple FFT sizes helps capture both coarse and fine spectral details
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        S_true = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_pred = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_true)
        mag_p = tf.abs(S_pred)
        # Using log1p (log(1+x)) is more stable than log for small magnitudes
        loss += tf.reduce_mean((tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2)
    return loss
        
def combined_loss(y_true, y_pred):
    """
    A combination of time-domain (L1) and frequency-domain (multi-STFT) loss.
    This provides a balanced optimization target.
    """
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    stft_loss = multi_stft_log_mse(y_true, y_pred)
    # These weights can be tuned. 0.8 for time-domain encourages waveform matching,
    # while 0.2 for spectral ensures frequency content is also corrected.
    return 0.5 * l1_loss + 0.5 * stft_loss
    #return 0.8 * l1_loss + 0.2 * stft_loss

# ───────────────────── MODEL ARCHITECTURE ───────────────────────────────────
def build_resnet(filters=32, kernel_size=63, dilations=(1, 2, 4, 8, 16, 32, 64), dynamic_time=False):
    """
    Builds a WaveNet-like residual model with dilated separable convolutions.
    This architecture has a large receptive field with relatively few parameters.
    """
    # Use None for the time dimension to allow for variable-length input during inference
    T = None if dynamic_time else SEG_LEN
    inp = tf.keras.Input(shape=(T, 1), name="piezo_in")
    
    # Initial 1x1 convolution to expand to the desired number of channels (filters)
    x = tf.keras.layers.Conv1D(filters, 1, padding='same', name="init_pw")(inp)
    
    # Stack of dilated residual blocks
    for idx, d in enumerate(dilations):
        res = x
        # SeparableConv1D is more efficient than a standard Conv1D
        x = tf.keras.layers.SeparableConv1D(
                filters, kernel_size,
                dilation_rate=d,
                padding='same',
                activation='relu',
                name=f"sep_{idx}"
            )(x)
        # 1x1 convolution at the end of the block
        x = tf.keras.layers.Conv1D(filters, 1, padding='same', name=f"pw_{idx}")(x)
        # Residual connection
        x = tf.keras.layers.Add(name=f"res_add_{idx}")([x, res])
        
    # Final 1x1 convolution to collapse back to a single audio channel
    x = tf.keras.layers.Conv1D(1, 1, padding='same', name="final_pw")(x)
    # Final residual connection back to the original input signal
    out = tf.keras.layers.Add(name="res_out")([x, inp])
    
    return tf.keras.Model(inp, out, name="piezo2mic_resnet")

# ───────────────────── MAIN TRAINING FUNCTION ────────────────────────────────
def train():
    # Split files into training and validation sets
    n = len(pairs)
    idx = list(range(n))
    if n >= 2:
        tr_i, va_i = train_test_split(idx, test_size=0.1, random_state=42)
    else:
        # If only one pair, use it for both training and validation
        tr_i = va_i = idx
        
    train_pairs = [pairs[i] for i in tr_i]
    val_pairs   = [pairs[i] for i in va_i]
    train_durs  = [sf.info(s).frames for s, _ in train_pairs]
    val_durs    = [sf.info(s).frames for s, _ in val_pairs]

    def approx_steps(durs):
        # Approximate how many segments can be drawn from the audio files per epoch
        # Using a stride of half the segment length for data augmentation
        return sum(max((d - SEG_LEN) // (SEG_LEN // 2) + 1, 0) for d in durs) // BATCH_SIZE

    steps_per_epoch  = max(approx_steps(train_durs), 1)
    validation_steps = max(approx_steps(val_durs), 1)
    print(f"≈ {steps_per_epoch} training steps/epoch, {validation_steps} validation steps")

    # On-the-fly data generator using tf.data.Dataset
    def segment_generator(pairs_list, durs_list):
        """Yields random (source_segment, target_segment) pairs."""
        while True:
            # Pick a random file
            i = random.randrange(len(pairs_list))
            sp, tp = pairs_list[i]
            dur    = durs_list[i]
            # Pick a random start offset
            offset = random.randrange(max(dur - SEG_LEN, 1))
            
            x, _ = sf.read(sp, start=offset, frames=SEG_LEN, dtype='float32', always_2d=True)
            y, _ = sf.read(tp, start=offset, frames=SEG_LEN, dtype='float32', always_2d=True)
            
            # Pad if the read segment is too short (end of file)
            if x.shape[0] < SEG_LEN:
                pad = SEG_LEN - x.shape[0]
                x = np.pad(x, ((0, pad), (0, 0)))
                y = np.pad(y, ((0, pad), (0, 0)))

            # Simple normalization: remove DC offset per segment
            x = x - np.mean(x)
            y = y - np.mean(y)
            
            yield x, y

    sig = tf.TensorSpec((SEG_LEN, 1), tf.float32)
    train_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_pairs, train_durs),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_pairs, val_durs),
        output_signature=(sig, sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # Build and compile the model
    model = build_resnet(dynamic_time=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=combined_loss,
                  metrics=['mae'])
    model.summary()

    # Set up callbacks and train
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

    # Save the final trained model
    model.save(os.path.join(OUT_DIR, "piezo2mic_resnet_trained.keras"))
    print("✅ Keras model saved to:", os.path.join(OUT_DIR, "piezo2mic_resnet_trained.keras"))
    return model

# ─────────────────────────── SCRIPT ENTRYPOINT ──────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("--- Training new model ---")
        model = train()
    else:
        model_path = os.path.join(OUT_DIR, "piezo2mic_resnet_trained.keras")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train one first using:")
            print(f"python {sys.argv[0]} --train")
            sys.exit(1)
        print(f"--- Loading existing model from {model_path} ---")
        model = tf.keras.models.load_model(model_path, compile=False)

    # Build an inference-specific model with a dynamic time axis
    print("\nBuilding inference model with dynamic input shape...")
    inf_model = build_resnet(dynamic_time=True)
    inf_model.set_weights(model.get_weights())
    inf_model_path = os.path.join(OUT_DIR, "piezo2mic_resnet_dynamic.keras")
    inf_model.save(inf_model_path)
    print("✅ Inference model saved to:", inf_model_path)

    # Generate a demo audio file to test the model
    print("\nGenerating demo audio file...")
    FRAMES = DEMO_SEC * SR
    piezo_path, mic_path = pairs[0]
    
    x_long, _ = sf.read(piezo_path, frames=FRAMES, dtype='float32', always_2d=True)
    y_ref, _ = sf.read(mic_path, frames=FRAMES, dtype='float32', always_2d=True)
    
    x_long = x_long - np.mean(x_long)
    x_in = x_long[np.newaxis, ...] # Add batch dimension -> (1, T, 1)
    
    print("Running inference on demo file...")
    y_pred = inf_model.predict(x_in, batch_size=1)[0, :, 0] # Remove batch/channel dims

    sf.write(os.path.join(OUT_DIR, "demo_input_piezo.wav"), x_long, SR)
    sf.write(os.path.join(OUT_DIR, "demo_output_predicted.wav"), y_pred, SR)
    sf.write(os.path.join(OUT_DIR, "demo_reference_mic.wav"), y_ref, SR)
    print("✅ Wrote demo files to", OUT_DIR)

