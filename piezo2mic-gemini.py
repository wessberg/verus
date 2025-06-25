import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import glob
import time

# -- CONSTANT PARAMETERS --
SAMPLE_RATE                  = 48000
CHUNK_SEC                    = 5
SRC_DIR                      = "./input/source"
TGT_DIR                      = "./input/target"
OUT_DIR                      = "./out-gemini"

# Training & Model Parameters
VALIDATION_SPLIT             = 0.2
FRAME_LENGTH                 = 2048 # Receptive field of the model in samples
BATCH_SIZE                   = 32
EPOCHS                       = 200
PATIENCE                     = 10 # For Early Stopping

# Demo Generation Parameters
DEMO_SEC                     = 120
DEMO_OFFSET_SEC              = 0
DEMO_PAIR_INDEX              = 0
SAVE_MODEL_AND_WEIGHTS       = True # If true, both INT8 and FP32 models will be saved

# -- HELPER FUNCTIONS --

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Loads an audio file, converts to mono and float32."""
    try:
        audio, sr = sf.read(file_path, dtype='float32')
        if sr != sample_rate:
            raise ValueError(f"Incorrect sample rate: {file_path} has {sr}Hz, expected {sample_rate}Hz")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Downmix to mono
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_dataset(source_files, target_files):
    """Creates a tf.data.Dataset from pairs of audio files."""
    source_chunks = []
    target_chunks = []

    print(f"Loading and chunking {len(source_files)} file pairs...")
    for src_path, tgt_path in zip(source_files, target_files):
        src_audio = load_audio(src_path)
        tgt_audio = load_audio(tgt_path)

        if src_audio is None or tgt_audio is None:
            continue

        # Ensure audio lengths are the same, trim to the shorter one
        min_len = min(len(src_audio), len(tgt_audio))
        src_audio = src_audio[:min_len]
        tgt_audio = tgt_audio[:min_len]

        chunk_samples = CHUNK_SEC * SAMPLE_RATE
        for i in range(0, min_len - chunk_samples + 1, chunk_samples):
            source_chunks.append(src_audio[i:i + chunk_samples])
            target_chunks.append(tgt_audio[i:i + chunk_samples])

    if not source_chunks:
        raise ValueError("No audio chunks were created. Check your input files and CHUNK_SEC.")

    # Create a dataset of the large audio chunks
    dataset = tf.data.Dataset.from_tensor_slices((source_chunks, target_chunks))

    # ---- START: CORRECTED SECTION ----
    # Define a function to apply tf.signal.frame, which is graph-compatible
    def chunk_to_frames(src_chunk, tgt_chunk):
        # The frame_step is equal to the frame_length for non-overlapping frames from chunks
        src_frames = tf.signal.frame(src_chunk, FRAME_LENGTH, FRAME_LENGTH, pad_end=False)
        tgt_frames = tf.signal.frame(tgt_chunk, FRAME_LENGTH, FRAME_LENGTH, pad_end=False)
        return src_frames, tgt_frames

    # Map this function across all chunks. This creates batches of frames.
    dataset = dataset.map(chunk_to_frames, num_parallel_calls=tf.data.AUTOTUNE)

    # Use unbatch() to flatten the batches of frames into a single stream of individual frames.
    # This is the key step that makes the data ready for the model.
    dataset = dataset.unbatch()
    # ---- END: CORRECTED SECTION ----

    # Add channel dimension for the Conv1D layers
    dataset = dataset.map(lambda src, tgt: (tf.expand_dims(src, -1), tf.expand_dims(tgt, -1)))

    return dataset

def build_low_latency_model(input_shape):
    """
    Builds an improved lightweight, causal model with a ~50k parameter budget.
    This version is deeper, wider, and uses residual connections for better training.
    """
    # Parameters for the improved model
    FILTERS = 24  # Increased from 16 for more feature learning capacity
    KERNEL_SIZE = 7
    # Increased depth and receptive field with higher dilation rates
    DILATION_RATES = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16] 

    inp = tf.keras.layers.Input(shape=input_shape)

    # 1. Initial "stem" convolution to expand channels from 1 to FILTERS
    # This prepares the tensor for the main residual blocks.
    x = tf.keras.layers.Conv1D(
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        padding='causal',
        name='stem_conv'
    )(inp)

    # 2. Stack of residual blocks with dilated convolutions
    for i, dilation_rate in enumerate(DILATION_RATES):
        # Store the input for the residual connection
        residual = x
        
        # Main causal, dilated convolution
        x = tf.keras.layers.Conv1D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu', # Activation is now inside the block
            name=f'd_conv_{i}'
        )(x)
        
        # Add the residual (input) to the output of the convolution
        x = tf.keras.layers.Add(name=f'residual_add_{i}')([x, residual])

    # 3. Output layer to reconstruct the signal
    # 1x1 convolution is efficient and acts as a final linear transformation
    output = tf.keras.layers.Conv1D(
        filters=1, 
        kernel_size=1, 
        padding='causal',
        activation='tanh', # Tanh activation to keep output in [-1, 1] range
        name='output_conv'
    )(x)

    model = tf.keras.Model(inp, output)
    return model

# -- MAIN EXECUTION --
if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")

    # 1. Setup Environment
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub_dir in [SRC_DIR, TGT_DIR]:
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            print(f"Created directory: {sub_dir}. Please add your .wav files.")

    # 2. Find and match audio files
    source_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.wav")))
    target_files = sorted(glob.glob(os.path.join(TGT_DIR, "*.wav")))

    if not source_files or not target_files:
        print("Error: Source or target directory is empty. Exiting.")
        exit()

    if len(source_files) != len(target_files):
        print("Warning: Mismatch in the number of source and target files.")
        min_files = min(len(source_files), len(target_files))
        source_files = source_files[:min_files]
        target_files = target_files[:min_files]

    print(f"Found {len(source_files)} audio pairs for training.")

    # 3. Create Train/Validation Split from file list
    indices = np.arange(len(source_files))
    np.random.shuffle(indices)
    split_point = int(len(indices) * (1 - VALIDATION_SPLIT))
    train_indices, val_indices = indices[:split_point], indices[split_point:]

    train_src_files = [source_files[i] for i in train_indices]
    train_tgt_files = [target_files[i] for i in train_indices]
    val_src_files = [source_files[i] for i in val_indices]
    val_tgt_files = [target_files[i] for i in val_indices]

    # 4. Create tf.data.Dataset pipelines
    try:
        train_dataset = create_dataset(train_src_files, train_tgt_files)
        val_dataset = create_dataset(val_src_files, val_tgt_files)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        exit()

    # Calculate dataset sizes for steps_per_epoch
    # tf.data.UNKNOWN_CARDINALITY is -2, so we handle it
    train_cardinality = train_dataset.cardinality().numpy()
    if train_cardinality < 0:
        print("Calculating training set size...")
        train_cardinality = sum(1 for _ in train_dataset)

    val_cardinality = val_dataset.cardinality().numpy()
    if val_cardinality < 0:
        print("Calculating validation set size...")
        val_cardinality = sum(1 for _ in val_dataset)


    if train_cardinality == 0 or val_cardinality == 0:
        print("Error: Training or validation dataset is empty. Check audio file lengths and chunking settings.")
        exit()

    # Prepare datasets for training
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # 5. Build and Compile Model
    model_input_shape = (FRAME_LENGTH, 1)
    model = build_low_latency_model(model_input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    print("\n--- Model Summary ---")
    model.summary()
    print("---------------------\n")

    # 6. Train the Model
    steps_per_epoch = train_cardinality // BATCH_SIZE
    validation_steps = val_cardinality // BATCH_SIZE

    if steps_per_epoch == 0 or validation_steps == 0:
        print("Error: Not enough data to form a single batch. Decrease BATCH_SIZE or add more/longer audio files.")
        exit()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    )

    print("--- Starting Training ---")
    start_time = time.time()

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=[early_stopping],
        verbose=1
    )

    print(f"--- Training Finished in {time.time() - start_time:.2f} seconds ---")

    # 7. Save Model and Weights
    if SAVE_MODEL_AND_WEIGHTS:
        print("\n--- Saving Models and Weights ---")

        # Save FP32 Keras model
        fp32_model_path = os.path.join(OUT_DIR, "model_fp32.keras")
        model.save(fp32_model_path)
        print(f"FP32 model saved to: {fp32_model_path}")

        # Save INT8 quantized model (TFLite)
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Create a representative dataset for quantization calibration
            def representative_dataset_gen():
                # Use a small portion of the validation dataset for calibration
                val_data_for_calib = create_dataset(val_src_files, val_tgt_files)
                for input_value, _ in val_data_for_calib.batch(1).take(100):
                    yield [input_value]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            tflite_model_quant = converter.convert()

            int8_model_path = os.path.join(OUT_DIR, "model_int8.tflite")
            with open(int8_model_path, 'wb') as f:
                f.write(tflite_model_quant)
            print(f"INT8 quantized TFLite model saved to: {int8_model_path}")

        except Exception as e:
            print(f"Could not create INT8 model. Error: {e}")
            print("Skipping INT8 model generation.")


    # 8. Generate and Save Demo Audio
    print("\n--- Generating Demo Audio ---")
    if DEMO_PAIR_INDEX >= len(source_files):
        print(f"Warning: DEMO_PAIR_INDEX ({DEMO_PAIR_INDEX}) is out of bounds. Using index 0 instead.")
        DEMO_PAIR_INDEX = 0

    src_demo_path = source_files[DEMO_PAIR_INDEX]
    tgt_demo_path = target_files[DEMO_PAIR_INDEX]

    src_audio_full = load_audio(src_demo_path)
    tgt_audio_full = load_audio(tgt_demo_path)

    start_sample = DEMO_OFFSET_SEC * SAMPLE_RATE
    end_sample = start_sample + (DEMO_SEC * SAMPLE_RATE)

    src_demo = src_audio_full[start_sample:end_sample]
    tgt_demo = tgt_audio_full[start_sample:end_sample]

    pad_amount = FRAME_LENGTH - (len(src_demo) % FRAME_LENGTH)
    src_demo_padded = np.pad(src_demo, (0, pad_amount), 'constant')

    src_demo_reshaped = src_demo_padded.reshape(-1, FRAME_LENGTH, 1)

    pred_demo_reshaped = model.predict(src_demo_reshaped, batch_size=BATCH_SIZE)

    pred_demo_padded = pred_demo_reshaped.flatten()
    pred_demo = pred_demo_padded[:len(src_demo)]

    sf.write(os.path.join(OUT_DIR, "demo_01_source_piezo.wav"), src_demo, SAMPLE_RATE)
    sf.write(os.path.join(OUT_DIR, "demo_02_target_mic.wav"), tgt_demo, SAMPLE_RATE)
    sf.write(os.path.join(OUT_DIR, "demo_03_predicted_mic.wav"), pred_demo, SAMPLE_RATE)

    print(f"Demo audio files saved in '{OUT_DIR}' directory.")
    print("Script finished successfully.")