import sounddevice as sd
import numpy as np
import tensorflow as tf
import time
import threading
import queue

gpus = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices([], "GPU")

# ─────── SETTINGS ───────────────────────────────────────────────────────────
SR         = 48000        # must match your model’s training SR
BLOCK_SIZE = 8096 * 4          # try 64, 128, 256…  smaller = lower I/O latency, more callback overhead
CHANNELS   = 1            # mono
QUEUE_MAX  = 4                             # ← buffer a few blocks

MODEL_PATH = "./out/piezo2mic_resnet_dynamic_time.keras"  # or SavedModel dir
# ─────────────────────────────────────────────────────────────────────────────

# 1) List all audio devices
devices = sd.query_devices()
print("\nAvailable audio devices:\n")
for idx, dev in enumerate(devices):
    print(f"  {idx:>2}: {dev['name']:<40}  in:{dev['max_input_channels']:<2}  out:{dev['max_output_channels']:<2}")
print()

# 2) Pick an input device
while True:
    try:
        inp = 2 # int(input("Select INPUT device index: "))
        if devices[inp]['max_input_channels'] > 0:
            break
        else:
            print("  → That device has no input channels. Try again.")
    except (ValueError, IndexError):
        print("  → Invalid index. Try again.")

# 3) Pick an output device
while True:
    try:
        outp = 2 # int(input("Select OUTPUT device index: "))
        if devices[outp]['max_output_channels'] > 0:
            break
        else:
            print("  → That device has no output channels. Try again.")
    except (ValueError, IndexError):
        print("  → Invalid index. Try again.")

print(f"\nUsing INPUT #{inp}: {devices[inp]['name']}")
print(f"Using OUTPUT#{outp}: {devices[outp]['name']}\n")


# 2) load and JIT-warm your TF model
print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)
@tf.function
def infer(x):
    return model(x, training=False)
_ = infer(tf.zeros((1, BLOCK_SIZE, 1), dtype=tf.float32))

# 3) compute approximate receptive-field (for history buffer size)
def compute_rf(m):
    rf = 1
    for L in m.layers:
        if isinstance(L, (tf.keras.layers.Conv1D, tf.keras.layers.SeparableConv1D)):
            k = L.kernel_size[0]
            d = L.dilation_rate[0] if hasattr(L, "dilation_rate") else 1
            rf += (k - 1) * d
    return rf

RF = compute_rf(model)
print(f"Receptive-field ≃ {RF} samples → {RF/SR*1000:.1f} ms history\n")

# 4) queues + DC-block state
in_q  = queue.Queue(maxsize=QUEUE_MAX)
out_q = queue.Queue(maxsize=QUEUE_MAX)
dc_state = 0.0
HPF_ALPHA = 0.995

# 5) background worker thread
def worker():
    global dc_state
    # start history as zeros
    history = np.zeros((RF,1), dtype=np.float32)
    while True:
        block = in_q.get()   # (BLOCK_SIZE,1)
        # 5a) concatenate history + new block
        cat = np.concatenate([history, block], axis=0)
        # 5b) run through TF
        y_all = infer(tf.constant(cat[None,...]))        # shape (1, RF+BLOCK_SIZE,1)
        y_all = y_all.numpy()[0,:,0]                    # → (RF+BLOCK_SIZE,)
        # 5c) our new output is only the tail BLOCK_SIZE
        y_out = y_all[-BLOCK_SIZE:].copy()
        # 5d) advance history for next round
        history = cat[-RF:]
        # 5e) enqueue result (blocking if the consumer is slow)
        out_q.put(y_out)

threading.Thread(target=worker, daemon=True).start()

# 6) audio callback (non-blocking)
def callback(indata, outdata, frames, time_info, status):
    global dc_state
    if status.input_underflow or status.output_underflow:
        print("UNDERFLOW/OVERFLOW:", status)
    # 6a) DC-block each new sample (one-pole high-pass)
    x_raw = indata[:frames,0].astype(np.float32)
    x_hp  = np.empty_like(x_raw)
    for i, s in enumerate(x_raw):
        dc_state = HPF_ALPHA * dc_state + (1 - HPF_ALPHA) * s
        x_hp[i]  = s - dc_state
    blk = x_hp.reshape(frames,1)

    # 6b) push to worker (drop if queue full)
    try:
        in_q.put_nowait(blk)
    except queue.Full:
        pass

    # 6c) pull a processed block (or else repeat last_out)
    try:
        out = out_q.get_nowait()
        callback.last_out = out
    except queue.Empty:
        out = getattr(callback, "last_out", np.zeros((frames,),dtype=np.float32))

    # 6d) write to output (and zero any extra channels)
    outdata[:] = 0.0
    outdata[:frames,0] = out

# 7) open low-latency full-duplex stream
print(f"Starting stream @ {SR} Hz, block={BLOCK_SIZE}…\n")
with sd.Stream(samplerate=SR,
               blocksize=BLOCK_SIZE,
               device=(inp, outp),
               dtype='float32',
               latency='low',
               channels=CHANNELS,
               callback=callback):
    print("Streaming… Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nDone.")