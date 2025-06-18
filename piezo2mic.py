import os, glob, random, sys, logging, re
import numpy as np
import soundfile as sf
import tensorflow as tf
from tflite import (
    Model,
    TensorType,
    BuiltinOperator
)
from sklearn.model_selection import train_test_split
from collections import defaultdict


# ───────────────────────── PARAMETERS ─────────────────────────────────────────

FILTERS      = 32
KERNEL_SIZE  = 31
DILATIONS    = (1,2,4,8,16,32)
RECEPTIVE_FIELD = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
TRAIN_BLOCK_SIZE = 4096
SR           = 48000
SEG_LEN      = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE
BATCH_SIZE   = 32
EPOCHS       = 15
PATIENCE     = 3
SHUFFLE_BUF  = 1000
SRC_DIR      = "./input/source"
TGT_DIR      = "./input/target"
OUT_DIR      = "./out"
DEMO_SEC     = 60 * 2

STATIC_MODEL_BASENAME      = "piezo2mic_resnet"
DYNAMIC_MODEL_BASENAME     = STATIC_MODEL_BASENAME + "_dynamic"
QUANT_MODEL_BASENAME       = DYNAMIC_MODEL_BASENAME + "_qweight"

STATIC_MODEL_FILENAME      = STATIC_MODEL_BASENAME + ".keras"
DYNAMIC_MODEL_FILENAME     = DYNAMIC_MODEL_BASENAME + ".keras"
QUANT_MODEL_FILENAME       = QUANT_MODEL_BASENAME + ".keras"

os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────── DATA LOADING ────────────────────────────────────────
print("Looking for audio pairs…")
srcs = sorted(glob.glob(os.path.join(SRC_DIR, "*_source.wav")))
pairs = []
for s in srcs:
    t = s.replace("_source.wav", "_target.wav").replace(SRC_DIR, TGT_DIR)
    if os.path.exists(t):
        pairs.append((s,t))
if not pairs:
    print("Error: no _source/_target pairs found"); sys.exit(1)
print(f"Found {len(pairs)} pairs.")

# ───────────────────────── LOSS FUNCS ─────────────────────────────────────────
def multi_stft_log_mse(y_true, y_pred):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    loss = 0.0
    for n_fft, hop in ((512,128),(1024,256),(2048,512)):
        S_true = tf.signal.stft(yt, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        S_pred = tf.signal.stft(yp, frame_length=n_fft, frame_step=hop, fft_length=n_fft)
        mag_t = tf.abs(S_true); mag_p = tf.abs(S_pred)
        loss += tf.reduce_mean((tf.math.log1p(mag_t) - tf.math.log1p(mag_p))**2)
    return loss

def combined_loss(y_true, y_pred):
    l1   = tf.reduce_mean(tf.abs(y_true - y_pred))
    stft = multi_stft_log_mse(y_true, y_pred)
    return 0.5*l1 + 0.5*stft

# ───────────────────────── MODEL ARCH ─────────────────────────────────────────
def build_resnet(filters=FILTERS,
                 kernel_size=KERNEL_SIZE,
                 dilations=DILATIONS,
                 dynamic_time=False):
    T = None if dynamic_time else SEG_LEN
    inp = tf.keras.Input(shape=(T,1), name="piezo_in")
    x   = tf.keras.layers.Conv1D(filters,1,padding='same',name="init_pw")(inp)

    for idx,d in enumerate(dilations):
        res    = x
        pad_amt= (kernel_size-1)*d
        x      = tf.keras.layers.ZeroPadding1D((pad_amt,0),name=f"pad_{idx}")(x)
        x      = tf.keras.layers.SeparableConv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=d,
                    padding='valid',
                    activation='relu',
                    name=f"sep_{idx}"
                 )(x)
        x      = tf.keras.layers.Conv1D(filters,1,padding='same',name=f"pw_{idx}")(x)
        x      = tf.keras.layers.Add(name=f"res_add_{idx}")([x,res])

    x   = tf.keras.layers.Conv1D(1,1,padding='same',name="final_pw")(x)
    out = tf.keras.layers.Add(name="res_out")([x, inp])
    return tf.keras.Model(inp,out,name="piezo2mic_resnet_causal")

# ─────────────────── MANUAL WEIGHT-ONLY QUANTIZER ────────────────────────────
def quantize_weights_inplace(model):
    """
    Per-tensor linear quantization of all float32 weights down to 8-bit levels.
    We compute min/max of each weight tensor, map to [0..255], round, then
    dequantize back to float32 (so your saved Keras model still has float32
    weights but only 256 distinct levels).
    """
    new_vars = []
    for w in model.get_weights():
        w_min, w_max = w.min(), w.max()
        if w_max <= w_min:
            new_vars.append(w)
            continue
        scale = (w_max - w_min) / 255.0
        # map to 0..255
        q = np.round((w - w_min) / scale).astype(np.uint8)
        # dequantize
        w_dq = q.astype(np.float32) * scale + w_min
        new_vars.append(w_dq)
    model.set_weights(new_vars)
    
def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations):
    """
    tflite_model_buf : bytes  ← tflite_model = converter.convert()
    outdir            : str    ← where to write .npy + _quant.txt
    dilations         : tuple  ← your (1,2,4,8,16,32)
    """
    os.makedirs(outdir, exist_ok=True)

    # ── load flatbuffer ─────────────────────────────────────────────
    model = Model.GetRootAsModel(tflite_model_buf, 0)
    subg  = model.Subgraphs(0)   # type: _SubGraph

    # ── build opcode → builtin mapping ─────────────────────────────
    opcodes = {}
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        opcodes[i] = oc.BuiltinCode()

    # ── helper to read a tensor’s raw buffer + quant params ─────────
    def load_tensor(tidx):
        t = subg.Tensors(tidx)           # type: _Tensor
        b = model.Buffers(t.Buffer())    # type: _Buffer
        raw = b.DataAsNumpy()            # view of bytes

        # pick numpy dtype
        ty = t.Type()
        if   ty == TensorType.INT8:   dt = np.int8
        elif ty == TensorType.INT32:  dt = np.int32
        else:
            raise ValueError(f"unexpected tensor type {ty}")

        arr = np.frombuffer(raw, dtype=dt).reshape(t.ShapeAsNumpy())

        # pull quant params
        qp = t.Quantization()
        scales      = [qp.Scale(i)      for i in range(qp.ScaleLength())]
        zero_points = [qp.ZeroPoint(i)  for i in range(qp.ZeroPointLength())]
        qmat = np.stack([scales, zero_points], axis=1)

        return arr, qmat

    sep_counter   = 0   # counts DEPTHWISE_CONV_2D → dw0..dw5
    conv2_counter = 0   # counts all CONV_2D → init,pwsep,pw,final

    # ── walk the operators in topo order ────────────────────────────
    for oi in range(subg.OperatorsLength()):
        op   = subg.Operators(oi)
        code = opcodes[op.OpcodeIndex()]

        # 1) depthwise halves of your SeparableConv1Ds
        if code == BuiltinOperator.DEPTHWISE_CONV_2D:
            # inputs: [in, filter, bias]  (bias is usually zero-vector or absent)
            w_idx = op.Inputs(1)
            b_idx = op.Inputs(2) if op.InputsLength() > 2 else None

            # save w
            w_arr, w_q = load_tensor(w_idx)
            fn = os.path.join(outdir, f"dw{sep_counter}_w_int8.npy")
            np.save(fn, w_arr)
            np.savetxt(fn.replace(".npy","_quant.txt"), w_q,
                       fmt="%g", delimiter=",",
                       header="scale,zero_point", comments="")

            # save bias if present
            if b_idx is not None:
                b_arr, b_q = load_tensor(b_idx)
                fn = os.path.join(outdir, f"dw{sep_counter}_b_int32.npy")
                np.save(fn, b_arr)
                np.savetxt(fn.replace(".npy","_quant.txt"), b_q,
                           fmt="%g", delimiter=",",
                           header="scale,zero_point", comments="")

            print(f"→ depthwise dw{sep_counter}")
            sep_counter += 1
            continue

        # 2) all the 1×1 convs: init_pw, the pointwise half of each sep,
        #    the residual pw_i, and your final_pw
        if code == BuiltinOperator.CONV_2D:
            w_idx, b_idx = op.Inputs(1), op.Inputs(2)
            conv2_counter += 1
            cnt = conv2_counter
            N   = len(dilations)

            # decide tag by counting pattern:
            #   conv2_counter == 1           → init_pw
            #   conv2_counter == 2*(N+1)     → final_pw
            #   otherwise evens               → pwsep[i]
            #             odds (>1)          → pw[i]
            if cnt == 1:
                tag = "init_pw"
            elif cnt == 2*(N+1):
                tag = "final_pw"
            elif cnt % 2 == 0:
                i = cnt//2 - 1
                tag = f"pwsep{i}"
            else:
                i = (cnt - 3)//2
                tag = f"pw{i}"

            # weight
            w_arr, w_q = load_tensor(w_idx)
            fn = os.path.join(outdir, f"{tag}_w_int8.npy")
            np.save(fn, w_arr)
            np.savetxt(fn.replace(".npy","_quant.txt"), w_q,
                       fmt="%g", delimiter=",",
                       header="scale,zero_point", comments="")

            # bias
            b_arr, b_q = load_tensor(b_idx)
            fn = os.path.join(outdir, f"{tag}_b_int32.npy")
            np.save(fn, b_arr)
            np.savetxt(fn.replace(".npy","_quant.txt"), b_q,
                       fmt="%g", delimiter=",",
                       header="scale,zero_point", comments="")

            print(f"→ conv2d #{cnt} → {tag}")
            continue

        # skip everything else
        continue

    # ── sanity checks ───────────────────────────────────────────────
    assert sep_counter   == len(dilations),   f"got {sep_counter} depthwise_conv2d ops, expected {len(dilations)}"
    # init + 2*N convs (pwsep & pw) + final = 1 + 2*N + 1 = 2*(N+1)
    assert conv2_counter == 2*(len(dilations)+1), f"got {conv2_counter} conv2d ops, expected {2*(len(dilations)+1)}"

    print("✅ saved all int8 kernels + int32 biases in", outdir)


# ───────────────────────── TRAINING ────────────────────────────────────────────
def train():
    # split train/val
    n  = len(pairs)
    idx= list(range(n))
    if n>=2:
        tr_i,va_i = train_test_split(idx,test_size=0.1,random_state=42)
    else:
        tr_i=va_i=idx
    train_pairs = [pairs[i] for i in tr_i]
    val_pairs   = [pairs[i] for i in va_i]
    train_durs  = [sf.info(s).frames for s,_ in train_pairs]
    val_durs    = [sf.info(s).frames for s,_ in val_pairs]

    def approx_steps(durs):
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # data generator
    def segment_generator(pairs_list, durs_list):
        while True:
            i       = random.randrange(len(pairs_list))
            sp,tp   = pairs_list[i]
            dur     = durs_list[i]
            offset  = random.randrange(max(dur-SEG_LEN,1))
            x,_     = sf.read(sp, start=offset, frames=SEG_LEN,
                              dtype='float32',always_2d=True)
            y,_     = sf.read(tp, start=offset, frames=SEG_LEN,
                              dtype='float32',always_2d=True)
            if x.shape[0]<SEG_LEN:
                pad=SEG_LEN-x.shape[0]
                x=np.pad(x,((0,pad),(0,0)))
                y=np.pad(y,((0,pad),(0,0)))
            x = x - np.mean(x)
            y = y - np.mean(y)
            yield x, y
            
    # 2) create a representative dataset generator
    def representative_data_gen():
        """Yields 100 random segments of shape (1, SEG_LEN, 1) from your source files."""
        for _ in range(100):
            # pick a random example
            i = random.randrange(len(pairs))
            src_path = pairs[i][0]
            dur      = train_durs[i]

            # pick a random offset so we can read SEG_LEN frames
            offset = random.randrange(max(dur - SEG_LEN, 1))

            # read exactly SEG_LEN frames from the source (piezo) file
            x, _ = sf.read(src_path,
                        start=offset,
                        frames=SEG_LEN,
                        dtype="float32",
                        always_2d=True)

            # if we’re too short (EOF), pad with zeros
            if x.shape[0] < SEG_LEN:
                pad_amt = SEG_LEN - x.shape[0]
                x = np.pad(x, ((0, pad_amt), (0, 0)))

            # mean‑remove, reshape to (1, SEG_LEN, 1)
            x = x - np.mean(x, axis=0, keepdims=True)
            x = x.reshape(1, SEG_LEN, 1)

            yield [x]

    sig = tf.TensorSpec((SEG_LEN,1),tf.float32)
    train_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_pairs,train_durs),
        output_signature=(sig,sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_pairs,val_durs),
        output_signature=(sig,sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # 1) Full-window FP32
    print("\n≈ Training FP32 model")
    static_model = build_resnet(dynamic_time=False)
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

    # 2) Dynamic FP32
    print("\n≈ Building + saving FP32 dynamic-time model")
    dynamic_model = build_resnet(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)

    # 3) Manual weight-only quantize
    print("\n≈ Quantizing weights (8-bit) in dynamic model")
    quant_model = tf.keras.models.clone_model(dynamic_model)
    quant_model.set_weights(dynamic_model.get_weights())
    quantize_weights_inplace(quant_model)
    path_q   = os.path.join(OUT_DIR, QUANT_MODEL_FILENAME)
    quant_model.save(path_q)
    print("✅ Saved weight-quantized dynamic model:", path_q)
    
    # Export layer weights into .npy files
    weights_dir = os.path.join(OUT_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1×1 Init PW
    w0, b0 = dynamic_model.get_layer("init_pw").get_weights()
    
    np.save(os.path.join(weights_dir, "init_pw_w.npy"), w0.astype(np.float32))
    np.save(os.path.join(weights_dir, "init_pw_b.npy"), b0.astype(np.float32))

    # Depthwise + PW for each residual block
    for idx, d in enumerate(DILATIONS):
        sep = dynamic_model.get_layer(f"sep_{idx}")
        pw  = dynamic_model.get_layer(f"pw_{idx}")

        # sep is the tf.keras.layers.SeparableConv1D
        # get_weights() returns [depthwise_kernel, pointwise_kernel, bias]
        dw_w, pw_w, pw_b = sep.get_weights()

        # save the depthwise kernel for your StatefulDepthwiseDilated1D
        np.save(os.path.join(weights_dir, f"dw{idx}_w.npy"), dw_w.astype(np.float32))

        # save the internal pointwise (and its bias) as a separate 1×1 conv
        np.save(os.path.join(weights_dir, f"pwsep{idx}_w.npy"), pw_w.astype(np.float32))
        np.save(os.path.join(weights_dir, f"pwsep{idx}_b.npy"), pw_b.astype(np.float32))
        
        pw = dynamic_model.get_layer(f"pw_{idx}")
        pw_w2, pw_b2 = pw.get_weights()
        np.save(os.path.join(weights_dir, f"pw{idx}_w.npy"), pw_w2.astype(np.float32))
        np.save(os.path.join(weights_dir, f"pw{idx}_b.npy"), pw_b2.astype(np.float32))

    # final 1×1
    wf, bf = dynamic_model.get_layer("final_pw").get_weights()
    path_final_pw_w = os.path.join(weights_dir, "final_pw_w.npy")
    path_final_pw_b = os.path.join(weights_dir, "final_pw_b.npy")
    np.save(path_final_pw_w, wf.astype(np.float32))
    np.save(path_final_pw_b, bf.astype(np.float32))
    
    # Now produce a TFLite INT8 model
    int8_weights_dir = os.path.join(OUT_DIR, "int8_weights")
    os.makedirs(int8_weights_dir, exist_ok=True)
    
    # 3) convert to TFLite with full INT8 quantization
    print("\n≈ Converting dynamic model to TFLite INT8 model for INT8 weight extraction")
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # ensure full integer (no float ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,    # plus their int8 versions
    ]
    
    # set I/O to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    extract_and_save_int8_weights(tflite_model, int8_weights_dir, DILATIONS)

    return (static_model, dynamic_model, quant_model)

# ───────────────────────── MAIN ────────────────────────────────────────────────
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--train":
        print("–––– TRAINING ––––")
        _, dynamic_model, quant_model = train()
    else:
        dpath = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
        qpath = os.path.join(OUT_DIR, QUANT_MODEL_FILENAME)
        if not os.path.exists(dpath) or not os.path.exists(qpath):
            print("Dynamic model not found. run: python piezo2mic.py --train")
            sys.exit(1)
        print("Loading dynamic model:", dpath)
        print("Loading quantized model:", qpath)
        dynamic_model = tf.keras.models.load_model(dpath, compile=False)
        quant_model   = tf.keras.models.load_model(qpath, compile=False)

    # ───────── DEMO ──────────────────────────────────────────────────────────────
    print("\nGenerating demo…")
    FRAMES = DEMO_SEC * SR
    pth_src, pth_tgt = pairs[0]
    x_long,_ = sf.read(pth_src, frames=FRAMES, dtype='float32', always_2d=True)
    y_ref,_  = sf.read(pth_tgt, frames=FRAMES, dtype='float32', always_2d=True)
    x_long   = x_long - np.mean(x_long)
    x_in     = x_long[np.newaxis,...]

    print("Running inference…")
    y_pred_dynamic = dynamic_model.predict(x_in, batch_size=1)[0,:,0]
    y_pred_quant = quant_model.predict(x_in, batch_size=1)[0,:,0]

    sf.write(os.path.join(OUT_DIR,"demo_input_piezo.wav"),      x_long, SR)
    sf.write(os.path.join(OUT_DIR,"demo_reference_mic.wav"),    y_ref,   SR)
    
    sf.write(os.path.join(OUT_DIR,"demo_output_pred_fp32.wav"), y_pred_dynamic,  SR)
    sf.write(os.path.join(OUT_DIR,"demo_output_pred_quant.wav"), y_pred_quant,  SR)
    print("✅ Demo files in", OUT_DIR)