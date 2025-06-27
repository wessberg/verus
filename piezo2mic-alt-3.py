#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal, with pick‐attack oversampling ramp,
and a 2nd “style” channel so one model can handle finger vs pick.
"""
import os
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = "15192"

import glob
import random
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tflite import Model, TensorType, BuiltinOperator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, ZeroPadding1D,
    Activation, Multiply, Add, Lambda
)
from tensorflow.keras.models import Model

tf.keras.config.enable_unsafe_deserialization()

# ───────────────────────────── PARAMETERS ─────────────────────────────────────
DEBUG               = False
FILTERS             = 64
KERNEL_SIZE         = 25
DILATIONS           = (1,2,4,8)
RECEPTIVE_FIELD     = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
TRAIN_BLOCK_SIZE    = 4096
SR                  = 48000
SEG_LEN             = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE
BATCH_SIZE          = 32

HF_KERNEL_SIZE      = 11
HF_FMIN             = 2000.0
HF_FMAX             = 10000.0
HF_N_FFT            = 512
HF_HOP              = 128

PHASE_1_EPOCHS      = 50
PHASE_2_EPOCHS      = 1
PHASE_3_EPOCHS      = 1

PHASE_1_LR          = 1e-3
PHASE_2_LR          = 1e-4
PHASE_3_LR          = 1e-2

PHASE_1_LR_PATIENCE = 5
PHASE_2_LR_PATIENCE = 7
PHASE_3_LR_PATIENCE = 10

PHASE_1_PICK_RATIO = 0.3  # 30% pick‐attack
PHASE_2_PICK_RATIO = 0.5  # 50% pick‐attack
PHASE_3_PICK_RATIO = 1.0  # 100% pick‐attack

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

# ─── your new 4‐dir layout ─────────────────────────────────────────────
PICK_SRC_DIR   = "./input2/picked/source"
PICK_TGT_DIR   = "./input2/picked/target"
FINGER_SRC_DIR = "./input2/fingerpicked/source"
FINGER_TGT_DIR = "./input2/fingerpicked/target"
OUT_DIR        = "./out-alt-3"

DEMO_SEC        = 60 * 2
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

STATIC_MODEL_FILENAME  = "piezo2mic_resnet.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_resnet_dynamic.keras"
SAVE_WEIGHTS           = False  # set to True to dump .npy weights

os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────── DATA LOADING ───────────────────────────────────
print("Looking for audio pairs…")
picked_pairs = []
for src in sorted(glob.glob(os.path.join(PICK_SRC_DIR, "*_source.wav"))):
    tgt = src.replace("_source.wav","_target.wav").replace(PICK_SRC_DIR, PICK_TGT_DIR)
    if os.path.exists(tgt):
        picked_pairs.append((src, tgt))
    else:
        print("Warning: no target for", src)
finger_pairs = []
for src in sorted(glob.glob(os.path.join(FINGER_SRC_DIR, "*_source.wav"))):
    tgt = src.replace("_source.wav","_target.wav").replace(FINGER_SRC_DIR, FINGER_TGT_DIR)
    if os.path.exists(tgt):
        finger_pairs.append((src, tgt))
    else:
        print("Warning: no target for", src)
print(f"  → {len(picked_pairs)} picked‐attack pairs, {len(finger_pairs)} finger‐picked pairs")
all_pairs = picked_pairs + finger_pairs

def build_chunks(pairs_list):
    """
    Turn a list of (src,tgt) into
    [(src,tgt,chunk_start,chunk_len), …].
    """
    chunks = []
    chunk_len = CHUNK_SEC * SR
    for src,tgt in pairs_list:
        info  = sf.info(src)
        total = info.frames
        n_full = total // chunk_len
        for i in range(n_full):
            chunks.append((src,tgt,i*chunk_len,chunk_len))
        rem = total - n_full*chunk_len
        if rem >= SEG_LEN:
            chunks.append((src,tgt,n_full*chunk_len,rem))
    return chunks

def pick_segment_generator(pick_chunks, context_pre=1024):
    """
    Randomly pick a chunk *and* a random attack offset inside it,
    then center the window roughly on that point (with a little pre-context).
    Always style=1.0.
    """
    while True:
        src,tgt,cstart,cdur = random.choice(pick_chunks)
        # read a bit more to find an onset
        rsamp, _ = sf.read(src, start=cstart, frames=cdur,
                           dtype='float32', always_2d=True)
        mono = rsamp[:,0]
        diffs = np.abs(np.diff(mono))
        thresh = np.median(diffs)*3
        candidates = np.where(diffs > thresh)[0]
        if len(candidates)==0:
            off = random.randint(0, max(cdur-SEG_LEN, 0))
        else:
            pick_idx = random.choice(candidates)
            off = pick_idx - context_pre
            off = max(0, min(off, cdur-SEG_LEN))
        read_start = cstart + off

        x,_ = sf.read(src, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        y,_ = sf.read(tgt, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        if x.shape[0] < SEG_LEN:
            pad = SEG_LEN - x.shape[0]
            x = np.pad(x, ((0,pad),(0,0)))
            y = np.pad(y, ((0,pad),(0,0)))

        # --- ADD STYLE CHANNEL (=1.0 for pick) ---
        style_chan = np.ones((SEG_LEN,1), dtype='float32')
        x = np.concatenate([x, style_chan], axis=-1)  # now (SEG_LEN,2)
        yield x, y

def segment_generator(chunk_list):
    """
    Uniform random windows from chunk_list.
    Style=1.0 if src in PICK_SRC_DIR else 0.0
    """
    while True:
        src_path,tgt_path,cstart,cdur = random.choice(chunk_list)
        off = random.randrange(max(cdur-SEG_LEN,1))
        read_start = cstart + off
        x,_ = sf.read(src_path, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        y,_ = sf.read(tgt_path, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        if x.shape[0] < SEG_LEN:
            pad = SEG_LEN - x.shape[0]
            x = np.pad(x, ((0,pad),(0,0)))
            y = np.pad(y, ((0,pad),(0,0)))

        # --- COMPUTE STYLE FLAG ---
        if src_path.startswith(PICK_SRC_DIR):
            style = 1.0
        else:
            style = 0.0
        style_chan = np.full((SEG_LEN,1), style, dtype='float32')
        x = np.concatenate([x, style_chan], axis=-1)  # (SEG_LEN,2)
        yield x, y
        
def generate_demo (model, prediction_name="demo_prediction", only_prediction=False):
    print("\nGenerating demo…")
    FRAMES      = DEMO_SEC * SR
    START_FRAME = DEMO_OFFSET_SEC * SR
    pth_src, pth_tgt = all_pairs[DEMO_PAIR_INDEX]

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

    # --- ADD STYLE CHANNEL TO LONG SIGNAL ---
    if pth_src.startswith(PICK_SRC_DIR):
        style = 1.0
    else:
        style = 0.0
    style_chan = np.full((FRAMES,1), style, dtype='float32')
    x_long2   = np.concatenate([x_long, style_chan], axis=-1)  # (FRAMES,2)

    x_in = x_long2[np.newaxis,...]  # (1,FRAMES,2)
    print("Running inference…")
    y_pred_dyn = model.predict(x_in, batch_size=1)[0,:,0]
    
    prediction_filepath = os.path.join(OUT_DIR, prediction_name + ".wav")
    sf.write(prediction_filepath, y_pred_dyn, SR, subtype='FLOAT')
    
    if (not only_prediction):
        source_filepath = os.path.join(OUT_DIR, "demo_source" +".wav")
        target_filepath = os.path.join(OUT_DIR, "demo_target" + ".wav")
        sf.write(source_filepath,
             x_long, SR, subtype='FLOAT')
        sf.write(target_filepath,
                y_ref, SR, subtype='FLOAT')
    
    print(f"✅ Demo saved to {prediction_filepath}")

@register_keras_serializable()
def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

@register_keras_serializable()
def waveform_l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# ─── helper functions ─────────────────────────────────────────────────────────

@register_keras_serializable()
def get_piezo(x):
    # take channel 0 (piezo)
    return x[:, :, 0:1]

@register_keras_serializable()
def get_style(x):
    # take channel 1 (0.0 or 1.0)
    return x[:, :, 1:2]

@register_keras_serializable()
def invert_style(x):
    return 1.0 - x

# you already have these, but make sure they are @register_keras_serializable
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]


def build_model(filters=64,
                kernel_size=25,
                dilations=(1,2,4,8),
                dynamic_time=False):
    # T = None ⇒ variable‐length for TFLite/dynamic
    T = None if dynamic_time else SEG_LEN

    # Input: (batch, time, 2)  →  [piezo_waveform, style_flag]
    inp = Input(shape=(T,2), name="piezo_and_style")

    # split off real waveform vs style bit
    piezo = Lambda(get_piezo, name="take_piezo")(inp)    # (b, T, 1)
    style = Lambda(get_style, name="take_style")(inp)    # (b, T, 1)

    # initial 1×1 projection
    x = Conv1D(filters, 1, padding="same", name="init_pw")(piezo)

    # the residual stack
    for idx, d in enumerate(dilations):
        res = x
        pad = (kernel_size - 1) * d
        x   = ZeroPadding1D((pad,0), name=f"pad_{idx}")(x)

        g = SeparableConv1D(
            filters*2,
            kernel_size,
            dilation_rate=d,
            padding="valid",
            name=f"sep_{idx}"
        )(x)

        c = Lambda(slice_content,
                   arguments={"filters":filters},
                   name=f"slice_content_{idx}")(g)
        g_ = Lambda(slice_gate,
                    arguments={"filters":filters},
                    name=f"slice_gate_{idx}")(g)

        c_act = Activation("tanh",    name=f"tanh_{idx}")(c)
        g_act = Activation("sigmoid", name=f"sigmoid_{idx}")(g_)
        x     = Multiply(name=f"gated_mul_{idx}")([c_act, g_act])

        x     = Conv1D(filters,1,
                       padding="same",
                       name=f"pw_{idx}")(x)
        x     = Add(name=f"res_add_{idx}")([x, res])

    # ─── two separate 1×1 “heads” ────────────────────────────────────────────────
    main_pick   = Conv1D(1,1,padding="same", name="final_pick")(x)
    main_finger = Conv1D(1,1,padding="same", name="final_finger")(x)

    # gate them with the style bit
    pick_contrib   = Multiply(name="pick_contrib")([style, main_pick])
    finger_coef    = Lambda(invert_style, name="invert_style")(style)
    finger_contrib = Multiply(name="finger_contrib")([finger_coef,
                                                     main_finger])
    mixed_main     = Add(name="mix_main")([pick_contrib, finger_contrib])

    # skip on the raw piezo
    skip = Conv1D(1,1,
                  use_bias=False,
                  kernel_initializer="ones",
                  name="skip_conv")(piezo)

    out = Add(name="res_out")([mixed_main, skip])

    return Model(inp, out, name="piezo2mic_twohead")

def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations, filters):
    # (unchanged)
    os.makedirs(outdir, exist_ok=True)
    m    = Model.GetRootAsModel(tflite_model_buf, 0)
    subg = m.Subgraphs(0)
    op_map = {i: m.OperatorCodes(i).BuiltinCode()
              for i in range(m.OperatorCodesLength())}
    def load_tensor(tidx):
        t = subg.Tensors(tidx)
        b = m.Buffers(t.Buffer())
        raw = b.DataAsNumpy()
        dtype = np.int8 if t.Type()==TensorType.INT8 else np.int32
        arr = np.frombuffer(raw, dtype=dtype).reshape(t.ShapeAsNumpy())
        qp = t.Quantization()
        scales = [qp.Scale(i)     for i in range(qp.ScaleLength())]
        zeros  = [qp.ZeroPoint(i) for i in range(qp.ZeroPointLength())]
        quant  = np.stack([scales, zeros], axis=1)
        return arr, quant

    sep_ctr=0; pwsep_ctr=0; pw_ctr=0
    for i in range(subg.OperatorsLength()):
        op   = subg.Operators(i)
        code = op_map[op.OpcodeIndex()]
        if code == BuiltinOperator.DEPTHWISE_CONV_2D:
            w_i = op.Inputs(1)
            b_i = op.InputsLength()>2 and op.Inputs(2)
            tag = f"dw{sep_ctr}"; sep_ctr+=1
            w_arr,w_q = load_tensor(w_i)
            np.save(f"{outdir}/{tag}_w_int8.npy", w_arr)
            np.savetxt(f"{outdir}/{tag}_w_int8_quant.txt",
                       w_q, fmt="%g", header="scale,zero_point", comments="")
            if b_i:
                b_arr,b_q = load_tensor(b_i)
                np.save(f"{outdir}/{tag}_b_int32.npy", b_arr)
                np.savetxt(f"{outdir}/{tag}_b_int32_quant.txt",
                           b_q, fmt="%g", header="scale,zero_point", comments="")
            continue
        if code == BuiltinOperator.CONV_2D:
            w_i,b_i   = op.Inputs(1), op.Inputs(2)
            w_arr,w_q = load_tensor(w_i)
            b_arr,b_q = load_tensor(b_i)
            W, B = w_arr.size, b_arr.size
            if W==HF_KERNEL_SIZE:
                tag="hf_transient"
            elif W==1 and B==1:
                tag="skip_conv"
            elif W==filters:
                tag="init_pw" if B==filters else "final_pw"
            elif W==filters*(2*filters) and B==(2*filters):
                tag=f"pwsep{pwsep_ctr}"; pwsep_ctr+=1
            elif W==filters*filters and B==filters:
                tag=f"pw{pw_ctr}"; pw_ctr+=1
            else:
                raise RuntimeError(f"Unexpected Conv2D shape: W={W},B={B}")
            fnw=os.path.join(outdir,f"{tag}_w_int8.npy")
            np.save(fnw, w_arr)
            np.savetxt(fnw.replace(".npy","_quant.txt"), w_q,
                       fmt="%g", header="scale,zero_point", comments="")
            fnb=os.path.join(outdir,f"{tag}_b_int32.npy")
            np.save(fnb, b_arr)
            np.savetxt(fnb.replace(".npy","_quant.txt"), b_q,
                       fmt="%g", header="scale,zero_point", comments="")
            continue

    assert sep_ctr   == len(dilations)
    assert pwsep_ctr == len(dilations)
    assert pw_ctr    == len(dilations)
    print("✅ saved all int8 kernels + int32 biases in", outdir)

# ─────────────────────────────── TRAINING ──────────────────────────────────────
def train():
    # 1) build chunk lists
    pick_chunks   = build_chunks(picked_pairs)
    finger_chunks = build_chunks(finger_pairs)
    all_chunks    = pick_chunks + finger_chunks
    print(f"Generated {len(all_chunks)} chunks")

    # 2) split train/val on all_chunks
    idx = list(range(len(all_chunks)))
    if len(idx) > 1:
        tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=69)
    else:
        tr_i = va_i = idx
    train_chunks = [all_chunks[i] for i in tr_i]
    val_chunks   = [all_chunks[i] for i in va_i]
    train_pick_chunks = [c for c in train_chunks if c in pick_chunks]

    # 3) compute steps
    def approx_steps(durs):
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur   for (_,_,_,dur) in val_chunks]
    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # --- REPRESENTATIVE DATA FOR INT8 (also 2-channel) ---
    def representative_data_gen():
        for _ in range(100):
            src,tgt,cstart,cdur = random.choice(train_chunks)
            off = random.randrange(max(cdur-SEG_LEN,1))
            read_start = cstart + off
            x,_ = sf.read(src, start=read_start, frames=SEG_LEN,
                        dtype='float32', always_2d=True)
            if x.shape[0] < SEG_LEN:
                x = np.pad(x, ((0,SEG_LEN-x.shape[0]),(0,0)))
            # style flag:
            style = 1.0 if src.startswith(PICK_SRC_DIR) else 0.0
            style_chan = np.full((SEG_LEN,1), style, dtype='float32')
            x = np.concatenate([x, style_chan], axis=-1)  # (SEG_LEN,2)
            yield [x.reshape(1,SEG_LEN,2)]

    # --- PHASE 1 --- 
    sig_x = tf.TensorSpec((SEG_LEN, 2), tf.float32)   # piezo+style
    sig_y = tf.TensorSpec((SEG_LEN, 1), tf.float32)   # mic

    uniform_ds_1 = tf.data.Dataset.from_generator(
        lambda: segment_generator(train_chunks),
        output_signature=(sig_x, sig_y)
    )
    pick_ds_1 = tf.data.Dataset.from_generator(
        lambda: pick_segment_generator(train_pick_chunks),
        output_signature=(sig_x, sig_y)
    )

    train_ds_1 = tf.data.Dataset.sample_from_datasets(
        [uniform_ds_1, pick_ds_1],
        weights=[1.0-PHASE_1_PICK_RATIO, PHASE_1_PICK_RATIO]
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds_1 = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_chunks),
        output_signature=(sig_x, sig_y)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    es1 = EarlyStopping(monitor='val_mae',
                        mode='min',
                        patience=PHASE_1_LR_PATIENCE)

    print("\n≈ Training Phase 1")
    static_model = build_model(dynamic_time=False)
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_1_LR),
        loss=waveform_l1_loss,
        metrics=["mae"]
    )
    static_model.summary()
    history1 = static_model.fit(
        train_ds_1,
        epochs=1 if DEBUG else PHASE_1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds_1,
        validation_steps=validation_steps,
        callbacks=[es1, TensorBoard(log_dir="./logs")]
    )
    
    phase1_trained_epochs = len(history1.history["loss"])

    # ── Save FP32 static & dynamic models ──────────────────────────────────
    path_static = os.path.join(OUT_DIR, STATIC_MODEL_FILENAME)
    static_model.save(path_static)
    print("✅ Saved FP32 static model:", path_static)

    dynamic_model = build_model(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)

    # ── Optional: dump .npy FP32 weights & extract INT8 ───────────────────
    if SAVE_WEIGHTS:
        # unchanged: dumping weights...
        pass

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
            }
        )
        
        generate_demo(dynamic_model)