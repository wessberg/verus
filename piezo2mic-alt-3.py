#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal, with pick‐attack oversampling ramp,
and a 2nd “style” channel so one model can handle finger vs pick.
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
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import (
    Input, Conv1D, SeparableConv1D, ZeroPadding1D,
    Activation, Multiply, Add, Lambda, Dense
)

# Try to enforce locked seeds so we can more reliably change config options and trust that the change in results was an effect of the change.
SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']  = '1'      # try to force TF→deterministic kernels
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'     # on some TF versions also needed

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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
EMBED_DIM           = 16

ALPHA_TRANSIENT  = 0.0   # how strongly to up-weight the ∣Δx∣ spikes
BETA_HF          = 0.0  # scale of the high-pass loss
HP_PRE_EMPHASIS  = 0.85 # coefficient for 1-sample 1st-order high-pass


PHASE_1_EPOCHS      = 50
#PHASE_2_EPOCHS      = 1
#PHASE_3_EPOCHS      = 1

PHASE_1_LR          = 1e-3
#PHASE_2_LR          = 1e-4
#PHASE_3_LR          = 1e-2

PHASE_1_LR_PATIENCE = 3
#PHASE_2_LR_PATIENCE = 7
#PHASE_3_LR_PATIENCE = 10

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

PICK_SRC_DIR   = "./input2/picked/source"
PICK_TGT_DIR   = "./input2/picked/target"
FINGER_SRC_DIR = "./input2/fingerpicked/source"
FINGER_TGT_DIR = "./input2/fingerpicked/target"

OUT_DIR        = "./out-alt-3"
CHECKPOINT_OUT_DIR = os.path.join(OUT_DIR, "checkpoints")
MODELS_OUT_DIR = os.path.join(OUT_DIR, "models")
DEMOS_OUT_DIR = os.path.join(OUT_DIR, "demos")

DEMO_SEC        = 60 * 2
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

STATIC_MODEL_FILENAME  = "piezo2mic_resnet.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_resnet_dynamic.keras"
SAVE_WEIGHTS           = False  # set to True to dump .npy weights

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_OUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUT_DIR, exist_ok=True)
os.makedirs(DEMOS_OUT_DIR, exist_ok=True)

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

audio_cache = {}
for (src,tgt) in all_pairs:
    audio_cache[src] = sf.read(src, dtype='float32', always_2d=True)[0]
    audio_cache[tgt] = sf.read(tgt, dtype='float32', always_2d=True)[0]

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

pick_chunks   = build_chunks(picked_pairs)
finger_chunks = build_chunks(finger_pairs)
all_chunks    = pick_chunks + finger_chunks
print(f"Generated {len(all_chunks)} chunks")

# --- Generators ---
def pick_segment_generator(pick_chunks, context_pre=1024):
    
    while True:
        src,tgt,cstart,cdur = random.choice(pick_chunks)
        src_wav = audio_cache[src]
        tgt_wav = audio_cache[tgt]
        
        window = src_wav[cstart : cstart + cdur, 0]
        diffs = np.abs(np.diff(window))
        
        if (diffs.size):
            thresh = np.median(diffs) * 3.0
            spikes = np.where(diffs > thresh)[0]
            
            if (spikes.size):
                idx = random.choice(spikes)
                off = max(0, min(int(idx-context_pre), cdur-SEG_LEN))
            else:
                off = random.randint(0, max(cdur-SEG_LEN, 0))
        else:
            off = random.randint(0, max(cdur-SEG_LEN, 0))

        read_start = cstart + off
        x = src_wav[read_start : read_start + SEG_LEN]
        y = tgt_wav[read_start : read_start + SEG_LEN]
        
        if x.shape[0]<SEG_LEN:
            p=SEG_LEN - x.shape[0]
            x=np.pad(x,((0,p),(0,0)))
            y=np.pad(y,((0,p),(0,0)))

        # style channel
        style = np.ones((SEG_LEN,1),dtype='float32')
        x = np.concatenate([x,style],axis=-1)
        yield x,y

def finger_segment_generator(finger_chunks, context_pre=1024):
    while True:
        src,tgt,cstart,cdur = random.choice(finger_chunks)
        src_wav = audio_cache[src]
        tgt_wav = audio_cache[tgt]

        window = src_wav[cstart : cstart + cdur, 0]
        diffs = np.abs(np.diff(window))
        
        if (diffs.size):
            thresh = np.median(diffs) * 3.0
            spikes = np.where(diffs > thresh)[0]
            if (spikes.size):
                idx = random.choice(spikes)
                off = max(0, min(int(idx-context_pre), cdur-SEG_LEN))
            else:
                off = random.randint(0, max(cdur-SEG_LEN, 0))
        else:
            off = random.randint(0, max(cdur-SEG_LEN, 0))
            

        thresh = np.median(diffs)*3; candidates = np.where(diffs>thresh)[0]
        if len(candidates)==0:
            off = random.randint(0, max(cdur-SEG_LEN,0))
        else:
            idx = random.choice(candidates)
            off = max(0, min(idx-context_pre, cdur-SEG_LEN))

        read_start = cstart + off
        x = src_wav[read_start : read_start + SEG_LEN]
        y = tgt_wav[read_start : read_start + SEG_LEN]

        if x.shape[0]<SEG_LEN:
            p=SEG_LEN-x.shape[0]
            x=np.pad(x,((0,p),(0,0)))
            y=np.pad(y,((0,p),(0,0)))

        style = np.zeros((SEG_LEN,1),dtype='float32')
        x = np.concatenate([x,style],axis=-1)
        yield x,y

def segment_generator(chunk_list):
    """Uniform random windows, style=1.0 if pick-path else 0.0."""
    while True:
        src, tgt, cstart, cdur = random.choice(chunk_list)
        src_wav = audio_cache[src]
        tgt_wav = audio_cache[tgt]

        off = random.randrange(max(cdur - SEG_LEN, 1))
        read_start = cstart + off
        x = src_wav[read_start : read_start + SEG_LEN]
        y = tgt_wav[read_start : read_start + SEG_LEN]

        if x.shape[0] < SEG_LEN:
            p = SEG_LEN - x.shape[0]
            x = np.pad(x, ((0,p),(0,0)))
            y = np.pad(y, ((0,p),(0,0)))

        style_flag = 1.0 if src.startswith(PICK_SRC_DIR) else 0.0
        style = np.full((SEG_LEN,1), style_flag, dtype='float32')
        x = np.concatenate([x, style], axis=-1)

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
    
    prediction_filepath = os.path.join(DEMOS_OUT_DIR, prediction_name + ".wav")
    sf.write(prediction_filepath, y_pred_dyn, SR, subtype='FLOAT')
    
    if (not only_prediction):
        source_filepath = os.path.join(DEMOS_OUT_DIR, "demo_source" +".wav")
        target_filepath = os.path.join(DEMOS_OUT_DIR, "demo_target" + ".wav")
        sf.write(source_filepath,
             x_long, SR, subtype='FLOAT')
        sf.write(target_filepath,
                y_ref, SR, subtype='FLOAT')
    
    print(f"✅ Demo saved to {prediction_filepath}")

@register_keras_serializable()
def waveform_l1_loss(y_true, y_pred):
    return tf.abs(y_true - y_pred)

@register_keras_serializable()
def hf_loss(y_true, y_pred):
    α = HP_PRE_EMPHASIS
    # 1st‐order high-pass (pre-emphasis):
    true_hp = y_true[:,1:,:] - α*y_true[:,:-1,:]
    pred_hp = y_pred[:,1:,:] - α*y_pred[:,:-1,:]
    true_hp = tf.pad(true_hp, [[0,0],[1,0],[0,0]])
    pred_hp = tf.pad(pred_hp, [[0,0],[1,0],[0,0]])
    return tf.abs(true_hp - pred_hp)

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    # returns (batch, T, 1)
    td = waveform_l1_loss(y_true, y_pred)
    hf = hf_loss(y_true, y_pred)
    return td + BETA_HF * hf

# 3) A proper named metric instead of a lambda:
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    # This just reduces hf_loss to a scalar
    return tf.reduce_mean(hf_loss(y_true, y_pred))

# ──────────────────────────────────────────────────────────────────────────────
@register_keras_serializable()
def add_sw(x, y):
    # x: (batch, T, 2) → x[...,0]=piezo, x[...,1]=style flag (0=finger,1=pick)
    piezo = x[...,0]            # (batch, T)
    style = x[...,1]            # (batch, T)

    # 1) extract one scalar per example in the batch
    #    (all time‐samples are identical so max or mean both work)
    style_flag = tf.reduce_max(style, axis=1)   # (batch,)

    # 2) define your base weights for each style
    #    here we set both to 1.0 (i.e. no style-bias), but you can
    #    make pick_weight != finger_weight if you like.
    pick_weight   = 1.0
    finger_weight = 1.0

    #    this yields a shape (batch,) where finger examples get
    #    finger_weight, pick examples get pick_weight
    base_scalar = pick_weight*style_flag + finger_weight*(1.0 - style_flag)
    #––– if you truly want them *exactly* equal, you can just do:
    #    base_scalar = tf.ones_like(style_flag)

    # 3) build your transient re-weighting term exactly as before
    Δ = tf.abs(piezo[:,1:] - piezo[:,:-1])        # (batch, T-1)
    Δ = tf.pad(Δ, [[0,0],[1,0]], constant_values=0)  # back to (batch, T)
    m = tf.reduce_max(Δ, axis=1, keepdims=True)   # (batch,1)
    m = tf.maximum(m, 1e-6)                       # avoid div0
    wn = 1.0 + ALPHA_TRANSIENT * (Δ / m)          # (batch, T)

    # 4) combine them
    sw = base_scalar[:, None] * wn                # (batch, T)
    sw = tf.expand_dims(sw, axis=-1)              # (batch, T, 1)

    return x, y, sw

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
def extract_style_scalar(x):
    # x has shape (batch, T, 1) → we grab the first time‐step value
    # but keep the last dim so output is (batch, 1)
    return x[:, 0:1, 0:1]       # now shape (batch, 1)

@register_keras_serializable()
def expand_time_axis(x):
    # x has shape (batch, features) → we want (batch, 1, features)
    return tf.expand_dims(x, axis=1)

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

@register_keras_serializable()
def slice_piezo(x):
    # Takes channel 0 of a (batch, time, 2) tensor
    return x[:, :, 0:1]

@register_keras_serializable()
def slice_film_gamma(x, filters):
    # x shape: (batch, T, 2*filters) → return (batch, T, filters)
    return x[:, :, :filters]

@register_keras_serializable()
def slice_film_beta(x, filters):
    # x shape: (batch, T, 2*filters) → return (batch, T, filters)
    return x[:, :, filters:]

@register_keras_serializable()
def slice_style(x):
    # Takes channel 1 of a (batch, time, 2) tensor
    return x[:, :, 1:2]

@register_keras_serializable()
def invert(x):
    return 1.0 - x


def build_model(dynamic_time=False):
    T = None if dynamic_time else SEG_LEN
    inp = Input(shape=(T,2), name="piezo_plus_style")
    style = Lambda(slice_style, name="take_style")(inp)                   # (batch, T, 1)
    piezo = Lambda(slice_piezo, name="take_piezo")(inp)
    # derive scalar style flag
    style_scalar = Lambda(extract_style_scalar, name="style_scalar")(style)  # (batch,)
    # style embedding
    style_embed = Dense(EMBED_DIM, activation='linear', name="style_embed")(style_scalar)
    # initial 1x1 projection on both channels
    x = Conv1D(FILTERS, 1, padding='same', name="init_pw")(inp)

    for idx, d in enumerate(DILATIONS):
        res = x
        pad = (KERNEL_SIZE - 1) * d
        x = ZeroPadding1D((pad,0), name=f"pad_{idx}")(x)
        g = SeparableConv1D(FILTERS*2, KERNEL_SIZE, dilation_rate=d,
                            padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':FILTERS},
                   name=f"slice_content_{idx}")(g)
        g2 = Lambda(slice_gate, arguments={'filters':FILTERS},
                    name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([
                Activation('tanh', name=f"tanh_{idx}")(c),
                Activation('sigmoid', name=f"sigmoid_{idx}")(g2)
            ])
        x = Conv1D(FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        # FiLM modulation

        film_params = Dense(FILTERS*2, name=f"film_dense_{idx}")(style_embed)
        gamma = Lambda(slice_film_gamma,
                    arguments={'filters': FILTERS},
                    name=f"film_gamma_{idx}")(film_params)
        beta  = Lambda(slice_film_beta,
                    arguments={'filters': FILTERS},
                    name=f"film_beta_{idx}")(film_params)

        x = Add(name=f"film_apply_{idx}")([
            Multiply(name=f"film_scale_{idx}")([x, gamma]),
            beta
        ])

    main_pick   = Conv1D(1,1,padding='same',name="final_pick")(x)
    main_finger = Conv1D(1,1,padding='same',name="final_finger")(x)
    pick_c   = Multiply(name="pick_contrib")([style, main_pick])
    finger_c = Multiply(name="finger_contrib")([
        Lambda(invert, name="invert_style")(style), main_finger])
    mixed    = Add(name="mix_main")([pick_c, finger_c])
    skip     = Conv1D(1,1,use_bias=False,
                      kernel_initializer='ones',
                      name="skip_conv")(piezo)
    out = Add(name="res_out")([mixed, skip])
    return tf.keras.models.Model(inp, out, name="piezo2mic_conditional")

def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations, filters):
    # TODO
    print("TODO")

# ─────────────────────────────── TRAINING ──────────────────────────────────────
def train():

    # train/val split
    idx = list(range(len(all_chunks)))

    if len(idx)>1:
        tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=SEED)
    else:
        raise RuntimeError("Not enough data to split into train/val sets.")

    train_chunks  = [all_chunks[i] for i in tr_i]
    val_chunks    = [all_chunks[i] for i in va_i]
    train_pick_chunks   = [c for c in train_chunks if c in pick_chunks]
    train_finger_chunks = [c for c in train_chunks if c in finger_chunks]
    val_pick_chunks     = [c for c in val_chunks   if c in pick_chunks]
    val_finger_chunks   = [c for c in val_chunks   if c in finger_chunks]
    

    def approx_steps(durs):
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE
    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur for (_,_,_,dur) in val_chunks]
    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    # datasets
    sig_x = tf.TensorSpec((SEG_LEN,2), tf.float32)
    sig_y = tf.TensorSpec((SEG_LEN,1), tf.float32)

    pick_ds    = tf.data.Dataset.from_generator(
        lambda: pick_segment_generator(train_pick_chunks), output_signature=(sig_x,sig_y)
    )
    finger_ds  = tf.data.Dataset.from_generator(
        lambda: finger_segment_generator(train_finger_chunks), output_signature=(sig_x,sig_y)
    )

    train_ds_1 = tf.data.Dataset.sample_from_datasets(
        [pick_ds, finger_ds], weights=[0.5, 0.5]
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)

    train_ds_1 = train_ds_1.map(add_sw).repeat().prefetch(tf.data.AUTOTUNE)

    # validation streams
    val_pick_ds_1 = tf.data.Dataset.from_generator(
        lambda: pick_segment_generator(val_pick_chunks),
        output_signature=(sig_x, sig_y)
    )
    val_finger_ds_1 = tf.data.Dataset.from_generator(
        lambda: finger_segment_generator(val_finger_chunks),
        output_signature=(sig_x, sig_y)
    )
    
    val_ds_1 = tf.data.Dataset.sample_from_datasets(
        [val_pick_ds_1, val_finger_ds_1],
        weights=[0.5, 0.5],
        seed=SEED
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    # compile & train
    static_model = build_model(dynamic_time=False)
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_1_LR),
        loss=waveform_l1_loss,
        metrics=["mae"]
    )
    
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_1_LR),
        loss=combined_loss,
        metrics=[
                "mae",
                mean_hf_metric
        ]
    )
    static_model.summary()
    
    # Early stopping callback
    es1 = EarlyStopping(monitor='val_mae', mode='min', verbose=1, patience=PHASE_1_LR_PATIENCE, restore_best_weights=True)
    
    # ModelCheckpoint: save whenever val_loss improves
    cp1 = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_OUT_DIR,
                            "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras"),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    history1 = static_model.fit(
        train_ds_1,
        epochs=3 if DEBUG else PHASE_1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds_1,
        validation_steps=validation_steps,
        callbacks=[es1, cp1, TensorBoard(log_dir="./logs")]
    )
    
    

    # style-specific evaluation
    #print("\nEvaluating pick style on val set:")
    #static_model.evaluate(val_pick_ds_1, steps=validation_steps)
    #print("\nEvaluating finger style on val set:")
    #static_model.evaluate(val_finger_ds_1, steps=validation_steps)
    
    phase1_trained_epochs = len(history1.history["loss"])

    # ── Save FP32 static & dynamic models ──────────────────────────────────
    path_static = os.path.join(MODELS_OUT_DIR, STATIC_MODEL_FILENAME)
    static_model.save(path_static)
    print("✅ Saved FP32 static model:", path_static)

    dynamic_model = build_model(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(MODELS_OUT_DIR, DYNAMIC_MODEL_FILENAME)
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
        dpath = os.path.join(MODELS_OUT_DIR, DYNAMIC_MODEL_FILENAME)
        if not os.path.exists(dpath):
            print("Dynamic model not found. run: python piezo2mic.py --train")
            sys.exit(1)
        dynamic_model = tf.keras.models.load_model(
            dpath,
            compile=False
        )
        
        generate_demo(dynamic_model)