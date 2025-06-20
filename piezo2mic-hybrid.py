#!/usr/bin/env python3
"""
Hybrid IR‐DSP + low‐latency NN for piezo → mic’d acoustic guitar.

Usage:
  python piezo2mic_hybrid.py --train   # estimate low-pass IR + train
  python piezo2mic_hybrid.py           # run demo, dump aligned WAVs
"""
import os, sys, glob, random
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, SeparableConv1D, ZeroPadding1D,
    Add, Multiply, Activation, Lambda
)
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.utils import register_keras_serializable

# ───────────────────────── PARAMETERS ──────────────────────────────────────────
SR               = 48000
FILTERS          = 48             # more capacity in residual net
KERNEL_SIZE      = 17
DILATIONS        = (1,2,4,8)      # RF ≃ 241 samples ≃ 5 ms
TRAIN_BLOCK_SIZE = 4096
BATCH_SIZE       = 32
EPOCHS           = 20
PATIENCE         = 5
SHUFFLE_BUF      = 1000

IR_LEN         = 2048
IR_FFT_SIZE    = 4096
IR_CUTOFF_HZ   = 2000.0           # low-pass cutoff for body IR

DEMO_SEC            = 60 * 2
DEMO_OFFSET_SEC     = 0
DEMO_PAIR_INDEX     = 0

SRC_DIR    = "./input/source"
TGT_DIR    = "./input/target"
OUT_DIR    = "./out-hybrid"
MODEL_FILE = "piezo2mic_hybrid.keras"
IR_FILE    = "ir.npy"

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────── UTILITIES ───────────────────────────────────────────
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]


def estimate_ir(pairs, ir_len, fft_size, cutoff_hz=None):
    """
    Estimate global IR by least-squares deconvolution,
    then optionally low-pass it below cutoff_hz with a cosine taper.
    """
    # accumulate X*Y and |X|^2
    num = np.zeros(fft_size//2+1, dtype=np.complex64)
    den = np.zeros(fft_size//2+1, dtype=np.float32)

    for sp, tp in pairs:
        N = sf.info(sp).frames
        off = random.randrange(max(N - ir_len, 1))
        x,_ = sf.read(sp, start=off, frames=ir_len,
                      dtype='float32', always_2d=True)
        y,_ = sf.read(tp, start=off, frames=ir_len,
                      dtype='float32', always_2d=True)
        if x.shape[0] < ir_len:
            p = ir_len - x.shape[0]
            x = np.pad(x, ((0,p),(0,0)))
            y = np.pad(y, ((0,p),(0,0)))
        x = x[:,0];  y = y[:,0]

        w  = np.hanning(ir_len)
        Xf = np.fft.rfft(x*w, n=fft_size)
        Yf = np.fft.rfft(y*w, n=fft_size)
        num += np.conj(Xf) * Yf
        den += np.abs(Xf)**2

    Hf = num / (den + 1e-8)

    # smooth low-pass in frequency if requested
    if cutoff_hz is not None:
        bins  = Hf.shape[0]
        freqs = np.linspace(0, SR/2, bins)
        roll  = cutoff_hz * 0.1
        mask  = np.zeros_like(freqs, dtype=np.float32)
        # full passband
        mask[freqs <= cutoff_hz] = 1.0
        # cosine taper in transition band
        idx = np.logical_and(freqs > cutoff_hz, freqs < cutoff_hz + roll)
        mask[idx] = 0.5 * (1.0 + np.cos(np.pi * (freqs[idx] - cutoff_hz) / roll))
        Hf *= mask.astype(np.complex64)

    h_full = np.fft.irfft(Hf, n=fft_size)
    h = h_full[:ir_len]
    return h


@register_keras_serializable()
class FixedIR(tf.keras.layers.Layer):
    """
    A fixed, causal FIR conv whose taps = the low-pass IR.
    We reverse taps so that tf.nn.conv1d does a true causal conv.
    """
    def __init__(self, ir, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ir_taps = np.array(ir, dtype=np.float32)
        self.ir_len  = int(self.ir_taps.shape[0])
        rev = self.ir_taps[::-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.ir_len,1,1),
            initializer=tf.keras.initializers.Constant(
                rev.reshape(self.ir_len,1,1)
            ),
            trainable=False
        )

    def call(self, x):
        xpad = tf.pad(x, [[0,0],[self.ir_len-1,0],[0,0]])
        return tf.nn.conv1d(xpad, self.kernel, stride=1, padding='VALID')

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ir": self.ir_taps.tolist()})
        return cfg


@register_keras_serializable()
class HFBYPASS(tf.keras.layers.Layer):
    """
    A tiny causal FIR for instant pick-attack bypass.
    """
    def __init__(self, kernel_size=33, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ks = int(kernel_size)
        # start at zero so bypass only learns what is missing
        self.hf_taps = self.add_weight(
            name="hf_taps",
            shape=(self.ks,1,1),
            initializer=Zeros(),
            trainable=True
        )

    def call(self, x):
        xpad = tf.pad(x, [[0,0],[self.ks-1,0],[0,0]])
        k    = tf.reverse(self.hf_taps, axis=[0])
        return tf.nn.conv1d(xpad, k, stride=1, padding='VALID')

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernel_size": self.ks})
        return cfg


def residual_branch(inp, filters, ksize, dilations):
    """
    Tiny gated ResNet → predicts the mid-to-high-freq residual.
    """
    x = Conv1D(filters, 1, padding='same', name="init_pw")(inp)
    for i, d in enumerate(dilations):
        res = x
        pad = (ksize-1)*d
        x   = ZeroPadding1D((pad,0), name=f"pad_{i}")(x)
        g   = SeparableConv1D(2*filters,
                              ksize,
                              dilation_rate=d,
                              padding='valid',
                              name=f"sep_{i}"
                             )(x)
        c   = Lambda(slice_content, arguments={'filters':filters},
                     name=f"content_{i}")(g)
        ga  = Lambda(slice_gate, arguments={'filters':filters},
                     name=f"gate_{i}")(g)
        x   = Multiply(name=f"gated_mul_{i}")([
                  Activation('tanh')(c),
                  Activation('sigmoid')(ga)
              ])
        x   = Conv1D(filters,1,padding='same',name=f"pw_{i}")(x)
        x   = Add(name=f"res_add_{i}")([x,res])
    return Conv1D(1,1,padding='same', name="residual_raw")(x)


def build_hybrid_model(ir):
    """
    piezo_in → FixedIR → body
           → HFBYPASS → hf
           → residual_branch → raw_resid → resid_gain → scaled_resid
         sum([body, hf, scaled_resid]) → hybrid_out
    """
    inp        = Input((None,1), name="piezo_in")
    body       = FixedIR(ir, name="ir_conv")(inp)
    hf         = HFBYPASS(33, name="hf_bypass")(inp)
    raw_resid  = residual_branch(inp, FILTERS, KERNEL_SIZE, DILATIONS)
    scaled_res = Conv1D(1,1,
                       use_bias=False,
                       kernel_initializer=Zeros(),    # start resid at zero
                       name="resid_gain")(raw_resid)
    out        = Add(name="hybrid_out")([body, hf, scaled_res])
    return Model(inp, out, name="piezo2mic_hybrid")


# ─────────────────────────────────── TRAIN ──────────────────────────────────────
def train():
    # gather file pairs
    srcs = sorted(glob.glob(os.path.join(SRC_DIR,"*_source.wav")))
    pairs=[]
    for s in srcs:
        t = s.replace("_source.wav","_target.wav").replace(SRC_DIR,TGT_DIR)
        if os.path.exists(t): pairs.append((s,t))
    if not pairs:
        sys.exit("❌ no source/target pairs found")

    # estimate or load low-pass IR
    ir_path = os.path.join(OUT_DIR, IR_FILE)
    if os.path.exists(ir_path):
        h_ir = np.load(ir_path); print("Loaded IR:",ir_path)
    else:
        print("Estimating & low-passing IR…")
        h_ir = estimate_ir(pairs, IR_LEN, IR_FFT_SIZE, cutoff_hz=IR_CUTOFF_HZ)
        np.save(ir_path, h_ir)
        print("Saved IR →",ir_path)

    # train/val split
    from sklearn.model_selection import train_test_split
    n = len(pairs)
    if n>=2:
        tr,va = train_test_split(list(range(n)), test_size=0.1, random_state=42)
    else:
        tr=va=list(range(n))
    train_pairs = [pairs[i] for i in tr]
    val_pairs   = [pairs[i] for i in va]
    train_d  = [sf.info(s).frames for s,_ in train_pairs]
    val_d    = [sf.info(t).frames for _,t in val_pairs]

    # receptive-field + segment length
    RF      = 1 + (KERNEL_SIZE-1)*sum(DILATIONS)
    SEG_LEN = RF + TRAIN_BLOCK_SIZE
    print(f"RF={RF} samples ({1000*RF/SR:.1f}ms), SEG_LEN={SEG_LEN}")

    def steps(durs):
        hop = SEG_LEN//2
        cnt = sum(max((d-SEG_LEN)//hop+1,0) for d in durs)
        return max(cnt//BATCH_SIZE,1)
    s_epoch = steps(train_d)
    v_steps = steps(val_d)
    print(f"→ {s_epoch} train steps/epoch, {v_steps} val steps")

    # data generator
    def gen(plist, dlist):
        while True:
            i      = random.randrange(len(plist))
            sp,tp  = plist[i]
            dur    = dlist[i]
            off    = random.randrange(max(dur-SEG_LEN,1))
            x,_ = sf.read(sp, start=off, frames=SEG_LEN,
                          dtype='float32', always_2d=True)
            y,_ = sf.read(tp, start=off, frames=SEG_LEN,
                          dtype='float32', always_2d=True)
            if x.shape[0]<SEG_LEN:
                p = SEG_LEN - x.shape[0]
                x = np.pad(x, ((0,p),(0,0)))
                y = np.pad(y, ((0,p),(0,0)))
            yield x, y

    sig    = tf.TensorSpec((SEG_LEN,1), tf.float32)
    ds_tr = (tf.data.Dataset
             .from_generator(lambda: gen(train_pairs, train_d),
                             output_signature=(sig,sig))
             .shuffle(SHUFFLE_BUF)
             .batch(BATCH_SIZE)
             .repeat()
             .prefetch(tf.data.AUTOTUNE))
    ds_val= (tf.data.Dataset
             .from_generator(lambda: gen(val_pairs,   val_d),
                             output_signature=(sig,sig))
             .batch(BATCH_SIZE)
             .repeat()
             .prefetch(tf.data.AUTOTUNE))

    # build & train
    model = build_hybrid_model(h_ir)

    # --- hybrid loss: MAE + small multires STFT-magnitude term -----------
    def mrstft_loss(y_true, y_pred):
        def one(n_fft, hop):
            # remove final dim
            yt = tf.squeeze(y_true, -1)
            yp = tf.squeeze(y_pred, -1)
            st_t = tf.signal.stft(yt,  frame_length=n_fft, frame_step=hop)
            st_p = tf.signal.stft(yp,  frame_length=n_fft, frame_step=hop)
            mag_t = tf.abs(st_t)
            mag_p = tf.abs(st_p)
            fro = tf.norm(mag_t - mag_p, ord='fro') / (tf.norm(mag_t, ord='fro')+1e-8)
            return tf.reduce_mean(tf.abs(mag_t - mag_p)) + fro
        return one(512,128) + one(1024,256)

    def hybrid_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred)) + 0.1 * mrstft_loss(y_true, y_pred)

    model.compile(optimizer='adam', loss=hybrid_loss)
    model.summary()

    es = tf.keras.callbacks.EarlyStopping(
         monitor='val_loss', patience=PATIENCE, restore_best_weights=True
    )
    model.fit(ds_tr,
              epochs=EPOCHS,
              steps_per_epoch=s_epoch,
              validation_data=ds_val,
              validation_steps=v_steps,
              callbacks=[es])

    model.save(os.path.join(OUT_DIR, MODEL_FILE))
    print("✅ Saved hybrid model →",MODEL_FILE)
    return model, h_ir


# ─────────────────────────────────── DEMO ──────────────────────────────────────
def demo():
    mpath = os.path.join(OUT_DIR, MODEL_FILE)
    if not os.path.exists(mpath):
        sys.exit("❌ Run with --train first")

    print("Loading hybrid model…")
    model = tf.keras.models.load_model(
        mpath, compile=False,
        custom_objects={
            'FixedIR'      : FixedIR,
            'HFBYPASS'     : HFBYPASS,
            'slice_content': slice_content,
            'slice_gate'   : slice_gate
        }
    )

    # sub‐models for aligned outputs
    body_model   = Model(inputs=model.input,
                         outputs=model.get_layer('ir_conv').output)
    hf_model     = Model(inputs=model.input,
                         outputs=model.get_layer('hf_bypass').output)
    resid_model  = Model(inputs=model.input,
                         outputs=model.get_layer('resid_gain').output)

    # pick a middle pair
    srcs = sorted(glob.glob(os.path.join(SRC_DIR,"*_source.wav")))
    pairs = [(s, s.replace("_source.wav","_target.wav")
                .replace(SRC_DIR,TGT_DIR))
             for s in srcs
             if os.path.exists(s.replace("_source.wav","_target.wav")
                               .replace(SRC_DIR,TGT_DIR))]

    FRAMES      = DEMO_SEC * SR
    START_FRAME = DEMO_OFFSET_SEC * SR
    p_src,p_tgt = pairs[DEMO_PAIR_INDEX]

    x_long,_= sf.read(p_src, start=START_FRAME, frames=FRAMES,
                      dtype='float32', always_2d=True)
    y_ref,_ = sf.read(p_tgt, start=START_FRAME, frames=FRAMES,
                      dtype='float32', always_2d=True)
    x1 = x_long[:,0]; y1 = y_ref[:,0]
    x_in = x1.reshape(1,-1,1)

    body   = body_model .predict(x_in, batch_size=1)[0,:,0]
    hf     = hf_model   .predict(x_in, batch_size=1)[0,:,0]
    resid  = resid_model.predict(x_in, batch_size=1)[0,:,0]
    attack = hf + resid
    hybrid = model        .predict(x_in, batch_size=1)[0,:,0]

    # global demo gain so hybrid ≈ reference
    gain = np.mean(np.abs(y1)) / (np.mean(np.abs(hybrid))+1e-9)
    body   *= gain
    attack *= gain
    hybrid *= gain

    # write WAVs
    sf.write(os.path.join(OUT_DIR,"demo_input_piezo.wav"),    x1,    SR)
    sf.write(os.path.join(OUT_DIR,"demo_reference_mic.wav"), y1,    SR)
    sf.write(os.path.join(OUT_DIR,"demo_body_only.wav"),      body,  SR)
    sf.write(os.path.join(OUT_DIR,"demo_attack_only.wav"),    attack,SR)
    sf.write(os.path.join(OUT_DIR,"demo_hybrid.wav"),         hybrid,SR)

    print("✅ Demo files in", OUT_DIR)


# ────────────────────────────────── MAIN ───────────────────────────────────────
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--train":
        train()
    demo()