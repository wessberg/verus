#!/usr/bin/env python3
"""
Train a causal ResNet to map an acoustic‐guitar piezo pickup signal 
to the equivalent mic’d signal, with pick‐attack oversampling ramp.
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
from tensorflow.keras import regularizers

# ───────────────────────────── PARAMETERS ─────────────────────────────────────
FILTERS             = 64
KERNEL_SIZE         = 25
DILATIONS           = (1,2,4,8)
RECEPTIVE_FIELD     = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
TRAIN_BLOCK_SIZE    = 4096
SR                  = 48000
SEG_LEN             = RECEPTIVE_FIELD + TRAIN_BLOCK_SIZE
BATCH_SIZE          = 32

HF_KERNEL_SIZE      = 11
HF_FMIN             = 3000.0
HF_FMAX             = 15000.0
HF_N_FFT            = 512
HF_HOP              = 128

PHASE_1_EPOCHS      = 5
PHASE_2_EPOCHS      = 5
PHASE_3_EPOCHS      = 100
DEBUG_STEPS         = False

PHASE_1_LR          = 1e-3
PHASE_2_LR          = 1e-4
PHASE_3_LR          = 1e-2

PHASE_1_LR_PATIENCE = 5
PHASE_2_LR_PATIENCE = 3
PHASE_3_LR_PATIENCE = 10

PHASE_1_PICK_RATIO = 0.1  # 10% pick‐attack
PHASE_2_PICK_RATIO = 0.5  # 50% pick‐attack
PHASE_3_PICK_RATIO = 1.0  # 100% pick‐attack

SHUFFLE_BUF         = 1000
CHUNK_SEC           = 5

# ─── your new 4‐dir layout ─────────────────────────────────────────────
PICK_SRC_DIR   = "./input2/picked/source"
PICK_TGT_DIR   = "./input2/picked/target"
FINGER_SRC_DIR = "./input2/fingerpicked/source"
FINGER_TGT_DIR = "./input2/fingerpicked/target"
OUT_DIR        = "./out"

DEMO_SEC        = 60 * 2
DEMO_OFFSET_SEC = 0
DEMO_PAIR_INDEX = 0

STATIC_MODEL_FILENAME  = "piezo2mic_resnet.keras"
DYNAMIC_MODEL_FILENAME = "piezo2mic_resnet_dynamic.keras"
SAVE_WEIGHTS           = False  # set to True to dump .npy weights

os.makedirs(OUT_DIR, exist_ok=True)

# ─── phase1_weights: get your tone & broad HF in place ──────────────
phase1_weights = {
    'w_l1'  : 0.50,
    'w_stft': 0.05,
    'w_fw'  : 0.05,
    'w_grad': 0.50,
    'w_pre' : 0.30,
    'w_mel' : 0.10,
    'w_hf'  : 5.00,
    'w_tr'  :150.00,
    'w_hw'  : 0.00
}
# ─── phase2_weights: tighten HF and refine attack shape ─────────────
phase2_weights = {
    'w_l1'  : 0.30,
    'w_stft': 0.10,
    'w_fw'  : 0.20,
    'w_grad': 0.00,
    'w_pre' : 0.40,
    'w_mel' : 0.10,
    'w_hf'  : 2.00,
    'w_tr'  :10.00,
    'w_hw'  : 0.00
}
# ─── phase3_weights: freeze all but bypass, sculpt the positive spike ─
phase3_weights = {
    'w_l1'  : 0.00,
    'w_stft': 0.00,
    'w_fw'  : 0.00,
    'w_grad': 0.00,
    'w_pre' : 0.00,
    'w_mel' : 0.00,
    'w_hf'  :10.00,
    'w_tr'  :200.00,
    'w_hw'  :100.00
}

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

def attack_segment_generator(uniform_chunks,
                             pick_chunks,
                             pick_prob=0.2,
                             N=10):
    """
    With probability `pick_prob` do best-of-N scan on `pick_chunks`,
    otherwise uniform random on `uniform_chunks`.
    Yields (x_window, y_window) of shape (SEG_LEN,1).
    """
    while True:
        if random.random() < pick_prob:
            # best-of-N from pick_chunks
            best, best_d = None, -1.0
            for _ in range(N):
                src,tgt,cstart,cdur = random.choice(pick_chunks)
                off = random.randrange(max(cdur-SEG_LEN,1))
                x,_ = sf.read(src, start=cstart+off, frames=SEG_LEN,
                              dtype='float32', always_2d=True)
                if x.shape[0] < SEG_LEN:
                    x = np.pad(x, ((0,SEG_LEN-x.shape[0]),(0,0)))
                d = np.max(np.abs(np.diff(x[:,0])))
                if d > best_d:
                    best_d, best = d, (src,tgt,cstart+off)
            src_path,tgt_path,read_start = best
        else:
            # uniform random from all_chunks
            src_path,tgt_path,cstart,cdur = random.choice(uniform_chunks)
            off = random.randrange(max(cdur-SEG_LEN,1))
            read_start = cstart + off

        x,_ = sf.read(src_path, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        if x.shape[0] < SEG_LEN:
            x = np.pad(x, ((0,SEG_LEN-x.shape[0]),(0,0)))
        y,_ = sf.read(tgt_path, start=read_start, frames=SEG_LEN,
                      dtype='float32', always_2d=True)
        yield x, y

def segment_generator(chunk_list):
    """
    Uniform random windows from chunk_list.
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
        yield x, y
        
def generate_demo (model, prediction_name="demo_prediction", only_prediction=False):
        # DEMO
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

        x_in = x_long[np.newaxis,...]
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

# ─────────────────────────────────────────────────────────────────────────────
@register_keras_serializable()
def half_wave_loss(y_true, y_pred, threshold=0.0):
    dy_t   = y_true[:,1:,:] - y_true[:,:-1,:]
    dy_p   = y_pred[:,1:,:] - y_pred[:,:-1,:]
    pos_t  = tf.nn.relu(dy_t - threshold)
    pos_p  = tf.nn.relu(dy_p - threshold)
    return tf.reduce_mean(tf.abs(pos_t - pos_p))

@register_keras_serializable()
class EnvGate(tf.keras.layers.Layer):
    """Sigmoidal gating [0..1] from |dx| to a soft pick‐mask."""
    def __init__(self, shift=20e-3, slope=800.0, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift
        self.slope = slope

    def call(self, x):
        # x is abs(dx)
        return tf.sigmoid((x - self.shift) * self.slope)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"shift": self.shift, "slope": self.slope})
        return cfg

@register_keras_serializable()
class HFGainLayer(tf.keras.layers.Layer):
    """
    A positive, bounded scalar gain γ = clip(exp(log_γ), 0, max_gain).
    You can initialize init_gain>1.0 or at inference do layer.log_gamma.assign(...)
    """
    def __init__(self, init_gain=1.0, max_gain=20.0, **kwargs):
        super().__init__(**kwargs)
        self.init_gain = init_gain
        self.max_gain  = max_gain

    def build(self, input_shape):
        # trainable log‐gamma
        self.log_gamma = self.add_weight(
            name="log_gamma",
            shape=(),
            initializer=tf.keras.initializers.Constant(np.log(self.init_gain)),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        gamma = tf.clip_by_value(tf.exp(self.log_gamma), 0.0, self.max_gain)
        return x * gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_gain": self.init_gain,
                    "max_gain":  self.max_gain})
        return cfg

@register_keras_serializable()
def transient_loss(y_true, y_pred,
                   threshold=1e-3, slope=2000.0):
    dy_t = y_true[:,1:,:] - y_true[:,:-1,:]
    dy_p = y_pred[:,1:,:] - y_pred[:,:-1,:]
    mask = tf.sigmoid((tf.abs(dy_t) - threshold) * slope)
    err  = tf.abs(dy_t - dy_p) * mask
    err  = tf.pad(err, [[0,0],[1,0],[0,0]])
    return tf.reduce_mean(err)

@register_keras_serializable()
class AbsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class EnvScaleLayer(tf.keras.layers.Layer):
    def __init__(self, shift=1e-3, slope=200.0, **kw):
        super().__init__(**kw)
        self.shift = shift
        self.slope = slope
    def call(self, x):
        return (x - self.shift) * self.slope
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'shift':self.shift,'slope':self.slope})
        return cfg

@register_keras_serializable()
class Diff1D(tf.keras.layers.Layer):
    def call(self, x):
        d = x[:,1:,:] - x[:,:-1,:]
        return tf.pad(d, [[0,0],[1,0],[0,0]])
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))
@register_keras_serializable()
def stft_metric(y_true, y_pred):
    return multi_res_stft_loss(y_true, y_pred)
@register_keras_serializable()
def fw_metric(y_true, y_pred):
    return freq_weighted_stft_loss(y_true, y_pred)
@register_keras_serializable()
def grad_metric(y_true, y_pred):
    return gradient_loss(y_true, y_pred)
@register_keras_serializable()
def pre_metric(y_true, y_pred):
    return pre_emphasis_loss(y_true, y_pred)
@register_keras_serializable()
def mel_metric(y_true, y_pred):
    return mel_loss(y_true, y_pred)
@register_keras_serializable()
def hf_metric(y_true, y_pred):
    return band_limited_stft_loss(y_true, y_pred)
@register_keras_serializable()
def tf_metric(y_true, y_pred):
    return transient_loss(y_true, y_pred)
@register_keras_serializable()
def hw_metric(y_true, y_pred):
    return half_wave_loss(y_true, y_pred)

@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]
@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

def pre_emphasis(x, μ=0.99):
    x0 = x[:, :1, :]
    x1 = x[:, 1:, :] - μ * x[:, :-1, :]
    return tf.concat([x0, x1], axis=1)
def pre_emphasis_loss(y_true, y_pred, μ=0.99):
    yt = pre_emphasis(y_true, μ)
    yp = pre_emphasis(y_pred, μ)
    return tf.reduce_mean(tf.abs(yt - yp))

def band_limited_stft_loss(y_true, y_pred):
    yt = tf.squeeze(y_true, -1)
    yp = tf.squeeze(y_pred, -1)
    S_t = tf.signal.stft(yt, frame_length=HF_N_FFT, frame_step=HF_HOP, fft_length=HF_N_FFT)
    S_p = tf.signal.stft(yp, frame_length=HF_N_FFT, frame_step=HF_HOP, fft_length=HF_N_FFT)
    mag_t = tf.abs(S_t)
    mag_p = tf.abs(S_p)
    freq_bins = tf.shape(mag_t)[-1]
    freqs = tf.linspace(0.0, SR/2, freq_bins)
    mask = tf.logical_and(freqs >= HF_FMIN, freqs <= HF_FMAX)
    mask = tf.cast(mask, mag_t.dtype)[tf.newaxis,tf.newaxis,:]
    diff = tf.abs(mag_t - mag_p) * mask
    return tf.reduce_mean(diff)

def freq_weighted_stft_loss(y_true,y_pred,α=1.5):
    yt = tf.squeeze(y_true, -1); yp = tf.squeeze(y_pred, -1)
    total=0.0
    specs=[(512,128),(1024,256),(2048,512)]
    for n_fft, hop in specs:
        S_t=tf.signal.stft(yt,frame_length=n_fft,frame_step=hop,fft_length=n_fft)
        S_p=tf.signal.stft(yp,frame_length=n_fft,frame_step=hop,fft_length=n_fft)
        mag_t,mag_p=tf.abs(S_t),tf.abs(S_p)
        diff=(tf.math.log1p(mag_t)-tf.math.log1p(mag_p))**2
        freq_bins=tf.shape(mag_t)[-1]
        freqs=tf.linspace(0.0,SR/2,freq_bins)
        w=tf.reshape((freqs/(SR/2))**α,[1,1,-1])
        total+=tf.reduce_mean(diff*w)
    return total/len(specs)

def multi_res_stft_loss(y_true,y_pred):
    yt = tf.squeeze(y_true, -1); yp = tf.squeeze(y_pred, -1)
    specs=[(64,16),(128,32),(256,64),(512,128),(1024,256),(2048,512)]
    mag_loss=0.0; comp_loss=0.0
    for n_fft,hop in specs:
        S_t=tf.signal.stft(yt,frame_length=n_fft,frame_step=hop,fft_length=n_fft)
        S_p=tf.signal.stft(yp,frame_length=n_fft,frame_step=hop,fft_length=n_fft)
        mag_t,mag_p=tf.abs(S_t),tf.abs(S_p)
        mag_loss  += tf.reduce_mean((tf.math.log1p(mag_t)-tf.math.log1p(mag_p))**2)
        comp_loss += tf.reduce_mean((tf.math.real(S_t)-tf.math.real(S_p))**2 + (tf.math.imag(S_t)-tf.math.imag(S_p))**2)
    return (mag_loss+comp_loss)/len(specs)

def gradient_loss(y_true,y_pred):
    dy_t=y_true[:,1:,:]-y_true[:,:-1,:]
    dy_p=y_pred[:,1:,:]-y_pred[:,:-1,:]
    return tf.reduce_mean(tf.abs(dy_t-dy_p))

def mel_loss(y_true,y_pred,n_mels=64,n_fft=1024,hop=256):
    yt=tf.squeeze(y_true,-1); yp=tf.squeeze(y_pred,-1)
    S_t=tf.abs(tf.signal.stft(yt,frame_length=n_fft,frame_step=hop,fft_length=n_fft))
    S_p=tf.abs(tf.signal.stft(yp,frame_length=n_fft,frame_step=hop,fft_length=n_fft))
    num_bins=n_fft//2+1
    mel_w=tf.signal.linear_to_mel_weight_matrix(n_mels,num_bins,SR,50.0,SR/2)
    M_t=tf.tensordot(S_t,mel_w,axes=[[2],[0]])
    M_p=tf.tensordot(S_p,mel_w,axes=[[2],[0]])
    return tf.reduce_mean(tf.abs(tf.math.log1p(M_t)-tf.math.log1p(M_p)))

def make_combined_loss(w_l1,w_stft,w_fw,w_grad,w_pre,w_mel,w_hf,w_tr,w_hw):
    @register_keras_serializable()
    def _loss(y_true,y_pred):
        l1   = tf.reduce_mean(tf.abs(y_true-y_pred))
        stft = multi_res_stft_loss(y_true,y_pred)
        fw   = freq_weighted_stft_loss(y_true,y_pred)
        grad = gradient_loss(y_true,y_pred)
        pre  = pre_emphasis_loss(y_true,y_pred)
        mel  = mel_loss(y_true,y_pred)
        hf   = band_limited_stft_loss(y_true,y_pred)
        tr   = transient_loss(y_true,y_pred)
        hw   = half_wave_loss(y_true,y_pred)
        return (w_l1*l1 + w_stft*stft + w_fw*fw + w_grad*grad +
                w_pre*pre + w_mel*mel + w_hf*hf + w_tr*tr + w_hw*hw)
    return _loss

def build_resnet_gated(filters=FILTERS, kernel_size=KERNEL_SIZE,
                       dilations=DILATIONS, dynamic_time=False):
    T = None if dynamic_time else SEG_LEN
    inp = tf.keras.Input(shape=(T,1), name="piezo_in")
    x = tf.keras.layers.Conv1D(filters,1,padding='same',name="init_pw")(inp)
    for idx,d in enumerate(dilations):
        res = x
        pad = (kernel_size-1)*d
        x   = tf.keras.layers.ZeroPadding1D((pad,0),name=f"pad_{idx}")(x)
        g   = tf.keras.layers.SeparableConv1D(filters*2,kernel_size,
                    dilation_rate=d,padding='valid',use_bias=True,
                    name=f"sep_{idx}")(x)
        content = tf.keras.layers.Lambda(slice_content,
                    arguments={'filters':filters},
                    name=f"content_{idx}")(g)
        gate    = tf.keras.layers.Lambda(slice_gate,
                    arguments={'filters':filters},
                    name=f"gate_{idx}")(g)
        c_act = tf.keras.layers.Activation('tanh',name=f"tanh_{idx}")(content)
        g_act = tf.keras.layers.Activation('sigmoid',name=f"sigmoid_{idx}")(gate)
        x = tf.keras.layers.Multiply(name=f"gated_mul_{idx}")([c_act,g_act])
        x = tf.keras.layers.Conv1D(filters,1,padding='same',name=f"pw_{idx}")(x)
        x = tf.keras.layers.Add(name=f"res_add_{idx}")([x,res])
    final = tf.keras.layers.Conv1D(1,1,padding='same',name="final_pw")(x)
    skip  = tf.keras.layers.Conv1D(1,1,use_bias=False,
                kernel_initializer=tf.keras.initializers.Ones(),
                name="skip_conv")(inp)
    main_out = tf.keras.layers.Add(name="res_out")([final,skip])
    
    # 1) compute first‐difference
    dx     = Diff1D(name="diff1d_in")(inp)
    # 2) absolute slope
    abs_dx = AbsLayer(name="abs_dx")(dx)

    # 3) a smooth [0..1] gate around each pick transient
    gate = EnvGate(shift=20e-3, slope=800.0, name="env_gate")(abs_dx)

    # 4) small bypass‐filter that only “sees” that gate
    hf_transient = tf.keras.layers.Conv1D(
        filters=1,
        kernel_size=HF_KERNEL_SIZE,
        padding='causal',
        use_bias=False,
        kernel_initializer='zeros',
        kernel_regularizer=regularizers.l1(1e-3),
        name='hf_transient'
    )(gate)

    # 5) trainable, bounded scalar gain on that bypass
    hf_transient = HFGainLayer(
        init_gain=1.0,    # start at unity
        max_gain=20.0,    # never exceed ×20
        name="hf_gain"
    )(hf_transient)

    # 6) your “air” bypass unchanged
    hf_air = tf.keras.layers.Conv1D(
        1, 31, padding='causal', use_bias=False,
        kernel_initializer='zeros',
        kernel_regularizer=regularizers.l1(1e-3),
        name='hf_air'
    )(inp)
    hf_air = tf.keras.layers.Dense(
        1, use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(5.0),
        name='air_gain'
    )(hf_air)

    # 7) merge
    out = tf.keras.layers.Add(name="add_hf")([main_out,
                                            hf_transient,
                                            hf_air])
    return tf.keras.Model(inp, out, name="piezo2mic_resnet_gated")

def extract_and_save_int8_weights(tflite_model_buf, outdir, dilations, filters):
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
    # val_pick_chunks   = [c for c in val_chunks   if c in pick_chunks]

    # 3) compute steps
    def approx_steps(durs):
        if (DEBUG_STEPS):
            return 500
        
        return sum(max((d-SEG_LEN)//(SEG_LEN//2)+1,0) for d in durs)//BATCH_SIZE

    train_durs = [dur for (_,_,_,dur) in train_chunks]
    val_durs   = [dur   for (_,_,_,dur) in val_chunks]
    steps_per_epoch  = max(approx_steps(train_durs),1)
    validation_steps = max(approx_steps(val_durs),1)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps")

    sig = tf.TensorSpec((SEG_LEN,1), tf.float32)
    
    def representative_data_gen():
        """
        For TFLite int8 quant: random uniform windows.
        """
        for _ in range(100):
            src,tgt,cstart,cdur = random.choice(train_chunks)
            off = random.randrange(max(cdur-SEG_LEN,1))
            read_start = cstart + off
            x,_ = sf.read(src, start=read_start, frames=SEG_LEN,
                        dtype='float32', always_2d=True)
            if x.shape[0] < SEG_LEN:
                x = np.pad(x, ((0,SEG_LEN-x.shape[0]),(0,0)))
            yield [x.reshape(1,SEG_LEN,1)]

    # ── PHASE 1: mostly uniform windows so the net learns the broad tone ──
    train_ds_1 = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(all_chunks,
                                         train_pick_chunks,
                                         pick_prob=PHASE_1_PICK_RATIO,
                                         N=10),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: segment_generator(val_chunks),
        output_signature=(sig, sig)
    ).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    print("\n≈ Training Phase 1")
    static_model = build_resnet_gated(dynamic_time=False)
    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_1_LR),
        loss=make_combined_loss(**phase1_weights),
        metrics=['mae',
                 l1_metric, stft_metric, fw_metric, grad_metric,
                 pre_metric, mel_metric, hf_metric, tf_metric, hw_metric]
    )
    static_model.summary()
    static_model.fit(
        train_ds_1,
        epochs=PHASE_1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[TensorBoard(log_dir="./logs")]
    )

    # ── PHASE 2: 50/50 uniform vs pick-attacks ─────────────────────────────
    train_ds_2 = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(all_chunks,
                                         train_pick_chunks,
                                         pick_prob=PHASE_2_PICK_RATIO,
                                         N=10),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    es2 = EarlyStopping(monitor='val_loss',
                        patience=PHASE_2_LR_PATIENCE,
                        restore_best_weights=True)

    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_2_LR),
        loss=make_combined_loss(**phase2_weights),
        metrics=['mae',
                 l1_metric, stft_metric, fw_metric, grad_metric,
                 pre_metric, mel_metric, hf_metric, tf_metric, hw_metric]
    )
    print("\n≈ Training Phase 2")
    static_model.fit(
        train_ds_2,
        initial_epoch=PHASE_1_EPOCHS,
        epochs=PHASE_1_EPOCHS + PHASE_2_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[es2, TensorBoard(log_dir="./logs")]
    )

    # ── PHASE 3: 100% pick-attacks, freeze all but bypass layers ───────────
    for layer in static_model.layers:
        layer.trainable = (layer.name in ['hf_transient','hf_gain','hf_air','air_gain'])

    train_ds_3 = tf.data.Dataset.from_generator(
        lambda: attack_segment_generator(all_chunks,
                                         train_pick_chunks,
                                         pick_prob=PHASE_3_PICK_RATIO,
                                         N=10),
        output_signature=(sig, sig)
    ).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

    es3 = EarlyStopping(monitor='half_wave_loss',
                        mode='min',
                        patience=PHASE_3_LR_PATIENCE,
                        restore_best_weights=True)

    static_model.compile(
        optimizer=tf.keras.optimizers.Adam(PHASE_3_LR),
        loss=make_combined_loss(**phase3_weights),
        metrics=[hf_metric, tf_metric, half_wave_loss]
    )
    print("\n≈ Training Phase 3")
    static_model.fit(
        train_ds_3,
        epochs=PHASE_3_EPOCHS,
        steps_per_epoch=200,
        callbacks=[es3],
    )

    # ── Save FP32 static & dynamic models ──────────────────────────────────
    path_static = os.path.join(OUT_DIR, STATIC_MODEL_FILENAME)
    static_model.save(path_static)
    print("✅ Saved FP32 static model:", path_static)

    dynamic_model = build_resnet_gated(dynamic_time=True)
    dynamic_model.set_weights(static_model.get_weights())
    path_dyn = os.path.join(OUT_DIR, DYNAMIC_MODEL_FILENAME)
    dynamic_model.save(path_dyn)
    print("✅ Saved FP32 dynamic model:", path_dyn)

    # ── Optional: dump .npy FP32 weights & extract INT8 ───────────────────
    if SAVE_WEIGHTS:
        weights_dir = os.path.join(OUT_DIR, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        # init_pw
        w0,b0 = dynamic_model.get_layer("init_pw").get_weights()
        np.save(os.path.join(weights_dir,"init_pw_w.npy"), w0.astype(np.float32))
        np.save(os.path.join(weights_dir,"init_pw_b.npy"), b0.astype(np.float32))

        # depthwise + pwsep + pw per block
        for idx,_ in enumerate(DILATIONS):
            sep = dynamic_model.get_layer(f"sep_{idx}")
            dw, pw_w1, pw_b1 = sep.get_weights()
            np.save(os.path.join(weights_dir,f"dw{idx}_w.npy"),   dw.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pwsep{idx}_w.npy"),pw_w1.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pwsep{idx}_b.npy"),pw_b1.astype(np.float32))
            pw = dynamic_model.get_layer(f"pw_{idx}")
            w2,b2 = pw.get_weights()
            np.save(os.path.join(weights_dir,f"pw{idx}_w.npy"),   w2.astype(np.float32))
            np.save(os.path.join(weights_dir,f"pw{idx}_b.npy"),   b2.astype(np.float32))

        # final_pw
        wf,bf = dynamic_model.get_layer("final_pw").get_weights()
        np.save(os.path.join(weights_dir,"final_pw_w.npy"), wf.astype(np.float32))
        np.save(os.path.join(weights_dir,"final_pw_b.npy"), bf.astype(np.float32))

        # skip_conv
        sw = dynamic_model.get_layer("skip_conv").get_weights()[0]
        np.save(os.path.join(weights_dir,"skip_conv_w.npy"), sw.astype(np.float32))

        # hf_transient (strip out-of-channel dim)
        hf_kernel = dynamic_model.get_layer("hf_transient").get_weights()[0]
        hf_kernel = hf_kernel[:,:,0]
        np.save(os.path.join(weights_dir,"hf_transient_w.npy"),
                hf_kernel.astype(np.float32))

        print("\n≈ Converting dynamic model to TFLite INT8 model for weight extraction")
        converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        extract_and_save_int8_weights(
            tflite_model,
            os.path.join(OUT_DIR,"int8_weights"),
            dilations=DILATIONS,
            filters=FILTERS
        )

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
                 'AbsLayer': AbsLayer,
                 'Diff1D': Diff1D,
                 'EnvScaleLayer': EnvScaleLayer,
                 'slice_content': slice_content,
                 'slice_gate':    slice_gate
            }
        )
        
        generate_demo(dynamic_model)
