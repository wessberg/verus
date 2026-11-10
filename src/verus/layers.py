import numpy as np
import tensorflow as tf
from scipy.signal import firwin
from tensorflow.keras import constraints  # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Layer, Multiply, Reshape)  # pyright: ignore[reportMissingImports]

from .config import (
    HP_PRE_EMPHASIS_HF,
    OFFLINE_HF_HP_ORDER,
    OFFLINE_HP_CUTOFF_HZ,
    OFFLINE_SAMPLE_RATE,
    PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
    PICK_RESO_MODAL_DECAYS_MS_REALTIME,
    PICK_RESO_MODAL_LEN_REALTIME,
    REALTIME_IR_DOWNSAMPLE_FACTOR,
    REALTIME_IR_FIR_SIZE,
    REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
    REALTIME_PICK_RESO_DS_TAPS,
    REALTIME_SAMPLE_RATE,
)
from .dsp import (
    _causal_conv1d,
    _crop_time,
    _decimate_modal_kernels,
    _exp_smoothing_kernel,
    _get_minphase_bp_kernel,
    _get_minphase_hp_kernel,
    _interp_lowpass_kernel,
    _interp_polyphase_apply,
    _interp_polyphase_filters,
    _interp_polyphase_len,
    _ir_fir_tf,
    _ir_prototype_kernel,
    _make_lpf_kernel,
    _make_modal_kernels,
    _take_last,
    _to_minimum_phase,
    hp_cutoff_for_sr,
)

@register_keras_serializable()
def apply_runtime_gain(x, knob_scalar, name):
    # knob_scalar: (B,1) → broadcast to (B,1,1) and multiply over time
    k3 = Reshape((1,1), name=f"{name}_expand")(knob_scalar)
    return Multiply(name=f"{name}_mul")([x, k3])

@register_keras_serializable()
class StreamingLinearRamp(tf.keras.layers.Layer):
    """
    Streaming linear fade-in from 0→1 over 'ramp_len' samples, with explicit state.
    inputs: [ref (B,T,1), s_count (B,1,1)]
      - 'ref' is only used to get T; it is not modified.
    outputs: [ramp (B,T,1), s_next (B,1,1)]
    """
    def __init__(self, ramp_len, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.ramp_len = int(max(1, ramp_len))

    def call(self, inputs):
        ref, s_count = inputs
        T = tf.shape(ref)[1]                         # time length this hop
        # previous count (how many samples already emitted, clamped to ramp_len)
        c0 = tf.cast(tf.squeeze(s_count, axis=[1,2]), tf.int32)  # (B,)
        c0c = tf.clip_by_value(c0, 0, self.ramp_len)

        # ramp indices for this hop: c0 + [1..T]
        idx = tf.range(T, dtype=tf.int32)[None, :] + c0c[:, None] + 1
        r = tf.clip_by_value(tf.cast(idx, ref.dtype) / float(self.ramp_len), 0.0, 1.0)
        ramp = r[:, :, None]                         # (B,T,1)

        # next state: clamp to ramp_len so it stops at 1.0 after ramp_len samples
        c_next = tf.clip_by_value(c0c + T, 0, self.ramp_len)
        s_next = tf.reshape(tf.cast(c_next, ref.dtype), (-1, 1, 1))
        return [ramp, s_next]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ramp_len": self.ramp_len})
        return cfg

@register_keras_serializable()
class StreamingScalarEMA(tf.keras.layers.Layer):
    """
    Streaming EMA for a scalar per-batch value (shape (B,1)).
    inputs: [x (B,1), s_prev (B,1)] or [x, s_prev, ref (B,T,C)]
    outputs: [y (B,1), s_next (B,1)],   y = a * s_prev + (1 - a) * x

    'alpha' is the coefficient per 'ref_hop'-sample hop. When a reference tensor is supplied the
    coefficient is rescaled to the ACTUAL hop length T: a = alpha ** (T / ref_hop), so the EMA has
    the same time constant whether the graph is stepped 32 samples at a time (pedal/TFLite) or
    1024 samples at a time (dynamic-T Keras inference). Without 'ref' (legacy graphs) a = alpha.
    """
    def __init__(self, alpha=0.98, ref_hop=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.ref_hop = None if ref_hop is None else int(ref_hop)

    def call(self, inputs):
        if len(inputs) == 3:
            x, s_prev, ref = inputs
        else:
            x, s_prev = inputs
            ref = None
        if ref is not None and self.ref_hop:
            T = tf.cast(tf.shape(ref)[1], x.dtype)
            a = tf.exp(tf.math.log(tf.constant(self.alpha, dtype=x.dtype)) * T / float(self.ref_hop))
        else:
            a = tf.constant(self.alpha, dtype=x.dtype)
        y = a * s_prev + (1.0 - a) * x
        return [y, y]  # y is also the next state

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha, "ref_hop": self.ref_hop})
        return cfg

@register_keras_serializable()
class StreamingStyleLerp(tf.keras.layers.Layer):
    """
    Expand a per-hop style scalar into a per-sample envelope by linearly
    interpolating from the previous-hop scalar to the current-hop scalar.

    Inputs:
      [ref (B,T,1), style_scalar (B,1), s_prev (B,1,1)]
    Outputs:
      [style_time (B,T,1), s_next (B,1,1)]
    """
    def call(self, inputs):
        ref, style_scalar, s_prev = inputs
        T = tf.shape(ref)[1]

        # prev/current scalars shaped (B,)
        prev = tf.squeeze(s_prev, axis=[1, 2])
        curr = tf.squeeze(style_scalar, axis=-1)

        # r: (1,T) in [0,1], use (T-1) so last sample == curr
        denom = tf.maximum(tf.cast(T - 1, ref.dtype), tf.constant(1.0, dtype=ref.dtype))
        r = tf.cast(tf.range(T), ref.dtype)[None, :] / denom

        # y[n] = prev + r[n]*(curr - prev)
        delta = curr - prev
        y = prev[:, None] + r * delta[:, None]     # (B,T)
        y3 = y[:, :, None]                         # (B,T,1)

        s_next = tf.reshape(curr, (-1, 1, 1))      # (B,1,1)
        return [y3, s_next]

    def get_config(self):
        return super().get_config()
    
@register_keras_serializable()
class PickResoDownsampleFIR(Layer):
    """
    Decimate by integer factor with a short, causal min-phase FIR (for pickreso branch).
    """
    def __init__(self, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_DS_TAPS, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)

    def build(self, input_shape):
        cutoff = 0.45 / float(self.factor)
        taps = firwin(self.num_taps, cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
        taps_min = _to_minimum_phase(taps)
        taps_min = taps_min / max(np.sum(taps_min), 1e-12)
        k = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        self.kernel = self.add_weight(
            name="ds_lpf_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=self.factor, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        out_T = (T + self.factor - 1) // self.factor  # ceil(T/factor)
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"factor": self.factor, "num_taps": self.num_taps})
        return cfg

@register_keras_serializable()
class LowrateModalBank(tf.keras.layers.Layer):
    """
    Trainable low-rate modal bank: conv1d with kernel shape (L_low,1,M).
    Initialized by decimating the default modal kernels or by a provided kernel_init.
    """
    def __init__(self, num_modes, length_lr, kernel_init=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_modes  = int(num_modes)
        self.length_lr  = int(length_lr)
        self._kernel_init_np = None if kernel_init is None else np.asarray(kernel_init, dtype=np.float32)

    def build(self, input_shape):
        if self._kernel_init_np is None:
            K_full = _make_modal_kernels(PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
                                         PICK_RESO_MODAL_DECAYS_MS_REALTIME,
                                         PICK_RESO_MODAL_LEN_REALTIME,
                                         REALTIME_SAMPLE_RATE)
            K_lr = _decimate_modal_kernels(K_full, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
        else:
            K_lr = self._kernel_init_np

        # Ensure expected shape (length_lr,1,num_modes) with center-crop or symmetric pad
        K_lr = np.asarray(K_lr, dtype=np.float32)
        want = (self.length_lr, 1, self.num_modes)
        if K_lr.shape != want:
            L_have = K_lr.shape[0]
            if L_have > self.length_lr:
                # Center-crop
                drop = L_have - self.length_lr
                start = drop // 2
                end = start + self.length_lr
                K_lr = K_lr[start:end, :, :]
            elif L_have < self.length_lr:
                # Symmetric pad on time axis
                pad = self.length_lr - L_have
                pL = pad // 2
                pR = pad - pL
                K_lr = np.pad(K_lr, ((pL, pR), (0,0), (0,0)), mode='constant')
        assert K_lr.shape == want, f"LowrateModalBank kernel shape mismatch: {K_lr.shape} != {want}"

        self.kernel = self.add_weight(
            name="modal_kernels_lr",
            shape=want,
            initializer=tf.keras.initializers.Constant(K_lr.astype(np.float32)),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        return (b, t, self.num_modes)

    def get_config(self):
        return {
            "num_modes": self.num_modes,
            "length_lr": self.length_lr,
            **super().get_config()
        }

@register_keras_serializable()
class CausalPickDBMask(tf.keras.layers.Layer):
    """
    Causal pick detector in dB domain with soft-knee logistic mapping.

    Pipeline:
      x → HPF → |.| → fast EMA = env_hi
      ref = (|HPF(x)| slow EMA)  if ref_is_hp else (|x| slow EMA)
      R_db = 20*log10(env_hi+eps) - 20*log10(ref+eps)
      m0   = sigmoid(slope * (R_db - thr_db)), where slope ~ ln(9)/knee_db (so ±knee/2 spans ~10%-90%)
      asym attack/release smoothing on m0
      optional post compression: y = 1 - (1 - m)^p with makeup
      optional floor: y = floor + (1 - floor) * y

    Notes:
      - thr_db around 0 dB means mask ≈ 0.5 when HF and ref are equal.
      - knee_db controls softness; 8–14 dB tends to sound natural.
      - ref_is_hp=True measures contrast of fast HF vs slower HF (recommended).
        If False, ref is broadband slow env (more conservative).
    """
    def __init__(self,
                 sample_rate,
                 hp_cut_hz=3500.0,
                 hp_order=129,
                 env_fast_ms=0.25,
                 env_slow_ms=10.0,
                 thr_db=0.0,
                 knee_db=12.0,
                 smooth_attack_ms=0.06,
                 smooth_release_ms=6.0,
                 ref_is_hp=True,
                 post_comp_p=1.6,
                 post_comp_gain=1.15,
                 floor=0.00,
                 out_gain=1.0,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate       = int(sample_rate)
        self.hp_cut_hz         = float(hp_cut_hz)
        self.hp_order          = int(hp_order)
        self.env_fast_ms       = float(env_fast_ms)
        self.env_slow_ms       = float(env_slow_ms)
        self.thr_db            = float(thr_db)
        self.knee_db           = float(knee_db)
        self.smooth_attack_ms  = float(smooth_attack_ms)
        self.smooth_release_ms = float(smooth_release_ms)
        self.ref_is_hp         = bool(ref_is_hp)
        self.post_comp_p       = float(post_comp_p)
        self.post_comp_gain    = float(post_comp_gain)
        self.floor             = float(floor)
        self.out_gain          = float(out_gain)

    def build(self, input_shape):
        # HPF taps
        k_hp = _get_minphase_hp_kernel(self.sample_rate, self.hp_cut_hz, self.hp_order)
        self.k_hp = self.add_weight(
            name="dbmask_hp_taps", shape=k_hp.shape,
            initializer=tf.keras.initializers.Constant(k_hp), trainable=False
        )

        # EMA taps (reversed for conv1d)
        def ema_ms(ms, scale, Lmax):
            smpls = max(1, int(round((float(ms)/1000.0) * self.sample_rate)))
            return _exp_smoothing_kernel(smpls, scale=scale, Lmax=Lmax)

        k_fast = ema_ms(self.env_fast_ms, scale=3.0, Lmax=1024)
        k_slow = ema_ms(self.env_slow_ms, scale=3.0, Lmax=4096)
        k_att  = ema_ms(self.smooth_attack_ms,  scale=4.0, Lmax=2048)
        k_rel  = ema_ms(self.smooth_release_ms, scale=4.0, Lmax=2048)

        self.k_fast = self.add_weight(
            name="dbmask_env_fast", shape=k_fast.shape,
            initializer=tf.keras.initializers.Constant(k_fast), trainable=False
        )
        self.k_slow = self.add_weight(
            name="dbmask_env_slow", shape=k_slow.shape,
            initializer=tf.keras.initializers.Constant(k_slow), trainable=False
        )
        self.k_att  = self.add_weight(
            name="dbmask_att", shape=k_att.shape,
            initializer=tf.keras.initializers.Constant(k_att), trainable=False
        )
        self.k_rel  = self.add_weight(
            name="dbmask_rel", shape=k_rel.shape,
            initializer=tf.keras.initializers.Constant(k_rel), trainable=False
        )

        # Precompute sigmoid slope so that ±knee/2 → ~0.1/0.9
        # slope ≈ ln(9)/knee_db
        slope_val = float(np.log(9.0) / max(self.knee_db, 1e-3))
        self.slope = self.add_weight(
            name="dbmask_sigmoid_slope", shape=(1,),
            initializer=tf.keras.initializers.Constant([slope_val]), trainable=False
        )
        self.thr_db_w = self.add_weight(
            name="dbmask_thr_db", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.thr_db]), trainable=False
        )

        # Output gain as a weight so you can tweak via set_weights if needed
        self.gain = self.add_weight(
            name="dbmask_out_gain", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.out_gain]), trainable=False
        )

        super().build(input_shape)

    def call(self, x):
        # x: (B,T,1)
        eps = tf.constant(1e-8, dtype=x.dtype)

        # Sidechain: HPF → abs
        hp     = _causal_conv1d(x, self.k_hp)
        abs_hp = tf.abs(hp)

        # Envelopes
        env_hi = _causal_conv1d(abs_hp, self.k_fast)
        if self.ref_is_hp:
            env_lo = _causal_conv1d(abs_hp, self.k_slow)
        else:
            env_lo = _causal_conv1d(tf.abs(x), self.k_slow)

        # Contrast in dB
        # R_db = 20 * log10(env_hi+eps) - 20*log10(env_lo+eps)
        ln10 = tf.math.log(tf.constant(10.0, dtype=x.dtype))
        r_db = (20.0 / ln10) * (tf.math.log(env_hi + eps) - tf.math.log(env_lo + eps))

        # Soft-knee logistic around thr_db with slope tied to knee_db
        z  = (r_db - self.thr_db_w[0]) * self.slope[0]
        m0 = tf.math.sigmoid(z)

        # Asym attack/release smoothing
        prev   = tf.pad(m0, [[0,0],[1,0],[0,0]])[:, :-1, :]
        rising = tf.cast(m0 > prev, x.dtype)
        m_att  = _causal_conv1d(m0, self.k_att)
        m_rel  = _causal_conv1d(m0, self.k_rel)
        m1     = tf.clip_by_value(rising * m_att + (1.0 - rising) * m_rel, 0.0, 1.0)

        # Optional post compression (saturating), then floor and final gain
        if self.post_comp_p != 1.0:
            p = tf.constant(self.post_comp_p, dtype=x.dtype)
            y = 1.0 - tf.pow(1.0 - m1, p)
            if self.post_comp_gain != 1.0:
                y = tf.clip_by_value(tf.constant(self.post_comp_gain, dtype=x.dtype) * y, 0.0, 1.0)
        else:
            y = m1

        if self.floor > 0.0:
            fl = tf.constant(self.floor, dtype=x.dtype)
            y = fl + (1.0 - fl) * y

        return tf.clip_by_value(y * self.gain[0], 0.0, 1.0)

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "hp_cut_hz": self.hp_cut_hz,
            "hp_order": self.hp_order,
            "env_fast_ms": self.env_fast_ms,
            "env_slow_ms": self.env_slow_ms,
            "thr_db": self.thr_db,
            "knee_db": self.knee_db,
            "smooth_attack_ms": self.smooth_attack_ms,
            "smooth_release_ms": self.smooth_release_ms,
            "ref_is_hp": self.ref_is_hp,
            "post_comp_p": self.post_comp_p,
            "post_comp_gain": self.post_comp_gain,
            "floor": self.floor,
            "out_gain": self.out_gain,
            **super().get_config()
        }

@register_keras_serializable()
def _pick_mask_engine(inp,
                         sample_rate,
                         prefix,
                         hp_order=129,
                         env_fast_ms=0.25,
                         env_slow_ms=10.0,
                         thr_db=0.0,
                         knee_db=12.0,
                         smooth_attack_ms=0.06,
                         smooth_release_ms=6.0,
                         ref_is_hp=True,
                         post_comp_p=1.6,
                         post_comp_gain=1.15,
                         floor=0.00,
                         out_gain=1.0):
    """
    Replacement for _pick_mask_engine using dB-domain detector + soft-knee + post compression.
    """
    return CausalPickDBMask(
        sample_rate=sample_rate,
        hp_cut_hz=hp_cutoff_for_sr(sample_rate),
        hp_order=hp_order,
        env_fast_ms=env_fast_ms,
        env_slow_ms=env_slow_ms,
        thr_db=thr_db,
        knee_db=knee_db,
        smooth_attack_ms=smooth_attack_ms,
        smooth_release_ms=smooth_release_ms,
        ref_is_hp=bool(ref_is_hp),
        post_comp_p=post_comp_p,
        post_comp_gain=post_comp_gain,
        floor=floor,
        out_gain=out_gain,
        name=f"{prefix}_pick_mask_db"
    )(inp)

@register_keras_serializable()
class SoftClip1D(tf.keras.layers.Layer):
    """
    Per-sample soft clip: y = tanh(drive * x) / tanh(drive) * limit
    Use limit<=1 to bound, drive>0 controls hardness.
    """
    def __init__(self, drive=2.0, limit=0.98, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.drive = float(drive)
        self.limit = float(limit)
    def call(self, x):
        d = tf.constant(self.drive, dtype=x.dtype)
        lim = tf.constant(self.limit, dtype=x.dtype)
        return lim * tf.math.tanh(d * x) / tf.math.tanh(d)
    def get_config(self):
        return {"drive": self.drive, "limit": self.limit, **super().get_config()}

@register_keras_serializable()
class LearnedModalBank(tf.keras.layers.Layer):
    def __init__(self, centers_hz, decays_ms, length, sample_rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.centers_hz = [float(x) for x in centers_hz]
        self.decays_ms  = [float(x) for x in decays_ms]
        self.length     = int(length)
        self.sample_rate= int(sample_rate)

    def build(self, input_shape):
        K = _make_modal_kernels(self.centers_hz, self.decays_ms, self.length, self.sample_rate)
        self.kernel = self.add_weight(
            name="modal_kernels_trainable",
            shape=K.shape,
            initializer=tf.keras.initializers.Constant(K),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        shp = tf.shape(x); b = shp[0]; t = shp[1]; c = tf.reduce_prod(shp[2:])
        x3 = tf.reshape(x, [b, t, c])
        pad = self.length - 1
        x_pad = tf.pad(x3, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.centers_hz))

    def get_config(self):
        return {
            "centers_hz": self.centers_hz,
            "decays_ms":  self.decays_ms,
            "length":     self.length,
            "sample_rate":self.sample_rate,
            **super().get_config()
        }

@register_keras_serializable()
class ZeroBias(constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)
    def get_config(self):
        return {}

@register_keras_serializable()
class ConcatTime(tf.keras.layers.Layer):
    """Concat [state, x] along time axis."""
    def call(self, inputs):
        s, x = inputs
        return tf.concat([s, x], axis=1)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class TakeLastN(tf.keras.layers.Layer):
    def __init__(self, n, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n = int(n)
    def call(self, x):
        return _take_last(x, self.n)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n": self.n})
        return cfg

@register_keras_serializable()
class StreamFixedFIR(tf.keras.layers.Layer):
    """
    Generic streaming fixed FIR with explicit state.
    Inputs: [x, state] where state has shape (L-1, C_in)
    Kernel shape: (L, 1, C_out) (time-reversed taps for conv1d)
    Returns: [y (B,T,C_out), state_next (B,L-1,C_in)]
    If taps_init is provided (np.array or tf.Tensor of shape (L,1,C_out)), it is used
    as the initializer instead of zeros. This avoids 'silent branch' when no copy-by-name occurs.
    """
    def __init__(self, kernel_len, out_ch, weight_name, trainable_kernel=False, taps_init=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kernel_len = int(kernel_len)
        self.out_ch = int(out_ch)
        self.weight_name = str(weight_name)
        self.trainable_kernel = bool(trainable_kernel)
        # Keep init only in memory; do not serialize to JSON config
        if taps_init is None:
            self._taps_init_np = None
        else:
            if tf.is_tensor(taps_init):
                self._taps_init_np = taps_init.numpy().astype(np.float32)
            else:
                self._taps_init_np = np.asarray(taps_init, dtype=np.float32)

    def build(self, input_shape):
        if self._taps_init_np is not None:
            k = self._taps_init_np
        else:
            k = np.zeros((self.kernel_len, 1, self.out_ch), dtype=np.float32)

        # Safety: enforce expected shape if out_ch==1 and a scalar kernel provided
        if k.shape != (self.kernel_len, 1, self.out_ch):
            # Try to reshape/broadcast if possible
            try:
                k = np.reshape(k, (self.kernel_len, 1, self.out_ch)).astype(np.float32)
            except Exception:
                # fallback to zeros with warning
                k = np.zeros((self.kernel_len, 1, self.out_ch), dtype=np.float32)

        self.kernel = self.add_weight(
            name=self.weight_name,
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=self.trainable_kernel
        )
        super().build(input_shape)

    def call(self, inputs):
        x, state = inputs  # x: (B,T,C_in), state: (B,L-1,C_in)
        T = tf.shape(x)[1]
        x_cat = tf.concat([state, x], axis=1)  # (B, L-1+T, C_in)
        y_full = tf.nn.conv1d(x_cat, filters=self.kernel, stride=1, padding='VALID')  # ideally (B, T, C_out)
        y = y_full[:, -T:, :]
        s_next = x_cat[:, - (self.kernel_len - 1):, :]
        return [y, s_next]

    def compute_output_shape(self, input_shapes):
        x_shp, _ = input_shapes
        b, t, _ = x_shp
        return [(b, t, self.out_ch), (b, self.kernel_len - 1, x_shp[2])]

    def get_config(self):
        # do not serialize taps_init (weights persist in the saved model anyway)
        return {
            "kernel_len": self.kernel_len,
            "out_ch": self.out_ch,
            "weight_name": self.weight_name,
            "trainable_kernel": self.trainable_kernel,
            **super().get_config()
        }

@register_keras_serializable()
class StreamingRealtimeIrDownsampleFIR(tf.keras.layers.Layer):
    """
    Streaming decimate-by-N with causal FIR.
    If use_ir_prototype=True, use the exact same taps as the non-streaming IR branch
    (global _ir_fir_kernel), guaranteeing equivalence for the IR path.
      inputs: [x (B,T,1), state (B,L-1,1)]
      outputs: [y_ds (B, ceil(T/N), 1), state_next (B,L-1,1)]
    """
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, kernel_size=REALTIME_IR_FIR_SIZE, use_ir_prototype=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.kernel_size = int(kernel_size)
        self.use_ir_prototype = bool(use_ir_prototype)

    def build(self, input_shape):
        if self.use_ir_prototype:
            k = _ir_prototype_kernel().astype(np.float32)  # (L,1,1), already reversed for conv1d
        else:
            cutoff = 0.45 / float(self.factor)
            taps = firwin(self.kernel_size, cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
            taps_min = _to_minimum_phase(taps)
            taps_min = taps_min / max(np.sum(taps_min), 1e-12)
            k = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)

        self.kernel = self.add_weight(
            name="ds_lpf_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        # Inputs are [x_fullrate, state] where state has (L-1) previous full-rate samples
        x, state = inputs  # x: (B,T,1), state: (B,L-1,1)
        # Concatenate state + new hop
        x_cat = tf.concat([state, x], axis=1)  # (B, L-1+T, 1)
        # Strided causal conv (decimator)
        y_full = tf.nn.conv1d(x_cat, filters=self.kernel, stride=self.factor, padding='VALID')  # (B, ?, 1)

        # Keep only the "new" low-rate samples for this hop:
        # n_new = ceil(T/factor)
        T = tf.shape(x)[1]
        n_new = tf.cast((T + self.factor - 1) // self.factor, tf.int32)
        y = y_full[:, -n_new:, :]  # (B, n_new, 1)

        # Next state is last (L-1) full-rate samples
        s_next = x_cat[:, - (tf.shape(self.kernel)[0] - 1):, :]
        return [y, s_next]

    def get_config(self):
        return {"factor": self.factor, "kernel_size": self.kernel_size, "use_ir_prototype": self.use_ir_prototype, **super().get_config()}

@register_keras_serializable()
class StreamingFIRInterpByN(tf.keras.layers.Layer):
    """
    Streaming version of FIRInterpByN (identical output, polyphase):
      inputs:  [x_low (B,Tl,1), state_low (B,S,1)]   S >= ceil(L/F) - 1 low-rate samples of history
      outputs: [y_up (B,Tl*F,1), state_next_low (B,S,1)]
    The state only has to cover the FIR's reach at the LOW rate: ceil(L/F)-1 samples
    (_interp_polyphase_len - 1). Graphs built before 2026-09-22 declared L-1 (8x too many) and
    zero-stuffed + filtered all of them every hop (~25% of the pedal model's MACs for the IR branch);
    such graphs still work: only the last J-1 state samples are read and the state keeps its length.
    """
    def __init__(self, factor, num_taps=257, cutoff=None, window=('kaiser', 9.0), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)
        self.cutoff = float(0.45 / self.factor) if cutoff is None else float(cutoff)
        self.window = window

    def build(self, input_shape):
        k = _interp_lowpass_kernel(self.num_taps, self.cutoff, self.window)
        self.kernel = self.add_weight(
            name="fir_interp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        x_low, state_low = inputs
        J = _interp_polyphase_len(self.num_taps, self.factor)
        S = int(state_low.shape[1])          # state inputs always have a static length
        if S < J - 1:
            raise ValueError(f"{self.name}: state has {S} samples, needs >= {J - 1}")
        hist = state_low[:, S - (J - 1):, :]
        y = _interp_polyphase_apply(tf.concat([hist, x_low], axis=1),
                                    _interp_polyphase_filters(self.kernel, self.factor), self.factor)
        if S == 0:
            return [y, state_low]
        s_next = tf.concat([state_low, x_low], axis=1)[:, -S:, :]
        return [y, s_next]

    def compute_output_shape(self, input_shapes):
        x_shape, s_shape = input_shapes
        t = x_shape[1]
        return [(x_shape[0], None if t is None else t * self.factor, 1), tuple(s_shape)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "factor": self.factor,
            "num_taps": self.num_taps,
            "cutoff": self.cutoff,
            "window": self.window
        })
        return cfg

@register_keras_serializable()
class StreamingPickDBMask(tf.keras.layers.Layer):
    """
    Streaming analytic pick detector in dB domain with soft-knee + post compression.
    Now includes a 1-sample continuity state for 'prev(m0)' so rising/decay selection
    is seamless across hops.

    Inputs:
      [ x, s_hp, s_fast, s_slow, s_att, s_rel, s_prev_m0 ]
        - x: (B,T,1)
        - s_*: streaming FIR states (B,L-1,1)
        - s_prev_m0: (B,1,1) last m0 sample from previous hop

    Outputs:
      [ y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_m0_n ]
        - y: (B,T,1) mask in [0,1]
        - s_prev_m0_n: (B,1,1) last m0 of this hop
    """
    def __init__(self,
                 sample_rate,
                 hp_cut_hz=3500.0,
                 hp_order=129,
                 env_fast_ms=0.25,
                 env_slow_ms=10.0,
                 thr_db=0.0,
                 knee_db=12.0,
                 smooth_attack_ms=0.06,
                 smooth_release_ms=6.0,
                 ref_is_hp=True,
                 post_comp_p=1.6,
                 post_comp_gain=1.15,
                 floor=0.00,
                 out_gain=1.0,
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate       = int(sample_rate)
        self.hp_cut_hz         = float(hp_cut_hz)
        self.hp_order          = int(hp_order)
        self.env_fast_ms       = float(env_fast_ms)
        self.env_slow_ms       = float(env_slow_ms)
        self.thr_db            = float(thr_db)
        self.knee_db           = float(knee_db)
        self.smooth_attack_ms  = float(smooth_attack_ms)
        self.smooth_release_ms = float(smooth_release_ms)
        self.ref_is_hp         = bool(ref_is_hp)
        self.post_comp_p       = float(post_comp_p)
        self.post_comp_gain    = float(post_comp_gain)
        self.floor             = float(floor)
        self.out_gain          = float(out_gain)

    def build(self, input_shape):
        k_hp = _get_minphase_hp_kernel(self.sample_rate, self.hp_cut_hz, self.hp_order)
        self.k_hp = self.add_weight(
            name="dbmask_hp_taps", shape=k_hp.shape,
            initializer=tf.keras.initializers.Constant(k_hp), trainable=False
        )
        def ema_ms(ms, scale, Lmax):
            smpls = max(1, int(round((float(ms)/1000.0) * self.sample_rate)))
            return _exp_smoothing_kernel(smpls, scale=scale, Lmax=Lmax)
        k_fast = ema_ms(self.env_fast_ms, scale=3.0, Lmax=1024)
        k_slow = ema_ms(self.env_slow_ms, scale=3.0, Lmax=4096)
        k_att  = ema_ms(self.smooth_attack_ms,  scale=4.0, Lmax=2048)
        k_rel  = ema_ms(self.smooth_release_ms, scale=4.0, Lmax=2048)

        self.k_fast = self.add_weight(
            name="dbmask_env_fast", shape=k_fast.shape,
            initializer=tf.keras.initializers.Constant(k_fast), trainable=False
        )
        self.k_slow = self.add_weight(
            name="dbmask_env_slow", shape=k_slow.shape,
            initializer=tf.keras.initializers.Constant(k_slow), trainable=False
        )
        self.k_att  = self.add_weight(
            name="dbmask_att", shape=k_att.shape,
            initializer=tf.keras.initializers.Constant(k_att), trainable=False
        )
        self.k_rel  = self.add_weight(
            name="dbmask_rel", shape=k_rel.shape,
            initializer=tf.keras.initializers.Constant(k_rel), trainable=False
        )

        slope_val = float(np.log(9.0) / max(self.knee_db, 1e-3))
        self.slope = self.add_weight(
            name="dbmask_sigmoid_slope", shape=(1,),
            initializer=tf.keras.initializers.Constant([slope_val]), trainable=False
        )
        self.thr_db_w = self.add_weight(
            name="dbmask_thr_db", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.thr_db]), trainable=False
        )
        self.gain = self.add_weight(
            name="dbmask_out_gain", shape=(1,),
            initializer=tf.keras.initializers.Constant([self.out_gain]), trainable=False
        )
        super().build(input_shape)

    def _stream_conv(self, x, k, s):
        x_cat = tf.concat([s, x], axis=1)  # (B, L-1+T, C)
        y = tf.nn.conv1d(x_cat, filters=k, stride=1, padding='VALID')
        T = tf.shape(x)[1]
        y = y[:, -T:, :]
        s_next = x_cat[:, tf.shape(x_cat)[1] - (tf.shape(k)[0] - 1):, :]
        return y, s_next

    def call(self, inputs):
        # Backward-compat: accept 6 or 7 inputs
        if len(inputs) == 6:
            x, s_hp, s_fast, s_slow, s_att, s_rel = inputs
            s_prev_m0 = None
        else:
            x, s_hp, s_fast, s_slow, s_att, s_rel, s_prev_m0 = inputs

        eps = tf.constant(1e-8, dtype=x.dtype)

        hp, s_hp_n = self._stream_conv(x, self.k_hp, s_hp)
        abs_hp = tf.abs(hp)

        env_hi, s_fast_n = self._stream_conv(abs_hp, self.k_fast, s_fast)
        if self.ref_is_hp:
            env_lo, s_slow_n = self._stream_conv(abs_hp, self.k_slow, s_slow)
        else:
            abs_x = tf.abs(x)
            env_lo, s_slow_n = self._stream_conv(abs_x, self.k_slow, s_slow)

        ln10 = tf.math.log(tf.constant(10.0, dtype=x.dtype))
        r_db = (20.0 / ln10) * (tf.math.log(env_hi + eps) - tf.math.log(env_lo + eps))
        z  = (r_db - self.thr_db_w[0]) * self.slope[0]
        m0 = tf.math.sigmoid(z)

        # Continuous prev across hops
        if s_prev_m0 is None:
            prev = tf.pad(m0, [[0,0],[1,0],[0,0]])[:, :-1, :]
        else:
            prev = tf.concat([s_prev_m0, m0[:, :-1, :]], axis=1)
        s_prev_m0_n = m0[:, -1:, :]

        rising = tf.cast(m0 > prev, x.dtype)
        m_att, s_att_n = self._stream_conv(m0, self.k_att, s_att)
        m_rel, s_rel_n = self._stream_conv(m0, self.k_rel, s_rel)
        m1 = tf.clip_by_value(rising * m_att + (1.0 - rising) * m_rel, 0.0, 1.0)

        if self.post_comp_p != 1.0:
            p = tf.constant(self.post_comp_p, dtype=x.dtype)
            y = 1.0 - tf.pow(1.0 - m1, p)
            if self.post_comp_gain != 1.0:
                y = tf.clip_by_value(tf.constant(self.post_comp_gain, dtype=x.dtype) * y, 0.0, 1.0)
        else:
            y = m1

        if self.floor > 0.0:
            fl = tf.constant(self.floor, dtype=x.dtype)
            y = fl + (1.0 - fl) * y

        y = tf.clip_by_value(y * self.gain[0], 0.0, 1.0)

        if s_prev_m0 is None:
            return [y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n]
        return [y, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_m0_n]

    def get_config(self):
        return {
            "sample_rate": self.sample_rate,
            "hp_cut_hz": self.hp_cut_hz,
            "hp_order": self.hp_order,
            "env_fast_ms": self.env_fast_ms,
            "env_slow_ms": self.env_slow_ms,
            "thr_db": self.thr_db,
            "knee_db": self.knee_db,
            "smooth_attack_ms": self.smooth_attack_ms,
            "smooth_release_ms": self.smooth_release_ms,
            "ref_is_hp": self.ref_is_hp,
            "post_comp_p": self.post_comp_p,
            "post_comp_gain": self.post_comp_gain,
            "floor": self.floor,
            "out_gain": self.out_gain,
            **super().get_config()
        }

@register_keras_serializable()
class StreamingDiffPos(tf.keras.layers.Layer):
    """
    Streaming positive first difference with 1-sample continuity state:
      inputs: [x (B,T,1), s_prev (B,1,1)]
      outputs: [y (B,T,1), s_next (B,1,1)]
      y[n] = relu(x[n] - x[n-1]) with x[-1] coming from s_prev.
    """
    def call(self, inputs):
        x, s_prev = inputs
        prev = tf.concat([s_prev, x[:, :-1, :]], axis=1)
        y = tf.nn.relu(x - prev)
        s_next = x[:, -1:, :]
        return [y, s_next]

    def get_config(self):
        return super().get_config()
    
@register_keras_serializable()
class PreEmphasisStream(tf.keras.layers.Layer):
    """
    Streaming 1st-order pre-emphasis with explicit state:
      inputs: [x (B,T,1), s (B,1,1)] where s is previous x[-1]
      outputs: [y (B,T,1), s_next (B,1,1)]
      y[n] = x[n] - alpha * x[n-1]
    """
    def __init__(self, alpha=HP_PRE_EMPHASIS_HF, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs):
        x, s = inputs
        prev = tf.concat([s, x[:, :-1, :]], axis=1)
        y = x - tf.constant(self.alpha, dtype=x.dtype) * prev
        s_next = x[:, -1:, :]
        return [y, s_next]

    def get_config(self):
        return {"alpha": self.alpha, **super().get_config()}

# ────────────────────────────── DATA PIPELINE HELPERS ───────────────────────

@register_keras_serializable()
class Abs1D(tf.keras.layers.Layer):
    def call(self, x):
        return tf.abs(x)

# ────────────────────────────── CUSTOM LAYERS ────────────────────────────────

@register_keras_serializable()
class ConstScale(tf.keras.layers.Layer):
    def __init__(self, scale, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = float(scale)
    def call(self, x):
        return x * self.scale
    def get_config(self):
        return {"scale": self.scale, **super().get_config()}
                        
@register_keras_serializable()
class FIRInterpByN(tf.keras.layers.Layer):
    """
    Upsample by integer factor N: zero insertion + fixed causal FIR (anti-imaging), computed in
    polyphase form (identical result, N× fewer multiplications).
    - factor=N
    - num_taps: e.g. 257
    - cutoff is normalized to Nyquist of the upsampled rate as a fraction in (0,1).
    Taps are min-phase (see _interp_lowpass_kernel). Input must have 1 channel.
    """
    def __init__(self, factor, num_taps=257, cutoff=None, window=('kaiser', 9.0), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.num_taps = int(num_taps)
        self.cutoff = float(0.45 / self.factor) if cutoff is None else float(cutoff)
        # Store as-is; _interp_window() normalises Keras TrackedList on load
        self.window = window

    def build(self, input_shape):
        k = _interp_lowpass_kernel(self.num_taps, self.cutoff, self.window)
        self.kernel = self.add_weight(
            name="fir_interp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, 1) → (B, T*F, 1); zero history before the first sample (causal)
        J = _interp_polyphase_len(self.num_taps, self.factor)
        x_pad = tf.pad(x, [[0, 0], [J - 1, 0], [0, 0]])
        return _interp_polyphase_apply(x_pad, _interp_polyphase_filters(self.kernel, self.factor), self.factor)

    def compute_output_shape(self, input_shape):
        b, t, c = input_shape
        return (b, None if t is None else t * self.factor, c)

    def get_config(self):
        # Return JSON-serializable fields; tuples will become lists in JSON, which is fine
        return {
            "factor": self.factor,
            "num_taps": self.num_taps,
            "cutoff": self.cutoff,
            "window": self.window,
            **super().get_config()
        }

@register_keras_serializable()
class BandpassFIR(tf.keras.layers.Layer):
    def __init__(self, low_hz, high_hz, order, sample_rate, window="hamming", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.order = int(order)
        self.sample_rate = int(sample_rate)
        self.window = str(window)

    def build(self, input_shape):
        k = _get_minphase_bp_kernel(self.sample_rate, self.low_hz, self.high_hz, self.order, window=self.window)
        self.taps = self.add_weight(
            name="bp_taps",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )

    def call(self, x):
        pad = tf.shape(self.taps)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "order": self.order,
            "sample_rate": self.sample_rate,
            "window": self.window
        })
        return cfg

@register_keras_serializable()
class HFHighpass(tf.keras.layers.Layer):
    def __init__(self, taps=None, sample_rate=None, cutoff_hz=OFFLINE_HP_CUTOFF_HZ, order=OFFLINE_HF_HP_ORDER, **kwargs):
        """
        Causal min-phase FIR high-pass, implemented as conv1d with left padding.
        - If 'taps' is provided, it is used as-is (Tensor/ndarray of shape [L,1,1], already time-reversed for conv1d).
        - Otherwise, taps are generated from (sample_rate, cutoff_hz, order) via _get_minphase_hp_kernel.
        Taps are saved as non-trainable weights for TFLite export.
        """
        super().__init__(**kwargs)
        self.sample_rate = None if sample_rate is None else int(sample_rate)
        self.cutoff_hz   = float(cutoff_hz)
        self.order       = int(order)
        self._taps_init  = None if taps is None else tf.convert_to_tensor(taps, dtype=tf.float32)

    def build(self, input_shape):
        if self._taps_init is None:
            # Fallback to a known SR if none provided (should not happen if you pass sample_rate explicitly)
            sr = self.sample_rate if self.sample_rate is not None else OFFLINE_SAMPLE_RATE
            init_taps = _get_minphase_hp_kernel(sr, self.cutoff_hz, self.order)
        else:
            init_taps = self._taps_init

        self.taps = self.add_weight(
            name="hp_taps",
            shape=init_taps.shape,
            initializer=tf.keras.initializers.Constant(init_taps),
            trainable=False
        )

    def call(self, x):
        pad = tf.shape(self.taps)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.taps, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "cutoff_hz":   self.cutoff_hz,
            "order":       self.order
        })
        # taps are stored as weights; no need to serialize them in config
        return cfg

@register_keras_serializable()
class RealtimeIrDownsampleFIR(Layer):
    def __init__(self, factor=REALTIME_IR_DOWNSAMPLE_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.factor = int(factor)
        self.kernel_size = REALTIME_IR_FIR_SIZE

    def call(self, x):
        p = x[..., 0:1]
        pad = self.kernel_size - 1
        p_pad = tf.pad(p, [[0, 0], [pad, 0], [0, 0]])
        return tf.nn.conv1d(p_pad, filters=_ir_fir_tf, stride=self.factor, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, T, _ = input_shape
        if T is None:
            return (b, None, 1)
        # With left pad (L-1) and VALID conv: out_T = floor((T-1)/factor) + 1 = ceil(T/factor)
        out_T = (T + self.factor - 1) // self.factor
        return (b, out_T, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'factor': self.factor})
        return cfg
    
@register_keras_serializable()
class MatchTimeToRef(Layer):
    """
    Crops or pads the time axis of x to match the time length of ref.
    Always returns exactly T_ref samples (no control flow/cond needed).
    """
    def call(self, inputs):
        x, ref = inputs
        T_ref = tf.shape(ref)[1]
        pad_amt = tf.maximum(0, T_ref - tf.shape(x)[1])
        x_pad = tf.pad(x, [[0,0],[0,pad_amt],[0,0]])
        return x_pad[:, :T_ref, :]

    def compute_output_shape(self, input_shapes):
        x_shape, ref_shape = input_shapes
        return (x_shape[0], ref_shape[1], x_shape[2])
    

@register_keras_serializable()
class IrScale(Layer):
    def __init__(self, init=0.1, **kwargs):
        super().__init__(**kwargs)
        self.init = init
    def build(self, input_shape):
        self.scale = self.add_weight(
            name="ir_scale",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True
        )
    def call(self, x):
        return x * self.scale
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init": self.init})
        return cfg
    
@register_keras_serializable()
class BlendScalar(Layer):
    """
    Blend two scalar tensors y = (1 - alpha) * student + alpha * teacher.
    alpha in [0,1]. If trainable=False, it's a fixed convex combination.
    Inputs must be broadcast-compatible (typically shape (B,1)).
    """
    def __init__(self, init_alpha=0.5, trainable=False, **kwargs):
        # Trainability is a LAYER-level flag: a variable created with add_weight(trainable=False)
        # can never be re-enabled by `layer.trainable = True` in Keras 3 (it lives in the
        # non-trainable list), which silently defeated set_trainable_layers(enable_style_blend).
        super().__init__(trainable=bool(trainable), **kwargs)
        self.init_alpha = float(init_alpha)
        self._trainable_alpha = bool(trainable)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.init_alpha),
            trainable=True
        )

    def call(self, inputs):
        student, teacher = inputs
        a = tf.clip_by_value(self.alpha, 0.0, 1.0)
        return (1.0 - a) * student + a * teacher

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "init_alpha": self.init_alpha,
            "trainable": self._trainable_alpha
        })
        return cfg

# Add near your other custom layers
@register_keras_serializable()
class LossTether(tf.keras.layers.Layer):
    """
    Connects auxiliary tensors (e.g., outputs of KD layers that only add losses)
    to the model graph without altering the primary tensor.
    Inputs: [primary, aux1, aux2, ...]
    Output: primary (unchanged)
    This ensures all aux tensors are 'reachable' so their add_loss is kept.
    """
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return inputs[0]
        return inputs
    
    
@register_keras_serializable()
class CausalExpSmoother(tf.keras.layers.Layer):
    """
    Causal exponential moving average implemented as a fixed FIR (left-padded conv1d).
    time_const_ms is the EMA time constant in milliseconds.
    """
    def __init__(self, time_const_ms, sample_rate, scale=4.0, Lmax=4096, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.time_const_ms = float(time_const_ms)
        self.sample_rate   = int(sample_rate)
        self.scale         = float(scale)
        self.Lmax          = int(Lmax)

    def build(self, input_shape):
        tau_samples = max(1, int(round((self.time_const_ms / 1000.0) * self.sample_rate)))
        k = _exp_smoothing_kernel(tau_samples, scale=self.scale, Lmax=self.Lmax)
        # IMPORTANT: pass a NumPy value to Constant initializer to avoid cross-graph capture
        k_init = k.numpy() if tf.is_tensor(k) else np.array(k, dtype=np.float32, copy=False)
        self.kernel = self.add_weight(
            name="exp_smooth_taps",
            shape=k_init.shape,
            initializer=tf.keras.initializers.Constant(k_init),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'time_const_ms': self.time_const_ms,
            'sample_rate': self.sample_rate,
            'scale': self.scale,
            'Lmax': self.Lmax
        })
        return cfg

@register_keras_serializable()
class AdjustableAlphaMixin:
    def build_alpha(self, init_alpha, name="alpha"):
        self.alpha_var = self.add_weight(
            name=name,
            shape=(),
            initializer=tf.keras.initializers.Constant(float(init_alpha)),
            trainable=False
        )
    def set_alpha(self, new_alpha):
        self.alpha_var.assign(float(new_alpha))
    def get_config(self):
        return {}
    
@register_keras_serializable()
class HFLowbandPenalty(Layer, AdjustableAlphaMixin):
    """
    Penalize low-band energy in the student's HF branch: L1(LPF(hf_final)).
    Returns input unchanged; training-only constraint.
    """
    def __init__(self, alpha, cutoff_hz=2800.0, order=129, sample_rate=REALTIME_SAMPLE_RATE, window="hamming",
                 core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self.sample_rate = int(sample_rate)
        self.window = str(window)
        # Only the fully-contexted core of a training segment is penalised (see *_LOSS_CORE)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

    def build(self, input_shape):
        k = _make_lpf_kernel(self.sample_rate, self.cutoff_hz, self.order, self.window)
        self.kernel = self.add_weight(
            name="lp_taps_for_penalty",
            shape=k.shape,
            initializer=tf.keras.initializers.Constant(k),
            trainable=False
        )
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, x, training=None):
        if not training:
            return x
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
        lp = tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')
        loss = tf.reduce_mean(tf.abs(_crop_time(lp, self.core_start, self.core_end)))
        self.add_loss(self.alpha_var * loss)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'alpha': float(self.init_alpha),
            'cutoff_hz': self.cutoff_hz,
            'order': self.order,
            'sample_rate': self.sample_rate,
            'window': self.window,
            'core_start': self.core_start,
            'core_end': self.core_end
        })
        return cfg

@register_keras_serializable()
class GaussianWeights1DLayer(tf.keras.layers.Layer):
    def __init__(self, centers, alpha=0.002, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._centers_np = np.asarray(centers, dtype=np.float32).reshape(1, -1)
        self.alpha = float(alpha)

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=self._centers_np.shape,
            initializer=tf.keras.initializers.Constant(self._centers_np),
            trainable=False
        )
        super().build(input_shape)

    def call(self, knob_scalar):
        # knob_scalar: (B,1), centers: (1,N)
        x = tf.abs(knob_scalar - self.centers)            # (B,N)
        w = tf.exp(-self.alpha * x)                       # (B,N)
        w = w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-8)
        return w

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "centers": self._centers_np.tolist(),
            "alpha": self.alpha
        })
        return cfg


@register_keras_serializable()
class OuterWeightsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        w_a, w_b = inputs  # w_a: (B,Na), w_b: (B,Nb)
        a = tf.expand_dims(w_a, axis=2)  # (B,Na,1)
        b = tf.expand_dims(w_b, axis=1)  # (B,1,Nb)
        w = a * b                        # (B,Na,Nb)
        shape = tf.shape(w)
        return tf.reshape(w, [shape[0], shape[1] * shape[2]])  # (B, Na*Nb)


@register_keras_serializable()
class ClampScalarLayer(tf.keras.layers.Layer):
    def __init__(self, lo, hi, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.lo = float(lo)
        self.hi = float(hi)

    def call(self, x):
        return tf.clip_by_value(x, self.lo, self.hi)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lo": self.lo, "hi": self.hi})
        return cfg


@register_keras_serializable()
class DbToLinScalarLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.pow(10.0, x / 20.0)


@register_keras_serializable()
class AddScalarLayer(tf.keras.layers.Layer):
    def __init__(self, value, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = float(value)

    def call(self, x):
        return x + self.value

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"value": self.value})
        return cfg


@register_keras_serializable()
class ReduceSumAxis2Layer(tf.keras.layers.Layer):
    def __init__(self, keepdims=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.keepdims = bool(keepdims)

    def call(self, x):
        return tf.reduce_sum(x, axis=2, keepdims=self.keepdims)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"keepdims": self.keepdims})
        return cfg


@register_keras_serializable()
def gaussian_weights_1d(knob_scalar, centers, alpha=0.002):
    return GaussianWeights1DLayer(centers=centers, alpha=alpha)(knob_scalar)

@register_keras_serializable()
def outer_weights(w_a, w_b):
    return OuterWeightsLayer()([w_a, w_b])

@register_keras_serializable()
def clamp_scalar(x, lo, hi):
    return ClampScalarLayer(lo=lo, hi=hi)(x)

@register_keras_serializable()
def db_to_lin_scalar(x):
    return DbToLinScalarLayer()(x)

@register_keras_serializable()
def phase_polarity_gain(switch01):
    """
    switch01: (B,1), 0 → +1, 1 → -1. 
    """
    s = tf.clip_by_value(switch01, 0.0, 1.0)
    return 1.0 - 2.0 * s

@register_keras_serializable()
class FixedFIRBank(tf.keras.layers.Layer):
    """
    Non-streaming, fixed FIR bank: taps shape (L,1,K) reversed for conv1d (cross-correlation).
    Outputs (B,T,K). Left-padded by (L-1).

    Keras-serialization-friendly:
      - taps_3d is optional. If absent, we create a zero-initialized (L,1,K) kernel and rely on
        weight restoration from the saved model. This allows load_model to succeed even when only
        L and K are present in config (the usual case).
      - If taps_3d is provided (during graph construction), we use it as the initializer.
    """
    def __init__(self, taps_3d=None, L=None, K=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._taps_init = None
        # L and K may be None when building brand-new layers; they are present when deserializing
        self.L = None if L is None else int(L)
        self.K = None if K is None else int(K)

        if taps_3d is not None:
            k = np.asarray(taps_3d, dtype=np.float32)
            if k.ndim != 3 or k.shape[1] != 1:
                raise ValueError(f"taps_3d must have shape (L,1,K); got {k.shape}")
            self._taps_init = k
            self.L = int(k.shape[0])
            self.K = int(k.shape[2])

    def build(self, input_shape):
        if self._taps_init is not None:
            shape = self._taps_init.shape
            initializer = tf.keras.initializers.Constant(self._taps_init)
        else:
            # When loading from a saved model, weights will be restored after construction.
            # Create a kernel with the correct shape so weight loading can succeed.
            if self.L is None or self.K is None:
                # As a last resort, default to (3,1,1) (will be overwritten by weight loading).
                L = 3 if self.L is None else int(self.L)
                K = 1 if self.K is None else int(self.K)
            else:
                L = int(self.L)
                K = int(self.K)
            shape = (L, 1, K)
            initializer = tf.keras.initializers.Zeros()

        self.kernel = self.add_weight(
            name="firbank_taps",
            shape=shape,
            initializer=initializer,
            trainable=False
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B,T,1) → pad left by (L-1)
        pad = tf.shape(self.kernel)[0] - 1
        x_pad = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
        return tf.nn.conv1d(x_pad, filters=self.kernel, stride=1, padding='VALID')

    def compute_output_shape(self, input_shape):
        b, t, _ = input_shape
        k = self.K if self.K is not None else (int(self.kernel.shape[2]) if hasattr(self, "kernel") else None)
        return (b, t, k)

    def get_config(self):
        # We only need to serialize L and K. taps are stored as weights.
        cfg = super().get_config()
        cfg.update({
            "L": self.L,
            "K": self.K
        })
        return cfg

@register_keras_serializable()
class TileStyle(Layer):
    def call(self, inputs):
        style_scalar, ref = inputs
        # ref has shape (B, T, C); we need to tile a (B,1,1) scalar across time T
        T = tf.shape(ref)[1]
        # reshape with -1 for the (possibly dynamic) batch dim
        s = tf.reshape(style_scalar, (-1, 1, 1))
        multiples = tf.stack([1, T, 1])  # build a Tensor, not a Python tuple with a Tensor inside
        return tf.tile(s, multiples)

    def compute_output_shape(self, input_shapes):
        # (batch, time, channels) -> channels=1 for style mask
        return (input_shapes[1][0], input_shapes[1][1], 1)

@register_keras_serializable()
class InvertStyle(Layer):
    def call(self, style_time):
        return 1.0 - style_time
    def compute_output_shape(self, input_shape):
        return input_shape
