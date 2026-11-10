import numpy as np
import tensorflow as tf
from scipy.signal import firwin
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Conv1D, Layer)  # pyright: ignore[reportMissingImports]

from .config import (
    OFFLINE_SAMPLE_RATE,
    REALTIME_SAMPLE_RATE,
)
from .layers import (
    AdjustableAlphaMixin,
)

    
# ─── Knowledge-distillation layers (realtime training graph only) ────────────
# These attach annealed KD losses (see KDAlphaScheduler) and return the student tensor
# unchanged, so they can be wired in-line on the student's output path. The losses are added in
# TRAINING only: they are not part of the objective that val_loss / EarlyStopping / checkpoint
# selection should track, and their alphas anneal, which would make val_loss fall by schedule.

_kd_decimator_cache = {}

def _kd_decimator_taps(factor):
    """
    Symmetric (zero-phase) anti-alias FIR for the teacher → student-rate decimation: EXACTLY the
    filter scipy.signal.resample_poly(x, 1, factor) uses, i.e. the one the student's own 48k
    targets were made with, so the KD target is band-limited like the real target.
    Returns (taps, half_length).
    """
    f = int(factor)
    hit = _kd_decimator_cache.get(f)
    if hit is None:
        half = 10 * f
        taps = firwin(2 * half + 1, 1.0 / float(f), window=('kaiser', 5.0)).astype(np.float32)
        hit = (taps, half)
        _kd_decimator_cache[f] = hit
    return hit

KD_TEACHER_DECIM_MARGIN = _kd_decimator_taps(OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE)[1]

def _kd_align(student, teacher, factor, core_start, core_end):
    """
    Bring a teacher tensor (B, factor*T [+ margin], C) to the student rate and crop both tensors to
    the loss core [core_start, core_end) (student-rate samples). Student sample n corresponds to
    teacher sample factor*n (x48 = resample_poly(x96, 1, 2)), so the teacher is low-passed with the
    resample_poly filter centred on factor*n and decimated — the previous 2-tap avg-pool sat half a
    96k-sample late and let 24-48 kHz teacher content alias into the KD target at only -3..-10 dB.
    Needs KD_TEACHER_DECIM_MARGIN teacher samples beyond factor*core_end (the training graph keeps
    them when it crops the teacher's right context); anything missing is zero-padded.
    """
    f = int(factor)
    taps, H = _kd_decimator_taps(f)
    n_s = tf.shape(student)[1]
    cs = 0 if core_start is None else int(core_start)
    ce = n_s if core_end is None else tf.minimum(int(core_end), n_s)
    s = student[:, cs:ce, :]

    t_lo = f * cs - H                       # first teacher sample the filter needs
    t_hi = f * (ce - 1) + H + 1             # one past the last
    pad_l = max(0, -t_lo)
    pad_r = tf.maximum(0, t_hi - tf.shape(teacher)[1])
    tp = tf.pad(teacher, [[0, 0], [pad_l, pad_r], [0, 0]])
    seg = tp[:, t_lo + pad_l:t_hi + pad_l, :]

    # Depthwise strided FIR: fold channels into the batch so one (L,1,1) kernel serves all of them
    b, c = tf.shape(seg)[0], tf.shape(seg)[2]
    x = tf.reshape(tf.transpose(seg, [0, 2, 1]), [b * c, -1, 1])
    k = tf.constant(taps[::-1].reshape(-1, 1, 1), dtype=seg.dtype)
    y = tf.nn.conv1d(x, filters=k, stride=f, padding="VALID")     # (b*c, ce-cs, 1)
    t = tf.transpose(tf.reshape(y, [b, c, -1]), [0, 2, 1])
    return s, t


@register_keras_serializable()
class KDWaveformDistill(Layer, AdjustableAlphaMixin):
    """
    L1 KD between a student signal (B,T,1) and a teacher signal (B,factor*T,1), evaluated on the
    loss core only. Returns the student unchanged.
    """
    def __init__(self, alpha, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        super().build(input_shape)

    def call(self, inputs, training=None):
        student, teacher = inputs
        if not training:
            return student
        s, t = _kd_align(student, teacher, self.factor, self.core_start, self.core_end)
        loss = tf.reduce_mean(tf.abs(s - t))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "factor": self.factor,
                "core_start": self.core_start, "core_end": self.core_end, **super().get_config()}


@register_keras_serializable()
class KDSTFTDistill(Layer, AdjustableAlphaMixin):
    """
    Multi-resolution STFT-magnitude L1 KD between student audio (B,T,1) and teacher audio
    (B,factor*T,1), evaluated on the loss core only. Returns the student unchanged.
    """
    SCALES = ((256, 64), (1024, 256))  # (win/n_fft, hop)

    def __init__(self, alpha, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        self._wins = [tf.constant(tf.reshape(tf.signal.hann_window(w, periodic=True), (1, 1, w)))
                      for (w, _) in self.SCALES]
        super().build(input_shape)

    def call(self, inputs, training=None):
        student, teacher = inputs
        if not training:
            return student
        s3, t3 = _kd_align(student, teacher, self.factor, self.core_start, self.core_end)
        s = tf.squeeze(s3, axis=-1)
        t = tf.squeeze(t3, axis=-1)

        losses = []
        for (win, hop), w in zip(self.SCALES, self._wins):
            fs = tf.signal.frame(s, frame_length=win, frame_step=hop, pad_end=True) * w
            ft = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True) * w
            mag_s = tf.abs(tf.signal.rfft(fs, fft_length=[win]))
            mag_t = tf.abs(tf.signal.rfft(ft, fft_length=[win]))
            losses.append(tf.reduce_mean(tf.abs(mag_s - mag_t)))
        loss = tf.add_n(losses) / float(len(losses))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "factor": self.factor,
                "core_start": self.core_start, "core_end": self.core_end, **super().get_config()}


@register_keras_serializable()
class KDFeatureDistill(Layer, AdjustableAlphaMixin):
    """
    Feature-space KD: projects student features (B,T,Cs) to the teacher width with a trainable
    1×1 conv, avg-pools teacher features (B,factor*T,Ct) down to the student rate, and adds an
    L1 loss on the loss core. Returns the student features unchanged.
    """
    def __init__(self, alpha, teacher_channels, factor=2, core_start=None, core_end=None, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self.teacher_channels = int(teacher_channels)
        self.factor = int(factor)
        self.core_start = None if core_start is None else int(core_start)
        self.core_end = None if core_end is None else int(core_end)
        self.proj = Conv1D(self.teacher_channels, 1, padding="same",
                           name=f"{self.name}_proj")

    def build(self, input_shape):
        self.build_alpha(self.init_alpha, name="alpha")
        # Build the projection up front: call() skips it outside training, and weights created
        # lazily inside the first train step would not be known to the optimizer.
        self.proj.build(tuple(input_shape[0]))
        super().build(input_shape)

    def call(self, inputs, training=None):
        student, teacher = inputs
        if not training:
            return student
        p = self.proj(student)
        s, t = _kd_align(p, teacher, self.factor, self.core_start, self.core_end)
        loss = tf.reduce_mean(tf.abs(s - t))
        self.add_loss(self.alpha_var * loss)
        return student

    def get_config(self):
        return {"alpha": self.init_alpha, "teacher_channels": self.teacher_channels,
                "factor": self.factor, "core_start": self.core_start, "core_end": self.core_end,
                **super().get_config()}
