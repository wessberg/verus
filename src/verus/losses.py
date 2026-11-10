import numpy as np
import tensorflow as tf
from scipy.signal import firwin
from tensorflow.keras.losses import Loss  # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]

from . import config
from .config import (
    COMPLEX_STFT_WEIGHT,
    D2_WEIGHT,
    HF_ONSET_GAIN,
    HP_PRE_EMPHASIS_HF,
    LMAG_FLOOR_DB,
    LMAG_SCALES_OFFLINE,
    LMAG_SCALES_REALTIME,
    OFFLINE_HP_CUTOFF_HZ,
    OFFLINE_LOSS_CORE,
    OFFLINE_PICK_RESO_BP_ORDER,
    OFFLINE_SAMPLE_RATE,
    PHASE_LOSS_LPF_TAPS,
    PICK_RESO_MODAL_CENTERS_HZ_OFFLINE,
    PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
    REALTIME_LOSS_CORE,
    REALTIME_PICK_RESO_BP_ORDER,
    STFT_HF_ONSET_GAIN,
    STFT_HIGH_BOOST,
    ULTRALOW_MAIN_CUTOFF_HZ,
    ULTRALOW_MAIN_ORDER,
    ULTRALOW_MAIN_WEIGHT,
)
from .adversarial import (
    lowband_error_penalty,
)
from .blocks import (
    gate_gain_signal,
    gate_weight_dynamic,
)
from .dsp import (
    _causal_conv1d,
    _crop_time,
    _get_minphase_bp_kernel,
    _make_lpf_kernel,
    hp_cutoff_for_sr,
)

# ────────────────────────────── LOSSES ───────────────────────────────────────


def _loss_core_from_kind_or_sr(sample_rate):
    """Default loss core (segment sample range that enters the losses) for a model rate."""
    return OFFLINE_LOSS_CORE if int(sample_rate) == OFFLINE_SAMPLE_RATE else REALTIME_LOSS_CORE

@register_keras_serializable()
class CombinedLoss(Loss):
    """
    Full training loss. 'core' = (start, end) sample range of the segment that enters the loss
    (defaults to the kind's *_LOSS_CORE); pass core=(None, None) to use the whole segment.
    """
    def __init__(self, sample_rate, extra_small_cplx=False, core=None, phase_cutoff_hz=None,
                 name="combined_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.extra_small_cplx = bool(extra_small_cplx)
        core = _loss_core_from_kind_or_sr(sample_rate) if core is None else tuple(core)
        self.core = (None if core[0] is None else int(core[0]), None if core[1] is None else int(core[1]))
        self.phase_cutoff_hz = None if phase_cutoff_hz is None else float(phase_cutoff_hz)
    def call(self, y_true, y_pred):
        return combined_loss(y_true, y_pred, self.sample_rate,
                             extra_small_cplx=self.extra_small_cplx, core=self.core,
                             phase_cutoff_hz=self.phase_cutoff_hz)
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "extra_small_cplx": self.extra_small_cplx,
            "core": list(self.core),
            "phase_cutoff_hz": self.phase_cutoff_hz
        })
        return config

@register_keras_serializable()
class FinetuneCombinedLossLite(tf.keras.losses.Loss):
    """
    Lightweight loss for strict finetune:
      L = l1_w * L1 + hf_w * HF(preemph) + stft_w * 1-scale MR-STFT + env_w * RMS-env
          + ulb_w * L1( LPF_{<ulb_cut}(y_pred - y_true) )

    Single STFT scale by default to keep it very cheap.
    """
    def __init__(self,
                 sample_rate,
                 stft_scales=((1024, 256, 1024),),  # single scale by default
                 l1_w=1.0, hf_w=2.0, stft_w=1.0, env_w=0.0,
                 onset_gain=HF_ONSET_GAIN,
                 ulb_w=0.0,
                 ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                 ulb_order=None,
                 core=None,
                 name="finetune_combined_loss_lite",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = int(sample_rate)
        core = _loss_core_from_kind_or_sr(sample_rate) if core is None else tuple(core)
        self.core = (None if core[0] is None else int(core[0]), None if core[1] is None else int(core[1]))
        self.stft_scales = tuple((int(n), int(h), int(w)) for (n, h, w) in stft_scales)
        self.l1_w  = float(l1_w)
        self.hf_w  = float(hf_w)
        self.stft_w= float(stft_w)
        self.env_w = float(env_w)
        self.onset_gain = float(onset_gain)

        # ULB term
        self.ulb_w     = float(ulb_w)
        self.ulb_cut_hz= float(ulb_cut_hz)
        if ulb_order is None:
            self.ulb_order = int(ULTRALOW_MAIN_ORDER)
        else:
            self.ulb_order = int(ulb_order)

        # Precompute Hann windows (shape (1,1,win)) as constants
        self._wins = []
        for _, _, win in self.stft_scales:
            w = tf.signal.hann_window(win, periodic=True)
            w = tf.reshape(w, (1, 1, win))
            self._wins.append(tf.constant(w, dtype=tf.float32))

        # LPF kernel for ultra-lowband error (as a constant)
        self._ulb_kernel = _make_lpf_kernel(self.sample_rate,
                                            cutoff_hz=self.ulb_cut_hz,
                                            order=self.ulb_order,
                                            window="hamming")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "sample_rate": self.sample_rate,
            "stft_scales": self.stft_scales,
            "l1_w": self.l1_w,
            "hf_w": self.hf_w,
            "stft_w": self.stft_w,
            "env_w": self.env_w,
            "onset_gain": self.onset_gain,
            "ulb_w": self.ulb_w,
            "ulb_cut_hz": self.ulb_cut_hz,
            "ulb_order": self.ulb_order,
            "core": list(self.core)
        })
        return cfg

    def _mrstft(self, y_true, y_pred, gate_env):
        t = tf.squeeze(y_true, axis=-1)
        p = tf.squeeze(y_pred, axis=-1)

        losses = []
        for (n_fft, hop, win), w in zip(self.stft_scales, self._wins):
            ft = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)  # (B,frames,win)
            fp = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
            ftw = ft * w
            fpw = fp * w
            st_t = tf.signal.rfft(ftw, fft_length=[n_fft])
            st_p = tf.signal.rfft(fpw, fft_length=[n_fft])
            mag_t = tf.abs(st_t)
            mag_p = tf.abs(st_p)
            diff = tf.abs(mag_t - mag_p)  # (B,frames,F)

            if gate_env is not None:
                ge = tf.squeeze(gate_env, axis=-1)
                gef = tf.signal.frame(ge, frame_length=win, frame_step=hop, pad_end=True)
                ge_w = tf.reduce_mean(gef, axis=-1, keepdims=True)
                diff = diff * (1.0 + self.onset_gain * ge_w)

            losses.append(tf.reduce_mean(diff, axis=[1, 2]))  # (B,)

        loss = tf.add_n(losses) / float(len(losses))
        return loss[:, None, None]

    def _ulb(self, y_true, y_pred):
        if self.ulb_w == 0.0:
            # Shape-compatible zero so downstream weighting is clean
            return tf.zeros_like(y_true)
        err = y_pred - y_true
        k = self._ulb_kernel  # (L,1,1), already reversed for conv1d
        pad = tf.shape(k)[0] - 1
        err_pad = tf.pad(err, [[0,0],[pad,0],[0,0]])
        lp_err = tf.nn.conv1d(err_pad, filters=k, stride=1, padding='VALID')
        return tf.abs(lp_err)

    def call(self, y_true, y_pred):
        cs, ce = self.core
        gate_env = gate_weight_dynamic(y_true, self.sample_rate)  # (B,T,1) in [0,1], warmed up on the full segment
        def core_mean(m):
            return tf.reduce_mean(_crop_time(m, cs, ce), axis=1, keepdims=True)  # (B,1,1)
        td = core_mean(waveform_l1_loss(y_true, y_pred) * (0.2 + 0.8 * gate_env))
        hf = core_mean(hf_loss(y_true, y_pred) * (1.0 + self.onset_gain * gate_env))
        ulb = core_mean(self._ulb(y_true, y_pred))
        yt_c, yp_c, g_c = _crop_time(y_true, cs, ce), _crop_time(y_pred, cs, ce), _crop_time(gate_env, cs, ce)
        spec = self._mrstft(yt_c, yp_c, g_c) if self.stft_scales else 0.0
        env = rms_env_loss(yt_c, yp_c) if self.env_w != 0.0 else 0.0

        return (self.l1_w  * td
              + self.hf_w  * hf
              + self.stft_w* spec
              + self.env_w * env
              + self.ulb_w * ulb)

@register_keras_serializable()
class LogMelLoss(Loss):
    def __init__(self, sample_rate, name="log_mel_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
    def call(self, y_true, y_pred):
        return log_mel_loss(y_true, y_pred, self.sample_rate)
    def get_config(self):
        config = super().get_config()
        config.update({"sample_rate": self.sample_rate})
        return config

@register_keras_serializable()
def waveform_l1_loss(y_true, y_pred):
    return tf.abs(y_true - y_pred)

@register_keras_serializable()
def band_limited_l1_loss_weighted(y_true, y_pred, sample_rate,
                                  low_hz, high_hz,
                                  order_off=OFFLINE_PICK_RESO_BP_ORDER, order_rt=REALTIME_PICK_RESO_BP_ORDER,
                                  onset_boost=4.0,
                                  thr_db=-60.0,
                                  ratio=6.0,
                                  attack_ms=0.3,
                                  hold_ms=20.0,
                                  release_ms=200.0):
    """
    L1 in 0.7–2 kHz, weighted by an envelope computed from the band-passed TARGET,
    so training emphasizes true resonance events even if the piezo has none.
    Returns (B,T,1).
    """
    order = order_off if int(sample_rate) == OFFLINE_SAMPLE_RATE else order_rt
    k = _get_minphase_bp_kernel(sample_rate, low_hz, high_hz, order)

    t_bp = _causal_conv1d(y_true, k)
    p_bp = _causal_conv1d(y_pred, k)

    # Build a resonance envelope from the TARGET band, not from the piezo HF gate.
    g = gate_gain_signal(
        t_bp,
        sample_rate=sample_rate,
        threshold_db=thr_db,
        ratio=ratio,
        attack_ms=attack_ms,
        hold_ms=hold_ms,
        release_ms=release_ms,
        cutoff_hz=max(200.0, 0.5 * low_hz),  # HP sidechain near band’s lower edge
        hp_order=129,
        env_ms=1.0,
        depth_smooth_ms=20.0,
        env_lmax=512,
        depth_lmax=512
    )

    return tf.abs(t_bp - p_bp) * (1.0 + onset_boost * g)

@register_keras_serializable()
def resonance_band_loss(y_true, y_pred, sample_rate):
    if int(sample_rate) == OFFLINE_SAMPLE_RATE:
        lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE)
        hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE)
        order = OFFLINE_PICK_RESO_BP_ORDER
    else:
        lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
        hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
        order = REALTIME_PICK_RESO_BP_ORDER
    return band_limited_l1_loss_weighted(
        y_true, y_pred, sample_rate,
        low_hz=lo, high_hz=hi,
        order_off=order, order_rt=order,
        onset_boost=5.0,  # you can lift this to emphasize pick onsets more
        thr_db=-60.0, ratio=6.0, attack_ms=0.3, hold_ms=20.0, release_ms=200.0
    )

@register_keras_serializable()
def hf_loss(y_true, y_pred):
    α = HP_PRE_EMPHASIS_HF
    true_hp = y_true[:,1:,:] - α*y_true[:,:-1,:]
    pred_hp = y_pred[:,1:,:] - α*y_pred[:,:-1,:]
    true_hp = tf.pad(true_hp, [[0,0],[1,0],[0,0]])
    pred_hp = tf.pad(pred_hp, [[0,0],[1,0],[0,0]])
    return tf.abs(true_hp - pred_hp)

def _build_freq_weight(n_fft, sample_rate, split_hz, high_boost):
    freqs = np.linspace(0, sample_rate/2.0, n_fft//2 + 1, dtype=np.float32)
    w = np.ones_like(freqs, dtype=np.float32)
    w[freqs >= split_hz] = high_boost
    return tf.constant(w[None, None, :], dtype=tf.float32)

@register_keras_serializable()
def multi_scale_stft_loss(y_true, y_pred):
    scales = [
        (128,   32,  128),
        (256,   64,  256),
        (512,  128,  512),
        (1024, 256, 1024),
        (2048, 512, 2048),
        (8192, 2048, 8192)
    ]
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    per_scale_losses = []
    for n_fft, hop, win in scales:
        frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
        frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
        window = tf.signal.hann_window(win, periodic=True)
        window = tf.reshape(window, (1,1,win))
        win_t = frames_t * window
        win_p = frames_p * window
        st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
        st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
        mag_t = tf.abs(st_t)
        mag_p = tf.abs(st_p)
        loss_sf = tf.reduce_mean(tf.abs(mag_t - mag_p), axis=[1,2])
        per_scale_losses.append(loss_sf)
    loss_stack = tf.stack(per_scale_losses, axis=0)
    loss_batch = tf.reduce_mean(loss_stack, axis=0)
    return loss_batch[:, None, None]

@register_keras_serializable()
def mrstft_mag_loss_weighted_onset(y_true, y_pred, sample_rate,
                                   split_hz=OFFLINE_HP_CUTOFF_HZ,
                                   high_boost=STFT_HIGH_BOOST,
                                   onset_env=None,
                                   onset_gain=STFT_HF_ONSET_GAIN):
    scales = [
        (128,   32,  128),
        (256,   64,  256),
        (512,  128,  512),
        (1024, 256, 1024),
        (2048, 512, 2048),
        (8192, 2048, 8192)
    ]
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    per_scale_losses = []
    for n_fft, hop, win in scales:
        frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
        frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
        window = tf.signal.hann_window(win, periodic=True)
        window = tf.reshape(window, (1,1,win))
        win_t = frames_t * window
        win_p = frames_p * window
        st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
        st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
        mag_t = tf.abs(st_t)
        mag_p = tf.abs(st_p)
        diff = tf.abs(mag_t - mag_p)  # (B, frames, F)
        wf = _build_freq_weight(n_fft, sample_rate, split_hz, high_boost)
        diff = diff * wf
        if onset_env is not None:
            env_f = tf.signal.frame(tf.squeeze(onset_env, axis=-1), frame_length=win, frame_step=hop, pad_end=True)
            env_w = tf.reduce_mean(env_f, axis=-1, keepdims=True)
            fw = 1.0 + onset_gain * env_w
            diff = diff * fw
        loss_sf = tf.reduce_mean(diff, axis=[1,2])
        per_scale_losses.append(loss_sf)
    loss_stack = tf.stack(per_scale_losses, axis=0)
    loss_batch = tf.reduce_mean(loss_stack, axis=0)
    return loss_batch[:, None, None]

@register_keras_serializable()
def complex_stft_loss_onset(y_true, y_pred, sample_rate, n_fft=256, hop=64, win=256, onset_env=None, onset_gain=STFT_HF_ONSET_GAIN,
                            max_hz=None):
    """Complex STFT L1 (real + imag). max_hz: only bins below it (phase-sensitive band, see OFFLINE_PHASE_LOSS_CUTOFF_HZ)."""
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    frames_t = tf.signal.frame(t, frame_length=win, frame_step=hop, pad_end=True)
    frames_p = tf.signal.frame(p, frame_length=win, frame_step=hop, pad_end=True)
    window = tf.signal.hann_window(win, periodic=True)
    window = tf.reshape(window, (1,1,win))
    win_t = frames_t * window
    win_p = frames_p * window
    st_t = tf.signal.rfft(win_t, fft_length=[n_fft])
    st_p = tf.signal.rfft(win_p, fft_length=[n_fft])
    diff = tf.abs(tf.math.real(st_t - st_p)) + tf.abs(tf.math.imag(st_t - st_p))
    if max_hz is not None:
        n_keep = max(1, min(n_fft // 2 + 1, int(float(max_hz) / (float(sample_rate) / n_fft)) + 1))
        diff = diff[..., :n_keep]
    if onset_env is not None:
        env_f = tf.signal.frame(tf.squeeze(onset_env, axis=-1), frame_length=win, frame_step=hop, pad_end=True)
        env_w = tf.reduce_mean(env_f, axis=-1, keepdims=True)
        fw = 1.0 + onset_gain * env_w
        diff = diff * fw
    loss = tf.reduce_mean(diff, axis=[1,2])
    return loss[:, None, None]

@register_keras_serializable()
def mrstft_logmag_loss(y_true, y_pred, sample_rate, scales, min_hz=0.0, onset_env=None,
                       onset_gain=STFT_HF_ONSET_GAIN, floor_db=LMAG_FLOOR_DB):
    """
    Multi-resolution log-magnitude STFT L1 over the bins >= min_hz, onset weighted. Short windows
    (1.3-11 ms) resolve the attack; the log compresses the dynamic range so the quiet HF of an attack
    counts as much as the loud fundamental. The floor (floor_db re the target's loudest bin, per
    example) keeps the loss from chasing hiss and silence. Phase-blind by construction.
    """
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    losses = []
    for n_fft, hop in scales:
        n_fft, hop = int(n_fft), int(hop)
        w = tf.reshape(tf.signal.hann_window(n_fft, periodic=True), (1, 1, n_fft))
        mt = tf.abs(tf.signal.rfft(tf.signal.frame(t, n_fft, hop, pad_end=True) * w, fft_length=[n_fft]))
        mp = tf.abs(tf.signal.rfft(tf.signal.frame(p, n_fft, hop, pad_end=True) * w, fft_length=[n_fft]))
        k0 = min(n_fft // 2, int(np.ceil(float(min_hz) / (float(sample_rate) / n_fft))))
        mt, mp = mt[..., k0:], mp[..., k0:]
        floor = tf.maximum(tf.reduce_max(mt, axis=[1, 2], keepdims=True) * (10.0 ** (float(floor_db) / 20.0)), 1e-7)
        d = tf.abs(tf.math.log(mt + floor) - tf.math.log(mp + floor))
        if onset_env is not None:
            ef = tf.signal.frame(tf.squeeze(onset_env, axis=-1), n_fft, hop, pad_end=True)
            d = d * (1.0 + onset_gain * tf.reduce_mean(ef, axis=-1, keepdims=True))
        losses.append(tf.reduce_mean(d, axis=[1, 2]))
    return (tf.add_n(losses) / float(len(losses)))[:, None, None]

_phase_lpf_cache = {}

def _phase_loss_lpf_kernel(sample_rate, cutoff_hz):
    """Linear-phase (symmetric) low-pass for the phase-sensitive loss band; used zero-phase."""
    key = (int(sample_rate), float(cutoff_hz))
    k = _phase_lpf_cache.get(key)
    if k is None:
        taps = PHASE_LOSS_LPF_TAPS if int(sample_rate) >= OFFLINE_SAMPLE_RATE else (PHASE_LOSS_LPF_TAPS // 2) | 1
        h = firwin(int(taps), float(cutoff_hz) / (float(sample_rate) / 2.0), window="hamming")
        k = (h / np.sum(h)).astype(np.float32).reshape(-1, 1, 1)
        _phase_lpf_cache[key] = k
    return k

def _zero_phase_lpf(x, kernel_np):
    """'SAME' symmetric FIR (non-causal, loss-side only). Edge effects stay in the segment's context."""
    k = tf.constant(kernel_np, dtype=x.dtype)
    h = (int(kernel_np.shape[0]) - 1) // 2
    return tf.nn.conv1d(tf.pad(x, [[0, 0], [h, h], [0, 0]]), filters=k, stride=1, padding='VALID')

@register_keras_serializable()
def second_diff_l1_onset(y_true, y_pred, onset_env=None):
    def d2(x):
        return x[:,2:,:] - 2.0*x[:,1:-1,:] + x[:,:-2,:]
    t2 = d2(y_true)
    p2 = d2(y_pred)
    sd = tf.abs(t2 - p2)
    sd = tf.pad(sd, [[0,0],[1,1],[0,0]])
    if onset_env is not None:
        sd = sd * (1.0 + HF_ONSET_GAIN * onset_env)
    return sd

@register_keras_serializable()
def log_mel_loss(y_true, y_pred, sample_rate):
    n_fft   = 1024
    hop     = 256
    win     = 1024
    n_mels  = 64
    fmin    = 20.0
    fmax    = float(sample_rate//2)
    t   = tf.squeeze(y_true, axis=-1)
    p   = tf.squeeze(y_pred, axis=-1)
    st_t = tf.signal.stft(t,   frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
    st_p = tf.signal.stft(p,   frame_length=win, frame_step=hop, fft_length=n_fft, pad_end=True)
    mag_t = tf.abs(st_t)
    mag_p = tf.abs(st_p)
    fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft//2+1,
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax
    )
    mel_t = tf.tensordot(mag_t, fb, axes=[[2],[0]])
    mel_p = tf.tensordot(mag_p, fb, axes=[[2],[0]])
    log_t = tf.math.log(mel_t + 1e-6)
    log_p = tf.math.log(mel_p + 1e-6)
    mel_diff = tf.reduce_mean(tf.abs(log_t - log_p), axis=[1,2])
    return mel_diff[:,None,None]

@register_keras_serializable()
def rms_env_loss(y_true, y_pred):
    env_win  = 1024
    env_hop  = 512
    t = tf.squeeze(y_true, axis=-1)
    p = tf.squeeze(y_pred, axis=-1)
    ft   = tf.signal.frame(t, frame_length=env_win, frame_step=env_hop, pad_end=True)
    fp   = tf.signal.frame(p, frame_length=env_win, frame_step=env_hop, pad_end=True)
    rms_t = tf.sqrt(tf.reduce_mean(ft**2, axis=2) + 1e-12)
    rms_p = tf.sqrt(tf.reduce_mean(fp**2, axis=2) + 1e-12)
    env_diff = tf.reduce_mean(tf.abs(rms_t - rms_p), axis=1)
    return env_diff[:,None,None]

def _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=False, core=(None, None),
                         phase_cutoff_hz=None):
    """
    All CombinedLoss terms as (B,1,1) tensors. Per-sample terms are computed on the FULL segment
    (their internal filters / onset gate are then warmed up by the context) and averaged over the
    loss core only; spectral terms are computed on the cropped core. Shared by combined_loss and
    the loss-weight calibration so both see exactly the same quantities.

    phase_cutoff_hz: if set, the phase-sensitive terms (l1, hf, d2, cplx) only see the band below it
    (zero-phase low-pass of both signals / complex bins below it) and 'lmag' supervises the band
    above it on log magnitude. None = legacy full-band behaviour ('lmag' = 0).
    """
    cs, ce = core
    gate_env = gate_weight_dynamic(y_true, sample_rate)

    def core_mean(m):
        return tf.reduce_mean(_crop_time(m, cs, ce), axis=1, keepdims=True)  # (B,1,1)

    if phase_cutoff_hz:
        k_lp = _phase_loss_lpf_kernel(sample_rate, phase_cutoff_hz)
        yt_ph, yp_ph = _zero_phase_lpf(y_true, k_lp), _zero_phase_lpf(y_pred, k_lp)
    else:
        yt_ph, yp_ph = y_true, y_pred

    td = core_mean(waveform_l1_loss(yt_ph, yp_ph))
    hf = core_mean(hf_loss(yt_ph, yp_ph) * (1.0 + HF_ONSET_GAIN * gate_env))
    d2 = core_mean(second_diff_l1_onset(yt_ph, yp_ph, onset_env=gate_env))
    bl = core_mean(resonance_band_loss(y_true, y_pred, sample_rate))
    ulb = core_mean(lowband_error_penalty(y_true, y_pred, sample_rate,
                                          cutoff_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                          order=ULTRALOW_MAIN_ORDER))

    yt_c = _crop_time(y_true, cs, ce)
    yp_c = _crop_time(y_pred, cs, ce)
    g_c  = _crop_time(gate_env, cs, ce)
    spec = mrstft_mag_loss_weighted_onset(yt_c, yp_c, sample_rate,
                                          split_hz=hp_cutoff_for_sr(sample_rate),
                                          high_boost=STFT_HIGH_BOOST,
                                          onset_env=g_c, onset_gain=STFT_HF_ONSET_GAIN)
    mel = log_mel_loss(yt_c, yp_c, sample_rate)
    env = rms_env_loss(yt_c, yp_c)
    cplx = complex_stft_loss_onset(yt_c, yp_c, sample_rate, n_fft=256, hop=64, win=256, onset_env=g_c,
                                   max_hz=phase_cutoff_hz)
    if extra_small_cplx:
        cplx_64 = complex_stft_loss_onset(yt_c, yp_c, sample_rate, n_fft=64, hop=16, win=64, onset_env=g_c,
                                          max_hz=phase_cutoff_hz)
        cplx = 0.5 * (cplx + cplx_64)
    if phase_cutoff_hz:
        scales = LMAG_SCALES_OFFLINE if int(sample_rate) >= OFFLINE_SAMPLE_RATE else LMAG_SCALES_REALTIME
        lmag = mrstft_logmag_loss(yt_c, yp_c, sample_rate, scales, min_hz=phase_cutoff_hz, onset_env=g_c)
    else:
        lmag = tf.zeros_like(td)

    return {'l1': td, 'hf': hf, 'stft': spec, 'mel': mel, 'env': env,
            'cplx': cplx, 'd2': d2, 'resbp_bl': bl, 'ulb': ulb, 'lmag': lmag}

@register_keras_serializable()
def combined_loss(y_true, y_pred, sample_rate, extra_small_cplx=False, core=(None, None), phase_cutoff_hz=None):
    t = _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=extra_small_cplx, core=core,
                             phase_cutoff_hz=phase_cutoff_hz)
    return (config.L1_WEIGHT       * t['l1']
          + config.LMAG_WEIGHT     * t['lmag']
          + config.HF_WEIGHT       * t['hf']
          + config.STFT_WEIGHT     * t['stft']
          + config.MEL_WEIGHT      * t['mel']
          + config.RMS_ENV_WEIGHT  * t['env']
          + COMPLEX_STFT_WEIGHT * t['cplx']
          + D2_WEIGHT       * t['d2']
          + config.RESBP_BAND_LOSS_WEIGHT * t['resbp_bl']
          + ULTRALOW_MAIN_WEIGHT   * t['ulb'])

# Replace mean_hf_metric with this version
@register_keras_serializable()
def mean_hf_metric(y_true, y_pred):
    vals = hf_loss(y_true, y_pred)  # (B, T, 1)
    # Return per-sample metric with shape (B, 1, 1), compatible with (B, T, 1) sample weights
    return tf.reduce_mean(vals, axis=[1, 2], keepdims=True)
