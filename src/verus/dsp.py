import math
import numpy as np
import scipy.signal as spsig
import tensorflow as tf
from scipy.signal import firwin
from scipy.signal import minimum_phase
from scipy.signal import resample_poly
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]

from .config import (
    HP_PRE_EMPHASIS_HF,
    OFFLINE_HP_CUTOFF_HZ,
    OFFLINE_SAMPLE_RATE,
    REALTIME_HP_CUTOFF_HZ,
    REALTIME_IR_CUTOFF,
    REALTIME_IR_FIR_SIZE,
    REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
    REALTIME_SAMPLE_RATE,
)

# Set (only) by legacy_minphase_filters() while loading .keras files saved before the Sept-2026
# min-phase fix: their fixed-FIR weights have the old half lengths, so the layers must rebuild
# their kernels the old way for the shapes to match.
_LEGACY_MINPHASE_HALF = False

class legacy_minphase_filters:
    """Context manager: design min-phase FIRs the pre-Sept-2026 way (half taps, sqrt magnitude)."""
    def __enter__(self):
        global _LEGACY_MINPHASE_HALF
        self._prev = _LEGACY_MINPHASE_HALF
        _LEGACY_MINPHASE_HALF = True
        return self
    def __exit__(self, *exc):
        global _LEGACY_MINPHASE_HALF
        _LEGACY_MINPHASE_HALF = self._prev
        return False

def _to_minimum_phase(h_linear):
    """
    Convert a linear-phase FIR prototype to a minimum-phase FIR with the SAME magnitude response
    and (about) the same number of taps.

    scipy's minimum_phase(method='homomorphic') defaults to half=True, which returns HALF the
    taps AND the SQUARE ROOT of the magnitude response (i.e. half the stopband attenuation in dB).
    Every min-phase filter in this file used to go through that path, so a "60 dB" design shipped
    as a ~30 dB filter. Prefer half=False (scipy >= 1.14); on older scipy fall back to the
    autocorrelation trick (|H|^2 prototype -> sqrt -> |H|), which is equivalent.
    """
    h = np.asarray(h_linear, dtype=np.float64)
    if _LEGACY_MINPHASE_HALF:
        return np.asarray(minimum_phase(h, method='homomorphic'), dtype=np.float32)
    try:
        hm = minimum_phase(h, method='homomorphic', half=False)
    except TypeError:  # scipy < 1.14: no 'half' kwarg
        hm = minimum_phase(np.convolve(h, h[::-1]), method='homomorphic')
    return np.asarray(hm, dtype=np.float32)

def hp_cutoff_for_sr(sr):
    sr = int(sr)
    if sr == OFFLINE_SAMPLE_RATE:
        return OFFLINE_HP_CUTOFF_HZ
    elif sr == REALTIME_SAMPLE_RATE:
        return REALTIME_HP_CUTOFF_HZ

    raise ValueError(f"Unsupported sample rate: {sr}")

def resample_to_model_sr(audio, audio_sample_rate: int, model_sample_rate: int):
    g = math.gcd(audio_sample_rate, model_sample_rate)
    up   = model_sample_rate // g
    down = audio_sample_rate // g
    # Ensure float32 output (SciPy may return float64)
    y = resample_poly(audio.astype(np.float32, copy=False), up, down).astype(np.float32, copy=False)
    return y


def _len_lr_from_full(L_full, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR):
    return (int(L_full) - 1) // int(factor) + 1

_minphase_decimator_len_cache = {}

def _minphase_decimator_len(num_taps, factor):
    """
    Actual tap count of the min-phase decimator kernel built by
    StreamingRealtimeIrDownsampleFIR / PickResoDownsampleFIR (scipy's minimum_phase roughly
    halves the prototype length). Probing the real design keeps state declarations exact
    across scipy versions.
    """
    key = (int(num_taps), int(factor), bool(_LEGACY_MINPHASE_HALF))
    L = _minphase_decimator_len_cache.get(key)
    if L is None:
        cutoff = 0.45 / float(factor)
        taps = firwin(int(num_taps), cutoff=cutoff, window=('kaiser', 8.0), pass_zero=True).astype(np.float32)
        L = int(len(_to_minimum_phase(taps)))
        _minphase_decimator_len_cache[key] = L
    return L

def _decimate_modal_kernels(kernel_full_rev, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR):
    """
    Given trained modal kernels (reversed for conv1d) of shape (L_full,1,M),
    build low-rate reversed kernels by taking every 'factor'-th tap along time.
    Returns shape (L_low,1,M).

    Important: we renormalize each mode to preserve its original L2 norm so the
    post-mix level doesn't collapse after decimation.
    """
    K = np.asarray(kernel_full_rev, dtype=np.float32)
    L_full = K.shape[0]
    M = K.shape[2]
    F = int(factor)
    L_low = (int(L_full) - 1) // F + 1
    idx = np.arange(L_low, dtype=np.int32) * F
    K_low = K[idx, :, :].copy()  # (L_low, 1, M)

    # Renormalize each mode to the original L2 norm
    eps = 1e-12
    for m in range(M):
        orig_l2 = float(np.sqrt(np.sum(K[:, 0, m]**2, dtype=np.float64)))
        low_l2  = float(np.sqrt(np.sum(K_low[:, 0, m]**2, dtype=np.float64)))
        if low_l2 > eps and orig_l2 > eps:
            K_low[:, 0, m] *= (orig_l2 / low_l2)

    return K_low.astype(np.float32)
    
    
# ─────────────────────────────────────────────────────────────────────────────
# NEW helper ops/layers for streaming, Keras-serializable, TFLite-friendly
# ─────────────────────────────────────────────────────────────────────────────

@register_keras_serializable()
def _take_last(x, n):
    return x[:, -int(n):, :]

def _crop_time(x, core_start, core_end):
    """
    x[:, core_start:core_end, :] with bounds clamped to the (possibly dynamic) time length, so the
    same layer config is safe on shorter inputs (never yields an empty slice on non-empty input).
    (None, None) → x unchanged.
    """
    if core_start is None and core_end is None:
        return x
    T = tf.shape(x)[1]
    ce = T if core_end is None else tf.minimum(int(core_end), T)
    cs = 0 if core_start is None else tf.minimum(int(core_start), tf.maximum(ce - 1, 0))
    return x[:, cs:ce, :]

def _interp_window(window):
    """Normalise a (possibly Keras-deserialised TrackedList) window spec to something firwin takes."""
    win = window
    try:
        if not isinstance(win, (str, tuple)):
            win = tuple(win)
    except TypeError:
        win = 'hann'
    return win

def _interp_lowpass_kernel(num_taps, cutoff, window):
    """
    Anti-imaging low-pass for FIRInterpByN / StreamingFIRInterpByN, reversed for conv1d, (L,1,1).
    Min-phase like every other FIR in the graph (same magnitude and length as the linear-phase
    design, but ~no group delay): the linear-phase interpolators delayed the whole IR branch by
    (257-1)/2 = 128 samples (2.7 ms) and the pick-resonance branch by 0.7 ms relative to the main
    path, and a causal model cannot learn to undo a delay. Under legacy_minphase_filters() the old
    linear-phase design is kept (only its shape matters there: the trained taps are loaded).
    """
    try:
        taps = firwin(int(num_taps), cutoff=float(cutoff), window=_interp_window(window), pass_zero=True)
    except Exception:
        taps = firwin(int(num_taps), cutoff=float(cutoff), window=('kaiser', 9.0), pass_zero=True)
    taps = np.asarray(taps, dtype=np.float64)
    if not _LEGACY_MINPHASE_HALF:
        taps = np.asarray(_to_minimum_phase(taps), dtype=np.float64)
    taps = taps / max(float(np.sum(taps)), 1e-12)
    return taps[::-1].reshape(-1, 1, 1).astype(np.float32)

def _interp_polyphase_len(num_taps, factor):
    """Taps per polyphase branch: ceil(L / F). The streaming state needs this minus one low-rate samples."""
    return (int(num_taps) + int(factor) - 1) // int(factor)

def _interp_polyphase_filters(kernel_rev, factor):
    """
    Polyphase form of zero-stuff-by-F + FIR h:  y[i*F + r] = sum_j h[r + j*F] * x[i - j].
    Returns conv1d filters (J, 1, F) (cross-correlation layout) built from the reversed (L,1,1)
    kernel weight, so the saved weight format is unchanged. Computing this way skips the F-1 of
    every F multiplications that hit an inserted zero (8x fewer MACs); the ops are constant-folded
    by the TFLite converter.
    """
    F = int(factor)
    L = int(kernel_rev.shape[0])
    J = (L + F - 1) // F
    h = tf.reverse(kernel_rev[:, 0, 0], axis=[0])                 # forward taps
    h = tf.pad(h, [[0, J * F - L]])
    P = tf.reshape(h, [J, F])                                      # P[j, r] = h[r + j*F]
    return tf.reshape(tf.reverse(P, axis=[0]), [J, 1, F])

def _interp_polyphase_apply(x_hist, filters, factor):
    """x_hist: (B, (J-1)+Tl, 1) low-rate samples incl. J-1 of history → (B, Tl*F, 1)."""
    y = tf.nn.conv1d(x_hist, filters=filters, stride=1, padding='VALID')   # (B, Tl, F)
    b, tl = tf.shape(y)[0], tf.shape(y)[1]
    return tf.reshape(y, [b, tl * int(factor), 1])

@register_keras_serializable()
def pre_emphasis(x, alpha=HP_PRE_EMPHASIS_HF):
    """
    Simple 1st-order high-pass (pre-emphasis): y[n] = x[n] - alpha * x[n-1]
    x shape: (B, T, C)
    """
    x_shift = tf.pad(x, [[0,0],[1,0],[0,0]])[:, :-1, :]
    return x - alpha * x_shift

def make_sinc_filter():
    """
    Return a 1-D linear-phase prototype (no min-phase, no reshape).
    The downstream code will do minimum_phase + reverse + reshape.
    """
    N = int(REALTIME_IR_FIR_SIZE)
    if N < 2:
        raise ValueError("REALTIME_IR_FIR_SIZE must be >= 2")

    # Normalized cutoff for decimate-by-factor LPF: REALTIME_IR_CUTOFF is already 1/factor
    t = np.arange(N, dtype=np.float32) - (N - 1) / 2.0
    h = np.sinc(2.0 * REALTIME_IR_CUTOFF * t).astype(np.float32)

    # Sharper window than Hamming (optional)
    w = np.kaiser(N, beta=9.0).astype(np.float32)
    h *= w

    # Normalize sum to 1 to preserve DC
    s = np.sum(h, dtype=np.float64)
    h = (h / s).astype(np.float32)
    return h  # 1-D float32

_ir_fir     = make_sinc_filter()
h_lin = _ir_fir
h_min = _to_minimum_phase(h_lin)
h_min /= np.sum(h_min)
h_min = h_min.astype(np.float32)

# For tf.nn.conv1d (cross-correlation), use the time-reversed kernel to implement convolution
_ir_fir_kernel = h_min[::-1].reshape(-1, 1, 1).astype(np.float32)

# Legacy (pre-Sept-2026) variant: half taps, sqrt magnitude — only for loading old .keras files
with legacy_minphase_filters():
    _h_min_legacy = _to_minimum_phase(h_lin)
_h_min_legacy /= np.sum(_h_min_legacy)
_ir_fir_kernel_legacy = _h_min_legacy.astype(np.float32)[::-1].reshape(-1, 1, 1)

def _ir_prototype_kernel():
    return _ir_fir_kernel_legacy if _LEGACY_MINPHASE_HALF else _ir_fir_kernel

_ir_fir_tf  = tf.constant(_ir_fir_kernel, dtype=tf.float32)  # downsampling (analysis) FIR

def _make_lpf_kernel(sample_rate, cutoff_hz=2800.0, order=129, window="hamming"):
    taps = firwin(order, cutoff_hz / (sample_rate / 2.0), window=window, pass_zero=True).astype(np.float32)
    taps = taps / np.sum(taps + 1e-12)
    # reverse for tf.nn.conv1d cross-correlation
    return tf.constant(taps[::-1].reshape(-1, 1, 1), dtype=tf.float32)

@register_keras_serializable()
def _lp_rev_padded(sr, cut_hz, target_len, window='hamming'):
    """
    Build a linear-phase LPF (reversed taps for conv1d), exact length target_len.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    wn = float(cut_hz) / (float(sr) / 2.0)
    taps = firwin(nt, cutoff=wn, window=window, pass_zero=True).astype(np.float32)
    taps = taps / np.maximum(np.sum(taps), 1e-12)
    k_rev = taps[::-1].reshape(nt, 1, 1).astype(np.float32)
    return k_rev

# ─── Gate sidechain HPF (min-phase, causal) cache ────────────────────────────
_gate_hp_kernel_cache = {}

def _get_minphase_hp_kernel(sample_rate, cutoff_hz, order=129):
    key = (int(sample_rate), float(cutoff_hz), int(order), bool(_LEGACY_MINPHASE_HALF))
    k_np = _gate_hp_kernel_cache.get(key)
    if k_np is None:
        taps = firwin(
        order,
        cutoff=cutoff_hz / (sample_rate / 2.0),
        window="hamming",
        pass_zero=False
        ).astype(np.float32)
        taps_min = _to_minimum_phase(taps)
        k_np = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        _gate_hp_kernel_cache[key] = k_np
    # Always materialize a new constant in the current graph.
    return tf.constant(k_np, dtype=tf.float32)

# ─── Band-pass (700–2000 Hz) causal min-phase FIR ────────────────────────────
_bp_kernel_cache = {}
def _get_minphase_bp_kernel(sample_rate, f_lo, f_hi, order=257, window="hamming"):
    """
    Build a causal min-phase band-pass FIR kernel for tf.nn.conv1d (cross-correlation),
    i.e. time-reversed taps. We do not normalize DC since it's a band-pass.
    """
    key = (int(sample_rate), float(f_lo), float(f_hi), int(order), str(window), bool(_LEGACY_MINPHASE_HALF))
    k_np = _bp_kernel_cache.get(key)
    if k_np is None:
        nyq = float(sample_rate) / 2.0
        lo  = float(f_lo) / nyq
        hi  = float(f_hi) / nyq
        taps = firwin(order, [lo, hi], pass_zero=False, window=window).astype(np.float32)
        # Convert to minimum-phase (same magnitude response as the linear-phase design)
        taps_min = _to_minimum_phase(taps)
        # Reverse for conv1d (cross-correlation) to emulate convolution
        k_np = taps_min[::-1].reshape(-1, 1, 1).astype(np.float32)
        _bp_kernel_cache[key] = k_np
    return tf.constant(k_np, dtype=tf.float32)

def _causal_conv1d(x, kernel):
    pad = tf.shape(kernel)[0] - 1
    x_pad = tf.pad(x, [[0,0],[pad,0],[0,0]])
    return tf.nn.conv1d(x_pad, filters=kernel, stride=1, padding='VALID')

# ─── Gate smoothing kernels (causal FIR) ─────────────────────────────────────
_gate_smooth_cache = {}  # (tau_samples, scale, Lmax) -> kernel np.array
_gate_ones_cache   = {}  # (L,) -> kernel np.array

@register_keras_serializable()
def gate_onset(g):
    # g: (B,T,1) ∈ [0,1]; return rising edges as positive envelope
    g_prev = tf.pad(g, [[0,0],[1,0],[0,0]])[:, :-1, :]
    return tf.nn.relu(g - g_prev)

def _make_modal_kernels(centers_hz, decays_ms, length, sample_rate):
    L = int(length)
    t = np.arange(L, dtype=np.float32)
    kernels = []
    for f, d_ms in zip(centers_hz, decays_ms):
        tau_s = max(1.0, (float(d_ms) / 1000.0) * float(sample_rate))
        env = np.exp(-t / tau_s).astype(np.float32)
        s   = np.sin(2.0 * np.pi * float(f) * t / float(sample_rate)).astype(np.float32)
        h = env * s
        h /= np.maximum(np.sqrt(np.sum(h*h, dtype=np.float32)), 1e-6)
        kernels.append(h)
    K = np.stack(kernels, axis=1)      # (L, M)
    K = K[::-1, :]                     # reverse for conv1d (cross-correlation → convolution)
    return K.reshape(L, 1, -1).astype(np.float32)

def _exp_smoothing_kernel(samples, scale=4.0, Lmax=2048):
    """
    Causal exponential FIR approximation of an EMA with time-constant 'samples'.
    Kernel length L ≈ scale * samples, clipped to [8, Lmax].
    Sum is normalized to 1. Returned reversed (for conv1d cross-correlation).
    """
    s = max(1, int(samples))
    L = int(np.clip(np.ceil(scale * s), 8, Lmax))
    key = (s, float(scale), int(Lmax))
    k_np = _gate_smooth_cache.get(key)
    if k_np is None:
        a = np.exp(-1.0 / float(s))
        taps = (1.0 - a) * (a ** np.arange(L, dtype=np.float32))
        taps = taps / np.sum(taps)  # normalize exactly
        k_np = taps[::-1].reshape(-1, 1, 1).astype(np.float32)  # reverse for tf.nn.conv1d
        _gate_smooth_cache[key] = k_np
    return tf.constant(k_np, dtype=tf.float32)

def _ones_kernel(length):
    """
    Causal moving-sum kernel of length 'length' (reversed for conv1d).
    """
    L = max(1, int(length))
    k_np = _gate_ones_cache.get(L)
    if k_np is None:
        taps = np.ones((L,), dtype=np.float32)
        k_np = taps[::-1].reshape(-1, 1, 1).astype(np.float32)
        _gate_ones_cache[L] = k_np
    return tf.constant(k_np, dtype=tf.float32)

def _db_to_lin(db):
    return float(10.0 ** (float(db) / 20.0))

def _corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.size, b.size)
    if n < 8: return 0.0
    a = a[:n]; b = b[:n]
    den = np.sqrt((np.sum(a*a)+1e-12) * (np.sum(b*b)+1e-12))
    if den <= 0: return 0.0
    return float(np.sum(a*b) / den)

@register_keras_serializable()
def _hp_minphase_rev_padded(sr, cut_hz, target_len):
    """
    Build conv1d-ready reversed min-phase HPF, exact length target_len.
    If minimum_phase returns fewer taps, left-pad zeros so reversed taps are right-aligned.
    Returns (target_len,1,1) float32.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    # HPF firwin prefers odd numtaps; keep design odd but do not change target_len
    design_nt = nt if (nt % 2) == 1 else (nt - 1 if nt > 3 else 3)
    wn = float(cut_hz) / (float(sr) / 2.0)
    taps = firwin(design_nt, cutoff=wn, pass_zero=False, window="hamming").astype(np.float32)
    taps_min = _to_minimum_phase(taps)
    k_rev_small = taps_min[::-1].reshape(-1)   # reversed for conv1d
    M = int(k_rev_small.shape[0])
    k_full = np.zeros((nt,), dtype=np.float32)
    if M >= nt:
        k_full[:] = k_rev_small[-nt:]
    else:
        pad = nt - M
        k_full[pad:] = k_rev_small
    return k_full.reshape(nt, 1, 1).astype(np.float32)

@register_keras_serializable()
def _bp_minphase_rev_padded(sr, low_hz, high_hz, target_len):
    """
    Build conv1d-ready reversed min-phase BP, exact length target_len.
    Left-pad zeros if minimum_phase returns fewer taps.
    Returns (target_len,1,1) float32.
    """
    nt = int(target_len)
    if nt < 3:
        nt = 3
    design_nt = nt if (nt % 2) == 1 else (nt - 1 if nt > 3 else 3)
    nyq = float(sr) / 2.0
    lo = float(low_hz) / nyq
    hi = float(high_hz) / nyq
    taps = firwin(design_nt, [lo, hi], pass_zero=False, window="hamming").astype(np.float32)
    taps_min = _to_minimum_phase(taps)
    k_rev_small = taps_min[::-1].reshape(-1)
    M = int(k_rev_small.shape[0])
    k_full = np.zeros((nt,), dtype=np.float32)
    if M >= nt:
        k_full[:] = k_rev_small[-nt:]
    else:
        pad = nt - M
        k_full[pad:] = k_rev_small
    return k_full.reshape(nt, 1, 1).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FAST PICKATTACK AUDITION: non-streaming realtime graph + in-place param tweak
# ─────────────────────────────────────────────────────────────────────────────

def _fit_rev_kernel_length(k_rev_np, target_len):
    """
    Ensure reversed FIR k_rev_np (L',1,1) matches target_len (L,1,1) by left-padding zeros
    or tail-cropping (keep most recent taps).
    """
    k = np.asarray(k_rev_np, dtype=np.float32).reshape(-1)
    Lwant = int(target_len)
    Lhave = int(k.shape[0])
    if Lhave == Lwant:
        return k.reshape(Lwant, 1, 1).astype(np.float32)
    out = np.zeros((Lwant,), dtype=np.float32)
    if Lhave > Lwant:
        out[:] = k[-Lwant:]
    else:
        out[Lwant - Lhave:] = k
    return out.reshape(Lwant, 1, 1).astype(np.float32)

def _exp_kernel_rev_ms(ms, sr, scale, Lwant):
    """
    Build reversed EMA kernel for ms, then pad/crop to Lwant.
    """
    s = max(1, int(round((float(ms)/1000.0) * float(sr))))
    k = _exp_smoothing_kernel(s, scale=scale, Lmax=max(Lwant, 8))
    k_np = (k.numpy() if tf.is_tensor(k) else np.asarray(k, np.float32)).reshape(-1)
    return _fit_rev_kernel_length(k_np, Lwant)

def _rms(x):
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0

def _bp(x, sr, lo, hi):
    hi = min(float(hi), 0.45 * sr)
    return spsig.sosfiltfilt(spsig.butter(4, [float(lo), hi], btype="band", fs=sr, output="sos"), np.asarray(x, np.float64))
