import math
import numpy as np
import scipy.signal as spsig

from .config import (
    KNOB_DEFAULT_HIGH_SHELF_GAIN_DB,
    KNOB_DEFAULT_HIGH_SHELF_HZ,
    KNOB_DEFAULT_HIGH_SHELF_SLOPE_DB_PER_OCT,
    KNOB_DEFAULT_LOW_CUT_HZ,
    KNOB_DEFAULT_LOW_CUT_SLOPE_DB_PER_OCT,
    KNOB_DEFAULT_LOW_SHELF_GAIN_DB,
    KNOB_DEFAULT_LOW_SHELF_HZ,
    KNOB_DEFAULT_LOW_SHELF_SLOPE_DB_PER_OCT,
    KNOB_DEFAULT_MID_GAIN_DB,
    KNOB_DEFAULT_MID_HZ,
    KNOB_DEFAULT_MID_Q,
)

# ───────── Master EQ (runtime IIR, NOT part of the model graph) ─────────
# The master EQ used to be four in-graph FIR banks (LC/LS/mid/HS blended over a knob grid).
# 33–193-tap FIRs cannot realise 20–300 Hz cuts (the bank filters were near-transparent) and the
# banks were ~35% of the realtime model's MACs, so the EQ now lives in the runtimes as RBJ
# biquads: apply_master_eq_iir() here (used by --process --eq ...) and the identical
# MasterEqIIR in runtime/src/fx/master_eq.hpp. Training and export are EQ-free. Defaults below are the
# neutral position (LC 20 Hz = bypass, all gains 0 dB) and are mirrored by the C++ Knobs.

def master_eq_defaults():
    return {
        "lc_hz": KNOB_DEFAULT_LOW_CUT_HZ, "lc_slope": KNOB_DEFAULT_LOW_CUT_SLOPE_DB_PER_OCT,
        "ls_gain_db": KNOB_DEFAULT_LOW_SHELF_GAIN_DB, "ls_hz": KNOB_DEFAULT_LOW_SHELF_HZ,
        "ls_slope": KNOB_DEFAULT_LOW_SHELF_SLOPE_DB_PER_OCT,
        "mid_gain_db": KNOB_DEFAULT_MID_GAIN_DB, "mid_hz": KNOB_DEFAULT_MID_HZ, "mid_q": KNOB_DEFAULT_MID_Q,
        "hs_gain_db": KNOB_DEFAULT_HIGH_SHELF_GAIN_DB, "hs_hz": KNOB_DEFAULT_HIGH_SHELF_HZ,
        "hs_slope": KNOB_DEFAULT_HIGH_SHELF_SLOPE_DB_PER_OCT,
    }

def _rbj_shelf_q_from_slope(gain_db, slope_dbpoct):
    # RBJ shelf slope parameter S in (0, 1]; 12 dB/oct ~ S=1 (steepest without overshoot)
    A = 10.0 ** (float(gain_db) / 40.0)
    S = min(1.0, max(0.25, float(slope_dbpoct) / 12.0))
    return 1.0 / math.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

def _rbj_biquad(kind, sr, f0, q, gain_db=0.0):
    """RBJ cookbook biquad -> normalized (b0,b1,b2,a1,a2). kind: hp1|hp2|ls|hs|peak"""
    sr = float(sr); f0 = min(max(float(f0), 1.0), 0.49 * sr)
    w0 = 2.0 * math.pi * f0 / sr
    if kind == "hp1":
        # first-order high-pass (6 dB/oct), bilinear
        k = math.tan(w0 / 2.0)
        n = 1.0 / (1.0 + k)
        return (n, -n, 0.0, (k - 1.0) * n, 0.0)
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * max(float(q), 1e-3))
    A = 10.0 ** (float(gain_db) / 40.0)
    if kind == "hp2":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "peak":
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    elif kind == "ls":
        sa = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cw + sa); b1 = 2 * A * ((A - 1) - (A + 1) * cw); b2 = A * ((A + 1) - (A - 1) * cw - sa)
        a0 = (A + 1) + (A - 1) * cw + sa;       a1 = -2 * ((A - 1) + (A + 1) * cw);    a2 = (A + 1) + (A - 1) * cw - sa
    elif kind == "hs":
        sa = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + sa); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - sa)
        a0 = (A + 1) - (A - 1) * cw + sa;       a1 = 2 * ((A - 1) - (A + 1) * cw);     a2 = (A + 1) - (A - 1) * cw - sa
    else:
        raise ValueError(kind)
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

# Butterworth Q values for cascaded 2nd-order sections (12/24/36 dB/oct)
_BUTTERWORTH_Q = {12.0: [0.70710678], 24.0: [0.54119610, 1.30656296], 36.0: [0.51763809, 0.70710678, 1.93185165]}

def master_eq_sos(sr, lc_hz=None, lc_slope=None, ls_gain_db=None, ls_hz=None, ls_slope=None,
                  mid_gain_db=None, mid_hz=None, mid_q=None, hs_gain_db=None, hs_hz=None, hs_slope=None):
    """
    Build the master EQ as a list of biquad sections (scipy 'sos' rows [b0,b1,b2,1,a1,a2]).
    Semantics mirror runtime/src/fx/master_eq.hpp MasterEqIIR exactly:
      - low cut: bypass at <= KNOB_DEFAULT_LOW_CUT_HZ; slope 6 = 1st-order, 12/24/36 = Butterworth cascade
      - low/high shelf and mid bell: RBJ; 0 dB gain = bypass
    Returns an empty list when everything is at the neutral position.
    """
    d = master_eq_defaults()
    v = lambda x, k: d[k] if x is None else float(x)
    lc_hz, lc_slope = v(lc_hz, "lc_hz"), v(lc_slope, "lc_slope")
    ls_gain_db, ls_hz, ls_slope = v(ls_gain_db, "ls_gain_db"), v(ls_hz, "ls_hz"), v(ls_slope, "ls_slope")
    mid_gain_db, mid_hz, mid_q = v(mid_gain_db, "mid_gain_db"), v(mid_hz, "mid_hz"), v(mid_q, "mid_q")
    hs_gain_db, hs_hz, hs_slope = v(hs_gain_db, "hs_gain_db"), v(hs_hz, "hs_hz"), v(hs_slope, "hs_slope")
    sos = []
    def add(c):
        b0, b1, b2, a1, a2 = c
        sos.append([b0, b1, b2, 1.0, a1, a2])
    if lc_hz > KNOB_DEFAULT_LOW_CUT_HZ + 1e-6:
        # Ascending candidates, first minimum wins on ties — exactly MasterEqIIR::nearestSlope (a set
        # iteration order made e.g. 9 dB/oct resolve differently from the pedal)
        slope = min((6.0, 12.0, 24.0, 36.0), key=lambda s: abs(s - lc_slope))
        if slope == 6.0:
            add(_rbj_biquad("hp1", sr, lc_hz, 0.0))
        else:
            for q in _BUTTERWORTH_Q[slope]:
                add(_rbj_biquad("hp2", sr, lc_hz, q))
    if abs(ls_gain_db) > 1e-6:
        add(_rbj_biquad("ls", sr, ls_hz, _rbj_shelf_q_from_slope(ls_gain_db, ls_slope), ls_gain_db))
    if abs(mid_gain_db) > 1e-6:
        add(_rbj_biquad("peak", sr, mid_hz, mid_q, mid_gain_db))
    if abs(hs_gain_db) > 1e-6:
        add(_rbj_biquad("hs", sr, hs_hz, _rbj_shelf_q_from_slope(hs_gain_db, hs_slope), hs_gain_db))
    return sos

def apply_master_eq_iir(x, sr, **knobs):
    """Apply master_eq_sos(sr, **knobs) to a mono float array (no-op at the neutral position)."""
    sos = master_eq_sos(sr, **knobs)
    if not sos:
        return np.asarray(x, dtype=np.float32)
    return spsig.sosfilt(np.asarray(sos, dtype=np.float64), np.asarray(x, dtype=np.float64)).astype(np.float32)

def parse_master_eq_arg(spec):
    """'--eq lc_hz=80,lc_slope=12,hs_gain_db=2' -> dict of master_eq_sos kwargs (validated)."""
    out = {}
    if not spec:
        return out
    valid = set(master_eq_defaults().keys())
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--eq entries must be key=value, got '{item}'")
        k, val = item.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise ValueError(f"--eq: unknown key '{k}'. Valid: {sorted(valid)}")
        out[k] = float(val)
    return out
