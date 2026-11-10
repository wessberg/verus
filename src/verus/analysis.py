import json
import numpy as np
import os
import scipy.signal as spsig
import soundfile as sf
import time

from .config import (
    ANALYSIS_EXCERPT_POSITIONS,
    ANALYSIS_OUT_DIR,
    ANALYSIS_PREROLL_SEC,
    ATTACK_BANDS,
    ATTACK_POST_MS,
    ATTACK_PRE_MS,
    ATTACK_TIMING_GRID_MS,
    OFFLINE_SAMPLE_RATE,
    PICKUP_MAGNETIC,
    PICKUP_PIEZO,
    REALTIME_SAMPLE_RATE,
)
from .data import (
    _read_pair_segment,
    load_training_data,
    pairs_by_group,
)
from .dsp import (
    _bp,
    _corr,
)
from .inference import (
    KerasStreamingRunner,
)
from .knobs import (
    prepare_demo_knob_feeds,
    prepare_process_knob_feeds,
)
from .modelio import (
    load_keras_model,
)
from .process import (
    _iter_offline_blocks,
    _make_process_style_arrays,
)
from .stems import (
    _select_stems_model_path,
)

def _crest(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0: return 0.0
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    peak = np.max(np.abs(x))
    return float(peak / (rms + 1e-12))

def _quantiles(x, qs=(0.5, 0.9, 0.99, 0.999)):
    x = np.abs(np.asarray(x, dtype=np.float32))
    return {f"q{int(q*1000):03d}": float(np.quantile(x, q)) for q in qs}

def _duty_cycle(m, thr=0.2):
    m = np.asarray(m, dtype=np.float32)
    if m.size == 0: return 0.0
    return float(np.mean((m > float(thr)).astype(np.float32)))

def _sum_rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32)**2) + 1e-12))

def _metrics_for_core(outs_dict, y_target_core=None):
    # outs must contain keys: y_pred, main_core, pickattack, pickreso, main_pick, main_finger
    y   = outs_dict.get("y_pred",    None)
    main= outs_dict.get("main_core", None)
    patt= outs_dict.get("pickattack",None)
    pres= outs_dict.get("pickreso",  None)
    mp  = outs_dict.get("main_pick", None)
    mf  = outs_dict.get("main_finger", None)
    msk = outs_dict.get("mask",      None)

    m = {}
    # RMS and crest
    for k, sig in [("full", y), ("main", main), ("pickattack", patt), ("pick_reso", pres),
                   ("pick", mp), ("fing", mf)]:
        if sig is not None:
            m[f"rms_{k}"]   = _sum_rms(sig)
            m[f"crest_{k}"] = _crest(sig)
            m.update({f"{k}_{q}":v for q,v in _quantiles(sig).items()})

    # Sum-check error
    if y is not None and main is not None:
        est = np.zeros_like(y)
        if patt is not None: est = est + patt
        if pres is not None: est = est + pres
        est = est + main
        m["rms_sum_err"] = _sum_rms(y - est)

    # Mask stats
    if msk is not None:
        m["mask_mean"]  = float(np.mean(msk))
        m["mask_duty20"] = _duty_cycle(msk, thr=0.2)
        m["mask_duty50"] = _duty_cycle(msk, thr=0.5)
        m.update({f"mask_{q}": v for q, v in _quantiles(msk).items()})

    # Pick/finger correlation
    if mp is not None and mf is not None:
        m["corr_pick_fing"] = _corr(mp, mf)

    # Optional: overshoot vs target (HF-ish proxy)
    if (y_target_core is not None) and (patt is not None):
        yt = np.asarray(y_target_core, dtype=np.float32)
        pa = np.asarray(patt,         dtype=np.float32)
        eps = 1e-6
        # Compare 99th percentile magnitudes normalized by RMS
        q_patt = np.quantile(np.abs(pa), 0.99)
        q_tgt  = np.quantile(np.abs(yt), 0.99)
        m["overshoot_q99_ratio"] = float(q_patt / (q_tgt + eps))

    return m

def detect_attacks(y, sr, min_gap_s=0.07):
    """Onsets (sample indices) of the MIC signal: spectral flux of the 3-12 kHz envelope, refined to
    the 20 %-of-peak crossing. Same detector as the Sept-2026 investigation."""
    e = np.abs(_bp(y, sr, 3000, 12000))
    e = np.maximum(spsig.sosfiltfilt(spsig.butter(2, 400, fs=sr, output="sos"), e), 0.0)
    hop = max(1, int(sr // 1000))
    flux = np.maximum(0, np.diff(np.log(e[::hop] + 1e-7)))
    thr = np.median(flux) + 6 * np.median(np.abs(flux - np.median(flux)))
    peaks, _ = spsig.find_peaks(flux, height=thr, distance=max(1, int(min_gap_s * sr / hop)))
    out, w0, w1 = [], int(0.005 * sr), int(0.010 * sr)
    for pk_i in peaks:
        c = pk_i * hop
        lo = max(0, c - w0)
        seg = e[lo: c + w1]
        if seg.size == 0:
            continue
        pk = int(np.argmax(seg))
        rel = np.nonzero(seg[:pk + 1] >= 0.2 * seg[pk])[0]
        out.append(lo + (int(rel[0]) if rel.size else pk))
    return np.asarray(out, dtype=int)

def _attack_windows(ons, sr, n):
    pre, post = int(ATTACK_PRE_MS * sr / 1000), int(ATTACK_POST_MS * sr / 1000)
    ons = ons[(ons - pre >= 0) & (ons + post <= n)]
    return ons, pre, post

def _band_attack_between(sig, ons, sr, lo, hi):
    h = _bp(sig, sr, lo, hi) ** 2
    ons, pre, post = _attack_windows(ons, sr, len(h))
    mask = np.zeros(len(h), bool)
    for o in ons:
        mask[o - pre:o + post] = True
    return float(h[mask].mean()) if mask.any() else 0.0, float(h[~mask].mean()) if (~mask).any() else 0.0

def _attack_coherence(a, b, ons, sr, bands):
    ons, pre, post = _attack_windows(ons, sr, min(len(a), len(b)))
    L = pre + post
    w = spsig.windows.tukey(L, 0.2)
    Sab = Saa = Sbb = 0.0
    for o in ons:
        A = np.fft.rfft(w * a[o - pre:o + post]); B = np.fft.rfft(w * b[o - pre:o + post])
        Sab = Sab + np.conj(A) * B; Saa = Saa + np.abs(A) ** 2; Sbb = Sbb + np.abs(B) ** 2
    f = np.fft.rfftfreq(L, 1.0 / sr)
    C = np.abs(Sab) ** 2 / (Saa * Sbb + 1e-30) if len(ons) else np.zeros_like(f)
    return {f"{lo}-{hi}": float(np.mean(C[(f >= lo) & (f < hi)])) for lo, hi in bands}

def _attack_timing(sig, ons, sr, ref_energy):
    h = _bp(sig, sr, 6000, 12000) ** 2
    ons = ons[(ons - int(0.002 * sr) >= 0) & (ons + int(0.060 * sr) <= len(h))]
    out = []
    for a, b in ATTACK_TIMING_GRID_MS:
        v = np.mean([h[o + int(a * sr / 1000): o + int(b * sr / 1000)].mean() for o in ons]) if len(ons) else 0.0
        out.append(float(10 * np.log10(v / ref_energy + 1e-30)))
    return out

def attack_metrics(x, y, y_model, sr, boundaries=()):
    """Attack-fidelity metrics of y_model vs the mic y (x = pickup input), all at sample rate sr.
    boundaries: sample indices where excerpts were concatenated (no onsets are taken near them)."""
    n = min(len(x), len(y), len(y_model))
    x, y, y_model = (np.asarray(v[:n], np.float64) for v in (x, y, y_model))
    ons = detect_attacks(y, sr)
    guard = int(0.070 * sr)
    for b in boundaries:
        ons = ons[np.abs(ons - int(b)) > guard]
    db = lambda a, b: float(10 * np.log10((a + 1e-30) / (b + 1e-30)))
    att, btw = {}, {}
    for lo, hi in ATTACK_BANDS:
        ta, tb = _band_attack_between(y, ons, sr, lo, hi)
        ma, mb = _band_attack_between(y_model, ons, sr, lo, hi)
        att[f"{lo}-{hi}"], btw[f"{lo}-{hi}"] = db(ma, ta), db(mb, tb)
    h = _bp(y, sr, 6000, 12000) ** 2
    ok = ons[(ons + int(0.001 * sr)) <= len(h)]
    ref = float(np.mean([h[o:o + int(0.001 * sr)].mean() for o in ok])) if len(ok) else 1.0
    hf = [k for k in att if int(k.split("-")[0]) >= 6000]
    coh_m = _attack_coherence(y_model, y, ons, sr, ATTACK_BANDS)
    return {
        "n_attacks": int(len(ons)),
        "attack_db": att, "between_db": btw,
        "coh_model": coh_m, "coh_piezo": _attack_coherence(x, y, ons, sr, ATTACK_BANDS),
        "timing_grid_ms": [list(g) for g in ATTACK_TIMING_GRID_MS],
        "timing_db": {"mic": _attack_timing(y, ons, sr, ref), "piezo": _attack_timing(x * np.sqrt(np.mean(y**2) / (np.mean(x**2) + 1e-30)), ons, sr, ref),
                      "model": _attack_timing(y_model, ons, sr, ref)},
        "summary": {"attack_hf_db": float(np.mean([att[k] for k in hf])),
                    "between_hf_db": float(np.mean([btw[k] for k in hf])),
                    "attack_hf_coherence": float(np.mean([coh_m[k] for k in hf]))},
    }

def _analysis_excerpts(sec):
    """(group name, src, tgt, style_flag, pickup_flag, [(offset_sec, dur), ...]) per training group:
    len(ANALYSIS_EXCERPT_POSITIONS) excerpts of the first take's held-out tail (attack statistics vary
    a lot along a take — a single excerpt is not representative)."""
    load_training_data()
    out = []
    for (pickup_name, style_name), pairs in pairs_by_group.items():
        if not pairs:
            continue
        src, tgt = pairs[0]
        dur = sf.info(src).duration
        each = max(2.0, float(sec) / len(ANALYSIS_EXCERPT_POSITIONS))
        spans = [(f * dur, min(each, dur - f * dur)) for f in ANALYSIS_EXCERPT_POSITIONS if dur - f * dur > 1.0]
        out.append((f"{pickup_name}/{style_name}", src, tgt, 1.0 if style_name == "pick" else 0.0,
                    PICKUP_MAGNETIC if pickup_name == "magnetic" else PICKUP_PIEZO, spans))
    return out

def analyze_attacks(kind, use_finetuned=False, sec=20.0, as_trained=False):
    sr = OFFLINE_SAMPLE_RATE if kind == "offline" else REALTIME_SAMPLE_RATE
    path = _select_stems_model_path(kind, use_finetuned=use_finetuned, prefer_static=False)
    if path is None or not os.path.exists(path):
        print(f"[attacks] no {kind} model found"); return None
    print(f"[attacks] {kind}: {path} ({'0 dB knobs, as trained' if as_trained else 'runtime knobs, as --process / the pedal'})")
    model = load_keras_model(path, compile=False)
    runner = KerasStreamingRunner(model) if kind == "realtime" else None
    report = {"kind": kind, "model": path, "model_mtime": os.path.getmtime(path), "as_trained": bool(as_trained),
              "time": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}
    for name, src, tgt, style, pickup, spans in _analysis_excerpts(sec):
        if as_trained:
            knobs = prepare_demo_knob_feeds(kind, pickattack=1.0, pickreso=1.0, main=1.0, ir=1.0, pickup=pickup)
        else:
            knobs = prepare_process_knob_feeds(kind, use_finetuned=use_finetuned, pickup=pickup)
        xs, ys, ms, bounds = [], [], [], []
        pre = int(ANALYSIS_PREROLL_SEC * sr)
        for off, dur in spans:
            x, y = _read_pair_segment(src, tgt, sr, max(0.0, off - ANALYSIS_PREROLL_SEC), dur + ANALYSIS_PREROLL_SEC)
            if kind == "offline":
                st, sfo, _ = _make_process_style_arrays(style)
                y_m = np.concatenate([b for b, _ in _iter_offline_blocks(model, x, knobs, st, sfo)])
            else:
                y_m = runner.run(x, knobs, style_true=style)["audio"]
            n = min(len(x), len(y), len(y_m))
            bounds.append(sum(len(v) for v in xs))
            xs.append(x[pre:n]); ys.append(y[pre:n]); ms.append(y_m[pre:n])
        m = attack_metrics(np.concatenate(xs), np.concatenate(ys), np.concatenate(ms), sr, boundaries=bounds[1:])
        m["excerpts"] = {"src": src, "spans": [[round(o, 3), round(d, 3)] for o, d in spans]}
        report["groups"][name] = m
        _print_attack_metrics(name, m, sum(d for _, d in spans))
    os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
    out_json = os.path.join(ANALYSIS_OUT_DIR, f"attacks_{kind}{'_finetuned' if use_finetuned else ''}"
                                              f"{'_astrained' if as_trained else ''}.json")
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                prev = json.load(f)
            print(f"\n[attacks] change vs previous run ({prev.get('time')}, {os.path.basename(prev.get('model', '?'))}):")
            for name, m in report["groups"].items():
                p = prev.get("groups", {}).get(name)
                if p and p.get("excerpts") != m["excerpts"]:
                    print(f"   {name:<16} (previous run scored different excerpts — not comparable)")
                elif p:
                    d = {k: m["summary"][k] - p["summary"][k] for k in m["summary"]}
                    print(f"   {name:<16} attack HF {d['attack_hf_db']:+.1f} dB   between-attack HF {d['between_hf_db']:+.1f} dB"
                          f"   attack HF coherence {d['attack_hf_coherence']:+.3f}")
        except Exception as e:
            print(f"[attacks] could not compare with {out_json}: {e}")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[attacks] written {out_json}")
    return report

def _print_attack_metrics(name, m, dur):
    print(f"\n=== {name}  ({dur:.0f} s held-out, {m['n_attacks']} attacks) ===")
    print(f"{'band (Hz)':<14}{'attack':>9}{'between':>9}{'coh model':>11}{'coh piezo':>11}   (dB = model vs mic)")
    for k in m["attack_db"]:
        print(f"{k:<14}{m['attack_db'][k]:>+9.1f}{m['between_db'][k]:>+9.1f}{m['coh_model'][k]:>11.2f}{m['coh_piezo'][k]:>11.2f}")
    g = m["timing_grid_ms"]
    print("6-12 kHz after the mic onset (dB re the mic's first ms):")
    print(f"{'  window ms':<14}" + "".join(f"{f'{a}..{b}':>8}" for a, b in g))
    for who in ("mic", "piezo", "model"):
        print(f"{'  ' + who:<14}" + "".join(f"{v:>8.1f}" for v in m["timing_db"][who]))
    s = m["summary"]
    print(f"summary: attack HF (6-20 kHz) {s['attack_hf_db']:+.1f} dB, between attacks {s['between_hf_db']:+.1f} dB, "
          f"attack HF coherence {s['attack_hf_coherence']:.2f}")
