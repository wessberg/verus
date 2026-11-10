import json
import numpy as np
import os
import soundfile as sf
import tensorflow as tf

from .config import (
    MODELS_REALTIME_OUT_DIR,
    REALTIME_SAMPLE_RATE,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_T,
)
from .data import (
    pickup_flag_for_path,
)
from .dsp import (
    _exp_kernel_rev_ms,
    _hp_minphase_rev_padded,
    _rms,
    resample_to_model_sr,
)
from .export import (
    _convert_keras_to_tflite_fp32_checked,
    build_realtime_streaming_export_for_tflite,
)
from .knobs import (
    _filter_feed_to_model_inputs,
    _resolve_demo_knobs,
    prepare_demo_knob_feeds,
)
from .modelio import (
    load_keras_model,
)
from .models import (
    build_realtime_streaming_model,
    copy_weights_by_name,
)
from .stems import (
    _tflite_run_full_audio,
)
from .tflite import (
    TFLiteRunner,
)

def _patch_pickmask_layer_streaming(layer,
                                    sample_rate,
                                    env_fast_ms=None,
                                    env_slow_ms=None,
                                    smooth_attack_ms=None,
                                    smooth_release_ms=None,
                                    thr_db=None,
                                    knee_db=None,
                                    post_comp_p=None,
                                    post_comp_gain=None,
                                    floor=None,
                                    out_gain=None):
    """
    Patch StreamingPickDBMask in-place:
      - set attributes: post_comp_p, post_comp_gain, floor
      - update scalar weights: thr_db, sigmoid slope (from knee), out_gain
      - rebuild envelope taps (keep lengths)
    Call BEFORE converting to TFLite.
    """
    # Attributes (captured as constants at conversion)
    if post_comp_p is not None:    layer.post_comp_p   = float(post_comp_p)
    if post_comp_gain is not None: layer.post_comp_gain= float(post_comp_gain)
    if floor is not None:          layer.floor         = float(floor)

    ws = layer.get_weights()
    if not ws:
        return
    # name->index map
    wmap = {w.name.split('/')[-1].split(':')[0]: i for i, w in enumerate(layer.weights)}

    def _set_scalar_weight(key, valf):
        if key in wmap and valf is not None:
            i = wmap[key]
            ws[i] = np.array([float(valf)], dtype=np.float32).reshape(ws[i].shape)

    _set_scalar_weight("dbmask_thr_db", thr_db)
    if knee_db is not None:
        slope_val = np.log(9.0) / max(float(knee_db), 1e-3)
        _set_scalar_weight("dbmask_sigmoid_slope", slope_val)
    _set_scalar_weight("dbmask_out_gain", out_gain)

    def _maybe_set_kernel(key, new_ms, scale):
        if key not in wmap or new_ms is None:
            return
        i = wmap[key]
        L = int(ws[i].shape[0])
        ws[i] = _exp_kernel_rev_ms(new_ms, sample_rate, scale, L)

    _maybe_set_kernel("dbmask_env_fast", env_fast_ms, 3.0)
    _maybe_set_kernel("dbmask_env_slow", env_slow_ms, 3.0)
    _maybe_set_kernel("dbmask_att",      smooth_attack_ms, 4.0)
    _maybe_set_kernel("dbmask_rel",      smooth_release_ms,4.0)

    layer.set_weights(ws)

def _patch_streaming_posthp_taps(layer, sample_rate, cutoff_hz):
    """
    Patch StreamFixedFIR 'realtime_pickattack_post_hp' taps to a new cutoff, keeping length.
    """
    if cutoff_hz is None:
        return
    ws = layer.get_weights()
    if not ws:
        return
    k0 = ws[0]
    L = int(k0.shape[0]) if k0.ndim == 3 else int(k0.shape[0])
    k_new = _hp_minphase_rev_padded(int(sample_rate), float(cutoff_hz), int(L))
    k_new = k_new.numpy() if tf.is_tensor(k_new) else np.asarray(k_new, np.float32)
    if k_new.shape != k0.shape:
        k_new = k_new.reshape(k0.shape)
    ws[0] = k_new.astype(np.float32)
    layer.set_weights(ws)

def _patch_softclip_attrs(layer, drive=None, limit=None):
    if drive is not None:
        layer.drive = float(drive)
    if limit is not None:
        layer.limit = float(limit)

def build_streaming_rt_with_params(load_from_path=None,
                                   softclip_drive=None, softclip_limit=None,
                                   env_fast_ms=None, env_slow_ms=None,
                                   smooth_attack_ms=None, smooth_release_ms=None,
                                   thr_db=None, knee_db=None,
                                   post_comp_p=None, post_comp_gain=None,
                                   mask_floor=None, mask_out_gain=None,
                                   posthp_cutoff_hz=None,
                                   static_T=REALTIME_TFLITE_T):
    """
    Build a streaming realtime model (static_T), copy weights from saved streaming model,
    then patch pickattack parameters in-place so conversion will capture them.
    """
    if load_from_path is None:
        load_from_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    base_stream = load_keras_model(load_from_path, compile=False)

    m = build_realtime_streaming_model(dynamic_time=False, static_T=int(static_T))
    copy_weights_by_name(m, base_stream, verbose=False)

    # Mask (StreamingPickDBMask) and post-HPF (StreamFixedFIR) and softclip attrs
    try:
        pm = m.get_layer("realtime_pick_mask_db")
        _patch_pickmask_layer_streaming(
            pm, sample_rate=REALTIME_SAMPLE_RATE,
            env_fast_ms=env_fast_ms, env_slow_ms=env_slow_ms,
            smooth_attack_ms=smooth_attack_ms, smooth_release_ms=smooth_release_ms,
            thr_db=thr_db, knee_db=knee_db, post_comp_p=post_comp_p, post_comp_gain=post_comp_gain,
            floor=mask_floor, out_gain=mask_out_gain
        )
    except Exception as e:
        print(f"[sweep/tflite] mask patch skipped: {e}")

    try:
        hp = m.get_layer("realtime_pickattack_post_hp")
        _patch_streaming_posthp_taps(hp, sample_rate=REALTIME_SAMPLE_RATE, cutoff_hz=posthp_cutoff_hz)
    except Exception as e:
        print(f"[sweep/tflite] post-HPF patch skipped: {e}")

    try:
        sc = m.get_layer("realtime_pickattack_softclip")
        _patch_softclip_attrs(sc, drive=softclip_drive, limit=softclip_limit)
    except Exception as e:
        print(f"[sweep/tflite] softclip patch skipped: {e}")

    return m

def export_streaming_rt_to_tflite(m_stream, out_tflite_path, T_fixed=REALTIME_TFLITE_T):
    """
    Wrap the streaming model for TFLite (bundle states), convert to FP32, save.
    """
    export = build_realtime_streaming_export_for_tflite(m_stream)
    _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="realtime", T_fixed=int(T_fixed))
    return out_tflite_path

def _render_pickattack_diff_keras_streaming(
    m_stream,
    x_mod,
    y_mod,
    use_finetuned,
    out_dir,
    apply_demo_boosts=True,
    pickup=None
):
    """
    STREAMING Keras large-T: isolate pickattack by knob-gating (no subtraction).
      - full:    demo knobs as-is
      - pa_only: main=0, pickattack=demo (or 1.0), pickreso=0, ir=0
    """
    os.makedirs(out_dir, exist_ok=True)

    # Zero states for streaming graph
    zero_states = _make_zero_state_feeds_for_streaming_model(m_stream)

    # Find fixed T and pad one shot
    in_piezo = None
    for t in m_stream.inputs:
        nm = t.name.split(':')[0].split('/')[-1]
        if nm == "realtime_piezo_input":
            in_piezo = t
            break
    if in_piezo is None:
        raise RuntimeError("Streaming model is missing 'realtime_piezo_input'.")

    T_fixed = int(in_piezo.shape[1]) if in_piezo.shape[1] is not None else len(x_mod)
    core = min(T_fixed, len(x_mod))

    x_pad = np.zeros((1, T_fixed, 1), dtype=np.float32)
    x_pad[0, :core, 0] = x_mod[:core]

    # Predicted style
    st_true  = np.zeros((1, 1), dtype=np.float32)
    st_force = np.zeros((1, 1), dtype=np.float32)

    # Base/demo knobs for full mix
    base_knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned, pickup=pickup)

    # Demo pickattack linear gain (or unity if you prefer)
    demo_pa_lin = _resolve_demo_knobs("realtime", use_finetuned=use_finetuned)["pickattack"]
    pa_on_val   = float(demo_pa_lin) if apply_demo_boosts else 1.0

    # Pickattack-only knobs: main=0, PR=0, IR=0, PA=pa_on_val
    pa_knobs = prepare_demo_knob_feeds(
        kind="realtime",
        use_finetuned=use_finetuned,
        main=0.0,
        pickattack=pa_on_val,
        pickreso=0.0,
        ir=0.0,
        pickup=pickup
    )

    def _predict_with_knobs(knobs_dict):
        feed = {
            "realtime_piezo_input": x_pad,
            "realtime_style_true_input":  st_true,
            "realtime_style_force_input": st_force,
            **zero_states,
            **knobs_dict
        }
        feed = _filter_feed_to_model_inputs(m_stream, feed)
        outs = m_stream.predict(feed, batch_size=1, verbose=0)
        y = np.asarray(outs[0], dtype=np.float32)[0, :, 0]
        return y[:core]

    # Full mix (demo knobs)
    y_full = _predict_with_knobs(base_knobs)

    # Pickattack-only mix (direct isolation)
    y_pa_only = _predict_with_knobs(pa_knobs)

    # Writes
    sf.write(os.path.join(out_dir, "pred_full_keras.wav"), y_full, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "stem_pickattack_only_keras.wav"), y_pa_only, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    if y_mod is not None:
        sf.write(os.path.join(out_dir, "mic_target.wav"), y_mod[:core], REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "piezo_source.wav"), x_mod[:core], REALTIME_SAMPLE_RATE, subtype='FLOAT')

    pa_db = 20.0 * np.log10(pa_on_val + 1e-12)
    print(f"[keras/stream] wrote full and pickattack stem | PA linear={pa_on_val:.4f} ({pa_db:.2f} dB) | "
          f"rms full={_rms(y_full):.6f} pa_only={_rms(y_pa_only):.6f}")
    
def _render_pickattack_diff_tflite(
    tflite_path,
    src_path,
    tgt_path=None,
    sec=10.0,
    offset_sec=0.0,
    out_dir="./out/ab_tflite",
    hop=None,
    use_finetuned=False,
    apply_demo_boosts=True,
    pickup=None
):
    """
    TFLite: isolate pickattack by knob-gating (no subtraction).
      - full:    demo knobs as-is
      - pa_only: main=0, pickattack=demo (or 1.0), pickreso=0, ir=0
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load/resample
    info = sf.info(src_path)
    sr_in = info.samplerate
    start = int(round(offset_sec * sr_in))
    frames = int(round(sec * sr_in))

    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0].astype(np.float32, copy=False)
    x_mod = resample_to_model_sr(x, sr_in, REALTIME_SAMPLE_RATE) if sr_in != REALTIME_SAMPLE_RATE else x

    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0].astype(np.float32, copy=False)
        y_mod = resample_to_model_sr(y, sr_in, REALTIME_SAMPLE_RATE) if sr_in != REALTIME_SAMPLE_RATE else y

    runner = TFLiteRunner(tflite_path)

    # Demo PA lin or unity
    demo_pa_lin = _resolve_demo_knobs("realtime", use_finetuned=use_finetuned)["pickattack"]
    pa_on_val   = float(demo_pa_lin) if apply_demo_boosts else 1.0
    core_hop    = hop if hop is not None else REALTIME_TFLITE_T

    # Base knobs → full mix
    base_knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned, pickup=pickup)

    # PA-only knobs: main=0, PR=0, IR=0, PA=pa_on_val
    pa_knobs = prepare_demo_knob_feeds(
        kind="realtime",
        use_finetuned=use_finetuned,
        main=0.0,
        pickattack=pa_on_val,
        pickreso=0.0,
        ir=0.0,
        pickup=pickup
    )

    # Full (demo knobs)
    y_full, _ = _tflite_run_full_audio(
        runner, kind="realtime", x_mod=x_mod, sample_rate=REALTIME_SAMPLE_RATE,
        style_mode={"mode": "pred", "value": None},
        knobs=base_knobs,
        fixed_hop=core_hop
    )

    # Pickattack-only (direct isolation)
    y_pa_only, _ = _tflite_run_full_audio(
        runner, kind="realtime", x_mod=x_mod, sample_rate=REALTIME_SAMPLE_RATE,
        style_mode={"mode": "pred", "value": None},
        knobs=pa_knobs,
        fixed_hop=core_hop
    )

    # Writes
    sf.write(os.path.join(out_dir, "pred_full_tflite.wav"), y_full, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "stem_pickattack_only_tflite.wav"), y_pa_only, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    if y_mod is not None:
        sf.write(os.path.join(out_dir, "mic_target.wav"), y_mod, REALTIME_SAMPLE_RATE, subtype='FLOAT')
    sf.write(os.path.join(out_dir, "piezo_source.wav"), x_mod, REALTIME_SAMPLE_RATE, subtype='FLOAT')

    pa_db = 20.0 * np.log10(pa_on_val + 1e-12)
    print(f"[tflite] wrote full and pickattack stem | PA linear={pa_on_val:.4f} ({pa_db:.2f} dB) | "
          f"rms full={_rms(y_full):.6f} pa_only={_rms(y_pa_only):.6f}")

def _load_and_resample_pair_for_realtime(src_path, tgt_path, sec, offset_sec, sample_rate=REALTIME_SAMPLE_RATE):
    info = sf.info(src_path)
    file_sr = info.samplerate
    start  = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0].astype(np.float32, copy=False)
    if file_sr != sample_rate:
        x_mod = resample_to_model_sr(x, file_sr, sample_rate)
    else:
        x_mod = x

    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0].astype(np.float32, copy=False)
        y_mod = resample_to_model_sr(y, file_sr, sample_rate) if (file_sr != sample_rate) else y
    return x_mod, y_mod

def _make_zero_state_feeds_for_streaming_model(model):
    feed = {}
    for t in model.inputs:
        nm = t.name.split(':')[0].split('/')[-1]
        if nm.startswith("realtime_state_"):
            s1 = int(t.shape[1]) if t.shape[1] is not None else 0
            s2 = int(t.shape[2]) if t.shape[2] is not None else 1
            feed[nm] = np.zeros((1, s1, s2), dtype=np.float32)
    return feed

def sweep_pickattack_params_with_tflite(
    src_path,
    tgt_path=None,
    param_grid=None,
    sec=10.0,
    offset_sec=0.0,
    base_out_dir="./out/fast_sweeps_ab",
    load_streaming_from=None,
    hop=REALTIME_TFLITE_T,
    use_finetuned=False,
    apply_demo_boosts=True,
    keras_stream_T=60000
):
    """
    For each param combo:
      1) Build a streaming Keras model with static_T = max(keras_stream_T, len(x_mod)) and render A/B in one predict().
      2) Build a streaming model with static_T=32 and export TFLite; render A/B (difference-of-passes) via the runner.

    keras_stream_T: lower bound for the Keras streaming static T (int). The actual T will be max(keras_stream_T, len(x_mod)).
    """
    # Load/resample once
    x_mod, y_mod = _load_and_resample_pair_for_realtime(
        src_path=src_path,
        tgt_path=tgt_path,
        sec=sec,
        offset_sec=offset_sec,
        sample_rate=REALTIME_SAMPLE_RATE
    )
    core_len = int(x_mod.shape[0])
    T_build = int(max(int(keras_stream_T), core_len))

    # Default param grid
    if param_grid is None:
        param_grid = {
            "softclip_drive": [1.6, 1.8, 2.0],
            "knee_db": [6.0, 10.0, 14.0],
            "smooth_release_ms": [24.0, 36.0],
            "posthp_cutoff_hz": [2800.0, 3200.0, 3600.0]
        }

    # Build all combos
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    def _recurse(i, cur):
        if i == len(keys):
            combos.append(dict(cur))
            return
        k = keys[i]
        for v in values[i]:
            cur[k] = v
            _recurse(i+1, cur)
    _recurse(0, {})

    # Ensure base directory
    os.makedirs(base_out_dir, exist_ok=True)

    print(f"[sweep/ab] {len(combos)} variants… (Keras streaming T={T_build}, TFLite T={REALTIME_TFLITE_T})")

    # Iterate variants
    for idx, cfg in enumerate(combos):
        # Folder name with params
        tag_parts = [f"{k}={str(cfg[k]).replace('.','p')}" for k in keys if k in cfg]
        tag = "_".join(tag_parts) if tag_parts else f"combo_{idx:03d}"
        out_dir = os.path.join(base_out_dir, f"{idx:03d}_{tag}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) Keras STREAMING large-T model (build JIT and patch params)
        m_stream_large = build_streaming_rt_with_params(
            load_from_path=load_streaming_from,
            softclip_drive=cfg.get("softclip_drive", None),
            softclip_limit=None,
            env_fast_ms=cfg.get("env_fast_ms", None),
            env_slow_ms=cfg.get("env_slow_ms", None),
            smooth_attack_ms=cfg.get("smooth_attack_ms", None),
            smooth_release_ms=cfg.get("smooth_release_ms", None),
            thr_db=cfg.get("thr_db", None),
            knee_db=cfg.get("knee_db", None),
            post_comp_p=cfg.get("post_comp_p", None),
            post_comp_gain=cfg.get("post_comp_gain", None),
            mask_floor=cfg.get("mask_floor", None),
            mask_out_gain=cfg.get("mask_out_gain", None),
            posthp_cutoff_hz=cfg.get("posthp_cutoff_hz", None),
            static_T=T_build
        )
        out_keras = os.path.join(out_dir, "keras")
        os.makedirs(out_keras, exist_ok=True)
        _render_pickattack_diff_keras_streaming(
            m_stream=m_stream_large,
            x_mod=x_mod,
            y_mod=y_mod,
            use_finetuned=use_finetuned,
            out_dir=out_keras,
            apply_demo_boosts=apply_demo_boosts,
            pickup=pickup_flag_for_path(src_path)
        )

        # 2) Streaming T=32 → TFLite export and render via runner (unchanged)
        m_stream_t32 = build_streaming_rt_with_params(
            load_from_path=load_streaming_from,
            softclip_drive=cfg.get("softclip_drive", None),
            softclip_limit=None,
            env_fast_ms=cfg.get("env_fast_ms", None),
            env_slow_ms=cfg.get("env_slow_ms", None),
            smooth_attack_ms=cfg.get("smooth_attack_ms", None),
            smooth_release_ms=cfg.get("smooth_release_ms", None),
            thr_db=cfg.get("thr_db", None),
            knee_db=cfg.get("knee_db", None),
            post_comp_p=cfg.get("post_comp_p", None),
            post_comp_gain=cfg.get("post_comp_gain", None),
            mask_floor=cfg.get("mask_floor", None),
            mask_out_gain=cfg.get("mask_out_gain", None),
            posthp_cutoff_hz=cfg.get("posthp_cutoff_hz", None),
            static_T=REALTIME_TFLITE_T
        )
        tflite_dir = os.path.join(out_dir, "tflite")
        os.makedirs(tflite_dir, exist_ok=True)
        tfl_path = os.path.join(tflite_dir, "model.tflite")
        export_streaming_rt_to_tflite(m_stream_t32, tfl_path, T_fixed=REALTIME_TFLITE_T)

        _render_pickattack_diff_tflite(
            tflite_path=tfl_path,
            src_path=src_path,
            tgt_path=tgt_path,
            sec=sec,
            offset_sec=offset_sec,
            out_dir=tflite_dir,
            hop=hop,
            use_finetuned=use_finetuned,
            apply_demo_boosts=apply_demo_boosts,
            pickup=pickup_flag_for_path(src_path)
        )

        with open(os.path.join(out_dir, "params.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[sweep/ab] Variant {idx+1}/{len(combos)} → {out_dir}")
