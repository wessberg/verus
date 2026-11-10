import math
import numpy as np
import os
import soundfile as sf
import sys
import tensorflow as tf
from tensorflow.keras.layers import (Activation)  # pyright: ignore[reportMissingImports]

from .config import (
    DEMO_PAIR_INDEX,
    FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME,
    FINETUNE_REALTIME_STATIC_MODEL_FILENAME,
    FINETUNE_SRC_DIR,
    LONG_DEMO_OFFSET_SEC,
    LONG_DEMO_SEC,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_STATIC_MODEL_FILENAME,
    REALTIME_DYNAMIC_MODEL_FILENAME,
    REALTIME_SAMPLE_RATE,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_T,
    STEMS_FINETUNE_OFFLINE_OUT_DIR,
    STEMS_FINETUNE_REALTIME_OUT_DIR,
    STEMS_OFFLINE_OUT_DIR,
    STEMS_REALTIME_OUT_DIR,
    STYLE_FLAG_PROB_THRESHOLD,
)
from .contexts import (
    compute_streaming_ctx_constants,
)
from .data import (
    _read_pair_segment,
    all_pairs,
    assert_has_pairs,
    pickup_flag_for_path,
    style_flag_for_path,
)
from .dsp import (
    resample_to_model_sr,
)
from .inference import (
    KerasStreamingRunner,
    _get_model_context_for_streaming,
)
from .knobs import (
    prepare_demo_knob_feeds,
)
from .modelio import (
    load_keras_model,
)
from .tflite import (
    TFLiteRunner,
    _select_tflite_demo_paths,
)

def _select_stems_model_path(kind, use_finetuned=False, prefer_static=False):
    """
    Select a saved Keras model filepath for stems/demos.
    - kind: 'offline' or 'realtime'
    - use_finetuned: prefer finetuned artifacts if present
    - prefer_static: if True, prefer 'static' artifacts; else prefer 'dynamic' artifacts
      Note: realtime static is the streaming T={REALTIME_TFLITE_T} graph; offline static uses fixed T train graph.
    """
    if kind == "offline":
        # Offline artifacts
        finetuned_dyn  = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        finetuned_stat = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
        baseline_dyn   = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
        baseline_stat  = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)

        if use_finetuned:
            if prefer_static and os.path.exists(finetuned_stat):
                return finetuned_stat
            if os.path.exists(finetuned_dyn):
                return finetuned_dyn
            # fallback to baseline if finetuned missing
            if prefer_static and os.path.exists(baseline_stat):
                return baseline_stat
            if os.path.exists(baseline_dyn):
                return baseline_dyn
            return None

        # baseline
        if prefer_static and os.path.exists(baseline_stat):
            return baseline_stat
        if os.path.exists(baseline_dyn):
            return baseline_dyn
        return None

    elif kind == "realtime":
        # Realtime artifacts (streaming)
        finetuned_dyn  = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        finetuned_stat = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        baseline_dyn   = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        baseline_stat  = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)

        if use_finetuned:
            if prefer_static and os.path.exists(finetuned_stat):
                return finetuned_stat
            if os.path.exists(finetuned_dyn):
                return finetuned_dyn
            # fallback to baseline if finetuned missing
            if prefer_static and os.path.exists(baseline_stat):
                return baseline_stat
            if os.path.exists(baseline_dyn):
                return baseline_dyn
            return None

        # baseline
        if prefer_static and os.path.exists(baseline_stat):
            return baseline_stat
        if os.path.exists(baseline_dyn):
            return baseline_dyn
        return None

    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

def render_isolated_stems_realtime_runtime(model,
                                           path_src,
                                           path_tgt,
                                           out_dir,
                                           sample_rate,
                                           sec=LONG_DEMO_SEC,
                                           offset_sec=LONG_DEMO_OFFSET_SEC,
                                           hop=REALTIME_TFLITE_T,
                                           use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)

    def rms(x):
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0

    # Read/resample
    x_mod, y_mod = _read_pair_segment(path_src, path_tgt, sample_rate, offset_sec, sec)

    # Build a tap-model that exposes post‑knob heads and main
    taps = []
    found = {}
    def tap(lname, tname):
        try:
            t = model.get_layer(lname).output
            taps.append(Activation('linear', name=tname)(t))
            found[tname] = True
            return True
        except Exception:
            found[tname] = False
            return False

    tap("realtime_pickattack_user_gain_mul", "tap_pickattack")
    tap("realtime_pickreso_user_gain_mul",   "tap_pickreso")
    tap("realtime_ir_user_gain_mul",         "tap_ir")
    # Prefer post-user-gain main; fall back to pre-user main if needed
    tap("realtime_main_user_gain_mul",       "tap_main_user")
    tap("realtime_main",                     "tap_main")

    tap_model = tf.keras.Model(inputs=model.inputs, outputs=list(model.outputs) + taps, name=model.name + "_taps")
    runner = KerasStreamingRunner(tap_model, hop=(hop if tap_model.inputs[0].shape[1] is not None else None))
    tap_idx = {n: runner.out_index[n] for n in found if found[n] and n in runner.out_index}

    knobs = prepare_demo_knob_feeds(kind="realtime", use_finetuned=use_finetuned,
                                    pickup=pickup_flag_for_path(path_src))

    # Single-pass streamer
    def run_stream(style_true=None, style_force=False):
        res = runner.run(x_mod, knobs, style_true=(style_true if style_force else None), taps=tap_idx)
        t = res["taps"]
        y_full = res["audio"]
        y_main = t.get("tap_main_user", t.get("tap_main", y_full))
        z = np.zeros_like(y_full)
        return {"full": y_full, "main": y_main,
                "pa": t.get("tap_pickattack", z), "pr": t.get("tap_pickreso", z), "ir": t.get("tap_ir", z)}

    # DC removal + peak protection for stems
    def finalize(sig, headroom_db=1.0, dc_remove=True):
        s = np.asarray(sig, np.float32)
        if s.size == 0:
            return s
        if dc_remove:
            s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)) + 1e-12)
        ceiling = 10.0 ** (float(-headroom_db) / 20.0)
        if peak > ceiling:
            s = s * (ceiling / peak)
        return s

    # Passes and writers
    passes = {
        "pred":        {"style_true": None, "style_force": False},
        "forced_pick": {"style_true": 1.0,  "style_force": True},
        "forced_fing": {"style_true": 0.0,  "style_force": True},
    }
    writers = {}
    def open_writers(key):
        if key == "pred":
            writers["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            writers["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            writers["full_pred"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        elif key == "forced_pick":
            writers["full_forced_pick"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            writers["full_forced_finger"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"main_core_{key}"] = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only_{key}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"pickattack_{key}"] = sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only_{key}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"pickreso_{key}"]   = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only_{key}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"ir_{key}"]         = sf.SoundFile(os.path.join(out_dir, f"stem_ir_lowpass_{key}.wav"),     mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        writers[f"sumcheck_{key}"]   = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck_{key}.wav"),  mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

    print("[stems/realtime] taps found:", found)

    for key, cfg in passes.items():
        open_writers(key)
        res = run_stream(style_true=cfg["style_true"], style_force=cfg["style_force"])

        # Finalize stems (DC removal + peak protection)
        main_f = finalize(res["main"], headroom_db=1.0, dc_remove=True)
        pa_f   = finalize(res["pa"],   headroom_db=1.0, dc_remove=True)
        pr_f   = finalize(res["pr"],   headroom_db=1.0, dc_remove=True)
        ir_f   = finalize(res["ir"],   headroom_db=1.0, dc_remove=True)

        # Write full
        if key == "pred":
            writers["src"].write(x_mod); writers["tgt"].write(y_mod); writers["full_pred"].write(res["full"])
        elif key == "forced_pick":
            writers["full_forced_pick"].write(res["full"])
        else:
            writers["full_forced_finger"].write(res["full"])

        # Write stems (finalized)
        writers[f"main_core_{key}"].write(main_f)
        writers[f"pickattack_{key}"].write(pa_f)
        writers[f"pickreso_{key}"].write(pr_f)
        writers[f"ir_{key}"].write(ir_f)

        # Sumcheck (use finalized stems)
        sum_est = np.clip(main_f + pa_f + pr_f + ir_f, -1.0, 1.0)
        writers[f"sumcheck_{key}"].write(sum_est)

        print(f"[stems/realtime][{key}] wrote RMS main={rms(main_f):.6f} pa={rms(pa_f):.6f} pr={rms(pr_f):.6f} ir={rms(ir_f):.6f}")

    for w in list(writers.values()):
        try: w.close()
        except Exception: pass

    print(f"✅ Realtime stems (pred + forced_pick + forced_fing) written to {out_dir}")

def render_isolated_stems_offline_runtime(model,
                                          path_src,
                                          path_tgt,
                                          out_dir,
                                          sample_rate,
                                          sec=LONG_DEMO_SEC,
                                          offset_sec=LONG_DEMO_OFFSET_SEC,
                                          block_sec=12.0,
                                          resample_margin_ms=50.0,
                                          use_finetuned=False):
    os.makedirs(out_dir, exist_ok=True)
    knob_base = prepare_demo_knob_feeds(kind="offline", use_finetuned=use_finetuned,
                                        pickup=pickup_flag_for_path(path_src))

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    # Style inference for file naming; forced passes are always emitted
    style_true_flag = style_flag_for_path(path_src)

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    def filter_feed(d):
        names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
        return {k: v for k, v in d.items() if k in names}

    # Writers
    def open_writers(pass_key):
        w = {}
        if pass_key == "pred":
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        elif pass_key == "forced_pick":
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

        # Stems (direct passes)
        suffix = "" if pass_key == "pred" else ("_forced_pick" if pass_key == "forced_pick" else "_forced_finger")
        w["main"]      = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only{'' if pass_key=='pred' else suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickattack"]= sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only{'' if pass_key=='pred' else suffix}.wav"), mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["pickreso"]  = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only{'' if pass_key=='pred' else suffix}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        w["sum"]       = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck{'' if pass_key=='pred' else suffix}.wav"),   mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
        return w

    def close_writers(w):
        for v in w.values():
            try: v.close()
            except: pass

    # Make a copy with branch knobs zeroed helper
    def iso_knobs(base, which):
        k = {kk: base[kk].astype(np.float32, copy=True) for kk in base.keys()}
        # zeros
        k["offline_knob_main"]       = np.array([[0.0]], np.float32)
        k["offline_knob_pickattack"] = np.array([[0.0]], np.float32)
        k["offline_knob_pickreso"]   = np.array([[0.0]], np.float32)
        # select
        if which == "main":
            k["offline_knob_main"] = knob_base["offline_knob_main"]
        elif which == "pickattack":
            k["offline_knob_pickattack"] = knob_base["offline_knob_pickattack"]
        elif which == "pickreso":
            k["offline_knob_pickreso"] = knob_base["offline_knob_pickreso"]
        return k

    passes = [
        ("pred",         np.zeros((1,1), np.float32), np.zeros((1,1), np.float32)),
        ("forced_pick",  np.ones((1,1),  np.float32), np.ones((1,1),  np.float32)),
        ("forced_finger",np.zeros((1,1), np.float32), np.ones((1,1),  np.float32)),
    ]

    produced_model = 0
    for pass_key, st_true, st_force in passes:
        w = open_writers(pass_key)
        produced_model = 0

        while produced_model < want_total_model:
            core_model = min(block_core_model, want_total_model - produced_model)
            left_this = left_ctx_model if produced_model > 0 else 0
            right_this = right_ctx_model if (produced_model + core_model) < want_total_model else 0

            core_start_file_off = int(round(produced_model * (file_sr / float(sample_rate))))
            core_start_file = START_FRAME + core_start_file_off

            left_ctx_file = int(math.ceil(left_this * (file_sr / float(sample_rate))))
            right_ctx_file = int(math.ceil(right_this * (file_sr / float(sample_rate))))
            core_file_len = int(math.ceil(core_model * (file_sr / float(sample_rate))))

            read_a = max(START_FRAME, core_start_file - left_ctx_file - resample_margin_file)
            read_b = min(END_FRAME, core_start_file + core_file_len + right_ctx_file + resample_margin_file)
            n_read = max(0, read_b - read_a)
            if n_read <= 0:
                break

            x_block_2d, _ = sf.read(path_src, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            x_file = x_block_2d[:, 0]
            y_block_2d, _ = sf.read(path_tgt, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0]

            if file_sr != sample_rate:
                x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
                y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
            else:
                x_resamp = x_file
                y_resamp = y_file

            left_off_file = (core_start_file - left_ctx_file) - read_a
            left_off_model = int(round(left_off_file * (sample_rate / float(file_sr))))
            inp_start = max(0, min(left_off_model, len(x_resamp)))
            inp_end = max(inp_start, min(inp_start + left_this + core_model + right_this, len(x_resamp)))
            if inp_end <= inp_start:
                break

            x_inp = x_resamp[inp_start:inp_end]
            core_start_in_inp = min(left_this, len(x_inp))
            core_end_in_inp = min(core_start_in_inp + core_model, len(x_inp))
            core_len = max(0, core_end_in_inp - core_start_in_inp)
            if core_len <= 0:
                break

            x_demo = x_inp[None, :, None].astype(np.float32)

            # Build knob sets
            k_full = {kk: vv for kk, vv in knob_base.items()}
            k_main = iso_knobs(knob_base, "main")
            k_pa   = iso_knobs(knob_base, "pickattack")
            k_pr   = iso_knobs(knob_base, "pickreso")

            base_feed = {
                "offline_piezo_input": x_demo,
                "offline_style_true_input":  st_true,
                "offline_style_force_input": st_force,
            }
            # Run passes (index the outputs; the runtime model may expose 2 or 3 of them)
            def run_with(knobs):
                fd = filter_feed({**base_feed, **knobs})
                outs = model.predict(fd, batch_size=1, verbose=0)
                if not isinstance(outs, (list, tuple)):
                    outs = [outs]
                y = np.asarray(outs[0], np.float32)[0, :, 0]
                return y[core_start_in_inp:core_end_in_inp]

            y_full = run_with(k_full)
            y_main = run_with(k_main)
            y_pa   = run_with(k_pa)
            y_pr   = run_with(k_pr)

            # write
            w["full"].write(y_full)
            w["main"].write(y_main)
            w["pickattack"].write(y_pa)
            w["pickreso"].write(y_pr)
            # sumcheck:
            y_sum = np.clip(y_main + y_pa + y_pr, -1.0, 1.0)
            w["sum"].write(y_sum)

            # src/tgt for pred
            if pass_key == "pred":
                w["src"].write(x_resamp[inp_start + core_start_in_inp:inp_start + core_end_in_inp])
                w["tgt"].write(y_resamp[inp_start + core_start_in_inp:inp_start + core_end_in_inp])

            produced_model += core_len

        close_writers(w)

    print(f"✅ Offline stems (pred + forced_pick + forced_finger) written to {out_dir}")

def run_render_stems_offline(use_finetuned=False):
    sample_rate = OFFLINE_SAMPLE_RATE
    
    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="offline", use_finetuned=bool(use_finetuned))
    if dpath is None:
        print("[stems/offline] No offline dynamic models found."); sys.exit(1)
    print("[stems/offline] Using:", dpath)
    offline_dyn = load_keras_model(dpath, compile=False)

    out_dir = STEMS_FINETUNE_OFFLINE_OUT_DIR if use_finetuned else STEMS_OFFLINE_OUT_DIR

    # Use the runtime renderer (no diffing, uses knob_main/pickattack/pickreso)
    render_isolated_stems_offline_runtime(model=offline_dyn,
                                          path_src=src,
                                          path_tgt=tgt,
                                          out_dir=out_dir,
                                          sample_rate=sample_rate,
                                          sec=LONG_DEMO_SEC,
                                          offset_sec=LONG_DEMO_OFFSET_SEC,
                                          block_sec=12.0,
                                          resample_margin_ms=50.0,
                                          use_finetuned=use_finetuned)


def run_render_stems_realtime(use_finetuned=False, prefer_static=False):
    print("[stems/realtime] run_render_stems_realtime (runtime streaming path)")
    sample_rate = REALTIME_SAMPLE_RATE

    if use_finetuned:
        src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        src, tgt = all_pairs[DEMO_PAIR_INDEX]

    dpath = _select_stems_model_path(kind="realtime",
                                     use_finetuned=bool(use_finetuned),
                                     prefer_static=bool(prefer_static))
    if dpath is None:
        print("[stems/realtime] No realtime models found."); sys.exit(1)
    print("[stems/realtime] Using:", dpath)
    realtime_m = load_keras_model(dpath, compile=False)

    out_dir = STEMS_FINETUNE_REALTIME_OUT_DIR if use_finetuned else STEMS_REALTIME_OUT_DIR

    # Use the runtime streaming renderer that taps post‑knob heads
    render_isolated_stems_realtime_runtime(model=realtime_m,
                                           path_src=src,
                                           path_tgt=tgt,
                                           out_dir=out_dir,
                                           sample_rate=sample_rate,
                                           sec=LONG_DEMO_SEC,
                                           offset_sec=LONG_DEMO_OFFSET_SEC,
                                           hop=REALTIME_TFLITE_T,
                                           use_finetuned=use_finetuned)


def _stems_tflite_outdir(kind, use_finetuned=False):
    base = STEMS_FINETUNE_REALTIME_OUT_DIR if (use_finetuned and kind == "realtime") else \
           STEMS_FINETUNE_OFFLINE_OUT_DIR  if (use_finetuned and kind == "offline")  else \
           STEMS_REALTIME_OUT_DIR          if (kind == "realtime")                   else \
           STEMS_OFFLINE_OUT_DIR
    out_dir = os.path.join(base, "tflite")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _tflite_schedule(kind, runner, T_fixed, core_len=None):
    """
    Decide left/core/right schedule for a single pass.
    Returns dict with keys: is_stream, T_fixed, left, core, right.
    """
    is_stream = runner.is_streaming()
    if is_stream:
        if core_len is None:
            core_len = T_fixed
        # A streaming hop < T would feed the zero-padded tail of every block into the model
        # states and then re-feed the same samples next hop — state corruption, not just
        # "reduced style stability". Only hop == T is valid mid-stream (the final partial
        # block is fine because nothing follows it).
        if int(core_len) != int(T_fixed):
            raise ValueError(
                f"Streaming models require hop == T ({T_fixed}); got hop={core_len}. "
                f"A smaller hop corrupts the streaming states.")
        return {"is_stream": True, "T_fixed": T_fixed, "left": 0, "core": int(core_len), "right": 0}
    # Windowed (offline TFLite, or any non-streaming)
    if core_len is None:
        core_len = max(1, min(T_fixed // 2, 8192 if kind == "offline" else 512))
    lc_m, rc_m = compute_streaming_ctx_constants(kind)
    left_this  = min(int(lc_m), max(0, T_fixed - 1))
    right_this = min(int(rc_m), max(0, T_fixed - 1 - left_this))
    if (T_fixed - left_this - right_this) < core_len:
        need = core_len - (T_fixed - left_this - right_this)
        r_dec = min(right_this, need); right_this -= r_dec; need -= r_dec
        if need > 0:
            l_dec = min(left_this, need); left_this -= l_dec
    return {"is_stream": False, "T_fixed": T_fixed, "left": int(left_this), "core": int(core_len), "right": int(right_this)}

def _tflite_run_full_audio(runner, kind, x_mod, sample_rate, style_mode, knobs, fixed_hop=None):
    """
    Run one full pass and return mono audio vector.
    style_mode: dict with keys:
      - mode = 'pred' or 'forced'
      - value = None (pred) or float scalar in [0,1] (forced)
    knobs: dict of knob tensors keyed by model input name
    """
    # Determine T/hop
    piezo_key = f"{kind}_piezo_input"
    T_fixed = runner.input_length_T(piezo_key)
    sched = _tflite_schedule(kind, runner, T_fixed, core_len=(fixed_hop if runner.is_streaming() else None))
    is_stream = sched["is_stream"]
    core_len  = sched["core"]
    left      = sched["left"]
    right     = sched["right"]

    # Ensure we start with zero states for streaming
    if runner.is_streaming():
        runner.reset_states()

    # Build per-pass constant style inputs
    if style_mode["mode"] == "pred":
        st_true  = np.zeros((1,1), dtype=np.float32)
        st_force = np.zeros((1,1), dtype=np.float32)
    else:
        v = float(np.clip(style_mode.get("value", 0.0), 0.0, 1.0))
        st_true  = np.array([[v]], dtype=np.float32)
        st_force = np.ones((1,1), dtype=np.float32)

    # Output buffers, also return style probability average on 'pred'
    y_full = []
    prob_sum = 0.0
    prob_count = 0

    pos = 0
    N = len(x_mod)
    while pos < N:
        core = min(core_len, N - pos)
        x_in = np.zeros((1, T_fixed, 1), dtype=np.float32)
        if is_stream:
            x_in[0, :core, 0] = x_mod[pos:pos+core]
        else:
            # left / core / right
            ltake = min(left, pos)
            rtake = min(right, N - (pos + core))
            if ltake > 0:
                x_in[0, left - ltake:left, 0] = x_mod[pos - ltake:pos]
            x_in[0, left:left + core, 0] = x_mod[pos:pos + core]
            if rtake > 0:
                xs = pos + core; xe = xs + rtake
                x_in[0, left + core:left + core + rtake, 0] = x_mod[xs:xe]

        # Make feed
        feed = {piezo_key: x_in}
        if kind == "realtime":
            feed["realtime_style_true_input"]  = st_true
            feed["realtime_style_force_input"] = st_force
        else:
            feed["offline_style_true_input"]  = st_true
            feed["offline_style_force_input"] = st_force
        for k, v in knobs.items():
            feed[k] = v.astype(np.float32, copy=False)

        y_pred, s_pred = runner.invoke(feed)

        if is_stream:
            y_core = y_pred[0, :core, 0]
        else:
            y_core = y_pred[0, left:left + core, 0]

        y_full.append(np.asarray(y_core, dtype=np.float32))
        if style_mode["mode"] == "pred":
            prob_sum   += float(np.asarray(s_pred, dtype=np.float32)[0, 0]) * float(core)
            prob_count += core

        pos += core

    y = np.concatenate(y_full, axis=0) if y_full else np.zeros((0,), np.float32)
    avg_prob = (prob_sum / float(prob_count)) if prob_count > 0 else None
    return y, avg_prob

def render_isolated_stems_tflite(kind, use_finetuned=False, hop=None):
    """
    Render stems from the TFLite model using direct isolation:
      - main only:      main=base, pa=0, pr=0, ir=0
      - pickattack only:main=0,   pa=base, pr=0, ir=0
      - pickreso only:  main=0,   pa=0,    pr=base, ir=0
      - IR only (rt):   main=0,   pa=0,    pr=0, ir=base

    Writes WAVs under out/demos/stems/<kind>/tflite/.
    """
    assert kind in ("offline", "realtime")
    # Choose pair
    if use_finetuned:
        path_src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
        path_tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    else:
        assert_has_pairs()
        path_src, path_tgt = all_pairs[DEMO_PAIR_INDEX]

    # Style flag for display (only used to decide if forced pass is written)
    style_true_flag = style_flag_for_path(path_src)

    # Sample rate and TFLite path
    sr = OFFLINE_SAMPLE_RATE if (kind == "offline") else REALTIME_SAMPLE_RATE
    tflite_path = _select_tflite_demo_paths(kind=kind, use_finetuned=use_finetuned)
    if not os.path.exists(tflite_path):
        print(f"[stems/tflite] TFLite not found at {tflite_path}. Run --export-tflite first.")
        return
    print(f"[stems/tflite] Using: {tflite_path}")

    # Read/resample audio
    info = sf.info(path_src)
    file_sr = info.samplerate
    start = int(round(LONG_DEMO_OFFSET_SEC * file_sr))
    frames = int(round(LONG_DEMO_SEC * file_sr))
    x_blk_2d, _ = sf.read(path_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_blk_2d, _ = sf.read(path_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_file = x_blk_2d[:, 0]; y_file = y_blk_2d[:, 0]
    x_mod = x_file if file_sr == sr else resample_to_model_sr(x_file, file_sr, sr)
    y_mod = y_file if file_sr == sr else resample_to_model_sr(y_file, file_sr, sr)

    # Runner and default knobs
    runner = TFLiteRunner(tflite_path)
    base_knobs = prepare_demo_knob_feeds(kind=kind, use_finetuned=use_finetuned,
                                         pickup=pickup_flag_for_path(path_src))

    # Helpers to build per-stem knob sets
    def clone_knobs(k):
        return {kk: k[kk].astype(np.float32, copy=True) for kk in k.keys()}

    def knobs_main_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
            k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
            # keep realtime_knob_main as base
        else:
            k["offline_knob_pickattack"]  = np.array([[0.0]], np.float32)
            k["offline_knob_pickreso"]    = np.array([[0.0]], np.float32)
            # keep offline_knob_main as base
        return k

    def knobs_pa_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
            k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
            # keep pa
        else:
            k["offline_knob_main"]        = np.array([[0.0]], np.float32)
            k["offline_knob_pickreso"]    = np.array([[0.0]], np.float32)
            # keep pa
        return k

    def knobs_pr_only():
        k = clone_knobs(base_knobs)
        if kind == "realtime":
            k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
            k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
            k["realtime_knob_ir"]         = np.array([[0.0]], np.float32)
        else:
            k["offline_knob_main"]        = np.array([[0.0]], np.float32)
            k["offline_knob_pickattack"]  = np.array([[0.0]], np.float32)
        return k

    def knobs_ir_only():
        k = clone_knobs(base_knobs)
        # realtime only
        k["realtime_knob_main"]       = np.array([[0.0]], np.float32)
        k["realtime_knob_pickattack"] = np.array([[0.0]], np.float32)
        k["realtime_knob_pickreso"]   = np.array([[0.0]], np.float32)
        return k

    def finalize(sig, headroom_db=1.0, dc_remove=True):
        s = np.asarray(sig, np.float32)
        if s.size == 0: return s
        if dc_remove: s = s - float(np.mean(s))
        peak = float(np.max(np.abs(s)) + 1e-12)
        ceiling = 10.0 ** (-float(headroom_db) / 20.0)
        if peak > ceiling: s = s * (ceiling / peak)
        return s

    def rms(x):
        x = np.asarray(x, np.float32)
        return float(np.sqrt(np.mean(x*x) + 1e-12)) if x.size else 0.0

    out_dir = _stems_tflite_outdir(kind, use_finetuned=use_finetuned)

    def open_writers(pass_key):
        w = {}
        if pass_key == "pred":
            w["src"]  = sf.SoundFile(os.path.join(out_dir, "piezo_source.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
            w["tgt"]  = sf.SoundFile(os.path.join(out_dir, "mic_target.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_predicted_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        elif pass_key == "forced_pick":
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_pick_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        else:
            w["full"] = sf.SoundFile(os.path.join(out_dir, "pred_full_forced_finger_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        # Stems
        suffix = "" if pass_key == "pred" else ("_forced_pick" if pass_key == "forced_pick" else "_forced_finger")
        w["main"]      = sf.SoundFile(os.path.join(out_dir, f"stem_main_core_only{'' if pass_key=='pred' else suffix}_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["pickattack"]= sf.SoundFile(os.path.join(out_dir, f"stem_pickattack_only{'' if pass_key=='pred' else suffix}_tflite.wav"), mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["pickreso"]  = sf.SoundFile(os.path.join(out_dir, f"stem_pickreso_only{'' if pass_key=='pred' else suffix}_tflite.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        if kind == "realtime":
            w["ir"]    = sf.SoundFile(os.path.join(out_dir, f"stem_ir_lowpass{'' if pass_key=='pred' else suffix}_tflite.wav"),      mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        w["sum"]       = sf.SoundFile(os.path.join(out_dir, f"pred_full_sumcheck{'' if pass_key=='pred' else suffix}_tflite.wav"),   mode='w', samplerate=sr, channels=1, subtype='FLOAT')
        return w

    def close_all(w):
        for v in w.values():
            try: v.close()
            except: pass

    passes = [
        ("pred",         {"mode":"pred",   "value": None}),
        ("forced_pick",  {"mode":"forced", "value": 1.0}),
        ("forced_finger",{"mode":"forced", "value": 0.0})
    ]

    for key, sty in passes:
        w = open_writers(key)

        # Full mix (base knobs)
        y_full, p_avg = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=base_knobs, fixed_hop=hop)
        w["full"].write(y_full)

        # Direct stems by one-pass isolation
        y_main, _ = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_main_only(), fixed_hop=hop)
        y_pa, _   = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_pa_only(),   fixed_hop=hop)
        y_pr, _   = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_pr_only(),   fixed_hop=hop)
        if kind == "realtime":
            y_ir, _ = _tflite_run_full_audio(runner, kind, x_mod, sr, style_mode=sty, knobs=knobs_ir_only(), fixed_hop=hop)
        else:
            y_ir = np.zeros_like(y_main)

        # Finalize and write
        main_f = finalize(y_main)
        pa_f   = finalize(y_pa)
        pr_f   = finalize(y_pr)
        ir_f   = finalize(y_ir)

        if key == "pred":
            w["src"].write(x_mod); w["tgt"].write(y_mod)

        w["main"].write(main_f)
        w["pickattack"].write(pa_f)
        w["pickreso"].write(pr_f)
        if kind == "realtime":
            w["ir"].write(ir_f)

        # Sumcheck
        y_sum = np.clip(main_f + pa_f + pr_f + (ir_f if kind == "realtime" else 0.0), -1.0, 1.0)
        w["sum"].write(y_sum)

        if key == "pred" and p_avg is not None:
            lbl = "pick" if p_avg > STYLE_FLAG_PROB_THRESHOLD else "finger"
            print(f"[stems/tflite][pred] avg style={lbl} ({p_avg:.4f})")
        print(f"[stems/tflite][{key}] RMS main={rms(main_f):.6f} pa={rms(pa_f):.6f} pr={rms(pr_f):.6f}" + (f" ir={rms(ir_f):.6f}" if kind=="realtime" else ""))

        close_all(w)

    print(f"✅ TFLite stems ({kind}) written to {out_dir}")

def run_render_stems_tflite(args):
    """
    CLI front-end: --render-stems-tflite [--offline-only|--realtime-only] [--finetuned] [--both|--all] [--hop=N]
    """
    # Parse flags from args list (strings)
    argset = set(args)
    do_offline   = ("--realtime-only" not in argset)
    do_realtime  = ("--offline-only"  not in argset)
    use_finetuned = ("--finetuned"    in argset)
    do_both       = ("--both"         in argset) or ("--all" in argset)

    # Optional hop
    hop = None
    for a in list(argset):
        if a.startswith("--hop="):
            try:
                hop = int(a.split("=",1)[1])
            except Exception:
                hop = None

    # Render
    if do_offline:
        if do_both:
            render_isolated_stems_tflite("offline", use_finetuned=False, hop=None)
            render_isolated_stems_tflite("offline", use_finetuned=True,  hop=None)
        else:
            render_isolated_stems_tflite("offline", use_finetuned=use_finetuned, hop=None)

    if do_realtime:
        if do_both:
            render_isolated_stems_tflite("realtime", use_finetuned=False, hop=hop)
            render_isolated_stems_tflite("realtime", use_finetuned=True,  hop=hop)
        else:
            render_isolated_stems_tflite("realtime", use_finetuned=use_finetuned, hop=hop)
