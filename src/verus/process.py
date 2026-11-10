import numpy as np
import os
import scipy.signal as spsig
import soundfile as sf
import sys

from .config import (
    OFFLINE_SAMPLE_RATE,
    PICKUP_MAGNETIC,
    PICKUP_PIEZO,
    REALTIME_SAMPLE_RATE,
)
from .eq import (
    apply_master_eq_iir,
    master_eq_sos,
)
from .inference import (
    KerasStreamingRunner,
    _get_model_context_for_streaming,
    _style_label,
)
from .knobs import (
    _filter_feed_to_model_inputs,
    prepare_process_knob_feeds,
)

def _cli_arg_value(args, flag, default=None):
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else default
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _open_output_writer(path, sample_rate):
    _ensure_parent_dir(path)
    ext = os.path.splitext(path)[1].lower()
    kwargs = dict(mode="w", samplerate=int(sample_rate), channels=1)
    if ext == ".wav":
        kwargs["subtype"] = "FLOAT"
    return sf.SoundFile(path, **kwargs)

def _load_input_mono(input_path):
    x2d, sr = sf.read(input_path, dtype="float32", always_2d=True)
    if x2d.size == 0:
        raise RuntimeError(f"Input file is empty: {input_path}")
    if x2d.shape[1] > 1:
        print(f"[process] '{input_path}' has {x2d.shape[1]} channels; using channel 0.")
    return x2d[:, 0].astype(np.float32, copy=False), int(sr)


def _normalize_process_style(forced_style):
    """
    Normalize a requested process style to:
      - None  -> auto
      - 1.0   -> pick / picked
      - 0.0   -> finger
    Accepts strings or numeric values.
    """
    if forced_style is None:
        return None

    if isinstance(forced_style, (int, float, np.integer, np.floating)):
        v = float(forced_style)
        if abs(v - 1.0) <= 1e-7:
            return 1.0
        if abs(v - 0.0) <= 1e-7:
            return 0.0
        raise ValueError("forced_style must be None, 0/1, or one of: auto, pick, picked, finger")

    s = str(forced_style).strip().lower()
    if s in ("", "auto", "autodetect", "none", "pred", "predicted", "detect", "classifier"):
        return None
    if s in ("pick", "picked", "p", "1", "true", "yes"):
        return 1.0
    if s in ("finger", "fingerstyle", "f", "0", "false", "no"):
        return 0.0

    raise ValueError(
        f"Invalid style '{forced_style}'. Valid values are: auto, pick, picked, finger"
    )


def _parse_process_style_arg(args_list):
    """
    CLI parser for process style.
    Supports:
      --style <auto|pick|picked|finger>
      --style=<...>
      --force-style <...>
      --force-style=<...>
      --pick / --picked
      --finger
    """
    args = set(args_list)

    raw_style = _cli_arg_value(args_list, "--style")
    raw_force_style = _cli_arg_value(args_list, "--force-style")

    if raw_style is not None and str(raw_style).startswith("--"):
        print("ERROR: --style requires a value: auto, pick, picked, or finger")
        sys.exit(1)

    if raw_force_style is not None and str(raw_force_style).startswith("--"):
        print("ERROR: --force-style requires a value: auto, pick, picked, or finger")
        sys.exit(1)

    short_values = []
    if "--pick" in args or "--picked" in args:
        short_values.append("pick")
    if "--finger" in args:
        short_values.append("finger")

    raw_values = [v for v in (raw_style, raw_force_style) if v is not None] + short_values
    if not raw_values:
        return None

    try:
        normalized = [_normalize_process_style(v) for v in raw_values]
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    uniq = {("auto" if v is None else _style_label(v)) for v in normalized}
    if len(uniq) > 1:
        print("ERROR: conflicting style flags. Use only one of --style/--force-style/--pick/--finger.")
        sys.exit(1)

    return normalized[0]


def _normalize_process_pickup(pickup):
    """
    Normalize a requested pickup type to 0.0 (piezo) or 1.0 (magnetic).
    'electric' is an alias for magnetic (electric guitars use magnetic pickups).
    """
    if pickup is None:
        return PICKUP_PIEZO
    if isinstance(pickup, (int, float, np.integer, np.floating)):
        v = float(pickup)
        if abs(v - PICKUP_PIEZO) <= 1e-7:
            return PICKUP_PIEZO
        if abs(v - PICKUP_MAGNETIC) <= 1e-7:
            return PICKUP_MAGNETIC
        raise ValueError("pickup must be one of: piezo, magnetic, electric (or 0/1)")
    s = str(pickup).strip().lower()
    if s in ("", "piezo", "p", "0"):
        return PICKUP_PIEZO
    if s in ("magnetic", "mag", "m", "electric", "e", "1"):
        return PICKUP_MAGNETIC
    raise ValueError(f"Invalid pickup '{pickup}'. Valid values are: piezo, magnetic, electric")


def _parse_process_pickup_arg(args_list):
    """
    CLI parser for pickup type: --pickup <piezo|magnetic|electric> (or --pickup=<...>).
    """
    raw = _cli_arg_value(args_list, "--pickup")
    if raw is not None and str(raw).startswith("--"):
        print("ERROR: --pickup requires a value: piezo, magnetic, or electric")
        sys.exit(1)
    if raw is None:
        return PICKUP_PIEZO
    try:
        return _normalize_process_pickup(raw)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def _make_process_style_arrays(forced_style):
    """
    Returns:
      style_true_input  shape (1,1)
      style_force_input shape (1,1)
      normalized_style  None|0.0|1.0
    """
    forced_style = _normalize_process_style(forced_style)

    if forced_style is None:
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            None
        )

    return (
        np.array([[forced_style]], dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        forced_style
    )


def _format_process_write_message(label, output_path, sample_rate, avg_prob=None, forced_style=None):
    msg = f"✅ {label} written: {output_path} ({int(sample_rate)} Hz"

    if forced_style is None:
        if avg_prob is not None:
            msg += f", avg style={_style_label(avg_prob)} ({avg_prob:.4f})"
    else:
        msg += f", forced style={_style_label(forced_style)}"
        if avg_prob is not None:
            msg += f", classifier avg={_style_label(avg_prob)} ({avg_prob:.4f})"

    msg += ")"
    return msg

def _resolve_process_output_paths(input_path, output_arg, do_offline, do_realtime):
    if not do_offline and not do_realtime:
        raise ValueError("Nothing selected; choose offline, realtime, or both.")

    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    looks_like_dir = output_arg.endswith(os.sep) or output_arg.endswith("/") or os.path.isdir(output_arg)
    root, ext = os.path.splitext(output_arg)
    out = {}

    if looks_like_dir:
        os.makedirs(output_arg, exist_ok=True)
        if do_offline:
            out["offline"] = os.path.join(output_arg, f"{input_stem}_offline.wav")
        if do_realtime:
            out["realtime"] = os.path.join(output_arg, f"{input_stem}_realtime.wav")
        return out

    if ext:
        if do_offline and do_realtime:
            out["offline"] = root + "_offline" + ext
            out["realtime"] = root + "_realtime" + ext
        elif do_offline:
            out["offline"] = output_arg
        else:
            out["realtime"] = output_arg
    else:
        if do_offline and do_realtime:
            out["offline"] = output_arg + "_offline.wav"
            out["realtime"] = output_arg + "_realtime.wav"
        elif do_offline:
            out["offline"] = output_arg + ".wav"
        else:
            out["realtime"] = output_arg + ".wav"

    for p in out.values():
        _ensure_parent_dir(p)
    return out

def _iter_offline_blocks(model, x_mod, knobs, style_true_arr, style_force_arr, block_sec=12.0):
    """Run the (dynamic) offline model over x_mod in blocks with full left/right context.
    Yields (y_core float32, style_prob) per block."""
    left_ctx, right_ctx = _get_model_context_for_streaming(OFFLINE_SAMPLE_RATE, model=model)
    block_core = max(1, int(round(block_sec * OFFLINE_SAMPLE_RATE)))
    pos, N = 0, len(x_mod)
    while pos < N:
        core = min(block_core, N - pos)
        lctx = min(left_ctx, pos)
        rctx = min(right_ctx, N - (pos + core))
        x_in = x_mod[pos - lctx: pos + core + rctx][None, :, None].astype(np.float32, copy=False)
        feed = {
            "offline_piezo_input": x_in,
            "offline_style_true_input": style_true_arr,
            "offline_style_force_input": style_force_arr,
            **knobs
        }
        outs = model.predict(_filter_feed_to_model_inputs(model, feed), batch_size=1, verbose=0)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        y_pred = np.asarray(outs[0], dtype=np.float32)[0, :, 0]
        style_prob = float(np.asarray(outs[1], dtype=np.float32)[0, 0]) if len(outs) > 1 else 0.0
        yield y_pred[lctx:lctx + core], style_prob
        pos += core

def process_audio_array_offline(model, x_mod, output_path, use_finetuned=False, block_sec=12.0, forced_style=None, pickup=None,
                                eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)
    style_true_arr, style_force_arr, _ = _make_process_style_arrays(forced_style)

    knobs = prepare_process_knob_feeds("offline", use_finetuned=use_finetuned,
                                       pickup=_normalize_process_pickup(pickup))
    writer = _open_output_writer(output_path, OFFLINE_SAMPLE_RATE)
    eq_state, eq_sos = None, None
    prob_sum = 0.0
    prob_count = 0

    for y_core, style_prob in _iter_offline_blocks(model, x_mod, knobs, style_true_arr, style_force_arr, block_sec):
        core = len(y_core)
        if eq_knobs:
            # Stateful IIR across blocks so block seams stay continuous
            if eq_state is None:
                eq_sos = np.asarray(master_eq_sos(OFFLINE_SAMPLE_RATE, **eq_knobs), dtype=np.float64)
                eq_state = spsig.sosfilt_zi(eq_sos) * 0.0 if len(eq_sos) else None
            if eq_state is not None:
                y_core, eq_state = spsig.sosfilt(eq_sos, y_core.astype(np.float64), zi=eq_state)
                y_core = y_core.astype(np.float32)
        writer.write(y_core)
        prob_sum += style_prob * float(core)
        prob_count += core

    writer.close()
    return (prob_sum / float(prob_count)) if prob_count > 0 else None

def process_audio_array_realtime(model, x_mod, output_path, use_finetuned=False, forced_style=None, pickup=None,
                                 eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)

    knobs = prepare_process_knob_feeds("realtime", use_finetuned=use_finetuned,
                                       pickup=_normalize_process_pickup(pickup))
    runner = KerasStreamingRunner(model)  # hop = T for static graphs, KERAS_STREAM_HOP for dynamic
    res = runner.run(x_mod, knobs, style_true=forced_style)
    y = res["audio"]
    if eq_knobs:
        y = apply_master_eq_iir(y, REALTIME_SAMPLE_RATE, **eq_knobs)
    writer = _open_output_writer(output_path, REALTIME_SAMPLE_RATE)
    writer.write(y)
    writer.close()
    if forced_style is not None:
        # Classifier average for reference (a cheap second pass is not worth it; report n/a)
        return None
    return res["avg_prob"]
