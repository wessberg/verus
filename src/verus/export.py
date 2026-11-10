import numpy as np
import os
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Activation, Input)  # pyright: ignore[reportMissingImports]

from .config import (
    FINETUNE_OFFLINE_STATIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_TFLITE_FP32_FILENAME,
    FINETUNE_REALTIME_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_TFLITE_FP32_FILENAME,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_STATIC_MODEL_FILENAME,
    OFFLINE_TFLITE_FP32_FILENAME,
    OFFLINE_TFLITE_T,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_FP32_FILENAME,
    REALTIME_TFLITE_T,
)
from .modelio import (
    load_keras_model,
)
from .models import (
    build_offline_model,
    build_realtime_streaming_model,
    copy_weights_by_name,
)

def _wrap_fixed_T_for_tflite(dyn_model, kind, T_fixed):
    """
    Wrap a dynamic-time runtime graph for fixed-T TFLite export.
    offline → outputs: [audio, style, dry]
    realtime → done by build_realtime_streaming_export_for_tflite
    """
    if kind == "offline":
        inp  = Input(shape=(T_fixed,1), name="offline_piezo_input")
        st_t = Input(shape=(1,),       name="offline_style_true_input")
        st_f = Input(shape=(1,),       name="offline_style_force_input")
        pkup = Input(shape=(1,),       name="offline_pickup_input")
        k_main = Input(shape=(1,),     name="offline_knob_main")
        k_pa = Input(shape=(1,),       name="offline_knob_pickattack")
        k_pr = Input(shape=(1,),       name="offline_knob_pickreso")
        k_dw = Input(shape=(1,),       name="offline_knob_drywet")
        k_og  = Input(shape=(1,), name="offline_knob_outgain")
        k_pi  = Input(shape=(1,), name="offline_knob_phaseinv")

        outs = dyn_model([inp, st_t, st_f, pkup, k_main, k_pa, k_pr, k_dw, k_og, k_pi])
        if not isinstance(outs, (list, tuple)) or len(outs) < 2:
            raise RuntimeError("Unexpected outputs from offline dynamic model during TFLite wrapping.")
        # Export [audio, style] only. The old "dry" output was a copy of the input (the models
        # are zero-latency) and, being a second [1,T,1] float tensor, made shape-based output
        # discovery in the runtimes ambiguous.
        y = Activation('linear', name="offline_audio_out")(outs[0])
        s = Activation('linear', name="offline_style_prob_out")(outs[1])

        export = tf.keras.Model(
            inputs=[inp, st_t, st_f, pkup, k_main, k_pa, k_pr, k_dw, k_og, k_pi],
            outputs=[y, s],
            name=dyn_model.name + "_tflite_export"
        )
        return export

    elif kind == "realtime":
        raise RuntimeError("_wrap_fixed_T_for_tflite(kind='realtime') is not used; "
                           "use build_realtime_streaming_export_for_tflite instead.")
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

@register_keras_serializable()
def build_realtime_streaming_export_for_tflite(rt_stream_model, strict=True):
    import re
    import numpy as np
    import tensorflow as tf

    def norm_tensor_name(t):
        nm = t.name.split(':')[0]
        if '/' in nm:
            nm = nm.split('/', 1)[0]
        if nm.startswith("serving_default_"):
            nm = nm[len("serving_default_"):]
        return nm

    def producer_layer_name(t):
        kh = getattr(t, "_keras_history", None)
        try:
            layer = kh[0] if isinstance(kh, (tuple, list)) else getattr(kh, "layer", None)
            return getattr(layer, "name", None)
        except Exception:
            return None

    # audio/style from model outputs. The old "dry" export output is intentionally dropped: it
    # was a copy of the input and, as a second [1,T,1] float tensor, made the shape-based output
    # discovery in TFLiteRunner / the C++ pedal ambiguous (audio vs dry only differed by name).
    if len(rt_stream_model.outputs) < 1:
        raise ValueError("[export/tflite] realtime streaming model has no outputs.")
    audio = rt_stream_model.outputs[0]
    style = None
    if len(rt_stream_model.outputs) >= 2:
        style = rt_stream_model.outputs[1]

    # If style missing, synthesize a zero scalar
    if style is None:
        style = tf.keras.layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], 1), tf.float32),
            name="realtime_style_prob")(audio)

    inputs = list(rt_stream_model.inputs)

    # Discover streaming state inputs in lex order, and lengths
    state_in_meta = []
    for t in inputs:
        nm = norm_tensor_name(t)
        if nm.startswith("realtime_state_"):
            L = int(t.shape[1]) if t.shape[1] is not None else 0
            C = int(t.shape[2]) if t.shape[2] is not None else 1
            state_in_meta.append((nm, L * C))
    state_in_meta.sort(key=lambda p: p[0])

    outs = list(rt_stream_model.outputs)
    out_by_prod = {}
    for t in outs:
        pn = producer_layer_name(t)
        if pn:
            out_by_prod[pn] = t
    out_by_name = {norm_tensor_name(t): t for t in outs}

    flat_list = []
    declared_lengths = []
    missing = []

    for (nm, expected_len) in state_in_meta:
        nm_next = nm + "_next"

        t = out_by_prod.get(nm_next, None)
        if t is None:
            t = out_by_name.get(nm_next, None)
        if t is None:
            # Try removing numeric suffixes
            base_map_prod = {re.sub(r"_\d+$", "", k): v for (k, v) in out_by_prod.items()}
            t = base_map_prod.get(nm_next, None)
            if t is None:
                base_map_name = {re.sub(r"_\d+$", "", k): v for (k, v) in out_by_name.items()}
                t = base_map_name.get(nm_next, None)

        exp_len = int(expected_len)

        if t is None:
            missing.append(nm_next)
            if strict:
                continue
            # Synthesize zeros (B,exp_len) tied to batch of 'audio'
            z = tf.keras.layers.Lambda(
                lambda a, e=exp_len: tf.tile(tf.zeros((1, e), dtype=tf.float32),
                                             [tf.shape(a)[0], 1]),
                name=f"{nm}_zeros")(audio)
            flat_list.append(z)
            declared_lengths.append(exp_len)
            continue

        flat = tf.keras.layers.Flatten(name=f"{nm}_flatten")(t)

        got_len = flat.shape[1]
        if got_len is None:
            Ls = t.shape[1] or 0
            Cs = t.shape[2] or 1
            got_len = int(Ls) * int(Cs)
        got_len = int(got_len)

        delta = exp_len - got_len
        if delta > 0:
            flat_fixed = tf.keras.layers.Lambda(
                lambda f, d=delta: tf.concat(
                    [tf.zeros((tf.shape(f)[0], d), dtype=f.dtype), f], axis=1),
                name=f"{nm}_padleft")(flat)
        elif delta < 0:
            flat_fixed = tf.keras.layers.Lambda(
                lambda f, e=exp_len: f[:, -e:],
                name=f"{nm}_tailcrop")(flat)
        else:
            flat_fixed = flat

        flat_list.append(flat_fixed)
        declared_lengths.append(exp_len)

    if strict and missing:
        miss_str = "\n  - ".join(missing)
        raise ValueError(f"[export/tflite] Missing '*_next' outputs for {len(missing)} state(s):\n  - {miss_str}\n"
                         "Export cannot proceed safely.")

    # Pack bundle/splits
    if flat_list:
        bundle = tf.keras.layers.Concatenate(axis=1, name="realtime_states_bundle_out")(flat_list)
    else:
        bundle = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 1), tf.float32),
                                        name="realtime_states_bundle_out")(audio)

    lens_np = np.array(declared_lengths, dtype=np.int32).reshape(1, -1)
    splits = tf.keras.layers.Lambda(lambda _: tf.constant(lens_np, dtype=tf.int32),
                                    name="realtime_states_splits_out")(audio)

    # Final named outputs
    y_out = tf.keras.layers.Activation('linear', name="realtime_audio_out")(audio)
    s_out = tf.keras.layers.Activation('linear', name="realtime_style_prob_out")(style)

    export = tf.keras.models.Model(
        inputs=inputs,
        outputs=[y_out, s_out, bundle, splits],
        name=rt_stream_model.name + "_tflite_export"
    )

    try:
        total = int(np.sum(lens_np))
        print("[export/tflite] Streaming realtime export wrapper")
        print(f"  states discovered: {len(state_in_meta)} (lex order by input name)")
        for (nm, exp_len) in state_in_meta[:8]:
            flag = "" if (nm + "_next") not in missing else " (MISSING_NEXT)"
            print(f"    - {nm:<40} len={exp_len:<6d}{flag}")
        if len(state_in_meta) > 8:
            print(f"    ... {len(state_in_meta)-8} more")
        print("[export/tflite] outputs:")
        for t_ in export.outputs:
            print("  -", t_.name.split(':')[0], "shape=", t_.shape)
        print(f"[export/tflite] sum(splits)={total}")
        if missing and not strict:
            print(f"[export/tflite] WARNING: {len(missing)} state(s) had MISSING_NEXT and were zero-padded in bundle.")
    except Exception:
        pass

    return export

def _inspect_tflite_file(path):
    try:
        intr = Interpreter(model_path=path)
        intr.allocate_tensors()
        outs = intr.get_output_details()
        ins  = intr.get_input_details()
        out_list = [(str(d["name"]), tuple(int(x) for x in d["shape"])) for d in outs]
        in_list  = [(str(d["name"]), tuple(int(x) for x in d["shape"])) for d in ins]
        print("[TFLite/inspect] Inputs:")
        for n, s in in_list:
            print("  ", n, s)
        print("[TFLite/inspect] Outputs:")
        for n, s in out_list:
            print("  ", n, s)
        return in_list, out_list
    except Exception as e:
        print(f"[TFLite/inspect] Could not inspect TFLite file: {e}")
        return [], []


def _validate_tflite_outputs_by_shape(kind, outs, T_fixed=None):
    """
    offline : exactly one 3D float output (audio) + a (1,1) style output
    realtime: exactly one 3D float output (audio), (1,1) style, 2D bundle, int splits
    Exactly ONE 3D output is a hard requirement: the runtimes pick the audio tensor by shape.
    """
    shapes = [s for (_, s) in outs]
    three_d = [s for s in shapes if len(s) == 3]
    two_d   = [s for s in shapes if len(s) == 2]
    if len(three_d) != 1:
        raise RuntimeError(f"[TFLite/export] {kind.upper()} expected exactly ONE 3D (audio) output, got {len(three_d)}: {outs}")
    if (1, 1) not in two_d:
        raise RuntimeError(f"[TFLite/export] {kind.upper()} expected a (1,1) style output; got: {outs}")
    if kind == "realtime":
        if len(shapes) < 4:
            raise RuntimeError(f"[TFLite/export] REALTIME expected ≥4 outputs (audio, style, bundle, splits) but found {len(shapes)}: {outs}")
        if T_fixed is not None and three_d[0][1] != int(T_fixed):
            print(f"[TFLite/export] WARNING: REALTIME audio T={three_d[0][1]} != fixed T={T_fixed}. Continuing.")
    return True

def verify_tflite_bundle_file(path, strict=True):
    """
    Inspect a realtime TFLite and verify:
      - 4 outputs: audio (3D), style (1,1), bundle (2D float), splits (2D int)
      - sum(splits) == bundle length
      - sum(L*C of 'realtime_state_*' inputs) == bundle length (optional; exporter may omit states with len=0)
    Chooses bundle/splits by SHAPE and DTYPE (robust to generic names and duplicate 'StatefulPartitionedCall' bases).
    """
    try:
        intr = Interpreter(model_path=path)
        intr.allocate_tensors()
    except Exception as e:
        raise RuntimeError(f"[verify] Failed to open {path}: {e}")

    ins  = intr.get_input_details()
    outs = intr.get_output_details()

    # Streaming state inputs
    state_inputs = []
    for d in ins:
        nm = str(d["name"])
        if b"realtime_state_" in d["name"] if isinstance(d["name"], bytes) else "realtime_state_" in nm:
            state_inputs.append(d)
    exp_sum = 0
    for d in state_inputs:
        s = tuple(int(x) for x in d["shape"])
        L = s[1] if len(s) > 1 else 0
        C = s[2] if len(s) > 2 else 1
        exp_sum += (L * C)

    # Pick outputs by shape/dtype
    def is_int(arr_dtype):
        return np.issubdtype(arr_dtype, np.integer)

    # Run once so get_tensor works for all consts
    intr.invoke()

    audio_detail = None
    bundle_detail = None
    splits_detail = None

    # Audio = the only 3D output (an export with two 3D outputs is ambiguous for the runtimes)
    three_d = [d for d in outs if len(tuple(int(x) for x in d["shape"])) == 3]
    if len(three_d) != 1:
        msg = f"[verify] expected exactly one 3D (audio) output, found {len(three_d)} — re-run --export-tflite"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
    audio_detail = three_d[0] if three_d else None

    # Bundle = largest 2D float output (excluding (1,1))
    best_total = -1
    for d in outs:
        shp = tuple(int(x) for x in d["shape"])
        if len(shp) == 2 and shp != (1,1):
            arr = intr.get_tensor(d["index"])
            if arr.dtype == np.float32:
                total = int(np.prod(shp))
                if total > best_total:
                    best_total = total
                    bundle_detail = d

    # Splits = (1,N) or (N,) integer vector where N == #states
    n_states = len(state_inputs)
    for d in outs:
        shp = tuple(int(x) for x in d["shape"])
        if len(shp) == 2 and shp[0] == 1 and shp[1] == n_states:
            arr = intr.get_tensor(d["index"])
            if is_int(arr.dtype):
                splits_detail = d
                break
    if splits_detail is None:
        for d in outs:
            shp = tuple(int(x) for x in d["shape"])
            if (len(shp) == 1 and shp[0] == n_states) or (len(shp) == 2 and shp == (n_states,1)):
                arr = intr.get_tensor(d["index"])
                if is_int(arr.dtype):
                    splits_detail = d
                    break

    # Print a summary
    print("[verify] TFLite:", path)
    print(f"[verify] inputs(states): {n_states} expected bundle len: {exp_sum}")
    for d in state_inputs[:8]:
        print(f"  in: {d['name']} {tuple(int(x) for x in d['shape'])}")
    if len(state_inputs) > 8:
        print(f"  ... {len(state_inputs)-8} more")

    print("[verify] outputs:")
    for d in outs:
        print(f"  out: {d['name']} {tuple(int(x) for x in d['shape'])}")

    if bundle_detail is None or splits_detail is None:
        msg = "[verify] Missing bundle or splits (by shape). Exporter likely failed to pack states."
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
        return

    b = intr.get_tensor(bundle_detail["index"])
    if b.ndim == 2:
        bundle_len = int(np.prod(b.shape[1:]))
    elif b.ndim == 1:
        bundle_len = int(b.shape[0])
    else:
        bundle_len = int(b.size)
    print(f"[verify] bundle len (tensor): {bundle_len}")

    s = intr.get_tensor(splits_detail["index"])
    s = s.reshape(1, -1).astype(np.int32, copy=False)
    s_sum = int(np.sum(s))
    s_cnt = int(s.shape[1])
    print(f"[verify] splits count: {s_cnt}, sum(splits): {s_sum}")

    if s_cnt != n_states:
        msg = f"[verify] splits count ({s_cnt}) != #state inputs ({n_states})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)

    if s_sum != bundle_len:
        msg = f"[verify] sum(splits) ({s_sum}) != bundle len ({bundle_len})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)

    if exp_sum != bundle_len:
        msg = f"[verify] sum(L*C of inputs) ({exp_sum}) != bundle len ({bundle_len})"
        if strict: raise RuntimeError(msg)
        print("WARN:", msg)
    else:
        print("[verify] OK: bundle matches streaming state sizes.")

def _convert_keras_to_tflite_fp32_checked(model, out_path, validate_kind=None, T_fixed=None):
    """
    Convert Keras model to FP32 TFLite, write to out_path, then inspect and validate by SHAPES.
    validate_kind: 'offline' or 'realtime' or None (skip validation).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflm = converter.convert()
    except Exception as e:
        if validate_kind == "realtime":
            # The pedal links the builtin op resolver only (no Flex delegate): a SELECT_TF_OPS
            # model would load-fail on the Pi, so do not silently produce one.
            raise RuntimeError(f"[TFLite] Realtime export needs non-builtin ops ({e}). "
                               "The C++ pedal cannot run SELECT_TF_OPS models; fix the graph instead.") from e
        print(f"[TFLite] WARNING: builtin conversion failed ({e}); retrying with SELECT_TF_OPS. "
              "This file needs the Flex delegate (Python/ai_edge_litert only).")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflm = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflm)
    print(f"✅ Saved FP32 TFLite: {out_path}")

    if validate_kind:
        _, outs = _inspect_tflite_file(out_path)
        _validate_tflite_outputs_by_shape(validate_kind, outs, T_fixed=T_fixed)

def _build_dyn_model_for_export(kind):
    """
    Build the dynamic-time runtime graph (teacherless) for export.
    Realtime no longer uses this (we export the streaming/stateful graph directly).
    """
    if kind == "offline":
        dyn = build_offline_model(dynamic_time=True)
        for l in dyn.layers:
            l.trainable = False
        return dyn
    elif kind == "realtime":
        raise RuntimeError("_build_dyn_model_for_export is not used for realtime. "
                           "Use build_realtime_streaming_model with a fixed T instead.")
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

def _export_one_tflite(kind, static_src_path, out_tflite_path, T_fixed):
    """
    Export a single FP32 TFLite for a given 'kind' from a static Keras model.
    - offline: wrap fixed‑T and validate [3D audio, 2D style]
    - realtime: wrap streaming and validate [3D audio, 2D style, 2D bundle]
    """
    if not os.path.exists(static_src_path):
        print(f"[TFLite/export] Missing source model: {static_src_path}")
        return False

    if kind == "offline":
        print(f"[TFLite/export] Loading offline static: {static_src_path}")
        static_m = load_keras_model(static_src_path, compile=False)

        dyn = _build_dyn_model_for_export(kind="offline")
        copy_weights_by_name(dyn, static_m, verbose=True)

        export = _wrap_fixed_T_for_tflite(dyn, kind="offline", T_fixed=T_fixed)
        _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="offline", T_fixed=T_fixed)
        return True

    # kind == "realtime"
    print(f"[TFLite/export] Loading realtime streaming static: {static_src_path}")
    rt_static = load_keras_model(static_src_path, compile=False)

    T_fixed = int(T_fixed)
    t_in = None
    try:
        t_in = int(rt_static.inputs[0].shape[1]) if rt_static.inputs else None
    except Exception:
        t_in = None
    if (t_in is not None) and (t_in not in (T_fixed,)):
        rt_fixedT = build_realtime_streaming_model(static_T=T_fixed)
        copy_weights_by_name(rt_fixedT, rt_static, verbose=True)
        rt_static = rt_fixedT

    export = build_realtime_streaming_export_for_tflite(rt_static)

    # Write to a distinct filename to avoid confusion with older files
    base, ext = os.path.splitext(out_tflite_path)
    _convert_keras_to_tflite_fp32_checked(export, out_tflite_path, validate_kind="realtime", T_fixed=T_fixed)
    print(f"✅ Saved realtime TFLite: {out_tflite_path}")
    return True

def export_tflite_all():
    # OFFLINE
    offline_static_src = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
    offline_tflite_out = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_TFLITE_FP32_FILENAME)
    ok_off = _export_one_tflite(kind="offline",
                       static_src_path=offline_static_src,
                       out_tflite_path=offline_tflite_out,
                       T_fixed=OFFLINE_TFLITE_T)

    offline_ft_src = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
    offline_ft_out = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_TFLITE_FP32_FILENAME)
    ok_off_ft = False
    if os.path.exists(offline_ft_src):
        ok_off_ft = _export_one_tflite(kind="offline",
                           static_src_path=offline_ft_src,
                           out_tflite_path=offline_ft_out,
                           T_fixed=OFFLINE_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned OFFLINE (not found): {offline_ft_src}")

    # REALTIME baseline
    rt_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    rt_tflite_out = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TFLITE_FP32_FILENAME)
    ok_rt = _export_one_tflite(kind="realtime",
                       static_src_path=rt_static_src,
                       out_tflite_path=rt_tflite_out,
                       T_fixed=REALTIME_TFLITE_T)

    # REALTIME finetuned
    rt_ft_static_src = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    rt_ft_out = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TFLITE_FP32_FILENAME)
    ok_rt_ft = False
    if os.path.exists(rt_ft_static_src):
        ok_rt_ft = _export_one_tflite(kind="realtime",
                           static_src_path=rt_ft_static_src,
                           out_tflite_path=rt_ft_out,
                           T_fixed=REALTIME_TFLITE_T)
    else:
        print(f"[TFLite/export] Skipping finetuned REALTIME (not found): {rt_ft_static_src}")

    print("\n[TFLite/export] Summary:")
    print(f"  offline baseline exported: {ok_off}")
    print(f"  offline finetuned exported: {ok_off_ft}")
    print(f"  realtime baseline exported: {ok_rt} -> {rt_tflite_out if ok_rt else '(old file may still exist!)'}")
    print(f"  realtime finetuned exported: {ok_rt_ft} -> {rt_ft_out if ok_rt_ft else '(old file may still exist!)'}")
