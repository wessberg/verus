import os
import tensorflow as tf

from .dsp import (
    legacy_minphase_filters,
)

def load_keras_model(path, compile=False):
    """
    tf.keras.models.load_model with a fallback for .keras files saved before the Sept-2026
    min-phase fix (their fixed-FIR weights are half the current length, so a plain load fails on a
    shape mismatch). Such models are loaded with the legacy filter designs — their own trained
    behaviour is preserved exactly; retrain or --migrate-* to get the corrected filters.
    """
    try:
        return tf.keras.models.load_model(path, compile=compile)
    except ValueError as e:
        if "could not be loaded" not in str(e):
            raise
    print(f"[load] {os.path.basename(path)} predates the Sept-2026 min-phase fix (half-length FIR "
          "weights); loading it with the legacy filter designs. Retrain (or --migrate-*) to get "
          "the corrected filters.")
    with legacy_minphase_filters():
        return tf.keras.models.load_model(path, compile=compile)

def _tensor_basename(x):
    nm = getattr(x, "name", str(x))
    nm = nm.split(":")[0]
    if "/" in nm:
        nm = nm.split("/", 1)[0]
    if nm.startswith("serving_default_"):
        nm = nm[len("serving_default_"):]
    return nm

def _producer_layer_name(t):
    # Keras 2.x: t._keras_history.layer; Keras 3: tuple(layer, node_index, tensor_index)
    kh = getattr(t, "_keras_history", None)
    try:
        if isinstance(kh, (tuple, list)) and len(kh) >= 1:
            layer = kh[0]
        else:
            layer = getattr(kh, "layer", None)
        return getattr(layer, "name", None)
    except Exception:
        return None

def _output_names_from_model(model):
    names = []
    for t in model.outputs:
        nm = _producer_layer_name(t)
        if not nm:
            nm = _tensor_basename(t)
        names.append(nm)
    return names

def _shape_tuple(x):
    s = getattr(x, "shape", None)
    if s is None and hasattr(x, "get_shape"):
        s = x.get_shape()
    try:
        return tuple(s.as_list())
    except Exception:
        try:
            return tuple(s)
        except Exception:
            return ()

def _print_model_io_summary(model):
    print("== Inputs ==")
    for t in model.inputs:
        print(f"  {_tensor_basename(t):<56} shape={_shape_tuple(t)}")
    print("== Outputs ==")
    for t in model.outputs:
        base = _tensor_basename(t)
        prod = _producer_layer_name(t)
        if prod:
            print(f"  {base:<32} <- {prod:<48} shape={_shape_tuple(t)}")
        else:
            print(f"  {base:<32} shape={_shape_tuple(t)}")

    in_names  = [_tensor_basename(t) for t in model.inputs]
    out_names = _output_names_from_model(model)
    state_in_names  = [n for n in in_names  if n.startswith("realtime_state_")]
    state_out_names = [n for n in out_names if n.startswith("realtime_state_") and n.endswith("_next")]
    print(f"Streaming states: inputs={len(state_in_names)} outputs={len(state_out_names)}")
    if state_in_names:
        print("  first inputs:", state_in_names[:5])
    if state_out_names:
        print("  first outputs:", state_out_names[:5])
