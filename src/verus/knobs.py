import numpy as np
import tensorflow as tf

from .config import (
    FINETUNE_OFFLINE_PICK_ATTACK_DEMO_BOOST_DB,
    FINETUNE_OFFLINE_PICK_RESO_DEMO_BOOST_DB,
    FINETUNE_REALTIME_IR_DEMO_BOOST_DB,
    FINETUNE_REALTIME_PICK_ATTACK_DEMO_BOOST_DB,
    FINETUNE_REALTIME_PICK_RESO_DEMO_BOOST_DB,
    KNOB_DEFAULT_DRY_PCT,
    KNOB_DEFAULT_IR_SCALE,
    KNOB_DEFAULT_MAIN_SCALE,
    KNOB_DEFAULT_OUT_GAIN,
    KNOB_DEFAULT_PHASE_INVERT,
    KNOB_DEFAULT_PICK_ATTACK_SCALE,
    KNOB_DEFAULT_PICK_RESO_SCALE,
    OFFLINE_PICK_ATTACK_DEMO_BOOST_DB,
    OFFLINE_PICK_ATTACK_RUNTIME_BASELINE_DB,
    OFFLINE_PICK_RESO_DEMO_BOOST_DB,
    OFFLINE_PICK_RESO_RUNTIME_BASELINE_DB,
    PICKUP_PIEZO,
    REALTIME_IR_DEMO_BOOST_DB,
    REALTIME_PICK_ATTACK_DEMO_BOOST_DB,
    REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB,
    REALTIME_PICK_RESO_DEMO_BOOST_DB,
    REALTIME_PICK_RESO_RUNTIME_BASELINE_DB,
)
from .dsp import (
    _db_to_lin,
)
from .eq import (
    master_eq_defaults,
)
from .modelio import (
    _tensor_basename,
)

def _common_knob_defaults():
    # In-graph knobs only. The master EQ is a runtime effect (see master_eq_defaults()).
    return {
        "dry_pct": KNOB_DEFAULT_DRY_PCT,
        "outgain": KNOB_DEFAULT_OUT_GAIN,
        "phaseinv": KNOB_DEFAULT_PHASE_INVERT,
        "main": KNOB_DEFAULT_MAIN_SCALE,
        "pickattack": KNOB_DEFAULT_PICK_ATTACK_SCALE,
        "pickreso": KNOB_DEFAULT_PICK_RESO_SCALE
    }

def realtime_knob_defaults():
    d = _common_knob_defaults()
    return {
        **d,
        "ir": KNOB_DEFAULT_IR_SCALE
    }

def offline_knob_defaults():
    d = _common_knob_defaults()
    return {
        **d
    }

def _resolve_demo_knobs(kind, use_finetuned=False):
    # Return (pickattack_lin, pickreso_lin, ir_lin)
    if kind == "offline":
        defaults = offline_knob_defaults()
        if use_finetuned:
            return {
                **defaults,
                "pickattack": _db_to_lin(FINETUNE_OFFLINE_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(FINETUNE_OFFLINE_PICK_RESO_DEMO_BOOST_DB)
            }
        else:
            return {
                **defaults,
                "pickattack": _db_to_lin(OFFLINE_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(OFFLINE_PICK_RESO_DEMO_BOOST_DB)
            }
    else:
        defaults = realtime_knob_defaults()
        if use_finetuned:
            return {
                **defaults,
                "pickattack": _db_to_lin(FINETUNE_REALTIME_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(FINETUNE_REALTIME_PICK_RESO_DEMO_BOOST_DB),
                "ir": _db_to_lin(FINETUNE_REALTIME_IR_DEMO_BOOST_DB)
            }
        else:
            return {
                **defaults,
                "pickattack": _db_to_lin(REALTIME_PICK_ATTACK_DEMO_BOOST_DB),
                "pickreso": _db_to_lin(REALTIME_PICK_RESO_DEMO_BOOST_DB),
                "ir": _db_to_lin(REALTIME_IR_DEMO_BOOST_DB)
            }

def prepare_training_knob_feeds(kind, piezo_input, style_true, style_force, pickup=None):
    b = tf.shape(piezo_input)[0]
    if pickup is None:
        pickup = tf.zeros((b, 1), tf.float32)

    if kind == "offline":
        knobs = offline_knob_defaults()
        return {
            'offline_piezo_input': piezo_input,
            'offline_style_true_input': style_true,
            'offline_style_force_input': style_force,
            'offline_pickup_input': pickup,
            'offline_knob_main':       tf.ones((b,1), tf.float32) * knobs["main"],
            'offline_knob_pickattack': tf.ones((b,1), tf.float32) * knobs["pickattack"],
            'offline_knob_pickreso':   tf.ones((b,1), tf.float32) * knobs["pickreso"],
            'offline_knob_drywet':     tf.ones((b,1), tf.float32) * knobs["dry_pct"],
            'offline_knob_outgain':    tf.ones((b,1), tf.float32) * knobs["outgain"],
            'offline_knob_phaseinv':   tf.ones((b,1), tf.float32) * knobs["phaseinv"],
        }
    else:
        knobs = realtime_knob_defaults()
        return {
            'realtime_piezo_input': piezo_input,
            'realtime_style_true_input': style_true,
            'realtime_style_force_input': style_force,
            'realtime_pickup_input': pickup,
            'realtime_knob_main':       tf.ones((b,1), tf.float32) * knobs["main"],
            'realtime_knob_pickattack': tf.ones((b,1), tf.float32) * knobs["pickattack"],
            'realtime_knob_pickreso':   tf.ones((b,1), tf.float32) * knobs["pickreso"],
            'realtime_knob_ir':         tf.ones((b,1), tf.float32) * knobs["ir"],
            'realtime_knob_drywet':     tf.ones((b,1), tf.float32) * knobs["dry_pct"],
            'realtime_knob_outgain':    tf.ones((b,1), tf.float32) * knobs["outgain"],
            'realtime_knob_phaseinv':   tf.ones((b,1), tf.float32) * knobs["phaseinv"],
        }

def _legacy_master_eq_knob_feeds(kind):
    """
    Neutral values for the in-graph master-EQ knob inputs that models saved before Sept 2026 still
    have. Every consumer filters feeds down to the model's actual inputs, so current (EQ-free)
    models simply ignore these keys, while legacy models get a transparent EQ.
    """
    d = master_eq_defaults()
    m = {"lc_hz": "lc_hz", "lc_slope": "lc_slope_dbpoct", "ls_gain_db": "ls_gain_db", "ls_hz": "ls_hz",
         "ls_slope": "ls_slope_dbpoct", "mid_gain_db": "mid_gain_db", "mid_hz": "mid_hz", "mid_q": "mid_q",
         "hs_gain_db": "hs_gain_db", "hs_hz": "hs_hz", "hs_slope": "hs_slope_dbpoct"}
    return {f"{kind}_knob_{suffix}": np.array([[float(d[k])]], dtype=np.float32) for k, suffix in m.items()}

def prepare_demo_knob_feeds(kind, use_finetuned=False, pickattack=None, pickreso=None, ir=None, main=None, pickup=None):
    knobs = _resolve_demo_knobs(kind, use_finetuned=use_finetuned)
    pickup_val = PICKUP_PIEZO if pickup is None else float(pickup)

    feeds = dict(_legacy_master_eq_knob_feeds(kind))
    if kind == "realtime":
        feeds["realtime_pickup_input"]    = np.array([[pickup_val]], dtype=np.float32)
        feeds["realtime_knob_main"]       = np.array([[main if main is not None else knobs["main"]]], dtype=np.float32)
        feeds["realtime_knob_pickattack"] = np.array([[pickattack if pickattack is not None else knobs["pickattack"]]], dtype=np.float32)
        feeds["realtime_knob_pickreso"]   = np.array([[pickreso if pickreso is not None else knobs["pickreso"]]], dtype=np.float32)
        feeds["realtime_knob_ir"]         = np.array([[ir if ir is not None else knobs["ir"]]], dtype=np.float32)
        feeds["realtime_knob_drywet"]     = np.array([[knobs["dry_pct"]]], dtype=np.float32)
        feeds["realtime_knob_outgain"]    = np.array([[knobs["outgain"]]], dtype=np.float32)
        feeds["realtime_knob_phaseinv"]   = np.array([[knobs["phaseinv"]]], dtype=np.float32)
    elif kind == "offline":
        feeds["offline_pickup_input"]     = np.array([[pickup_val]], dtype=np.float32)
        feeds["offline_knob_main"]        = np.array([[main if main is not None else knobs["main"]]], dtype=np.float32)
        feeds["offline_knob_pickattack"]  = np.array([[pickattack if pickattack is not None else knobs["pickattack"]]], dtype=np.float32)
        feeds["offline_knob_pickreso"]    = np.array([[pickreso if pickreso is not None else knobs["pickreso"]]], dtype=np.float32)
        feeds["offline_knob_drywet"]      = np.array([[knobs["dry_pct"]]], dtype=np.float32)
        feeds["offline_knob_outgain"]     = np.array([[knobs["outgain"]]], dtype=np.float32)
        feeds["offline_knob_phaseinv"]    = np.array([[knobs["phaseinv"]]], dtype=np.float32)
    return feeds

def _filter_feed_to_model_inputs(model, feed):
    in_names = [_tensor_basename(t) for t in model.inputs]
    return {k: v for (k, v) in feed.items() if k in in_names}

def prepare_process_knob_feeds(kind, use_finetuned=False, pickup=None):
    # --process uses the same runtime baseline make-up gains as the demos and the C++ pedal
    # (see *_RUNTIME_BASELINE_DB), so a file rendered here matches the pedal at 0 dB knobs.
    if kind == "realtime":
        return prepare_demo_knob_feeds(
            kind="realtime",
            use_finetuned=use_finetuned,
            main=KNOB_DEFAULT_MAIN_SCALE,
            pickattack=_db_to_lin(REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB),
            pickreso=_db_to_lin(REALTIME_PICK_RESO_RUNTIME_BASELINE_DB),
            ir=KNOB_DEFAULT_IR_SCALE,
            pickup=pickup
        )
    return prepare_demo_knob_feeds(
        kind="offline",
        use_finetuned=use_finetuned,
        main=KNOB_DEFAULT_MAIN_SCALE,
        pickattack=_db_to_lin(OFFLINE_PICK_ATTACK_RUNTIME_BASELINE_DB),
        pickreso=_db_to_lin(OFFLINE_PICK_RESO_RUNTIME_BASELINE_DB),
        pickup=pickup
    )
