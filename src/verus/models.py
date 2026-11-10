import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import constraints  # pyright: ignore[reportMissingImports]
from tensorflow.keras import regularizers  # pyright: ignore[reportMissingImports]
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Dense, GlobalAveragePooling1D, Input, Lambda, Layer, Multiply, Reshape, SeparableConv1D, ZeroPadding1D)  # pyright: ignore[reportMissingImports]

from .config import (
    COLOREQ_BANDS_OFFLINE_BODY,
    COLOREQ_BANDS_REALTIME_BODY,
    HP_PRE_EMPHASIS_HF,
    KD_TEACHER_LEFT_CTX_SAMPLES,
    KD_TEACHER_RIGHT_CTX_SAMPLES,
    MODELS_OFFLINE_OUT_DIR,
    OFFLINE_COLOREQ_ORDER,
    OFFLINE_DILATIONS,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_EMBED_DIM,
    OFFLINE_FILM_BETA_MAG,
    OFFLINE_FILM_BETA_SCALE,
    OFFLINE_FILM_GAMMA_MAG,
    OFFLINE_FILM_GAMMA_SCALE,
    OFFLINE_FILTERS,
    OFFLINE_HF_HP_ORDER,
    OFFLINE_KERNEL_SIZE,
    OFFLINE_LOOKAHEAD_SAMPLES,
    OFFLINE_LOSS_CORE,
    OFFLINE_PICK_ATTACK_CUTOFF_HZ,
    OFFLINE_PICK_ATTACK_SOFTCLIP_DRIVE,
    OFFLINE_PICK_MASK_ENV_FAST_MS,
    OFFLINE_PICK_MASK_ENV_SLOW_MS,
    OFFLINE_PICK_MASK_FLOOR,
    OFFLINE_PICK_MASK_KNEE_DB,
    OFFLINE_PICK_MASK_OUT_GAIN,
    OFFLINE_PICK_MASK_POST_COMP_GAIN,
    OFFLINE_PICK_MASK_POST_COMP_P,
    OFFLINE_PICK_MASK_SMOOTH_ATTACK_MS,
    OFFLINE_PICK_MASK_SMOOTH_RELEASE_MS,
    OFFLINE_PICK_MASK_THRESHOLD_DB,
    OFFLINE_PRE_BLOCKS,
    OFFLINE_PRE_KERNEL,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_SEG_LEN,
    PICKUP_ADAPTER_DILATIONS,
    PICKUP_ADAPTER_FILTERS,
    PICKUP_ADAPTER_KERNEL,
    PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
    PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
    PICK_ATTACK_LOWBAND_PENALTY_ORDER,
    PICK_ATTACK_STYLE_SMOOTH_MS,
    PICK_LIMITER_DRIVE,
    PICK_LIMITER_HEADROOM_DB,
    PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
    PICK_RESO_LOWBAND_PENALTY_ORDER,
    PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
    PICK_RESO_MODAL_LEN_REALTIME,
    REALTIME_AUDIO_STFT_DISTILL_ALPHA,
    REALTIME_COLOREQ_ORDER,
    REALTIME_DILATIONS,
    REALTIME_DISTILL_ALPHA,
    REALTIME_EMBED_DIM,
    REALTIME_FEAT_DISTILL_ALPHA,
    REALTIME_FILM_BETA_MAG,
    REALTIME_FILM_BETA_SCALE,
    REALTIME_FILM_GAMMA_MAG,
    REALTIME_FILM_GAMMA_SCALE,
    REALTIME_FILTERS,
    REALTIME_HF_HP_ORDER,
    REALTIME_IR_DILATIONS,
    REALTIME_IR_DOWNSAMPLE_FACTOR,
    REALTIME_IR_FIR_SIZE,
    REALTIME_IR_KERNEL,
    REALTIME_KERNEL_SIZE,
    REALTIME_LOSS_CORE,
    REALTIME_LP_ORDER,
    REALTIME_PICK_ATTACK_CUTOFF_HZ,
    REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE,
    REALTIME_PICK_MASK_ENV_FAST_MS,
    REALTIME_PICK_MASK_ENV_SLOW_MS,
    REALTIME_PICK_MASK_FLOOR,
    REALTIME_PICK_MASK_KNEE_DB,
    REALTIME_PICK_MASK_OUT_GAIN,
    REALTIME_PICK_MASK_POST_COMP_GAIN,
    REALTIME_PICK_MASK_POST_COMP_P,
    REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
    REALTIME_PICK_MASK_SMOOTH_RELEASE_MS,
    REALTIME_PICK_MASK_THRESHOLD_DB,
    REALTIME_PICK_RESO_BP_ORDER,
    REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
    REALTIME_PICK_RESO_DS_TAPS,
    REALTIME_PICK_RESO_INTERP_TAPS,
    REALTIME_PICK_RESO_SOFTCLIP_DRIVE,
    REALTIME_PRE_BLOCKS,
    REALTIME_PRE_KERNEL,
    REALTIME_SAMPLE_RATE,
    REALTIME_SEG_LEN,
    REALTIME_STEM_FINGER_DISTILL_ALPHA,
    REALTIME_STEM_PICK_DISTILL_ALPHA,
    REALTIME_STYLE_BLEND_INIT,
    REALTIME_STYLE_BLEND_TRAINABLE,
    REALTIME_TFLITE_T,
    RESBP_STRIKE_ENV_MS_REALTIME,
    STYLE_EMA_ALPHA_PER_HOP,
    STYLE_EMA_REF_HOP,
)
from .blocks import (
    _pick_attack,
    _pickup_adapter,
    _reson_learn_offline,
    _reson_learn_realtime,
    add_coloreq,
    clip_01x4,
    compute_pads,
    film_apply,
    gated_sep_block,
    glu_sep_block,
    invert,
    slice_content,
    slice_gate,
)
from .dsp import (
    _bp_minphase_rev_padded,
    _exp_smoothing_kernel,
    _get_minphase_hp_kernel,
    _hp_minphase_rev_padded,
    _interp_polyphase_len,
    _ir_prototype_kernel,
    _len_lr_from_full,
    _minphase_decimator_len,
    hp_cutoff_for_sr,
    pre_emphasis,
)
from .kd import (
    KDFeatureDistill,
    KDSTFTDistill,
    KDWaveformDistill,
    KD_TEACHER_DECIM_MARGIN,
)
from .layers import (
    Abs1D,
    BlendScalar,
    ConcatTime,
    ConstScale,
    FIRInterpByN,
    InvertStyle,
    IrScale,
    LossTether,
    MatchTimeToRef,
    PreEmphasisStream,
    RealtimeIrDownsampleFIR,
    SoftClip1D,
    StreamFixedFIR,
    StreamingDiffPos,
    StreamingFIRInterpByN,
    StreamingLinearRamp,
    StreamingPickDBMask,
    StreamingRealtimeIrDownsampleFIR,
    StreamingScalarEMA,
    StreamingStyleLerp,
    TakeLastN,
    TileStyle,
    ZeroBias,
    _pick_mask_engine,
    apply_runtime_gain,
    phase_polarity_gain,
)
from .modelio import (
    _tensor_basename,
    load_keras_model,
)

        
def _seed_cond_embed_from_legacy_style_embed(dst_model, src_model, verbose=False):
    """
    Legacy-compat shim: old graphs conditioned FiLM on a Dense named '<prefix>_style_embed'
    (input dim 1: style only); new graphs use '<prefix>_cond_embed' (input dim 2: style +
    pickup). Seed the new kernel's style row from the legacy weights and zero the pickup row,
    so a migrated model behaves IDENTICALLY to the old one at pickup=0 instead of getting a
    randomly initialized conditioning path.
    """
    seeded = 0
    for prefix in ("offline", "realtime"):
        try:
            dst = dst_model.get_layer(f"{prefix}_cond_embed")
            src = src_model.get_layer(f"{prefix}_style_embed")
        except Exception:
            continue
        sw = src.get_weights()  # [(1,E), (E,)]
        dw = dst.get_weights()  # [(2,E), (E,)]
        if len(sw) != 2 or len(dw) != 2:
            continue
        if sw[0].ndim != 2 or dw[0].ndim != 2:
            continue
        if sw[0].shape[1] != dw[0].shape[1] or dw[0].shape[0] != sw[0].shape[0] + 1:
            continue
        if np.shape(sw[1]) != np.shape(dw[1]):
            continue
        k = np.zeros_like(dw[0])
        k[:sw[0].shape[0], :] = sw[0]
        dst.set_weights([k, sw[1]])
        seeded += 1
    if verbose and seeded:
        print(f"[copy_weights_by_name] seeded {seeded} cond_embed layer(s) from legacy style_embed")
    return seeded

def copy_weights_by_name(dst_model, src_model, verbose=False):
    """
    Copy weights layer-by-layer by matching layer names and shapes.
    Returns the number of layers whose weights were copied.
    Leaves non-matching layers untouched.
    Also seeds new-style '<prefix>_cond_embed' layers from legacy '<prefix>_style_embed'
    weights when copying from an old-format model (no-op otherwise).
    """
    src_layers = {l.name: l for l in src_model.layers}
    matched = 0
    for l in dst_model.layers:
        if isinstance(l, FrozenTeacher):
            # The embedded KD teacher is always loaded fresh from the offline model on disk;
            # never overwrite it with the (possibly stale) teacher baked into an old checkpoint.
            continue
        ref = src_layers.get(l.name)
        if ref is None:
            continue
        sw = ref.get_weights()
        dw = l.get_weights()
        if len(sw) != len(dw):
            continue
        # Check shape compatibility
        ok = True
        for s, d in zip(sw, dw):
            if np.shape(s) != np.shape(d):
                ok = False
                break
        if not ok:
            continue
        l.set_weights(sw)
        matched += 1
    matched += _seed_cond_embed_from_legacy_style_embed(dst_model, src_model, verbose=verbose)
    if verbose:
        print(f"[copy_weights_by_name] copied {matched} layer(s) by name into '{dst_model.name}'")
    return matched

# ────────────────────────────── MODEL BUILDERS ───────────────────────────────

@register_keras_serializable()
class FrozenTeacher(Layer):
    """
    Runs a frozen Keras model in INFERENCE mode regardless of the outer training flag.

    Keras 3's functional graph replays a nested model call with the OUTER `training` value
    (it overrides the `training=False` recorded at graph-construction time), so a teacher that
    is simply called inside the student's training graph runs with training=True and its
    training-only layers (HFLowbandPenalty.add_loss) pollute the student's objective. This
    wrapper pins training=False for the wrapped model.
    """
    def __init__(self, model, **kwargs):
        kwargs.setdefault("trainable", False)
        super().__init__(**kwargs)
        self.model = model
        self.model.trainable = False

    def call(self, inputs, training=None):
        return self.model(inputs, training=False)

    def get_config(self):
        cfg = super().get_config()
        cfg["model"] = tf.keras.utils.serialize_keras_object(self.model)
        return cfg

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model, **config)


def load_offline_teacher(path=None):
    """
    Load the (non-causal, dynamic-time) offline model as the frozen KD teacher.

    NOTE: earlier versions distilled from a "causalized" clone (all look-ahead moved to the left
    pad). That is NOT the same network delayed: each residual block's conv path gets delayed by
    its right pad while the skip path does not, so the clone's residual sums are misaligned by
    0.5–12 ms per block. The student was being pulled toward a broken teacher. The real teacher
    is used now; it receives extra left/right context (KD_TEACHER_*_CTX_SAMPLES) and its outputs
    are cropped back onto the student's segment inside build_realtime_training_model.
    """
    if path is None:
        path = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    teacher = load_keras_model(path, compile=False)
    in_names = [_tensor_basename(t) for t in teacher.inputs]
    if any("knob_lc_hz" in n for n in in_names) or "offline_pickup_input" not in in_names:
        raise RuntimeError(
            f"{path} predates the current offline graph (it still has in-graph master-EQ knob "
            "inputs / lacks the pickup input). Retrain it with --train-offline, or rebuild it from "
            "the old weights with --migrate-offline, before running --train-realtime / --fine-tune.")
    teacher.trainable = False
    for l in teacher.layers:
        l.trainable = False
    return teacher

@register_keras_serializable()
def build_offline_model(dynamic_time=False,
                                  lookahead_samples=OFFLINE_LOOKAHEAD_SAMPLES,
                                  name="verus_offline",
                                  coloreq_trainable=False,
                                  include_dry_output=True):
    """
    Offline model. During training, set include_dry_output=False to keep 2 outputs:
      [final mix, style]
    For runtime/export, include_dry_output=True (default): expose 3 outputs:
      [final mix, style, dry]
    """
    T   = None if dynamic_time else OFFLINE_SEG_LEN
    inp = Input(shape=(T,1), name="offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="offline_style_true_input")
    style_force_in = Input(shape=(1,), name="offline_style_force_input")
    pickup_in      = Input(shape=(1,), name="offline_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="offline_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="offline_knob_phaseinv")

    # Loss core for the training-only lowband penalties (static training graph only)
    core = OFFLINE_LOSS_CORE if not dynamic_time else (None, None)

    # Pickup-conditional input adapter (pass-through at pickup=0). All analysis/synthesis
    # branches read the adapted signal; the dry paths keep the raw input.
    x_core = _pickup_adapter(inp, pickup_in, prefix="offline")

    # Style
    cls         = Conv1D(16,3, padding="same", activation="relu", name="offline_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="offline_style_gap")(cls)
    style_logit = Dense(1, name="offline_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="offline_style_prob")(style_logit)

    one_minus_force = Lambda(invert, name="offline_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="offline_style_used_scalar")([
        Multiply(name="offline_style_pred_part")([one_minus_force, style_prob]),
        Multiply(name="offline_style_true_part")([style_force_in, style_true_in]),
    ])
    style_time = TileStyle(name="offline_style_time_used")([style_used_scalar, inp])
    # FiLM conditioning on both axes (style + pickup)
    cond = Concatenate(name="offline_cond_concat")([style_used_scalar, pickup_in])
    style_embed= Dense(OFFLINE_EMBED_DIM, activation="linear", name="offline_cond_embed")(cond)

    x = Conv1D(OFFLINE_FILTERS, 1, padding="same", name="offline_init_pw")(x_core)
    for i in range(OFFLINE_PRE_BLOCKS):
        x = gated_sep_block(x, OFFLINE_FILTERS, OFFLINE_PRE_KERNEL, 1, prefix=f"offline_pre_{i}", lookahead=lookahead_samples)
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_pre_{i}",
                       gamma_scale=OFFLINE_FILM_GAMMA_SCALE, beta_scale=OFFLINE_FILM_BETA_SCALE,
                       gamma_mag=OFFLINE_FILM_GAMMA_MAG, beta_mag=OFFLINE_FILM_BETA_MAG)

    for idx, d in enumerate(OFFLINE_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=OFFLINE_KERNEL_SIZE, dilation=d, lookahead_samples=lookahead_samples)
        x = ZeroPadding1D((pad_left, pad_right), name=f"pad_{idx}")(x)
        g = SeparableConv1D(2*OFFLINE_FILTERS, OFFLINE_KERNEL_SIZE, dilation_rate=d, padding='valid', name=f"sep_{idx}")(x)
        c = Lambda(slice_content, arguments={'filters':OFFLINE_FILTERS}, name=f"slice_content_{idx}")(g)
        g2= Lambda(slice_gate,    arguments={'filters':OFFLINE_FILTERS}, name=f"slice_gate_{idx}")(g)
        x = Multiply(name=f"gated_mul_{idx}")([Activation('tanh', name=f"tanh_{idx}")(c),
                                               Activation('sigmoid', name=f"sigmoid_{idx}")(g2)])
        x = Conv1D(OFFLINE_FILTERS,1,padding='same', name=f"pw_{idx}")(x)
        x = Add(name=f"res_add_{idx}")([x, res])
        x = film_apply(x, style_embed, OFFLINE_FILTERS, prefix=f"offline_dil_{idx}",
                       gamma_scale=OFFLINE_FILM_GAMMA_SCALE, beta_scale=OFFLINE_FILM_BETA_SCALE,
                       gamma_mag=OFFLINE_FILM_GAMMA_MAG, beta_mag=OFFLINE_FILM_BETA_MAG)

    main_pick   = Conv1D(1,1,padding="same",name="offline_final_pick")(x)
    main_finger = Conv1D(1,1,padding="same",name="offline_final_finger")(x)
    inv_style   = InvertStyle(name="offline_invert_style")(style_time)
    pick_c      = Multiply(name="offline_pick_contrib")([style_time, main_pick])
    finger_c    = Multiply(name="offline_finger_contrib")([inv_style,   main_finger])
    main_core   = Add(name="offline_main_core")([pick_c, finger_c])

    # Analytic pick mask and heads (fed from the adapted signal so the fixed dB-domain
    # thresholds see piezo-like HF content regardless of pickup type)
    mask = _pick_mask_engine(x_core,
                             thr_db=OFFLINE_PICK_MASK_THRESHOLD_DB,
                             knee_db=OFFLINE_PICK_MASK_KNEE_DB,
                             ref_is_hp=True,
                             post_comp_p=OFFLINE_PICK_MASK_POST_COMP_P,
                             post_comp_gain=OFFLINE_PICK_MASK_POST_COMP_GAIN,
                             floor=OFFLINE_PICK_MASK_FLOOR,
                             out_gain=OFFLINE_PICK_MASK_OUT_GAIN,
                             env_fast_ms=OFFLINE_PICK_MASK_ENV_FAST_MS,
                             env_slow_ms=OFFLINE_PICK_MASK_ENV_SLOW_MS,
                             hp_order=OFFLINE_HF_HP_ORDER,
                             sample_rate=OFFLINE_SAMPLE_RATE,
                             smooth_attack_ms=OFFLINE_PICK_MASK_SMOOTH_ATTACK_MS,
                             smooth_release_ms=OFFLINE_PICK_MASK_SMOOTH_RELEASE_MS,
                             prefix="offline")

    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="offline_hf_pre_emph")(x_core)
    pickattack, pa_pen = _pick_attack("offline", hf_in, mask, style_time, filters=16,
                                      sample_rate=OFFLINE_SAMPLE_RATE,
                                      softclip_drive=OFFLINE_PICK_ATTACK_SOFTCLIP_DRIVE,
                                      lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                                      lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                                      lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                                      cutoff_hz=OFFLINE_PICK_ATTACK_CUTOFF_HZ,
                                      core=core)
    pickreso, pr_pen = _reson_learn_offline(x_core, mask, style_time, core=core)

    knob_pa_in  = Input(shape=(1,), name="offline_knob_pickattack")
    knob_pr_in  = Input(shape=(1,), name="offline_knob_pickreso")
    knob_main_in= Input(shape=(1,), name="offline_knob_main")
    knob_dw_in  = Input(shape=(1,), name="offline_knob_drywet")

    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="offline_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="offline_pickreso_user_gain")
    main_scaled = apply_runtime_gain(main_core, knob_main_in, name="offline_main_user_gain")

    out_prehp = Add(name="offline_with_picklearn_prehp")([main_scaled, pickattack, pickreso])
    out_prehp = LossTether(name="offline_pick_lowband_penalties")([out_prehp, pa_pen, pr_pen])

    # ColorEQ (pre dry/wet)
    out_col = add_coloreq(out_prehp, bands=COLOREQ_BANDS_OFFLINE_BODY, sample_rate=OFFLINE_SAMPLE_RATE,
                          name_prefix="offline_coloreq", order=OFFLINE_COLOREQ_ORDER, trainable_scales=bool(coloreq_trainable))

    # Dry/wet mix
    wet_gain   = Lambda(invert, name="offline_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(out_col, wet_gain,   name="offline_wet_gain")
    dry_scaled = apply_runtime_gain(inp,     knob_dw_in, name="offline_dry_gain")
    y_mix      = Add(name="offline_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + output gain
    phase_gain = Lambda(phase_polarity_gain, name="offline_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="offline_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="offline_output_gain")

    # Dry passthrough (raw piezo, time-aligned)
    dry_out = Activation('linear', name="offline_dry_out")(inp)

    inputs_list = [inp, style_true_in, style_force_in, pickup_in,
                   knob_main_in, knob_pa_in, knob_pr_in, knob_dw_in,
                   knob_outgain, knob_phaseinv]

    if include_dry_output:
        return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob, dry_out], name=name)
    else:
        return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob], name=name)

@register_keras_serializable()
def build_realtime_training_model(offline=None,
                        dynamic_time=False,
                        name="verus_realtime",
                        coloreq_trainable=False):
    T   = None if dynamic_time else REALTIME_SEG_LEN
    inp = Input(shape=(T,1), name="realtime_piezo_input")

    offline_inp = None
    if offline is not None:
        offline_inp = Input(shape=(None, 1), name="realtime_offline_piezo_input")

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")
    pickup_in      = Input(shape=(1,), name="realtime_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="realtime_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="realtime_knob_phaseinv")

    knob_main_in = Input(shape=(1,), name="realtime_knob_main")
    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    # Loss core (student-rate samples) for KD and the lowband penalties: static training graph only
    core = REALTIME_LOSS_CORE if not dynamic_time else (None, None)

    # Pickup-conditional input adapter (pass-through at pickup=0)
    x_core = _pickup_adapter(inp, pickup_in, prefix="realtime")

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)
    style_prob  = Activation("sigmoid", name="realtime_style_prob")(style_logit)

    kd_factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE

    style_pred_source = style_prob
    teacher_audio = None
    teacher_pick = None
    teacher_finger = None
    teacher_feats = None
    if offline is not None:
        teacher = offline
        teacher.trainable = False

        # Teacher wrapper exposing audio, style, the pick/finger heads, and two FiLM-block
        # feature maps for knowledge distillation. The teacher is the real (non-causal) offline
        # model; it gets KD_TEACHER_*_CTX_SAMPLES of extra context around the student's segment
        # (see segment_generator_dual) and its time-domain outputs are cropped back below.
        audio_out, style_prob_out = teacher.outputs[0], teacher.outputs[1]
        t_feat_mid_idx = len(OFFLINE_DILATIONS) // 2
        t_feat_last_idx = len(OFFLINE_DILATIONS) - 1
        teacher_combo = FrozenTeacher(tf.keras.Model(
            inputs=teacher.inputs,
            outputs=[audio_out, style_prob_out,
                     teacher.get_layer("offline_final_pick").output,
                     teacher.get_layer("offline_final_finger").output,
                     teacher.get_layer(f"offline_dil_{t_feat_mid_idx}_film_apply").output,
                     teacher.get_layer(f"offline_dil_{t_feat_last_idx}_film_apply").output],
            name="offline_teacher_combo_model"
        ), name="offline_teacher_combo")

        # IMPORTANT: feed inputs in the exact order of build_offline_model(inputs_list)
        teacher_inputs = [
            offline_inp,                # offline_piezo_input (student segment ×2 + teacher context)
            style_true_in,              # offline_style_true_input
            style_force_in,             # offline_style_force_input
            pickup_in,                  # offline_pickup_input

            knob_main_in,               # offline_knob_main
            knob_pa_in,                 # offline_knob_pickattack
            knob_pr_in,                 # offline_knob_pickreso
            knob_dw_in,                 # offline_knob_drywet

            knob_outgain,               # offline_knob_outgain
            knob_phaseinv,              # offline_knob_phaseinv
        ]

        # FrozenTeacher pins training=False: the teacher's training-only penalty layers must not
        # add (constant) losses to the student's objective.
        (teacher_audio, offline_style_prob,
         teacher_pick, teacher_finger,
         t_feat_mid, t_feat_last) = teacher_combo(teacher_inputs)

        # Crop the teacher's context so every teacher tensor is time-aligned with the student's
        # segment (96k: 2× the student length). KD_TEACHER_DECIM_MARGIN samples of the right
        # context are kept: _kd_align's zero-phase decimation filter reaches that far past the
        # last loss-core sample.
        t_crop = (int(KD_TEACHER_LEFT_CTX_SAMPLES), int(KD_TEACHER_RIGHT_CTX_SAMPLES - KD_TEACHER_DECIM_MARGIN))
        teacher_audio  = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_audio_crop")(teacher_audio)
        teacher_pick   = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_pick_crop")(teacher_pick)
        teacher_finger = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_finger_crop")(teacher_finger)
        t_feat_mid     = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_feat0_crop")(t_feat_mid)
        t_feat_last    = tf.keras.layers.Cropping1D(t_crop, name="kd_teacher_feat1_crop")(t_feat_last)
        teacher_feats = (t_feat_mid, t_feat_last)

        # Blend student style prob with teacher’s style prob
        style_pred_source = BlendScalar(
            init_alpha=REALTIME_STYLE_BLEND_INIT,
            trainable=REALTIME_STYLE_BLEND_TRAINABLE,
            name="realtime_style_blend"
        )([style_prob, offline_style_prob])

    one_minus_force = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_pred_source]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])
    style_scalar_b3 = Reshape((1, 1), name="realtime_style_scalar_broadcast")(style_used_scalar)
    # FiLM conditioning on both axes (style + pickup)
    cond            = Concatenate(name="realtime_cond_concat")([style_used_scalar, pickup_in])
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_cond_embed")(cond)

    kd_feat_mid_idx  = len(REALTIME_DILATIONS) // 2
    kd_feat_last_idx = len(REALTIME_DILATIONS) - 1

    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(x_core)
    for i in range(REALTIME_PRE_BLOCKS):
        x = glu_sep_block(x, REALTIME_FILTERS, REALTIME_PRE_KERNEL, 1,
                          prefix=f"realtime_pre_{i}", lookahead=0,
                          residual_scale=0.35, bias_constraint=BC)
        x = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}",
                       gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                       gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)

    for i, d in enumerate(REALTIME_DILATIONS):
        res = x
        pad_left, pad_right = compute_pads(kernel=REALTIME_KERNEL_SIZE, dilation=d, lookahead_samples=0)
        x = ZeroPadding1D((pad_left, pad_right), name=f"realtime_pad_{i}")(x)
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid",
                            bias_constraint=BC, name=f"realtime_sep_{i}")(x)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        xb = Multiply(name=f"realtime_gate_mul_{i}")([c, Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pw_{i}")(xb)
        xb = ConstScale(0.35, name=f"realtime_res_scale_{i}")(xb)
        x  = Add(name=f"realtime_res_add_{i}")([xb, res])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)
        # Feature-space knowledge distillation (returns x unchanged)
        if teacher_feats is not None and i == kd_feat_mid_idx:
            x = KDFeatureDistill(alpha=REALTIME_FEAT_DISTILL_ALPHA, teacher_channels=OFFLINE_FILTERS,
                                 factor=kd_factor, core_start=core[0], core_end=core[1],
                                 name="feat_distill_0")([x, teacher_feats[0]])
        if teacher_feats is not None and i == kd_feat_last_idx:
            x = KDFeatureDistill(alpha=REALTIME_FEAT_DISTILL_ALPHA, teacher_channels=OFFLINE_FILTERS,
                                 factor=kd_factor, core_start=core[0], core_end=core[1],
                                 name="feat_distill_1")([x, teacher_feats[1]])

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)
    # Stem-level knowledge distillation against the teacher's pick/finger heads
    if teacher_pick is not None:
        out_pick = KDWaveformDistill(alpha=REALTIME_STEM_PICK_DISTILL_ALPHA, factor=kd_factor,
                                     core_start=core[0], core_end=core[1],
                                     name="stem_pick_distill")([out_pick, teacher_pick])
    if teacher_finger is not None:
        out_finger = KDWaveformDistill(alpha=REALTIME_STEM_FINGER_DISTILL_ALPHA, factor=kd_factor,
                                       core_start=core[0], core_end=core[1],
                                       name="stem_finger_distill")([out_finger, teacher_finger])
    inv_style  = Lambda(invert, name="realtime_invert_style")(style_scalar_b3)
    pick_c     = Multiply(name="realtime_pick_contrib")([style_scalar_b3, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style,     out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch (non-streaming)
    ir = RealtimeIrDownsampleFIR(name="realtime_ir_downsample")(x_core)
    for i, d in enumerate(REALTIME_IR_DILATIONS):
        pad = (REALTIME_IR_KERNEL - 1) * d
        ir = ZeroPadding1D((pad, 0), name=f"realtime_ir_pad_{i}")(ir)
        ir = SeparableConv1D(filters=1, kernel_size=REALTIME_IR_KERNEL, dilation_rate=d, padding="valid",
                             depthwise_regularizer=regularizers.l2(1e-4),
                             pointwise_regularizer=regularizers.l2(1e-4),
                             depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                             pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                             bias_constraint=BC, name=f"realtime_ir_sep_{i}")(ir)
    ir = IrScale(init=0.6, name="realtime_ir_scale")(ir)
    ir = FIRInterpByN(factor=REALTIME_IR_DOWNSAMPLE_FACTOR, num_taps=max(193, REALTIME_LP_ORDER),
                      cutoff=0.45/REALTIME_IR_DOWNSAMPLE_FACTOR, window=('kaiser', 9.0),
                      name="realtime_ir_interp")(ir)
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir, inp])

    # Pick mask and heads (fed from the adapted signal)
    mask = _pick_mask_engine(x_core,
                             thr_db=REALTIME_PICK_MASK_THRESHOLD_DB,
                             knee_db=REALTIME_PICK_MASK_KNEE_DB,
                             ref_is_hp=True,
                             post_comp_p=REALTIME_PICK_MASK_POST_COMP_P,
                             post_comp_gain=REALTIME_PICK_MASK_POST_COMP_GAIN,
                             floor=REALTIME_PICK_MASK_FLOOR,
                             out_gain=REALTIME_PICK_MASK_OUT_GAIN,
                             env_fast_ms=REALTIME_PICK_MASK_ENV_FAST_MS,
                             env_slow_ms=REALTIME_PICK_MASK_ENV_SLOW_MS,
                             hp_order=REALTIME_HF_HP_ORDER,
                             sample_rate=REALTIME_SAMPLE_RATE,
                             smooth_attack_ms=REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
                             smooth_release_ms=REALTIME_PICK_MASK_SMOOTH_RELEASE_MS,
                             prefix="realtime")
    hf_in = Lambda(pre_emphasis, arguments={'alpha': HP_PRE_EMPHASIS_HF}, name="realtime_hf_pre_emph")(x_core)
    pickattack, pa_pen = _pick_attack("realtime", hf_in, mask, style_scalar_b3,
                                      filters=16, sample_rate=REALTIME_SAMPLE_RATE,
                                      softclip_drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE,
                                      lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                                      lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                                      lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                                      cutoff_hz=REALTIME_PICK_ATTACK_CUTOFF_HZ,
                                      core=core)
    pickreso, pr_pen = _reson_learn_realtime(x_core, mask, style_scalar_b3, core=core)

    pickattack = apply_runtime_gain(pickattack, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(pickreso,   knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,         knob_ir_in, name="realtime_ir_user_gain")
    
    y_main_scaled = apply_runtime_gain(y_main, knob_main_in, name="realtime_main_user_gain")

    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main_scaled, ir, pickattack, pickreso])
    y_prehp  = LossTether(name="realtime_pick_lowband_penalties")([y_prehp, pa_pen, pr_pen])

    y_wet = add_coloreq(y_prehp, bands=COLOREQ_BANDS_REALTIME_BODY, sample_rate=REALTIME_SAMPLE_RATE,
                        name_prefix="realtime_coloreq", order=REALTIME_COLOREQ_ORDER, trainable_scales=bool(coloreq_trainable))
    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_mix      = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + Output gain
    phase_gain = Lambda(phase_polarity_gain, name="realtime_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="realtime_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="realtime_output_gain")

    # Audio-level knowledge distillation against the teacher's output (loss core only).
    # Both sides see identical knob values during training, so the comparison is level-matched.
    if teacher_audio is not None:
        y_out = KDWaveformDistill(alpha=REALTIME_DISTILL_ALPHA, factor=kd_factor,
                                  core_start=core[0], core_end=core[1],
                                  name="audio_distill")([y_out, teacher_audio])
        y_out = KDSTFTDistill(alpha=REALTIME_AUDIO_STFT_DISTILL_ALPHA, factor=kd_factor,
                              core_start=core[0], core_end=core[1],
                              name="audio_stft_distill")([y_out, teacher_audio])

    inputs_list = [inp, style_true_in, style_force_in, pickup_in]
    if offline_inp is not None:
        inputs_list = [inp, offline_inp, style_true_in, style_force_in, pickup_in]
    inputs_list += [knob_main_in, knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in,
                    knob_outgain, knob_phaseinv]

    return tf.keras.models.Model(inputs=inputs_list, outputs=[y_out, style_prob], name=name)

@register_keras_serializable()
def build_realtime_streaming_model(dynamic_time=False,
                                   name="verus_realtime",
                                   coloreq_trainable=False,
                                   static_T=REALTIME_TFLITE_T):
    if dynamic_time:
        T = None
    else:
        T = int(static_T)

    inp = Input(shape=(T,1), name="realtime_piezo_input")
    state_inputs = []
    state_outputs = []

    style_true_in  = Input(shape=(1,), name="realtime_style_true_input")
    style_force_in = Input(shape=(1,), name="realtime_style_force_input")
    pickup_in      = Input(shape=(1,), name="realtime_pickup_input")  # 0=piezo, 1=magnetic (user-set)

    knob_main_in = Input(shape=(1,), name="realtime_knob_main")
    knob_pa_in = Input(shape=(1,), name="realtime_knob_pickattack")
    knob_pr_in = Input(shape=(1,), name="realtime_knob_pickreso")
    knob_ir_in = Input(shape=(1,), name="realtime_knob_ir")
    knob_dw_in = Input(shape=(1,), name="realtime_knob_drywet")

    # Pickup-conditional input adapter (streaming; pass-through at pickup=0).
    # Layer names mirror the training graph so copy_weights_by_name transfers the weights.
    a = inp
    for i, d in enumerate(PICKUP_ADAPTER_DILATIONS):
        pad = (PICKUP_ADAPTER_KERNEL - 1) * d
        ch_in = 1 if i == 0 else PICKUP_ADAPTER_FILTERS
        s_ad = Input(shape=(pad, ch_in), name=f"realtime_state_pickup_adapt_c{i+1}")
        state_inputs.append(s_ad)
        a_cat = ConcatTime(name=f"realtime_pickup_adapt_c{i+1}_concat")([s_ad, a])
        a = SeparableConv1D(PICKUP_ADAPTER_FILTERS, PICKUP_ADAPTER_KERNEL, dilation_rate=d,
                            padding="valid", name=f"realtime_pickup_adapt_c{i+1}")(a_cat)
        a = Activation("relu", name=f"realtime_pickup_adapt_c{i+1}_relu")(a)
        s_ad_n = TakeLastN(pad, name=f"realtime_state_pickup_adapt_c{i+1}_next")(a_cat)
        state_outputs.append(s_ad_n)
    a = Conv1D(1, 1, padding="same", kernel_initializer="zeros", name="realtime_pickup_adapt_out")(a)
    p3 = Reshape((1, 1), name="realtime_pickup_scalar_b3")(pickup_in)
    delta = Multiply(name="realtime_pickup_adapt_gate")([a, p3])
    x_core = Add(name="realtime_pickup_adapt_add")([inp, delta])

    # Style head
    cls         = Conv1D(16, 3, padding="same", activation="relu", name="realtime_style_conv")(x_core)
    cls         = GlobalAveragePooling1D(name="realtime_style_gap")(cls)
    style_logit = Dense(1, name="realtime_style_logit")(cls)

    # EMA of per-hop style LOGIT. The state 'realtime_state_style_ema' stores the logit, so a
    # zero state corresponds to probability 0.5 (unbiased at reset). Applying sigmoid AFTER the
    # EMA matches the training-time pooling (sigmoid of a linear pool over ~90 ms) far better
    # than an EMA of per-32-sample sigmoids, which is both noisy and biased toward 0.5.
    s_style = Input(shape=(1,1), name="realtime_state_style_ema")
    state_inputs.append(s_style)
    s_style_flat = Reshape((1,), name="realtime_state_style_ema_flat")(s_style)
    # Hop-invariant EMA: alpha is defined per STYLE_EMA_REF_HOP samples and rescaled to the actual
    # hop length (T), so a dynamic-T graph stepped 1024 samples at a time tracks style exactly like
    # the T=32 pedal graph.
    style_logit_ema, s_style_next_flat = StreamingScalarEMA(
        alpha=STYLE_EMA_ALPHA_PER_HOP, ref_hop=STYLE_EMA_REF_HOP,
        name="realtime_style_logit_ema")([style_logit, s_style_flat, inp])
    s_style_next = Reshape((1,1), name="realtime_state_style_ema_expand")(s_style_next_flat)
    state_outputs.append(Activation('linear', name="realtime_state_style_ema_next")(s_style_next))
    style_prob_ema = Activation("sigmoid", name="realtime_style_prob")(style_logit_ema)

    # Select predicted vs true style
    one_minus_force   = Lambda(invert, name="realtime_one_minus_force")(style_force_in)
    style_used_scalar = Add(name="realtime_style_used_scalar")([
        Multiply(name="realtime_style_pred_part")([one_minus_force, style_prob_ema]),
        Multiply(name="realtime_style_true_part")([style_force_in, style_true_in]),
    ])

    # FiLM conditioning on both axes (style + pickup)
    cond            = Concatenate(name="realtime_cond_concat")([style_used_scalar, pickup_in])
    style_embed     = Dense(REALTIME_EMBED_DIM, activation="linear", name="realtime_cond_embed")(cond)

    # NEW: per-sample style_time via streaming linear interpolation between hops
    s_style_prev = Input(shape=(1,1), name="realtime_state_style_prev")
    state_inputs.append(s_style_prev)
    style_time, s_style_prev_next = StreamingStyleLerp(name="realtime_style_time_stream")([inp, style_used_scalar, s_style_prev])
    
        # NEW: per-sample smoothing of the style envelope for the pickattack head only.
    # This removes the subtle “zipper/bitcrush” artifact caused by per-hop envelope slope resets.
    tau_pa = max(1, int(round((PICK_ATTACK_STYLE_SMOOTH_MS / 1000.0) * REALTIME_SAMPLE_RATE)))
    L_pa   = int(np.clip(np.ceil(3.0 * tau_pa), 8, 1024))  # keep the state small
    s_style_pa = Input(shape=(L_pa - 1, 1), name="realtime_state_pickattack_style_smooth")
    state_inputs.append(s_style_pa)
    k_pa_smooth = _exp_smoothing_kernel(tau_pa, scale=3.0, Lmax=L_pa)
    style_time_pa, s_style_pa_n = StreamFixedFIR(
        kernel_len=L_pa,
        out_ch=1,
        weight_name="exp_smooth_taps",
        trainable_kernel=False,
        taps_init=k_pa_smooth,
        name="realtime_pickattack_style_smooth"
    )([style_time, s_style_pa])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_style_smooth_next")(s_style_pa_n))
    
    state_outputs.append(Activation('linear', name="realtime_state_style_prev_next")(s_style_prev_next))
    inv_style_time = Lambda(invert, name="realtime_invert_style_time")(style_time)

    # Main stack
    BC = ZeroBias()
    x = Conv1D(REALTIME_FILTERS, 1, padding="same", name="realtime_init_pw", bias_constraint=BC)(x_core)
    for i in range(REALTIME_PRE_BLOCKS):
        pad_len = (REALTIME_PRE_KERNEL - 1) * 1
        s_in = Input(shape=(pad_len, REALTIME_FILTERS), name=f"realtime_state_pre_{i}")
        state_inputs.append(s_in)
        x_cat = ConcatTime(name=f"realtime_pre_{i}_concat")([s_in, x])
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_PRE_KERNEL, dilation_rate=1, padding="valid",
                            bias_constraint=BC, name=f"realtime_pre_{i}_sep")(x_cat)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_pre_{i}_slice_c")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_pre_{i}_slice_g")(g)
        xb = Multiply(name=f"realtime_pre_{i}_gate_mul")([c, Activation("sigmoid", name=f"realtime_pre_{i}_sigmoid")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pre_{i}_pw")(xb)
        xb = ConstScale(0.35, name=f"realtime_pre_{i}_res_scale")(xb)
        x  = Add(name=f"realtime_pre_{i}_res_add")([xb, x])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_pre_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)
        s_out = TakeLastN(pad_len, name=f"realtime_state_pre_{i}_next")(x_cat)
        state_outputs.append(s_out)

    for i, d in enumerate(REALTIME_DILATIONS):
        pad_len = (REALTIME_KERNEL_SIZE - 1) * d
        s_in = Input(shape=(pad_len, REALTIME_FILTERS), name=f"realtime_state_sep_{i}")
        state_inputs.append(s_in)
        x_cat = ConcatTime(name=f"realtime_padcat_{i}")([s_in, x])
        g = SeparableConv1D(2*REALTIME_FILTERS, REALTIME_KERNEL_SIZE, dilation_rate=d, padding="valid",
                            bias_constraint=BC, name=f"realtime_sep_{i}")(x_cat)
        c  = Lambda(slice_content, arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_c_{i}")(g)
        g2 = Lambda(slice_gate,    arguments={"filters": REALTIME_FILTERS}, name=f"realtime_slice_g_{i}")(g)
        xb = Multiply(name=f"realtime_gate_mul_{i}")([c, Activation("sigmoid", name=f"realtime_sigmoid_{i}")(g2)])
        xb = Conv1D(REALTIME_FILTERS, 1, padding="same", bias_constraint=BC, name=f"realtime_pw_{i}")(xb)
        xb = ConstScale(0.35, name=f"realtime_res_scale_{i}")(xb)
        x  = Add(name=f"realtime_res_add_{i}")([xb, x])
        x  = film_apply(x, style_embed, REALTIME_FILTERS, prefix=f"realtime_dil_{i}",
                        gamma_scale=REALTIME_FILM_GAMMA_SCALE, beta_scale=REALTIME_FILM_BETA_SCALE,
                        gamma_mag=REALTIME_FILM_GAMMA_MAG, beta_mag=REALTIME_FILM_BETA_MAG)
        s_out = TakeLastN(pad_len, name=f"realtime_state_sep_{i}_next")(x_cat)
        state_outputs.append(s_out)

    out_pick   = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_pick")(x)
    out_finger = Conv1D(1, 1, padding="same", bias_constraint=BC, name="realtime_final_finger")(x)

    pick_c     = Multiply(name="realtime_pick_contrib")([style_time, out_pick])
    finger_c   = Multiply(name="realtime_finger_contrib")([inv_style_time, out_finger])
    y_main     = Add(name="realtime_main")([pick_c, finger_c])

    # IR branch (streaming). The decimator uses the min-phase prototype kernel, whose actual
    # length is roughly half of REALTIME_IR_FIR_SIZE (scipy minimum_phase halves the taps) —
    # declare the state from the real kernel so it matches the next-state the layer emits.
    L_ir_ds = int(_ir_prototype_kernel().shape[0])
    s_ir_ds = Input(shape=(L_ir_ds - 1, 1), name="realtime_state_ir_ds")
    state_inputs.append(s_ir_ds)
    ir_ds, s_ir_ds_next = StreamingRealtimeIrDownsampleFIR(
        factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
        kernel_size=REALTIME_IR_FIR_SIZE,
        use_ir_prototype=True,
        name="realtime_ir_downsample"
    )([x_core, s_ir_ds])
    state_outputs.append(Activation('linear', name="realtime_state_ir_ds_next")(s_ir_ds_next))

    ir_lr = ir_ds
    for i, d in enumerate(REALTIME_IR_DILATIONS):
        pad = (REALTIME_IR_KERNEL - 1) * d
        s_in = Input(shape=(pad, 1), name=f"realtime_state_ir_sep_{i}")
        state_inputs.append(s_in)
        ir_cat = ConcatTime(name=f"realtime_ir_padcat_{i}")([s_in, ir_lr])
        ir_lr = SeparableConv1D(filters=1, kernel_size=REALTIME_IR_KERNEL, dilation_rate=d, padding="valid",
                                depthwise_regularizer=regularizers.l2(1e-4),
                                pointwise_regularizer=regularizers.l2(1e-4),
                                depthwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                                pointwise_constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0),
                                bias_constraint=BC, name=f"realtime_ir_sep_{i}")(ir_cat)
        s_out = TakeLastN(pad, name=f"realtime_state_ir_sep_{i}_next")(ir_cat)
        state_outputs.append(s_out)
    ir_lr = IrScale(init=0.6, name="realtime_ir_scale")(ir_lr)

    s_ir_up = Input(shape=(_interp_polyphase_len(max(193, REALTIME_LP_ORDER), REALTIME_IR_DOWNSAMPLE_FACTOR) - 1, 1),
                    name="realtime_state_ir_interp")
    state_inputs.append(s_ir_up)
    ir_up, s_ir_up_next = StreamingFIRInterpByN(factor=REALTIME_IR_DOWNSAMPLE_FACTOR,
                                                num_taps=max(193, REALTIME_LP_ORDER),
                                                cutoff=0.45/REALTIME_IR_DOWNSAMPLE_FACTOR,
                                                window=('kaiser', 9.0),
                                                name="realtime_ir_interp")(inputs=[ir_lr, s_ir_up])
    state_outputs.append(Activation('linear', name="realtime_state_ir_interp_next")(s_ir_up_next))
    ir = MatchTimeToRef(name="realtime_ir_matchlen")([ir_up, inp])

    # Pick mask (fed from the adapted signal)
    def ema_len(ms, sr, scale, Lmax):
        s = max(1, int(round((float(ms)/1000.0) * float(sr))))
        L = int(np.clip(np.ceil(scale * s), 8, Lmax))
        return L
    # scipy's minimum_phase halves the tap count, so derive the true HPF kernel length from the
    # kernel itself; the declared state must match the next-state StreamingPickDBMask emits.
    _k_hp_probe = _get_minphase_hp_kernel(REALTIME_SAMPLE_RATE,
                                          hp_cutoff_for_sr(REALTIME_SAMPLE_RATE),
                                          REALTIME_HF_HP_ORDER)
    L_hp   = int(_k_hp_probe.shape[0])
    L_fast = ema_len(REALTIME_PICK_MASK_ENV_FAST_MS, REALTIME_SAMPLE_RATE, 3.0, 1024)
    L_slow = ema_len(REALTIME_PICK_MASK_ENV_SLOW_MS, REALTIME_SAMPLE_RATE, 3.0, 4096)
    L_att  = ema_len(REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,  REALTIME_SAMPLE_RATE, 4.0, 2048)
    L_rel  = ema_len(REALTIME_PICK_MASK_SMOOTH_RELEASE_MS, REALTIME_SAMPLE_RATE, 4.0, 2048)

    s_m_hp   = Input(shape=(L_hp-1,   1), name="realtime_state_pickmask_hp")
    s_m_fast = Input(shape=(L_fast-1, 1), name="realtime_state_pickmask_fast")
    s_m_slow = Input(shape=(L_slow-1, 1), name="realtime_state_pickmask_slow")
    s_m_att  = Input(shape=(L_att-1,  1), name="realtime_state_pickmask_att")
    s_m_rel  = Input(shape=(L_rel-1,  1), name="realtime_state_pickmask_rel")
    s_m_prev = Input(shape=(1,1),         name="realtime_state_pickmask_prev")
    state_inputs += [s_m_hp, s_m_fast, s_m_slow, s_m_att, s_m_rel, s_m_prev]

    mask_outs = StreamingPickDBMask(
        sample_rate=REALTIME_SAMPLE_RATE,
        hp_cut_hz=hp_cutoff_for_sr(REALTIME_SAMPLE_RATE),
        hp_order=REALTIME_HF_HP_ORDER,
        env_fast_ms=REALTIME_PICK_MASK_ENV_FAST_MS,
        env_slow_ms=REALTIME_PICK_MASK_ENV_SLOW_MS,
        thr_db=REALTIME_PICK_MASK_THRESHOLD_DB,
        knee_db=REALTIME_PICK_MASK_KNEE_DB,
        smooth_attack_ms=REALTIME_PICK_MASK_SMOOTH_ATTACK_MS,
        smooth_release_ms=REALTIME_PICK_MASK_SMOOTH_RELEASE_MS,
        ref_is_hp=True,
        post_comp_p=REALTIME_PICK_MASK_POST_COMP_P,
        post_comp_gain=REALTIME_PICK_MASK_POST_COMP_GAIN,
        floor=REALTIME_PICK_MASK_FLOOR,
        out_gain=REALTIME_PICK_MASK_OUT_GAIN,
        name="realtime_pick_mask_db"
    )([x_core, s_m_hp, s_m_fast, s_m_slow, s_m_att, s_m_rel, s_m_prev])
    mask, s_hp_n, s_fast_n, s_slow_n, s_att_n, s_rel_n, s_prev_n = mask_outs
    state_outputs += [
        Activation('linear', name="realtime_state_pickmask_hp_next")(s_hp_n),
        Activation('linear', name="realtime_state_pickmask_fast_next")(s_fast_n),
        Activation('linear', name="realtime_state_pickmask_slow_next")(s_slow_n),
        Activation('linear', name="realtime_state_pickmask_att_next")(s_att_n),
        Activation('linear', name="realtime_state_pickmask_rel_next")(s_rel_n),
        Activation('linear', name="realtime_state_pickmask_prev_next")(s_prev_n),
    ]

    # Pre-emphasis
    s_pre = Input(shape=(1,1), name="realtime_state_preemph")
    state_inputs.append(s_pre)
    hf_in, s_pre_next = PreEmphasisStream(alpha=HP_PRE_EMPHASIS_HF, name="realtime_hf_pre_emph_stream")([x_core, s_pre])
    state_outputs.append(Activation('linear', name="realtime_state_preemph_next")(s_pre_next))

    # Pick-attack tiny CNN (smoothed style gating)
    x_pa = Multiply(name="realtime_picklearn_gate")([hf_in, mask])

    pad_c1 = (7 - 1) * 1
    s_c1 = Input(shape=(pad_c1, 1), name="realtime_state_picklearn_c1")
    state_inputs.append(s_c1)
    x_cat1 = ConcatTime(name="realtime_picklearn_c1_concat")([s_c1, x_pa])
    x_pa1 = SeparableConv1D(16, 7, dilation_rate=1, padding="valid", name="realtime_picklearn_c1")(x_cat1)
    x_pa1 = tf.keras.layers.LeakyReLU(negative_slope=0.0, name="realtime_picklearn_c1_relu")(x_pa1)
    s_c1n = TakeLastN(pad_c1, name="realtime_state_picklearn_c1_next")(x_cat1)
    state_outputs.append(s_c1n)

    pad_c2 = (7 - 1) * 2
    s_c2 = Input(shape=(pad_c2, 16), name="realtime_state_picklearn_c2")
    state_inputs.append(s_c2)
    x_cat2 = ConcatTime(name="realtime_picklearn_c2_concat")([s_c2, x_pa1])
    x_pa2 = SeparableConv1D(16, 7, dilation_rate=2, padding="valid", name="realtime_picklearn_c2")(x_cat2)
    x_pa2 = tf.keras.layers.LeakyReLU(negative_slope=0.0, name="realtime_picklearn_c2_relu")(x_pa2)
    s_c2n = TakeLastN(pad_c2, name="realtime_state_picklearn_c2_next")(x_cat2)
    state_outputs.append(s_c2n)

    x_pa_out = Conv1D(1, 1, padding="same", name="realtime_picklearn_out")(x_pa2)
    # NEW: gate with smoothed time envelope
    xg = Multiply(name="realtime_pickattack_gated")([x_pa_out, style_time_pa])
    xg = SoftClip1D(drive=REALTIME_PICK_ATTACK_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickattack_softclip")(xg)

    # Fade-in ramp to suppress startup pop
    s_pa_ramp = Input(shape=(1,1), name="realtime_state_pickattack_ramp")
    state_inputs.append(s_pa_ramp)
    ramp_pa, s_pa_ramp_n = StreamingLinearRamp(ramp_len=int(0.003 * REALTIME_SAMPLE_RATE),
                                               name="realtime_pickattack_ramp")([xg, s_pa_ramp])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_ramp_next")(s_pa_ramp_n))
    xg = Multiply(name="realtime_pickattack_ramp_apply")([xg, ramp_pa])

    # Post-HPF + scale + limit
    s_hp_pa = Input(shape=(PICK_ATTACK_LOWBAND_PENALTY_ORDER - 1, 1), name="realtime_state_pickattack_posthp")
    state_inputs.append(s_hp_pa)
    L_pa = int(PICK_ATTACK_LOWBAND_PENALTY_ORDER)
    hp_pa_k_full = _hp_minphase_rev_padded(REALTIME_SAMPLE_RATE, REALTIME_PICK_ATTACK_CUTOFF_HZ, L_pa)
    xh_pa, s_hp_pa_n = StreamFixedFIR(kernel_len=L_pa, out_ch=1, weight_name="hp_taps",
                                      trainable_kernel=False, taps_init=hp_pa_k_full,
                                      name="realtime_pickattack_post_hp")([xg, s_hp_pa])
    state_outputs.append(Activation('linear', name="realtime_state_pickattack_posthp_next")(s_hp_pa_n))
    y_pa = IrScale(init=1.0, name="realtime_pickattack_scale")(xh_pa)
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y_pa = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=limit_lin, name="realtime_pickattack_limit")(y_pa)

    # Pick-reso streaming (smoothed style gating)
    s_on_prev = Input(shape=(1,1), name="realtime_state_pickreso_onset_prev")
    state_inputs.append(s_on_prev)
    onset, s_on_prev_n = StreamingDiffPos(name="realtime_pickreso_onset_stream")([mask, s_on_prev])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_onset_prev_next")(s_on_prev_n))

    s_on = int(round((RESBP_STRIKE_ENV_MS_REALTIME / 1000.0) * REALTIME_SAMPLE_RATE))
    L_env = int(np.clip(np.ceil(4.0 * max(1, s_on)), 8, 512))
    s_env = Input(shape=(L_env - 1, 1), name="realtime_state_pickreso_onset_smooth")
    state_inputs.append(s_env)
    k_on = _exp_smoothing_kernel(max(1, s_on), scale=4.0, Lmax=L_env)
    strike, s_env_n = StreamFixedFIR(kernel_len=L_env, out_ch=1, weight_name="exp_smooth_taps",
                                     trainable_kernel=False, taps_init=k_on,
                                     name="realtime_pickreso_onset_pulse")([onset, s_env])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_onset_smooth_next")(s_env_n))

    s_bp = Input(shape=(REALTIME_PICK_RESO_BP_ORDER - 1, 1), name="realtime_state_pickreso_ctrl_bp")
    state_inputs.append(s_bp)
    bp_lo = 0.9 * min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    bp_hi = 1.1 * max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    bp_k_full = _bp_minphase_rev_padded(REALTIME_SAMPLE_RATE, bp_lo, bp_hi, REALTIME_PICK_RESO_BP_ORDER)
    bp_out, s_bp_n = StreamFixedFIR(kernel_len=REALTIME_PICK_RESO_BP_ORDER, out_ch=1, weight_name="bp_taps",
                                    trainable_kernel=False, taps_init=bp_k_full,
                                    name="realtime_pickreso_ctrl_bp")([x_core, s_bp])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_bp_next")(s_bp_n))
    amp_raw = Abs1D()(bp_out)

    # 3 ms time constant in samples
    tau_env2 = max(1, int(round((3.0 / 1000.0) * REALTIME_SAMPLE_RATE)))
    # FIR length ≈ 3 × tau, clipped to [8, 256]
    s_env2_len = int(np.clip(np.ceil(3.0 * tau_env2), 8, 256))

    s_env2 = Input(shape=(s_env2_len - 1, 1), name="realtime_state_pickreso_ctrl_env")
    state_inputs.append(s_env2)
    k_env2 = _exp_smoothing_kernel(tau_env2, scale=4.0, Lmax=s_env2_len)
    amp, s_env2_n = StreamFixedFIR(kernel_len=s_env2_len, out_ch=1, weight_name="exp_smooth_taps",
                                trainable_kernel=False, taps_init=k_env2,
                                name="realtime_pickreso_ctrl_env")([amp_raw, s_env2])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ctrl_env_next")(s_env2_n))
    amp = Lambda(clip_01x4, name="realtime_pickreso_ctrl_gain")(amp)

    exc = Multiply(name="realtime_pickreso_excite")([strike, amp])

    # Same min-phase halving applies to the pickreso decimator kernel
    L_reso_ds = _minphase_decimator_len(REALTIME_PICK_RESO_DS_TAPS, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    s_ds8 = Input(shape=(L_reso_ds - 1, 1), name="realtime_state_pickreso_ds8")
    state_inputs.append(s_ds8)
    ds8, s_ds8_n = StreamingRealtimeIrDownsampleFIR(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                                    kernel_size=REALTIME_PICK_RESO_DS_TAPS,
                                                    use_ir_prototype=False,
                                                    name="realtime_pickreso_ds8")([exc, s_ds8])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_ds8_next")(s_ds8_n))

    num_modes = len(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    L_lr = _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    s_mlr = Input(shape=(L_lr - 1, 1), name="realtime_state_pickreso_modal_lr")
    state_inputs.append(s_mlr)
    modes_lr, s_mlr_n = StreamFixedFIR(kernel_len=L_lr, out_ch=num_modes, weight_name="modal_kernels_lr",
                                       trainable_kernel=True, name="realtime_pickreso_modal_lr")([ds8, s_mlr])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_modal_lr_next")(s_mlr_n))

    mixed_lr = Conv1D(1, 1, use_bias=True, padding="same", name="realtime_pickreso_mix")(modes_lr)

    s_interp = Input(shape=(_interp_polyphase_len(REALTIME_PICK_RESO_INTERP_TAPS, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR) - 1, 1),
                     name="realtime_state_pickreso_interp")
    state_inputs.append(s_interp)
    up, s_interp_n = StreamingFIRInterpByN(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                           num_taps=REALTIME_PICK_RESO_INTERP_TAPS,
                                           cutoff=0.45/REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
                                           window=('kaiser', 9.0),
                                           name="realtime_pickreso_interp")([mixed_lr, s_interp])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_interp_next")(s_interp_n))
    mixed = MatchTimeToRef(name="realtime_pickreso_matchlen")([up, inp])

    # NEW: gate with smoothed style envelope
    xg_reso = Multiply(name="realtime_pickreso_gated")([mixed, style_time])

    s_hp_reso = Input(shape=(PICK_RESO_LOWBAND_PENALTY_ORDER - 1, 1), name="realtime_state_pickreso_posthp")
    state_inputs.append(s_hp_reso)
    L_pr = int(PICK_RESO_LOWBAND_PENALTY_ORDER)
    hp_pr_k_full = _hp_minphase_rev_padded(REALTIME_SAMPLE_RATE, PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ, L_pr)
    xh_reso, s_hp_reso_n = StreamFixedFIR(kernel_len=L_pr, out_ch=1, weight_name="hp_taps",
                                          trainable_kernel=False, taps_init=hp_pr_k_full,
                                          name="realtime_pickreso_post_hp")([xg_reso, s_hp_reso])
    state_outputs.append(Activation('linear', name="realtime_state_pickreso_posthp_next")(s_hp_reso_n))
    xh_reso = SoftClip1D(drive=REALTIME_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickreso_softclip")(xh_reso)
    y_reso = IrScale(init=1.0, name="realtime_pickreso_scale")(xh_reso)
    y_reso = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0), name="realtime_pickreso_limit")(y_reso)
    y_reso = IrScale(init=1.0, name="realtime_pickreso_calib")(y_reso)

    # Mix stems and coloreq
    y_main_scaled = apply_runtime_gain(y_main, knob_main_in, name="realtime_main_user_gain")
    pickattack = apply_runtime_gain(y_pa, knob_pa_in, name="realtime_pickattack_user_gain")
    pickreso   = apply_runtime_gain(y_reso, knob_pr_in, name="realtime_pickreso_user_gain")
    ir         = apply_runtime_gain(ir,   knob_ir_in,   name="realtime_ir_user_gain")
    y_prehp  = Add(name="realtime_with_picklearn_prehp")([y_main_scaled, ir, pickattack, pickreso])

    # Color EQ streaming bandpasses
    coloreq_states_in  = []
    coloreq_states_out = []
    y_terms = [y_prehp]
    for bi, (lo, hi) in enumerate(COLOREQ_BANDS_REALTIME_BODY):
        s_b = Input(shape=(REALTIME_COLOREQ_ORDER - 1, 1), name=f"realtime_state_coloreq_bp_{bi}")
        coloreq_states_in.append(s_b)
        # taps_init MUST be given: the training graph's BandpassFIR weight has the min-phase
        # kernel's own length, so copy_weights_by_name would not transfer it into this streaming
        # FIR if the lengths ever differed. Without an initializer the streaming coloreq bands were
        # all-zero and any fine-tuned coloreq gain was silently lost in the exported models.
        bp_y, s_b_n = StreamFixedFIR(kernel_len=REALTIME_COLOREQ_ORDER, out_ch=1, weight_name="bp_taps",
                                     trainable_kernel=False,
                                     taps_init=_bp_minphase_rev_padded(REALTIME_SAMPLE_RATE, lo, hi, REALTIME_COLOREQ_ORDER),
                                     name=f"realtime_coloreq_bp_{bi}")([y_prehp, s_b])
        coloreq_states_out.append(Activation('linear', name=f"realtime_state_coloreq_bp_{bi}_next")(s_b_n))
        g = IrScale(init=0.0, name=f"realtime_coloreq_g_{bi}", trainable=bool(coloreq_trainable))(bp_y)
        y_terms.append(g)

    y_prehp = LossTether(name="realtime_pick_lowband_penalties_stream")([y_prehp])
    y_wet = Add(name="realtime_coloreq_sum")(y_terms)

    # Output/phase knobs (post dry/wet). The master EQ is a runtime effect (see master_eq_sos).
    knob_outgain = Input(shape=(1,), name="realtime_knob_outgain")
    knob_phaseinv= Input(shape=(1,), name="realtime_knob_phaseinv")

    wet_gain   = Lambda(invert, name="realtime_wet_from_drywet")(knob_dw_in)
    wet_scaled = apply_runtime_gain(y_wet,  wet_gain,    name="realtime_wet_gain")
    dry_scaled = apply_runtime_gain(inp,    knob_dw_in,  name="realtime_dry_gain")
    y_mix      = Add(name="realtime_drywet_mix")([wet_scaled, dry_scaled])

    # Phase invert + Output gain
    phase_gain = Lambda(phase_polarity_gain, name="realtime_phase_invert_gain")(knob_phaseinv)
    y_phi = apply_runtime_gain(y_mix, phase_gain, name="realtime_phase_invert")
    y_out = apply_runtime_gain(y_phi, knob_outgain, name="realtime_output_gain")

    # Dry passthrough (raw piezo, time-aligned)
    dry_out = Activation('linear', name="realtime_dry_out")(inp)

    inputs_list  = [inp, style_true_in, style_force_in, pickup_in,
                    knob_main_in, knob_pa_in, knob_pr_in, knob_ir_in, knob_dw_in,
                    knob_outgain, knob_phaseinv]
    inputs_list += state_inputs + coloreq_states_in

    # Style output is the EMA-smoothed probability (matches what drives FiLM), not the noisy
    # per-hop estimate — runtimes can consume it directly.
    outputs_list = [y_out, style_prob_ema, dry_out] + state_outputs + coloreq_states_out

    return tf.keras.models.Model(inputs=inputs_list, outputs=outputs_list, name=name)

def build_offline_debug_model(model):
    used = set([l.name for l in model.layers])

    def unique_alias(base):
        root = f"dbg_{base}"
        name = root
        k = 1
        while name in used:
            name = f"{root}_{k}"
            k += 1
        used.add(name)
        return name

    outs = []
    def add(alias, lname):
        try:
            t = model.get_layer(lname).output
        except Exception:
            return
        outs.append(Activation('linear', name=unique_alias(alias))(t))

    # Full mix AFTER dry/wet so stems 'full' respects knob
    add("y_pred",       "offline_drywet_mix")
    # Fallback:
    if not outs:
        add("y_pred",   "offline_with_picklearn_prehp")

    add("main_core",    "offline_main_core")
    add("main_pick",    "offline_pick_contrib")
    add("main_finger",  "offline_finger_contrib")
    add("pickreso",     "offline_pickreso_user_gain_mul")
    add("pickattack",   "offline_pickattack_user_gain_mul")
    add("style_pred",   "offline_style_prob")
    add("mask",         "offline_pick_mask_db")
    add("style_time",   "offline_style_time_used")

    return tf.keras.Model(inputs=model.inputs, outputs=outs, name="offline_debug_model")
