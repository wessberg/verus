import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Activation, Add, Conv1D, Dense, Lambda, Multiply, Reshape, SeparableConv1D, ZeroPadding1D)  # pyright: ignore[reportMissingImports]

from .config import (
    OFFLINE_GATE_ATTACK_MS,
    OFFLINE_GATE_HOLD_MS,
    OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ,
    OFFLINE_GATE_RATIO,
    OFFLINE_GATE_RELEASE_MS,
    OFFLINE_GATE_THRESHOLD_DB,
    OFFLINE_PICK_RESO_BP_ORDER,
    OFFLINE_PICK_RESO_SOFTCLIP_DRIVE,
    OFFLINE_SAMPLE_RATE,
    PICKUP_ADAPTER_DILATIONS,
    PICKUP_ADAPTER_FILTERS,
    PICKUP_ADAPTER_KERNEL,
    PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
    PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
    PICK_ATTACK_LOWBAND_PENALTY_ORDER,
    PICK_LIMITER_DRIVE,
    PICK_LIMITER_HEADROOM_DB,
    PICK_RESO_LOWBAND_PENALTY_ALPHA,
    PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
    PICK_RESO_LOWBAND_PENALTY_ORDER,
    PICK_RESO_MODAL_CENTERS_HZ_OFFLINE,
    PICK_RESO_MODAL_CENTERS_HZ_REALTIME,
    PICK_RESO_MODAL_DECAYS_MS_OFFLINE,
    PICK_RESO_MODAL_LEN_OFFLINE,
    PICK_RESO_MODAL_LEN_REALTIME,
    REALTIME_GATE_ATTACK_MS,
    REALTIME_GATE_HOLD_MS,
    REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ,
    REALTIME_GATE_RATIO,
    REALTIME_GATE_RELEASE_MS,
    REALTIME_GATE_THRESHOLD_DB,
    REALTIME_PICK_RESO_BP_ORDER,
    REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
    REALTIME_PICK_RESO_DS_TAPS,
    REALTIME_PICK_RESO_INTERP_TAPS,
    REALTIME_PICK_RESO_SOFTCLIP_DRIVE,
    REALTIME_SAMPLE_RATE,
    RESBP_STRIKE_ENV_MS_OFFLINE,
    RESBP_STRIKE_ENV_MS_REALTIME,
)
from .dsp import (
    _causal_conv1d,
    _exp_smoothing_kernel,
    _get_minphase_hp_kernel,
    _len_lr_from_full,
    _ones_kernel,
    gate_onset,
)
from .layers import (
    Abs1D,
    BandpassFIR,
    CausalExpSmoother,
    ConstScale,
    FIRInterpByN,
    HFHighpass,
    HFLowbandPenalty,
    IrScale,
    LearnedModalBank,
    LowrateModalBank,
    MatchTimeToRef,
    PickResoDownsampleFIR,
    SoftClip1D,
)

def _pick_attack(prefix,
                 hf_in,
                 mask,
                 style_time,
                 sample_rate,
                 filters=16,
                 softclip_drive=2.0,
                 lowband_penalty_alpha=PICK_ATTACK_LOWBAND_PENALTY_ALPHA,
                 lowband_penalty_cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                 lowband_penalty_order=PICK_ATTACK_LOWBAND_PENALTY_ORDER,
                 cutoff_hz=PICK_ATTACK_LOWBAND_PENALTY_CUTOFF_HZ,
                 core=(None, None)):
    """
    Tiny causal HF synthesis gated by the analytic pick mask, then style-gated,
    then a runtime HPF safety-net, then final scale. 'core' = loss core for the training-only
    lowband penalty (see *_LOSS_CORE).

    Returns:
      - y:    post-style, post-HPF, post-scale stem named '{prefix}_pickattack_scale'
      - patt: training-only lowband penalty attached to the post-scale stem
    """
    # Mask-gated input for the tiny CNN
    x = Multiply(name=f"{prefix}_picklearn_gate")([hf_in, mask])
    x = causal_sep_conv(x, filters, 7, 1, name=f"{prefix}_picklearn_c1", activation="relu")
    x = causal_sep_conv(x, filters, 7, 2, name=f"{prefix}_picklearn_c2", activation="relu")
    x = Conv1D(1, 1, padding="same", name=f"{prefix}_picklearn_out")(x)

    # Style gating
    xg = Multiply(name=f"{prefix}_pickattack_gated")([x, style_time])
    
    xg = SoftClip1D(drive=softclip_drive, limit=0.98, name=f"{prefix}_pickattack_softclip")(xg)

    # Runtime HPF safety-net (cut below penalty_cutoff_hz)
    xh = HFHighpass(sample_rate=sample_rate,
                    cutoff_hz=cutoff_hz,
                    order=lowband_penalty_order,
                    name=f"{prefix}_pickattack_post_hp")(xg)

    # Final scale (stems are POST-HPF as requested)
    y = IrScale(init=1.0, name=f"{prefix}_pickattack_scale")(xh)
    
    # Post-scale peak protection with ≥1 dB headroom
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE,
                   limit=limit_lin,
                   name=f"{prefix}_pickattack_limit")(y)

    # Training-only lowband penalty on the post-scale signal
    patt = HFLowbandPenalty(alpha=lowband_penalty_alpha,
                            cutoff_hz=lowband_penalty_cutoff_hz,
                            order=lowband_penalty_order,
                            sample_rate=sample_rate,
                            core_start=core[0], core_end=core[1],
                            name=f"{prefix}_pickattack_lowband_penalty")(y)
    return y, patt

def _reson_learn_offline(inp, mask, style_time, core=(None, None)):
    sr = OFFLINE_SAMPLE_RATE
    onset = Lambda(gate_onset, name="offline_pickreso_onset")(mask)
    strike = CausalExpSmoother(time_const_ms=RESBP_STRIKE_ENV_MS_OFFLINE, sample_rate=sr, Lmax=4096,
                               name="offline_pickreso_onset_pulse")(onset)
    ana = BandpassFIR(low_hz=0.9*min(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE),
                      high_hz=1.1*max(PICK_RESO_MODAL_CENTERS_HZ_OFFLINE),
                      order=OFFLINE_PICK_RESO_BP_ORDER, sample_rate=sr,
                      name="offline_pickreso_ctrl_bp")(inp)
    amp = CausalExpSmoother(time_const_ms=4.0, sample_rate=sr, Lmax=1024,
                            name="offline_pickreso_ctrl_env")(Abs1D()(ana))
    amp = Lambda(clip_01x4, name="offline_pickreso_ctrl_gain")(amp)
    exc = Multiply(name="offline_pickreso_excite")([strike, amp])

    modes = LearnedModalBank(centers_hz=PICK_RESO_MODAL_CENTERS_HZ_OFFLINE,
                             decays_ms =PICK_RESO_MODAL_DECAYS_MS_OFFLINE,
                             length    =PICK_RESO_MODAL_LEN_OFFLINE,
                             sample_rate=sr,
                             name="offline_pickreso_modal")(exc)
    mixed = Conv1D(1, 1, use_bias=True, padding="same", name="offline_pickreso_mix")(modes)

    # Style-gate first
    xg = Multiply(name="offline_pickreso_gated")([mixed, style_time])

    # Then runtime HPF safety-net at the requested cutoff
    xh = HFHighpass(sample_rate=sr,
                    cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                    order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                    name="offline_pickreso_post_hp")(xg)
    
    xh = SoftClip1D(drive=OFFLINE_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="offline_pickreso_softclip")(xh)

    # Final scale (POST-HPF)
    y = IrScale(init=1.0, name="offline_pickreso_scale")(xh)
    
    # Post-scale peak protection with ≥1 dB headroom
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE,
                   limit=limit_lin,
                   name="offline_pickreso_limit")(y)

    # Training-only lowband penalty on the post-scale stem
    patt = HFLowbandPenalty(alpha=PICK_RESO_LOWBAND_PENALTY_ALPHA,
                            cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                            order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                            sample_rate=OFFLINE_SAMPLE_RATE,
                            core_start=core[0], core_end=core[1],
                            name="offline_pickreso_lowband_penalty")(y)
    return y, patt

def _reson_learn_realtime(inp, mask, style_time, core=(None, None)):
    sr = REALTIME_SAMPLE_RATE
    onset = Lambda(gate_onset, name="realtime_pickreso_onset")(mask)
    strike = CausalExpSmoother(time_const_ms=RESBP_STRIKE_ENV_MS_REALTIME, sample_rate=sr, Lmax=512,
                               name="realtime_pickreso_onset_pulse")(onset)
    ana = BandpassFIR(low_hz=0.9*min(PICK_RESO_MODAL_CENTERS_HZ_REALTIME),
                      high_hz=1.1*max(PICK_RESO_MODAL_CENTERS_HZ_REALTIME),
                      order=REALTIME_PICK_RESO_BP_ORDER, sample_rate=sr,
                      name="realtime_pickreso_ctrl_bp")(inp)
    amp = CausalExpSmoother(time_const_ms=3.0, sample_rate=sr, Lmax=256,
                            name="realtime_pickreso_ctrl_env")(Abs1D()(ana))
    amp = Lambda(clip_01x4, name="realtime_pickreso_ctrl_gain")(amp)
    exc = Multiply(name="realtime_pickreso_excite")([strike, amp])

    # DS → low-rate modal bank → 1x1 mix → upsample → match T
    ds = PickResoDownsampleFIR(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_DS_TAPS,
                               name="realtime_pickreso_ds8")(exc)
    num_modes = len(PICK_RESO_MODAL_CENTERS_HZ_REALTIME)
    L_lr = _len_lr_from_full(PICK_RESO_MODAL_LEN_REALTIME, factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
    modes_lr = LowrateModalBank(num_modes=num_modes, length_lr=L_lr,
                                name="realtime_pickreso_modal_lr")(ds)
    mixed_lr = Conv1D(1, 1, use_bias=True, padding="same", name="realtime_pickreso_mix")(modes_lr)

    up = FIRInterpByN(factor=REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, num_taps=REALTIME_PICK_RESO_INTERP_TAPS,
                      cutoff=0.45/REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR, window=('kaiser', 9.0),
                      name="realtime_pickreso_interp")(mixed_lr)
    mixed = MatchTimeToRef(name="realtime_pickreso_matchlen")([up, inp])

    xg = Multiply(name="realtime_pickreso_gated")([mixed, style_time])
    xh = HFHighpass(sample_rate=sr,
                    cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                    order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                    name="realtime_pickreso_post_hp")(xg)
    xh = SoftClip1D(drive=REALTIME_PICK_RESO_SOFTCLIP_DRIVE, limit=0.98, name="realtime_pickreso_softclip")(xh)
    y = IrScale(init=1.0, name="realtime_pickreso_scale")(xh)

    # Post-scale peak protection
    limit_lin = 10.0 ** (-PICK_LIMITER_HEADROOM_DB / 20.0)
    y = SoftClip1D(drive=PICK_LIMITER_DRIVE, limit=limit_lin, name="realtime_pickreso_limit")(y)

    # NEW: calibration scale (migration will set its weight so knob dB → same loudness as legacy)
    y = IrScale(init=1.0, name="realtime_pickreso_calib")(y)

    patt = HFLowbandPenalty(alpha=PICK_RESO_LOWBAND_PENALTY_ALPHA,
                            cutoff_hz=PICK_RESO_LOWBAND_PENALTY_CUTOFF_HZ,
                            order=PICK_RESO_LOWBAND_PENALTY_ORDER,
                            sample_rate=REALTIME_SAMPLE_RATE,
                            core_start=core[0], core_end=core[1],
                            name="realtime_pickreso_lowband_penalty")(y)
    return y, patt

@register_keras_serializable()
def causal_sep_conv(x, filters, kernel, dilation, name, activation=None,
                    bias_constraint=None, bias_initializer=None):
    pad_left = (int(kernel) - 1) * int(dilation)
    x = ZeroPadding1D((pad_left, 0), name=f"{name}_pad")(x)
    x = SeparableConv1D(
        filters=filters,
        kernel_size=kernel,
        dilation_rate=dilation,
        padding="valid",
        bias_constraint=bias_constraint,
        bias_initializer=bias_initializer,
        name=name
    )(x)
    if activation:
        x = Activation(activation, name=f"{name}_{activation}")(x)
    return x

@register_keras_serializable()
def add_coloreq(x, bands, sample_rate, name_prefix, order=65, trainable_scales=False):
    """
    Post-EQ inside the base model graph:
      y = x + sum_i g_i * BP_i(x)
    - g_i initialized to 0.0 (identity). When trainable_scales=False, they remain frozen.
    """
    terms = [x]
    for i, (lo, hi) in enumerate(bands):
        bp = BandpassFIR(low_hz=lo, high_hz=hi, order=order, sample_rate=sample_rate,
                         name=f"{name_prefix}_bp_{i}")(x)
        terms.append(IrScale(init=0.0, name=f"{name_prefix}_g_{i}", trainable=trainable_scales)(bp))
    return Add(name=f"{name_prefix}_sum")(terms)

@register_keras_serializable()
def gate_gain_signal(
    x, sample_rate,
    threshold_db,
    ratio,
    attack_ms,
    hold_ms,
    release_ms,
    cutoff_hz,
    hp_order=129,
    min_gain_db=-60.0,
    env_ms=None,
    depth_smooth_ms=None,
    # NEW: allow per-call Lmax
    env_lmax=None,
    depth_lmax=None
):
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    kernel_hp = _get_minphase_hp_kernel(sample_rate, cutoff_hz, hp_order)
    sc = _causal_conv1d(x, kernel_hp)

    det_raw = tf.abs(sc) + 1e-12
    if env_ms is not None and env_ms > 0.0:
        env_smpl = max(1, int(round((float(env_ms) / 1000.0) * float(sample_rate))))
        k_env = _exp_smoothing_kernel(env_smpl, scale=2.0, Lmax=(env_lmax or 512))
        det = _causal_conv1d(det_raw, k_env)
    else:
        det = det_raw

    LN10 = tf.math.log(tf.constant(10.0, dtype=tf.float32))
    level_db = 20.0 / LN10 * tf.math.log(det)

    atk_s = max(1, int(round((float(attack_ms)  / 1000.0) * float(sample_rate))))
    rel_s = max(1, int(round((float(release_ms) / 1000.0) * float(sample_rate))))
    hld_s = max(0, int(round((float(hold_ms)    / 1000.0) * float(sample_rate))))

    thr_db  = tf.constant(float(threshold_db), dtype=tf.float32)
    ratio_f = tf.constant(float(ratio),        dtype=tf.float32)

    above = tf.cast(level_db >= thr_db, tf.float32)
    if hld_s > 0:
        k_hold = _ones_kernel(hld_s)
        pad = hld_s - 1
        above_pad = tf.pad(above, [[0,0],[pad,0],[0,0]])
        hold_sum = tf.nn.conv1d(above_pad, filters=k_hold, stride=1, padding='VALID')
        open_or_hold = tf.cast(hold_sum > 0.0, tf.float32)
    else:
        open_or_hold = above

    delta_db = tf.nn.relu(thr_db - level_db[...,0])
    attn_db  = tf.minimum(ratio_f * delta_db, tf.constant(120.0, tf.float32))
    ALPHA = tf.constant(np.log(10.0)/20.0, dtype=tf.float32)
    g_close_raw = tf.exp(-ALPHA * attn_db)[:, :, None]
    if min_gain_db is not None:
        floor = tf.constant(10.0 ** (float(min_gain_db)/20.0), dtype=tf.float32)
        g_close_raw = tf.maximum(g_close_raw, floor)

    if depth_smooth_ms is not None and depth_smooth_ms > 0.0:
        d_smpl = max(1, int(round((float(depth_smooth_ms) / 1000.0) * float(sample_rate))))
        k_depth = _exp_smoothing_kernel(d_smpl, scale=2.0, Lmax=(depth_lmax or 512))
        g_close_s = _causal_conv1d(g_close_raw, k_depth)
    else:
        g_close_s = g_close_raw

    k_att = _exp_smoothing_kernel(atk_s, scale=4.0, Lmax=2048)
    k_rel = _exp_smoothing_kernel(rel_s, scale=4.0, Lmax=2048)
    g_att_open = _causal_conv1d(open_or_hold, k_att)
    g_rel_open = _causal_conv1d(open_or_hold, k_rel)

    g_prev_open = tf.pad(open_or_hold, [[0,0],[1,0],[0,0]])[:, :-1, :]
    inc_open = tf.cast(open_or_hold > g_prev_open, tf.float32)
    g_env = inc_open * g_att_open + (1.0 - inc_open) * g_rel_open
    g_env = tf.clip_by_value(g_env, 0.0, 1.0)

    g = g_env + (1.0 - g_env) * g_close_s
    g = tf.clip_by_value(g, 0.0, 1.0)
    return g

@register_keras_serializable()
def gate_weight_offline(y_true):
    return gate_gain_signal(
        y_true,
        sample_rate=OFFLINE_SAMPLE_RATE,
        threshold_db=OFFLINE_GATE_THRESHOLD_DB,
        ratio=OFFLINE_GATE_RATIO,
        attack_ms=OFFLINE_GATE_ATTACK_MS,
        hold_ms=OFFLINE_GATE_HOLD_MS,
        release_ms=OFFLINE_GATE_RELEASE_MS,
        cutoff_hz=OFFLINE_GATE_HP_SIDECHAIN_LOW_CUT_HZ
    )

@register_keras_serializable()
def gate_weight_realtime(y_true):
    return gate_gain_signal(
        y_true,
        sample_rate=REALTIME_SAMPLE_RATE,
        threshold_db=REALTIME_GATE_THRESHOLD_DB,
        ratio=REALTIME_GATE_RATIO,
        attack_ms=REALTIME_GATE_ATTACK_MS,
        hold_ms=REALTIME_GATE_HOLD_MS,
        release_ms=REALTIME_GATE_RELEASE_MS,
        cutoff_hz=REALTIME_GATE_HP_SIDECHAIN_LOW_CUT_HZ
    )

@register_keras_serializable()
def gate_weight_dynamic(y_true, sample_rate):
    sr = int(sample_rate)
    if sr == OFFLINE_SAMPLE_RATE:
        return gate_weight_offline(y_true)
    elif sr == REALTIME_SAMPLE_RATE:
        return gate_weight_realtime(y_true)
    else:
        # default to realtime params if an unknown SR is used
        return gate_weight_realtime(y_true)

# Slice helpers
@register_keras_serializable()
def slice_content(x, filters):
    return x[:, :, :filters]

@register_keras_serializable()
def slice_gate(x, filters):
    return x[:, :, filters:]

@register_keras_serializable()
def slice_film_gamma(x, filters):
    return x[..., :filters]

@register_keras_serializable()
def slice_film_beta(x, filters):
    return x[..., filters:]

@register_keras_serializable()
def invert(x):
    return 1.0 - x

@register_keras_serializable()
def clip_01x4(z):
    # Scale by 4 and clamp to [0, 1]
    return tf.clip_by_value(4.0 * z, 0.0, 1.0)

@register_keras_serializable()
def film_gamma_stable(raw, scale=1.0, mag=0.1):
    # γ = 1 + mag * tanh(scale * raw)
    return 1.0 + float(mag) * tf.tanh(scale * raw)

@register_keras_serializable()
def film_beta_stable(raw, scale=1.0, mag=0.1):
    # β = mag * tanh(scale * raw)
    return float(mag) * tf.tanh(scale * raw)

def compute_pads(kernel, dilation, lookahead_samples):
    P_tot = (kernel - 1) * dilation
    pad_right = min(int(lookahead_samples), P_tot)
    pad_left  = P_tot - pad_right
    return pad_left, pad_right

@register_keras_serializable()
def glu_sep_block(x, filters, kernel, dilation, prefix, lookahead=0, residual_scale=0.35, bias_constraint=None):
    """
    Gated Linear Unit version of the residual block:
      y = res + s * PW( (DW(x))_content * sigmoid((DW(x))_gate) )
    where content is linear (no tanh), gate is sigmoid, and s<1 is a fixed residual scale.
    This greatly reduces harmonic generation vs. tanh.
    """
    res = x
    pad_left, pad_right = compute_pads(kernel=kernel, dilation=dilation, lookahead_samples=lookahead)
    x   = ZeroPadding1D((pad_left, pad_right), name=f"{prefix}_pad")(x)
    g   = SeparableConv1D(2*filters, kernel, dilation_rate=dilation, padding="valid",
                          bias_constraint=bias_constraint, name=f"{prefix}_sep")(x)
    c   = Lambda(slice_content, arguments={"filters": filters}, name=f"{prefix}_slice_c")(g)   # linear content
    g2  = Lambda(slice_gate,    arguments={"filters": filters}, name=f"{prefix}_slice_g")(g)   # gate logits
    x   = Multiply(name=f"{prefix}_gate_mul")([c, Activation("sigmoid", name=f"{prefix}_sigmoid")(g2)])
    x   = Conv1D(filters, 1, padding="same", bias_constraint=bias_constraint, name=f"{prefix}_pw")(x)
    # Residual scaling to keep the stack linear-ish and avoid runaway gain
    x   = ConstScale(residual_scale, name=f"{prefix}_res_scale")(x)
    x   = Add(name=f"{prefix}_res_add")([x, res])
    return x

def gated_sep_block(x, filters, kernel, dilation, prefix, lookahead=0, bias_constraint=None):
    res = x
    pad_left, pad_right = compute_pads(kernel=kernel, dilation=dilation, lookahead_samples=lookahead)
    x   = ZeroPadding1D((pad_left, pad_right), name=f"{prefix}_pad")(x)
    g   = SeparableConv1D(2*filters, kernel, dilation_rate=dilation, padding="valid",
                          bias_constraint=bias_constraint, name=f"{prefix}_sep")(x)
    c   = Lambda(slice_content, arguments={"filters": filters}, name=f"{prefix}_slice_c")(g)
    g2  = Lambda(slice_gate,    arguments={"filters": filters}, name=f"{prefix}_slice_g")(g)
    x   = Multiply(name=f"{prefix}_gate_mul")([Activation("tanh", name=f"{prefix}_tanh")(c),
                                               Activation("sigmoid", name=f"{prefix}_sigmoid")(g2)])
    x   = Conv1D(filters, 1, padding="same", bias_constraint=bias_constraint, name=f"{prefix}_pw")(x)
    x   = Add(name=f"{prefix}_res_add")([x, res])
    return x

def film_apply(x, style_embed, filters, prefix,
               gamma_scale=1.0, beta_scale=1.0,
               gamma_mag=0.1, beta_mag=0.1):
    film_params = Dense(filters*2, name=f"{prefix}_film_dense")(style_embed)
    gamma_raw = Lambda(slice_film_gamma, arguments={'filters': filters},
                       name=f"{prefix}_film_gamma_raw")(film_params)
    beta_raw  = Lambda(slice_film_beta,  arguments={'filters': filters},
                       name=f"{prefix}_film_beta_raw")(film_params)

    gamma = Lambda(film_gamma_stable,
                   arguments={'scale': float(gamma_scale), 'mag': float(gamma_mag)},
                   name=f"{prefix}_film_gamma")(gamma_raw)
    beta  = Lambda(film_beta_stable,
                   arguments={'scale': float(beta_scale), 'mag': float(beta_mag)},
                   name=f"{prefix}_film_beta")(beta_raw)

    gamma3 = Reshape((1, filters), name=f"{prefix}_film_gamma_expand")(gamma)
    beta3  = Reshape((1, filters), name=f"{prefix}_film_beta_expand")(beta)
    return Add(name=f"{prefix}_film_apply")([
        Multiply(name=f"{prefix}_film_scale")([x, gamma3]),
        beta3
    ])
    
def _pickup_adapter(inp, pickup_scalar, prefix):
    """
    Pickup-conditional input adapter: y = x + pickup * Delta(x).
    - Exact pass-through for piezo (pickup = 0).
    - The delta network's output conv is zero-initialized, so magnetic input also starts as a
      pass-through and learns the magnetic→piezo-domain correction during training.
    Fully causal; adds pickup_adapter_left_context() samples of left context to every branch fed
    from its output.
    """
    a = inp
    for i, d in enumerate(PICKUP_ADAPTER_DILATIONS):
        a = causal_sep_conv(a, PICKUP_ADAPTER_FILTERS, PICKUP_ADAPTER_KERNEL, d,
                            name=f"{prefix}_pickup_adapt_c{i+1}", activation="relu")
    a = Conv1D(1, 1, padding="same", kernel_initializer="zeros",
               name=f"{prefix}_pickup_adapt_out")(a)
    p3 = Reshape((1, 1), name=f"{prefix}_pickup_scalar_b3")(pickup_scalar)
    delta = Multiply(name=f"{prefix}_pickup_adapt_gate")([a, p3])
    return Add(name=f"{prefix}_pickup_adapt_add")([inp, delta])
