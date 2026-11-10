import math
import numpy as np
import os
import soundfile as sf
import tensorflow as tf

from .config import (
    KERAS_STREAM_HOP,
    LONG_DEMO_OFFSET_SEC,
    LONG_DEMO_PAIR_INDEX,
    LONG_DEMO_SEC,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_STATIC_MODEL_FILENAME,
    REALTIME_DYNAMIC_MODEL_FILENAME,
    REALTIME_IR_DOWNSAMPLE_FACTOR,
    REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR,
    REALTIME_SAMPLE_RATE,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_T,
    STYLE_FLAG_PROB_THRESHOLD,
)
from .contexts import (
    compute_realtime_streaming_context,
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
from .knobs import (
    prepare_demo_knob_feeds,
)
from .modelio import (
    _output_names_from_model,
    _tensor_basename,
    load_keras_model,
)

# ─────────────────────────────────────────────────────────────────────────────

class KerasStreamingRunner:
    """
    Hop-wise driver for a self-contained STREAMING realtime Keras model (static T or dynamic T).

    - discovers the realtime_state_* inputs and their *_next outputs (by producer-layer name,
      with an ordered fallback), zero-initialises them and carries them across hops;
    - steps the model through a tf.function (model.predict() per 32-sample hop costs ~10 ms of
      overhead each, i.e. ~45 min for a 3-minute file);
    - hop == T for static-T models (anything else corrupts the states); dynamic-T models are
      stepped KERAS_STREAM_HOP samples at a time. Style tracking is hop-invariant (see
      StreamingScalarEMA), and every decimator needs hop % 8 == 0;
    - optional output taps (by output index) are collected per hop, e.g. for stems;
    - the step runs on `device` (default CPU): these per-hop graphs are ~1500 tiny ops, and on
      tensorflow-metal the per-op launch latency made a 32-sample hop cost ~60 ms vs ~25 ms on
      the CPU (measured Sept 2026). Pass device=None to use TF's default placement.
    """
    def __init__(self, model, hop=None, device="/CPU:0"):
        self.model = model
        self.device = device
        self.in_names = [_tensor_basename(t) for t in model.inputs]
        piezo = next((t for t in model.inputs if _tensor_basename(t) == "realtime_piezo_input"), None)
        if piezo is None:
            raise RuntimeError("KerasStreamingRunner: model has no 'realtime_piezo_input'.")
        self.T = int(piezo.shape[1]) if piezo.shape[1] is not None else None
        if self.T is None:
            self.hop = int(hop or KERAS_STREAM_HOP)
        else:
            if hop is not None and int(hop) != self.T:
                raise ValueError(f"Streaming model with static T={self.T} requires hop == T (got {hop}).")
            self.hop = self.T
        f = max(REALTIME_IR_DOWNSAMPLE_FACTOR, REALTIME_PICK_RESO_DOWNSAMPLE_FACTOR)
        if self.hop % f != 0:
            raise ValueError(f"hop ({self.hop}) must be a multiple of the decimation factor ({f}).")

        self.state_names = [n for n in self.in_names if n.startswith("realtime_state_")]
        if not self.state_names:
            raise RuntimeError("KerasStreamingRunner: model is not STREAMING (no realtime_state_* inputs).")
        self.state_shapes = {}
        for t in model.inputs:
            nm = _tensor_basename(t)
            if nm in self.state_names:
                self.state_shapes[nm] = (1,
                                         int(t.shape[1]) if t.shape[1] is not None else 0,
                                         int(t.shape[2]) if t.shape[2] is not None else 1)
        self.out_names = _output_names_from_model(model)
        self.out_index = {n: i for i, n in enumerate(self.out_names)}
        self.state_out_idx = {n: self.out_index[n + "_next"] for n in self.state_names
                              if (n + "_next") in self.out_index}
        if len(self.state_out_idx) != len(self.state_names):
            # Ordered fallback: state outputs follow the non-state outputs, in state-input order
            n_non = len(model.outputs) - len(self.state_names)
            if n_non < 2:
                raise RuntimeError("KerasStreamingRunner: could not map state outputs.")
            self.state_out_idx = {n: n_non + i for i, n in enumerate(self.state_names)}
        self.states = None
        self.reset()

        def _step_fn(feed):
            if self.device:
                with tf.device(self.device):
                    return model(feed, training=False)
            return model(feed, training=False)
        self._step = tf.function(_step_fn, reduce_retracing=True)

    def reset(self):
        self.states = {n: np.zeros(s, dtype=np.float32) for n, s in self.state_shapes.items()}

    def run(self, x_mod, knob_feeds, style_true=None, taps=None):
        """
        Stream x_mod (1-D float array) through the model.
          style_true: None → classifier ('auto'); 0.0/1.0 → forced style.
          taps: {tag: output_index} of extra outputs to collect ((T,) or (T,C) per hop).
        Returns {'audio': (N,), 'avg_prob': float|None, 'taps': {tag: (N,...)}}.
        """
        self.reset()
        x_mod = np.asarray(x_mod, dtype=np.float32)
        N = len(x_mod)
        hop = self.hop
        if style_true is None:
            st_true = np.zeros((1, 1), np.float32); st_force = np.zeros((1, 1), np.float32)
        else:
            st_true = np.array([[float(style_true)]], np.float32); st_force = np.ones((1, 1), np.float32)
        const = {k: np.asarray(v, np.float32) for k, v in knob_feeds.items() if k in self.in_names}
        const["realtime_style_true_input"] = st_true
        const["realtime_style_force_input"] = st_force
        taps = taps or {}
        audio, tap_parts = [], {t: [] for t in taps}
        prob_sum, prob_count = 0.0, 0
        pos = 0
        while pos < N:
            core = min(hop, N - pos)
            xb = np.zeros((1, hop, 1), np.float32)
            xb[0, :core, 0] = x_mod[pos:pos + core]
            feed = dict(const)
            feed["realtime_piezo_input"] = xb
            feed.update(self.states)
            outs = self._step(feed)
            outs = [np.asarray(o) for o in outs]
            audio.append(outs[0][0, :core, 0])
            if style_true is None:
                prob_sum += float(outs[1][0, 0]) * core
                prob_count += core
            for tag, idx in taps.items():
                v = outs[idx]
                tap_parts[tag].append(v[0, :core, 0] if (v.ndim == 3 and v.shape[2] == 1) else v[0, :core])
            for n, idx in self.state_out_idx.items():
                self.states[n] = outs[idx]
            pos += core
        cat = lambda parts: np.concatenate(parts, axis=0) if parts else np.zeros((0,), np.float32)
        return {
            "audio": cat(audio),
            "avg_prob": (prob_sum / float(prob_count)) if prob_count > 0 else None,
            "taps": {t: cat(p) for t, p in tap_parts.items()},
        }

def _extract_audio_io(model, batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Unexpected dataset element")
    x = batch[0]
    y = batch[1]
    # Audio target
    if isinstance(y, (tuple, list)):
        y_true_audio = y[0]
    elif isinstance(y, dict):
        out_names = [t.name.split(':')[0].split('/')[-1] for t in model.outputs]
        y_true_audio = y[out_names[0]]
    else:
        y_true_audio = y

    # Feed inputs as a list in the model’s input order to avoid the KeyError
    if isinstance(x, dict):
        in_names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
        try:
            x = [x[k] for k in in_names]
        except KeyError:
            # Be tolerant to any prefixes (e.g., serving_default_) that might appear
            x_norm = {k.split(':')[0].split('/')[-1]: v for k, v in x.items()}
            missing = [k for k in in_names if k not in x_norm]
            if missing:
                raise KeyError(f"Batch is missing inputs required by the model: {missing}. "
                            f"Batch keys={list(x.keys())}, expected={in_names}")
            x = [x_norm[k] for k in in_names]

    y_pred_audio, _ = model(x, training=False)
    return y_true_audio, y_pred_audio


def _get_model_context_for_streaming(model_sample_rate: int, model=None):
    """
    Returns (left_ctx_model, right_ctx_model) in samples at the model's sample-rate.
    For realtime, if a model is provided, compute branch-aware left context automatically.
    """
    if model_sample_rate == OFFLINE_SAMPLE_RATE:
        # Left = sum of the per-layer left pads (+ colour EQ + input adapter),
        # Right = SUM of per-layer bounded look-aheads (additive across layers)
        return compute_streaming_ctx_constants("offline")
    elif model_sample_rate == REALTIME_SAMPLE_RATE:
        return compute_realtime_streaming_context(model, sample_rate=REALTIME_SAMPLE_RATE, verbose=True)
    else:
        # Fallback: treat as causal with tiny RF (safe default)
        return 1024, 0  # adjust if you add other sample-rates

def generate_demo_from_files(model,
                             kind,
                             out_dir,
                             sample_rate,
                             path_src,
                             path_target,
                             prediction_name="demo_pred",
                             only_prediction=False,
                             offset_sec=LONG_DEMO_OFFSET_SEC,
                             sec=LONG_DEMO_SEC,
                             block_sec=12.0,
                             resample_margin_ms=50.0,
                             use_finetuned=False,
                             pickup=None):
    os.makedirs(out_dir, exist_ok=True)
    knob_feeds = prepare_demo_knob_feeds(kind, use_finetuned=use_finetuned,
                                         pickup=pickup_flag_for_path(path_src) if pickup is None else pickup)
    style_true_flag = style_flag_for_path(path_src)

    if kind == "realtime":
        # Streaming graphs need their state inputs fed and carried hop-to-hop; a windowed
        # predict() call is not possible (Keras refuses a feed with missing inputs).
        runner = KerasStreamingRunner(model)
        x_mod, y_mod = _read_pair_segment(path_src, path_target if not only_prediction else None,
                                          sample_rate, offset_sec, sec)
        res = runner.run(x_mod, knob_feeds, style_true=None)
        style_pred_path = os.path.join(out_dir, prediction_name + "_style_pred.wav")
        sf.write(style_pred_path, res["audio"], sample_rate, subtype='FLOAT')
        if style_true_flag is not None:
            y_true = runner.run(x_mod, knob_feeds, style_true=style_true_flag)["audio"]
            sf.write(os.path.join(out_dir, prediction_name + "_style_true.wav"), y_true, sample_rate, subtype='FLOAT')
        if not only_prediction:
            sf.write(os.path.join(out_dir, "demo_source.wav"), x_mod, sample_rate, subtype='FLOAT')
            if y_mod is not None:
                sf.write(os.path.join(out_dir, "demo_target.wav"), y_mod, sample_rate, subtype='FLOAT')
        p = res["avg_prob"]
        print(f"✅ Demo saved to {style_pred_path}. Predicted style (avg): {_style_label(p)} "
              f"({p:.4f})" if p is not None else f"✅ Demo saved to {style_pred_path}.")
        return

    info = sf.info(path_src)
    file_sr = info.samplerate
    START_FRAME = int(round(offset_sec * file_sr))
    TOTAL_FRAMES = int(round(sec * file_sr))
    END_FRAME = START_FRAME + TOTAL_FRAMES

    g_norm = 1.0

    left_ctx_model, right_ctx_model = _get_model_context_for_streaming(sample_rate, model=model)
    resample_margin_file = int(round((resample_margin_ms / 1000.0) * file_sr))
    block_core_model = max(1, int(round(block_sec * sample_rate)))
    want_total_model = int(round(TOTAL_FRAMES * (sample_rate / float(file_sr))))

    style_pred_path = os.path.join(out_dir, prediction_name + "_style_pred.wav")
    style_true_path = os.path.join(out_dir, prediction_name + "_style_true.wav")

    w_pred = sf.SoundFile(style_pred_path, mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_true = None
    if style_true_flag is not None:
        w_true = sf.SoundFile(style_true_path, mode='w', samplerate=sample_rate, channels=1, subtype='FLOAT')

    w_src = None
    w_tgt = None
    if not only_prediction:
        w_src = sf.SoundFile(os.path.join(out_dir, "demo_source.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')
        w_tgt = sf.SoundFile(os.path.join(out_dir, "demo_target.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')

    produced_model = 0
    prob_sum = 0.0
    prob_count = 0

    # Helper: filter a feed dict to the model’s actual input names
    model_input_names = [t.name.split(':')[0].split('/')[-1] for t in model.inputs]
    def filter_feed(d):
        return {k: v for k, v in d.items() if k in model_input_names}

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
        x_file = x_block_2d[:, 0] * g_norm
        if not only_prediction:
            y_block_2d, _ = sf.read(path_target, start=read_a, frames=n_read, dtype='float32', always_2d=True)
            y_file = y_block_2d[:, 0] * g_norm

        if file_sr != sample_rate:
            x_resamp = resample_to_model_sr(x_file, file_sr, sample_rate)
            if not only_prediction:
                y_resamp = resample_to_model_sr(y_file, file_sr, sample_rate)
        else:
            x_resamp = x_file
            if not only_prediction:
                y_resamp = y_file

        left_off_file = (core_start_file - left_ctx_file) - read_a
        left_off_model = int(round(left_off_file * (sample_rate / float(file_sr))))
        inp_start = max(0, min(left_off_model, len(x_resamp)))
        inp_end = max(inp_start, min(inp_start + left_this + core_model + right_this, len(x_resamp)))
        x_inp = x_resamp[inp_start:inp_end]
        if x_inp.size == 0:
            break

        core_start_in_inp = min(left_this, len(x_inp))
        core_end_in_inp = min(core_start_in_inp + core_model, len(x_inp))
        core_len = max(0, core_end_in_inp - core_start_in_inp)
        if core_len <= 0:
            break

        x_demo = x_inp[None, :, None].astype(np.float32)

        # Predicted-style pass with knobs
        dummy_style = np.zeros((1, 1), dtype=np.float32)
        force_zero  = np.zeros((1, 1), dtype=np.float32)

        base_feed = {
            f"{kind}_piezo_input": x_demo,
            f"{kind}_style_true_input": dummy_style,
            f"{kind}_style_force_input": force_zero,
        }
        base_feed.update(knob_feeds)
        feed = filter_feed(base_feed)

        outs_pred = model.predict(feed, batch_size=1, verbose=0)
        if not isinstance(outs_pred, (list, tuple)):
            outs_pred = [outs_pred]
        y_inf_pred = np.asarray(outs_pred[0], np.float32)[0, :, 0]
        style_pred_out = np.asarray(outs_pred[1], np.float32)
        y_core_pred = y_inf_pred[core_start_in_inp:core_end_in_inp]
        w_pred.write(y_core_pred)

        prob_sum += float(style_pred_out[0, 0]) * float(core_len)
        prob_count += core_len

        # True-style pass (if known)
        if w_true is not None:
            style_true = np.array([[style_true_flag]], dtype=np.float32)
            force_one  = np.ones((1, 1), dtype=np.float32)
            true_feed = {
                f"{kind}_piezo_input": x_demo,
                f"{kind}_style_true_input": style_true,
                f"{kind}_style_force_input": force_one,
            }
            true_feed.update(knob_feeds)
            feed_true = filter_feed(true_feed)
            outs_true = model.predict(feed_true, batch_size=1, verbose=0)
            if not isinstance(outs_true, (list, tuple)):
                outs_true = [outs_true]
            y_inf_true = np.asarray(outs_true[0], np.float32)[0, :, 0]
            y_core_true = y_inf_true[core_start_in_inp:core_end_in_inp]
            w_true.write(y_core_true)

        if not only_prediction:
            core_start_global = inp_start + core_start_in_inp
            core_end_global = inp_start + core_end_in_inp
            x_core = x_resamp[core_start_global:core_end_global]
            y_core = y_resamp[core_start_global:core_end_global]
            w_src.write(x_core)
            w_tgt.write(y_core)

        produced_model += core_len

    w_pred.close()
    if w_true is not None:
        w_true.close()
    if w_src is not None:
        w_src.close()
    if w_tgt is not None:
        w_tgt.close()

    if prob_count > 0:
        p = prob_sum / float(prob_count)
        style_pred_name = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ Demo saved to {style_pred_path}. Predicted style (avg): {style_pred_name} ({p:.4f})")
        if w_true is not None:
            style_true_name = 'pick' if style_true_flag >= STYLE_FLAG_PROB_THRESHOLD else 'finger'
            print(f"✅ Demo saved to {style_true_path}. True style: {style_true_name}")
    else:
        print(f"✅ Demo saved to {style_pred_path}.")
def generate_demo(model, kind, out_dir, sample_rate, prediction_name="demo_pred", only_prediction=False):
    """
    Offline: unchanged (windowed context).
    Realtime: requires a self-contained STREAMING model with fixed T=32.
              This function never builds any graph; it throws if the model is not streaming or not T=32.
    """
    assert_has_pairs()
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]
    os.makedirs(out_dir, exist_ok=True)

    # Offline: windowed dynamic graph. Realtime: streaming graph (static T=32 → hop 32, exactly
    # like the pedal; dynamic T → KERAS_STREAM_HOP), both via KerasStreamingRunner.
    return generate_demo_from_files(
        model           = model,
        kind            = kind,
        out_dir         = out_dir,
        sample_rate     = sample_rate,
        path_src        = pth_src,
        path_target     = pth_tgt,
        prediction_name = prediction_name,
        only_prediction = only_prediction,
        offset_sec      = LONG_DEMO_OFFSET_SEC,
        sec             = LONG_DEMO_SEC
    )
        
def prepare_and_generate_demo(kind, sample_rate, dynamic_model_filepath, demo_out_dir, prefer_static=False):
    """
    For demos, only load saved artifacts:
      - offline: dynamic offline model (non-streaming) by default; if prefer_static=True and static exists, use it
      - realtime: static T=32 streaming model (self-contained) is preferred; prefer_static=True is recommended
    This function never builds graphs; it loads saved .keras models and runs generate_demo.
    """
    # Prefer saved artifacts by kind
    prefer = None
    if kind == "offline":
        if prefer_static:
            cand = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
            if os.path.exists(cand):
                prefer = cand
        if prefer is None:
            cand_dyn = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
            if os.path.exists(cand_dyn):
                prefer = cand_dyn

    elif kind == "realtime":
        # Realtime demo: by default the DYNAMIC streaming model, stepped KERAS_STREAM_HOP samples
        # at a time (~30x faster; hop-invariant style EMA, verified identical to 5e-5). With
        # --static (prefer_static) the static T=32 model is stepped exactly like the pedal.
        cand_static = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
        cand_dyn    = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
        if prefer_static and os.path.exists(cand_static):
            prefer = cand_static
        elif os.path.exists(cand_dyn):
            prefer = cand_dyn
        elif os.path.exists(cand_static):
            prefer = cand_static
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

    dpath = prefer if prefer is not None else dynamic_model_filepath
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"[demo] Missing model for kind='{kind}'. Expected at: {dpath}. "
                                f"For realtime, train/export the static T={REALTIME_TFLITE_T} streaming model first.")
    model = load_keras_model(dpath, compile=False)
    model.summary()

    generate_demo(model=model, kind=kind, out_dir=demo_out_dir, sample_rate=sample_rate)

def _style_label(p):
    if p is None:
        return "unknown"
    return "pick" if p >= STYLE_FLAG_PROB_THRESHOLD else "finger"
