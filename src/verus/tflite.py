import numpy as np
import os
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter

from .config import (
    FINETUNE_OFFLINE_TFLITE_FP32_FILENAME,
    FINETUNE_REALTIME_TFLITE_FP32_FILENAME,
    LONG_DEMOS_OFFLINE_OUT_DIR,
    LONG_DEMOS_REALTIME_OUT_DIR,
    LONG_DEMO_OFFSET_SEC,
    LONG_DEMO_PAIR_INDEX,
    LONG_DEMO_SEC,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_TFLITE_FP32_FILENAME,
    OFFLINE_TFLITE_T,
    REALTIME_SAMPLE_RATE,
    REALTIME_TFLITE_FP32_FILENAME,
    REALTIME_TFLITE_T,
    STYLE_FLAG_PROB_THRESHOLD,
)
from .contexts import (
    compute_streaming_ctx_constants,
)
from .data import (
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

def _select_tflite_demo_paths(kind, use_finetuned=False):
    if kind == "offline":
        if use_finetuned:
            return os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_TFLITE_FP32_FILENAME)
        return os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_TFLITE_FP32_FILENAME)
    elif kind == "realtime":
        if use_finetuned:
            return os.path.join(
                MODELS_REALTIME_OUT_DIR,
                FINETUNE_REALTIME_TFLITE_FP32_FILENAME
            )
        return os.path.join(
            MODELS_REALTIME_OUT_DIR,
            REALTIME_TFLITE_FP32_FILENAME
        )
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")

def _tflite_demo_blockwise(kind, tflite_runner, sample_rate, x_mod, y_mod=None,
                           style_true_flag=None, use_finetuned=False, out_dir=".",
                           fixed_hop=None, pickup=None):
    """
    TFLite demo runner.
    - Streaming: T=hop, left=0, right=0; TFLiteRunner keeps states.
    - Otherwise: windowed schedule.
    """
    os.makedirs(out_dir, exist_ok=True)

    if kind == "realtime" and tflite_runner.is_streaming() and not tflite_runner.has_bundle():
        print("⚠️  [TFLite/demo] This realtime TFLite does not expose 'realtime_states_bundle_out' by name; "
              "the runner will try to find the bundle by shape. If you still hear artifacts, re-run --export-tflite.")

    piezo_key = f"{kind}_piezo_input"
    T_fixed = tflite_runner.input_length_T(piezo_key)

    is_stream = tflite_runner.is_streaming()

    if fixed_hop is not None:
        core_len = int(fixed_hop)
        if core_len < 1 or core_len > T_fixed:
            raise ValueError(f"fixed_hop must be in [1, {T_fixed}], got {fixed_hop}")
        if is_stream and core_len != T_fixed:
            # See _tflite_schedule: hop < T pushes zero-padding into the streaming states.
            raise ValueError(
                f"Streaming models require hop == T ({T_fixed}); got hop={core_len}. "
                f"A smaller hop corrupts the streaming states.")
    else:
        core_len = T_fixed if is_stream else max(1, min(T_fixed // 2, 8192 if kind == "offline" else 512))

    if is_stream:
        print(f"[TFLite/demo] STREAMING: T={T_fixed}, hop={core_len}, left=0, right=0")
        left_this, right_this = 0, 0
    else:
        lc_m, rc_m = compute_streaming_ctx_constants(kind)
        left_this  = min(int(lc_m), max(0, T_fixed - 1))
        right_this = min(int(rc_m), max(0, T_fixed - 1 - left_this))
        if (T_fixed - left_this - right_this) < core_len:
            need = core_len - (T_fixed - left_this - right_this)
            r_dec = min(right_this, need); right_this -= r_dec; need -= r_dec
            if need > 0:
                l_dec = min(left_this, need); left_this -= l_dec
        print(f"[TFLite/demo] WINDOWED: T={T_fixed}, left={left_this}, core={core_len}, right={right_this}")

    knob_feeds = prepare_demo_knob_feeds(kind=kind, use_finetuned=use_finetuned, pickup=pickup)
    
    # DEBUG: isolate main path only (no IR, no pick branches; fully wet)
    #if kind == "realtime":
        #knob_feeds["realtime_knob_ir"]         = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_pickattack"] = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_pickreso"]   = np.array([[0.0]], dtype=np.float32)
        #knob_feeds["realtime_knob_drywet"]     = np.array([[0.0]], dtype=np.float32)
    
    style_pred_path = os.path.join(out_dir, "demo_pred_style_pred_tflite.wav")
    style_true_path = os.path.join(out_dir, "demo_pred_style_true_tflite.wav")

    w_src = sf.SoundFile(os.path.join(out_dir, "demo_source.wav"), mode='w',
                         samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_tgt = None
    if y_mod is not None:
        w_tgt = sf.SoundFile(os.path.join(out_dir, "demo_target.wav"), mode='w',
                             samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_pred = sf.SoundFile(style_pred_path, mode='w',
                          samplerate=sample_rate, channels=1, subtype='FLOAT')
    w_true = None
    if style_true_flag is not None:
        w_true = sf.SoundFile(style_true_path, mode='w',
                              samplerate=sample_rate, channels=1, subtype='FLOAT')

    pos = 0
    prob_sum = 0.0
    prob_count = 0
    N = len(x_mod)

    # Helper to run one pass (predicted style or forced style)
    def run_pass(force_style=None):
        nonlocal pos, prob_sum, prob_count

        # Reset globals
        pos = 0
        if tflite_runner.is_streaming():
            tflite_runner.reset_states()

        while pos < N:
            core = min(core_len, N - pos)

            x_in = np.zeros((1, T_fixed, 1), dtype=np.float32)
            if is_stream:
                # stream: fill the first 'core' samples, leave the rest zeros
                x_in[0, :core, 0] = x_mod[pos:pos + core]
            else:
                # windowed: left/core/right fill
                ltake = min(left_this, pos)
                rtake = min(right_this, N - (pos + core))
                if ltake > 0:
                    x_in[0, left_this - ltake:left_this, 0] = x_mod[pos - ltake:pos]
                x_in[0, left_this:left_this + core, 0] = x_mod[pos:pos + core]
                if rtake > 0:
                    xs = pos + core; xe = xs + rtake
                    x_in[0, left_this + core:left_this + core + rtake, 0] = x_mod[xs:xe]

            base_feed = {k: v.astype(np.float32) for k, v in knob_feeds.items()}
            base_feed[f"{kind}_piezo_input"] = x_in

            if force_style is None:
                base_feed[f"{kind}_style_true_input"]  = np.zeros((1,1), np.float32)
                base_feed[f"{kind}_style_force_input"] = np.zeros((1,1), np.float32)
            else:
                base_feed[f"{kind}_style_true_input"]  = np.array([[float(force_style)]], dtype=np.float32)
                base_feed[f"{kind}_style_force_input"] = np.ones((1,1), dtype=np.float32)

            y_pred, s_pred = tflite_runner.invoke(base_feed)

            if is_stream:
                y_core = y_pred[0, :core, 0]
            else:
                y_core = y_pred[0, left_this:left_this + core, 0]

            if force_style is None:
                w_pred.write(y_core)
                prob_sum   += float(np.asarray(s_pred, dtype=np.float32)[0, 0]) * float(core)
                prob_count += core
                # Write src/tgt only on the predicted-style pass; writing on every pass would
                # duplicate the reference files' content.
                w_src.write(x_mod[pos:pos + core])
                if w_tgt is not None:
                    w_tgt.write(y_mod[pos:pos + core])
            else:
                w_true.write(y_core)

            pos += core

    # Predicted-style pass
    run_pass(force_style=None)
    # Forced-style pass if we have a ground truth style
    if style_true_flag is not None:
        run_pass(force_style=style_true_flag)

    w_src.close()
    if w_tgt is not None: w_tgt.close()
    w_pred.close()
    if w_true is not None: w_true.close()

    if prob_count > 0:
        p = prob_sum / float(prob_count)
        label = 'pick' if p > STYLE_FLAG_PROB_THRESHOLD else 'finger'
        print(f"✅ TFLite demo written to {style_pred_path}. Predicted style (avg): {label} ({p:.4f})")
    
class TFLiteRunner:
    """
    Robust TFLite runner for ai_edge_litert:
      - Uses a single packed bundle + splits (lengths) to update states.
      - Finds bundle and splits by SHAPE (not by name), so it survives TFLite's generic output names.
      - Splits order: lexicographic state-input name order, independent of TFLite input reordering.
      - Falls back to declared vs lex-sorted order if splits are absent.
      - Ensures the 3D piezo block is fed even if input names are generic.
      - Throttles noisy warnings.
      - Deep-copies outputs to avoid internal buffer aliasing.
    """
    def __init__(self, model_path, debug=True):
        self.debug = bool(debug)
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()

        # Cache IO
        self.inp_details = self.interp.get_input_details()
        self.out_details = self.interp.get_output_details()
        self._in_details_by_idx  = {d["index"]: d for d in self.inp_details}
        self._out_details_by_idx = {d["index"]: d for d in self.out_details}

        # Name maps (normalized) for inputs only; outputs have generic names, so we avoid name maps for them.
        self.in_map = { self._norm_name_input(d["name"]): d["index"] for d in self.inp_details }
        self._in_details_by_name  = { self._norm_name_input(d["name"]): d for d in self.inp_details }

        # Kind
        self._kind = "realtime" if any("realtime_piezo_input" in k for k in self.in_map.keys()) else (
                     "offline"  if any("offline_piezo_input"  in k for k in self.in_map.keys()) else "unknown")

        # Streaming discovery
        self._discover_stream_states()

        # Expected bundle length = sum(L*C) over state inputs (from TFLite input shapes)
        self._expected_bundle_len = 0
        for nm in self.state_in_names:
            dinfo = self._in_details_by_name.get(nm, None)
            if dinfo is None:
                continue
            shp = tuple(int(x) for x in dinfo["shape"])
            L = shp[1] if len(shp) > 1 else 0
            C = shp[2] if len(shp) > 2 else 1
            self._expected_bundle_len += (L * C)

        # Shape-based discovery for bundle and splits
        # Bundle: 2D output with largest total length (excluding (1,1))
        # Splits: 2D output (1, N_states) or (N_states,) if TFLite flattens, prefer exact row-major (1,N)
        self._shape_bundle_candidate_name = None
        self._shape_splits_candidate_name = None
        self._shape_bundle_len = 0
        self._shape_splits_len = 0
        self._shape_splits_count = 0

        # Evaluate outputs by shape without relying on names
        for d in self.out_details:
            nm = str(d["name"])  # keep raw (unique)
            shp = tuple(int(x) for x in d["shape"])
            if len(shp) == 2:
                if shp != (1, 1):
                    total = int(shp[0] * shp[1])
                    # Candidate for bundle: pick the largest 2D (excluding style)
                    if total > self._shape_bundle_len:
                        self._shape_bundle_len = total
                        self._shape_bundle_candidate_name = nm
                # Candidate for splits: (1, N_states)
                if shp[0] == 1 and shp[1] == len(self.state_in_names):
                    self._shape_splits_candidate_name = nm
                    self._shape_splits_len = int(shp[1])
                    self._shape_splits_count = int(shp[1])
            elif len(shp) == 1:
                # Sometimes TFLite can collapse (1,N) to (N,), accept that too for splits
                if shp[0] == len(self.state_in_names) and self._shape_splits_candidate_name is None:
                    self._shape_splits_candidate_name = nm
                    self._shape_splits_len = int(shp[0])
                    self._shape_splits_count = int(shp[0])

        # Fallback ordered indices: [audio, style] + states... (very rare)
        self._ordered_state_out_idx = {}
        if self._is_streaming:
            for i, nm in enumerate(self.state_in_names):
                self._ordered_state_out_idx[nm] = 2 + i

        # Precompute lex-sorted name list for robust splitting
        self.state_in_names_sorted = sorted(list(self.state_in_names))

        # Dynamic decision for bundle splitting order ('declared' or 'sorted')
        self._bundle_order_mode = None

        # Throttled warnings
        self._warn_counts = {}
        self._piezo_fallback_logged = False

        self._print_io_summary()

        # Quick self test (zero-input)
        try:
            self._quick_self_test()
        except Exception as e:
            print(f"[TFLiteRunner] Self-test skipped: {e}")

    # ------------- helpers -------------
    def _norm_name_input(self, n):
        """
        Normalize input tensor names:
          - Strip ':0' suffix
          - Drop 'serving_default_' prefix
          - Remove TF scope prefixes ending with '/'
        """
        n = str(n)
        # Strip ':0' or ':<num>'
        if ":" in n:
            n = n.split(":", 1)[0]
        if n.startswith("serving_default_"):
            n = n[len("serving_default_"):]
        # Remove any leading scopes like 'StatefulPartitionedCall/...'
        if "/" in n:
            n = n.split("/")[-1]
        return n

    def _discover_stream_states(self):
        # Input names in interpreter-declared order (normalized)
        in_order_names = [ self._norm_name_input(d["name"]) for d in self.inp_details ]
        self.state_in_names = [n for n in in_order_names if n.startswith("realtime_state_")]

        # Name-based output map (unreliable with generic TFLite names) → leave empty
        self.state_out_map = {}
        self._is_streaming = len(self.state_in_names) > 0

        # Allocate zero states by input shapes
        self._state = {}
        self._state_shapes = {}
        for nm in self.state_in_names:
            dinfo = self._in_details_by_name.get(nm, None)
            if dinfo is None:
                continue
            shp = tuple(int(x) for x in dinfo["shape"])
            if len(shp) != 3:
                shp = (1, shp[1] if len(shp) > 1 else 0, 1)
            self._state[nm] = np.zeros(shp, dtype=np.float32)
            self._state_shapes[nm] = shp

    def _print_io_summary(self, max_layer_names=48):
        print("== TFLite IO Summary ==")
        print(f"  kind: {self._kind}")
        print(f"  inputs ({len(self.inp_details)}):")
        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            shp = tuple(int(x) for x in d["shape"])
            print(f"    {nm:<48} shape={shp}")
        print(f"  outputs ({len(self.out_details)}):")
        for d in self.out_details:
            nm = str(d["name"])
            shp = tuple(int(x) for x in d["shape"])
            print(f"    {nm:<48} shape={shp}")
        print(f"  streaming: {self._is_streaming} (states={len(self.state_in_names)})")
        if self._is_streaming:
            bundle_note = "present(by shape)" if self._shape_bundle_candidate_name is not None else "absent"
            splits_note = "present(by shape)" if self._shape_splits_candidate_name is not None else "absent"
            print(f"  expected bundle length: {self._expected_bundle_len}")
            print(f"  packed bundle: {bundle_note}")
            print(f"  splits vector: {splits_note}")
            print("  state inputs (in declared order):")
            for nm in self.state_in_names[:max_layer_names]:
                shp = tuple(int(x) for x in self._in_details_by_name[nm]["shape"])
                print(f"    {nm:<48} shape={shp}")
            if len(self.state_in_names) > max_layer_names:
                print(f"    ... {len(self.state_in_names)-max_layer_names} more")
        print("========================")

    def is_streaming(self):
        return bool(self._is_streaming)

    def has_bundle(self):
        # A bundle is considered present if shape-based candidate exists
        return bool(self._shape_bundle_candidate_name)

    def reset_states(self):
        for k in self._state.keys():
            self._state[k].fill(0.0)

    def input_length_T(self, piezo_key):
        # Prefer named piezo input if present
        if piezo_key in self.in_map:
            shp = self._in_details_by_name[piezo_key]["shape"]
            if len(shp) >= 2:
                return int(shp[1])
        # Fallback: any 3D input
        for d in self.inp_details:
            shp = d["shape"]
            if len(shp) == 3 and shp[1] > 0:
                return int(shp[1])
        # Last resort: first input second dim
        return int(self.inp_details[0]["shape"][1])

    def _piezo_T_from_feed(self, feed):
        for k, v in feed.items():
            # feed keys already normalized by caller (we normalize below too)
            nk = self._norm_name_input(k)
            if "piezo_input" in nk:
                arr = np.asarray(v)
                if arr.ndim >= 2:
                    return int(arr.shape[1])
        for v in feed.values():
            arr = np.asarray(v)
            if arr.ndim == 3:
                return int(arr.shape[1])
        return None

    def _normalize_audio_style(self, audio, style):
        a = np.array(audio, dtype=np.float32, copy=True)
        s = np.array(style, dtype=np.float32, copy=True)
        if a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        elif a.ndim >= 4:
            a = a.reshape((a.shape[0], a.shape[1], -1))[:, :, :1]
        if s.ndim == 0:
            s = s.reshape((1, 1))
        elif s.ndim == 1:
            s = s.reshape((1, -1))[:, :1]
        elif s.ndim >= 3:
            s = s.reshape((s.shape[0], -1))[:, :1]
        return a, s

    def _select_audio_style(self, outs_list, T_cur):
        # Prefer 3D as audio (by name if the export kept names, else match T), 2D(1,1) as style
        three_d = [(n,a) for (n,a) in outs_list if np.asarray(a).ndim == 3]
        two_d   = [(n,a) for (n,a) in outs_list if np.asarray(a).ndim == 2]

        audio = None
        if three_d:
            named = [t for t in three_d if "audio_out" in str(t[0])]
            cand = named or [t for t in three_d if (T_cur is None or int(np.asarray(t[1]).shape[1]) == int(T_cur))]
            if len(three_d) > 1 and not named:
                self._warn_throttled('ambiguous_3d',
                    f"⚠️  [TFLiteRunner] {len(three_d)} 3D outputs and none named '*audio_out' — "
                    "audio selection is ambiguous; re-run --export-tflite.")
            audio = (cand or three_d)[0][1]
        else:
            # Fallback to largest tensor
            audio = max(outs_list, key=lambda t: np.asarray(t[1]).size)[1]

        style = None
        for (n,a) in two_d:
            arr = np.asarray(a)
            if arr.shape == (1,1):
                style = arr
                break
        if style is None:
            style = np.zeros((1,1), dtype=np.float32)
        return audio, style

    def _pick_bundle_from_outs(self, outs_list):
        # Choose 2D float32 output with largest length, excluding (1,1)
        best = None
        best_len = 0
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if arr.ndim == 2 and arr.dtype == np.float32 and tuple(arr.shape) != (1, 1):
                total = int(np.prod(arr.shape))
                if total > best_len:
                    best_len = total
                    best = arr
        # Fallback: any 2D (non (1,1))
        if best is None:
            for (n, a) in outs_list:
                arr = np.asarray(a)
                if arr.ndim == 2 and tuple(arr.shape) != (1, 1):
                    total = int(np.prod(arr.shape))
                    if total > best_len:
                        best_len = total
                        best = arr
        return best

    def _pick_splits_from_outs(self, outs_list):
        # Choose a (1,N_states) or (N_states,) integer vector; cast to int32
        target = len(self.state_in_names)
        # Prefer (1, target) int
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if arr.ndim == 2 and arr.shape == (1, target) and np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int32, copy=False)
        # Fallback: (target,) or (target,1) int
        for (n, a) in outs_list:
            arr = np.asarray(a)
            if ((arr.ndim == 1 and arr.shape[0] == target) or
                (arr.ndim == 2 and arr.shape == (target, 1))) and np.issubdtype(arr.dtype, np.integer):
                return arr.reshape(1, -1).astype(np.int32, copy=False)
        return None

    def _warn_throttled(self, key, msg, first_n=3, every=200):
        c = self._warn_counts.get(key, 0)
        self._warn_counts[key] = c + 1
        if c < first_n or (every > 0 and (c % every) == 0):
            print(msg)

    def _try_update_bundle_with_order(self, names_order, bundle_vec, lengths=None):
        """
        Attempt to update states using a given names_order (list of names),
        bundle_vec flattened (1,N) or (N,), and optional 'lengths' (1,M) int32.
        If lengths is None, it will slice by TFLite input shapes.
        Returns (updated_count, consumed_len).
        """
        vec = np.asarray(bundle_vec, dtype=np.float32)
        if vec.ndim == 2:
            vec = vec[0]
        elif vec.ndim >= 3:
            vec = vec.reshape(vec.shape[0], -1)[0]

        offset = 0
        updated = 0
        for idx, nm in enumerate(names_order):
            shp = self._state_shapes.get(nm, (1,0,1))
            if lengths is not None:
                lens = np.asarray(lengths).astype(np.int64)
                if lens.ndim == 2:
                    lens = lens[0]
                need = int(lens[idx]) if idx < lens.size else 0
            else:
                need = int(shp[1] * shp[2])

            if need <= 0:
                # Missing '_next' on exporter side or zero-size state: skip
                continue

            if offset + need > vec.size:
                self._warn_throttled('bundle_short',
                    f"⚠️  [TFLiteRunner] bundle too short for state '{nm}': need {need}, have {max(0, vec.size - offset)}")
                break

            seg = vec[offset:offset+need].reshape((1, int(shp[1]), int(shp[2]))).astype(np.float32, copy=False)
            self._state[nm] = seg
            offset += need
            updated += 1

        return updated, offset

    def _update_states_from_bundle(self, bundle, splits=None):
        """
        Update states from bundle. If 'splits' (lengths) is present, prefer lexicographic ORDER with those lengths.
        Else auto-try declared order then sorted; cache whichever works best.
        """
        # With splits: use lexicographic order deterministically
        if splits is not None:
            try:
                ss = np.asarray(splits)
                total = int(np.sum(ss))
                # Commented out because it completely spams the logs:
                # if self.debug:
                #     print(f"[TFLiteRunner] Using bundle with splits: sum={total}, expected≈{self._expected_bundle_len}")
                # If total != expected, we still proceed; zeros in splits indicate missing '_next' states.
            except Exception:
                pass
            updated, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=splits)
            self._last_state_updates_count = updated
            self._bundle_order_mode = 'sorted'
            if updated != len(self.state_in_names):
                self._warn_throttled('bundle_partial_sorted',
                    f"⚠️  [TFLiteRunner] bundle (with splits) updated {updated}/{len(self.state_in_names)} states.")
            # Commented out because it completely spams the logs:
            # else:
            #     if self.debug:
            #         print(f"[TFLiteRunner] bundle (with splits) updated all {updated} states.")
            return updated

        # No splits: reuse mode if already decided
        if self._bundle_order_mode == 'declared':
            upd, _ = self._try_update_bundle_with_order(self.state_in_names, bundle, lengths=None)
            self._last_state_updates_count = upd
            return upd
        if self._bundle_order_mode == 'sorted':
            upd, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=None)
            self._last_state_updates_count = upd
            return upd

        # First-time: try declared then sorted
        upd_decl, _ = self._try_update_bundle_with_order(self.state_in_names, bundle, lengths=None)
        upd_sort, _ = self._try_update_bundle_with_order(self.state_in_names_sorted, bundle, lengths=None)

        if upd_sort > upd_decl:
            self._bundle_order_mode = 'sorted'
            self._last_state_updates_count = upd_sort
            if upd_sort != len(self.state_in_names):
                self._warn_throttled('bundle_partial_sorted',
                    f"⚠️  [TFLiteRunner] bundle updated {upd_sort}/{len(self.state_in_names)} states with sorted order.")
            else:
                if self.debug:
                    print(f"[TFLiteRunner] bundle updated all {upd_sort} states (sorted order).")
            return upd_sort
        else:
            self._bundle_order_mode = 'declared'
            self._last_state_updates_count = upd_decl
            if upd_decl != len(self.state_in_names):
                self._warn_throttled('bundle_partial_decl',
                    f"⚠️  [TFLiteRunner] bundle updated {upd_decl}/{len(self.state_in_names)} states with declared order.")
            else:
                if self.debug:
                    print(f"[TFLiteRunner] bundle updated all {upd_decl} states (declared order).")
            return upd_decl

    # ------------- inference -------------
    def invoke(self, feed_dict):
        # Normalize feed and inject states if missing
        feed = { self._norm_name_input(k): np.asarray(v, np.float32, order="C") for k, v in feed_dict.items() }
        if self._is_streaming:
            for nm in self.state_in_names:
                if nm not in feed:
                    feed[nm] = self._state[nm]

        T_cur = self._piezo_T_from_feed(feed)

        # Set inputs by name; remember whether the 3D piezo was actually set
        set_piezo = False
        piezo_block = None

        # Try to remember any 3D array from the feed (expected to be the piezo block)
        for v in feed.values():
            arr = np.asarray(v)
            if arr.ndim == 3 and (T_cur is None or int(arr.shape[1]) == int(T_cur)):
                piezo_block = arr
                break

        for k, v in feed.items():
            nm = self._norm_name_input(k)
            if nm in self.in_map:
                idx = self.in_map[nm]
                # Track if this is the 3D piezo feed
                shp = tuple(int(x) for x in self._in_details_by_idx[idx]["shape"])
                if len(shp) == 3 and np.asarray(v).ndim == 3 and (T_cur is None or int(v.shape[1]) == int(T_cur)):
                    set_piezo = True
                self.interp.set_tensor(idx, v)

        # Fallback mapping: if we didn't set any 3D piezo by name, map the 3D block by shape
        if (not set_piezo) and (piezo_block is not None):
            for d in self.inp_details:
                nm_raw = str(d["name"])
                shp = tuple(int(x) for x in d["shape"])
                if len(shp) == 3 and (T_cur is None or shp[1] == T_cur or shp[1] == -1) and shp[2] == 1:
                    try:
                        self.interp.set_tensor(d["index"], piezo_block)
                        if not self._piezo_fallback_logged:
                            print(f"[TFLiteRunner] Mapped 3D piezo block to unnamed input '{nm_raw}' shape={shp} (fallback by shape).")
                            self._piezo_fallback_logged = True
                        break
                    except Exception:
                        continue

        self.interp.invoke()

        # Deep-copy outputs
        outs_list = []
        for d in self.out_details:
            idx = d["index"]
            name = str(d["name"])
            val = np.copy(self.interp.get_tensor(idx))
            outs_list.append((name, val))
        self._last_state_updates_count = 0

        # Try packed bundle path first (by shape)
        if self._is_streaming:
            bundle = self._pick_bundle_from_outs(outs_list)
            splits = self._pick_splits_from_outs(outs_list)
            if bundle is not None:
                self._update_states_from_bundle(bundle, splits=splits)

                if self._is_streaming:
                    total_states = len(self.state_in_names)
                    updated = int(getattr(self, "_last_state_updates_count", 0))
                    if updated != total_states:
                        self._warn_throttled('bundle_partial_update',
                            f"[TFLiteRunner] bundle updated {updated}/{total_states} states (check export wiring).")

        # Name-based per-state mapping is skipped (generic TFLite names).
        # Ordered fallback [audio, style, states...]
        if self._is_streaming and self._last_state_updates_count < len(self.state_in_names):
            for in_nm in self.state_in_names:
                idx = self._ordered_state_out_idx.get(in_nm, None)
                if idx is None or idx >= len(outs_list):
                    continue
                cand = outs_list[idx][1]
                want = self._state_shapes.get(in_nm, None)
                if (want is None) or (tuple(cand.shape) == tuple(want)):
                    self._state[in_nm] = np.array(cand, dtype=np.float32, copy=True)
            self._last_state_updates_count = len(self.state_in_names)

        if self._is_streaming and self._last_state_updates_count == 0:
            self._warn_throttled('no_state_update',
                "⚠️  [TFLiteRunner] WARNING: states did NOT update; expect tonal/bitcrushed noise. "
                "Ensure your realtime TFLite includes a states bundle and (optionally) a splits vector.")

        # Select audio/style from outs_list
        audio_arr, style_arr = self._select_audio_style(outs_list, T_cur)
        return self._normalize_audio_style(audio_arr, style_arr)

    def _prepare_default_feed(self, T):
        feed = {}
        piezo_key = "realtime_piezo_input" if self._kind == "realtime" else "offline_piezo_input"
        if piezo_key in self.in_map:
            feed[piezo_key] = np.zeros((1, int(T), 1), dtype=np.float32)

        st_true = "realtime_style_true_input" if self._kind == "realtime" else "offline_style_true_input"
        st_force= "realtime_style_force_input" if self._kind == "realtime" else "offline_style_force_input"
        pkup    = "realtime_pickup_input" if self._kind == "realtime" else "offline_pickup_input"
        if st_true in self.in_map:  feed[st_true]  = np.zeros((1,1), np.float32)
        if st_force in self.in_map: feed[st_force] = np.zeros((1,1), np.float32)
        if pkup in self.in_map:     feed[pkup]     = np.zeros((1,1), np.float32)  # default: piezo

        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            if "knob_" in nm:
                val = 1.0
                feed[nm] = np.array([[val]], dtype=np.float32)

        if self._is_streaming:
            for nm in self.state_in_names:
                feed[nm] = self._state[nm]
        return feed

    def _quick_self_test(self):
        if not self._is_streaming:
            return
        T = None
        # Try to get T from piezo input
        for d in self.inp_details:
            nm = self._norm_name_input(d["name"])
            if "piezo_input" in nm:
                shp = d["shape"]
                T = int(shp[1]) if len(shp) > 1 else None
                break
        if T is None:
            # Try any 3D input
            for d in self.inp_details:
                shp = tuple(int(x) for x in d["shape"])
                if len(shp) == 3 and shp[1] > 0:
                    T = shp[1]
                    break
        if T is None or T <= 0:
            return
        feed = self._prepare_default_feed(T)
        _ = self.invoke(feed)

def generate_demo_tflite(kind, use_finetuned=False, out_dir=None, hop=None):
    """
    Run the TFLite model in a streaming fashion and write WAVs.
    kind: 'offline' or 'realtime'
    uses T=32.
    hop: if provided (int), use exact pedal-style schedule:
         left = T - hop, right = 0, core = hop.
    """
    assert_has_pairs()
    pth_src, pth_tgt = all_pairs[LONG_DEMO_PAIR_INDEX]
    style_true_flag = style_flag_for_path(pth_src)

    if kind == "offline":
        sr = OFFLINE_SAMPLE_RATE
        base_out_dir = LONG_DEMOS_OFFLINE_OUT_DIR
        subdir = f"t{OFFLINE_TFLITE_T}"
    else:
        sr = REALTIME_SAMPLE_RATE
        base_out_dir = LONG_DEMOS_REALTIME_OUT_DIR
        subdir = f"t{REALTIME_TFLITE_T}"

    if out_dir is None:
        out_dir = os.path.join(base_out_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    info = sf.info(pth_src)
    file_sr = info.samplerate
    start = int(round(LONG_DEMO_OFFSET_SEC * file_sr))
    frames = int(round(LONG_DEMO_SEC * file_sr))

    x_long, _ = sf.read(pth_src, start=start, frames=frames, dtype='float32', always_2d=True)
    y_long, _ = sf.read(pth_tgt, start=start, frames=frames, dtype='float32', always_2d=True)
    x_src = x_long[:,0]
    y_src = y_long[:,0]
    x_mod = x_src if file_sr == sr else resample_to_model_sr(x_src, file_sr, sr)
    y_mod = y_src if file_sr == sr else resample_to_model_sr(y_src, file_sr, sr)

    tflite_path = _select_tflite_demo_paths(kind=kind, use_finetuned=use_finetuned)
    if not os.path.exists(tflite_path):
        print(f"[TFLite/demo] TFLite model not found at {tflite_path}. Run --export-tflite first.")
        return

    print(f"[TFLite/demo] Using: {tflite_path}")
    runner = TFLiteRunner(tflite_path)

    _tflite_demo_blockwise(kind, runner, sample_rate=sr,
                           x_mod=x_mod, y_mod=y_mod, style_true_flag=style_true_flag,
                           use_finetuned=use_finetuned, out_dir=out_dir,
                           fixed_hop=hop, pickup=pickup_flag_for_path(pth_src))
