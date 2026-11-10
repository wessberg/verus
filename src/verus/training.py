import json
import numpy as np
import os
import soundfile as sf
import sys
import tensorflow as tf
import time
from tensorflow.keras.callbacks import (Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, TerminateOnNaN)  # pyright: ignore[reportMissingImports]

from . import config
from .config import (
    SEED,
    BASE_TRAIN_GAIN_DB_HI,
    BASE_TRAIN_GAIN_DB_LO,
    CALIB_SEGMENTS,
    CALIB_WARMUP_STEPS,
    CHECKPOINT_OFFLINE_OUT_DIR,
    CHECKPOINT_REALTIME_OUT_DIR,
    DEBUG,
    DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
    DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
    DEMO_OFFSET_SEC,
    DEMO_PAIR_INDEX,
    DEMO_SEC,
    DISC_LR,
    DISC_RESOLUTIONS_OFFLINE,
    EPOCHS,
    FINETUNE_BATCH_SIZE,
    FINETUNE_DEMOS_OUT_DIR,
    FINETUNE_DEMO_OFFSET_SEC,
    FINETUNE_DEMO_SEC,
    FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_LR,
    FINETUNE_OFFLINE_STATIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_STEPS_CAP,
    FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME,
    FINETUNE_REALTIME_LR,
    FINETUNE_REALTIME_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_STEPS_CAP,
    FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME,
    FINETUNE_SRC_DIR,
    FINGER_CHUNK_RATIO,
    KD_ALPHA_SCHEDULE,
    KD_STYLE_BLEND_SCHEDULE,
    KERAS_STREAM_HOP,
    LAMBDA_STYLE_RATIO,
    LOGS_OFFLINE_OUT_DIR,
    LOGS_REALTIME_OUT_DIR,
    LR,
    LR_FACTOR,
    LR_PATIENCE,
    LR_PLATEAU_PATIENCE,
    MIN_LR,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_ADVERSARIAL,
    OFFLINE_ADV_RAMP_EPOCHS,
    OFFLINE_ADV_START_EPOCH,
    OFFLINE_ADV_WEIGHT,
    OFFLINE_BATCH_SIZE,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_EPOCHS,
    OFFLINE_FINETUNE_EPOCHS,
    OFFLINE_FM_WEIGHT,
    OFFLINE_LOSS_CORE,
    OFFLINE_MAX_STEPS_PER_EPOCH,
    OFFLINE_PHASE_LOSS_CUTOFF_HZ,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_SEG_LEN,
    OFFLINE_STATIC_MODEL_FILENAME,
    PICKUP_DIR_NAMES,
    PICKUP_MAGNETIC,
    PICKUP_PIEZO,
    PICK_CHUNK_RATIO,
    REALTIME_BATCH_SIZE,
    REALTIME_DYNAMIC_MODEL_FILENAME,
    REALTIME_FINETUNE_EPOCHS,
    REALTIME_LOSS_CORE,
    REALTIME_MAX_STEPS_PER_EPOCH,
    REALTIME_PHASE_LOSS_CUTOFF_HZ,
    REALTIME_SAMPLE_RATE,
    REALTIME_SEG_LEN,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TRAINABLE_MODEL_FILENAME,
    SHUFFLE_BUF,
    STYLE_DIR_NAMES,
    ULTRALOW_MAIN_CUTOFF_HZ,
    ULTRALOW_MAIN_ORDER,
    VAL_MAX_SEGMENTS,
)
from .adversarial import (
    AdversarialRamp,
    AdversarialTrainer,
    GeneratorCheckpoint,
    build_offline_discriminators,
    disc_weights_path,
)
from .data import (
    _extract_epoch_from_ckpt,
    _read_pair_segment,
    add_shared_random_gain_dual,
    add_shared_random_gain_offline,
    add_shared_random_gain_realtime,
    add_sw_offline,
    add_sw_offline_finetune_nf,
    add_sw_realtime,
    add_sw_realtime_dual,
    add_sw_realtime_dual_nf,
    all_pairs,
    approx_steps_from_chunks,
    approx_steps_from_length,
    assert_has_pairs,
    build_chunks,
    denoise_target_if_enabled,
    kd_teacher_segment_len,
    make_mixed_ds_from_arrays_offline,
    make_mixed_ds_from_arrays_realtime_dual,
    pairs_by_group,
    pickup_flag_for_path,
    precache_finetune_resampled_pair,
    segment_generator,
    segment_generator_dual,
    style_flag_for_path,
)
from .inference import (
    KerasStreamingRunner,
    _extract_audio_io,
    _style_label,
    generate_demo_from_files,
)
from .knobs import (
    prepare_demo_knob_feeds,
)
from .layers import (
    IrScale,
)
from .losses import (
    CombinedLoss,
    FinetuneCombinedLossLite,
    LogMelLoss,
    _combined_loss_terms,
    _loss_core_from_kind_or_sr,
    mean_hf_metric,
    multi_scale_stft_loss,
    rms_env_loss,
)
from .modelio import (
    _tensor_basename,
    load_keras_model,
)
from .models import (
    build_offline_model,
    build_realtime_streaming_model,
    build_realtime_training_model,
    copy_weights_by_name,
    load_offline_teacher,
)

class DemoCallback(Callback):
    """
    Per-epoch listening demo. Offline: dynamic-time graph, single shot. Realtime: the actual
    streaming graph, stepped hop-wise through KerasStreamingRunner (a single-shot pass through
    the streaming graph would apply exactly one EMA update to the style logit and mix pick/finger
    ~50/50 regardless of what the classifier says).
    """
    def __init__(self, sample_rate, demo_checkpoints_out_dir, build_model_fn, build_model_fn_args_dynamic={}, use_safe_weight_copy=False):
        super().__init__()
        assert_has_pairs()
        demo_src, demo_tgt = all_pairs[DEMO_PAIR_INDEX]
        self.style_true_flag = style_flag_for_path(demo_src)
        self.pickup_flag = pickup_flag_for_path(demo_src)
        self.src_path = demo_src
        self.tgt_path = demo_tgt
        self.demo_sec = DEMO_SEC
        self.offset_sec = DEMO_OFFSET_SEC
        self.build_model_fn = build_model_fn
        self.build_model_fn_args_dynamic = build_model_fn_args_dynamic
        self.sample_rate = sample_rate
        self.kind = "offline" if int(sample_rate) == int(OFFLINE_SAMPLE_RATE) else "realtime"
        self.every = 1
        self.demos_out_dir = demo_checkpoints_out_dir
        self.use_safe_weight_copy = use_safe_weight_copy
        os.makedirs(self.demos_out_dir, exist_ok=True)
        self.static_model = None
        self.dynamic_model = None
        self.runner = None

    def set_model(self, model):
        super().set_model(model)
        self.static_model = getattr(model, "generator", model)   # AdversarialTrainer wraps the generator
        self.dynamic_model = self.build_model_fn(**self.build_model_fn_args_dynamic, dynamic_time=True)
        self._sync_weights()
        if self.kind == "realtime":
            self.runner = KerasStreamingRunner(self.dynamic_model, hop=KERAS_STREAM_HOP)

    def _sync_weights(self):
        if self.use_safe_weight_copy:
            copy_weights_by_name(self.dynamic_model, self.static_model)
        else:
            self.dynamic_model.set_weights(self.static_model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every != 0:
            return
        self._sync_weights()
        kind = self.kind
        x_mod, y_mod = _read_pair_segment(self.src_path, self.tgt_path, self.sample_rate,
                                          self.offset_sec, self.demo_sec)
        knob_feeds = prepare_demo_knob_feeds(kind=kind, use_finetuned=False, pickup=self.pickup_flag)

        if kind == "realtime":
            res = self.runner.run(x_mod, knob_feeds, style_true=None)
            y_pred_pass = res["audio"]; style_pred = res["avg_prob"]
            y_true_pass = self.runner.run(x_mod, knob_feeds, style_true=self.style_true_flag)["audio"] \
                          if self.style_true_flag is not None else None
        else:
            in_names = [_tensor_basename(t) for t in self.dynamic_model.inputs]
            def predict(style_flag):
                feed = {f"{kind}_piezo_input": x_mod[None, :, None].astype(np.float32),
                        f"{kind}_style_true_input": np.array([[0.0 if style_flag is None else style_flag]], np.float32),
                        f"{kind}_style_force_input": np.array([[0.0 if style_flag is None else 1.0]], np.float32),
                        **knob_feeds}
                outs = self.dynamic_model.predict({k: v for k, v in feed.items() if k in in_names},
                                                  batch_size=1, verbose=0)
                return np.asarray(outs[0], np.float32)[0, :, 0], float(np.asarray(outs[1], np.float32)[0, 0])
            y_pred_pass, style_pred = predict(None)
            y_true_pass = predict(self.style_true_flag)[0] if self.style_true_flag is not None else None

        style_pred_name = _style_label(style_pred)
        style_true_name = _style_label(self.style_true_flag) if self.style_true_flag is not None else None

        epoch_dir = os.path.join(self.demos_out_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        sf.write(os.path.join(epoch_dir, "demo_source.wav"), x_mod, self.sample_rate, subtype='FLOAT')
        if y_mod is not None:
            sf.write(os.path.join(epoch_dir, "demo_target.wav"), y_mod, self.sample_rate, subtype='FLOAT')
        sf.write(os.path.join(epoch_dir, "demo_pred_style_pred.wav"), y_pred_pass, self.sample_rate, subtype='FLOAT')
        if y_true_pass is not None:
            sf.write(os.path.join(epoch_dir, "demo_pred_style_true.wav"), y_true_pass, self.sample_rate, subtype='FLOAT')

        p_txt = f"{style_pred:.3f}" if style_pred is not None else "n/a"
        print(f"[DemoCallback] epoch {epoch+1}: predicted style={style_pred_name} ({p_txt}), "
              f"true style={style_true_name}, written to {epoch_dir}")

def analyze_loss_scales(model, batch, sample_rate, extra_small_cplx=False, core=None, phase_cutoff_hz=None):
    """Mean of every CombinedLoss term on one batch, computed exactly as the loss computes them."""
    y_true, y_pred = _extract_audio_io(model, batch)
    core = _loss_core_from_kind_or_sr(sample_rate) if core is None else core
    terms = _combined_loss_terms(y_true, y_pred, sample_rate, extra_small_cplx=extra_small_cplx, core=core,
                                 phase_cutoff_hz=phase_cutoff_hz)
    return {k: float(tf.reduce_mean(v).numpy()) for k, v in terms.items()}

def _measure_recon_loss(model, dataset, loss, n_batches=3):
    """Mean reconstruction loss (a compiled CombinedLoss instance) of `model` over a few batches."""
    vals = []
    for batch in dataset.take(n_batches):
        y_true, y_pred = _extract_audio_io(model, batch)
        vals.append(float(tf.reduce_mean(loss(y_true, y_pred)).numpy()))
    return float(np.mean(vals)) if vals else 1.0

def calibrate_loss_weights(model, dataset, sample_rate, n_batches=5,
                                      normalize_keys=('l1','stft','mel','env','hf','resbp_bl','lmag'),
                                      declared=None, extra_small_cplx=False, core=None, phase_cutoff_hz=None):
    if declared is None:
        declared = {
            'l1': config.L1_WEIGHT, 'stft': config.STFT_WEIGHT, 'mel': config.MEL_WEIGHT, 'env': config.RMS_ENV_WEIGHT, 'hf': config.HF_WEIGHT,
            'resbp_bl': config.RESBP_BAND_LOSS_WEIGHT, 'lmag': config.LMAG_WEIGHT
        }
    sums = {k:0.0 for k in normalize_keys}
    count = 0
    for batch in dataset.take(n_batches):
        sc = analyze_loss_scales(model, batch, sample_rate, extra_small_cplx=extra_small_cplx, core=core,
                                 phase_cutoff_hz=phase_cutoff_hz)
        for k in normalize_keys:
            sums[k] += float(sc[k])
        count += 1
    avgs = {k: (sums[k] / max(count,1)) for k in normalize_keys}
    print("Average consistent loss scales:", {k: round(v, 6) for k,v in avgs.items()})
    ref = float(avgs['l1']) if avgs['l1'] > 0 else 1.0
    new_w = {}
    for k in normalize_keys:
        bs = float(avgs[k])
        new_w[k] = (declared[k] / bs * ref) if bs > 0 else 0.0
    return new_w, avgs

class TrainStateCallback(tf.keras.callbacks.Callback):
    """Persist the learning rate at the end of every epoch (1-based, like the checkpoint names)."""
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        try:
            state = {}
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    state = json.load(f)
            lr = float(np.asarray(tf.keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)))
            state.setdefault("lr_by_epoch", {})[str(epoch + 1)] = lr
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[train_state] could not persist training state: {e}")

class KDAlphaScheduler(tf.keras.callbacks.Callback):
    """
    Linearly anneal alpha variables of KD layers and style blend over epochs.
    targets: dict of layer_name -> (start_alpha, end_alpha, start_epoch, end_epoch)
    """
    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def on_epoch_begin(self, epoch, logs=None):
        for lname, (a0, a1, e0, e1) in self.targets.items():
            layer = None
            try:
                layer = self.model.get_layer(lname)
            except Exception:
                continue
            if layer is None:
                continue
            # progress
            if epoch <= e0:
                a = a0
            elif epoch >= e1:
                a = a1
            else:
                p = (epoch - e0) / max(1, (e1 - e0))
                a = a0 + (a1 - a0) * p
            # Set alpha if supported
            if hasattr(layer, "set_alpha"):
                layer.set_alpha(a)
            else:
                # Style blend: its "alpha" is a trainable weight[0]
                if lname == "realtime_style_blend":
                    w = layer.get_weights()
                    if w and len(w[0].shape) == 1 and w[0].shape[0] == 1:
                        w[0][0] = float(a)
                        layer.set_weights(w)

def fine_tune(offline=True, realtime=True, pickup=PICKUP_PIEZO):
    """
    Fine-tune on input/finetune/{source,target}.wav. 'pickup' declares the source's pickup type
    (0 = piezo, 1 = magnetic/electric) exactly like --process --pickup: a magnetic/electric DI must
    be fine-tuned through the magnetic conditioning path, otherwise the piezo path is bent toward a
    foreign domain and the pickup input stops meaning anything.
    """
    pickup = float(pickup)
    print(f"[finetune] pickup: {'magnetic' if pickup >= 0.5 else 'piezo'}")
    # IO
    src = os.path.join(FINETUNE_SRC_DIR, "source.wav")
    tgt = os.path.join(FINETUNE_SRC_DIR, "target.wav")
    if not os.path.exists(src) or not os.path.exists(tgt):
        print("ERROR: finetune source/target not found in", FINETUNE_SRC_DIR); sys.exit(1)

    x_full, sr_x = sf.read(src, dtype='float32', always_2d=True)
    y_full, sr_y = sf.read(tgt, dtype='float32', always_2d=True)
    if sr_x != sr_y:
        print(f"Warning: finetune pair SR mismatch: {sr_x} vs {sr_y}")
    y_full = denoise_target_if_enabled(y_full, sr_y)

    cached = precache_finetune_resampled_pair(
        x_full, y_full, file_sr=sr_x,
        sr_rt=REALTIME_SAMPLE_RATE, sr_off=OFFLINE_SAMPLE_RATE
    )

    # Lightweight fine-tune losses (restricted to the contexted core of each segment)
    loss_offlite = FinetuneCombinedLossLite(sample_rate=OFFLINE_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0, ulb_w=0.20,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER,
                                            core=OFFLINE_LOSS_CORE)
    loss_rt_lite = FinetuneCombinedLossLite(sample_rate=REALTIME_SAMPLE_RATE,
                                            stft_scales=((1024,256,1024),),
                                            l1_w=0.6, hf_w=2.0, stft_w=0.8, env_w=0.0,
                                            ulb_cut_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                                            ulb_order=ULTRALOW_MAIN_ORDER,
                                            core=REALTIME_LOSS_CORE)

    off_dyn_export = None
    rt_dyn_export  = None

    if offline:
        print("\n🔧 Finetuning OFFLINE base (coloreq + film + scales only)…")
        base_off_dyn = load_keras_model(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build the OFFLINE training graph. include_dry_output=False keeps it at 2 outputs so
        # the 2-entry loss list below matches (the dry output is a runtime/export concern).
        off_static_train = build_offline_model(include_dry_output=False)
        copy_weights_by_name(off_static_train, base_off_dyn, verbose=True)

        # Unfreeze only FiLM + coloreq
        off_static_train = set_trainable_layers(
            off_static_train,
            enable_film=True,
            enable_coloreq=True
        )

        # Dataset
        dsO = make_mixed_ds_from_arrays_offline(
            cached["x96"], cached["y96"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len=OFFLINE_SEG_LEN
        ).map(lambda x, y: add_sw_offline_finetune_nf(x, y, pickup=pickup)
        ).map(add_shared_random_gain_offline(db_lo=-30.0, db_hi=+18.0)
        ).prefetch(tf.data.AUTOTUNE)

        steps_off = approx_steps_from_length(
            len(cached["x96"]), OFFLINE_SEG_LEN, FINETUNE_BATCH_SIZE, cap=FINETUNE_OFFLINE_STEPS_CAP
        )

        off_static_train.compile(optimizer=tf.keras.optimizers.Adam(FINETUNE_OFFLINE_LR),
                                 loss=[loss_offlite, "binary_crossentropy"],
                                 loss_weights=[1.0, 0.0],
                                 metrics=[["mae"], []])
        off_static_train.fit(dsO, epochs=OFFLINE_FINETUNE_EPOCHS, steps_per_epoch=steps_off, verbose=1)

        # Export OFFLINE dynamic
        off_dyn_export = build_offline_model(dynamic_time=True)
        copy_weights_by_name(off_dyn_export, off_static_train, verbose=True)

        ft_offline_dyn_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        off_dyn_export.save(ft_offline_dyn_path)
        print("✅ Saved finetuned offline model (dynamic):", ft_offline_dyn_path)

        ft_offline_static_path = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
        off_static_train.save(ft_offline_static_path)
        print("✅ Saved finetuned offline model (static):", ft_offline_static_path)

    if realtime:
        print("\n🔧 Finetuning REALTIME base (coloreq + film + scales only)…")

        # Distill from the fine-tuned offline model when it was adapted to THIS pair (fine-tuned in
        # this run, or newer than the pair's files); otherwise from the base offline model.
        ft_off_dyn = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        ft_fresh = os.path.exists(ft_off_dyn) and (
            offline or os.path.getmtime(ft_off_dyn) > max(os.path.getmtime(src), os.path.getmtime(tgt)))
        teacher_path = ft_off_dyn if ft_fresh else None
        offline_teacher = load_offline_teacher(teacher_path)
        print(f"[finetune] KD teacher: {teacher_path or os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)}")

        base_rt_dyn = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME), compile=False)

        # Build KD training graph (teacher embedded) and copy weights from baseline
        rt_static_train = build_realtime_training_model(
            offline=offline_teacher
        )
        copy_weights_by_name(rt_static_train, base_rt_dyn, verbose=True)

        # The streaming model does not contain the KD feature projections (feat_distill_*_proj);
        # without them the feature KD would compare the teacher to a RANDOM frozen projection.
        # Seed them from the base trainable model when it exists, else disable feature KD.
        base_rt_trainable_path = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME)
        feat_kd_seeded = 0
        if os.path.exists(base_rt_trainable_path):
            base_rt_trainable = load_keras_model(base_rt_trainable_path, compile=False)
            src_layers = {l.name: l for l in base_rt_trainable.layers}
            for lname in ("feat_distill_0", "feat_distill_1"):
                try:
                    dst, src_l = rt_static_train.get_layer(lname), src_layers.get(lname)
                    if src_l is not None and [w.shape for w in src_l.get_weights()] == [w.shape for w in dst.get_weights()]:
                        dst.set_weights(src_l.get_weights()); feat_kd_seeded += 1
                except Exception:
                    pass
            del base_rt_trainable
        if feat_kd_seeded < 2:
            print("[finetune] KD feature projections unavailable → feature KD disabled for this fine-tune.")
            for lname in ("feat_distill_0", "feat_distill_1"):
                try:
                    rt_static_train.get_layer(lname).set_alpha(0.0)
                except Exception:
                    pass

        # Unfreeze only FiLM + coloreq (and scales if desired)
        rt_static_train = set_trainable_layers(
            rt_static_train,
            enable_film=True,
            enable_coloreq=True
        )

        # Fine-tune distills from the base teacher at the end-of-anneal KD strengths
        for lname, (_, a_end, _, _) in KD_ALPHA_SCHEDULE.items():
            if lname.startswith("feat_distill") and feat_kd_seeded < 2:
                continue
            try:
                rt_static_train.get_layer(lname).set_alpha(a_end)
            except Exception:
                pass

        # KD dual-rate dataset
        dsRT_dual = make_mixed_ds_from_arrays_realtime_dual(
            cached["x48"], cached["x96"], cached["y48"],
            batch_size=FINETUNE_BATCH_SIZE,
            seg_len_rt=REALTIME_SEG_LEN
        ).map(lambda x, y: add_sw_realtime_dual_nf(x, y, pickup=pickup)
        ).map(add_shared_random_gain_dual(db_lo=-30.0, db_hi=+18.0)
        ).prefetch(tf.data.AUTOTUNE)

        steps_rt = approx_steps_from_length(
            len(cached["x48"]), REALTIME_SEG_LEN, FINETUNE_BATCH_SIZE, cap=FINETUNE_REALTIME_STEPS_CAP
        )

        rt_static_train.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_REALTIME_LR, clipnorm=1.0),
                                loss=[loss_rt_lite, "binary_crossentropy"],
                                loss_weights=[1.0, 0.0],
                                metrics=[["mae"], []])
        rt_static_train.fit(dsRT_dual, epochs=REALTIME_FINETUNE_EPOCHS, steps_per_epoch=steps_rt, verbose=1)
        
        ft_rt_trainable_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME)
        rt_static_train.save(ft_rt_trainable_path)
        print("✅ Saved finetuned realtime training model:", ft_rt_trainable_path)

        # Export self-contained streaming runtime artifacts (dynamic + static)
        rt_dyn_stream = build_realtime_streaming_model(dynamic_time=True)
        copy_weights_by_name(rt_dyn_stream, rt_static_train, verbose=True)
        ft_rt_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
        rt_dyn_stream.save(ft_rt_dyn_path)
        print("✅ Saved finetuned realtime streaming dynamic model:", ft_rt_dyn_path)

        rt_static_stream = build_realtime_streaming_model()
        copy_weights_by_name(rt_static_stream, rt_static_train, verbose=True)
        ft_rt_static_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        rt_static_stream.save(ft_rt_static_path)
        print("✅ Saved finetuned realtime streaming static model:", ft_rt_static_path)

        # Return dynamic stream (for optional demo)
        rt_dyn_export = rt_dyn_stream

    return off_dyn_export, rt_dyn_export

def set_trainable_layers(model,
                         enable_film=False,
                         enable_scales=False,
                         enable_coloreq=False,
                         enable_mask_head=False,
                         enable_heads=False,
                         enable_style_head=False,
                         enable_style_blend=False):
    """
    Freeze everything, then selectively unfreeze groups:
      - enable_film:    Dense layers generating FiLM params (name contains 'film_dense')
      - enable_scales:  All IrScale except coloreq (pickattack_scale, pickreso_scale, ir_scale, etc.)
      - enable_coloreq: Only coloreq IrScale bands (name starts with '*_coloreq_g_')
      - enable_mask_head: hf mask tiny conv stack (offline_*_hf_mask_* and realtime_* equivalents)
      - enable_heads:   final pick/finger heads (Conv1D layers named '*_final_pick'/'*_final_finger')
      - enable_style_head: style classifier tower (conv/gap/logit/prob)
      - enable_style_blend: realtime_style_blend scalar (if present)
    Works for both offline_* and realtime_* prefixes.
    """
    # 1) Freeze all
    for l in model.layers:
        l.trainable = False

    def is_coloreq_scale(layer):
        return isinstance(layer, IrScale) and ("_coloreq_g_" in layer.name)

    def is_regular_scale(layer):
        # Any IrScale that is not a coloreq scale
        return isinstance(layer, IrScale) and not is_coloreq_scale(layer)

    trainable_names = []

    # 2) Film generators (shared for offline/realtime by name convention)
    if enable_film:
        for l in model.layers:
            if "film_dense" in l.name:
                l.trainable = True
                trainable_names.append(l.name)

    # 3) Scales (all IrScale except coloreq)
    if enable_scales:
        for l in model.layers:
            if is_regular_scale(l):
                l.trainable = True
                trainable_names.append(l.name)

    # 4) COLOREQ band scales only
    if enable_coloreq:
        for l in model.layers:
            if is_coloreq_scale(l):
                l.trainable = True
                trainable_names.append(l.name)

    # 5) Pick-attack tiny CNN (offline/realtime prefixes)
    if enable_mask_head:
        for lname in [
            "offline_picklearn_c1", "offline_picklearn_c2", "offline_picklearn_out",
            "realtime_picklearn_c1","realtime_picklearn_c2","realtime_picklearn_out"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

    # 6) Final pick/finger heads (Conv1D)
    if enable_heads:
        for lname in [
            "offline_final_pick", "offline_final_finger",
            "realtime_final_pick","realtime_final_finger"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

    # 7) Style classifier tower (disable by default to keep invariance)
    if enable_style_head:
        for lname in [
            "offline_style_conv", "offline_style_gap", "offline_style_logit", "offline_style_prob",
            "realtime_style_conv","realtime_style_gap","realtime_style_logit","realtime_style_prob"
        ]:
            try:
                layer = model.get_layer(lname)
            except Exception:
                continue
            layer.trainable = True
            trainable_names.append(layer.name)

        # conditioning embedding Dense (style + pickup); legacy name kept for old graphs
        for lname in ["offline_cond_embed","realtime_cond_embed",
                      "offline_style_embed","realtime_style_embed"]:
            try:
                layer = model.get_layer(lname)
                layer.trainable = True
                trainable_names.append(layer.name)
            except Exception:
                pass

    # 8) Realtime style_blend scalar (optional)
    if enable_style_blend:
        try:
            layer = model.get_layer("realtime_style_blend")
            layer.trainable = True
            trainable_names.append(layer.name)
        except Exception:
            pass

    print(f"[trainable] enabled: {sorted(set(trainable_names))}")
    return model

def _split_chunks_train_val(chunks, val_fraction=0.2):
    """
    Split chunks into train/val by CONTIGUOUS time regions per source file: the tail of each
    file is held out. A random chunk shuffle would put val chunks seconds away from train
    chunks of the same take, which leaks and makes val_loss (EarlyStopping/checkpoints)
    optimistic.
    """
    by_src = {}
    for c in chunks:
        by_src.setdefault(c[0], []).append(c)
    train, val = [], []
    for src in sorted(by_src):
        cs = sorted(by_src[src], key=lambda c: c[2])
        n_val = int(round(len(cs) * val_fraction))
        if len(cs) > 1 and n_val == 0:
            n_val = 1
        cut = len(cs) - n_val
        train += cs[:cut]
        val   += cs[cut:]
    return train, val

def train_shared(
    kind,
    seg_len,
    sample_rate,
    batch_size,
    build_model_fn,
    checkpoint_out_dir,
    logs_out_dir,
    demo_checkpoints_out_dir,
    static_model_filepath,
    dynamic_model_filepath,
    resume=False,
    resume_ckpt_path=None,
    build_model_fn_args_static={},
    build_model_fn_args_dynamic={},
    dual_offline=False,
    trainable_model_filepath=None,
    epochs=EPOCHS,
    max_steps_per_epoch=None,
    init_weights_path=None
):

    assert_has_pairs()

    # Chunks per (pickup, style) group; groups without data are simply absent, so a
    # piezo-only corpus behaves exactly like before the pickup axis existed.
    group_specs = []   # (style_name, pickup_flag, train_chunks, val_chunks)
    train_chunks_all, val_chunks_all = [], []
    for pickup_name in PICKUP_DIR_NAMES:
        for style_name in STYLE_DIR_NAMES:
            pairs = pairs_by_group.get((pickup_name, style_name), [])
            if not pairs:
                continue
            chunks = build_chunks(pairs_list=pairs, seg_len=seg_len, sample_rate=sample_rate)
            if not chunks:
                continue
            tr, va = _split_chunks_train_val(chunks, val_fraction=0.2)
            pickup_flag = PICKUP_MAGNETIC if pickup_name == "magnetic" else PICKUP_PIEZO
            group_specs.append((style_name, pickup_flag, tr, va))
            train_chunks_all += tr
            val_chunks_all   += va

    if not group_specs or not val_chunks_all or not train_chunks_all:
        raise RuntimeError("Not enough data to split into train/val sets.")

    steps_per_epoch  = approx_steps_from_chunks(chunks=train_chunks_all, seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(max_steps_per_epoch))
    validation_steps = approx_steps_from_chunks(chunks=val_chunks_all,   seg_len=seg_len, model_sample_rate=sample_rate, batch_size=batch_size)
    print(f"≈ {steps_per_epoch} train steps/epoch, {validation_steps} val steps "
          f"({len(group_specs)} pickup×style group(s))")

    sig_y = (
        tf.TensorSpec((seg_len, 1), tf.float32),
        tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.float32)
    )

    # Per-group sampling weights: each style keeps its configured ratio, split evenly across
    # the pickups that actually have data for that style.
    style_ratio = {"pick": PICK_CHUNK_RATIO, "finger": FINGER_CHUNK_RATIO}
    n_by_style = {s: sum(1 for (sn, _, _, _) in group_specs if sn == s) for s in STYLE_DIR_NAMES}
    weights = [style_ratio[sn] / max(1, n_by_style[sn]) for (sn, _, _, _) in group_specs]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    if kind == 'realtime' and dual_offline:
        # offline:realtime SR = 96k:48k = 2:1
        offline_factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE
        assert offline_factor == 2, "This code assumes 96k offline and 48k realtime."

        sig_x = (
            tf.TensorSpec((seg_len, 1), tf.float32),                                      # realtime 48k
            tf.TensorSpec((kd_teacher_segment_len(seg_len, offline_factor), 1), tf.float32)  # offline 96k (+ctx)
        )

        def make_gen(chunks, style_name, pickup_flag):
            return lambda: segment_generator_dual(
                seg_len_realtime=seg_len, chunks=chunks,
                is_pick_chunks=(style_name == "pick"),
                realtime_sample_rate=REALTIME_SAMPLE_RATE, offline_sample_rate=OFFLINE_SAMPLE_RATE,
                pickup_flag=pickup_flag)

        sw_map_fn = add_sw_realtime_dual
        gain_map  = add_shared_random_gain_dual(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI)
    else:
        sig_x = tf.TensorSpec((seg_len, 1), tf.float32)

        def make_gen(chunks, style_name, pickup_flag):
            return lambda: segment_generator(
                seg_len=seg_len, chunks=chunks,
                is_pick_chunks=(style_name == "pick"),
                model_sample_rate=sample_rate,
                pickup_flag=pickup_flag)

        sw_map_fn = add_sw_offline if kind == 'offline' else add_sw_realtime
        gain_map  = (add_shared_random_gain_offline(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI)
                     if kind == 'offline'
                     else add_shared_random_gain_realtime(BASE_TRAIN_GAIN_DB_LO, BASE_TRAIN_GAIN_DB_HI))

    train_gens = [tf.data.Dataset.from_generator(make_gen(tr, sn, pf), output_signature=(sig_x, sig_y))
                  for (sn, pf, tr, _) in group_specs]
    val_gens   = [tf.data.Dataset.from_generator(make_gen(va, sn, pf), output_signature=(sig_x, sig_y))
                  for (sn, pf, _, va) in group_specs]

    train_ds = tf.data.Dataset.sample_from_datasets(
        train_gens, weights=weights
    ).shuffle(SHUFFLE_BUF, seed=SEED, reshuffle_each_iteration=True).batch(batch_size)

    # Shared random gain augmentation (train only) → level-invariant model + analytic branches
    train_ds = train_ds.map(sw_map_fn).map(gain_map).repeat().prefetch(tf.data.AUTOTUNE)

    # Validation set: a FIXED, cached set of batches. The generators draw random offsets, so an
    # uncached val pipeline would present different audio every epoch and EarlyStopping /
    # ModelCheckpoint / ReduceLROnPlateau would react to sampling noise instead of progress.
    validation_steps = max(1, min(validation_steps, VAL_MAX_SEGMENTS[kind] // batch_size))
    # sw_map_fn draws the random 50% style-force mask, so it must run BEFORE .cache(): mapped after
    # the cache it re-drew the mask every epoch and the "fixed" val set still changed per epoch.
    val_raw = tf.data.Dataset.sample_from_datasets(
        val_gens, weights=weights, seed=SEED
    ).batch(batch_size=batch_size).take(validation_steps).map(sw_map_fn).cache()
    t_val = time.time()
    n_val = sum(1 for _ in val_raw)  # fill the cache once so every later pass is identical
    print(f"[val] cached {n_val} fixed validation batch(es) in {time.time() - t_val:.1f}s")
    val_ds = val_raw.repeat().prefetch(tf.data.AUTOTUNE)

    if resume and resume_ckpt_path:
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        ckpt_model = load_keras_model(resume_ckpt_path, compile=False)
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        copy_weights_by_name(static_model, ckpt_model, verbose=True)
        initial_epoch = _extract_epoch_from_ckpt(resume_ckpt_path)
    else:
        static_model = build_model_fn(**build_model_fn_args_static, dynamic_time=False)
        initial_epoch = 0
        if init_weights_path:
            n = copy_weights_by_name(static_model, load_keras_model(init_weights_path, compile=False), verbose=True)
            print(f"[init] warm start: {n} layer(s) from {init_weights_path}")

    # Loss-scale calibration. Random-init loss magnitudes are not representative, so run a brief
    # warmup with the declared weights first and calibrate against the warmed-up model. The
    # calibrated weights are persisted next to the checkpoints and reloaded on resume: re-running
    # the calibration against a partially trained model would silently change the objective
    # mid-run (the ratios between terms drift as the model improves).
    loss_core = OFFLINE_LOSS_CORE if kind == 'offline' else REALTIME_LOSS_CORE
    phase_cutoff = OFFLINE_PHASE_LOSS_CUTOFF_HZ if kind == 'offline' else REALTIME_PHASE_LOSS_CUTOFF_HZ
    use_adversarial = (kind == 'offline') and bool(OFFLINE_ADVERSARIAL)
    weights_json = os.path.join(checkpoint_out_dir, "loss_weights.json")
    new_w = None
    if resume and resume_ckpt_path and os.path.exists(weights_json):
        with open(weights_json, "r") as f:
            new_w = json.load(f)
        if new_w.get("_phase_cutoff_hz") != phase_cutoff:
            print(f"[calib] WARNING: {weights_json} was calibrated for phase cutoff "
                  f"{new_w.get('_phase_cutoff_hz')} (now {phase_cutoff}) — recalibrating; the objective changes.")
            new_w = None
        else:
            print(f"[calib] Reusing persisted loss weights from {weights_json}")
    if new_w is None:
        if not (resume and resume_ckpt_path) and not init_weights_path:
            warmup_steps = max(1, min(CALIB_WARMUP_STEPS, steps_per_epoch))
            print(f"[calib] Warmup for {warmup_steps} step(s) before loss-weight calibration…")
            static_model.compile(
                optimizer=tf.keras.optimizers.Adam(LR),
                loss=[CombinedLoss(sample_rate, extra_small_cplx=(kind == 'offline'), core=loss_core,
                                   phase_cutoff_hz=phase_cutoff),
                      "binary_crossentropy"],
                loss_weights=[1.0, LAMBDA_STYLE_RATIO]
            )
            static_model.fit(train_ds, epochs=1, steps_per_epoch=warmup_steps, verbose=1)
        new_w, avgs = calibrate_loss_weights(
            model=static_model, dataset=val_ds, sample_rate=sample_rate,
            n_batches=max(1, min(validation_steps, -(-CALIB_SEGMENTS // batch_size))),
            extra_small_cplx=(kind == 'offline'), core=loss_core, phase_cutoff_hz=phase_cutoff
        )
        # Scale reference for the adversarial weights: the calibrated reconstruction loss at this point
        new_w["_recon_ref"] = float(sum(new_w[k] * avgs[k] for k in avgs))
        new_w["_phase_cutoff_hz"] = phase_cutoff
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        with open(weights_json, "w") as f:
            json.dump({k: (None if v is None else float(v)) for k, v in new_w.items()}, f, indent=2)
    config.L1_WEIGHT              = new_w['l1']
    config.STFT_WEIGHT            = new_w['stft']
    config.MEL_WEIGHT             = new_w['mel']
    config.RMS_ENV_WEIGHT         = new_w['env']
    config.HF_WEIGHT              = new_w['hf']
    config.RESBP_BAND_LOSS_WEIGHT = new_w['resbp_bl']  # requires the matching `global` declaration above
    config.LMAG_WEIGHT            = new_w.get('lmag', 0.0)
    print(f"→ normalized ({kind}) weights:"
        f" L1={config.L1_WEIGHT:.6f}"
        f" STFT={config.STFT_WEIGHT:.6f}"
        f" MEL={config.MEL_WEIGHT:.6f}"
        f" ENV={config.RMS_ENV_WEIGHT:.6f}"
        f" HF={config.HF_WEIGHT:.6f}"
        f" RESBP_BL={config.RESBP_BAND_LOSS_WEIGHT:.6f}"
        f" LMAG={config.LMAG_WEIGHT:.6f}"
        + (f" (phase-sensitive terms below {phase_cutoff:.0f} Hz)" if phase_cutoff else ""))

    # Instantiate the loss with the per-model smoothing (loss restricted to the contexted core)
    _combined_loss = CombinedLoss(sample_rate, extra_small_cplx=(kind == 'offline'), core=loss_core,
                                  phase_cutoff_hz=phase_cutoff)
    _log_mel_loss  = LogMelLoss(sample_rate)

    # Learning rate: on resume continue with the LR that was in effect at the end of the resumed
    # checkpoint's epoch (ReduceLROnPlateau may have lowered it); restarting at LR undid that.
    state_json = os.path.join(checkpoint_out_dir, "train_state.json")
    lr0 = LR
    if resume and resume_ckpt_path and os.path.exists(state_json):
        try:
            with open(state_json, "r") as f:
                lr0 = float(json.load(f).get("lr_by_epoch", {}).get(str(initial_epoch), LR))
            print(f"[resume] continuing at learning rate {lr0:.3g} (epoch {initial_epoch})")
        except Exception as e:
            print(f"[resume] could not read {state_json} ({e}); using LR={LR}")
    elif not resume and os.path.exists(state_json):
        os.remove(state_json)

    ckpt_pattern = os.path.join(checkpoint_out_dir, "best_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras")
    if use_adversarial:
        recon_ref = new_w.get("_recon_ref")
        if not recon_ref:
            recon_ref = _measure_recon_loss(static_model, val_ds, _combined_loss)
            new_w["_recon_ref"] = recon_ref
            with open(weights_json, "w") as f:
                json.dump({k: (None if v is None else float(v)) for k, v in new_w.items()}, f, indent=2)
        discriminators = build_offline_discriminators()
        if resume and resume_ckpt_path:
            for i, d in enumerate(discriminators):
                dp = disc_weights_path(resume_ckpt_path, i)
                if os.path.exists(dp):
                    d.build((None, None, 1))
                    d.load_weights(dp)
                    print(f"[adversarial] resumed discriminator {i} from {dp}")
        fit_model = AdversarialTrainer(static_model, discriminators, recon_loss=_combined_loss,
                                       style_weight=LAMBDA_STYLE_RATIO,
                                       adv_weight=OFFLINE_ADV_WEIGHT * recon_ref,
                                       fm_weight=OFFLINE_FM_WEIGHT * recon_ref,
                                       core=loss_core, d_learning_rate=DISC_LR)
        fit_model.compile(optimizer=tf.keras.optimizers.Adam(lr0), jit_compile=False)
        print(f"[adversarial] {len(discriminators)} spectrogram discriminator(s) {DISC_RESOLUTIONS_OFFLINE}; "
              f"reconstruction reference {recon_ref:.4g} → adv weight {OFFLINE_ADV_WEIGHT * recon_ref:.4g}, "
              f"feature-matching weight {OFFLINE_FM_WEIGHT * recon_ref:.4g} (from epoch {OFFLINE_ADV_START_EPOCH + 1})")
        cp = GeneratorCheckpoint(ckpt_pattern, monitor='val_loss')
    else:
        static_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr0),
            loss=[ _combined_loss, "binary_crossentropy" ],
            loss_weights = [ 1.0,  LAMBDA_STYLE_RATIO ],
            metrics = [
                [ 'mae', _log_mel_loss, mean_hf_metric, multi_scale_stft_loss, rms_env_loss ],
                [ 'accuracy' ]
            ]
        )
        fit_model = static_model
        cp = ModelCheckpoint(filepath=ckpt_pattern, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    static_model.summary()

    # Adversarial runs keep their LAST weights (the reconstruction val_loss would roll back exactly what
    # the discriminator adds); every epoch is checkpointed — --promote-offline-checkpoint picks one.
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=LR_PATIENCE,
                       restore_best_weights=not use_adversarial)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR, patience=LR_PLATEAU_PATIENCE, min_lr=MIN_LR, mode='min', verbose=1)
    terminate_on_nan = TerminateOnNaN()
    tb = TensorBoard(log_dir=logs_out_dir, write_graph=False)

    # Use streaming realtime for demo (teacherless), keep offline demo builder unchanged
    if kind == 'realtime':
        demo_build_fn = build_realtime_streaming_model
        demo_build_args = {}  # teacherless streaming
        demo_safe_copy = True
    else:
        demo_build_fn = build_model_fn
        demo_build_args = build_model_fn_args_dynamic
        demo_safe_copy = False

    demo_cb = DemoCallback(sample_rate=sample_rate,
                           build_model_fn=demo_build_fn,
                           build_model_fn_args_dynamic=demo_build_args,
                           demo_checkpoints_out_dir=demo_checkpoints_out_dir,
                           use_safe_weight_copy=demo_safe_copy)

    cbs = [es, cp, reduce_lr, terminate_on_nan, tb, demo_cb, TrainStateCallback(state_json)]

    if kind == 'realtime':
        cbs.append(KDAlphaScheduler({**KD_ALPHA_SCHEDULE, **KD_STYLE_BLEND_SCHEDULE}))
    if use_adversarial:
        cbs.append(AdversarialRamp(OFFLINE_ADV_START_EPOCH, OFFLINE_ADV_RAMP_EPOCHS))

    fit_model.fit(train_ds,
                     initial_epoch=initial_epoch,
                     epochs=1 if DEBUG else epochs,
                     steps_per_epoch=steps_per_epoch,
                     validation_data=val_ds,
                     validation_steps=validation_steps,
                     callbacks=cbs)

    # Export runtime artifacts
    if kind == 'offline':
        # Keep original offline exports (unchanged)
        static_model.save(static_model_filepath)
        print("✅ Saved FP32 static model:", static_model_filepath)
        dynamic_model = build_model_fn(**build_model_fn_args_dynamic, dynamic_time=True)
        dynamic_model.set_weights(static_model.get_weights())
        dynamic_model.save(dynamic_model_filepath)
        print("✅ Saved FP32 dynamic model:", dynamic_model_filepath)
        return static_model, dynamic_model
    
    # For realtime, first save the trainable version to disk (which should have the offline teacher baked into its graph)
    if trainable_model_filepath is not None:
        static_model.save(trainable_model_filepath)
        print("✅ Saved FP32 trainable static model (with teacher):", trainable_model_filepath)

    # kind == 'realtime': always export self-contained streaming runtime models (the KD training
    # graph is NOT written to static_model_filepath — that path holds the streaming T=32 model)
    rt_dyn_stream = build_realtime_streaming_model(dynamic_time=True)
    copy_weights_by_name(rt_dyn_stream, static_model, verbose=True)

    rt_dyn_stream.save(dynamic_model_filepath)
    print("✅ Saved realtime streaming dynamic model:", dynamic_model_filepath)

    rt_static_stream = build_realtime_streaming_model()
    copy_weights_by_name(rt_static_stream, static_model, verbose=True)

    rt_static_stream.save(static_model_filepath)
    print("✅ Saved realtime streaming static model:", static_model_filepath)

    return static_model, rt_dyn_stream

def promote_offline_checkpoint(ckpt_path):
    """Write the baseline offline static + dynamic models from a training checkpoint (e.g. the epoch of
    an adversarial run that sounds / measures best), replacing the ones train_offline exported."""
    ck = load_keras_model(ckpt_path, compile=False)
    static = build_offline_model(include_dry_output=False)
    n = copy_weights_by_name(static, ck, verbose=True)
    dyn = build_offline_model(dynamic_time=True)
    dyn.set_weights(static.get_weights())
    static.save(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME))
    dyn.save(os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME))
    print(f"✅ Promoted {os.path.basename(ckpt_path)} ({n} layers) to {MODELS_OFFLINE_OUT_DIR}/"
          f"{{{OFFLINE_STATIC_MODEL_FILENAME}, {OFFLINE_DYNAMIC_MODEL_FILENAME}}}")

def train_offline(resume=False, resume_ckpt_path=None, init_weights_path=None):
    """init_weights_path: warm start (weights copied by layer name, e.g. from the previous offline
    model) instead of a random init; ignored on resume."""
    train_shared(
        kind='offline',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        epochs=OFFLINE_EPOCHS,
        max_steps_per_epoch=OFFLINE_MAX_STEPS_PER_EPOCH,
        init_weights_path=None if resume else init_weights_path,
        seg_len=OFFLINE_SEG_LEN,
        sample_rate=OFFLINE_SAMPLE_RATE,
        batch_size=OFFLINE_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
        build_model_fn=build_offline_model,
        build_model_fn_args_static={ "include_dry_output": False },   # training: 2 outputs
        build_model_fn_args_dynamic={ "include_dry_output": True },    # runtime/export: 3 outputs
        checkpoint_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
        logs_out_dir=LOGS_OFFLINE_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR
    )

def train_realtime(resume=False, resume_ckpt_path=None):
    offline_dynamic_model_filepath = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    if not os.path.exists(offline_dynamic_model_filepath):
        print("Offline model not found, run --train-offline first")
        sys.exit(1)

    # Load the offline teacher (the real non-causal model; it gets extra context per segment)
    offline_teacher = load_offline_teacher(offline_dynamic_model_filepath)
    print("Loaded offline teacher (non-causal, frozen) from", offline_dynamic_model_filepath)

    # Train the baseline realtime KD model; export streaming runtime artifacts
    train_shared(
        kind='realtime',
        resume=resume,
        resume_ckpt_path=resume_ckpt_path,
        seg_len=REALTIME_SEG_LEN,
        sample_rate=REALTIME_SAMPLE_RATE,
        batch_size=REALTIME_BATCH_SIZE,
        static_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
        dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
        trainable_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME),
        build_model_fn=build_realtime_training_model,                   # KD graph for training
        build_model_fn_args_static={ 'offline': offline_teacher },      # teacher embedded
        build_model_fn_args_dynamic={ 'offline': offline_teacher },     # teacher embedded
        checkpoint_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
        logs_out_dir=LOGS_REALTIME_OUT_DIR,
        demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
        dual_offline=True,
        max_steps_per_epoch=REALTIME_MAX_STEPS_PER_EPOCH
    )

def generate_fine_tune_demo(model, kind, sample_rate, pickup=PICKUP_PIEZO):
    out_dir = os.path.join(FINETUNE_DEMOS_OUT_DIR, kind)
    os.makedirs(out_dir, exist_ok=True)
    return generate_demo_from_files(
        model           = model,
        kind            = kind,
        sample_rate     = sample_rate,
        offset_sec      = FINETUNE_DEMO_OFFSET_SEC,
        sec             = FINETUNE_DEMO_SEC,
        path_src        = os.path.join(FINETUNE_SRC_DIR, "source.wav"),
        path_target     = os.path.join(FINETUNE_SRC_DIR, "target.wav"),
        out_dir         = out_dir,
        prediction_name = "demo_pred",
        use_finetuned=True,
        pickup=pickup
    )
