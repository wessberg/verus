import os
import sys
import time

from . import config

from .config import (
    CHECKPOINT_OFFLINE_OUT_DIR,
    CHECKPOINT_REALTIME_OUT_DIR,
    DEMO_CHECKPOINTS_OFFLINE_OUT_DIR,
    DEMO_CHECKPOINTS_REALTIME_OUT_DIR,
    DEMO_PAIR_INDEX,
    FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME,
    FINETUNE_REALTIME_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_TFLITE_FP32_FILENAME,
    LOGS_OFFLINE_OUT_DIR,
    LOGS_REALTIME_OUT_DIR,
    LONG_DEMOS_OFFLINE_OUT_DIR,
    LONG_DEMOS_REALTIME_OUT_DIR,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_STATIC_MODEL_FILENAME,
    REALTIME_DYNAMIC_MODEL_FILENAME,
    REALTIME_PICK_ATTACK_CUTOFF_HZ,
    REALTIME_PICK_MASK_KNEE_DB,
    REALTIME_SAMPLE_RATE,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_FP32_FILENAME,
    REALTIME_TFLITE_T,
)
from .analysis import (
    analyze_attacks,
)
from .data import (
    all_pairs,
    assert_has_pairs,
    cleanup_and_prepare_train,
)
from .dsp import (
    resample_to_model_sr,
)
from .eq import (
    parse_master_eq_arg,
)
from .export import (
    _inspect_tflite_file,
    export_tflite_all,
    verify_tflite_bundle_file,
)
from .inference import (
    _style_label,
    prepare_and_generate_demo,
)
from .migrate import (
    migrate_offline_models_inplace,
    migrate_realtime_models_inplace,
)
from .modelio import (
    _print_model_io_summary,
    load_keras_model,
)
from .pickattack import (
    sweep_pickattack_params_with_tflite,
)
from .process import (
    _cli_arg_value,
    _format_process_write_message,
    _load_input_mono,
    _normalize_process_pickup,
    _normalize_process_style,
    _parse_process_pickup_arg,
    _parse_process_style_arg,
    _resolve_process_output_paths,
    process_audio_array_offline,
    process_audio_array_realtime,
)
from .stems import (
    _select_stems_model_path,
    run_render_stems_offline,
    run_render_stems_realtime,
    run_render_stems_tflite,
)
from .tflite import (
    generate_demo_tflite,
)
from .training import (
    fine_tune,
    generate_fine_tune_demo,
    promote_offline_checkpoint,
    train_offline,
    train_realtime,
)

def run_process_cli(input_path, output_arg, do_offline=True, do_realtime=True, use_finetuned=False, forced_style=None, pickup=None,
                    eq_knobs=None):
    forced_style = _normalize_process_style(forced_style)
    pickup = _normalize_process_pickup(pickup)
    eq_knobs = dict(eq_knobs or {})

    if not os.path.exists(input_path):
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    x_raw, input_sr = _load_input_mono(input_path)
    outputs = _resolve_process_output_paths(input_path, output_arg, do_offline, do_realtime)

    if forced_style is None:
        print("[process] style mode: auto")
    else:
        print(f"[process] style mode: forced {_style_label(forced_style)}")
    print(f"[process] pickup: {'magnetic' if pickup >= 0.5 else 'piezo'}")
    if eq_knobs:
        print(f"[process] master EQ: {eq_knobs}")

    if do_offline:
        offline_model_path = _select_stems_model_path("offline", use_finetuned=use_finetuned, prefer_static=False)
        if offline_model_path is None or not os.path.exists(offline_model_path):
            print("ERROR: offline model not found. Run --train-offline first.")
            sys.exit(1)

        print("[process/offline] Loading:", offline_model_path)
        offline_model = load_keras_model(offline_model_path, compile=False)
        x_off = x_raw if input_sr == OFFLINE_SAMPLE_RATE else resample_to_model_sr(x_raw, input_sr, OFFLINE_SAMPLE_RATE)
        p_off = process_audio_array_offline(
            model=offline_model,
            x_mod=x_off,
            output_path=outputs["offline"],
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        print(_format_process_write_message(
            label="Offline",
            output_path=outputs["offline"],
            sample_rate=OFFLINE_SAMPLE_RATE,
            avg_prob=p_off,
            forced_style=forced_style
        ))

    if do_realtime:
        # The dynamic streaming graph is stepped KERAS_STREAM_HOP samples at a time (the style EMA
        # is hop-invariant); the T=32 static graph would need ~90k model calls for a 1-minute file.
        realtime_model_path = _select_stems_model_path("realtime", use_finetuned=use_finetuned, prefer_static=False)
        if realtime_model_path is None or not os.path.exists(realtime_model_path):
            print("ERROR: realtime model not found. Run --train-realtime first.")
            sys.exit(1)

        print("[process/realtime] Loading:", realtime_model_path)
        realtime_model = load_keras_model(realtime_model_path, compile=False)
        x_rt = x_raw if input_sr == REALTIME_SAMPLE_RATE else resample_to_model_sr(x_raw, input_sr, REALTIME_SAMPLE_RATE)
        p_rt = process_audio_array_realtime(
            model=realtime_model,
            x_mod=x_rt,
            output_path=outputs["realtime"],
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        print(_format_process_write_message(
            label="Realtime",
            output_path=outputs["realtime"],
            sample_rate=REALTIME_SAMPLE_RATE,
            avg_prob=p_rt,
            forced_style=forced_style
        ))

def main():
    config.ensure_output_dirs()
    if "--cpu" in sys.argv:
        sys.argv.remove("--cpu")   # applied at import (next to `import tensorflow`)
    if len(sys.argv)>1 and sys.argv[1]=="--train-offline":
        init_from = None
        if "--init-from" in sys.argv:
            i = sys.argv.index("--init-from")
            if i + 1 >= len(sys.argv) or not os.path.exists(sys.argv[i + 1]):
                print("Usage: --train-offline [--init-from <offline_model.keras>]"); sys.exit(1)
            init_from = sys.argv[i + 1]
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_OFFLINE_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_OFFLINE_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_OFFLINE_OUT_DIR
        )
        if resume:
            if init_from:
                print(f"[init] checkpoints found — resuming them; --init-from {init_from} is ignored")
            train_offline(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_offline(init_weights_path=init_from)
    elif len(sys.argv)>1 and sys.argv[1]=="--train-realtime":
        resume, resume_ckpt = cleanup_and_prepare_train(
            logs_out_dir=LOGS_REALTIME_OUT_DIR,
            checkpoints_out_dir=CHECKPOINT_REALTIME_OUT_DIR,
            demo_checkpoints_out_dir=DEMO_CHECKPOINTS_REALTIME_OUT_DIR
        )
        if resume:
            train_realtime(resume=resume, resume_ckpt_path=resume_ckpt)
        else:
            train_realtime()
    elif (len(sys.argv)>1 and sys.argv[1]=="--demo"):
        print("Generating demo audio files...")
        args = set(sys.argv[2:])
        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)

        # Flags:
        # --static-realtime or --static: step the static T=32 streaming model exactly like the pedal
        #                                (default: dynamic streaming model at KERAS_STREAM_HOP, much faster)
        # --static-offline: force offline static artifact (if present)
        prefer_static_rt  = ("--static-realtime" in args) or ("--static" in args)
        prefer_static_off = ("--static-offline" in args)

        time_start = time.time()

        # Offline baseline: load saved offline model (dynamic by default unless --static-offline)
        if do_offline:
            prepare_and_generate_demo(
                kind="offline",
                sample_rate=OFFLINE_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_OFFLINE_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
                prefer_static=prefer_static_off
            )

        # Realtime baseline: dynamic streaming model by default; --static/--static-realtime → T=32 model
        if do_realtime:
            prepare_and_generate_demo(
                kind="realtime",
                sample_rate=REALTIME_SAMPLE_RATE,
                demo_out_dir=LONG_DEMOS_REALTIME_OUT_DIR,
                dynamic_model_filepath=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
                prefer_static=prefer_static_rt
            )

        time_end = time.time()
        print(f"✅ Demo generation complete (took {time_end - time_start:.1f} sec).")
        sys.exit(0)
    
    elif len(sys.argv)>1 and sys.argv[1] in ("--sweep-pickattack", "--foo"):
        # Developer tool: A/B the pick-attack detector parameters through Keras streaming + TFLite
        assert_has_pairs()
        sweep_pickattack_params_with_tflite(
            src_path=all_pairs[DEMO_PAIR_INDEX][0],
            tgt_path=all_pairs[DEMO_PAIR_INDEX][1],
            param_grid={
                "knee_db":       [REALTIME_PICK_MASK_KNEE_DB], # [4.0],
                "posthp_cutoff_hz": [REALTIME_PICK_ATTACK_CUTOFF_HZ] # [1000.0, 3000.0, 4000.0, 5000.0, 6000.0]
            },
            sec=5.0,
            offset_sec=480.0,
            base_out_dir="./out/fast_sweeps_ab",
            load_streaming_from=os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),  # or None for default
            hop=REALTIME_TFLITE_T,
            keras_stream_T=120000  # large T for Keras streaming
        )
    
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune":
        # --fine-tune [--offline-only|--realtime-only] [--pickup piezo|magnetic|electric]
        ft_args = sys.argv[2:]
        ft_pickup = _parse_process_pickup_arg(ft_args)
        offline_dyn, realtime_dyn = fine_tune(offline="--realtime-only" not in ft_args,
                                              realtime="--offline-only" not in ft_args,
                                              pickup=ft_pickup)
        
        if (offline_dyn is not None):
            generate_fine_tune_demo(model=offline_dyn, kind="offline", sample_rate=OFFLINE_SAMPLE_RATE, pickup=ft_pickup)
        if (realtime_dyn is not None):
            generate_fine_tune_demo(model=realtime_dyn, kind="realtime", sample_rate=REALTIME_SAMPLE_RATE, pickup=ft_pickup)

        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--fine-tune-demo":
        # Load finetuned models if present and render both
        offline_dyn_path  = os.path.join(MODELS_OFFLINE_OUT_DIR,  FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)
        realtime_dyn_path = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)

        offline_dyn  = load_keras_model(offline_dyn_path,  compile=False) if os.path.exists(offline_dyn_path)  else None
        realtime_dyn = load_keras_model(realtime_dyn_path, compile=False) if os.path.exists(realtime_dyn_path) else None

        if offline_dyn is None and realtime_dyn is None:
            print("Fine-tuned dynamic models not found, run --fine-tune first")
            sys.exit(1)

        ft_pickup = _parse_process_pickup_arg(sys.argv[2:])
        if offline_dyn is not None:
            generate_fine_tune_demo(model=offline_dyn, kind="offline", sample_rate=OFFLINE_SAMPLE_RATE, pickup=ft_pickup)
        if realtime_dyn is not None:
            generate_fine_tune_demo(model=realtime_dyn, kind="realtime", sample_rate=REALTIME_SAMPLE_RATE, pickup=ft_pickup)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--render-stems":
        # Flags (can be combined):
        # --finetuned        → finetuned dynamic/static models (if present)
        # --both             → baseline + finetuned
        # --all              → baseline + finetuned
        # --offline-only / --realtime-only limits branches
        # New:
        # --static or --static-realtime → force realtime STATIC T=32 model for stems
        args = set(sys.argv[2:])

        do_offline   = ("--realtime-only" not in args)
        do_realtime  = ("--offline-only"  not in args)
        do_finetuned = ("--finetuned"     in args)
        do_both      = ("--both"          in args)
        do_all       = ("--all"           in args)

        # New static realtime selector
        prefer_static_rt = ("--static-realtime" in args) or ("--static" in args)

        # OFFLINE: baseline or finetuned (existing behavior)
        if do_offline:
            if do_all:
                run_render_stems_offline(use_finetuned=False)
                run_render_stems_offline(use_finetuned=True)
            elif do_both:
                run_render_stems_offline(use_finetuned=False)
                run_render_stems_offline(use_finetuned=True)
            else:
                run_render_stems_offline(use_finetuned=do_finetuned)

        # REALTIME: baseline / finetuned; prefer static T=32 if requested
        if do_realtime:
            if do_all:
                run_render_stems_realtime(use_finetuned=False, prefer_static=prefer_static_rt)
                run_render_stems_realtime(use_finetuned=True,  prefer_static=prefer_static_rt)
            elif do_both:
                run_render_stems_realtime(use_finetuned=False, prefer_static=prefer_static_rt)
                run_render_stems_realtime(use_finetuned=True,  prefer_static=prefer_static_rt)
            else:
                run_render_stems_realtime(use_finetuned=do_finetuned, prefer_static=prefer_static_rt)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--render-stems-tflite":
        # Same options as --render-stems:
        #   --offline-only / --realtime-only
        #   --finetuned
        #   --both | --all
        #   --hop=N  (for realtime streaming hop)
        run_render_stems_tflite(sys.argv[2:])
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--print-layers":
        
        paths = [
            os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
        ]

        for p in paths:
            print("[offline/persist] loading:", p)
            m = load_keras_model(p, compile=False)
            
            for l in m.layers:
                # Print layer name
                print(f"Layer: {l.name}, Type: {type(l)}")
            print("-----")
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-realtime":
        args = set(sys.argv[2:])
        use_finetuned = ("--finetuned" in args)
        path = os.path.join(
            MODELS_REALTIME_OUT_DIR,
            FINETUNE_REALTIME_STATIC_MODEL_FILENAME if use_finetuned else REALTIME_STATIC_MODEL_FILENAME
        )
        print("[inspect-realtime] Loading:", path)
        m = load_keras_model(path, compile=False)
        _print_model_io_summary(m)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-tflite-file":
        if len(sys.argv) < 3:
            print("Usage: --inspect-tflite-file <path.tflite>"); sys.exit(1)
        path = sys.argv[2]
        _inspect_tflite_file(path)
        try:
            verify_tflite_bundle_file(path, strict=False)
        except Exception as e:
            print("verify_tflite_bundle_file (non-strict) raised:", e)
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--inspect-tflite-all":
        # Convenience: check both baseline and finetuned realtime TFLite files if present
        paths = [
            os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TFLITE_FP32_FILENAME),
            os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TFLITE_FP32_FILENAME),
        ]
        for p in paths:
            if os.path.exists(p):
                print("\n=== Inspecting:", p, "===")
                _inspect_tflite_file(p)
                try:
                    verify_tflite_bundle_file(p, strict=False)
                except Exception as e:
                    print("verify_tflite_bundle_file (non-strict) raised:", e)
            else:
                print("\n=== Missing:", p, "===")
        sys.exit(0)
    
    elif len(sys.argv)>1 and sys.argv[1]=="--export-tflite":
        
        # load the streaming model to use for conversion
        #m = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME), compile=False)
        #debug_streaming_state_lengths(m)
        
        
        print("Exporting FP32 TFLite models (offline T=40960, realtime T=32)…")
        export_tflite_all()
        sys.exit(0)

    elif len(sys.argv)>1 and sys.argv[1]=="--demo-tflite":
        args = set(sys.argv[2:])
        do_offline    = ("--realtime-only" not in args)
        do_realtime   = ("--offline-only"  not in args)
        use_finetuned = ("--finetuned"     in args)
        do_both       = ("--both" in args)
        
        # Measure the time it takes to do this:
        time_start = time.time()

        # Optional hop parsing: --hop=16
        hop = None
        for a in list(args):
            if a.startswith("--hop="):
                try:
                    hop = int(a.split("=",1)[1])
                except Exception:
                    hop = None

        if do_offline:
            generate_demo_tflite(kind="offline", use_finetuned=use_finetuned)

        if do_realtime:
            if do_both:
                # T=32 and large demo-T, both
                generate_demo_tflite(kind="realtime", use_finetuned=use_finetuned, hop=hop)
            else:
                generate_demo_tflite(kind="realtime", use_finetuned=use_finetuned, hop=hop)
        time_end = time.time()
        print(f"[TFLite/demo] All done in {time_end - time_start:.1f} seconds.")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == "--process":
        args_list = sys.argv[2:]
        args = set(args_list)

        input_path = _cli_arg_value(args_list, "--input")
        output_arg = _cli_arg_value(args_list, "--output")

        if input_path is not None and str(input_path).startswith("--"):
            input_path = None
        if output_arg is not None and str(output_arg).startswith("--"):
            output_arg = None

        forced_style = _parse_process_style_arg(args_list)
        pickup = _parse_process_pickup_arg(args_list)
        try:
            eq_knobs = parse_master_eq_arg(_cli_arg_value(args_list, "--eq"))
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if input_path is None or output_arg is None:
            print(
                "Usage:\n"
                "  verus --process --input <input_audio> --output <output_path> "
                "[--offline-only|--realtime-only|--both] [--finetuned] "
                "[--style <auto|pick|picked|finger>] [--pickup <piezo|magnetic|electric>]\n"
                "  [--eq lc_hz=80,lc_slope=12,ls_gain_db=0,ls_hz=200,ls_slope=12,mid_gain_db=0,mid_hz=1000,mid_q=1,"
                "hs_gain_db=0,hs_hz=4000,hs_slope=12]\n\n"
                "Style aliases:\n"
                "  --force-style <auto|pick|picked|finger>\n"
                "  --pick / --picked\n"
                "  --finger\n\n"
                "Notes:\n"
                "  - If no style is provided, style is auto-detected (same behavior as today).\n"
                "  - If a style is forced, the render uses that style.\n"
                "  - --pickup declares the source pickup type (default piezo). Use magnetic (or\n"
                "    its alias electric) for magnetic-pickup sources, e.g. electric guitars.\n"
                "  - The printed style average is still the classifier prediction, for reference.\n"
                "  - If neither --offline-only nor --realtime-only is passed, both models are run.\n"
                "  - If both models are run, --output is treated as a prefix or directory and\n"
                "    '_offline' / '_realtime' are appended automatically.\n"
                "  - Output is written at the model sample rate: offline=96k, realtime=48k.\n"
                "  - --eq applies the same post-model master EQ the pedal runs (RBJ biquads); default = off.\n"
            )
            sys.exit(1)

        do_offline = ("--realtime-only" not in args)
        do_realtime = ("--offline-only" not in args)

        if not do_offline and not do_realtime:
            print("ERROR: choose --offline-only, --realtime-only, or neither/--both.")
            sys.exit(1)

        use_finetuned = ("--finetuned" in args)

        run_process_cli(
            input_path=input_path,
            output_arg=output_arg,
            do_offline=do_offline,
            do_realtime=do_realtime,
            use_finetuned=use_finetuned,
            forced_style=forced_style,
            pickup=pickup,
            eq_knobs=eq_knobs
        )
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--analyze-attacks":
        # --analyze-attacks [--offline-only|--realtime-only] [--finetuned] [--as-trained] [--sec N]
        args_list = sys.argv[2:]
        sec = float(_cli_arg_value(args_list, "--sec", 20.0))
        for kind in ("offline", "realtime"):
            if ("--realtime-only" in args_list and kind == "offline") or ("--offline-only" in args_list and kind == "realtime"):
                continue
            analyze_attacks(kind, use_finetuned="--finetuned" in args_list, sec=sec,
                            as_trained="--as-trained" in args_list)
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--promote-offline-checkpoint":
        # --promote-offline-checkpoint <out/checkpoints/offline/best_model_epoch_NN_...keras>
        if len(sys.argv) < 3 or not os.path.exists(sys.argv[2]):
            print("Usage: --promote-offline-checkpoint <checkpoint.keras>"); sys.exit(1)
        promote_offline_checkpoint(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-realtime":
        migrate_realtime_models_inplace()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-offline":
        migrate_offline_models_inplace()
        sys.exit(0)
    elif len(sys.argv)>1 and sys.argv[1]=="--migrate-all":
        migrate_offline_models_inplace()
        migrate_realtime_models_inplace()
        sys.exit(0)
    else:
        print("No command given.")
        sys.exit(1)
    
if __name__=="__main__":
    main()
