import os

from .config import (
    FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME,
    FINETUNE_OFFLINE_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME,
    FINETUNE_REALTIME_STATIC_MODEL_FILENAME,
    FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME,
    MODELS_OFFLINE_OUT_DIR,
    MODELS_REALTIME_OUT_DIR,
    OFFLINE_DYNAMIC_MODEL_FILENAME,
    OFFLINE_STATIC_MODEL_FILENAME,
    REALTIME_DYNAMIC_MODEL_FILENAME,
    REALTIME_STATIC_MODEL_FILENAME,
    REALTIME_TFLITE_T,
    REALTIME_TRAINABLE_MODEL_FILENAME,
)
from .modelio import (
    load_keras_model,
)
from .models import (
    build_offline_model,
    build_realtime_streaming_model,
    build_realtime_training_model,
    copy_weights_by_name,
    load_offline_teacher,
)

    
def migrate_offline_models_inplace():
    """
    Add master EQ/phase/output knobs to offline models by rebuilding graphs and copying weights by name.
    Saves to the same baseline/finetuned filenames.
    """
    dst_offline_static_baseline  = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME)
    dst_offline_dynamic_baseline = os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME)
    dst_offline_static_ft        = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_STATIC_MODEL_FILENAME)
    dst_offline_dynamic_ft       = os.path.join(MODELS_OFFLINE_OUT_DIR, FINETUNE_OFFLINE_DYNAMIC_MODEL_FILENAME)

    # Load old
    if not os.path.exists(dst_offline_static_baseline) or not os.path.exists(dst_offline_dynamic_baseline):
        print("[migrate/offline] Missing baseline offline models; run --train-offline first.")
        return

    off_static_old  = load_keras_model(dst_offline_static_baseline,  compile=False)
    off_dynamic_old = load_keras_model(dst_offline_dynamic_baseline, compile=False)

    # Rebuild new graphs with new knobs
    off_static_new  = build_offline_model(dynamic_time=False)
    off_dynamic_new = build_offline_model(dynamic_time=True)

    copy_weights_by_name(off_static_new,  off_static_old,  verbose=True)
    copy_weights_by_name(off_dynamic_new, off_dynamic_old, verbose=True)

    off_static_new.save(dst_offline_static_baseline)
    off_dynamic_new.save(dst_offline_dynamic_baseline)
    print("✅ [migrate/offline] Saved updated baseline offline static/dynamic.")

    # Finetuned (if present)
    if os.path.exists(dst_offline_static_ft):
        off_static_ft_old  = load_keras_model(dst_offline_static_ft,  compile=False)
        off_dynamic_ft_old = load_keras_model(dst_offline_dynamic_ft, compile=False) if os.path.exists(dst_offline_dynamic_ft) else None

        off_static_ft_new  = build_offline_model(dynamic_time=False)
        copy_weights_by_name(off_static_ft_new, off_static_ft_old, verbose=True)
        off_static_ft_new.save(dst_offline_static_ft)
        print("✅ [migrate/offline] Saved updated finetuned offline static.")

        if off_dynamic_ft_old is not None:
            off_dynamic_ft_new = build_offline_model(dynamic_time=True)
            copy_weights_by_name(off_dynamic_ft_new, off_dynamic_ft_old, verbose=True)
            off_dynamic_ft_new.save(dst_offline_dynamic_ft)
            print("✅ [migrate/offline] Saved updated finetuned offline dynamic.")

def migrate_realtime_models_inplace():
    # Destination paths
    destination_realtime_baseline_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_finetuned_training_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME)
    destination_realtime_static_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_baseline_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_DYNAMIC_MODEL_FILENAME)
    destination_realtime_static_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_STATIC_MODEL_FILENAME)
    destination_realtime_dynamic_finetuned_streaming_model = os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_DYNAMIC_MODEL_FILENAME)
    
    # Guard against missing prerequisites instead of crashing mid-migration
    required = [
        os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME),
        os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME),
        os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_STATIC_MODEL_FILENAME),
        os.path.join(MODELS_OFFLINE_OUT_DIR, OFFLINE_DYNAMIC_MODEL_FILENAME),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[migrate/realtime] Missing prerequisite model(s):")
        for p in missing:
            print("  -", p)
        print("[migrate/realtime] Aborting migration.")
        return

    # Load the two old models to migrate from. We can use those to generate the dynamic models and eventually flatten all weights
    realtime_static_baseline_training_model_old = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)
    realtime_static_finetuned_training_model_old = load_keras_model(os.path.join(MODELS_REALTIME_OUT_DIR, FINETUNE_REALTIME_TRAINABLE_MODEL_FILENAME), compile=False)

    # Load the offline teacher (non-causal, dynamic-time; used by every training graph)
    offline_teacher = load_offline_teacher()

    # Now, moving into the actual migration, first rebuild training graphs
    realtime_static_baseline_training_model = build_realtime_training_model(offline=offline_teacher)
    realtime_dynamic_baseline_training_model = build_realtime_training_model(offline=offline_teacher, dynamic_time=True)
    
    realtime_static_finetuned_training_model = build_realtime_training_model(offline=offline_teacher)
    realtime_dynamic_finetuned_training_model = build_realtime_training_model(offline=offline_teacher, dynamic_time=True)
    
    # Now copy weights by name from the training models to the new training graphs
    copy_weights_by_name(realtime_static_baseline_training_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_baseline_training_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_static_finetuned_training_model, realtime_static_finetuned_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_finetuned_training_model, realtime_static_finetuned_training_model_old, verbose=True)    
    
    # Now write the streaming-friendly, stateful flat models
    realtime_static_baseline_streaming_model = build_realtime_streaming_model()
    realtime_dynamic_baseline_streaming_model = build_realtime_streaming_model(dynamic_time=True)
    realtime_static_finetuned_streaming_model = build_realtime_streaming_model()
    realtime_dynamic_finetuned_streaming_model = build_realtime_streaming_model(dynamic_time=True)
    
    copy_weights_by_name(realtime_static_baseline_streaming_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_baseline_streaming_model, realtime_static_baseline_training_model_old, verbose=True)
    copy_weights_by_name(realtime_static_finetuned_streaming_model, realtime_static_finetuned_training_model_old, verbose=True)
    copy_weights_by_name(realtime_dynamic_finetuned_streaming_model, realtime_static_finetuned_training_model_old, verbose=True)
    
    # Save the new training model to disk
    realtime_static_baseline_training_model.save(destination_realtime_baseline_training_model)
    realtime_static_finetuned_training_model.save(destination_realtime_finetuned_training_model)
    
    print(f"[migrate] Successfully wrote updated training models")
    
    realtime_static_baseline_streaming_model.save(destination_realtime_static_baseline_streaming_model)
    realtime_dynamic_baseline_streaming_model.save(destination_realtime_dynamic_baseline_streaming_model)
    realtime_static_finetuned_streaming_model.save(destination_realtime_static_finetuned_streaming_model)
    realtime_dynamic_finetuned_streaming_model.save(destination_realtime_dynamic_finetuned_streaming_model)

    print(f"[migrate] Successfully wrote updated streaming models (T={REALTIME_TFLITE_T})")
    print("✅ Migration complete.")
