# Model outputs

The model has 5 outputs:

1. Processed mono audio
    - Shape: [1, T, 1]
    - Name: realtime_audio_out

2. Style probability (pick vs finger)
    - Shape: [1, 1]
    - Name: realtime_style_prob_out

3. Unprocessed / dry mono signal, time‑aligned to model I/O
    - Shape: [1, T, 1]
    - Name: realtime_dry_out_export

4. Packed streaming state bundle
    - Shape: [1, N] (2‑D float)
    - Name: realtime_states_bundle_out

5. Splits (lengths of each state in the bundle)
    - Shape: [1, Nstates] or [Nstates] (int32)
    - Name: realtime_states_splits_out

# Python dependencies

`pip install soundfile tflite`

# Dead code removal

`vulture piezo2mic.py`

# Timeout prevention for ROCm

./rocm-idle-protection.sh