# Piezo2Mic — Training, Inference & Deployment Guide

A runbook for picking this project back up after time away. Last major overhaul: **July 2026**
(see "Current state" at the bottom for what changed and why a retrain is needed).

---

## 1. What this is (30-second refresher)

Two neural models convert a guitar **pickup signal** into the sound of the **same guitar mic'd up**:

| Model | Rate | Latency | Purpose |
|---|---|---|---|
| **Offline** | 96 kHz | non-causal (~64 ms look-ahead) | Quality reference / teacher / offline file processing |
| **Realtime** | 48 kHz | fully causal, streamed in 32-sample hops | The pedal (Raspberry Pi + HiFiBerry, TFLite) |

The realtime model is **knowledge-distilled** from the offline teacher during training
(waveform, STFT, stem, and feature KD — see `KD_ALPHA_SCHEDULE` in `piezo2mic.py`).

Two conditioning axes:
- **Playing style** (pick=1 / finger=0) — classified by the model at inference, can be forced.
- **Pickup type** (piezo=0 / magnetic=1) — **user-set, never classified** (you know what you
  plugged in). A gated input-adapter network normalizes magnetic input into the piezo-like
  domain; it is an exact pass-through at pickup=0. "electric" is an alias for magnetic —
  this is the electric→acoustic path.

Everything lives in one file: `piezo2mic.py`. The pedal runtime is `cpp/src/main.cpp`.
The desktop converter app is `piezo2mic_app/` (PySide6, shells out to `--process`).

---

## 2. Command quick reference

```bash
# Training (in order)
python piezo2mic.py --train-offline          # teacher first (~hours)
python piezo2mic.py --train-realtime         # student, needs the offline model
python piezo2mic.py --fine-tune              # optional: adapt to a foreign recording
python piezo2mic.py --export-tflite          # write the .tflite files for the Pi

# Listening checks
python piezo2mic.py --demo                   # Keras demos → out/demos/long/
python piezo2mic.py --demo-tflite            # same but through the exported TFLite
python piezo2mic.py --render-stems           # per-branch stems (main/pickattack/pickreso/IR)
python piezo2mic.py --render-stems-tflite

# Sanity / debugging
python piezo2mic.py --inspect-tflite-all     # I/O + streaming-state bundle verification
python piezo2mic.py --inspect-realtime       # Keras streaming model I/O summary
python piezo2mic.py --print-layers

# File processing
python piezo2mic.py --process --input in.wav --output out.wav \
    [--offline-only|--realtime-only] [--finetuned] \
    [--style auto|pick|finger] [--pickup piezo|magnetic|electric]

# Desktop UI (pip install pyside6)
python -m piezo2mic_app
```

---

## 3. Environment

```bash
pip install -r requirements.txt        # TF ≥2.19 (Keras 3 API required), ai-edge-litert,
                                       # numpy, scipy, soundfile; pyside6 only for the UI
```

- The code targets **Keras 3** (`LeakyReLU(negative_slope=...)`, registered serializables).
  Do not downgrade below TF 2.19 without checking.
- On Apple Silicon, `tensorflow-metal` works and was used for development.
- Training data is loaded **lazily** — importing `piezo2mic.py` or running `--process` no
  longer reads the corpus into RAM; only training/demo/stems commands do.

---

## 4. Training data

### Layout

```
input/
  piezo/
    pick/    source/take1_source.wav …   target/take1_target.wav …
    finger/  source/…                    target/…
  magnetic/
    pick/    source/…                    target/…
    finger/  source/…                    target/…
  finetune/
    source.wav + target.wav              (single pair, for --fine-tune)
```

- Files pair by basename: `takeN_source.wav` ↔ `takeN_target.wav`.
- The **legacy layout** `input/pick/…` and `input/finger/…` is still read and treated as
  piezo, so old data keeps working untouched.
- Groups that have no data are simply skipped — a piezo-only corpus trains exactly as before.

### Recording requirements

- **96 kHz / 24-bit**, source and target **time-aligned before training** (there is no
  auto-alignment in the pipeline).
- At least ~1 hour per style, roughly 50/50 pick vs finger.
- Source = pickup DI (piezo under-saddle, or magnetic soundhole pickup). Target = mic.
- **For magnetic support, record triplets**: piezo + magnetic soundhole pickup + mic
  *simultaneously* on the same acoustic guitar. Every take then yields two training pairs
  sharing one identical mic target, and later enables the cross-pickup consistency loss
  (`TODO(data)` marker in `load_training_data()` — not implemented until triplets exist).
- Train/val split is automatic: the **last ~20% of each file** is held out (contiguous tail,
  no leakage). Longer takes → more meaningful validation.

---

## 5. Training runbook

### Step 0 — fresh start vs resume

Training **auto-resumes** from the newest checkpoint in `out/checkpoints/<kind>/`.
Logs and demo checkpoints are wiped automatically at start; checkpoints are not.

```bash
# To force a fresh run:
rm -f out/checkpoints/offline/*.keras     # before --train-offline
rm -f out/checkpoints/realtime/*.keras    # before --train-realtime
```

### Step 1 — offline teacher

```bash
python piezo2mic.py --train-offline
```

What happens: 60-step warmup → loss-weight calibration (printed, includes the resonance-band
weight) → up to 40 epochs, EarlyStopping(patience=8, restores best), ReduceLROnPlateau.
Per-epoch listening demos land in `out/demos/checkpoints/offline/epoch_NNN/`.

Writes: `out/models/offline/piezo2mic_baseline_offline.keras` (static) and
`…_offline_dynamic.keras` (any-length, used as the KD teacher and for `--process`).

### Step 2 — realtime student (KD)

```bash
python piezo2mic.py --train-realtime      # requires the offline dynamic model
```

The offline teacher is causalized and embedded in the training graph; KD alphas anneal per
`KD_ALPHA_SCHEDULE`. Per-epoch demos use the actual streaming graph, so they sound like the
pedal will.

Writes to `out/models/realtime/`:
- `piezo2mic_baseline_trainable_realtime.keras` — KD training graph (teacher baked in)
- `piezo2mic_baseline_realtime.keras` — **streaming static T=32** (this is what gets exported)
- `piezo2mic_baseline_realtime_dynamic.keras` — streaming, any length (Python inference)

### Step 3 — listen before exporting

```bash
python piezo2mic.py --demo
python piezo2mic.py --render-stems       # is the pick attack/resonance sitting right?
```

### Step 4 — optional fine-tune on a foreign recording

Put an aligned `source.wav`/`target.wav` in `input/finetune/` (any guitar/mic you don't
control). Only FiLM + coloreq unfreeze; the mic target is lightly denoised; the realtime
model keeps distilling from the base teacher at end-of-anneal strengths.

```bash
python piezo2mic.py --fine-tune            # or --offline-only / --realtime-only
python piezo2mic.py --fine-tune-demo
```

### Step 5 — export TFLite for the Pi

```bash
python piezo2mic.py --export-tflite
python piezo2mic.py --inspect-tflite-all   # must print "OK: bundle matches streaming state sizes"
python piezo2mic.py --demo-tflite          # final listening check through the real artifact
```

Produces in `out/models/…`: `piezo2mic_baseline_realtime_t32_fp32.tflite` (the pedal model),
`piezo2mic_baseline_offline_t40960_fp32.tflite`, plus `finetuned` variants when present.

The realtime TFLite has 5 outputs (documented in `readme.md`): audio, style prob
(EMA-smoothed), dry passthrough, packed **states bundle**, and **splits** (per-state lengths).
The C++ runtime discovers all of this by shape/name — no code changes needed for re-exports.

---

## 6. Inference on a computer

```bash
# Acoustic with piezo pickup (default):
python piezo2mic.py --process --input in.wav --output out.wav --offline-only

# Electric guitar / magnetic pickup:
python piezo2mic.py --process --input di.wav --output out.wav --offline-only --pickup electric

# Force a style instead of auto-detect:
python piezo2mic.py --process … --style pick
```

Output sample rates: offline 96 k, realtime 48 k. **Level note:** since July 2026, `--process`
applies the same pick-branch make-up gains as the pedal and demos
(`*_RUNTIME_BASELINE_DB` constants), so files match the pedal at 0 dB knobs.

Desktop app: `python -m piezo2mic_app` — input/output pickers, style, model, and pickup
dropdowns; runs the same backend in a worker process.

---

## 7. Raspberry Pi deployment

Full build instructions (PREEMPT_RT kernel, bazel-built TFLite with XNNPACK, flatbuffers,
PortAudio, CMake flags): **`cpp/readme.md`**. HiFiBerry DAC8x/ADC8x config
(`/etc/asound.conf`, `config.txt` overlays): **`readme.md`** at the repo root.

```bash
cd cpp/build && cmake --build .
sudo ./stream_audio_cpp \
  --tflite ../../out/models/realtime/piezo2mic_baseline_realtime_t32_fp32.tflite \
  --blocksize 32 --input-device 0 --output-device 0
```

- `--blocksize 32` is **mandatory** for streaming models (hop must equal T; the binary
  refuses otherwise — a smaller hop corrupts the streaming states).
- Channel map, FX loop, reverb/delay/chorus/gate are documented in the header comment of
  `cpp/src/main.cpp`.
- Keyboard: `b` bypass, `q/a` pick-attack ±1 dB, `w/s` pick-reso, `e/d` IR, `r/f` wet,
  `t` style mode, `g` guided auto, `l` FX loop, **`u` pickup piezo↔magnetic**, `p` print.
- After a `git pull` on the Pi, **recompile** — the July 2026 changes touched the runtime
  (style handling, hop guard, pickup input, boost consistency). Note: those C++ changes were
  never compiled on-target; fix any trivial build breaks before gigging.

### Constants that must stay in sync (Python ↔ C++)

| `piezo2mic.py` | `cpp/src/main.cpp` | Value |
|---|---|---|
| `REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB` | `kPickAttackBaselineBoostDB` | 22 dB |
| `REALTIME_PICK_RESO_RUNTIME_BASELINE_DB` | `kPickResoBaselineBoostDB` | 32 dB |
| `KNOB_DEFAULT_LOW_CUT_SLOPE_DB_PER_OCT` | `Knobs.lc_slope_dbpoct` default | 6 dB/oct |

(Finetuned models were auditioned with an 8 dB pickreso boost — the pedal can't detect
"finetuned" from a TFLite file, so adjust the C++ constant manually if you deploy one.)

---

## 8. Sharp edges when modifying the code

- **Streaming state contract**: the exporter packs states into the bundle in **lexicographic
  order of the `realtime_state_*` input names**; the Python `TFLiteRunner` and the C++ runtime
  slice it back in the same order. Renaming a state input silently reorders the bundle for
  old binaries. Always re-run `--inspect-tflite-all` after touching states.
- **scipy `minimum_phase` halves FIR tap counts.** Any streaming state whose kernel goes
  through min-phase must be declared from the *actual* kernel length (see `_k_hp_probe`,
  `L_ir_ds`, `_minphase_decimator_len`) — declaring from the design order is off by 2×.
- **hop == T** everywhere for streaming inference. The Python paths and the C++ binary both
  enforce it now; don't relax it.
- **Old checkpoints/models**: they still load, and `copy_weights_by_name` seeds the new
  `*_cond_embed` (style+pickup) layer from legacy `*_style_embed` weights so behavior is
  preserved at pickup=0. But new capabilities (KD, adapter, pickup) only exist in **newly
  trained** weights — migrating is a stopgap, retraining is the real path.
- The style state `realtime_state_style_ema` stores the classifier **logit** (zero state =
  probability 0.5); the exported style output is already the smoothed probability.

---

## 9. Magnetic / electric→acoustic roadmap (where this was left)

1. **Record triplets** (piezo + magnetic soundhole pickup + mic, simultaneously, same
   acoustic guitar) → drop into `input/piezo/…` and `input/magnetic/…`.
2. Retrain offline → realtime. The adapter (`*_pickup_adapt_*`) learns the magnetic→piezo
   correction; it is zero-initialized so piezo behavior is untouched.
3. Implement the **cross-pickup consistency loss** at the `TODO(data)` marker in
   `load_training_data()` once triplet pairing is possible
   (`model(piezo_src) ≈ model(magnetic_src)` for the same take).
4. Try electric guitar DI through `--process --pickup electric`; expect a domain gap
   (longer sustain, no body coupling) → collect electric DI and fine-tune if needed.

---

## 10. Current state (as of 2026-07-05)

A full review + fix pass landed (uncommitted at time of writing): reinstated knowledge
distillation (was dead code), fixed a silent loss-weight bug, leak-free train/val split,
gain augmentation in base training, warmup loss calibration, three streaming-state size
bugs, offline block-seam right-context fix, hop guards, the pickup axis end-to-end, lazy
data loading, and level-parity between `--process`/demos/pedal. Verified by a smoke test
(graph builds, KD losses fire, 4-hop streaming state roundtrip, TFLite export + strict
bundle verification).

**The saved models in `out/models/` predate all of this.** They still run, but none of the
improvements are in their weights — the first thing to do when picking this up is a full
retrain (Section 5), then re-export, then rebuild the C++ on the Pi.
