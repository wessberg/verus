# Piezo2Mic — Training, Inference & Deployment Guide

A runbook for picking this project back up after time away. Last major overhaul: **September 2026**
(see "Current state" at the bottom for what changed and why a retrain is needed).

---

## 1. What this is (30-second refresher)

Two neural models convert a guitar **pickup signal** into the sound of the **same guitar mic'd up**:

| Model | Rate | Latency | Purpose |
|---|---|---|---|
| **Offline** | 96 kHz | non-causal (~64 ms look-ahead) | Quality reference / teacher / offline file processing |
| **Realtime** | 48 kHz | fully causal, streamed in 32-sample hops | The pedal (Raspberry Pi + HiFiBerry, TFLite) |

The realtime model is **knowledge-distilled** from the offline teacher during training
(waveform, STFT, stem, and feature KD — see `KD_ALPHA_SCHEDULE` in `piezo2mic.py`). The teacher
is the real, non-causal offline model; it is fed extra context around each training segment and
its outputs are cropped back onto the student's segment.

Two conditioning axes:
- **Playing style** (pick=1 / finger=0) — classified by the model at inference, can be forced.
- **Pickup type** (piezo=0 / magnetic=1) — **user-set, never classified** (you know what you
  plugged in). A gated input-adapter network normalizes magnetic input into the piezo-like
  domain; it is an exact pass-through at pickup=0. "electric" is an alias for magnetic —
  this is the electric→acoustic path.

The models are **EQ-free**: the master EQ (low cut / low shelf / mid / high shelf) is a runtime
effect implemented as RBJ biquads in both the C++ pedal (`MasterEqIIR`) and Python
(`--process --eq ...`), with identical formulas. Phase invert, output gain and dry/wet stay in
the model as exact, trivial ops.

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
python piezo2mic.py --demo [--static]        # Keras demos → out/demos/long/ (--static: step T=32 like the pedal)
python piezo2mic.py --demo-tflite            # same but through the exported TFLite
python piezo2mic.py --render-stems           # per-branch stems (main/pickattack/pickreso/IR)
python piezo2mic.py --render-stems-tflite

# Sanity / debugging
python piezo2mic.py --inspect-tflite-all     # I/O + streaming-state bundle verification
python piezo2mic.py --inspect-realtime       # Keras streaming model I/O summary
python piezo2mic.py --print-layers
python piezo2mic.py --sweep-pickattack       # A/B the pick detector parameters (dev tool)

# File processing
python piezo2mic.py --process --input in.wav --output out.wav \
    [--offline-only|--realtime-only] [--finetuned] \
    [--style auto|pick|finger] [--pickup piezo|magnetic|electric] \
    [--eq lc_hz=80,lc_slope=12,ls_gain_db=0,mid_gain_db=0,hs_gain_db=0,...]

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
- scipy ≥ 1.14 is preferred (`minimum_phase(..., half=False)`); older versions fall back to an
  equivalent autocorrelation design automatically.
- Training data is loaded **lazily** — importing `piezo2mic.py` or running `--process` does not
  read the corpus into RAM; only training/demo/stems commands do.

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
- Style and pickup are read from the **directory names** (`pick`/`finger`, `piezo`/`magnetic`),
  not from substrings anywhere in the path.

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

Training **auto-resumes** from the newest checkpoint in `out/checkpoints/<kind>/`, and reuses
the loss-weight calibration stored next to it (`loss_weights.json`), so the objective does not
change mid-run. Logs and demo checkpoints are wiped automatically at start; checkpoints are not.

```bash
# To force a fresh run:
rm -f out/checkpoints/offline/*.keras  out/checkpoints/offline/loss_weights.json   # before --train-offline
rm -f out/checkpoints/realtime/*.keras out/checkpoints/realtime/loss_weights.json  # before --train-realtime
```

### How a training segment is built (both models)

```
[ left context | loss core (8192 samples) | right context (offline only) ]
```

Only the **core** enters the losses (and the KD terms, and the low-band penalties): every branch
sees real history there and the offline model sees its full look-ahead. The context sizes are
derived from the architecture constants:

| | left context | core | right context | segment |
|---|---|---|---|---|
| offline (96 k) | 19 416 (sum of per-layer left pads + adapter) | 8192 | 6170 (sum of look-aheads) | 33 778 |
| realtime (48 k) | 8430 (pick-reso branch incl. the mask's 85 ms envelope) | 8192 | 0 | 16 622 |

The realtime student's teacher copy of the segment is the same span at 96 k plus the offline
model's own context (19 416 + 6170) → 58 830 samples; the teacher outputs are cropped back to
the student's span (`Cropping1D`) before KD. Expect the realtime KD stage to cost about as much
per step as an offline training step; `REALTIME_BATCH_SIZE` is 8 for that reason.

The validation set is a **fixed, cached** set of at most `VAL_MAX_BATCHES` batches, so
EarlyStopping / ModelCheckpoint / ReduceLROnPlateau react to progress, not sampling noise.

### Step 1 — offline teacher

```bash
python piezo2mic.py --train-offline
```

What happens: 60-step warmup → loss-weight calibration (printed and saved to
`out/checkpoints/offline/loss_weights.json`) → up to 40 epochs, EarlyStopping(patience=8,
restores best), ReduceLROnPlateau. Per-epoch listening demos land in
`out/demos/checkpoints/offline/epoch_NNN/`.

Writes: `out/models/offline/piezo2mic_baseline_offline.keras` (static) and
`…_offline_dynamic.keras` (any-length, used as the KD teacher and for `--process`).

### Step 2 — realtime student (KD)

```bash
python piezo2mic.py --train-realtime      # requires the offline dynamic model
```

The offline teacher is embedded frozen (in inference mode) in the training graph; KD alphas
anneal per `KD_ALPHA_SCHEDULE`. Per-epoch demos step the actual streaming graph hop-wise, so
they track style exactly like the pedal will.

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
model keeps distilling from the base teacher at end-of-anneal strengths (feature KD is seeded
from the base trainable model, or disabled if that file is missing).

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

The realtime TFLite has **4 outputs** (documented in `readme.md`): audio, style prob
(EMA-smoothed), packed **states bundle**, and **splits** (per-state lengths). There is exactly
one 3-D output so the runtimes can pick the audio tensor by shape. The export refuses to fall
back to `SELECT_TF_OPS` for realtime models — the pedal only links the builtin ops.

---

## 6. Inference on a computer

```bash
# Acoustic with piezo pickup (default):
python piezo2mic.py --process --input in.wav --output out.wav --offline-only

# Electric guitar / magnetic pickup:
python piezo2mic.py --process --input di.wav --output out.wav --offline-only --pickup electric

# Force a style instead of auto-detect:
python piezo2mic.py --process … --style pick

# Apply the pedal's master EQ (same biquads as the C++):
python piezo2mic.py --process … --eq lc_hz=80,lc_slope=24,hs_gain_db=2
```

Output sample rates: offline 96 k, realtime 48 k. **Level note:** `--process` applies the same
pick-branch make-up gains as the pedal and demos (`*_RUNTIME_BASELINE_DB` constants), so files
match the pedal at 0 dB knobs. The realtime path runs the dynamic streaming graph 1024 samples
per hop through `KerasStreamingRunner`; the style EMA is hop-invariant, so this matches the
pedal's 32-sample stepping to within float noise and is ~30× faster than before.

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
- The master EQ lives in the pedal (`MasterEqIIR`, driven by the existing `gKnobs` EQ fields /
  web UI) and no longer in the model; old TFLite exports that still have EQ inputs get neutral
  values.
- Runtime hygiene (Sept 2026): the inference thread sleeps on a semaphore instead of spinning a
  core; DI and processed audio are pushed in lockstep and popped together (no drift after an
  xrun); a backlog beyond one hop is trimmed (latency cannot creep); all FX buffers are allocated
  before the stream starts; telemetry (`xruns`, `inDrops`, `resyncs`, proc time) is printed by
  the main thread, never from RT threads.
- After a `git pull` on the Pi, **recompile**. The September 2026 C++ changes were syntax-checked
  against stub headers only (no TFLite/PortAudio toolchain on the dev Mac); fix any trivial
  build breaks before gigging.

### Constants that must stay in sync (Python ↔ C++)

| `piezo2mic.py` | `cpp/src/main.cpp` | Value |
|---|---|---|
| `REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB` | `kPickAttackBaselineBoostDB` | 22 dB |
| `REALTIME_PICK_RESO_RUNTIME_BASELINE_DB` | `kPickResoBaselineBoostDB` | 32 dB |
| `master_eq_sos()` formulas / `master_eq_defaults()` | `MasterEqIIR` / `Knobs` EQ defaults | LC 20 Hz = bypass, 6 dB/oct, all gains 0 dB |

(Finetuned models were auditioned with an 8 dB pickreso boost — the pedal can't detect
"finetuned" from a TFLite file, so adjust the C++ constant manually if you deploy one.
The pedal's output gain defaults to −3 dB; Python uses 0 dB.)

---

## 8. Sharp edges when modifying the code

- **Streaming state contract**: the exporter packs states into the bundle in **lexicographic
  order of the `realtime_state_*` input names**; the Python `TFLiteRunner` and the C++ runtime
  slice it back in the same order. Renaming a state input silently reorders the bundle for
  old binaries. Always re-run `--inspect-tflite-all` after touching states.
- **Min-phase filters**: every FIR goes through `_to_minimum_phase()`, which keeps the designed
  magnitude AND length. scipy's raw `minimum_phase()` default (`half=True`) halves both the taps
  and the dB attenuation — that is what the code used before September 2026, so all "129-tap
  60 dB" filters were actually 65-tap 30 dB filters. Streaming states are still declared from
  the actual kernel lengths (`_k_hp_probe`, `_minphase_decimator_len`), so they follow.
- **Loss core**: `CombinedLoss`, `FinetuneCombinedLossLite`, the KD layers and
  `HFLowbandPenalty` all crop to `*_LOSS_CORE`. If you change a kernel length or dilation,
  `REALTIME_TRAIN_LEFT_CTX` / `OFFLINE_TRAIN_LEFT_CTX` update automatically; if you add a branch
  with a longer memory, extend `_realtime_train_left_context()`.
- **Teacher context**: `segment_generator_dual` hands the teacher `KD_TEACHER_LEFT_CTX_SAMPLES`
  + 2×segment + `KD_TEACHER_RIGHT_CTX_SAMPLES` samples; `build_realtime_training_model` crops.
  Never "causalize" the offline model by moving its padding — that misaligns every residual
  block (the conv path is delayed, the skip path is not).
- **hop == T** everywhere for static streaming inference. The Python paths and the C++ binary
  both enforce it. Dynamic-T streaming graphs may be stepped with any multiple of 8
  (`KERAS_STREAM_HOP`); the style EMA rescales its coefficient to the hop length.
- **Streaming coloreq taps** are initialised explicitly (`taps_init`) because the training
  graph's `BandpassFIR` weight has a different length and `copy_weights_by_name` skips it.
- **Old checkpoints/models**: they still load through `load_keras_model()` — a pre-Sept-2026
  file fails the plain load (its fixed-FIR weights are half the current length) and is then
  reloaded with the legacy filter designs (`legacy_minphase_filters()`), so it behaves exactly as
  it did; the legacy master-EQ layer classes are kept for deserialization and the knob feeds carry
  neutral values for the old in-graph EQ inputs. `copy_weights_by_name` seeds the new
  `*_cond_embed` layer from legacy `*_style_embed` weights and skips the embedded teacher. But the
  new graphs have no master EQ, different state sizes and properly designed filters — migrating
  old weights is a stopgap, retraining is the real path.
- The style state `realtime_state_style_ema` stores the classifier **logit** (zero state =
  probability 0.5); the exported style output is already the smoothed probability.
- `BlendScalar` trainability is a layer-level flag (Keras 3 cannot re-enable a variable created
  with `trainable=False`).

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

## 10. Current state (as of 2026-09-02)

A second review + fix pass landed on top of the July 2026 overhaul:

- **KD teacher**: the causalized teacher clone (misaligned residual blocks) is gone; the real
  non-causal offline model is the teacher, fed with context and cropped.
- **Losses on the contexted core only**; segments carry the full left context (the realtime IR
  branch alone needed more history than the old 4457-sample segment had).
- **Min-phase filters now have their designed magnitude** (they had the square root of it).
  Expect the analytic pick detector to fire a little more eagerly with the sharper 2.5 kHz
  sidechain HPF; re-tune `REALTIME_PICK_MASK_THRESHOLD_DB` by ear (`--sweep-pickattack`).
- **Master EQ moved out of the model** into the runtimes (C++ biquads, `--process --eq`).
  ~35% fewer MACs in the realtime model; the low-cut now actually cuts.
- Streaming coloreq taps, fine-tune KD projections, `BlendScalar` unfreezing, hop-invariant
  style EMA, `KerasStreamingRunner` for every Keras streaming path (per-epoch demos, fine-tune
  demos, `--process`), fixed cached validation set, persisted loss-weight calibration,
  per-sample gain augmentation, TFLite export without the redundant dry output.
- C++: lockstep DI/processed rings with backlog trimming, semaphore-driven inference thread,
  no RT-thread printing, DSP prewarm, post-model master EQ, cached input indices.

**The saved models in `out/models/` predate all of this.** They still run, but none of the
improvements are in their weights — the first thing to do when picking this up is a full
retrain (Section 5), then re-export, then rebuild the C++ on the Pi.
