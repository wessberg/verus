# Verus — Training, Inference & Deployment Guide

A runbook: how the system works and how to run it. Planned work is in `ROADMAP.md`.
**No trained models are checked in:** `out/models/` is created by training (§5), and the first
run is from scratch.

---

## 1. What this is (30-second refresher)

Two neural models convert a guitar **pickup signal** into the sound of the **same guitar mic'd up**:

| Model | Rate | Latency | Purpose |
|---|---|---|---|
| **Offline** | 96 kHz | non-causal (~64 ms look-ahead) | Quality reference / teacher / offline file processing |
| **Realtime** | 48 kHz | fully causal, streamed in 32-sample hops | The pedal (Raspberry Pi + HiFiBerry, TFLite) |

The realtime model is **knowledge-distilled** from the offline teacher during training
(waveform, STFT, stem, and feature KD — see `KD_ALPHA_SCHEDULE` in `src/verus/config.py`). The teacher
is the real, non-causal offline model; it is fed extra context around each training segment, its
outputs are cropped back onto the student's segment and brought to 48 kHz with the same zero-phase
anti-alias filter `resample_poly` used for the 48k targets. KD losses are **training-only** —
`val_loss` (EarlyStopping, checkpoints, LR plateau) is the real objective only.

Two conditioning axes:
- **Playing style** (pick=1 / finger=0) — classified by the model at inference, can be forced.
- **Pickup type** (piezo=0 / magnetic=1) — **set by you, never classified** (you know what you
  plugged in). A gated input-adapter network normalizes magnetic input into the piezo-like
  domain; it is an exact pass-through at pickup=0. "electric" is an alias for magnetic —
  this is the electric→acoustic path.

The models are **EQ-free**: the master EQ (low cut / low shelf / mid / high shelf) is a runtime
effect implemented as RBJ biquads in both the C++ pedal (`MasterEqIIR`) and Python
(`--process --eq ...`), with identical formulas. Phase invert, output gain and dry/wet stay in
the model as exact, trivial ops.

The Python package is `src/verus/` (run it with `verus ...` after `pip install -e .`, or
`PYTHONPATH=src python -m verus ...`; always from the repository root, because `input/` and `out/` are
resolved relative to the working directory). The pedal runtime is `runtime/` (source layout: `runtime/README.md`). The desktop
converter app is `src/verus/app/` (PySide6, runs `--process`).

| Module | Holds |
|---|---|
| `config.py` | every constant (rates, model sizes, loss weights, paths, filenames) and the derived context lengths |
| `dsp.py`, `eq.py` | min-phase and FIR design, resampling; the master EQ biquads |
| `layers.py`, `blocks.py`, `kd.py` | custom Keras layers; graph-building blocks (gates, FiLM, pick branches); distillation layers |
| `losses.py`, `adversarial.py` | the loss terms; the spectrogram discriminators and adversarial trainer |
| `data.py`, `knobs.py`, `contexts.py` | corpus loading and segment generators; runtime knob feeds; streaming context sizes |
| `models.py`, `modelio.py`, `migrate.py` | offline / realtime / streaming model builders; loading and tensor-name helpers; migrating old `.keras` files |
| `training.py` | `--train-offline`, `--train-realtime`, `--fine-tune`, callbacks |
| `export.py`, `tflite.py` | TFLite export and verification; the Python TFLite runner |
| `inference.py`, `process.py` | demos and the Keras streaming runner; `--process` |
| `stems.py`, `analysis.py`, `pickattack.py` | stem rendering; attack analysis; pick-attack parameter sweep |
| `cli.py` | argument handling and the `main()` entry point |

---

## 2. Command quick reference

```bash
# Training (in order)
verus --train-offline [--init-from <offline.keras>]   # teacher first (§5 Step 1)
verus --train-realtime         # student, needs the offline model
verus --fine-tune [--pickup piezo|magnetic|electric]   # optional: adapt to a foreign recording
verus --export-tflite          # write the .tflite files for the Pi

# Listening checks
verus --demo [--static]        # Keras demos → out/demos/long/ (--static: step T=32 like the pedal)
verus --demo-tflite            # same but through the exported TFLite
verus --render-stems           # per-branch stems (main/pickattack/pickreso/IR)
verus --render-stems-tflite

# Sanity / debugging
verus --analyze-attacks        # pick/finger ATTACK fidelity vs the mic (see §5, "Measuring the attack")
verus --promote-offline-checkpoint <ckpt.keras>   # export a chosen offline checkpoint
verus --inspect-tflite-all     # I/O + streaming-state bundle verification
verus --inspect-realtime       # Keras streaming model I/O summary
verus --print-layers
verus --sweep-pickattack       # A/B the pick detector parameters (dev tool)

# File processing
verus --process --input in.wav --output out.wav \
    [--offline-only|--realtime-only] [--finetuned] \
    [--style auto|pick|finger] [--pickup piezo|magnetic|electric] \
    [--eq lc_hz=80,lc_slope=12,ls_gain_db=0,mid_gain_db=0,hs_gain_db=0,...]

# Electric guitar: match the acoustic's magnetic pickup to an electric DI (§4, "EQ-match recordings")
python tools/eq_match.py input/eqmatch/acoustic_magnetic.wav input/eqmatch/electric_humbucker.wav

# Desktop UI (pip install pyside6)
python -m verus.app
```

---

## 3. Environment

```bash
pip install -r requirements.txt        # TF ≥2.19 (Keras 3 API required), ai-edge-litert,
                                       # numpy, scipy, soundfile; pyside6 only for the UI
```

- The code targets **Keras 3** (`LeakyReLU(negative_slope=...)`, registered serializables).
  Do not downgrade below TF 2.19 without checking.
- On Apple Silicon, `tensorflow-metal` works and was used for development. For training on the
  16 GB dev Mac pass **`--cpu`** (any command; hides the GPU before TF initialises): measured on the
  offline step, the Metal GPU was no faster (1.6–2.0 vs 2.0 s/step at batch 1) and — CPU and GPU
  share the unified memory — took ~1.4 GB more of it, pushing the machine into swap.
- **Memory (16 GB Mac, 2026-09-23).** Offline training step, CPU, adversarial objective: batch 1
  ≈ 5.3–7.7 GB peak RSS, batch 2 ≈ 7.4 GB and swapping, batch 4 > 8 GB (it swapped the machine
  until the SSD filled and macOS crashed). Hence `OFFLINE_BATCH_SIZE = 1`. With the usual apps
  open only ~6–10 GB is available; close what you can, keep ≥ 15 GB of disk free (swap lives on
  it), and run long jobs under a guard that kills them when swap grows (see §5 Step 1).
- scipy ≥ 1.14 is preferred (`minimum_phase(..., half=False)`); older versions fall back to an
  equivalent autocorrelation design automatically.
- The training corpus is **never held in RAM**: `load_training_data()` only scans the pairs and
  every training segment is read from its WAV on demand (`_LazyMono`; bit-identical to the old
  whole-file arrays at the file rate, ≤ 4e-9 where it resamples). As float32 the 4 × 1 h × 96 kHz
  takes were 5.5 GB.

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
- **For magnetic support, record the pickups together**: piezo + magnetic soundhole pickup + the
  mics *simultaneously* on the same acoustic guitar ("triplets"; with two mics, the four-track
  session below). Every take then yields two training pairs sharing one identical mic target, and
  later enables the cross-pickup consistency loss (`TODO(data)` marker in `load_training_data()` —
  not implemented until such takes exist).
- Train/val split is automatic: the **last ~20% of each file** is held out (contiguous tail,
  no leakage). Longer takes → more meaningful validation.

### The four-track session (piezo + magnetic + two mics)

The next corpus is recorded the way acoustic guitar is usually recorded — one mic in front of the
soundhole, one at the shoulder — so the model can output that stereo pair (`ROADMAP.md`). **What to play, how, and the session step by step:
`RECORDING.md`.** The requirements:

- **Four tracks at once**, through one interface (one clock), 96 kHz / 24-bit: piezo DI, magnetic
  soundhole pickup DI (both on high-impedance inputs), **mic 1 in front of the soundhole**, **mic 2
  at the shoulder pointing down at the 12th fret**. ≥ 1 h per style, as above.
- Keep the mic positions fixed for the whole session and write them down (mic models, distances,
  angles; a photo helps) next to the takes — the model learns this exact setup.
- The soundhole pickup with the same settings as the EQ-match pair (below).
- **Alignment** (a prep tool is planned, see `ROADMAP.md`; by hand until then — details and
  the reasons: `RECORDING.md` "Preparing the files"): move the two mic tracks *together, as a
  group*, until the earlier of the two lines up with the piezo, measured on the bridge taps (the old
  takes' midband onsets line up at a median of 0.0 ms fingerstyle, −1.5 ms picked). Never nudge one mic against the other — their relative delay
  is the stereo image. The two pickups need no shift (same clock, no acoustic path). Each take
  starts with three knuckle taps on the bridge as a reference (`RECORDING.md`).
- **Files** — the same take has the same name under `piezo/` and `magnetic/`; that is how the triplet
  pairing (planned) will find it:

  ```
  input/piezo/<style>/source/takeN_source.wav     piezo, mono
  input/magnetic/<style>/source/takeN_source.wav  magnetic, mono
  input/piezo/<style>/target/takeN_target.wav     stereo: channel 1 = soundhole mic, channel 2 = 12th-fret mic
  input/magnetic/<style>/target/takeN_target.wav  a hard link to the file above (ln: no extra disk)
  ```

  Until stereo training exists the loader reads channel 1 only — the soundhole mic, the
  closest to where the old corpus's mic stood (a small condenser some distance off the soundhole,
  angled).
- **Disk:** 2 h of four tracks is ~8.3 GB in the DAW session and ~8.3 GB again once exported into
  `input/`; training needs ≥ 6 GB free (`tools/train_guard.py`).
  Record the session to an external drive and free space before exporting, or keep `input/` on an
  external SSD (training reads per segment, so that's fine; the hard link then needs both files on
  that drive).

### EQ-match recordings (fake electric sources)

An electric guitar can never have an aligned mic target (the acoustic's mic can't hear the
electric's performance), so the electric case is trained from **fake electric** sources: the
acoustic's magnetic recordings run through an EQ that matches them to a real electric DI
(`ROADMAP.md`). The EQ comes from `tools/eq_match.py`, which compares the long-term spectra of two
recordings of the same material, one per guitar:

- Both through the **same interface input and gain**, a high-impedance instrument input — ideally
  the kind of front end the pedal will have (a passive pickup's tone depends on the load). Same cable, 96 kHz / 24-bit, mono, unprocessed (no amp sim, EQ, compression or
  normalizing). Set the gain so the louder guitar peaks around −6 dBFS; don't change it in between.
- **Electric:** the neck humbucker (its position is closest to the soundhole's, so the match is most
  faithful), volume and tone where you'd play them. A second take on the bridge pickup, as a
  separate file, is a bonus.
- **Acoustic:** the soundhole pickup only (any body sensor or mic blend off), with exactly the
  settings the magnetic triplets will use — the EQ is only valid for that signal.
- **The same material on both, ≥ 5 minutes each, in the same style** (picked with picked): all six
  strings and the part of the neck you use, chords and single notes, similar dynamics, same tuning.
  The match is limited by *what was played*: in a synthetic test, different 5-min passages put the
  curve ~1.5 dB RMS off, 1 min ~4 dB, picked vs fingerstyle ~4 dB.
- **~5 s of silence** (strings muted, hands off) at the start and end: the quietest frames give each
  file's noise floor (hum, hiss), which is subtracted.
- Save as `input/eqmatch/electric_humbucker.wav` and `input/eqmatch/acoustic_magnetic.wav`
  (training doesn't read `input/eqmatch/`).

`python tools/eq_match.py input/eqmatch/acoustic_magnetic.wav input/eqmatch/electric_humbucker.wav` writes
to `out/eqmatch/`: the curve (`eq_curve.csv`), the correction as minimum-phase FIRs at 96 and 48 kHz
(`electric_match_ir_{96k,48k}.wav`, loadable in any IR/convolution plugin), `acoustic_as_electric.wav`
(the acoustic through the FIR at the electric's level — A/B it against the electric DI by ear) and
`eq_match.png`. It prints a **split-half check**: curves from the first and second halves of the
recordings; the full curve is off by roughly half that figure, and above ~3 dB it asks for longer or
more alike recordings. Options: `--help`.

---

## 5. Training runbook

### Step 0 — fresh start vs resume

Training **auto-resumes** from the newest checkpoint in `out/checkpoints/<kind>/`, and reuses
the loss-weight calibration stored next to it (`loss_weights.json`), so the objective does not
change mid-run. Logs and demo checkpoints are wiped automatically at start; checkpoints are not.

```bash
# To force a fresh run:
rm -f out/checkpoints/offline/*.keras out/checkpoints/offline/*.weights.h5 out/checkpoints/offline/loss_weights.json   # before --train-offline
rm -f out/checkpoints/realtime/*.keras out/checkpoints/realtime/loss_weights.json  # before --train-realtime
```

On resume the learning rate continues from where ReduceLROnPlateau had it at the resumed
checkpoint's epoch (`out/checkpoints/<kind>/train_state.json`, written every epoch; removed
automatically on a fresh start).

### How a training segment is built (both models)

```
[ left context | loss core (8192 samples) | right context (offline only) ]
```

Only the **core** enters the losses (and the KD terms, and the low-band penalties): every branch
sees real history there and the offline model sees its full look-ahead. The context sizes are
derived from the architecture constants:

| | left context | core | right context | segment |
|---|---|---|---|---|
| offline (96 k) | 19 512 (sum of per-layer left pads + colour EQ 96 + adapter 24) | 8192 | 6170 (sum of look-aheads) | 33 874 |
| realtime (48 k) | 8494 (pick-reso branch incl. the mask's 85 ms envelope, + colour EQ 64 + adapter 24) | 8192 | 0 | 16 686 |

The colour EQ filters the SUM of the branches and the pickup adapter feeds all of them, so both
add to the longest branch (`_offline_train_left_context`, `_realtime_train_left_context`).

The realtime student's teacher copy of the segment is the same span at 96 k plus the offline
model's own context (19 512 + 6170) → 59 054 samples; the teacher outputs are cropped back to
the student's span (`Cropping1D`, keeping `KD_TEACHER_DECIM_MARGIN` = 20 samples on the right for
the decimation filter) before KD. Expect the realtime KD stage to cost about as much per step as
an offline training step; `REALTIME_BATCH_SIZE` is 8 for that reason.

The validation set is a **fixed, cached** set of at most `VAL_MAX_SEGMENTS` segments (256 offline,
512 realtime, whatever the batch size) — including the
random 50 % style-forcing mask, which is drawn once before caching — so EarlyStopping /
ModelCheckpoint / ReduceLROnPlateau react to progress, not sampling noise.

### Step 1 — offline teacher

```bash
# from scratch (a machine with headroom; see Budget below)
verus --train-offline
# 16 GB Mac: CPU, batch 1
verus --train-offline --cpu
# warm start from an earlier offline model (`--init-from`, see below)
verus --train-offline --cpu --init-from <offline.keras>
```

What happens: (from scratch only: 60-step warmup →) loss-weight calibration over the 256 fixed
validation segments (`CALIB_SEGMENTS`; printed and saved to `out/checkpoints/offline/loss_weights.json`) → `OFFLINE_EPOCHS` (12) epochs of
`OFFLINE_MAX_STEPS_PER_EPOCH` (2000) steps, EarlyStopping(patience=8), ReduceLROnPlateau. Per-epoch
listening demos land in `out/demos/checkpoints/offline/epoch_NNN/`.

**Budget.** At batch 1 on the dev Mac's CPU a step is ~2.0 s, so an epoch is ~70 min and the 12
epochs ~14 h; one full pass over the corpus would be ~31k steps (~17 h). A from-scratch run needs
far more than 12 epochs — that is a job for a bigger machine (batch 4, `OFFLINE_EPOCHS` ≈ 40,
`OFFLINE_MAX_STEPS_PER_EPOCH = None` restores the old full-pass epochs). `--init-from` copies weights
by layer name and shape from an earlier offline model: everything is carried over except layers that
are new (e.g. the pickup adapter; `cond_embed` is seeded from the old style embedding) and fixed
FIRs whose length changed (they keep the current designs). It is ignored when checkpoints exist
(the run resumes them instead).

**Run it guarded** — `tools/train_guard.py` samples system memory, swap and disk every 0.5 s, logs a
trend line every 5 min, keeps the Mac awake (`caffeinate -i -w`) and kills the run if available
memory < 1.5 GB, swap grows > 1.5 GB or free disk < 6 GB; a killed run resumes from its last epoch
with the same command. (Closing the lid still sleeps a MacBook.)

```bash
python tools/train_guard.py --log out/logs/train_offline_guard.log -- \
    verus --train-offline --cpu \
    > out/logs/train_offline.log 2>&1
```

**Objective (since 2026-09-23), aimed at the pick attack.** The mic's attack HF is mostly not
phase-predictable from the piezo (and the pick's contact click precedes the string release the
piezo registers by ~6–7 ms), so phase-sensitive losses above a few kHz can only be minimised by
predicting ~nothing — the old models were 12–13 dB short of the mic's attack HF on picked playing.
Now:
- the phase-sensitive terms (waveform L1, pre-emphasis HF, second difference, complex STFT) only
  see the band below `OFFLINE_PHASE_LOSS_CUTOFF_HZ` (3 kHz; zero-phase low-pass of both signals);
- above it the model is supervised on magnitude only: the existing MR-STFT and log-mel terms plus
  `lmag`, a short-window (1.3–11 ms), onset-weighted log-magnitude MR-STFT (`LMAG_WEIGHT`);
- an adversarial spectrogram discriminator (3 resolutions, LSGAN + feature matching;
  `OFFLINE_ADVERSARIAL`) asks for HF that is *plausible* rather than phase-exact. The discriminators
  train from epoch 1, the generator's adversarial gradient ramps in over epochs 2–3
  (`OFFLINE_ADV_START_EPOCH`, `OFFLINE_ADV_RAMP_EPOCHS`); weights are relative to the calibrated
  reconstruction loss (`OFFLINE_ADV_WEIGHT`, `OFFLINE_FM_WEIGHT`).

Training then logs `recon`, `style`, `adv`, `fm`, `d_loss`, `d_real`, `d_fake`; `val_loss` is
reconstruction + style only (comparable across runs). Healthy: `d_real` near 1 and `d_fake` near 0
but not pinned (the generator is fooling it sometimes); if `d_loss` collapses to ~0 for many epochs,
lower `DISC_LR` or raise `OFFLINE_FM_WEIGHT`.

**Checkpoint selection differs for adversarial runs.** The adversarial term trades a little
reconstruction for plausible HF, so `val_loss` (reconstruction) is the wrong criterion for picking
the model: EarlyStopping still stops a run whose reconstruction stagnates, but does NOT roll back to
the best-`val_loss` epoch, and EVERY epoch is checkpointed (generator `*.keras` + discriminators
`*.discN.weights.h5`). The exported model is the last epoch. To use another epoch:

```bash
verus --analyze-attacks --offline-only --as-trained     # judge the exported (last) epoch
verus --promote-offline-checkpoint out/checkpoints/offline/best_model_epoch_NN_val_loss_X.keras
``` Set `OFFLINE_ADVERSARIAL = False` (and/or
`OFFLINE_PHASE_LOSS_CUTOFF_HZ = None`) to get the previous objective back. The realtime student is
unchanged for now (it cannot place a click it only learns about 6–7 ms later — see the roadmap).

### Measuring the attack

`val_loss` barely reflects the attack (on picked takes the attacks are ~17 % of the time but a third
of the mic's HF energy). `--analyze-attacks` renders three excerpts of the held-out tail of each
training group and prints, per band: attack-window energy vs the mic, energy *between* attacks vs
the mic (over-boosted pick branches leak there), phase coherence model↔mic at the attacks (piezo↔mic
as reference), and the onset-triggered 6–12 kHz time course of mic / piezo / model. Default knobs
are the runtime ones (what `--process` and the pedal play); `--as-trained` uses 0 dB knobs, i.e. the
model itself. Each run is saved to `out/analysis/` and diffed against the previous one.

```bash
verus --analyze-attacks [--offline-only|--realtime-only] [--finetuned] [--as-trained] [--sec 21]
```

Targets for picked playing: attack HF (6–20 kHz) within a few dB of the mic with `--as-trained`,
between-attack HF not above the mic, and the model's 6–12 kHz energy peaking in the first 2 ms after
the mic onset (the old model peaked at 4–8 ms, on the string release).

Writes: `out/models/offline/verus_baseline_offline.keras` (static) and
`…_offline_dynamic.keras` (any-length, used as the KD teacher and for `--process`).

### Step 2 — realtime student (KD)

```bash
verus --train-realtime      # requires the offline dynamic model
```

The offline teacher is embedded frozen (in inference mode) in the training graph; KD alphas
anneal per `KD_ALPHA_SCHEDULE` over the whole run (`KD_ANNEAL_END_EPOCH = EPOCHS`). The style
used for non-forced samples is a scheduled blend of student and teacher style probability that
reaches 0 (student only — exactly what inference has) for the last quarter of training
(`KD_STYLE_BLEND_SCHEDULE`; it is not a trainable weight any more). Per-epoch demos step the actual
streaming graph hop-wise.

Writes to `out/models/realtime/`:
- `verus_baseline_trainable_realtime.keras` — KD training graph (teacher baked in)
- `verus_baseline_realtime.keras` — **streaming static T=32** (this is what gets exported)
- `verus_baseline_realtime_dynamic.keras` — streaming, any length (Python inference)

### Step 3 — listen before exporting

```bash
verus --demo
verus --render-stems       # is the pick attack/resonance sitting right?
```

### Step 4 — optional fine-tune on a foreign recording

Put an aligned `source.wav`/`target.wav` in `input/finetune/` (any guitar/mic you don't
control). Only FiLM + coloreq unfreeze; the mic target is lightly denoised; the realtime
model keeps distilling at end-of-anneal strengths — from the fine-tuned offline model when it was
fine-tuned on this pair (same run, or newer than the pair's files), otherwise from the base
offline model (feature KD is seeded from the base trainable model, or disabled if that file is
missing).

**Declare the source's pickup** like `--process` does: a magnetic/electric DI must be fine-tuned
through the magnetic conditioning path (`--pickup magnetic`, alias `electric`); the default is
piezo. Run the fine-tuned model with the same `--pickup` afterwards. (Fine-tuning needs a mic target
aligned with the source, so it can't adapt the model to an electric guitar — see §4, "EQ-match
recordings".)

```bash
verus --fine-tune [--offline-only|--realtime-only] [--pickup piezo|magnetic|electric]
verus --fine-tune-demo [--pickup ...]
```

### Step 5 — export TFLite for the Pi

```bash
verus --export-tflite
verus --inspect-tflite-all   # must print "OK: bundle matches streaming state sizes"
verus --demo-tflite          # final listening check through the real artifact
```

Produces in `out/models/…`: `verus_baseline_realtime_t32_fp32.tflite` (the pedal model),
`verus_baseline_offline_t40960_fp32.tflite`, plus `finetuned` variants when present.

The realtime TFLite has **4 outputs** (documented in `README.md`): audio, style prob
(EMA-smoothed), packed **states bundle**, and **splits** (per-state lengths). There is exactly
one 3-D output so the runtimes can pick the audio tensor by shape. The export refuses to fall
back to `SELECT_TF_OPS` for realtime models — the pedal only links the builtin ops.

---

## 6. Inference on a computer

```bash
# Acoustic with piezo pickup (default):
verus --process --input in.wav --output out.wav --offline-only

# Electric guitar / magnetic pickup:
verus --process --input di.wav --output out.wav --offline-only --pickup electric

# Force a style instead of auto-detect:
verus --process … --style pick

# Apply the pedal's master EQ (same biquads as the C++):
verus --process … --eq lc_hz=80,lc_slope=24,hs_gain_db=2
```

Output sample rates: offline 96 k, realtime 48 k. **Level note:** `--process` applies the same
pick-branch make-up gains as the pedal and demos (`*_RUNTIME_BASELINE_DB` constants), so files
match the pedal at 0 dB knobs. The realtime path runs the dynamic streaming graph 1024 samples
per hop through `KerasStreamingRunner` (~30× faster than 32-sample stepping). With a forced style
it is identical to the pedal's 32-sample stepping (to ~4e-6); in **auto** style the classifier
logit is pooled per hop before the (hop-rescaled) EMA, so the style trajectory — and hence the
output — can differ slightly from the pedal. Use `--demo --static` for pedal-exact rendering.

Desktop app: `python -m verus.app` — input/output pickers, style, model, and pickup
dropdowns; runs the same backend in a worker process.

---

## 7. Raspberry Pi deployment

Full build instructions (PREEMPT_RT kernel, bazel-built TFLite with XNNPACK, flatbuffers,
PortAudio, CMake flags): **`runtime/README.md`**. HiFiBerry DAC8x/ADC8x config
(`/etc/asound.conf`, `config.txt` overlays): **`README.md`** at the repo root.

```bash
cd runtime/build && cmake --build .
sudo ./verus-runtime \
  --tflite ../../out/models/realtime/verus_baseline_realtime_t32_fp32.tflite \
  --blocksize 32 --input-device 0 --output-device 0
```

- `--blocksize 32` is **mandatory** for streaming models (hop must equal T; the binary
  refuses otherwise — a smaller hop corrupts the streaming states). The model is loaded,
  validated and warmed up before the stream starts (a bad model/blocksize exits cleanly; the
  warm-up prints the measured ms/hop vs the 0.67 ms budget).
- `--latency-hops N` (default 1): processed hops queued ahead of the audio callback. With 1 the
  inference thread must finish each hop within one 0.67 ms period; every miss is a dropped hop
  (a click, counted as `xruns`). Each extra hop adds 0.67 ms of latency and one period of
  scheduling-jitter headroom (a jitter buffer re-establishes it after underruns/bypass). The
  binary prints a hint when xruns accumulate.
- `--webui [port]` enables the browser UI on 127.0.0.1 (default 8088; now also has pickup and
  FX-loop switches).
- Channel map, FX loop, reverb/delay/chorus/gate are documented in the header comment of
  `runtime/src/main.cpp`.
- Keyboard: `b` bypass, `q/a` pick-attack ±1 dB, `w/s` pick-reso, `e/d` IR, `r/f` wet,
  `t` style mode, `g` guided auto, `l` FX loop, **`u` pickup piezo↔magnetic**, `p` print.
- The master EQ lives in the pedal (`MasterEqIIR`, driven by the existing `gKnobs` EQ fields /
  web UI) and no longer in the model; old TFLite exports that still have EQ inputs get neutral
  values.
- Runtime hygiene: the inference thread (SCHED_FIFO 35) sleeps on a semaphore instead of spinning
  a core; the PortAudio callback thread sets itself to SCHED_FIFO 40 (it used to inherit main's 20,
  i.e. run BELOW the inference thread); main/keyboard/UI threads are not RT any more. DI and
  processed audio are pushed in lockstep and only ever consumed together by the same count (the
  old per-ring flush could leave DI a hop ahead after a bypass toggle); backlog beyond
  `latency-hops` is trimmed; the processed signal is sanitised (NaN/Inf → 0, ±12 dBFS clamp)
  before any stateful FX; all FX buffers are allocated before the stream starts; the model's
  state-bundle copy plan is precomputed (no per-hop allocation); telemetry (`xruns`, `paXruns`,
  `inDrops`, `resyncs`, proc time) is printed by the main thread, never from RT threads.
- After a `git pull` on the Pi, **recompile**. Before the source was split into headers
  (2026-09-22), the runtime was compiled warning-free (`-Wall -Wextra`) with GCC 12 on Debian
  Bookworm arm64 — the Raspberry Pi OS toolchain — against the real TFLite 2.20 / flatbuffers
  24.3.25 / PortAudio headers, and its TFLite wrapper was run against exported models: output
  bit-identical to the Python `TFLiteRunner`. (An earlier version did not compile with GCC 12:
  `std::sinf`/`std::expf` only exist in libstdc++ ≥ 14.) The split was then checked against stub
  TFLite / PortAudio headers only (it compiles to the same machine code as the single file);
  **repeat the Docker compile check with the real headers** before relying on it. Not yet run on
  real audio hardware.

### Constants that must stay in sync (Python ↔ C++)

| `src/verus/` | `runtime/src/` | Value |
|---|---|---|
| `REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB` (`config.py`) | `kPickAttackBaselineBoostDB` (`state.hpp`) | 22 dB |
| `REALTIME_PICK_RESO_RUNTIME_BASELINE_DB` (`config.py`) | `kPickResoBaselineBoostDB` (`state.hpp`) | 32 dB |
| `master_eq_sos()` formulas / `master_eq_defaults()` (`eq.py`) | `MasterEqIIR` (`fx/master_eq.hpp`) / `Knobs` EQ defaults (`knobs.hpp`) | LC 20 Hz = bypass, 6 dB/oct, all gains 0 dB |

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
- **Streaming coloreq taps** are initialised explicitly (`taps_init`) so the streaming FIR never
  depends on `copy_weights_by_name` finding a same-shaped training weight.
- **Upsamplers** (`FIRInterpByN` / `StreamingFIRInterpByN`, IR and pick-resonance branches) are
  polyphase (identical result, 8× fewer MACs; the zero-stuffed version was ~50 % of the pedal
  model's per-hop time) and min-phase (the linear-phase 257-tap design delayed the whole IR branch
  by 2.7 ms vs the main path). Their streaming state is `ceil(L/F)-1` low-rate samples (32 and 8),
  not `L-1`; older streaming graphs with the long states still load and produce the same output.
- **KD alignment**: student sample n ↔ teacher sample 2n. `_kd_align` low-passes the teacher with
  the `resample_poly` 2:1 filter centred on 2n (not the old 2-tap avg-pool, which was half a
  96k-sample late and aliased 24–48 kHz teacher content into the KD target at ~-5 dB).
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
- **Offline objective**: `loss_weights.json` records `_phase_cutoff_hz` (a mismatch on resume forces a
  recalibration, with a warning) and `_recon_ref` (the scale the adversarial weights are relative to).
  `AdversarialTrainer` is a training wrapper only — never save it; `GeneratorCheckpoint` saves the
  generator (a normal `.keras`) and the discriminators' weights next to it.
- On Apple Silicon one offline GPU (tensorflow-metal) training run produced a NaN loss once and
  could not be reproduced; CPU is the safe choice for long runs, `TerminateOnNaN` stops a bad one.
- `BlendScalar` trainability is a layer-level flag (Keras 3 cannot re-enable a variable created
  with `trainable=False`).

---

## 9. Roadmap

Planned work (the magnetic / electric roadmap, the pick-attack steps, the retrain, desktop app and
plugins) is in [`ROADMAP.md`](ROADMAP.md).
