# Roadmap

Planned and open work, roughly in order. Nothing here is a promise; it is where the project is
heading. How the current system works: [TRAINING.md](TRAINING.md).

## Data

- **Record the four-track session**: piezo + magnetic soundhole pickup + soundhole mic + 12th-fret
  mic, simultaneously, 96 kHz / 24-bit, ≥ 1 h per style. Step by step: [RECORDING.md](RECORDING.md).
- **Record the EQ-match pair**: the same material on an electric (humbucker DI) and on the acoustic's
  magnetic pickup, ≥ 5 min each, same style, into `input/eqmatch/`.
- **Check the pedal's input impedance.** A passive humbucker needs ~1 MΩ to keep its resonance; into a
  20 kΩ line input it rolls off from ~700 Hz. The HiFiBerry ADC8x input impedance is unknown (line
  inputs are commonly 10–50 kΩ). The piezo wants a high impedance too, so one Hi-Z buffer in front of
  the ADC would serve both.
- **Session prep tool**: raw booth exports → `input/`. Aligns on the three bridge taps (median of the
  onset offsets; not by cross-correlation of the playing, which is ambiguous on this guitar), checks
  polarity, applies one gain constant per track, writes the stereo target, cuts out logged noise and
  checks every take (clipping, hum, DC, rumble, noise floor, levels). The rules it implements are in
  [RECORDING.md](RECORDING.md) "Preparing the files". The same alignment is needed for end-user
  fine-tuning.

## Training

- **EQ match, then fake-electric sources.** Run `tools/eq_match.py` on the pair above and check the
  split-half figure (≲ 3 dB) by ear. Then apply the matched FIR on the fly to a fraction of the
  magnetic training segments (same mic target, pickup flag = magnetic), in both segment generators,
  at both sample rates, with a fixed share in the validation set. Optionally randomize around the
  matched curve so the model covers other humbuckers and P90s. No extra audio on disk.
- **Cross-pickup consistency loss**: `model(piezo_src) ≈ model(magnetic_src)` (and ≈
  `model(fake_electric_src)`) for the same take. Needs triplet pairing of the four-track takes (see
  the `TODO(data)` marker in `load_training_data()`).
- **Stereo output**: the target is the two-mic pair as recorded, the model outputs both channels, and
  the player chooses stereo or mono.
  - Loader: stereo targets (channel 1 = soundhole, channel 2 = 12th fret); `_LazyMono` reads channel 1
    only today.
  - Losses for the image, not just each channel: per-channel losses only constrain inter-channel phase
    below `OFFLINE_PHASE_LOSS_CUTOFF_HZ` (3 kHz), so also supervise mid (L+R) and side (L−R)
    magnitudes and give the discriminators both channels.
  - Mono: summing spaced mics comb-filters (~20–30 cm → first notch around 0.5–1 kHz). Offer a blend
    control and/or a delay-aligned sum.
  - Model/runtime: a two-channel output head, C++ and `--process` writing stereo, parity checks per
    channel, and a measurement of the realtime cost on the Pi.
- **First full training run and export** on the new corpus (offline teacher first, ~14 h at 12 × 2000
  steps on a 16 GB Mac, which is far fewer steps than from-scratch needs; run under `tools/train_guard.py`), then
  `--analyze-attacks --as-trained`, `--train-realtime`, `--export-tflite`, `--inspect-tflite-all`.
  Memory of the realtime KD step has not been measured; probe it with a short guarded run first.
- **Pick-click generator branch (offline)**: an onset-triggered excitation placed ~6–7 ms *before* the
  detected release (the offline model has look-ahead), trained with magnitude/adversarial terms only;
  then drop the boosted pick-attack branch and its 38 dB baseline.
- **Realtime click timing**: emit the click at the release (causal, ~6–7 ms late) or accept a few ms
  of look-ahead. Either way the student's KD/HF losses must tolerate the offset, and waveform KD above
  3 kHz should go.
- **Magnetic pickup as a pick-contact cue**: the soundhole pickup sits near where the pick hits and may
  see the string pushed before the release. Test on the four-track recordings.

## Pedal

- **First run on the Pi + HiFiBerry.** The pedal builds with GCC 12 (verified in Docker) but has not
  been run on the real hardware. Watch `Proc=` against the 0.67 ms budget (32 samples at 48 kHz) and
  `xruns/paXruns/resyncs`; try `--latency-hops 2` if xruns appear. A Pi 4 may be throughput-bound.
- **Re-tune by ear** after the retrain: the pick detector (`REALTIME_PICK_MASK_THRESHOLD_DB` = 26,
  knee 4, via `--sweep-pickattack`), the realtime pick-branch baselines (22 / 32 dB, synced with C++)
  and the offline pick-attack baseline.
- **Electric sustain**: electric strings ring much longer than an acoustic's and EQ can't change that.
  A DSP expander after the model with a "decay" knob, in C++ and in `--process` (identical formulas,
  like the master EQ).

## Desktop app, DAW plugins, end-user fine-tuning

One C++ audio core (the pedal's streaming runtime, EQ, FX) shared by the pedal, plugins and app; one
fine-tuning engine shared by all three.

- **DAW plugin (VST3 / AU) with the realtime model.** JUCE, so the C++ in `runtime/src/`
  carries over (JUCE is available under AGPLv3, which fits this project's license; check the terms before shipping). Needs a FIFO that runs the model in 32-sample hops for any
  host block size, resampling for host rates other than 48 kHz, reported latency, no allocation or
  locks on the audio thread, and TFLite builds for macOS / Windows. On an M-series Mac the model runs
  at ~0.23 ms per 32-sample hop.
- **The offline model in the DAW via ARA 2.** The offline model is non-causal (~64 ms look-ahead),
  96 kHz and heavy, so it doesn't fit a realtime plugin; ARA 2 lets a plugin read a clip's whole audio,
  render it in the background and play it back in sync. Today: bounce the track and run
  `--process --offline-only`.
- **End-user fine-tuning engine.** Adapt the model to a user's own guitar and mic (today:
  `--fine-tune`, [TRAINING.md](TRAINING.md) §5 Step 4, with a manually aligned pair). Needs a headless
  engine in its own process, automatic alignment and input checks, bounded time and memory on ordinary
  machines, and a model bundle with metadata (offline + realtime models, base version, pickup,
  runtime baselines) so the pedal can tell a fine-tuned model from the base model.
- **Standalone desktop app + pedal integration**: realtime play-through, offline file processing (what
  `src/verus/app/` does today), fine-tuning, and finding the pedal / pushing and pulling models
  (the pedal only has a local web UI today, `--webui`).

## Ideas

- **Piezo saddles on an electric** (e.g. Graph Tech Ghost) sense string force at the bridge, much like
  an acoustic's under-saddle piezo, so the existing piezo path would nearly apply.
- **Unpaired adversarial stage on real electric DI**, only if fake-electric sources still sound clearly
  non-acoustic. Risky: without a paired target a discriminator can invent sound that wasn't played.
- **A separate "electric" pickup condition**, only if the model itself should shorten sustain; the
  pickup input is a 0/1 scalar today.
