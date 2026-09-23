# Backlog

Everything not done yet, in priority order. Ids are stable: keep them when reordering, never reuse
one. **[user]** = needs the user (recordings, hardware, listening, a decision). When an item is done,
delete it and say so in that round's file. How this fits with the rounds: `AGENTS.md`.

## Now — blocked on the user

### B-1 [user] Record the magnetic triplets
Piezo + magnetic soundhole pickup + mic, **simultaneously**, on the same acoustic, 96 kHz / 24-bit,
aligned, ≥ 1 h per style (pick / finger). Layout and requirements: `TRAINING.md` §4. Use the same
soundhole-pickup settings as the EQ-match recording (B-2). Unblocks B-6, B-7, B-8, B-13.

### B-2 [user] Record the EQ-match pair
The same material on the electric (humbucker DI) and on the acoustic's magnetic pickup, ≥ 5 min each,
same style, into `input/eqmatch/`. Instructions: `TRAINING.md` §4 "EQ-match recordings". Why ≥ 5 min
and the same material: round 07's synthetic test (the curve is limited by what was played). Unblocks B-5.

### B-3 [user] Check the pedal's input impedance
A passive humbucker needs ~1 MΩ to keep its resonance; into a 20 kΩ line input it rolls off from
~700 Hz. The HiFiBerry ADC8x input impedance is unknown (line inputs are commonly 10–50 kΩ). The piezo
needs high impedance too, so one Hi-Z buffer in front of the ADC would serve both. Record B-2's electric
DI through the same kind of input the pedal will have. (Round 07.)

## Next — once the recordings exist

### B-5 Run the EQ match and A/B it by ear
`python eq_match.py input/eqmatch/acoustic_magnetic.wav input/eqmatch/electric_humbucker.wav` →
`out/eqmatch/`. Check the split-half figure (≲ 3 dB), look at `eq_match.png`, and have the user compare
`acoustic_as_electric.wav` with the electric DI. Done when the user accepts the match (or it's clear
what's missing).

### B-6 Fake-electric sources in training
Apply the matched FIR on the fly to a fraction (e.g. half) of the magnetic training segments, same mic
target, pickup flag = magnetic ("electric" is an alias). Design:
- In `segment_generator` / `segment_generator_dual`: the 96 kHz FIR on offline-rate slices, the 48 kHz
  FIR on realtime-rate slices (both designed from the same curve by `eq_match.py`); read the FIR length
  of extra samples before each slice and convolve "valid", so segments have no filter start-up.
- The fixed validation set gets a fixed share of fake-electric segments (seeded).
- Optionally randomize around the matched curve (small tilt, a pot/cable resonance shift) so the model
  doesn't learn one exact curve and covers other humbuckers and P90s.
- No extra audio on disk (the disk is ~97 % full).
Done when training includes them and `--process --pickup electric` on the user's electric DI has been
listened to.

### B-7 Cross-pickup consistency loss
`model(piezo_src) ≈ model(magnetic_src)` (and ≈ `model(fake_electric_src)`) for the same take; the
`TODO(data)` marker in `load_training_data()`. Needs triplet pairing (B-1).

### B-8 Full retrain → export
Offline warm start under the guard with the Mac otherwise idle (`TRAINING.md` §5 Step 1; ~13–14 h),
then `--analyze-attacks --as-trained` (compare with `out/analysis/`), listen, `--train-realtime`,
`--export-tflite`, `--inspect-tflite-all` ("OK: bundle matches streaming state sizes"). The realtime KD
step's memory has **not** been measured — probe it with a short guarded run first
(`REALTIME_BATCH_SIZE` = 8 won't fit). Checkpoints are per epoch; pick adversarial epochs with
`--promote-offline-checkpoint`.

## After the retrain

### B-9 [user] First run on the Pi hardware
The pedal builds with GCC 12 (verified in Docker) but has never run on the real Pi + HiFiBerry. Watch
`Proc=` against the 0.67 ms budget (32 samples at 48 kHz) and `xruns/paXruns/resyncs`; try
`--latency-hops 2` if xruns appear. A Pi 4 may be throughput-bound even after round 05's ~2x speedup
(0.23 ms/hop on the M-series Mac). Build/run: `cpp/readme.md`, `TRAINING.md` §7.

### B-10 [user] Re-tune by ear
The pick detector (`REALTIME_PICK_MASK_THRESHOLD_DB` = 26, knee 4, via `--sweep-pickattack`), the
realtime pick-branch baselines (22 / 32 dB, synced with C++), and the offline pick-attack runtime
baseline (the round-06 pilot was ~5 dB over-bright between attacks with the +38 dB boost). Min-phase
upsamplers and the corrected KD may also shift levels slightly.

## Later

### B-11 Pick-click generator branch (offline)
An onset-triggered excitation placed ~6–7 ms **before** the detected release (the offline model has
the look-ahead), trained only with the magnitude/adversarial terms; then drop the boosted pick-attack
branch and its 38 dB baseline. Why: round 06.

### B-12 Realtime click timing
Decide between emitting the click at the release (causal, ~6–7 ms late) and a few ms of look-ahead
(latency). Either way the student's KD/HF losses must tolerate the offset (e.g. envelope losses with a
±7 ms tolerance) or it learns to suppress the click; waveform KD above 3 kHz should go (same argument as
round 06's offline split).

### B-13 Magnetic pickup as a pick-contact cue
The soundhole pickup sits near where the pick hits and may see the string pushed before the release —
test on the triplets (B-1). Could help the causal model with B-12.

### B-14 Electric sustain: a decay control on the pedal
Electric strings ring much longer than an acoustic's, and EQ can't change that. A DSP expander after the
model with a "decay" knob, in C++ and in `--process` (identical formulas, like the master EQ). Simpler
and adjustable per guitar, versus teaching the model a guessed sustain transform. (Round 07.)

### B-15 Unpaired adversarial stage on real electric DI
Only if B-6 still sounds clearly non-acoustic on an electric. Real electric DI in; the existing
spectrogram discriminators judge the output against the mic recordings; a content loss ties the output
to what was played (e.g. stay close to the B-6 model's output). The only way to train on real electric
input — risky: without a paired target a discriminator can invent sound that wasn't played.

## Future — desktop app, DAW plugins, end-user fine-tuning

The plan (user, round 07): besides the pedal, **standalone desktop apps and DAW plugins**, both able to
**create fine-tuned models**; the pedal gets a desktop integration where its user can fine-tune too.
One C++ audio core (the pedal's streaming runtime, EQ, FX) shared by the pedal, the plugins and the
app; one fine-tuning engine (B-20) shared by the app, the plugins and the pedal integration.
**Name:** the user decided on a name for the pedal in an earlier conversation that isn't recorded
anywhere (the code says "Piezo2Mic") — ask, then record it in `AGENTS.md`.

### B-18 DAW plugin (VST3 / AU) with the realtime model
The pedal as a plugin, for tracking and monitoring through the realtime model. Fine-tuning from the
plugin: B-20.
- JUCE (VST3, AU, standalone; AAX later) so the C++ from `cpp/src/main.cpp` carries over: the TFLite
  streaming wrapper and state bundle, `MasterEqIIR`, the baseline constants. The synced-constants
  table (`TRAINING.md` §7) then spans three runtimes. Check JUCE's licence terms before shipping.
- What differs from the pedal: the host's block size varies → a FIFO that runs the model in its
  32-sample hops; host rates other than 48 kHz → resample in and out; report the added latency to
  the host (delay compensation); no allocation or locks on the audio thread; reset the streaming
  states on sample-rate change / playback start; many instances at once.
- TFLite built for macOS / Windows (2.20 with XNNPACK, as on the pedal). The M-series Mac ran
  0.23 ms per 32-sample hop, i.e. ~3.7 ms per 512-sample block against a 10.7 ms budget — fine on
  desktops, but measure.
- Parameters: the pedal's knobs (pick attack, pick reso, IR, wet, style mode, pickup, master EQ),
  automatable.
Done when it runs in a DAW without dropouts at 64–512-sample buffers and its output matches the Python
TFLite runner with a forced style (as the C++ pedal does, round 05).

### B-19 The offline model in the DAW, Melodyne-style (ARA 2)
The offline model is non-causal (~64 ms look-ahead), 96 kHz and heavy, so it doesn't fit a normal
realtime plugin. Melodyne's deep DAW integration is **ARA 2** (Audio Random Access: Celemony's SDK,
supported by JUCE): the plugin reads a clip's whole audio from the host, renders it in the background
and plays the result back in sync with the timeline; edits follow the clip. That fits the offline
model: full context in both directions, no latency limit, re-render when a knob changes (the knobs
are model inputs). Most major DAWs support ARA 2 (e.g. Cubase, Studio One, Logic, Reaper); Ableton
Live didn't, last I knew — check the current list.
Fallbacks: (a) a "high quality" mode in B-18's plugin that runs the offline model with a large
reported latency — only if it runs faster than realtime on a desktop CPU (not measured yet); usable
for mixing, not for tracking. (b) What exists today: bounce the track, run it through the desktop app
(`--process --offline-only`), import the result.
Builds on B-18. A future improvement, not near-term.

### B-20 End-user fine-tuning engine
Let a user adapt the model to *their* guitar and mic: record a pickup + mic pair, fine-tune, and get
a model for the app, the plugins and the pedal. Today this is `--fine-tune` (`TRAINING.md` §5 Step 4:
FiLM + colour EQ unfreeze, offline then realtime KD, then TFLite export), run by hand with a manually
aligned `input/finetune/` pair. For end users it needs:
- **A headless engine in its own process**, driven by the app and the plugins (never training inside
  a DAW's audio process). The straightforward route is packaging the existing Python/TensorFlow
  fine-tune path (TF is Apache-2.0); the alternative, porting training to a C++-embeddable runtime,
  is a much bigger job. Progress, cancel and resume (the LR is already persisted across resume).
- **Automatic alignment and checks** of the recorded pair: the pipeline has no auto-alignment today
  (`TRAINING.md` §4). Estimate the pickup→mic delay and polarity (cross-correlation), and reject
  clipping, too little material or too much noise, with a message saying what to fix.
- **Bounded compute and memory on ordinary machines:** measure a fine-tune's time and peak memory
  first (the dev Mac lessons: AGENTS.md); fine-tune runs the offline teacher for the realtime KD.
- **A model bundle with metadata**: offline + realtime models plus the base model version, pickup,
  and runtime baselines. This also fixes round 05's "the pedal can't tell a fine-tuned model from a
  TFLite file" (8 vs 32 dB pick-reso baseline).
- **Collecting the pair inside a DAW:** the plugin on the pickup track with the mic track as a sidechain
  input, or, with ARA (B-19), reading both clips directly.

### B-21 Standalone desktop app + pedal integration
The product's standalone app. It covers:
- realtime play-through via the audio interface (a JUCE standalone build of B-18 gives this);
- offline file processing (what the PySide6 `piezo2mic_app/` does today);
- fine-tuning (B-20);
- the pedal's desktop integration: find the pedal, push models to it and pull them back, edit
  settings. The pedal only has a local web UI today (`--webui`, 127.0.0.1).

Decision to make: a JUCE app that shares its code with the plugin, with the fine-tuning engine as a
bundled helper (recommended), or keep growing the PySide6 app.

## Ideas / parked

### B-16 Piezo saddles on the electric
Bridge piezo saddles (e.g. Graph Tech Ghost) sense string force at the bridge, much like the acoustic's
under-saddle piezo — the existing piezo path would nearly apply. The highest-quality electric→acoustic
route; hardware, the user's call.

### B-17 A separate "electric" pickup condition
Only worth it if the model itself should shorten sustain (B-14 done in the model instead of DSP): a
short-window realtime model can't tell from the input that a note will ring too long. The pickup input
is a 0/1 scalar today; a third value would need a second scalar or one-hot.
