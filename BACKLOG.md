# Backlog

Everything not done yet, in priority order. Ids are stable: keep them when reordering, never reuse
one. **[user]** = needs the user (recordings, hardware, listening, a decision). When an item is done,
delete it and say so in that round's file. How this fits with the rounds: `AGENTS.md`.

## Now — blocked on the user

### B-1 [user] Record the four-track session (the "magnetic triplets", now with two mics)
Piezo + magnetic soundhole pickup + soundhole mic + 12th-fret mic, **simultaneously**, on the acoustic,
96 kHz / 24-bit, ≥ 1 h per style (pick / finger). **The session step by step, what to play: `RECORDING.md`**
(eight 15-min takes alternating styles + an EQ-match take, ~3 h in the booth); requirements:
`TRAINING.md` §4 "The four-track session". Same soundhole-pickup settings as the EQ-match pair (B-2).
**Free disk first:** ~16 GB with the DAW session and the exports, 12 GB free on 2026-09-23 (round 08).
Unblocks B-6, B-7, B-8, B-13, B-22.

### B-2 [user] Record the EQ-match pair
The same material on the electric (humbucker DI) and on the acoustic's magnetic pickup, ≥ 5 min each,
same style, into `input/eqmatch/`. The acoustic half is take 9 of the booth session; the electric
half is played along to it at home (`RECORDING.md`). Requirements: `TRAINING.md` §4 "EQ-match recordings". Why ≥ 5 min
and the same material: round 07's synthetic test (the curve is limited by what was played). Unblocks B-5.

### B-3 [user] Check the pedal's input impedance
A passive humbucker needs ~1 MΩ to keep its resonance; into a 20 kΩ line input it rolls off from
~700 Hz. The HiFiBerry ADC8x input impedance is unknown (line inputs are commonly 10–50 kΩ). The piezo
needs high impedance too, so one Hi-Z buffer in front of the ADC would serve both. Record B-2's electric
DI through the same kind of input the pedal will have. (Round 07.)

## Next — once the recordings exist

### B-23 Session prep: raw booth exports → `input/`
`RECORDING.md` has the user export each take raw: four mono WAVs (`takeNN_<style>_piezo`,
`_magnetic`, `_mic_soundhole`, `_mic_12th`), unaligned and untrimmed, plus room tone and a log. A
standalone tool (like `eq_match.py`) implementing `RECORDING.md` "Preparing the files":
- **aligns on the three bridge taps** (median of the three onset offsets, piezo vs the earlier mic)
  and shifts **both** mics by it; checks with midband (80 Hz–1 kHz) onsets after silence, median
  ≈ 0 ms like the old corpus. Not by cross-correlation of the playing: ambiguous by 2.9 ms on this
  guitar (body resonance ~175 Hz) and pulled by the 6.5 ms high-band pick-contact lead (round 08);
- checks polarity (low-band correlation at zero offset positive, as in the old corpus; else invert
  the mic pair together); reports the inter-mic delay and the offset's consistency across takes;
- one gain constant per track for the whole session (pickups to the old piezo level, the mic pair
  with one shared gain to the old mic-vs-piezo relation); no per-take normalization;
- writes `input/piezo|magnetic/<style>/{source,target}/takeNN_*.wav` with a stereo target
  (channel 1 = soundhole, channel 2 = 12th fret) and the magnetic target as a hard link;
- cuts the noise times from the log (1 s before to 2 s after; splits the take there, 5 ms fades,
  drops pieces < 30 s) and trims the leading taps. The log is the booth sheet's database
  (https://claude.ai/artifact/6FcdUY47gjuzV5oNALoJjj; `ArtifactData`): `takes/t01`…`t11` with `startedAt`/`endedAt` (epoch ms), `start`,
  `length`, `pick`, `capo`, `noise` ("mm:ss, mm:ss", time into the take, tapped by hand — so ± 1–2 s:
  look for the transient near each mark), `notes`; `booth/setup` `fields` (mic models, distances,
  gains, guitar controls); `booth/checks` `items`;
- checks every take: clipping, pickup hum, DC, rumble, noise floor against the room tone, levels,
  style and length, and says what's wrong;
- writes take 9's magnetic track as `input/eqmatch/acoustic_magnetic.wav` (silence kept, taps removed).
Streams in blocks (the disk and memory rules in `AGENTS.md`). The same alignment is needed for
end-user fine-tuning (B-20). First thing to do when the recordings arrive.

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

### B-22 Stereo output: the user's two-mic setup as the target
Proposed and decided by the user (round 08): the target is recorded as they record acoustic guitar
anyway — one mic in front of the soundhole, one at the shoulder pointing down at the 12th fret — and
the model outputs both channels; the player chooses stereo or mono. Why it fits: the pedal already has a stereo path
(the mono model output is copied to L/R before the stereo FX; DAC output 8 is the mono sum), and the
two mics differ mostly in ways the pickup signal can predict (body and low end at the soundhole,
string and fretboard at the 12th fret, different arrival times). What it needs:
- **Recording:** `TRAINING.md` §4 "The four-track session" (both mics as separate tracks, the pair
  aligned to the pickups as a group — their relative delay is the stereo image).
- **Loader:** stereo targets (channel 1 = soundhole, channel 2 = 12th fret); `_LazyMono` reads
  channel 1 only today.
- **Losses for the image, not just each channel:** per-channel losses only constrain the
  inter-channel phase below `OFFLINE_PHASE_LOSS_CUTOFF_HZ` (3 kHz); above it both channels are trained
  on magnitude alone, so their relationship would be arbitrary (too narrow or phasey). Also supervise
  mid (L+R) and side (L−R) magnitudes, and give the discriminators both channels.
- **Mono:** summing spaced mics comb-filters (~20–30 cm spacing → ~0.5–1 ms → first notch around
  0.5–1 kHz) — the same as summing the real mics. Offer a blend control (ends = a single mic, no
  comb) and/or an aligned sum (delay the earlier mic by the measured inter-mic delay before summing).
- **Model/runtime:** a two-channel output head; the realtime cost on the Pi rises by however much of
  the output stage is per channel — measure (B-9). C++ and `--process` write stereo; parity checks
  per channel.
- **Old data:** the 2 h corpus's mic (a small condenser some distance off the soundhole, angled)
  matches neither position, so it only warm-starts: initialize both output channels from the old
  single output head, then train on the four-track session.
Implement before B-8 (the retrain), together with B-6 and B-7.

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
