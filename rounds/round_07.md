# Round 07 — Memory-safe training; electric-guitar strategy; EQ match; rounds

- **Date:** 2026-09-23
- **Agent:** Claude Opus 5.5 (Claude Code)
- **Goal:** run the full offline retrain that crashed the Mac at the end of round 06, without
  endangering the machine. Redirected by the user along the way: the retrain was postponed, then this
  round planned electric-guitar support, built the EQ-match tool, and set up the rounds structure.
- **Commits:** rounds 05–07 in one commit, "Rounds 05-07: review fixes, pick-attack objective, memory-safe training, EQ match, rounds docs", on `review-fixes-and-magnetic-pickup`,
  pushed and merged into `master` (fast-forward) at the end of this round. It changed `.gitignore`,
  `TRAINING.md`, `cpp/CMakeLists.txt`, `cpp/readme.md`, `cpp/src/main.cpp`, `piezo2mic.py`,
  `requirements.txt`, `readme.md` and added `train_guard.py`, `eq_match.py`, `AGENTS.md`, `CLAUDE.md`,
  `BACKLOG.md`, `rounds/`, `out/models/offline/legacy_2025-10/` (model backup; identical blobs to the
  tracked Oct-2025 files, so no repo growth).

## What happened
1. **Root cause of the crash:** the whole corpus (4 × 1 h at 96 kHz plus resampled 48 kHz copies,
   ~5.5 GB) was held in RAM, and the offline adversarial step at batch 4 needs > 8 GB on top. With the
   user's other apps open only ~6–10 GB of the 16 GB is available; swap filled the ~97 % full SSD.
2. **Corpus read from disk per segment:** `_LazyMono` (in `piezo2mic.py`) reads each training slice with
   soundfile and resamples it over a polyphase-aligned window, so slices equal the old whole-file
   arrays with no per-segment edge transients; `load_training_data()` reads headers only; the
   in-RAM caches are gone.
3. **Memory probes** (offline adversarial step, see the table below) → CPU, batch 1.
   `--cpu` hides the GPU; it has to be applied right after `import tensorflow` (in `main()` it raised
   "Visible devices cannot be modified after being initialized").
4. **Budget:** one pass over the training chunks is 31 158 steps at batch 1 — weeks on this CPU from
   scratch. So: warm start from the Oct-2025 offline model (`--train-offline --init-from <keras>`,
   copied by layer name, skips the warm-up) and `OFFLINE_EPOCHS` = 12 × `OFFLINE_MAX_STEPS_PER_EPOCH` =
   2000 (~13–14 h). Validation capped by `VAL_MAX_SEGMENTS` (offline 256, realtime 512). The Oct-2025
   offline model is backed up in `out/models/offline/legacy_2025-10/`.
5. **`train_guard.py`:** runs a command, samples system available memory, swap and free disk every
   0.5 s, keeps the Mac awake (`caffeinate`), and kills the job's process tree past a limit.
6. **Attempt 1** was stopped after epoch 1: the loss-weight calibration averaged a fixed 5 *batches*,
   i.e. 5 segments at batch 1, and its weights were far off (STFT 0.296 vs 0.0227, HF 183 vs 57.1 on 256
   segments). Fix: `CALIB_SEGMENTS` = 256.
7. **Attempt 2** ran cleanly (~4.6 GB RSS, 1.9 s/step) until the guard stopped it when the user opened
   other apps and swap grew 1.51 → 3.25 GB — the guard doing its job.
8. The user **postponed the retrain** until the magnetic triplets are recorded ("no point before
   that"). Stale checkpoints / `loss_weights.json` / `train_state.json` were deleted.
9. **Electric strategy.** The user asked about EQ'ing the acoustic's magnetic soundhole pickup toward
   a passive humbucker, then clarified the real goal: **the pedal should make an electric guitar sound
   acoustic**, with one model also serving the acoustic. Their soundhole-pickup recordings still sound
   clearly acoustic, not dull like an electric DI. Discussed below; the user chose to record the same
   material on the electric (humbucker DI) and on the acoustic (its magnetic humbucking soundhole
   pickup), EQ-match the acoustic to the electric, and use that as a "fake electric" source ("it'll
   have to do").
10. **`eq_match.py`** written and tested on synthetic data (below).
11. **Rounds structure** set up at the user's request: `AGENTS.md`, `CLAUDE.md`, `BACKLOG.md`,
    `rounds/INDEX.md`, rounds 01–06 written retroactively; `TRAINING.md` §9/§9b/§10 (roadmap and state)
    moved into `BACKLOG.md` and the round files.
12. The user asked for a DAW plugin in the backlog: the realtime model as a VST3/AU plugin (B-18) and
    the offline model integrated Melodyne-style via ARA 2 (B-19). They also asked for the pedal's
    name, decided in an earlier conversation: it isn't recorded in the repo, the agent's notes or the
    local session history (only "Piezo2Mic" in the code) — asked the user. Then: the plugin must be
    able to create fine-tuned models, and the pedal will get a desktop integration with fine-tuning,
    so standalone apps and plugins are both required → B-20 (end-user fine-tuning engine) and B-21
    (standalone app + pedal integration), grouped with B-18/B-19 under "Future" in `BACKLOG.md`.
13. At the user's request: committed everything (rounds 05–07 in one commit), pushed the branch, merged
    it into `master` (a fast-forward: `master` had not moved since `c4d1510e`) and pushed `master`.

## Why
- **CPU, batch 1:** the only measured configuration that doesn't swap with the user's apps open; Metal
  was no faster.
- **Warm start** instead of from scratch: the architecture changes since Oct 2025 are additive
  (pickup adapter zero-initialized, cond embed seeded from the style embed), so the old weights are a
  valid starting point and save weeks.
- **Postponing** (user's call): the retrain after the triplets would be needed anyway.
- **EQ match, not a randomized RLC simulation:** I suggested simulating passive pickups with randomized
  coil/cable/load values (covers humbucker, P90 and single coil without owning them); the user
  preferred matching their own electric by measurement. Randomizing around the matched curve can
  still be added (B-6).
- **"electric" stays an alias of "magnetic":** the EQ difference is audible in the input, so a separate
  condition adds nothing — unless the model is also taught to shorten sustain, which a short-window
  realtime model can't infer from the input (B-17).

## What we learned
**Dev Mac memory** (offline adversarial step):

| configuration | peak RSS | speed | swap |
|---|---|---|---|
| CPU batch 1 | 5.3–7.7 GB (≈4.6 GB steady) | 1.9–2.0 s/step | none |
| CPU batch 2 | 7.4 GB | 3.7 s/step | +0.6 GB |
| CPU batch 4 | > 8 GB | — | swaps (killed) |
| Metal GPU batch 1 | not visible in RSS | 1.6–2.0 s/step | +1.4 GB |

Apple Silicon memory is unified: Metal allocations don't show in the process RSS, so judge by system
available memory and `vm.swapusage`. The loss-weight calibration needs ~256 segments (5 were ~10x off
on STFT, ~3x on HF; 64-segment slices spread 2x).

**Why the acoustic's magnetic pickup still sounds acoustic** — mostly the string vibration, not the
transducer:
- the strings are anchored to a vibrating top that drains energy at its resonances (faster, body-shaped
  decay), and the pickup is mounted on that top, so it also senses body vibration (some soundhole
  pickups add a body sensor or mic on purpose);
- bronze-wound strings: only the steel core is magnetic;
- soundhole pickups are usually active or voiced bright — no passive resonance.

**A passive pickup** is a resonant low-pass: coil L and R, cable + coil C, pot/input load. A humbucker
(~4.5 H) into ~1 MΩ with a few metres of cable peaks around 2–4 kHz and falls ~12 dB/oct above.
Into a 20 kΩ line input it rolls off from ~700 Hz (R/2πL) — the pedal needs a high-impedance input
(B-3). Single coils (~2.5 H) peak higher and sharper; P90s sit in between.

**Pickup position comb:** fixed *per string* (it depends on the pickup-to-bridge distance and the
string's wave speed), not per fretted note — this corrects a claim made earlier in this round. The
soundhole sits about 1/4 of the scale from the bridge, roughly a neck-pickup position; a bridge
pickup's comb can't be synthesized from one mono signal (six strings, six different combs).

**EQ can't fix:** sustain / decay (electric strings ring longer), string character, the body
vibration in the soundhole signal, bridge-pickup combs. **No aligned mic target can exist for an
electric guitar**, so `--fine-tune --pickup electric` doesn't apply to one (the old roadmap said
"collect electric DI and fine-tune"); electric support must come from fake-electric sources (B-6) and,
if needed, an unpaired adversarial stage on real electric DI (B-15).

**`eq_match.py` on synthetic data:** "acoustic" = 3 min of the piezo pick take; "electric" = piezo
audio through a textbook humbucker circuit (4.5 H, 8 kΩ, 470 pF, 200 kΩ load: peak 3.2 kHz, −18 dB at
10 kHz), +6 dB, with 50 Hz hum (−62 dBFS) and hiss, 5 s of silence at each end. Error of the recovered
curve vs the circuit, shape only, 80 Hz–12 kHz:

| electric recording | 1/3 oct | 1/6 oct | 1/12 oct |
|---|---|---|---|
| same performance | 0.21 dB RMS (max 0.8) | 0.07 (0.4) | 0.04 (0.2) |
| different passage, 1 min | 3.7 (14.1) | 4.2 (15.9) | 4.8 (15.4) |
| different passage, 3 min | 1.7 (4.9) | 2.2 (6.4) | 2.8 (8.1) |
| different passage, 5 min | 1.5 (4.2) | 1.9 (5.6) | 2.4 (7.2) |
| fingerstyle vs picked, 3 min | 3.9 (10.0) | 4.1 (10.4) | 4.7 (12.4) |

So the match is limited by *what was played*, not by the method: record ≥ 5 min of the same material
in the same style on both guitars; default smoothing 1/3 octave. Hum and hiss were subtracted exactly
(0.0 dB error below 300 Hz). The level offset was recovered exactly (+6.0 dB). The tool's split-half
check (curves from each half of the recordings) flags the bad cases: 0.1 dB same performance, 2.3 dB
at 5 min, 3.8–4.9 dB for 1 min / 3 min / finger-vs-pick; the full curve is off by roughly half the
split-half figure. Runtime 3 s, 170 MB peak for 3-min files (streamed in blocks).

## State at the end
- Nothing running. `out/checkpoints/offline/` empty. **`out/models/` = the 2025-10-19 models**, which
  predate rounds 03–07's fixes.
- Logs of both attempts: `out/logs/train_offline_20260923_attempt1{,_guard}.log`,
  `out/logs/train_offline_20260923{,_guard}.log`.
- New tools: `train_guard.py`, `eq_match.py` (`TRAINING.md` §2, §4, §5).
- No magnetic or electric recordings exist yet; `input/` holds the piezo pick/finger takes and the
  fine-tune pair only.
- Everything is committed and on `master` (also on `origin`).

## Next
1. The user records the EQ-match pair (B-2) and the magnetic triplets (B-1), and checks the pedal's
   input impedance (B-3). Recording instructions: `TRAINING.md` §4.
2. Run `eq_match.py` on the pair and A/B `acoustic_as_electric.wav` against the electric DI with the
   user (B-5).
3. Then fake-electric sources in training (B-6), the cross-pickup consistency loss (B-7), and the
   retrain (B-8) under `train_guard.py` with the Mac otherwise idle.
4. Record the pedal's name in `AGENTS.md` once the user gives it (B-18).
