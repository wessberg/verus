# Round 08 — Stereo two-mic target (idea)

- **Date:** 2026-09-23
- **Agent:** Claude Opus 5.5 (Claude Code)
- **Goal:** answer the user's proposal to train toward their stereo two-mic recording setup instead of
  one mic, before the magnetic triplets are recorded.
- **Commits:** "Round 08: stereo two-mic target, booth session guide and booth sheet" on `master`, pushed
  (`RECORDING.md`, `rounds/round_08.md`, `rounds/INDEX.md`, `BACKLOG.md`, `TRAINING.md`, `AGENTS.md`; the
  booth sheet is a published artifact, not in the repo).

## What happened
1. The user proposed recording the target in stereo the way they record acoustic guitar anyway (one
   mic at the shoulder pointing down at the 12th fret, one in front of the soundhole), having the
   model produce both, and letting the player choose stereo or a mono sum.
2. Checked the pedal: it already has a stereo signal path (`cpp/src/main.cpp`: the mono model output
   is copied to L/R, then stereo reverb/delay/chorus and the FX loop; main L/R out; DAC output 8 = mono
   sum of main L/R).
3. Recommended it and recorded the design as BACKLOG B-22; B-1 (the triplet recording) now says to
   decide first, and that recording both mics as separate tracks keeps the option open.
4. The user answered: the old corpus's mic was a small condenser some distance off the soundhole,
   at an angle; the interface has 8 channels. Taken as a go for B-22. Wrote the four-track session
   into `TRAINING.md` §4 (setup, group alignment, file layout with a hard-linked shared target,
   disk) and updated B-1 and B-22.
5. The user asked for a guide from entering the booth to leaving it: what to play, how, pick or
   fingers, for how long. Wrote `RECORDING.md` (preparation, setup, a fixed start/end for every take,
   eight 15-min takes alternating styles with a block plan per style, an EQ-match take, raw exports,
   the electric half at home, a log sheet) and added B-23 (a prep tool from the raw exports to
   `input/`), so the user doesn't align anything by hand.
6. The user asked for a tickbox page with a clean design, and more detail on preparing the files
   afterwards (splitting, phase correction, RMS normalization). To ground the alignment, polarity and
   level rules, measured the old corpus (details under "What we learned"), then wrote `RECORDING.md`
   "Preparing the files" and sharpened B-23.
7. Published the **booth sheet**: https://claude.ai/artifact/6FcdUY47gjuzV5oNALoJjj (private; `db` capability). All of `RECORDING.md`'s booth
   steps as tickboxes, the take list with block times, a per-take Start/Stop timer that shows the
   current block (and keeps the screen awake), a **Mark noise** button that appends the time into the
   take's log, setup-note fields and a take log. State is mirrored on the device and synced to the
   page's database when online, so a flaky booth connection loses nothing; B-23 reads the noise
   marks from it. Checked: one look at 400 px width with a take running (then tightened the running
   strip and the take header); after publishing, both collections (`booth`, `takes`) list — empty,
   since the page writes only when used. Also fixed `RECORDING.md`'s fingerstyle blocks, which added
   up to 17.5 min instead of 15.

## Why
- Recording both mics costs nothing extra in the session that's about to happen and keeps every option
  open (either mic alone, a blend, stereo); re-recording later would cost a whole session.
- The differences between the two mics are mostly predictable from the pickup signal (radiation
  pattern of body vs strings, distances), so a shared core with a two-channel head should learn them.
- **Channel 1 = soundhole mic:** until stereo training exists the loader reads channel 1 only, and
  the soundhole mic is the closest to the old corpus's mic — so an interim mono run stays comparable.
- **Old corpus = warm start only:** its mic matches neither new position; both new output channels
  start from the old single head.
- **Hard-linked target:** the piezo and magnetic pairs of a take share one stereo target file without
  a second copy on a nearly full disk.

- **Takes are single-style and each covers all the material, ending in repertoire:** the style label
  is the folder, and the last 20 % of every take's chunks is the validation set
  (`_split_chunks_train_val`), so each take's tail should be what matters most.
- **Styles alternate take by take** so string ageing, temperature and fatigue don't correlate with style.
- **Bridge taps, not pick attacks, as the alignment reference:** the mic hears the pick's contact
  ~6.5 ms before the piezo sees the release (round 06).
- **Raw exports + a prep tool (B-23)** instead of aligning in the DAW: one alignment rule applied
  the same way to every take, and it's needed for end-user fine-tuning anyway.
- **Electric half played along to take 9** in headphones: the closest two performances can get
  without being the same one.

## What we learned
- **The old corpus is aligned; the pick-contact lead is real** (round 06 stands). Measured on the 1 h
  pick/finger takes:
  - onsets of notes after ≥ 200 ms of silence (316 picked, 428 fingerstyle), envelope crossing
    20 dB below the local peak: **midband 80 Hz–1 kHz, mic minus piezo median −1.5 ms picked
    (IQR −4.6 … +1.1), 0.0 ms fingerstyle (−2.0 … +1.5)**; **4–12 kHz: −6.5 ms in both** (IQR about
    −7/−8 … −5). So only the high band leads — the contact click — not the whole signal.
  - Cross-correlation of the playing is **ambiguous**: in 80 Hz–1 kHz it shows near-equal peaks at
    ~0 samples (positive, +0.38 … +0.54) and ~−275 samples (2.9 ms, negative) — half a period of a
    ~175 Hz body resonance — and which wins changes along the take. Broadband measures (GCC-PHAT,
    transfer-function impulse response, phase-slope group delay) put the peak at −4 … −6 ms and about
    half of the impulse response's energy before zero, but that is the high-band lead plus unwrap
    errors at coherence 0.2–0.6 (group delay per 5-min block jumps between ≈ −1.8 and −5.6 ms); the
    one strongly coherent band, 80–250 Hz (0.83–0.91), gives +0.2 … +0.8 ms. → align the new takes on
    the bridge taps, check with midband onsets.
  - **Polarity:** positive at zero offset (low band). **Levels (gated playing RMS):** piezo −28.4 /
    −29.2 dBFS, mic −26.6 / −31.7 dBFS (mic minus piezo +1.8 dB picked, −2.4 dB fingerstyle); the
    mic peaked at −1.7 / −1.8 dBFS — too little headroom.
- With the round-06 objective, per-channel losses constrain the inter-channel phase only below 3 kHz;
  above it the stereo relationship would be unsupervised → needs mid/side magnitude terms and a
  two-channel discriminator (B-22).
- Summing two spaced mics comb-filters (~0.5–1 ms apart → first notch ~0.5–1 kHz); the model would
  reproduce exactly that. Mono is better served by a blend control or a delay-aligned sum.
- **Disk:** 2 h of four tracks at 96 kHz / 24-bit is ~8.3 GB in the DAW session plus ~8.3 GB exported;
  12 GB were free (`df`, 2026-09-23), and training needs ≥ 6 GB free.

## State at the end
No code changes; recording instructions, the booth sheet (https://claude.ai/artifact/6FcdUY47gjuzV5oNALoJjj) and backlog updated. No new
recordings yet; the booth sheet's database is empty.

## Next
1. The user records the booth session per `RECORDING.md` (B-1, including the acoustic half of B-2),
   then the electric half at home.
2. B-23 (prep the raw exports into `input/`), B-5 (EQ match by ear), then B-22 + B-6 + B-7 in code,
   ahead of the retrain (B-8).
