# Round 01 — Initial development: piezo→mic models

- **Date:** 2025-06-10 → 2025-10-19
- **Agent:** the user (commits by Frederik Wessberg)
- **Goal:** a neural model that turns an acoustic guitar's under-saddle piezo signal into the sound of
  the same guitar mic'd up, fast enough to run as a pedal.
- **Commits:** `a9eee6af` … `147e08ed` (~340 commits, mostly "feat: more work")

*Written retroactively in round 07 from git history and file dates. The commit messages carry almost
no information, so this is a coarse outline, not a record of decisions.*

## What happened
- 2025-06: first models (`out/piezo2mic_resnet.keras`, "finished training" 2025-06-24).
- 2025-07: experiments in a separate `piezo2mic-tweaked.py` (a stateful GRU; workarounds for ROCm, so
  training ran on an AMD GPU at some point); the fine-tune pair `input/finetune/` dates from 2025-07-27.
- 2025-08: the current training takes were recorded (`input/pick/…`, `input/finger/…`: 1 h each,
  96 kHz, piezo source + mic target, file dates 2025-08-10).
- 2025-09 → 2025-10: the offline-teacher / realtime-student structure with knowledge distillation,
  several retrains ("retrained" 2025-09-10), a C++ TFLite runtime in `cpp2/` ("it works" 2025-10-17).
- 2025-10-19: the models that are still in `out/models/` were trained and exported.

## What we learned
Not recorded. The design as it stands is described in `TRAINING.md` §1.

## State at the end
Offline (96 kHz) and realtime (48 kHz, T=32 streaming) models trained on the pick/finger takes; C++
runtime working on a computer.

## Next
Pi deployment (round 02).
