# Round 03 — First review pass; magnetic-pickup conditioning

- **Date:** 2026-07-05
- **Agent:** Claude (Claude Code)
- **Goal:** review `piezo2mic.py` and the pedal, fix what was found, and add a pickup-type input so the
  model can take a magnetic pickup as well as the piezo.
- **Commits:** `e80bc943` "Fix all review findings and add magnetic-pickup conditioning"

*Written retroactively in round 07 from the previous agent's notes and `TRAINING.md`.*

## What happened
- Knowledge distillation reinstated for the realtime student.
- Streaming-state size bugs fixed; the style state stores the classifier logit (EMA on the logit, zero =
  probability 0.5).
- **Pickup axis:** `*_pickup_input` (0 = piezo, 1 = magnetic, user-set, never classified), a gated
  delta input adapter (`*_pickup_adapt_*`, zero-initialized, exact pass-through at pickup 0) and FiLM on
  `*_cond_embed` (seeded from the old `*_style_embed` by `copy_weights_by_name`). Data layout
  `input/<pickup>/<style>/{source,target}`; the old `input/<style>` layout is read as piezo.
- Training files scanned lazily (headers); the audio was still loaded whole into RAM (changed in round 07).
- Leak-free train/val split (last ~20 % of each file); Python↔C++ level parity constants.

## State at the end
Code fixed and committed; no retrain (the models are still from 2025-10-19). No magnetic recordings.

## Next
Second review pass (round 04).
