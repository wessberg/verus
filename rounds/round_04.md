# Round 04 — Second review pass: real teacher, min-phase filters, EQ out of the model

- **Date:** 2026-09-02
- **Agent:** Claude (Claude Code)
- **Goal:** second review and fix pass over training, export and the pedal.
- **Commits:** `02a62b88` "fix: bug fixes"

*Written retroactively in round 07 from the previous agent's notes and `TRAINING.md`.*

## What happened
- The KD teacher is the real non-causal offline model (`FrozenTeacher`), fed extra context and cropped
  back onto the student's segment, instead of a causalized clone.
- Losses apply to the contexted core of each segment only (`*_LOSS_CORE`).
- Every FIR is minimum-phase via `_to_minimum_phase()` with its designed magnitude (not the square root).
- The master EQ left the models: RBJ biquads in C++ (`MasterEqIIR`) and Python (`master_eq_sos`,
  `--process --eq`), identical formulas. The models are EQ-free.
- Streaming colour-EQ taps (`taps_init`), fine-tune KD projections, hop-invariant style EMA,
  `KerasStreamingRunner` for all Keras streaming inference, persisted loss-weight calibration
  (`loss_weights.json`), per-sample gain augmentation, TFLite export without the redundant dry output.
- C++: lockstep rings, a semaphore-driven inference thread, model prewarm.

## State at the end
Committed; no retrain.

## Next
Third review pass (round 05).
